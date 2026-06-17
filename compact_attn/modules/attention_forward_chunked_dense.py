import os
from typing import Dict, Optional, Tuple, Union

import torch
from flash_attn import flash_attn_varlen_func

from compact_attn.kernels.varlen.compactattn_pack_kernels import (
    can_use_triton_pack,
    can_use_triton_scatter,
    gather_3d_to_cat,
    scatter_cat_to_3d,
    triton_available,
)
from compact_attn.kernels.varlen.indexed_dense_prefill_varlen import (
    build_paged_kv_cache_from_keep_block_fast,
    build_paged_kv_cache_from_block_mask,
    can_use_indexed_dense_prefill,
    flash_attn_indexed_prefill_from_paged_kv,
    flashinfer_indexed_prefill_from_paged_kv,
    indexed_dense_available,
)
from compact_attn.kernels.varlen.flashinfer_zero_copy_prefill import (
    flashinfer_prefill_cudnn_one_shot,
    flashinfer_prefill_zero_copy,
    flashinfer_prefill_zero_copy_per_query,
    flashinfer_prefill_zero_copy_subgroup,
)
from compact_attn.kernels.varlen.fa2_indexed_prefill import (
    can_use_fa2_indexed_prefill,
    fa2_indexed_available,
    run_fa2_indexed_prefill,
)
from compact_attn.kernels.varlen.indexed_dense_prefill_direct_triton import (
    can_use_direct_indexed_prefill,
    direct_indexed_prefill_available,
    run_direct_indexed_prefill,
)
from compact_attn.kernels.varlen.indexed_dense_cache_fill_cuda import (
    build_keep_curr_fast_cuda,
    build_keep_past_fast_cuda,
    can_use_cuda_keep_block_builder_fast,
    can_use_cuda_keep_block_fast,
)
from compact_attn.modules.common import _upad_input, pad_input, repeat_kv
from compact_attn.modules.dense_prefill import dense_prefill_full_kv

COMPACTATTN_VERSION = "CompactAttn-PagedOpt-v1.7"
_CHUNK_CAUSAL_MASK_CACHE: Dict[Tuple[str, int, int], torch.Tensor] = {}
_CHUNK_DIAG_INDEX_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


def _cuda_elapsed_ms(fn, enabled: bool = True):
    if not enabled:
        return fn(), 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    return out, float(start.elapsed_time(end))


def _cuda_elapsed_and_peak(fn, device: torch.device, *, measure_timing: bool = True, measure_peak: bool = False):
    if not measure_timing and not measure_peak:
        return fn(), 0.0, {}

    start = end = None
    if measure_timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    allocated_before = reserved_before = None
    if measure_peak and device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

    if measure_timing:
        start.record()
    out = fn()
    if measure_timing:
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
    else:
        elapsed_ms = 0.0

    if measure_peak and device.type == "cuda":
        if not measure_timing:
            torch.cuda.synchronize(device)
        mem_stats = {
            "peak_alloc_mb": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0),
            "alloc_delta_mb": float(torch.cuda.memory_allocated(device) - allocated_before) / (1024.0 * 1024.0),
            "reserved_delta_mb": float(torch.cuda.memory_reserved(device) - reserved_before) / (1024.0 * 1024.0),
        }
    else:
        mem_stats = {}
    return out, elapsed_ms, mem_stats


def _debug_env_mode(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def _debug_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _compactattn_effective_threshold(threshold: Union[float, torch.Tensor]):
    if isinstance(threshold, torch.Tensor):
        return threshold
    override = os.environ.get("SEERATTN_DEBUG_COMPACTATTN_THRESHOLD_OVERRIDE", None)
    if override is None:
        return float(threshold)
    try:
        return float(override)
    except ValueError:
        return float(threshold)


def _compactattn_query_block_mode() -> str:
    mode = _debug_env_mode("SEERATTN_DEBUG_COMPACTATTN_QUERY_BLOCK_MODE", "union")
    return mode if mode in {"union", "last"} else "union"


def _compactattn_kv_head_mode() -> str:
    mode = _debug_env_mode("SEERATTN_DEBUG_COMPACTATTN_KV_HEAD_MODE", "union")
    return mode if mode in {"union", "first"} else "union"


def _compactattn_current_chunk_full_open() -> bool:
    return not _debug_env_bool("SEERATTN_DEBUG_COMPACTATTN_DISABLE_CURRENT_OPEN", False)


def _compactattn_keep_recent_blocks(default: int) -> int:
    raw = os.environ.get("SEERATTN_DEBUG_COMPACTATTN_KEEP_RECENT_OVERRIDE", None)
    if raw is None:
        return int(default)
    try:
        return max(int(raw), 0)
    except ValueError:
        return int(default)


def _compactattn_peak_mem_debug() -> bool:
    return _debug_env_bool("SEERATTN_DEBUG_COMPACTATTN_PEAK_MEM", False)


def _compactattn_enable_full_kv_reuse_path() -> bool:
    return _debug_env_bool("SEERATTN_DEBUG_COMPACTATTN_ENABLE_FULL_KV_REUSE", False)


def _compactattn_enable_fused_builder_v2() -> bool:
    return _debug_env_bool("SEERATTN_DEBUG_COMPACTATTN_ENABLE_FUSED_BUILDER_V2", False)


def _empty_detail_stats() -> Dict[str, float]:
    return {
        "col_input_sanitize_ms": 0.0,
        "col_select_ratio_ms": 0.0,
        "col_index_keep_flat_sanitize_ms": 0.0,
        "col_index_sel_blocks_sum_ms": 0.0,
        "col_index_zero_row_fix_ms": 0.0,
        "col_index_sel_lens_ms": 0.0,
        "col_index_pages_per_row_ms": 0.0,
        "col_index_page_count_stats_ms": 0.0,
        "repeat_kv_ms": 0.0,
        "upad_input_ms": 0.0,
        "pad_output_ms": 0.0,
        "col_select_mask_ms": 0.0,
        "col_select_mask_past_ms": 0.0,
        "col_select_mask_curr_ms": 0.0,
        "col_keep_expand_ms": 0.0,
        "col_repeat_kv_ms": 0.0,
        "col_pack_prepare_ms": 0.0,
        "col_gather_qkv_ms": 0.0,
        "col_cu_seqlens_ms": 0.0,
        "col_unpack_scatter_ms": 0.0,
        "col_select_mask_fastpath_calls": 0.0,
        "col_select_mask_generic_calls": 0.0,
        "col_pack_impl_torch_calls": 0.0,
        "col_pack_impl_triton_calls": 0.0,
        "col_pack_impl_fallback_calls": 0.0,
        "col_index_build_ms": 0.0,
        "col_index_union_block_ms": 0.0,
        "col_index_block_table_ms": 0.0,
        "col_index_table_fill_ms": 0.0,
        "col_index_table_kernel_ms": 0.0,
        "col_index_compact_ms": 0.0,
        "col_index_compact_kernel_ms": 0.0,
        "col_index_compact_nonzero_ms": 0.0,
        "col_index_compact_post_ms": 0.0,
        "col_index_compact_fused_calls": 0.0,
        "col_index_compact_fallback_calls": 0.0,
        "col_index_compact_fused_post_ms": 0.0,
        "col_index_src_layout_ms": 0.0,
        "col_index_cache_fill_ms": 0.0,
        "col_index_cache_fill_kernel_ms": 0.0,
        "col_index_cache_fill_cuda_calls": 0.0,
        "col_index_cache_fill_cuda_fallback_calls": 0.0,
        "col_index_cache_fill_backend_id": 0.0,
        "col_index_cache_fill_triton_calls": 0.0,
        "col_index_cache_fill_torch_calls": 0.0,
        "col_index_cache_fill_fallback_calls": 0.0,
        "col_index_cache_fill_small_calls": 0.0,
        "col_index_cache_fill_medium_calls": 0.0,
        "col_index_cache_fill_large_calls": 0.0,
        "col_index_cache_fill_variant_id": 0.0,
        "col_index_cache_fill_tuned_calls": 0.0,
        "col_index_cache_fill_launch_blocks": 0.0,
        "col_index_cache_fill_effective_blocks": 0.0,
        "col_workspace_capacity_pages": 0.0,
        "col_workspace_required_pages": 0.0,
        "col_workspace_page_pool_capacity_pages": 0.0,
        "col_workspace_page_pool_required_pages": 0.0,
        "col_workspace_total_pages_used": 0.0,
        "col_workspace_max_pages_used": 0.0,
        "col_workspace_growth_events": 0.0,
        "col_workspace_k_alloc_mb": 0.0,
        "col_workspace_v_alloc_mb": 0.0,
        "col_workspace_block_table_alloc_mb": 0.0,
        "col_index_build_peak_alloc_mb": 0.0,
        "col_index_build_peak_reserved_mb": 0.0,
        "col_index_build_alloc_delta_mb": 0.0,
        "col_index_build_reserved_delta_mb": 0.0,
        "col_scratch_pool_growth_events": 0.0,
        "col_scratch_pool_pos_alloc_mb": 0.0,
        "col_scratch_pool_meta_alloc_mb": 0.0,
        "col_index_cache_fill_current_tail_ms": 0.0,
        "col_index_attn_q_layout_ms": 0.0,
        "col_index_attn_q_layout_cuda_calls": 0.0,
        "col_index_attn_out_layout_ms": 0.0,
        "col_index_attn_cache_seqlens_cast_ms": 0.0,
        "col_index_attn_block_table_cast_ms": 0.0,
        "col_index_attn_q_layout_peak_alloc_mb": 0.0,
        "col_index_attn_q_layout_peak_reserved_mb": 0.0,
        "col_index_attn_flash_kvcache_peak_alloc_mb": 0.0,
        "col_index_attn_flash_kvcache_peak_reserved_mb": 0.0,
        "col_index_attn_out_layout_peak_alloc_mb": 0.0,
        "col_index_attn_out_layout_peak_reserved_mb": 0.0,
        "col_index_kernel_peak_alloc_mb": 0.0,
        "col_index_kernel_peak_reserved_mb": 0.0,
        "col_index_kernel_alloc_delta_mb": 0.0,
        "col_index_kernel_reserved_delta_mb": 0.0,
        "col_indexed_dense_kernel_ms": 0.0,
        "col_indexed_dense_calls": 0.0,
        "col_indexed_dense_fallback_calls": 0.0,
        "col_tail_dense_fallback_calls": 0.0,
        "col_full_paged_kv_reuse_calls": 0.0,
        "col_full_paged_kv_init_calls": 0.0,
        "col_full_paged_kv_append_ms": 0.0,
        "col_index_page_table_only_ms": 0.0,
        "col_selected_kv_materialize_calls": 0.0,
        "col_selected_kv_materialize_ms": 0.0,
        "col_fused_builder_calls": 0.0,
        "col_fused_builder_ms": 0.0,
        "col_fused_builder_table_ms": 0.0,
        "col_fused_builder_fill_ms": 0.0,
        "col_fused_builder_row_tiled_calls": 0.0,
        "col_fused_builder_tail_fused_calls": 0.0,
        "col_builder_peak_alloc_mb": 0.0,
        "col_builder_peak_reserved_mb": 0.0,
        "col_direct_index_build_ms": 0.0,
        "col_direct_kernel_ms": 0.0,
        "col_direct_calls": 0.0,
        "col_direct_fallback_calls": 0.0,
        "col_indexed_impl_fa2_calls": 0.0,
        "col_fa2_indexed_kernel_ms": 0.0,
        "col_fa2_indexed_calls": 0.0,
        "col_fa2_indexed_fallback_calls": 0.0,
        "col_fa2_indexed_v2_short_calls": 0.0,
        "col_fa2_indexed_v2_long_calls": 0.0,
        "col_fa2_indexed_v2_fallback_calls": 0.0,
        "col_fa2_indexed_v2_kernel_ms": 0.0,
        "col_fa2_indexed_v3_short_calls": 0.0,
        "col_fa2_indexed_v3_long_calls": 0.0,
        "col_fa2_indexed_v3_fallback_calls": 0.0,
        "col_fa2_indexed_v3_split_k": 0.0,
        "col_fa2_indexed_v3_past_kernel_ms": 0.0,
        "col_fa2_indexed_v3_reduce_ms": 0.0,
        "col_fa2_indexed_v3_current_kernel_ms": 0.0,
        "col_fa2_indexed_v3_kernel_ms": 0.0,
    }


def _get_chunk_causal_mask(q_blocks: int, curr_k_blocks: int, device: torch.device) -> torch.Tensor:
    key = (str(device), int(q_blocks), int(curr_k_blocks))
    cached = _CHUNK_CAUSAL_MASK_CACHE.get(key, None)
    if cached is not None and cached.device == device:
        return cached
    mask = torch.tril(
        torch.ones((q_blocks, curr_k_blocks), dtype=torch.bool, device=device)
    )
    _CHUNK_CAUSAL_MASK_CACHE[key] = mask
    return mask


def _get_chunk_diag_index(q_blocks: int, device: torch.device) -> torch.Tensor:
    key = (str(device), int(q_blocks))
    cached = _CHUNK_DIAG_INDEX_CACHE.get(key, None)
    if cached is not None and cached.device == device:
        return cached
    idx = torch.arange(q_blocks, device=device)
    _CHUNK_DIAG_INDEX_CACHE[key] = idx
    return idx


def _ensure_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones((batch_size, kv_len), dtype=torch.bool, device=device)
    return attention_mask.to(torch.bool)


def _dense_prefill_full_kv(
    query_states: torch.Tensor,  # [B, Q, Hq, D]
    key_states: torch.Tensor,  # [B, K, Hkv, D]
    value_states: torch.Tensor,  # [B, K, Hkv, D]
    attention_mask: Optional[torch.Tensor],  # [B, K]
    softmax_scale: float,
    num_key_value_groups: int,
    fallback_used: float = 1.0,
    measure_timing: bool = True,
    attn_module=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    out, shared_stats = dense_prefill_full_kv(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        softmax_scale=softmax_scale,
        num_key_value_groups=num_key_value_groups,
        fallback_used=fallback_used,
        measure_timing=measure_timing,
        attn_module=attn_module,
    )
    stats = _empty_detail_stats()
    stats.update(shared_stats)
    return out, stats


def _build_keep_block_mask(
    attn_gate_score: torch.Tensor,  # [B, H, Qb, Kb]
    block_attention_mask: Optional[torch.Tensor],  # [B, 1, Qb, Kb]
    threshold: Union[float, torch.Tensor],
    keep_recent_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keep_recent_blocks = _compactattn_keep_recent_blocks(keep_recent_blocks)
    bsz, num_heads, _, _ = attn_gate_score.shape
    device = attn_gate_score.device

    if block_attention_mask is not None:
        if block_attention_mask.shape[-2:] != attn_gate_score.shape[-2:]:
            block_attention_mask = block_attention_mask[
                ...,
                -attn_gate_score.shape[-2]:,
                -attn_gate_score.shape[-1]:,
            ]
        valid_mask = block_attention_mask.to(torch.bool).expand_as(attn_gate_score)
    else:
        valid_mask = torch.tril(torch.ones_like(attn_gate_score, dtype=torch.bool, device=device))

    masked_score = attn_gate_score.masked_fill(~valid_mask, float("-inf"))
    query_block_mode = _compactattn_query_block_mode()
    if query_block_mode == "last":
        col_score = masked_score[:, :, -1, :]  # [B, H, Kb]
        valid_columns = valid_mask[:, :, -1, :]  # [B, H, Kb]
    else:
        col_score = masked_score.amax(dim=2)  # [B, H, Kb]
        valid_columns = valid_mask.any(dim=2)  # [B, H, Kb]

    if isinstance(threshold, torch.Tensor):
        threshold_cmp = threshold.to(device=device, dtype=col_score.dtype)
        if threshold_cmp.dim() == 0:
            threshold_cmp = threshold_cmp.view(1, 1, 1)
        elif threshold_cmp.dim() == 1:
            threshold_cmp = threshold_cmp.view(-1, 1, 1)
        else:
            raise ValueError(
                f"Expected scalar or 1D batch threshold tensor, got shape={tuple(threshold.shape)}"
            )
    else:
        threshold_cmp = threshold

    keep_block = (col_score > threshold_cmp) & valid_columns

    if keep_recent_blocks > 0:
        # For each (B, H), keep the last N valid columns without Python loops.
        rev_cum = torch.cumsum(valid_columns.to(torch.int32).flip(dims=[-1]), dim=-1).flip(dims=[-1])
        tail_mask = (rev_cum > 0) & (rev_cum <= keep_recent_blocks) & valid_columns
        keep_block = keep_block | tail_mask

    return keep_block, valid_columns, col_score, valid_mask


def _apply_keep_recent_to_keep_block(
    keep_block: torch.Tensor,  # [B, H, Kb]
    valid_block: Optional[torch.Tensor],  # [B, H, Kb]
    keep_recent_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep_recent_blocks = _compactattn_keep_recent_blocks(keep_recent_blocks)
    if valid_block is None:
        valid_columns = torch.ones_like(keep_block, dtype=torch.bool, device=keep_block.device)
    else:
        valid_columns = valid_block.to(torch.bool)

    keep_block = keep_block.to(torch.bool)
    if keep_recent_blocks <= 0:
        return keep_block, valid_columns

    # A projected-selection path with no explicit valid mask means
    # every column is valid and "keep the last N valid blocks" degenerates to a tail slice.
    if valid_block is None:
        recent = min(int(keep_recent_blocks), int(keep_block.shape[-1]))
        if recent > 0:
            keep_block[..., -recent:] = True
        return keep_block, valid_columns

    rev_cum = torch.cumsum(valid_columns.to(torch.int32).flip(dims=[-1]), dim=-1).flip(dims=[-1])
    tail_mask = (rev_cum > 0) & (rev_cum <= keep_recent_blocks) & valid_columns
    return keep_block | tail_mask, valid_columns


def _expand_blocks_to_tokens(keep_block: torch.Tensor, block_size: int, kv_len: int) -> torch.Tensor:
    return keep_block.repeat_interleave(block_size, dim=-1)[..., :kv_len]


def _maybe_pad_indexed_dense_tail_inputs(
    key_states: torch.Tensor,  # [B, K, Hkv, D]
    value_states: torch.Tensor,  # [B, K, Hkv, D]
    keep_block_kv: torch.Tensor,  # [B, Hkv, Kb]
    attention_mask: torch.Tensor,  # [B, K]
    block_size: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    kv_len = int(key_states.shape[1])
    if kv_len <= 0 or (kv_len % block_size) == 0:
        return None
    if not _compactattn_current_chunk_full_open():
        return None

    pad_tokens = int(((kv_len + block_size - 1) // block_size) * block_size - kv_len)
    if pad_tokens <= 0:
        return None

    bsz, _, num_kv_heads, head_dim = key_states.shape
    device = key_states.device
    dtype = key_states.dtype
    mask_dtype = attention_mask.dtype

    pad_k = torch.zeros((bsz, pad_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    pad_v = torch.zeros((bsz, pad_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    padded_key_states = torch.cat([key_states, pad_k], dim=1)
    padded_value_states = torch.cat([value_states, pad_v], dim=1)
    padded_attention_mask = torch.cat(
        [attention_mask, torch.zeros((bsz, pad_tokens), device=device, dtype=mask_dtype)],
        dim=1,
    )

    kv_blocks_aligned = padded_key_states.shape[1] // block_size
    tail_blocks = kv_blocks_aligned - keep_block_kv.shape[-1]
    if tail_blocks < 0:
        return None

    padded_keep_block_kv = keep_block_kv.to(torch.bool)
    if tail_blocks > 0:
        # The missing aligned block is the final partially-filled current chunk.
        # Mark it selected so the paged indexed path can materialize it instead
        # of falling back to dense on every layer tail.
        tail_keep = torch.ones(
            (bsz, keep_block_kv.shape[1], tail_blocks),
            device=device,
            dtype=torch.bool,
        )
        padded_keep_block_kv = torch.cat([padded_keep_block_kv, tail_keep], dim=-1)
    return (
        padded_key_states,
        padded_value_states,
        padded_keep_block_kv,
        padded_attention_mask,
        pad_tokens,
    )


def _can_use_chunked_mask_fast_path(
    attn_gate_score: torch.Tensor,
    block_attention_mask: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    q_len: int,
    kv_len: int,
    block_size: int,
    skip_content_checks: bool = False,
) -> bool:
    # --- Cheap CPU-only shape/alignment checks (no GPU sync) ---
    if kv_len <= q_len:
        return False
    if block_size <= 0:
        return False
    past_len = kv_len - q_len
    if (q_len % block_size) != 0 or (kv_len % block_size) != 0 or (past_len % block_size) != 0:
        return False

    q_blocks = q_len // block_size
    kv_blocks = kv_len // block_size
    if attn_gate_score.shape[2] != q_blocks or attn_gate_score.shape[3] != kv_blocks:
        return False

    if block_attention_mask is not None:
        if (
            block_attention_mask.shape[0] != attn_gate_score.shape[0]
            or block_attention_mask.shape[2] != q_blocks
            or block_attention_mask.shape[3] != kv_blocks
        ):
            return False

    # --- Expensive GPU→host content checks (skippable after first verification) ---
    if skip_content_checks:
        return True

    if not bool(attention_mask.to(torch.bool).all().item()):
        return False

    if block_attention_mask is not None:
        past_k_blocks = (kv_len - q_len) // block_size
        if past_k_blocks > 0:
            # Fast path assumes every past key block is valid for all query blocks.
            if not bool(block_attention_mask.to(torch.bool)[..., :past_k_blocks].all().item()):
                return False
        curr_k_blocks = kv_blocks - past_k_blocks
        if curr_k_blocks > 0:
            expected_curr = _get_chunk_causal_mask(q_blocks, curr_k_blocks, attn_gate_score.device)
            curr_mask = block_attention_mask.to(torch.bool)[..., past_k_blocks:kv_blocks]
            if not bool((curr_mask == expected_curr.view(1, 1, q_blocks, curr_k_blocks)).all().item()):
                return False
    return True


# Cache for fast-path content verification.  Keyed on (device, q_len, block_size).
# Once verified for a given (device, q_len, block_size), content checks are
# skipped for subsequent chunks in the same sequence (the conditions are
# invariant: attention_mask is all-ones, block_attention_mask follows causal).
_FAST_PATH_CONTENT_VERIFIED: Dict[Tuple[str, int, int], bool] = {}


def clear_fast_path_content_cache() -> None:
    """Reset fast-path verification cache (call between sequences)."""
    _FAST_PATH_CONTENT_VERIFIED.clear()


def _build_keep_block_mask_chunked_fast_path(
    attn_gate_score: torch.Tensor,  # [B, H, Qb, Kb]
    block_attention_mask: Optional[torch.Tensor],  # [B, 1, Qb, Kb]
    threshold: float,
    keep_recent_blocks: int,
    q_len: int,
    kv_len: int,
    block_size: int,
    measure_timing: bool = False,
) -> Tuple[torch.Tensor, bool, float, float]:
    keep_recent_blocks = _compactattn_keep_recent_blocks(keep_recent_blocks)
    bsz, num_heads, q_blocks, kv_blocks = attn_gate_score.shape
    past_len = kv_len - q_len
    past_k_blocks = past_len // block_size
    curr_k_blocks = kv_blocks - past_k_blocks
    device = attn_gate_score.device

    use_cuda_fast = can_use_cuda_keep_block_fast(attn_gate_score)

    query_block_mode = _compactattn_query_block_mode()

    def _run_past():
        if past_k_blocks <= 0:
            return torch.zeros((bsz, num_heads, 0), dtype=torch.bool, device=device)
        if use_cuda_fast:
            return build_keep_past_fast_cuda(
                attn_gate_score=attn_gate_score,
                threshold=threshold,
                past_k_blocks=past_k_blocks,
            )
        if query_block_mode == "last":
            past_col_score = attn_gate_score[:, :, -1, :past_k_blocks]
        else:
            past_col_score = attn_gate_score[..., :past_k_blocks].amax(dim=2)
        return past_col_score > threshold

    keep_past, select_mask_past_ms = _cuda_elapsed_ms(_run_past, enabled=measure_timing)

    def _run_current():
        if curr_k_blocks <= 0:
            return torch.zeros((bsz, num_heads, 0), dtype=torch.bool, device=device)
        if use_cuda_fast:
            return build_keep_curr_fast_cuda(
                attn_gate_score=attn_gate_score,
                threshold=threshold,
                past_k_blocks=past_k_blocks,
                curr_k_blocks=curr_k_blocks,
            )
        attn_curr = attn_gate_score[..., past_k_blocks:kv_blocks]  # [B, H, Qb, Qb]
        if query_block_mode == "last":
            col_score_curr = attn_curr[:, :, -1, :]
        elif curr_k_blocks == q_blocks:
            # For current-chunk columns, valid rows are suffixes (i >= j). Use suffix-max+diag
            # instead of constructing a large expanded triangular mask per head.
            suffix_max = torch.flip(
                torch.cummax(torch.flip(attn_curr, dims=[2]), dim=2).values, dims=[2]
            )
            diag_idx = _get_chunk_diag_index(curr_k_blocks, device)
            col_score_curr = suffix_max[:, :, diag_idx, diag_idx]
        else:
            causal_curr = _get_chunk_causal_mask(q_blocks, curr_k_blocks, device)
            col_score_curr = attn_curr.masked_fill(
                ~causal_curr.view(1, 1, q_blocks, curr_k_blocks), float("-inf")
            ).amax(dim=2)
        return col_score_curr > threshold

    keep_curr, select_mask_curr_ms = _cuda_elapsed_ms(_run_current, enabled=measure_timing)

    keep_block = torch.empty((bsz, num_heads, kv_blocks), dtype=torch.bool, device=device)
    if past_k_blocks > 0:
        keep_block[..., :past_k_blocks] = keep_past
    if curr_k_blocks > 0:
        keep_block[..., past_k_blocks:kv_blocks] = keep_curr

    if keep_recent_blocks > 0:
        recent = min(int(keep_recent_blocks), int(kv_blocks))
        if recent > 0:
            keep_block[..., kv_blocks - recent : kv_blocks] = True

    return keep_block, True, select_mask_past_ms, select_mask_curr_ms


def _build_kv_head_union_keep_block_mask(
    keep_block: torch.Tensor,  # [B, Hq, Kb]
    q_len: int,
    kv_len: int,
    block_size: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    past_len = kv_len - q_len
    past_k_blocks = past_len // block_size
    kv_blocks = kv_len // block_size
    bsz, num_q_heads, _ = keep_block.shape
    num_kv_heads = num_q_heads // num_key_value_groups

    keep_past_q = keep_block[..., :past_k_blocks]  # [B, Hq, past_k_blocks]
    kv_head_mode = _compactattn_kv_head_mode()
    if past_k_blocks > 0:
        keep_past_view = keep_past_q.view(
            bsz, num_kv_heads, num_key_value_groups, past_k_blocks
        )
        if kv_head_mode == "first":
            keep_past_kv = keep_past_view[:, :, 0, :]
        else:
            keep_past_kv = keep_past_view.any(dim=2)  # [B, Hkv, past_k_blocks]
    else:
        keep_past_kv = torch.zeros(
            (bsz, num_kv_heads, 0), dtype=torch.bool, device=keep_block.device
        )

    curr_k_blocks = kv_blocks - past_k_blocks
    if _compactattn_current_chunk_full_open():
        keep_curr_kv = torch.ones(
            (bsz, num_kv_heads, curr_k_blocks), dtype=torch.bool, device=keep_block.device
        )
    else:
        keep_curr_q = keep_block[..., past_k_blocks:kv_blocks]
        keep_curr_view = keep_curr_q.view(
            bsz, num_kv_heads, num_key_value_groups, curr_k_blocks
        )
        if kv_head_mode == "first":
            keep_curr_kv = keep_curr_view[:, :, 0, :]
        else:
            keep_curr_kv = keep_curr_view.any(dim=2)
    return torch.cat([keep_past_kv, keep_curr_kv], dim=-1)


def _build_q_head_keep_block_mask(
    keep_block: torch.Tensor,  # [B, Hq, Kb]
    q_len: int,
    kv_len: int,
    block_size: int,
) -> torch.Tensor:
    past_len = kv_len - q_len
    past_k_blocks = past_len // block_size
    kv_blocks = kv_len // block_size
    keep_past_q = keep_block[..., :past_k_blocks]
    curr_k_blocks = kv_blocks - past_k_blocks
    if _compactattn_current_chunk_full_open():
        keep_curr_q = torch.ones(
            (*keep_block.shape[:2], curr_k_blocks),
            dtype=torch.bool,
            device=keep_block.device,
        )
    else:
        keep_curr_q = keep_block[..., past_k_blocks:kv_blocks]
    return torch.cat([keep_past_q, keep_curr_q], dim=-1).contiguous()


def _project_short_qhead_keep_block_to_kv_full_width(
    keep_block: torch.Tensor,  # [B, Hq, Kb_short]
    expected_kv_blocks: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    bsz, num_q_heads, short_kv_blocks = keep_block.shape
    if short_kv_blocks > expected_kv_blocks:
        raise ValueError(
            f"short keep_block width {short_kv_blocks} exceeds expected KV blocks {expected_kv_blocks}"
        )
    num_kv_heads = num_q_heads // num_key_value_groups
    keep_view = keep_block.view(
        bsz, num_kv_heads, num_key_value_groups, short_kv_blocks
    )
    if _compactattn_kv_head_mode() == "first":
        keep_short_kv = keep_view[:, :, 0, :]
    else:
        keep_short_kv = keep_view.any(dim=2)

    keep_full_kv = torch.zeros(
        (bsz, num_kv_heads, expected_kv_blocks),
        dtype=torch.bool,
        device=keep_block.device,
    )
    if short_kv_blocks > 0:
        keep_full_kv[..., expected_kv_blocks - short_kv_blocks : expected_kv_blocks] = keep_short_kv
    return keep_full_kv


def _build_kv_head_union_past_block_indices(
    keep_block: torch.Tensor,  # [B, Hq, Kb]
    q_len: int,
    kv_len: int,
    block_size: int,
    num_key_value_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    past_len = kv_len - q_len
    past_k_blocks = past_len // block_size
    bsz, num_q_heads, _ = keep_block.shape
    num_kv_heads = num_q_heads // num_key_value_groups
    device = keep_block.device

    if past_k_blocks <= 0:
        empty_indices = torch.full(
            (bsz, num_kv_heads, 1), -1, dtype=torch.int32, device=device
        )
        empty_counts = torch.zeros((bsz, num_kv_heads), dtype=torch.int32, device=device)
        return empty_indices, empty_counts

    keep_past_q = keep_block[..., :past_k_blocks]  # [B, Hq, past_k_blocks]
    keep_past_view = keep_past_q.view(
        bsz, num_kv_heads, num_key_value_groups, past_k_blocks
    )
    if _compactattn_kv_head_mode() == "first":
        keep_past_kv = keep_past_view[:, :, 0, :]
    else:
        keep_past_kv = keep_past_view.any(dim=2)  # [B, Hkv, past_k_blocks]

    rows = bsz * num_kv_heads
    keep_flat = keep_past_kv.reshape(rows, past_k_blocks)
    counts_flat = keep_flat.sum(dim=-1, dtype=torch.int32)
    # Avoid host sync from max().item(); use fixed-width past block capacity.
    indices_flat = torch.full((rows, past_k_blocks), -1, dtype=torch.int32, device=device)
    pos = torch.nonzero(keep_flat, as_tuple=False)
    if pos.numel() > 0:
        row_idx = pos[:, 0].to(torch.long)
        blk_idx = pos[:, 1].to(torch.int32)
        offsets = torch.cumsum(counts_flat.to(torch.int64), dim=0) - counts_flat.to(torch.int64)
        local_rank = torch.arange(pos.shape[0], device=device, dtype=torch.long) - offsets[row_idx]
        indices_flat[row_idx, local_rank] = blk_idx

    return (
        indices_flat.view(bsz, num_kv_heads, past_k_blocks),
        counts_flat.view(bsz, num_kv_heads),
    )


def _build_kv_head_past_block_indices_from_kv_keep_block(
    keep_block_kv: torch.Tensor,  # [B, Hkv, Kb]
    q_len: int,
    kv_len: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    past_len = kv_len - q_len
    past_k_blocks = past_len // block_size
    bsz, num_kv_heads, _ = keep_block_kv.shape
    device = keep_block_kv.device

    if past_k_blocks <= 0:
        empty_indices = torch.full(
            (bsz, num_kv_heads, 1), -1, dtype=torch.int32, device=device
        )
        empty_counts = torch.zeros((bsz, num_kv_heads), dtype=torch.int32, device=device)
        return empty_indices, empty_counts

    keep_past_kv = keep_block_kv[..., :past_k_blocks].to(torch.bool).contiguous()
    rows = bsz * num_kv_heads
    keep_flat = keep_past_kv.reshape(rows, past_k_blocks)
    counts_flat = keep_flat.sum(dim=-1, dtype=torch.int32)
    indices_flat = torch.full((rows, past_k_blocks), -1, dtype=torch.int32, device=device)
    pos = torch.nonzero(keep_flat, as_tuple=False)
    if pos.numel() > 0:
        row_idx = pos[:, 0].to(torch.long)
        blk_idx = pos[:, 1].to(torch.int32)
        offsets = torch.cumsum(counts_flat.to(torch.int64), dim=0) - counts_flat.to(torch.int64)
        local_rank = torch.arange(pos.shape[0], device=device, dtype=torch.long) - offsets[row_idx]
        indices_flat[row_idx, local_rank] = blk_idx

    return (
        indices_flat.view(bsz, num_kv_heads, past_k_blocks),
        counts_flat.view(bsz, num_kv_heads),
    )


def _run_indexed_dense_chunked_prefill(
    query_states: torch.Tensor,  # [B, Q, Hq, D]
    key_states: torch.Tensor,  # [B, K, Hkv, D]
    value_states: torch.Tensor,  # [B, K, Hkv, D]
    keep_block: Optional[torch.Tensor],  # [B, Hq, Kb]
    keep_block_kv_preprojected: Optional[torch.Tensor],
    attention_mask: torch.Tensor,  # [B, K]
    block_size: int,
    num_key_value_groups: int,
    softmax_scale: float,
    page_block_size: int = 256,
    cache_fill_backend: str = "auto",
    measure_timing: bool = True,
    indexed_impl: str = "fa2_paged",
    k_hf: Optional[torch.Tensor] = None,  # [B, Hkv, K, D] heads-first, for fi_zero_copy
    v_hf: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    if indexed_impl in {"fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"}:
        if k_hf is None or v_hf is None:
            raise ValueError(f"{indexed_impl} requires k_hf and v_hf (heads-first KV cache)")
        # Resolve keep_block_kv [B, Hkv, Kb] — already available or reduce from [B, Hq, Kb]
        q_len_zc = query_states.shape[1]
        kv_len_zc = key_states.shape[1]
        if kv_len_zc % block_size != 0:
            return None, {
                "col_index_build_ms": 0.0,
                "col_indexed_dense_kernel_ms": 0.0,
                "dense_kernel_ms": 0.0,
                "gather_pack_ms": 0.0,
                "pack_prepare_ms": 0.0,
                "gather_qkv_ms": 0.0,
                "cu_seqlens_ms": 0.0,
                "unpack_scatter_ms": 0.0,
                "pack_impl_torch_calls": 0.0,
                "pack_impl_triton_calls": 0.0,
                "pack_impl_fallback_calls": 0.0,
                "col_indexed_dense_calls": 0.0,
                "col_indexed_dense_fallback_calls": 1.0,
                "col_fi_zero_copy_calls": 0.0,
                "col_fi_zero_copy_misaligned_tail_calls": 1.0,
            }
        if indexed_impl == "fi_zero_copy_per_query":
            if keep_block is None:
                raise ValueError("fi_zero_copy_per_query requires q-head keep_block")
            keep_block_q = _build_q_head_keep_block_mask(
                keep_block=keep_block,
                q_len=q_len_zc,
                kv_len=kv_len_zc,
                block_size=block_size,
            )
            zero_copy_fn = flashinfer_prefill_zero_copy_per_query
            out, zc_stats = zero_copy_fn(
                q=query_states.contiguous(),
                k_hf=k_hf,
                v_hf=v_hf,
                keep_block_q=keep_block_q,
                num_key_value_groups=num_key_value_groups,
                softmax_scale=softmax_scale,
                block_size=block_size,
                measure_timing=measure_timing,
            )
        elif indexed_impl == "fi_zero_copy_subgroup":
            if keep_block_kv_preprojected is None:
                raise ValueError("fi_zero_copy_subgroup requires subgroup keep_block")
            query_subgroup_size = int(os.environ.get("SEER_ZC_QUERY_SUBGROUP_SIZE", "4"))
            out, zc_stats = flashinfer_prefill_zero_copy_subgroup(
                q=query_states.contiguous(),
                k_hf=k_hf,
                v_hf=v_hf,
                keep_block_group=keep_block_kv_preprojected.to(torch.bool).contiguous(),
                num_key_value_groups=num_key_value_groups,
                query_subgroup_size=query_subgroup_size,
                softmax_scale=softmax_scale,
                block_size=block_size,
                measure_timing=measure_timing,
            )
        else:
            if keep_block_kv_preprojected is not None:
                keep_block_kv = keep_block_kv_preprojected.to(torch.bool).contiguous()
            else:
                keep_block_kv = _build_kv_head_union_keep_block_mask(
                    keep_block=keep_block,
                    q_len=q_len_zc,
                    kv_len=kv_len_zc,
                    block_size=block_size,
                    num_key_value_groups=num_key_value_groups,
                )
            zero_copy_fn = (
                flashinfer_prefill_cudnn_one_shot
                if indexed_impl == "cudnn_one_shot"
                else flashinfer_prefill_zero_copy
            )
            out, zc_stats = zero_copy_fn(
                q=query_states.contiguous(),
                k_hf=k_hf,
                v_hf=v_hf,
                keep_block_kv=keep_block_kv,
                num_key_value_groups=num_key_value_groups,
                softmax_scale=softmax_scale,
                block_size=block_size,
                measure_timing=measure_timing,
            )
        zc_metadata_ms = float(zc_stats.get("zc_metadata_ms", 0.0))
        zc_attn_ms = float(zc_stats.get("zc_attn_ms", 0.0))
        empty_stats: Dict[str, float] = {
            "col_index_build_ms": zc_metadata_ms,
            "col_index_union_block_ms": 0.0,
            "col_index_block_table_ms": 0.0,
            "col_index_table_fill_ms": 0.0,
            "col_index_table_kernel_ms": 0.0,
            "col_index_compact_ms": 0.0,
            "col_index_compact_kernel_ms": 0.0,
            "col_index_compact_nonzero_ms": 0.0,
            "col_index_compact_post_ms": 0.0,
            "col_index_compact_fused_calls": 0.0,
            "col_index_compact_fallback_calls": 0.0,
            "col_index_compact_fused_post_ms": 0.0,
            "col_index_src_layout_ms": 0.0,
            "col_index_cache_fill_ms": 0.0,
            "col_index_cache_fill_kernel_ms": 0.0,
            "col_indexed_dense_kernel_ms": zc_attn_ms,
            "dense_kernel_ms": zc_attn_ms,
            "gather_pack_ms": 0.0,
            "pack_prepare_ms": 0.0,
            "gather_qkv_ms": 0.0,
            "cu_seqlens_ms": 0.0,
            "unpack_scatter_ms": 0.0,
            "pack_impl_torch_calls": 0.0,
            "pack_impl_triton_calls": 0.0,
            "pack_impl_fallback_calls": 0.0,
            "col_indexed_dense_calls": 1.0,
            "col_indexed_dense_fallback_calls": 0.0,
            "col_index_attn_q_workspace_alloc_mb": 0.0,
            "col_index_attn_out_workspace_alloc_mb": 0.0,
            "col_index_attn_q_workspace_growth_events": 0.0,
            "col_index_attn_cache_seqlens_copy_calls": 0.0,
            "col_index_attn_block_table_copy_calls": 0.0,
            "col_index_attn_cache_seqlens_cast_ms": 0.0,
            "col_index_attn_block_table_cast_ms": 0.0,
            "col_index_attn_q_layout_ms": float(zc_stats.get("zc_q_layout_ms", 0.0)),
            "col_index_attn_out_layout_ms": 0.0,
            "col_selected_kv_materialize_calls": 0.0,
            "col_selected_kv_materialize_ms": 0.0,
            "col_indexed_impl_fa2_calls": 0.0,
            "col_fi_zero_copy_calls": 1.0,
            "col_fi_zero_copy_metadata_ms": zc_metadata_ms,
            "col_fi_zero_copy_q_layout_ms": float(zc_stats.get("zc_q_layout_ms", 0.0)),
            "col_fi_zero_copy_wrapper_init_ms": float(zc_stats.get("zc_wrapper_init_ms", 0.0)),
            "col_fi_zero_copy_wrapper_init_host_ms": float(
                zc_stats.get("zc_wrapper_init_host_ms", 0.0)
            ),
            "col_fi_zero_copy_plan_ms": float(zc_stats.get("zc_plan_ms", 0.0)),
            "col_fi_zero_copy_plan_host_ms": float(zc_stats.get("zc_plan_host_ms", 0.0)),
            "col_fi_zero_copy_run_ms": float(zc_stats.get("zc_run_ms", 0.0)),
            "col_fi_zero_copy_attn_ms": zc_attn_ms,
            "col_fi_zero_copy_total_ms": float(zc_stats.get("zc_total_ms", 0.0)),
            "col_fi_zero_copy_rows": float(zc_stats.get("zc_rows", 0.0)),
            "col_fi_zero_copy_kv_blocks": float(zc_stats.get("zc_kv_blocks", 0.0)),
            "col_fi_zero_copy_selected_pages": float(zc_stats.get("zc_selected_pages", 0.0)),
            "col_fi_zero_copy_q_tokens": float(zc_stats.get("zc_q_tokens", 0.0)),
        }
        return out, empty_stats
    if indexed_impl == "fi_paged":
        page_block_size = 64
    q_len = query_states.shape[1]
    kv_len = key_states.shape[1]
    device = query_states.device
    peak_mem_debug = _compactattn_peak_mem_debug()
    past_len = kv_len - q_len
    past_k_blocks = max(past_len // block_size, 0)
    curr_k_blocks = q_len // block_size
    full_block_aligned = (
        q_len > 0
        and kv_len >= q_len
        and (q_len % block_size) == 0
        and (kv_len % block_size) == 0
        and (past_len % block_size) == 0
    )
    use_preprojected_keep_block = keep_block_kv_preprojected is not None
    keep_block_fast_builder = keep_block_kv_preprojected if use_preprojected_keep_block else keep_block
    fast_builder_num_key_value_groups = 1 if use_preprojected_keep_block else num_key_value_groups
    use_fused_builder_v2 = _compactattn_enable_fused_builder_v2()
    use_fast_builder = False
    if cache_fill_backend in {"auto", "cuda"} and full_block_aligned:
        use_fast_builder = (
            keep_block_fast_builder is not None
            and can_use_cuda_keep_block_builder_fast(
                keep_block=keep_block_fast_builder,
                k=key_states,
                v=value_states,
                block_size=block_size,
                page_block_size=page_block_size,
                num_key_value_groups=fast_builder_num_key_value_groups,
                past_k_blocks=past_k_blocks,
                curr_k_blocks=curr_k_blocks,
            )
        )

    if use_fast_builder:
        def _run_build():
            return build_paged_kv_cache_from_keep_block_fast(
                k=key_states.contiguous(),
                v=value_states.contiguous(),
                keep_block=keep_block_fast_builder.contiguous(),
                num_key_value_groups=fast_builder_num_key_value_groups,
                q_len=q_len,
                block_size=block_size,
                page_block_size=page_block_size,
                cache_fill_backend=cache_fill_backend,
                prefer_fused_builder_v2=use_fused_builder_v2,
                measure_timing=measure_timing,
            )

        (k_cache, v_cache, block_table, cache_seqlens, index_timing), _, build_mem = _cuda_elapsed_and_peak(
            _run_build,
            device,
            measure_timing=False,
            measure_peak=peak_mem_debug,
        )
        col_index_union_block_ms = float(index_timing.get("index_union_block_ms", 0.0))
    else:
        if keep_block_kv_preprojected is not None:
            def _run_preprojected_keep_block_kv():
                keep_block_kv_local = keep_block_kv_preprojected.to(torch.bool).contiguous()
                if _compactattn_current_chunk_full_open():
                    keep_block_kv_local[..., past_k_blocks : past_k_blocks + curr_k_blocks] = True
                return keep_block_kv_local

            keep_block_kv, col_index_union_block_ms = _cuda_elapsed_ms(
                _run_preprojected_keep_block_kv,
                enabled=measure_timing,
            )
        else:
            keep_block_kv, col_index_union_block_ms = _cuda_elapsed_ms(
                lambda: _build_kv_head_union_keep_block_mask(
                    keep_block=keep_block,
                    q_len=q_len,
                    kv_len=kv_len,
                    block_size=block_size,
                    num_key_value_groups=num_key_value_groups,
                ),
                enabled=measure_timing,
            )

        indexed_key_states = key_states
        indexed_value_states = value_states
        indexed_attention_mask = attention_mask
        padded_tail_calls = 0.0
        padded_tail_tokens = 0.0

        can_run_indexed_dense = can_use_indexed_dense_prefill(
            q=query_states,
            k=indexed_key_states,
            v=indexed_value_states,
            keep_block_kv=keep_block_kv,
            block_size=block_size,
            page_block_size=page_block_size,
        )
        if not can_run_indexed_dense:
            padded_tail = _maybe_pad_indexed_dense_tail_inputs(
                key_states=key_states,
                value_states=value_states,
                keep_block_kv=keep_block_kv,
                attention_mask=attention_mask,
                block_size=block_size,
            )
            if padded_tail is not None:
                (
                    indexed_key_states,
                    indexed_value_states,
                    keep_block_kv,
                    indexed_attention_mask,
                    pad_tokens,
                ) = padded_tail
                padded_tail_calls = 1.0
                padded_tail_tokens = float(pad_tokens)
                can_run_indexed_dense = can_use_indexed_dense_prefill(
                    q=query_states,
                    k=indexed_key_states,
                    v=indexed_value_states,
                    keep_block_kv=keep_block_kv,
                    block_size=block_size,
                    page_block_size=page_block_size,
                )

        if not can_run_indexed_dense:
            return None, {
                    "col_index_build_ms": col_index_union_block_ms,
                    "col_index_union_block_ms": col_index_union_block_ms,
                    "col_index_block_table_ms": 0.0,
                    "col_index_table_fill_ms": 0.0,
                    "col_index_table_kernel_ms": 0.0,
                    "col_index_compact_ms": 0.0,
                    "col_index_compact_kernel_ms": 0.0,
                    "col_index_compact_nonzero_ms": 0.0,
                    "col_index_compact_post_ms": 0.0,
                    "col_index_compact_fused_calls": 0.0,
                    "col_index_compact_fallback_calls": 0.0,
                    "col_index_compact_fused_post_ms": 0.0,
                    "col_index_src_layout_ms": 0.0,
                    "col_index_cache_fill_ms": 0.0,
                    "col_index_cache_fill_kernel_ms": 0.0,
                    "col_index_cache_fill_cuda_calls": 0.0,
                    "col_index_cache_fill_cuda_fallback_calls": 0.0,
                    "col_index_cache_fill_backend_id": 0.0,
                    "col_index_cache_fill_triton_calls": 0.0,
                    "col_index_cache_fill_torch_calls": 0.0,
                    "col_index_cache_fill_fallback_calls": 0.0,
                    "col_index_cache_fill_small_calls": 0.0,
                    "col_index_cache_fill_medium_calls": 0.0,
                    "col_index_cache_fill_large_calls": 0.0,
                    "col_index_cache_fill_variant_id": 0.0,
                    "col_index_cache_fill_tuned_calls": 0.0,
                    "col_indexed_dense_kernel_ms": 0.0,
                    "dense_kernel_ms": 0.0,
                    "gather_pack_ms": 0.0,
                    "pack_prepare_ms": 0.0,
                    "gather_qkv_ms": 0.0,
                    "cu_seqlens_ms": 0.0,
                    "unpack_scatter_ms": 0.0,
                    "pack_impl_torch_calls": 0.0,
                    "pack_impl_triton_calls": 0.0,
                    "pack_impl_fallback_calls": 0.0,
                    "col_indexed_dense_calls": 0.0,
                    "col_indexed_dense_fallback_calls": 1.0,
                    "col_full_paged_kv_reuse_calls": 0.0,
                    "col_full_paged_kv_init_calls": 0.0,
                    "col_full_paged_kv_append_ms": 0.0,
                    "col_index_page_table_only_ms": 0.0,
                "col_selected_kv_materialize_calls": 0.0,
                "col_selected_kv_materialize_ms": 0.0,
                "col_fused_builder_calls": 0.0,
                "col_fused_builder_ms": 0.0,
                "col_fused_builder_table_ms": 0.0,
                "col_fused_builder_fill_ms": 0.0,
                "col_builder_peak_alloc_mb": 0.0,
                "col_builder_peak_reserved_mb": 0.0,
                    "col_direct_index_build_ms": 0.0,
                    "col_direct_kernel_ms": 0.0,
                    "col_direct_calls": 0.0,
                    "col_direct_fallback_calls": 0.0,
                    "col_indexed_impl_fa2_calls": 0.0,
                    "col_fa2_indexed_kernel_ms": 0.0,
                    "col_fa2_indexed_calls": 0.0,
                    "col_fa2_indexed_fallback_calls": 0.0,
                "col_index_padded_tail_calls": padded_tail_calls,
                "col_index_padded_tail_tokens": padded_tail_tokens,
            }

        def _run_build():
            return build_paged_kv_cache_from_block_mask(
                k=indexed_key_states.contiguous(),
                v=indexed_value_states.contiguous(),
                keep_block_kv=keep_block_kv.contiguous(),
                block_size=block_size,
                page_block_size=page_block_size,
                cache_fill_backend=cache_fill_backend,
                prefer_fused_builder_v2=False,
                measure_timing=measure_timing,
            )

        (k_cache, v_cache, block_table, cache_seqlens, index_timing), _, build_mem = _cuda_elapsed_and_peak(
            _run_build,
            device,
            measure_timing=False,
            measure_peak=peak_mem_debug,
        )
        index_timing = dict(index_timing)
        index_timing["index_padded_tail_calls"] = padded_tail_calls
        index_timing["index_padded_tail_tokens"] = padded_tail_tokens
    col_index_block_table_ms = float(index_timing.get("index_block_table_ms", 0.0))
    col_index_table_fill_ms = float(index_timing.get("index_table_fill_ms", col_index_block_table_ms))
    col_index_table_kernel_ms = float(
        index_timing.get("index_table_kernel_ms", col_index_table_fill_ms)
    )
    col_index_compact_ms = float(index_timing.get("index_compact_ms", 0.0))
    col_index_compact_kernel_ms = float(
        index_timing.get("index_compact_kernel_ms", col_index_compact_ms)
    )
    col_index_compact_nonzero_ms = float(index_timing.get("index_compact_nonzero_ms", 0.0))
    col_index_compact_post_ms = float(index_timing.get("index_compact_post_ms", 0.0))
    col_index_compact_fused_calls = float(index_timing.get("index_compact_fused_calls", 0.0))
    col_index_compact_fallback_calls = float(index_timing.get("index_compact_fallback_calls", 0.0))
    col_index_compact_fused_post_ms = float(index_timing.get("index_compact_fused_post_ms", 0.0))
    col_index_src_layout_ms = float(index_timing.get("index_src_layout_ms", 0.0))
    col_index_cache_fill_ms = float(index_timing.get("index_cache_fill_ms", 0.0))
    col_index_cache_fill_kernel_ms = float(index_timing.get("index_cache_fill_kernel_ms", 0.0))
    col_index_cache_fill_cuda_calls = float(index_timing.get("index_cache_fill_cuda_calls", 0.0))
    col_index_cache_fill_cuda_fallback_calls = float(
        index_timing.get("index_cache_fill_cuda_fallback_calls", 0.0)
    )
    col_index_cache_fill_backend_id = float(index_timing.get("index_cache_fill_backend_id", 0.0))
    col_index_cache_fill_triton_calls = float(index_timing.get("index_cache_fill_triton_calls", 0.0))
    col_index_cache_fill_torch_calls = float(index_timing.get("index_cache_fill_torch_calls", 0.0))
    col_index_cache_fill_fallback_calls = float(index_timing.get("index_cache_fill_fallback_calls", 0.0))
    col_index_cache_fill_small_calls = float(index_timing.get("index_cache_fill_small_calls", 0.0))
    col_index_cache_fill_medium_calls = float(index_timing.get("index_cache_fill_medium_calls", 0.0))
    col_index_cache_fill_large_calls = float(index_timing.get("index_cache_fill_large_calls", 0.0))
    col_index_cache_fill_variant_id = float(index_timing.get("index_cache_fill_variant_id", 0.0))
    col_index_cache_fill_tuned_calls = float(index_timing.get("index_cache_fill_tuned_calls", 0.0))
    col_index_cache_fill_launch_blocks = float(index_timing.get("index_cache_fill_launch_blocks", 0.0))
    col_index_cache_fill_effective_blocks = float(index_timing.get("index_cache_fill_effective_blocks", 0.0))
    col_index_padded_tail_calls = float(index_timing.get("index_padded_tail_calls", 0.0))
    col_index_padded_tail_tokens = float(index_timing.get("index_padded_tail_tokens", 0.0))
    col_workspace_capacity_pages = float(index_timing.get("workspace_capacity_pages", 0.0))
    col_workspace_required_pages = float(index_timing.get("workspace_required_pages", 0.0))
    col_workspace_page_pool_capacity_pages = float(
        index_timing.get("workspace_page_pool_capacity_pages", 0.0)
    )
    col_workspace_page_pool_required_pages = float(
        index_timing.get("workspace_page_pool_required_pages", 0.0)
    )
    col_workspace_total_pages_used = float(index_timing.get("workspace_total_pages_used", 0.0))
    col_workspace_max_pages_used = float(index_timing.get("workspace_max_pages_used", 0.0))
    col_workspace_growth_events = float(index_timing.get("workspace_growth_events", 0.0))
    col_workspace_k_alloc_mb = float(index_timing.get("workspace_k_alloc_mb", 0.0))
    col_workspace_v_alloc_mb = float(index_timing.get("workspace_v_alloc_mb", 0.0))
    col_workspace_block_table_alloc_mb = float(index_timing.get("workspace_block_table_alloc_mb", 0.0))
    col_index_build_peak_alloc_mb = float(build_mem.get("peak_alloc_mb", 0.0))
    col_index_build_peak_reserved_mb = float(build_mem.get("peak_reserved_mb", 0.0))
    col_index_build_alloc_delta_mb = float(build_mem.get("alloc_delta_mb", 0.0))
    col_index_build_reserved_delta_mb = float(build_mem.get("reserved_delta_mb", 0.0))
    col_scratch_pool_growth_events = float(index_timing.get("scratch_pool_growth_events", 0.0))
    col_scratch_pool_pos_alloc_mb = float(index_timing.get("scratch_pool_pos_alloc_mb", 0.0))
    col_scratch_pool_meta_alloc_mb = float(index_timing.get("scratch_pool_meta_alloc_mb", 0.0))
    col_fused_builder_calls = float(index_timing.get("fused_builder_calls", 0.0))
    col_fused_builder_ms = float(index_timing.get("fused_builder_ms", 0.0))
    col_fused_builder_table_ms = float(index_timing.get("fused_builder_table_ms", 0.0))
    col_fused_builder_fill_ms = float(index_timing.get("fused_builder_fill_ms", 0.0))
    col_fused_builder_row_tiled_calls = float(index_timing.get("fused_builder_row_tiled_calls", 0.0))
    col_fused_builder_tail_fused_calls = float(index_timing.get("fused_builder_tail_fused_calls", 0.0))
    col_selected_kv_materialize_calls = float(index_timing.get("selected_kv_materialize_calls", 0.0))
    col_selected_kv_materialize_ms = float(index_timing.get("selected_kv_materialize_ms", 0.0))
    col_builder_peak_alloc_mb = float(build_mem.get("peak_alloc_mb", 0.0))
    col_builder_peak_reserved_mb = float(build_mem.get("peak_reserved_mb", 0.0))
    col_index_build_ms = (
        col_index_union_block_ms
        + col_index_table_fill_ms
        + col_index_compact_ms
        + col_index_src_layout_ms
        + col_index_cache_fill_ms
    )

    def _run_kernel():
        if indexed_impl == "fi_paged":
            return flashinfer_indexed_prefill_from_paged_kv(
                q=query_states.contiguous(),
                k_cache=k_cache,
                v_cache=v_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                num_key_value_groups=num_key_value_groups,
                softmax_scale=softmax_scale,
                measure_timing=measure_timing,
            )
        return flash_attn_indexed_prefill_from_paged_kv(
            q=query_states.contiguous(),
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            num_key_value_groups=num_key_value_groups,
            softmax_scale=softmax_scale,
            measure_timing=measure_timing,
        )

    (out, kernel_timing), indexed_dense_kernel_ms, kernel_mem = _cuda_elapsed_and_peak(
        _run_kernel,
        device,
        measure_timing=measure_timing,
        measure_peak=peak_mem_debug,
    )
    col_index_attn_q_workspace_alloc_mb = float(kernel_timing.get("index_attn_q_workspace_alloc_mb", 0.0))
    col_index_attn_out_workspace_alloc_mb = float(kernel_timing.get("index_attn_out_workspace_alloc_mb", 0.0))
    col_index_attn_q_workspace_growth_events = float(
        kernel_timing.get("index_attn_q_workspace_growth_events", 0.0)
    )
    col_index_attn_cache_seqlens_copy_calls = float(
        kernel_timing.get("index_attn_cache_seqlens_copy_calls", 0.0)
    )
    col_index_attn_block_table_copy_calls = float(
        kernel_timing.get("index_attn_block_table_copy_calls", 0.0)
    )
    col_index_attn_q_layout_ms = float(kernel_timing.get("index_attn_q_layout_ms", 0.0))
    col_index_attn_q_layout_cuda_calls = float(kernel_timing.get("index_attn_q_layout_cuda_calls", 0.0))
    col_index_attn_out_layout_ms = float(kernel_timing.get("index_attn_out_layout_ms", 0.0))
    col_index_attn_q_layout_peak_alloc_mb = float(
        kernel_timing.get("index_attn_q_layout_peak_alloc_mb", 0.0)
    )
    col_index_attn_q_layout_peak_reserved_mb = float(
        kernel_timing.get("index_attn_q_layout_peak_reserved_mb", 0.0)
    )
    col_index_attn_flash_kvcache_peak_alloc_mb = float(
        kernel_timing.get("index_attn_flash_kvcache_peak_alloc_mb", 0.0)
    )
    col_index_attn_flash_kvcache_peak_reserved_mb = float(
        kernel_timing.get("index_attn_flash_kvcache_peak_reserved_mb", 0.0)
    )
    col_index_attn_out_layout_peak_alloc_mb = float(
        kernel_timing.get("index_attn_out_layout_peak_alloc_mb", 0.0)
    )
    col_index_attn_out_layout_peak_reserved_mb = float(
        kernel_timing.get("index_attn_out_layout_peak_reserved_mb", 0.0)
    )
    col_index_kernel_peak_alloc_mb = float(kernel_mem.get("peak_alloc_mb", 0.0))
    col_index_kernel_peak_reserved_mb = float(kernel_mem.get("peak_reserved_mb", 0.0))
    col_index_kernel_alloc_delta_mb = float(kernel_mem.get("alloc_delta_mb", 0.0))
    col_index_kernel_reserved_delta_mb = float(kernel_mem.get("reserved_delta_mb", 0.0))
    col_selected_kv_materialize_ms = (
        col_index_compact_ms
        + col_index_src_layout_ms
        + col_index_cache_fill_ms
    )
    return out, {
        "col_index_build_ms": col_index_build_ms,
        "col_index_union_block_ms": col_index_union_block_ms,
        "col_index_block_table_ms": col_index_block_table_ms,
        "col_index_table_fill_ms": col_index_table_fill_ms,
        "col_index_table_kernel_ms": col_index_table_kernel_ms,
        "col_index_compact_ms": col_index_compact_ms,
        "col_index_compact_kernel_ms": col_index_compact_kernel_ms,
        "col_index_compact_nonzero_ms": col_index_compact_nonzero_ms,
        "col_index_compact_post_ms": col_index_compact_post_ms,
        "col_index_compact_fused_calls": col_index_compact_fused_calls,
        "col_index_compact_fallback_calls": col_index_compact_fallback_calls,
        "col_index_compact_fused_post_ms": col_index_compact_fused_post_ms,
        "col_index_src_layout_ms": col_index_src_layout_ms,
        "col_index_cache_fill_ms": col_index_cache_fill_ms,
        "col_index_cache_fill_kernel_ms": col_index_cache_fill_kernel_ms,
        "col_index_cache_fill_cuda_calls": col_index_cache_fill_cuda_calls,
        "col_index_cache_fill_cuda_fallback_calls": col_index_cache_fill_cuda_fallback_calls,
        "col_index_cache_fill_backend_id": col_index_cache_fill_backend_id,
        "col_index_cache_fill_triton_calls": col_index_cache_fill_triton_calls,
        "col_index_cache_fill_torch_calls": col_index_cache_fill_torch_calls,
        "col_index_cache_fill_fallback_calls": col_index_cache_fill_fallback_calls,
        "col_index_cache_fill_small_calls": col_index_cache_fill_small_calls,
        "col_index_cache_fill_medium_calls": col_index_cache_fill_medium_calls,
        "col_index_cache_fill_large_calls": col_index_cache_fill_large_calls,
        "col_index_cache_fill_variant_id": col_index_cache_fill_variant_id,
        "col_index_cache_fill_tuned_calls": col_index_cache_fill_tuned_calls,
        "col_index_cache_fill_launch_blocks": col_index_cache_fill_launch_blocks,
        "col_index_cache_fill_effective_blocks": col_index_cache_fill_effective_blocks,
        "col_index_padded_tail_calls": col_index_padded_tail_calls,
        "col_index_padded_tail_tokens": col_index_padded_tail_tokens,
        "col_index_cache_fill_current_tail_ms": float(index_timing.get("index_cache_fill_current_tail_ms", 0.0)),
        "col_workspace_capacity_pages": col_workspace_capacity_pages,
        "col_workspace_required_pages": col_workspace_required_pages,
        "col_workspace_page_pool_capacity_pages": col_workspace_page_pool_capacity_pages,
        "col_workspace_page_pool_required_pages": col_workspace_page_pool_required_pages,
        "col_workspace_total_pages_used": col_workspace_total_pages_used,
        "col_workspace_max_pages_used": col_workspace_max_pages_used,
        "col_workspace_growth_events": col_workspace_growth_events,
        "col_workspace_k_alloc_mb": col_workspace_k_alloc_mb,
        "col_workspace_v_alloc_mb": col_workspace_v_alloc_mb,
        "col_workspace_block_table_alloc_mb": col_workspace_block_table_alloc_mb,
        "col_index_build_peak_alloc_mb": col_index_build_peak_alloc_mb,
        "col_index_build_peak_reserved_mb": col_index_build_peak_reserved_mb,
        "col_index_build_alloc_delta_mb": col_index_build_alloc_delta_mb,
        "col_index_build_reserved_delta_mb": col_index_build_reserved_delta_mb,
        "col_scratch_pool_growth_events": col_scratch_pool_growth_events,
        "col_scratch_pool_pos_alloc_mb": col_scratch_pool_pos_alloc_mb,
        "col_scratch_pool_meta_alloc_mb": col_scratch_pool_meta_alloc_mb,
        "col_index_attn_q_workspace_alloc_mb": col_index_attn_q_workspace_alloc_mb,
        "col_index_attn_out_workspace_alloc_mb": col_index_attn_out_workspace_alloc_mb,
        "col_index_attn_q_workspace_growth_events": col_index_attn_q_workspace_growth_events,
        "col_index_attn_cache_seqlens_copy_calls": col_index_attn_cache_seqlens_copy_calls,
        "col_index_attn_block_table_copy_calls": col_index_attn_block_table_copy_calls,
        "col_index_attn_q_layout_ms": col_index_attn_q_layout_ms,
        "col_index_attn_q_layout_cuda_calls": col_index_attn_q_layout_cuda_calls,
        "col_index_attn_out_layout_ms": col_index_attn_out_layout_ms,
        "col_index_attn_q_layout_peak_alloc_mb": col_index_attn_q_layout_peak_alloc_mb,
        "col_index_attn_q_layout_peak_reserved_mb": col_index_attn_q_layout_peak_reserved_mb,
        "col_index_attn_flash_kvcache_peak_alloc_mb": col_index_attn_flash_kvcache_peak_alloc_mb,
        "col_index_attn_flash_kvcache_peak_reserved_mb": col_index_attn_flash_kvcache_peak_reserved_mb,
        "col_index_attn_out_layout_peak_alloc_mb": col_index_attn_out_layout_peak_alloc_mb,
        "col_index_attn_out_layout_peak_reserved_mb": col_index_attn_out_layout_peak_reserved_mb,
        "col_index_kernel_peak_alloc_mb": col_index_kernel_peak_alloc_mb,
        "col_index_kernel_peak_reserved_mb": col_index_kernel_peak_reserved_mb,
        "col_index_kernel_alloc_delta_mb": col_index_kernel_alloc_delta_mb,
        "col_index_kernel_reserved_delta_mb": col_index_kernel_reserved_delta_mb,
        "col_indexed_dense_kernel_ms": indexed_dense_kernel_ms,
        "dense_kernel_ms": indexed_dense_kernel_ms,
        "gather_pack_ms": 0.0,
        "pack_prepare_ms": 0.0,
        "gather_qkv_ms": 0.0,
        "cu_seqlens_ms": 0.0,
        "unpack_scatter_ms": 0.0,
        "pack_impl_torch_calls": 0.0,
        "pack_impl_triton_calls": 0.0,
        "pack_impl_fallback_calls": 0.0,
        "col_indexed_dense_calls": 1.0,
        "col_indexed_dense_fallback_calls": 0.0,
        "col_full_paged_kv_reuse_calls": 0.0,
        "col_full_paged_kv_init_calls": 0.0,
        "col_full_paged_kv_append_ms": 0.0,
        "col_index_page_table_only_ms": 0.0,
        "col_selected_kv_materialize_calls": col_selected_kv_materialize_calls,
        "col_selected_kv_materialize_ms": col_selected_kv_materialize_ms,
        "col_fused_builder_calls": col_fused_builder_calls,
        "col_fused_builder_ms": col_fused_builder_ms,
        "col_fused_builder_table_ms": col_fused_builder_table_ms,
        "col_fused_builder_fill_ms": col_fused_builder_fill_ms,
        "col_fused_builder_row_tiled_calls": col_fused_builder_row_tiled_calls,
        "col_fused_builder_tail_fused_calls": col_fused_builder_tail_fused_calls,
        "col_builder_peak_alloc_mb": col_builder_peak_alloc_mb,
        "col_builder_peak_reserved_mb": col_builder_peak_reserved_mb,
        "col_direct_index_build_ms": 0.0,
        "col_direct_kernel_ms": 0.0,
        "col_direct_calls": 0.0,
        "col_direct_fallback_calls": 0.0,
        "col_indexed_impl_fa2_calls": 1.0,
        "col_fa2_indexed_kernel_ms": 0.0,
        "col_fa2_indexed_calls": 0.0,
        "col_fa2_indexed_fallback_calls": 0.0,
    }


def _run_fa2_indexed_chunked_prefill(
    query_states: torch.Tensor,  # [B, Q, Hq, D]
    key_states: torch.Tensor,  # [B, K, Hkv, D]
    value_states: torch.Tensor,  # [B, K, Hkv, D]
    keep_block: Optional[torch.Tensor],  # [B, Hq, Kb]
    keep_block_kv_preprojected: Optional[torch.Tensor],
    attention_mask: torch.Tensor,  # [B, K]
    block_size: int,
    num_key_value_groups: int,
    softmax_scale: float,
    page_block_size: int = 256,
    measure_timing: bool = True,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    q_len = query_states.shape[1]
    kv_len = key_states.shape[1]
    past_len = kv_len - q_len
    (past_block_indices, past_block_counts), col_index_build_ms = _cuda_elapsed_ms(
        lambda: (
            _build_kv_head_past_block_indices_from_kv_keep_block(
                keep_block_kv=keep_block_kv_preprojected,
                q_len=q_len,
                kv_len=kv_len,
                block_size=block_size,
            )
            if keep_block_kv_preprojected is not None
            else _build_kv_head_union_past_block_indices(
                keep_block=keep_block,
                q_len=q_len,
                kv_len=kv_len,
                block_size=block_size,
                num_key_value_groups=num_key_value_groups,
            )
        ),
        enabled=measure_timing,
    )

    if not can_use_fa2_indexed_prefill(
        q=query_states,
        k=key_states,
        v=value_states,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        block_size=block_size,
        num_key_value_groups=num_key_value_groups,
    ):
        return None, {
            "col_index_build_ms": col_index_build_ms,
            "col_index_union_block_ms": 0.0,
            "col_index_block_table_ms": 0.0,
            "col_index_table_fill_ms": 0.0,
            "col_index_table_kernel_ms": 0.0,
            "col_index_compact_ms": col_index_build_ms,
            "col_index_compact_kernel_ms": col_index_build_ms,
            "col_index_compact_nonzero_ms": 0.0,
            "col_index_compact_post_ms": 0.0,
            "col_index_src_layout_ms": 0.0,
            "col_index_cache_fill_ms": 0.0,
            "col_index_cache_fill_kernel_ms": 0.0,
            "col_index_cache_fill_triton_calls": 0.0,
            "col_index_cache_fill_torch_calls": 0.0,
            "col_index_cache_fill_fallback_calls": 0.0,
            "col_index_cache_fill_small_calls": 0.0,
            "col_index_cache_fill_medium_calls": 0.0,
            "col_index_cache_fill_large_calls": 0.0,
            "col_index_cache_fill_variant_id": 0.0,
            "col_index_cache_fill_tuned_calls": 0.0,
            "col_indexed_dense_kernel_ms": 0.0,
            "dense_kernel_ms": 0.0,
            "gather_pack_ms": 0.0,
            "pack_prepare_ms": 0.0,
            "gather_qkv_ms": 0.0,
            "cu_seqlens_ms": 0.0,
            "unpack_scatter_ms": 0.0,
            "pack_impl_torch_calls": 0.0,
            "pack_impl_triton_calls": 0.0,
            "pack_impl_fallback_calls": 0.0,
            "col_indexed_dense_calls": 0.0,
            "col_indexed_dense_fallback_calls": 1.0,
            "col_full_paged_kv_reuse_calls": 0.0,
            "col_full_paged_kv_init_calls": 0.0,
            "col_full_paged_kv_append_ms": 0.0,
            "col_index_page_table_only_ms": 0.0,
            "col_selected_kv_materialize_calls": 0.0,
            "col_selected_kv_materialize_ms": 0.0,
            "col_direct_index_build_ms": 0.0,
            "col_direct_kernel_ms": 0.0,
            "col_direct_calls": 0.0,
            "col_direct_fallback_calls": 0.0,
            "col_indexed_impl_fa2_calls": 0.0,
            "col_fa2_indexed_kernel_ms": 0.0,
            "col_fa2_indexed_calls": 0.0,
            "col_fa2_indexed_fallback_calls": 1.0,
            "col_fa2_indexed_v2_short_calls": 0.0,
            "col_fa2_indexed_v2_long_calls": 0.0,
            "col_fa2_indexed_v2_fallback_calls": 1.0,
            "col_fa2_indexed_v2_kernel_ms": 0.0,
            "col_fa2_indexed_v3_short_calls": 0.0,
            "col_fa2_indexed_v3_long_calls": 0.0,
            "col_fa2_indexed_v3_fallback_calls": 1.0,
            "col_fa2_indexed_v3_split_k": 0.0,
            "col_fa2_indexed_v3_past_kernel_ms": 0.0,
            "col_fa2_indexed_v3_reduce_ms": 0.0,
            "col_fa2_indexed_v3_current_kernel_ms": 0.0,
            "col_fa2_indexed_v3_kernel_ms": 0.0,
        }

    def _run_kernel():
        return run_fa2_indexed_prefill(
            q=query_states.contiguous(),
            k=key_states.contiguous(),
            v=value_states.contiguous(),
            past_block_indices=past_block_indices.contiguous(),
            past_block_counts=past_block_counts.contiguous(),
            past_len=past_len,
            block_size=block_size,
            num_key_value_groups=num_key_value_groups,
            softmax_scale=softmax_scale,
        )

    kernel_out, fa2_indexed_kernel_ms = _cuda_elapsed_ms(_run_kernel, enabled=measure_timing)
    out, kernel_meta = kernel_out
    impl = str(kernel_meta.get("impl", "v2_fallback"))
    variant = str(kernel_meta.get("variant", "unknown"))
    is_v3 = 1.0 if impl == "v3" else 0.0
    v2_short_calls = 1.0 if (is_v3 == 0.0 and variant == "short") else 0.0
    v2_long_calls = 1.0 if (is_v3 == 0.0 and variant == "long") else 0.0
    v3_short_calls = 1.0 if (is_v3 == 1.0 and variant == "short") else 0.0
    v3_long_calls = 1.0 if (is_v3 == 1.0 and variant == "long") else 0.0
    v3_fallback_calls = 1.0 if is_v3 == 0.0 else 0.0
    v3_split_k = float(kernel_meta.get("split_k", 0.0))
    v3_past_ms = float(kernel_meta.get("v3_past_ms", 0.0)) if is_v3 == 1.0 else 0.0
    v3_reduce_ms = float(kernel_meta.get("v3_reduce_ms", 0.0)) if is_v3 == 1.0 else 0.0
    v3_current_ms = float(kernel_meta.get("v3_current_ms", 0.0)) if is_v3 == 1.0 else 0.0
    v3_kernel_ms = float(kernel_meta.get("v3_total_ms", 0.0)) if is_v3 == 1.0 else 0.0
    v2_kernel_ms = fa2_indexed_kernel_ms if is_v3 == 0.0 else 0.0
    return out, {
        "col_index_build_ms": col_index_build_ms,
        "col_index_union_block_ms": 0.0,
        "col_index_block_table_ms": 0.0,
        "col_index_table_fill_ms": 0.0,
        "col_index_table_kernel_ms": 0.0,
        "col_index_compact_ms": col_index_build_ms,
        "col_index_compact_kernel_ms": col_index_build_ms,
        "col_index_compact_nonzero_ms": 0.0,
        "col_index_compact_post_ms": 0.0,
        "col_index_src_layout_ms": 0.0,
        "col_index_cache_fill_ms": 0.0,
        "col_index_cache_fill_triton_calls": 0.0,
        "col_index_cache_fill_torch_calls": 0.0,
        "col_index_cache_fill_fallback_calls": 0.0,
        "col_indexed_dense_kernel_ms": fa2_indexed_kernel_ms,
        "dense_kernel_ms": fa2_indexed_kernel_ms,
        "gather_pack_ms": 0.0,
        "pack_prepare_ms": 0.0,
        "gather_qkv_ms": 0.0,
        "cu_seqlens_ms": 0.0,
        "unpack_scatter_ms": 0.0,
        "pack_impl_torch_calls": 0.0,
        "pack_impl_triton_calls": 0.0,
        "pack_impl_fallback_calls": 0.0,
        "col_indexed_dense_calls": 1.0,
        "col_indexed_dense_fallback_calls": 0.0,
        "col_full_paged_kv_reuse_calls": 1.0,
        "col_full_paged_kv_init_calls": 0.0,
        "col_full_paged_kv_append_ms": 0.0,
        "col_index_page_table_only_ms": col_index_build_ms,
        "col_selected_kv_materialize_calls": 0.0,
        "col_selected_kv_materialize_ms": 0.0,
        "col_direct_index_build_ms": 0.0,
        "col_direct_kernel_ms": 0.0,
        "col_direct_calls": 0.0,
        "col_direct_fallback_calls": 0.0,
        "col_indexed_impl_fa2_calls": 0.0,
        "col_fa2_indexed_kernel_ms": fa2_indexed_kernel_ms,
        "col_fa2_indexed_calls": 1.0,
        "col_fa2_indexed_fallback_calls": 0.0,
        "col_fa2_indexed_v2_short_calls": v2_short_calls,
        "col_fa2_indexed_v2_long_calls": v2_long_calls,
        "col_fa2_indexed_v2_fallback_calls": v3_fallback_calls,
        "col_fa2_indexed_v2_kernel_ms": v2_kernel_ms,
        "col_fa2_indexed_v3_short_calls": v3_short_calls,
        "col_fa2_indexed_v3_long_calls": v3_long_calls,
        "col_fa2_indexed_v3_fallback_calls": v3_fallback_calls,
        "col_fa2_indexed_v3_split_k": v3_split_k,
        "col_fa2_indexed_v3_past_kernel_ms": v3_past_ms,
        "col_fa2_indexed_v3_reduce_ms": v3_reduce_ms,
        "col_fa2_indexed_v3_current_kernel_ms": v3_current_ms,
        "col_fa2_indexed_v3_kernel_ms": v3_kernel_ms,
    }


def _run_indexed_dense_chunked_prefill_direct(
    query_states: torch.Tensor,  # [B, Q, Hq, D]
    key_states: torch.Tensor,  # [B, K, Hkv, D]
    value_states: torch.Tensor,  # [B, K, Hkv, D]
    keep_block: Optional[torch.Tensor],  # [B, Hq, Kb]
    keep_block_kv_preprojected: Optional[torch.Tensor],
    block_size: int,
    num_key_value_groups: int,
    softmax_scale: float,
    measure_timing: bool = True,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    q_len = query_states.shape[1]
    kv_len = key_states.shape[1]
    past_len = kv_len - q_len

    (past_block_indices, past_block_counts), col_direct_index_build_ms = _cuda_elapsed_ms(
        lambda: (
            _build_kv_head_past_block_indices_from_kv_keep_block(
                keep_block_kv=keep_block_kv_preprojected,
                q_len=q_len,
                kv_len=kv_len,
                block_size=block_size,
            )
            if keep_block_kv_preprojected is not None
            else _build_kv_head_union_past_block_indices(
                keep_block=keep_block,
                q_len=q_len,
                kv_len=kv_len,
                block_size=block_size,
                num_key_value_groups=num_key_value_groups,
            )
        ),
        enabled=measure_timing,
    )

    if not can_use_direct_indexed_prefill(
        q=query_states,
        k=key_states,
        v=value_states,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        block_size=block_size,
        num_key_value_groups=num_key_value_groups,
    ):
        return None, {
            "col_index_build_ms": col_direct_index_build_ms,
            "col_index_union_block_ms": 0.0,
            "col_index_block_table_ms": 0.0,
            "col_index_table_fill_ms": 0.0,
            "col_index_table_kernel_ms": 0.0,
            "col_index_compact_ms": 0.0,
            "col_index_compact_kernel_ms": 0.0,
            "col_index_compact_nonzero_ms": 0.0,
            "col_index_compact_post_ms": 0.0,
            "col_index_src_layout_ms": 0.0,
            "col_index_cache_fill_ms": 0.0,
            "col_index_cache_fill_triton_calls": 0.0,
            "col_index_cache_fill_torch_calls": 0.0,
            "col_index_cache_fill_fallback_calls": 0.0,
            "col_indexed_dense_kernel_ms": 0.0,
            "dense_kernel_ms": 0.0,
            "gather_pack_ms": 0.0,
            "pack_prepare_ms": 0.0,
            "gather_qkv_ms": 0.0,
            "cu_seqlens_ms": 0.0,
            "unpack_scatter_ms": 0.0,
            "pack_impl_torch_calls": 0.0,
            "pack_impl_triton_calls": 0.0,
            "pack_impl_fallback_calls": 0.0,
            "col_indexed_dense_calls": 0.0,
            "col_indexed_dense_fallback_calls": 1.0,
            "col_full_paged_kv_reuse_calls": 0.0,
            "col_full_paged_kv_init_calls": 0.0,
            "col_full_paged_kv_append_ms": 0.0,
            "col_index_page_table_only_ms": 0.0,
            "col_selected_kv_materialize_calls": 0.0,
            "col_selected_kv_materialize_ms": 0.0,
            "col_direct_index_build_ms": col_direct_index_build_ms,
            "col_direct_kernel_ms": 0.0,
            "col_direct_calls": 0.0,
            "col_direct_fallback_calls": 1.0,
            "col_indexed_impl_fa2_calls": 0.0,
            "col_fa2_indexed_kernel_ms": 0.0,
            "col_fa2_indexed_calls": 0.0,
            "col_fa2_indexed_fallback_calls": 0.0,
        }

    def _run_kernel():
        return run_direct_indexed_prefill(
            q=query_states.contiguous(),
            k=key_states.contiguous(),
            v=value_states.contiguous(),
            past_block_indices=past_block_indices.contiguous(),
            past_block_counts=past_block_counts.contiguous(),
            past_len=past_len,
            block_size=block_size,
            num_key_value_groups=num_key_value_groups,
            softmax_scale=softmax_scale,
        )

    out, col_direct_kernel_ms = _cuda_elapsed_ms(_run_kernel, enabled=measure_timing)
    return out, {
        "col_index_build_ms": col_direct_index_build_ms,
        "col_index_union_block_ms": 0.0,
        "col_index_block_table_ms": 0.0,
        "col_index_table_fill_ms": 0.0,
        "col_index_table_kernel_ms": 0.0,
        "col_index_compact_ms": 0.0,
        "col_index_compact_kernel_ms": 0.0,
        "col_index_compact_nonzero_ms": 0.0,
        "col_index_compact_post_ms": 0.0,
        "col_index_src_layout_ms": 0.0,
        "col_index_cache_fill_ms": 0.0,
        "col_index_cache_fill_triton_calls": 0.0,
        "col_index_cache_fill_torch_calls": 0.0,
        "col_index_cache_fill_fallback_calls": 0.0,
        "col_indexed_dense_kernel_ms": col_direct_kernel_ms,
        "dense_kernel_ms": col_direct_kernel_ms,
        "gather_pack_ms": 0.0,
        "pack_prepare_ms": 0.0,
        "gather_qkv_ms": 0.0,
        "cu_seqlens_ms": 0.0,
        "unpack_scatter_ms": 0.0,
        "pack_impl_torch_calls": 0.0,
        "pack_impl_triton_calls": 0.0,
        "pack_impl_fallback_calls": 0.0,
        "col_indexed_dense_calls": 1.0,
        "col_indexed_dense_fallback_calls": 0.0,
        "col_full_paged_kv_reuse_calls": 1.0,
        "col_full_paged_kv_init_calls": 0.0,
        "col_full_paged_kv_append_ms": 0.0,
        "col_index_page_table_only_ms": col_direct_index_build_ms,
        "col_selected_kv_materialize_calls": 0.0,
        "col_selected_kv_materialize_ms": 0.0,
        "col_direct_index_build_ms": col_direct_index_build_ms,
        "col_direct_kernel_ms": col_direct_kernel_ms,
        "col_direct_calls": 1.0,
        "col_direct_fallback_calls": 0.0,
        "col_indexed_impl_fa2_calls": 0.0,
        "col_fa2_indexed_kernel_ms": 0.0,
        "col_fa2_indexed_calls": 0.0,
        "col_fa2_indexed_fallback_calls": 0.0,
    }


def _pack_selected_dense_inputs(
    query_states: torch.Tensor,  # [B, Q, H, D]
    key_states: torch.Tensor,  # [B, K, H, D]
    value_states: torch.Tensor,  # [B, K, H, D]
    keep_token_mask: torch.Tensor,  # [B, H, K]
    attention_mask: torch.Tensor,  # [B, K]
    measure_timing: bool = True,
) -> Tuple[Optional[Tuple[torch.Tensor, ...]], float]:
    bsz, q_len, num_heads, head_dim = query_states.shape
    device = query_states.device

    def _run_pack_prepare():
        # Flatten (B, H) into a ragged batch dimension to avoid Python per-(b,h) loops.
        bh = bsz * num_heads
        q_by_head = query_states.permute(0, 2, 1, 3).contiguous()  # [B, H, Q, D]
        k_by_head = key_states.permute(0, 2, 1, 3).contiguous()  # [B, H, K, D]
        v_by_head = value_states.permute(0, 2, 1, 3).contiguous()  # [B, H, K, D]
        q_flat = q_by_head.reshape(bh, q_len, head_dim)
        k_flat = k_by_head.reshape(bh, k_by_head.shape[2], head_dim)
        v_flat = v_by_head.reshape(bh, v_by_head.shape[2], head_dim)

        q_mask_bh = (
            attention_mask[:, -q_len:]
            .unsqueeze(1)
            .expand(bsz, num_heads, q_len)
            .reshape(bh, q_len)
        )
        k_mask_bh = (keep_token_mask & attention_mask.unsqueeze(1)).reshape(bh, -1)

        k_len_bh = k_mask_bh.sum(dim=-1, dtype=torch.int64)
        zero_k_rows = k_len_bh == 0
        if zero_k_rows.any():
            valid_k_per_b = attention_mask.to(torch.bool)
            has_valid_k_per_b = valid_k_per_b.any(dim=-1)
            b_idx = torch.arange(bh, device=device, dtype=torch.long) // num_heads
            fix_rows = zero_k_rows & has_valid_k_per_b[b_idx]
            if fix_rows.any():
                # Last valid token per batch row for safety keep.
                last_valid_k_per_b = valid_k_per_b.to(torch.int64).shape[-1] - 1 - torch.argmax(
                    valid_k_per_b.to(torch.int64).flip(dims=[-1]), dim=-1
                )
                k_mask_bh[fix_rows, last_valid_k_per_b[b_idx[fix_rows]]] = True
                k_len_bh = k_mask_bh.sum(dim=-1, dtype=torch.int64)

        q_len_bh = q_mask_bh.sum(dim=-1, dtype=torch.int64)
        active_rows = (q_len_bh > 0) & (k_len_bh > 0)

        if not active_rows.any():
            return None

        q_flat_active = q_flat[active_rows]  # [BHa, Q, D]
        k_flat_active = k_flat[active_rows]  # [BHa, K, D]
        v_flat_active = v_flat[active_rows]  # [BHa, K, D]
        q_mask_active = q_mask_bh[active_rows]  # [BHa, Q]
        k_mask_active = k_mask_bh[active_rows]  # [BHa, K]
        q_len_active = q_len_bh[active_rows].to(torch.int32)
        k_len_active = k_len_bh[active_rows].to(torch.int32)
        return (
            bh,
            q_flat_active,
            k_flat_active,
            v_flat_active,
            q_mask_active,
            k_mask_active,
            q_len_active,
            k_len_active,
            active_rows,
        )

    return _cuda_elapsed_ms(_run_pack_prepare, enabled=measure_timing)


def _pack_and_run_selected_dense_torch(
    query_states: torch.Tensor,  # [B, Q, H, D]
    key_states: torch.Tensor,  # [B, K, H, D]
    value_states: torch.Tensor,  # [B, K, H, D]
    keep_token_mask: torch.Tensor,  # [B, H, K]
    attention_mask: torch.Tensor,  # [B, K]
    softmax_scale: float,
    measure_timing: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz, q_len, num_heads, head_dim = query_states.shape
    device = query_states.device
    pack_out, pack_prepare_ms = _pack_selected_dense_inputs(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        keep_token_mask=keep_token_mask,
        attention_mask=attention_mask,
        measure_timing=measure_timing,
    )
    if pack_out is None:
        out_zero = torch.zeros((bsz, q_len, num_heads, head_dim), device=device, dtype=query_states.dtype)
        return out_zero, {
            "pack_prepare_ms": pack_prepare_ms,
            "gather_qkv_ms": 0.0,
            "cu_seqlens_ms": 0.0,
            "dense_kernel_ms": 0.0,
            "unpack_scatter_ms": 0.0,
            "gather_pack_ms": pack_prepare_ms,
            "pack_impl_torch_calls": 1.0,
            "pack_impl_triton_calls": 0.0,
            "pack_impl_fallback_calls": 0.0,
        }

    (
        bh,
        q_flat_active,
        k_flat_active,
        v_flat_active,
        q_mask_active,
        k_mask_active,
        q_len_active,
        k_len_active,
        active_rows,
    ) = pack_out

    def _run_gather_qkv():
        # Gather ragged Q/K/V without host-side loops.
        q_pos = torch.nonzero(q_mask_active, as_tuple=False)
        k_pos = torch.nonzero(k_mask_active, as_tuple=False)
        q_row, q_tok = q_pos[:, 0], q_pos[:, 1]
        k_row, k_tok = k_pos[:, 0], k_pos[:, 1]
        q_cat = q_flat_active[q_row, q_tok].unsqueeze(1).contiguous()  # [sum_q, 1, D]
        k_cat = k_flat_active[k_row, k_tok].unsqueeze(1).contiguous()  # [sum_k, 1, D]
        v_cat = v_flat_active[k_row, k_tok].unsqueeze(1).contiguous()  # [sum_k, 1, D]
        return q_row, q_tok, q_cat, k_cat, v_cat

    gather_out, gather_qkv_ms = _cuda_elapsed_ms(_run_gather_qkv, enabled=measure_timing)
    q_row, q_tok, q_cat, k_cat, v_cat = gather_out

    def _run_cu_seqlens():
        cu_q = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(q_len_active, dim=0, dtype=torch.int32)]
        )
        cu_k = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(k_len_active, dim=0, dtype=torch.int32)]
        )
        max_seqlen_q = int(q_len_active.max().item())
        max_seqlen_k = int(k_len_active.max().item())
        return cu_q, cu_k, max_seqlen_q, max_seqlen_k

    cu_out, cu_seqlens_ms = _cuda_elapsed_ms(_run_cu_seqlens, enabled=measure_timing)
    cu_q, cu_k, max_seqlen_q, max_seqlen_k = cu_out

    def _run_kernel():
        return flash_attn_varlen_func(
            q_cat,
            k_cat,
            v_cat,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
        )

    out_unpad, dense_kernel_ms = _cuda_elapsed_ms(_run_kernel, enabled=measure_timing)

    def _run_unpack_scatter():
        out_flat_active = torch.zeros(
            (q_flat_active.shape[0], q_len, head_dim), device=device, dtype=query_states.dtype
        )
        out_flat_active[q_row, q_tok, :] = out_unpad[:, 0, :]
        out_flat = torch.zeros((bh, q_len, head_dim), device=device, dtype=query_states.dtype)
        out_flat[active_rows] = out_flat_active
        return out_flat.view(bsz, num_heads, q_len, head_dim).permute(0, 2, 1, 3).contiguous()

    out, unpack_scatter_ms = _cuda_elapsed_ms(_run_unpack_scatter, enabled=measure_timing)
    gather_pack_ms = pack_prepare_ms + gather_qkv_ms + cu_seqlens_ms + unpack_scatter_ms
    return out, {
        "pack_prepare_ms": pack_prepare_ms,
        "gather_qkv_ms": gather_qkv_ms,
        "cu_seqlens_ms": cu_seqlens_ms,
        "dense_kernel_ms": dense_kernel_ms,
        "unpack_scatter_ms": unpack_scatter_ms,
        "gather_pack_ms": gather_pack_ms,
        "pack_impl_torch_calls": 1.0,
        "pack_impl_triton_calls": 0.0,
        "pack_impl_fallback_calls": 0.0,
    }


def _pack_and_run_selected_dense_triton(
    query_states: torch.Tensor,  # [B, Q, H, D]
    key_states: torch.Tensor,  # [B, K, H, D]
    value_states: torch.Tensor,  # [B, K, H, D]
    keep_token_mask: torch.Tensor,  # [B, H, K]
    attention_mask: torch.Tensor,  # [B, K]
    softmax_scale: float,
    measure_timing: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz, q_len, num_heads, head_dim = query_states.shape
    device = query_states.device
    pack_out, pack_prepare_ms = _pack_selected_dense_inputs(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        keep_token_mask=keep_token_mask,
        attention_mask=attention_mask,
        measure_timing=measure_timing,
    )
    if pack_out is None:
        out_zero = torch.zeros((bsz, q_len, num_heads, head_dim), device=device, dtype=query_states.dtype)
        return out_zero, {
            "pack_prepare_ms": pack_prepare_ms,
            "gather_qkv_ms": 0.0,
            "cu_seqlens_ms": 0.0,
            "dense_kernel_ms": 0.0,
            "unpack_scatter_ms": 0.0,
            "gather_pack_ms": pack_prepare_ms,
            "pack_impl_torch_calls": 0.0,
            "pack_impl_triton_calls": 1.0,
            "pack_impl_fallback_calls": 0.0,
        }

    (
        bh,
        q_flat_active,
        k_flat_active,
        v_flat_active,
        q_mask_active,
        k_mask_active,
        q_len_active,
        k_len_active,
        active_rows,
    ) = pack_out

    def _run_gather_qkv():
        q_pos = torch.nonzero(q_mask_active, as_tuple=False)
        k_pos = torch.nonzero(k_mask_active, as_tuple=False)
        q_row = q_pos[:, 0].contiguous()
        q_tok = q_pos[:, 1].contiguous()
        k_row = k_pos[:, 0].contiguous()
        k_tok = k_pos[:, 1].contiguous()

        q_cat = torch.empty((q_row.numel(), 1, head_dim), device=device, dtype=query_states.dtype)
        k_cat = torch.empty((k_row.numel(), 1, head_dim), device=device, dtype=query_states.dtype)
        v_cat = torch.empty((k_row.numel(), 1, head_dim), device=device, dtype=query_states.dtype)

        use_triton = (
            triton_available()
            and can_use_triton_pack(q_flat_active, q_row, q_tok)
            and can_use_triton_pack(k_flat_active, k_row, k_tok)
            and can_use_triton_pack(v_flat_active, k_row, k_tok)
        )
        fallback_used = 0.0
        if use_triton:
            gather_3d_to_cat(q_flat_active, q_row, q_tok, q_cat)
            gather_3d_to_cat(k_flat_active, k_row, k_tok, k_cat)
            gather_3d_to_cat(v_flat_active, k_row, k_tok, v_cat)
        else:
            q_cat.copy_(q_flat_active[q_row, q_tok].unsqueeze(1).contiguous())
            k_cat.copy_(k_flat_active[k_row, k_tok].unsqueeze(1).contiguous())
            v_cat.copy_(v_flat_active[k_row, k_tok].unsqueeze(1).contiguous())
            fallback_used = 1.0
        return q_row, q_tok, q_cat, k_cat, v_cat, use_triton, fallback_used

    gather_out, gather_qkv_ms = _cuda_elapsed_ms(_run_gather_qkv, enabled=measure_timing)
    q_row, q_tok, q_cat, k_cat, v_cat, use_triton, fallback_used = gather_out

    def _run_cu_seqlens():
        cu_q = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(q_len_active, dim=0, dtype=torch.int32)]
        )
        cu_k = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(k_len_active, dim=0, dtype=torch.int32)]
        )
        max_seqlen_q = int(q_len_active.max().item())
        max_seqlen_k = int(k_len_active.max().item())
        return cu_q, cu_k, max_seqlen_q, max_seqlen_k

    cu_out, cu_seqlens_ms = _cuda_elapsed_ms(_run_cu_seqlens, enabled=measure_timing)
    cu_q, cu_k, max_seqlen_q, max_seqlen_k = cu_out

    def _run_kernel():
        return flash_attn_varlen_func(
            q_cat,
            k_cat,
            v_cat,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
        )

    out_unpad, dense_kernel_ms = _cuda_elapsed_ms(_run_kernel, enabled=measure_timing)

    def _run_unpack_scatter():
        out_flat_active = torch.zeros(
            (q_flat_active.shape[0], q_len, head_dim), device=device, dtype=query_states.dtype
        )
        use_triton_scatter = use_triton and can_use_triton_scatter(out_unpad, q_row, q_tok, out_flat_active)
        fallback_scatter = 0.0
        if use_triton_scatter:
            scatter_cat_to_3d(out_unpad, q_row, q_tok, out_flat_active)
        else:
            out_flat_active[q_row, q_tok, :] = out_unpad[:, 0, :]
            fallback_scatter = 1.0
        out_flat = torch.zeros((bh, q_len, head_dim), device=device, dtype=query_states.dtype)
        out_flat[active_rows] = out_flat_active
        out = out_flat.view(bsz, num_heads, q_len, head_dim).permute(0, 2, 1, 3).contiguous()
        return out, fallback_scatter

    unpack_out, unpack_scatter_ms = _cuda_elapsed_ms(_run_unpack_scatter, enabled=measure_timing)
    out, fallback_scatter = unpack_out
    gather_pack_ms = pack_prepare_ms + gather_qkv_ms + cu_seqlens_ms + unpack_scatter_ms
    return out, {
        "pack_prepare_ms": pack_prepare_ms,
        "gather_qkv_ms": gather_qkv_ms,
        "cu_seqlens_ms": cu_seqlens_ms,
        "dense_kernel_ms": dense_kernel_ms,
        "unpack_scatter_ms": unpack_scatter_ms,
        "gather_pack_ms": gather_pack_ms,
        "pack_impl_torch_calls": 0.0,
        "pack_impl_triton_calls": 1.0,
        "pack_impl_fallback_calls": float(max(fallback_used, fallback_scatter)),
    }


def _pack_and_run_selected_dense(
    query_states: torch.Tensor,  # [B, Q, H, D]
    key_states: torch.Tensor,  # [B, K, H, D]
    value_states: torch.Tensor,  # [B, K, H, D]
    keep_token_mask: torch.Tensor,  # [B, H, K]
    attention_mask: torch.Tensor,  # [B, K]
    softmax_scale: float,
    pack_impl: str = "torch",
    measure_timing: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if pack_impl == "torch":
        return _pack_and_run_selected_dense_torch(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            keep_token_mask=keep_token_mask,
            attention_mask=attention_mask,
            softmax_scale=softmax_scale,
            measure_timing=measure_timing,
        )
    if pack_impl == "triton":
        if triton_available():
            return _pack_and_run_selected_dense_triton(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                keep_token_mask=keep_token_mask,
                attention_mask=attention_mask,
                softmax_scale=softmax_scale,
                measure_timing=measure_timing,
            )
        out, stats = _pack_and_run_selected_dense_torch(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            keep_token_mask=keep_token_mask,
            attention_mask=attention_mask,
            softmax_scale=softmax_scale,
            measure_timing=measure_timing,
        )
        stats["pack_impl_torch_calls"] = 1.0
        stats["pack_impl_triton_calls"] = 0.0
        stats["pack_impl_fallback_calls"] = 1.0
        return out, stats
    raise ValueError(f"Unsupported pack_impl: {pack_impl}")


def chunked_prefill_column_dense_attention_from_keep_block(
    query_states: torch.Tensor,  # [B, Q, Hq, D]
    key_states: torch.Tensor,  # [B, K, Hkv, D]
    value_states: torch.Tensor,  # [B, K, Hkv, D]
    keep_block: Optional[torch.Tensor],  # [B, Hq, Kb]
    attention_mask: Optional[torch.Tensor],  # [B, K]
    query_length: int,
    softmax_scale: float,
    block_size: int = 64,
    num_key_value_groups: int = 1,
    pack_impl: str = "torch",
    indexed_impl: str = "fa2_paged",
    cache_fill_backend: str = "auto",
    return_stats: bool = False,
    valid_block: Optional[torch.Tensor] = None,  # [B, Hq, Kb]
    allow_indexed_dense: bool = False,
    keep_block_kv_preprojected: Optional[torch.Tensor] = None,  # [B, Hkv, Kb]
    attn_module=None,
    k_hf: Optional[torch.Tensor] = None,  # [B, Hkv, K, D] heads-first, for fi_zero_copy
    v_hf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    if query_length <= 1:
        raise ValueError("chunked_prefill_column_dense_attention_from_keep_block expects query_length > 1.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if indexed_impl not in {"fa2_paged", "triton_direct", "fa2_indexed", "fi_paged", "fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"}:
        raise ValueError(f"Unsupported indexed_impl: {indexed_impl}")

    bsz, q_len, num_q_heads, _ = query_states.shape
    kv_len = key_states.shape[1]
    kv_blocks = (kv_len + block_size - 1) // block_size
    num_kv_heads = key_states.shape[2]
    if keep_block is None and keep_block_kv_preprojected is None:
        raise ValueError("Either keep_block or keep_block_kv_preprojected must be provided.")
    if keep_block is not None and keep_block.shape != (bsz, num_q_heads, kv_blocks):
        raise ValueError(
            f"keep_block shape mismatch: expected {(bsz, num_q_heads, kv_blocks)}, got {tuple(keep_block.shape)}"
        )
    if indexed_impl == "fi_zero_copy_subgroup":
        query_subgroup_size = int(os.environ.get("SEER_ZC_QUERY_SUBGROUP_SIZE", "4"))
        if num_key_value_groups % query_subgroup_size != 0:
            raise ValueError(
                "SEER_ZC_QUERY_SUBGROUP_SIZE must divide num_key_value_groups: "
                f"{query_subgroup_size} vs {num_key_value_groups}"
            )
        expected_kv_projected_shape = (
            bsz,
            num_kv_heads * (num_key_value_groups // query_subgroup_size),
            kv_blocks,
        )
    else:
        expected_kv_projected_shape = (bsz, num_kv_heads, kv_blocks)
    if keep_block_kv_preprojected is not None and keep_block_kv_preprojected.shape != expected_kv_projected_shape:
        raise ValueError(
            "keep_block_kv_preprojected shape mismatch: "
            f"expected {expected_kv_projected_shape}, got {tuple(keep_block_kv_preprojected.shape)}"
        )
    if valid_block is not None:
        if keep_block is None:
            raise ValueError("valid_block requires keep_block to be provided.")
        if valid_block.shape != keep_block.shape:
            raise ValueError(
                f"valid_block shape mismatch: expected {tuple(keep_block.shape)}, got {tuple(valid_block.shape)}"
            )

    device = query_states.device
    measure_timing = bool(return_stats)
    (
        attention_mask,
        keep_block_expanded,
        valid_block,
        keep_block_kv_preprojected,
    ), col_input_sanitize_ms = _cuda_elapsed_ms(
        lambda: (
            _ensure_attention_mask(attention_mask, bsz, kv_len, device),
            None if keep_block is None else keep_block.to(torch.bool).contiguous(),
            None if valid_block is None else valid_block.to(torch.bool).contiguous(),
            None
            if keep_block_kv_preprojected is None
            else keep_block_kv_preprojected.to(torch.bool).contiguous(),
        ),
        enabled=measure_timing,
    )

    if return_stats:
        def _compute_select_ratio():
            if keep_block_kv_preprojected is not None:
                selected_blocks_local = float(keep_block_kv_preprojected.sum().item())
                valid_blocks_local = float(keep_block_kv_preprojected.numel())
            elif keep_block_expanded is None:
                selected_blocks_local = 0.0
                valid_blocks_local = 0.0
            else:
                selected_blocks_local = float(keep_block_expanded.sum().item())
                if valid_block is None:
                    valid_blocks_local = float(keep_block_expanded.numel())
                else:
                    valid_blocks_local = float(valid_block.sum().item())
            select_ratio_local = selected_blocks_local / valid_blocks_local if valid_blocks_local > 0 else 1.0
            return selected_blocks_local, valid_blocks_local, select_ratio_local

        (selected_blocks, valid_blocks, select_ratio), col_select_ratio_ms = _cuda_elapsed_ms(
            _compute_select_ratio,
            enabled=measure_timing,
        )
    else:
        selected_blocks = 0.0
        valid_blocks = 0.0
        select_ratio = 0.0
        col_select_ratio_ms = 0.0

    use_indexed_dense = False
    indexed_dense_fallback_calls = 0.0
    tail_dense_fallback_calls = 0.0
    direct_impl_fallback_calls = 0.0
    col_repeat_kv_ms = 0.0
    col_keep_expand_ms = 0.0
    pack_stats: Dict[str, float]

    if pack_impl == "indexed_dense":
        can_attempt_indexed = (
            allow_indexed_dense
            and indexed_dense_available()
            and query_states.dtype in (torch.float16, torch.bfloat16)
            and key_states.dtype == query_states.dtype
            and value_states.dtype == query_states.dtype
            and query_states.shape[-1] == 128
            and query_states.is_cuda
            and key_states.is_cuda
            and value_states.is_cuda
        )
        if can_attempt_indexed:
            if indexed_impl == "triton_direct":
                if direct_indexed_prefill_available():
                    direct_out, direct_stats = _run_indexed_dense_chunked_prefill_direct(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        keep_block=keep_block_expanded,
                        keep_block_kv_preprojected=keep_block_kv_preprojected,
                        block_size=block_size,
                        num_key_value_groups=num_key_value_groups,
                        softmax_scale=softmax_scale,
                        measure_timing=measure_timing,
                    )
                    if direct_out is not None:
                        out = direct_out
                        pack_stats = direct_stats
                        use_indexed_dense = True
                    else:
                        direct_impl_fallback_calls += float(
                            direct_stats.get("col_direct_fallback_calls", 1.0)
                        )
                else:
                    direct_impl_fallback_calls += 1.0
            elif indexed_impl == "fa2_indexed":
                if fa2_indexed_available():
                    fa2_out, fa2_stats = _run_fa2_indexed_chunked_prefill(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        keep_block=keep_block_expanded,
                        keep_block_kv_preprojected=keep_block_kv_preprojected,
                        attention_mask=attention_mask,
                        block_size=block_size,
                        num_key_value_groups=num_key_value_groups,
                        softmax_scale=softmax_scale,
                        measure_timing=measure_timing,
                    )
                    if fa2_out is not None:
                        out = fa2_out
                        pack_stats = fa2_stats
                        use_indexed_dense = True
                    else:
                        indexed_dense_fallback_calls += float(
                            fa2_stats.get("col_fa2_indexed_fallback_calls", 1.0)
                        )
                else:
                    indexed_dense_fallback_calls += 1.0
            elif _compactattn_enable_full_kv_reuse_path():
                if fa2_indexed_available():
                    fa2_out, fa2_stats = _run_fa2_indexed_chunked_prefill(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        keep_block=keep_block_expanded,
                        keep_block_kv_preprojected=keep_block_kv_preprojected,
                        attention_mask=attention_mask,
                        block_size=block_size,
                        num_key_value_groups=num_key_value_groups,
                        softmax_scale=softmax_scale,
                        measure_timing=measure_timing,
                    )
                    if fa2_out is not None:
                        out = fa2_out
                        pack_stats = fa2_stats
                        use_indexed_dense = True
                    else:
                        indexed_dense_fallback_calls += float(
                            fa2_stats.get("col_fa2_indexed_fallback_calls", 1.0)
                        )
                if not use_indexed_dense and direct_indexed_prefill_available():
                    direct_out, direct_stats = _run_indexed_dense_chunked_prefill_direct(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        keep_block=keep_block_expanded,
                        keep_block_kv_preprojected=keep_block_kv_preprojected,
                        block_size=block_size,
                        num_key_value_groups=num_key_value_groups,
                        softmax_scale=softmax_scale,
                        measure_timing=measure_timing,
                    )
                    if direct_out is not None:
                        out = direct_out
                        pack_stats = direct_stats
                        use_indexed_dense = True
                    else:
                        direct_impl_fallback_calls += float(
                            direct_stats.get("col_direct_fallback_calls", 1.0)
                        )
            if not use_indexed_dense:
                indexed_out, indexed_stats = _run_indexed_dense_chunked_prefill(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    keep_block=keep_block_expanded,
                    keep_block_kv_preprojected=keep_block_kv_preprojected,
                    attention_mask=attention_mask,
                    block_size=block_size,
                    num_key_value_groups=num_key_value_groups,
                    softmax_scale=softmax_scale,
                    cache_fill_backend=cache_fill_backend,
                    measure_timing=measure_timing,
                    indexed_impl=indexed_impl,
                    k_hf=k_hf,
                    v_hf=v_hf,
                )
                if indexed_out is not None:
                    out = indexed_out
                    pack_stats = indexed_stats
                    use_indexed_dense = True
                else:
                    indexed_dense_fallback_calls += float(
                        indexed_stats.get("col_indexed_dense_fallback_calls", 1.0)
                    )
        else:
            indexed_dense_fallback_calls = 1.0

    use_dense_for_indexed_misaligned_tail = (
        pack_impl == "indexed_dense"
        and not use_indexed_dense
        and indexed_dense_fallback_calls > 0.0
        and allow_indexed_dense
        and indexed_dense_available()
        and query_states.dtype in (torch.float16, torch.bfloat16)
        and key_states.dtype == query_states.dtype
        and value_states.dtype == query_states.dtype
        and query_states.shape[-1] == 128
        and query_states.is_cuda
        and key_states.is_cuda
        and value_states.is_cuda
        and (kv_len % block_size) != 0
    )

    if use_dense_for_indexed_misaligned_tail:
        out, dense_tail_stats = _dense_prefill_full_kv(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            softmax_scale=softmax_scale,
            num_key_value_groups=num_key_value_groups,
            fallback_used=1.0,
            measure_timing=measure_timing,
            attn_module=attn_module,
        )
        pack_stats = {
            "pack_prepare_ms": 0.0,
            "gather_qkv_ms": 0.0,
            "cu_seqlens_ms": 0.0,
            "unpack_scatter_ms": 0.0,
            "pack_impl_torch_calls": 0.0,
            "pack_impl_triton_calls": 0.0,
            "pack_impl_fallback_calls": 0.0,
        }
        pack_stats.update(dense_tail_stats)
        tail_dense_fallback_calls = 1.0
        use_indexed_dense = True

    if not use_indexed_dense:
        if keep_block_expanded is None:
            keep_block_expanded = keep_block_kv_preprojected.repeat_interleave(num_key_value_groups, dim=1)
        keep_token, col_keep_expand_ms_0 = _cuda_elapsed_ms(
            lambda: _expand_blocks_to_tokens(keep_block_expanded, block_size=block_size, kv_len=kv_len),
            enabled=measure_timing,
        )

        query_valid = attention_mask[:, -q_len:]
        current_chunk_full_open = _compactattn_current_chunk_full_open()

        def _run_keep_expand_post():
            keep_token_local = keep_token
            if current_chunk_full_open:
                keep_token_local[:, :, kv_len - q_len : kv_len] |= query_valid.unsqueeze(1)
            keep_token_local &= attention_mask.unsqueeze(1)
            return keep_token_local

        keep_token, col_keep_expand_ms_1 = _cuda_elapsed_ms(
            _run_keep_expand_post, enabled=measure_timing
        )
        col_keep_expand_ms = col_keep_expand_ms_0 + col_keep_expand_ms_1

        def _run_repeat_kv():
            key_states_full = repeat_kv(
                key_states.transpose(1, 2).contiguous(), num_key_value_groups
            ).transpose(1, 2).contiguous()
            value_states_full = repeat_kv(
                value_states.transpose(1, 2).contiguous(), num_key_value_groups
            ).transpose(1, 2).contiguous()
            return key_states_full, value_states_full

        (key_states_full, value_states_full), col_repeat_kv_ms = _cuda_elapsed_ms(
            _run_repeat_kv, enabled=measure_timing
        )

        out, pack_stats = _pack_and_run_selected_dense(
            query_states=query_states,
            key_states=key_states_full,
            value_states=value_states_full,
            keep_token_mask=keep_token,
            attention_mask=attention_mask,
            softmax_scale=softmax_scale,
            pack_impl=pack_impl if pack_impl in {"torch", "triton"} else "torch",
            measure_timing=measure_timing,
        )
        if pack_impl == "indexed_dense":
            indexed_dense_fallback_calls = 1.0

        if return_stats:
            selected_tokens = float(keep_token.sum().item())
            valid_tokens = float((attention_mask.unsqueeze(1).expand(-1, num_q_heads, -1)).sum().item())
        else:
            selected_tokens = 0.0
            valid_tokens = 0.0
    else:
        if return_stats:
            selected_tokens = float(selected_blocks * block_size)
            valid_tokens = float(valid_blocks * block_size)
        else:
            selected_tokens = 0.0
            valid_tokens = 0.0

    if not return_stats:
        return out, None

    stats = _empty_detail_stats()
    stats.update(
        {
            "select_ratio": float(select_ratio),
            "selected_blocks": selected_blocks,
            "valid_blocks": valid_blocks,
            "selected_tokens": selected_tokens,
            "valid_tokens": valid_tokens,
            "gather_pack_ms": float(pack_stats["gather_pack_ms"]),
            "dense_kernel_ms": float(pack_stats["dense_kernel_ms"]),
            "fallback_used": float(max(pack_stats.get("fallback_used", 0.0), tail_dense_fallback_calls)),
            "path": "chunked_selected_dense_tail" if tail_dense_fallback_calls > 0.0 else "chunked_selected",
            "col_keep_expand_ms": col_keep_expand_ms,
            "col_input_sanitize_ms": col_input_sanitize_ms,
            "col_select_ratio_ms": col_select_ratio_ms,
            "col_repeat_kv_ms": col_repeat_kv_ms,
            "col_pack_prepare_ms": float(pack_stats["pack_prepare_ms"]),
            "col_gather_qkv_ms": float(pack_stats["gather_qkv_ms"]),
            "col_cu_seqlens_ms": float(pack_stats["cu_seqlens_ms"]),
            "col_unpack_scatter_ms": float(pack_stats["unpack_scatter_ms"]),
            "col_debug_query_block_last": 1.0 if _compactattn_query_block_mode() == "last" else 0.0,
            "col_debug_kv_head_first": 1.0 if _compactattn_kv_head_mode() == "first" else 0.0,
            "col_debug_current_open_disabled": 0.0 if _compactattn_current_chunk_full_open() else 1.0,
            "col_pack_impl_torch_calls": float(pack_stats["pack_impl_torch_calls"]),
            "col_pack_impl_triton_calls": float(pack_stats["pack_impl_triton_calls"]),
            "col_pack_impl_fallback_calls": float(pack_stats["pack_impl_fallback_calls"]),
            "col_index_build_ms": float(pack_stats.get("col_index_build_ms", 0.0)),
            "col_index_keep_flat_sanitize_ms": float(
                pack_stats.get("index_keep_flat_sanitize_ms", 0.0)
            ),
            "col_index_sel_blocks_sum_ms": float(
                pack_stats.get("index_sel_blocks_sum_ms", 0.0)
            ),
            "col_index_zero_row_fix_ms": float(
                pack_stats.get("index_zero_row_fix_ms", 0.0)
            ),
            "col_index_sel_lens_ms": float(pack_stats.get("index_sel_lens_ms", 0.0)),
            "col_index_pages_per_row_ms": float(
                pack_stats.get("index_pages_per_row_ms", 0.0)
            ),
            "col_index_page_count_stats_ms": float(
                pack_stats.get("index_page_count_stats_ms", 0.0)
            ),
            "col_index_union_block_ms": float(pack_stats.get("col_index_union_block_ms", 0.0)),
            "col_index_block_table_ms": float(pack_stats.get("col_index_block_table_ms", 0.0)),
            "col_index_table_fill_ms": float(pack_stats.get("col_index_table_fill_ms", 0.0)),
            "col_index_table_kernel_ms": float(pack_stats.get("col_index_table_kernel_ms", 0.0)),
            "col_index_compact_ms": float(pack_stats.get("col_index_compact_ms", 0.0)),
            "col_index_compact_kernel_ms": float(pack_stats.get("col_index_compact_kernel_ms", 0.0)),
            "col_index_compact_nonzero_ms": float(
                pack_stats.get("col_index_compact_nonzero_ms", 0.0)
            ),
            "col_index_compact_post_ms": float(
                pack_stats.get("col_index_compact_post_ms", 0.0)
            ),
            "col_index_compact_fused_calls": float(
                pack_stats.get("col_index_compact_fused_calls", 0.0)
            ),
            "col_index_compact_fallback_calls": float(
                pack_stats.get("col_index_compact_fallback_calls", 0.0)
            ),
            "col_index_compact_fused_post_ms": float(
                pack_stats.get("col_index_compact_fused_post_ms", 0.0)
            ),
            "col_index_src_layout_ms": float(pack_stats.get("col_index_src_layout_ms", 0.0)),
            "col_index_cache_fill_ms": float(pack_stats.get("col_index_cache_fill_ms", 0.0)),
            "col_index_cache_fill_kernel_ms": float(
                pack_stats.get("col_index_cache_fill_kernel_ms", 0.0)
            ),
            "col_index_cache_fill_cuda_calls": float(
                pack_stats.get("col_index_cache_fill_cuda_calls", 0.0)
            ),
            "col_index_cache_fill_cuda_fallback_calls": float(
                pack_stats.get("col_index_cache_fill_cuda_fallback_calls", 0.0)
            ),
            "col_index_cache_fill_backend_id": float(
                pack_stats.get("col_index_cache_fill_backend_id", 0.0)
            ),
            "col_index_cache_fill_triton_calls": float(pack_stats.get("col_index_cache_fill_triton_calls", 0.0)),
            "col_index_cache_fill_torch_calls": float(pack_stats.get("col_index_cache_fill_torch_calls", 0.0)),
            "col_index_cache_fill_fallback_calls": float(
                pack_stats.get("col_index_cache_fill_fallback_calls", 0.0)
            ),
            "col_index_cache_fill_small_calls": float(
                pack_stats.get("col_index_cache_fill_small_calls", 0.0)
            ),
            "col_index_cache_fill_medium_calls": float(
                pack_stats.get("col_index_cache_fill_medium_calls", 0.0)
            ),
            "col_index_cache_fill_large_calls": float(
                pack_stats.get("col_index_cache_fill_large_calls", 0.0)
            ),
            "col_index_cache_fill_variant_id": float(
                pack_stats.get("col_index_cache_fill_variant_id", 0.0)
            ),
            "col_index_cache_fill_tuned_calls": float(
                pack_stats.get("col_index_cache_fill_tuned_calls", 0.0)
            ),
            "col_index_cache_fill_current_tail_ms": float(
                pack_stats.get("col_index_cache_fill_current_tail_ms", 0.0)
            ),
            "col_workspace_capacity_pages": float(pack_stats.get("col_workspace_capacity_pages", 0.0)),
            "col_workspace_required_pages": float(pack_stats.get("col_workspace_required_pages", 0.0)),
            "col_workspace_growth_events": float(pack_stats.get("col_workspace_growth_events", 0.0)),
            "col_workspace_k_alloc_mb": float(pack_stats.get("col_workspace_k_alloc_mb", 0.0)),
            "col_workspace_v_alloc_mb": float(pack_stats.get("col_workspace_v_alloc_mb", 0.0)),
            "col_workspace_block_table_alloc_mb": float(
                pack_stats.get("col_workspace_block_table_alloc_mb", 0.0)
            ),
            "col_indexed_dense_kernel_ms": float(pack_stats.get("col_indexed_dense_kernel_ms", 0.0)),
            "col_indexed_dense_calls": float(pack_stats.get("col_indexed_dense_calls", 0.0)),
            "col_indexed_dense_fallback_calls": float(
                pack_stats.get("col_indexed_dense_fallback_calls", 0.0) + indexed_dense_fallback_calls
            ),
            "col_fi_zero_copy_calls": float(pack_stats.get("col_fi_zero_copy_calls", 0.0)),
            "col_fi_zero_copy_metadata_ms": float(
                pack_stats.get("col_fi_zero_copy_metadata_ms", 0.0)
            ),
            "col_fi_zero_copy_q_layout_ms": float(
                pack_stats.get("col_fi_zero_copy_q_layout_ms", 0.0)
            ),
            "col_fi_zero_copy_wrapper_init_ms": float(
                pack_stats.get("col_fi_zero_copy_wrapper_init_ms", 0.0)
            ),
            "col_fi_zero_copy_wrapper_init_host_ms": float(
                pack_stats.get("col_fi_zero_copy_wrapper_init_host_ms", 0.0)
            ),
            "col_fi_zero_copy_plan_ms": float(pack_stats.get("col_fi_zero_copy_plan_ms", 0.0)),
            "col_fi_zero_copy_plan_host_ms": float(
                pack_stats.get("col_fi_zero_copy_plan_host_ms", 0.0)
            ),
            "col_fi_zero_copy_run_ms": float(pack_stats.get("col_fi_zero_copy_run_ms", 0.0)),
            "col_fi_zero_copy_attn_ms": float(pack_stats.get("col_fi_zero_copy_attn_ms", 0.0)),
            "col_fi_zero_copy_total_ms": float(pack_stats.get("col_fi_zero_copy_total_ms", 0.0)),
            "col_fi_zero_copy_rows": float(pack_stats.get("col_fi_zero_copy_rows", 0.0)),
            "col_fi_zero_copy_kv_blocks": float(pack_stats.get("col_fi_zero_copy_kv_blocks", 0.0)),
            "col_fi_zero_copy_selected_pages": float(
                pack_stats.get("col_fi_zero_copy_selected_pages", 0.0)
            ),
            "col_fi_zero_copy_q_tokens": float(pack_stats.get("col_fi_zero_copy_q_tokens", 0.0)),
            "col_tail_dense_fallback_calls": float(
                pack_stats.get("col_tail_dense_fallback_calls", 0.0) + tail_dense_fallback_calls
            ),
            "col_full_paged_kv_reuse_calls": float(
                pack_stats.get("col_full_paged_kv_reuse_calls", 0.0)
            ),
            "col_full_paged_kv_init_calls": float(
                pack_stats.get("col_full_paged_kv_init_calls", 0.0)
            ),
            "col_full_paged_kv_append_ms": float(
                pack_stats.get("col_full_paged_kv_append_ms", 0.0)
            ),
            "col_index_page_table_only_ms": float(
                pack_stats.get("col_index_page_table_only_ms", 0.0)
            ),
            "col_selected_kv_materialize_calls": float(
                pack_stats.get("col_selected_kv_materialize_calls", 0.0)
            ),
            "col_selected_kv_materialize_ms": float(
                pack_stats.get("col_selected_kv_materialize_ms", 0.0)
            ),
            "col_fused_builder_calls": float(pack_stats.get("col_fused_builder_calls", 0.0)),
            "col_fused_builder_ms": float(pack_stats.get("col_fused_builder_ms", 0.0)),
            "col_fused_builder_table_ms": float(pack_stats.get("col_fused_builder_table_ms", 0.0)),
            "col_fused_builder_fill_ms": float(pack_stats.get("col_fused_builder_fill_ms", 0.0)),
            "col_builder_peak_alloc_mb": float(pack_stats.get("col_builder_peak_alloc_mb", 0.0)),
            "col_builder_peak_reserved_mb": float(pack_stats.get("col_builder_peak_reserved_mb", 0.0)),
            "col_direct_index_build_ms": float(pack_stats.get("col_direct_index_build_ms", 0.0)),
            "col_direct_kernel_ms": float(pack_stats.get("col_direct_kernel_ms", 0.0)),
            "col_direct_calls": float(pack_stats.get("col_direct_calls", 0.0)),
            "col_direct_fallback_calls": float(
                pack_stats.get("col_direct_fallback_calls", 0.0) + direct_impl_fallback_calls
            ),
            "col_indexed_impl_fa2_calls": float(pack_stats.get("col_indexed_impl_fa2_calls", 0.0)),
            "col_fa2_indexed_kernel_ms": float(pack_stats.get("col_fa2_indexed_kernel_ms", 0.0)),
            "col_fa2_indexed_calls": float(pack_stats.get("col_fa2_indexed_calls", 0.0)),
            "col_fa2_indexed_fallback_calls": float(
                pack_stats.get("col_fa2_indexed_fallback_calls", 0.0)
            ),
            "col_fa2_indexed_v2_short_calls": float(
                pack_stats.get("col_fa2_indexed_v2_short_calls", 0.0)
            ),
            "col_fa2_indexed_v2_long_calls": float(
                pack_stats.get("col_fa2_indexed_v2_long_calls", 0.0)
            ),
            "col_fa2_indexed_v2_fallback_calls": float(
                pack_stats.get("col_fa2_indexed_v2_fallback_calls", 0.0)
            ),
            "col_fa2_indexed_v2_kernel_ms": float(
                pack_stats.get("col_fa2_indexed_v2_kernel_ms", 0.0)
            ),
            "col_fa2_indexed_v3_short_calls": float(
                pack_stats.get("col_fa2_indexed_v3_short_calls", 0.0)
            ),
            "col_fa2_indexed_v3_long_calls": float(
                pack_stats.get("col_fa2_indexed_v3_long_calls", 0.0)
            ),
            "col_fa2_indexed_v3_fallback_calls": float(
                pack_stats.get("col_fa2_indexed_v3_fallback_calls", 0.0)
            ),
            "col_fa2_indexed_v3_split_k": float(
                pack_stats.get("col_fa2_indexed_v3_split_k", 0.0)
            ),
            "col_fa2_indexed_v3_past_kernel_ms": float(
                pack_stats.get("col_fa2_indexed_v3_past_kernel_ms", 0.0)
            ),
            "col_fa2_indexed_v3_reduce_ms": float(
                pack_stats.get("col_fa2_indexed_v3_reduce_ms", 0.0)
            ),
            "col_fa2_indexed_v3_current_kernel_ms": float(
                pack_stats.get("col_fa2_indexed_v3_current_kernel_ms", 0.0)
            ),
            "col_fa2_indexed_v3_kernel_ms": float(
                pack_stats.get("col_fa2_indexed_v3_kernel_ms", 0.0)
            ),
            "col_index_attn_cache_seqlens_cast_ms": float(
                pack_stats.get("index_attn_cache_seqlens_cast_ms", 0.0)
            ),
            "col_index_attn_block_table_cast_ms": float(
                pack_stats.get("index_attn_block_table_cast_ms", 0.0)
            ),
            "col_index_attn_q_layout_ms": float(pack_stats.get("col_index_attn_q_layout_ms", 0.0)),
            "col_index_attn_out_layout_ms": float(pack_stats.get("col_index_attn_out_layout_ms", 0.0)),
            "compactattn_version": COMPACTATTN_VERSION,
        }
    )
    return out, stats


def chunked_prefill_column_dense_attention_forward(
    query_states: torch.Tensor,  # [B, Q, Hq, D]
    key_states: torch.Tensor,  # [B, K, Hkv, D]
    value_states: torch.Tensor,  # [B, K, Hkv, D]
    attention_mask: Optional[torch.Tensor],  # [B, K]
    query_length: int,
    softmax_scale: Optional[float] = None,
    attn_gate_score: Optional[torch.Tensor] = None,  # [B, Hq|Hkv, Qb, Kb]
    block_attention_mask: Optional[torch.Tensor] = None,  # [B, 1, Qb, Kb]
    threshold: Union[float, torch.Tensor] = 0.0,
    keep_recent_blocks: int = 2,
    block_size: int = 64,
    num_key_value_groups: int = 1,
    attn_gate_is_kv_group_aware: bool = False,
    pack_impl: str = "torch",
    indexed_impl: str = "fa2_paged",
    cache_fill_backend: str = "auto",
    return_stats: bool = False,
    attn_module=None,
    k_hf: Optional[torch.Tensor] = None,  # [B, Hkv, K, D] heads-first, for fi_zero_copy
    v_hf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    if attn_gate_score is None:
        raise ValueError("attn_gate_score is required for chunked column-dense attention.")
    if query_length <= 1:
        raise ValueError("chunked_prefill_column_dense_attention_forward expects query_length > 1.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if indexed_impl not in {"fa2_paged", "triton_direct", "fa2_indexed", "fi_paged", "fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"}:
        raise ValueError(f"Unsupported indexed_impl: {indexed_impl}")

    threshold = _compactattn_effective_threshold(threshold)

    bsz, q_len, num_q_heads, _ = query_states.shape
    kv_len = key_states.shape[1]
    device = query_states.device
    measure_timing = bool(return_stats)
    attention_mask_was_none = attention_mask is None
    attention_mask = _ensure_attention_mask(attention_mask, bsz, kv_len, device)
    expected_kv_blocks = (kv_len + block_size - 1) // block_size

    # Standard chunked-prefill calls usually arrive without an explicit token mask.
    # In that case we know attention_mask is all-ones, so the fast-path can be used
    # directly after cheap shape/alignment checks. Custom masks still go through the
    # cached verifier to preserve fallback behavior.
    common_default_fast_path = attention_mask_was_none and block_attention_mask is None
    threshold_is_tensor = isinstance(threshold, torch.Tensor) and threshold.numel() > 1

    if common_default_fast_path and not threshold_is_tensor:
        use_fast_path = _can_use_chunked_mask_fast_path(
            attn_gate_score=attn_gate_score,
            block_attention_mask=block_attention_mask,
            attention_mask=attention_mask,
            q_len=q_len,
            kv_len=kv_len,
            block_size=block_size,
            skip_content_checks=True,
        )
    elif not threshold_is_tensor:
        # Use cached content-verification to avoid repeated GPU→host syncs.
        # Content checks (attention_mask all-ones, block_attention_mask causal)
        # are invariant across chunks of the same sequence.
        cache_key = (str(device), int(q_len), int(block_size))
        content_verified = _FAST_PATH_CONTENT_VERIFIED.get(cache_key, False)
        use_fast_path = _can_use_chunked_mask_fast_path(
            attn_gate_score=attn_gate_score,
            block_attention_mask=block_attention_mask,
            attention_mask=attention_mask,
            q_len=q_len,
            kv_len=kv_len,
            block_size=block_size,
            skip_content_checks=content_verified,
        )
        if use_fast_path and not content_verified:
            _FAST_PATH_CONTENT_VERIFIED[cache_key] = True
    else:
        use_fast_path = False

    col_select_mask_past_ms = 0.0
    col_select_mask_curr_ms = 0.0
    valid_is_all_ones = False
    keep_block_kv_preprojected = None
    if use_fast_path:
        keep_block, valid_is_all_ones, col_select_mask_past_ms, col_select_mask_curr_ms = (
            _build_keep_block_mask_chunked_fast_path(
                attn_gate_score=attn_gate_score,
                block_attention_mask=block_attention_mask,
                threshold=threshold,
                keep_recent_blocks=keep_recent_blocks,
                q_len=q_len,
                kv_len=kv_len,
                block_size=block_size,
                measure_timing=measure_timing,
            )
        )
        col_select_mask_ms = col_select_mask_past_ms + col_select_mask_curr_ms
        fastpath_calls = 1.0
        generic_calls = 0.0
        valid_block = None
        if attn_gate_is_kv_group_aware:
            keep_block_kv_preprojected = keep_block
            keep_block = None
        elif keep_block.shape[-1] != expected_kv_blocks:
            keep_block_kv_preprojected = _project_short_qhead_keep_block_to_kv_full_width(
                keep_block=keep_block,
                expected_kv_blocks=expected_kv_blocks,
                num_key_value_groups=num_key_value_groups,
            )
            keep_block = None
        else:
            keep_block_kv_preprojected = _build_kv_head_union_keep_block_mask(
                keep_block=keep_block,
                q_len=q_len,
                kv_len=kv_len,
                block_size=block_size,
                num_key_value_groups=num_key_value_groups,
            )
    else:
        keep_block_out, col_select_mask_ms = _cuda_elapsed_ms(
            lambda: _build_keep_block_mask(
                attn_gate_score=attn_gate_score,
                block_attention_mask=block_attention_mask,
                threshold=threshold,
                keep_recent_blocks=keep_recent_blocks,
            ),
            enabled=measure_timing,
        )
        keep_block, valid_block, _, _ = keep_block_out
        valid_is_all_ones = False
        fastpath_calls = 0.0
        generic_calls = 1.0
        col_select_mask_curr_ms = col_select_mask_ms
        if attn_gate_is_kv_group_aware:
            keep_block_kv_preprojected = keep_block
            keep_block = None
            valid_block = None
        elif keep_block.shape[-1] != expected_kv_blocks:
            keep_block_kv_preprojected = _project_short_qhead_keep_block_to_kv_full_width(
                keep_block=keep_block,
                expected_kv_blocks=expected_kv_blocks,
                num_key_value_groups=num_key_value_groups,
            )
            keep_block = None
            valid_block = None

    out, stats = chunked_prefill_column_dense_attention_from_keep_block(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        keep_block=keep_block,
        attention_mask=attention_mask,
        query_length=query_length,
        softmax_scale=softmax_scale,
        block_size=block_size,
        num_key_value_groups=num_key_value_groups,
        pack_impl=pack_impl,
        indexed_impl=indexed_impl,
        cache_fill_backend=cache_fill_backend,
        return_stats=return_stats,
        valid_block=None if valid_is_all_ones else valid_block,
        allow_indexed_dense=(pack_impl == "indexed_dense"),
        keep_block_kv_preprojected=keep_block_kv_preprojected,
        attn_module=attn_module,
        k_hf=k_hf,
        v_hf=v_hf,
    )
    if not return_stats:
        return out, None

    assert stats is not None
    stats.update(
        {
            "col_select_mask_ms": col_select_mask_ms,
            "col_select_mask_past_ms": col_select_mask_past_ms,
            "col_select_mask_curr_ms": col_select_mask_curr_ms,
            "col_select_mask_fastpath_calls": fastpath_calls,
            "col_select_mask_generic_calls": generic_calls,
            "compactattn_version": COMPACTATTN_VERSION,
        }
    )
    return out, stats
