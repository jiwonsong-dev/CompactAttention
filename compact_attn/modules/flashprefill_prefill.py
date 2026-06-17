import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from compact_attn.flashprefill_vendor import (
    flash_prefill_chunked,
    flash_prefill_chunked_from_mean_k,
    flash_prefill_score_chunked,
    flash_prefill_score_chunked_from_mean_k,
    flash_prefill_select_chunked,
    flash_prefill_select_chunked_from_mean_k,
)
from compact_attn.modules.attention_forward_chunked_dense import (
    chunked_prefill_column_dense_attention_from_keep_block,
)
from compact_attn.modules.dense_prefill import _cuda_elapsed_ms, dense_prefill_full_kv
from compact_attn.modules.common import repeat_kv

_FLASHPREFILL_DEBUG_PRINTED = False


def _flashprefill_use_strided_select_k() -> bool:
    return os.environ.get("SEER_FLASHPREFILL_STRIDED_SELECT_K", "0") == "1"


def _flashprefill_use_fused_kv_union() -> bool:
    return os.environ.get("SEER_FLASHPREFILL_FUSED_KV_UNION", "1") != "0"


def _flashprefill_debug_tensor(name: str, tensor: torch.Tensor) -> str:
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    return (
        f"{name}:type={type(tensor).__name__},shape={tuple(tensor.shape)},"
        f"local_shape={tuple(local.shape)},device={local.device},stride={local.stride()}"
    )


def _maybe_print_flashprefill_debug(*tensors: tuple[str, torch.Tensor]) -> None:
    global _FLASHPREFILL_DEBUG_PRINTED
    if _FLASHPREFILL_DEBUG_PRINTED:
        return
    if os.environ.get("SEER_DEBUG_FLASHPREFILL_INPUTS", "0") != "1":
        return
    _FLASHPREFILL_DEBUG_PRINTED = True
    rank = os.environ.get("RANK", "?")
    msg = " | ".join(_flashprefill_debug_tensor(name, tensor) for name, tensor in tensors)
    print(f"[flashprefill debug rank={rank}] {msg}", flush=True)


def _flashprefill_active_mask_from_selection(
    block_index: torch.Tensor,  # [B, Qb, Kb, Hq]
    counts: torch.Tensor,  # [B, Qb, Hq]
) -> torch.Tensor:
    bsz, query_blocks, key_blocks, num_q_heads = block_index.shape
    device = block_index.device
    indices = block_index.permute(0, 3, 1, 2).contiguous()  # [B, Hq, Qb, Kb]
    valid = (
        torch.arange(key_blocks, device=device, dtype=counts.dtype)
        .view(1, 1, 1, key_blocks)
        < counts.permute(0, 2, 1).unsqueeze(-1)
    )

    active = torch.zeros((bsz, num_q_heads, query_blocks, key_blocks), dtype=torch.bool, device=device)
    active.scatter_(dim=-1, index=indices.masked_fill(~valid, 0), src=valid)
    return active


def _flashprefill_pairwise_jaccard_mean(active_past: torch.Tensor) -> float:
    _, _, query_blocks, _ = active_past.shape
    if query_blocks <= 1:
        return 0.0

    pairwise = []
    for q0 in range(query_blocks):
        a = active_past[:, :, q0, :]
        for q1 in range(q0 + 1, query_blocks):
            b = active_past[:, :, q1, :]
            union = (a | b).sum(dim=-1)
            if not bool((union > 0).any().item()):
                continue
            inter = (a & b).sum(dim=-1)
            jaccard = inter.to(torch.float32) / union.clamp_min(1).to(torch.float32)
            pairwise.append(jaccard[union > 0].mean())

    if not pairwise:
        return 0.0
    return float(torch.stack(pairwise).mean().item())


def _flashprefill_selection_profile_stats(
    *,
    active: torch.Tensor,  # [B, Hq, Qb, Kb]
    keep_block: torch.Tensor,  # [B, Hq, Kb]
    keep_block_kv_preprojected: torch.Tensor,  # [B, Hkv, Kb]
    q_len: int,
    kv_len: int,
    block_size: int,
) -> Dict[str, float]:
    past_len = max(kv_len - q_len, 0)
    past_k_blocks = max((past_len + block_size - 1) // block_size, 0)
    curr_k_blocks = max(active.shape[-1] - past_k_blocks, 0)

    active_past = active[..., :past_k_blocks]
    keep_past_q = keep_block[..., :past_k_blocks]
    keep_past_kv = keep_block_kv_preprojected[..., :past_k_blocks]
    keep_curr_kv = keep_block_kv_preprojected[..., past_k_blocks : past_k_blocks + curr_k_blocks]

    def _ratio(x: torch.Tensor) -> float:
        total = x.numel()
        if total <= 0:
            return 0.0
        return float(x.sum().item()) / float(total)

    return {
        "flashprefill_pre_union_full_density": _ratio(active),
        "flashprefill_pre_union_past_density": _ratio(active_past),
        "flashprefill_post_query_union_full_density": _ratio(keep_block),
        "flashprefill_post_query_union_past_density": _ratio(keep_past_q),
        "flashprefill_post_kv_union_full_density": _ratio(keep_block_kv_preprojected),
        "flashprefill_post_kv_union_past_density": _ratio(keep_past_kv),
        "flashprefill_current_chunk_open_density": _ratio(keep_curr_kv),
        "flashprefill_pre_union_selected_blocks_full": float(active.sum().item()),
        "flashprefill_pre_union_selected_blocks_past": float(active_past.sum().item()),
        "flashprefill_post_query_union_selected_blocks_full": float(keep_block.sum().item()),
        "flashprefill_post_query_union_selected_blocks_past": float(keep_past_q.sum().item()),
        "flashprefill_post_kv_union_selected_blocks_full": float(keep_block_kv_preprojected.sum().item()),
        "flashprefill_post_kv_union_selected_blocks_past": float(keep_past_kv.sum().item()),
        "flashprefill_query_block_pairwise_jaccard_mean": _flashprefill_pairwise_jaccard_mean(active_past),
    }


def _flashprefill_selection_profile_stats_from_scores(
    *,
    output_score: torch.Tensor,  # [B, Qb, Kb, Hq]
    keep_block_kv_preprojected: torch.Tensor,  # [B, Hkv or Hkv*num_subgroups, Kb]
    q_len: int,
    kv_len: int,
    block_size: int,
    num_key_value_groups: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block: int,
    query_subgroup_size: Optional[int],
) -> Dict[str, float]:
    bsz, query_blocks, key_blocks, num_q_heads = output_score.shape
    device = output_score.device
    num_kv_heads = num_q_heads // num_key_value_groups
    past_len = max(kv_len - q_len, 0)
    past_k_blocks = max((past_len + block_size - 1) // block_size, 0)
    curr_k_blocks = max(key_blocks - past_k_blocks, 0)

    tail_query_blocks = min(max(last_n_block, 0), query_blocks - 1) if query_blocks > 1 else 0
    pooled_query_blocks = max(query_blocks - tail_query_blocks, 1)
    score_hq = output_score[:, :pooled_query_blocks].permute(0, 3, 1, 2).contiguous()

    k_ids = torch.arange(key_blocks, device=device).view(1, 1, 1, key_blocks)
    q_ids = torch.arange(
        past_k_blocks,
        past_k_blocks + pooled_query_blocks,
        device=device,
    ).view(1, 1, pooled_query_blocks, 1)
    dist = q_ids - k_ids
    mask_causal = dist >= 0
    mask_sink = k_ids < int(attention_sink)
    mask_window = mask_causal & (dist < int(window_size))
    finite_all = torch.isfinite(score_hq) & mask_causal
    row_score = torch.where(finite_all, score_hq, torch.zeros_like(score_hq))
    max_val = row_score.max(dim=-1, keepdim=True).values
    active = finite_all & (row_score >= (max_val * float(alpha)))
    active = (active | mask_sink | mask_window) & mask_causal

    keep_block = active.any(dim=2)
    if curr_k_blocks > 0:
        keep_block[..., past_k_blocks:key_blocks] = True

    if query_subgroup_size is not None:
        subgroup_size = int(query_subgroup_size)
        if num_key_value_groups % subgroup_size != 0:
            raise ValueError(
                "query_subgroup_size must divide num_key_value_groups: "
                f"{subgroup_size} vs {num_key_value_groups}"
            )
        num_subgroups = num_key_value_groups // subgroup_size
        keep_subgroup = keep_block.view(
            bsz, num_kv_heads, num_subgroups, subgroup_size, key_blocks
        ).any(dim=3)
        if curr_k_blocks > 0:
            keep_subgroup[..., past_k_blocks:key_blocks] = True
        keep_kv_after_subgroup = keep_subgroup.any(dim=2)
    else:
        num_subgroups = 1
        keep_subgroup = keep_block.view(
            bsz, num_kv_heads, num_key_value_groups, key_blocks
        ).any(dim=2).unsqueeze(2)
        keep_kv_after_subgroup = keep_subgroup.squeeze(2)

    keep_past_q = keep_block[..., :past_k_blocks]
    keep_past_subgroup = keep_subgroup[..., :past_k_blocks]
    keep_past_kv_after_subgroup = keep_kv_after_subgroup[..., :past_k_blocks]
    active_past = active[..., :past_k_blocks]
    preprojected_past = keep_block_kv_preprojected[..., :past_k_blocks]

    def _sum(x: torch.Tensor) -> float:
        return float(x.sum().item())

    def _numel(x: torch.Tensor) -> float:
        return float(x.numel())

    return {
        "flashprefill_pre_union_full_density": _sum(active) / _numel(active) if active.numel() else 0.0,
        "flashprefill_pre_union_past_density": _sum(active_past) / _numel(active_past) if active_past.numel() else 0.0,
        "flashprefill_post_query_union_full_density": _sum(keep_block) / _numel(keep_block) if keep_block.numel() else 0.0,
        "flashprefill_post_query_union_past_density": _sum(keep_past_q) / _numel(keep_past_q) if keep_past_q.numel() else 0.0,
        "flashprefill_post_subgroup_union_full_density": _sum(keep_subgroup) / _numel(keep_subgroup) if keep_subgroup.numel() else 0.0,
        "flashprefill_post_subgroup_union_past_density": _sum(keep_past_subgroup) / _numel(keep_past_subgroup) if keep_past_subgroup.numel() else 0.0,
        "flashprefill_post_kv_union_full_density": _sum(keep_kv_after_subgroup) / _numel(keep_kv_after_subgroup) if keep_kv_after_subgroup.numel() else 0.0,
        "flashprefill_post_kv_union_past_density": _sum(keep_past_kv_after_subgroup) / _numel(keep_past_kv_after_subgroup) if keep_past_kv_after_subgroup.numel() else 0.0,
        "flashprefill_pre_union_selected_blocks_full": _sum(active),
        "flashprefill_pre_union_valid_blocks_full": _numel(active),
        "flashprefill_pre_union_selected_blocks_past": _sum(active_past),
        "flashprefill_pre_union_valid_blocks_past": _numel(active_past),
        "flashprefill_post_query_union_selected_blocks_full": _sum(keep_block),
        "flashprefill_post_query_union_valid_blocks_full": _numel(keep_block),
        "flashprefill_post_query_union_selected_blocks_past": _sum(keep_past_q),
        "flashprefill_post_query_union_valid_blocks_past": _numel(keep_past_q),
        "flashprefill_post_subgroup_union_selected_blocks_full": _sum(keep_subgroup),
        "flashprefill_post_subgroup_union_valid_blocks_full": _numel(keep_subgroup),
        "flashprefill_post_subgroup_union_selected_blocks_past": _sum(keep_past_subgroup),
        "flashprefill_post_subgroup_union_valid_blocks_past": _numel(keep_past_subgroup),
        "flashprefill_post_kv_union_selected_blocks_full": _sum(keep_kv_after_subgroup),
        "flashprefill_post_kv_union_valid_blocks_full": _numel(keep_kv_after_subgroup),
        "flashprefill_post_kv_union_selected_blocks_past": _sum(keep_past_kv_after_subgroup),
        "flashprefill_post_kv_union_valid_blocks_past": _numel(keep_past_kv_after_subgroup),
        "flashprefill_preprojected_union_selected_blocks_full": _sum(keep_block_kv_preprojected),
        "flashprefill_preprojected_union_valid_blocks_full": _numel(keep_block_kv_preprojected),
        "flashprefill_preprojected_union_selected_blocks_past": _sum(preprojected_past),
        "flashprefill_preprojected_union_valid_blocks_past": _numel(preprojected_past),
        "flashprefill_query_subgroup_size": float(query_subgroup_size or num_key_value_groups),
        "flashprefill_query_subgroups": float(num_subgroups),
    }


def _flashprefill_dense_fallback(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    softmax_scale: float,
    num_key_value_groups: int,
    measure_timing: bool,
    attn_module,
):
    _maybe_print_flashprefill_debug(
        ("fallback_q", query_states),
        ("fallback_k", key_states),
        ("fallback_v", value_states),
    )

    def _run_sdpa():
        bsz, q_len, _, _ = query_states.shape
        kv_len = key_states.shape[1]
        q = query_states.transpose(1, 2).contiguous()
        k = repeat_kv(key_states.transpose(1, 2).contiguous(), num_key_value_groups)
        v = repeat_kv(value_states.transpose(1, 2).contiguous(), num_key_value_groups)

        q_abs = torch.arange(kv_len - q_len, kv_len, device=query_states.device).view(q_len, 1)
        k_abs = torch.arange(kv_len, device=query_states.device).view(1, kv_len)
        causal = k_abs <= q_abs
        if attention_mask is not None:
            pad = attention_mask.to(device=query_states.device)
            if pad.dim() == 2:
                causal = causal.view(1, 1, q_len, kv_len) & pad[:, None, None, :].to(torch.bool)
            else:
                causal = causal.view(1, 1, q_len, kv_len)
        else:
            causal = causal.view(1, 1, q_len, kv_len)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal,
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        )
        return out.transpose(1, 2).contiguous()

    out, dense_kernel_ms = _cuda_elapsed_ms(_run_sdpa, enabled=measure_timing and query_states.is_cuda)
    stats = {
        "repeat_kv_ms": 0.0,
        "upad_input_ms": 0.0,
        "pad_output_ms": 0.0,
        "dense_kernel_ms": float(dense_kernel_ms),
        "gather_pack_ms": 0.0,
        "fallback_used": 1.0,
        "flashprefill_sdpa_fallback_calls": 1.0,
    }
    return out, stats


@triton.jit
def _flashprefill_kv_union_from_scores_kernel(
    score_ptr,
    keep_ptr,
    stride_sb,
    stride_sq,
    stride_sk,
    stride_sh,
    stride_ob,
    stride_oh,
    stride_ok,
    num_q_heads: tl.constexpr,
    key_blocks: tl.constexpr,
    past_k_blocks: tl.constexpr,
    pooled_query_blocks: tl.constexpr,
    attention_sink: tl.constexpr,
    window_size: tl.constexpr,
    alpha: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // (num_q_heads // group_size)
    hkv = pid_bh % (num_q_heads // group_size)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < key_blocks

    # Current-chunk KV blocks are always visible in the compactattn projection.
    keep = offs_k >= past_k_blocks

    for g in range(0, group_size):
        hq = hkv * group_size + g
        for q in range(0, pooled_query_blocks):
            q_abs = past_k_blocks + q
            row_max = tl.full((), 0.0, tl.float32)
            for kb0 in range(0, key_blocks, BLOCK_K):
                scan_k = kb0 + tl.arange(0, BLOCK_K)
                vals = tl.load(
                    score_ptr
                    + b * stride_sb
                    + q * stride_sq
                    + scan_k * stride_sk
                    + hq * stride_sh,
                    mask=scan_k < key_blocks,
                    other=0.0,
                ).to(tl.float32)
                vals = tl.where(scan_k <= q_abs, vals, 0.0)
                row_max = tl.maximum(row_max, tl.max(vals, axis=0))

            vals = tl.load(
                score_ptr
                + b * stride_sb
                + q * stride_sq
                + offs_k * stride_sk
                + hq * stride_sh,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            causal = offs_k <= q_abs
            score_keep = (vals >= (row_max * alpha)) & causal
            sink_keep = (offs_k < attention_sink) & causal
            window_keep = ((q_abs - offs_k) >= 0) & ((q_abs - offs_k) < window_size)
            keep = keep | score_keep | sink_keep | window_keep

    tl.store(
        keep_ptr + b * stride_ob + hkv * stride_oh + offs_k * stride_ok,
        keep,
        mask=mask_k,
    )


@triton.jit
def _flashprefill_subgroup_union_from_scores_kernel(
    score_ptr,
    keep_ptr,
    stride_sb,
    stride_sq,
    stride_sk,
    stride_sh,
    stride_ob,
    stride_oh,
    stride_ok,
    num_q_heads: tl.constexpr,
    key_blocks: tl.constexpr,
    past_k_blocks: tl.constexpr,
    pooled_query_blocks: tl.constexpr,
    attention_sink: tl.constexpr,
    window_size: tl.constexpr,
    alpha: tl.constexpr,
    group_size: tl.constexpr,
    subgroup_size: tl.constexpr,
    num_subgroups: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_bg = tl.program_id(1)
    num_kv_heads = num_q_heads // group_size
    b = pid_bg // (num_kv_heads * num_subgroups)
    group_row = pid_bg % (num_kv_heads * num_subgroups)
    hkv = group_row // num_subgroups
    subgroup = group_row % num_subgroups
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < key_blocks

    # Current-chunk KV blocks are always visible in the compactattn projection.
    keep = offs_k >= past_k_blocks

    for sg in range(0, subgroup_size):
        hq = hkv * group_size + subgroup * subgroup_size + sg
        for q in range(0, pooled_query_blocks):
            q_abs = past_k_blocks + q
            row_max = tl.full((), 0.0, tl.float32)
            for kb0 in range(0, key_blocks, BLOCK_K):
                scan_k = kb0 + tl.arange(0, BLOCK_K)
                vals = tl.load(
                    score_ptr
                    + b * stride_sb
                    + q * stride_sq
                    + scan_k * stride_sk
                    + hq * stride_sh,
                    mask=scan_k < key_blocks,
                    other=0.0,
                ).to(tl.float32)
                vals = tl.where(scan_k <= q_abs, vals, 0.0)
                row_max = tl.maximum(row_max, tl.max(vals, axis=0))

            vals = tl.load(
                score_ptr
                + b * stride_sb
                + q * stride_sq
                + offs_k * stride_sk
                + hq * stride_sh,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            causal = offs_k <= q_abs
            score_keep = (vals >= (row_max * alpha)) & causal
            sink_keep = (offs_k < attention_sink) & causal
            window_keep = ((q_abs - offs_k) >= 0) & ((q_abs - offs_k) < window_size)
            keep = keep | score_keep | sink_keep | window_keep

    tl.store(
        keep_ptr + b * stride_ob + group_row * stride_oh + offs_k * stride_ok,
        keep,
        mask=mask_k,
    )


def _flashprefill_keep_block_kv_from_scores_fused(
    output_score: torch.Tensor,  # [B, Qb, Kb, Hq]
    *,
    q_len: int,
    kv_len: int,
    block_size: int,
    num_key_value_groups: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block: int,
) -> torch.Tensor:
    bsz, query_blocks, key_blocks, num_q_heads = output_score.shape
    past_len = max(kv_len - q_len, 0)
    past_k_blocks = max((past_len + block_size - 1) // block_size, 0)
    num_kv_heads = num_q_heads // num_key_value_groups
    tail_query_blocks = min(max(last_n_block, 0), query_blocks - 1) if query_blocks > 1 else 0
    pooled_query_blocks = max(query_blocks - tail_query_blocks, 1)
    keep_block_kv = torch.empty((bsz, num_kv_heads, key_blocks), dtype=torch.bool, device=output_score.device)
    block_k = 64
    grid = (triton.cdiv(key_blocks, block_k), bsz * num_kv_heads)
    _flashprefill_kv_union_from_scores_kernel[grid](
        output_score,
        keep_block_kv,
        *output_score.stride(),
        *keep_block_kv.stride(),
        num_q_heads,
        key_blocks,
        past_k_blocks,
        pooled_query_blocks,
        int(attention_sink),
        int(window_size),
        float(alpha),
        int(num_key_value_groups),
        BLOCK_K=block_k,
    )
    return keep_block_kv


def _flashprefill_query_subgroup_size(num_key_value_groups: int) -> int:
    subgroup_size = int(os.environ.get("SEER_ZC_QUERY_SUBGROUP_SIZE", "4"))
    if subgroup_size <= 0:
        raise ValueError("SEER_ZC_QUERY_SUBGROUP_SIZE must be positive")
    if num_key_value_groups % subgroup_size != 0:
        raise ValueError(
            "SEER_ZC_QUERY_SUBGROUP_SIZE must divide num_key_value_groups: "
            f"{subgroup_size} vs {num_key_value_groups}"
        )
    return subgroup_size


def _flashprefill_keep_block_subgroup_from_scores_fused(
    output_score: torch.Tensor,  # [B, Qb, Kb, Hq]
    *,
    q_len: int,
    kv_len: int,
    block_size: int,
    num_key_value_groups: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block: int,
    query_subgroup_size: int,
) -> torch.Tensor:
    bsz, query_blocks, key_blocks, num_q_heads = output_score.shape
    past_len = max(kv_len - q_len, 0)
    past_k_blocks = max((past_len + block_size - 1) // block_size, 0)
    num_kv_heads = num_q_heads // num_key_value_groups
    if num_key_value_groups % query_subgroup_size != 0:
        raise ValueError(
            f"query_subgroup_size={query_subgroup_size} must divide G={num_key_value_groups}"
        )
    num_subgroups = num_key_value_groups // query_subgroup_size
    tail_query_blocks = min(max(last_n_block, 0), query_blocks - 1) if query_blocks > 1 else 0
    pooled_query_blocks = max(query_blocks - tail_query_blocks, 1)
    keep_block_group = torch.empty(
        (bsz, num_kv_heads * num_subgroups, key_blocks),
        dtype=torch.bool,
        device=output_score.device,
    )
    block_k = 64
    grid = (triton.cdiv(key_blocks, block_k), bsz * num_kv_heads * num_subgroups)
    _flashprefill_subgroup_union_from_scores_kernel[grid](
        output_score,
        keep_block_group,
        *output_score.stride(),
        *keep_block_group.stride(),
        num_q_heads,
        key_blocks,
        past_k_blocks,
        pooled_query_blocks,
        int(attention_sink),
        int(window_size),
        float(alpha),
        int(num_key_value_groups),
        int(query_subgroup_size),
        int(num_subgroups),
        BLOCK_K=block_k,
    )
    return keep_block_group


def _flashprefill_keep_block_from_selection(
    block_index: torch.Tensor,  # [B, Qb, Kb, Hq]
    counts: torch.Tensor,  # [B, Qb, Hq]
    *,
    q_len: int,
    kv_len: int,
    block_size: int,
    num_key_value_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, query_blocks, key_blocks, num_q_heads = block_index.shape
    device = block_index.device
    past_len = max(kv_len - q_len, 0)
    past_k_blocks = max((past_len + block_size - 1) // block_size, 0)
    num_kv_heads = num_q_heads // num_key_value_groups

    active = _flashprefill_active_mask_from_selection(block_index, counts)
    keep_block = active.any(dim=2)  # [B, Hq, Kb]
    if past_k_blocks < key_blocks:
        keep_block[..., past_k_blocks:key_blocks] = True

    if past_k_blocks > 0:
        keep_past_view = keep_block[..., :past_k_blocks].view(
            bsz, num_kv_heads, num_key_value_groups, past_k_blocks
        )
        keep_past_kv = keep_past_view.any(dim=2)
    else:
        keep_past_kv = torch.zeros(
            (bsz, num_kv_heads, 0), dtype=torch.bool, device=device
        )

    curr_k_blocks = max(key_blocks - past_k_blocks, 0)
    keep_curr_kv = torch.ones(
        (bsz, num_kv_heads, curr_k_blocks), dtype=torch.bool, device=device
    )
    keep_block_kv_preprojected = torch.cat([keep_past_kv, keep_curr_kv], dim=-1)
    return keep_block, keep_block_kv_preprojected


def _flashprefill_keep_block_from_scores_rowwise(
    output_score: torch.Tensor,  # [B, Qb, Kb, Hq]
    *,
    q_len: int,
    kv_len: int,
    block_size: int,
    num_key_value_groups: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block: int,
    min_budget: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    score_hq = output_score.permute(0, 3, 1, 2).contiguous()  # [B, Hq, Qb, Kb]
    bsz, num_q_heads, query_blocks, key_blocks = score_hq.shape
    device = output_score.device
    past_len = max(kv_len - q_len, 0)
    past_k_blocks = max((past_len + block_size - 1) // block_size, 0)
    q_block_offset = past_k_blocks
    num_kv_heads = num_q_heads // num_key_value_groups

    tail_query_blocks = min(max(last_n_block, 0), query_blocks - 1) if query_blocks > 1 else 0
    pooled_query_blocks = max(query_blocks - tail_query_blocks, 1)

    k_ids = torch.arange(key_blocks, device=device).view(1, 1, 1, key_blocks)
    q_ids = torch.arange(
        q_block_offset,
        q_block_offset + pooled_query_blocks,
        device=device,
    ).view(1, 1, pooled_query_blocks, 1)
    dist = q_ids - k_ids
    mask_causal = dist >= 0
    mask_sink = k_ids < attention_sink
    mask_window = mask_causal & (dist < window_size)

    # Official FlashPrefill applies the alpha threshold independently for each
    # query block and query head, then execution-specific projection may union
    # those selected blocks. Do not threshold after q-block pooling.
    row_score = score_hq[:, :, :pooled_query_blocks, :]
    finite_all = torch.isfinite(row_score) & mask_causal
    row_score = torch.where(finite_all, row_score, torch.zeros_like(row_score))

    if min_budget > 0:
        k = min(int(min_budget), int(key_blocks))
        topk_vals, topk_indices = torch.topk(row_score, k=k, dim=-1)
        max_val = topk_vals[..., :1]
        keep_row = finite_all & (row_score >= (max_val * alpha))
        keep_row.scatter_(-1, topk_indices, True)
    else:
        max_val = row_score.max(dim=-1, keepdim=True).values
        keep_row = finite_all & (row_score >= (max_val * alpha))

    keep_row = (keep_row | mask_sink | mask_window) & mask_causal
    keep_all_q = keep_row.any(dim=2)

    if past_k_blocks > 0:
        keep_past_q = keep_all_q[..., :past_k_blocks]
    else:
        keep_past_q = torch.zeros((bsz, num_q_heads, 0), dtype=torch.bool, device=device)

    curr_k_blocks = max(key_blocks - past_k_blocks, 0)
    keep_curr_q = torch.ones((bsz, num_q_heads, curr_k_blocks), dtype=torch.bool, device=device)
    keep_block = torch.cat([keep_past_q, keep_curr_q], dim=-1)
    if past_k_blocks > 0:
        keep_past_view = keep_past_q.view(bsz, num_kv_heads, num_key_value_groups, past_k_blocks)
        keep_past_kv = keep_past_view.any(dim=2)
    else:
        keep_past_kv = torch.zeros((bsz, num_kv_heads, 0), dtype=torch.bool, device=device)
    keep_curr_kv = torch.ones((bsz, num_kv_heads, curr_k_blocks), dtype=torch.bool, device=device)
    keep_block_kv_preprojected = torch.cat([keep_past_kv, keep_curr_kv], dim=-1)
    return keep_block, keep_block_kv_preprojected


def flashprefill_block_sparse_prefill_full_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    softmax_scale: float,
    num_key_value_groups: int,
    block_size: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block: int,
    min_budget: int,
    measure_timing: bool = False,
    profile_selection: bool = False,
    attn_module=None,
    mean_k_cache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    q_len = int(query_states.shape[1])
    kv_len = int(key_states.shape[1])

    if q_len <= 1 or kv_len <= q_len:
        out, stats = _flashprefill_dense_fallback(
            query_states,
            key_states,
            value_states,
            attention_mask,
            softmax_scale=softmax_scale,
            num_key_value_groups=num_key_value_groups,
            measure_timing=measure_timing,
            attn_module=attn_module,
        )
        stats = dict(stats)
        stats.update(
            {
                "flashprefill_fused_total_ms": 0.0,
                "flashprefill_select_replay_ms": 0.0,
                "flashprefill_estimated_attention_ms": 0.0,
                "flashprefill_kernel_ms": 0.0,
                "flashprefill_selected_blocks": 0.0,
                "flashprefill_selected_blocks_pre_union": 0.0,
                "flashprefill_pre_union_past_density": 0.0,
                "flashprefill_post_query_union_past_density": 0.0,
                "flashprefill_post_kv_union_past_density": 0.0,
                "flashprefill_current_chunk_open_density": 0.0,
                "flashprefill_pre_union_selected_blocks_past": 0.0,
                "flashprefill_post_query_union_selected_blocks_past": 0.0,
                "flashprefill_post_kv_union_selected_blocks_past": 0.0,
                "flashprefill_query_block_pairwise_jaccard_mean": 0.0,
                "flashprefill_mean_k_cache_used": 0.0,
                "flashprefill_query_blocks": 0.0,
                "flashprefill_key_blocks": 0.0,
            }
        )
        return out, stats

    if attention_mask is not None:
        mask = attention_mask.to(device=query_states.device)
        if (mask == 0).any():
            return _flashprefill_dense_fallback(
                query_states,
                key_states,
                value_states,
                attention_mask,
                softmax_scale=softmax_scale,
                num_key_value_groups=num_key_value_groups,
                measure_timing=measure_timing,
                attn_module=attn_module,
            )

    if not query_states.is_cuda or query_states.dtype not in (torch.float16, torch.bfloat16):
        return _flashprefill_dense_fallback(
            query_states,
            key_states,
            value_states,
            attention_mask,
            softmax_scale=softmax_scale,
            num_key_value_groups=num_key_value_groups,
            measure_timing=measure_timing,
            attn_module=attn_module,
        )

    _maybe_print_flashprefill_debug(
        ("q", query_states),
        ("k", key_states),
        ("v", value_states),
    )

    out = torch.empty_like(query_states).contiguous()
    _, fused_total_ms = _cuda_elapsed_ms(
        lambda: (
            flash_prefill_chunked_from_mean_k(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                out,
                mean_k_cache.contiguous(),
                block_size=block_size,
                attention_sink=attention_sink,
                window_size=window_size,
                alpha=alpha,
                last_n_block_full=last_n_block,
                min_budget=min_budget,
            )
            if mean_k_cache is not None
            else flash_prefill_chunked(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                out,
                block_size=block_size,
                attention_sink=attention_sink,
                window_size=window_size,
                alpha=alpha,
                last_n_block_full=last_n_block,
                min_budget=min_budget,
            )
        ),
        enabled=measure_timing and query_states.is_cuda,
    )

    profile_stats = {}
    selected_blocks = 0.0
    selected_blocks_pre_union = 0.0
    valid_blocks = 0.0
    query_blocks = float((q_len + block_size - 1) // block_size)
    key_blocks = float((kv_len + block_size - 1) // block_size)
    select_replay_ms = 0.0
    profile_stats_ms = 0.0
    estimated_attention_ms = 0.0
    if profile_selection:
        (selection_outputs, select_replay_ms) = _cuda_elapsed_ms(
            lambda: (
                flash_prefill_select_chunked_from_mean_k(
                    query_states.contiguous(),
                    mean_k_cache.contiguous(),
                    key_len=kv_len,
                    block_size=block_size,
                    attention_sink=attention_sink,
                    window_size=window_size,
                    alpha=alpha,
                    last_n_block_full=last_n_block,
                    min_budget=min_budget,
                )
                if mean_k_cache is not None
                else flash_prefill_select_chunked(
                    query_states.contiguous(),
                    key_states.contiguous(),
                    block_size=block_size,
                    attention_sink=attention_sink,
                    window_size=window_size,
                    alpha=alpha,
                    last_n_block_full=last_n_block,
                    min_budget=min_budget,
                )
            ),
            enabled=True,
        )
        _, block_index, counts = selection_outputs
        keep_block, keep_block_kv_preprojected = _flashprefill_keep_block_from_selection(
            block_index=block_index,
            counts=counts,
            q_len=q_len,
            kv_len=kv_len,
            block_size=block_size,
            num_key_value_groups=num_key_value_groups,
        )
        active = _flashprefill_active_mask_from_selection(block_index, counts)
        (profile_stats, profile_stats_ms) = _cuda_elapsed_ms(
            lambda: _flashprefill_selection_profile_stats(
                active=active,
                keep_block=keep_block,
                keep_block_kv_preprojected=keep_block_kv_preprojected,
                q_len=q_len,
                kv_len=kv_len,
                block_size=block_size,
            ),
            enabled=measure_timing and query_states.is_cuda,
        )
        selected_blocks = float(keep_block.sum().item())
        selected_blocks_pre_union = float(counts.sum().item())
        valid_blocks = float(keep_block.numel())
        query_blocks = float(block_index.shape[1])
        key_blocks = float(block_index.shape[2])
        estimated_attention_ms = max(float(fused_total_ms) - float(select_replay_ms), 0.0)

    stats = {
        "repeat_kv_ms": 0.0,
        "upad_input_ms": 0.0,
        "pad_output_ms": 0.0,
        "dense_kernel_ms": 0.0,
        "gather_pack_ms": 0.0,
        "fallback_used": 0.0,
        "flashprefill_fused_total_ms": float(fused_total_ms),
        "flashprefill_kernel_ms": float(fused_total_ms),
        "flashprefill_select_replay_ms": float(select_replay_ms),
        "flashprefill_estimated_attention_ms": float(estimated_attention_ms),
        "flashprefill_selected_blocks": float(selected_blocks),
        "flashprefill_selected_blocks_pre_union": float(selected_blocks_pre_union),
        "valid_blocks": float(valid_blocks),
        "select_ratio": float(selected_blocks / valid_blocks) if valid_blocks > 0 else 0.0,
        "flashprefill_profile_stats_ms": float(profile_stats_ms),
        "flashprefill_mean_k_cache_used": 1.0 if mean_k_cache is not None else 0.0,
        "flashprefill_query_blocks": float(query_blocks),
        "flashprefill_key_blocks": float(key_blocks),
    }
    stats.update(profile_stats)
    return out, stats


def flashprefill_compactattn_prefill_full_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    softmax_scale: float,
    num_key_value_groups: int,
    block_size: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block: int,
    min_budget: int,
    pack_impl: str,
    indexed_impl: str,
    cache_fill_backend: str,
    measure_timing: bool = False,
    attn_module=None,
    k_hf: Optional[torch.Tensor] = None,
    v_hf: Optional[torch.Tensor] = None,
    mean_k_cache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    q_len = int(query_states.shape[1])
    kv_len = int(key_states.shape[1])

    if q_len <= 1 or kv_len <= q_len:
        out, stats = _flashprefill_dense_fallback(
            query_states,
            key_states,
            value_states,
            attention_mask,
            softmax_scale=softmax_scale,
            num_key_value_groups=num_key_value_groups,
            measure_timing=measure_timing,
            attn_module=attn_module,
        )
        stats = dict(stats)
        stats.update(
            {
                "flashprefill_select_ms": 0.0,
                "flashprefill_selected_blocks": 0.0,
                "flashprefill_selected_blocks_pre_union": 0.0,
                "flashprefill_pre_union_past_density": 0.0,
                "flashprefill_post_query_union_past_density": 0.0,
                "flashprefill_post_kv_union_past_density": 0.0,
                "flashprefill_current_chunk_open_density": 0.0,
                "flashprefill_pre_union_selected_blocks_past": 0.0,
                "flashprefill_post_query_union_selected_blocks_past": 0.0,
                "flashprefill_post_kv_union_selected_blocks_past": 0.0,
                "flashprefill_query_block_pairwise_jaccard_mean": 0.0,
                "flashprefill_query_blocks": 0.0,
                "flashprefill_key_blocks": 0.0,
            }
        )
        return out, stats

    if attention_mask is not None:
        mask = attention_mask.to(device=query_states.device)
        if (mask == 0).any():
            return _flashprefill_dense_fallback(
                query_states,
                key_states,
                value_states,
                attention_mask,
                softmax_scale=softmax_scale,
                num_key_value_groups=num_key_value_groups,
                measure_timing=measure_timing,
                attn_module=attn_module,
            )

    if not query_states.is_cuda or query_states.dtype not in (torch.float16, torch.bfloat16):
        return _flashprefill_dense_fallback(
            query_states,
            key_states,
            value_states,
            attention_mask,
            softmax_scale=softmax_scale,
            num_key_value_groups=num_key_value_groups,
            measure_timing=measure_timing,
            attn_module=attn_module,
        )

    _maybe_print_flashprefill_debug(
        ("q", query_states),
        ("k", key_states),
        ("v", value_states),
    )

    (select_inputs, select_input_layout_ms) = _cuda_elapsed_ms(
        lambda: (
            query_states.contiguous(),
            key_states
            if mean_k_cache is not None or _flashprefill_use_strided_select_k()
            else key_states.contiguous(),
        ),
        enabled=measure_timing and query_states.is_cuda,
    )
    select_q, select_k = select_inputs
    use_subgroup_union = indexed_impl == "fi_zero_copy_subgroup"
    use_fused_kv_union = (
        _flashprefill_use_fused_kv_union()
        and indexed_impl != "fi_zero_copy_per_query"
        and int(min_budget) == 0
        and (
            use_subgroup_union
            or not measure_timing
            or os.environ.get("SEER_FLASHPREFILL_DISABLE_PROFILE_STATS", "0") == "1"
        )
    )
    if use_fused_kv_union:
        (output_score, select_ms) = _cuda_elapsed_ms(
            lambda: (
                flash_prefill_score_chunked_from_mean_k(
                    select_q,
                    mean_k_cache.contiguous(),
                    key_len=kv_len,
                    block_size=block_size,
                )
                if mean_k_cache is not None
                else flash_prefill_score_chunked(
                    select_q,
                    select_k,
                    block_size=block_size,
                )
            ),
            enabled=measure_timing and query_states.is_cuda,
        )
        block_index = None
        counts = None
    else:
        (selection_outputs, select_ms) = _cuda_elapsed_ms(
            lambda: (
                flash_prefill_select_chunked_from_mean_k(
                    select_q,
                    mean_k_cache.contiguous(),
                    key_len=kv_len,
                    block_size=block_size,
                    attention_sink=attention_sink,
                    window_size=window_size,
                    alpha=alpha,
                    last_n_block_full=last_n_block,
                    min_budget=min_budget,
                )
                if mean_k_cache is not None
                else flash_prefill_select_chunked(
                    select_q,
                    select_k,
                    block_size=block_size,
                    attention_sink=attention_sink,
                    window_size=window_size,
                    alpha=alpha,
                    last_n_block_full=last_n_block,
                    min_budget=min_budget,
                )
            ),
            enabled=measure_timing and query_states.is_cuda,
        )
        output_score, block_index, counts = selection_outputs
    (keep_outputs, union_ms) = _cuda_elapsed_ms(
        lambda: (
            (
                None,
                (
                    _flashprefill_keep_block_subgroup_from_scores_fused(
                        output_score=output_score,
                        q_len=q_len,
                        kv_len=kv_len,
                        block_size=block_size,
                        num_key_value_groups=num_key_value_groups,
                        attention_sink=attention_sink,
                        window_size=window_size,
                        alpha=alpha,
                        last_n_block=last_n_block,
                        query_subgroup_size=_flashprefill_query_subgroup_size(
                            num_key_value_groups
                        ),
                    )
                    if use_subgroup_union
                    else _flashprefill_keep_block_kv_from_scores_fused(
                        output_score=output_score,
                        q_len=q_len,
                        kv_len=kv_len,
                        block_size=block_size,
                        num_key_value_groups=num_key_value_groups,
                        attention_sink=attention_sink,
                        window_size=window_size,
                        alpha=alpha,
                        last_n_block=last_n_block,
                    )
                ),
            )
            if use_fused_kv_union
            else _flashprefill_keep_block_from_scores_rowwise(
                output_score=output_score,
                q_len=q_len,
                kv_len=kv_len,
                block_size=block_size,
                num_key_value_groups=num_key_value_groups,
                attention_sink=attention_sink,
                window_size=window_size,
                alpha=alpha,
                last_n_block=last_n_block,
                min_budget=min_budget,
            )
        ),
        enabled=measure_timing and query_states.is_cuda,
    )
    keep_block, keep_block_kv_preprojected = keep_outputs
    profile_stats = {}
    profile_stats_ms = 0.0
    if measure_timing and os.environ.get("SEER_FLASHPREFILL_DISABLE_PROFILE_STATS", "0") != "1":
        if counts is not None and block_index is not None:
            active = _flashprefill_active_mask_from_selection(block_index, counts)
            (profile_stats, profile_stats_ms) = _cuda_elapsed_ms(
                lambda: _flashprefill_selection_profile_stats(
                    active=active,
                    keep_block=keep_block,
                    keep_block_kv_preprojected=keep_block_kv_preprojected,
                    q_len=q_len,
                    kv_len=kv_len,
                    block_size=block_size,
                ),
                enabled=query_states.is_cuda,
            )
        else:
            (profile_stats, profile_stats_ms) = _cuda_elapsed_ms(
                lambda: _flashprefill_selection_profile_stats_from_scores(
                    output_score=output_score,
                    keep_block_kv_preprojected=keep_block_kv_preprojected,
                    q_len=q_len,
                    kv_len=kv_len,
                    block_size=block_size,
                    num_key_value_groups=num_key_value_groups,
                    attention_sink=attention_sink,
                    window_size=window_size,
                    alpha=alpha,
                    last_n_block=last_n_block,
                    query_subgroup_size=(
                        _flashprefill_query_subgroup_size(num_key_value_groups)
                        if use_subgroup_union
                        else None
                    ),
                ),
                enabled=query_states.is_cuda,
            )

    if indexed_impl in {"fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"}:
        if k_hf is None or v_hf is None:
            k_hf = key_states.transpose(1, 2).contiguous()
            v_hf = value_states.transpose(1, 2).contiguous()

    attn_output, compactattn_stats = chunked_prefill_column_dense_attention_from_keep_block(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        keep_block=keep_block,
        attention_mask=attention_mask,
        query_length=q_len,
        softmax_scale=softmax_scale,
        block_size=block_size,
        num_key_value_groups=num_key_value_groups,
        pack_impl=pack_impl,
        indexed_impl=indexed_impl,
        cache_fill_backend=cache_fill_backend,
        return_stats=measure_timing,
        valid_block=None,
        allow_indexed_dense=True,
        keep_block_kv_preprojected=keep_block_kv_preprojected,
        k_hf=k_hf,
        v_hf=v_hf,
    )

    if not measure_timing or compactattn_stats is None:
        if not measure_timing:
            return attn_output, {}
        stats = {
            "flashprefill_select_ms": float(select_ms),
            "flashprefill_select_input_layout_ms": float(select_input_layout_ms),
            "flashprefill_selected_blocks": float(keep_block_kv_preprojected.sum().item()),
            "flashprefill_selected_blocks_pre_union": float(counts.sum().item()) if counts is not None else 0.0,
            "flashprefill_union_ms": float(union_ms),
            "flashprefill_fused_kv_union_calls": float(1.0 if use_fused_kv_union else 0.0),
            "flashprefill_mean_k_cache_used": 1.0 if mean_k_cache is not None else 0.0,
            "flashprefill_query_blocks": float(output_score.shape[1]),
            "flashprefill_key_blocks": float(output_score.shape[2]),
            "flashprefill_pooled_query_blocks": float(max(output_score.shape[1] - min(max(last_n_block, 0), output_score.shape[1] - 1) if output_score.shape[1] > 1 else 1, 1)),
            "flashprefill_profile_stats_ms": float(profile_stats_ms),
        }
        stats.update(profile_stats)
        return attn_output, stats

    stats = dict(compactattn_stats)
    stats.update(
        {
            "flashprefill_select_ms": float(select_ms),
            "flashprefill_select_input_layout_ms": float(select_input_layout_ms),
            "flashprefill_union_ms": float(union_ms),
            "flashprefill_fused_kv_union_calls": float(1.0 if use_fused_kv_union else 0.0),
            "flashprefill_mean_k_cache_used": 1.0 if mean_k_cache is not None else 0.0,
            "flashprefill_selected_blocks": float(keep_block_kv_preprojected.sum().item()),
            "flashprefill_selected_blocks_pre_union": float(counts.sum().item()) if counts is not None else 0.0,
            "flashprefill_query_blocks": float(output_score.shape[1]),
            "flashprefill_key_blocks": float(output_score.shape[2]),
            "flashprefill_pooled_query_blocks": float(max(output_score.shape[1] - min(max(last_n_block, 0), output_score.shape[1] - 1) if output_score.shape[1] > 1 else 1, 1)),
            "flashprefill_profile_stats_ms": float(profile_stats_ms),
        }
    )
    stats.update(profile_stats)
    return attn_output, stats
