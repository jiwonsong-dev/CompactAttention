import os
os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
from typing import Callable, Dict, Optional, Tuple

import torch
from flash_attn import flash_attn_with_kvcache
from compact_attn.kernels.varlen.indexed_dense_cache_fill_kernels import (
    DEFAULT_CACHE_FILL_MIN_BLOCKS_FOR_TRITON,
    cache_fill_blocks_to_paged_kv_from_pos_with_rank,
    cache_fill_blocks_to_paged_kv_from_kv_strided,
    can_use_triton_cache_fill_from_pos_with_rank,
    can_use_triton_cache_fill_from_kv,
    select_cache_fill_launch_config,
    triton_available as cache_fill_triton_available,
)
from compact_attn.kernels.varlen.indexed_dense_cache_fill_cuda import (
    build_past_indices_and_metadata_from_keep_block_cuda_out,
    build_selected_indices_and_metadata_from_keep_block_cuda_out,
    build_block_table_cuda,
    cache_fill_from_selected_indices_row_tiled_cuda,
    cache_fill_current_tail_cuda,
    cache_fill_from_past_indices_compact_cuda,
    compact_keep_blocks_and_build_table_cuda_out,
    compact_keep_blocks_cuda_out,
    cache_fill_blocks_to_paged_kv_from_kv_strided_cuda,
    cache_fill_blocks_to_paged_kv_from_pos_with_local_rank_cuda,
    cache_fill_blocks_to_paged_kv_from_pos_with_rank_cuda,
    can_use_cuda_build_block_table,
    can_use_cuda_cache_fill_from_pos_with_local_rank,
    can_use_cuda_cache_fill_from_pos_with_rank,
    can_use_cuda_cache_fill_from_selected_indices_row_tiled,
    can_use_cuda_pack_q_for_indexed_prefill,
    can_use_cuda_cache_fill_from_row_blk_dst,
    can_use_cuda_compact_keep_blocks,
    can_use_cuda_compact_keep_blocks_and_build_table,
    can_use_cuda_keep_block_builder_fast,
    cuda_cache_fill_available,
    pack_q_for_indexed_prefill_cuda,
)


def indexed_dense_available() -> bool:
    return True


def _flashinfer_attention_backend() -> str:
    override = os.environ.get("SEER_FLASHINFER_ATTENTION_BACKEND", "").strip()
    if override:
        return override
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
            if major == 9:
                return "fa3"
        except Exception:
            pass
    return "fa2"


def can_use_indexed_dense_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_block_kv: torch.Tensor,
    block_size: int,
    page_block_size: int,
) -> bool:
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda and keep_block_kv.is_cuda):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if keep_block_kv.ndim != 3:
        return False
    if q.shape[-1] != 128:
        return False
    if k.shape[-1] != 128 or v.shape[-1] != 128:
        return False
    if page_block_size <= 0 or (page_block_size % 256) != 0:
        return False
    if block_size <= 0 or (page_block_size % block_size) != 0:
        return False
    bsz, q_len, num_q_heads, _ = q.shape
    if k.shape[0] != bsz or v.shape[0] != bsz:
        return False
    if k.shape[1] < q_len or v.shape[1] < q_len:
        return False
    if (k.shape[1] % block_size) != 0:
        return False
    if keep_block_kv.shape[0] != bsz:
        return False
    if keep_block_kv.shape[1] != k.shape[2]:
        return False
    if keep_block_kv.shape[2] != (k.shape[1] // block_size):
        return False
    if num_q_heads % k.shape[2] != 0:
        return False
    return True


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


def _debug_peak_mem_enabled() -> bool:
    raw = os.environ.get("SEERATTN_DEBUG_COMPACTATTN_PEAK_MEM", None)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _cuda_peak_memory_stats(fn, device: torch.device, enabled: bool = False):
    if not enabled or device.type != "cuda":
        return fn(), {}
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = fn()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    return out, {
        "alloc_before_mb": float(allocated_before) / (1024.0 * 1024.0),
        "alloc_after_mb": float(allocated_after) / (1024.0 * 1024.0),
        "alloc_delta_mb": float(allocated_after - allocated_before) / (1024.0 * 1024.0),
        "peak_alloc_mb": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
        "reserved_before_mb": float(reserved_before) / (1024.0 * 1024.0),
        "reserved_after_mb": float(reserved_after) / (1024.0 * 1024.0),
        "reserved_delta_mb": float(reserved_after - reserved_before) / (1024.0 * 1024.0),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0),
    }


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
            "alloc_before_mb": float(allocated_before) / (1024.0 * 1024.0),
            "alloc_after_mb": float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0),
            "alloc_delta_mb": float(torch.cuda.memory_allocated(device) - allocated_before) / (1024.0 * 1024.0),
            "peak_alloc_mb": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
            "reserved_before_mb": float(reserved_before) / (1024.0 * 1024.0),
            "reserved_after_mb": float(torch.cuda.memory_reserved(device)) / (1024.0 * 1024.0),
            "reserved_delta_mb": float(torch.cuda.memory_reserved(device) - reserved_before) / (1024.0 * 1024.0),
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0),
        }
    else:
        mem_stats = {}
    return out, elapsed_ms, mem_stats


_PAGED_KV_WORKSPACES: Dict[Tuple[int, str, int, int, int], Dict[str, torch.Tensor]] = {}
_INDEXED_ATTN_WORKSPACES: Dict[Tuple[int, str, int, int, int, int, int], Dict[str, torch.Tensor]] = {}
_INDEXED_BUILD_SCRATCH_WORKSPACES: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
_WORKSPACE_ALIGN_PAGES = 16
_WORKSPACE_MIN_CAPACITY_PAGES = 16
_WORKSPACE_HEADROOM_PAGES = 8
_WORKSPACE_MIN_GROW_PAGES = 16
_WORKSPACE_MAX_EXTRA_PAGES = 64


def _compactattn_enable_fused_builder_v2() -> bool:
    raw = os.environ.get("SEERATTN_DEBUG_COMPACTATTN_ENABLE_FUSED_BUILDER_V2", None)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def clear_indexed_dense_workspaces(device: Optional[torch.device] = None) -> Dict[str, int]:
    if device is None:
        paged = len(_PAGED_KV_WORKSPACES)
        attn = len(_INDEXED_ATTN_WORKSPACES)
        scratch = len(_INDEXED_BUILD_SCRATCH_WORKSPACES)
        _PAGED_KV_WORKSPACES.clear()
        _INDEXED_ATTN_WORKSPACES.clear()
        _INDEXED_BUILD_SCRATCH_WORKSPACES.clear()
        return {
            "paged_workspaces_cleared": int(paged),
            "attn_workspaces_cleared": int(attn),
            "scratch_workspaces_cleared": int(scratch),
        }

    device_idx = int(device.index) if device.index is not None else -1

    def _clear_matching(store: Dict, key_matches) -> int:
        removed = 0
        for key in list(store.keys()):
            if key_matches(key):
                del store[key]
                removed += 1
        return removed

    paged = _clear_matching(_PAGED_KV_WORKSPACES, lambda key: int(key[0]) == device_idx)
    attn = _clear_matching(_INDEXED_ATTN_WORKSPACES, lambda key: int(key[0]) == device_idx)
    scratch = _clear_matching(_INDEXED_BUILD_SCRATCH_WORKSPACES, lambda key: int(key[0]) == device_idx)
    return {
        "paged_workspaces_cleared": int(paged),
        "attn_workspaces_cleared": int(attn),
        "scratch_workspaces_cleared": int(scratch),
    }


def _align_up_pages(value: int, align: int) -> int:
    value = int(value)
    align = max(int(align), 1)
    return ((value + align - 1) // align) * align


def _next_workspace_capacity_pages(required_capacity: int, current_capacity: int) -> int:
    required_capacity = max(int(required_capacity), 1)
    current_capacity = max(int(current_capacity), 0)
    if current_capacity == 0:
        return _align_up_pages(
            max(required_capacity, _WORKSPACE_MIN_CAPACITY_PAGES),
            _WORKSPACE_ALIGN_PAGES,
        )

    grow_pages = max(current_capacity // 4, _WORKSPACE_MIN_GROW_PAGES)
    candidate = max(required_capacity + _WORKSPACE_HEADROOM_PAGES, current_capacity + grow_pages)
    candidate = min(candidate, required_capacity + _WORKSPACE_MAX_EXTRA_PAGES)
    new_capacity = _align_up_pages(candidate, _WORKSPACE_ALIGN_PAGES)
    return max(new_capacity, required_capacity)


def _get_or_create_paged_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    rows: int,
    max_pages_upper: int,
    total_pages_upper: int,
    page_block_size: int,
    head_dim: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    device_idx = int(device.index) if device.index is not None else -1
    key = (device_idx, str(dtype), int(rows), int(page_block_size), int(head_dim))
    ws = _PAGED_KV_WORKSPACES.get(key, None)
    current_block_table_capacity = 0 if ws is None else int(ws["block_table_capacity"])
    current_page_pool_capacity = 0 if ws is None else int(ws["page_pool_capacity"])
    required_block_table_capacity = max(int(max_pages_upper), 1)
    required_page_pool_capacity = max(int(total_pages_upper), 1)
    growth_events = 0.0
    if current_block_table_capacity >= required_block_table_capacity:
        new_block_table_capacity = current_block_table_capacity
    else:
        new_block_table_capacity = _next_workspace_capacity_pages(
            required_block_table_capacity, current_block_table_capacity
        )
        growth_events = 1.0
    if current_page_pool_capacity >= required_page_pool_capacity:
        new_page_pool_capacity = current_page_pool_capacity
    else:
        new_page_pool_capacity = _next_workspace_capacity_pages(
            required_page_pool_capacity, current_page_pool_capacity
        )
        growth_events = 1.0
    if growth_events > 0.0:
        ws = {
            "k_cache": torch.empty(
                (new_page_pool_capacity, page_block_size, 1, head_dim), device=device, dtype=dtype
            ),
            "v_cache": torch.empty(
                (new_page_pool_capacity, page_block_size, 1, head_dim), device=device, dtype=dtype
            ),
            "block_table_flat": torch.empty(
                (rows * new_block_table_capacity,), device=device, dtype=torch.int32
            ),
            "block_table_capacity": int(new_block_table_capacity),
            "page_pool_capacity": int(new_page_pool_capacity),
        }
        _PAGED_KV_WORKSPACES[key] = ws
    workspace_stats = {
        "workspace_capacity_pages": float(ws["block_table_capacity"]),
        "workspace_required_pages": float(required_block_table_capacity),
        "workspace_page_pool_capacity_pages": float(ws["page_pool_capacity"]),
        "workspace_page_pool_required_pages": float(required_page_pool_capacity),
        "workspace_growth_events": float(growth_events),
        "workspace_k_alloc_mb": float(ws["k_cache"].numel() * ws["k_cache"].element_size()) / (1024.0 * 1024.0),
        "workspace_v_alloc_mb": float(ws["v_cache"].numel() * ws["v_cache"].element_size()) / (1024.0 * 1024.0),
        "workspace_block_table_alloc_mb": float(
            ws["block_table_flat"].numel() * ws["block_table_flat"].element_size()
        )
        / (1024.0 * 1024.0),
    }
    return ws, workspace_stats


def _get_or_create_indexed_attn_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    bsz: int,
    q_len: int,
    num_kv_heads: int,
    num_key_value_groups: int,
    head_dim: int,
    use_out_flat: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    device_idx = int(device.index) if device.index is not None else -1
    key = (
        device_idx,
        str(dtype),
        int(bsz),
        int(q_len),
        int(num_kv_heads),
        int(num_key_value_groups),
        int(head_dim),
    )
    ws = _INDEXED_ATTN_WORKSPACES.get(key, None)
    if ws is None:
        num_q_heads = int(num_kv_heads) * int(num_key_value_groups)
        ws = {
            "q_group_5d": torch.empty(
                (bsz, num_kv_heads, q_len, num_key_value_groups, head_dim),
                device=device,
                dtype=dtype,
            ),
        }
        if use_out_flat:
            ws["out_flat"] = torch.empty(
                (bsz, q_len, num_q_heads, head_dim),
                device=device,
                dtype=dtype,
            )
        _INDEXED_ATTN_WORKSPACES[key] = ws
        growth_events = 1.0
    else:
        num_q_heads = int(num_kv_heads) * int(num_key_value_groups)
        if use_out_flat and "out_flat" not in ws:
            ws["out_flat"] = torch.empty(
                (bsz, q_len, num_q_heads, head_dim),
                device=device,
                dtype=dtype,
            )
        growth_events = 0.0
    workspace_stats = {
        "attn_q_workspace_alloc_mb": float(
            ws["q_group_5d"].numel() * ws["q_group_5d"].element_size()
        )
        / (1024.0 * 1024.0),
        "attn_out_workspace_alloc_mb": float(
            ws["out_flat"].numel() * ws["out_flat"].element_size()
            if "out_flat" in ws
            else 0
        )
        / (1024.0 * 1024.0),
        "attn_q_workspace_growth_events": float(growth_events),
    }
    return ws, workspace_stats


def _next_scratch_capacity(required_capacity: int, current_capacity: int, *, align: int, minimum: int) -> int:
    required_capacity = max(int(required_capacity), 1)
    current_capacity = max(int(current_capacity), 0)
    if current_capacity == 0:
        return _align_up_pages(max(required_capacity, minimum), align)
    grow = max(current_capacity // 4, align)
    candidate = max(required_capacity, current_capacity + grow)
    return _align_up_pages(candidate, align)


def _get_or_create_indexed_build_scratch_workspace(
    *,
    device: torch.device,
    rows: int,
    kv_blocks: int,
    max_pages_upper: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    device_idx = int(device.index) if device.index is not None else -1
    key = (device_idx, int(rows))
    ws = _INDEXED_BUILD_SCRATCH_WORKSPACES.get(key, None)
    current_kv_capacity = 0 if ws is None else int(ws["kv_blocks_capacity"])
    current_pages_capacity = 0 if ws is None else int(ws["max_pages_capacity"])
    required_kv_capacity = max(int(kv_blocks), 1)
    required_pages_capacity = max(int(max_pages_upper), 1)

    growth_events = 0.0
    kv_capacity = current_kv_capacity
    pages_capacity = current_pages_capacity
    if current_kv_capacity < required_kv_capacity:
        kv_capacity = _next_scratch_capacity(required_kv_capacity, current_kv_capacity, align=64, minimum=64)
        growth_events = 1.0
    if current_pages_capacity < required_pages_capacity:
        pages_capacity = _next_scratch_capacity(required_pages_capacity, current_pages_capacity, align=16, minimum=16)
        growth_events = 1.0

    if growth_events > 0.0:
        max_sel = max(int(rows) * kv_capacity, 1)
        ws = {
            "sel_blocks_i32": torch.empty((rows,), device=device, dtype=torch.int32),
            "pages_per_row_i32": torch.empty((rows,), device=device, dtype=torch.int32),
            "page_offsets_i32": torch.empty((rows,), device=device, dtype=torch.int32),
            "cache_seqlens_i32": torch.empty((rows,), device=device, dtype=torch.int32),
            "selected_offsets_i32": torch.empty((rows,), device=device, dtype=torch.int32),
            "past_block_indices_i32": torch.empty((rows, kv_capacity), device=device, dtype=torch.int32),
            "pos_i64": torch.empty((max_sel, 2), device=device, dtype=torch.int64),
            "local_rank_i32": torch.empty((max_sel,), device=device, dtype=torch.int32),
            "kv_blocks_capacity": int(kv_capacity),
            "max_pages_capacity": int(pages_capacity),
        }
        _INDEXED_BUILD_SCRATCH_WORKSPACES[key] = ws

    meta_tensors = (
        ws["sel_blocks_i32"],
        ws["pages_per_row_i32"],
        ws["page_offsets_i32"],
        ws["cache_seqlens_i32"],
        ws["selected_offsets_i32"],
        ws["past_block_indices_i32"],
    )
    workspace_stats = {
        "scratch_pool_growth_events": float(growth_events),
        "scratch_pool_pos_alloc_mb": float(
            (ws["pos_i64"].numel() * ws["pos_i64"].element_size())
            + (ws["local_rank_i32"].numel() * ws["local_rank_i32"].element_size())
        )
        / (1024.0 * 1024.0),
        "scratch_pool_meta_alloc_mb": float(
            sum(t.numel() * t.element_size() for t in meta_tensors)
        )
        / (1024.0 * 1024.0),
        "scratch_pool_kv_blocks_capacity": float(ws["kv_blocks_capacity"]),
        "scratch_pool_max_pages_capacity": float(ws["max_pages_capacity"]),
    }
    return ws, workspace_stats


def build_paged_kv_cache_from_keep_block_fast(
    k: torch.Tensor,  # [B, K, Hkv, D]
    v: torch.Tensor,  # [B, K, Hkv, D]
    keep_block: torch.Tensor,  # [B, Hq, Kb]
    num_key_value_groups: int,
    q_len: int,
    block_size: int = 64,
    page_block_size: int = 256,
    cache_fill_backend: str = "auto",
    prefer_fused_builder_v2: bool = False,
    measure_timing: bool = True,
    stage_callback: Optional[Callable[[str, Optional[Dict[str, float]]], None]] = None,
):
    bsz, kv_len, num_kv_heads, head_dim = k.shape
    rows = bsz * num_kv_heads
    device = k.device
    kv_blocks = kv_len // block_size
    past_len = kv_len - q_len
    past_k_blocks = past_len // block_size
    curr_k_blocks = kv_blocks - past_k_blocks
    max_pages_upper = max((kv_len + page_block_size - 1) // page_block_size, 1)

    build_ws, build_workspace_stats = _get_or_create_indexed_build_scratch_workspace(
        device=device,
        rows=rows,
        kv_blocks=kv_blocks,
        max_pages_upper=max_pages_upper,
    )
    page_offsets = build_ws["page_offsets_i32"]
    past_block_indices = build_ws["past_block_indices_i32"]
    past_block_counts = build_ws["sel_blocks_i32"]
    pages_per_row = build_ws["pages_per_row_i32"]
    cache_seqlens = build_ws["cache_seqlens_i32"]
    selected_offsets = build_ws["selected_offsets_i32"]

    def _run_meta():
        if prefer_fused_builder_v2:
            build_selected_indices_and_metadata_from_keep_block_cuda_out(
                keep_block=keep_block,
                num_key_value_groups=num_key_value_groups,
                past_k_blocks=past_k_blocks,
                curr_k_blocks=curr_k_blocks,
                block_size=block_size,
                page_block_size=page_block_size,
                selected_block_indices=past_block_indices,
                selected_block_counts=past_block_counts,
                pages_per_row=pages_per_row,
                cache_seqlens=cache_seqlens,
            )
        else:
            build_past_indices_and_metadata_from_keep_block_cuda_out(
                keep_block=keep_block,
                num_key_value_groups=num_key_value_groups,
                past_k_blocks=past_k_blocks,
                curr_k_blocks=curr_k_blocks,
                block_size=block_size,
                page_block_size=page_block_size,
                past_block_indices=past_block_indices,
                past_block_counts=past_block_counts,
                pages_per_row=pages_per_row,
                cache_seqlens=cache_seqlens,
            )
        return None

    _, fast_build_ms = _cuda_elapsed_ms(_run_meta, enabled=measure_timing)

    # Use upper bounds to avoid host-device synchronization (.item() calls).
    # Workspace is cached and grows monotonically, so over-allocation is a
    # one-time cost.  flash_attn_with_kvcache only accesses pages referenced
    # in block_table (bounded by cache_seqlens), so extra capacity is safe.
    max_pages_used = max_pages_upper
    total_pages_used = rows * max_pages_upper
    ws, workspace_stats = _get_or_create_paged_workspace(
        device=device,
        dtype=k.dtype,
        rows=rows,
        max_pages_upper=max_pages_used,
        total_pages_upper=total_pages_used,
        page_block_size=page_block_size,
        head_dim=head_dim,
    )
    k_cache = ws["k_cache"][:total_pages_used]
    v_cache = ws["v_cache"][:total_pages_used]
    block_table = ws["block_table_flat"][: rows * max_pages_used].view(rows, max_pages_used)

    use_row_tiled_v2 = (
        prefer_fused_builder_v2
        and cache_fill_backend == "cuda"
        and bsz > 1
        and block_size == 64
        and page_block_size == 256
        and head_dim == 128
        and can_use_cuda_cache_fill_from_selected_indices_row_tiled(
            k=k,
            v=v,
            selected_block_indices=past_block_indices,
            selected_block_counts=past_block_counts,
            page_offsets=page_offsets,
            k_cache_flat=k_cache.reshape(-1, head_dim),
            v_cache_flat=v_cache.reshape(-1, head_dim),
            block_size=block_size,
            page_block_size=page_block_size,
        )
    )

    _, table_fill_ms = _cuda_elapsed_ms(
        lambda: build_block_table_cuda(
            pages_per_row=pages_per_row,
            block_table=block_table,
            page_offsets=page_offsets,
        ),
        enabled=measure_timing,
    )
    table_kernel_ms = table_fill_ms

    if stage_callback is not None:
        stage_callback(
            "block_table_done",
            {
                "rows": float(rows),
                "kv_blocks": float(kv_blocks),
                "table_fill_ms": float(table_fill_ms),
            },
        )

    if past_block_counts.numel() > 0 and not use_row_tiled_v2:
        torch.cumsum(past_block_counts, dim=0, dtype=torch.int32, out=selected_offsets)
        selected_offsets.sub_(past_block_counts)
    else:
        selected_offsets.zero_()
    if use_row_tiled_v2:
        row_tiled_blocks_per_tile = 4
        if past_block_indices.size(1) >= 1024:
            row_tiled_workers_per_row = 16
        elif past_block_indices.size(1) >= 512:
            row_tiled_workers_per_row = 8
        else:
            row_tiled_workers_per_row = 4
        num_sel_blocks = rows * row_tiled_workers_per_row
        launch_config = {
            "small_calls": 0.0,
            "medium_calls": 0.0,
            "large_calls": 0.0,
            "variant_id": 30.0,
            "tuned_calls": 0.0,
        }
    else:
        row_tiled_blocks_per_tile = 0
        row_tiled_workers_per_row = 0
        if past_block_counts.numel() > 0:
            # Use upper bound to avoid a host-device sync
            # from .item().  The CUDA cache-fill kernel checks bounds per thread
            # block, so excess launches are harmless (immediate early-return).
            num_sel_blocks = rows * (kv_blocks if prefer_fused_builder_v2 else past_k_blocks)
        else:
            num_sel_blocks = 0
        launch_config = select_cache_fill_launch_config(
            num_sel_blocks=max(num_sel_blocks, 1),
            block_size=int(block_size),
            head_dim=int(head_dim),
            dtype=k.dtype,
        )

    def _run_fill_cache_backend():
        if use_row_tiled_v2:
            _, past_kernel_ms = _cuda_elapsed_ms(
                lambda: cache_fill_from_selected_indices_row_tiled_cuda(
                    k=k,
                    v=v,
                    selected_block_indices=past_block_indices,
                    selected_block_counts=past_block_counts,
                    page_offsets=page_offsets,
                    k_cache_flat=k_cache.reshape(-1, head_dim),
                    v_cache_flat=v_cache.reshape(-1, head_dim),
                    block_size=block_size,
                    page_block_size=page_block_size,
                ),
                enabled=measure_timing,
            )
        else:
            _, past_kernel_ms = _cuda_elapsed_ms(
                lambda: cache_fill_from_past_indices_compact_cuda(
                    k=k,
                    v=v,
                    past_block_indices=past_block_indices,
                    past_block_counts=past_block_counts,
                    selected_offsets=selected_offsets,
                    page_offsets=page_offsets,
                    k_cache_flat=k_cache.reshape(-1, head_dim),
                    v_cache_flat=v_cache.reshape(-1, head_dim),
                    total_selected=num_sel_blocks,
                    block_size=block_size,
                    page_block_size=page_block_size,
                ),
                enabled=measure_timing,
            )
        current_kernel_ms = 0.0
        if not prefer_fused_builder_v2:
            _, current_kernel_ms = _cuda_elapsed_ms(
                lambda: cache_fill_current_tail_cuda(
                    k=k,
                    v=v,
                    past_block_counts=past_block_counts,
                    page_offsets=page_offsets,
                    k_cache_flat=k_cache.reshape(-1, head_dim),
                    v_cache_flat=v_cache.reshape(-1, head_dim),
                    past_k_blocks=past_k_blocks,
                    curr_k_blocks=curr_k_blocks,
                    block_size=block_size,
                    page_block_size=page_block_size,
                ),
                enabled=measure_timing,
            )
        return {
            "index_cache_fill_cuda_calls": 1.0,
            "index_cache_fill_cuda_fallback_calls": 0.0,
            "index_cache_fill_backend_id": 3.0,
            "index_cache_fill_triton_calls": 0.0,
            "index_cache_fill_torch_calls": 0.0,
            "index_cache_fill_fallback_calls": 0.0,
            "index_cache_fill_kernel_ms": float(past_kernel_ms + current_kernel_ms),
            "index_cache_fill_small_calls": float(launch_config["small_calls"]),
            "index_cache_fill_medium_calls": float(launch_config["medium_calls"]),
            "index_cache_fill_large_calls": float(launch_config["large_calls"]),
            "index_cache_fill_variant_id": float(launch_config["variant_id"]) + 10.0,
            "index_cache_fill_tuned_calls": float(launch_config["tuned_calls"]),
            "index_cache_fill_current_tail_ms": float(current_kernel_ms),
            "index_cache_fill_launch_blocks": float(num_sel_blocks),
            "index_cache_fill_effective_blocks": float(num_sel_blocks),
        }

    cache_fill_backend, cache_fill_ms = _cuda_elapsed_ms(
        _run_fill_cache_backend, enabled=measure_timing
    )
    if stage_callback is not None:
        stage_callback(
            "cache_fill_done",
            {
                "cache_fill_ms": float(cache_fill_ms),
                "cache_fill_kernel_ms": float(cache_fill_backend.get("index_cache_fill_kernel_ms", 0.0)),
                "cache_fill_triton_calls": 0.0,
                "cache_fill_torch_calls": 0.0,
                "cache_fill_fallback_calls": 0.0,
            },
        )

    timings = {
        "index_block_table_ms": table_fill_ms,
        "index_table_fill_ms": table_fill_ms,
        "index_table_kernel_ms": table_kernel_ms,
        "index_compact_ms": 0.0,
        "index_compact_kernel_ms": 0.0,
        "index_compact_nonzero_ms": 0.0,
        "index_compact_post_ms": 0.0,
        "index_compact_fused_calls": 1.0,
        "index_compact_fallback_calls": 0.0,
        "index_compact_fused_post_ms": 0.0,
        "index_src_layout_ms": 0.0,
        "index_cache_fill_ms": cache_fill_ms,
        "index_cache_fill_kernel_ms": float(cache_fill_backend.get("index_cache_fill_kernel_ms", 0.0)),
        "index_build_ms": fast_build_ms + table_fill_ms + cache_fill_ms,
        "index_union_block_ms": fast_build_ms,
        "index_cache_fill_cuda_calls": float(cache_fill_backend.get("index_cache_fill_cuda_calls", 0.0)),
        "index_cache_fill_cuda_fallback_calls": float(
            cache_fill_backend.get("index_cache_fill_cuda_fallback_calls", 0.0)
        ),
        "index_cache_fill_backend_id": float(cache_fill_backend.get("index_cache_fill_backend_id", 0.0)),
        "index_cache_fill_triton_calls": float(cache_fill_backend.get("index_cache_fill_triton_calls", 0.0)),
        "index_cache_fill_torch_calls": float(cache_fill_backend.get("index_cache_fill_torch_calls", 0.0)),
        "index_cache_fill_fallback_calls": float(cache_fill_backend.get("index_cache_fill_fallback_calls", 0.0)),
        "index_cache_fill_small_calls": float(cache_fill_backend.get("index_cache_fill_small_calls", 0.0)),
        "index_cache_fill_medium_calls": float(cache_fill_backend.get("index_cache_fill_medium_calls", 0.0)),
        "index_cache_fill_large_calls": float(cache_fill_backend.get("index_cache_fill_large_calls", 0.0)),
        "index_cache_fill_variant_id": float(cache_fill_backend.get("index_cache_fill_variant_id", 0.0)),
        "index_cache_fill_tuned_calls": float(cache_fill_backend.get("index_cache_fill_tuned_calls", 0.0)),
        "index_cache_fill_current_tail_ms": float(cache_fill_backend.get("index_cache_fill_current_tail_ms", 0.0)),
        "index_cache_fill_launch_blocks": float(cache_fill_backend.get("index_cache_fill_launch_blocks", 0.0)),
        "index_cache_fill_effective_blocks": float(cache_fill_backend.get("index_cache_fill_effective_blocks", 0.0)),
        "workspace_capacity_pages": float(workspace_stats["workspace_capacity_pages"]),
        "workspace_required_pages": float(workspace_stats["workspace_required_pages"]),
        "workspace_page_pool_capacity_pages": float(workspace_stats["workspace_page_pool_capacity_pages"]),
        "workspace_page_pool_required_pages": float(workspace_stats["workspace_page_pool_required_pages"]),
        "workspace_growth_events": float(workspace_stats["workspace_growth_events"]),
        "workspace_k_alloc_mb": float(workspace_stats["workspace_k_alloc_mb"]),
        "workspace_v_alloc_mb": float(workspace_stats["workspace_v_alloc_mb"]),
        "workspace_block_table_alloc_mb": float(workspace_stats["workspace_block_table_alloc_mb"]),
        "workspace_total_pages_used": float(total_pages_used),
        "workspace_max_pages_used": float(max_pages_used),
        "scratch_pool_growth_events": float(build_workspace_stats["scratch_pool_growth_events"]),
        "scratch_pool_pos_alloc_mb": float(build_workspace_stats["scratch_pool_pos_alloc_mb"]),
        "scratch_pool_meta_alloc_mb": float(build_workspace_stats["scratch_pool_meta_alloc_mb"]),
        "fused_builder_calls": 1.0 if prefer_fused_builder_v2 else 0.0,
        "fused_builder_ms": float(fast_build_ms + table_fill_ms + cache_fill_ms) if prefer_fused_builder_v2 else 0.0,
        "fused_builder_table_ms": float(table_fill_ms) if prefer_fused_builder_v2 else 0.0,
        "fused_builder_fill_ms": float(cache_fill_ms) if prefer_fused_builder_v2 else 0.0,
        "fused_builder_row_tiled_calls": 1.0 if use_row_tiled_v2 else 0.0,
        "fused_builder_tail_fused_calls": 1.0 if use_row_tiled_v2 else 0.0,
        "selected_kv_materialize_calls": 0.0 if prefer_fused_builder_v2 else 1.0,
        "selected_kv_materialize_ms": 0.0 if prefer_fused_builder_v2 else float(cache_fill_ms),
    }
    return k_cache, v_cache, block_table, cache_seqlens.to(torch.int32), timings


def build_paged_kv_cache_from_block_mask(
    k: torch.Tensor,  # [B, K, Hkv, D]
    v: torch.Tensor,  # [B, K, Hkv, D]
    keep_block_kv: torch.Tensor,  # [B, Hkv, Kb]
    block_size: int = 64,
    page_block_size: int = 256,
    cache_fill_impl: str = "auto",
    cache_fill_backend: str = "auto",
    cache_fill_min_blocks_for_triton: int = DEFAULT_CACHE_FILL_MIN_BLOCKS_FOR_TRITON,
    prefer_fused_builder_v2: bool = False,
    measure_timing: bool = True,
    stage_callback: Optional[Callable[[str, Optional[Dict[str, float]]], None]] = None,
):
    bsz, kv_len, num_kv_heads, head_dim = k.shape
    rows = bsz * num_kv_heads
    device = k.device
    kv_blocks = kv_len // block_size

    keep_flat, keep_flat_sanitize_ms = _cuda_elapsed_ms(
        lambda: keep_block_kv.reshape(rows, kv_blocks).to(torch.bool),
        enabled=measure_timing,
    )
    max_pages_upper = max((kv_len + page_block_size - 1) // page_block_size, 1)
    build_ws, build_workspace_stats = _get_or_create_indexed_build_scratch_workspace(
        device=device,
        rows=rows,
        kv_blocks=kv_blocks,
        max_pages_upper=max_pages_upper,
    )
    sel_blocks = build_ws["sel_blocks_i32"]
    _, sel_blocks_sum_ms = _cuda_elapsed_ms(
        lambda: torch.sum(keep_flat, dim=-1, dtype=torch.int32, out=sel_blocks),
        enabled=measure_timing,
    )
    zero_rows = sel_blocks == 0
    zero_row_fix_ms = 0.0
    if zero_rows.any():
        def _run_zero_row_fix():
            keep_flat[zero_rows, kv_blocks - 1] = True
            torch.sum(keep_flat, dim=-1, dtype=torch.int32, out=sel_blocks)
        _, zero_row_fix_ms = _cuda_elapsed_ms(_run_zero_row_fix, enabled=measure_timing)
    sel_lens = build_ws["cache_seqlens_i32"]
    def _run_sel_lens():
        sel_lens.copy_(sel_blocks)
        sel_lens.mul_(int(block_size))
    _, sel_lens_ms = _cuda_elapsed_ms(_run_sel_lens, enabled=measure_timing)

    pages_per_row = build_ws["pages_per_row_i32"]
    def _run_pages_per_row():
        pages_per_row.copy_(sel_lens)
        pages_per_row.add_(int(page_block_size - 1))
        torch.div(pages_per_row, int(page_block_size), rounding_mode="floor", out=pages_per_row)
    _, pages_per_row_ms = _cuda_elapsed_ms(_run_pages_per_row, enabled=measure_timing)
    # Use upper bounds to avoid host-device synchronization.
    max_pages_used = max_pages_upper
    total_pages_used = rows * max_pages_upper
    page_count_stats_ms = 0.0
    ws, workspace_stats = _get_or_create_paged_workspace(
        device=device,
        dtype=k.dtype,
        rows=rows,
        max_pages_upper=max_pages_used,
        total_pages_upper=total_pages_used,
        page_block_size=page_block_size,
        head_dim=head_dim,
    )
    k_cache = ws["k_cache"][:total_pages_used]
    v_cache = ws["v_cache"][:total_pages_used]
    block_table = ws["block_table_flat"][: rows * max_pages_used].view(rows, max_pages_used)
    page_slot = torch.arange(max_pages_used, device=device, dtype=torch.int32).unsqueeze(0)
    page_offsets = build_ws["page_offsets_i32"]
    pos_buf = build_ws["pos_i64"]
    local_rank_buf = build_ws["local_rank_i32"]

    backend_req = str(cache_fill_backend or cache_fill_impl or "auto")
    if backend_req not in {"auto", "cuda", "triton"}:
        backend_req = "auto"

    use_cuda_compact = backend_req in {"auto", "cuda"} and can_use_cuda_compact_keep_blocks(
        keep_flat=keep_flat,
        sel_blocks=sel_blocks,
    )
    use_cuda_fused_builder = backend_req in {"auto", "cuda"} and can_use_cuda_compact_keep_blocks_and_build_table(
        keep_flat=keep_flat,
        sel_blocks=sel_blocks,
        pages_per_row=pages_per_row,
        block_table=block_table,
        page_offsets=page_offsets,
    )
    prefer_cuda_builder = (not use_cuda_fused_builder) and backend_req in {"auto", "cuda"} and can_use_cuda_build_block_table(
        pages_per_row=pages_per_row,
        block_table=block_table,
        page_offsets=page_offsets,
    )

    keep_prefix_rank = None
    local_rank = None
    compact_nonzero_ms = 0.0
    compact_post_ms = 0.0
    compact_fused_post_ms = 0.0
    table_fill_ms = 0.0
    table_kernel_ms = 0.0
    if use_cuda_fused_builder:
        nsel, compact_kernel_ms = _cuda_elapsed_ms(
            lambda: compact_keep_blocks_and_build_table_cuda_out(
                keep_flat=keep_flat,
                sel_blocks=sel_blocks,
                pages_per_row=pages_per_row,
                pos=pos_buf,
                local_rank=local_rank_buf,
                block_table=block_table,
                page_offsets=page_offsets,
            ),
            enabled=measure_timing,
        )
        pos = pos_buf[:nsel]
        local_rank = local_rank_buf[:nsel]
    else:
        def _run_build_table():
            if prefer_cuda_builder:
                build_block_table_cuda(
                    pages_per_row=pages_per_row,
                    block_table=block_table,
                    page_offsets=page_offsets,
                )
                return page_offsets
            page_offsets_local = torch.cumsum(pages_per_row, dim=0, dtype=torch.int32) - pages_per_row
            valid_page = page_slot < pages_per_row.unsqueeze(1)
            block_ids = page_offsets_local.unsqueeze(1) + page_slot
            block_table.copy_(torch.where(valid_page, block_ids, torch.zeros_like(block_ids)))
            page_offsets.copy_(page_offsets_local)
            return page_offsets

        page_offsets, table_fill_ms = _cuda_elapsed_ms(_run_build_table, enabled=measure_timing)
        table_kernel_ms = table_fill_ms
        if use_cuda_compact:
            nsel, compact_kernel_ms = _cuda_elapsed_ms(
                lambda: compact_keep_blocks_cuda_out(
                    keep_flat=keep_flat,
                    sel_blocks=sel_blocks,
                    pos=pos_buf,
                    local_rank=local_rank_buf,
                ),
                enabled=measure_timing,
            )
            pos = pos_buf[:nsel]
            local_rank = local_rank_buf[:nsel]
        else:
            keep_prefix_rank, compact_kernel_ms = _cuda_elapsed_ms(
                lambda: (torch.cumsum(keep_flat.to(torch.int32), dim=-1, dtype=torch.int32) - 1).contiguous(),
                enabled=measure_timing,
            )

            pos, compact_nonzero_ms = _cuda_elapsed_ms(
                lambda: torch.nonzero(keep_flat, as_tuple=False).contiguous(), enabled=measure_timing
            )

    if stage_callback is not None:
        stage_callback(
            "block_table_done",
            {
                "rows": float(rows),
                "kv_blocks": float(kv_blocks),
                "table_fill_ms": float(table_fill_ms),
            },
        )

    k_cache_flat = k_cache.reshape(-1, head_dim)
    v_cache_flat = v_cache.reshape(-1, head_dim)
    num_sel_blocks = int(pos.shape[0])
    attempted_triton = False
    attempted_cuda = False
    use_triton_fused = False
    use_cuda_fused = False
    use_cuda_local_rank_fused = False
    if num_sel_blocks > 0:
        if backend_req == "cuda":
            attempted_cuda = True
            if use_cuda_compact:
                use_cuda_local_rank_fused = can_use_cuda_cache_fill_from_pos_with_local_rank(
                    k=k,
                    v=v,
                    pos=pos,
                    local_rank=local_rank,
                    page_offsets=page_offsets,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                    page_block_size=page_block_size,
                )
            else:
                use_cuda_fused = can_use_cuda_cache_fill_from_pos_with_rank(
                    k=k,
                    v=v,
                    pos=pos,
                    keep_prefix_rank=keep_prefix_rank,
                    page_offsets=page_offsets,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                    page_block_size=page_block_size,
                )
        elif backend_req == "triton":
            attempted_triton = True
            use_triton_fused = can_use_triton_cache_fill_from_pos_with_rank(
                k=k,
                v=v,
                pos=pos,
                keep_prefix_rank=keep_prefix_rank,
                page_offsets=page_offsets,
                k_cache_flat=k_cache_flat,
                v_cache_flat=v_cache_flat,
                block_size=block_size,
                page_block_size=page_block_size,
            )
        elif backend_req == "auto":
            attempted_cuda = cuda_cache_fill_available()
            if attempted_cuda:
                if use_cuda_compact:
                    use_cuda_local_rank_fused = can_use_cuda_cache_fill_from_pos_with_local_rank(
                        k=k,
                        v=v,
                        pos=pos,
                        local_rank=local_rank,
                        page_offsets=page_offsets,
                        k_cache_flat=k_cache_flat,
                        v_cache_flat=v_cache_flat,
                        block_size=block_size,
                        page_block_size=page_block_size,
                    )
                else:
                    use_cuda_fused = can_use_cuda_cache_fill_from_pos_with_rank(
                        k=k,
                        v=v,
                        pos=pos,
                        keep_prefix_rank=keep_prefix_rank,
                        page_offsets=page_offsets,
                        k_cache_flat=k_cache_flat,
                        v_cache_flat=v_cache_flat,
                        block_size=block_size,
                        page_block_size=page_block_size,
                    )
            attempted_triton = cache_fill_triton_available() and not use_cuda_compact
            if not use_cuda_fused and not use_cuda_local_rank_fused:
                use_triton_fused = can_use_triton_cache_fill_from_pos_with_rank(
                    k=k,
                    v=v,
                    pos=pos,
                    keep_prefix_rank=keep_prefix_rank,
                    page_offsets=page_offsets,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                    page_block_size=page_block_size,
                )

    compact_fused_calls = 1.0 if (use_triton_fused or use_cuda_fused or use_cuda_local_rank_fused) else 0.0
    compact_fallback_calls = 0.0 if (use_triton_fused or use_cuda_fused or use_cuda_local_rank_fused) else 1.0

    if use_triton_fused or use_cuda_fused or use_cuda_local_rank_fused:
        compact_out = None
    else:
        def _run_index_compact_post():
            if pos.numel() == 0:
                return None
            row_idx = pos[:, 0].to(torch.long).contiguous()
            blk_idx = pos[:, 1].to(torch.long).contiguous()
            if keep_prefix_rank is not None:
                local_block_rank = keep_prefix_rank[row_idx, blk_idx].to(torch.long).contiguous()
            else:
                local_block_rank = local_rank.to(torch.long).contiguous()
            row_token_base = (
                page_offsets[row_idx].to(torch.long).contiguous() * page_block_size
                + local_block_rank * block_size
            ).contiguous()
            return row_idx, blk_idx, row_token_base

        compact_out, compact_post_ms = _cuda_elapsed_ms(_run_index_compact_post, enabled=measure_timing)

    compact_tail_ms = compact_nonzero_ms + compact_post_ms
    compact_ms = compact_kernel_ms + compact_tail_ms
    src_layout_ms = 0.0
    if stage_callback is not None:
        stage_callback(
            "compact_done",
            {
                "compact_ms": float(compact_ms),
                "compact_kernel_ms": float(compact_kernel_ms),
                "compact_nonzero_ms": float(compact_nonzero_ms),
                "compact_post_ms": float(compact_post_ms),
                "num_selected_blocks": float(num_sel_blocks),
            },
        )

    def _run_fill_cache_backend():
        if num_sel_blocks == 0:
            return {
                "index_cache_fill_cuda_calls": 0.0,
                "index_cache_fill_cuda_fallback_calls": 0.0,
                "index_cache_fill_backend_id": 0.0,
                "index_cache_fill_triton_calls": 0.0,
                "index_cache_fill_torch_calls": 0.0,
                "index_cache_fill_fallback_calls": 0.0,
                "index_cache_fill_kernel_ms": 0.0,
                "index_cache_fill_small_calls": 0.0,
                "index_cache_fill_medium_calls": 0.0,
                "index_cache_fill_large_calls": 0.0,
                "index_cache_fill_variant_id": 0.0,
                "index_cache_fill_tuned_calls": 0.0,
            }

        launch_config = select_cache_fill_launch_config(
            num_sel_blocks=int(num_sel_blocks),
            block_size=int(block_size),
            head_dim=int(head_dim),
            dtype=k.dtype,
        )

        if use_cuda_fused:
            _, kernel_ms = _cuda_elapsed_ms(
                lambda: cache_fill_blocks_to_paged_kv_from_pos_with_rank_cuda(
                    k=k,
                    v=v,
                    pos=pos,
                    keep_prefix_rank=keep_prefix_rank,
                    page_offsets=page_offsets,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                    page_block_size=page_block_size,
                ),
                enabled=measure_timing,
            )
            return {
                "index_cache_fill_cuda_calls": 1.0,
                "index_cache_fill_cuda_fallback_calls": 0.0,
                "index_cache_fill_backend_id": 2.0,
                "index_cache_fill_triton_calls": 0.0,
                "index_cache_fill_torch_calls": 0.0,
                "index_cache_fill_fallback_calls": 0.0,
                "index_cache_fill_kernel_ms": float(kernel_ms),
                "index_cache_fill_small_calls": float(launch_config["small_calls"]),
                "index_cache_fill_medium_calls": float(launch_config["medium_calls"]),
                "index_cache_fill_large_calls": float(launch_config["large_calls"]),
                "index_cache_fill_variant_id": float(launch_config["variant_id"]),
                "index_cache_fill_tuned_calls": float(launch_config["tuned_calls"]),
            }

        if use_cuda_local_rank_fused:
            _, kernel_ms = _cuda_elapsed_ms(
                lambda: cache_fill_blocks_to_paged_kv_from_pos_with_local_rank_cuda(
                    k=k,
                    v=v,
                    pos=pos,
                    local_rank=local_rank,
                    page_offsets=page_offsets,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                    page_block_size=page_block_size,
                ),
                enabled=measure_timing,
            )
            return {
                "index_cache_fill_cuda_calls": 1.0,
                "index_cache_fill_cuda_fallback_calls": 0.0,
                "index_cache_fill_backend_id": 2.0,
                "index_cache_fill_triton_calls": 0.0,
                "index_cache_fill_torch_calls": 0.0,
                "index_cache_fill_fallback_calls": 0.0,
                "index_cache_fill_kernel_ms": float(kernel_ms),
                "index_cache_fill_small_calls": float(launch_config["small_calls"]),
                "index_cache_fill_medium_calls": float(launch_config["medium_calls"]),
                "index_cache_fill_large_calls": float(launch_config["large_calls"]),
                "index_cache_fill_variant_id": float(launch_config["variant_id"]),
                "index_cache_fill_tuned_calls": float(launch_config["tuned_calls"]),
            }

        if use_triton_fused:
            _, kernel_ms = _cuda_elapsed_ms(
                lambda: cache_fill_blocks_to_paged_kv_from_pos_with_rank(
                    k=k,
                    v=v,
                    pos=pos,
                    keep_prefix_rank=keep_prefix_rank,
                    page_offsets=page_offsets,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                    page_block_size=page_block_size,
                    launch_config=launch_config,
                ),
                enabled=measure_timing,
            )
            return {
                "index_cache_fill_cuda_calls": 0.0,
                "index_cache_fill_cuda_fallback_calls": 0.0,
                "index_cache_fill_backend_id": 1.0,
                "index_cache_fill_triton_calls": 1.0,
                "index_cache_fill_torch_calls": 0.0,
                "index_cache_fill_fallback_calls": 0.0,
                "index_cache_fill_kernel_ms": float(kernel_ms),
                "index_cache_fill_small_calls": float(launch_config["small_calls"]),
                "index_cache_fill_medium_calls": float(launch_config["medium_calls"]),
                "index_cache_fill_large_calls": float(launch_config["large_calls"]),
                "index_cache_fill_variant_id": float(launch_config["variant_id"]),
                "index_cache_fill_tuned_calls": float(launch_config["tuned_calls"]),
            }

        if compact_out is None:
            return {
                "index_cache_fill_cuda_calls": 0.0,
                "index_cache_fill_cuda_fallback_calls": 0.0,
                "index_cache_fill_backend_id": 0.0,
                "index_cache_fill_triton_calls": 0.0,
                "index_cache_fill_torch_calls": 0.0,
                "index_cache_fill_fallback_calls": 0.0,
                "index_cache_fill_kernel_ms": 0.0,
                "index_cache_fill_small_calls": 0.0,
                "index_cache_fill_medium_calls": 0.0,
                "index_cache_fill_large_calls": 0.0,
                "index_cache_fill_variant_id": 0.0,
                "index_cache_fill_tuned_calls": 0.0,
            }

        row_idx, blk_idx, row_token_base = compact_out
        use_cuda = False
        use_triton = False
        attempted_triton_local = attempted_triton
        attempted_cuda_local = attempted_cuda

        if backend_req == "cuda":
            attempted_cuda_local = True
            use_cuda = can_use_cuda_cache_fill_from_row_blk_dst(
                k=k,
                v=v,
                row_idx=row_idx,
                blk_idx=blk_idx,
                dst_token_base=row_token_base,
                k_cache_flat=k_cache_flat,
                v_cache_flat=v_cache_flat,
                block_size=block_size,
            )
        elif backend_req == "triton":
            attempted_triton_local = True
            use_triton = can_use_triton_cache_fill_from_kv(
                k=k,
                v=v,
                row_idx=row_idx,
                blk_idx=blk_idx,
                dst_token_base=row_token_base,
                k_cache_flat=k_cache_flat,
                v_cache_flat=v_cache_flat,
                block_size=block_size,
            )
        elif backend_req == "auto":
            attempted_cuda_local = cuda_cache_fill_available()
            if attempted_cuda_local:
                use_cuda = can_use_cuda_cache_fill_from_row_blk_dst(
                    k=k,
                    v=v,
                    row_idx=row_idx,
                    blk_idx=blk_idx,
                    dst_token_base=row_token_base,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                )
            attempted_triton_local = cache_fill_triton_available()
            if not use_cuda:
                use_triton = can_use_triton_cache_fill_from_kv(
                    k=k,
                    v=v,
                    row_idx=row_idx,
                    blk_idx=blk_idx,
                    dst_token_base=row_token_base,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                )

        if use_cuda:
            _, kernel_ms = _cuda_elapsed_ms(
                lambda: cache_fill_blocks_to_paged_kv_from_kv_strided_cuda(
                    k=k,
                    v=v,
                    row_idx=row_idx,
                    blk_idx=blk_idx,
                    dst_token_base=row_token_base,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                ),
                enabled=measure_timing,
            )
            return {
                "index_cache_fill_cuda_calls": 1.0,
                "index_cache_fill_cuda_fallback_calls": 0.0,
                "index_cache_fill_backend_id": 2.0,
                "index_cache_fill_triton_calls": 0.0,
                "index_cache_fill_torch_calls": 0.0,
                "index_cache_fill_fallback_calls": 0.0,
                "index_cache_fill_kernel_ms": float(kernel_ms),
                "index_cache_fill_small_calls": float(launch_config["small_calls"]),
                "index_cache_fill_medium_calls": float(launch_config["medium_calls"]),
                "index_cache_fill_large_calls": float(launch_config["large_calls"]),
                "index_cache_fill_variant_id": float(launch_config["variant_id"]),
                "index_cache_fill_tuned_calls": float(launch_config["tuned_calls"]),
            }

        if use_triton:
            _, kernel_ms = _cuda_elapsed_ms(
                lambda: cache_fill_blocks_to_paged_kv_from_kv_strided(
                    k=k,
                    v=v,
                    row_idx=row_idx,
                    blk_idx=blk_idx,
                    dst_token_base=row_token_base,
                    k_cache_flat=k_cache_flat,
                    v_cache_flat=v_cache_flat,
                    block_size=block_size,
                    launch_config=launch_config,
                ),
                enabled=measure_timing,
            )
            return {
                "index_cache_fill_cuda_calls": 0.0,
                "index_cache_fill_cuda_fallback_calls": 0.0,
                "index_cache_fill_backend_id": 1.0,
                "index_cache_fill_triton_calls": 1.0,
                "index_cache_fill_torch_calls": 0.0,
                "index_cache_fill_fallback_calls": 0.0,
                "index_cache_fill_kernel_ms": float(kernel_ms),
                "index_cache_fill_small_calls": float(launch_config["small_calls"]),
                "index_cache_fill_medium_calls": float(launch_config["medium_calls"]),
                "index_cache_fill_large_calls": float(launch_config["large_calls"]),
                "index_cache_fill_variant_id": float(launch_config["variant_id"]),
                "index_cache_fill_tuned_calls": float(launch_config["tuned_calls"]),
            }

        # Torch fallback without relayout materialization.
        token_offsets = torch.arange(block_size, device=device, dtype=torch.long).unsqueeze(0)
        flat_token_idx = (row_token_base.unsqueeze(1) + token_offsets).reshape(-1)
        b_idx = torch.div(row_idx, num_kv_heads, rounding_mode="floor")
        h_idx = torch.remainder(row_idx, num_kv_heads)
        tok = blk_idx.unsqueeze(1) * block_size + token_offsets
        k_sel = k[b_idx.unsqueeze(1), tok, h_idx.unsqueeze(1), :].reshape(-1, head_dim)
        v_sel = v[b_idx.unsqueeze(1), tok, h_idx.unsqueeze(1), :].reshape(-1, head_dim)
        k_cache_flat[flat_token_idx] = k_sel
        v_cache_flat[flat_token_idx] = v_sel
        return {
            "index_cache_fill_cuda_calls": 0.0,
            "index_cache_fill_cuda_fallback_calls": 1.0
            if attempted_cuda_local and not use_cuda
            else 0.0,
            "index_cache_fill_backend_id": 0.0,
            "index_cache_fill_triton_calls": 0.0,
            "index_cache_fill_torch_calls": 1.0,
            "index_cache_fill_fallback_calls": 1.0 if attempted_triton_local else 0.0,
            "index_cache_fill_kernel_ms": 0.0,
            "index_cache_fill_small_calls": 0.0,
            "index_cache_fill_medium_calls": 0.0,
            "index_cache_fill_large_calls": 0.0,
            "index_cache_fill_variant_id": 0.0,
            "index_cache_fill_tuned_calls": 0.0,
        }

    cache_fill_backend, cache_fill_ms = _cuda_elapsed_ms(
        _run_fill_cache_backend, enabled=measure_timing
    )
    fused_builder_path_active = bool(
        prefer_fused_builder_v2
        and use_cuda_fused_builder
        and use_cuda_local_rank_fused
    )
    selected_kv_materialize_calls = 0.0 if fused_builder_path_active else (1.0 if num_sel_blocks > 0 else 0.0)
    selected_kv_materialize_ms = 0.0 if fused_builder_path_active else float(compact_ms + cache_fill_ms)
    if stage_callback is not None:
        stage_callback(
            "cache_fill_done",
            {
                "cache_fill_ms": float(cache_fill_ms),
                "cache_fill_kernel_ms": float(cache_fill_backend.get("index_cache_fill_kernel_ms", 0.0)),
                "cache_fill_triton_calls": float(cache_fill_backend.get("index_cache_fill_triton_calls", 0.0)),
                "cache_fill_torch_calls": float(cache_fill_backend.get("index_cache_fill_torch_calls", 0.0)),
                "cache_fill_fallback_calls": float(cache_fill_backend.get("index_cache_fill_fallback_calls", 0.0)),
            },
        )
    timings = {
        "index_keep_flat_sanitize_ms": keep_flat_sanitize_ms,
        "index_sel_blocks_sum_ms": sel_blocks_sum_ms,
        "index_zero_row_fix_ms": zero_row_fix_ms,
        "index_sel_lens_ms": sel_lens_ms,
        "index_pages_per_row_ms": pages_per_row_ms,
        "index_page_count_stats_ms": page_count_stats_ms,
        "index_block_table_ms": table_fill_ms,
        "index_table_fill_ms": table_fill_ms,
        "index_table_kernel_ms": table_kernel_ms,
        "index_compact_ms": compact_ms,
        "index_compact_kernel_ms": compact_kernel_ms,
        "index_compact_nonzero_ms": compact_nonzero_ms,
        "index_compact_post_ms": compact_post_ms,
        "index_compact_fused_calls": compact_fused_calls,
        "index_compact_fallback_calls": compact_fallback_calls,
        "index_compact_fused_post_ms": compact_fused_post_ms,
        "index_src_layout_ms": src_layout_ms,
        "index_cache_fill_ms": cache_fill_ms,
        "index_cache_fill_kernel_ms": float(cache_fill_backend.get("index_cache_fill_kernel_ms", 0.0)),
        "index_build_ms": table_fill_ms + compact_ms + src_layout_ms + cache_fill_ms,
        "index_cache_fill_cuda_calls": float(cache_fill_backend.get("index_cache_fill_cuda_calls", 0.0)),
        "index_cache_fill_cuda_fallback_calls": float(
            cache_fill_backend.get("index_cache_fill_cuda_fallback_calls", 0.0)
        ),
        "index_cache_fill_backend_id": float(cache_fill_backend.get("index_cache_fill_backend_id", 0.0)),
        "index_cache_fill_triton_calls": float(cache_fill_backend.get("index_cache_fill_triton_calls", 0.0)),
        "index_cache_fill_torch_calls": float(cache_fill_backend.get("index_cache_fill_torch_calls", 0.0)),
        "index_cache_fill_fallback_calls": float(cache_fill_backend.get("index_cache_fill_fallback_calls", 0.0)),
        "index_cache_fill_small_calls": float(cache_fill_backend.get("index_cache_fill_small_calls", 0.0)),
        "index_cache_fill_medium_calls": float(cache_fill_backend.get("index_cache_fill_medium_calls", 0.0)),
        "index_cache_fill_large_calls": float(cache_fill_backend.get("index_cache_fill_large_calls", 0.0)),
        "index_cache_fill_variant_id": float(cache_fill_backend.get("index_cache_fill_variant_id", 0.0)),
        "index_cache_fill_tuned_calls": float(cache_fill_backend.get("index_cache_fill_tuned_calls", 0.0)),
        "workspace_capacity_pages": float(workspace_stats["workspace_capacity_pages"]),
        "workspace_required_pages": float(workspace_stats["workspace_required_pages"]),
        "workspace_page_pool_capacity_pages": float(workspace_stats["workspace_page_pool_capacity_pages"]),
        "workspace_page_pool_required_pages": float(workspace_stats["workspace_page_pool_required_pages"]),
        "workspace_growth_events": float(workspace_stats["workspace_growth_events"]),
        "workspace_k_alloc_mb": float(workspace_stats["workspace_k_alloc_mb"]),
        "workspace_v_alloc_mb": float(workspace_stats["workspace_v_alloc_mb"]),
        "workspace_block_table_alloc_mb": float(workspace_stats["workspace_block_table_alloc_mb"]),
        "workspace_total_pages_used": float(total_pages_used),
        "workspace_max_pages_used": float(max_pages_used),
        "scratch_pool_growth_events": float(build_workspace_stats["scratch_pool_growth_events"]),
        "scratch_pool_pos_alloc_mb": float(build_workspace_stats["scratch_pool_pos_alloc_mb"]),
        "scratch_pool_meta_alloc_mb": float(build_workspace_stats["scratch_pool_meta_alloc_mb"]),
        "fused_builder_calls": 1.0 if fused_builder_path_active else 0.0,
        "fused_builder_ms": float(compact_kernel_ms + cache_fill_ms) if fused_builder_path_active else 0.0,
        "fused_builder_table_ms": float(compact_kernel_ms) if fused_builder_path_active else 0.0,
        "fused_builder_fill_ms": float(cache_fill_ms) if fused_builder_path_active else 0.0,
        "selected_kv_materialize_calls": selected_kv_materialize_calls,
        "selected_kv_materialize_ms": selected_kv_materialize_ms,
    }
    return k_cache, v_cache, block_table, sel_lens.to(torch.int32), timings


def build_paged_kv_cache_from_token_mask(
    k: torch.Tensor,  # [B, K, Hkv, D]
    v: torch.Tensor,  # [B, K, Hkv, D]
    keep_token_mask_kv: torch.Tensor,  # [B, Hkv, K]
    page_block_size: int = 256,
):
    bsz, kv_len, num_kv_heads, head_dim = k.shape
    rows = bsz * num_kv_heads
    device = k.device

    mask_flat = keep_token_mask_kv.reshape(rows, kv_len)
    sel_lens = mask_flat.sum(dim=-1, dtype=torch.int32)
    zero_rows = sel_lens == 0
    if zero_rows.any():
        mask_flat[zero_rows, kv_len - 1] = True
        sel_lens = mask_flat.sum(dim=-1, dtype=torch.int32)

    pages_per_row = torch.div(sel_lens + (page_block_size - 1), page_block_size, rounding_mode="floor")
    max_pages = int(pages_per_row.max().item())
    total_pages = int(pages_per_row.sum().item())
    total_pages = max(total_pages, 1)
    max_pages = max(max_pages, 1)

    k_cache = torch.zeros((total_pages, page_block_size, 1, head_dim), device=device, dtype=k.dtype)
    v_cache = torch.zeros((total_pages, page_block_size, 1, head_dim), device=device, dtype=v.dtype)
    block_table = torch.zeros((rows, max_pages), device=device, dtype=torch.int32)

    page_offsets = torch.cumsum(pages_per_row, dim=0, dtype=torch.int32) - pages_per_row
    page_slot = torch.arange(max_pages, device=device, dtype=torch.int32).unsqueeze(0)  # [1, max_pages]
    valid_page = page_slot < pages_per_row.unsqueeze(1)  # [rows, max_pages]
    if valid_page.any():
        block_ids = page_offsets.unsqueeze(1) + page_slot
        block_table[valid_page] = block_ids[valid_page]

    # Flatten KV to [rows, K, D] with rows = (b, kv_head).
    k_flat = k.permute(0, 2, 1, 3).reshape(rows, kv_len, head_dim).contiguous()
    v_flat = v.permute(0, 2, 1, 3).reshape(rows, kv_len, head_dim).contiguous()

    # Select all kept tokens across rows in row-major order.
    pos = torch.nonzero(mask_flat, as_tuple=False)  # [Nsel, 2], (row, token)
    row_idx = pos[:, 0].to(torch.long)
    tok_idx = pos[:, 1].to(torch.long)

    # Per-row rank (0-based) for each kept token.
    # `nonzero(mask_flat)` is row-major, so tokens for each row are contiguous.
    sel_offsets = torch.cumsum(sel_lens.to(torch.int64), dim=0) - sel_lens.to(torch.int64)
    local_rank = torch.arange(pos.shape[0], device=device, dtype=torch.long) - sel_offsets[row_idx]
    page_in_row = torch.div(local_rank, page_block_size, rounding_mode="floor")
    offset_in_page = torch.remainder(local_rank, page_block_size)
    global_page = page_offsets[row_idx].to(torch.long) + page_in_row

    k_sel = k_flat[row_idx, tok_idx, :]
    v_sel = v_flat[row_idx, tok_idx, :]
    k_cache[global_page, offset_in_page, 0, :] = k_sel
    v_cache[global_page, offset_in_page, 0, :] = v_sel

    return k_cache, v_cache, block_table, sel_lens


def flash_attn_indexed_prefill_from_paged_kv(
    q: torch.Tensor,  # [B, Q, Hq, D]
    k_cache: torch.Tensor,  # [num_pages, page_block, 1, D]
    v_cache: torch.Tensor,  # [num_pages, page_block, 1, D]
    block_table: torch.Tensor,  # [B*Hkv, max_pages]
    cache_seqlens: torch.Tensor,  # [B*Hkv], int32
    num_key_value_groups: int,
    softmax_scale: float,
    measure_timing: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz, q_len, num_q_heads, head_dim = q.shape
    rows = block_table.shape[0]
    num_kv_heads = rows // bsz
    peak_mem_enabled = _debug_peak_mem_enabled()
    use_direct_out_layout = _compactattn_enable_fused_builder_v2()

    attn_ws, attn_workspace_stats = _get_or_create_indexed_attn_workspace(
        device=q.device,
        dtype=q.dtype,
        bsz=bsz,
        q_len=q_len,
        num_kv_heads=num_kv_heads,
        num_key_value_groups=num_key_value_groups,
        head_dim=head_dim,
        use_out_flat=not use_direct_out_layout,
    )
    out_flat = attn_ws.get("out_flat", None)
    if bsz == 1 and q.is_contiguous():
        q_group = q.squeeze(0).as_strided(
            (num_kv_heads, q_len, num_key_value_groups, head_dim),
            (
                num_key_value_groups * head_dim,
                num_kv_heads * num_key_value_groups * head_dim,
                head_dim,
                1,
            ),
        )
        q_layout_ms = 0.0
        q_layout_mem = {}
        q_layout_cuda_calls = 0.0
    else:
        q_group_5d = attn_ws["q_group_5d"]
        use_cuda_q_pack = use_direct_out_layout and can_use_cuda_pack_q_for_indexed_prefill(
            q=q,
            q_group=q_group_5d,
            num_kv_heads=num_kv_heads,
            num_key_value_groups=num_key_value_groups,
        )

        def _run_q_layout():
            if use_cuda_q_pack:
                pack_q_for_indexed_prefill_cuda(
                    q=q,
                    q_group=q_group_5d,
                    num_kv_heads=num_kv_heads,
                    num_key_value_groups=num_key_value_groups,
                )
                return q_group_5d
            return q_group_5d.copy_(
                q.view(bsz, q_len, num_kv_heads, num_key_value_groups, head_dim).permute(0, 2, 1, 3, 4)
            )
        _, q_layout_ms, q_layout_mem = _cuda_elapsed_and_peak(
            _run_q_layout,
            q.device,
            measure_timing=measure_timing,
            measure_peak=peak_mem_enabled,
        )
        q_group = q_group_5d.view(rows, q_len, num_key_value_groups, head_dim)
        q_layout_cuda_calls = 1.0 if use_cuda_q_pack else 0.0

    cache_seqlens_i32, cache_seqlens_cast_ms = _cuda_elapsed_ms(
        lambda: (
            cache_seqlens
            if cache_seqlens.dtype == torch.int32 and cache_seqlens.is_contiguous()
            else cache_seqlens.to(torch.int32).contiguous()
        ),
        enabled=measure_timing,
    )
    block_table_i32, block_table_cast_ms = _cuda_elapsed_ms(
        lambda: (
            block_table
            if block_table.dtype == torch.int32 and block_table.is_contiguous()
            else block_table.to(torch.int32).contiguous()
        ),
        enabled=measure_timing,
    )
    cache_seqlens_copy_calls = 0.0 if cache_seqlens_i32.data_ptr() == cache_seqlens.data_ptr() else 1.0
    block_table_copy_calls = 0.0 if block_table_i32.data_ptr() == block_table.data_ptr() else 1.0

    def _run_flash_attn():
        return flash_attn_with_kvcache(
            q=q_group,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens_i32,
            block_table=block_table_i32,
            softmax_scale=softmax_scale,
            causal=True,
        )
    out_group, flash_kvcache_ms, flash_attn_mem = _cuda_elapsed_and_peak(
        _run_flash_attn,
        q.device,
        measure_timing=measure_timing,
        measure_peak=peak_mem_enabled,
    )

    def _run_out_layout():
        out = (
            out_group.view(bsz, num_kv_heads, q_len, num_key_value_groups, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(bsz, q_len, num_q_heads, head_dim)
        )
        if out_flat is None:
            return out
        return out_flat.copy_(out)

    out, out_layout_ms, out_layout_mem = _cuda_elapsed_and_peak(
        _run_out_layout,
        q.device,
        measure_timing=measure_timing,
        measure_peak=peak_mem_enabled,
    )
    return out, {
        "index_attn_q_workspace_alloc_mb": float(attn_workspace_stats["attn_q_workspace_alloc_mb"]),
        "index_attn_out_workspace_alloc_mb": float(attn_workspace_stats["attn_out_workspace_alloc_mb"]),
        "index_attn_q_workspace_growth_events": float(attn_workspace_stats["attn_q_workspace_growth_events"]),
        "index_attn_cache_seqlens_copy_calls": float(cache_seqlens_copy_calls),
        "index_attn_block_table_copy_calls": float(block_table_copy_calls),
        "index_attn_cache_seqlens_cast_ms": float(cache_seqlens_cast_ms),
        "index_attn_block_table_cast_ms": float(block_table_cast_ms),
        "index_attn_q_layout_ms": float(q_layout_ms),
        "index_attn_q_layout_cuda_calls": float(q_layout_cuda_calls),
        "index_attn_flash_kvcache_ms": float(flash_kvcache_ms),
        "index_attn_out_layout_ms": float(out_layout_ms),
        "index_attn_q_layout_peak_alloc_mb": float(q_layout_mem.get("peak_alloc_mb", 0.0)),
        "index_attn_q_layout_peak_reserved_mb": float(q_layout_mem.get("peak_reserved_mb", 0.0)),
        "index_attn_flash_kvcache_peak_alloc_mb": float(flash_attn_mem.get("peak_alloc_mb", 0.0)),
        "index_attn_flash_kvcache_peak_reserved_mb": float(flash_attn_mem.get("peak_reserved_mb", 0.0)),
        "index_attn_out_layout_peak_alloc_mb": float(out_layout_mem.get("peak_alloc_mb", 0.0)),
        "index_attn_out_layout_peak_reserved_mb": float(out_layout_mem.get("peak_reserved_mb", 0.0)),
    }


_FI_WORKSPACES: Dict[int, torch.Tensor] = {}


def _get_or_create_fi_workspace(device: torch.device) -> torch.Tensor:
    device_idx = int(device.index) if device.index is not None else 0
    if device_idx not in _FI_WORKSPACES:
        _FI_WORKSPACES[device_idx] = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device=device
        )
    return _FI_WORKSPACES[device_idx]


def flashinfer_indexed_prefill_from_paged_kv(
    q: torch.Tensor,  # [B, Q, Hq, D]
    k_cache: torch.Tensor,  # [num_pages, page_block_size, 1, D]
    v_cache: torch.Tensor,  # [num_pages, page_block_size, 1, D]
    block_table: torch.Tensor,  # [B*Hkv, max_pages]
    cache_seqlens: torch.Tensor,  # [B*Hkv], int32
    num_key_value_groups: int,
    softmax_scale: float,
    measure_timing: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    import flashinfer
    bsz, q_len, num_q_heads, head_dim = q.shape
    rows = block_table.shape[0]
    num_kv_heads = rows // bsz
    page_block_size = k_cache.shape[1]
    G = num_key_value_groups
    device = q.device

    cache_seqlens_i32 = (
        cache_seqlens
        if cache_seqlens.dtype == torch.int32 and cache_seqlens.is_contiguous()
        else cache_seqlens.to(torch.int32).contiguous()
    )
    block_table_i32 = (
        block_table
        if block_table.dtype == torch.int32 and block_table.is_contiguous()
        else block_table.to(torch.int32).contiguous()
    )

    pages_per_row = cache_seqlens_i32 // page_block_size
    kv_indptr = torch.zeros(rows + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = pages_per_row.cumsum(0)
    max_pages = block_table_i32.shape[1]
    valid_mask = torch.arange(max_pages, device=device).unsqueeze(0) < pages_per_row.unsqueeze(1)
    kv_indices = block_table_i32[valid_mask]
    kv_last_page_len = torch.full((rows,), page_block_size, dtype=torch.int32, device=device)
    qo_indptr = torch.arange(0, (rows + 1) * q_len, q_len, dtype=torch.int32, device=device)

    if bsz == 1 and q.is_contiguous():
        q_fi = q.squeeze(0).as_strided(
            (num_kv_heads, q_len, G, head_dim),
            (G * head_dim, num_kv_heads * G * head_dim, head_dim, 1),
        ).reshape(rows * q_len, G, head_dim)
    else:
        q_fi = (
            q.view(bsz, q_len, num_kv_heads, G, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(rows * q_len, G, head_dim)
        )
    if not q_fi.is_contiguous():
        q_fi = q_fi.contiguous()

    workspace = _get_or_create_fi_workspace(device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        backend=_flashinfer_attention_backend(),
    )
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last_page_len,
        num_qo_heads=G,
        num_kv_heads=1,
        head_dim_qk=head_dim,
        page_size=page_block_size,
        causal=True,
        sm_scale=softmax_scale,
        q_data_type=q.dtype,
    )

    def _run_fi():
        return wrapper.run(q_fi, (k_cache, v_cache))

    out_fi, _, _ = _cuda_elapsed_and_peak(
        _run_fi, device, measure_timing=measure_timing, measure_peak=False
    )

    out = (
        out_fi.reshape(bsz, num_kv_heads, q_len, G, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(bsz, q_len, num_q_heads, head_dim)
    )
    return out, {
        "index_attn_q_workspace_alloc_mb": 0.0,
        "index_attn_out_workspace_alloc_mb": 0.0,
        "index_attn_q_workspace_growth_events": 0.0,
        "index_attn_cache_seqlens_copy_calls": 0.0,
        "index_attn_block_table_copy_calls": 0.0,
        "index_attn_cache_seqlens_cast_ms": 0.0,
        "index_attn_block_table_cast_ms": 0.0,
        "index_attn_q_layout_ms": 0.0,
        "index_attn_q_layout_cuda_calls": 0.0,
        "index_attn_out_layout_ms": 0.0,
        "index_attn_q_layout_peak_alloc_mb": 0.0,
        "index_attn_q_layout_peak_reserved_mb": 0.0,
        "index_attn_flash_kvcache_peak_alloc_mb": 0.0,
        "index_attn_flash_kvcache_peak_reserved_mb": 0.0,
        "index_attn_out_layout_peak_alloc_mb": 0.0,
        "index_attn_out_layout_peak_reserved_mb": 0.0,
    }
