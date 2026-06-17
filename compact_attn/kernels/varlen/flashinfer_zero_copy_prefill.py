"""Zero-copy FlashInfer paged KV attention for heads-first KV cache.

KV is stored as [bsz, Hkv, kv_len, D] (heads-first). Because the position
dimension (dim=2) is contiguous within each head, a zero-copy view
  k_hf.view(bsz*Hkv*n_blocks, block_size, 1, D)
exposes FlashInfer pages without any cache-fill materialization.

Memory bandwidth comparison (50% sparsity, all data in HBM):
  fi_paged  (with materializ.): read(50%) + write(50%) + read(50%) = 150% data movement
  fi_zero_copy:                  read(50%)                          =  50% data movement
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import torch

from compact_attn.kernels.varlen.indexed_dense_prefill_varlen import _get_or_create_fi_workspace
from compact_attn.kernels.varlen.indexed_dense_cache_fill_cuda import (
    build_flashinfer_kv_indices_cuda_out,
    build_flashinfer_kv_indices_per_query_cuda_out,
    build_selected_indices_from_kv_keep_block_cuda_out,
    can_use_cuda_build_flashinfer_kv_indices,
    can_use_cuda_build_selected_indices_from_kv_keep_block,
    can_use_cuda_pack_q_for_indexed_prefill,
    pack_q_for_indexed_prefill_cuda,
)

_ZC_CALL_COUNT = 0
_ZC_STATIC_METADATA: Dict[Tuple[int, str, int, int, int, int], Dict[str, torch.Tensor]] = {}
_ZC_WRAPPERS: Dict[Tuple[int, str, int, int, int], object] = {}
_ZC_GRAPH_WRAPPERS: Dict[Tuple[int, str, int, int, int, int, int, int], object] = {}
_ZC_METADATA_WORKSPACES: Dict[Tuple[int, int], Dict[str, object]] = {}
_ZC_Q_LAYOUT_WORKSPACES: Dict[Tuple[int, str, int, int, int, int, int], Dict[str, object]] = {}
_ZC_OUTPUT_WORKSPACES: Dict[Tuple[int, str, int, int, int, int], torch.Tensor] = {}
_CUDNN_BLOCK_TABLE_WORKSPACES: Dict[Tuple[int, int, int], torch.Tensor] = {}
_CUDNN_SEQ_LEN_WORKSPACES: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
_CUDNN_OFFSET_WORKSPACES: Dict[Tuple[int, int, int, int, int], torch.Tensor] = {}
_CUDNN_OUTPUT_WORKSPACES: Dict[Tuple[int, str, int, int, int, int], torch.Tensor] = {}
_CUDNN_LSE_WORKSPACES: Dict[Tuple[int, int, int, int], torch.Tensor] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def _env_int_with_fallback(name: str, fallback_name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is not None and value.strip() != "":
        return int(value)
    return _env_int(fallback_name, default)


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


def _next_power_of_two(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


def _cuda_elapsed_ms(fn, *, enabled: bool, device: torch.device):
    if not enabled:
        return fn(), 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(device)
    start.record(stream)
    out = fn()
    end.record(stream)
    end.synchronize()
    return out, float(start.elapsed_time(end))


def _host_and_cuda_elapsed_ms(fn, *, enabled: bool, device: torch.device):
    if not enabled:
        return fn(), 0.0, 0.0
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(device)
    host_start = time.perf_counter()
    start.record(stream)
    out = fn()
    end.record(stream)
    end.synchronize()
    host_ms = (time.perf_counter() - host_start) * 1000.0
    return out, float(start.elapsed_time(end)), float(host_ms)


def _device_key(device: torch.device) -> int:
    return int(device.index) if device.index is not None else -1


def _get_static_metadata(
    *,
    device: torch.device,
    rows: int,
    n_blocks: int,
    q_len: int,
    block_size: int,
) -> Dict[str, torch.Tensor]:
    key = (
        _device_key(device),
        str(device.type),
        int(rows),
        int(n_blocks),
        int(q_len),
        int(block_size),
    )
    entry = _ZC_STATIC_METADATA.get(key)
    if entry is None:
        row_base = (torch.arange(rows, device=device, dtype=torch.int32) * n_blocks).unsqueeze(1)
        blk_ids = torch.arange(n_blocks, device=device, dtype=torch.int32).unsqueeze(0)
        entry = {
            "global_ids": row_base + blk_ids,
            "kv_last_page_len": torch.full((rows,), block_size, dtype=torch.int32, device=device),
            "qo_indptr": torch.arange(0, (rows + 1) * q_len, q_len, dtype=torch.int32, device=device),
        }
        _ZC_STATIC_METADATA[key] = entry
    return entry


def _get_wrapper(
    *,
    flashinfer,
    workspace: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    num_key_value_groups: int,
    head_dim: int,
    block_size: int,
    rows: int,
    q_len: int,
    kv_indices_capacity: int,
):
    if _env_flag("SEER_ZC_FLASHINFER_CUDA_GRAPH", False):
        key = (
            _device_key(device),
            str(dtype),
            int(num_key_value_groups),
            int(head_dim),
            int(block_size),
            int(rows),
            int(q_len),
            int(kv_indices_capacity),
        )
        wrapper = _ZC_GRAPH_WRAPPERS.get(key)
        if wrapper is None:
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                kv_layout="NHD",
                use_cuda_graph=True,
                backend=_flashinfer_attention_backend(),
                qo_indptr_buf=torch.empty((rows + 1,), device=device, dtype=torch.int32),
                paged_kv_indptr_buf=torch.empty((rows + 1,), device=device, dtype=torch.int32),
                paged_kv_indices_buf=torch.empty(
                    (max(kv_indices_capacity, 1),), device=device, dtype=torch.int32
                ),
                paged_kv_last_page_len_buf=torch.empty((rows,), device=device, dtype=torch.int32),
            )
            _ZC_GRAPH_WRAPPERS[key] = wrapper
        return wrapper

    key = (
        _device_key(device),
        str(dtype),
        int(num_key_value_groups),
        int(head_dim),
        int(block_size),
    )
    wrapper = _ZC_WRAPPERS.get(key)
    if wrapper is None:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend=_flashinfer_attention_backend(),
        )
        _ZC_WRAPPERS[key] = wrapper
    return wrapper


def _next_metadata_capacity(required_capacity: int, current_capacity: int, *, align: int = 64) -> int:
    required_capacity = max(int(required_capacity), 1)
    current_capacity = max(int(current_capacity), 0)
    if current_capacity == 0:
        return ((required_capacity + align - 1) // align) * align
    if current_capacity >= required_capacity:
        return current_capacity
    grown = max(current_capacity + max(current_capacity // 4, align), required_capacity)
    return ((grown + align - 1) // align) * align


def _get_or_create_metadata_workspace(
    *,
    device: torch.device,
    rows: int,
    kv_blocks: int,
) -> Dict[str, object]:
    key = (_device_key(device), int(rows))
    ws = _ZC_METADATA_WORKSPACES.get(key)
    current_capacity = 0 if ws is None else int(ws["kv_blocks_capacity"])
    required_capacity = max(int(kv_blocks), 1)
    capacity = _next_metadata_capacity(required_capacity, current_capacity)
    if ws is None or capacity != current_capacity:
        ws = {
            "selected_block_indices_i32": torch.empty(
                (rows, capacity), device=device, dtype=torch.int32
            ),
            "selected_block_counts_i32": torch.empty((rows,), device=device, dtype=torch.int32),
            "kv_indptr_i32": torch.empty((rows + 1,), device=device, dtype=torch.int32),
            "kv_indices_i32": torch.empty((rows * capacity,), device=device, dtype=torch.int32),
            "kv_blocks_capacity": int(capacity),
        }
        _ZC_METADATA_WORKSPACES[key] = ws
    return ws


def _get_or_create_q_layout_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    bsz: int,
    q_len: int,
    num_kv_heads: int,
    num_key_value_groups: int,
    head_dim: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    key = (
        _device_key(device),
        str(dtype),
        int(bsz),
        int(q_len),
        int(num_kv_heads),
        int(num_key_value_groups),
        int(head_dim),
    )
    ws = _ZC_Q_LAYOUT_WORKSPACES.get(key)
    if ws is None:
        ws = {
            "q_group_5d": torch.empty(
                (bsz, num_kv_heads, q_len, num_key_value_groups, head_dim),
                device=device,
                dtype=dtype,
            ),
        }
        _ZC_Q_LAYOUT_WORKSPACES[key] = ws
        growth_events = 1.0
    else:
        growth_events = 0.0
    stats = {
        "q_workspace_alloc_mb": float(
            ws["q_group_5d"].numel() * ws["q_group_5d"].element_size()
        )
        / (1024.0 * 1024.0),
        "q_workspace_growth_events": float(growth_events),
    }
    return ws, stats


def _get_or_create_output_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    rows: int,
    q_len: int,
    num_key_value_groups: int,
    head_dim: int,
) -> torch.Tensor:
    key = (
        _device_key(device),
        str(dtype),
        int(rows),
        int(q_len),
        int(num_key_value_groups),
        int(head_dim),
    )
    out = _ZC_OUTPUT_WORKSPACES.get(key)
    if out is None:
        out = torch.empty(
            (rows * q_len, num_key_value_groups, head_dim),
            device=device,
            dtype=dtype,
        )
        _ZC_OUTPUT_WORKSPACES[key] = out
    return out


def _get_or_create_cudnn_block_table_workspace(
    *,
    device: torch.device,
    rows: int,
    max_pages: int,
) -> torch.Tensor:
    key = (_device_key(device), int(rows), int(max_pages))
    block_table = _CUDNN_BLOCK_TABLE_WORKSPACES.get(key)
    if block_table is None:
        block_table = torch.empty((rows, max_pages), device=device, dtype=torch.int32)
        _CUDNN_BLOCK_TABLE_WORKSPACES[key] = block_table
    return block_table


def _resolve_cudnn_graph_max_pages(*, exact_max_pages: int, n_blocks: int) -> int:
    graph_max_pages = max(int(exact_max_pages), 1)
    if _env_flag("SEER_CUDNN_MAX_PAGES_FULL_BLOCKS", False):
        graph_max_pages = max(graph_max_pages, int(n_blocks))
    # cuDNN's frontend graph cache keys include max_sequence_kv.  Exact selected
    # widths create too many per-layer/chunk keys for sparse prefill, so bucket
    # by default.  Set SEER_CUDNN_MAX_PAGES_BUCKET_SIZE=0 to reproduce exact mode.
    bucket_size = _env_int("SEER_CUDNN_MAX_PAGES_BUCKET_SIZE", 8)
    if bucket_size > 0:
        graph_max_pages = ((graph_max_pages + bucket_size - 1) // bucket_size) * bucket_size
    if _env_flag("SEER_CUDNN_MAX_PAGES_POWER2", False):
        graph_max_pages = _next_power_of_two(graph_max_pages)
    return min(max(graph_max_pages, 1), max(int(n_blocks), 1))


def _get_or_create_cudnn_seq_len_workspaces(
    *,
    device: torch.device,
    rows: int,
) -> Dict[str, torch.Tensor]:
    key = (_device_key(device), int(rows))
    ws = _CUDNN_SEQ_LEN_WORKSPACES.get(key)
    if ws is None:
        ws = {
            "actual_seq_lens_q": torch.empty((rows, 1, 1, 1), device=device, dtype=torch.int32),
            "actual_seq_lens_kv": torch.empty((rows, 1, 1, 1), device=device, dtype=torch.int32),
        }
        _CUDNN_SEQ_LEN_WORKSPACES[key] = ws
    return ws


def _get_or_create_cudnn_offsets(
    *,
    device: torch.device,
    rows: int,
    q_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    key = (_device_key(device), int(rows), int(q_len), int(num_heads), int(head_dim))
    offsets = _CUDNN_OFFSET_WORKSPACES.get(key)
    if offsets is None:
        token_offsets = torch.arange(
            0,
            (rows + 1) * q_len,
            q_len,
            dtype=torch.int32,
            device=device,
        )
        offsets = (token_offsets * int(num_heads) * int(head_dim)).contiguous()
        _CUDNN_OFFSET_WORKSPACES[key] = offsets
    return offsets


def _get_or_create_cudnn_output_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    rows: int,
    q_len: int,
    num_key_value_groups: int,
    head_dim: int,
) -> torch.Tensor:
    key = (
        _device_key(device),
        str(dtype),
        int(rows),
        int(q_len),
        int(num_key_value_groups),
        int(head_dim),
    )
    out = _CUDNN_OUTPUT_WORKSPACES.get(key)
    if out is None:
        out = torch.empty(
            (rows * q_len, num_key_value_groups, head_dim),
            device=device,
            dtype=dtype,
        )
        _CUDNN_OUTPUT_WORKSPACES[key] = out
    return out


def _get_or_create_cudnn_lse_workspace(
    *,
    device: torch.device,
    rows: int,
    q_len: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    key = (_device_key(device), int(rows), int(q_len), int(num_key_value_groups))
    lse = _CUDNN_LSE_WORKSPACES.get(key)
    if lse is None:
        lse = torch.empty(
            (rows, q_len, num_key_value_groups),
            device=device,
            dtype=torch.float32,
        )
        _CUDNN_LSE_WORKSPACES[key] = lse
    return lse


def _build_q_fi_layout(
    *,
    q: torch.Tensor,
    bsz: int,
    q_len: int,
    num_kv_heads: int,
    num_key_value_groups: int,
    head_dim: int,
) -> torch.Tensor:
    rows = bsz * num_kv_heads
    if bsz == 1 and q.is_contiguous():
        q_fi = q.squeeze(0).as_strided(
            (num_kv_heads, q_len, num_key_value_groups, head_dim),
            (num_key_value_groups * head_dim, num_kv_heads * num_key_value_groups * head_dim, head_dim, 1),
        ).reshape(rows * q_len, num_key_value_groups, head_dim)
    else:
        q_ws, _ = _get_or_create_q_layout_workspace(
            device=q.device,
            dtype=q.dtype,
            bsz=bsz,
            q_len=q_len,
            num_kv_heads=num_kv_heads,
            num_key_value_groups=num_key_value_groups,
            head_dim=head_dim,
        )
        q_group_5d = q_ws["q_group_5d"]
        if can_use_cuda_pack_q_for_indexed_prefill(
            q=q,
            q_group=q_group_5d,
            num_kv_heads=num_kv_heads,
            num_key_value_groups=num_key_value_groups,
        ):
            pack_q_for_indexed_prefill_cuda(
                q=q,
                q_group=q_group_5d,
                num_kv_heads=num_kv_heads,
                num_key_value_groups=num_key_value_groups,
            )
        else:
            q_group_5d.copy_(
                q.view(bsz, q_len, num_kv_heads, num_key_value_groups, head_dim).permute(0, 2, 1, 3, 4)
            )
        q_fi = q_group_5d.view(rows * q_len, num_key_value_groups, head_dim)
    if not q_fi.is_contiguous():
        q_fi = q_fi.contiguous()
    return q_fi


def _build_index_metadata_reference_from_keep_block_kv(
    *,
    keep_block_kv: torch.Tensor,
    q_len: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
    bsz, num_kv_heads, n_blocks = keep_block_kv.shape
    rows = bsz * num_kv_heads
    device = keep_block_kv.device
    static_metadata = _get_static_metadata(
        device=device,
        rows=rows,
        n_blocks=n_blocks,
        q_len=q_len,
        block_size=block_size,
    )
    keep_flat = keep_block_kv.view(rows, n_blocks)
    pages_per_row = keep_flat.sum(dim=1).to(torch.int32)
    kv_indptr = torch.zeros(rows + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = pages_per_row.cumsum(0)
    kv_indices = static_metadata["global_ids"][keep_flat]
    return (
        pages_per_row,
        kv_indptr,
        kv_indices,
        static_metadata["kv_last_page_len"],
        static_metadata["qo_indptr"],
        None,
    )


def _can_use_fast_index_metadata_builder(
    *,
    keep_block_kv: torch.Tensor,
) -> bool:
    if not (keep_block_kv.is_cuda and keep_block_kv.is_contiguous() and keep_block_kv.dtype == torch.bool):
        return False
    rows = int(keep_block_kv.shape[0]) * int(keep_block_kv.shape[1])
    n_blocks = int(keep_block_kv.shape[2])
    ws = _get_or_create_metadata_workspace(device=keep_block_kv.device, rows=rows, kv_blocks=n_blocks)
    return (
        can_use_cuda_build_selected_indices_from_kv_keep_block(
            keep_block_kv=keep_block_kv,
            selected_block_indices=ws["selected_block_indices_i32"],
            selected_block_counts=ws["selected_block_counts_i32"],
        )
        and can_use_cuda_build_flashinfer_kv_indices(
            selected_block_indices=ws["selected_block_indices_i32"],
            selected_block_counts=ws["selected_block_counts_i32"],
            kv_indptr=ws["kv_indptr_i32"],
            kv_indices=ws["kv_indices_i32"],
        )
    )


def _build_index_metadata_fast_from_keep_block_kv(
    *,
    keep_block_kv: torch.Tensor,
    q_len: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, num_kv_heads, n_blocks = keep_block_kv.shape
    rows = bsz * num_kv_heads
    device = keep_block_kv.device
    static_metadata = _get_static_metadata(
        device=device,
        rows=rows,
        n_blocks=n_blocks,
        q_len=q_len,
        block_size=block_size,
    )
    ws = _get_or_create_metadata_workspace(device=device, rows=rows, kv_blocks=n_blocks)
    selected_block_indices = ws["selected_block_indices_i32"]
    selected_block_counts = ws["selected_block_counts_i32"]
    kv_indptr = ws["kv_indptr_i32"]
    kv_indices = ws["kv_indices_i32"]
    build_selected_indices_from_kv_keep_block_cuda_out(
        keep_block_kv=keep_block_kv,
        selected_block_indices=selected_block_indices,
        selected_block_counts=selected_block_counts,
    )
    kv_indptr.zero_()
    torch.cumsum(selected_block_counts, dim=0, dtype=torch.int32, out=kv_indptr[1:])
    build_flashinfer_kv_indices_cuda_out(
        selected_block_indices=selected_block_indices,
        selected_block_counts=selected_block_counts,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_blocks=n_blocks,
    )
    return (
        selected_block_counts,
        kv_indptr,
        kv_indices,
        static_metadata["kv_last_page_len"],
        static_metadata["qo_indptr"],
        selected_block_counts,
    )


def _build_index_metadata_reference_from_keep_block_q(
    *,
    keep_block_q: torch.Tensor,
    num_key_value_groups: int,
    q_len: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
    bsz, num_q_heads, n_blocks = keep_block_q.shape
    if num_q_heads % num_key_value_groups != 0:
        raise ValueError(
            f"Hq={num_q_heads} must be divisible by G={num_key_value_groups}"
        )
    num_kv_heads = num_q_heads // num_key_value_groups
    rows = bsz * num_q_heads
    device = keep_block_q.device
    static_metadata = _get_static_metadata(
        device=device,
        rows=rows,
        n_blocks=n_blocks,
        q_len=q_len,
        block_size=block_size,
    )

    keep_flat = keep_block_q.view(rows, n_blocks)
    pages_per_row = keep_flat.sum(dim=1).to(torch.int32)
    kv_indptr = torch.zeros(rows + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = pages_per_row.cumsum(0)

    block_ids = torch.arange(n_blocks, device=device, dtype=torch.int32)
    q_head = torch.arange(num_q_heads, device=device, dtype=torch.int32)
    kv_head = torch.div(q_head, int(num_key_value_groups), rounding_mode="floor")
    batch_offsets = (
        torch.arange(bsz, device=device, dtype=torch.int32)
        * int(num_kv_heads)
        * int(n_blocks)
    )
    head_offsets = kv_head * int(n_blocks)
    page_ids = (
        batch_offsets[:, None, None]
        + head_offsets[None, :, None]
        + block_ids[None, None, :]
    )
    kv_indices = page_ids.reshape(rows, n_blocks)[keep_flat]
    return (
        pages_per_row,
        kv_indptr,
        kv_indices,
        static_metadata["kv_last_page_len"],
        static_metadata["qo_indptr"],
        None,
    )


def _can_use_fast_index_metadata_builder_q(
    *,
    keep_block_q: torch.Tensor,
) -> bool:
    if not (keep_block_q.is_cuda and keep_block_q.is_contiguous() and keep_block_q.dtype == torch.bool):
        return False
    rows = int(keep_block_q.shape[0]) * int(keep_block_q.shape[1])
    n_blocks = int(keep_block_q.shape[2])
    ws = _get_or_create_metadata_workspace(device=keep_block_q.device, rows=rows, kv_blocks=n_blocks)
    return (
        can_use_cuda_build_selected_indices_from_kv_keep_block(
            keep_block_kv=keep_block_q,
            selected_block_indices=ws["selected_block_indices_i32"],
            selected_block_counts=ws["selected_block_counts_i32"],
        )
        and can_use_cuda_build_flashinfer_kv_indices(
            selected_block_indices=ws["selected_block_indices_i32"],
            selected_block_counts=ws["selected_block_counts_i32"],
            kv_indptr=ws["kv_indptr_i32"],
            kv_indices=ws["kv_indices_i32"],
        )
    )


def _build_index_metadata_fast_from_keep_block_q(
    *,
    keep_block_q: torch.Tensor,
    num_key_value_groups: int,
    q_len: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, num_q_heads, n_blocks = keep_block_q.shape
    if num_q_heads % num_key_value_groups != 0:
        raise ValueError(
            f"Hq={num_q_heads} must be divisible by G={num_key_value_groups}"
        )
    num_kv_heads = num_q_heads // num_key_value_groups
    rows = bsz * num_q_heads
    device = keep_block_q.device
    static_metadata = _get_static_metadata(
        device=device,
        rows=rows,
        n_blocks=n_blocks,
        q_len=q_len,
        block_size=block_size,
    )
    ws = _get_or_create_metadata_workspace(device=device, rows=rows, kv_blocks=n_blocks)
    selected_block_indices = ws["selected_block_indices_i32"]
    selected_block_counts = ws["selected_block_counts_i32"]
    kv_indptr = ws["kv_indptr_i32"]
    kv_indices = ws["kv_indices_i32"]
    build_selected_indices_from_kv_keep_block_cuda_out(
        keep_block_kv=keep_block_q,
        selected_block_indices=selected_block_indices,
        selected_block_counts=selected_block_counts,
    )
    kv_indptr.zero_()
    torch.cumsum(selected_block_counts, dim=0, dtype=torch.int32, out=kv_indptr[1:])
    build_flashinfer_kv_indices_per_query_cuda_out(
        selected_block_indices=selected_block_indices,
        selected_block_counts=selected_block_counts,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_blocks=n_blocks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_key_value_groups=num_key_value_groups,
    )
    return (
        selected_block_counts,
        kv_indptr,
        kv_indices,
        static_metadata["kv_last_page_len"],
        static_metadata["qo_indptr"],
        selected_block_counts,
    )


def debug_compare_index_metadata_builders(
    *,
    keep_block_kv: torch.Tensor,
    q_len: int,
    block_size: int = 64,
) -> Dict[str, object]:
    ref_pages_per_row, ref_kv_indptr, ref_kv_indices, _, _, _ = (
        _build_index_metadata_reference_from_keep_block_kv(
            keep_block_kv=keep_block_kv,
            q_len=q_len,
            block_size=block_size,
        )
    )
    if not _can_use_fast_index_metadata_builder(keep_block_kv=keep_block_kv):
        return {
            "fast_builder_available": False,
            "rows": int(keep_block_kv.shape[0] * keep_block_kv.shape[1]),
            "kv_blocks": int(keep_block_kv.shape[2]),
        }
    fast_counts, fast_kv_indptr, fast_kv_indices, _, _, _ = (
        _build_index_metadata_fast_from_keep_block_kv(
            keep_block_kv=keep_block_kv,
            q_len=q_len,
            block_size=block_size,
        )
    )
    selected_pages = int(fast_kv_indptr[-1].item())
    fast_kv_indices_valid = fast_kv_indices[:selected_pages]
    return {
        "fast_builder_available": True,
        "rows": int(keep_block_kv.shape[0] * keep_block_kv.shape[1]),
        "kv_blocks": int(keep_block_kv.shape[2]),
        "selected_pages": int(selected_pages),
        "pages_per_row_match": bool(torch.equal(ref_pages_per_row, fast_counts)),
        "kv_indptr_match": bool(torch.equal(ref_kv_indptr, fast_kv_indptr)),
        "kv_indices_match": bool(torch.equal(ref_kv_indices, fast_kv_indices_valid)),
        "pages_per_row_max_abs_diff": int(
            (ref_pages_per_row - fast_counts).abs().max().item()
        ) if ref_pages_per_row.numel() > 0 else 0,
        "kv_indptr_max_abs_diff": int(
            (ref_kv_indptr - fast_kv_indptr).abs().max().item()
        ) if ref_kv_indptr.numel() > 0 else 0,
        "kv_indices_max_abs_diff": int(
            (ref_kv_indices - fast_kv_indices_valid).abs().max().item()
        ) if ref_kv_indices.numel() > 0 else 0,
    }


def flashinfer_prefill_zero_copy_per_query(
    q: torch.Tensor,              # [bsz, q_len, Hq, D]
    k_hf: torch.Tensor,           # [bsz, Hkv, kv_len, D] contiguous
    v_hf: torch.Tensor,           # [bsz, Hkv, kv_len, D] contiguous
    keep_block_q: torch.Tensor,   # [bsz, Hq, n_blocks] bool
    num_key_value_groups: int,
    softmax_scale: float,
    block_size: int = 64,
    measure_timing: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Zero-copy FlashInfer prefill with a query-head-specific block table.

    This experimental path avoids KV-group union by treating each query head as
    its own pseudo-batch row. K/V pages are still the heads-first KV-cache pages;
    multiple query rows may reference the same KV-head page without copying it.
    """
    import flashinfer

    bsz, q_len, Hq, D = q.shape
    _, Hkv, kv_len, _ = k_hf.shape
    G = num_key_value_groups
    rows = bsz * Hq
    n_blocks = kv_len // block_size
    device = q.device

    if Hq != Hkv * G:
        raise ValueError(f"Hq={Hq} must equal Hkv={Hkv} * G={G}")
    if keep_block_q.shape != (bsz, Hq, n_blocks):
        raise ValueError(
            f"keep_block_q shape {tuple(keep_block_q.shape)} does not match "
            f"(B,Hq,n_blocks)=({bsz},{Hq},{n_blocks})"
        )
    if kv_len % block_size != 0:
        raise ValueError(f"kv_len={kv_len} must be divisible by block_size={block_size} for fi_zero_copy_per_query")
    if not k_hf.is_contiguous() or not v_hf.is_contiguous():
        raise ValueError("k_hf and v_hf must be contiguous for zero-copy page views")

    k_pages = k_hf.view(bsz * Hkv * n_blocks, block_size, D).unsqueeze(2)
    v_pages = v_hf.view(bsz * Hkv * n_blocks, block_size, D).unsqueeze(2)

    build_metadata_fn = (
        lambda: _build_index_metadata_fast_from_keep_block_q(
            keep_block_q=keep_block_q,
            num_key_value_groups=G,
            q_len=q_len,
            block_size=block_size,
        )
    ) if _can_use_fast_index_metadata_builder_q(keep_block_q=keep_block_q) else (
        lambda: _build_index_metadata_reference_from_keep_block_q(
            keep_block_q=keep_block_q,
            num_key_value_groups=G,
            q_len=q_len,
            block_size=block_size,
        )
    )

    (
        pages_per_row,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        qo_indptr,
        selected_block_counts,
    ), zc_metadata_ms = _cuda_elapsed_ms(
        build_metadata_fn,
        enabled=measure_timing,
        device=device,
    )
    if selected_block_counts is not None:
        selected_pages = int(kv_indptr[-1].item())
        kv_indices_for_plan = kv_indices[:selected_pages]
    else:
        selected_pages = int(kv_indices.numel())
        kv_indices_for_plan = kv_indices

    q_fi, zc_q_layout_ms = _cuda_elapsed_ms(
        lambda: q.permute(0, 2, 1, 3).contiguous().view(rows * q_len, 1, D),
        enabled=measure_timing,
        device=device,
    )

    workspace = _get_or_create_fi_workspace(device)

    wrapper, zc_wrapper_init_ms, zc_wrapper_init_host_ms = _host_and_cuda_elapsed_ms(
        lambda: _get_wrapper(
            flashinfer=flashinfer,
            workspace=workspace,
            device=device,
            dtype=q.dtype,
            num_key_value_groups=1,
            head_dim=D,
            block_size=block_size,
            rows=rows,
            q_len=q_len,
            kv_indices_capacity=max(selected_pages, 1),
        ),
        enabled=measure_timing,
        device=device,
    )

    def _run_plan():
        fixed_split_size = _env_optional_int("SEER_ZC_FIXED_SPLIT_SIZE")
        disable_split_kv = _env_flag("SEER_ZC_DISABLE_SPLIT_KV", False)
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices_for_plan,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=D,
            page_size=block_size,
            causal=True,
            sm_scale=softmax_scale,
            q_data_type=q.dtype,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    _, zc_plan_ms, zc_plan_host_ms = _host_and_cuda_elapsed_ms(
        _run_plan,
        enabled=measure_timing,
        device=device,
    )

    out_fi_buf = (
        _get_or_create_output_workspace(
            device=device,
            dtype=q.dtype,
            rows=rows,
            q_len=q_len,
            num_key_value_groups=1,
            head_dim=D,
        )
        if _env_flag("SEER_ZC_PREALLOC_OUT", True)
        else None
    )
    out_fi, zc_run_ms = _cuda_elapsed_ms(
        lambda: wrapper.run(q_fi, (k_pages, v_pages), out=out_fi_buf),
        enabled=measure_timing,
        device=device,
    )
    out = out_fi.view(bsz, Hq, q_len, D).permute(0, 2, 1, 3).contiguous()
    zc_attn_ms = zc_q_layout_ms + zc_wrapper_init_ms + zc_plan_ms + zc_run_ms
    zc_total_ms = zc_metadata_ms + zc_attn_ms
    return out, {
        "zc_metadata_ms": float(zc_metadata_ms),
        "zc_q_layout_ms": float(zc_q_layout_ms),
        "zc_wrapper_init_ms": float(zc_wrapper_init_ms),
        "zc_wrapper_init_host_ms": float(zc_wrapper_init_host_ms),
        "zc_plan_ms": float(zc_plan_ms),
        "zc_plan_host_ms": float(zc_plan_host_ms),
        "zc_run_ms": float(zc_run_ms),
        "zc_attn_ms": float(zc_attn_ms),
        "zc_total_ms": float(zc_total_ms),
        "zc_rows": float(rows),
        "zc_kv_blocks": float(n_blocks),
        "zc_selected_pages": float(selected_pages),
        "zc_q_tokens": float(rows * q_len),
        "zc_pages_per_row_sum": float(selected_pages),
        "zc_plan_full_kv_indices": 0.0,
        "zc_plan_full_kv_indices_min_blocks": -1.0,
        "zc_prealloc_out": float(out_fi_buf is not None),
        "zc_fixed_split_size": float(_env_optional_int("SEER_ZC_FIXED_SPLIT_SIZE") or -1),
        "zc_disable_split_kv": float(_env_flag("SEER_ZC_DISABLE_SPLIT_KV", False)),
        "zc_per_query_block_table": 1.0,
    }


def flashinfer_prefill_zero_copy_subgroup(
    q: torch.Tensor,                    # [bsz, q_len, Hq, D]
    k_hf: torch.Tensor,                 # [bsz, Hkv, kv_len, D] contiguous
    v_hf: torch.Tensor,                 # [bsz, Hkv, kv_len, D] contiguous
    keep_block_group: torch.Tensor,     # [bsz, Hkv*num_subgroups, n_blocks] bool
    num_key_value_groups: int,
    query_subgroup_size: int,
    softmax_scale: float,
    block_size: int = 64,
    measure_timing: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Zero-copy FlashInfer prefill with query-head subgroup block tables.

    This path is the middle point between KV-head union and per-query-head rows:
    each KV head is split into G / query_subgroup_size pseudo rows, and each row
    executes query_subgroup_size query heads over its own selected page list.
    """
    import flashinfer

    bsz, q_len, Hq, D = q.shape
    _, Hkv, kv_len, _ = k_hf.shape
    G = int(num_key_value_groups)
    subgroup_size = int(query_subgroup_size)
    if subgroup_size <= 0 or G % subgroup_size != 0:
        raise ValueError(
            f"query_subgroup_size={subgroup_size} must be positive and divide G={G}"
        )
    num_subgroups = G // subgroup_size
    rows = bsz * Hkv * num_subgroups
    n_blocks = kv_len // block_size
    device = q.device

    if Hq != Hkv * G:
        raise ValueError(f"Hq={Hq} must equal Hkv={Hkv} * G={G}")
    if keep_block_group.shape != (bsz, Hkv * num_subgroups, n_blocks):
        raise ValueError(
            f"keep_block_group shape {tuple(keep_block_group.shape)} does not match "
            f"(B,Hkv*num_subgroups,n_blocks)=({bsz},{Hkv * num_subgroups},{n_blocks})"
        )
    if kv_len % block_size != 0:
        raise ValueError(f"kv_len={kv_len} must be divisible by block_size={block_size} for fi_zero_copy_subgroup")
    if not k_hf.is_contiguous() or not v_hf.is_contiguous():
        raise ValueError("k_hf and v_hf must be contiguous for zero-copy page views")

    k_pages = k_hf.view(bsz * Hkv * n_blocks, block_size, D).unsqueeze(2)
    v_pages = v_hf.view(bsz * Hkv * n_blocks, block_size, D).unsqueeze(2)

    # Reuse the per-query metadata builder by treating subgroup rows as
    # pseudo query heads. The page-id mapping then resolves each subgroup row
    # back to its owning KV head.
    build_metadata_fn = (
        lambda: _build_index_metadata_fast_from_keep_block_q(
            keep_block_q=keep_block_group,
            num_key_value_groups=num_subgroups,
            q_len=q_len,
            block_size=block_size,
        )
    ) if _can_use_fast_index_metadata_builder_q(keep_block_q=keep_block_group) else (
        lambda: _build_index_metadata_reference_from_keep_block_q(
            keep_block_q=keep_block_group,
            num_key_value_groups=num_subgroups,
            q_len=q_len,
            block_size=block_size,
        )
    )

    (
        pages_per_row,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        qo_indptr,
        selected_block_counts,
    ), zc_metadata_ms = _cuda_elapsed_ms(
        build_metadata_fn,
        enabled=measure_timing,
        device=device,
    )
    # Subgroup execution has many more planning rows than the KV-head path. Avoiding
    # exact selected-page slicing removes a host sync and stabilizes FlashInfer plan
    # inputs without changing the subgroup keep mask semantics.
    full_kv_indices_min_blocks = _env_int_with_fallback(
        "SEER_ZC_SUBGROUP_PLAN_FULL_KV_INDICES_MIN_BLOCKS",
        "SEER_ZC_PLAN_FULL_KV_INDICES_MIN_BLOCKS",
        1,
    )
    use_full_kv_indices = (
        selected_block_counts is not None
        and _env_flag("SEER_ZC_PLAN_FULL_KV_INDICES", True)
        and n_blocks >= full_kv_indices_min_blocks
    )
    if selected_block_counts is not None and use_full_kv_indices:
        selected_pages = -1
        kv_indices_for_plan = kv_indices
    elif selected_block_counts is not None:
        selected_pages = int(kv_indptr[-1].item())
        kv_indices_for_plan = kv_indices[:selected_pages]
    else:
        selected_pages = int(kv_indices.numel())
        kv_indices_for_plan = kv_indices

    def _build_q_fi():
        q_local = q if q.is_contiguous() else q.contiguous()
        return (
            q_local.view(bsz, q_len, Hkv, num_subgroups, subgroup_size, D)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(rows * q_len, subgroup_size, D)
        )

    q_fi, zc_q_layout_ms = _cuda_elapsed_ms(
        _build_q_fi,
        enabled=measure_timing,
        device=device,
    )

    workspace = _get_or_create_fi_workspace(device)

    wrapper, zc_wrapper_init_ms, zc_wrapper_init_host_ms = _host_and_cuda_elapsed_ms(
        lambda: _get_wrapper(
            flashinfer=flashinfer,
            workspace=workspace,
            device=device,
            dtype=q.dtype,
            num_key_value_groups=subgroup_size,
            head_dim=D,
            block_size=block_size,
            rows=rows,
            q_len=q_len,
            kv_indices_capacity=kv_indices.numel(),
        ),
        enabled=measure_timing,
        device=device,
    )

    def _run_plan():
        fixed_split_size = _env_optional_int("SEER_ZC_FIXED_SPLIT_SIZE")
        disable_split_kv = _env_flag("SEER_ZC_DISABLE_SPLIT_KV", False)
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices_for_plan,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=subgroup_size,
            num_kv_heads=1,
            head_dim_qk=D,
            page_size=block_size,
            causal=True,
            sm_scale=softmax_scale,
            q_data_type=q.dtype,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    _, zc_plan_ms, zc_plan_host_ms = _host_and_cuda_elapsed_ms(
        _run_plan,
        enabled=measure_timing,
        device=device,
    )

    out_fi_buf = (
        _get_or_create_output_workspace(
            device=device,
            dtype=q.dtype,
            rows=rows,
            q_len=q_len,
            num_key_value_groups=subgroup_size,
            head_dim=D,
        )
        if _env_flag("SEER_ZC_PREALLOC_OUT", True)
        else None
    )
    out_fi, zc_run_ms = _cuda_elapsed_ms(
        lambda: wrapper.run(q_fi, (k_pages, v_pages), out=out_fi_buf),
        enabled=measure_timing,
        device=device,
    )
    out = (
        out_fi.reshape(bsz, Hkv, num_subgroups, q_len, subgroup_size, D)
        .permute(0, 3, 1, 2, 4, 5)
        .reshape(bsz, q_len, Hq, D)
    )
    zc_attn_ms = zc_q_layout_ms + zc_wrapper_init_ms + zc_plan_ms + zc_run_ms
    zc_total_ms = zc_metadata_ms + zc_attn_ms
    return out, {
        "zc_metadata_ms": float(zc_metadata_ms),
        "zc_q_layout_ms": float(zc_q_layout_ms),
        "zc_wrapper_init_ms": float(zc_wrapper_init_ms),
        "zc_wrapper_init_host_ms": float(zc_wrapper_init_host_ms),
        "zc_plan_ms": float(zc_plan_ms),
        "zc_plan_host_ms": float(zc_plan_host_ms),
        "zc_run_ms": float(zc_run_ms),
        "zc_attn_ms": float(zc_attn_ms),
        "zc_total_ms": float(zc_total_ms),
        "zc_rows": float(rows),
        "zc_kv_blocks": float(n_blocks),
        "zc_selected_pages": float(selected_pages),
        "zc_q_tokens": float(rows * q_len),
        "zc_pages_per_row_sum": float(selected_pages),
        "zc_plan_full_kv_indices": float(use_full_kv_indices),
        "zc_plan_full_kv_indices_min_blocks": float(full_kv_indices_min_blocks),
        "zc_prealloc_out": float(out_fi_buf is not None),
        "zc_fixed_split_size": float(_env_optional_int("SEER_ZC_FIXED_SPLIT_SIZE") or -1),
        "zc_disable_split_kv": float(_env_flag("SEER_ZC_DISABLE_SPLIT_KV", False)),
        "zc_query_subgroup_size": float(subgroup_size),
        "zc_query_subgroups": float(num_subgroups),
    }


def flashinfer_prefill_zero_copy(
    q: torch.Tensor,             # [bsz, q_len, Hq, D]
    k_hf: torch.Tensor,         # [bsz, Hkv, kv_len, D]  contiguous
    v_hf: torch.Tensor,         # [bsz, Hkv, kv_len, D]  contiguous
    keep_block_kv: torch.Tensor, # [bsz, Hkv, n_blocks]   bool
    num_key_value_groups: int,
    softmax_scale: float,
    block_size: int = 64,
    measure_timing: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Batched prefill attention with zero-copy access to a heads-first KV cache.

    No materialization (cache-fill) step.  k_hf / v_hf are viewed as FlashInfer
    pages in-place; only the selected blocks are read by the kernel.

    Args:
        q:             [bsz, q_len, Hq, D]        query (after RoPE, seq-first)
        k_hf:          [bsz, Hkv, kv_len, D]      full accumulated KV (heads-first, contiguous)
        v_hf:          [bsz, Hkv, kv_len, D]
        keep_block_kv: [bsz, Hkv, n_blocks] bool  per-KV-head block selection mask
        num_key_value_groups: G = Hq // Hkv
        softmax_scale: 1 / sqrt(D)
        block_size:    tokens per block (default 64)
        measure_timing: ignored for now (reserved for future profiling)

    Returns:
        attn_output:  [bsz, q_len, Hq, D]
        stats:        timing dict (empty for now)
    """
    global _ZC_CALL_COUNT
    _ZC_CALL_COUNT += 1
    if _ZC_CALL_COUNT == 1 or os.environ.get("SEER_ZC_DEBUG", "0") == "1":
        rank = os.environ.get("RANK", "0")
        print(f"[fi_zero_copy rank={rank}] call #{_ZC_CALL_COUNT} k_hf={tuple(k_hf.shape)} contiguous={k_hf.is_contiguous()}", flush=True)

    import flashinfer

    bsz, q_len, Hq, D = q.shape
    _, Hkv, kv_len, _ = k_hf.shape
    G = num_key_value_groups
    rows = bsz * Hkv
    n_blocks = kv_len // block_size
    device = q.device

    if kv_len % block_size != 0:
        raise ValueError(f"kv_len={kv_len} must be divisible by block_size={block_size} for fi_zero_copy")
    if not k_hf.is_contiguous():
        raise ValueError("k_hf must be contiguous for zero-copy page view")

    # --- Zero-copy page view ---
    # k_hf [bsz, Hkv, kv_len, D]  (contiguous) is equivalent to
    # [bsz, Hkv, n_blocks, block_size, D].  Flatten first three dims then
    # unsqueeze the head dim: [rows*n_blocks, block_size, 1, D].
    k_pages = k_hf.view(rows * n_blocks, block_size, D).unsqueeze(2)  # zero-copy
    v_pages = v_hf.view(rows * n_blocks, block_size, D).unsqueeze(2)  # zero-copy

    # --- Index construction from keep_block_kv ---
    build_index_metadata_fn = (
        lambda: _build_index_metadata_fast_from_keep_block_kv(
            keep_block_kv=keep_block_kv,
            q_len=q_len,
            block_size=block_size,
        )
    ) if _can_use_fast_index_metadata_builder(keep_block_kv=keep_block_kv) else (
        lambda: _build_index_metadata_reference_from_keep_block_kv(
            keep_block_kv=keep_block_kv,
            q_len=q_len,
            block_size=block_size,
        )
    )

    (
        pages_per_row_or_counts,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        qo_indptr,
        selected_block_counts,
    ), zc_metadata_ms = _cuda_elapsed_ms(
        build_index_metadata_fn,
        enabled=measure_timing,
        device=device,
    )
    # Full-buffer planning avoids a sync from exact slicing, but it is slower at short
    # context widths. Keep early chunks on exact slices and switch once history is wider.
    full_kv_indices_min_blocks = _env_int("SEER_ZC_PLAN_FULL_KV_INDICES_MIN_BLOCKS", 257)
    use_full_kv_indices = (
        selected_block_counts is not None
        and _env_flag("SEER_ZC_PLAN_FULL_KV_INDICES", True)
        and n_blocks >= full_kv_indices_min_blocks
    )
    if selected_block_counts is not None and use_full_kv_indices:
        selected_pages = -1
        kv_indices_for_plan = kv_indices
    elif selected_block_counts is not None:
        selected_pages = int(kv_indptr[-1].item())
        kv_indices_for_plan = kv_indices[:selected_pages]
    else:
        selected_pages = int(kv_indices.numel())
        kv_indices_for_plan = kv_indices

    # --- Query layout: [bsz, q_len, Hq, D] → [rows*q_len, G, D] ---
    # For pseudo-batch row r = b*Hkv + h, the G query heads are h*G … (h+1)*G.
    def _build_q_fi():
        return _build_q_fi_layout(
            q=q,
            bsz=bsz,
            q_len=q_len,
            num_kv_heads=Hkv,
            num_key_value_groups=G,
            head_dim=D,
        )

    q_fi, zc_q_layout_ms = _cuda_elapsed_ms(
        _build_q_fi,
        enabled=measure_timing,
        device=device,
    )

    # --- FlashInfer BatchPrefill ---
    workspace = _get_or_create_fi_workspace(device)

    def _make_wrapper():
        return _get_wrapper(
            flashinfer=flashinfer,
            workspace=workspace,
            device=device,
            dtype=q.dtype,
            num_key_value_groups=G,
            head_dim=D,
            block_size=block_size,
            rows=rows,
            q_len=q_len,
            kv_indices_capacity=kv_indices.numel(),
        )

    wrapper, zc_wrapper_init_ms, zc_wrapper_init_host_ms = _host_and_cuda_elapsed_ms(
        _make_wrapper,
        enabled=measure_timing,
        device=device,
    )

    def _run_plan():
        fixed_split_size = _env_optional_int("SEER_ZC_FIXED_SPLIT_SIZE")
        disable_split_kv = _env_flag("SEER_ZC_DISABLE_SPLIT_KV", False)
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices_for_plan,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=G,
            num_kv_heads=1,
            head_dim_qk=D,
            page_size=block_size,
            causal=True,
            sm_scale=softmax_scale,
            q_data_type=q.dtype,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    _, zc_plan_ms, zc_plan_host_ms = _host_and_cuda_elapsed_ms(
        _run_plan,
        enabled=measure_timing,
        device=device,
    )

    out_fi_buf = (
        _get_or_create_output_workspace(
            device=device,
            dtype=q.dtype,
            rows=rows,
            q_len=q_len,
            num_key_value_groups=G,
            head_dim=D,
        )
        if _env_flag("SEER_ZC_PREALLOC_OUT", True)
        else None
    )

    out_fi, zc_run_ms = _cuda_elapsed_ms(
        lambda: wrapper.run(q_fi, (k_pages, v_pages), out=out_fi_buf),
        enabled=measure_timing,
        device=device,
    )

    # --- Output layout: [rows*q_len, G, D] → [bsz, q_len, Hq, D] ---
    out = (
        out_fi.reshape(bsz, Hkv, q_len, G, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(bsz, q_len, Hq, D)
    )
    zc_attn_ms = zc_q_layout_ms + zc_wrapper_init_ms + zc_plan_ms + zc_run_ms
    zc_total_ms = zc_metadata_ms + zc_attn_ms
    return out, {
        "zc_metadata_ms": float(zc_metadata_ms),
        "zc_q_layout_ms": float(zc_q_layout_ms),
        "zc_wrapper_init_ms": float(zc_wrapper_init_ms),
        "zc_wrapper_init_host_ms": float(zc_wrapper_init_host_ms),
        "zc_plan_ms": float(zc_plan_ms),
        "zc_plan_host_ms": float(zc_plan_host_ms),
        "zc_run_ms": float(zc_run_ms),
        "zc_attn_ms": float(zc_attn_ms),
        "zc_total_ms": float(zc_total_ms),
        "zc_rows": float(rows),
        "zc_kv_blocks": float(n_blocks),
        "zc_selected_pages": float(selected_pages),
        "zc_q_tokens": float(rows * q_len),
        "zc_pages_per_row_sum": float(selected_pages),
        "zc_plan_full_kv_indices": float(use_full_kv_indices),
        "zc_plan_full_kv_indices_min_blocks": float(full_kv_indices_min_blocks),
        "zc_prealloc_out": float(out_fi_buf is not None),
        "zc_fixed_split_size": float(_env_optional_int("SEER_ZC_FIXED_SPLIT_SIZE") or -1),
        "zc_disable_split_kv": float(_env_flag("SEER_ZC_DISABLE_SPLIT_KV", False)),
    }


def flashinfer_prefill_cudnn_one_shot(
    q: torch.Tensor,              # [bsz, q_len, Hq, D]
    k_hf: torch.Tensor,           # [bsz, Hkv, kv_len, D] contiguous
    v_hf: torch.Tensor,           # [bsz, Hkv, kv_len, D] contiguous
    keep_block_kv: torch.Tensor,  # [bsz, Hkv, n_blocks] bool
    num_key_value_groups: int,
    softmax_scale: float,
    block_size: int = 64,
    measure_timing: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Plan-free cuDNN paged prefill over the same zero-copy heads-first KV cache.

    This path uses the same selected KV pages as ``flashinfer_prefill_zero_copy``
    but calls ``cudnn_batch_prefill_with_kv_cache`` directly.  cuDNN expects KV
    pages as [num_pages, num_kv_heads, page_size, head_dim], so the zero-copy
    page view is [rows * n_blocks, 1, block_size, D].
    """
    bsz, q_len, Hq, D = q.shape
    _, Hkv, kv_len, _ = k_hf.shape
    G = num_key_value_groups
    rows = bsz * Hkv
    n_blocks = kv_len // block_size
    device = q.device

    if kv_len % block_size != 0:
        raise ValueError(f"kv_len={kv_len} must be divisible by block_size={block_size} for cudnn_one_shot")
    if not k_hf.is_contiguous() or not v_hf.is_contiguous():
        raise ValueError("k_hf and v_hf must be contiguous for cudnn_one_shot page views")
    if Hq != Hkv * G:
        raise ValueError(f"Hq={Hq} must equal Hkv={Hkv} * G={G}")

    k_pages = k_hf.view(rows * n_blocks, block_size, D).unsqueeze(1)
    v_pages = v_hf.view(rows * n_blocks, block_size, D).unsqueeze(1)

    def _build_metadata():
        if _can_use_fast_index_metadata_builder(keep_block_kv=keep_block_kv):
            ws = _get_or_create_metadata_workspace(
                device=keep_block_kv.device,
                rows=rows,
                kv_blocks=n_blocks,
            )
            selected_block_indices = ws["selected_block_indices_i32"]
            selected_block_counts = ws["selected_block_counts_i32"]
            build_selected_indices_from_kv_keep_block_cuda_out(
                keep_block_kv=keep_block_kv,
                selected_block_indices=selected_block_indices,
                selected_block_counts=selected_block_counts,
            )
            exact_max_pages = max(int(selected_block_counts.max().item()), 1)
            graph_max_pages = _resolve_cudnn_graph_max_pages(
                exact_max_pages=exact_max_pages,
                n_blocks=n_blocks,
            )
            block_table = _get_or_create_cudnn_block_table_workspace(
                device=device,
                rows=rows,
                max_pages=graph_max_pages,
            )
            local_blocks = selected_block_indices[:, :exact_max_pages]
            row_offsets = _get_static_metadata(
                device=device,
                rows=rows,
                n_blocks=n_blocks,
                q_len=q_len,
                block_size=block_size,
            )["global_ids"][:, :1]
            block_table.zero_()
            block_table[:, :exact_max_pages].copy_(local_blocks)
            block_table[:, :exact_max_pages].add_(row_offsets)
            slot_ids = torch.arange(graph_max_pages, device=device, dtype=torch.int32).unsqueeze(0)
            block_table.masked_fill_(slot_ids >= selected_block_counts.unsqueeze(1), 0)
            pages_per_row = selected_block_counts
        else:
            pages_per_row, kv_indptr, kv_indices, _, _, _ = (
                _build_index_metadata_reference_from_keep_block_kv(
                    keep_block_kv=keep_block_kv,
                    q_len=q_len,
                    block_size=block_size,
                )
            )
            exact_max_pages = max(int(pages_per_row.max().item()), 1)
            graph_max_pages = _resolve_cudnn_graph_max_pages(
                exact_max_pages=exact_max_pages,
                n_blocks=n_blocks,
            )
            block_table = _get_or_create_cudnn_block_table_workspace(
                device=device,
                rows=rows,
                max_pages=graph_max_pages,
            )
            block_table.zero_()
            # Fallback path is validation-oriented; production runs should use the CUDA builder.
            for row in range(rows):
                start = int(kv_indptr[row].item())
                end = int(kv_indptr[row + 1].item())
                if end > start:
                    block_table[row, : end - start].copy_(kv_indices[start:end])

        seq_ws = _get_or_create_cudnn_seq_len_workspaces(device=device, rows=rows)
        actual_seq_lens_q = seq_ws["actual_seq_lens_q"]
        actual_seq_lens_kv = seq_ws["actual_seq_lens_kv"]
        actual_seq_lens_q.fill_(int(q_len))
        actual_seq_lens_kv.view(rows).copy_(pages_per_row * int(block_size))
        q_offsets = _get_or_create_cudnn_offsets(
            device=device,
            rows=rows,
            q_len=q_len,
            num_heads=G,
            head_dim=D,
        )
        return (
            block_table,
            actual_seq_lens_q,
            actual_seq_lens_kv,
            q_offsets,
            pages_per_row,
            exact_max_pages,
            graph_max_pages,
        )

    (
        block_table,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        q_offsets,
        pages_per_row,
        exact_max_pages,
        graph_max_pages,
    ), cudnn_metadata_ms, cudnn_metadata_host_ms = _host_and_cuda_elapsed_ms(
        _build_metadata,
        enabled=measure_timing,
        device=device,
    )

    q_fi, cudnn_q_layout_ms = _cuda_elapsed_ms(
        lambda: _build_q_fi_layout(
            q=q,
            bsz=bsz,
            q_len=q_len,
            num_kv_heads=Hkv,
            num_key_value_groups=G,
            head_dim=D,
        ),
        enabled=measure_timing,
        device=device,
    )

    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

    workspace = _get_or_create_fi_workspace(device)
    out_fi = _get_or_create_cudnn_output_workspace(
        device=device,
        dtype=q.dtype,
        rows=rows,
        q_len=q_len,
        num_key_value_groups=G,
        head_dim=D,
    )
    lse = _get_or_create_cudnn_lse_workspace(
        device=device,
        rows=rows,
        q_len=q_len,
        num_key_value_groups=G,
    )

    def _run_cudnn():
        return cudnn_batch_prefill_with_kv_cache(
            q_fi,
            k_pages,
            v_pages,
            softmax_scale,
            workspace,
            max_token_per_sequence=q_len,
            max_sequence_kv=int(graph_max_pages) * int(block_size),
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_table,
            causal=True,
            return_lse=True,
            batch_offsets_q=q_offsets,
            batch_offsets_o=q_offsets,
            out=out_fi,
            lse=lse,
            is_cuda_graph_compatible=True,
        )[0]

    out_fi, cudnn_run_ms = _cuda_elapsed_ms(
        _run_cudnn,
        enabled=measure_timing,
        device=device,
    )

    out = (
        out_fi.reshape(bsz, Hkv, q_len, G, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(bsz, q_len, Hq, D)
    )
    cudnn_attn_ms = cudnn_q_layout_ms + cudnn_run_ms
    cudnn_total_ms = cudnn_metadata_ms + cudnn_attn_ms
    # Avoid an extra host sync in the hot path.  ``max_pages`` already requires
    # one sync today; the selected-page count is diagnostic-only.
    selected_pages = -1
    return out, {
        "zc_metadata_ms": float(cudnn_metadata_ms),
        "zc_metadata_host_ms": float(cudnn_metadata_host_ms),
        "zc_q_layout_ms": float(cudnn_q_layout_ms),
        "zc_wrapper_init_ms": 0.0,
        "zc_wrapper_init_host_ms": 0.0,
        "zc_plan_ms": 0.0,
        "zc_plan_host_ms": 0.0,
        "zc_run_ms": float(cudnn_run_ms),
        "zc_attn_ms": float(cudnn_attn_ms),
        "zc_total_ms": float(cudnn_total_ms),
        "zc_rows": float(rows),
        "zc_kv_blocks": float(n_blocks),
        "zc_selected_pages": float(selected_pages),
        "zc_q_tokens": float(rows * q_len),
        "zc_pages_per_row_sum": float(selected_pages),
        "zc_cudnn_one_shot": 1.0,
        "zc_cudnn_max_pages": float(graph_max_pages),
        "zc_cudnn_exact_max_pages": float(exact_max_pages),
        "zc_cudnn_graph_max_pages": float(graph_max_pages),
        "zc_cudnn_bucket_size": float(_env_int("SEER_CUDNN_MAX_PAGES_BUCKET_SIZE", 8)),
        "zc_cudnn_power2_bucket": float(_env_flag("SEER_CUDNN_MAX_PAGES_POWER2", False)),
        "zc_cudnn_full_blocks": float(_env_flag("SEER_CUDNN_MAX_PAGES_FULL_BLOCKS", False)),
    }
