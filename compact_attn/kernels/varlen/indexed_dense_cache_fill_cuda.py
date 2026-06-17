from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Optional

import torch


@lru_cache(maxsize=1)
def _load_extension() -> Optional[object]:
    try:
        return importlib.import_module("compact_attn._C_compactattn")
    except Exception:
        return None


def cuda_cache_fill_available() -> bool:
    return _load_extension() is not None


def can_use_cuda_build_block_table(
    pages_per_row: torch.Tensor,
    block_table: torch.Tensor,
    page_offsets: torch.Tensor,
) -> bool:
    if not cuda_cache_fill_available():
        return False
    if not (pages_per_row.is_cuda and block_table.is_cuda and page_offsets.is_cuda):
        return False
    if not (pages_per_row.is_contiguous() and block_table.is_contiguous() and page_offsets.is_contiguous()):
        return False
    if pages_per_row.dtype != torch.int32 or block_table.dtype != torch.int32 or page_offsets.dtype != torch.int32:
        return False
    if pages_per_row.ndim != 1 or block_table.ndim != 2 or page_offsets.ndim != 1:
        return False
    if block_table.shape[0] != pages_per_row.shape[0] or page_offsets.shape[0] != pages_per_row.shape[0]:
        return False
    return True


def can_use_cuda_compact_keep_blocks(keep_flat: torch.Tensor, sel_blocks: torch.Tensor) -> bool:
    if not cuda_cache_fill_available():
        return False
    if not (keep_flat.is_cuda and sel_blocks.is_cuda):
        return False
    if not (keep_flat.is_contiguous() and sel_blocks.is_contiguous()):
        return False
    if keep_flat.dtype != torch.bool or sel_blocks.dtype != torch.int32:
        return False
    if keep_flat.ndim != 2 or sel_blocks.ndim != 1:
        return False
    if keep_flat.shape[0] != sel_blocks.shape[0]:
        return False
    return True


def can_use_cuda_compact_keep_blocks_and_build_table(
    keep_flat: torch.Tensor,
    sel_blocks: torch.Tensor,
    pages_per_row: torch.Tensor,
    block_table: torch.Tensor,
    page_offsets: torch.Tensor,
) -> bool:
    return can_use_cuda_compact_keep_blocks(keep_flat, sel_blocks) and can_use_cuda_build_block_table(
        pages_per_row=pages_per_row,
        block_table=block_table,
        page_offsets=page_offsets,
    )


def can_use_cuda_keep_block_fast(attn_gate_score: torch.Tensor) -> bool:
    if not cuda_cache_fill_available():
        return False
    if not (attn_gate_score.is_cuda and attn_gate_score.is_contiguous()):
        return False
    if attn_gate_score.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if attn_gate_score.ndim != 4:
        return False
    return True


def can_use_cuda_build_selected_indices_from_kv_keep_block(
    keep_block_kv: torch.Tensor,
    selected_block_indices: torch.Tensor,
    selected_block_counts: torch.Tensor,
) -> bool:
    if not cuda_cache_fill_available():
        return False
    if not (keep_block_kv.is_cuda and selected_block_indices.is_cuda and selected_block_counts.is_cuda):
        return False
    if not (
        keep_block_kv.is_contiguous()
        and selected_block_indices.is_contiguous()
        and selected_block_counts.is_contiguous()
    ):
        return False
    if keep_block_kv.dtype != torch.bool:
        return False
    if selected_block_indices.dtype != torch.int32 or selected_block_counts.dtype != torch.int32:
        return False
    if keep_block_kv.ndim != 3 or selected_block_indices.ndim != 2 or selected_block_counts.ndim != 1:
        return False
    rows = keep_block_kv.shape[0] * keep_block_kv.shape[1]
    if selected_block_indices.shape[0] != rows or selected_block_counts.shape[0] != rows:
        return False
    if selected_block_indices.shape[1] < keep_block_kv.shape[2]:
        return False
    return True


def can_use_cuda_build_flashinfer_kv_indices(
    selected_block_indices: torch.Tensor,
    selected_block_counts: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
) -> bool:
    if not cuda_cache_fill_available():
        return False
    if not (
        selected_block_indices.is_cuda
        and selected_block_counts.is_cuda
        and kv_indptr.is_cuda
        and kv_indices.is_cuda
    ):
        return False
    if not (
        selected_block_indices.is_contiguous()
        and selected_block_counts.is_contiguous()
        and kv_indptr.is_contiguous()
        and kv_indices.is_contiguous()
    ):
        return False
    if (
        selected_block_indices.dtype != torch.int32
        or selected_block_counts.dtype != torch.int32
        or kv_indptr.dtype != torch.int32
        or kv_indices.dtype != torch.int32
    ):
        return False
    if (
        selected_block_indices.ndim != 2
        or selected_block_counts.ndim != 1
        or kv_indptr.ndim != 1
        or kv_indices.ndim != 1
    ):
        return False
    rows = selected_block_counts.shape[0]
    if selected_block_indices.shape[0] != rows or kv_indptr.shape[0] != rows + 1:
        return False
    return True


def can_use_cuda_cache_fill_from_pos_with_rank(
    k: torch.Tensor,
    v: torch.Tensor,
    pos: torch.Tensor,
    keep_prefix_rank: torch.Tensor,
    page_offsets: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
    page_block_size: int,
) -> bool:
    if not cuda_cache_fill_available():
        return False
    if block_size <= 0 or page_block_size <= 0 or (page_block_size % block_size) != 0:
        return False
    if k.dtype not in (torch.float16, torch.bfloat16):
        return False
    if v.dtype != k.dtype or k_cache_flat.dtype != k.dtype or v_cache_flat.dtype != k.dtype:
        return False
    if not (k.is_cuda and v.is_cuda and pos.is_cuda and keep_prefix_rank.is_cuda and page_offsets.is_cuda):
        return False
    if not (k_cache_flat.is_cuda and v_cache_flat.is_cuda):
        return False
    if not (k.is_contiguous() and v.is_contiguous()):
        return False
    if not (pos.is_contiguous() and keep_prefix_rank.is_contiguous() and page_offsets.is_contiguous()):
        return False
    if not (k_cache_flat.is_contiguous() and v_cache_flat.is_contiguous()):
        return False
    if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        return False
    if k.shape[-1] != 128:
        return False
    if pos.ndim != 2 or pos.shape[1] != 2 or pos.dtype != torch.int64:
        return False
    if keep_prefix_rank.ndim != 2 or keep_prefix_rank.dtype != torch.int32:
        return False
    if page_offsets.ndim != 1 or page_offsets.dtype != torch.int32:
        return False
    if k_cache_flat.ndim != 2 or v_cache_flat.ndim != 2:
        return False
    if k_cache_flat.shape[1] != 128 or v_cache_flat.shape[1] != 128:
        return False
    return True


def can_use_cuda_cache_fill_from_pos_with_local_rank(
    k: torch.Tensor,
    v: torch.Tensor,
    pos: torch.Tensor,
    local_rank: torch.Tensor,
    page_offsets: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
    page_block_size: int,
) -> bool:
    if not cuda_cache_fill_available():
        return False
    if block_size <= 0 or page_block_size <= 0 or (page_block_size % block_size) != 0:
        return False
    if k.dtype not in (torch.float16, torch.bfloat16):
        return False
    if v.dtype != k.dtype or k_cache_flat.dtype != k.dtype or v_cache_flat.dtype != k.dtype:
        return False
    if not (k.is_cuda and v.is_cuda and pos.is_cuda and local_rank.is_cuda and page_offsets.is_cuda):
        return False
    if not (k_cache_flat.is_cuda and v_cache_flat.is_cuda):
        return False
    if not (k.is_contiguous() and v.is_contiguous()):
        return False
    if not (pos.is_contiguous() and local_rank.is_contiguous() and page_offsets.is_contiguous()):
        return False
    if not (k_cache_flat.is_contiguous() and v_cache_flat.is_contiguous()):
        return False
    if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        return False
    if k.shape[-1] != 128:
        return False
    if pos.ndim != 2 or pos.shape[1] != 2 or pos.dtype != torch.int64:
        return False
    if local_rank.ndim != 1 or local_rank.dtype != torch.int32:
        return False
    if page_offsets.ndim != 1 or page_offsets.dtype != torch.int32:
        return False
    if pos.shape[0] != local_rank.shape[0]:
        return False
    if k_cache_flat.ndim != 2 or v_cache_flat.ndim != 2:
        return False
    if k_cache_flat.shape[1] != 128 or v_cache_flat.shape[1] != 128:
        return False
    return True


def can_use_cuda_cache_fill_from_row_blk_dst(
    k: torch.Tensor,
    v: torch.Tensor,
    row_idx: torch.Tensor,
    blk_idx: torch.Tensor,
    dst_token_base: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
) -> bool:
    if not cuda_cache_fill_available():
        return False
    if block_size <= 0:
        return False
    if k.dtype not in (torch.float16, torch.bfloat16):
        return False
    if v.dtype != k.dtype or k_cache_flat.dtype != k.dtype or v_cache_flat.dtype != k.dtype:
        return False
    if not (k.is_cuda and v.is_cuda and row_idx.is_cuda and blk_idx.is_cuda and dst_token_base.is_cuda):
        return False
    if not (k_cache_flat.is_cuda and v_cache_flat.is_cuda):
        return False
    if not (k.is_contiguous() and v.is_contiguous()):
        return False
    if not (row_idx.is_contiguous() and blk_idx.is_contiguous() and dst_token_base.is_contiguous()):
        return False
    if not (k_cache_flat.is_contiguous() and v_cache_flat.is_contiguous()):
        return False
    if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        return False
    if k.shape[-1] != 128:
        return False
    if row_idx.ndim != 1 or row_idx.dtype != torch.int64:
        return False
    if blk_idx.ndim != 1 or blk_idx.dtype != torch.int64:
        return False
    if dst_token_base.ndim != 1 or dst_token_base.dtype != torch.int64:
        return False
    if row_idx.numel() != blk_idx.numel() or row_idx.numel() != dst_token_base.numel():
        return False
    if k_cache_flat.ndim != 2 or v_cache_flat.ndim != 2:
        return False
    if k_cache_flat.shape[1] != 128 or v_cache_flat.shape[1] != 128:
        return False
    return True


def can_use_cuda_cache_fill_from_selected_indices_row_tiled(
    k: torch.Tensor,
    v: torch.Tensor,
    selected_block_indices: torch.Tensor,
    selected_block_counts: torch.Tensor,
    page_offsets: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
    page_block_size: int,
) -> bool:
    if not cuda_cache_fill_available():
        return False
    if block_size <= 0 or page_block_size <= 0 or (page_block_size % block_size) != 0:
        return False
    if k.dtype not in (torch.float16, torch.bfloat16):
        return False
    if v.dtype != k.dtype or k_cache_flat.dtype != k.dtype or v_cache_flat.dtype != k.dtype:
        return False
    if not (
        k.is_cuda
        and v.is_cuda
        and selected_block_indices.is_cuda
        and selected_block_counts.is_cuda
        and page_offsets.is_cuda
    ):
        return False
    if not (k_cache_flat.is_cuda and v_cache_flat.is_cuda):
        return False
    if not (k.is_contiguous() and v.is_contiguous()):
        return False
    if not (
        selected_block_indices.is_contiguous()
        and selected_block_counts.is_contiguous()
        and page_offsets.is_contiguous()
    ):
        return False
    if not (k_cache_flat.is_contiguous() and v_cache_flat.is_contiguous()):
        return False
    if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        return False
    if k.shape[-1] != 128:
        return False
    if selected_block_indices.ndim != 2 or selected_block_indices.dtype != torch.int32:
        return False
    if selected_block_counts.ndim != 1 or selected_block_counts.dtype != torch.int32:
        return False
    if page_offsets.ndim != 1 or page_offsets.dtype != torch.int32:
        return False
    if selected_block_indices.shape[0] != selected_block_counts.shape[0]:
        return False
    if selected_block_indices.shape[0] != page_offsets.shape[0]:
        return False
    if k_cache_flat.ndim != 2 or v_cache_flat.ndim != 2:
        return False
    if k_cache_flat.shape[1] != 128 or v_cache_flat.shape[1] != 128:
        return False
    return True


def can_use_cuda_pack_q_for_indexed_prefill(
    q: torch.Tensor,
    q_group: torch.Tensor,
    num_kv_heads: int,
    num_key_value_groups: int,
) -> bool:
    if not cuda_cache_fill_available():
        return False
    if num_kv_heads <= 0 or num_key_value_groups <= 0:
        return False
    if not (q.is_cuda and q_group.is_cuda):
        return False
    if not (q.is_contiguous() and q_group.is_contiguous()):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16) or q_group.dtype != q.dtype:
        return False
    if q.ndim != 4 or q_group.ndim != 5:
        return False
    if q.shape[-1] != 128 or q_group.shape[-1] != 128:
        return False
    if q.shape[2] != num_kv_heads * num_key_value_groups:
        return False
    if q_group.shape[0] != q.shape[0]:
        return False
    if q_group.shape[1] != num_kv_heads:
        return False
    if q_group.shape[2] != q.shape[1]:
        return False
    if q_group.shape[3] != num_key_value_groups:
        return False
    return True


def can_use_cuda_keep_block_builder_fast(
    *,
    keep_block: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    page_block_size: int,
    num_key_value_groups: int,
    past_k_blocks: int,
    curr_k_blocks: int,
) -> bool:
    if not cuda_cache_fill_available():
        return False
    if block_size <= 0 or page_block_size <= 0 or (page_block_size % block_size) != 0:
        return False
    if num_key_value_groups <= 0 or past_k_blocks < 0 or curr_k_blocks < 0:
        return False
    if not (keep_block.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if not (keep_block.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False
    if keep_block.dtype != torch.bool:
        return False
    if k.dtype not in (torch.float16, torch.bfloat16) or v.dtype != k.dtype:
        return False
    if keep_block.ndim != 3 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        return False
    if k.shape[-1] != 128 or keep_block.shape[0] != k.shape[0]:
        return False
    if keep_block.shape[1] != k.shape[2] * num_key_value_groups:
        return False
    if keep_block.shape[2] < past_k_blocks + curr_k_blocks:
        return False
    return True


def cache_fill_blocks_to_paged_kv_from_pos_with_rank_cuda(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    pos: torch.Tensor,
    keep_prefix_rank: torch.Tensor,
    page_offsets: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
    page_block_size: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.cache_fill_from_pos_rank(
        k,
        v,
        pos,
        keep_prefix_rank,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        int(block_size),
        int(page_block_size),
    )


def cache_fill_blocks_to_paged_kv_from_pos_with_local_rank_cuda(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    pos: torch.Tensor,
    local_rank: torch.Tensor,
    page_offsets: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
    page_block_size: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.cache_fill_from_pos_local_rank(
        k,
        v,
        pos,
        local_rank,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        int(block_size),
        int(page_block_size),
    )


def build_block_table_cuda(
    *,
    pages_per_row: torch.Tensor,
    block_table: torch.Tensor,
    page_offsets: torch.Tensor,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.build_block_table(pages_per_row, block_table, page_offsets)


def pack_q_for_indexed_prefill_cuda(
    *,
    q: torch.Tensor,
    q_group: torch.Tensor,
    num_kv_heads: int,
    num_key_value_groups: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.pack_q_for_indexed_prefill(
        q,
        q_group,
        int(num_kv_heads),
        int(num_key_value_groups),
    )


def compact_keep_blocks_cuda(
    *,
    keep_flat: torch.Tensor,
    sel_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    return ext.compact_keep_blocks(keep_flat, sel_blocks)


def compact_keep_blocks_cuda_out(
    *,
    keep_flat: torch.Tensor,
    sel_blocks: torch.Tensor,
    pos: torch.Tensor,
    local_rank: torch.Tensor,
) -> int:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    return int(ext.compact_keep_blocks_out(keep_flat, sel_blocks, pos, local_rank))


def compact_keep_blocks_and_build_table_cuda(
    *,
    keep_flat: torch.Tensor,
    sel_blocks: torch.Tensor,
    pages_per_row: torch.Tensor,
    block_table: torch.Tensor,
    page_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    return ext.compact_keep_blocks_and_build_table(
        keep_flat,
        sel_blocks,
        pages_per_row,
        block_table,
        page_offsets,
    )


def compact_keep_blocks_and_build_table_cuda_out(
    *,
    keep_flat: torch.Tensor,
    sel_blocks: torch.Tensor,
    pages_per_row: torch.Tensor,
    pos: torch.Tensor,
    local_rank: torch.Tensor,
    block_table: torch.Tensor,
    page_offsets: torch.Tensor,
) -> int:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    return int(
        ext.compact_keep_blocks_and_build_table_out(
            keep_flat,
            sel_blocks,
            pages_per_row,
            pos,
            local_rank,
            block_table,
            page_offsets,
        )
    )


def build_keep_past_fast_cuda(
    *,
    attn_gate_score: torch.Tensor,
    threshold: float,
    past_k_blocks: int,
) -> torch.Tensor:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    return ext.build_keep_past_fast(attn_gate_score, float(threshold), int(past_k_blocks))


def build_keep_curr_fast_cuda(
    *,
    attn_gate_score: torch.Tensor,
    threshold: float,
    past_k_blocks: int,
    curr_k_blocks: int,
) -> torch.Tensor:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    return ext.build_keep_curr_fast(
        attn_gate_score,
        float(threshold),
        int(past_k_blocks),
        int(curr_k_blocks),
    )


def build_past_indices_and_metadata_from_keep_block_cuda(
    *,
    keep_block: torch.Tensor,
    num_key_value_groups: int,
    past_k_blocks: int,
    curr_k_blocks: int,
    block_size: int,
    page_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    return ext.build_past_indices_and_metadata_from_keep_block(
        keep_block,
        int(num_key_value_groups),
        int(past_k_blocks),
        int(curr_k_blocks),
        int(block_size),
        int(page_block_size),
    )


def build_past_indices_and_metadata_from_keep_block_cuda_out(
    *,
    keep_block: torch.Tensor,
    num_key_value_groups: int,
    past_k_blocks: int,
    curr_k_blocks: int,
    block_size: int,
    page_block_size: int,
    past_block_indices: torch.Tensor,
    past_block_counts: torch.Tensor,
    pages_per_row: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.build_past_indices_and_metadata_from_keep_block_out(
        keep_block,
        int(num_key_value_groups),
        int(past_k_blocks),
        int(curr_k_blocks),
        int(block_size),
        int(page_block_size),
        past_block_indices,
        past_block_counts,
        pages_per_row,
        cache_seqlens,
    )


def build_selected_indices_and_metadata_from_keep_block_cuda_out(
    *,
    keep_block: torch.Tensor,
    num_key_value_groups: int,
    past_k_blocks: int,
    curr_k_blocks: int,
    block_size: int,
    page_block_size: int,
    selected_block_indices: torch.Tensor,
    selected_block_counts: torch.Tensor,
    pages_per_row: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.build_selected_indices_and_metadata_from_keep_block_out(
        keep_block,
        int(num_key_value_groups),
        int(past_k_blocks),
        int(curr_k_blocks),
        int(block_size),
        int(page_block_size),
        selected_block_indices,
        selected_block_counts,
        pages_per_row,
        cache_seqlens,
    )


def build_selected_indices_from_kv_keep_block_cuda_out(
    *,
    keep_block_kv: torch.Tensor,
    selected_block_indices: torch.Tensor,
    selected_block_counts: torch.Tensor,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.build_selected_indices_from_kv_keep_block_out(
        keep_block_kv,
        selected_block_indices,
        selected_block_counts,
    )


def build_flashinfer_kv_indices_cuda_out(
    *,
    selected_block_indices: torch.Tensor,
    selected_block_counts: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_blocks: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.build_flashinfer_kv_indices(
        selected_block_indices,
        selected_block_counts,
        kv_indptr,
        kv_indices,
        int(kv_blocks),
    )


def build_flashinfer_kv_indices_per_query_cuda_out(
    *,
    selected_block_indices: torch.Tensor,
    selected_block_counts: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_key_value_groups: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.build_flashinfer_kv_indices_per_query(
        selected_block_indices,
        selected_block_counts,
        kv_indptr,
        kv_indices,
        int(kv_blocks),
        int(num_q_heads),
        int(num_kv_heads),
        int(num_key_value_groups),
    )


def cache_fill_from_past_indices_cuda(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    past_block_indices: torch.Tensor,
    past_block_counts: torch.Tensor,
    page_offsets: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    active_past_k_blocks: int,
    block_size: int,
    page_block_size: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.cache_fill_from_past_indices(
        k,
        v,
        past_block_indices,
        past_block_counts,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        int(active_past_k_blocks),
        int(block_size),
        int(page_block_size),
    )


def cache_fill_from_past_indices_compact_cuda(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    past_block_indices: torch.Tensor,
    past_block_counts: torch.Tensor,
    selected_offsets: torch.Tensor,
    page_offsets: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    total_selected: int,
    block_size: int,
    page_block_size: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.cache_fill_from_past_indices_compact(
        k,
        v,
        past_block_indices,
        past_block_counts,
        selected_offsets,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        int(total_selected),
        int(block_size),
        int(page_block_size),
    )


def cache_fill_current_tail_cuda(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    past_block_counts: torch.Tensor,
    page_offsets: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    past_k_blocks: int,
    curr_k_blocks: int,
    block_size: int,
    page_block_size: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.cache_fill_current_tail(
        k,
        v,
        past_block_counts,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        int(past_k_blocks),
        int(curr_k_blocks),
        int(block_size),
        int(page_block_size),
    )


def cache_fill_from_selected_indices_row_tiled_cuda(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_block_indices: torch.Tensor,
    selected_block_counts: torch.Tensor,
    page_offsets: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
    page_block_size: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.cache_fill_from_selected_indices_row_tiled(
        k,
        v,
        selected_block_indices,
        selected_block_counts,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        int(block_size),
        int(page_block_size),
    )


def cache_fill_blocks_to_paged_kv_from_kv_strided_cuda(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    row_idx: torch.Tensor,
    blk_idx: torch.Tensor,
    dst_token_base: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
) -> None:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("compact_attn._C_compactattn is not available")
    ext.cache_fill_from_row_blk_dst(
        k,
        v,
        row_idx,
        blk_idx,
        dst_token_base,
        k_cache_flat,
        v_cache_flat,
        int(block_size),
    )
