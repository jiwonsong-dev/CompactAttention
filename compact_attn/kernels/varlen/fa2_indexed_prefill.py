from typing import Dict, Tuple

import torch
from flash_attn import flash_attn_with_kvcache
from compact_attn.kernels.varlen.fa2_indexed_prefill_triton import (
    can_use_fa2_indexed_prefill,
    fa2_indexed_triton_available,
    run_fa2_indexed_prefill_triton,
)


def fa2_indexed_available() -> bool:
    return fa2_indexed_triton_available()


def run_fa2_indexed_prefill(
    q: torch.Tensor,  # [B, Q, Hq, D]
    k: torch.Tensor,  # [B, K, Hkv, D]
    v: torch.Tensor,  # [B, K, Hkv, D]
    past_block_indices: torch.Tensor,  # [B, Hkv, Smax], -1 padded
    past_block_counts: torch.Tensor,  # [B, Hkv]
    past_len: int,
    block_size: int,
    num_key_value_groups: int,
    softmax_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    return run_fa2_indexed_prefill_triton(
        q=q,
        k=k,
        v=v,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        block_size=block_size,
        num_key_value_groups=num_key_value_groups,
        softmax_scale=softmax_scale,
    )


def run_fa2_indexed_prefill_from_paged_kv(
    q: torch.Tensor,  # [B, Q, Hq, D]
    k_cache: torch.Tensor,  # [num_pages, page_block, 1, D]
    v_cache: torch.Tensor,  # [num_pages, page_block, 1, D]
    block_table: torch.Tensor,  # [B*Hkv, max_pages]
    cache_seqlens: torch.Tensor,  # [B*Hkv]
    num_key_value_groups: int,
    softmax_scale: float,
) -> torch.Tensor:
    # NOTE: v0 delegates to FA2 paged-kv call while keeping a dedicated API surface.
    bsz, q_len, num_q_heads, head_dim = q.shape
    rows = block_table.shape[0]
    num_kv_heads = rows // bsz

    q_group = (
        q.view(bsz, q_len, num_kv_heads, num_key_value_groups, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(rows, q_len, num_key_value_groups, head_dim)
        .contiguous()
    )
    out_group = flash_attn_with_kvcache(
        q=q_group,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens.to(torch.int32).contiguous(),
        block_table=block_table.to(torch.int32).contiguous(),
        softmax_scale=softmax_scale,
        causal=True,
    )
    out = (
        out_group.view(bsz, num_kv_heads, q_len, num_key_value_groups, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(bsz, q_len, num_q_heads, head_dim)
        .contiguous()
    )
    return out
