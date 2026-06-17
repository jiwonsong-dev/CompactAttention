import os
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F
from compact_attn.kernels.attn_pooling_kernel_2d import attn_with_pooling

from compact_attn.modules.common import (
    repeat_kv,
)


def _get_distill_impl(head_dim: int) -> str:
    impl = os.environ.get("SEERATTN_DISTILL_IMPL", "").strip().lower()
    if impl in {"kernel", "triton"}:
        return "kernel"
    if impl in {"torch", "chunked"}:
        return "torch"
    if head_dim >= 256:
        return "torch"
    return "kernel"


def _attention_distill_forward_chunked_torch(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    softmax_scale: Optional[float],
    block_size: int,
):
    bsz, num_heads, seq_len, head_dim = query_states.shape
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    q_chunk_tokens = int(
        os.environ.get(
            "SEERATTN_DISTILL_Q_CHUNK_TOKENS",
            str(max(block_size, block_size * 2)),
        )
    )
    q_chunk_tokens = max(block_size, (q_chunk_tokens // block_size) * block_size)

    num_k_blocks = (seq_len + block_size - 1) // block_size
    pad_k_tokens = num_k_blocks * block_size - seq_len
    if pad_k_tokens > 0:
        key_states_padded = F.pad(key_states, (0, 0, 0, pad_k_tokens))
    else:
        key_states_padded = key_states

    attn_output = torch.empty_like(query_states)
    mask_ground_truth = torch.zeros(
        (bsz, num_heads, (seq_len + block_size - 1) // block_size, num_k_blocks),
        device=query_states.device,
        dtype=torch.float32,
    )

    key_positions = torch.arange(seq_len, device=query_states.device)

    for q_start in range(0, seq_len, q_chunk_tokens):
        q_end = min(q_start + q_chunk_tokens, seq_len)
        q_chunk = query_states[:, :, q_start:q_end, :]

        scores = torch.matmul(
            q_chunk.to(torch.float32),
            key_states.transpose(-1, -2).to(torch.float32),
        )
        scores.mul_(float(softmax_scale))

        q_positions = torch.arange(q_start, q_end, device=query_states.device)
        causal_mask = key_positions.view(1, 1, 1, seq_len) <= q_positions.view(1, 1, q_end - q_start, 1)
        scores.masked_fill_(~causal_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        output_chunk = torch.matmul(probs.to(value_states.dtype), value_states)
        attn_output[:, :, q_start:q_end, :] = output_chunk

        if pad_k_tokens > 0:
            probs_for_pool = F.pad(probs, (0, pad_k_tokens))
        else:
            probs_for_pool = probs
        probs_for_pool = probs_for_pool.view(
            bsz,
            num_heads,
            q_end - q_start,
            num_k_blocks,
            block_size,
        )
        block_max_per_row = probs_for_pool.amax(dim=-1)

        q_block_start = q_start // block_size
        q_chunk_len = q_end - q_start
        q_blocks_in_chunk = (q_chunk_len + block_size - 1) // block_size
        q_pad_tokens = q_blocks_in_chunk * block_size - q_chunk_len
        if q_pad_tokens > 0:
            block_max_per_row = F.pad(block_max_per_row, (0, 0, 0, q_pad_tokens))
        block_max_per_row = block_max_per_row.view(
            bsz,
            num_heads,
            q_blocks_in_chunk,
            block_size,
            num_k_blocks,
        )
        mask_ground_truth[:, :, q_block_start : q_block_start + q_blocks_in_chunk, :] = block_max_per_row.amax(dim=3)

    return attn_output, mask_ground_truth

def reduce_mask_ground_truth_by_kv_group(
    mask_ground_truth: torch.Tensor,
    num_key_value_groups: int,
    pooling: str = "max",
) -> torch.Tensor:
    if num_key_value_groups <= 1:
        return mask_ground_truth
    if mask_ground_truth.shape[1] % num_key_value_groups != 0:
        raise ValueError(
            f"Ground-truth head dimension {mask_ground_truth.shape[1]} is not divisible by "
            f"num_key_value_groups={num_key_value_groups}."
        )
    bsz, num_q_heads, q_blocks, k_blocks = mask_ground_truth.shape
    num_kv_heads = num_q_heads // num_key_value_groups
    grouped = mask_ground_truth.view(
        bsz,
        num_kv_heads,
        num_key_value_groups,
        q_blocks,
        k_blocks,
    )
    if pooling == "max":
        return grouped.max(dim=2).values
    if pooling == "mean":
        return grouped.mean(dim=2)
    raise ValueError(f"Unsupported kv-group pooling: {pooling}")


def attention_distill_forward(
    query_states: torch.Tensor, ## [batch, seq_len, num_heads, head_dim]
    key_states: torch.Tensor, ## [batch, seq_len, num_heads, head_dim]
    value_states: torch.Tensor, ## [batch, seq_len, num_heads, head_dim]
    softmax_scale: Optional[float] = None,
    block_size: Optional[int] = None,
    num_key_value_groups: Optional[int] = 1,
    kv_group_aware_query: bool = False,
    kv_group_pooling: str = "max",
    **kwargs,
):

    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()


    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    

    distill_impl = _get_distill_impl(query_states.shape[-1])
    if distill_impl == "torch":
        attn_output, mask_ground_truth = _attention_distill_forward_chunked_torch(
            query_states,
            key_states,
            value_states,
            softmax_scale=softmax_scale,
            block_size=block_size,
        )
    else:
        attn_output, mask_ground_truth = attn_with_pooling(
            query_states,
            key_states,
            value_states,
            True,
            softmax_scale,
            block_size,
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    if kv_group_aware_query:
        mask_ground_truth = reduce_mask_ground_truth_by_kv_group(
            mask_ground_truth,
            num_key_value_groups=num_key_value_groups,
            pooling=kv_group_pooling,
        )

    return attn_output, mask_ground_truth

