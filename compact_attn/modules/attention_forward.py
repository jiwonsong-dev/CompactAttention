import os
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func, flash_attn_varlen_func
from compact_attn.kernels.block_sparse_attn import block_sparse_triton_fn
from compact_attn.kernels.varlen.block_sparse_attn_varlen_2d import block_2d_sparse_attn_varlen_func
import os
import math

from compact_attn.modules.common import (
    repeat_kv_varlen,
    repeat_kv,
    pad_input,
    _upad_input,
)
from compact_attn.modules.dense_prefill import _normalize_dense_padding_mask

def sparse_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    softmax_scale: Optional[float] = None,
    attn_gate_score: Optional[torch.Tensor] = None,
    sparsity_method: Optional[str] = None,
    threshold: Optional[float] = None,
    nz_ratio: Optional[float] = None,
    last_block_dense: Optional[bool] = None,
    block_size: Optional[int] = None,
    num_key_value_groups: Optional[int] = None,
    profile_file: Optional[str] = None,
    block_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    force_all_sparse_blocks = os.environ.get("SEERATTN_DEBUG_FORCE_ALL_SPARSE_BLOCKS", "0") == "1"


    if query_length > 1:
        if sparsity_method == "nz_ratio":
            _, _, _, key_block_len = attn_gate_score.shape
            topk_nz_ratio = 1 - math.sqrt(1 - nz_ratio)
            topk = int(topk_nz_ratio * key_block_len)
            topk = 1 if topk == 0 else topk
            sparse_index = torch.topk(attn_gate_score, topk, dim=-1).indices
            gate_mask = torch.full_like(attn_gate_score, False, dtype=torch.bool)
            gate_mask.scatter_(-1, sparse_index, True)
        elif sparsity_method == "threshold":
            gate_mask = attn_gate_score > threshold

        if force_all_sparse_blocks:
            gate_mask = torch.ones_like(attn_gate_score, dtype=torch.bool)

        if last_block_dense:
            gate_mask[:, :, -2:, :] = True

        if block_attention_mask is not None:
            gate_mask = gate_mask & block_attention_mask.to(torch.bool)
        else:
            gate_mask.tril_()

        if profile_file is not None:
            total_size = gate_mask.numel()
            with open(profile_file, "a") as f:
                f.write(f"{query_length}: {gate_mask.sum().item() / total_size}\n")

        if attention_mask is None:
            kv_length = key_states.shape[1]
            if kv_length == query_length:
                query_states = query_states.transpose(1, 2).contiguous()
                key_states = key_states.transpose(1, 2).contiguous()
                value_states = value_states.transpose(1, 2).contiguous()
                key_states = repeat_kv(key_states, num_key_value_groups)
                value_states = repeat_kv(value_states, num_key_value_groups)
                attn_output = block_sparse_triton_fn( 
                    query_states,
                    key_states,
                    value_states,
                    block_sparse_mask=gate_mask,
                    sm_scale=softmax_scale,
                    BLOCK_M=block_size,
                    BLOCK_N=block_size,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
            else:
                batch_size = query_states.shape[0]
                cu_seqlens_q = torch.arange(
                    0,
                    (batch_size + 1) * query_length,
                    step=query_length,
                    dtype=torch.int32,
                    device=query_states.device,
                )
                cu_seqlens_k = torch.arange(
                    0,
                    (batch_size + 1) * kv_length,
                    step=kv_length,
                    dtype=torch.int32,
                    device=query_states.device,
                )
                query_states = query_states.reshape(batch_size * query_length, query_states.shape[2], query_states.shape[3])
                key_states = key_states.reshape(batch_size * kv_length, key_states.shape[2], key_states.shape[3])
                value_states = value_states.reshape(batch_size * kv_length, value_states.shape[2], value_states.shape[3])
                key_states = repeat_kv_varlen(key_states, num_key_value_groups)
                value_states = repeat_kv_varlen(value_states, num_key_value_groups)
                attn_output = block_2d_sparse_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_k,
                    cu_seqlens_q,
                    kv_length,
                    softmax_scale,
                    gate_mask,
                    block_size,
                )
                attn_output = attn_output.reshape(batch_size, query_length, -1, value_states.shape[-1])

        else:
            batch_size = query_states.shape[0]
            attention_mask = _normalize_dense_padding_mask(
                attention_mask,
                batch_size=batch_size,
                kv_len=key_states.shape[1],
                device=query_states.device,
            )
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            key_states = repeat_kv_varlen(key_states, num_key_value_groups)
            value_states = repeat_kv_varlen(value_states, num_key_value_groups)

            attn_output_unpad = block_2d_sparse_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_k,
                cu_seqlens_q,
                max_seqlen_in_batch_k,
                softmax_scale,
                gate_mask,
                block_size,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    else:
        if attention_mask is None:
            key_states = repeat_kv(key_states.transpose(1, 2).contiguous(), num_key_value_groups).transpose(1, 2).contiguous()
            value_states = repeat_kv(value_states.transpose(1, 2).contiguous(), num_key_value_groups).transpose(1, 2).contiguous()
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                softmax_scale=softmax_scale,
                causal=True,
            )
        else:
            attention_mask = _normalize_dense_padding_mask(
                attention_mask,
                batch_size=query_states.shape[0],
                kv_len=key_states.shape[1],
                device=query_states.device,
            )
            has_padding = bool((attention_mask == 0).any().item())
            if not has_padding:
                key_states = repeat_kv(key_states.transpose(1, 2).contiguous(), num_key_value_groups).transpose(1, 2).contiguous()
                value_states = repeat_kv(value_states.transpose(1, 2).contiguous(), num_key_value_groups).transpose(1, 2).contiguous()
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
            else:
                batch_size = query_states.shape[0]
                query_states_unpad, key_states_unpad, value_states_unpad, _, cu_seq_lens, max_seq_lens = _upad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )
                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
                key_states_unpad = repeat_kv_varlen(key_states_unpad, num_key_value_groups)
                value_states_unpad = repeat_kv_varlen(value_states_unpad, num_key_value_groups)
                attn_output_unpad = flash_attn_varlen_func(
                    query_states_unpad,
                    key_states_unpad,
                    value_states_unpad,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
                attn_output = attn_output_unpad.view(batch_size, query_length, attn_output_unpad.shape[1], attn_output_unpad.shape[2])

    return attn_output
