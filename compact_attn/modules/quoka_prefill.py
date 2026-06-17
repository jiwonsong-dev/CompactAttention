from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from compact_attn.modules.dense_prefill import dense_prefill_full_kv


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


def _gather_headwise_sequence(
    tensor: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    return torch.gather(
        tensor,
        dim=2,
        index=indices.unsqueeze(-1).expand(*indices.shape, tensor.shape[-1]),
    )


def select_quoka_past_key_value(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    query_ratio: float,
    kv_budget_ratio: float,
    num_key_value_groups: int,
    past_attention_mask: Optional[torch.Tensor] = None,
    score_chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    bsz, q_len, num_query_heads, head_dim = query_states.shape
    kv_len = key_states.shape[1]
    num_key_value_heads = key_states.shape[2]
    past_len = kv_len - q_len
    if past_len <= 0:
        raise ValueError("QUOKA selection requires past context")
    if num_query_heads != num_key_value_heads * int(num_key_value_groups):
        raise ValueError("Unexpected GQA layout for QUOKA selection")
    if score_chunk_size <= 0:
        raise ValueError("score_chunk_size must be positive")

    if past_attention_mask is not None:
        past_attention_mask = past_attention_mask.to(device=query_states.device, dtype=torch.bool)
        if past_attention_mask.shape != (bsz, past_len):
            raise ValueError("past_attention_mask has unexpected shape")
        valid_past_lens = past_attention_mask.sum(dim=-1)
        if int(valid_past_lens.min().item()) <= 0:
            raise ValueError("No valid past tokens available for QUOKA selection")
        # Use the minimum valid length as budget so all rows can fill their topk
        # (padding positions are masked to -inf, so topk picks from valid tokens only)
        effective_past_len = int(valid_past_lens.min().item())
    else:
        effective_past_len = past_len

    query_count = max(1, min(q_len, int(query_ratio * q_len)))
    kv_budget = max(1, min(effective_past_len, int(kv_budget_ratio * effective_past_len)))

    query_per_head = query_states.permute(0, 2, 1, 3).contiguous()
    mean_query = query_per_head.mean(dim=2, keepdim=True)
    query_scores = F.cosine_similarity(
        F.normalize(query_per_head, dim=-1, eps=1e-6),
        F.normalize(mean_query, dim=-1, eps=1e-6),
        dim=-1,
    )
    selected_query_idx = torch.topk(
        query_scores,
        k=query_count,
        dim=-1,
        largest=False,
        sorted=False,
    ).indices
    selected_queries = _gather_headwise_sequence(query_per_head, selected_query_idx)
    # Normalize BEFORE averaging (matches QUOKA paper Algorithm 1 line 6→8).
    selected_queries = F.normalize(selected_queries, dim=-1, eps=1e-6)
    selected_queries = selected_queries.reshape(
        bsz,
        num_key_value_heads,
        num_key_value_groups,
        query_count,
        head_dim,
    ).mean(dim=2)

    past_keys = key_states[:, :past_len].permute(0, 2, 1, 3).contiguous()
    past_values = value_states[:, :past_len].permute(0, 2, 1, 3).contiguous()
    token_scores = torch.empty(
        (bsz, num_key_value_heads, past_len),
        device=query_states.device,
        dtype=query_states.dtype,
    )

    for start in range(0, past_len, score_chunk_size):
        end = min(start + score_chunk_size, past_len)
        key_chunk = F.normalize(past_keys[:, :, start:end, :], dim=-1, eps=1e-6)
        score_chunk = torch.einsum("bhqd,bhkd->bhqk", selected_queries, key_chunk)
        token_scores[:, :, start:end] = score_chunk.amax(dim=2)

    if past_attention_mask is not None:
        token_scores = token_scores.masked_fill(~past_attention_mask[:, None, :], torch.finfo(token_scores.dtype).min)

    selected_token_idx = torch.topk(
        token_scores,
        k=kv_budget,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    selected_token_idx = torch.sort(selected_token_idx, dim=-1).values

    selected_past_keys = _gather_headwise_sequence(past_keys, selected_token_idx).permute(0, 2, 1, 3).contiguous()
    selected_past_values = _gather_headwise_sequence(past_values, selected_token_idx).permute(0, 2, 1, 3).contiguous()

    stats = {
        "query_count": int(query_count),
        "kv_budget": int(kv_budget),
        "past_len": int(past_len),
        "assembled_kv_len": int(kv_budget + q_len),
        "selected_idx_sorted": int(bool(torch.all(selected_token_idx[..., 1:] >= selected_token_idx[..., :-1]))),
    }
    return selected_past_keys, selected_past_values, stats


def quoka_dense_prefill_full_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    softmax_scale: float,
    num_key_value_groups: int,
    query_ratio: float,
    kv_budget_ratio: float,
    score_chunk_size: int = 4096,
    measure_timing: bool = False,
    attn_module=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    q_len = int(query_states.shape[1])
    kv_len = int(key_states.shape[1])
    past_len = kv_len - q_len
    if q_len <= 1 or past_len <= 0:
        out, stats = dense_prefill_full_kv(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            softmax_scale=softmax_scale,
            num_key_value_groups=num_key_value_groups,
            fallback_used=0.0,
            measure_timing=measure_timing,
            attn_module=attn_module,
        )
        stats = dict(stats)
        stats.update(
            {
                "quoka_query_count": 0.0,
                "quoka_kv_budget": 0.0,
                "quoka_selection_ms": 0.0,
                "quoka_assembled_kv_len": float(kv_len),
            }
        )
        return out, stats

    past_attention_mask = None
    if attention_mask is not None:
        past_attention_mask = attention_mask[:, :past_len]

    selection_out, selection_ms = _cuda_elapsed_ms(
        lambda: select_quoka_past_key_value(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            query_ratio=query_ratio,
            kv_budget_ratio=kv_budget_ratio,
            num_key_value_groups=num_key_value_groups,
            past_attention_mask=past_attention_mask,
            score_chunk_size=score_chunk_size,
        ),
        enabled=measure_timing and query_states.is_cuda,
    )
    selected_past_keys, selected_past_values, select_stats = selection_out

    current_keys = key_states[:, past_len:, :, :].contiguous()
    current_values = value_states[:, past_len:, :, :].contiguous()
    assembled_keys = torch.cat((selected_past_keys, current_keys), dim=1)
    assembled_values = torch.cat((selected_past_values, current_values), dim=1)

    out, dense_stats = dense_prefill_full_kv(
        query_states=query_states,
        key_states=assembled_keys,
        value_states=assembled_values,
        attention_mask=None,
        softmax_scale=softmax_scale,
        num_key_value_groups=num_key_value_groups,
        fallback_used=0.0,
        measure_timing=measure_timing,
        attn_module=attn_module,
    )
    dense_stats = dict(dense_stats)
    dense_stats.update(
        {
            "quoka_query_count": float(select_stats["query_count"]),
            "quoka_kv_budget": float(select_stats["kv_budget"]),
            "quoka_selection_ms": float(selection_ms),
            "quoka_assembled_kv_len": float(select_stats["assembled_kv_len"]),
            "quoka_selected_idx_sorted": float(select_stats["selected_idx_sorted"]),
        }
    )
    return out, dense_stats
