"""LLaMA compactattn variant with heads-first KV cache layout.

Stores KV as [bsz, Hkv, kv_len, D] (heads-first) so that a zero-copy view
  k_hf.view(bsz*Hkv*n_blocks, block_size, 1, D)
exposes FlashInfer pages without any materialization.

Phase 1: HeadsFirstDynamicCache + LlamaSeerAttentionChunkedDenseHF that
         stores heads-first but permutes back to seq-first for all existing
         paths.  Output is numerically identical to seer_compactattn.

Phase 2: When compactattn_indexed_impl is "fi_zero_copy" or "cudnn_one_shot",
         _do_chunked_compactattn_attention()
         passes k_hf/v_hf directly to chunked_prefill_column_dense_attention_forward,
         bypassing the cache-fill materialization step entirely.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from compact_attn.modules.attention_forward_chunked_dense import chunked_prefill_column_dense_attention_forward
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn_chunked_dense import (
    LlamaSeerAttentionChunkedDense,
    SeerAttnLlamaChunkedDenseForCausalLM,
    _cuda_elapsed_ms,
)


class HeadsFirstDynamicCache(DynamicCache):
    """DynamicCache that stores KV tensors in [bsz, Hkv, kv_len, D] format.

    Callers must pass heads-first tensors to update() and will receive
    heads-first tensors back.  DynamicCache.update() uses dim=-2 for
    concatenation, which is the seq dim in both layouts:
      seq-first   [bsz, seq, Hkv*D]  → dim=-2 is seq  ✓
      heads-first [bsz, Hkv, seq, D] → dim=-2 is seq  ✓
    No override of update() is needed.
    """

    def get_heads_first(self, layer_idx: int):
        """Return (k, v) in [bsz, Hkv, kv_len, D] format."""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class LlamaSeerAttentionChunkedDenseHF(LlamaSeerAttentionChunkedDense):
    """Attention module with heads-first KV cache storage and optional zero-copy attention.

    _do_kv_cache_update():
      - Permutes key/value to [bsz, Hkv, seq, D] before storing in HeadsFirstDynamicCache.
      - Stores k_hf/v_hf on self for use by _do_chunked_compactattn_attention.
      - Permutes the accumulated cache back to seq-first for all dense paths.

    _do_chunked_compactattn_attention():
      - If indexed_impl is "fi_zero_copy" or "cudnn_one_shot": passes k_hf/v_hf to
        chunked_prefill_column_dense_attention_forward, bypassing cache-fill.
      - Otherwise: delegates to base class (standard fa2_paged / fi_paged path).
    """

    def _do_kv_cache_update(self, key_states, value_states, past_key_value, sin, cos, cache_position, collect_stats):
        def _run():
            q_len_in = key_states.shape[1]
            k_hf_in = key_states.permute(0, 2, 1, 3).contiguous()   # [bsz, Hkv, seq, D]
            v_hf_in = value_states.permute(0, 2, 1, 3).contiguous()
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_hf, v_hf = past_key_value.update(k_hf_in, v_hf_in, self.layer_idx, cache_kwargs)
            kv_len_acc = k_hf.shape[2]
            # For zero-copy backends on chunked-prefill chunks (kv_len > q_len), the
            # returned key_states is only read for its .shape[1] — skip the
            # expensive full-KV permute+copy and return a non-contiguous view.
            # The first chunk (kv_len_acc == q_len_in) still needs contiguous
            # for the force_dense FA2 path.
            if kv_len_acc > q_len_in and self.compactattn_indexed_impl in {"fi_zero_copy", "cudnn_one_shot"}:
                k_sf = k_hf.permute(0, 2, 1, 3)   # zero-cost view
                v_sf = v_hf.permute(0, 2, 1, 3)
            else:
                k_sf = k_hf.permute(0, 2, 1, 3).contiguous()   # [bsz, kv_len, Hkv, D]
                v_sf = v_hf.permute(0, 2, 1, 3).contiguous()
            return k_sf, v_sf, k_hf, v_hf
        (k_sf, v_sf, k_hf, v_hf), ms = _cuda_elapsed_ms(_run, enabled=collect_stats)
        self._last_k_hf = k_hf  # [bsz, Hkv, kv_len, D]
        self._last_v_hf = v_hf
        return k_sf, v_sf, ms

    def _do_chunked_compactattn_attention(self, query_states, key_states, value_states, **kwargs):
        if (self.compactattn_indexed_impl in {"fi_zero_copy", "cudnn_one_shot"}
                and getattr(self, "_last_k_hf", None) is not None):
            return chunked_prefill_column_dense_attention_forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                k_hf=self._last_k_hf,
                v_hf=self._last_v_hf,
                **kwargs,
            )
        return chunked_prefill_column_dense_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            **kwargs,
        )


class SeerAttnLlamaChunkedDenseHFForCausalLM(SeerAttnLlamaChunkedDenseForCausalLM):
    """SeerAttention LLaMA compactattn model with heads-first KV cache storage.

    When compactattn_indexed_impl is "fi_zero_copy" or "cudnn_one_shot", the compactattn attention kernel
    reads directly from the heads-first cache without any cache-fill materialization.
    """

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("seerattn_compactattn_indexed_impl", "fi_zero_copy")
        model = SeerAttnLlamaChunkedDenseForCausalLM.from_pretrained(*args, **kwargs)
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if isinstance(attn, LlamaSeerAttentionChunkedDense):
                attn.__class__ = LlamaSeerAttentionChunkedDenseHF
        model.__class__ = cls
        return model

    def forward(self, *args, past_key_values=None, **kwargs):
        if past_key_values is None:
            past_key_values = HeadsFirstDynamicCache()
        return super().forward(*args, past_key_values=past_key_values, **kwargs)
