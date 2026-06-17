"""Load base models with QUOKA attention monkey-patched in.

Loads the stock HuggingFace model
(LlamaForCausalLM / Qwen2ForCausalLM / Qwen3ForCausalLM)
and replace each layer's attention forward with QUOKA's query/KV-budget
selection path.  Because no extra parameters (attention gates etc.) are
added, ``device_map="auto"`` splits layers at clean boundaries and avoids
the intra-layer device-mismatch bug that the SeerAttn-inherited classes
trigger on multi-GPU setups.
"""
from __future__ import annotations

from types import MethodType
from typing import Callable, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    Gemma3ForCausalLM,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb as qwen3_moe_apply_rotary_pos_emb
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb as gemma3_apply_rotary_pos_emb

from compact_attn.modules.dense_prefill import dense_prefill_full_kv
from compact_attn.modules.quoka_prefill import quoka_dense_prefill_full_kv


# ---------------------------------------------------------------------------
# QUOKA attention forward (model-family agnostic core)
# ---------------------------------------------------------------------------

def _quoka_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    rope_fn: Callable = None,
    **kwargs,
):
    config = self.config
    head_dim = self.head_dim
    num_kv_groups = self.num_key_value_groups
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    # --- QKV projection (identical to the base model) ---
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    # Qwen3 applies RMSNorm to Q and K after projection.
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    # Transpose to (B, H, S, D) for RoPE + cache.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # --- RoPE ---
    cos, sin = position_embeddings
    query_states, key_states = rope_fn(query_states, key_states, cos, sin)

    # --- Cache update ---
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    q_len = int(query_states.shape[2])
    kv_len = int(key_states.shape[2])
    collect_stats = bool(
        getattr(config, "seerattn_profile_quoka_components", False)
        or getattr(self, "quoka_detailed_timing", False)
    )
    self._quoka_last_stats = None

    force_dense = (
        (not self.training)
        and (q_len > 1)
        and bool(getattr(config, "seerattn_chunked_prefill_force_dense", False))
    )
    use_quoka = (
        (not self.training)
        and (q_len > 1)
        and (kv_len > q_len)
        and not force_dense
        and getattr(self, "_quoka_allow_sparse_prefill", True)
    )

    # Transpose to (B, S, H, D) for the quoka / dense prefill helpers.
    query_bshd = query_states.transpose(1, 2)
    key_bshd = key_states.transpose(1, 2)
    value_bshd = value_states.transpose(1, 2)

    if use_quoka:
        attn_output, quoka_stats = quoka_dense_prefill_full_kv(
            query_states=query_bshd,
            key_states=key_bshd,
            value_states=value_bshd,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            num_key_value_groups=num_kv_groups,
            query_ratio=float(getattr(config, "seerattn_quoka_query_ratio", 0.25)),
            kv_budget_ratio=float(getattr(config, "seerattn_quoka_kv_budget_ratio", 0.25)),
            score_chunk_size=int(getattr(config, "seerattn_quoka_score_chunk_size", 4096)),
            measure_timing=collect_stats,
            attn_module=self,
        )
    else:
        attn_output, quoka_stats = dense_prefill_full_kv(
            query_states=query_bshd,
            key_states=key_bshd,
            value_states=value_bshd,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            num_key_value_groups=num_kv_groups,
            fallback_used=0.0,
            measure_timing=collect_stats,
            attn_module=self,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    if collect_stats and isinstance(quoka_stats, dict):
        self._quoka_last_stats = dict(quoka_stats)
    return attn_output, None


# ---------------------------------------------------------------------------
# Layer patching
# ---------------------------------------------------------------------------

def _patch_quoka_layers(model, *, rope_fn: Callable, layer_predicate: Optional[Callable] = None):
    """Replace every decoder-layer's ``self_attn.forward`` with QUOKA."""
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None and language_model is not None:
        layers = getattr(language_model, "layers", None)
    if layers is None:
        raise AttributeError("Could not find decoder layers for QUOKA patching")
    for layer in layers:
        allow_sparse_prefill = True
        if layer_predicate is not None:
            allow_sparse_prefill = bool(layer_predicate(layer))
        if not hasattr(layer.self_attn, "_quoka_original_forward"):
            layer.self_attn._quoka_original_forward = layer.self_attn.forward
        layer.self_attn._quoka_allow_sparse_prefill = allow_sparse_prefill
        layer.self_attn._quoka_last_stats = None
        layer.self_attn.quoka_detailed_timing = False

        def _make_forward(rfn):
            def fwd(self, *args, _rfn=rfn, **kwargs):
                return _quoka_attention_forward(self, *args, rope_fn=_rfn, **kwargs)
            return fwd

        layer.self_attn.forward = MethodType(_make_forward(rope_fn), layer.self_attn)
    return model


# ---------------------------------------------------------------------------
# Model-specific loaders
# ---------------------------------------------------------------------------

def _set_quoka_config(model, *, query_ratio, kv_budget_ratio, score_chunk_size):
    cfg = model.config
    setattr(cfg, "seerattn_quoka_query_ratio", float(query_ratio))
    setattr(cfg, "seerattn_quoka_kv_budget_ratio", float(kv_budget_ratio))
    setattr(cfg, "seerattn_quoka_score_chunk_size", int(score_chunk_size))
    setattr(cfg, "seerattn_chunked_prefill_force_dense", False)
    setattr(cfg, "seerattn_chunked_prefill_final_dense_tail_blocks", 0)


def load_quoka_llama_model(
    name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    query_ratio: float = 0.25,
    kv_budget_ratio: float = 0.25,
    score_chunk_size: int = 4096,
    device_map=None,
    tp_plan=None,
    **_ignored,
):
    kwargs = dict(
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    if tp_plan is not None:
        kwargs["tp_plan"] = tp_plan
    else:
        kwargs["device_map"] = device_map
    model = LlamaForCausalLM.from_pretrained(name_or_path, **kwargs).eval()
    _set_quoka_config(
        model,
        query_ratio=query_ratio,
        kv_budget_ratio=kv_budget_ratio,
        score_chunk_size=score_chunk_size,
    )
    return _patch_quoka_layers(model, rope_fn=llama_apply_rotary_pos_emb)


def load_quoka_qwen2_model(
    name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    query_ratio: float = 0.25,
    kv_budget_ratio: float = 0.25,
    score_chunk_size: int = 4096,
    qwen_long_context: bool = False,
    qwen_long_context_max_position_embeddings: int = 131072,
    qwen_yarn_factor: float = 4.0,
    qwen_original_max_position_embeddings: int = 32768,
    device_map=None,
    tp_plan=None,
    **_ignored,
):
    config = None
    if qwen_long_context:
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        config.max_position_embeddings = int(qwen_long_context_max_position_embeddings)
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": float(qwen_yarn_factor),
            "original_max_position_embeddings": int(qwen_original_max_position_embeddings),
        }
    kwargs = dict(
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    if tp_plan is not None:
        kwargs["tp_plan"] = tp_plan
    else:
        kwargs["device_map"] = device_map
    model = Qwen2ForCausalLM.from_pretrained(name_or_path, **kwargs).eval()
    _set_quoka_config(
        model,
        query_ratio=query_ratio,
        kv_budget_ratio=kv_budget_ratio,
        score_chunk_size=score_chunk_size,
    )
    return _patch_quoka_layers(model, rope_fn=qwen2_apply_rotary_pos_emb)


def load_quoka_qwen3_model(
    name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    query_ratio: float = 0.25,
    kv_budget_ratio: float = 0.25,
    score_chunk_size: int = 4096,
    qwen_long_context: bool = False,
    qwen_long_context_max_position_embeddings: int = 131072,
    qwen_yarn_factor: float = 4.0,
    qwen_original_max_position_embeddings: int = 32768,
    device_map=None,
    tp_plan=None,
    **_ignored,
):
    config = None
    if qwen_long_context:
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        config.max_position_embeddings = int(qwen_long_context_max_position_embeddings)
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": float(qwen_yarn_factor),
            "original_max_position_embeddings": int(qwen_original_max_position_embeddings),
        }
    kwargs = dict(
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    if tp_plan is not None:
        kwargs["tp_plan"] = tp_plan
    else:
        kwargs["device_map"] = device_map
    model = Qwen3ForCausalLM.from_pretrained(name_or_path, **kwargs).eval()
    _set_quoka_config(
        model,
        query_ratio=query_ratio,
        kv_budget_ratio=kv_budget_ratio,
        score_chunk_size=score_chunk_size,
    )
    return _patch_quoka_layers(model, rope_fn=qwen3_apply_rotary_pos_emb)


def load_quoka_qwen3_moe_model(
    name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    query_ratio: float = 0.25,
    kv_budget_ratio: float = 0.25,
    score_chunk_size: int = 4096,
    qwen_long_context: bool = False,
    qwen_long_context_max_position_embeddings: int = 131072,
    qwen_yarn_factor: float = 4.0,
    qwen_original_max_position_embeddings: int = 32768,
    device_map=None,
    tp_plan=None,
    **_ignored,
):
    config = None
    if qwen_long_context:
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        config.max_position_embeddings = int(qwen_long_context_max_position_embeddings)
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": float(qwen_yarn_factor),
            "original_max_position_embeddings": int(qwen_original_max_position_embeddings),
        }
    kwargs = dict(
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    if tp_plan is not None:
        kwargs["tp_plan"] = tp_plan
    else:
        kwargs["device_map"] = device_map
    model = Qwen3MoeForCausalLM.from_pretrained(name_or_path, **kwargs).eval()
    _set_quoka_config(
        model,
        query_ratio=query_ratio,
        kv_budget_ratio=kv_budget_ratio,
        score_chunk_size=score_chunk_size,
    )
    return _patch_quoka_layers(model, rope_fn=qwen3_moe_apply_rotary_pos_emb)


def load_quoka_gemma3_model(
    name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    query_ratio: float = 0.25,
    kv_budget_ratio: float = 0.25,
    score_chunk_size: int = 4096,
    device_map=None,
    tp_plan=None,
    **_ignored,
):
    input_config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    base_model = getattr(input_config, "base_model", name_or_path)
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    if getattr(config, "model_type", None) == "gemma3":
        config = config.text_config
    kwargs = dict(
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    if tp_plan is not None:
        kwargs["tp_plan"] = tp_plan
    else:
        kwargs["device_map"] = device_map
    model = Gemma3ForCausalLM.from_pretrained(base_model, **kwargs).eval()
    _set_quoka_config(
        model,
        query_ratio=query_ratio,
        kv_budget_ratio=kv_budget_ratio,
        score_chunk_size=score_chunk_size,
    )
    return _patch_quoka_layers(
        model,
        rope_fn=gemma3_apply_rotary_pos_emb,
        layer_predicate=lambda layer: getattr(layer, "attention_type", None) == "full_attention",
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def load_quoka_model(name_or_path: str, **kwargs):
    model_type = getattr(
        AutoConfig.from_pretrained(name_or_path, trust_remote_code=True), "model_type", None
    )
    if model_type == "llama":
        return load_quoka_llama_model(name_or_path, **kwargs)
    if model_type == "qwen2":
        return load_quoka_qwen2_model(name_or_path, **kwargs)
    if model_type == "qwen3":
        return load_quoka_qwen3_model(name_or_path, **kwargs)
    if model_type == "qwen3_moe":
        return load_quoka_qwen3_moe_model(name_or_path, **kwargs)
    if model_type in {"gemma3", "gemma3_text"}:
        return load_quoka_gemma3_model(name_or_path, **kwargs)
    raise ValueError(f"Unsupported model_type for QUOKA: {model_type!r}")
