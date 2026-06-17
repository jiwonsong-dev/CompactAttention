"""Load stock models with dense attention monkey-patched to shared helpers.

This keeps the plain dense baseline on top of the stock HuggingFace model
while routing prefill + dense-style decode through `dense_prefill_full_kv(...)`.
That gives us a single dense backend selector surface without introducing the
extra gate / sparse state carried by the Seer-derived classes.
"""
from __future__ import annotations

from types import MethodType
from typing import Callable, Optional, Tuple

import torch
from transformers import AutoConfig, LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb as qwen3_moe_apply_rotary_pos_emb

from compact_attn.modules.dense_prefill import dense_prefill_full_kv


def _dense_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    rope_fn: Callable = None,
    **kwargs,
):
    del kwargs
    head_dim = self.head_dim
    num_kv_groups = self.num_key_value_groups
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = rope_fn(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    profile_dense = bool(getattr(self.config, "seerattn_profile_dense_components", False))
    attn_output, dense_stats = dense_prefill_full_kv(
        query_states=query_states.transpose(1, 2),
        key_states=key_states.transpose(1, 2),
        value_states=value_states.transpose(1, 2),
        attention_mask=attention_mask,
        softmax_scale=self.scaling,
        num_key_value_groups=num_kv_groups,
        fallback_used=0.0,
        measure_timing=profile_dense,
        attn_module=self,
    )
    self._dense_last_stats = dense_stats if profile_dense else None
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _decoder_layers(model):
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None and language_model is not None:
        layers = getattr(language_model, "layers", None)
    if layers is None:
        raise AttributeError("Could not find decoder layers for dense patching")
    return layers


def _patch_dense_layers(model, *, rope_fn: Callable):
    for layer in _decoder_layers(model):
        if not hasattr(layer.self_attn, "_dense_original_forward"):
            layer.self_attn._dense_original_forward = layer.self_attn.forward
        if not hasattr(layer.self_attn, "_dense_last_stats"):
            layer.self_attn._dense_last_stats = None

        def _make_forward(rfn):
            def fwd(self, *args, _rfn=rfn, **kwargs):
                return _dense_attention_forward(self, *args, rope_fn=_rfn, **kwargs)

            return fwd

        layer.self_attn.forward = MethodType(_make_forward(rope_fn), layer.self_attn)
    return model


def _load_hf_model(
    cls,
    name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    device_map=None,
    tp_plan=None,
    config=None,
):
    kwargs = {
        "config": config,
        "torch_dtype": torch_dtype,
        "attn_implementation": "flash_attention_2",
    }
    if tp_plan is not None:
        kwargs["tp_plan"] = tp_plan
    else:
        kwargs["device_map"] = device_map
    return cls.from_pretrained(name_or_path, **kwargs).eval()


def load_dense_llama_model(
    name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map=None,
    tp_plan=None,
    dense_backend: str = "flashinfer",
    **_ignored,
):
    model = _load_hf_model(
        LlamaForCausalLM,
        name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        tp_plan=tp_plan,
    )
    setattr(model.config, "seerattn_dense_backend", str(dense_backend))
    return _patch_dense_layers(model, rope_fn=llama_apply_rotary_pos_emb)


def load_dense_qwen2_model(
    name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map=None,
    tp_plan=None,
    dense_backend: str = "flashinfer",
    config=None,
    **_ignored,
):
    model = _load_hf_model(
        Qwen2ForCausalLM,
        name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        tp_plan=tp_plan,
        config=config,
    )
    setattr(model.config, "seerattn_dense_backend", str(dense_backend))
    return _patch_dense_layers(model, rope_fn=qwen2_apply_rotary_pos_emb)


def load_dense_qwen3_model(
    name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map=None,
    tp_plan=None,
    dense_backend: str = "flashinfer",
    config=None,
    **_ignored,
):
    model = _load_hf_model(
        Qwen3ForCausalLM,
        name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        tp_plan=tp_plan,
        config=config,
    )
    setattr(model.config, "seerattn_dense_backend", str(dense_backend))
    return _patch_dense_layers(model, rope_fn=qwen3_apply_rotary_pos_emb)


def load_dense_qwen3_moe_model(
    name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map=None,
    tp_plan=None,
    dense_backend: str = "flashinfer",
    config=None,
    **_ignored,
):
    model = _load_hf_model(
        Qwen3MoeForCausalLM,
        name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        tp_plan=tp_plan,
        config=config,
    )
    setattr(model.config, "seerattn_dense_backend", str(dense_backend))
    return _patch_dense_layers(model, rope_fn=qwen3_moe_apply_rotary_pos_emb)


def load_dense_model(name_or_path: str, **kwargs):
    model_type = getattr(
        AutoConfig.from_pretrained(name_or_path, trust_remote_code=True), "model_type", None
    )
    if model_type == "llama":
        return load_dense_llama_model(name_or_path, **kwargs)
    if model_type == "qwen2":
        return load_dense_qwen2_model(name_or_path, **kwargs)
    if model_type == "qwen3":
        return load_dense_qwen3_model(name_or_path, **kwargs)
    if model_type == "qwen3_moe":
        return load_dense_qwen3_moe_model(name_or_path, **kwargs)
    raise ValueError(f"Unsupported model_type for dense FlashInfer wrapper: {model_type!r}")
