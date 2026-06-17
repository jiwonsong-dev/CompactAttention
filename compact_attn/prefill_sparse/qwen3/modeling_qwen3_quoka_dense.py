# coding=utf-8
from typing import Optional, Tuple

import copy

import torch
from torch import nn
from einops import rearrange
from flash_attn.layers.rotary import apply_rotary_emb_func
from transformers.cache_utils import Cache

from compact_attn.modules.common import apply_rotary_pos_emb
from compact_attn.modules.dense_prefill import dense_prefill_full_kv
from compact_attn.modules.quoka_prefill import quoka_dense_prefill_full_kv

from .configuration_qwen3_seerattn import SeerAttnQwen3Config
from .modeling_qwen3_seerattn import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    SeerAttnQwen3ForCausalLM,
    SeerAttnQwen3Model,
    SeerAttnQwen3PreTrainedModel,
    _sync_qwen3_base_vocab_size,
)
from compact_attn.prefill_sparse.qwen.modeling_qwen2_seerattn import SeerAttnQwen2ForCausalLM


class Qwen3QuokaDenseAttention(nn.Module):
    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.quoka_query_ratio = float(getattr(config, "seerattn_quoka_query_ratio", 0.25))
        self.quoka_kv_budget_ratio = float(getattr(config, "seerattn_quoka_kv_budget_ratio", 0.25))
        self.quoka_score_chunk_size = int(getattr(config, "seerattn_quoka_score_chunk_size", 4096))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        q_len = hidden_states.shape[1]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = rearrange(query_states, "... (h d) -> ... h d", d=self.head_dim)
        key_states = rearrange(key_states, "... (h d) -> ... h d", d=self.head_dim)
        value_states = rearrange(value_states, "... (h d) -> ... h d", d=self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        if self.config.use_flash_rope:
            query_states = apply_rotary_emb_func(
                query_states, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len
            )
            key_states = apply_rotary_emb_func(
                key_states, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=2
            )

        current_key_states = key_states
        current_value_states = value_states
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states.flatten(-2, -1),
                value_states.flatten(-2, -1),
                self.layer_idx,
                cache_kwargs,
            )
            key_states = rearrange(key_states, "... (h d) -> ... h d", d=self.head_dim)
            value_states = rearrange(value_states, "... (h d) -> ... h d", d=self.head_dim)
        else:
            key_states = current_key_states
            value_states = current_value_states

        force_dense_prefill = (
            (not self.training)
            and (q_len > 1)
            and bool(getattr(self.config, "seerattn_chunked_prefill_force_dense", False))
        )
        quoka_prefill = (
            (not self.training)
            and (q_len > 1)
            and (key_states.shape[1] > q_len)
            and not force_dense_prefill
        )

        if quoka_prefill:
            attn_output, _ = quoka_dense_prefill_full_kv(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                softmax_scale=self.scaling,
                num_key_value_groups=self.num_key_value_groups,
                query_ratio=self.quoka_query_ratio,
                kv_budget_ratio=self.quoka_kv_budget_ratio,
                score_chunk_size=self.quoka_score_chunk_size,
                measure_timing=False,
                attn_module=self,
            )
        else:
            attn_output, _ = dense_prefill_full_kv(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                softmax_scale=self.scaling,
                num_key_value_groups=self.num_key_value_groups,
                fallback_used=0.0,
                measure_timing=False,
                attn_module=self,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not kwargs.get("output_attentions", False):
            return attn_output, 0.0, None, None, None
        return attn_output, 0.0, None, None, None


class SeerAttnQwen3QuokaDenseDecoderLayer(nn.Module):
    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3QuokaDenseAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, seerattn_mask_loss, self_attn_weights, mask_gate_prediction, mask_ground_truth = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, seerattn_mask_loss)
        if output_attentions:
            outputs += (self_attn_weights, mask_gate_prediction, mask_ground_truth)
        return outputs


class SeerAttnQwen3QuokaDenseModel(SeerAttnQwen3Model):
    def __init__(self, config: SeerAttnQwen3Config):
        SeerAttnQwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                SeerAttnQwen3QuokaDenseDecoderLayer(config=config, layer_idx=layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        block_config = copy.deepcopy(config)
        block_config.hidden_size = config.seerattn_gate_hidden_size * config.num_attention_heads
        self.block_rotary_emb = Qwen3RotaryEmbedding(config=block_config)
        self.gradient_checkpointing = False
        self.post_init()


class SeerAttnQwen3QuokaDenseForCausalLM(SeerAttnQwen3ForCausalLM):
    def __init__(self, config: SeerAttnQwen3Config):
        super().__init__(config)
        if not hasattr(config, "seerattn_quoka_query_ratio"):
            setattr(config, "seerattn_quoka_query_ratio", 0.25)
        if not hasattr(config, "seerattn_quoka_kv_budget_ratio"):
            setattr(config, "seerattn_quoka_kv_budget_ratio", 0.25)
        if not hasattr(config, "seerattn_quoka_score_chunk_size"):
            setattr(config, "seerattn_quoka_score_chunk_size", 4096)
        self.model = SeerAttnQwen3QuokaDenseModel(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.pop("load_gate", None)
        quoka_query_ratio = kwargs.pop("seerattn_quoka_query_ratio", 0.25)
        quoka_kv_budget_ratio = kwargs.pop("seerattn_quoka_kv_budget_ratio", 0.25)
        quoka_score_chunk_size = kwargs.pop("seerattn_quoka_score_chunk_size", 4096)
        force_dense_prefill = kwargs.pop("seerattn_chunked_prefill_force_dense", False)
        final_dense_tail_blocks = kwargs.pop("seerattn_chunked_prefill_final_dense_tail_blocks", 0)
        incoming_config = kwargs.pop("config", None)
        if incoming_config is None:
            config = SeerAttnQwen3Config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif isinstance(incoming_config, SeerAttnQwen3Config):
            config = incoming_config
        else:
            config = SeerAttnQwen3Config.from_dict(incoming_config.to_dict())
        base_model = getattr(config, "base_model", pretrained_model_name_or_path)
        _sync_qwen3_base_vocab_size(config, base_model, kwargs)
        for key in list(kwargs.keys()):
            if hasattr(config, key) and key != "torch_dtype":
                setattr(config, key, kwargs.pop(key))
        setattr(config, "seerattn_quoka_query_ratio", float(quoka_query_ratio))
        setattr(config, "seerattn_quoka_kv_budget_ratio", float(quoka_kv_budget_ratio))
        setattr(config, "seerattn_quoka_score_chunk_size", int(quoka_score_chunk_size))
        setattr(config, "seerattn_chunked_prefill_force_dense", bool(force_dense_prefill))
        setattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks", int(final_dense_tail_blocks))

        return super(SeerAttnQwen2ForCausalLM, cls).from_pretrained(
            base_model,
            config=config,
            *model_args,
            **kwargs,
        )


__all__ = ["SeerAttnQwen3QuokaDenseForCausalLM"]
