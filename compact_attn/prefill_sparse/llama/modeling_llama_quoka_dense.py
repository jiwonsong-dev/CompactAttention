# coding=utf-8
from typing import Optional, Tuple, Union

import torch
from torch import nn
from einops import rearrange
from flash_attn.layers.rotary import apply_rotary_emb_func
from transformers.cache_utils import Cache, DynamicCache

from compact_attn.modules.common import apply_rotary_pos_emb
from compact_attn.modules.dense_prefill import dense_prefill_full_kv
from compact_attn.modules.quoka_prefill import quoka_dense_prefill_full_kv
from compact_attn.prefill_sparse.llama.configuration_llama_seerattn import SeerAttnLlamaConfig
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn import (
    LlamaRotaryEmbedding,
    SeerAttnLlamaDecoderLayer,
    SeerAttnLlamaForCausalLM,
    SeerAttnLlamaModel,
    _build_position_ids_from_attention_mask,
)
from compact_attn.utils import BaseModelOutputWithPastAndSeer


class LlamaQuokaDenseAttention(nn.Module):
    def __init__(self, config: SeerAttnLlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
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

        cos, sin = position_embeddings
        use_flash_rope = bool(self.config.use_flash_rope) and cos.dim() == 2
        if use_flash_rope:
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


class SeerAttnLlamaQuokaDenseDecoderLayer(SeerAttnLlamaDecoderLayer):
    def __init__(self, config: SeerAttnLlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = LlamaQuokaDenseAttention(config=config, layer_idx=layer_idx)


class SeerAttnLlamaQuokaDenseModel(SeerAttnLlamaModel):
    def __init__(self, config: SeerAttnLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                SeerAttnLlamaQuokaDenseDecoderLayer(config=config, layer_idx=layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        seer_batch_kv_lens: Optional[torch.LongTensor] = None,
        seer_batch_query_lens: Optional[torch.LongTensor] = None,
        seer_batch_query_start: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndSeer]:
        if attention_mask is not None:
            if not (attention_mask == 0).any().item():
                input_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
                if attention_mask.shape[-1] == input_length:
                    attention_mask = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            try:
                past_key_values = DynamicCache(config=self.config)
            except TypeError:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = _build_position_ids_from_attention_mask(
                attention_mask=attention_mask,
                cache_position=cache_position,
                query_length=inputs_embeds.shape[1],
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_mask_gate_predictions = () if output_attentions else None
        all_mask_ground_truths = () if output_attentions else None
        total_mask_loss = 0.0

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                seer_batch_kv_lens=seer_batch_kv_lens,
                seer_batch_query_lens=seer_batch_query_lens,
                seer_batch_query_start=seer_batch_query_start,
            )

            hidden_states = layer_outputs[0]
            total_mask_loss += layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)
                all_mask_gate_predictions += (layer_outputs[3],)
                all_mask_ground_truths += (layer_outputs[4],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPastAndSeer(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mask_gate_predictions=all_mask_gate_predictions,
            mask_ground_truths=all_mask_ground_truths,
            mask_loss=total_mask_loss,
        )
        return output if return_dict else output.to_tuple()


class SeerAttnLlamaQuokaDenseForCausalLM(SeerAttnLlamaForCausalLM):
    def __init__(self, config: SeerAttnLlamaConfig):
        super().__init__(config)
        if not hasattr(config, "seerattn_quoka_query_ratio"):
            setattr(config, "seerattn_quoka_query_ratio", 0.25)
        if not hasattr(config, "seerattn_quoka_kv_budget_ratio"):
            setattr(config, "seerattn_quoka_kv_budget_ratio", 0.25)
        if not hasattr(config, "seerattn_quoka_score_chunk_size"):
            setattr(config, "seerattn_quoka_score_chunk_size", 4096)
        self.model = SeerAttnLlamaQuokaDenseModel(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        quoka_query_ratio = kwargs.pop("seerattn_quoka_query_ratio", 0.25)
        quoka_kv_budget_ratio = kwargs.pop("seerattn_quoka_kv_budget_ratio", 0.25)
        quoka_score_chunk_size = kwargs.pop("seerattn_quoka_score_chunk_size", 4096)

        config = SeerAttnLlamaConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        base_model = getattr(config, "base_model", pretrained_model_name_or_path)
        for key in list(kwargs.keys()):
            if hasattr(config, key) and key != "torch_dtype":
                setattr(config, key, kwargs.pop(key))
        setattr(config, "seerattn_quoka_query_ratio", float(quoka_query_ratio))
        setattr(config, "seerattn_quoka_kv_budget_ratio", float(quoka_kv_budget_ratio))
        setattr(config, "seerattn_quoka_score_chunk_size", int(quoka_score_chunk_size))

        return super(SeerAttnLlamaForCausalLM, cls).from_pretrained(
            base_model,
            config=config,
            *model_args,
            **kwargs,
        )
