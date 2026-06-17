# coding=utf-8
from typing import Optional, Tuple, Union

import torch
from torch import nn
from einops import rearrange
from flash_attn.layers.rotary import apply_rotary_emb_func
from transformers.cache_utils import Cache, DynamicCache

from compact_attn.modules.common import apply_rotary_pos_emb
from compact_attn.modules.dense_prefill import dense_prefill_full_kv
from compact_attn.modules.flashprefill_prefill import flashprefill_compactattn_prefill_full_kv
from compact_attn.flashprefill_vendor import flash_prefill_compute_mean_k
from compact_attn.prefill_sparse.llama.configuration_llama_seerattn import SeerAttnLlamaConfig
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn import (
    LlamaRotaryEmbedding,
    SeerAttnLlamaDecoderLayer,
    SeerAttnLlamaForCausalLM,
    SeerAttnLlamaModel,
    _build_position_ids_from_attention_mask,
)
from compact_attn.utils import BaseModelOutputWithPastAndSeer


def _record_cuda_event_pair(fn, pending_pairs, key: str, enabled: bool = True):
    if not enabled or pending_pairs is None:
        return fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    pending_pairs.append((key, start, end))
    return out


def _wait_async_collective(tensor: torch.Tensor) -> torch.Tensor:
    wait = getattr(tensor, "wait", None)
    if callable(wait):
        return wait()
    return tensor


class HeadsFirstDynamicCache(DynamicCache):
    """DynamicCache that stores KV tensors as [B, Hkv, K, D]."""


class LlamaFlashPrefillCompactAttnAttention(nn.Module):
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

        self.compactattn_pack_impl = str(
            getattr(config, "seerattn_compactattn_pack_impl", "indexed_dense")
        )
        if self.compactattn_pack_impl not in {"torch", "triton", "indexed_dense"}:
            self.compactattn_pack_impl = "indexed_dense"
        self.compactattn_indexed_impl = str(
            getattr(config, "seerattn_compactattn_indexed_impl", "fa2_paged")
        )
        if self.compactattn_indexed_impl not in {
            "fa2_paged",
            "triton_direct",
            "fa2_indexed",
            "fi_paged",
            "fi_zero_copy",
            "fi_zero_copy_per_query",
            "fi_zero_copy_subgroup",
            "cudnn_one_shot",
        }:
            self.compactattn_indexed_impl = "fa2_paged"
        self.compactattn_cache_fill_backend = str(
            getattr(config, "seerattn_compactattn_cache_fill_backend", "cuda")
        )
        if self.compactattn_cache_fill_backend not in {"auto", "cuda", "triton"}:
            self.compactattn_cache_fill_backend = "cuda"
        self.compactattn_debug = bool(
            getattr(
                config,
                "seerattn_flashprefill_compactattn_debug",
                getattr(config, "seerattn_compactattn_debug", False),
            )
        )
        self.compactattn_detailed_timing = bool(
            getattr(config, "seerattn_flashprefill_detailed_timing", False)
        )
        self._compactattn_last_stats = None
        self._compactattn_pending_timing_pairs = []
        self._last_k_hf = None
        self._last_v_hf = None
        self._flashprefill_mean_k_cache_parts = []
        self._flashprefill_mean_k_cache_token_len = 0

    def _update_flashprefill_mean_k_cache(
        self,
        current_key_states: torch.Tensor,
        *,
        full_kv_len: int,
        block_size: int,
    ) -> Optional[torch.Tensor]:
        if self.training or current_key_states is None or current_key_states.shape[1] <= 1:
            return None
        q_len = int(current_key_states.shape[1])
        start_token = int(full_kv_len) - q_len
        if start_token < 0 or start_token % int(block_size) != 0:
            self._flashprefill_mean_k_cache_parts = []
            self._flashprefill_mean_k_cache_token_len = 0
            return None
        current_mean_k = flash_prefill_compute_mean_k(
            current_key_states.contiguous(),
            block_size=int(block_size),
        )
        if start_token == 0:
            self._flashprefill_mean_k_cache_parts = [current_mean_k]
            self._flashprefill_mean_k_cache_token_len = int(full_kv_len)
        elif self._flashprefill_mean_k_cache_token_len == start_token:
            self._flashprefill_mean_k_cache_parts.append(current_mean_k)
            self._flashprefill_mean_k_cache_token_len = int(full_kv_len)
        else:
            self._flashprefill_mean_k_cache_parts = []
            self._flashprefill_mean_k_cache_token_len = 0
            return None
        return torch.cat(self._flashprefill_mean_k_cache_parts, dim=1)

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
        collect_stats = bool(self.compactattn_debug)
        pending_timing_pairs = [] if collect_stats else None

        query_states, key_states, value_states = _record_cuda_event_pair(
            lambda: (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ),
            pending_timing_pairs,
            "qkv_proj_ms",
            enabled=collect_stats,
        )

        query_states = rearrange(query_states, "... (h d) -> ... h d", d=self.head_dim)
        key_states = rearrange(key_states, "... (h d) -> ... h d", d=self.head_dim)
        value_states = rearrange(value_states, "... (h d) -> ... h d", d=self.head_dim)

        cos, sin = position_embeddings
        def _run_rope():
            use_flash_rope = bool(self.config.use_flash_rope) and cos.dim() == 2
            if use_flash_rope:
                query_states_out = apply_rotary_emb_func(
                    query_states, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len
                )
                key_states_out = apply_rotary_emb_func(
                    key_states, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len
                )
            else:
                query_states_out, key_states_out = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=2
                )
            return query_states_out, key_states_out

        query_states, key_states = _record_cuda_event_pair(
            _run_rope,
            pending_timing_pairs,
            "rope_ms",
            enabled=collect_stats,
        )
        current_key_states = key_states

        if past_key_value is not None:
            def _run_cache_update():
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                if (
                    self.compactattn_indexed_impl in {"fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"}
                    and isinstance(past_key_value, HeadsFirstDynamicCache)
                ):
                    q_len_in = key_states.shape[1]
                    k_hf_in = key_states.permute(0, 2, 1, 3).contiguous()
                    v_hf_in = value_states.permute(0, 2, 1, 3).contiguous()
                    k_hf, v_hf = past_key_value.update(
                        k_hf_in,
                        v_hf_in,
                        self.layer_idx,
                        cache_kwargs,
                    )
                    self._last_k_hf = k_hf
                    self._last_v_hf = v_hf
                    if k_hf.shape[2] > q_len_in:
                        return k_hf.permute(0, 2, 1, 3), v_hf.permute(0, 2, 1, 3)
                    return (
                        k_hf.permute(0, 2, 1, 3).contiguous(),
                        v_hf.permute(0, 2, 1, 3).contiguous(),
                    )
                self._last_k_hf = None
                self._last_v_hf = None
                key_states_out, value_states_out = past_key_value.update(
                    key_states.flatten(-2, -1),
                    value_states.flatten(-2, -1),
                    self.layer_idx,
                    cache_kwargs,
                )
                key_states_out = rearrange(key_states_out, "... (h d) -> ... h d", d=self.head_dim)
                value_states_out = rearrange(value_states_out, "... (h d) -> ... h d", d=self.head_dim)
                return key_states_out, value_states_out

            key_states, value_states = _record_cuda_event_pair(
                _run_cache_update,
                pending_timing_pairs,
                "cache_update_ms",
                enabled=collect_stats,
            )

        force_dense_prefill = (
            (not self.training)
            and (q_len > 1)
            and bool(getattr(self.config, "seerattn_chunked_prefill_force_dense", False))
        )
        block_size = int(getattr(self.config, "seerattn_flashprefill_block_size", 128))
        mean_k_cache = self._update_flashprefill_mean_k_cache(
            current_key_states,
            full_kv_len=int(key_states.shape[1]),
            block_size=block_size,
        )
        flashprefill_prefill = (
            (not self.training)
            and (q_len > 1)
            and (key_states.shape[1] > q_len)
            and not force_dense_prefill
        )

        if flashprefill_prefill:
            attn_output, compactattn_stats = _record_cuda_event_pair(
                lambda: flashprefill_compactattn_prefill_full_kv(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    attention_mask=attention_mask,
                    softmax_scale=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    block_size=block_size,
                    attention_sink=int(getattr(self.config, "seerattn_flashprefill_attention_sink", 2)),
                    window_size=int(getattr(self.config, "seerattn_flashprefill_window_size", 4)),
                    alpha=float(getattr(self.config, "seerattn_flashprefill_alpha", 0.06)),
                    last_n_block=int(getattr(self.config, "seerattn_flashprefill_last_n_block", 2)),
                    min_budget=int(getattr(self.config, "seerattn_flashprefill_min_budget", 0)),
                    pack_impl=self.compactattn_pack_impl,
                    indexed_impl=self.compactattn_indexed_impl,
                    cache_fill_backend=self.compactattn_cache_fill_backend,
                    measure_timing=self.compactattn_detailed_timing,
                    attn_module=self,
                    k_hf=getattr(self, "_last_k_hf", None),
                    v_hf=getattr(self, "_last_v_hf", None),
                    mean_k_cache=mean_k_cache,
                ),
                pending_timing_pairs,
                "attention_stack_ms",
                enabled=collect_stats,
            )
            path_label = "flashprefill_chunked_selected"
        else:
            attn_output, compactattn_stats = _record_cuda_event_pair(
                lambda: dense_prefill_full_kv(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    attention_mask=attention_mask,
                    softmax_scale=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    fallback_used=0.0,
                    measure_timing=self.compactattn_detailed_timing,
                    attn_module=self,
                ),
                pending_timing_pairs,
                "attention_stack_ms",
                enabled=collect_stats,
            )
            path_label = "dense_fallback"

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = _record_cuda_event_pair(
            lambda: self.o_proj(attn_output),
            pending_timing_pairs,
            "o_proj_ms",
            enabled=collect_stats,
        )
        if not bool(getattr(self.config, "seerattn_defer_async_collective_wait", True)):
            attn_output = _wait_async_collective(attn_output)
        if collect_stats and isinstance(compactattn_stats, dict):
            merged_stats = dict(compactattn_stats)
            merged_stats.update(
                {
                    "path": path_label,
                    "q_len": float(q_len),
                    "kv_len": float(key_states.shape[1]),
                    "layer_idx": float(self.layer_idx),
                }
            )
            self._compactattn_last_stats = merged_stats
            self._compactattn_pending_timing_pairs = pending_timing_pairs or []
        else:
            self._compactattn_last_stats = None
            self._compactattn_pending_timing_pairs = []

        if not kwargs.get("output_attentions", False):
            return attn_output, 0.0, None, None, None
        return attn_output, 0.0, None, None, None


class SeerAttnLlamaFlashPrefillCompactAttnDecoderLayer(SeerAttnLlamaDecoderLayer):
    def __init__(self, config: SeerAttnLlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = LlamaFlashPrefillCompactAttnAttention(config=config, layer_idx=layer_idx)


class SeerAttnLlamaFlashPrefillCompactAttnModel(SeerAttnLlamaModel):
    def __init__(self, config: SeerAttnLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                SeerAttnLlamaFlashPrefillCompactAttnDecoderLayer(config=config, layer_idx=layer_idx)
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
            if str(getattr(self.config, "seerattn_compactattn_indexed_impl", "")) in {
                "fi_zero_copy",
                "fi_zero_copy_per_query",
                "fi_zero_copy_subgroup",
                "cudnn_one_shot",
            }:
                past_key_values = HeadsFirstDynamicCache()
            else:
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


class SeerAttnLlamaFlashPrefillCompactAttnForCausalLM(SeerAttnLlamaForCausalLM):
    def __init__(self, config: SeerAttnLlamaConfig):
        super().__init__(config)
        defaults = {
            "seerattn_dense_backend": "flashinfer",
            "seerattn_flashprefill_alpha": 0.06,
            "seerattn_flashprefill_block_size": 128,
            "seerattn_flashprefill_attention_sink": 2,
            "seerattn_flashprefill_window_size": 4,
            "seerattn_flashprefill_last_n_block": 2,
            "seerattn_flashprefill_min_budget": 0,
            "seerattn_compactattn_pack_impl": "indexed_dense",
            "seerattn_compactattn_indexed_impl": "fa2_paged",
            "seerattn_compactattn_cache_fill_backend": "cuda",
            "seerattn_defer_async_collective_wait": True,
        }
        for key, value in defaults.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        if not hasattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks"):
            setattr(
                config,
                "seerattn_chunked_prefill_final_dense_tail_blocks",
                0,
            )
        self.model = SeerAttnLlamaFlashPrefillCompactAttnModel(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        dense_backend = kwargs.pop("seerattn_dense_backend", "flashinfer")
        flashprefill_alpha = kwargs.pop("seerattn_flashprefill_alpha", 0.06)
        flashprefill_block_size = kwargs.pop("seerattn_flashprefill_block_size", 128)
        flashprefill_attention_sink = kwargs.pop("seerattn_flashprefill_attention_sink", 2)
        flashprefill_window_size = kwargs.pop("seerattn_flashprefill_window_size", 4)
        flashprefill_last_n_block = kwargs.pop("seerattn_flashprefill_last_n_block", 2)
        flashprefill_min_budget = kwargs.pop("seerattn_flashprefill_min_budget", 0)
        compactattn_pack_impl = kwargs.pop("seerattn_compactattn_pack_impl", "indexed_dense")
        compactattn_indexed_impl = kwargs.pop("seerattn_compactattn_indexed_impl", "fa2_paged")
        compactattn_cache_fill_backend = kwargs.pop("seerattn_compactattn_cache_fill_backend", "cuda")
        compactattn_debug = kwargs.pop("seerattn_flashprefill_compactattn_debug", False)
        flashprefill_detailed_timing = kwargs.pop("seerattn_flashprefill_detailed_timing", False)
        final_dense_tail_blocks = kwargs.pop("seerattn_chunked_prefill_final_dense_tail_blocks", None)
        defer_async_collective_wait = kwargs.pop("seerattn_defer_async_collective_wait", True)

        config = SeerAttnLlamaConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        base_model = getattr(config, "base_model", pretrained_model_name_or_path)
        for key in list(kwargs.keys()):
            if hasattr(config, key) and key != "torch_dtype":
                setattr(config, key, kwargs.pop(key))
        setattr(config, "seerattn_dense_backend", str(dense_backend))
        setattr(config, "seerattn_flashprefill_alpha", float(flashprefill_alpha))
        setattr(config, "seerattn_flashprefill_block_size", int(flashprefill_block_size))
        setattr(config, "seerattn_flashprefill_attention_sink", int(flashprefill_attention_sink))
        setattr(config, "seerattn_flashprefill_window_size", int(flashprefill_window_size))
        setattr(config, "seerattn_flashprefill_last_n_block", int(flashprefill_last_n_block))
        setattr(config, "seerattn_flashprefill_min_budget", int(flashprefill_min_budget))
        setattr(config, "seerattn_compactattn_pack_impl", str(compactattn_pack_impl))
        setattr(config, "seerattn_compactattn_indexed_impl", str(compactattn_indexed_impl))
        setattr(config, "seerattn_compactattn_cache_fill_backend", str(compactattn_cache_fill_backend))
        setattr(config, "seerattn_flashprefill_compactattn_debug", bool(compactattn_debug))
        setattr(config, "seerattn_flashprefill_detailed_timing", bool(flashprefill_detailed_timing))
        setattr(config, "seerattn_defer_async_collective_wait", bool(defer_async_collective_wait))
        if final_dense_tail_blocks is None:
            final_dense_tail_blocks = 0
        setattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks", int(final_dense_tail_blocks))

        return super(SeerAttnLlamaForCausalLM, cls).from_pretrained(
            base_model,
            config=config,
            *model_args,
            **kwargs,
        )
