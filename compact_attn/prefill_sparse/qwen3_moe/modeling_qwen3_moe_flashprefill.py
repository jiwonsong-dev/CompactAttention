# coding=utf-8
"""FlashPrefill / CompactAttention(FP) wrappers for HF Qwen3-MoE models.

The model architecture is kept identical to Hugging Face's Qwen3-MoE
implementation except for the self-attention module.  MoE blocks, router
logic, embeddings, norms, and LM head are inherited unchanged.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    apply_rotary_pos_emb,
)

from compact_attn.modules.dense_prefill import dense_prefill_full_kv
from compact_attn.modules.flashprefill_prefill import (
    flashprefill_block_sparse_prefill_full_kv,
    flashprefill_compactattn_prefill_full_kv,
)
from compact_attn.flashprefill_vendor import flash_prefill_compute_mean_k
from compact_attn.prefill_sparse.llama.modeling_llama_flashprefill_compactattn import (
    HeadsFirstDynamicCache,
    _record_cuda_event_pair,
    _wait_async_collective,
)


class Qwen3MoeFlashPrefillAttention(Qwen3MoeAttention):
    """Qwen3-MoE attention with FlashPrefill selection.

    ``execution_mode=block_sparse`` uses FlashPrefill's block-sparse kernel.
    ``execution_mode=compactattn`` uses the CompactAttention(FP) execution path,
    including FlashInfer zero-copy when ``seerattn_compactattn_indexed_impl`` is
    ``fi_zero_copy``.
    """

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.flashprefill_execution_mode = str(
            getattr(config, "seerattn_flashprefill_execution_mode", "block_sparse")
        )
        if self.flashprefill_execution_mode not in {"block_sparse", "compactattn"}:
            self.flashprefill_execution_mode = "block_sparse"

        self.compactattn_pack_impl = str(
            getattr(config, "seerattn_compactattn_pack_impl", "indexed_dense")
        )
        if self.compactattn_pack_impl not in {"torch", "triton", "indexed_dense"}:
            self.compactattn_pack_impl = "indexed_dense"
        self.compactattn_indexed_impl = str(
            getattr(config, "seerattn_compactattn_indexed_impl", "fa2_paged")
        )
        if self.compactattn_indexed_impl == "fi_zero_copy":
            self.compactattn_indexed_impl = "fi_zero_copy_subgroup"
            setattr(config, "seerattn_compactattn_indexed_impl", self.compactattn_indexed_impl)
            os.environ.setdefault("SEER_ZC_QUERY_SUBGROUP_SIZE", "4")
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

        self.flashprefill_debug = bool(
            getattr(config, "seerattn_flashprefill_block_sparse_debug", False)
        )
        self.flashprefill_profile_selection = bool(
            getattr(config, "seerattn_flashprefill_block_sparse_profile_selection", False)
        )
        self.compactattn_debug = bool(
            getattr(
                config,
                "seerattn_flashprefill_compactattn_debug",
                getattr(config, "seerattn_compactattn_debug", False),
            )
        )
        self.flashprefill_detailed_timing = bool(
            getattr(config, "seerattn_flashprefill_detailed_timing", False)
        )
        self.compactattn_detailed_timing = bool(
            getattr(
                config,
                "seerattn_compactattn_detailed_timing",
                self.flashprefill_detailed_timing,
            )
        )
        # Keep the LLaMA FlashPrefill block-sparse profiler attribute names so
        # shared benchmark tooling can enable Qwen3-MoE diagnostics too.
        self.block_sparse_debug = self.flashprefill_debug
        self.block_sparse_profile_selection = self.flashprefill_profile_selection
        self.block_sparse_detailed_timing = self.flashprefill_detailed_timing
        self._compactattn_last_stats = None
        self._compactattn_pending_timing_pairs = []
        self._block_sparse_last_stats = None
        self._block_sparse_pending_timing_pairs = []
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
        if os.environ.get("SEER_FLASHPREFILL_DISABLE_MEAN_K_CACHE", "0") == "1":
            self._flashprefill_mean_k_cache_parts = []
            self._flashprefill_mean_k_cache_token_len = 0
            return None
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
        q_len = int(hidden_states.shape[1])
        collect_stats = bool(
            self.compactattn_debug
            or self.flashprefill_debug
            or getattr(self, "block_sparse_debug", False)
        )
        pending_timing_pairs = [] if collect_stats else None
        hidden_shape = (*input_shape, -1, self.head_dim)

        def _qkv_proj():
            q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
            k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
            v = self.v_proj(hidden_states).view(hidden_shape)
            return q, k, v

        query_states, key_states, value_states = _record_cuda_event_pair(
            _qkv_proj,
            pending_timing_pairs,
            "qkv_proj_ms",
            enabled=collect_stats,
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _record_cuda_event_pair(
            lambda: apply_rotary_pos_emb(query_states, key_states, cos, sin),
            pending_timing_pairs,
            "rope_ms",
            enabled=collect_stats,
        )

        # Keep a sequence-first copy of the current chunk for FlashPrefill's
        # pooled-K scoring cache.  This matches the LLaMA FP path.
        current_key_states = key_states.transpose(1, 2).contiguous()

        if past_key_value is not None:

            def _cache_update():
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                if (
                    self.flashprefill_execution_mode == "compactattn"
                    and self.compactattn_indexed_impl in {"fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"}
                    and isinstance(past_key_value, HeadsFirstDynamicCache)
                ):
                    k_hf_in = key_states.contiguous()
                    v_hf_in = value_states.contiguous()
                    k_hf, v_hf = past_key_value.update(
                        k_hf_in,
                        v_hf_in,
                        self.layer_idx,
                        cache_kwargs,
                    )
                    self._last_k_hf = k_hf
                    self._last_v_hf = v_hf
                    k_sf = k_hf.transpose(1, 2)
                    v_sf = v_hf.transpose(1, 2)
                    if k_hf.shape[2] <= q_len:
                        k_sf = k_sf.contiguous()
                        v_sf = v_sf.contiguous()
                    return k_sf, v_sf

                self._last_k_hf = None
                self._last_v_hf = None
                k_out, v_out = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )
                return k_out.transpose(1, 2).contiguous(), v_out.transpose(1, 2).contiguous()

            key_states_sf, value_states_sf = _record_cuda_event_pair(
                _cache_update,
                pending_timing_pairs,
                "cache_update_ms",
                enabled=collect_stats,
            )
        else:
            key_states_sf = key_states.transpose(1, 2).contiguous()
            value_states_sf = value_states.transpose(1, 2).contiguous()
            self._last_k_hf = None
            self._last_v_hf = None

        query_states_sf = query_states.transpose(1, 2).contiguous()
        kv_len = int(key_states_sf.shape[1])
        force_dense_prefill = (
            (not self.training)
            and (q_len > 1)
            and bool(getattr(self.config, "seerattn_chunked_prefill_force_dense", False))
        )
        flashprefill_prefill = (
            (not self.training)
            and (q_len > 1)
            and (kv_len > q_len)
            and not force_dense_prefill
        )
        block_size = int(getattr(self.config, "seerattn_flashprefill_block_size", 128))
        mean_k_cache = self._update_flashprefill_mean_k_cache(
            current_key_states,
            full_kv_len=kv_len,
            block_size=block_size,
        )

        if flashprefill_prefill and self.flashprefill_execution_mode == "block_sparse":
            attn_output, stats = _record_cuda_event_pair(
                lambda: flashprefill_block_sparse_prefill_full_kv(
                    query_states=query_states_sf,
                    key_states=key_states_sf,
                    value_states=value_states_sf,
                    attention_mask=attention_mask,
                    softmax_scale=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    block_size=block_size,
                    attention_sink=int(getattr(self.config, "seerattn_flashprefill_attention_sink", 2)),
                    window_size=int(getattr(self.config, "seerattn_flashprefill_window_size", 4)),
                    alpha=float(getattr(self.config, "seerattn_flashprefill_alpha", 0.01)),
                    last_n_block=int(getattr(self.config, "seerattn_flashprefill_last_n_block", 2)),
                    min_budget=int(getattr(self.config, "seerattn_flashprefill_min_budget", 0)),
                    measure_timing=bool(
                        self.flashprefill_detailed_timing
                        or getattr(self, "block_sparse_detailed_timing", False)
                    ),
                    profile_selection=bool(
                        self.flashprefill_profile_selection
                        or getattr(self, "block_sparse_profile_selection", False)
                    ),
                    attn_module=self,
                    mean_k_cache=mean_k_cache,
                ),
                pending_timing_pairs,
                "attention_stack_ms",
                enabled=collect_stats,
            )
            path_label = "flashprefill_chunked_selected"
        elif flashprefill_prefill and self.flashprefill_execution_mode == "compactattn":
            attn_output, stats = _record_cuda_event_pair(
                lambda: flashprefill_compactattn_prefill_full_kv(
                    query_states=query_states_sf,
                    key_states=key_states_sf,
                    value_states=value_states_sf,
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
                    measure_timing=self.flashprefill_detailed_timing or self.compactattn_detailed_timing,
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
            attn_output, stats = _record_cuda_event_pair(
                lambda: dense_prefill_full_kv(
                    query_states=query_states_sf,
                    key_states=key_states_sf,
                    value_states=value_states_sf,
                    attention_mask=attention_mask,
                    softmax_scale=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    fallback_used=0.0,
                    measure_timing=self.flashprefill_detailed_timing or self.compactattn_detailed_timing,
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

        if collect_stats and isinstance(stats, dict):
            merged_stats = dict(stats)
            merged_stats.update(
                {
                    "path": path_label,
                    "q_len": float(q_len),
                    "kv_len": float(kv_len),
                    "layer_idx": float(self.layer_idx),
                }
            )
            if self.flashprefill_execution_mode == "block_sparse":
                self._block_sparse_last_stats = merged_stats
                self._block_sparse_pending_timing_pairs = pending_timing_pairs or []
                self._compactattn_last_stats = None
                self._compactattn_pending_timing_pairs = []
            else:
                self._compactattn_last_stats = merged_stats
                self._compactattn_pending_timing_pairs = pending_timing_pairs or []
                self._block_sparse_last_stats = None
                self._block_sparse_pending_timing_pairs = []
        else:
            self._compactattn_last_stats = None
            self._compactattn_pending_timing_pairs = []
            self._block_sparse_last_stats = None
            self._block_sparse_pending_timing_pairs = []

        return attn_output, None


class Qwen3MoeFlashPrefillDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = Qwen3MoeFlashPrefillAttention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        router_logits = None
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        hidden_states = residual + hidden_states
        if output_router_logits:
            return hidden_states, router_logits
        return hidden_states


class Qwen3MoeFlashPrefillModel(Qwen3MoeModel):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                Qwen3MoeFlashPrefillDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            if str(getattr(self.config, "seerattn_compactattn_indexed_impl", "")) in {
                "fi_zero_copy",
                "fi_zero_copy_per_query",
                "fi_zero_copy_subgroup",
                "cudnn_one_shot",
            } and str(getattr(self.config, "seerattn_flashprefill_execution_mode", "")) == "compactattn":
                past_key_values = HeadsFirstDynamicCache()
            else:
                past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # CompactAttention chunked-prefill runtime uses exact same-length,
        # no-padding batches.  Passing the HF causal mask would force the FP
        # kernels into dense fallback, while the kernels already implement the
        # causal prefill constraint internally.
        layer_attention_mask = None
        if attention_mask is not None and (attention_mask == 0).any().item():
            layer_attention_mask = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        output_router_logits = bool(kwargs.pop("output_router_logits", False))
        router_logits = [] if output_router_logits else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                output_router_logits=output_router_logits,
                **kwargs,
            )
            if output_router_logits:
                hidden_states, layer_router_logits = hidden_states
                router_logits.append(layer_router_logits)

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            router_logits=tuple(router_logits) if router_logits is not None else None,
        )


class SeerAttnQwen3MoeFlashPrefillForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config: Qwen3MoeConfig):
        _set_flashprefill_defaults(config, execution_mode="block_sparse")
        super().__init__(config)
        self.model = Qwen3MoeFlashPrefillModel(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = Qwen3MoeConfig.from_pretrained(pretrained_model_name_or_path, **_config_kwargs(kwargs))
        _set_flashprefill_config_from_kwargs(config, kwargs, execution_mode="block_sparse")
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **kwargs,
        )


class SeerAttnQwen3MoeFlashPrefillCompactAttnForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config: Qwen3MoeConfig):
        _set_flashprefill_defaults(config, execution_mode="compactattn")
        super().__init__(config)
        self.model = Qwen3MoeFlashPrefillModel(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = Qwen3MoeConfig.from_pretrained(pretrained_model_name_or_path, **_config_kwargs(kwargs))
        _set_flashprefill_config_from_kwargs(config, kwargs, execution_mode="compactattn")
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **kwargs,
        )


def _config_kwargs(kwargs: dict) -> dict:
    return {
        key: kwargs[key]
        for key in (
            "cache_dir",
            "force_download",
            "local_files_only",
            "revision",
            "token",
            "trust_remote_code",
        )
        if key in kwargs
    }


def _set_flashprefill_defaults(config: Qwen3MoeConfig, *, execution_mode: str) -> None:
    defaults = {
        "seerattn_dense_backend": "flashinfer",
        "seerattn_flashprefill_execution_mode": execution_mode,
        "seerattn_flashprefill_alpha": 0.06 if execution_mode == "compactattn" else 0.01,
        "seerattn_flashprefill_block_size": 128,
        "seerattn_flashprefill_attention_sink": 2,
        "seerattn_flashprefill_window_size": 4,
        "seerattn_flashprefill_last_n_block": 2,
        "seerattn_flashprefill_min_budget": 0,
        "seerattn_compactattn_pack_impl": "indexed_dense",
        "seerattn_compactattn_indexed_impl": "fa2_paged",
        "seerattn_compactattn_cache_fill_backend": "cuda",
        "seerattn_chunked_prefill_final_dense_tail_blocks": 0,
        "seerattn_defer_async_collective_wait": True,
    }
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)


def _set_flashprefill_config_from_kwargs(config: Qwen3MoeConfig, kwargs: dict, *, execution_mode: str) -> None:
    _set_flashprefill_defaults(config, execution_mode=execution_mode)
    if "use_cache" in kwargs:
        config.use_cache = bool(kwargs.pop("use_cache"))
    mapping = {
        "seerattn_dense_backend": str,
        "seerattn_flashprefill_alpha": float,
        "seerattn_flashprefill_block_size": int,
        "seerattn_flashprefill_attention_sink": int,
        "seerattn_flashprefill_window_size": int,
        "seerattn_flashprefill_last_n_block": int,
        "seerattn_flashprefill_min_budget": int,
        "seerattn_compactattn_pack_impl": str,
        "seerattn_compactattn_indexed_impl": str,
        "seerattn_compactattn_cache_fill_backend": str,
        "seerattn_flashprefill_compactattn_debug": bool,
        "seerattn_flashprefill_block_sparse_debug": bool,
        "seerattn_flashprefill_block_sparse_profile_selection": bool,
        "seerattn_flashprefill_detailed_timing": bool,
        "seerattn_chunked_prefill_force_dense": bool,
        "seerattn_chunked_prefill_final_dense_tail_blocks": int,
        "seerattn_defer_async_collective_wait": bool,
    }
    for key, caster in mapping.items():
        if key in kwargs:
            setattr(config, key, caster(kwargs.pop(key)))
    setattr(config, "seerattn_flashprefill_execution_mode", execution_mode)


__all__ = [
    "SeerAttnQwen3MoeFlashPrefillForCausalLM",
    "SeerAttnQwen3MoeFlashPrefillCompactAttnForCausalLM",
]
