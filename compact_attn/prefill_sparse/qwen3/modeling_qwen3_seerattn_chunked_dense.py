import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from huggingface_hub import hf_hub_download
from flash_attn.layers.rotary import apply_rotary_emb_func
from transformers.cache_utils import Cache

from compact_attn.kernels.varlen.indexed_dense_prefill_varlen import clear_indexed_dense_workspaces
from compact_attn.modules.attention_distill import attention_distill_forward
from compact_attn.modules.attention_forward import sparse_flash_attention_forward
from compact_attn.modules.common import apply_rotary_pos_emb
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn import LlamaSeerAttention
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn_chunked_dense import (
    COMPACTATTN_VERSION,
    LlamaSeerAttentionChunkedDense,
    _clear_indexed_dense_workspaces_for_model,
    _cuda_elapsed_ms,
    _dense_prefill_full_kv,
    _load_gate_weights_with_optional_q_reinit,
    _note_stage_begin,
    _note_stage_done,
    chunked_prefill_column_dense_attention_forward,
    chunked_prefill_column_dense_attention_from_keep_block,
    clear_fast_path_content_cache,
)
from compact_attn.prefill_sparse.qwen.modeling_qwen2_seerattn import SeerAttnQwen2ForCausalLM

from .configuration_qwen3_seerattn import SeerAttnQwen3Config
from .modeling_qwen3_seerattn import (
    SeerAttnQwen3Attention,
    SeerAttnQwen3DecoderLayer,
    SeerAttnQwen3ForCausalLM,
    SeerAttnQwen3Model,
    _resolve_qwen_seer_device_map,
    _sync_qwen3_base_vocab_size,
)


class Qwen3SeerAttentionChunkedDense(SeerAttnQwen3Attention):
    _can_use_chunked_gate_cache = LlamaSeerAttention._can_use_chunked_gate_cache
    _build_chunked_gate_blocks = LlamaSeerAttention._build_chunked_gate_blocks

    _resolve_compactattn_threshold = LlamaSeerAttentionChunkedDense._resolve_compactattn_threshold
    _compactattn_chunked_gate_cache_store = LlamaSeerAttentionChunkedDense._compactattn_chunked_gate_cache_store
    _compactattn_cached_k_blocks_from_current = (
        LlamaSeerAttentionChunkedDense._compactattn_cached_k_blocks_from_current
    )
    _compactattn_gate_cache_capacity = LlamaSeerAttentionChunkedDense._compactattn_gate_cache_capacity
    _compactattn_get_cached_k_blocks = LlamaSeerAttentionChunkedDense._compactattn_get_cached_k_blocks
    _compactattn_append_cached_k_blocks = LlamaSeerAttentionChunkedDense._compactattn_append_cached_k_blocks
    _append_compactattn_chunked_gate_key_cache = (
        LlamaSeerAttentionChunkedDense._append_compactattn_chunked_gate_key_cache
    )
    _gate_k_get = LlamaSeerAttentionChunkedDense._gate_k_get
    _gate_k_append = LlamaSeerAttentionChunkedDense._gate_k_append
    _compute_chunked_gate_from_cache = LlamaSeerAttentionChunkedDense._compute_chunked_gate_from_cache
    _compute_compactattn_gate_output = LlamaSeerAttentionChunkedDense._compute_compactattn_gate_output
    _expand_compactattn_gate_scores_for_q_heads = (
        LlamaSeerAttentionChunkedDense._expand_compactattn_gate_scores_for_q_heads
    )
    _compute_compactattn_mask_loss = LlamaSeerAttentionChunkedDense._compute_compactattn_mask_loss
    _maybe_record_compactattn_threshold_calibration = (
        LlamaSeerAttentionChunkedDense._maybe_record_compactattn_threshold_calibration
    )

    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.compactattn_kv_group_aware_gate = bool(
            getattr(config, "seerattn_compactattn_kv_group_aware_gate", False)
        )
        self.compactattn_threshold = float(
            getattr(config, "seerattn_compactattn_threshold", config.seerattn_threshold)
        )
        self.compactattn_threshold_schedule = getattr(
            config, "seerattn_compactattn_threshold_schedule", None
        )
        self.compactattn_use_chunked_gate_cache = bool(
            getattr(config, "seerattn_compactattn_use_chunked_gate_cache", True)
        )
        self.compactattn_keep_recent_blocks = int(
            getattr(config, "seerattn_compactattn_keep_recent_blocks", 2)
        )
        self.compactattn_release_indexed_workspaces = bool(
            getattr(config, "seerattn_compactattn_release_indexed_workspaces", True)
        )
        self.compactattn_pack_impl = str(
            getattr(config, "seerattn_compactattn_pack_impl", "indexed_dense")
        )
        if self.compactattn_pack_impl not in {"torch", "triton", "indexed_dense"}:
            self.compactattn_pack_impl = "torch"
        self.compactattn_indexed_impl = str(
            getattr(config, "seerattn_compactattn_indexed_impl", "fa2_paged")
        )
        if self.compactattn_indexed_impl not in {"fa2_paged", "triton_direct", "fa2_indexed"}:
            self.compactattn_indexed_impl = "fa2_paged"
        self.compactattn_cache_fill_backend = str(
            getattr(config, "seerattn_compactattn_cache_fill_backend", "auto")
        )
        if self.compactattn_cache_fill_backend not in {"auto", "cuda", "triton"}:
            self.compactattn_cache_fill_backend = "auto"
        self.compactattn_version = COMPACTATTN_VERSION
        self.compactattn_debug = bool(getattr(config, "seerattn_compactattn_debug", False))
        self.compactattn_disable_first_chunk_dense = bool(
            getattr(config, "seerattn_compactattn_disable_first_chunk_dense", False)
        )
        # Auto dense bypass: skip sparse path when kv_len is too short for
        # the compactattn overhead to pay off.  0 = disabled (original behavior).
        self.compactattn_auto_dense_kv_threshold = int(
            getattr(config, "seerattn_compactattn_auto_dense_kv_threshold", 0)
        )
        raw_env = os.environ.get("SEERATTN_COMPACTATTN_AUTO_DENSE_KV_THRESHOLD", None)
        if raw_env is not None:
            try:
                self.compactattn_auto_dense_kv_threshold = int(raw_env)
            except ValueError:
                pass
        self._compactattn_last_stats = None
        self._compactattn_last_train_loss_stats = None
        self._compactattn_threshold_calibration_callback = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        block_position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        collect_stats = bool(self.compactattn_debug)
        input_shape = hidden_states.shape[:-1]
        q_len = hidden_states.shape[1]
        _note_stage_begin(self, "qkv_proj")
        (query_states, key_states, value_states), qkv_proj_ms = _cuda_elapsed_ms(
            lambda: (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ),
            enabled=collect_stats,
        )
        _note_stage_done(self, "qkv_proj", {"qkv_proj_ms": float(qkv_proj_ms)})

        query_states = rearrange(query_states, "... (h d) -> ... h d", d=self.head_dim)
        key_states = rearrange(key_states, "... (h d) -> ... h d", d=self.head_dim)
        value_states = rearrange(value_states, "... (h d) -> ... h d", d=self.head_dim)

        # Qwen3 applies per-head RMSNorm on Q/K before RoPE. Reusing the Llama
        # chunked-dense forward without this step shifts the gate/attention path.
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if self.config.use_flash_rope:
            query_states_nope = query_states.clone()
            key_states_nope = key_states.clone()
        else:
            query_states_nope = query_states
            key_states_nope = key_states

        cos, sin = position_embeddings

        def _run_rope():
            if self.config.use_flash_rope:
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

        _note_stage_begin(self, "rope")
        (query_states, key_states), rope_ms = _cuda_elapsed_ms(
            _run_rope, enabled=collect_stats
        )
        _note_stage_done(self, "rope", {"rope_ms": float(rope_ms)})

        cache_update_ms = 0.0
        if past_key_value is not None:

            def _run_cache_update():
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states_out, value_states_out = past_key_value.update(
                    key_states.flatten(-2, -1),
                    value_states.flatten(-2, -1),
                    self.layer_idx,
                    cache_kwargs,
                )
                key_states_out = rearrange(key_states_out, "... (h d) -> ... h d", d=self.head_dim)
                value_states_out = rearrange(value_states_out, "... (h d) -> ... h d", d=self.head_dim)
                return key_states_out, value_states_out

            _note_stage_begin(self, "cache_update")
            (key_states, value_states), cache_update_ms = _cuda_elapsed_ms(
                _run_cache_update, enabled=collect_stats
            )
            _note_stage_done(self, "cache_update", {"cache_update_ms": float(cache_update_ms)})

        kv_len = key_states.shape[1]
        # Auto dense bypass: when kv_len is short, the per-chunk compactattn
        # overhead (gate + mask + paged build) exceeds the savings from
        # sparse execution.  Force dense to avoid the regression.
        auto_dense_bypass = (
            (not self.training)
            and (q_len > 1)
            and self.compactattn_auto_dense_kv_threshold > 0
            and kv_len <= self.compactattn_auto_dense_kv_threshold
        )
        force_dense_prefill = (
            (not self.training)
            and (q_len > 1)
            and (bool(getattr(self.config, "seerattn_chunked_prefill_force_dense", False))
                 or auto_dense_bypass)
        )
        chunked_prefill = (not self.training) and (q_len > 1) and (kv_len > q_len) and (not auto_dense_bypass)
        full_prefill_selected = (
            (not self.training)
            and (q_len > 1)
            and (kv_len == q_len)
            and self.compactattn_disable_first_chunk_dense
        )
        full_prefill_dense = (
            (not self.training)
            and (q_len > 1)
            and (kv_len == q_len)
            and (not self.compactattn_disable_first_chunk_dense)
        )
        _note_stage_done(
            self,
            "path_flags",
            {
                "q_len": float(q_len),
                "kv_len": float(kv_len),
                "force_dense_prefill": float(force_dense_prefill),
                "chunked_prefill": float(chunked_prefill),
                "full_prefill_selected": float(full_prefill_selected),
                "full_prefill_dense": float(full_prefill_dense),
            },
        )
        path_label = "other"
        compactattn_stats = None

        if force_dense_prefill:
            _note_stage_begin(self, "gate_compute")
            attn_gate_output = None
            self._append_compactattn_chunked_gate_key_cache(
                key_states_nope=key_states_nope,
                past_key_value=past_key_value,
                block_position_embeddings=block_position_embeddings,
                cache_position=cache_position,
            )
            _note_stage_done(self, "gate_compute")
        elif chunked_prefill:
            if collect_stats:
                setattr(self.attn_gate, "__seer_layer_idx", int(self.layer_idx))
                setattr(
                    self.attn_gate,
                    "__seer_chunk_idx",
                    int((kv_len // q_len) - 1) if q_len > 0 else -1,
                )
            _note_stage_begin(self, "gate_compute")
            attn_gate_output = None
            if self.compactattn_use_chunked_gate_cache:
                attn_gate_output = self._compute_chunked_gate_from_cache(
                    query_states_nope=query_states_nope,
                    key_states_nope=key_states_nope,
                    past_key_value=past_key_value,
                    block_attention_mask=block_attention_mask,
                    block_position_embeddings=block_position_embeddings,
                    cache_position=cache_position,
                    use_softmax=self.config.seerattn_sparsity_method == "threshold",
                )
            if attn_gate_output is None:
                attn_gate_output = self._compute_compactattn_gate_output(
                    query_states_nope,
                    key_states_nope,
                    block_attention_mask,
                    block_position_embeddings,
                    use_softmax=self.config.seerattn_sparsity_method == "threshold",
                )
                if self.compactattn_use_chunked_gate_cache:
                    self._append_compactattn_chunked_gate_key_cache(
                        key_states_nope=key_states_nope,
                        past_key_value=past_key_value,
                        block_position_embeddings=block_position_embeddings,
                        cache_position=cache_position,
                    )
            _note_stage_done(self, "gate_compute")
        elif full_prefill_selected:
            if collect_stats:
                setattr(self.attn_gate, "__seer_layer_idx", int(self.layer_idx))
                setattr(self.attn_gate, "__seer_chunk_idx", 0)
            _note_stage_begin(self, "gate_compute")
            attn_gate_output = self._compute_compactattn_gate_output(
                query_states_nope,
                key_states_nope,
                block_attention_mask,
                block_position_embeddings,
                use_softmax=not self.training and self.config.seerattn_sparsity_method == "threshold",
            )
            self._append_compactattn_chunked_gate_key_cache(
                key_states_nope=key_states_nope,
                past_key_value=past_key_value,
                block_position_embeddings=block_position_embeddings,
                cache_position=cache_position,
            )
            _note_stage_done(self, "gate_compute")
        elif full_prefill_dense:
            _note_stage_begin(self, "gate_compute")
            attn_gate_output = None
            self._append_compactattn_chunked_gate_key_cache(
                key_states_nope=key_states_nope,
                past_key_value=past_key_value,
                block_position_embeddings=block_position_embeddings,
                cache_position=cache_position,
            )
            _note_stage_done(self, "gate_compute")
        else:
            if collect_stats:
                setattr(self.attn_gate, "__seer_layer_idx", int(self.layer_idx))
                setattr(self.attn_gate, "__seer_chunk_idx", -1)
            _note_stage_begin(self, "gate_compute")
            attn_gate_output = self._compute_compactattn_gate_output(
                query_states_nope,
                key_states_nope,
                block_attention_mask,
                block_position_embeddings,
                use_softmax=not self.training and self.config.seerattn_sparsity_method == "threshold",
            )
            self._append_compactattn_chunked_gate_key_cache(
                key_states_nope=key_states_nope,
                past_key_value=past_key_value,
                block_position_embeddings=block_position_embeddings,
                cache_position=cache_position,
            )
            _note_stage_done(self, "gate_compute")

        if self.training:
            attn_output, ground_truth_mask = attention_distill_forward(
                query_states,
                key_states,
                value_states,
                softmax_scale=self.scaling,
                block_size=self.config.seerattn_gate_block_size,
                num_key_value_groups=self.num_key_value_groups,
                kv_group_aware_query=self.compactattn_kv_group_aware_gate,
            )
            self._maybe_record_compactattn_threshold_calibration(
                attn_gate_output,
                ground_truth_mask,
                block_attention_mask,
                query_states,
                key_states,
                q_len=q_len,
                kv_len=kv_len,
            )
        else:
            if force_dense_prefill:
                _note_stage_begin(self, "attn_dispatch")
                attn_output, dense_stats = _dense_prefill_full_kv(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    attention_mask=attention_mask,
                    softmax_scale=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    fallback_used=0.0,
                    measure_timing=collect_stats,
                    attn_module=self,
                )
                _note_stage_done(self, "attn_dispatch", {"attn_path_dense": 1.0})
                path_label = "forced_dense"
                if collect_stats:
                    compactattn_stats = {
                        "dense_kernel_ms": float(dense_stats.get("dense_kernel_ms", 0.0)),
                        "gather_pack_ms": float(dense_stats.get("gather_pack_ms", 0.0)),
                        "fallback_used": 0.0,
                        "selected_blocks": 0.0,
                        "valid_blocks": 0.0,
                        "selected_tokens": 0.0,
                        "valid_tokens": 0.0,
                        "repeat_kv_ms": float(dense_stats.get("repeat_kv_ms", 0.0)),
                        "upad_input_ms": float(dense_stats.get("upad_input_ms", 0.0)),
                        "pad_output_ms": float(dense_stats.get("pad_output_ms", 0.0)),
                        "path": path_label,
                        "compactattn_version": self.compactattn_version,
                    }
                else:
                    compactattn_stats = None
            elif chunked_prefill or full_prefill_selected:
                _note_stage_begin(self, "attn_dispatch")
                attn_output, compactattn_stats = chunked_prefill_column_dense_attention_forward(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        attention_mask=attention_mask,
                        query_length=q_len,
                        softmax_scale=self.scaling,
                        attn_gate_score=attn_gate_output,
                        block_attention_mask=block_attention_mask,
                        threshold=self._resolve_compactattn_threshold(kv_len),
                        keep_recent_blocks=self.compactattn_keep_recent_blocks,
                        block_size=self.config.seerattn_gate_block_size,
                        num_key_value_groups=self.num_key_value_groups,
                        attn_gate_is_kv_group_aware=self.compactattn_kv_group_aware_gate,
                        pack_impl=self.compactattn_pack_impl,
                        indexed_impl=self.compactattn_indexed_impl,
                        cache_fill_backend=self.compactattn_cache_fill_backend,
                        return_stats=collect_stats,
                        attn_module=self,
                    )
                if collect_stats and isinstance(compactattn_stats, dict):
                    path_label = str(
                        compactattn_stats.get(
                            "path",
                            "first_chunk_selected" if full_prefill_selected else "chunked_selected",
                        )
                    )
                else:
                    path_label = "first_chunk_selected" if full_prefill_selected else "chunked_selected"
                _note_stage_done(self, "attn_dispatch", {"attn_path_selected": 1.0})
            elif full_prefill_dense:
                _note_stage_begin(self, "attn_dispatch")
                attn_output, dense_stats = _dense_prefill_full_kv(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    attention_mask=attention_mask,
                    softmax_scale=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    fallback_used=0.0,
                    measure_timing=collect_stats,
                    attn_module=self,
                )
                _note_stage_done(self, "attn_dispatch", {"attn_path_dense": 1.0})
                path_label = "first_chunk_dense"
                if collect_stats:
                    compactattn_stats = {
                        "dense_kernel_ms": float(dense_stats.get("dense_kernel_ms", 0.0)),
                        "gather_pack_ms": float(dense_stats.get("gather_pack_ms", 0.0)),
                        "fallback_used": 0.0,
                        "selected_blocks": 0.0,
                        "valid_blocks": 0.0,
                        "selected_tokens": 0.0,
                        "valid_tokens": 0.0,
                        "repeat_kv_ms": float(dense_stats.get("repeat_kv_ms", 0.0)),
                        "upad_input_ms": float(dense_stats.get("upad_input_ms", 0.0)),
                        "pad_output_ms": float(dense_stats.get("pad_output_ms", 0.0)),
                        "path": path_label,
                        "compactattn_version": self.compactattn_version,
                    }
                else:
                    compactattn_stats = None
            else:
                _note_stage_begin(self, "attn_dispatch")
                sparse_attn_gate_output = self._expand_compactattn_gate_scores_for_q_heads(attn_gate_output)
                attn_output = sparse_flash_attention_forward(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    query_length=q_len,
                    softmax_scale=self.scaling,
                    attn_gate_score=sparse_attn_gate_output,
                    sparsity_method=self.config.seerattn_sparsity_method,
                    threshold=self.config.seerattn_threshold,
                    nz_ratio=self.config.seerattn_nz_ratio,
                    last_block_dense=self.config.seerattn_last_block_dense,
                    block_size=self.config.seerattn_gate_block_size,
                    num_key_value_groups=self.num_key_value_groups,
                    profile_file=self.profile_file,
                    block_attention_mask=block_attention_mask,
                )
                _note_stage_done(self, "attn_dispatch", {"attn_path_sparse": 1.0})
                path_label = "other"
                compactattn_stats = {"path": path_label} if collect_stats else None

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        _note_stage_begin(self, "o_proj")
        attn_output, o_proj_ms = _cuda_elapsed_ms(
            lambda: self.o_proj(attn_output), enabled=collect_stats
        )
        _note_stage_done(self, "o_proj", {"o_proj_ms": float(o_proj_ms)})

        if self.training:
            self._compactattn_last_stats = None
            mask_loss, attn_gate_output, ground_truth_mask = self._compute_compactattn_mask_loss(
                attn_gate_output=attn_gate_output,
                ground_truth_mask=ground_truth_mask,
                block_attention_mask=block_attention_mask,
            )
        else:
            mask_loss = 0.0
            attn_gate_output = None
            ground_truth_mask = None
            self._compactattn_last_train_loss_stats = None
            if collect_stats:
                merged_stats = {}
                if isinstance(compactattn_stats, dict):
                    merged_stats.update(compactattn_stats)
                merged_stats.update(
                    {
                        "qkv_proj_ms": float(qkv_proj_ms),
                        "rope_ms": float(rope_ms),
                        "cache_update_ms": float(cache_update_ms),
                        "o_proj_ms": float(o_proj_ms),
                        "path": path_label,
                        "col_first_chunk_calls": 1.0 if path_label == "first_chunk_dense" else 0.0,
                        "col_selected_path_calls": 1.0 if path_label == "chunked_selected" else 0.0,
                        "compactattn_version": self.compactattn_version,
                    }
                )
                self._compactattn_last_stats = merged_stats
            else:
                self._compactattn_last_stats = None

        if not kwargs.get("output_attentions", False):
            attn_gate_output = None
            ground_truth_mask = None
        return attn_output, mask_loss, None, attn_gate_output, ground_truth_mask


class SeerAttnQwen3ChunkedDenseDecoderLayer(SeerAttnQwen3DecoderLayer):
    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = Qwen3SeerAttentionChunkedDense(config=config, layer_idx=layer_idx)


class SeerAttnQwen3ChunkedDenseModel(SeerAttnQwen3Model):
    def __init__(self, config: SeerAttnQwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                SeerAttnQwen3ChunkedDenseDecoderLayer(config=config, layer_idx=layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()


class SeerAttnQwen3ChunkedDenseForCausalLM(SeerAttnQwen3ForCausalLM):
    _no_split_modules = [
        *SeerAttnQwen3ForCausalLM._no_split_modules,
        "SeerAttnQwen3ChunkedDenseDecoderLayer",
    ]

    def __init__(self, config: SeerAttnQwen3Config):
        super().__init__(config)
        if not hasattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks"):
            setattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks", 2)
        if not hasattr(config, "seerattn_compactattn_release_indexed_workspaces"):
            setattr(config, "seerattn_compactattn_release_indexed_workspaces", True)
        self.model = SeerAttnQwen3ChunkedDenseModel(config)

    def forward(self, *args, **kwargs):
        try:
            return super().forward(*args, **kwargs)
        finally:
            if bool(getattr(self.config, "seerattn_compactattn_release_indexed_workspaces", False)):
                _clear_indexed_dense_workspaces_for_model(self)
            clear_fast_path_content_cache()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, load_gate=True, *model_args, **kwargs):
        force_dense_prefill = kwargs.pop("seerattn_chunked_prefill_force_dense", False)
        compactattn_threshold = kwargs.pop("seerattn_compactattn_threshold", None)
        kwargs.pop("seerattn_compactattn_chunked_gate_head_pool", None)  # removed feature, ignore stale config keys
        compactattn_use_chunked_gate_cache = kwargs.pop("seerattn_compactattn_use_chunked_gate_cache", True)
        compactattn_keep_recent_blocks = kwargs.pop("seerattn_compactattn_keep_recent_blocks", 2)
        compactattn_kv_group_aware_gate = kwargs.pop("seerattn_compactattn_kv_group_aware_gate", None)
        compactattn_adjacent_align_lambda = kwargs.pop("seerattn_compactattn_adjacent_align_lambda", None)
        compactattn_reinit_q_branch_on_load = kwargs.pop(
            "seerattn_compactattn_reinit_q_branch_on_load", None
        )
        compactattn_pack_impl = kwargs.pop("seerattn_compactattn_pack_impl", "indexed_dense")
        compactattn_indexed_impl = kwargs.pop("seerattn_compactattn_indexed_impl", "fa2_paged")
        compactattn_cache_fill_backend = kwargs.pop("seerattn_compactattn_cache_fill_backend", "auto")
        compactattn_debug = kwargs.pop("seerattn_compactattn_debug", False)
        compactattn_release_indexed_workspaces = kwargs.pop(
            "seerattn_compactattn_release_indexed_workspaces", True
        )
        compactattn_disable_first_chunk_dense = kwargs.pop(
            "seerattn_compactattn_disable_first_chunk_dense", False
        )
        final_dense_tail_blocks = kwargs.pop("seerattn_chunked_prefill_final_dense_tail_blocks", 2)
        input_config = kwargs.pop("config", None)

        def _coerce_qwen3_config(base_config, *, base_model_name):
            if isinstance(base_config, SeerAttnQwen3Config):
                config = base_config
            else:
                config = SeerAttnQwen3Config(**base_config.to_dict())
            _sync_qwen3_base_vocab_size(config, base_model_name, kwargs)
            setattr(config, "seerattn_chunked_prefill_force_dense", bool(force_dense_prefill))
            setattr(
                config,
                "seerattn_chunked_prefill_final_dense_tail_blocks",
                int(final_dense_tail_blocks),
            )
            if compactattn_threshold is None:
                effective_threshold = config.seerattn_threshold
            else:
                effective_threshold = float(compactattn_threshold)
            setattr(config, "seerattn_compactattn_threshold", float(effective_threshold))
            setattr(config, "seerattn_compactattn_use_chunked_gate_cache", bool(compactattn_use_chunked_gate_cache))
            setattr(config, "seerattn_compactattn_keep_recent_blocks", int(compactattn_keep_recent_blocks))
            setattr(config, "seerattn_compactattn_pack_impl", str(compactattn_pack_impl))
            setattr(config, "seerattn_compactattn_indexed_impl", str(compactattn_indexed_impl))
            setattr(config, "seerattn_compactattn_cache_fill_backend", str(compactattn_cache_fill_backend))
            setattr(config, "seerattn_compactattn_debug", bool(compactattn_debug))
            setattr(
                config,
                "seerattn_compactattn_release_indexed_workspaces",
                bool(compactattn_release_indexed_workspaces),
            )
            setattr(
                config,
                "seerattn_compactattn_disable_first_chunk_dense",
                bool(compactattn_disable_first_chunk_dense),
            )
            return config

        if load_gate:
            config = input_config
            if config is None:
                config = SeerAttnQwen3Config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            base_model = getattr(config, "base_model", pretrained_model_name_or_path)
            config = _coerce_qwen3_config(config, base_model_name=base_model)
            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))
            if compactattn_kv_group_aware_gate is None:
                compactattn_kv_group_aware_gate = bool(
                    getattr(config, "seerattn_compactattn_kv_group_aware_gate", False)
                )
            if compactattn_adjacent_align_lambda is None:
                compactattn_adjacent_align_lambda = float(
                    getattr(config, "seerattn_compactattn_adjacent_align_lambda", 1e-3)
                )
            if compactattn_reinit_q_branch_on_load is None:
                compactattn_reinit_q_branch_on_load = bool(
                    getattr(config, "seerattn_compactattn_reinit_q_branch_on_load", False)
                )
            setattr(
                config,
                "seerattn_compactattn_kv_group_aware_gate",
                bool(compactattn_kv_group_aware_gate),
            )
            setattr(
                config,
                "seerattn_compactattn_adjacent_align_lambda",
                float(compactattn_adjacent_align_lambda),
            )
            setattr(
                config,
                "seerattn_compactattn_reinit_q_branch_on_load",
                bool(compactattn_reinit_q_branch_on_load),
            )
            kwargs["device_map"] = _resolve_qwen_seer_device_map(config, kwargs.get("device_map"))

            model = super(SeerAttnQwen2ForCausalLM, cls).from_pretrained(
                base_model, config=config, *model_args, **kwargs
            )

            if os.path.exists(pretrained_model_name_or_path):
                gate_weights = torch.load(
                    os.path.join(pretrained_model_name_or_path, "attn_gate_weights.pth")
                )
            else:
                try:
                    gate_weights = torch.load(
                        hf_hub_download(
                            repo_id=pretrained_model_name_or_path, filename="attn_gate_weights.pth"
                        )
                    )
                except Exception as exc:
                    raise ValueError("Could not load the attention gate weights.") from exc

            _load_gate_weights_with_optional_q_reinit(
                model,
                gate_weights,
                reinit_query_branch=bool(compactattn_reinit_q_branch_on_load),
            )
            print("Attention gate weights loaded successfully.")
        else:
            config = input_config
            if config is None:
                config = SeerAttnQwen3Config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            base_model = getattr(config, "base_model", pretrained_model_name_or_path)
            config = _coerce_qwen3_config(config, base_model_name=base_model)
            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))
            kwargs["device_map"] = _resolve_qwen_seer_device_map(config, kwargs.get("device_map"))

            model = super(SeerAttnQwen2ForCausalLM, cls).from_pretrained(
                base_model, config=config, *model_args, **kwargs
            )
            if compactattn_kv_group_aware_gate is not None:
                setattr(
                    model.config,
                    "seerattn_compactattn_kv_group_aware_gate",
                    bool(compactattn_kv_group_aware_gate),
                )
            if compactattn_adjacent_align_lambda is not None:
                setattr(
                    model.config,
                    "seerattn_compactattn_adjacent_align_lambda",
                    float(compactattn_adjacent_align_lambda),
                )
            if compactattn_reinit_q_branch_on_load is not None:
                setattr(
                    model.config,
                    "seerattn_compactattn_reinit_q_branch_on_load",
                    bool(compactattn_reinit_q_branch_on_load),
                )
            for layer in model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_use_chunked_gate_cache"):
                    layer.self_attn.compactattn_use_chunked_gate_cache = bool(compactattn_use_chunked_gate_cache)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_kv_group_aware_gate"):
                    layer.self_attn.compactattn_kv_group_aware_gate = bool(
                        getattr(model.config, "seerattn_compactattn_kv_group_aware_gate", False)
                    )
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_adjacent_align_lambda"):
                    layer.self_attn.compactattn_adjacent_align_lambda = float(
                        getattr(model.config, "seerattn_compactattn_adjacent_align_lambda", 1e-3)
                    )
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_pack_impl"):
                    layer.self_attn.compactattn_pack_impl = str(compactattn_pack_impl)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_indexed_impl"):
                    layer.self_attn.compactattn_indexed_impl = str(compactattn_indexed_impl)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_release_indexed_workspaces"):
                    layer.self_attn.compactattn_release_indexed_workspaces = bool(
                        compactattn_release_indexed_workspaces
                    )
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_cache_fill_backend"):
                    layer.self_attn.compactattn_cache_fill_backend = str(compactattn_cache_fill_backend)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_disable_first_chunk_dense"):
                    layer.self_attn.compactattn_disable_first_chunk_dense = bool(
                        compactattn_disable_first_chunk_dense
                    )
        return model


__all__ = ["SeerAttnQwen3ChunkedDenseForCausalLM"]
