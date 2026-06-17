import os

import torch
from torch import nn
from huggingface_hub import hf_hub_download

from compact_attn.kernels.varlen.indexed_dense_prefill_varlen import clear_indexed_dense_workspaces
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn import LlamaSeerAttention
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn_chunked_dense import (
    COMPACTATTN_VERSION,
    LlamaSeerAttentionChunkedDense,
    _clear_indexed_dense_workspaces_for_model,
    _load_gate_weights_with_optional_q_reinit,
)
from compact_attn.prefill_sparse.qwen.configuration_qwen2_seerattn import SeerAttnQwen2Config
from compact_attn.prefill_sparse.qwen.modeling_qwen2_seerattn import (
    SeerAttnQwen2Attention,
    SeerAttnQwen2DecoderLayer,
    SeerAttnQwen2ForCausalLM,
    SeerAttnQwen2Model,
    _resolve_qwen_seer_device_map,
    _sync_qwen_base_vocab_size,
)


class Qwen2SeerAttentionChunkedDense(SeerAttnQwen2Attention):
    _can_use_chunked_gate_cache = LlamaSeerAttention._can_use_chunked_gate_cache
    _build_chunked_gate_blocks = LlamaSeerAttention._build_chunked_gate_blocks

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
    forward = LlamaSeerAttentionChunkedDense.forward

    def __init__(self, config: SeerAttnQwen2Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.compactattn_kv_group_aware_gate = bool(
            getattr(config, "seerattn_compactattn_kv_group_aware_gate", False)
        )
        self.compactattn_threshold = float(
            getattr(config, "seerattn_compactattn_threshold", config.seerattn_threshold)
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


class SeerAttnQwen2ChunkedDenseDecoderLayer(SeerAttnQwen2DecoderLayer):
    def __init__(self, config: SeerAttnQwen2Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = Qwen2SeerAttentionChunkedDense(config=config, layer_idx=layer_idx)


class SeerAttnQwen2ChunkedDenseModel(SeerAttnQwen2Model):
    def __init__(self, config: SeerAttnQwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                SeerAttnQwen2ChunkedDenseDecoderLayer(config=config, layer_idx=layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()


class SeerAttnQwen2ChunkedDenseForCausalLM(SeerAttnQwen2ForCausalLM):
    _no_split_modules = [
        *SeerAttnQwen2ForCausalLM._no_split_modules,
        "SeerAttnQwen2ChunkedDenseDecoderLayer",
    ]

    def __init__(self, config: SeerAttnQwen2Config):
        super().__init__(config)
        if not hasattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks"):
            setattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks", 2)
        if not hasattr(config, "seerattn_compactattn_release_indexed_workspaces"):
            setattr(config, "seerattn_compactattn_release_indexed_workspaces", True)
        self.model = SeerAttnQwen2ChunkedDenseModel(config)

    def forward(self, *args, **kwargs):
        try:
            return super().forward(*args, **kwargs)
        finally:
            if bool(getattr(self.config, "seerattn_compactattn_release_indexed_workspaces", False)):
                _clear_indexed_dense_workspaces_for_model(self)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, load_gate=True, *model_args, **kwargs):
        force_dense_prefill = kwargs.pop("seerattn_chunked_prefill_force_dense", False)
        compactattn_threshold = kwargs.pop("seerattn_compactattn_threshold", None)
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

        if load_gate:
            config = SeerAttnQwen2Config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            base_model = config.base_model
            _sync_qwen_base_vocab_size(config, base_model, kwargs)

            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))

            if compactattn_threshold is None:
                compactattn_threshold = config.seerattn_threshold
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
            setattr(config, "seerattn_compactattn_threshold", float(compactattn_threshold))
            setattr(config, "seerattn_compactattn_use_chunked_gate_cache", bool(compactattn_use_chunked_gate_cache))
            setattr(config, "seerattn_compactattn_keep_recent_blocks", int(compactattn_keep_recent_blocks))
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
            setattr(
                config,
                "seerattn_chunked_prefill_final_dense_tail_blocks",
                int(final_dense_tail_blocks),
            )
            setattr(config, "seerattn_chunked_prefill_force_dense", bool(force_dense_prefill))
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
            kwargs["device_map"] = _resolve_qwen_seer_device_map(
                kwargs.get("config", None), kwargs.get("device_map")
            )
            model = super(SeerAttnQwen2ForCausalLM, cls).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
            setattr(model.config, "seerattn_compactattn_pack_impl", str(compactattn_pack_impl))
            setattr(model.config, "seerattn_compactattn_indexed_impl", str(compactattn_indexed_impl))
            setattr(model.config, "seerattn_compactattn_cache_fill_backend", str(compactattn_cache_fill_backend))
            setattr(model.config, "seerattn_compactattn_use_chunked_gate_cache", bool(compactattn_use_chunked_gate_cache))
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
            setattr(
                model.config,
                "seerattn_compactattn_release_indexed_workspaces",
                bool(compactattn_release_indexed_workspaces),
            )
            setattr(
                model.config,
                "seerattn_compactattn_disable_first_chunk_dense",
                bool(compactattn_disable_first_chunk_dense),
            )
            setattr(model.config, "seerattn_chunked_prefill_force_dense", bool(force_dense_prefill))
            setattr(
                model.config,
                "seerattn_chunked_prefill_final_dense_tail_blocks",
                int(final_dense_tail_blocks),
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
