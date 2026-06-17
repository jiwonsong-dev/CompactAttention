"""Load stock LLaMA models with method-specific attention modules replaced.

This is a profiling harness: the decoder layer / norm / MLP stack stays the
stock HuggingFace implementation, while each layer's ``self_attn`` module is
replaced with the requested sparse/compact attention implementation.  It avoids
comparing full custom model classes against stock dense classes when the goal is
attention speedup under a shared outer model convention.
"""
from __future__ import annotations

import copy
import os
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from types import MethodType

from torch import nn
from transformers import LlamaForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from compact_attn.prefill_sparse.llama.configuration_llama_seerattn import SeerAttnLlamaConfig
from compact_attn.prefill_sparse.llama.modeling_llama_flashprefill import (
    LlamaFlashPrefillBlockSparseAttention,
)
from compact_attn.prefill_sparse.llama.modeling_llama_flashprefill_compactattn import (
    LlamaFlashPrefillCompactAttnAttention,
)
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn import LlamaSeerAttention
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn_chunked_dense_hf import (
    LlamaSeerAttentionChunkedDenseHF,
)
from compact_attn.prefill_sparse.qwen3_moe.modeling_qwen3_moe_flashprefill import (
    Qwen3MoeFlashPrefillAttention,
)
from compact_attn.load_quoka import load_quoka_qwen3_moe_model


_SEER_RUNTIME_ATTRS = (
    "seerattn_threshold",
    "seerattn_sparsity_method",
    "seerattn_nz_ratio",
    "seerattn_last_block_dense",
    "seerattn_gate_type",
    "seerattn_gate_block_size",
    "seerattn_gate_hidden_size",
    "seerattn_gate_force_double",
    "seerattn_use_chunked_gate_cache",
    "seerattn_compactattn_threshold",
    "seerattn_compactattn_threshold_schedule",
    "seerattn_compactattn_keep_recent_blocks",
    "seerattn_compactattn_disable_first_chunk_dense",
    "seerattn_compactattn_chunked_gate_head_pool",
    "seerattn_compactattn_kv_group_aware_gate",
    "seerattn_compactattn_pack_impl",
    "seerattn_compactattn_indexed_impl",
    "seerattn_compactattn_cache_fill_backend",
    "seerattn_compactattn_release_indexed_workspaces",
    "seerattn_chunked_prefill_force_dense",
    "seerattn_chunked_prefill_final_dense_tail_blocks",
    "seerattn_dense_backend",
    "seerattn_flashprefill_alpha",
    "seerattn_flashprefill_block_size",
    "seerattn_flashprefill_attention_sink",
    "seerattn_flashprefill_window_size",
    "seerattn_flashprefill_last_n_block",
    "seerattn_flashprefill_min_budget",
    "seerattn_defer_async_collective_wait",
    "use_flash_rope",
)


def _adapt_attention_return_contract(attn: nn.Module) -> None:
    """Keep params under ``self_attn.*`` while returning HF's 2-tuple shape."""

    if hasattr(attn, "_seer_replacement_original_forward"):
        return
    original_forward = attn.forward
    attn._seer_replacement_original_forward = original_forward

    def _forward(self, *args, **kwargs):
        out = self._seer_replacement_original_forward(*args, **kwargs)
        if isinstance(out, tuple):
            attn_output = out[0]
            attn_weights = out[2] if len(out) > 2 else None
        else:
            attn_output = out
            attn_weights = None
        return attn_output, attn_weights

    attn.forward = MethodType(_forward, attn)


def _reuse_attention_projections(dst: nn.Module, src: nn.Module) -> None:
    """Reuse stock projection modules so TP DTensor sharding is preserved."""
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def _copy_seer_attrs_to_stock_config(stock_config, seer_config: SeerAttnLlamaConfig) -> None:
    for name in _SEER_RUNTIME_ATTRS:
        if hasattr(seer_config, name):
            setattr(stock_config, name, getattr(seer_config, name))


def _build_seer_config(
    model_id: str,
    base_model: str,
    *,
    variant: str,
    dense_backend: str,
    threshold: float,
    last_block_dense: bool,
    final_dense_tail_blocks: int,
    compactattn_keep_recent_blocks: int,
    compactattn_disable_first_chunk_dense: bool,
    compactattn_chunked_gate_head_pool: str,
    col_pack_impl: str,
    col_indexed_impl: str,
    col_cache_fill_backend: str,
    flashprefill_alpha: float,
    flashprefill_block_size: int,
    flashprefill_attention_sink: int,
    flashprefill_window_size: int,
    flashprefill_last_n_block: int,
    flashprefill_min_budget: int,
) -> SeerAttnLlamaConfig:
    cfg_source = model_id if variant in {"seer_block_sparse", "seer_compactattn", "seer_compactattn_hf"} else base_model
    cfg = SeerAttnLlamaConfig.from_pretrained(cfg_source)
    setattr(cfg, "base_model", base_model)
    setattr(cfg, "use_cache", True)
    setattr(cfg, "seerattn_dense_backend", str(dense_backend))
    setattr(cfg, "seerattn_sparsity_method", "threshold")
    setattr(cfg, "seerattn_threshold", float(threshold))
    setattr(cfg, "seerattn_last_block_dense", bool(last_block_dense))
    setattr(cfg, "seerattn_chunked_prefill_force_dense", False)
    setattr(cfg, "seerattn_chunked_prefill_final_dense_tail_blocks", int(final_dense_tail_blocks))
    setattr(cfg, "seerattn_compactattn_threshold", float(threshold))
    setattr(cfg, "seerattn_compactattn_threshold_schedule", None)
    setattr(cfg, "seerattn_compactattn_keep_recent_blocks", int(compactattn_keep_recent_blocks))
    setattr(cfg, "seerattn_compactattn_disable_first_chunk_dense", bool(compactattn_disable_first_chunk_dense))
    setattr(cfg, "seerattn_compactattn_chunked_gate_head_pool", str(compactattn_chunked_gate_head_pool))
    setattr(cfg, "seerattn_compactattn_pack_impl", str(col_pack_impl))
    setattr(cfg, "seerattn_compactattn_indexed_impl", str(col_indexed_impl))
    setattr(cfg, "seerattn_compactattn_cache_fill_backend", str(col_cache_fill_backend))
    setattr(cfg, "seerattn_flashprefill_alpha", float(flashprefill_alpha))
    setattr(cfg, "seerattn_flashprefill_block_size", int(flashprefill_block_size))
    setattr(cfg, "seerattn_gate_block_size", int(flashprefill_block_size))
    setattr(cfg, "seerattn_flashprefill_attention_sink", int(flashprefill_attention_sink))
    setattr(cfg, "seerattn_flashprefill_window_size", int(flashprefill_window_size))
    setattr(cfg, "seerattn_flashprefill_last_n_block", int(flashprefill_last_n_block))
    setattr(cfg, "seerattn_flashprefill_min_budget", int(flashprefill_min_budget))
    setattr(cfg, "seerattn_defer_async_collective_wait", bool(variant.startswith("flashprefill")))
    if variant.startswith("flashprefill"):
        setattr(cfg, "seerattn_gate_block_size", int(flashprefill_block_size))
    return cfg


def _build_qwen3_moe_flashprefill_config(
    base_model: str,
    *,
    variant: str,
    dense_backend: str,
    flashprefill_alpha: float,
    flashprefill_block_size: int,
    flashprefill_attention_sink: int,
    flashprefill_window_size: int,
    flashprefill_last_n_block: int,
    flashprefill_min_budget: int,
    col_pack_impl: str,
    col_indexed_impl: str,
    col_cache_fill_backend: str,
    final_dense_tail_blocks: int,
    qwen_long_context: bool = False,
    qwen_long_context_max_position_embeddings: int = 131072,
    qwen_yarn_factor: float = 4.0,
    qwen_original_max_position_embeddings: int = 32768,
) -> Qwen3MoeConfig:
    if variant not in {"flashprefill_block_sparse", "flashprefill_compactattn"}:
        raise ValueError(f"Unsupported Qwen3-MoE replacement variant: {variant}")
    cfg = Qwen3MoeConfig.from_pretrained(base_model)
    if qwen_long_context:
        cfg.max_position_embeddings = int(qwen_long_context_max_position_embeddings)
        cfg.rope_scaling = {
            "rope_type": "yarn",
            "factor": float(qwen_yarn_factor),
            "original_max_position_embeddings": int(qwen_original_max_position_embeddings),
        }
    setattr(cfg, "base_model", base_model)
    setattr(cfg, "use_cache", True)
    setattr(cfg, "seerattn_dense_backend", str(dense_backend))
    setattr(
        cfg,
        "seerattn_flashprefill_execution_mode",
        "compactattn" if variant == "flashprefill_compactattn" else "block_sparse",
    )
    setattr(cfg, "seerattn_flashprefill_alpha", float(flashprefill_alpha))
    setattr(cfg, "seerattn_flashprefill_block_size", int(flashprefill_block_size))
    setattr(cfg, "seerattn_flashprefill_attention_sink", int(flashprefill_attention_sink))
    setattr(cfg, "seerattn_flashprefill_window_size", int(flashprefill_window_size))
    setattr(cfg, "seerattn_flashprefill_last_n_block", int(flashprefill_last_n_block))
    setattr(cfg, "seerattn_flashprefill_min_budget", int(flashprefill_min_budget))
    setattr(cfg, "seerattn_compactattn_pack_impl", str(col_pack_impl))
    setattr(cfg, "seerattn_compactattn_indexed_impl", str(col_indexed_impl))
    setattr(cfg, "seerattn_compactattn_cache_fill_backend", str(col_cache_fill_backend))
    setattr(cfg, "seerattn_chunked_prefill_force_dense", False)
    setattr(cfg, "seerattn_chunked_prefill_final_dense_tail_blocks", int(final_dense_tail_blocks))
    setattr(cfg, "seerattn_defer_async_collective_wait", True)
    return cfg


def _mark_qwen3_moe_replacement_model(model: nn.Module, *, variant: str) -> nn.Module:
    """Annotate delegated Qwen3-MoE loaders as replacement-harness models."""

    setattr(model.config, "seerattn_attention_harness", "replacement")
    setattr(model.config, "seerattn_replacement_needs_block_gate_metadata", False)
    if variant == "flashprefill_compactattn":
        setattr(model.config, "seerattn_cache_layout", "heads_first_flashprefill")
    else:
        setattr(model.config, "seerattn_cache_layout", "dynamic")
    return model


def _load_gate_weights(model: nn.Module, gate_model_id: str) -> None:
    if os.path.exists(gate_model_id):
        path = os.path.join(gate_model_id, "attn_gate_weights.pth")
    else:
        path = hf_hub_download(repo_id=gate_model_id, filename="attn_gate_weights.pth")
    gate_weights = torch.load(path, map_location="cpu")
    model.load_state_dict(gate_weights, strict=False)


def load_llama_attention_replacement_model(
    *,
    model_id: str,
    base_model: str,
    variant: str,
    torch_dtype: torch.dtype,
    device_map=None,
    tp_plan: Optional[str] = None,
    dense_backend: str = "flashinfer",
    threshold: float = 3e-4,
    last_block_dense: bool = False,
    final_dense_tail_blocks: int = 0,
    compactattn_keep_recent_blocks: int = 2,
    compactattn_disable_first_chunk_dense: bool = False,
    compactattn_chunked_gate_head_pool: str = "avg",
    col_pack_impl: str = "indexed_dense",
    col_indexed_impl: str = "fa2_paged",
    col_cache_fill_backend: str = "auto",
    flashprefill_alpha: float = 0.01,
    flashprefill_block_size: int = 128,
    flashprefill_attention_sink: int = 2,
    flashprefill_window_size: int = 4,
    flashprefill_last_n_block: int = 2,
    flashprefill_min_budget: int = 0,
) -> nn.Module:
    load_kwargs = dict(torch_dtype=torch_dtype, attn_implementation="flash_attention_2")
    if tp_plan is not None:
        load_kwargs["tp_plan"] = tp_plan
    else:
        load_kwargs["device_map"] = device_map
    model = LlamaForCausalLM.from_pretrained(base_model, **load_kwargs).eval()

    seer_cfg = _build_seer_config(
        model_id,
        base_model,
        variant=variant,
        dense_backend=dense_backend,
        threshold=threshold,
        last_block_dense=last_block_dense,
        final_dense_tail_blocks=final_dense_tail_blocks,
        compactattn_keep_recent_blocks=compactattn_keep_recent_blocks,
        compactattn_disable_first_chunk_dense=compactattn_disable_first_chunk_dense,
        compactattn_chunked_gate_head_pool=compactattn_chunked_gate_head_pool,
        col_pack_impl=col_pack_impl,
        col_indexed_impl=col_indexed_impl,
        col_cache_fill_backend=col_cache_fill_backend,
        flashprefill_alpha=flashprefill_alpha,
        flashprefill_block_size=flashprefill_block_size,
        flashprefill_attention_sink=flashprefill_attention_sink,
        flashprefill_window_size=flashprefill_window_size,
        flashprefill_last_n_block=flashprefill_last_n_block,
        flashprefill_min_budget=flashprefill_min_budget,
    )
    _copy_seer_attrs_to_stock_config(model.config, seer_cfg)

    for layer in model.model.layers:
        src_attn = layer.self_attn
        if variant == "seer_block_sparse":
            impl = LlamaSeerAttention(seer_cfg, layer_idx=src_attn.layer_idx)
        elif variant in {"seer_compactattn", "seer_compactattn_hf"}:
            impl = LlamaSeerAttentionChunkedDenseHF(seer_cfg, layer_idx=src_attn.layer_idx)
        elif variant == "flashprefill_block_sparse":
            impl = LlamaFlashPrefillBlockSparseAttention(seer_cfg, layer_idx=src_attn.layer_idx)
        elif variant == "flashprefill_compactattn":
            impl = LlamaFlashPrefillCompactAttnAttention(seer_cfg, layer_idx=src_attn.layer_idx)
        else:
            raise ValueError(f"Unsupported attention replacement variant: {variant}")
        impl.to(device=src_attn.q_proj.weight.device, dtype=src_attn.q_proj.weight.dtype)
        _reuse_attention_projections(impl, src_attn)
        _adapt_attention_return_contract(impl)
        layer.self_attn = impl

    # ``model.eval()`` was called before replacing ``self_attn`` modules, so
    # newly inserted modules would otherwise remain in training mode and disable
    # prefill-only sparse paths guarded by ``not self.training``.
    model.eval()

    if variant in {"seer_block_sparse", "seer_compactattn", "seer_compactattn_hf"}:
        _load_gate_weights(model, model_id)

    # Marker consumed by the chunked-prefill runner.
    setattr(model.config, "seerattn_attention_harness", "replacement")
    setattr(
        model.config,
        "seerattn_replacement_needs_block_gate_metadata",
        variant in {"seer_block_sparse", "seer_compactattn", "seer_compactattn_hf"},
    )
    if variant in {"seer_compactattn", "seer_compactattn_hf"}:
        setattr(model.config, "seerattn_cache_layout", "heads_first_seer")
    elif variant == "flashprefill_compactattn":
        setattr(model.config, "seerattn_cache_layout", "heads_first_flashprefill")
    else:
        setattr(model.config, "seerattn_cache_layout", "dynamic")

    # Provide a block-rope module compatible with the Seer gate projection width.
    block_config = copy.deepcopy(seer_cfg)
    block_config.hidden_size = seer_cfg.seerattn_gate_hidden_size * seer_cfg.num_attention_heads
    model.model.block_rotary_emb = type(model.model.rotary_emb)(config=block_config)
    ref = model.model.embed_tokens.weight
    model.model.block_rotary_emb.to(device=ref.device, dtype=ref.dtype)
    return model


def load_qwen3_moe_attention_replacement_model(
    *,
    base_model: str,
    variant: str,
    torch_dtype: torch.dtype,
    device_map=None,
    tp_plan: Optional[str] = None,
    dense_backend: str = "flashinfer",
    final_dense_tail_blocks: int = 0,
    compactattn_keep_recent_blocks: int = 2,
    compactattn_disable_first_chunk_dense: bool = False,
    col_pack_impl: str = "indexed_dense",
    col_indexed_impl: str = "fa2_paged",
    col_cache_fill_backend: str = "auto",
    flashprefill_alpha: float = 0.01,
    flashprefill_block_size: int = 128,
    flashprefill_attention_sink: int = 2,
    flashprefill_window_size: int = 4,
    flashprefill_last_n_block: int = 2,
    flashprefill_min_budget: int = 0,
    qwen_long_context: bool = False,
    qwen_long_context_max_position_embeddings: int = 131072,
    qwen_yarn_factor: float = 4.0,
    qwen_original_max_position_embeddings: int = 32768,
    quoka_query_ratio: float = 0.25,
    quoka_kv_budget_ratio: float = 0.25,
) -> nn.Module:
    """Load stock Qwen3-MoE with replacement-harness attention variants."""

    supported_variants = {
        "quoka_dense",
        "flashprefill_block_sparse",
        "flashprefill_compactattn",
    }
    if variant not in supported_variants:
        raise ValueError(
            "Qwen3-MoE replacement currently supports only "
            f"{', '.join(sorted(supported_variants))}"
        )

    common_delegate_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        tp_plan=tp_plan,
        qwen_long_context=bool(qwen_long_context),
        qwen_long_context_max_position_embeddings=int(qwen_long_context_max_position_embeddings),
        qwen_yarn_factor=float(qwen_yarn_factor),
        qwen_original_max_position_embeddings=int(qwen_original_max_position_embeddings),
    )
    if variant == "quoka_dense":
        model = load_quoka_qwen3_moe_model(
            base_model,
            query_ratio=float(quoka_query_ratio),
            kv_budget_ratio=float(quoka_kv_budget_ratio),
            **common_delegate_kwargs,
        )
        return _mark_qwen3_moe_replacement_model(model.eval(), variant=variant)

    load_kwargs = dict(torch_dtype=torch_dtype, attn_implementation="flash_attention_2")
    if qwen_long_context:
        cfg = Qwen3MoeConfig.from_pretrained(base_model)
        cfg.max_position_embeddings = int(qwen_long_context_max_position_embeddings)
        cfg.rope_scaling = {
            "rope_type": "yarn",
            "factor": float(qwen_yarn_factor),
            "original_max_position_embeddings": int(qwen_original_max_position_embeddings),
        }
        load_kwargs["config"] = cfg
    if tp_plan is not None:
        load_kwargs["tp_plan"] = tp_plan
    else:
        load_kwargs["device_map"] = device_map
    model = Qwen3MoeForCausalLM.from_pretrained(base_model, **load_kwargs).eval()

    fp_cfg = _build_qwen3_moe_flashprefill_config(
        base_model,
        variant=variant,
        dense_backend=dense_backend,
        flashprefill_alpha=flashprefill_alpha,
        flashprefill_block_size=flashprefill_block_size,
        flashprefill_attention_sink=flashprefill_attention_sink,
        flashprefill_window_size=flashprefill_window_size,
        flashprefill_last_n_block=flashprefill_last_n_block,
        flashprefill_min_budget=flashprefill_min_budget,
        col_pack_impl=col_pack_impl,
        col_indexed_impl=col_indexed_impl,
        col_cache_fill_backend=col_cache_fill_backend,
        final_dense_tail_blocks=final_dense_tail_blocks,
        qwen_long_context=bool(qwen_long_context),
        qwen_long_context_max_position_embeddings=int(qwen_long_context_max_position_embeddings),
        qwen_yarn_factor=float(qwen_yarn_factor),
        qwen_original_max_position_embeddings=int(qwen_original_max_position_embeddings),
    )
    _copy_seer_attrs_to_stock_config(model.config, fp_cfg)
    setattr(
        model.config,
        "seerattn_flashprefill_execution_mode",
        "compactattn" if variant == "flashprefill_compactattn" else "block_sparse",
    )

    for layer in model.model.layers:
        src_attn = layer.self_attn
        impl = Qwen3MoeFlashPrefillAttention(fp_cfg, layer_idx=src_attn.layer_idx)
        impl.to(device=src_attn.q_proj.weight.device, dtype=src_attn.q_proj.weight.dtype)
        _reuse_attention_projections(impl, src_attn)
        _adapt_attention_return_contract(impl)
        layer.self_attn = impl

    # Replacement modules are inserted after the stock model has entered eval
    # mode; call eval again so FlashPrefill uses the inference/chunked path.
    model.eval()

    return _mark_qwen3_moe_replacement_model(model, variant=variant)
