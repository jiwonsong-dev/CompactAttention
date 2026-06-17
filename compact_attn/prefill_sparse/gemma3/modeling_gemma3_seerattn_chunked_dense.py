# coding=utf-8
from typing import Any, Optional, Tuple, Union

import copy
import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.gemma3.modeling_gemma3 import (
    ALL_ATTENTION_FUNCTIONS,
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3MLP,
    Gemma3PreTrainedModel,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
    Gemma3TextModel,
    apply_rotary_pos_emb,
    create_causal_mask,
    eager_attention_forward,
    create_sliding_window_causal_mask,
)
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg
from huggingface_hub import hf_hub_download
from einops import rearrange

from compact_attn.modules.attention_distill import attention_distill_forward, reduce_mask_ground_truth_by_kv_group
from compact_attn.modules.attention_forward import sparse_flash_attention_forward
from compact_attn.modules.attention_forward_chunked_dense import (
    COMPACTATTN_VERSION,
    chunked_prefill_column_dense_attention_forward,
)
from compact_attn.modules.dense_prefill import dense_prefill_full_kv
from compact_attn.prefill_sparse.attn_gate import ATTNGATE_CLASSES, MultiHeadLinear
from compact_attn.utils import BaseModelOutputWithPastAndSeer, CausalLMOutputWithPastAndSeer

from .configuration_gemma3_seerattn import SeerAttnGemma3Config


logger = logging.get_logger(__name__)


def _cuda_elapsed_ms(fn, enabled: bool = True):
    if not enabled:
        return fn(), 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    return out, float(start.elapsed_time(end))


def _normalize_compactattn_threshold_schedule(schedule: Optional[Any]) -> Optional[dict]:
    if schedule is None:
        return None
    if isinstance(schedule, str):
        schedule_path = os.path.expanduser(schedule)
        with open(schedule_path, "r", encoding="utf-8") as f:
            schedule = json.load(f)
    elif isinstance(schedule, tuple):
        schedule = list(schedule)

    if isinstance(schedule, list):
        schedule = {"entries": schedule}
    if not isinstance(schedule, dict):
        raise TypeError(
            "seerattn_compactattn_threshold_schedule must be None, a JSON path, a dict, or a list of entries; "
            f"got {type(schedule)!r}"
        )

    raw_entries = schedule.get("entries", None)
    if raw_entries is None:
        raise ValueError("Threshold schedule must contain an 'entries' field.")
    if not isinstance(raw_entries, list) or len(raw_entries) == 0:
        raise ValueError("Threshold schedule 'entries' must be a non-empty list.")

    entries = []
    threshold_lower_bound = float(schedule.get("threshold_lower_bound", 0.0))
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise TypeError(f"Schedule entry {idx} must be a dict, got {type(entry)!r}")
        if "max_kv_len" not in entry or "threshold" not in entry:
            raise ValueError(f"Schedule entry {idx} must contain 'max_kv_len' and 'threshold'.")
        max_kv_len = int(entry["max_kv_len"])
        threshold = max(float(entry["threshold"]), threshold_lower_bound)
        if max_kv_len <= 0:
            raise ValueError(f"Schedule entry {idx} has invalid max_kv_len={max_kv_len}.")
        entries.append({"max_kv_len": max_kv_len, "threshold": threshold})
    entries.sort(key=lambda item: item["max_kv_len"])
    return {**schedule, "entries": entries}


def _resolve_compactattn_threshold_from_schedule(
    schedule: Optional[dict],
    kv_len: int,
    default_threshold: float,
) -> float:
    if not schedule:
        return float(default_threshold)
    kv_len = int(kv_len)
    for entry in schedule["entries"]:
        if kv_len <= int(entry["max_kv_len"]):
            return float(entry["threshold"])
    return float(schedule["entries"][-1]["threshold"])


def _sync_gemma3_base_vocab_size(config: SeerAttnGemma3Config, base_model: str, kwargs) -> None:
    config_kwargs = {}
    for key in ("cache_dir", "force_download", "local_files_only", "revision", "token", "trust_remote_code"):
        if key in kwargs:
            config_kwargs[key] = kwargs[key]
    base_config = AutoConfig.from_pretrained(base_model, **config_kwargs)
    if getattr(base_config, "model_type", None) == "gemma3":
        config.vocab_size = base_config.text_config.vocab_size
    else:
        config.vocab_size = base_config.vocab_size


class SeerAttnGemma3Attention(nn.Module):
    def __init__(self, config: SeerAttnGemma3Config, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
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
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.attn_gate = ATTNGATE_CLASSES[config.seerattn_gate_type](
            config.seerattn_gate_block_size,
            self.head_dim,
            config.seerattn_gate_hidden_size,
            num_k_head=config.num_key_value_heads,
            num_q_head=config.num_attention_heads,
            force_double=config.seerattn_gate_force_double,
            use_flash_rope=False,
            kv_group_aware_query=bool(
                getattr(config, "seerattn_compactattn_kv_group_aware_gate", False)
            ),
        )
        self.compactattn_kv_group_aware_gate = bool(
            getattr(config, "seerattn_compactattn_kv_group_aware_gate", False)
        )
        self.compactattn_gate_layout = str(
            getattr(
                config,
                "seerattn_compactattn_gate_layout",
                "gqa_aware" if self.compactattn_kv_group_aware_gate else "old_union_qhead",
            )
        )
        self.mask_loss_func = torch.nn.KLDivLoss()
        self.block_sparse_debug = False
        self._seer_bs_last_stats = None
        self.compactattn_debug = bool(getattr(config, "seerattn_compactattn_debug", False))
        self._compactattn_last_stats = None
        self.compactattn_threshold = float(getattr(config, "seerattn_compactattn_threshold", 5e-4))
        self.compactattn_threshold_schedule = _normalize_compactattn_threshold_schedule(
            getattr(config, "seerattn_compactattn_threshold_schedule", None)
        )
        self.compactattn_keep_recent_blocks = int(
            getattr(config, "seerattn_compactattn_keep_recent_blocks", 2)
        )
        self.compactattn_disable_first_chunk_dense = bool(
            getattr(config, "seerattn_compactattn_disable_first_chunk_dense", False)
        )
        self.compactattn_pack_impl = str(
            getattr(config, "seerattn_compactattn_pack_impl", "indexed_dense")
        )
        if self.compactattn_pack_impl not in {"torch", "triton", "indexed_dense"}:
            self.compactattn_pack_impl = "indexed_dense"
        self.compactattn_indexed_impl = str(
            getattr(config, "seerattn_compactattn_indexed_impl", "fi_zero_copy")
        )
        if self.compactattn_indexed_impl not in {
            "fa2_paged",
            "triton_direct",
            "fa2_indexed",
            "fi_paged",
            "fi_zero_copy",
            "cudnn_one_shot",
        }:
            self.compactattn_indexed_impl = "fi_zero_copy"
        self.compactattn_cache_fill_backend = str(
            getattr(config, "seerattn_compactattn_cache_fill_backend", "auto")
        )
        if self.compactattn_cache_fill_backend not in {"auto", "cuda", "triton"}:
            self.compactattn_cache_fill_backend = "auto"
        self.compactattn_version = COMPACTATTN_VERSION
        self.profile_file = os.environ.get("PROFILE_FILE", None)

    def _is_compactattn_execution(self) -> bool:
        return str(getattr(self.config, "seerattn_gemma3_execution_mode", "block_sparse")) == "compactattn"

    def _resolve_compactattn_threshold(self, kv_len: int) -> float:
        return _resolve_compactattn_threshold_from_schedule(
            self.compactattn_threshold_schedule,
            kv_len=kv_len,
            default_threshold=self.compactattn_threshold,
        )

    def _do_chunked_compactattn_attention(self, query_states, key_states, value_states, **kwargs):
        return chunked_prefill_column_dense_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            **kwargs,
        )

    def _chunked_gate_cache_store(self, past_key_value: Optional[Cache]):
        if past_key_value is None:
            return None
        store = getattr(past_key_value, "_seer_chunked_gate_k_cache", None)
        if store is None:
            store = {}
            setattr(past_key_value, "_seer_chunked_gate_k_cache", store)
        return store

    def _gate_k_get(self, past_key_value: Optional[Cache]) -> Optional[torch.Tensor]:
        store = self._chunked_gate_cache_store(past_key_value)
        if store is None:
            return None
        return store.get(self.layer_idx, None)

    def _gate_k_append(
        self, past_key_value: Optional[Cache], k_blocks: torch.Tensor
    ) -> Optional[torch.Tensor]:
        store = self._chunked_gate_cache_store(past_key_value)
        if store is None:
            return None
        k_blocks = k_blocks.detach()
        if not k_blocks.is_contiguous():
            k_blocks = k_blocks.contiguous()
        prev = store.get(self.layer_idx, None)
        full = k_blocks if prev is None else torch.cat((prev, k_blocks), dim=1).contiguous()
        store[self.layer_idx] = full
        return full

    def _should_use_chunked_gate_cache(self, attention_mask: Optional[torch.Tensor]) -> bool:
        if not bool(getattr(self.config, "seerattn_use_chunked_gate_cache", True)):
            return False
        if attention_mask is None or attention_mask.dim() != 2:
            return True
        if attention_mask.shape[0] > 1:
            return False
        return True

    def _can_use_chunked_gate_cache(
        self,
        key_states_nope: torch.Tensor,
        past_key_value: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> bool:
        del block_position_embeddings
        if past_key_value is not None:
            try:
                chunk_start = int(past_key_value.get_seq_length())
            except Exception:
                chunk_start = -1
        elif cache_position is not None:
            chunk_start = int(cache_position[0].item())
        else:
            return False
        block_size = int(self.config.seerattn_gate_block_size)
        if block_size <= 0:
            return False
        q_len = int(key_states_nope.shape[1])
        if (chunk_start % block_size) != 0:
            return False
        if q_len < block_size:
            return False
        return True

    def _build_chunked_gate_blocks(
        self,
        query_states_nope: torch.Tensor,
        key_states_nope: torch.Tensor,
        block_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_blocks = self.attn_gate.compress_query_blocks(query_states_nope)
        k_blocks = self.attn_gate.compress_key_blocks(key_states_nope)
        q_blocks, k_blocks = self.attn_gate.apply_block_position_embeddings(
            q=q_blocks,
            k=k_blocks,
            position_embeddings=block_position_embeddings,
        )
        return q_blocks, k_blocks

    def _append_chunked_gate_key_cache(
        self,
        key_states_nope: torch.Tensor,
        past_key_value: Optional[Cache],
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache_position: Optional[torch.LongTensor],
    ) -> bool:
        if not self._can_use_chunked_gate_cache(
            key_states_nope=key_states_nope,
            past_key_value=past_key_value,
            cache_position=cache_position,
            block_position_embeddings=block_position_embeddings,
        ):
            return False
        store = self._chunked_gate_cache_store(past_key_value)
        if store is None:
            return False

        k_blocks = self.attn_gate.compress_key_blocks(key_states_nope)
        _, k_blocks = self.attn_gate.apply_block_position_embeddings(
            q=None,
            k=k_blocks,
            position_embeddings=block_position_embeddings,
        )
        k_blocks = k_blocks.detach().contiguous()
        prev = store.get(self.layer_idx, None)
        if prev is None:
            store[self.layer_idx] = k_blocks
        else:
            store[self.layer_idx] = torch.cat((prev, k_blocks), dim=1).contiguous()
        return True

    def _compute_chunked_gate_from_cache(
        self,
        query_states_nope: torch.Tensor,
        key_states_nope: torch.Tensor,
        past_key_value: Optional[Cache],
        block_attention_mask: Optional[torch.Tensor],
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache_position: Optional[torch.LongTensor],
        use_softmax: bool,
    ) -> Optional[torch.Tensor]:
        if block_attention_mask is None:
            return None
        if not self._can_use_chunked_gate_cache(
            key_states_nope=key_states_nope,
            past_key_value=past_key_value,
            cache_position=cache_position,
            block_position_embeddings=block_position_embeddings,
        ):
            return None
        prev_k_blocks = self._gate_k_get(past_key_value)
        if prev_k_blocks is None:
            return None

        q_blocks, current_k_blocks = self._build_chunked_gate_blocks(
            query_states_nope=query_states_nope,
            key_states_nope=key_states_nope,
            block_position_embeddings=block_position_embeddings,
        )

        if block_attention_mask.shape[-2] != q_blocks.shape[1]:
            return None
        full_k_len = int(prev_k_blocks.shape[1]) + int(current_k_blocks.shape[1])
        if block_attention_mask.shape[-1] != full_k_len:
            return None

        full_k_blocks = self._gate_k_append(past_key_value, current_k_blocks)
        if full_k_blocks is None:
            return None

        return self.attn_gate.score_compressed_blocks(
            q=q_blocks,
            k=full_k_blocks,
            attention_mask=block_attention_mask,
            use_softmax=use_softmax,
        )

    def _compute_full_layer_mask_loss(
        self,
        attn_gate_output: torch.Tensor,
        ground_truth_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_start = attn_gate_output.shape[2] // 4
        gate_scores = attn_gate_output[:, :, q_start:].to(torch.float32)
        finite_min = torch.finfo(gate_scores.dtype).min / 4
        finite_max = torch.finfo(gate_scores.dtype).max / 4
        gate_scores = torch.nan_to_num(gate_scores, nan=0.0, posinf=finite_max, neginf=finite_min)
        target_mask = ground_truth_mask[:, :, q_start:]
        if target_mask.shape[1] != gate_scores.shape[1]:
            if (
                self.compactattn_kv_group_aware_gate
                and target_mask.shape[1] == gate_scores.shape[1] * self.num_key_value_groups
            ):
                target_mask = reduce_mask_ground_truth_by_kv_group(
                    target_mask,
                    num_key_value_groups=self.num_key_value_groups,
                    pooling="max",
                )
            else:
                raise RuntimeError(
                    "Ground-truth head dimension does not match gate head dimension: "
                    f"target={tuple(target_mask.shape)} gate={tuple(gate_scores.shape)}"
                )
        target_mask = target_mask.to(torch.float32)
        gate_log_probs = F.log_softmax(gate_scores, dim=-1)
        mask_loss = self.mask_loss_func(gate_log_probs, target_mask)
        return mask_loss, gate_log_probs, target_mask

    def _maybe_record_compactattn_threshold_calibration(
        self,
        attn_gate_output: Optional[torch.Tensor],
        ground_truth_mask: Optional[torch.Tensor],
        block_attention_mask: Optional[torch.Tensor],
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        *,
        q_len: int,
        kv_len: int,
    ) -> None:
        callback = getattr(self, "_compactattn_threshold_calibration_callback", None)
        if callback is None or attn_gate_output is None or ground_truth_mask is None:
            return

        gate_probs = F.softmax(attn_gate_output.detach().to(torch.float32), dim=-1)
        target_mask = ground_truth_mask.detach().to(torch.float32)
        if target_mask.shape[1] != gate_probs.shape[1]:
            if (
                self.compactattn_kv_group_aware_gate
                and target_mask.shape[1] == gate_probs.shape[1] * self.num_key_value_groups
            ):
                target_mask = reduce_mask_ground_truth_by_kv_group(
                    target_mask,
                    num_key_value_groups=self.num_key_value_groups,
                    pooling="max",
                )
            else:
                raise RuntimeError(
                    "Calibration target head dimension does not match gate head dimension: "
                    f"target={tuple(target_mask.shape)} gate={tuple(gate_probs.shape)}"
                )
        callback(
            layer=self,
            gate_probs=gate_probs,
            target_mask=target_mask,
            block_attention_mask=(
                block_attention_mask.detach()
                if torch.is_tensor(block_attention_mask)
                else block_attention_mask
            ),
            query_states=query_states.detach(),
            key_states=key_states.detach(),
            softmax_scale=float(self.scaling),
            q_len=int(q_len),
            kv_len=int(kv_len),
        )

    def _native_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ):
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        return attn_output, attn_weights

    def _compactattn_inference_forward(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states_nope: torch.Tensor,
        key_states_nope: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        block_attention_mask: Optional[torch.Tensor],
        input_shape,
        qkv_proj_ms: float,
        collect_stats: bool,
    ):
        q_len = int(query_states.shape[1])
        kv_len = int(key_states.shape[1])
        use_chunked_gate_cache = self._should_use_chunked_gate_cache(attention_mask)
        force_dense_prefill = (
            q_len > 1
            and bool(getattr(self.config, "seerattn_chunked_prefill_force_dense", False))
        )
        chunked_prefill = q_len > 1 and kv_len > q_len
        full_prefill_selected = (
            q_len > 1
            and kv_len == q_len
            and self.compactattn_disable_first_chunk_dense
        )
        full_prefill_dense = (
            q_len > 1
            and kv_len == q_len
            and not self.compactattn_disable_first_chunk_dense
        )

        _gate_start = torch.cuda.Event(enable_timing=True) if collect_stats else None
        _gate_end = torch.cuda.Event(enable_timing=True) if collect_stats else None
        if collect_stats:
            _gate_start.record()

        if force_dense_prefill:
            attn_gate_output = None
            if use_chunked_gate_cache:
                self._append_chunked_gate_key_cache(
                    key_states_nope=key_states_nope,
                    past_key_value=past_key_value,
                    block_position_embeddings=block_position_embeddings,
                    cache_position=cache_position,
                )
        elif chunked_prefill:
            attn_gate_output = None
            if use_chunked_gate_cache:
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
                attn_gate_output = self.attn_gate(
                    query_states_nope,
                    key_states_nope,
                    block_attention_mask,
                    block_position_embeddings,
                    use_softmax=self.config.seerattn_sparsity_method == "threshold",
                )
                if use_chunked_gate_cache:
                    self._append_chunked_gate_key_cache(
                        key_states_nope=key_states_nope,
                        past_key_value=past_key_value,
                        block_position_embeddings=block_position_embeddings,
                        cache_position=cache_position,
                    )
        elif full_prefill_selected:
            attn_gate_output = self.attn_gate(
                query_states_nope,
                key_states_nope,
                block_attention_mask,
                block_position_embeddings,
                use_softmax=self.config.seerattn_sparsity_method == "threshold",
            )
            if use_chunked_gate_cache:
                self._append_chunked_gate_key_cache(
                    key_states_nope=key_states_nope,
                    past_key_value=past_key_value,
                    block_position_embeddings=block_position_embeddings,
                    cache_position=cache_position,
                )
        elif full_prefill_dense:
            attn_gate_output = None
            if use_chunked_gate_cache:
                self._append_chunked_gate_key_cache(
                    key_states_nope=key_states_nope,
                    past_key_value=past_key_value,
                    block_position_embeddings=block_position_embeddings,
                    cache_position=cache_position,
                )
        else:
            attn_gate_output = None

        if collect_stats:
            _gate_end.record()
            _gate_end.synchronize()
            gate_ms = float(_gate_start.elapsed_time(_gate_end))
        else:
            gate_ms = 0.0

        path_label = "other"
        if force_dense_prefill or full_prefill_dense or q_len == 1:
            attn_output, dense_stats = dense_prefill_full_kv(
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
            path_label = "forced_dense" if force_dense_prefill else "first_chunk_dense" if full_prefill_dense else "decode_dense"
            compactattn_stats = dict(dense_stats) if isinstance(dense_stats, dict) else {}
        elif chunked_prefill or full_prefill_selected:
            attn_output, compactattn_stats = self._do_chunked_compactattn_attention(
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
            path_label = "first_chunk_selected" if full_prefill_selected else "chunked_selected"
        else:
            attn_output, compactattn_stats = dense_prefill_full_kv(
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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output, o_proj_ms = _cuda_elapsed_ms(lambda: self.o_proj(attn_output), collect_stats)
        if collect_stats:
            merged_stats = {}
            if isinstance(compactattn_stats, dict):
                merged_stats.update(compactattn_stats)
            merged_stats.update(
                {
                    "qkv_proj_ms": float(qkv_proj_ms),
                    "gate_ms": float(gate_ms),
                    "o_proj_ms": float(o_proj_ms),
                    "path": path_label,
                    "compactattn_version": self.compactattn_version,
                }
            )
            self._compactattn_last_stats = merged_stats
        else:
            self._compactattn_last_stats = None
        self._seer_bs_last_stats = None
        return attn_output, query_states.new_zeros(()), None, None, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        collect_stats = (bool(self.block_sparse_debug) or bool(self.compactattn_debug)) and not self.training
        input_shape = hidden_states.shape[:-1]
        q_len = hidden_states.shape[1]

        (query_states, key_states, value_states), qkv_proj_ms = _cuda_elapsed_ms(
            lambda: (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)),
            collect_stats,
        )

        query_states = rearrange(query_states, "... (h d) -> ... h d", d=self.head_dim)
        key_states = rearrange(key_states, "... (h d) -> ... h d", d=self.head_dim)
        value_states = rearrange(value_states, "... (h d) -> ... h d", d=self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states_nope = query_states
        key_states_nope = key_states

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            cos,
            sin,
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states_native = past_key_value.update(
                key_states,
                value_states.transpose(1, 2),
                self.layer_idx,
                cache_kwargs,
            )
            value_states = value_states_native.transpose(1, 2)
        else:
            value_states_native = value_states.transpose(1, 2)

        query_states_sparse = query_states.transpose(1, 2).contiguous()
        key_states_sparse = key_states.transpose(1, 2).contiguous()
        value_states_sparse = value_states_native.transpose(1, 2).contiguous()

        # Gemma3 local/sliding layers are semantically local attention layers; keep
        # them dense/native and only apply Seer block sparsity to full-attention layers.
        if self.is_sliding:
            attn_output, attn_weights = self._native_attention_forward(
                query_states,
                key_states,
                value_states_native,
                attention_mask,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output, o_proj_ms = _cuda_elapsed_ms(lambda: self.o_proj(attn_output), collect_stats)
            if collect_stats:
                self._seer_bs_last_stats = {
                    "qkv_proj_ms": float(qkv_proj_ms),
                    "gate_ms": 0.0,
                    "sparse_attn_ms": 0.0,
                    "o_proj_ms": float(o_proj_ms),
                }
            return attn_output, hidden_states.new_zeros(()), attn_weights, None, None

        if self._is_compactattn_execution() and not self.training:
            return self._compactattn_inference_forward(
                query_states=query_states_sparse,
                key_states=key_states_sparse,
                value_states=value_states_sparse,
                query_states_nope=query_states_nope,
                key_states_nope=key_states_nope,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                block_position_embeddings=block_position_embeddings,
                block_attention_mask=block_attention_mask,
                input_shape=input_shape,
                qkv_proj_ms=qkv_proj_ms,
                collect_stats=collect_stats,
            )

        chunked_prefill = (not self.training) and (q_len > 1) and (key_states_sparse.shape[1] > q_len)
        use_chunked_gate_cache = self._should_use_chunked_gate_cache(attention_mask)
        force_dense_prefill = (
            (not self.training)
            and (q_len > 1)
            and bool(getattr(self.config, "seerattn_chunked_prefill_force_dense", False))
        )

        _gate_start = torch.cuda.Event(enable_timing=True) if collect_stats else None
        _gate_end = torch.cuda.Event(enable_timing=True) if collect_stats else None
        if collect_stats:
            _gate_start.record()

        if force_dense_prefill:
            attn_gate_output = None
            if use_chunked_gate_cache:
                self._append_chunked_gate_key_cache(
                    key_states_nope=key_states_nope,
                    past_key_value=past_key_value,
                    block_position_embeddings=block_position_embeddings,
                    cache_position=cache_position,
                )
        elif chunked_prefill:
            attn_gate_output = None
            if use_chunked_gate_cache:
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
                attn_gate_output = self.attn_gate(
                    query_states_sparse,
                    key_states_sparse,
                    block_attention_mask,
                    None,
                    use_softmax=self.config.seerattn_sparsity_method == "threshold",
                )
                if use_chunked_gate_cache:
                    self._append_chunked_gate_key_cache(
                        key_states_nope=key_states_nope,
                        past_key_value=past_key_value,
                        block_position_embeddings=block_position_embeddings,
                        cache_position=cache_position,
                    )
        else:
            if q_len == 1:
                attn_gate_output = None
            else:
                attn_gate_output = self.attn_gate(
                    query_states_nope,
                    key_states_nope,
                    block_attention_mask,
                    block_position_embeddings,
                    use_softmax=not self.training and self.config.seerattn_sparsity_method == "threshold",
                )
            if q_len > 1 and use_chunked_gate_cache:
                self._append_chunked_gate_key_cache(
                    key_states_nope=key_states_nope,
                    past_key_value=past_key_value,
                    block_position_embeddings=block_position_embeddings,
                    cache_position=cache_position,
                )

        if collect_stats:
            _gate_end.record()
            _gate_end.synchronize()
            gate_ms = float(_gate_start.elapsed_time(_gate_end))
        else:
            gate_ms = 0.0

        if self.training:
            attn_output, ground_truth_mask = attention_distill_forward(
                query_states_sparse,
                key_states_sparse,
                value_states_sparse,
                softmax_scale=self.scaling,
                block_size=self.config.seerattn_gate_block_size,
                num_key_value_groups=self.num_key_value_groups,
                kv_group_aware_query=self.compactattn_kv_group_aware_gate,
            )
            sparse_attn_ms = 0.0
        else:
            if force_dense_prefill or q_len == 1:
                attn_output, _ = dense_prefill_full_kv(
                    query_states=query_states_sparse,
                    key_states=key_states_sparse,
                    value_states=value_states_sparse,
                    attention_mask=attention_mask,
                    softmax_scale=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    fallback_used=0.0,
                    measure_timing=False,
                    attn_module=self,
                )
                sparse_attn_ms = 0.0
            else:
                attn_output, sparse_attn_ms = _cuda_elapsed_ms(
                    lambda: sparse_flash_attention_forward(
                        query_states_sparse,
                        key_states_sparse,
                        value_states_sparse,
                        attention_mask,
                        query_length=q_len,
                        softmax_scale=self.scaling,
                        attn_gate_score=attn_gate_output,
                        sparsity_method=self.config.seerattn_sparsity_method,
                        threshold=self.config.seerattn_threshold,
                        nz_ratio=self.config.seerattn_nz_ratio,
                        last_block_dense=self.config.seerattn_last_block_dense,
                        block_size=self.config.seerattn_gate_block_size,
                        num_key_value_groups=self.num_key_value_groups,
                        profile_file=self.profile_file,
                        block_attention_mask=block_attention_mask,
                    ),
                    collect_stats,
                )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output, o_proj_ms = _cuda_elapsed_ms(lambda: self.o_proj(attn_output), collect_stats)

        if collect_stats:
            self._seer_bs_last_stats = {
                "qkv_proj_ms": float(qkv_proj_ms),
                "gate_ms": float(gate_ms),
                "sparse_attn_ms": float(sparse_attn_ms),
                "o_proj_ms": float(o_proj_ms),
            }
        else:
            self._seer_bs_last_stats = None

        if self.training:
            mask_loss, attn_gate_output, ground_truth_mask = self._compute_full_layer_mask_loss(
                attn_gate_output=attn_gate_output,
                ground_truth_mask=ground_truth_mask,
            )
            self._maybe_record_compactattn_threshold_calibration(
                attn_gate_output=attn_gate_output,
                ground_truth_mask=ground_truth_mask,
                block_attention_mask=block_attention_mask,
                query_states=query_states_sparse,
                key_states=key_states_sparse,
                q_len=q_len,
                kv_len=int(key_states_sparse.shape[1]),
            )
        else:
            mask_loss = hidden_states.new_zeros(())
            attn_gate_output = None
            ground_truth_mask = None

        if not kwargs.get("output_attentions", False):
            attn_gate_output = None
            ground_truth_mask = None
        return attn_output, mask_loss, None, attn_gate_output, ground_truth_mask


class SeerAttnGemma3DecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config: SeerAttnGemma3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = SeerAttnGemma3Attention(config=config, layer_idx=layer_idx)

    @deprecate_kwarg("last_cache_position", version="4.53.0")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = (
            position_embeddings_local if self.self_attn.is_sliding else position_embeddings_global
        )
        hidden_states, seerattn_mask_loss, self_attn_weights, mask_gate_prediction, mask_ground_truth = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            block_position_embeddings=block_position_embeddings,
            block_attention_mask=block_attention_mask,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, seerattn_mask_loss)
        if output_attentions:
            outputs += (self_attn_weights, mask_gate_prediction, mask_ground_truth)
        return outputs


class SeerAttnGemma3PreTrainedModel(Gemma3PreTrainedModel):
    config_class = SeerAttnGemma3Config
    _no_split_modules = ["SeerAttnGemma3DecoderLayer"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MultiHeadLinear):
            std = self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=std)


class SeerAttnGemma3Model(SeerAttnGemma3PreTrainedModel):
    def __init__(self, config: SeerAttnGemma3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [SeerAttnGemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        block_config = copy.deepcopy(config)
        block_config.hidden_size = config.seerattn_gate_hidden_size * config.num_attention_heads
        block_config.head_dim = config.seerattn_gate_hidden_size
        self.block_rotary_emb = Gemma3RotaryEmbedding(config=block_config)
        self.gradient_checkpointing = False

        local_config = copy.deepcopy(config)
        local_config.rope_theta = config.rope_local_base_freq
        local_config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=local_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _seerattn_update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        batch_size, query_len = inputs_embeds.shape[:2]
        if query_len == 1:
            return None

        block_size = self.config.seerattn_gate_block_size
        device = inputs_embeds.device
        if attention_mask is None:
            if cache_position is not None:
                kv_len = int(cache_position[-1].item()) + 1
            else:
                kv_len = query_len
            attention_mask = torch.ones((batch_size, kv_len), dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(torch.bool)

        query_mask = attention_mask[:, -query_len:]
        query_valid_blocks = F.max_pool1d(
            query_mask.unsqueeze(1).to(torch.float32),
            kernel_size=block_size,
            stride=block_size,
            ceil_mode=True,
        ).squeeze(1).to(torch.bool)
        key_valid_blocks = F.max_pool1d(
            attention_mask.unsqueeze(1).to(torch.float32),
            kernel_size=block_size,
            stride=block_size,
            ceil_mode=True,
        ).squeeze(1).to(torch.bool)

        q_blocks = query_valid_blocks.shape[-1]
        k_blocks = key_valid_blocks.shape[-1]
        valid_q_lens = query_mask.sum(dim=-1, dtype=torch.int64)
        valid_k_lens = attention_mask.sum(dim=-1, dtype=torch.int64)
        past_lens = (valid_k_lens - valid_q_lens).clamp(min=0)

        q_block_end = (torch.arange(q_blocks, device=device, dtype=torch.int64) + 1) * block_size - 1
        q_block_end = q_block_end.unsqueeze(0).expand(batch_size, -1)
        q_block_end = torch.minimum(q_block_end, (valid_q_lens.unsqueeze(1) - 1).clamp(min=0))
        q_block_end = q_block_end + past_lens.unsqueeze(1)

        k_block_idx = torch.arange(k_blocks, device=device, dtype=torch.int64).view(1, 1, -1)
        causal_mask = k_block_idx <= torch.div(q_block_end.unsqueeze(-1), block_size, rounding_mode="floor")

        gate_mask = causal_mask
        gate_mask = gate_mask & query_valid_blocks.unsqueeze(-1)
        gate_mask = gate_mask & key_valid_blocks.unsqueeze(1)
        return gate_mask.unsqueeze(1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndSeer:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        raw_attention_mask = attention_mask if not isinstance(attention_mask, dict) else None
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        block_attention_mask = self._seerattn_update_causal_mask(raw_attention_mask, inputs_embeds, cache_position)
        hidden_states = inputs_embeds
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)
        block_position_ids = position_ids[:, 0::self.config.seerattn_gate_block_size]
        block_position_embeddings = self.block_rotary_emb(hidden_states, block_position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_mask_gate_predictions = () if output_attentions else None
        all_mask_ground_truths = () if output_attentions else None
        total_mask_loss = hidden_states.new_zeros(())

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_attention_mask = (
                causal_mask_mapping[decoder_layer.attention_type]
                if decoder_layer.attention_type == "sliding_attention"
                else raw_attention_mask
            )
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                block_position_embeddings=block_position_embeddings,
                block_attention_mask=block_attention_mask if decoder_layer.attention_type == "full_attention" else None,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            total_mask_loss = total_mask_loss + layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)
                all_mask_gate_predictions += (layer_outputs[3],)
                all_mask_ground_truths += (layer_outputs[4],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPastAndSeer(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mask_gate_predictions=all_mask_gate_predictions,
            mask_ground_truths=all_mask_ground_truths,
            mask_loss=total_mask_loss,
        )
        return output if return_dict else output.to_tuple()


class SeerAttnGemma3ChunkedDenseForCausalLM(Gemma3ForCausalLM):
    config_class = SeerAttnGemma3Config
    _no_split_modules = ["SeerAttnGemma3DecoderLayer"]

    def __init__(self, config: SeerAttnGemma3Config):
        super().__init__(config)
        self.model = SeerAttnGemma3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPastAndSeer:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPastAndSeer(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mask_gate_predictions=outputs.mask_gate_predictions,
            mask_ground_truths=outputs.mask_ground_truths,
            mask_loss=outputs.mask_loss,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, load_gate=True, *model_args, **kwargs):
        input_config = kwargs.pop("config", None)
        gate_type = kwargs.get("seerattn_gate_type", "Qavg_Kmaxminavg")
        gate_hidden_size = kwargs.get("seerattn_gate_hidden_size", 128)
        gate_force_double = kwargs.get("seerattn_gate_force_double", False)
        kv_group_aware_gate = kwargs.pop("seerattn_compactattn_kv_group_aware_gate", False)
        reinit_q_branch = kwargs.pop("seerattn_compactattn_reinit_q_branch_on_load", False)

        def _coerce_gemma3_config(base_config, *, base_model_name):
            if isinstance(base_config, SeerAttnGemma3Config):
                config = base_config
            else:
                if getattr(base_config, "model_type", None) == "gemma3":
                    base_config = base_config.text_config
                config_dict = base_config.to_dict()
                if config_dict.get("torch_dtype", None) == "auto":
                    config_dict["torch_dtype"] = None
                config = SeerAttnGemma3Config(**config_dict)
            _sync_gemma3_base_vocab_size(config, base_model_name, kwargs)
            config.seerattn_gate_type = gate_type
            config.seerattn_gate_hidden_size = gate_hidden_size
            config.seerattn_gate_force_double = gate_force_double
            config.seerattn_compactattn_kv_group_aware_gate = bool(kv_group_aware_gate)
            config.seerattn_compactattn_gate_layout = (
                "gqa_aware" if bool(kv_group_aware_gate) else "old_union_qhead"
            )
            config.seerattn_compactattn_reinit_q_branch_on_load = bool(reinit_q_branch)
            config.base_model = base_model_name
            return config

        if load_gate:
            config = input_config
            if config is None:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            base_model = getattr(config, "base_model", pretrained_model_name_or_path)
            config = _coerce_gemma3_config(config, base_model_name=base_model)
            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))
            model = super(Gemma3ForCausalLM, cls).from_pretrained(
                base_model,
                config=config,
                *model_args,
                **kwargs,
            )
            if os.path.exists(pretrained_model_name_or_path):
                gate_weights = torch.load(os.path.join(pretrained_model_name_or_path, "attn_gate_weights.pth"))
            else:
                gate_weights = torch.load(
                    hf_hub_download(repo_id=pretrained_model_name_or_path, filename="attn_gate_weights.pth")
                )
            filtered_gate_weights = {}
            for key, value in gate_weights.items():
                if "attn_gate" in key:
                    if reinit_q_branch and ".attn_gate.mask_linear_q.weight" in key:
                        continue
                    filtered_gate_weights[key] = value
            model.load_state_dict(filtered_gate_weights, strict=False)
            if reinit_q_branch:
                for layer in getattr(model.model, "layers", []):
                    attn_gate = getattr(getattr(layer, "self_attn", None), "attn_gate", None)
                    mask_linear_q = getattr(attn_gate, "mask_linear_q", None)
                    if mask_linear_q is not None:
                        model._init_weights(mask_linear_q)
            print("Attention gate weights loaded successfully.")
        else:
            config = input_config
            if config is None:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            base_model = getattr(config, "base_model", pretrained_model_name_or_path)
            config = _coerce_gemma3_config(config, base_model_name=base_model)
            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))
            model = super(Gemma3ForCausalLM, cls).from_pretrained(
                base_model,
                config=config,
                *model_args,
                **kwargs,
            )
        return model


__all__ = [
    "SeerAttnGemma3Config",
    "SeerAttnGemma3ChunkedDenseForCausalLM",
]
