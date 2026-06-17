import json
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from huggingface_hub import hf_hub_download
from einops import rearrange
from flash_attn.layers.rotary import apply_rotary_emb_func
from transformers.cache_utils import Cache

from compact_attn.modules.attention_distill import (
    attention_distill_forward,
    reduce_mask_ground_truth_by_kv_group,
)
from compact_attn.modules.attention_forward import sparse_flash_attention_forward
from compact_attn.modules.attention_forward_chunked_dense import (
    COMPACTATTN_VERSION,
    _dense_prefill_full_kv,
    chunked_prefill_column_dense_attention_forward,
    chunked_prefill_column_dense_attention_from_keep_block,
    clear_fast_path_content_cache,
)
from compact_attn.kernels.varlen.indexed_dense_prefill_varlen import clear_indexed_dense_workspaces
from compact_attn.modules.common import apply_rotary_pos_emb
from compact_attn.prefill_sparse.llama.configuration_llama_seerattn import SeerAttnLlamaConfig
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn import (
    LlamaSeerAttention,
    SeerAttnLlamaDecoderLayer,
    SeerAttnLlamaForCausalLM,
    SeerAttnLlamaModel,
)


def _normalize_compactattn_threshold_schedule(
    schedule: Optional[Any],
) -> Optional[dict]:
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
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise TypeError(f"Schedule entry {idx} must be a dict, got {type(entry)!r}")
        if "max_kv_len" not in entry or "threshold" not in entry:
            raise ValueError(f"Schedule entry {idx} must contain 'max_kv_len' and 'threshold'.")
        max_kv_len = int(entry["max_kv_len"])
        threshold = float(entry["threshold"])
        if max_kv_len <= 0:
            raise ValueError(f"Schedule entry {idx} has invalid max_kv_len={max_kv_len}.")
        threshold_lower_bound = float(schedule.get("threshold_lower_bound", 0.0))
        threshold = max(threshold, threshold_lower_bound)
        entries.append(
            {
                "max_kv_len": max_kv_len,
                "threshold": threshold,
            }
        )
    entries.sort(key=lambda item: item["max_kv_len"])
    return {
        **schedule,
        "entries": entries,
    }


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


def _maybe_get_stage_tracer(attn_module):
    tracer = getattr(attn_module, "__seer_stage_tracer", None)
    if tracer is None:
        return None
    sample_layer_idx = getattr(attn_module, "__seer_stage_sample_layer_idx", None)
    layer_idx = getattr(attn_module, "layer_idx", None)
    if sample_layer_idx is not None and layer_idx != sample_layer_idx:
        return None
    return tracer


def _note_stage_begin(attn_module, stage_name: str):
    tracer = _maybe_get_stage_tracer(attn_module)
    if tracer is None:
        return
    tracer.set_active_stage(stage_name)
    tracer.note_stage(f"{stage_name}_begin", sync=False)


def _note_stage_done(attn_module, stage_name: str, extra=None):
    tracer = _maybe_get_stage_tracer(attn_module)
    if tracer is None:
        return
    tracer.note_stage(f"{stage_name}_done", extra or {}, sync=False)
    tracer.set_active_stage(None)


def _clear_indexed_dense_workspaces_for_model(model: nn.Module) -> None:
    devices = []
    seen = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if not isinstance(tensor, torch.Tensor):
            continue
        device = tensor.device
        if device.type != "cuda":
            continue
        device_idx = int(device.index) if device.index is not None else -1
        if device_idx in seen:
            continue
        seen.add(device_idx)
        devices.append(device)
    if not devices:
        clear_indexed_dense_workspaces()
        return
    for device in devices:
        clear_indexed_dense_workspaces(device=device)


def _reinitialize_compactattn_query_branches(model: nn.Module) -> None:
    for layer in getattr(model.model, "layers", []):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue
        attn_gate = getattr(self_attn, "attn_gate", None)
        mask_linear_q = getattr(attn_gate, "mask_linear_q", None)
        if mask_linear_q is None:
            continue
        model._init_weights(mask_linear_q)


def _load_gate_weights_with_optional_q_reinit(
    model: nn.Module,
    gate_weights: dict,
    reinit_query_branch: bool,
) -> None:
    def _maybe_fold_kv_group_weights(checkpoint_value: torch.Tensor, target_value: torch.Tensor):
        if checkpoint_value.ndim != 3 or target_value.ndim != 3:
            return None
        old_heads, in_dim, out_dim = checkpoint_value.shape
        new_heads, target_in_dim, target_out_dim = target_value.shape
        if in_dim != target_in_dim or out_dim != target_out_dim:
            return None
        if old_heads % new_heads != 0:
            return None
        group_size = old_heads // new_heads
        return checkpoint_value.view(new_heads, group_size, in_dim, out_dim).mean(dim=1)

    filtered_gate_weights = {}
    model_state = model.state_dict()
    for key, value in gate_weights.items():
        target = model_state.get(key, None)
        if target is None:
            filtered_gate_weights[key] = value
            continue
        if target.shape == value.shape:
            filtered_gate_weights[key] = value
            continue
        if ".attn_gate.mask_linear_k.weight" in key:
            folded = _maybe_fold_kv_group_weights(value, target)
            if folded is not None:
                filtered_gate_weights[key] = folded
                continue
        if reinit_query_branch and ".attn_gate.mask_linear_q.weight" in key:
            continue
        raise ValueError(
            f"Gate weight shape mismatch for {key}: checkpoint {tuple(value.shape)} vs model {tuple(target.shape)}"
        )

    model.load_state_dict(filtered_gate_weights, strict=False)
    if reinit_query_branch:
        _reinitialize_compactattn_query_branches(model)


class LlamaSeerAttentionChunkedDense(LlamaSeerAttention):
    def __init__(self, config: SeerAttnLlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.compactattn_kv_group_aware_gate = bool(
            getattr(config, "seerattn_compactattn_kv_group_aware_gate", False)
        )
        self.compactattn_threshold = float(
            getattr(config, "seerattn_compactattn_threshold", 5e-4)
        )
        self.compactattn_threshold_schedule = _normalize_compactattn_threshold_schedule(
            getattr(config, "seerattn_compactattn_threshold_schedule", None)
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
        if self.compactattn_indexed_impl not in {"fa2_paged", "triton_direct", "fa2_indexed", "fi_paged", "fi_zero_copy", "cudnn_one_shot"}:
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

    def _resolve_compactattn_threshold(self, kv_len: int) -> float:
        return _resolve_compactattn_threshold_from_schedule(
            self.compactattn_threshold_schedule,
            kv_len=kv_len,
            default_threshold=self.compactattn_threshold,
        )

    def _expand_compactattn_gate_scores_for_q_heads(
        self, attn_gate_output: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if attn_gate_output is None or not self.compactattn_kv_group_aware_gate:
            return attn_gate_output
        return attn_gate_output.repeat_interleave(self.num_key_value_groups, dim=1)

    def _compute_compactattn_mask_loss(
        self,
        attn_gate_output: torch.Tensor,
        ground_truth_mask: torch.Tensor,
        block_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_start = attn_gate_output.shape[2] // 4
        gate_scores = attn_gate_output[:, :, q_start:].to(torch.float32)
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
                    f"target={tuple(target_mask.shape)} gate={tuple(gate_scores.shape)} "
                    f"kv_group_aware={self.compactattn_kv_group_aware_gate} "
                    f"num_key_value_groups={self.num_key_value_groups}"
                )
        target_mask = target_mask.to(torch.float32)
        gate_log_probs = F.log_softmax(gate_scores, dim=-1)
        kl_loss = self.mask_loss_func(gate_log_probs, target_mask)
        mask_loss = kl_loss
        self._compactattn_last_train_loss_stats = {
            "mask_kl_loss": float(kl_loss.detach().item()),
            "mask_loss_total": float(mask_loss.detach().item()),
        }
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
            block_attention_mask=block_attention_mask.detach() if torch.is_tensor(block_attention_mask) else block_attention_mask,
            query_states=query_states.detach(),
            key_states=key_states.detach(),
            softmax_scale=float(self.scaling),
            q_len=int(q_len),
            kv_len=int(kv_len),
        )

    def _compactattn_chunked_gate_cache_store(self, past_key_value: Optional[Cache]):
        if past_key_value is None:
            return None
        store = getattr(past_key_value, "_seer_compactattn_chunked_gate_k_cache", None)
        if store is None:
            store = {}
            setattr(past_key_value, "_seer_compactattn_chunked_gate_k_cache", store)
        return store

    # ---- Storage interface overrides: use compactattn buffer store ----

    def _gate_k_get(self, past_key_value: Optional[Cache]) -> Optional[torch.Tensor]:
        return self._compactattn_get_cached_k_blocks(past_key_value)

    def _gate_k_append(
        self, past_key_value: Optional[Cache], k_blocks: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return self._compactattn_append_cached_k_blocks(past_key_value, k_blocks)

    def _compactattn_cached_k_blocks_from_current(
        self,
        key_states_nope: torch.Tensor,
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        k_blocks = self.attn_gate.compress_key_blocks(key_states_nope)
        _, k_blocks = self.attn_gate.apply_block_position_embeddings(
            q=None,
            k=k_blocks,
            position_embeddings=block_position_embeddings,
        )
        return k_blocks.detach().contiguous()

    def _compactattn_gate_cache_capacity(self, required_len: int, current_capacity: int = 0) -> int:
        growth_blocks = 256
        if required_len <= 0:
            return 0
        capacity = max(int(current_capacity), growth_blocks)
        required_len = int(required_len)
        while capacity < required_len:
            capacity *= 2
        return capacity

    def _compactattn_get_cached_k_blocks(
        self,
        past_key_value: Optional[Cache],
    ) -> Optional[torch.Tensor]:
        store = self._compactattn_chunked_gate_cache_store(past_key_value)
        if store is None:
            return None
        entry = store.get(self.layer_idx, None)
        if entry is None:
            return None
        if torch.is_tensor(entry):
            return entry
        buffer = entry.get("buffer", None)
        length = int(entry.get("length", 0))
        if buffer is None or length <= 0:
            return None
        return buffer.narrow(1, 0, length)

    def _compactattn_append_cached_k_blocks(
        self,
        past_key_value: Optional[Cache],
        k_blocks: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        store = self._compactattn_chunked_gate_cache_store(past_key_value)
        if store is None:
            return None

        k_blocks = k_blocks.detach()
        if not k_blocks.is_contiguous():
            k_blocks = k_blocks.contiguous()

        entry = store.get(self.layer_idx, None)
        if entry is None:
            current_len = 0
            buffer = None
        elif torch.is_tensor(entry):
            buffer = entry
            current_len = int(buffer.shape[1])
        else:
            buffer = entry.get("buffer", None)
            current_len = int(entry.get("length", 0))

        append_len = int(k_blocks.shape[1])
        required_len = current_len + append_len
        if required_len <= 0:
            return None

        needs_realloc = (
            buffer is None
            or buffer.device != k_blocks.device
            or buffer.dtype != k_blocks.dtype
            or buffer.dim() != k_blocks.dim()
            or buffer.shape[0] != k_blocks.shape[0]
            or buffer.shape[2:] != k_blocks.shape[2:]
            or int(buffer.shape[1]) < required_len
        )

        if needs_realloc:
            current_capacity = 0 if buffer is None else int(buffer.shape[1])
            capacity = self._compactattn_gate_cache_capacity(required_len, current_capacity=current_capacity)
            new_shape = list(k_blocks.shape)
            new_shape[1] = capacity
            new_buffer = torch.empty(new_shape, device=k_blocks.device, dtype=k_blocks.dtype)
            if buffer is not None and current_len > 0:
                prev_view = buffer.narrow(1, 0, current_len)
                new_buffer.narrow(1, 0, current_len).copy_(prev_view)
            buffer = new_buffer

        buffer.narrow(1, current_len, append_len).copy_(k_blocks)
        store[self.layer_idx] = {"buffer": buffer, "length": required_len}
        return buffer.narrow(1, 0, required_len)

    def _append_compactattn_chunked_gate_key_cache(
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
        k_blocks = self._compactattn_cached_k_blocks_from_current(
            key_states_nope=key_states_nope,
            block_position_embeddings=block_position_embeddings,
        )
        if self._compactattn_append_cached_k_blocks(past_key_value, k_blocks) is None:
            return False
        return True

    def _compute_compactattn_gate_output(
        self,
        query_states_nope: torch.Tensor,
        key_states_nope: torch.Tensor,
        block_attention_mask: Optional[torch.Tensor],
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_softmax: bool,
    ) -> Optional[torch.Tensor]:
        return self.attn_gate(
            query_states_nope,
            key_states_nope,
            block_attention_mask,
            block_position_embeddings,
            use_softmax=use_softmax,
        )

    def _do_kv_cache_update(self, key_states, value_states, past_key_value, sin, cos, cache_position, collect_stats):
        """Update the KV cache and return (key_states, value_states, cache_update_ms).

        Subclasses can override to change storage format while preserving the
        seq-first [bsz, kv_len, Hkv, D] tensors returned to forward().
        """
        def _run():
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(
                key_states.flatten(-2, -1),
                value_states.flatten(-2, -1),
                self.layer_idx,
                cache_kwargs,
            )
            k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
            v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)
            return k, v
        (k_out, v_out), ms = _cuda_elapsed_ms(_run, enabled=collect_stats)
        return k_out, v_out, ms

    def _do_chunked_compactattn_attention(self, query_states, key_states, value_states, **kwargs):
        """Run compactattn attention.  Subclasses can override to inject alternative kernels."""
        return chunked_prefill_column_dense_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        block_position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        seer_batch_kv_lens: Optional[torch.LongTensor] = None,
        seer_batch_query_lens: Optional[torch.LongTensor] = None,
        seer_batch_query_start: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        collect_stats = bool(self.compactattn_debug)
        input_shape = hidden_states.shape[:-1]
        q_len = hidden_states.shape[1]
        self._reject_unsupported_batch_metadata(
            seer_batch_kv_lens=seer_batch_kv_lens,
            seer_batch_query_lens=seer_batch_query_lens,
            seer_batch_query_start=seer_batch_query_start,
        )
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

        cos, sin = position_embeddings
        use_flash_rope = bool(self.config.use_flash_rope) and cos.dim() == 2
        if use_flash_rope:
            query_states_nope = query_states.clone()
            key_states_nope = key_states.clone()
        else:
            query_states_nope = query_states
            key_states_nope = key_states
        def _run_rope():
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

        _note_stage_begin(self, "rope")
        (query_states, key_states), rope_ms = _cuda_elapsed_ms(
            _run_rope, enabled=collect_stats
        )
        _note_stage_done(self, "rope", {"rope_ms": float(rope_ms)})

        cache_update_ms = 0.0
        if past_key_value is not None:
            _note_stage_begin(self, "cache_update")
            key_states, value_states, cache_update_ms = self._do_kv_cache_update(
                key_states, value_states, past_key_value, sin, cos, cache_position, collect_stats
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
            _note_stage_begin(self, "gate_compute")
            if q_len == 1:
                attn_gate_output = None  # decode: gate irrelevant, attention is always dense
            else:
                if collect_stats:
                    setattr(self.attn_gate, "__seer_layer_idx", int(self.layer_idx))
                    setattr(self.attn_gate, "__seer_chunk_idx", -1)
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
                # decode (q_len==1): always dense regardless of batch layout
                attn_output, _ = _dense_prefill_full_kv(
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


class SeerAttnLlamaChunkedDenseDecoderLayer(SeerAttnLlamaDecoderLayer):
    def __init__(self, config: SeerAttnLlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = LlamaSeerAttentionChunkedDense(config=config, layer_idx=layer_idx)


class SeerAttnLlamaChunkedDenseModel(SeerAttnLlamaModel):
    def __init__(self, config: SeerAttnLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                SeerAttnLlamaChunkedDenseDecoderLayer(config=config, layer_idx=layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()


class SeerAttnLlamaChunkedDenseForCausalLM(SeerAttnLlamaForCausalLM):
    _no_split_modules = [
        *SeerAttnLlamaForCausalLM._no_split_modules,
        "SeerAttnLlamaChunkedDenseDecoderLayer",
    ]

    def __init__(self, config: SeerAttnLlamaConfig):
        super().__init__(config)
        if not hasattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks"):
            setattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks", 2)
        if not hasattr(config, "seerattn_compactattn_release_indexed_workspaces"):
            setattr(config, "seerattn_compactattn_release_indexed_workspaces", True)
        if not hasattr(config, "seerattn_profile_compactattn_cleanup"):
            setattr(config, "seerattn_profile_compactattn_cleanup", False)
        self._profile_defer_compactattn_cleanup = False
        self._profile_pending_compactattn_cleanup = False
        self.model = SeerAttnLlamaChunkedDenseModel(config)

    def _run_compactattn_cleanup(self) -> None:
        if bool(getattr(self.config, "seerattn_compactattn_release_indexed_workspaces", False)):
            _clear_indexed_dense_workspaces_for_model(self)
        clear_fast_path_content_cache()

    def _prepare_profiled_compactattn_cleanup(self) -> None:
        if bool(getattr(self.config, "seerattn_profile_compactattn_cleanup", False)):
            self._profile_defer_compactattn_cleanup = True
            self._profile_pending_compactattn_cleanup = True

    def _finish_profiled_compactattn_cleanup(self) -> None:
        if self._profile_pending_compactattn_cleanup:
            self._run_compactattn_cleanup()
            self._profile_pending_compactattn_cleanup = False

    def forward(self, *args, **kwargs):
        try:
            return super().forward(*args, **kwargs)
        finally:
            if self._profile_defer_compactattn_cleanup:
                self._profile_defer_compactattn_cleanup = False
            else:
                self._run_compactattn_cleanup()
                self._profile_pending_compactattn_cleanup = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, load_gate=True, *model_args, **kwargs):
        compactattn_threshold = kwargs.pop("seerattn_compactattn_threshold", None)
        compactattn_threshold_schedule = kwargs.pop("seerattn_compactattn_threshold_schedule", None)
        compactattn_use_chunked_gate_cache = kwargs.pop("seerattn_compactattn_use_chunked_gate_cache", True)
        compactattn_keep_recent_blocks = kwargs.pop("seerattn_compactattn_keep_recent_blocks", 2)
        compactattn_kv_group_aware_gate = kwargs.pop("seerattn_compactattn_kv_group_aware_gate", None)
        kwargs.pop("seerattn_compactattn_adjacent_align_lambda", None)
        kwargs.pop("seerattn_compactattn_union_penalty_lambda", None)
        kwargs.pop("seerattn_compactattn_union_penalty_margin", None)
        kwargs.pop("seerattn_compactattn_union_penalty_temperature", None)
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
        compactattn_chunked_gate_head_pool = kwargs.pop("seerattn_compactattn_chunked_gate_head_pool", "none")
        final_dense_tail_blocks = kwargs.pop("seerattn_chunked_prefill_final_dense_tail_blocks", 2)

        if load_gate:
            config = SeerAttnLlamaConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            base_model = config.base_model

            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))

            if compactattn_threshold is None:
                compactattn_threshold = config.seerattn_threshold
            if compactattn_threshold_schedule is None:
                compactattn_threshold_schedule = getattr(config, "seerattn_compactattn_threshold_schedule", None)
            if compactattn_kv_group_aware_gate is None:
                compactattn_kv_group_aware_gate = bool(
                    getattr(config, "seerattn_compactattn_kv_group_aware_gate", False)
                )
            if compactattn_reinit_q_branch_on_load is None:
                compactattn_reinit_q_branch_on_load = bool(
                    getattr(config, "seerattn_compactattn_reinit_q_branch_on_load", False)
                )
            setattr(config, "seerattn_compactattn_threshold", float(compactattn_threshold))
            setattr(
                config,
                "seerattn_compactattn_threshold_schedule",
                _normalize_compactattn_threshold_schedule(compactattn_threshold_schedule),
            )
            setattr(config, "seerattn_compactattn_use_chunked_gate_cache", bool(compactattn_use_chunked_gate_cache))
            setattr(config, "seerattn_compactattn_keep_recent_blocks", int(compactattn_keep_recent_blocks))
            setattr(
                config,
                "seerattn_compactattn_kv_group_aware_gate",
                bool(compactattn_kv_group_aware_gate),
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

            model = super(SeerAttnLlamaForCausalLM, cls).from_pretrained(
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
            config = SeerAttnLlamaConfig.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))

            if compactattn_threshold is None:
                compactattn_threshold = getattr(config, "seerattn_threshold", 5e-4)
            if compactattn_threshold_schedule is None:
                compactattn_threshold_schedule = getattr(config, "seerattn_compactattn_threshold_schedule", None)
            if compactattn_kv_group_aware_gate is None:
                compactattn_kv_group_aware_gate = bool(
                    getattr(config, "seerattn_compactattn_kv_group_aware_gate", False)
                )
            if compactattn_reinit_q_branch_on_load is None:
                compactattn_reinit_q_branch_on_load = bool(
                    getattr(config, "seerattn_compactattn_reinit_q_branch_on_load", False)
                )

            setattr(config, "seerattn_compactattn_threshold", float(compactattn_threshold))
            setattr(
                config,
                "seerattn_compactattn_threshold_schedule",
                _normalize_compactattn_threshold_schedule(compactattn_threshold_schedule),
            )
            setattr(config, "seerattn_compactattn_use_chunked_gate_cache", bool(compactattn_use_chunked_gate_cache))
            setattr(config, "seerattn_compactattn_keep_recent_blocks", int(compactattn_keep_recent_blocks))
            setattr(
                config,
                "seerattn_compactattn_kv_group_aware_gate",
                bool(compactattn_kv_group_aware_gate),
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

            model = super(SeerAttnLlamaForCausalLM, cls).from_pretrained(
                pretrained_model_name_or_path, config=config, *model_args, **kwargs
            )
            for layer in model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_use_chunked_gate_cache"):
                    layer.self_attn.compactattn_use_chunked_gate_cache = bool(compactattn_use_chunked_gate_cache)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "compactattn_kv_group_aware_gate"):
                    layer.self_attn.compactattn_kv_group_aware_gate = bool(
                        getattr(model.config, "seerattn_compactattn_kv_group_aware_gate", False)
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
