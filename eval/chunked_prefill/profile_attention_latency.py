#!/usr/bin/env python
"""Profile attention latency inside full-model prefill.

This script reuses the exact model-loading and prefill runners from
`test_unified_latency_sweep.py` and only layers a CUDA-event profiler on top of
each decoder layer's `self_attn.forward`. That keeps the measured execution path
aligned with the main latency sweep while decomposing end-to-end latency into:

- attention time
- non-attention time
- attention share of total latency
"""

import argparse
import csv
import gc
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.chunked_prefill.chunked_prefill_runtime import (
    DEFAULT_DENSE_BACKEND,
    _cleanup_after_benchmark_case,
    _cleanup_after_sequence_iteration,
    _enable_default_qwen_long_context,
    _synchronize_cuda_all_devices,
    build_input_batches,
    get_dtype,
    load_model,
    parse_seq_lens,
    print_config,
    resolve_vocab_size,
    resolve_last_block_dense_scope,
    resolve_model_id,
    resolve_variant,
    run_chunked_prefill_once,
    run_dense_prefill_once,
    validate_args,
)


class AttentionProfiler:
    """Record CUDA-event timing around every decoder layer attention forward."""

    def __init__(
        self,
        collect_compactattn_components: bool = False,
        collect_dense_components: bool = False,
        collect_block_sparse_components: bool = False,
        collect_quoka_components: bool = False,
    ) -> None:
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._wrappers: list[tuple[torch.nn.Module, Any]] = []
        self.collect_compactattn_components = bool(collect_compactattn_components)
        self.collect_dense_components = bool(collect_dense_components)
        self.collect_block_sparse_components = bool(collect_block_sparse_components)
        self.collect_quoka_components = bool(collect_quoka_components)
        self._compactattn_component_sums: dict[str, float] = {}
        self._dense_component_sums: dict[str, float] = {}
        self._block_sparse_component_sums: dict[str, float] = {}
        self._quoka_component_sums: dict[str, float] = {}

    @staticmethod
    def _wait_async_collective_output(obj: Any) -> Any:
        wait = getattr(obj, "wait", None)
        if callable(wait):
            return wait()
        if isinstance(obj, tuple):
            return tuple(AttentionProfiler._wait_async_collective_output(item) for item in obj)
        if isinstance(obj, list):
            return [AttentionProfiler._wait_async_collective_output(item) for item in obj]
        if isinstance(obj, dict):
            return {
                key: AttentionProfiler._wait_async_collective_output(value)
                for key, value in obj.items()
            }
        return obj

    @staticmethod
    def _find_cuda_tensor(obj: Any) -> Optional[torch.Tensor]:
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            return obj
        if isinstance(obj, (tuple, list)):
            for item in obj:
                found = AttentionProfiler._find_cuda_tensor(item)
                if found is not None:
                    return found
        if isinstance(obj, dict):
            for item in obj.values():
                found = AttentionProfiler._find_cuda_tensor(item)
                if found is not None:
                    return found
        return None

    def install(self, model) -> None:
        layers = _decoder_layers(model)
        last_layer_idx = len(layers) - 1
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            original = attn.forward
            profiler = self
            is_last_layer = layer_idx == last_layer_idx

            def _make_wrapper(orig_fn, *, layer_idx: int, is_last_layer: bool):
                def _profiled_forward(self, *args, **kwargs):
                    tensor = profiler._find_cuda_tensor(args)
                    if tensor is None:
                        tensor = profiler._find_cuda_tensor(kwargs)
                    if tensor is None:
                        return orig_fn(*args, **kwargs)

                    device = tensor.device
                    with torch.cuda.device(device):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        stream = torch.cuda.current_stream(device)
                        start.record(stream)
                        out = orig_fn(*args, **kwargs)
                        profiler._record_compactattn_components(self, layer_idx)
                        profiler._record_dense_components(self, layer_idx)
                        profiler._record_block_sparse_components(self, layer_idx)
                        profiler._record_quoka_components(self, layer_idx)
                        if (
                            is_last_layer
                            and hasattr(model, "_finish_profiled_compactattn_cleanup")
                        ):
                            model._finish_profiled_compactattn_cleanup()
                        end.record(stream)
                    profiler._events.append((start, end))
                    return out

                return _profiled_forward

            bound = _make_wrapper(
                original,
                layer_idx=layer_idx,
                is_last_layer=is_last_layer,
            ).__get__(attn, type(attn))
            attn.forward = bound
            self._wrappers.append((attn, original))

    def uninstall(self) -> None:
        for attn, original in self._wrappers:
            attn.forward = original
        self._wrappers.clear()

    def reset_run(self) -> None:
        self._events.clear()
        self._compactattn_component_sums.clear()
        self._dense_component_sums.clear()
        self._block_sparse_component_sums.clear()
        self._quoka_component_sums.clear()

    def finish_run(self) -> float:
        if not self._events:
            return 0.0
        _synchronize_cuda_all_devices()
        return sum(start.elapsed_time(end) for start, end in self._events)

    def _record_compactattn_components(self, attn_module, layer_idx: int) -> None:
        if not self.collect_compactattn_components:
            return
        cur = getattr(attn_module, "_compactattn_last_stats", None)
        if not isinstance(cur, dict):
            return
        self._compactattn_component_sums["compactattn_profiled_calls"] = (
            self._compactattn_component_sums.get("compactattn_profiled_calls", 0.0) + 1.0
        )
        self._compactattn_component_sums[f"compactattn_profiled_layer_{layer_idx}_calls"] = (
            self._compactattn_component_sums.get(f"compactattn_profiled_layer_{layer_idx}_calls", 0.0) + 1.0
        )
        for key, value in cur.items():
            if isinstance(value, (int, float)):
                prefixed = key if key.startswith("col_") else f"compactattn_{key}"
                self._compactattn_component_sums[prefixed] = (
                    self._compactattn_component_sums.get(prefixed, 0.0) + float(value)
                )
        pending_pairs = getattr(attn_module, "_compactattn_pending_timing_pairs", None)
        if pending_pairs:
            for key, start, end in pending_pairs:
                end.synchronize()
                prefixed = f"compactattn_internal_{key}"
                self._compactattn_component_sums[prefixed] = (
                    self._compactattn_component_sums.get(prefixed, 0.0)
                    + float(start.elapsed_time(end))
                )

    def _record_dense_components(self, attn_module, layer_idx: int) -> None:
        if not self.collect_dense_components:
            return
        cur = getattr(attn_module, "_dense_last_stats", None)
        if not isinstance(cur, dict):
            return
        self._dense_component_sums["dense_profiled_calls"] = (
            self._dense_component_sums.get("dense_profiled_calls", 0.0) + 1.0
        )
        self._dense_component_sums[f"dense_profiled_layer_{layer_idx}_calls"] = (
            self._dense_component_sums.get(f"dense_profiled_layer_{layer_idx}_calls", 0.0) + 1.0
        )
        for key, value in cur.items():
            if isinstance(value, (int, float)):
                prefixed = key if key.startswith("dense_") else f"dense_{key}"
                self._dense_component_sums[prefixed] = (
                    self._dense_component_sums.get(prefixed, 0.0) + float(value)
                )

    def _record_block_sparse_components(self, attn_module, layer_idx: int) -> None:
        if not self.collect_block_sparse_components:
            return
        cur = getattr(attn_module, "_block_sparse_last_stats", None)
        if not isinstance(cur, dict):
            return
        self._block_sparse_component_sums["block_sparse_profiled_calls"] = (
            self._block_sparse_component_sums.get("block_sparse_profiled_calls", 0.0) + 1.0
        )
        self._block_sparse_component_sums[f"block_sparse_profiled_layer_{layer_idx}_calls"] = (
            self._block_sparse_component_sums.get(f"block_sparse_profiled_layer_{layer_idx}_calls", 0.0) + 1.0
        )
        for key, value in cur.items():
            if isinstance(value, (int, float)):
                prefixed = key if key.startswith("block_sparse_") else f"block_sparse_{key}"
                self._block_sparse_component_sums[prefixed] = (
                    self._block_sparse_component_sums.get(prefixed, 0.0) + float(value)
                )
        pending_pairs = getattr(attn_module, "_block_sparse_pending_timing_pairs", None)
        if pending_pairs:
            for key, start, end in pending_pairs:
                end.synchronize()
                prefixed = f"block_sparse_internal_{key}"
                self._block_sparse_component_sums[prefixed] = (
                    self._block_sparse_component_sums.get(prefixed, 0.0)
                    + float(start.elapsed_time(end))
                )

    def _record_quoka_components(self, attn_module, layer_idx: int) -> None:
        if not self.collect_quoka_components:
            return
        cur = getattr(attn_module, "_quoka_last_stats", None)
        if not isinstance(cur, dict):
            return
        self._quoka_component_sums["quoka_profiled_calls"] = (
            self._quoka_component_sums.get("quoka_profiled_calls", 0.0) + 1.0
        )
        self._quoka_component_sums[f"quoka_profiled_layer_{layer_idx}_calls"] = (
            self._quoka_component_sums.get(f"quoka_profiled_layer_{layer_idx}_calls", 0.0)
            + 1.0
        )
        for key, value in cur.items():
            if isinstance(value, (int, float)):
                prefixed = key if key.startswith(("quoka_", "dense_")) else f"quoka_{key}"
                self._quoka_component_sums[prefixed] = (
                    self._quoka_component_sums.get(prefixed, 0.0) + float(value)
                )

    def component_snapshot(self) -> dict[str, float]:
        out = dict(self._compactattn_component_sums)
        out.update(self._dense_component_sums)
        out.update(self._block_sparse_component_sums)
        out.update(self._quoka_component_sums)
        return out


def _enable_compactattn_component_debug(model) -> None:
    for layer in _decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "compactattn_debug"):
            attn.compactattn_debug = True
        if attn is not None and hasattr(attn, "compactattn_detailed_timing"):
            attn.compactattn_detailed_timing = True


def _enable_dense_component_debug(model) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        setattr(cfg, "seerattn_profile_dense_components", True)


def _enable_block_sparse_component_debug(model) -> None:
    for layer in _decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "block_sparse_debug"):
            attn.block_sparse_debug = True
        if attn is not None and hasattr(attn, "block_sparse_detailed_timing"):
            attn.block_sparse_detailed_timing = True
        if attn is not None and hasattr(attn, "block_sparse_profile_selection"):
            attn.block_sparse_profile_selection = True


def _enable_quoka_component_debug(model) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        setattr(cfg, "seerattn_profile_quoka_components", True)
    for layer in _decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "quoka_detailed_timing"):
            attn.quoka_detailed_timing = True


def _set_deferred_async_collective_wait(model, enabled: bool) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        setattr(cfg, "seerattn_defer_async_collective_wait", bool(enabled))
    for layer in _decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        attn_cfg = getattr(attn, "config", None)
        if attn_cfg is not None:
            setattr(attn_cfg, "seerattn_defer_async_collective_wait", bool(enabled))


def _decoder_layers(model):
    model_body = getattr(model, "model", None)
    layers = getattr(model_body, "layers", None)
    if layers is not None:
        return list(layers)
    language_model = getattr(model_body, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is not None:
        return list(layers)
    raise AttributeError("Could not find decoder layers on model")


def _compactattn_component_summary(attn_ms: float, stats: dict[str, float]) -> dict[str, float]:
    selected_materialize_ms = float(stats.get("col_selected_kv_materialize_ms", 0.0))
    if selected_materialize_ms > 0.0:
        compaction_ms = selected_materialize_ms
    else:
        compaction_ms = (
            float(stats.get("col_index_compact_ms", 0.0))
            + float(stats.get("col_index_src_layout_ms", 0.0))
            + float(stats.get("col_index_cache_fill_ms", 0.0))
        )
    computation_ms = float(stats.get("compactattn_dense_kernel_ms", 0.0))
    other_ms = float(attn_ms) - compaction_ms - computation_ms
    if other_ms < 0.0 and abs(other_ms) < 1e-3:
        other_ms = 0.0
    return {
        "component_compaction_ms": compaction_ms,
        "component_computation_ms": computation_ms,
        "component_other_ms": other_ms,
    }


def _dense_component_summary(attn_ms: float, stats: dict[str, float]) -> dict[str, float]:
    contiguous_ms = float(stats.get("dense_contiguous_ms", 0.0))
    plan_ms = float(stats.get("dense_gather_pack_ms", 0.0))
    kernel_ms = float(
        stats.get("dense_kernel_ms", stats.get("dense_dense_kernel_ms", 0.0))
    )
    upad_ms = float(stats.get("dense_upad_input_ms", 0.0))
    pad_ms = float(stats.get("dense_pad_output_ms", 0.0))
    other_ms = float(attn_ms) - contiguous_ms - plan_ms - kernel_ms - upad_ms - pad_ms
    if other_ms < 0.0 and abs(other_ms) < 1e-3:
        other_ms = 0.0
    return {
        "component_contiguous_ms": contiguous_ms,
        "component_plan_ms": plan_ms,
        "component_kernel_ms": kernel_ms,
        "component_upad_ms": upad_ms,
        "component_pad_ms": pad_ms,
        "component_other_ms": other_ms,
    }


def _tp_global_max_dict(values: dict[str, float]) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return dict(values)
    if not values:
        return {}
    device = torch.device("cuda", torch.cuda.current_device())
    keys = sorted(values)
    tensor = torch.tensor([float(values[k]) for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return {k: float(v) for k, v in zip(keys, tensor.tolist())}

def _tp_global_max_ms(value: float) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    device = torch.device("cuda", torch.cuda.current_device())
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def parse_args():
    p = argparse.ArgumentParser(
        description="Profile attention latency inside full-model chunked prefill"
    )
    p.add_argument("--model", "--seer-model", dest="model", type=str, default=None)
    p.add_argument(
        "--execution-mode",
        type=str,
        required=True,
        choices=["dense", "block_sparse", "compactattn", "compactattention", "compact_attention"],
        help=(
            "Execution family. Prefer compactattention for the compacted dense "
            "execution path; compactattn remains as a backward-compatible alias."
        ),
    )
    p.add_argument(
        "--selection-method",
        type=str,
        required=True,
        choices=["none", "seer", "seer_hf", "quoka", "flashprefill"],
    )
    p.add_argument("--seq-lens", type=str, default="8192,16384,32768,65536,131072")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--dense-attn-impl",
        type=str,
        default="flash_attention_2",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    p.add_argument(
        "--dense-backend",
        type=str,
        default=DEFAULT_DENSE_BACKEND,
        choices=["flash_attn", "flashinfer", "fa3_direct"],
    )
    p.add_argument(
        "--dense-class-family",
        type=str,
        default="stock",
        choices=["stock", "seer", "flashprefill"],
        help=(
            "Class family used for dense baselines. Use seer or flashprefill "
            "to force dense attention through the corresponding custom Llama "
            "class when checking speedup conventions."
        ),
    )
    p.add_argument(
        "--attention-harness",
        type=str,
        default="legacy",
        choices=["legacy", "replacement"],
        help=(
            "Profiling harness. 'legacy' uses historical method-specific model "
            "loaders. 'replacement' uses a stock LLaMA model with self_attn "
            "modules replaced for supported methods."
        ),
    )
    p.add_argument(
        "--last-block-dense-scope",
        type=str,
        default=None,
        choices=["all_prefill_chunks", "final_prefill_chunk"],
    )
    p.add_argument(
        "--last-block-dense",
        dest="last_block_dense",
        action="store_true",
        default=None,
        help=(
            "Enable the Seer last-block-dense selector override. If omitted, "
            "the loaded model/config default is preserved."
        ),
    )
    p.add_argument(
        "--no-last-block-dense",
        dest="last_block_dense",
        action="store_false",
        help="Disable the Seer last-block-dense selector override.",
    )
    p.add_argument(
        "--final-chunk-dense-blocks",
        type=int,
        default=None,
        help=(
            "Unified final-chunk dense-row policy. Seer block-sparse supports "
            "0 or 2 and maps it to last_block_dense, Seer CompactAttention maps "
            "the value to final_dense_tail_blocks, and FlashPrefill variants map "
            "it to final-chunk last_n blocks."
        ),
    )
    p.add_argument("--dense-prefix-tokens", type=int, default=0)
    p.add_argument("--final-dense-tail-blocks", type=int, default=None)
    p.add_argument("--flashprefill-alpha", type=float, default=None)
    p.add_argument("--flashprefill-block-size", type=int, default=128)
    p.add_argument("--flashprefill-attention-sink", type=int, default=2)
    p.add_argument("--flashprefill-window-size", type=int, default=4)
    p.add_argument(
        "--flashprefill-last-n-block",
        type=int,
        default=0,
        help=(
            "FlashPrefill selector dense tail for the final scored chunk only. "
            "Intermediate chunks use 0."
        ),
    )
    p.add_argument("--flashprefill-min-budget", type=int, default=0)
    p.add_argument(
        "--defer-flashprefill-async-wait",
        action="store_true",
        help=(
            "Compatibility flag. FlashPrefill TP async waits are disabled by "
            "default in the profiling harness; this flag preserves that default."
        ),
    )
    p.add_argument(
        "--include-flashprefill-async-wait-in-attn",
        action="store_true",
        help=(
            "Legacy FlashPrefill accounting: include the explicit TP async "
            "collective wait inside attn_ms. By default no explicit wait is "
            "called, matching the other attention implementations."
        ),
    )
    p.add_argument("--compactattn-keep-recent-blocks", "--compactattention-keep-recent-blocks", type=int, default=2)
    p.add_argument("--compactattn-disable-first-chunk-dense", "--compactattention-disable-first-chunk-dense", action="store_true")
    p.add_argument(
        "--compactattn-chunked-gate-head-pool",
        "--compactattention-chunked-gate-head-pool",
        type=str,
        default="none",
        choices=["none", "avg", "max", "score_avg"],
    )
    p.add_argument(
        "--col-pack-impl",
        "--compactattention-pack-impl",
        type=str,
        default="indexed_dense",
        choices=["torch", "triton", "indexed_dense"],
    )
    p.add_argument(
        "--col-indexed-impl",
        "--compactattention-indexed-impl",
        type=str,
        default="auto",
        choices=["auto", "fa2_paged", "triton_direct", "fa2_indexed", "fi_paged", "fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"],
    )
    p.add_argument(
        "--col-cache-fill-backend",
        "--compactattention-cache-fill-backend",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "triton"],
    )
    p.add_argument("--quoka-query-ratio", type=float, default=0.25)
    p.add_argument("--quoka-kv-budget-ratio", type=float, default=0.25)
    p.add_argument("--prompt-source", type=str, default="ruler", choices=["random", "ruler"])
    p.add_argument("--input-schedule", type=str, default="fixed", choices=["fixed", "cycle"])
    p.add_argument("--task", type=str, default="vt")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--data-file", type=str, default=None)
    p.add_argument("--qwen-long-context", action="store_true")
    p.add_argument("--qwen-long-context-max-position-embeddings", type=int, default=131072)
    p.add_argument("--qwen-yarn-factor", type=float, default=4.0)
    p.add_argument("--qwen-original-max-position-embeddings", type=int, default=32768)
    p.add_argument("--device-map", type=str, default="auto", choices=["auto"])
    p.add_argument(
        "--profile-compactattn-components",
        action="store_true",
        help="Opt-in debug timing for compactattn internal compaction/computation split.",
    )
    p.add_argument(
        "--compactattn-components-csv",
        type=Path,
        default=None,
        help="Optional CSV output for --profile-compactattn-components.",
    )
    p.add_argument(
        "--profile-dense-components",
        action="store_true",
        help="Opt-in debug timing for dense FlashInfer internal copy/plan/kernel split.",
    )
    p.add_argument(
        "--dense-components-csv",
        type=Path,
        default=None,
        help="Optional CSV output for --profile-dense-components.",
    )
    p.add_argument(
        "--profile-block-sparse-components",
        action="store_true",
        help="Opt-in debug timing for FlashPrefill block-sparse selection/attention split.",
    )
    p.add_argument(
        "--block-sparse-components-csv",
        type=Path,
        default=None,
        help="Optional CSV output for --profile-block-sparse-components.",
    )
    p.add_argument(
        "--profile-quoka-components",
        action="store_true",
        help="Opt-in debug timing for QUOKA selection/dense-kernel components.",
    )
    p.add_argument(
        "--quoka-components-csv",
        type=Path,
        default=None,
        help="Optional CSV output for --profile-quoka-components.",
    )
    return p.parse_args()


def _mean_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}
    keys = sorted({key for item in dicts for key in item})
    out = {}
    for key in keys:
        out[key] = statistics.mean(float(item.get(key, 0.0)) for item in dicts)
    return out


def _write_component_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "seq_len",
        "total_ms",
        "attn_ms",
        "component_profile_attn_ms",
        "component_profile_total_ms",
        "component_compaction_ms",
        "component_computation_ms",
        "component_other_ms",
        "component_compaction_ratio",
        "component_computation_ratio",
        "component_other_ratio",
        "component_profiled_calls",
    ]
    extra_fieldnames = sorted(
        key for row in rows for key in row if key not in set(base_fieldnames)
    )
    fieldnames = base_fieldnames + extra_fieldnames
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_dense_component_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seq_len",
        "total_ms",
        "attn_ms",
        "component_profile_attn_ms",
        "component_profile_total_ms",
        "component_contiguous_ms",
        "component_plan_ms",
        "component_kernel_ms",
        "component_upad_ms",
        "component_pad_ms",
        "component_other_ms",
        "component_contiguous_ratio",
        "component_plan_ratio",
        "component_kernel_ratio",
        "component_other_ratio",
        "component_profiled_calls",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if _world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(_local_rank)
    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

    variant = resolve_variant(args)
    validate_args(args, variant)

    seq_lens = parse_seq_lens(args.seq_lens)
    for seq_len in seq_lens:
        if seq_len % args.chunk_size != 0:
            raise ValueError(f"seq_len={seq_len} should be divisible by chunk_size={args.chunk_size}")

    dtype = get_dtype(args.dtype)
    torch.manual_seed(args.seed)

    model_id = resolve_model_id(args, variant)
    model_cfg = AutoConfig.from_pretrained(model_id)
    _enable_default_qwen_long_context(args, model_cfg)
    model, base_model = load_model(args, variant, dtype, model_id, model_cfg)
    if variant in {"flashprefill_block_sparse", "flashprefill_compactattn"}:
        _set_deferred_async_collective_wait(
            model,
            enabled=not bool(args.include_flashprefill_async_wait_in_attn),
        )
    if args.profile_dense_components:
        if variant != "dense":
            raise ValueError("--profile-dense-components is only supported for dense execution")
        if args.dense_backend != "flashinfer":
            raise ValueError("--profile-dense-components currently profiles the dense FlashInfer backend")
        _enable_dense_component_debug(model)
    if args.profile_compactattn_components:
        if variant not in {"seer_compactattn", "seer_compactattn_hf", "flashprefill_compactattn"}:
            raise ValueError("--profile-compactattn-components is only supported for compactattn execution")
        _enable_compactattn_component_debug(model)
    if args.profile_block_sparse_components:
        if variant != "flashprefill_block_sparse":
            raise ValueError("--profile-block-sparse-components is only supported for flashprefill block_sparse")
        _enable_block_sparse_component_debug(model)
    if args.profile_quoka_components:
        if variant != "quoka_dense":
            raise ValueError("--profile-quoka-components is only supported for QUOKA")
        _enable_quoka_component_debug(model)
    if is_rank0:
        print_config(args, variant, model_id, base_model, seq_lens)

    device = next(model.parameters()).device
    vocab_size = resolve_vocab_size(model_cfg, model)
    tokenizer = None if args.prompt_source == "random" else AutoTokenizer.from_pretrained(base_model, use_fast=True)
    last_block_dense_scope = resolve_last_block_dense_scope(args, variant)

    use_chunked_runner = variant != "dense" or hasattr(
        getattr(model, "config", None), "seerattn_chunked_prefill_force_dense"
    )

    profiler = AttentionProfiler(
        collect_compactattn_components=bool(args.profile_compactattn_components),
        collect_dense_components=bool(args.profile_dense_components),
        collect_block_sparse_components=bool(args.profile_block_sparse_components),
        collect_quoka_components=bool(args.profile_quoka_components),
    )
    profiler.install(model)
    component_csv_rows: list[dict[str, float]] = []
    dense_component_csv_rows: list[dict[str, float]] = []
    block_sparse_component_csv_rows: list[dict[str, float]] = []
    quoka_component_csv_rows: list[dict[str, float]] = []

    if is_rank0:
        print("\n[results]")
        print("seq_len | total_ms | attn_ms | nonattn_ms | attn_frac | chunks | attn_per_chunk_ms")
        print("--------+----------+---------+------------+-----------+--------+------------------")

    for seq_len in seq_lens:
        input_batches = None
        try:
            input_batches = build_input_batches(
                args=args,
                seq_len=seq_len,
                batch_size=args.batch_size,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                num_inputs=args.warmup + args.runs,
            )

            def _get_input(iter_idx: int) -> torch.Tensor:
                source = input_batches[0] if len(input_batches) == 1 else input_batches[iter_idx]
                return source.to(device=device, non_blocking=True)

            for idx in range(args.warmup):
                input_ids = _get_input(idx)
                try:
                    if hasattr(model, "_prepare_profiled_compactattn_cleanup"):
                        model._prepare_profiled_compactattn_cleanup()
                    profiler.reset_run()
                    if use_chunked_runner:
                        _ = run_chunked_prefill_once(
                            model,
                            input_ids,
                            args.chunk_size,
                            last_block_dense_scope=last_block_dense_scope,
                            dense_prefix_tokens=args.dense_prefix_tokens,
                        )
                    else:
                        _ = run_dense_prefill_once(model, input_ids, args.chunk_size)
                    _ = profiler.finish_run()
                finally:
                    del input_ids
                    _cleanup_after_sequence_iteration(variant)

            totals = []
            attns = []
            component_runs = []
            for idx in range(args.runs):
                input_ids = _get_input(args.warmup + idx)
                try:
                    if hasattr(model, "_prepare_profiled_compactattn_cleanup"):
                        model._prepare_profiled_compactattn_cleanup()
                    profiler.reset_run()
                    if use_chunked_runner:
                        total_ms = run_chunked_prefill_once(
                            model,
                            input_ids,
                            args.chunk_size,
                            last_block_dense_scope=last_block_dense_scope,
                            dense_prefix_tokens=args.dense_prefix_tokens,
                        )
                    else:
                        total_ms = run_dense_prefill_once(model, input_ids, args.chunk_size)
                    attn_ms = profiler.finish_run()
                    global_total_ms = _tp_global_max_ms(total_ms)
                    global_attn_ms = _tp_global_max_ms(attn_ms)
                    totals.append(global_total_ms)
                    attns.append(global_attn_ms)
                    if is_rank0:
                        global_nonattn_ms = global_total_ms - global_attn_ms
                        global_attn_frac = (
                            global_attn_ms / global_total_ms * 100.0
                            if global_total_ms > 0.0
                            else 0.0
                        )
                        print(
                            f"[run] seq_len={seq_len} run={idx + 1}/{args.runs} "
                            f"total_ms={global_total_ms:.1f} "
                            f"attn_ms={global_attn_ms:.1f} "
                            f"nonattn_ms={global_nonattn_ms:.1f} "
                            f"attn_frac={global_attn_frac:.1f}%",
                            flush=True,
                        )
                    if args.profile_compactattn_components:
                        component_snapshot = profiler.component_snapshot()
                        local_component_summary = _compactattn_component_summary(
                            attn_ms,
                            component_snapshot,
                        )
                        local_component_summary["component_profile_attn_ms"] = float(attn_ms)
                        local_component_summary["component_profile_total_ms"] = float(total_ms)
                        local_component_summary["component_profiled_calls"] = float(
                            component_snapshot.get("compactattn_profiled_calls", 0.0)
                        )
                        local_component_summary.update(
                            {
                                key: float(value)
                                for key, value in component_snapshot.items()
                                if key.startswith("col_")
                            }
                        )
                        component_runs.append(_tp_global_max_dict(local_component_summary))
                    if args.profile_dense_components:
                        local_dense_summary = _dense_component_summary(
                            attn_ms,
                            profiler.component_snapshot(),
                        )
                        local_dense_summary["component_profile_attn_ms"] = float(attn_ms)
                        local_dense_summary["component_profile_total_ms"] = float(total_ms)
                        local_dense_summary["component_profiled_calls"] = float(
                            profiler.component_snapshot().get("dense_profiled_calls", 0.0)
                        )
                        component_runs.append(_tp_global_max_dict(local_dense_summary))
                    if args.profile_block_sparse_components:
                        component_snapshot = profiler.component_snapshot()
                        local_block_summary = {
                            "component_profile_attn_ms": float(attn_ms),
                            "component_profile_total_ms": float(total_ms),
                            "component_profiled_calls": float(
                                component_snapshot.get("block_sparse_profiled_calls", 0.0)
                            ),
                        }
                        local_block_summary.update(
                            {
                                key: float(value)
                                for key, value in component_snapshot.items()
                                if key.startswith("block_sparse_")
                            }
                        )
                        component_runs.append(_tp_global_max_dict(local_block_summary))
                    if args.profile_quoka_components:
                        component_snapshot = profiler.component_snapshot()
                        local_quoka_summary = {
                            "component_profile_attn_ms": float(attn_ms),
                            "component_profile_total_ms": float(total_ms),
                            "component_profiled_calls": float(
                                component_snapshot.get("quoka_profiled_calls", 0.0)
                            ),
                        }
                        local_quoka_summary.update(
                            {
                                key: float(value)
                                for key, value in component_snapshot.items()
                                if key.startswith(("quoka_", "dense_"))
                            }
                        )
                        component_runs.append(_tp_global_max_dict(local_quoka_summary))
                finally:
                    del input_ids
                    _cleanup_after_sequence_iteration(variant)

            mean_total = statistics.mean(totals)
            mean_attn = statistics.mean(attns)
            nonattn = mean_total - mean_attn
            attn_frac = (mean_attn / mean_total * 100.0) if mean_total > 0 else 0.0
            chunks = (seq_len + args.chunk_size - 1) // args.chunk_size
            attn_per_chunk = mean_attn / chunks if chunks > 0 else 0.0
            if is_rank0:
                print(
                    f"{seq_len:7d} | {mean_total:8.1f} | {mean_attn:7.1f} | "
                    f"{nonattn:10.1f} | {attn_frac:7.1f}% | {chunks:6d} | {attn_per_chunk:16.2f}"
                )
                if args.profile_compactattn_components and component_runs:
                    component_avg = _mean_dicts(component_runs)
                    profile_attn = float(component_avg.get("component_profile_attn_ms", mean_attn))
                    compaction = float(component_avg.get("component_compaction_ms", 0.0))
                    computation = float(component_avg.get("component_computation_ms", 0.0))
                    other = float(component_avg.get("component_other_ms", 0.0))
                    denom = profile_attn if profile_attn > 0.0 else compaction + computation + other
                    component_row = {
                        "seq_len": float(seq_len),
                        "total_ms": mean_total,
                        "attn_ms": mean_attn,
                        "component_profile_attn_ms": profile_attn,
                        "component_profile_total_ms": float(
                            component_avg.get("component_profile_total_ms", mean_total)
                        ),
                        "component_compaction_ms": compaction,
                        "component_computation_ms": computation,
                        "component_other_ms": other,
                        "component_compaction_ratio": compaction / denom if denom > 0.0 else 0.0,
                        "component_computation_ratio": computation / denom if denom > 0.0 else 0.0,
                        "component_other_ratio": other / denom if denom > 0.0 else 0.0,
                        "component_profiled_calls": float(
                            component_avg.get("component_profiled_calls", 0.0)
                        ),
                    }
                    component_row.update(
                        {
                            key: float(value)
                            for key, value in component_avg.items()
                            if key.startswith("col_")
                        }
                    )
                    component_csv_rows.append(component_row)
                    print(
                        "          components | "
                        f"compaction={compaction:.1f} ms | "
                        f"computation={computation:.1f} ms | "
                        f"other={other:.1f} ms"
                    )
                if args.profile_dense_components and component_runs:
                    component_avg = _mean_dicts(component_runs)
                    profile_attn = float(component_avg.get("component_profile_attn_ms", mean_attn))
                    contiguous = float(component_avg.get("component_contiguous_ms", 0.0))
                    plan = float(component_avg.get("component_plan_ms", 0.0))
                    kernel = float(component_avg.get("component_kernel_ms", 0.0))
                    upad = float(component_avg.get("component_upad_ms", 0.0))
                    pad = float(component_avg.get("component_pad_ms", 0.0))
                    other = float(component_avg.get("component_other_ms", 0.0))
                    denom = profile_attn if profile_attn > 0.0 else contiguous + plan + kernel + upad + pad + other
                    dense_component_row = {
                        "seq_len": float(seq_len),
                        "total_ms": mean_total,
                        "attn_ms": mean_attn,
                        "component_profile_attn_ms": profile_attn,
                        "component_profile_total_ms": float(
                            component_avg.get("component_profile_total_ms", mean_total)
                        ),
                        "component_contiguous_ms": contiguous,
                        "component_plan_ms": plan,
                        "component_kernel_ms": kernel,
                        "component_upad_ms": upad,
                        "component_pad_ms": pad,
                        "component_other_ms": other,
                        "component_contiguous_ratio": contiguous / denom if denom > 0.0 else 0.0,
                        "component_plan_ratio": plan / denom if denom > 0.0 else 0.0,
                        "component_kernel_ratio": kernel / denom if denom > 0.0 else 0.0,
                        "component_other_ratio": other / denom if denom > 0.0 else 0.0,
                        "component_profiled_calls": float(
                            component_avg.get("component_profiled_calls", 0.0)
                        ),
                    }
                    dense_component_csv_rows.append(dense_component_row)
                    print(
                        "          dense components | "
                        f"contiguous={contiguous:.1f} ms | "
                        f"plan={plan:.1f} ms | "
                        f"kernel={kernel:.1f} ms | "
                        f"other={other:.1f} ms"
                    )
                if args.profile_block_sparse_components and component_runs:
                    component_avg = _mean_dicts(component_runs)
                    block_row = {
                        "seq_len": float(seq_len),
                        "total_ms": mean_total,
                        "attn_ms": mean_attn,
                        "component_profile_attn_ms": float(
                            component_avg.get("component_profile_attn_ms", mean_attn)
                        ),
                        "component_profile_total_ms": float(
                            component_avg.get("component_profile_total_ms", mean_total)
                        ),
                        "component_profiled_calls": float(
                            component_avg.get("component_profiled_calls", 0.0)
                        ),
                    }
                    block_row.update(
                        {
                            key: float(value)
                            for key, value in component_avg.items()
                            if key.startswith("block_sparse_")
                        }
                    )
                    block_sparse_component_csv_rows.append(block_row)
                    print(
                        "          block_sparse components | "
                        f"fused={component_avg.get('block_sparse_flashprefill_fused_total_ms', 0.0):.1f} ms | "
                        f"select_replay={component_avg.get('block_sparse_flashprefill_select_replay_ms', 0.0):.1f} ms | "
                        f"est_attention={component_avg.get('block_sparse_flashprefill_estimated_attention_ms', 0.0):.1f} ms"
                    )
                if args.profile_quoka_components and component_runs:
                    component_avg = _mean_dicts(component_runs)
                    quoka_row = {
                        "seq_len": float(seq_len),
                        "total_ms": mean_total,
                        "attn_ms": mean_attn,
                        "component_profile_attn_ms": float(
                            component_avg.get("component_profile_attn_ms", mean_attn)
                        ),
                        "component_profile_total_ms": float(
                            component_avg.get("component_profile_total_ms", mean_total)
                        ),
                        "component_profiled_calls": float(
                            component_avg.get("component_profiled_calls", 0.0)
                        ),
                    }
                    quoka_row.update(
                        {
                            key: float(value)
                            for key, value in component_avg.items()
                            if key.startswith(("quoka_", "dense_"))
                        }
                    )
                    quoka_component_csv_rows.append(quoka_row)
                    print(
                        "          quoka components | "
                        f"selection={component_avg.get('quoka_selection_ms', 0.0):.1f} ms | "
                        f"dense_kernel={component_avg.get('dense_kernel_ms', 0.0):.1f} ms | "
                        f"assembled_kv={component_avg.get('quoka_assembled_kv_len', 0.0):.1f}"
                    )

        except RuntimeError as e:
            msg = str(e).split("\n", 1)[0]
            if is_rank0:
                print(f"{seq_len:7d} | fail: {msg}")
        except Exception as e:
            if is_rank0:
                print(f"{seq_len:7d} | fail: {e}")
        finally:
            if input_batches is not None:
                del input_batches
            _cleanup_after_benchmark_case(variant)
            gc.collect()
            torch.cuda.empty_cache()

    profiler.uninstall()
    if is_rank0 and args.compactattn_components_csv is not None and component_csv_rows:
        _write_component_csv(args.compactattn_components_csv, component_csv_rows)
        print(f"\n[compactattn_components_csv] {args.compactattn_components_csv}")
    if is_rank0 and args.dense_components_csv is not None and dense_component_csv_rows:
        _write_dense_component_csv(args.dense_components_csv, dense_component_csv_rows)
        print(f"\n[dense_components_csv] {args.dense_components_csv}")
    if is_rank0 and args.block_sparse_components_csv is not None and block_sparse_component_csv_rows:
        _write_component_csv(args.block_sparse_components_csv, block_sparse_component_csv_rows)
        print(f"\n[block_sparse_components_csv] {args.block_sparse_components_csv}")
    if is_rank0 and args.quoka_components_csv is not None and quoka_component_csv_rows:
        _write_component_csv(args.quoka_components_csv, quoka_component_csv_rows)
        print(f"\n[quoka_components_csv] {args.quoka_components_csv}")


if __name__ == "__main__":
    main()
