#!/usr/bin/env python
import argparse
import gc
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.tensor_parallel import _get_parameter_tp_plan, shard_and_distribute_module

from compact_attn import (
    SeerAttnLlamaChunkedDenseForCausalLM,
    SeerAttnLlamaChunkedDenseHFForCausalLM,
    SeerAttnLlamaFlashPrefillCompactAttnForCausalLM,
    SeerAttnLlamaFlashPrefillForCausalLM,
    SeerAttnLlamaForCausalLM,
    load_dense_llama_model,
    SeerAttnQwen2ChunkedDenseForCausalLM,
    SeerAttnQwen2ForCausalLM,
    SeerAttnQwen3ForCausalLM,
    SeerAttnQwen3ChunkedDenseForCausalLM,
    SeerAttnQwen3MoeFlashPrefillCompactAttnForCausalLM,
    SeerAttnQwen3MoeFlashPrefillForCausalLM,
    SeerAttnGemma3ChunkedDenseForCausalLM,
    load_dense_model,
    load_quoka_model,
)
from compact_attn.prefill_sparse.llama.configuration_llama_seerattn import SeerAttnLlamaConfig
from compact_attn.load_attention_replacement import (
    load_llama_attention_replacement_model,
    load_qwen3_moe_attention_replacement_model,
)
from compact_attn.kernels.varlen.indexed_dense_prefill_varlen import clear_indexed_dense_workspaces
from compact_attn.modules.attention_forward_chunked_dense import COMPACTATTN_VERSION, clear_fast_path_content_cache
from compact_attn.prefill_sparse.llama.modeling_llama_flashprefill_compactattn import (
    HeadsFirstDynamicCache as FlashPrefillHeadsFirstDynamicCache,
)
from compact_attn.prefill_sparse.llama.modeling_llama_seerattn_chunked_dense_hf import (
    HeadsFirstDynamicCache as SeerHeadsFirstDynamicCache,
)

SEER_GATE_MODEL_ID = "SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates"
BASE_LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SEER_GLOBAL_THRESHOLD = float(os.environ.get("SEER_GLOBAL_THRESHOLD", "3e-4"))
COMPACTATTN_SA_THRESHOLD = float(os.environ.get("COMPACTATTN_SA_THRESHOLD", "5e-4"))
DEFAULT_DENSE_BACKEND = "flashinfer"

VARIANT_MAP = {
    ("dense", "none"): "dense",
    ("dense", "quoka"): "quoka_dense",
    ("block_sparse", "seer"): "seer_block_sparse",
    ("block_sparse", "flashprefill"): "flashprefill_block_sparse",
    ("compactattn", "seer"): "seer_compactattn",
    ("compactattn", "seer_hf"): "seer_compactattn_hf",
    ("compactattn", "flashprefill"): "flashprefill_compactattn",
}

_JSONL_LINE_COUNT_CACHE = {}
_FA2_CHUNKED_DENSE_PATCHED = False


def canonical_execution_mode(execution_mode: str) -> str:
    """Map public execution aliases to the internal implementation family."""
    mode = str(execution_mode)
    if mode in {"compactattention", "compact_attention"}:
        return "compactattn"
    return mode


def _default_col_indexed_impl_for_args(args) -> str:
    """Resolve the default compactattn backend without breaking non-Seer paths."""
    if canonical_execution_mode(getattr(args, "execution_mode", None)) != "compactattn":
        return "fa2_paged"
    selection_method = str(getattr(args, "selection_method", ""))
    if selection_method in {"seer", "seer_hf", "flashprefill"}:
        return "fi_zero_copy"
    return "fa2_paged"


def _is_qwen3_moe_model_arg(args) -> bool:
    model = str(
        getattr(args, "model", None)
        or getattr(args, "seer_model", None)
        or ""
    ).lower()
    return "qwen3" in model and ("a3b" in model or "moe" in model)


def _apply_qwen3_fi_zero_copy_default(args) -> None:
    if canonical_execution_mode(getattr(args, "execution_mode", None)) != "compactattn":
        return
    if str(getattr(args, "selection_method", "")) != "flashprefill":
        return
    if not _is_qwen3_moe_model_arg(args):
        return
    if str(getattr(args, "col_indexed_impl", "")) != "fi_zero_copy":
        return
    args.col_indexed_impl = "fi_zero_copy_subgroup"
    os.environ.setdefault("SEER_ZC_QUERY_SUBGROUP_SIZE", "4")


def apply_backend_defaults(args) -> None:
    if not hasattr(args, "dense_backend") or getattr(args, "dense_backend") is None:
        args.dense_backend = DEFAULT_DENSE_BACKEND
    if not hasattr(args, "col_indexed_impl") or getattr(args, "col_indexed_impl") is None:
        args.col_indexed_impl = "auto"
    if str(args.col_indexed_impl) == "auto":
        args.col_indexed_impl = _default_col_indexed_impl_for_args(args)
    _apply_qwen3_fi_zero_copy_default(args)


def _needs_sequence_iteration_cleanup(variant: str) -> bool:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # TP sweeps reuse the same sharded model instance across warmup / measured
        # samples. For long multi-batch prompts, allocator fragmentation can build
        # up across iterations even when each individual prompt would fit. Treat
        # each sample as an independent request and aggressively release cached
        # CUDA allocator state between iterations.
        return True
    return False


def _cleanup_after_sequence_iteration(variant: str) -> None:
    if not _needs_sequence_iteration_cleanup(variant):
        return
    clear_fast_path_content_cache()
    gc.collect()
    if torch.cuda.is_available():
        _synchronize_cuda_all_devices()
        torch.cuda.empty_cache()


def _needs_benchmark_case_cleanup(variant: str) -> bool:
    return False


def _synchronize_cuda_all_devices() -> None:
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_idx)


def _cleanup_after_benchmark_case(variant: str) -> None:
    if not _needs_benchmark_case_cleanup(variant):
        return
    _synchronize_cuda_all_devices()
    clear_fast_path_content_cache()
    clear_indexed_dense_workspaces()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cleanup_after_seq_len_case() -> None:
    _synchronize_cuda_all_devices()
    clear_fast_path_content_cache()
    clear_indexed_dense_workspaces()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_retryable_cuda_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    retry_markers = (
        "cuda error: unspecified launch failure",
        "cuda error: an illegal memory access was encountered",
        "cuda out of memory",
        "out of memory",
    )
    return any(marker in msg for marker in retry_markers)


def _batch_select_chunked_gate_caches(past_key_values, keep_indices: torch.Tensor) -> None:
    if past_key_values is None:
        return

    keep_indices_cpu = keep_indices.to(device="cpu")
    for attr_name in (
        "_seer_chunked_gate_k_cache",
        "_seer_compactattn_chunked_gate_k_cache",
        "_seer_rowwise_gate_k_cache",
        "_seer_rowwise_gate_k_carry",
        "_seer_compactattn_rowwise_gate_k_cache",
        "_seer_compactattn_rowwise_gate_k_carry",
    ):
        store = getattr(past_key_values, attr_name, None)
        if not isinstance(store, dict):
            continue
        for layer_idx, cached in list(store.items()):
            if torch.is_tensor(cached):
                index = keep_indices_cpu.to(device=cached.device)
                store[layer_idx] = cached.index_select(0, index).contiguous()
            elif isinstance(cached, list):
                selected = []
                for idx in keep_indices_cpu.tolist():
                    entry = cached[int(idx)]
                    if torch.is_tensor(entry):
                        selected.append(entry.contiguous())
                    else:
                        selected.append(entry)
                store[layer_idx] = selected
            elif isinstance(cached, dict):
                buffer = cached.get("buffer", None)
                length = int(cached.get("length", 0))
                if torch.is_tensor(buffer):
                    index = keep_indices_cpu.to(device=buffer.device)
                    new_buffer = buffer.index_select(0, index).contiguous()
                    store[layer_idx] = {"buffer": new_buffer, "length": length}


def _attention_mask_rows_identical(attention_mask: torch.Tensor) -> bool:
    if attention_mask.ndim != 2 or attention_mask.shape[0] <= 1:
        return True
    return bool(torch.equal(attention_mask, attention_mask[:1].expand_as(attention_mask)))


def _normalize_uniform_batch_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    context: str,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError(f"{context} expects a 2D attention mask, got shape={tuple(attention_mask.shape)}")
    if not _attention_mask_rows_identical(attention_mask):
        raise ValueError(
            f"{context} currently supports exact same-length batches only; "
            "all attention-mask rows must be identical."
        )
    if not bool((attention_mask == 0).any().item()):
        return None
    return attention_mask




def parse_args():
    p = argparse.ArgumentParser(
        description="Unified chunked prefill latency sweep with configurable execution mode and selection method"
    )
    p.add_argument(
        "--model",
        "--seer-model",
        dest="model",
        type=str,
        default=None,
    )
    p.add_argument(
        "--execution-mode",
        type=str,
        required=True,
        choices=["dense", "block_sparse", "compactattn", "compactattention", "compact_attention"],
        help=(
            "Execution family: dense, block_sparse, or compactattention. "
            "compactattn is kept as a backward-compatible alias for compactattention."
        ),
    )
    p.add_argument(
        "--selection-method",
        type=str,
        required=True,
        choices=["none", "seer", "seer_hf", "quoka", "flashprefill"],
        help="Selection/scoring family to pair with the execution mode",
    )
    p.add_argument("--seq-lens", type=str, default="8192,16384,32768,65536,131072,262144")
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
        help=(
            "Dense baseline / dense fallback backend used when the shared dense "
            "prefill helper is active."
        ),
    )
    p.add_argument(
        "--dense-class-family",
        type=str,
        default="stock",
        choices=["stock", "seer", "flashprefill"],
        help=(
            "Class family used for dense baselines. 'stock' uses the HF model "
            "with dense attention monkey-patched. 'seer' and 'flashprefill' use "
            "the corresponding custom Llama classes with sparse selection forced "
            "off, for class-semantics-matched speedup checks."
        ),
    )
    p.add_argument(
        "--attention-harness",
        type=str,
        default="legacy",
        choices=["legacy", "replacement"],
        help=(
            "Profiling harness. 'legacy' uses the historical method-specific "
            "model loaders. 'replacement' loads the stock LLaMA model and "
            "replaces only self_attn modules for supported methods."
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
            "Intermediate chunks use 0 to avoid making each chunk tail dense."
        ),
    )
    p.add_argument("--flashprefill-min-budget", type=int, default=0)

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

    p.add_argument("--prompt-source", type=str, default="random", choices=["random", "ruler"])
    p.add_argument(
        "--input-schedule",
        type=str,
        default="cycle",
        choices=["fixed", "cycle"],
        help="How to source inputs across warmup/runs. 'cycle' uses different inputs by default.",
    )
    p.add_argument("--task", type=str, default="cwe")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--data-file", type=Path, default=None)
    p.add_argument("--qwen-long-context", action="store_true")
    p.add_argument("--qwen-long-context-max-position-embeddings", type=int, default=131072)
    p.add_argument("--qwen-yarn-factor", type=float, default=4.0)
    p.add_argument("--qwen-original-max-position-embeddings", type=int, default=32768)
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto"],
        help="Default 'auto': spread the model across all visible GPUs (CUDA_VISIBLE_DEVICES).",
    )
    p.add_argument(
        "--seq-len-isolation",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Run each seq_len in an isolated child process to avoid cross-case CUDA contamination.",
    )
    p.add_argument(
        "--isolated-seq-len-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


def _should_isolate_seq_len_cases(args, variant: str, seq_lens: List[int]) -> bool:
    if args.isolated_seq_len_child or len(seq_lens) <= 1:
        return False
    if args.seq_len_isolation == "on":
        return True
    if args.seq_len_isolation == "off":
        return False
    return variant == "dense" and args.device_map == "auto" and args.batch_size > 1


def _build_isolated_seq_len_cmd(seq_len: int) -> List[str]:
    cmd = [sys.executable, str(Path(__file__).resolve())]
    argv = sys.argv[1:]
    skip_next = False
    for idx, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token in {"--seq-lens", "--seq-len-isolation"}:
            skip_next = True
            continue
        if token == "--isolated-seq-len-child":
            continue
        if token.startswith("--seq-lens=") or token.startswith("--seq-len-isolation="):
            continue
        cmd.append(token)
    cmd.extend(["--seq-lens", str(seq_len), "--isolated-seq-len-child"])
    return cmd


def _parse_isolated_result_row(stdout: str, seq_len: int) -> str | None:
    prefix = f"{seq_len} | "
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line
    return None


def _format_child_failure(seq_len: int, completed: subprocess.CompletedProcess[str], reason: str) -> str:
    tail_stdout = "\n".join(completed.stdout.splitlines()[-8:])
    tail_stderr = "\n".join(completed.stderr.splitlines()[-8:])
    details = []
    if tail_stdout:
        details.append(f"stdout_tail=\n{tail_stdout}")
    if tail_stderr:
        details.append(f"stderr_tail=\n{tail_stderr}")
    detail_str = "\n".join(details)
    if detail_str:
        return f"{seq_len} | - | - | - | - | fail: {reason}\n{detail_str}"
    return f"{seq_len} | - | - | - | - | fail: {reason}"


def _run_isolated_seq_len_case(seq_len: int) -> str:
    cmd = _build_isolated_seq_len_cmd(seq_len)
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    result_row = _parse_isolated_result_row(completed.stdout, seq_len)
    if completed.returncode == 0 and result_row is not None:
        return result_row
    reason = f"child_exit={completed.returncode}"
    if result_row is None:
        reason += ", missing result row"
    return _format_child_failure(seq_len, completed, reason)


def get_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def parse_seq_lens(seq_lens: str) -> List[int]:
    out = []
    for x in seq_lens.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("No sequence lengths provided")
    return out


def default_ruler_data_root(model: str | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    model_l = str(model or "").lower()
    if "qwen3" in model_l and ("30b" in model_l or "a3b" in model_l):
        return repo_root / "results" / "ruler_qwen3_30b_a3b_instruct_2507"
    if "gemma" in model_l and ("12b" in model_l or "3-12" in model_l):
        return repo_root / "results" / "ruler_gemma_3_12b_it"
    return repo_root / "results" / "ruler_llama_3_1_8b"


def default_data_file(seq_len: int, task: str, model: str | None = None) -> Path:
    return (
        default_ruler_data_root(model)
        / "synthetic"
        / str(seq_len)
        / "data"
        / task
        / "validation.jsonl"
    )


def load_ruler_prompt(path: Path, sample_index: int) -> str:
    if not path.exists():
        raise FileNotFoundError(f"RULER data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == sample_index:
                record = json.loads(line)
                if "input" not in record:
                    raise KeyError(f"Missing 'input' field in {path}")
                return str(record["input"])

    raise IndexError(f"sample_index={sample_index} out of range for {path}")


def count_jsonl_lines(path: Path) -> int:
    cache_key = str(path.resolve())
    cached = _JSONL_LINE_COUNT_CACHE.get(cache_key, None)
    if cached is not None:
        return int(cached)

    with path.open("r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    _JSONL_LINE_COUNT_CACHE[cache_key] = int(count)
    return int(count)


def build_input_ids(
    *,
    args,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    tokenizer,
    sample_offset: int = 0,
) -> torch.Tensor:
    if args.prompt_source == "random":
        generator = torch.Generator(device="cpu")
        base_seed = int(args.seed) + (int(seq_len) * 104729)
        if args.input_schedule == "cycle":
            base_seed += int(sample_offset)
        generator.manual_seed(base_seed)
        return torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len),
            generator=generator,
            dtype=torch.long,
            device="cpu",
        )

    data_file = args.data_file if args.data_file is not None else default_data_file(
        seq_len,
        args.task,
        getattr(args, "model", None),
    )
    data_file = Path(data_file)
    num_samples = count_jsonl_lines(data_file)
    if num_samples <= 0:
        raise ValueError(f"No samples found in {data_file}")
    base_index = int(args.sample_index)
    if args.input_schedule == "cycle":
        base_index = (base_index + int(sample_offset)) % num_samples

    def _encode_sample(idx: int) -> torch.Tensor:
        prompt = load_ruler_prompt(data_file, idx % num_samples)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
        )["input_ids"]
        if enc.shape[1] == 0:
            raise ValueError(f"Tokenized prompt is empty for {data_file} idx={idx}")
        if enc.shape[1] < seq_len:
            pad = torch.zeros(1, seq_len - enc.shape[1], dtype=enc.dtype)
            enc = torch.cat([enc, pad], dim=1)
        return enc  # (1, seq_len)

    rows = [_encode_sample(base_index + i) for i in range(batch_size)]
    encoded = torch.cat(rows, dim=0)  # (batch_size, seq_len)
    return encoded.cpu()


def build_input_batches(
    *,
    args,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    tokenizer,
    num_inputs: int,
) -> List[torch.Tensor]:
    count = 1 if args.input_schedule == "fixed" else max(int(num_inputs), 1)
    return [
        build_input_ids(
            args=args,
            seq_len=seq_len,
            batch_size=batch_size,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            sample_offset=offset,
        )
        for offset in range(count)
    ]


def resolve_vocab_size(model_cfg, model) -> int:
    for cfg in (
        model_cfg,
        getattr(model_cfg, "text_config", None),
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        vocab_size = getattr(cfg, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)
    raise AttributeError("Could not resolve vocab_size from model or config")


def maybe_patch_plain_fa2_chunked_dense():
    global _FA2_CHUNKED_DENSE_PATCHED
    if _FA2_CHUNKED_DENSE_PATCHED:
        return

    try:
        from transformers import modeling_flash_attention_utils as hf_fa_utils
    except Exception:
        return

    orig_prepare = getattr(hf_fa_utils, "_prepare_from_posids", None)
    if orig_prepare is None:
        return
    if getattr(hf_fa_utils, "_seerattn_chunked_dense_safe_patch", False):
        _FA2_CHUNKED_DENSE_PATCHED = True
        return

    def _prepare_from_posids_chunked_dense_safe(query, key, value, position_ids):
        posids = position_ids if position_ids.device == query.device else position_ids.to(query.device)
        if posids.ndim == 1:
            posids = posids.unsqueeze(0)
        if posids.ndim == 2 and posids.numel() > 0:
            bsz = int(query.shape[0])
            q_len = int(query.shape[1])
            kv_len = int(key.shape[1])
            # Dense chunked-prefill passes full, no-padding segments. On the HF
            # FA2 path we can safely recover cu-seqlens directly from row
            # boundaries even when accelerate/multi-GPU mangles position_ids.
            query = query.contiguous().view(-1, query.size(-2), query.size(-1))
            key = key.contiguous().view(-1, key.size(-2), key.size(-1))
            value = value.contiguous().view(-1, value.size(-2), value.size(-1))
            flat_q_tokens = int(bsz * q_len)
            indices_q = torch.arange(flat_q_tokens, device=query.device, dtype=torch.int32)
            cu_q = torch.arange(
                0,
                flat_q_tokens + 1,
                q_len,
                device=query.device,
                dtype=torch.int32,
            )
            flat_k_tokens = int(bsz * kv_len)
            cu_k = torch.arange(
                0,
                flat_k_tokens + 1,
                kv_len,
                device=query.device,
                dtype=torch.int32,
            )
            return (
                query,
                key,
                value,
                indices_q,
                (cu_q, cu_k),
                (int(q_len), int(kv_len)),
            )
        return orig_prepare(query, key, value, posids)

    hf_fa_utils._prepare_from_posids = _prepare_from_posids_chunked_dense_safe
    hf_fa_utils._seerattn_chunked_dense_safe_patch = True
    _FA2_CHUNKED_DENSE_PATCHED = True


def resolve_variant(args) -> str:
    requested_execution_mode = getattr(args, "execution_mode", None)
    args.requested_execution_mode = requested_execution_mode
    args.execution_mode = canonical_execution_mode(requested_execution_mode)
    apply_backend_defaults(args)
    key = (args.execution_mode, args.selection_method)
    if (
        key == ("compactattn", "seer")
        and str(getattr(args, "col_indexed_impl", "")) in {"fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"}
    ):
        return "seer_compactattn_hf"
    if key not in VARIANT_MAP:
        raise ValueError(
            "Unsupported combination: "
            f"execution_mode={args.execution_mode}, selection_method={args.selection_method}"
        )
    return VARIANT_MAP[key]


def resolve_model_id(args, variant: str) -> str:
    if args.model:
        return str(args.model)
    if variant in {"seer_block_sparse", "seer_compactattn", "seer_compactattn_hf"}:
        return SEER_GATE_MODEL_ID
    return BASE_LLAMA_MODEL_ID


def display_variant_name(variant: str) -> str:
    if variant == "seer_compactattn":
        return "compactattention"
    if variant == "seer_compactattn_hf":
        return "compactattention_hf"
    return variant


def display_selection_method(args, variant: str) -> str:
    if variant in {"seer_compactattn", "seer_compactattn_hf"} and str(args.selection_method) in {"seer", "seer_hf"}:
        return "compactattention"
    return str(args.selection_method)


def resolve_threshold_schedule(args, variant: str, model_id: str):
    # Length-aware Seer threshold schedules are deprecated. Seer selectors now use
    # fixed method defaults resolved by resolve_seer_threshold(). Keep this function
    # as an import-compatible shim.
    return None


def resolve_seer_threshold(args, variant: str) -> float:
    if getattr(args, "threshold", None) is not None:
        return float(args.threshold)
    if variant in {"seer_compactattn", "seer_compactattn_hf"}:
        return COMPACTATTN_SA_THRESHOLD
    return SEER_GLOBAL_THRESHOLD


def resolve_flashprefill_alpha(args, variant: str) -> float:
    if getattr(args, "flashprefill_alpha", None) is not None:
        return float(args.flashprefill_alpha)
    if variant == "flashprefill_compactattn":
        return 0.06
    return 0.01


def resolve_final_chunk_dense_blocks(args, variant: str) -> Optional[int]:
    value = getattr(args, "final_chunk_dense_blocks", None)
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError("--final-chunk-dense-blocks must be non-negative")
    if variant == "seer_block_sparse" and value not in {0, 2}:
        raise ValueError(
            "Seer block-sparse currently supports final-chunk dense blocks as "
            "a boolean policy implemented as the final two query blocks. Use 0 or 2."
        )
    return value


def resolve_seer_last_block_dense(args, variant: str) -> Optional[bool]:
    unified = resolve_final_chunk_dense_blocks(args, variant)
    if unified is not None and variant == "seer_block_sparse":
        return unified > 0
    explicit = getattr(args, "last_block_dense", None)
    if explicit is not None:
        return bool(explicit)
    return None


def resolve_flashprefill_last_n_block(args, variant: str) -> int:
    unified = resolve_final_chunk_dense_blocks(args, variant)
    if unified is not None and variant in {"flashprefill_block_sparse", "flashprefill_compactattn"}:
        return unified
    return int(getattr(args, "flashprefill_last_n_block", 0))


def resolve_last_block_dense_scope(args, variant: str) -> str:
    if args.last_block_dense_scope is not None:
        return args.last_block_dense_scope
    if variant in {
        "seer_block_sparse",
        "seer_compactattn",
        "seer_compactattn_hf",
        "flashprefill_block_sparse",
        "flashprefill_compactattn",
    }:
        return "final_prefill_chunk"
    return "all_prefill_chunks"


def last_dense_scope_applies(scope: str, *, chunk_is_final: bool) -> bool:
    if scope == "all_prefill_chunks":
        return True
    if scope == "final_prefill_chunk":
        return bool(chunk_is_final)
    raise ValueError(f"Unsupported last-block-dense scope: {scope}")


def resolve_final_dense_tail_blocks(args, variant: str) -> int:
    if args.final_dense_tail_blocks is not None:
        return int(args.final_dense_tail_blocks)
    unified = resolve_final_chunk_dense_blocks(args, variant)
    if unified is not None and variant in {"seer_compactattn", "seer_compactattn_hf"}:
        return unified
    if variant in {"seer_compactattn", "seer_compactattn_hf"}:
        return 2
    return 0


def build_chunk_ranges(total_tokens: int, chunk_size: int, min_chunk_tokens: int):
    if total_tokens <= 0:
        return []
    if total_tokens <= chunk_size:
        return [(0, total_tokens)]
    chunk_lengths = []
    remaining = int(total_tokens)
    while remaining > 0:
        take = min(chunk_size, remaining)
        chunk_lengths.append(take)
        remaining -= take
    while len(chunk_lengths) >= 2 and chunk_lengths[-1] < min_chunk_tokens:
        chunk_lengths[-2] += chunk_lengths[-1]
        chunk_lengths.pop()
    out = []
    start = 0
    for length in chunk_lengths:
        end = start + int(length)
        out.append((start, end))
        start = end
    return out


def _primary_model_device(model: torch.nn.Module) -> torch.device:
    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embed_tokens is not None and hasattr(embed_tokens, "weight"):
        return embed_tokens.weight.device
    return next(model.parameters()).device


def _is_qwen_model_cfg(model_cfg) -> bool:
    return str(getattr(model_cfg, "model_type", "")).lower() in {"qwen2", "qwen3", "qwen3_moe"}


def _is_qwen3_model_cfg(model_cfg) -> bool:
    return str(getattr(model_cfg, "model_type", "")).lower() == "qwen3"


def _is_qwen3_moe_model_cfg(model_cfg) -> bool:
    return str(getattr(model_cfg, "model_type", "")).lower() == "qwen3_moe"


def _is_llama_model_cfg(model_cfg) -> bool:
    return str(getattr(model_cfg, "model_type", "")).lower() == "llama"


def _is_gemma3_model_cfg(model_cfg) -> bool:
    return str(getattr(model_cfg, "model_type", "")).lower() in {"gemma3", "gemma3_text"}


def _enable_default_qwen_long_context(args, model_cfg) -> None:
    # Qwen latency runs should default to the same YaRN setup we use in our
    # benchmark scripts, unless the caller already enabled it explicitly.
    if bool(getattr(args, "qwen_long_context", False)):
        return
    # Qwen3-MoE 2507 models are native long-context models.  Do not silently
    # retrofit the dense-Qwen YaRN defaults onto them.
    if _is_qwen3_moe_model_cfg(model_cfg):
        return
    if _is_qwen_model_cfg(model_cfg):
        args.qwen_long_context = True


def _build_qwen_long_context_config(args, model_id: str):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cfg.max_position_embeddings = int(args.qwen_long_context_max_position_embeddings)
    cfg.rope_scaling = {
        "rope_type": "yarn",
        "factor": float(args.qwen_yarn_factor),
        "original_max_position_embeddings": int(args.qwen_original_max_position_embeddings),
    }
    setattr(cfg, "base_model", str(model_id))
    return cfg


def _set_runtime_flag(model, name: str, value):
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, name):
        setattr(cfg, name, value)
    for layer in getattr(getattr(model, "model", None), "layers", []):
        self_attn = getattr(layer, "self_attn", None)
        attn_cfg = getattr(self_attn, "config", None)
        if attn_cfg is not None and hasattr(attn_cfg, name):
            setattr(attn_cfg, name, value)


def _set_runtime_threshold(model, value: float) -> None:
    value = float(value)
    _set_runtime_flag(model, "seerattn_threshold", value)
    for layer in getattr(getattr(model, "model", None), "layers", []):
        self_attn = getattr(layer, "self_attn", None)
        attn_cfg = getattr(self_attn, "config", None)
        if attn_cfg is not None and hasattr(attn_cfg, "seerattn_threshold"):
            setattr(attn_cfg, "seerattn_threshold", value)


def _dense_chunk_ranges(seq_len: int, chunk_size: int) -> list[tuple[int, int]]:
    return build_chunk_ranges(seq_len, chunk_size, chunk_size)


def _dense_forward_with_cache(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    past_seen_tokens: int,
):
    bsz, cur_chunk = input_ids.shape
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + cur_chunk,
        device=input_ids.device,
        dtype=torch.long,
    )
    position_ids = cache_position.unsqueeze(0).expand(bsz, -1)
    return model(
        input_ids=input_ids,
        attention_mask=None,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
        position_ids=position_ids,
        cache_position=cache_position,
    )


def _dense_generation_forward_with_cache(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    past_seen_tokens: int,
):
    return _dense_forward_with_cache(
        model,
        input_ids,
        past_key_values,
        past_seen_tokens,
    )


def run_dense_prefill_once(model, input_ids: torch.Tensor, chunk_size: int) -> float:
    _, seq_len = input_ids.shape
    past = None
    out = None
    past_seen = 0
    _synchronize_cuda_all_devices()
    t0 = time.perf_counter()
    with torch.no_grad():
        for start, end in _dense_chunk_ranges(seq_len, chunk_size):
            out = _dense_forward_with_cache(
                model,
                input_ids[:, start:end],
                past,
                past_seen,
            )
            past = out.past_key_values
            past_seen += end - start
    _synchronize_cuda_all_devices()
    elapsed = (time.perf_counter() - t0) * 1000.0
    del past, out  # break ModelOutput→DynamicCache ref chain before caller's gc.collect/empty_cache
    return elapsed


def _new_chunked_prefill_cache(model):
    layout = str(getattr(getattr(model, "config", None), "seerattn_cache_layout", "dynamic"))
    if layout == "heads_first_seer":
        return SeerHeadsFirstDynamicCache()
    if layout == "heads_first_flashprefill":
        return FlashPrefillHeadsFirstDynamicCache()
    return None


def _build_replacement_block_attention_mask(
    *,
    attention_mask: torch.Tensor | None,
    batch_size: int,
    query_len: int,
    kv_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if query_len == 1 or block_size <= 0:
        return None
    if attention_mask is None:
        full_mask = torch.ones((batch_size, kv_len), dtype=torch.bool, device=device)
    else:
        full_mask = attention_mask.to(device=device, dtype=torch.bool)
        kv_len = int(full_mask.shape[-1])

    query_mask = full_mask[:, -query_len:]
    query_valid_blocks = F.max_pool1d(
        query_mask.unsqueeze(1).to(torch.float32),
        kernel_size=block_size,
        stride=block_size,
        ceil_mode=True,
    ).squeeze(1).to(torch.bool)
    key_valid_blocks = F.max_pool1d(
        full_mask.unsqueeze(1).to(torch.float32),
        kernel_size=block_size,
        stride=block_size,
        ceil_mode=True,
    ).squeeze(1).to(torch.bool)

    q_blocks = int(query_valid_blocks.shape[-1])
    k_blocks = int(key_valid_blocks.shape[-1])
    valid_q_lens = query_mask.sum(dim=-1, dtype=torch.int64)
    valid_k_lens = full_mask.sum(dim=-1, dtype=torch.int64)
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


def _build_replacement_block_position_embeddings(
    model,
    position_ids: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    block_rotary_emb = getattr(getattr(model, "model", None), "block_rotary_emb", None)
    if block_rotary_emb is None or block_size <= 0:
        return None
    block_position_ids = position_ids[:, 0::block_size]
    try:
        ref = next(model.parameters())
    except StopIteration:
        return None
    dummy = torch.empty((), device=block_position_ids.device, dtype=ref.dtype)
    return block_rotary_emb(dummy, block_position_ids)


def run_chunked_prefill_once(
    model,
    input_ids: torch.Tensor,
    chunk_size: int,
    *,
    last_block_dense_scope: str,
    dense_prefix_tokens: int,
    attention_mask: torch.Tensor | None = None,
) -> float:
    _, seq_len = input_ids.shape
    attention_mask = _normalize_uniform_batch_attention_mask(
        attention_mask,
        context="chunked prefill runtime",
    )
    cfg = getattr(model, "config", None)
    gate_block_size = int(getattr(cfg, "seerattn_gate_block_size", 1))
    seer_threshold_default = float(getattr(cfg, "seerattn_threshold", SEER_GLOBAL_THRESHOLD))
    seer_threshold_schedule = None
    last_block_dense_default = bool(getattr(cfg, "seerattn_last_block_dense", False))
    force_dense_default = bool(getattr(cfg, "seerattn_chunked_prefill_force_dense", False))
    flashprefill_last_n_block_default = int(
        getattr(cfg, "seerattn_flashprefill_last_n_block", 0)
    )
    final_dense_tail_blocks = int(
        0
        if force_dense_default
        else getattr(cfg, "seerattn_chunked_prefill_final_dense_tail_blocks", 0)
    )

    def _run_segment(
        segment_input_ids: torch.Tensor,
        segment_attention_mask: torch.Tensor | None,
        start: int,
        end: int,
        past,
        past_seen: int,
        *,
        last_block_dense: bool,
        force_dense: bool,
        flashprefill_last_n_block: int,
    ):
        cur_chunk = end - start
        segment_bsz = segment_input_ids.shape[0]
        cache_position = torch.arange(
            past_seen,
            past_seen + cur_chunk,
            device=input_ids.device,
            dtype=torch.long,
        )
        if segment_attention_mask is not None:
            if not (segment_attention_mask == 0).any():
                segment_attention_mask = None
                position_ids = cache_position.unsqueeze(0).expand(segment_bsz, -1)
            else:
                position_ids = segment_attention_mask.to(torch.long).cumsum(-1) - 1
                position_ids.masked_fill_(segment_attention_mask == 0, 0)
                position_ids = position_ids[:, -cur_chunk:]
        else:
            position_ids = cache_position.unsqueeze(0).expand(segment_bsz, -1)
        _set_runtime_threshold(model, seer_threshold_default)
        _set_runtime_flag(model, "seerattn_last_block_dense", bool(last_block_dense))
        _set_runtime_flag(model, "seerattn_chunked_prefill_force_dense", bool(force_dense))
        _set_runtime_flag(
            model,
            "seerattn_flashprefill_last_n_block",
            int(flashprefill_last_n_block),
        )
        call_kwargs = dict(
            input_ids=segment_input_ids,
            attention_mask=segment_attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        if bool(
            getattr(
                getattr(model, "config", None),
                "seerattn_replacement_needs_block_gate_metadata",
                False,
            )
        ):
            block_attention_mask = _build_replacement_block_attention_mask(
                attention_mask=segment_attention_mask,
                batch_size=segment_bsz,
                query_len=cur_chunk,
                kv_len=end,
                block_size=gate_block_size,
                device=input_ids.device,
            )
            block_position_embeddings = _build_replacement_block_position_embeddings(
                model,
                position_ids,
                gate_block_size,
            )
            call_kwargs["block_attention_mask"] = block_attention_mask
            call_kwargs["block_position_embeddings"] = block_position_embeddings
        try:
            return model(**call_kwargs)
        finally:
            _set_runtime_flag(model, "seerattn_last_block_dense", last_block_dense_default)
            _set_runtime_flag(model, "seerattn_chunked_prefill_force_dense", force_dense_default)
            _set_runtime_flag(
                model,
                "seerattn_flashprefill_last_n_block",
                flashprefill_last_n_block_default,
            )

    past = _new_chunked_prefill_cache(model)
    past_seen = 0
    dense_prefix_tokens = max(0, int(dense_prefix_tokens))
    _synchronize_cuda_all_devices()
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            chunk_start = start
            end = min(start + chunk_size, seq_len)
            if attention_mask is not None:
                if not bool(attention_mask[0, start:end].any().item()):
                    break
                current_input_ids = input_ids[:, start:end]
                current_attention_mask = attention_mask[:, :end]
            else:
                current_input_ids = input_ids[:, start:end]
                current_attention_mask = None

            if start < dense_prefix_tokens:
                dense_end = min(end, dense_prefix_tokens)
                dense_input_ids = current_input_ids[:, : dense_end - start]
                dense_attention_mask = (
                    current_attention_mask[:, :dense_end] if current_attention_mask is not None else None
                )
                out = _run_segment(
                    dense_input_ids,
                    dense_attention_mask,
                    start,
                    dense_end,
                    past,
                    past_seen,
                    last_block_dense=False,
                    force_dense=True,
                    flashprefill_last_n_block=0,
                )
                past = out.past_key_values
                past_seen += dense_end - start
                start = dense_end
                if start >= end:
                    continue

            chunk_is_final = end == seq_len
            chunk_len = end - start
            if final_dense_tail_blocks > 0 and chunk_is_final:
                tail_tokens = min(chunk_len, final_dense_tail_blocks * gate_block_size)
                prefix_end = end - tail_tokens
                if 0 < (prefix_end - start) < gate_block_size:
                    prefix_end = start
                if prefix_end > start:
                    prefix_offset = start - chunk_start
                    out = _run_segment(
                        current_input_ids[:, prefix_offset : prefix_offset + (prefix_end - start)],
                        current_attention_mask[:, :prefix_end] if current_attention_mask is not None else None,
                        start,
                        prefix_end,
                        past,
                        past_seen,
                        last_block_dense=False,
                        force_dense=False,
                        flashprefill_last_n_block=0,
                    )
                    past = out.past_key_values
                    past_seen += prefix_end - start
                tail_offset = prefix_end - chunk_start
                out = _run_segment(
                    current_input_ids[:, tail_offset : tail_offset + (end - prefix_end)],
                    current_attention_mask,
                    prefix_end,
                    end,
                    past,
                    past_seen,
                    last_block_dense=False,
                    force_dense=True,
                    flashprefill_last_n_block=flashprefill_last_n_block_default,
                )
                past = out.past_key_values
                past_seen += end - prefix_end
                continue

            scope_applies = last_dense_scope_applies(
                last_block_dense_scope,
                chunk_is_final=chunk_is_final,
            )
            use_last_block_dense = (
                False
                if final_dense_tail_blocks > 0
                else last_block_dense_default and scope_applies
            )
            use_flashprefill_last_n_block = (
                flashprefill_last_n_block_default if scope_applies else 0
            )

            segment_offset = start - chunk_start
            out = _run_segment(
                current_input_ids[:, segment_offset : segment_offset + chunk_len],
                current_attention_mask,
                start,
                end,
                past,
                past_seen,
                last_block_dense=use_last_block_dense,
                force_dense=force_dense_default,
                flashprefill_last_n_block=use_flashprefill_last_n_block,
            )
            past = out.past_key_values
            past_seen += chunk_len

    _synchronize_cuda_all_devices()
    elapsed = (time.perf_counter() - t0) * 1000.0
    del past, out
    return elapsed


class ChunkedPrefillGenerationRunner:
    def __init__(
        self,
        model,
        tokenizer,
        chunk_size: int,
        max_new_tokens: int,
        last_block_dense_scope: str,
        dense_prefix_tokens: int,
        *,
        drop_full_attention_mask_for_dense: bool = False,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.max_new_tokens = int(max_new_tokens)
        self.last_block_dense_scope = str(last_block_dense_scope)
        self.dense_prefix_tokens = max(0, int(dense_prefix_tokens))
        self.drop_full_attention_mask_for_dense = bool(drop_full_attention_mask_for_dense)
        self.min_prefill_tokens = max(
            1,
            int(getattr(getattr(self.model, "config", None), "seerattn_gate_block_size", 1)),
        )
        self.device = next(model.parameters()).device
        self.has_last_block_dense = hasattr(getattr(self.model, "config", None), "seerattn_last_block_dense")
        self.last_block_dense_default = bool(
            getattr(getattr(self.model, "config", None), "seerattn_last_block_dense", False)
        )
        self.has_force_dense = hasattr(
            getattr(self.model, "config", None), "seerattn_chunked_prefill_force_dense"
        )
        self.force_dense_default = bool(
            getattr(getattr(self.model, "config", None), "seerattn_chunked_prefill_force_dense", False)
        )
        self.has_flashprefill_last_n_block = hasattr(
            getattr(self.model, "config", None), "seerattn_flashprefill_last_n_block"
        )
        self.flashprefill_last_n_block_default = int(
            getattr(getattr(self.model, "config", None), "seerattn_flashprefill_last_n_block", 0)
        )
        self.final_dense_tail_blocks = int(
            0
            if self.force_dense_default
            else getattr(
                getattr(self.model, "config", None),
                "seerattn_chunked_prefill_final_dense_tail_blocks",
                0,
            )
        )
        forward_sig = inspect.signature(self.model.forward)
        self.supports_logits_to_keep = "logits_to_keep" in forward_sig.parameters
        self.supports_cache_position = "cache_position" in forward_sig.parameters
        self.supports_position_ids = "position_ids" in forward_sig.parameters
        self.stop_markers = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            self.eos_token_ids = {eos_token_id}
        elif eos_token_id is None:
            self.eos_token_ids = set()
        else:
            self.eos_token_ids = set(int(x) for x in eos_token_id)

    def _decode_generated_batch(self, generated: torch.Tensor) -> list[str]:
        texts: list[str] = []
        for row in generated:
            tokens = []
            for tok in row.tolist():
                if tok in self.eos_token_ids:
                    break
                tokens.append(tok)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            for marker in self.stop_markers:
                text = text.split(marker)[0]
            texts.append(text.strip())
        return texts

    def _batch_has_uniform_prompt_lengths(self, attention_mask: torch.Tensor) -> bool:
        return _attention_mask_rows_identical(attention_mask)

    def _normalize_uniform_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return _normalize_uniform_batch_attention_mask(
            attention_mask,
            context="chunked multi-batch generation",
        )

    @torch.inference_mode()
    def _generate_from_encoded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[str]:
        prompt_len = int(input_ids.shape[1])
        batch_size = int(input_ids.shape[0])
        attention_mask = self._normalize_uniform_attention_mask(attention_mask)
        chunk_ranges = build_chunk_ranges(prompt_len, self.chunk_size, self.min_prefill_tokens)
        if (
            (self.has_last_block_dense or self.has_force_dense)
            and len(chunk_ranges) >= 2
            and (chunk_ranges[-1][1] - chunk_ranges[-1][0]) < self.chunk_size
        ):
            chunk_ranges[-2:] = [(chunk_ranges[-2][0], chunk_ranges[-1][1])]
        past = _new_chunked_prefill_cache(self.model)
        outputs = None
        seen = 0
        try:
            for chunk_idx, (start, end) in enumerate(chunk_ranges):
                if start < self.dense_prefix_tokens:
                    dense_end = min(end, self.dense_prefix_tokens)
                    outputs = self.run_prefill_segment(
                        input_ids=input_ids[:, start:dense_end],
                        attention_mask=None if attention_mask is None else attention_mask[:, :dense_end],
                        past_key_values=past,
                        past_seen_tokens=seen,
                        last_block_dense=False,
                        force_dense=True,
                        flashprefill_last_n_block=0,
                    )
                    past = outputs.past_key_values
                    seen += dense_end - start
                    start = dense_end
                    if start >= end:
                        continue
                chunk_is_final = chunk_idx == (len(chunk_ranges) - 1)
                chunk_len = end - start
                if self.final_dense_tail_blocks > 0 and chunk_is_final:
                    tail_tokens = self.final_dense_tail_blocks * self.min_prefill_tokens
                    tail_len = min(chunk_len, tail_tokens)
                    prefix_end = end - tail_len
                    if 0 < (prefix_end - start) < self.min_prefill_tokens:
                        prefix_end = start
                    if prefix_end > start:
                        outputs = self.run_prefill_segment(
                            input_ids=input_ids[:, start:prefix_end],
                            attention_mask=None if attention_mask is None else attention_mask[:, :prefix_end],
                            past_key_values=past,
                            past_seen_tokens=seen,
                            last_block_dense=False,
                            force_dense=False,
                            flashprefill_last_n_block=0,
                        )
                        past = outputs.past_key_values
                        seen += prefix_end - start
                    outputs = self.run_prefill_segment(
                        input_ids=input_ids[:, prefix_end:end],
                        attention_mask=None if attention_mask is None else attention_mask[:, :end],
                        past_key_values=past,
                        past_seen_tokens=seen,
                        last_block_dense=False,
                        force_dense=True,
                        flashprefill_last_n_block=self.flashprefill_last_n_block_default,
                    )
                    past = outputs.past_key_values
                    seen += end - prefix_end
                    continue

                scope_applies = last_dense_scope_applies(
                    self.last_block_dense_scope,
                    chunk_is_final=chunk_is_final,
                )
                use_last_block_dense = (
                    False
                    if self.final_dense_tail_blocks > 0
                    else self.last_block_dense_default and scope_applies
                )
                outputs = self.run_prefill_segment(
                    input_ids=input_ids[:, start:end],
                    attention_mask=None if attention_mask is None else attention_mask[:, :end],
                    past_key_values=past,
                    past_seen_tokens=seen,
                    last_block_dense=use_last_block_dense,
                    force_dense=self.force_dense_default,
                    flashprefill_last_n_block=(
                        self.flashprefill_last_n_block_default if scope_applies else 0
                    ),
                )
                past = outputs.past_key_values
                seen += end - start
            if outputs is None:
                return [""] * batch_size
            generated_ids = []
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_ids.append(next_token)
            finished = torch.zeros((batch_size,), dtype=torch.bool, device=next_token.device)
            for eos_id in self.eos_token_ids:
                finished |= next_token.eq(int(eos_id))
            if bool(finished.all().item()):
                generated = torch.stack(generated_ids, dim=1)
                return self._decode_generated_batch(generated)

            decode_attention_mask = attention_mask
            for _ in range(1, self.max_new_tokens):
                if decode_attention_mask is not None:
                    decode_attention_mask = torch.cat(
                        [
                            decode_attention_mask,
                            torch.ones(
                                (decode_attention_mask.shape[0], 1),
                                device=self.device,
                                dtype=decode_attention_mask.dtype,
                            ),
                        ],
                        dim=-1,
                    )
                step_ids = generated_ids[-1].unsqueeze(1)
                outputs = self.forward_with_cache(
                    input_ids=step_ids,
                    attention_mask=decode_attention_mask,
                    past_key_values=past,
                    past_seen_tokens=seen,
                )
                past = outputs.past_key_values
                seen += 1
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                generated_ids.append(next_token)
                for eos_id in self.eos_token_ids:
                    finished |= next_token.eq(int(eos_id))
                if bool(finished.all().item()):
                    break
            generated = torch.stack(generated_ids, dim=1)
            return self._decode_generated_batch(generated)
        finally:
            self.set_last_block_dense(self.last_block_dense_default)
            self.set_force_dense(self.force_dense_default)
            self.set_flashprefill_last_n_block(self.flashprefill_last_n_block_default)

    def set_last_block_dense(self, enabled: bool):
        if self.has_last_block_dense:
            _set_runtime_flag(self.model, "seerattn_last_block_dense", bool(enabled))

    def set_force_dense(self, enabled: bool):
        if self.has_force_dense:
            _set_runtime_flag(self.model, "seerattn_chunked_prefill_force_dense", bool(enabled))

    def set_flashprefill_last_n_block(self, value: int):
        if self.has_flashprefill_last_n_block:
            _set_runtime_flag(self.model, "seerattn_flashprefill_last_n_block", int(value))

    def forward_with_cache(self, input_ids, attention_mask, past_key_values, past_seen_tokens, *, logits_to_keep: int = 1):
        attention_mask = self._normalize_uniform_attention_mask(attention_mask)
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        cache_position = None
        if self.supports_cache_position:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
            kwargs["cache_position"] = cache_position
        if self.supports_position_ids:
            if cache_position is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                position_ids = position_ids[:, -input_ids.shape[1] :]
            else:
                if attention_mask is not None and (attention_mask == 0).any():
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 0)
                    position_ids = position_ids[:, -input_ids.shape[1] :]
                else:
                    position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
            kwargs["position_ids"] = position_ids
        else:
            position_ids = None
        if bool(
            getattr(
                getattr(self.model, "config", None),
                "seerattn_replacement_needs_block_gate_metadata",
                False,
            )
        ):
            query_len = int(input_ids.shape[1])
            if attention_mask is None:
                kv_len = past_seen_tokens + query_len
            else:
                kv_len = int(attention_mask.shape[-1])
            block_size = int(getattr(getattr(self.model, "config", None), "seerattn_gate_block_size", 1))
            kwargs["block_attention_mask"] = _build_replacement_block_attention_mask(
                attention_mask=attention_mask,
                batch_size=int(input_ids.shape[0]),
                query_len=query_len,
                kv_len=kv_len,
                block_size=block_size,
                device=input_ids.device,
            )
            if position_ids is not None:
                kwargs["block_position_embeddings"] = _build_replacement_block_position_embeddings(
                    self.model,
                    position_ids,
                    block_size,
                )
        if self.supports_logits_to_keep:
            kwargs["logits_to_keep"] = logits_to_keep
        return self.model(**kwargs)

    def run_prefill_segment(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        past_seen_tokens,
        *,
        last_block_dense: bool,
        force_dense: bool,
        flashprefill_last_n_block: int,
        logits_to_keep: int = 1,
    ):
        attention_mask = self._normalize_uniform_attention_mask(attention_mask)
        self.set_last_block_dense(last_block_dense)
        self.set_force_dense(force_dense)
        self.set_flashprefill_last_n_block(flashprefill_last_n_block)
        try:
            return self.forward_with_cache(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                past_seen_tokens=past_seen_tokens,
                logits_to_keep=logits_to_keep,
            )
        except Exception as exc:
            mask_shape = None if attention_mask is None else tuple(attention_mask.shape)
            mask_device = None if attention_mask is None else str(attention_mask.device)
            mask_has_zeros = False if attention_mask is None else bool((attention_mask == 0).any().item())
            raise RuntimeError(
                "Chunked prefill segment failed: "
                f"input_shape={tuple(input_ids.shape)}, "
                f"input_device={input_ids.device}, "
                f"attention_mask_shape={mask_shape}, "
                f"attention_mask_device={mask_device}, "
                f"attention_mask_has_zeros={mask_has_zeros}, "
                f"past_seen_tokens={past_seen_tokens}, "
                f"last_block_dense={bool(last_block_dense)}, "
                f"force_dense={bool(force_dense)}, "
                f"flashprefill_last_n_block={int(flashprefill_last_n_block)}"
            ) from exc
        finally:
            self.set_last_block_dense(self.last_block_dense_default)
            self.set_force_dense(self.force_dense_default)
            self.set_flashprefill_last_n_block(self.flashprefill_last_n_block_default)

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        return self._generate_from_encoded(input_ids, attention_mask)[0]

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0])]

        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        if not self._batch_has_uniform_prompt_lengths(attention_mask):
            # Mixed-length chunked generation remains experimental. For now, keep
            # multi-batch generation on the exact, same-length path only.
            return [self.generate(prompt) for prompt in prompts]
        return self._generate_from_encoded(input_ids, attention_mask)


class DenseChunkedPrefillRunner:
    def __init__(self, model, tokenizer, chunk_size: int, max_new_tokens: int):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.max_new_tokens = int(max_new_tokens)
        self.device = _primary_model_device(model)
        self.use_tp = bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
        self.stop_markers = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            self.eos_token_ids = {eos_token_id}
        elif eos_token_id is None:
            self.eos_token_ids = set()
        else:
            self.eos_token_ids = set(int(x) for x in eos_token_id)

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        if self.use_tp:
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = generated[0, input_ids.shape[1] :]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            for marker in self.stop_markers:
                text = text.split(marker)[0]
            return text.strip()
        prompt_len = int(input_ids.shape[1])
        past = None
        outputs = None
        seen = 0

        for start, end in _dense_chunk_ranges(prompt_len, self.chunk_size):
            outputs = _dense_generation_forward_with_cache(
                self.model,
                input_ids[:, start:end],
                past,
                seen,
            )
            past = outputs.past_key_values
            seen += end - start

        if outputs is None:
            return ""

        generated_ids = []
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_ids.append(next_token)
        if int(next_token.item()) in self.eos_token_ids:
            return self.tokenizer.decode(next_token, skip_special_tokens=True)

        for _ in range(1, self.max_new_tokens):
            step_ids = generated_ids[-1].unsqueeze(1)
            outputs = _dense_generation_forward_with_cache(
                self.model,
                step_ids,
                past,
                seen,
            )
            past = outputs.past_key_values
            seen += 1
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_ids.append(next_token)
            if int(next_token.item()) in self.eos_token_ids:
                break

        generated = torch.stack(generated_ids, dim=1)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        for marker in self.stop_markers:
            text = text.split(marker)[0]
        return text.strip()


def benchmark(
    variant: str,
    model,
    input_batches: List[torch.Tensor],
    device,
    chunk_size: int,
    warmup: int,
    runs: int,
    *,
    last_block_dense_scope: str,
    dense_prefix_tokens: int,
):
    use_chunked_runner = variant != "dense" or hasattr(
        getattr(model, "config", None), "seerattn_chunked_prefill_force_dense"
    )
    run_once = run_chunked_prefill_once if use_chunked_runner else run_dense_prefill_once
    if not input_batches:
        raise ValueError("input_batches must not be empty")

    def _get_input(iter_idx: int) -> torch.Tensor:
        source = input_batches[0] if len(input_batches) == 1 else input_batches[iter_idx]
        return source.to(device=device, non_blocking=True)

    for idx in range(warmup):
        input_ids = _get_input(idx)
        try:
            if not use_chunked_runner:
                _ = run_once(model, input_ids, chunk_size)
            else:
                _ = run_once(
                    model,
                    input_ids,
                    chunk_size,
                    last_block_dense_scope=last_block_dense_scope,
                    dense_prefix_tokens=dense_prefix_tokens,
                )
        finally:
            del input_ids
            _cleanup_after_sequence_iteration(variant)

    latencies = []
    actual_lens = []
    for idx in range(runs):
        input_ids = _get_input(warmup + idx)
        try:
            actual_lens.append(int(input_ids.shape[1]))
            if not use_chunked_runner:
                latencies.append(run_once(model, input_ids, chunk_size))
            else:
                latencies.append(
                    run_once(
                        model,
                        input_ids,
                        chunk_size,
                        last_block_dense_scope=last_block_dense_scope,
                        dense_prefix_tokens=dense_prefix_tokens,
                    )
                )
        finally:
            del input_ids
            _cleanup_after_sequence_iteration(variant)

    mean_ms = statistics.mean(latencies)
    std_ms = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    return latencies, mean_ms, std_ms, actual_lens


def _to_device(model, device_map):
    """Move model to GPU unless device_map='auto' already handled placement."""
    return model if device_map == "auto" else model.cuda()


def _uses_tp(device_map: str | None) -> bool:
    return bool(
        device_map == "auto"
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def _dense_uses_tp(device_map: str | None) -> bool:
    return _uses_tp(device_map)


def _quoka_uses_tp(device_map: str | None) -> bool:
    return _uses_tp(device_map)


def _seer_uses_tp(device_map: str | None) -> bool:
    return _uses_tp(device_map)


def _tp_local_tensor(param: torch.Tensor) -> torch.Tensor:
    if hasattr(param, "to_local"):
        return param.to_local()
    data = getattr(param, "data", param)
    if hasattr(data, "to_local"):
        return data.to_local()
    return data


def _reshard_model_parameters_for_tp(model) -> None:
    device_mesh = getattr(model, "_device_mesh", None)
    if device_mesh is None:
        return

    tp_plan = dict(getattr(model, "_tp_plan", None) or {})
    tp_plan.update(getattr(type(model), "_tp_plan", {}) or {})
    if not tp_plan:
        return

    rank = int(device_mesh.get_local_rank())
    named_parameters = list(model.named_parameters())
    for param_name, param in named_parameters:
        if "attn_gate." in param_name:
            continue
        if _get_parameter_tp_plan(param_name, tp_plan) is None:
            continue
        if hasattr(param, "to_local") or hasattr(getattr(param, "data", None), "to_local"):
            continue

        full_param = _tp_local_tensor(param).detach()
        shard_and_distribute_module(
            model,
            full_param,
            param,
            param_name,
            param.dtype,
            full_param.is_contiguous(),
            rank,
            device_mesh,
        )


def _slice_multihead_linear_heads(linear, start_head: int, end_head: int, target_device: torch.device) -> None:
    if linear is None:
        return
    weight = getattr(linear, "weight", None)
    if weight is None:
        return
    sliced = (
        weight.detach()
        .narrow(0, int(start_head), int(end_head - start_head))
        .contiguous()
        .to(device=target_device)
    )
    linear.weight = torch.nn.Parameter(sliced, requires_grad=weight.requires_grad)
    linear.num_head = int(end_head - start_head)


def _localize_seer_gate_for_tp(attn_layer, rank: int) -> None:
    attn_gate = getattr(attn_layer, "attn_gate", None)
    if attn_gate is None:
        return

    head_dim = int(getattr(attn_layer, "head_dim"))
    q_proj = getattr(attn_layer, "q_proj", None)
    k_proj = getattr(attn_layer, "k_proj", None)
    if q_proj is None or k_proj is None:
        return

    local_q_weight = _tp_local_tensor(q_proj.weight)
    local_kv_weight = _tp_local_tensor(k_proj.weight)
    local_q_heads = int(local_q_weight.shape[0]) // head_dim
    local_kv_heads = int(local_kv_weight.shape[0]) // head_dim
    full_q_heads = int(attn_gate.num_q_head)
    full_kv_heads = int(attn_gate.num_k_head)
    if local_q_heads <= 0 or local_kv_heads <= 0:
        return

    q_start = rank * local_q_heads
    kv_start = rank * local_kv_heads
    target_device = local_q_weight.device

    if local_q_heads == full_q_heads and local_kv_heads == full_kv_heads:
        attn_gate.to(target_device)
        return

    mask_linear_q = getattr(attn_gate, "mask_linear_q", None)
    if mask_linear_q is not None:
        if bool(getattr(attn_gate, "kv_group_aware_query", False)):
            _slice_multihead_linear_heads(mask_linear_q, kv_start, kv_start + local_kv_heads, target_device)
        else:
            _slice_multihead_linear_heads(mask_linear_q, q_start, q_start + local_q_heads, target_device)

    mask_linear_k = getattr(attn_gate, "mask_linear_k", None)
    if mask_linear_k is not None:
        current_heads = int(mask_linear_k.weight.shape[0])
        if current_heads == full_q_heads:
            _slice_multihead_linear_heads(mask_linear_k, q_start, q_start + local_q_heads, target_device)
        elif current_heads == full_kv_heads:
            _slice_multihead_linear_heads(mask_linear_k, kv_start, kv_start + local_kv_heads, target_device)
        else:
            raise RuntimeError(
                "Unsupported attn_gate.mask_linear_k head layout under TP: "
                f"current={current_heads} full_q={full_q_heads} full_kv={full_kv_heads}"
            )

    attn_gate.num_q_head = local_q_heads
    attn_gate.num_k_head = local_kv_heads
    attn_gate.num_key_value_groups = local_q_heads // local_kv_heads
    attn_gate.to(target_device)


def _localize_seer_gates_for_tp(model) -> None:
    if not dist.is_initialized():
        return
    rank = int(dist.get_rank())
    _reshard_model_parameters_for_tp(model)
    for layer in getattr(getattr(model, "model", None), "layers", []):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None:
            _localize_seer_gate_for_tp(self_attn, rank)


def load_model(args, variant: str, dtype: torch.dtype, model_id: str, model_cfg) -> Tuple[torch.nn.Module, str]:
    base_model = getattr(model_cfg, "base_model", model_id)
    is_qwen = _is_qwen_model_cfg(model_cfg)
    is_qwen3 = _is_qwen3_model_cfg(model_cfg)
    is_qwen3_moe = _is_qwen3_moe_model_cfg(model_cfg)
    is_gemma3 = _is_gemma3_model_cfg(model_cfg)
    qwen_long_context_cfg = None
    if is_qwen and bool(args.qwen_long_context):
        qwen_long_context_cfg = _build_qwen_long_context_config(args, base_model)
    if is_qwen3:
        seer_block_sparse_cls = SeerAttnQwen3ForCausalLM
    elif is_qwen:
        seer_block_sparse_cls = SeerAttnQwen2ForCausalLM
    elif is_gemma3:
        seer_block_sparse_cls = SeerAttnGemma3ChunkedDenseForCausalLM
    else:
        seer_block_sparse_cls = SeerAttnLlamaForCausalLM
    if is_qwen3:
        seer_compactattn_cls = SeerAttnQwen3ChunkedDenseForCausalLM
    elif is_qwen:
        seer_compactattn_cls = SeerAttnQwen2ChunkedDenseForCausalLM
    elif is_gemma3:
        seer_compactattn_cls = SeerAttnGemma3ChunkedDenseForCausalLM
    else:
        seer_compactattn_cls = SeerAttnLlamaChunkedDenseForCausalLM

    device_map = getattr(args, "device_map", None)

    if str(getattr(args, "attention_harness", "legacy")) == "replacement":
        llama_replacement_variants = {
            "seer_block_sparse",
            "seer_compactattn",
            "seer_compactattn_hf",
            "flashprefill_block_sparse",
            "flashprefill_compactattn",
        }
        qwen3_moe_replacement_variants = {
            "quoka_dense",
            "flashprefill_block_sparse",
            "flashprefill_compactattn",
        }
        use_replacement_loader = (
            (is_qwen3_moe and variant in qwen3_moe_replacement_variants)
            or (_is_llama_model_cfg(model_cfg) and variant in llama_replacement_variants)
        )
        if use_replacement_loader:
            if not (_is_llama_model_cfg(model_cfg) or is_qwen3_moe):
                raise ValueError(
                    "--attention-harness replacement is currently supported only for "
                    "Llama models and Qwen3-MoE attention variants"
                )
            replacement_kwargs = dict(
                torch_dtype=dtype,
                dense_backend=args.dense_backend,
                threshold=(
                    resolve_flashprefill_alpha(args, variant)
                    if variant.startswith("flashprefill")
                    else resolve_seer_threshold(args, variant)
                ),
                last_block_dense=(
                    seer_last_block_dense
                    if (seer_last_block_dense := resolve_seer_last_block_dense(args, variant)) is not None
                    else variant.startswith("seer_")
                ),
                final_dense_tail_blocks=resolve_final_dense_tail_blocks(args, variant),
                compactattn_keep_recent_blocks=args.compactattn_keep_recent_blocks,
                compactattn_disable_first_chunk_dense=args.compactattn_disable_first_chunk_dense,
                compactattn_chunked_gate_head_pool=args.compactattn_chunked_gate_head_pool,
                col_pack_impl=args.col_pack_impl,
                col_indexed_impl=args.col_indexed_impl,
                col_cache_fill_backend=args.col_cache_fill_backend,
                flashprefill_alpha=resolve_flashprefill_alpha(args, variant),
                flashprefill_block_size=int(args.flashprefill_block_size),
                flashprefill_attention_sink=int(args.flashprefill_attention_sink),
                flashprefill_window_size=int(args.flashprefill_window_size),
                flashprefill_last_n_block=resolve_flashprefill_last_n_block(args, variant),
                flashprefill_min_budget=int(args.flashprefill_min_budget),
            )
            if _uses_tp(device_map):
                replacement_kwargs["tp_plan"] = "auto"
            else:
                replacement_kwargs["device_map"] = device_map
            if is_qwen3_moe:
                qwen_replacement_kwargs = dict(replacement_kwargs)
                qwen_replacement_kwargs.pop("threshold", None)
                qwen_replacement_kwargs.pop("last_block_dense", None)
                qwen_replacement_kwargs.pop("compactattn_chunked_gate_head_pool", None)
                qwen_replacement_kwargs.update(
                    qwen_long_context=bool(args.qwen_long_context),
                    qwen_long_context_max_position_embeddings=int(
                        args.qwen_long_context_max_position_embeddings
                    ),
                    qwen_yarn_factor=float(args.qwen_yarn_factor),
                    qwen_original_max_position_embeddings=int(
                        args.qwen_original_max_position_embeddings
                    ),
                    quoka_query_ratio=float(args.quoka_query_ratio),
                    quoka_kv_budget_ratio=float(args.quoka_kv_budget_ratio),
                )
                model = load_qwen3_moe_attention_replacement_model(
                    base_model=base_model,
                    variant=variant,
                    **qwen_replacement_kwargs,
                )
            else:
                model = load_llama_attention_replacement_model(
                    model_id=model_id,
                    base_model=base_model,
                    variant=variant,
                    **replacement_kwargs,
                )
            if _uses_tp(device_map):
                _reshard_model_parameters_for_tp(model)
                _localize_seer_gates_for_tp(model)
            return _to_device(model, device_map), base_model

    if variant == "dense":
        dense_class_family = str(getattr(args, "dense_class_family", "stock"))
        if dense_class_family != "stock":
            if not _is_llama_model_cfg(model_cfg):
                raise ValueError(
                    f"--dense-class-family {dense_class_family} is currently supported only for Llama models"
                )
            dense_backend = str(
                getattr(args, "dense_backend", getattr(model_cfg, "seerattn_dense_backend", DEFAULT_DENSE_BACKEND))
            )
            dense_force_config = SeerAttnLlamaConfig.from_pretrained(base_model)
            setattr(dense_force_config, "seerattn_chunked_prefill_force_dense", True)
            setattr(dense_force_config, "seerattn_dense_backend", dense_backend)
            setattr(dense_force_config, "use_cache", True)
            if _uses_tp(device_map):
                setattr(dense_force_config, "fused_norm", False)

            dense_force_load_kwargs = dict(
                torch_dtype=dtype,
            )
            if _uses_tp(device_map):
                dense_force_load_kwargs["tp_plan"] = "auto"
            else:
                dense_force_load_kwargs["device_map"] = device_map

            if dense_class_family == "seer":
                model = SeerAttnLlamaForCausalLM.from_pretrained(
                    base_model,
                    load_gate=False,
                    config=dense_force_config,
                    **dense_force_load_kwargs,
                ).eval()
            elif dense_class_family == "flashprefill":
                flashprefill_force_kwargs = dict(dense_force_load_kwargs)
                flashprefill_force_kwargs["seerattn_chunked_prefill_force_dense"] = True
                if _uses_tp(device_map):
                    flashprefill_force_kwargs["fused_norm"] = False
                model = SeerAttnLlamaFlashPrefillForCausalLM.from_pretrained(
                    base_model,
                    seerattn_dense_backend=dense_backend,
                    **flashprefill_force_kwargs,
                ).eval()
            else:
                raise ValueError(f"Unhandled dense_class_family={dense_class_family}")
            return _to_device(model, device_map), base_model

        if args.dense_attn_impl == "flash_attention_2":
            dense_backend = str(
                getattr(args, "dense_backend", getattr(model_cfg, "seerattn_dense_backend", DEFAULT_DENSE_BACKEND))
            )
            dense_kwargs = dict(
                torch_dtype=dtype,
                dense_backend=dense_backend,
            )
            if qwen_long_context_cfg is not None:
                dense_kwargs["config"] = qwen_long_context_cfg
            if _dense_uses_tp(device_map):
                dense_kwargs["tp_plan"] = "auto"
            else:
                dense_kwargs["device_map"] = device_map
            try:
                model = load_dense_model(base_model, **dense_kwargs).eval()
            except ValueError:
                if not _is_llama_model_cfg(model_cfg):
                    raise
                model = load_dense_llama_model(base_model, **dense_kwargs).eval()
            return _to_device(model, device_map), base_model
        if args.dense_attn_impl == "flash_attention_2":
            maybe_patch_plain_fa2_chunked_dense()
        dense_kwargs = dict(
            attn_implementation=args.dense_attn_impl,
            config=qwen_long_context_cfg,
            torch_dtype=dtype,
        )
        if _dense_uses_tp(device_map):
            dense_kwargs["tp_plan"] = "auto"
        else:
            dense_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(base_model, **dense_kwargs).eval()
        return _to_device(model, device_map), base_model

    if is_qwen3_moe and variant in {"seer_block_sparse", "seer_compactattn", "seer_compactattn_hf"}:
        raise ValueError(
            "Qwen3-MoE Seer-gate variants are not implemented. "
            "Use --selection-method flashprefill for FlashPrefill or CompactAttention(FP)."
        )

    if variant == "seer_block_sparse":
        _seer_bs_kwargs = dict(
            load_gate=True,
            seerattn_sparsity_method="threshold",
            use_cache=True,
            seerattn_chunked_prefill_force_dense=False,
            seerattn_dense_backend=args.dense_backend,
        )
        if is_gemma3:
            _seer_bs_kwargs["seerattn_gemma3_execution_mode"] = "block_sparse"
        seer_last_block_dense = resolve_seer_last_block_dense(args, variant)
        if seer_last_block_dense is not None:
            _seer_bs_kwargs["seerattn_last_block_dense"] = bool(seer_last_block_dense)
        if qwen_long_context_cfg is not None:
            _seer_bs_kwargs["config"] = qwen_long_context_cfg
        _seer_bs_load_kwargs = dict(
            seerattn_threshold=resolve_seer_threshold(args, variant),
            seerattn_chunked_prefill_final_dense_tail_blocks=resolve_final_dense_tail_blocks(args, variant),
            torch_dtype=dtype,
        )
        if _seer_uses_tp(device_map):
            _seer_bs_load_kwargs["tp_plan"] = "auto"
            _seer_bs_kwargs["fused_norm"] = False
        else:
            _seer_bs_load_kwargs["device_map"] = device_map
        model = seer_block_sparse_cls.from_pretrained(
            model_id,
            **_seer_bs_kwargs,
            **_seer_bs_load_kwargs,
        ).eval()
        setattr(model.config, "seerattn_threshold", resolve_seer_threshold(args, variant))
        setattr(model.config, "seerattn_threshold_schedule", None)
        if _seer_uses_tp(device_map):
            _localize_seer_gates_for_tp(model)
        return _to_device(model, device_map), base_model

    if variant == "seer_compactattn":
        compactattn_threshold_schedule = None
        _seer_cd_kwargs = dict(
            load_gate=True,
            seerattn_sparsity_method="threshold",
            seerattn_compactattn_threshold_schedule=compactattn_threshold_schedule,
            use_cache=True,
            seerattn_chunked_prefill_force_dense=False,
            seerattn_dense_backend=args.dense_backend,
        )
        if is_gemma3:
            _seer_cd_kwargs["seerattn_gemma3_execution_mode"] = "compactattn"
        seer_last_block_dense = resolve_seer_last_block_dense(args, variant)
        if seer_last_block_dense is not None:
            _seer_cd_kwargs["seerattn_last_block_dense"] = bool(seer_last_block_dense)
        if qwen_long_context_cfg is not None:
            _seer_cd_kwargs["config"] = qwen_long_context_cfg
        _seer_cd_load_kwargs = dict(
            seerattn_threshold=resolve_seer_threshold(args, variant),
            seerattn_compactattn_threshold=resolve_seer_threshold(args, variant),
            seerattn_compactattn_keep_recent_blocks=args.compactattn_keep_recent_blocks,
            seerattn_compactattn_disable_first_chunk_dense=args.compactattn_disable_first_chunk_dense,
            seerattn_compactattn_chunked_gate_head_pool=args.compactattn_chunked_gate_head_pool,
            seerattn_compactattn_pack_impl=args.col_pack_impl,
            seerattn_compactattn_indexed_impl=args.col_indexed_impl,
            seerattn_compactattn_cache_fill_backend=args.col_cache_fill_backend,
            seerattn_chunked_prefill_final_dense_tail_blocks=resolve_final_dense_tail_blocks(args, variant),
            torch_dtype=dtype,
        )
        if _seer_uses_tp(device_map):
            _seer_cd_load_kwargs["tp_plan"] = "auto"
            _seer_cd_kwargs["fused_norm"] = False
        else:
            _seer_cd_load_kwargs["device_map"] = device_map
        model = seer_compactattn_cls.from_pretrained(
            model_id,
            **_seer_cd_kwargs,
            **_seer_cd_load_kwargs,
        ).eval()
        model.config.seerattn_threshold = resolve_seer_threshold(args, variant)
        model.config.seerattn_compactattn_threshold = resolve_seer_threshold(args, variant)
        model.config.seerattn_compactattn_threshold_schedule = None
        for layer in getattr(getattr(model, "model", None), "layers", []):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "compactattn_threshold_schedule"):
                self_attn.compactattn_threshold_schedule = None
        if _seer_uses_tp(device_map):
            _localize_seer_gates_for_tp(model)
        return _to_device(model, device_map), base_model

    if variant == "seer_compactattn_hf":
        compactattn_threshold_schedule = None
        _seer_cd_kwargs = dict(
            load_gate=True,
            seerattn_sparsity_method="threshold",
            seerattn_compactattn_threshold_schedule=compactattn_threshold_schedule,
            use_cache=True,
            seerattn_chunked_prefill_force_dense=False,
            seerattn_dense_backend=args.dense_backend,
        )
        if is_gemma3:
            _seer_cd_kwargs["seerattn_gemma3_execution_mode"] = "compactattn"
        seer_last_block_dense = resolve_seer_last_block_dense(args, variant)
        if seer_last_block_dense is not None:
            _seer_cd_kwargs["seerattn_last_block_dense"] = bool(seer_last_block_dense)
        if qwen_long_context_cfg is not None:
            _seer_cd_kwargs["config"] = qwen_long_context_cfg
        _seer_cd_load_kwargs = dict(
            seerattn_threshold=resolve_seer_threshold(args, variant),
            seerattn_compactattn_threshold=resolve_seer_threshold(args, variant),
            seerattn_compactattn_keep_recent_blocks=args.compactattn_keep_recent_blocks,
            seerattn_compactattn_disable_first_chunk_dense=args.compactattn_disable_first_chunk_dense,
            seerattn_compactattn_chunked_gate_head_pool=args.compactattn_chunked_gate_head_pool,
            seerattn_compactattn_pack_impl=args.col_pack_impl,
            seerattn_compactattn_indexed_impl=args.col_indexed_impl,
            seerattn_compactattn_cache_fill_backend=args.col_cache_fill_backend,
            seerattn_chunked_prefill_final_dense_tail_blocks=resolve_final_dense_tail_blocks(args, variant),
            torch_dtype=dtype,
        )
        if _seer_uses_tp(device_map):
            _seer_cd_load_kwargs["tp_plan"] = "auto"
            _seer_cd_kwargs["fused_norm"] = False
        else:
            _seer_cd_load_kwargs["device_map"] = device_map
        seer_cd_hf_cls = seer_compactattn_cls if is_gemma3 else SeerAttnLlamaChunkedDenseHFForCausalLM
        model = seer_cd_hf_cls.from_pretrained(
            model_id,
            **_seer_cd_kwargs,
            **_seer_cd_load_kwargs,
        ).eval()
        model.config.seerattn_threshold = resolve_seer_threshold(args, variant)
        model.config.seerattn_compactattn_threshold = resolve_seer_threshold(args, variant)
        model.config.seerattn_compactattn_threshold_schedule = None
        for layer in getattr(getattr(model, "model", None), "layers", []):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "compactattn_threshold_schedule"):
                self_attn.compactattn_threshold_schedule = None
        if _seer_uses_tp(device_map):
            _localize_seer_gates_for_tp(model)
        return _to_device(model, device_map), base_model

    if variant == "flashprefill_block_sparse":
        _fp_kwargs = dict(
            use_cache=True,
            seerattn_chunked_prefill_force_dense=False,
            seerattn_dense_backend=args.dense_backend,
            seerattn_flashprefill_alpha=resolve_flashprefill_alpha(args, variant),
            seerattn_flashprefill_block_size=int(args.flashprefill_block_size),
            seerattn_flashprefill_attention_sink=int(args.flashprefill_attention_sink),
            seerattn_flashprefill_window_size=int(args.flashprefill_window_size),
            seerattn_flashprefill_last_n_block=resolve_flashprefill_last_n_block(args, variant),
            seerattn_flashprefill_min_budget=int(args.flashprefill_min_budget),
            seerattn_defer_async_collective_wait=True,
            seerattn_chunked_prefill_final_dense_tail_blocks=resolve_final_dense_tail_blocks(args, variant),
            torch_dtype=dtype,
        )
        if _uses_tp(device_map):
            _fp_kwargs["tp_plan"] = "auto"
            _fp_kwargs["fused_norm"] = False
        else:
            _fp_kwargs["device_map"] = device_map
        fp_cls = SeerAttnQwen3MoeFlashPrefillForCausalLM if is_qwen3_moe else SeerAttnLlamaFlashPrefillForCausalLM
        if qwen_long_context_cfg is not None:
            _fp_kwargs["config"] = qwen_long_context_cfg
        model = fp_cls.from_pretrained(
            base_model,
            **_fp_kwargs,
        ).eval()
        model.config.seerattn_gate_block_size = int(args.flashprefill_block_size)
        return _to_device(model, device_map), base_model

    if variant == "flashprefill_compactattn":
        _fp_cd_kwargs = dict(
            use_cache=True,
            seerattn_chunked_prefill_force_dense=False,
            seerattn_dense_backend=args.dense_backend,
            seerattn_flashprefill_alpha=resolve_flashprefill_alpha(args, variant),
            seerattn_flashprefill_block_size=int(args.flashprefill_block_size),
            seerattn_flashprefill_attention_sink=int(args.flashprefill_attention_sink),
            seerattn_flashprefill_window_size=int(args.flashprefill_window_size),
            seerattn_flashprefill_last_n_block=resolve_flashprefill_last_n_block(args, variant),
            seerattn_flashprefill_min_budget=int(args.flashprefill_min_budget),
            seerattn_defer_async_collective_wait=True,
            seerattn_compactattn_pack_impl=args.col_pack_impl,
            seerattn_compactattn_indexed_impl=args.col_indexed_impl,
            seerattn_compactattn_cache_fill_backend=args.col_cache_fill_backend,
            seerattn_chunked_prefill_final_dense_tail_blocks=resolve_final_dense_tail_blocks(args, variant),
            torch_dtype=dtype,
        )
        if _uses_tp(device_map):
            _fp_cd_kwargs["tp_plan"] = "auto"
            _fp_cd_kwargs["fused_norm"] = False
        else:
            _fp_cd_kwargs["device_map"] = device_map
        fp_cd_cls = (
            SeerAttnQwen3MoeFlashPrefillCompactAttnForCausalLM
            if is_qwen3_moe
            else SeerAttnLlamaFlashPrefillCompactAttnForCausalLM
        )
        if qwen_long_context_cfg is not None:
            _fp_cd_kwargs["config"] = qwen_long_context_cfg
        model = fp_cd_cls.from_pretrained(
            base_model,
            **_fp_cd_kwargs,
        ).eval()
        model.config.seerattn_gate_block_size = int(args.flashprefill_block_size)
        return _to_device(model, device_map), base_model

    if variant == "quoka_dense":
        _quoka_kwargs = dict(
            torch_dtype=dtype,
            query_ratio=args.quoka_query_ratio,
            kv_budget_ratio=args.quoka_kv_budget_ratio,
            seerattn_dense_backend=args.dense_backend,
        )
        if _quoka_uses_tp(device_map):
            _quoka_kwargs["tp_plan"] = "auto"
        else:
            _quoka_kwargs["device_map"] = device_map
        if qwen_long_context_cfg is not None:
            _quoka_kwargs["qwen_long_context"] = True
            _quoka_kwargs["qwen_long_context_max_position_embeddings"] = int(
                args.qwen_long_context_max_position_embeddings
            )
            _quoka_kwargs["qwen_yarn_factor"] = float(args.qwen_yarn_factor)
            _quoka_kwargs["qwen_original_max_position_embeddings"] = int(
                args.qwen_original_max_position_embeddings
            )
        model = load_quoka_model(model_id, **_quoka_kwargs)
        return _to_device(model, device_map), base_model

    raise ValueError(f"Unhandled variant: {variant}")


def validate_args(args, variant: str):
    if args.quoka_query_ratio < 0:
        raise ValueError("--quoka-query-ratio must be non-negative")
    if args.quoka_kv_budget_ratio < 0:
        raise ValueError("--quoka-kv-budget-ratio must be non-negative")
    if (
        args.execution_mode == "compactattn"
        and args.col_indexed_impl in {"fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"}
        and variant not in {"seer_compactattn_hf", "flashprefill_compactattn"}
    ):
        raise ValueError(
            f"--col-indexed-impl {args.col_indexed_impl} requires "
            "--selection-method seer_hf or flashprefill so the heads-first KV cache is available"
        )


def print_config(args, variant: str, model_id: str, base_model: str, seq_lens: List[int]):
    print(f"model={model_id}")
    print(f"execution_mode={args.execution_mode}")
    print(f"selection_method={display_selection_method(args, variant)}")
    print(f"variant={display_variant_name(variant)}")
    if variant == "dense":
        print(f"base_model={base_model}")
    if args.execution_mode == "compactattn":
        print(f"compactattn_version={COMPACTATTN_VERSION}")
        print(f"compactattn_chunked_gate_head_pool={args.compactattn_chunked_gate_head_pool}")

    cfg_parts = [
        f"seq_lens={seq_lens}",
        f"chunk={args.chunk_size}",
        f"batch={args.batch_size}",
        f"dtype={args.dtype}",
        f"warmup={args.warmup}",
        f"runs={args.runs}",
        f"dense_prefix_tokens={args.dense_prefix_tokens}",
        f"prompt_source={args.prompt_source}",
        f"input_schedule={args.input_schedule}",
        f"attention_harness={getattr(args, 'attention_harness', 'legacy')}",
    ]
    if args.prompt_source == "ruler":
        default_data = default_data_file(seq_lens[0], args.task, model_id)
        cfg_parts.extend(
            [
                f"task={args.task}",
                f"sample_index={args.sample_index}",
                f"data_file={args.data_file}" if args.data_file is not None else "data_file=default_per_model_seq_len",
                f"default_data_root={default_data.parents[4]}",
            ]
        )

    if variant == "dense":
        cfg_parts.append(f"dense_attn_impl={args.dense_attn_impl}")
        cfg_parts.append(f"dense_backend={args.dense_backend}")
        cfg_parts.append(f"dense_class_family={getattr(args, 'dense_class_family', 'stock')}")
    if variant in {"seer_block_sparse", "seer_compactattn", "seer_compactattn_hf"}:
        cfg_parts.append(f"threshold={resolve_seer_threshold(args, variant)}")
    if variant == "seer_block_sparse":
        cfg_parts.append(f"last_block_dense_scope={resolve_last_block_dense_scope(args, variant)}")
    final_chunk_dense_blocks = resolve_final_chunk_dense_blocks(args, variant)
    if final_chunk_dense_blocks is not None:
        cfg_parts.append(f"final_chunk_dense_blocks={final_chunk_dense_blocks}")
    if variant in {"seer_block_sparse", "seer_compactattn", "seer_compactattn_hf"}:
        cfg_parts.append(f"dense_backend={args.dense_backend}")
    if variant in {"seer_compactattn", "seer_compactattn_hf", "flashprefill_compactattn"}:
        cfg_parts.extend(
            [
                f"keep_recent={args.compactattn_keep_recent_blocks}",
                f"disable_first_chunk_dense={args.compactattn_disable_first_chunk_dense}",
                f"pack={args.col_pack_impl}",
                f"indexed={args.col_indexed_impl}",
                f"cache_fill={args.col_cache_fill_backend}",
                f"final_dense_tail_blocks={resolve_final_dense_tail_blocks(args, variant)}",
            ]
        )
    if variant in {"flashprefill_block_sparse", "flashprefill_compactattn"}:
        cfg_parts.extend(
            [
                f"flashprefill_alpha={resolve_flashprefill_alpha(args, variant)}",
                f"flashprefill_block_size={args.flashprefill_block_size}",
                f"flashprefill_attention_sink={args.flashprefill_attention_sink}",
                f"flashprefill_window_size={args.flashprefill_window_size}",
                f"flashprefill_last_n_block={resolve_flashprefill_last_n_block(args, variant)}",
                f"flashprefill_last_n_block_scope={resolve_last_block_dense_scope(args, variant)}",
                f"flashprefill_min_budget={args.flashprefill_min_budget}",
                f"final_dense_tail_blocks={resolve_final_dense_tail_blocks(args, variant)}",
            ]
        )
    if args.qwen_long_context:
        cfg_parts.extend(
            [
                f"qwen_long_context={args.qwen_long_context}",
                f"qwen_long_context_max_position_embeddings={args.qwen_long_context_max_position_embeddings}",
                f"qwen_yarn_factor={args.qwen_yarn_factor}",
                f"qwen_original_max_position_embeddings={args.qwen_original_max_position_embeddings}",
            ]
        )
    if variant == "quoka_dense":
        cfg_parts.extend(
            [
                f"quoka_query_ratio={args.quoka_query_ratio}",
                f"quoka_kv_budget_ratio={args.quoka_kv_budget_ratio}",
                f"dense_backend={args.dense_backend}",
            ]
        )
    print("cfg=" + ",".join(cfg_parts))


def format_actual_lens(actual_lens: List[int]) -> str:
    if not actual_lens:
        return "-"
    if len(set(actual_lens)) == 1:
        return str(actual_lens[0])
    return f"{min(actual_lens)}~{max(actual_lens)}"


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    variant = resolve_variant(args)
    validate_args(args, variant)

    seq_lens = parse_seq_lens(args.seq_lens)
    for seq_len in seq_lens:
        if seq_len % args.chunk_size != 0:
            raise ValueError(f"seq_len={seq_len} should be divisible by chunk_size={args.chunk_size}")

    model_id = resolve_model_id(args, variant)
    model_cfg = AutoConfig.from_pretrained(model_id)
    _enable_default_qwen_long_context(args, model_cfg)
    base_model = getattr(model_cfg, "base_model", model_id)
    print_config(args, variant, model_id, base_model, seq_lens)

    if _should_isolate_seq_len_cases(args, variant, seq_lens):
        print("[seq-len-isolation] enabled")
        print("\n[results]")
        print("seq_len | actual_len | latencies_ms | mean_ms | std_ms | status")
        print("--------+------------+--------------+---------+--------+--------")
        for seq_len in seq_lens:
            print(_run_isolated_seq_len_case(seq_len))
        return

    dtype = get_dtype(args.dtype)
    torch.manual_seed(args.seed)

    model, base_model = load_model(args, variant, dtype, model_id, model_cfg)

    device = next(model.parameters()).device
    vocab_size = resolve_vocab_size(model_cfg, model)
    tokenizer = None if args.prompt_source == "random" else AutoTokenizer.from_pretrained(base_model, use_fast=True)

    print("\n[results]")
    print("seq_len | actual_len | latencies_ms | mean_ms | std_ms | status")
    print("--------+------------+--------------+---------+--------+--------")
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
            try:
                latencies, mean_ms, std_ms, actual_lens = benchmark(
                    variant=variant,
                    model=model,
                    input_batches=input_batches,
                    device=device,
                    chunk_size=args.chunk_size,
                    warmup=args.warmup,
                    runs=args.runs,
                    last_block_dense_scope=resolve_last_block_dense_scope(args, variant),
                    dense_prefix_tokens=args.dense_prefix_tokens,
                )
            except RuntimeError as e:
                if variant != "dense" or not _is_retryable_cuda_error(e):
                    raise
                print(f"[retry] dense seq_len={seq_len} after transient CUDA failure: {str(e).splitlines()[0]}")
                _cleanup_after_seq_len_case()
                latencies, mean_ms, std_ms, actual_lens = benchmark(
                    variant=variant,
                    model=model,
                    input_batches=input_batches,
                    device=device,
                    chunk_size=args.chunk_size,
                    warmup=args.warmup,
                    runs=args.runs,
                    last_block_dense_scope=resolve_last_block_dense_scope(args, variant),
                    dense_prefix_tokens=args.dense_prefix_tokens,
                )
            lat_str = ",".join(f"{x:.2f}" for x in latencies)
            actual_len = format_actual_lens(actual_lens)
            print(f"{seq_len} | {actual_len} | {lat_str} | {mean_ms:.2f} | {std_ms:.2f} | ok")
        except RuntimeError as e:
            msg = str(e).split("\n", 1)[0]
            print(f"{seq_len} | - | - | - | - | fail: {msg}")
        except Exception as e:
            print(f"{seq_len} | - | - | - | - | fail: {e}")
        finally:
            if input_batches is not None:
                del input_batches
            _cleanup_after_benchmark_case(variant)
            _cleanup_after_seq_len_case()


if __name__ == "__main__":
    main()
