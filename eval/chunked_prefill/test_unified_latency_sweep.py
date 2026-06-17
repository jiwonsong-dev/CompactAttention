#!/usr/bin/env python
import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.chunked_prefill.chunked_prefill_runtime import (
    DEFAULT_DENSE_BACKEND,
    _cleanup_after_benchmark_case,
    _cleanup_after_seq_len_case,
    _enable_default_qwen_long_context,
    benchmark,
    build_input_batches,
    display_selection_method,
    display_variant_name,
    get_dtype,
    load_model,
    parse_seq_lens,
    print_config,
    resolve_last_block_dense_scope,
    resolve_model_id,
    resolve_vocab_size,
    resolve_variant,
    validate_args,
)


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
            "compactattn remains as a backward-compatible alias."
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
    )
    p.add_argument(
        "--last-block-dense-scope",
        type=str,
        default=None,
        choices=["all_prefill_chunks", "final_prefill_chunk"],
    )
    p.add_argument("--dense-prefix-tokens", type=int, default=0)
    p.add_argument("--final-dense-tail-blocks", type=int, default=None)
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
    p.add_argument("--flashprefill-alpha", type=float, default=0.05)
    p.add_argument("--flashprefill-block-size", type=int, default=128)
    p.add_argument("--flashprefill-attention-sink", type=int, default=2)
    p.add_argument("--flashprefill-window-size", type=int, default=4)
    p.add_argument("--flashprefill-last-n-block", type=int, default=2)
    p.add_argument("--flashprefill-min-budget", type=int, default=0)
    p.add_argument("--prompt-source", type=str, default="ruler", choices=["random", "ruler"])
    p.add_argument(
        "--input-schedule",
        type=str,
        default="fixed",
        choices=["fixed", "cycle"],
        help="How to source inputs across warmup/runs. 'cycle' uses different inputs by default.",
    )
    p.add_argument("--task", type=str, default="vt")
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
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
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
    for token in argv:
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
    completed = subprocess.run(
        _build_isolated_seq_len_cmd(seq_len),
        capture_output=True,
        text=True,
    )
    result_row = _parse_isolated_result_row(completed.stdout, seq_len)
    if completed.returncode == 0 and result_row is not None:
        return result_row
    reason = f"child_exit={completed.returncode}"
    if result_row is None:
        reason += ", missing result row"
    return _format_child_failure(seq_len, completed, reason)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # Initialize the default process group so _uses_tp() / is_rank0 work correctly
    # before load_model is called. torchrun sets WORLD_SIZE/LOCAL_RANK/RANK env vars
    # but does not call init_process_group; we must do it here.
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

    if _should_isolate_seq_len_cases(args, variant, seq_lens):
        print(f"model={resolve_model_id(args, variant)}")
        print(f"execution_mode={args.execution_mode}")
        print(f"selection_method={display_selection_method(args, variant)}")
        print(f"variant={display_variant_name(variant)}")
        print(
            "config="
            f"seq_lens={seq_lens}, chunk={args.chunk_size}, batch={args.batch_size}, "
            f"dtype={args.dtype}, warmup={args.warmup}, runs={args.runs}, "
            f"dense_prefix_tokens={args.dense_prefix_tokens}, prompt_source={args.prompt_source}, "
            f"input_schedule={args.input_schedule}, seq_len_isolation=enabled"
        )
        print("\n[results]")
        print("seq_len | mean_ms | std_ms | actual_len | batch | status")
        print("------- | ------- | ------ | ---------- | ----- | ------")
        for seq_len in seq_lens:
            print(_run_isolated_seq_len_case(seq_len), flush=True)
        return

    model_id = resolve_model_id(args, variant)
    model_cfg = AutoConfig.from_pretrained(model_id)
    _enable_default_qwen_long_context(args, model_cfg)
    model, base_model = load_model(args, variant, dtype, model_id, model_cfg)
    if is_rank0:
        print_config(args, variant, model_id, base_model, seq_lens)

    device = next(model.parameters()).device
    vocab_size = resolve_vocab_size(model_cfg, model)
    tokenizer = None if args.prompt_source == "random" else AutoTokenizer.from_pretrained(base_model, use_fast=True)
    last_block_dense_scope = resolve_last_block_dense_scope(args, variant)

    if is_rank0:
        print("\n[results]")
        print("seq_len | mean_ms | std_ms | actual_len | batch | status")
        print("------- | ------- | ------ | ---------- | ----- | ------")

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
                _, mean_ms, std_ms, actual_lens = benchmark(
                    variant=variant,
                    model=model,
                    input_batches=input_batches,
                    device=device,
                    chunk_size=args.chunk_size,
                    warmup=args.warmup,
                    runs=args.runs,
                    last_block_dense_scope=last_block_dense_scope,
                    dense_prefix_tokens=args.dense_prefix_tokens,
                )
            except RuntimeError as exc:
                if _should_isolate_seq_len_cases(args, variant, [seq_len]):
                    raise
                if not getattr(args, "isolated_seq_len_child", False) and str(exc):
                    from eval.chunked_prefill.chunked_prefill_runtime import _is_retryable_cuda_error

                    if _is_retryable_cuda_error(exc):
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        _, mean_ms, std_ms, actual_lens = benchmark(
                            variant=variant,
                            model=model,
                            input_batches=input_batches,
                            device=device,
                            chunk_size=args.chunk_size,
                            warmup=args.warmup,
                            runs=args.runs,
                            last_block_dense_scope=last_block_dense_scope,
                            dense_prefix_tokens=args.dense_prefix_tokens,
                        )
                    else:
                        raise
                else:
                    raise

            if is_rank0:
                print(
                    f"{seq_len} | {mean_ms:.2f} | {std_ms:.2f} | "
                    f"{','.join(str(x) for x in actual_lens)} | {args.batch_size} | ok",
                    flush=True,
                )
        finally:
            del input_batches
            _cleanup_after_benchmark_case(variant)
            _cleanup_after_seq_len_case()


if __name__ == "__main__":
    main()
