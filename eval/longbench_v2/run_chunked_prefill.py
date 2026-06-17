#!/usr/bin/env python
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.chunked_prefill.chunked_prefill_runtime import (
    ChunkedPrefillGenerationRunner,
    DEFAULT_DENSE_BACKEND,
    DenseChunkedPrefillRunner,
    SEER_GLOBAL_THRESHOLD,
    _enable_default_qwen_long_context,
    apply_backend_defaults,
    get_dtype,
    load_model,
    resolve_last_block_dense_scope,
    resolve_model_id,
    resolve_variant,
)
from eval.longbench_v2.common import build_prompt, extract_answer, truncate_prompt_middle
from eval.longbench_v2.score import score_rows, write_csv
from eval.ruler.pred.run_chunked_prefill_task_once import maybe_init_variant_tp, variant_for_mode


def build_runtime_args(args: argparse.Namespace) -> Namespace:
    execution_mode, selection_method = variant_for_mode(args.mode)
    return Namespace(
        model=args.model,
        execution_mode=execution_mode,
        selection_method=selection_method,
        dense_attn_impl=args.dense_attn_impl,
        dense_backend=args.dense_backend,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        batch_size=1,
        warmup=0,
        runs=1,
        dtype=args.dtype,
        seed=1234,
        dense_prefix_tokens=args.dense_prefix_tokens,
        last_block_dense=args.last_block_dense,
        last_block_dense_scope=args.last_block_dense_scope,
        final_dense_tail_blocks=args.final_dense_tail_blocks,
        flashprefill_alpha=args.flashprefill_alpha,
        flashprefill_block_size=args.flashprefill_block_size,
        flashprefill_attention_sink=args.flashprefill_attention_sink,
        flashprefill_window_size=args.flashprefill_window_size,
        flashprefill_last_n_block=args.flashprefill_last_n_block,
        flashprefill_min_budget=args.flashprefill_min_budget,
        compactattn_keep_recent_blocks=args.compactattn_keep_recent_blocks,
        compactattn_disable_first_chunk_dense=args.compactattn_disable_first_chunk_dense,
        compactattn_chunked_gate_head_pool=args.compactattn_chunked_gate_head_pool,
        col_pack_impl=args.compactattn_pack_impl,
        col_indexed_impl=args.compactattn_indexed_impl,
        col_cache_fill_backend=args.compactattn_cache_fill_backend,
        quoka_query_ratio=args.quoka_query_ratio,
        quoka_kv_budget_ratio=args.quoka_kv_budget_ratio,
        prompt_source="longbench_v2",
        input_schedule="fixed",
        task="longbench_v2",
        sample_index=0,
        data_file=args.data_file,
        qwen_long_context=bool(args.qwen_long_context),
        qwen_long_context_max_position_embeddings=args.qwen_long_context_max_position_embeddings,
        qwen_yarn_factor=args.qwen_yarn_factor,
        qwen_original_max_position_embeddings=args.qwen_original_max_position_embeddings,
        device_map="auto",
    )


def load_longbench_v2(args: argparse.Namespace) -> list[dict]:
    if args.data_file is not None:
        path = Path(args.data_file)
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    else:
        from datasets import load_dataset

        data = list(load_dataset("THUDM/LongBench-v2", split=args.split))

    out = []
    for idx, item in enumerate(data):
        if args.length and str(item.get("length")) != args.length:
            continue
        if args.difficulty and str(item.get("difficulty")) != args.difficulty:
            continue
        if args.domain and str(item.get("domain")) != args.domain:
            continue
        if args.num_shards > 1 and (idx % args.num_shards) != args.shard_id:
            continue
        row = dict(item)
        row.setdefault("_id", row.get("id", idx))
        out.append(row)
        if args.limit is not None and len(out) >= args.limit:
            break
    return out


def make_runner(args: argparse.Namespace):
    runtime_args = build_runtime_args(args)
    apply_backend_defaults(runtime_args)
    variant = resolve_variant(runtime_args)
    using_variant_tp, rank = maybe_init_variant_tp(variant, runtime_args)

    model_id = resolve_model_id(runtime_args, variant)
    model_cfg = AutoConfig.from_pretrained(model_id)
    _enable_default_qwen_long_context(runtime_args, model_cfg)
    base_model = getattr(model_cfg, "base_model", model_id)
    model, base_model = load_model(runtime_args, variant, get_dtype(runtime_args.dtype), model_id, model_cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if variant == "dense":
        runner = DenseChunkedPrefillRunner(model, tokenizer, args.chunk_size, args.max_new_tokens)
    else:
        runner = ChunkedPrefillGenerationRunner(
            model,
            tokenizer,
            args.chunk_size,
            args.max_new_tokens,
            resolve_last_block_dense_scope(runtime_args, variant),
            args.dense_prefix_tokens,
            drop_full_attention_mask_for_dense=False,
        )
    return runner, variant, using_variant_tp, rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", required=True, choices=[
        "dense", "block_sparse", "compactattn", "compactattention", "seer_compactattention",
        "flashprefill_block_sparse", "flashprefill_compactattn", "flashprefill_compactattention",
        "quoka_dense",
    ])
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--length", choices=["short", "medium", "long"], default=None)
    parser.add_argument("--difficulty", choices=["easy", "hard"], default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--max-input-tokens", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--threshold", type=float, default=SEER_GLOBAL_THRESHOLD)
    parser.add_argument("--last-block-dense", dest="last_block_dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--last-block-dense-scope", choices=["all_prefill_chunks", "final_prefill_chunk"], default=None)
    parser.add_argument("--final-dense-tail-blocks", type=int, default=None)
    parser.add_argument("--dense-prefix-tokens", type=int, default=0)
    parser.add_argument("--dense-attn-impl", default="flash_attention_2")
    parser.add_argument("--dense-backend", default=DEFAULT_DENSE_BACKEND, choices=["flash_attn", "flashinfer"])
    parser.add_argument("--compactattn-disable-first-chunk-dense", "--compactattention-disable-first-chunk-dense", action="store_true")
    parser.add_argument("--compactattn-keep-recent-blocks", "--compactattention-keep-recent-blocks", type=int, default=2)
    parser.add_argument("--compactattn-chunked-gate-head-pool", "--compactattention-chunked-gate-head-pool", default="none", choices=["none", "avg", "max", "score_avg"])
    parser.add_argument("--compactattn-pack-impl", "--compactattention-pack-impl", default="indexed_dense", choices=["torch", "triton", "indexed_dense"])
    parser.add_argument("--compactattn-indexed-impl", "--compactattention-indexed-impl", default="auto", choices=["auto", "fa2_paged", "triton_direct", "fa2_indexed", "fi_paged", "fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"])
    parser.add_argument("--compactattn-cache-fill-backend", "--compactattention-cache-fill-backend", default="cuda", choices=["auto", "cuda", "triton"])
    parser.add_argument("--flashprefill-alpha", type=float, default=None)
    parser.add_argument("--flashprefill-block-size", type=int, default=128)
    parser.add_argument("--flashprefill-attention-sink", type=int, default=2)
    parser.add_argument("--flashprefill-window-size", type=int, default=4)
    parser.add_argument("--flashprefill-last-n-block", type=int, default=2)
    parser.add_argument("--flashprefill-min-budget", type=int, default=0)
    parser.add_argument("--quoka-query-ratio", type=float, default=0.25)
    parser.add_argument("--quoka-kv-budget-ratio", type=float, default=0.25)
    parser.add_argument("--qwen-long-context", action="store_true")
    parser.add_argument("--qwen-long-context-max-position-embeddings", type=int, default=131072)
    parser.add_argument("--qwen-yarn-factor", type=float, default=4.0)
    parser.add_argument("--qwen-original-max-position-embeddings", type=int, default=32768)
    args = parser.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0, --num-shards)")

    rows = load_longbench_v2(args)
    runner, variant, using_variant_tp, rank = make_runner(args)
    is_rank0 = rank == 0
    args.save_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.mode}.jsonl"
    pred_file = args.save_dir / output_name

    completed = set()
    if is_rank0 and pred_file.exists():
        for line in pred_file.open(encoding="utf-8"):
            if line.strip():
                completed.add(json.loads(line)["_id"])
    if using_variant_tp:
        payload = [sorted(completed)]
        dist.broadcast_object_list(payload, src=0)
        completed = set(payload[0])

    fout = pred_file.open("a", encoding="utf-8", buffering=1) if is_rank0 else None
    try:
        for item in rows:
            item_id = str(item["_id"])
            if item_id in completed:
                continue
            prompt = build_prompt(item)
            prompt, original_tokens, was_truncated = truncate_prompt_middle(
                runner.tokenizer, prompt, args.max_input_tokens
            )
            response = runner.generate(prompt)
            pred_answer = extract_answer(response)
            if is_rank0:
                row = {
                    "_id": item_id,
                    "domain": item.get("domain"),
                    "sub_domain": item.get("sub_domain"),
                    "difficulty": item.get("difficulty"),
                    "length": item.get("length"),
                    "answer": item.get("answer"),
                    "pred_answer": pred_answer,
                    "judge": pred_answer == str(item.get("answer", "")).strip().upper(),
                    "response": response,
                    "original_prompt_tokens": original_tokens,
                    "truncated": was_truncated,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"done id={item_id} pred={pred_answer} gold={row['answer']} "
                    f"judge={row['judge']} tokens={original_tokens} truncated={was_truncated}",
                    flush=True,
                )
    finally:
        if fout is not None:
            fout.close()

    if using_variant_tp:
        dist.barrier()

    if is_rank0:
        pred_rows = [json.loads(line) for line in pred_file.open(encoding="utf-8") if line.strip()]
        summary = score_rows(pred_rows)
        (args.save_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_csv(args.save_dir / "summary.csv", summary)
        print({"pred_file": str(pred_file), "summary": summary})

    if using_variant_tp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
