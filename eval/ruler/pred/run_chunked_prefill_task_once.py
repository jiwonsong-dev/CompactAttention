import argparse
import csv
import importlib
import json
import os
from argparse import Namespace
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import yaml
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
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

TASK_ORDER = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def ordered_tasks(rows_by_task):
    ordered = [task for task in TASK_ORDER if task in rows_by_task]
    ordered.extend(task for task in rows_by_task if task not in TASK_ORDER)
    return ordered


def load_summary_rows(summary_file: Path):
    rows = {}
    if not summary_file.exists():
        return rows

    with open(summary_file, "r", newline="", encoding="utf-8") as f:
        raw_rows = list(csv.reader(f))

    if not raw_rows:
        return rows

    if raw_rows[0] and raw_rows[0][0] == "Metric":
        tasks = raw_rows[0][1:]
        metrics = {row[0]: row[1:] for row in raw_rows[1:] if row}
        scores = metrics.get("Score", [])
        nulls = metrics.get("Nulls", [])
        for idx, task_name in enumerate(tasks):
            rows[task_name] = {
                "Tasks": task_name,
                "Score": scores[idx] if idx < len(scores) else "",
                "Nulls": nulls[idx] if idx < len(nulls) else "",
            }
        return rows

    with open(summary_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row.get("Tasks")
            if task_name:
                rows[task_name] = {
                    "Tasks": task_name,
                    "Score": row.get("Score", ""),
                    "Nulls": row.get("Nulls", ""),
                }
    return rows


def write_horizontal_summary(summary_file: Path, rows_by_task):
    ordered = ordered_tasks(rows_by_task)
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", *ordered])
        writer.writerow(["Score", *[rows_by_task[task]["Score"] for task in ordered]])
        writer.writerow(["Nulls", *[rows_by_task[task]["Nulls"] for task in ordered]])


def load_metric_and_tokens(task_name: str):
    with open("eval/ruler/synthetic.yaml", "r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)[task_name]
    data_mod = importlib.import_module("eval.ruler.data.synthetic.constants")
    eval_mod = importlib.import_module("eval.ruler.eval.synthetic.constants")
    merged = dict(data_mod.TASKS[task_cfg["task"]])
    merged.update(task_cfg)
    metric_fn = eval_mod.TASKS[task_cfg["task"]]["metric_fn"]
    return merged, metric_fn


def variant_for_mode(mode: str) -> tuple[str, str]:
    mapping = {
        "dense": ("dense", "none"),
        "block_sparse": ("block_sparse", "seer"),
        "compactattn": ("compactattn", "seer"),
        "compactattention": ("compactattn", "seer"),
        "seer_compactattention": ("compactattn", "seer"),
        "flashprefill_block_sparse": ("block_sparse", "flashprefill"),
        "flashprefill_compactattn": ("compactattn", "flashprefill"),
        "flashprefill_compactattention": ("compactattn", "flashprefill"),
        "quoka_dense": ("dense", "quoka"),
    }
    if mode not in mapping:
        raise ValueError(f"Unsupported mode: {mode}")
    return mapping[mode]


def build_runtime_args(args) -> Namespace:
    execution_mode, selection_method = variant_for_mode(args.mode)
    return Namespace(
        model=args.model,
        execution_mode=execution_mode,
        selection_method=selection_method,
        dense_attn_impl=args.dense_attn_impl,
        dense_backend=args.dense_backend,
        attention_harness=args.attention_harness,
        threshold=SEER_GLOBAL_THRESHOLD,
        chunk_size=args.chunk_size,
        batch_size=1,
        warmup=0,
        runs=1,
        dtype="bfloat16",
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
        prompt_source="ruler",
        input_schedule="fixed",
        task=args.task,
        sample_index=0,
        data_file=Path(args.data_file),
        qwen_long_context=bool(args.qwen_long_context),
        qwen_long_context_max_position_embeddings=args.qwen_long_context_max_position_embeddings,
        qwen_yarn_factor=args.qwen_yarn_factor,
        qwen_original_max_position_embeddings=args.qwen_original_max_position_embeddings,
        device_map="auto",
    )


def maybe_init_variant_tp(variant: str, runtime_args: Namespace) -> tuple[bool, int]:
    use_tp = (
        variant in {"dense", "quoka_dense", "seer_block_sparse", "seer_compactattn", "seer_compactattn_hf"}
        and getattr(runtime_args, "device_map", None) == "auto"
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    if not use_tp:
        return False, 0
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    torch.cuda.set_device(local_rank)
    return True, dist.get_rank()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--seer-model", dest="model", default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "dense",
            "block_sparse",
            "compactattn",
            "compactattention",
            "seer_compactattention",
            "flashprefill_block_sparse",
            "flashprefill_compactattn",
            "flashprefill_compactattention",
            "quoka_dense",
        ],
        required=True,
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--last-block-dense",
        dest="last_block_dense",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--last-block-dense-scope",
        type=str,
        default=None,
        choices=["all_prefill_chunks", "final_prefill_chunk"],
    )
    parser.add_argument("--final-dense-tail-blocks", type=int, default=None)
    parser.add_argument("--compactattn-disable-first-chunk-dense", "--compactattention-disable-first-chunk-dense", action="store_true")
    parser.add_argument("--compactattn-keep-recent-blocks", "--compactattention-keep-recent-blocks", type=int, default=2)
    parser.add_argument(
        "--compactattn-chunked-gate-head-pool",
        "--compactattention-chunked-gate-head-pool",
        type=str,
        default="none",
        choices=["none", "avg", "max", "score_avg"],
    )
    parser.add_argument(
        "--compactattn-pack-impl",
        "--compactattention-pack-impl",
        type=str,
        default="indexed_dense",
        choices=["torch", "triton", "indexed_dense"],
    )
    parser.add_argument(
        "--compactattn-indexed-impl",
        "--compactattention-indexed-impl",
        type=str,
        default="auto",
        choices=["auto", "fa2_paged", "triton_direct", "fa2_indexed", "fi_paged", "fi_zero_copy", "fi_zero_copy_per_query", "fi_zero_copy_subgroup", "cudnn_one_shot"],
    )
    parser.add_argument(
        "--compactattn-cache-fill-backend",
        "--compactattention-cache-fill-backend",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "triton"],
    )
    parser.add_argument("--flashprefill-alpha", type=float, default=None)
    parser.add_argument("--flashprefill-block-size", type=int, default=128)
    parser.add_argument("--flashprefill-attention-sink", type=int, default=2)
    parser.add_argument("--flashprefill-window-size", type=int, default=4)
    parser.add_argument(
        "--flashprefill-last-n-block",
        type=int,
        default=2,
        help=(
            "FlashPrefill selector dense tail for the final scored chunk only; "
            "intermediate chunks use 0."
        ),
    )
    parser.add_argument("--flashprefill-min-budget", type=int, default=0)
    parser.add_argument("--quoka-query-ratio", type=float, default=0.25)
    parser.add_argument("--quoka-kv-budget-ratio", type=float, default=0.25)
    parser.add_argument("--dense-prefix-tokens", type=int, default=0)
    parser.add_argument("--dense-attn-impl", type=str, default="flash_attention_2")
    parser.add_argument(
        "--dense-backend",
        type=str,
        default=DEFAULT_DENSE_BACKEND,
        choices=["flash_attn", "flashinfer"],
    )
    parser.add_argument(
        "--attention-harness",
        type=str,
        default="legacy",
        choices=["legacy", "replacement"],
    )
    parser.add_argument("--qwen-long-context", action="store_true")
    parser.add_argument("--qwen-long-context-max-position-embeddings", type=int, default=131072)
    parser.add_argument("--qwen-yarn-factor", type=float, default=4.0)
    parser.add_argument("--qwen-original-max-position-embeddings", type=int, default=32768)
    args = parser.parse_args()

    if args.final_dense_tail_blocks is not None and args.final_dense_tail_blocks < 0:
        parser.error("--final-dense-tail-blocks must be non-negative")
    if args.dense_prefix_tokens < 0:
        parser.error("--dense-prefix-tokens must be non-negative")
    if args.quoka_query_ratio < 0:
        parser.error("--quoka-query-ratio must be non-negative")
    if args.quoka_kv_budget_ratio < 0:
        parser.error("--quoka-kv-budget-ratio must be non-negative")

    task_cfg, metric_fn = load_metric_and_tokens(args.task)
    max_new_tokens = int(task_cfg["tokens_to_generate"])

    runtime_args = build_runtime_args(args)
    apply_backend_defaults(runtime_args)
    variant = resolve_variant(runtime_args)
    using_variant_tp, rank = maybe_init_variant_tp(variant, runtime_args)
    is_rank0 = rank == 0

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
        runner = DenseChunkedPrefillRunner(
            model,
            tokenizer,
            args.chunk_size,
            max_new_tokens,
        )
    else:
        runner = ChunkedPrefillGenerationRunner(
            model,
            tokenizer,
            args.chunk_size,
            max_new_tokens,
            resolve_last_block_dense_scope(runtime_args, variant),
            args.dense_prefix_tokens,
            drop_full_attention_mask_for_dense=False,
        )

    save_dir = Path(args.save_dir)
    if is_rank0:
        save_dir.mkdir(parents=True, exist_ok=True)
    pred_file = save_dir / f"{args.task}.jsonl"
    summary_file = save_dir / "summary.csv"

    completed = set()
    if is_rank0 and pred_file.exists():
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["index"])
    if using_variant_tp:
        payload = [sorted(completed)]
        dist.broadcast_object_list(payload, src=0)
        completed = set(payload[0])

    samples = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if is_rank0:
        fout = open(pred_file, "a", encoding="utf-8", buffering=1)
    else:
        fout = None
    try:
        for sample in samples:
            if sample["index"] in completed:
                continue
            pred = runner.generate(sample["input"])
            if is_rank0:
                row = {
                    "index": sample["index"],
                    "input": sample["input"],
                    "outputs": sample["outputs"],
                    "pred": pred,
                    "others": sample.get("others", {}),
                    "truncation": sample.get("truncation", -1),
                    "length": sample.get("length", -1),
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"done index={sample['index']} pred_head={pred[:80]!r}", flush=True)
    finally:
        if fout is not None:
            fout.close()

    if using_variant_tp:
        dist.barrier()

    if is_rank0 and os.environ.get("SKIP_TASK_SCORE", "0") != "1":
        predicts = []
        references = []
        nulls = 0
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                pred = row["pred"].strip()
                predicts.append(pred)
                references.append(row.get("outputs", [""]))
                if not pred:
                    nulls += 1
        score = metric_fn(predicts, references) if references and references[0][0] is not None else 0.0

        summary_rows = load_summary_rows(summary_file)
        summary_rows[args.task] = {
            "Tasks": args.task,
            "Score": score,
            "Nulls": f"{nulls}/{len(predicts)}",
        }
        write_horizontal_summary(summary_file, summary_rows)

        print(
            {
                "pred_file": str(pred_file),
                "summary_file": str(summary_file),
                "score": score,
                "nulls": f"{nulls}/{len(predicts)}",
                "dense_tp": variant == "dense" and using_variant_tp,
                "quoka_tp": variant == "quoka_dense" and using_variant_tp,
                "seer_tp": variant == "seer_block_sparse" and using_variant_tp,
                "compact_tp": variant in {"seer_compactattn", "seer_compactattn_hf"} and using_variant_tp,
            }
        )

    if using_variant_tp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
