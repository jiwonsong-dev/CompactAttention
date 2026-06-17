# LongBench v2 Chunked-Prefill Evaluation

This directory adds a local LongBench v2 runner for the same chunked-prefill
methods used by RULER:

- `dense`
- `quoka_dense`
- `block_sparse` / `seer_compactattention`
- `flashprefill_block_sparse` / `flashprefill_compactattention`

LongBench v2 is loaded from `THUDM/LongBench-v2` by default. It is a
multiple-choice benchmark; the scorer extracts `A/B/C/D` and reports the same
top-level breakdown as the official script: `overall`, `easy`, `hard`,
`short`, `medium`, and `long`.

## Example: Qwen3 Dense

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
python eval/longbench_v2/run_chunked_prefill.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --mode dense \
  --chunk-size 1024 \
  --max-input-tokens 131072 \
  --qwen-long-context \
  --save-dir results/longbench_v2/qwen3_dense_c1024
```

## Method Commands

Common Qwen3 options:

```bash
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
COMMON="--model $MODEL --chunk-size 1024 --max-input-tokens 131072 --qwen-long-context"
```

For LLaMA method-grid runs, `seer_bs` and `compact_sa` automatically use
`SEER_MODEL=SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates`, because those
methods require `attn_gate_weights.pth`. Dense, QUOKA, FlashPrefill, and
Compact FP use `MODEL`.

Dense FlashInfer:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python eval/longbench_v2/run_chunked_prefill.py \
  $COMMON --mode dense --dense-backend flashinfer \
  --save-dir results/longbench_v2/qwen3_dense_c1024
```

QUOKA:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python eval/longbench_v2/run_chunked_prefill.py \
  $COMMON --mode quoka_dense --dense-backend flashinfer \
  --quoka-query-ratio 0.25 --quoka-kv-budget-ratio 0.25 \
  --save-dir results/longbench_v2/qwen3_quoka_c1024
```

FlashPrefill block sparse:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python eval/longbench_v2/run_chunked_prefill.py \
  $COMMON --mode flashprefill_block_sparse \
  --flashprefill-alpha 0.01 --flashprefill-last-n-block 2 \
  --last-block-dense-scope final_prefill_chunk \
  --save-dir results/longbench_v2/qwen3_flashprefill_alpha001_c1024
```

CompactAttention FP:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python eval/longbench_v2/run_chunked_prefill.py \
  $COMMON --mode flashprefill_compactattention \
  --flashprefill-alpha 0.12 --flashprefill-last-n-block 2 \
  --last-block-dense-scope final_prefill_chunk \
  --final-dense-tail-blocks 0 \
  --compactattn-indexed-impl fi_zero_copy_subgroup \
  --compactattn-disable-first-chunk-dense \
  --save-dir results/longbench_v2/qwen3_compactfp_alpha012_c1024
```

## Example: Qwen3 CompactAttention FP

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
python eval/longbench_v2/run_chunked_prefill.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --mode flashprefill_compactattention \
  --chunk-size 1024 \
  --flashprefill-alpha 0.12 \
  --flashprefill-last-n-block 2 \
  --final-dense-tail-blocks 0 \
  --compactattn-indexed-impl fi_zero_copy_subgroup \
  --compactattn-disable-first-chunk-dense \
  --max-input-tokens 131072 \
  --qwen-long-context \
  --save-dir results/longbench_v2/qwen3_compactfp_alpha012_c1024
```

## Sharding

Use `--num-shards` and `--shard-id` to split the 503 examples across GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python eval/longbench_v2/run_chunked_prefill.py ... --num-shards 4 --shard-id 0
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD python eval/longbench_v2/run_chunked_prefill.py ... --num-shards 4 --shard-id 1
```

Use different `--output-name` values for concurrent shards to avoid interleaved
writes:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python eval/longbench_v2/run_chunked_prefill.py ... --num-shards 4 --shard-id 0 --output-name shard0.jsonl
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD python eval/longbench_v2/run_chunked_prefill.py ... --num-shards 4 --shard-id 1 --output-name shard1.jsonl
```

## Scoring

```bash
PYTHONPATH=$PWD python eval/longbench_v2/score.py \
  --pred-file results/longbench_v2/qwen3_dense_c1024/dense.jsonl
```

Multiple shard files can be scored together:

```bash
PYTHONPATH=$PWD python eval/longbench_v2/score.py \
  --pred-file results/longbench_v2/qwen3_dense_c1024/shard*.jsonl
```

The generated summaries are:

- `summary.json`
- `summary.csv`
