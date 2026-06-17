#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
SEER_MODEL="${SEER_MODEL:-SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates}"
OUT_ROOT="${OUT_ROOT:-results/longbench_v2/$(basename "$MODEL")_c${CHUNK_SIZE:-1024}}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-131072}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LIMIT="${LIMIT:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
SPLIT="${SPLIT:-train}"
METHODS="${METHODS:-dense quoka seer_bs compact_sa flashprefill_bs compact_fp}"

SEER_THRESHOLD="${SEER_THRESHOLD:-3e-4}"
COMPACT_SA_THRESHOLD="${COMPACT_SA_THRESHOLD:-5e-4}"
FLASHPREFILL_ALPHA="${FLASHPREFILL_ALPHA:-0.01}"
COMPACT_FP_ALPHA="${COMPACT_FP_ALPHA:-0.06}"

COMMON=(
  --chunk-size "$CHUNK_SIZE"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --split "$SPLIT"
  --num-shards "$NUM_SHARDS"
  --shard-id "$SHARD_ID"
)

if [[ -n "$LIMIT" ]]; then
  COMMON+=(--limit "$LIMIT")
fi

if [[ "$MODEL" == *Qwen3* || "$MODEL" == *Qwen* ]]; then
  COMMON+=(
    --qwen-long-context
    --qwen-long-context-max-position-embeddings "$MAX_INPUT_TOKENS"
    --qwen-yarn-factor "${QWEN_YARN_FACTOR:-4.0}"
    --qwen-original-max-position-embeddings "${QWEN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-32768}"
  )
fi

run_method() {
  local name="$1"
  local save_dir="$OUT_ROOT/$name"
  local method_model="$MODEL"
  if [[ "$name" == "seer_bs" || "$name" == "compact_sa" ]]; then
    method_model="$SEER_MODEL"
  fi
  mkdir -p "$save_dir"
  echo "[$(date '+%F %T')] start method=$name gpu=$GPU model=$method_model save_dir=$save_dir"
  case "$name" in
    dense)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PYTHON" eval/longbench_v2/run_chunked_prefill.py \
        "${COMMON[@]}" \
        --model "$method_model" \
        --mode dense \
        --dense-backend flashinfer \
        --save-dir "$save_dir" \
        --output-name "$name.jsonl"
      ;;
    quoka)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PYTHON" eval/longbench_v2/run_chunked_prefill.py \
        "${COMMON[@]}" \
        --model "$method_model" \
        --mode quoka_dense \
        --dense-backend flashinfer \
        --quoka-query-ratio "${QUOKA_QUERY_RATIO:-0.25}" \
        --quoka-kv-budget-ratio "${QUOKA_KV_BUDGET_RATIO:-0.25}" \
        --save-dir "$save_dir" \
        --output-name "$name.jsonl"
      ;;
    seer_bs)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PYTHON" eval/longbench_v2/run_chunked_prefill.py \
        "${COMMON[@]}" \
        --model "$method_model" \
        --mode block_sparse \
        --threshold "$SEER_THRESHOLD" \
        --last-block-dense \
        --last-block-dense-scope final_prefill_chunk \
        --save-dir "$save_dir" \
        --output-name "$name.jsonl"
      ;;
    compact_sa)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PYTHON" eval/longbench_v2/run_chunked_prefill.py \
        "${COMMON[@]}" \
        --model "$method_model" \
        --mode seer_compactattention \
        --threshold "$COMPACT_SA_THRESHOLD" \
        --last-block-dense \
        --last-block-dense-scope final_prefill_chunk \
        --final-dense-tail-blocks "${COMPACT_SA_DENSE_TAIL_BLOCKS:-2}" \
        --compactattn-indexed-impl "${COMPACT_SA_INDEXED_IMPL:-auto}" \
        --save-dir "$save_dir" \
        --output-name "$name.jsonl"
      ;;
    flashprefill_bs)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PYTHON" eval/longbench_v2/run_chunked_prefill.py \
        "${COMMON[@]}" \
        --model "$method_model" \
        --mode flashprefill_block_sparse \
        --flashprefill-alpha "$FLASHPREFILL_ALPHA" \
        --flashprefill-last-n-block 2 \
        --last-block-dense-scope final_prefill_chunk \
        --save-dir "$save_dir" \
        --output-name "$name.jsonl"
      ;;
    compact_fp)
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PYTHON" eval/longbench_v2/run_chunked_prefill.py \
        "${COMMON[@]}" \
        --model "$method_model" \
        --mode flashprefill_compactattention \
        --flashprefill-alpha "$COMPACT_FP_ALPHA" \
        --flashprefill-last-n-block 2 \
        --last-block-dense-scope final_prefill_chunk \
        --final-dense-tail-blocks 0 \
        --compactattn-indexed-impl "${COMPACT_FP_INDEXED_IMPL:-fi_zero_copy_subgroup}" \
        --compactattn-disable-first-chunk-dense \
        --compactattn-keep-recent-blocks 2 \
        --save-dir "$save_dir" \
        --output-name "$name.jsonl"
      ;;
    *)
      echo "Unknown method: $name" >&2
      exit 2
      ;;
  esac
  echo "[$(date '+%F %T')] done method=$name"
}

for method in $METHODS; do
  run_method "$method"
done

echo "[$(date '+%F %T')] completed methods: $METHODS"
