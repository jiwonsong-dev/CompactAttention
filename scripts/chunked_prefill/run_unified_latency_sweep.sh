#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 GPU EXECUTION_MODE SELECTION_METHOD [CHUNK_SIZE] [SEQ_LENS] [RUNS] [WARMUP] [CONDA_ENV] [MODEL]"
  echo "  GPU may be a single id (e.g. 0) or a comma-separated visible set (e.g. 0,1)."
  echo "  EXECUTION_MODE: dense | block_sparse | compactattention (compactattn alias still works)"
  echo "  SELECTION_METHOD: none | seer | seer_hf | quoka | flashprefill"
  echo "  Set BATCH_SIZE=N in the environment for multi-batch runs. Default: 1"
  echo "  Comma-separated GPU lists auto-enable --device-map auto; set DEVICE_MAP_AUTO=1 to force it."
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

GPU=$1
EXECUTION_MODE=$2
SELECTION_METHOD=$3
CHUNK_SIZE=${4:-1024}
SEQ_LENS=${5:-8192,16384,32768,65536,131072,262144}
RUNS=${6:-3}
WARMUP=${7:-1}
CONDA_ENV=${8:-compactattn}
VENV_DIR="${VENV_DIR:-/workspace/compactattn}"
MODEL=${9:-${SEER_MODEL:-}}

if [ -z "$MODEL" ]; then
  if [ "$SELECTION_METHOD" = "seer" ]; then
    MODEL="SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates"
  else
    MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
  fi
fi

run_python() {
  if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
    python "$@"
    return
  fi
  if [ -f "$VENV_DIR/bin/python" ]; then
    "$VENV_DIR/bin/python" "$@"
    return
  fi
  conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && [[ "$GPU" == *,* ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

BATCH_SIZE_VALUE=${BATCH_SIZE:-1}
DEVICE_MAP_AUTO_VALUE=${DEVICE_MAP_AUTO:-0}
USE_DEVICE_MAP_AUTO=0
if [[ "$GPU" == *,* ]]; then
  USE_DEVICE_MAP_AUTO=1
fi
if [ "$DEVICE_MAP_AUTO_VALUE" = "1" ]; then
  USE_DEVICE_MAP_AUTO=1
fi

EXTRA_ARGS=()
if [ -n "${SEER_LAST_BLOCK_DENSE_SCOPE:-}" ]; then
  EXTRA_ARGS+=(--last-block-dense-scope "${SEER_LAST_BLOCK_DENSE_SCOPE}")
fi
if [ -n "${SEERATTN_FINAL_DENSE_TAIL_BLOCKS:-}" ]; then
  EXTRA_ARGS+=(--final-dense-tail-blocks "${SEERATTN_FINAL_DENSE_TAIL_BLOCKS}")
fi
if [ "${COMPACTATTN_DISABLE_FIRST_CHUNK_DENSE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--compactattn-disable-first-chunk-dense)
fi
if [ -n "${PROMPT_SOURCE:-}" ]; then
  EXTRA_ARGS+=(--prompt-source "${PROMPT_SOURCE}")
fi
if [ -n "${TASK:-}" ]; then
  EXTRA_ARGS+=(--task "${TASK}")
fi
if [ -n "${SAMPLE_INDEX:-}" ]; then
  EXTRA_ARGS+=(--sample-index "${SAMPLE_INDEX}")
fi
if [ -n "${DATA_FILE:-}" ]; then
  EXTRA_ARGS+=(--data-file "${DATA_FILE}")
fi
if [ "$USE_DEVICE_MAP_AUTO" = "1" ]; then
  EXTRA_ARGS+=(--device-map auto)
fi

USE_TP_LAUNCH=0
if [ "$USE_DEVICE_MAP_AUTO" = "1" ]; then
  case "$EXECUTION_MODE:$SELECTION_METHOD" in
    dense:none|dense:quoka|block_sparse:seer|compactattn:seer|compactattention:seer)
      USE_TP_LAUNCH=1
      ;;
  esac
fi

PY_ARGS=(
  "$ROOT/eval/chunked_prefill/test_unified_latency_sweep.py"
  --model "$MODEL"
  --execution-mode "$EXECUTION_MODE"
  --selection-method "$SELECTION_METHOD"
  --seq-lens "$SEQ_LENS"
  --chunk-size "$CHUNK_SIZE"
  --batch-size "$BATCH_SIZE_VALUE"
  --warmup "$WARMUP"
  --runs "$RUNS"
  --dense-attn-impl "${DENSE_ATTN_IMPL:-flash_attention_2}"
  --dense-backend "${DENSE_BACKEND:-flashinfer}"
  --dense-prefix-tokens "${DENSE_PREFIX_TOKENS:-0}"
  --compactattn-keep-recent-blocks "${COMPACTATTN_KEEP_RECENT_BLOCKS:-2}"
  --compactattn-chunked-gate-head-pool "${COMPACTATTN_CHUNKED_GATE_HEAD_POOL:-none}"
  --col-pack-impl "${COMPACTATTN_PACK_IMPL:-indexed_dense}"
  --col-indexed-impl "${COMPACTATTN_INDEXED_IMPL:-auto}"
  --col-cache-fill-backend "${COMPACTATTN_CACHE_FILL_BACKEND:-cuda}"
  --quoka-query-ratio "${QUOKA_QUERY_RATIO:-0.25}"
  --quoka-kv-budget-ratio "${QUOKA_KV_BUDGET_RATIO:-0.25}"
  --input-schedule "${INPUT_SCHEDULE:-fixed}"
  "${EXTRA_ARGS[@]}"
)

if [ "$USE_TP_LAUNCH" = "1" ]; then
  NUM_GPUS=$(awk -F',' '{print NF}' <<<"$GPU")
  run_python -m torch.distributed.run --nproc_per_node "$NUM_GPUS" "${PY_ARGS[@]}"
else
  run_python "${PY_ARGS[@]}"
fi
