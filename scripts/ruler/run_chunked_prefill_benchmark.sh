#!/usr/bin/env bash
set -euo pipefail

# NOTE:
# This runner is for the current chunked-prefill decoder implementations in this repo,
# which are Llama-3.1/Llama-family specific. Use prepare_ruler_prebuilt_data.sh if you
# only want tokenizer-aligned synthetic data for other model families such as Qwen2.5.

SEER_GATE_MODEL_ID="SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates"
SEER_GATE_QWEN3_MODEL_ID="jiwonsong/SeerAttention-Qwen3-8B-AttnGates"
BASE_LLAMA_MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 MODEL|auto RESULTS_ROOT MODE GPU SEQ_LEN [NUM_SAMPLES] [CHUNK_SIZE] [CONDA_ENV] [TOKENIZER_PATH]"
  echo "MODE examples: dense, block_sparse, seer_compactattention, flashprefill_block_sparse, flashprefill_compactattention"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL=$1
RESULTS_ROOT=$2
MODE=$3
RUN_MODE=$MODE
GPU=$4
SEQ_LEN=$5
NUM_SAMPLES=${6:-25}
CHUNK_SIZE=${7:-1024}
CONDA_ENV=${8:-compactattn}
VENV_DIR="${VENV_DIR:-/workspace/compactattn}"
TOKENIZER_PATH=${9:-}
PREBUILT_DATA_ROOT=${RULER_PREBUILT_DATA_ROOT:-results/ruler_llama_3_1_8b}

select_default_model() {
  local mode=$1
  local tokenizer_hint=${2:-}
  local lowered
  lowered="$(printf '%s' "$tokenizer_hint" | tr '[:upper:]' '[:lower:]')"
  case "$mode" in
    block_sparse|compactattn|compactattention|seer_compactattention)
      if [[ "$lowered" == *qwen/qwen3-8b* ]] || [[ "$lowered" == *qwen3-8b* ]]; then
        printf '%s\n' "$SEER_GATE_QWEN3_MODEL_ID"
      else
        printf '%s\n' "$SEER_GATE_MODEL_ID"
      fi
      ;;
    flashprefill_block_sparse|flashprefill_compactattn|flashprefill_compactattention)
      printf '%s\n' "$BASE_LLAMA_MODEL_ID"
      ;;
    *)
      printf '%s\n' "$BASE_LLAMA_MODEL_ID"
      ;;
  esac
}

canonical_runner_mode() {
  case "$1" in
    compactattention|seer_compactattention)
      printf 'compactattn\n'
      ;;
    flashprefill_compactattention)
      printf 'flashprefill_compactattn\n'
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

RUN_MODE=$(canonical_runner_mode "$MODE")

resolve_model_checkpoint_dir() {
  local path=$1
  if [ ! -d "$path" ]; then
    printf '%s\n' "$path"
    return
  fi
  if [ -f "$path/config.json" ]; then
    printf '%s\n' "$path"
    return
  fi
  local matches=()
  while IFS= read -r line; do
    matches+=("$line")
  done < <(find "$path" -mindepth 2 -maxdepth 2 -type f -name config.json -printf '%h\n' | sort -u)
  if [ "${#matches[@]}" -eq 1 ]; then
    printf '%s\n' "${matches[0]}"
    return
  fi
  printf '%s\n' "$path"
}

if [ -z "$MODEL" ] || [ "$MODEL" = "auto" ]; then
  MODEL=$(select_default_model "$MODE" "$TOKENIZER_PATH")
fi
MODEL=$(resolve_model_checkpoint_dir "$MODEL")

normalize_ruler_dir() {
  local path=$1
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  elif [[ "$path" == results/* ]]; then
    printf '%s\n' "$ROOT/$path"
  else
    printf '%s\n' "$ROOT/results/ruler/$path"
  fi
}

run_python() {
  if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
    python "$@"
    return
  fi
  if [ -f "$VENV_DIR/bin/python" ]; then
    "$VENV_DIR/bin/python" "$@"
    return
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found, venv not found, and env '$CONDA_ENV' is not active" >&2
    exit 1
  fi
  conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

export PYTHONPATH=$ROOT
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPU

RESULTS_ROOT=$(normalize_ruler_dir "$RESULTS_ROOT")
PREBUILT_DATA_ROOT=$(normalize_ruler_dir "$PREBUILT_DATA_ROOT")

if [ -z "$TOKENIZER_PATH" ]; then
  TOKENIZER_PATH=$(run_python - "$MODEL" <<'PY'
from transformers import AutoConfig
import sys
model = sys.argv[1]
config = AutoConfig.from_pretrained(model)
print(getattr(config, "base_model", model))
PY
)
fi

THRESHOLD_TAG="5e-4"

MODEL_NAME_FOR_PATH=$(basename "$MODEL")
if [[ "$MODE" == *compactattention* ]]; then
  RESULT_BASENAME="${MODEL_NAME_FOR_PATH}_CompactAttentionChunkedPrefill_${MODE}_${THRESHOLD_TAG}"
else
  RESULT_BASENAME="${MODEL_NAME_FOR_PATH}_SeerAttnChunkedPrefill_${MODE}_${THRESHOLD_TAG}"
fi
RESULT_DIR="$RESULTS_ROOT/${RESULT_BASENAME}/synthetic/${SEQ_LEN}"
DATA_DIR="$RESULT_DIR/data"
PRED_DIR="$RESULT_DIR/pred"
mkdir -p "$DATA_DIR" "$PRED_DIR"

TASKS=(
  niah_single_1
  niah_single_2
  niah_single_3
  niah_multikey_1
  niah_multikey_2
  niah_multikey_3
  niah_multivalue
  niah_multiquery
  vt
  cwe
  fwe
  qa_1
  qa_2
)

if [ -n "${TASKS_CSV:-}" ]; then
  IFS=',' read -r -a TASKS <<< "$TASKS_CSV"
fi

for TASK in "${TASKS[@]}"; do
  EXTRA_ARGS=()
  if [ "${SEERATTN_LAST_BLOCK_DENSE:-1}" = "0" ]; then
    EXTRA_ARGS+=(--no-last-block-dense)
  fi
  if [ -n "${SEERATTN_LAST_BLOCK_DENSE_SCOPE:-}" ]; then
    EXTRA_ARGS+=(--last-block-dense-scope "$SEERATTN_LAST_BLOCK_DENSE_SCOPE")
  fi
  if [ -n "${SEERATTN_FINAL_DENSE_TAIL_BLOCKS:-}" ]; then
    EXTRA_ARGS+=(--final-dense-tail-blocks "$SEERATTN_FINAL_DENSE_TAIL_BLOCKS")
  fi
  if [ -n "${DENSE_PREFIX_TOKENS:-}" ]; then
    EXTRA_ARGS+=(--dense-prefix-tokens "$DENSE_PREFIX_TOKENS")
  fi
  if [ "$RUN_MODE" = "quoka_dense" ]; then
    EXTRA_ARGS+=(--quoka-query-ratio "${QUOKA_QUERY_RATIO:-0.25}")
    EXTRA_ARGS+=(--quoka-kv-budget-ratio "${QUOKA_KV_BUDGET_RATIO:-0.25}")
  fi
  if [ "$RUN_MODE" = "compactattn" ] || [ "$RUN_MODE" = "flashprefill_compactattn" ]; then
    if [ "${COMPACTATTENTION_DISABLE_FIRST_CHUNK_DENSE:-${COMPACTATTN_DISABLE_FIRST_CHUNK_DENSE:-0}}" = "1" ]; then
      EXTRA_ARGS+=(--compactattention-disable-first-chunk-dense)
    fi
    EXTRA_ARGS+=(--compactattention-keep-recent-blocks "${COMPACTATTENTION_KEEP_RECENT_BLOCKS:-${COMPACTATTN_KEEP_RECENT_BLOCKS:-2}}")
    EXTRA_ARGS+=(--compactattention-chunked-gate-head-pool "${COMPACTATTENTION_CHUNKED_GATE_HEAD_POOL:-${COMPACTATTN_CHUNKED_GATE_HEAD_POOL:-none}}")
    EXTRA_ARGS+=(--compactattention-pack-impl "${COMPACTATTENTION_PACK_IMPL:-${COMPACTATTN_PACK_IMPL:-indexed_dense}}")
    EXTRA_ARGS+=(--compactattention-indexed-impl "${COMPACTATTENTION_INDEXED_IMPL:-${COMPACTATTN_INDEXED_IMPL:-auto}}")
    EXTRA_ARGS+=(--compactattention-cache-fill-backend "${COMPACTATTENTION_CACHE_FILL_BACKEND:-${COMPACTATTN_CACHE_FILL_BACKEND:-cuda}}")
  fi
  if [ "$RUN_MODE" = "flashprefill_block_sparse" ] || [ "$RUN_MODE" = "flashprefill_compactattn" ]; then
    EXTRA_ARGS+=(--flashprefill-alpha "${FLASHPREFILL_ALPHA:-0.18}")
    EXTRA_ARGS+=(--flashprefill-block-size "${FLASHPREFILL_BLOCK_SIZE:-128}")
    EXTRA_ARGS+=(--flashprefill-attention-sink "${FLASHPREFILL_ATTENTION_SINK:-2}")
    EXTRA_ARGS+=(--flashprefill-window-size "${FLASHPREFILL_WINDOW_SIZE:-4}")
    EXTRA_ARGS+=(--flashprefill-last-n-block "${FLASHPREFILL_LAST_N_BLOCK:-2}")
    EXTRA_ARGS+=(--flashprefill-min-budget "${FLASHPREFILL_MIN_BUDGET:-0}")
  fi
  if [ "${QWEN_LONG_CONTEXT:-0}" = "1" ]; then
    EXTRA_ARGS+=(--qwen-long-context)
    EXTRA_ARGS+=(--qwen-long-context-max-position-embeddings "${QWEN_LONG_CONTEXT_MAX_POSITION_EMBEDDINGS:-131072}")
    EXTRA_ARGS+=(--qwen-yarn-factor "${QWEN_YARN_FACTOR:-4.0}")
    EXTRA_ARGS+=(--qwen-original-max-position-embeddings "${QWEN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-32768}")
  fi
  DATA_FILE="$DATA_DIR/$TASK/validation.jsonl"
  if [ -n "$PREBUILT_DATA_ROOT" ]; then
    PREBUILT_FILE="$PREBUILT_DATA_ROOT/synthetic/$SEQ_LEN/data/$TASK/validation.jsonl"
    if [ -f "$PREBUILT_FILE" ]; then
      DATA_FILE="$PREBUILT_FILE"
    fi
  fi
  if [ ! -f "$DATA_FILE" ]; then
    run_python "$ROOT/eval/ruler/data/prepare.py" \
      --save_dir "$DATA_DIR" \
      --benchmark synthetic \
      --task "$TASK" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --tokenizer_type hf \
      --max_seq_length "$SEQ_LEN" \
      --model_template_type llama-3 \
      --num_samples "$NUM_SAMPLES"
    DATA_FILE="$DATA_DIR/$TASK/validation.jsonl"
  fi

  run_python "$ROOT/eval/ruler/pred/run_chunked_prefill_task_once.py" \
    --model "$MODEL" \
    --task "$TASK" \
    --data-file "$DATA_FILE" \
    --save-dir "$PRED_DIR" \
    --mode "$RUN_MODE" \
    --chunk-size "$CHUNK_SIZE" \
    "${EXTRA_ARGS[@]}"
done

if [ "${SKIP_FINAL_SCORE:-0}" != "1" ]; then
  run_python "$ROOT/eval/ruler/pred/score_chunked_prefill_results.py" --data-dir "$PRED_DIR"
fi

echo "Saved results to $RESULT_DIR"
