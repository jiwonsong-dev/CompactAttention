#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 MODEL DATA_ROOT [NUM_SAMPLES] [CONDA_ENV] [TOKENIZER_PATH] [MODEL_TEMPLATE_TYPE]"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL=$1
DATA_ROOT=$2
NUM_SAMPLES=${3:-25}
CONDA_ENV=${4:-compactattn}
VENV_DIR="${VENV_DIR:-/workspace/compactattn}"
TOKENIZER_PATH=${5:-}
MODEL_TEMPLATE_TYPE=${6:-}

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

DATA_ROOT=$(normalize_ruler_dir "$DATA_ROOT")

SEQ_LENS=(8192 16384 32768 65536 131072)
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

ensure_nltk_resource() {
  local package=$1
  local resource=$2
  run_python - "$package" "$resource" <<'PY'
import sys
import nltk

package = sys.argv[1]
resource = sys.argv[2]

try:
    nltk.data.find(resource)
    print(f"[nltk] found {resource}")
except LookupError:
    print(f"[nltk] downloading {package} for {resource}")
    nltk.download(package)
PY
}

ensure_nltk_resource punkt tokenizers/punkt
ensure_nltk_resource punkt_tab tokenizers/punkt_tab/english

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

if [ -z "$MODEL_TEMPLATE_TYPE" ]; then
  MODEL_TEMPLATE_TYPE=$(run_python - "$MODEL" <<'PY'
import sys

model = sys.argv[1].lower()
if "qwen3" in model:
    print("hf-chat-no-thinking")
elif "qwen" in model:
    print("qwen2.5")
else:
    print("llama-3")
PY
)
fi

mkdir -p "$DATA_ROOT/synthetic"

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  SAVE_DIR="$DATA_ROOT/synthetic/$SEQ_LEN/data"
  mkdir -p "$SAVE_DIR"
  for TASK in "${TASKS[@]}"; do
    TARGET="$SAVE_DIR/$TASK/validation.jsonl"
    if [ -f "$TARGET" ]; then
      echo "[skip] seq=$SEQ_LEN task=$TASK existing=$TARGET"
      continue
    fi
    echo "[prepare] seq=$SEQ_LEN task=$TASK"
    run_python "$ROOT/eval/ruler/data/prepare.py" \
      --save_dir "$SAVE_DIR" \
      --benchmark synthetic \
      --task "$TASK" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --tokenizer_type hf \
      --max_seq_length "$SEQ_LEN" \
      --model_template_type "$MODEL_TEMPLATE_TYPE" \
      --num_samples "$NUM_SAMPLES"
  done
done

echo "Saved prebuilt data under $DATA_ROOT"
