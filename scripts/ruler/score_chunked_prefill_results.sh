#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 RESULT_DIR_OR_PRED_DIR [CONDA_ENV]"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
TARGET=$1
CONDA_ENV=${2:-seer_ruler}
VENV_DIR="${VENV_DIR:-/workspace/compactattn}"

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

run_python "$ROOT/eval/ruler/pred/score_chunked_prefill_results.py" --data-dir "$TARGET"
