#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 CHUNK_SIZE OUT_DIR GPU_DENSE GPU_SEER_BS GPU_COMPACTATTN [SEQ_LENS] [RUNS] [WARMUP] [CONDA_ENV] [MODEL_OVERRIDE]"
  echo "  Use 'skip' for a GPU slot to omit that row."
  echo "  Set BATCH_SIZE=N in the environment for multi-batch runs. Default: 1"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

CHUNK_SIZE=$1
OUT_DIR=$2
GPU_DENSE=$3
GPU_SEER_BS=$4
GPU_COMPACTATTN=$5
SEQ_LENS=${6:-8192,16384,32768,65536,131072,262144}
RUNS=${7:-3}
WARMUP=${8:-1}
CONDA_ENV=${9:-compactattn}
MODEL_OVERRIDE=${10:-}
BATCH_SIZE=${BATCH_SIZE:-1}

normalize_latency_dir() {
  local path=$1
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  elif [[ "$path" == results/* ]]; then
    printf '%s\n' "$ROOT/$path"
  else
    printf '%s\n' "$ROOT/results/latency/$path"
  fi
}

OUT_DIR=$(normalize_latency_dir "$OUT_DIR")

mkdir -p "$OUT_DIR"

launch_row() {
  local label=$1
  local gpu=$2
  local execution_mode=$3
  local selection_method=$4
  shift 2
  local log_path="$OUT_DIR/${label}_chunk${CHUNK_SIZE}.log"
  if [ "$gpu" = "skip" ]; then
    echo "[skip] $label"
    return
  fi
  echo "[launch] $label gpu=$gpu batch=$BATCH_SIZE log=$log_path"
  local cmd=(
    "$SCRIPT_DIR/run_unified_latency_sweep.sh"
    "$gpu"
    "$execution_mode"
    "$selection_method"
    "$CHUNK_SIZE"
    "$SEQ_LENS"
    "$RUNS"
    "$WARMUP"
    "$CONDA_ENV"
  )
  if [ -n "$MODEL_OVERRIDE" ]; then
    cmd+=("$MODEL_OVERRIDE")
  fi
  "${cmd[@]}" >"$log_path" 2>&1 &
}

launch_row dense "$GPU_DENSE" dense none
launch_row seer_block_sparse "$GPU_SEER_BS" block_sparse seer
launch_row compactattention "$GPU_COMPACTATTN" compactattn seer

wait
echo "[done] logs written to $OUT_DIR"
