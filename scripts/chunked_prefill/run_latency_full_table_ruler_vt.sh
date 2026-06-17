#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "Usage: $0 CHUNK_SIZE OUT_DIR GPU0 GPU1 GPU2 GPU3 [SEQ_LENS] [CONDA_ENV]"
  echo "  Runs the full VT real-input latency table across four GPUs."
  echo "  Use 'skip' for a GPU slot to omit that batch."
  echo "  Set BATCH_SIZE=N in the environment for multi-batch runs. Default: 1"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

CHUNK_SIZE=$1
OUT_DIR=$2
GPU0=$3
GPU1=$4
GPU2=$5
GPU3=$6
SEQ_LENS=${7:-8192,16384,32768,65536,131072,262144}
CONDA_ENV=${8:-compactattn}
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

run_batch() {
  local gpu=$1
  shift
  if [ "$gpu" = "skip" ]; then
    echo "[skip] batch"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" bash -lc "
set -euo pipefail
cd '$ROOT'
export PYTHONPATH='$ROOT'
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export BATCH_SIZE='$BATCH_SIZE'
export PROMPT_SOURCE=ruler
export INPUT_SCHEDULE=cycle
export DENSE_ATTN_IMPL=flash_attention_2
export COMPACTATTN_FALLBACK_RATIO=1.1
export COMPACTATTN_KEEP_RECENT_BLOCKS=2
export COMPACTATTN_PACK_IMPL=indexed_dense
export COMPACTATTN_INDEXED_IMPL=fa2_paged
export COMPACTATTN_CACHE_FILL_BACKEND=cuda
$*
" &
}

batch0_cmd=$(cat <<EOF
bash '$SCRIPT_DIR/run_unified_latency_sweep.sh' $GPU0 dense none '$CHUNK_SIZE' '$SEQ_LENS' 3 1 '$CONDA_ENV' > '$OUT_DIR/dense.log' 2>&1
bash '$SCRIPT_DIR/run_unified_latency_sweep.sh' $GPU0 dense quoka '$CHUNK_SIZE' '$SEQ_LENS' 3 1 '$CONDA_ENV' > '$OUT_DIR/quoka_dense.log' 2>&1
bash '$SCRIPT_DIR/run_unified_latency_sweep.sh' $GPU0 block_sparse seer '$CHUNK_SIZE' '$SEQ_LENS' 3 1 '$CONDA_ENV' > '$OUT_DIR/seer_block_sparse.log' 2>&1
bash '$SCRIPT_DIR/run_unified_latency_sweep.sh' $GPU0 compactattn seer '$CHUNK_SIZE' '$SEQ_LENS' 3 1 '$CONDA_ENV' > '$OUT_DIR/seer_compactattn.log' 2>&1
EOF
)

batch1_cmd=$(cat <<EOF
export DENSE_PREFIX_TOKENS=16384
bash '$SCRIPT_DIR/run_unified_latency_sweep.sh' $GPU1 block_sparse seer '$CHUNK_SIZE' '$SEQ_LENS' 3 1 '$CONDA_ENV' > '$OUT_DIR/seer_block_sparse_denseprefix16k.log' 2>&1
bash '$SCRIPT_DIR/run_unified_latency_sweep.sh' $GPU1 compactattn seer '$CHUNK_SIZE' '$SEQ_LENS' 3 1 '$CONDA_ENV' > '$OUT_DIR/seer_compactattn_denseprefix16k.log' 2>&1
EOF
)

batch2_cmd=""

batch3_cmd=""

echo "[launch] vt real-input full table -> $OUT_DIR"
run_batch "$GPU0" "$batch0_cmd"
run_batch "$GPU1" "$batch1_cmd"
run_batch "$GPU2" "$batch2_cmd"
run_batch "$GPU3" "$batch3_cmd"

wait
echo "[done] logs written to $OUT_DIR"
