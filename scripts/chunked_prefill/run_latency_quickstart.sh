#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

SEQ_LENS=8192,16384,32768,65536,131072,262144
RUNS=3
WARMUP=1
CONDA_ENV=compactattn
MODEL_OVERRIDE=
DENSE_GPU=skip
DENSE_PREFIX_TOKENS=
BATCH_SIZE=${BATCH_SIZE:-1}

usage() {
  cat <<'EOF'
Usage:
  single method:
    run_latency_quickstart.sh single MODE GPU [options]

  table sweep:
    run_latency_quickstart.sh table CHUNK_SIZE OUT_DIR GPU_SEER_BS GPU_SEER_CD [options]

Options:
  --chunk-size N            Chunk size for single-mode runs. Default: 1024
  --seq-lens CSV            Default: 8192,16384,32768,65536,131072,262144
  --runs N                  Default: 3
  --warmup N                Default: 1
  --batch-size N            Default: 1
  --conda-env NAME          Default: compactattn
  --model ID                Override automatic model choice for all launched rows
  --seer-model ID           Backward-compatible alias for --model
  --dense-gpu GPU           Add dense row to table mode. Default: skip
  --dense-prefix-tokens N   Apply the same dense prefix to sparse/compactattn/QUOKA wrappers
  -h, --help                Show this help

MODE:
  dense | seer_block_sparse | compactattention | quoka_dense

Examples:
  bash scripts/chunked_prefill/run_latency_quickstart.sh \
    single compactattention 0 --chunk-size 1024 --batch-size 8 --dense-prefix-tokens 16384

  bash scripts/chunked_prefill/run_latency_quickstart.sh \
    table 1024 h100_chunk1024 0 1 --batch-size 8 --conda-env compactattn --dense-prefix-tokens 16384

  bash scripts/chunked_prefill/run_latency_quickstart.sh \
    table 2048 h100_chunk2048 0 1 --dense-gpu 4

Advanced tuning knobs still work through env vars, for example:
  COMPACTATTN_INDEXED_IMPL, COMPACTATTN_CACHE_FILL_BACKEND, COMPACTATTN_KEEP_RECENT_BLOCKS,
  SEERATTN_FINAL_DENSE_TAIL_BLOCKS.
EOF
}

export_dense_prefix_env() {
  if [ -n "$DENSE_PREFIX_TOKENS" ]; then
    export DENSE_PREFIX_TOKENS="$DENSE_PREFIX_TOKENS"
  fi
  export BATCH_SIZE="$BATCH_SIZE"
}

mode_to_combo() {
  case "$1" in
    dense)
      printf 'dense none\n'
      ;;
    seer_block_sparse|block_sparse_final_chunk_last_block_dense)
      printf 'block_sparse seer\n'
      ;;
    seer_compactattn|compactattention|compactattn_final_dense_tail2)
      printf 'compactattn seer\n'
      ;;
    quoka_dense)
      printf 'dense quoka\n'
      ;;
    *)
      echo "Unknown mode: $1" >&2
      exit 1
      ;;
  esac
}

parse_common_opts() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --chunk-size)
        CHUNK_SIZE=$2
        shift 2
        ;;
      --seq-lens)
        SEQ_LENS=$2
        shift 2
        ;;
      --runs)
        RUNS=$2
        shift 2
        ;;
      --warmup)
        WARMUP=$2
        shift 2
        ;;
      --batch-size)
        BATCH_SIZE=$2
        shift 2
        ;;
      --conda-env)
        CONDA_ENV=$2
        shift 2
        ;;
      --model|--seer-model)
        MODEL_OVERRIDE=$2
        shift 2
        ;;
      --dense-gpu)
        DENSE_GPU=$2
        shift 2
        ;;
      --dense-prefix-tokens)
        DENSE_PREFIX_TOKENS=$2
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
}

run_single() {
  if [ "$#" -lt 2 ]; then
    usage >&2
    exit 1
  fi
  local mode=$1
  local gpu=$2
  shift 2

  CHUNK_SIZE=1024
  parse_common_opts "$@"
  export_dense_prefix_env

  local execution_mode selection_method
  read -r execution_mode selection_method <<<"$(mode_to_combo "$mode")"

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
  "${cmd[@]}"
}

run_table() {
  if [ "$#" -lt 4 ]; then
    usage >&2
    exit 1
  fi
  local chunk_size=$1
  local out_dir=$2
  local gpu_seer_bs=$3
  local gpu_seer_cd=$4
  shift 4

  parse_common_opts "$@"
  export_dense_prefix_env

  "$SCRIPT_DIR/run_latency_table_sweep.sh" \
    "$chunk_size" \
    "$out_dir" \
    "$DENSE_GPU" \
    "$gpu_seer_bs" \
    "$gpu_seer_cd" \
    "$SEQ_LENS" \
    "$RUNS" \
    "$WARMUP" \
    "$CONDA_ENV" \
    "$MODEL_OVERRIDE"
}

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 1
fi

COMMAND=$1
shift

case "$COMMAND" in
  single)
    run_single "$@"
    ;;
  table)
    run_table "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage >&2
    exit 1
    ;;
esac
