# Ruler Test

The RULER benchmark scripts is modified from [MInference's repo](https://github.com/microsoft/MInference/tree/main/experiments/ruler).

To run the RULER benchmark, you need first install the requirements:
```
    pip install Cython
    pip install -r requirements.txt
```

To reproduce SeerAttn results on RULER, run:
```
    bash eval_ruler.sh
```

Noted that the some package version like huggingface-hub in RULER might confict with others. If you find any errors, feel free to leave an issue.

## Chunked Prefill

For chunked-prefill runs in this repo, use the dedicated scripts below.

Important:
- The current chunked-prefill decoder wrappers and benchmark runner in this repo are Llama-3.1 / Llama-family specific.
- If you want reusable RULER data for another tokenizer family such as Qwen2.5, prepare that data separately and keep it in a separate root.

### Run a full 32K/64K benchmark

```bash
bash scripts/ruler/run_chunked_prefill_benchmark.sh \
  SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates \
  ./results \
  block_sparse \
  1 \
  32768 \
  25 \
  1024 \
  seer_ruler
```

Arguments:
- `MODEL`: checkpoint path or HF repo
- `RESULTS_ROOT`: root folder for outputs
- `MODE`: `dense`, `block_sparse`, `seer_compactattention`, or
  `flashprefill_compactattention`. The legacy `compactattn` mode is still accepted
  as an alias for Seer-based CompactAttention.
- `GPU`: physical GPU index
- `SEQ_LEN`: sequence length, e.g. `32768` or `65536`
- optional: `NUM_SAMPLES`, `CHUNK_SIZE`, `THRESHOLD`, `CONDA_ENV`, `TOKENIZER_PATH`

The script prepares task data, runs all 13 synthetic tasks, and writes:
- `data/`
- `pred/*.jsonl`
- `pred/summary.csv`

### Prebuild reproducible synthetic data

If you want fixed synthetic data prepared ahead of time for `8K/16K/32K/64K/128K`, generate it once and reuse it across runs:

```bash
bash scripts/ruler/prepare_ruler_prebuilt_data.sh \
  SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates \
  ./results/ruler_prebuilt_data_llama_3_1_8b \
  25 \
  seer_ruler
```

To prepare the prebuilt data:

```bash
bash scripts/ruler/prepare_ruler_prebuilt_data.sh \
  SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates results/ruler_llama_3_1_8b 25 compactattn
```

This defaults to:

- model: `SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates`
- output root: `results/ruler_llama_3_1_8b`
- samples per task: `25`
- conda env: `compactattn`

Example for Qwen-tokenizer-aligned prebuilt data:

```bash
bash scripts/ruler/prepare_ruler_prebuilt_data.sh \
  Qwen/Qwen2.5-7B-Instruct \
  results/ruler_qwen_2_5_7b \
  25 \
  compactattn \
  Qwen/Qwen2.5-7B-Instruct \
  qwen2.5
```

This creates:

```text
results/ruler_prebuilt_data_llama_3_1_8b/
  synthetic/
    8192/data/<task>/validation.jsonl
    16384/data/<task>/validation.jsonl
    32768/data/<task>/validation.jsonl
    65536/data/<task>/validation.jsonl
    131072/data/<task>/validation.jsonl
```

By convention, keep separate roots per tokenizer family, for example:
- `results/ruler_prebuilt_data_llama_3_1_8b`
- `results/ruler_prebuilt_data_qwen2_5_7b`

### Prebuild Qwen2.5 data

Example for Qwen2.5-7B-Instruct:

```bash
bash scripts/ruler/prepare_ruler_prebuilt_data.sh \
  Qwen/Qwen2.5-7B-Instruct \
  ./results/ruler_prebuilt_data_qwen2_5_7b \
  25 \
  seer_ruler \
  Qwen/Qwen2.5-7B-Instruct \
  qwen2.5
```

This only prepares tokenizer-aligned synthetic data. It does not imply that the current chunked-prefill decoder runner supports Qwen inference.

To force benchmark runs to reuse the prebuilt files instead of regenerating task data:

```bash
RULER_PREBUILT_DATA_ROOT=./results/ruler_prebuilt_data_llama_3_1_8b \
bash scripts/ruler/run_chunked_prefill_benchmark.sh \
  SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates \
  ./results \
  block_sparse \
  1 \
  32768 \
  25 \
  1024 \
  seer_ruler
```

### Recommended CompactAttention example

If you want Seer-based CompactAttention with first chunk processed by the selected path instead of dense prefill:

```bash
COMPACTATTN_DISABLE_FIRST_CHUNK_DENSE=1 \
bash scripts/ruler/run_chunked_prefill_benchmark.sh \
  SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates \
  ./results \
  seer_compactattention \
  2 \
  32768 \
  25 \
  1024 \
  seer_ruler
```

Optional CompactAttention environment knobs:
- Prefer the `COMPACTATTENTION_*` names; the older `COMPACTATTN_*` names remain as aliases.
- `COMPACTATTENTION_DISABLE_FIRST_CHUNK_DENSE=1`
- `COMPACTATTENTION_KEEP_RECENT_BLOCKS=2`
- `COMPACTATTENTION_PACK_IMPL=indexed_dense`
- `COMPACTATTENTION_INDEXED_IMPL=fi_zero_copy`
- `COMPACTATTENTION_CACHE_FILL_BACKEND=cuda`

### Score an existing result directory

```bash
bash scripts/ruler/score_chunked_prefill_results.sh \
  ./results/SeerAttention-Llama-3.1-8B-AttnGates_SeerAttnChunkedPrefill_block_sparse_5e-4/synthetic/32768/pred \
  seer_ruler
```

You can also pass the parent result directory instead of the `pred/` directory. The script writes a single `summary.csv` into the prediction directory.
