# Chunked-Prefill Latency Sweeps

These wrappers measure chunked-prefill attention latency for the compared methods
(dense / QUOKA / SeerAttention / FlashPrefill, with `block_sparse` and `compactattn`
execution). They are meant to be runnable right after `git clone`, as long as the
model and Python env are available.

> **TP:** a single GPU id runs **TP1** (the default). A comma-separated GPU set
> enables tensor parallelism; in the comparison table each GPU slot runs one method
> at TP1.

## Quickstart

High-level entrypoint:

```bash
bash scripts/chunked_prefill/run_latency_quickstart.sh --help
```

Two patterns:

```bash
# 1) single — one method at TP1 on GPU 0
bash scripts/chunked_prefill/run_latency_quickstart.sh \
  single compactattention 0 --chunk-size 1024 --batch-size 1 --conda-env compactattn

# 2) table — comparison sweep, one method per GPU slot (each TP1)
bash scripts/chunked_prefill/run_latency_quickstart.sh \
  table 1024 out_chunk1024 0 1 --batch-size 1 --conda-env compactattn
```

## Unified sweep (choose execution × selector separately)

```bash
bash scripts/chunked_prefill/run_unified_latency_sweep.sh 0 compactattn seer 1024
```

`<gpu> <exec_family> <selector_family> <chunk_size>`, where `exec_family ∈
{dense, block_sparse, compactattn}` and `selector_family ∈ {none, quoka, seer,
flashprefill}`. Set `BATCH_SIZE` in the environment for multi-batch runs; pass a
comma-separated GPU list to enable `--device-map auto`.

## Full comparison table (real VT input)

```bash
# one method per GPU slot (each TP1); use as many GPUs as you have free
bash scripts/chunked_prefill/run_latency_full_table_ruler_vt.sh \
  1024 vt_table_chunk1024 0 1
```

Runs the full method comparison on the RULER `vt` task, **one method per GPU slot
(each TP1)**. Defaults: `seq_lens = 8192,16384,32768,65536,131072,262144`,
`warmup=1`, `runs=3`, `input_schedule=cycle`.

## Real-input defaults

For real-input latency tables prefer `prompt_source=ruler`, `task=vt`,
`input_schedule=cycle`, `warmup=1`, `runs=3`:

- `vt` fills long-context target lengths more reliably than `cwe`.
- `cycle` rotates through different samples across warmup/measured runs, so the
  table is not overfit to a single prompt (methods differ in prompt sensitivity).

## Outputs

Results write to `results/latency/<OUT_DIR>/` with `summary.csv` and `summary.md`.
Useful env overrides: `BATCH_SIZE`, `COMPACTATTN_INDEXED_IMPL`,
`COMPACTATTN_CACHE_FILL_BACKEND`, `COMPACTATTN_KEEP_RECENT_BLOCKS`,
`SEERATTN_FINAL_DENSE_TAIL_BLOCKS`.

## Interpretation

These sweeps answer a specific question: in chunked prefill, does block-sparse
attention turn its FLOPs reduction into proportional wall-clock speedup? The working
conclusion is *not reliably* — even with the same selector, a slightly denser but
more regular `compactattn` (compact dense) execution is often faster than
`block_sparse`, because block-sparse pays extra non-linear overheads (irregular
metadata/indexing, lower kernel utilization, fixed costs that do not scale with the
reduced FLOPs). Both paths also pay the common chunked-prefill overheads (chunk loop,
selection/scoring, qkv/rope/cache-update/output projection).
