# CompactAttention: Accelerating Chunked Prefill with Block-Union KV Selection

Official implementation of the paper **"CompactAttention: Accelerating Chunked Prefill with Block-Union KV Selection"**

<p align="center">
  <a href="https://arxiv.org/abs/2605.16839"><img src="https://img.shields.io/badge/arXiv-2605.16839-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <img src="assets/pareto.png">
</p>

<p align="center">
  <img src="assets/method.png">
</p>

**CompactAttention** is a sparse-attention method designed for **chunked prefill** serving of long-context LLMs. It reframes 2D block-sparse masks as **KV-selection signals** and gathers a per-group KV block table so a dense FlashAttention kernel can do the work — recovering dense-kernel throughput while keeping full-attention accuracy.

## 🚀 Key Features

- **Built for Chunked Prefill**
  Targets the regime real serving systems run in — small query chunks attending to a long, growing KV cache — where one-shot sparse methods break down.

- **Block-Union KV Selection**
  Converts any 2D block-sparse mask into a GQA-aware per-group KV block table via Q-block union and intra-group union, with no sparse-kernel launch overhead.

- **Dense Kernel, Sparse Memory**
  Runs a dense FlashAttention over the selected KV blocks, trading a small amount of redundant work for much higher SM utilization at typical chunk sizes.

- **Selector-Agnostic**
  Plugs in on top of existing selectors — both training-free (FlashPrefill-style) and learned (SeerAttention-style AttnGate) families are supported.

- **Up to 2.72× Attention Speedup**
  On LLaMA-3.1-8B-Instruct at 128K context, matches dense-attention accuracy on RULER while delivering up to **2.72×** attention speedup.

## Key Results

<p align="center">
  <img src="assets/speedup.png">
</p>

<p align="center">
  <img src="assets/accuracy.png">
</p>

## Methods

| Method | Selector | Execution evaluated |
|---|---|---|
| **Dense** | — | full-kv (HuggingFace `flash_attention_2`) |
| **QUOKA** | dense-side baseline (no block-level structure) | dense |
| **SeerAttention** | learned block-level gate (AttnGate) | `block_sparse` and `compactattn` |
| **FlashPrefill** | training-free selector | `block_sparse` and `compactattn` |

`compactattn` is the Block-Union compact-dense execution path; `block_sparse` is the
conventional irregular sparse kernel given the same selected mask.

## Installation

**Prerequisites:** Linux with an NVIDIA GPU and a CUDA toolkit (`nvcc`, used to build
the `compact_attn._C_compactattn` extension). HuggingFace reference: `transformers==4.54.1`.

```bash
conda create -yn compactattn python=3.11
conda activate compactattn

# PyTorch matching your CUDA runtime first
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation

# Builds the CUDA extension and initializes the third_party/cutlass submodule
pip install -e .

# Verify the extension built
python -c "import compact_attn._C_compactattn; print('OK')"
```

## Datasets

Benchmark datasets are **not** redistributed; fetch them with the provided scripts
(see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing):

```bash
bash eval/ruler/data/synthetic/json/download_qa_dataset.sh
python eval/ruler/data/synthetic/json/download_paulgraham_essay.py
```

## Usage

### 1. Chunked-prefill latency

A single GPU id runs **TP1**; pass a comma-separated list (e.g. `0,1`) for tensor parallelism.

```bash
# one method at TP1: <gpu> <exec_family> <selector_family> <chunk_size>
bash scripts/chunked_prefill/run_unified_latency_sweep.sh 0 compactattn seer 1024

# high-level entrypoint (single method or a comparison table)
bash scripts/chunked_prefill/run_latency_quickstart.sh --help
```

`exec_family ∈ {dense, block_sparse, compactattn}`, `selector_family ∈
{none, quoka, seer, flashprefill}`. Details:
[scripts/chunked_prefill/README.md](scripts/chunked_prefill/README.md).

### 2. RULER accuracy

```bash
bash scripts/ruler/prepare_ruler_prebuilt_data.sh \
  SeerAttention/SeerAttention-Llama-3.1-8B-AttnGates results/ruler_llama_3_1_8b 25 compactattn
bash scripts/ruler/run_chunked_prefill_benchmark.sh
```

### 3. LongBench-v2

```bash
bash eval/longbench_v2/run_method_grid.sh
```

## Repository layout

- `compact_attn/` — core library: model integrations (`prefill_sparse/`), shared
  attention forward logic (`modules/`), kernels (`kernels/`, plus the
  `_C_compactattn` CUDA source under `csrc/`).
- `eval/` — benchmark drivers: `chunked_prefill/` (latency), `ruler/`, `longbench_v2/`.
- `scripts/` — high-level wrappers for latency sweeps and RULER.

## Citation

If you find CompactAttention relevant to your research, please cite our work:

```bibtex
@article{compactattention,
  title   = {CompactAttention: Accelerating Chunked Prefill with Block-Union KV Selection},
  author  = {Song, Jiwon and Jo, Dongwon and Kang, Beomseok and Kim, Jae-Joon},
  journal = {arXiv preprint arXiv:2605.16839},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE). This project builds on and adapts several
third-party projects; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

This project builds on or compares against
[SeerAttention](https://github.com/microsoft/SeerAttention) (base codebase + gate
scoring), [FlashAttention](https://github.com/Dao-AILab/flash-attention),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention),
[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass),
[RULER](https://github.com/NVIDIA/RULER), and
[LongBench](https://github.com/THUDM/LongBench).
