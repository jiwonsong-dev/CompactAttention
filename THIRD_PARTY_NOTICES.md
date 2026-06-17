# Third-Party Notices

CompactAttention is released under the MIT License (see [LICENSE](LICENSE)). It
builds on, vendors, or adapts code and data from the third-party projects listed
below. All trademarks and copyrights are the property of their respective owners.
Each component remains under its own license; please consult the upstream
projects for the authoritative license texts.

## Code

| Project | Upstream | License | Where it is used |
|---|---|---|---|
| **SeerAttention** | https://github.com/microsoft/SeerAttention | MIT | This repository began as a fork of SeerAttention. The Python package `compact_attn`, the block-level gate scoring, and much of the model-integration scaffolding derive from it. |
| **Block-Sparse-Attention** | https://github.com/mit-han-lab/Block-Sparse-Attention | MIT | `compact_attn/kernels/block_sparse_attention/` CUDA kernel. |
| **FlashAttention** | https://github.com/Dao-AILab/flash-attention | BSD-3-Clause | Dense / varlen attention kernels and helpers used across `compact_attn/modules/` and `compact_attn/kernels/`. |
| **FlashInfer** | https://github.com/flashinfer-ai/flashinfer | Apache-2.0 | FlashInfer-based execution backends (`fi_paged`, `fi_zero_copy`). |
| **NVIDIA CUTLASS** | https://github.com/NVIDIA/cutlass | BSD-3-Clause | Git submodule at `third_party/cutlass`, used to build the `compact_attn._C_compactattn` CUDA extension. |
| **HuggingFace Transformers** | https://github.com/huggingface/transformers | Apache-2.0 | Model-integration files under `compact_attn/prefill_sparse/` adapt Transformers modeling code. |
| **NVIDIA RULER** | https://github.com/NVIDIA/RULER | Apache-2.0 | `eval/ruler/` long-context accuracy benchmark harness. |
| **LongBench v2** | https://github.com/THUDM/LongBench | MIT | `eval/longbench_v2/` evaluation harness. |

## Datasets

The benchmark datasets are **not** redistributed in this repository. They are
fetched on demand by the provided download scripts and remain subject to their
original licenses:

- **HotpotQA** (CC BY-SA 4.0) and **SQuAD** (CC BY-SA 4.0) — fetched by
  `eval/ruler/data/synthetic/json/download_qa_dataset.sh`.
- **Paul Graham essays** (© Paul Graham, all rights reserved) — fetched from the
  author's site by `eval/ruler/data/synthetic/json/download_paulgraham_essay.py`.
  These are downloaded locally for evaluation only and are not redistributed.

## Citation of upstream methods

CompactAttention compares against and reuses ideas from SeerAttention
(arXiv:2410.13276). Please cite the original works when
appropriate (see the Acknowledgements section of the README).
