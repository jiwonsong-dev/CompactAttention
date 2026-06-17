import os
import subprocess

from setuptools import setup, find_packages

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except Exception:  # pragma: no cover - setup-time optional dependency
    torch = None
    BuildExtension = None
    CUDAExtension = None
    CUDA_HOME = None

# Initialize the CUTLASS submodule when building from a git checkout. Guarded so
# that installs from a source tarball (no .git directory) do not fail.
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_here, ".git")):
    subprocess.run(
        ["git", "submodule", "update", "--init", "third_party/cutlass"],
        check=False,
    )


ext_modules = []
cmdclass = {}

if CUDAExtension is not None and CUDA_HOME is not None:
    ext_modules.append(
        CUDAExtension(
            name="compact_attn._C_compactattn",
            sources=[
                "compact_attn/csrc/compactattn/cache_fill.cpp",
                "compact_attn/csrc/compactattn/cache_fill_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    )
    cmdclass["build_ext"] = BuildExtension


def _read_readme():
    try:
        with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


setup(
    name="compact-attention",
    version="0.1.0",
    description=(
        "CompactAttention: accelerating chunked prefill with "
        "block-union KV selection"
    ),
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    author="Jiwon Song",
    license="MIT",
    url="https://github.com/jiwonsong-dev/CompactAttention",
    project_urls={
        "Paper": "https://arxiv.org/abs/2605.16839",
        "Source": "https://github.com/jiwonsong-dev/CompactAttention",
    },
    python_requires=">=3.10",
    # NOTE: `seerattn_*` config fields and `SeerAttention` class names are kept
    # for compatibility with published *-AttnGates checkpoints.
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
