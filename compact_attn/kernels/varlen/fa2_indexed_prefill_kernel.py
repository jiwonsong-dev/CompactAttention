"""
Placeholder module for FA2 indexed prefill specialized kernels.

The v0 integration routes through `flash_attn_with_kvcache` via
`fa2_indexed_prefill.py`. This file intentionally exists as the dedicated
kernel anchor point for future lower-level implementations.
"""


def kernel_available() -> bool:
    return False
