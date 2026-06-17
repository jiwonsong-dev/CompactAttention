import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    class _DummyTriton:
        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def cdiv(x, y):
            return (x + y - 1) // y

    class _DummyTL:
        constexpr = int

    triton = _DummyTriton()
    tl = _DummyTL()
    _TRITON_AVAILABLE = False


def direct_indexed_prefill_available() -> bool:
    return bool(_TRITON_AVAILABLE)


def can_use_direct_indexed_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_block_indices: torch.Tensor,
    past_block_counts: torch.Tensor,
    past_len: int,
    block_size: int,
    num_key_value_groups: int,
) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if past_block_indices.ndim != 3 or past_block_counts.ndim != 2:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if not (past_block_indices.is_cuda and past_block_counts.is_cuda):
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False
    if not (past_block_indices.is_contiguous() and past_block_counts.is_contiguous()):
        return False
    if block_size != 64:
        return False
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return False
    bsz, q_len, num_q_heads, _ = q.shape
    if q_len != 1024:
        return False
    if k.shape[0] != bsz or v.shape[0] != bsz:
        return False
    kv_len = k.shape[1]
    if kv_len <= q_len or v.shape[1] != kv_len:
        return False
    if (q_len % block_size) != 0 or (kv_len % block_size) != 0:
        return False
    if past_len != (kv_len - q_len):
        return False
    if past_len < 0 or (past_len % block_size) != 0:
        return False
    num_kv_heads = k.shape[2]
    if num_q_heads % num_kv_heads != 0:
        return False
    if num_key_value_groups != (num_q_heads // num_kv_heads):
        return False
    if past_block_indices.shape[0] != bsz or past_block_indices.shape[1] != num_kv_heads:
        return False
    if past_block_counts.shape[0] != bsz or past_block_counts.shape[1] != num_kv_heads:
        return False
    return True


@triton.jit
def _direct_indexed_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    past_idx_ptr,
    past_counts_ptr,
    out_ptr,
    softmax_scale,
    past_len,
    q_len,
    kv_len,
    num_key_value_groups,
    stride_q_b,
    stride_q_q,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_k,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_k,
    stride_v_h,
    stride_v_d,
    stride_pi_b,
    stride_pi_h,
    stride_pi_s,
    stride_pc_b,
    stride_pc_h,
    stride_o_b,
    stride_o_q,
    stride_o_h,
    stride_o_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = offs_m < q_len

    q_ptrs = (
        q_ptr
        + pid_b * stride_q_b
        + offs_m[:, None] * stride_q_q
        + pid_hq * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    kv_head = pid_hq // num_key_value_groups
    past_count = tl.load(past_counts_ptr + pid_b * stride_pc_b + kv_head * stride_pc_h).to(tl.int32)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    i = 0
    while i < past_count:
        blk = tl.load(past_idx_ptr + pid_b * stride_pi_b + kv_head * stride_pi_h + i * stride_pi_s).to(tl.int32)
        if blk >= 0:
            key_pos = blk * BLOCK_N + offs_n
            key_mask = key_pos < past_len

            k_ptrs = (
                k_ptr
                + pid_b * stride_k_b
                + key_pos[None, :] * stride_k_k
                + kv_head * stride_k_h
                + offs_d[:, None] * stride_k_d
            )
            v_ptrs = (
                v_ptr
                + pid_b * stride_v_b
                + key_pos[:, None] * stride_v_k
                + kv_head * stride_v_h
                + offs_d[None, :] * stride_v_d
            )
            k_block = tl.load(k_ptrs, mask=key_mask[None, :], other=0.0)
            v_block = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)

            scores = tl.dot(q, k_block) * softmax_scale
            valid = q_mask[:, None] & key_mask[None, :]
            scores = tl.where(valid, scores, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        i += 1

    curr_blocks = q_len // BLOCK_N
    cb = 0
    while cb < curr_blocks:
        key_pos = past_len + cb * BLOCK_N + offs_n
        key_mask = key_pos < kv_len
        key_rel = cb * BLOCK_N + offs_n
        causal = key_rel[None, :] <= offs_m[:, None]

        k_ptrs = (
            k_ptr
            + pid_b * stride_k_b
            + key_pos[None, :] * stride_k_k
            + kv_head * stride_k_h
            + offs_d[:, None] * stride_k_d
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_v_b
            + key_pos[:, None] * stride_v_k
            + kv_head * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=key_mask[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)

        scores = tl.dot(q, k_block) * softmax_scale
        valid = q_mask[:, None] & key_mask[None, :] & causal
        scores = tl.where(valid, scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        cb += 1

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = (acc / l_safe[:, None]).to(out_ptr.dtype.element_ty)
    out = tl.where(q_mask[:, None], out, 0.0)

    out_ptrs = (
        out_ptr
        + pid_b * stride_o_b
        + offs_m[:, None] * stride_o_q
        + pid_hq * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(out_ptrs, out, mask=q_mask[:, None])


def run_direct_indexed_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_block_indices: torch.Tensor,
    past_block_counts: torch.Tensor,
    past_len: int,
    block_size: int,
    num_key_value_groups: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(q.shape[-1]))
    if not can_use_direct_indexed_prefill(
        q=q,
        k=k,
        v=v,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        block_size=block_size,
        num_key_value_groups=num_key_value_groups,
    ):
        raise ValueError("run_direct_indexed_prefill called with unsupported inputs.")

    bsz, q_len, num_q_heads, head_dim = q.shape
    out = torch.empty_like(q)
    block_m = 32
    grid = (bsz, num_q_heads, triton.cdiv(q_len, block_m))

    _direct_indexed_prefill_kernel[grid](
        q,
        k,
        v,
        past_block_indices,
        past_block_counts,
        out,
        float(softmax_scale),
        int(past_len),
        int(q_len),
        int(k.shape[1]),
        int(num_key_value_groups),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        past_block_indices.stride(0),
        past_block_indices.stride(1),
        past_block_indices.stride(2),
        past_block_counts.stride(0),
        past_block_counts.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_size,
        BLOCK_D=head_dim,
        num_warps=4,
        num_stages=2,
    )
    return out
