import math
from typing import Dict, Optional, Tuple

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


def fa2_indexed_triton_available() -> bool:
    return bool(_TRITON_AVAILABLE)


def can_use_fa2_indexed_prefill(
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
    if num_key_value_groups != 4:
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
    if num_q_heads != num_kv_heads * 4:
        return False
    if past_block_indices.shape[0] != bsz or past_block_indices.shape[1] != num_kv_heads:
        return False
    if past_block_counts.shape[0] != bsz or past_block_counts.shape[1] != num_kv_heads:
        return False
    return True


@triton.jit
def _online_update(
    q_block,
    k_block,
    v_block,
    valid_mask,
    m_i,
    l_i,
    acc,
    softmax_scale,
):
    scores = tl.dot(q_block, k_block) * softmax_scale
    scores = tl.where(valid_mask, scores, float("-inf"))
    m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
    p = tl.exp(scores - m_ij[:, None])
    l_ij = tl.sum(p, axis=1)
    alpha = tl.exp(m_i - m_ij)
    acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    return m_i, l_i, acc


@triton.jit
def _fa2_indexed_prefill_kernel_v2_g4(
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
    pid_hkv = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_mh = tl.arange(0, BLOCK_M * 4)
    offs_h = offs_mh // BLOCK_M
    offs_m_local = offs_mh - offs_h * BLOCK_M
    offs_m = pid_m * BLOCK_M + offs_m_local
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < q_len

    hq = pid_hkv * 4 + offs_h

    q_ptrs = (
        q_ptr
        + pid_b * stride_q_b
        + offs_m[:, None] * stride_q_q
        + hq[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M * 4], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M * 4], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M * 4, BLOCK_D], dtype=tl.float32)

    past_count = tl.load(past_counts_ptr + pid_b * stride_pc_b + pid_hkv * stride_pc_h).to(tl.int32)
    i = 0
    while i < past_count:
        blk = tl.load(
            past_idx_ptr + pid_b * stride_pi_b + pid_hkv * stride_pi_h + i * stride_pi_s
        ).to(tl.int32)
        key_pos = blk * BLOCK_N + offs_n
        key_mask = key_pos < past_len

        k_ptrs = (
            k_ptr
            + pid_b * stride_k_b
            + key_pos[None, :] * stride_k_k
            + pid_hkv * stride_k_h
            + offs_d[:, None] * stride_k_d
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_v_b
            + key_pos[:, None] * stride_v_k
            + pid_hkv * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=key_mask[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
        valid = q_mask[:, None] & key_mask[None, :]

        m_i, l_i, acc = _online_update(
            q_block, k_block, v_block, valid, m_i, l_i, acc, softmax_scale
        )
        i += 1

    curr_blocks = q_len // BLOCK_N
    q_tile_start = pid_m * BLOCK_M
    q_tile_end = q_tile_start + (BLOCK_M - 1)
    cb_first = q_tile_start // BLOCK_N
    cb_last = q_tile_end // BLOCK_N
    if cb_last >= curr_blocks:
        cb_last = curr_blocks - 1

    cb = 0
    while cb < cb_first:
        k_block_start = cb * BLOCK_N
        key_pos = past_len + k_block_start + offs_n
        key_mask = key_pos < kv_len

        k_ptrs = (
            k_ptr
            + pid_b * stride_k_b
            + key_pos[None, :] * stride_k_k
            + pid_hkv * stride_k_h
            + offs_d[:, None] * stride_k_d
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_v_b
            + key_pos[:, None] * stride_v_k
            + pid_hkv * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=key_mask[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
        valid = q_mask[:, None] & key_mask[None, :]

        m_i, l_i, acc = _online_update(
            q_block, k_block, v_block, valid, m_i, l_i, acc, softmax_scale
        )
        cb += 1

    cb = 0
    while cb <= (cb_last - cb_first):
        cur_cb = cb_first + cb

        k_block_start = cur_cb * BLOCK_N

        key_pos = past_len + k_block_start + offs_n
        key_mask = key_pos < kv_len

        k_ptrs = (
            k_ptr
            + pid_b * stride_k_b
            + key_pos[None, :] * stride_k_k
            + pid_hkv * stride_k_h
            + offs_d[:, None] * stride_k_d
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_v_b
            + key_pos[:, None] * stride_v_k
            + pid_hkv * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=key_mask[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
        base_valid = q_mask[:, None] & key_mask[None, :]
        causal = (k_block_start + offs_n)[None, :] <= offs_m[:, None]
        valid = base_valid & causal

        m_i, l_i, acc = _online_update(
            q_block, k_block, v_block, valid, m_i, l_i, acc, softmax_scale
        )
        cb += 1

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(out_ptr.dtype.element_ty)
    o = tl.where(q_mask[:, None], o, 0.0)

    o_ptrs = (
        out_ptr
        + pid_b * stride_o_b
        + offs_m[:, None] * stride_o_q
        + hq[:, None] * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, o, mask=q_mask[:, None])


@triton.jit
def _past_splitk_kernel_g4(
    q_ptr,
    k_ptr,
    v_ptr,
    past_idx_ptr,
    past_counts_ptr,
    ws_m_ptr,
    ws_l_ptr,
    ws_acc_ptr,
    softmax_scale,
    past_len,
    q_len,
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
    stride_wm_b,
    stride_wm_h,
    stride_wm_q,
    stride_wm_s,
    stride_wm_m,
    stride_wl_b,
    stride_wl_h,
    stride_wl_q,
    stride_wl_s,
    stride_wl_m,
    stride_wa_b,
    stride_wa_h,
    stride_wa_q,
    stride_wa_s,
    stride_wa_m,
    stride_wa_d,
    split_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    pid_z = tl.program_id(2)
    pid_s = pid_z % split_k
    pid_m = pid_z // split_k

    offs_mh = tl.arange(0, BLOCK_M * 4)
    offs_h = offs_mh // BLOCK_M
    offs_m_local = offs_mh - offs_h * BLOCK_M
    offs_m = pid_m * BLOCK_M + offs_m_local
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < q_len
    hq = pid_hkv * 4 + offs_h

    q_ptrs = (
        q_ptr
        + pid_b * stride_q_b
        + offs_m[:, None] * stride_q_q
        + hq[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # use a finite floor to avoid inf-inf in split merge when a split is empty
    m_i = tl.full([BLOCK_M * 4], -1.0e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M * 4], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M * 4, BLOCK_D], dtype=tl.float32)

    past_count = tl.load(past_counts_ptr + pid_b * stride_pc_b + pid_hkv * stride_pc_h).to(tl.int32)
    count_per_split = (past_count + split_k - 1) // split_k
    split_start = pid_s * count_per_split
    split_end = split_start + count_per_split
    if split_end > past_count:
        split_end = past_count

    i = split_start
    while i < split_end:
        blk = tl.load(past_idx_ptr + pid_b * stride_pi_b + pid_hkv * stride_pi_h + i * stride_pi_s).to(tl.int32)
        key_pos = blk * BLOCK_N + offs_n
        key_mask = key_pos < past_len

        k_ptrs = (
            k_ptr
            + pid_b * stride_k_b
            + key_pos[None, :] * stride_k_k
            + pid_hkv * stride_k_h
            + offs_d[:, None] * stride_k_d
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_v_b
            + key_pos[:, None] * stride_v_k
            + pid_hkv * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=key_mask[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
        valid = q_mask[:, None] & key_mask[None, :]
        m_i, l_i, acc = _online_update(q_block, k_block, v_block, valid, m_i, l_i, acc, softmax_scale)
        i += 1

    wm_ptrs = (
        ws_m_ptr
        + pid_b * stride_wm_b
        + pid_hkv * stride_wm_h
        + pid_m * stride_wm_q
        + pid_s * stride_wm_s
        + offs_mh * stride_wm_m
    )
    wl_ptrs = (
        ws_l_ptr
        + pid_b * stride_wl_b
        + pid_hkv * stride_wl_h
        + pid_m * stride_wl_q
        + pid_s * stride_wl_s
        + offs_mh * stride_wl_m
    )
    wa_ptrs = (
        ws_acc_ptr
        + pid_b * stride_wa_b
        + pid_hkv * stride_wa_h
        + pid_m * stride_wa_q
        + pid_s * stride_wa_s
        + offs_mh[:, None] * stride_wa_m
        + offs_d[None, :] * stride_wa_d
    )
    tl.store(wm_ptrs, m_i, mask=q_mask)
    tl.store(wl_ptrs, l_i, mask=q_mask)
    tl.store(wa_ptrs, acc, mask=q_mask[:, None])


@triton.jit
def _reduce_splitk_kernel_g4(
    ws_m_ptr,
    ws_l_ptr,
    ws_acc_ptr,
    ws_mr_ptr,
    ws_lr_ptr,
    ws_accr_ptr,
    q_len,
    stride_wm_b,
    stride_wm_h,
    stride_wm_q,
    stride_wm_s,
    stride_wm_m,
    stride_wl_b,
    stride_wl_h,
    stride_wl_q,
    stride_wl_s,
    stride_wl_m,
    stride_wa_b,
    stride_wa_h,
    stride_wa_q,
    stride_wa_s,
    stride_wa_m,
    stride_wa_d,
    stride_wmr_b,
    stride_wmr_h,
    stride_wmr_q,
    stride_wmr_m,
    stride_wlr_b,
    stride_wlr_h,
    stride_wlr_q,
    stride_wlr_m,
    stride_war_b,
    stride_war_h,
    stride_war_q,
    stride_war_m,
    stride_war_d,
    split_k,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_mh = tl.arange(0, BLOCK_M * 4)
    offs_d = tl.arange(0, BLOCK_D)
    offs_m_local = offs_mh % BLOCK_M
    offs_m = pid_m * BLOCK_M + offs_m_local
    q_mask = offs_m < q_len

    m_i = tl.full([BLOCK_M * 4], -1.0e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M * 4], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M * 4, BLOCK_D], dtype=tl.float32)

    s = 0
    while s < split_k:
        wm_ptrs = (
            ws_m_ptr
            + pid_b * stride_wm_b
            + pid_hkv * stride_wm_h
            + pid_m * stride_wm_q
            + s * stride_wm_s
            + offs_mh * stride_wm_m
        )
        wl_ptrs = (
            ws_l_ptr
            + pid_b * stride_wl_b
            + pid_hkv * stride_wl_h
            + pid_m * stride_wl_q
            + s * stride_wl_s
            + offs_mh * stride_wl_m
        )
        wa_ptrs = (
            ws_acc_ptr
            + pid_b * stride_wa_b
            + pid_hkv * stride_wa_h
            + pid_m * stride_wa_q
            + s * stride_wa_s
            + offs_mh[:, None] * stride_wa_m
            + offs_d[None, :] * stride_wa_d
        )
        m_s = tl.load(wm_ptrs, mask=q_mask, other=-1.0e9)
        l_s = tl.load(wl_ptrs, mask=q_mask, other=0.0)
        acc_s = tl.load(wa_ptrs, mask=q_mask[:, None], other=0.0)

        m_new = tl.maximum(m_i, m_s)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_s - m_new)
        l_i = alpha * l_i + beta * l_s
        acc = acc * alpha[:, None] + acc_s * beta[:, None]
        m_i = m_new

        s += 1

    wmr_ptrs = (
        ws_mr_ptr
        + pid_b * stride_wmr_b
        + pid_hkv * stride_wmr_h
        + pid_m * stride_wmr_q
        + offs_mh * stride_wmr_m
    )
    wlr_ptrs = (
        ws_lr_ptr
        + pid_b * stride_wlr_b
        + pid_hkv * stride_wlr_h
        + pid_m * stride_wlr_q
        + offs_mh * stride_wlr_m
    )
    war_ptrs = (
        ws_accr_ptr
        + pid_b * stride_war_b
        + pid_hkv * stride_war_h
        + pid_m * stride_war_q
        + offs_mh[:, None] * stride_war_m
        + offs_d[None, :] * stride_war_d
    )
    tl.store(wmr_ptrs, m_i, mask=q_mask)
    tl.store(wlr_ptrs, l_i, mask=q_mask)
    tl.store(war_ptrs, acc, mask=q_mask[:, None])


@triton.jit
def _current_chunk_kernel_g4(
    q_ptr,
    k_ptr,
    v_ptr,
    ws_mr_ptr,
    ws_lr_ptr,
    ws_accr_ptr,
    out_ptr,
    softmax_scale,
    past_len,
    q_len,
    kv_len,
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
    stride_wmr_b,
    stride_wmr_h,
    stride_wmr_q,
    stride_wmr_m,
    stride_wlr_b,
    stride_wlr_h,
    stride_wlr_q,
    stride_wlr_m,
    stride_war_b,
    stride_war_h,
    stride_war_q,
    stride_war_m,
    stride_war_d,
    stride_o_b,
    stride_o_q,
    stride_o_h,
    stride_o_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_mh = tl.arange(0, BLOCK_M * 4)
    offs_h = offs_mh // BLOCK_M
    offs_m_local = offs_mh - offs_h * BLOCK_M
    offs_m = pid_m * BLOCK_M + offs_m_local
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < q_len
    hq = pid_hkv * 4 + offs_h

    q_ptrs = (
        q_ptr
        + pid_b * stride_q_b
        + offs_m[:, None] * stride_q_q
        + hq[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    wmr_ptrs = (
        ws_mr_ptr
        + pid_b * stride_wmr_b
        + pid_hkv * stride_wmr_h
        + pid_m * stride_wmr_q
        + offs_mh * stride_wmr_m
    )
    wlr_ptrs = (
        ws_lr_ptr
        + pid_b * stride_wlr_b
        + pid_hkv * stride_wlr_h
        + pid_m * stride_wlr_q
        + offs_mh * stride_wlr_m
    )
    war_ptrs = (
        ws_accr_ptr
        + pid_b * stride_war_b
        + pid_hkv * stride_war_h
        + pid_m * stride_war_q
        + offs_mh[:, None] * stride_war_m
        + offs_d[None, :] * stride_war_d
    )

    m_i = tl.load(wmr_ptrs, mask=q_mask, other=-1.0e9)
    l_i = tl.load(wlr_ptrs, mask=q_mask, other=0.0)
    acc = tl.load(war_ptrs, mask=q_mask[:, None], other=0.0)

    curr_blocks = q_len // BLOCK_N
    q_tile_start = pid_m * BLOCK_M
    q_tile_end = q_tile_start + (BLOCK_M - 1)
    cb_first = q_tile_start // BLOCK_N
    cb_last = q_tile_end // BLOCK_N
    if cb_last >= curr_blocks:
        cb_last = curr_blocks - 1

    cb = 0
    while cb < cb_first:
        k_block_start = cb * BLOCK_N
        key_pos = past_len + k_block_start + offs_n
        key_mask = key_pos < kv_len

        k_ptrs = (
            k_ptr
            + pid_b * stride_k_b
            + key_pos[None, :] * stride_k_k
            + pid_hkv * stride_k_h
            + offs_d[:, None] * stride_k_d
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_v_b
            + key_pos[:, None] * stride_v_k
            + pid_hkv * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=key_mask[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
        valid = q_mask[:, None] & key_mask[None, :]
        m_i, l_i, acc = _online_update(q_block, k_block, v_block, valid, m_i, l_i, acc, softmax_scale)
        cb += 1

    cb = 0
    while cb <= (cb_last - cb_first):
        cur_cb = cb_first + cb
        k_block_start = cur_cb * BLOCK_N

        key_pos = past_len + k_block_start + offs_n
        key_mask = key_pos < kv_len

        k_ptrs = (
            k_ptr
            + pid_b * stride_k_b
            + key_pos[None, :] * stride_k_k
            + pid_hkv * stride_k_h
            + offs_d[:, None] * stride_k_d
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_v_b
            + key_pos[:, None] * stride_v_k
            + pid_hkv * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=key_mask[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=key_mask[:, None], other=0.0)
        base_valid = q_mask[:, None] & key_mask[None, :]
        causal = (k_block_start + offs_n)[None, :] <= offs_m[:, None]
        valid = base_valid & causal

        m_i, l_i, acc = _online_update(q_block, k_block, v_block, valid, m_i, l_i, acc, softmax_scale)
        cb += 1

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(out_ptr.dtype.element_ty)
    o = tl.where(q_mask[:, None], o, 0.0)

    o_ptrs = (
        out_ptr
        + pid_b * stride_o_b
        + offs_m[:, None] * stride_o_q
        + hq[:, None] * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, o, mask=q_mask[:, None])


def _select_variant(
    past_len: int,
    block_size: int,
    force_variant: Optional[str] = None,
) -> Tuple[str, int, int, int]:
    if force_variant == "short":
        return "short", 32, 8, 2
    if force_variant == "long":
        return "long", 32, 8, 2

    past_blocks = int(past_len // block_size)
    if past_blocks <= 128:
        return "short", 32, 8, 2
    return "long", 32, 8, 2


def _select_split_k(
    past_blocks: int,
    variant: str,
    force_split_k: Optional[int] = None,
) -> int:
    if force_split_k is not None:
        return max(1, min(int(force_split_k), 8))
    if past_blocks <= 32:
        return 1
    if variant == "short":
        return 2
    return 4


def _estimate_v3_workspace_bytes(
    bsz: int,
    num_kv_heads: int,
    q_tiles: int,
    split_k: int,
    block_m: int,
    head_dim: int,
) -> int:
    block_mh = block_m * 4
    partial = bsz * num_kv_heads * q_tiles * split_k * block_mh
    merged = bsz * num_kv_heads * q_tiles * block_mh
    # ws_m + ws_l + ws_acc + ws_mr + ws_lr + ws_accr, all fp32
    elems = partial * 2 + partial * head_dim + merged * 2 + merged * head_dim
    return int(elems * 4)


def _launch_fa2_indexed_prefill_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_block_indices: torch.Tensor,
    past_block_counts: torch.Tensor,
    past_len: int,
    softmax_scale: float,
    block_m: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    bsz, q_len, _, head_dim = q.shape
    num_kv_heads = k.shape[2]
    out = torch.empty_like(q)

    grid = (bsz, num_kv_heads, triton.cdiv(q_len, block_m))
    _fa2_indexed_prefill_kernel_v2_g4[grid](
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
        BLOCK_M=int(block_m),
        BLOCK_N=64,
        BLOCK_D=int(head_dim),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def _launch_fa2_indexed_prefill_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_block_indices: torch.Tensor,
    past_block_counts: torch.Tensor,
    past_len: int,
    softmax_scale: float,
    block_m: int,
    num_warps: int,
    num_stages: int,
    variant: str,
    force_split_k: Optional[int],
    max_workspace_bytes: int,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    bsz, q_len, _, head_dim = q.shape
    num_kv_heads = k.shape[2]
    q_tiles = triton.cdiv(q_len, block_m)
    past_blocks = past_len // 64
    split_k = _select_split_k(past_blocks=past_blocks, variant=variant, force_split_k=force_split_k)

    workspace_bytes = _estimate_v3_workspace_bytes(
        bsz=bsz,
        num_kv_heads=num_kv_heads,
        q_tiles=q_tiles,
        split_k=split_k,
        block_m=block_m,
        head_dim=head_dim,
    )
    if workspace_bytes > int(max_workspace_bytes):
        return None, {
            "impl": "v2_fallback",
            "fallback_reason": "workspace_cap",
            "variant": variant,
            "split_k": float(split_k),
            "v3_past_ms": 0.0,
            "v3_reduce_ms": 0.0,
            "v3_current_ms": 0.0,
            "v3_total_ms": 0.0,
        }

    try:
        block_mh = block_m * 4
        ws_m = torch.empty((bsz, num_kv_heads, q_tiles, split_k, block_mh), device=q.device, dtype=torch.float32)
        ws_l = torch.empty_like(ws_m)
        ws_acc = torch.empty((bsz, num_kv_heads, q_tiles, split_k, block_mh, head_dim), device=q.device, dtype=torch.float32)
        ws_mr = torch.empty((bsz, num_kv_heads, q_tiles, block_mh), device=q.device, dtype=torch.float32)
        ws_lr = torch.empty_like(ws_mr)
        ws_accr = torch.empty((bsz, num_kv_heads, q_tiles, block_mh, head_dim), device=q.device, dtype=torch.float32)
        out = torch.empty_like(q)
    except RuntimeError:
        return None, {
            "impl": "v2_fallback",
            "fallback_reason": "workspace_oom",
            "variant": variant,
            "split_k": float(split_k),
            "v3_past_ms": 0.0,
            "v3_reduce_ms": 0.0,
            "v3_current_ms": 0.0,
            "v3_total_ms": 0.0,
        }

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e3 = torch.cuda.Event(enable_timing=True)

    e0.record()
    grid_past = (bsz, num_kv_heads, q_tiles * split_k)
    _past_splitk_kernel_g4[grid_past](
        q,
        k,
        v,
        past_block_indices,
        past_block_counts,
        ws_m,
        ws_l,
        ws_acc,
        float(softmax_scale),
        int(past_len),
        int(q_len),
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
        ws_m.stride(0),
        ws_m.stride(1),
        ws_m.stride(2),
        ws_m.stride(3),
        ws_m.stride(4),
        ws_l.stride(0),
        ws_l.stride(1),
        ws_l.stride(2),
        ws_l.stride(3),
        ws_l.stride(4),
        ws_acc.stride(0),
        ws_acc.stride(1),
        ws_acc.stride(2),
        ws_acc.stride(3),
        ws_acc.stride(4),
        ws_acc.stride(5),
        int(split_k),
        BLOCK_M=int(block_m),
        BLOCK_N=64,
        BLOCK_D=int(head_dim),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    e1.record()

    grid_reduce = (bsz, num_kv_heads, q_tiles)
    _reduce_splitk_kernel_g4[grid_reduce](
        ws_m,
        ws_l,
        ws_acc,
        ws_mr,
        ws_lr,
        ws_accr,
        int(q_len),
        ws_m.stride(0),
        ws_m.stride(1),
        ws_m.stride(2),
        ws_m.stride(3),
        ws_m.stride(4),
        ws_l.stride(0),
        ws_l.stride(1),
        ws_l.stride(2),
        ws_l.stride(3),
        ws_l.stride(4),
        ws_acc.stride(0),
        ws_acc.stride(1),
        ws_acc.stride(2),
        ws_acc.stride(3),
        ws_acc.stride(4),
        ws_acc.stride(5),
        ws_mr.stride(0),
        ws_mr.stride(1),
        ws_mr.stride(2),
        ws_mr.stride(3),
        ws_lr.stride(0),
        ws_lr.stride(1),
        ws_lr.stride(2),
        ws_lr.stride(3),
        ws_accr.stride(0),
        ws_accr.stride(1),
        ws_accr.stride(2),
        ws_accr.stride(3),
        ws_accr.stride(4),
        int(split_k),
        BLOCK_M=int(block_m),
        BLOCK_D=int(head_dim),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    e2.record()

    grid_current = (bsz, num_kv_heads, q_tiles)
    _current_chunk_kernel_g4[grid_current](
        q,
        k,
        v,
        ws_mr,
        ws_lr,
        ws_accr,
        out,
        float(softmax_scale),
        int(past_len),
        int(q_len),
        int(k.shape[1]),
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
        ws_mr.stride(0),
        ws_mr.stride(1),
        ws_mr.stride(2),
        ws_mr.stride(3),
        ws_lr.stride(0),
        ws_lr.stride(1),
        ws_lr.stride(2),
        ws_lr.stride(3),
        ws_accr.stride(0),
        ws_accr.stride(1),
        ws_accr.stride(2),
        ws_accr.stride(3),
        ws_accr.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M=int(block_m),
        BLOCK_N=64,
        BLOCK_D=int(head_dim),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    e3.record()
    e3.synchronize()

    past_ms = float(e0.elapsed_time(e1))
    reduce_ms = float(e1.elapsed_time(e2))
    current_ms = float(e2.elapsed_time(e3))

    return out, {
        "impl": "v3",
        "fallback_reason": "",
        "variant": variant,
        "split_k": float(split_k),
        "v3_past_ms": past_ms,
        "v3_reduce_ms": reduce_ms,
        "v3_current_ms": current_ms,
        "v3_total_ms": past_ms + reduce_ms + current_ms,
    }


def run_fa2_indexed_prefill_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_block_indices: torch.Tensor,
    past_block_counts: torch.Tensor,
    past_len: int,
    block_size: int,
    num_key_value_groups: int,
    softmax_scale: Optional[float] = None,
    force_variant: Optional[str] = None,
    force_split_k: Optional[int] = None,
    max_workspace_bytes: int = 2 * 1024 * 1024 * 1024,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(q.shape[-1]))
    if not can_use_fa2_indexed_prefill(
        q=q,
        k=k,
        v=v,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        block_size=block_size,
        num_key_value_groups=num_key_value_groups,
    ):
        raise ValueError("run_fa2_indexed_prefill_triton called with unsupported inputs.")

    variant, block_m, num_warps, num_stages = _select_variant(
        past_len=past_len,
        block_size=block_size,
        force_variant=force_variant,
    )

    out_v3, v3_meta = _launch_fa2_indexed_prefill_v3(
        q=q,
        k=k,
        v=v,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        softmax_scale=float(softmax_scale),
        block_m=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
        variant=variant,
        force_split_k=force_split_k,
        max_workspace_bytes=max_workspace_bytes,
    )
    if out_v3 is not None:
        v3_meta.update(
            {
                "block_m": float(block_m),
                "num_warps": float(num_warps),
                "num_stages": float(num_stages),
            }
        )
        return out_v3, v3_meta

    out_v2 = _launch_fa2_indexed_prefill_v2(
        q=q,
        k=k,
        v=v,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        softmax_scale=float(softmax_scale),
        block_m=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    meta = {
        "impl": "v2_fallback",
        "fallback_reason": str(v3_meta.get("fallback_reason", "v3_unavailable")),
        "variant": variant,
        "split_k": float(v3_meta.get("split_k", 0.0)),
        "block_m": float(block_m),
        "num_warps": float(num_warps),
        "num_stages": float(num_stages),
        "v3_past_ms": 0.0,
        "v3_reduce_ms": 0.0,
        "v3_current_ms": 0.0,
        "v3_total_ms": 0.0,
    }
    return out_v2, meta


def run_fa2_indexed_prefill_triton_with_config(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_block_indices: torch.Tensor,
    past_block_counts: torch.Tensor,
    past_len: int,
    block_size: int,
    num_key_value_groups: int,
    block_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int = 1,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(q.shape[-1]))
    if not can_use_fa2_indexed_prefill(
        q=q,
        k=k,
        v=v,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        block_size=block_size,
        num_key_value_groups=num_key_value_groups,
    ):
        raise ValueError("run_fa2_indexed_prefill_triton_with_config called with unsupported inputs.")

    out_v3, _ = _launch_fa2_indexed_prefill_v3(
        q=q,
        k=k,
        v=v,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        softmax_scale=float(softmax_scale),
        block_m=int(block_m),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        variant="short" if (past_len // block_size) <= 128 else "long",
        force_split_k=int(split_k),
        max_workspace_bytes=8 * 1024 * 1024 * 1024,
    )
    if out_v3 is not None:
        return out_v3

    return _launch_fa2_indexed_prefill_v2(
        q=q,
        k=k,
        v=v,
        past_block_indices=past_block_indices,
        past_block_counts=past_block_counts,
        past_len=past_len,
        softmax_scale=float(softmax_scale),
        block_m=int(block_m),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
