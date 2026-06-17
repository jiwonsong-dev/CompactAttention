import math

import torch

DEFAULT_CACHE_FILL_MIN_BLOCKS_FOR_TRITON = 1

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and tl is not None


if triton_available():

    @triton.jit
    def _cache_fill_blocks_kernel(
        k_blocks_ptr,
        v_blocks_ptr,
        row_idx_ptr,
        blk_idx_ptr,
        dst_base_ptr,
        k_cache_ptr,
        v_cache_ptr,
        stride_k_row,
        stride_k_blk,
        stride_k_tok,
        stride_k_d,
        stride_v_row,
        stride_v_blk,
        stride_v_tok,
        stride_v_d,
        stride_kc_tok,
        stride_kc_d,
        stride_vc_tok,
        stride_vc_d,
        num_blocks,
        block_size,
        head_dim,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_blk = tl.program_id(0)
        if pid_blk >= num_blocks:
            return

        pid_t = tl.program_id(1)
        pid_d = tl.program_id(2)

        row = tl.load(row_idx_ptr + pid_blk).to(tl.int64)
        blk = tl.load(blk_idx_ptr + pid_blk).to(tl.int64)
        dst_base = tl.load(dst_base_ptr + pid_blk).to(tl.int64)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        t_mask = offs_t < block_size
        d_mask = offs_d < head_dim
        mask = t_mask[:, None] & d_mask[None, :]

        src_k = (
            k_blocks_ptr
            + row * stride_k_row
            + blk * stride_k_blk
            + offs_t[:, None] * stride_k_tok
            + offs_d[None, :] * stride_k_d
        )
        src_v = (
            v_blocks_ptr
            + row * stride_v_row
            + blk * stride_v_blk
            + offs_t[:, None] * stride_v_tok
            + offs_d[None, :] * stride_v_d
        )
        k_vals = tl.load(src_k, mask=mask, other=0.0)
        v_vals = tl.load(src_v, mask=mask, other=0.0)

        dst_k = (
            k_cache_ptr
            + (dst_base + offs_t)[:, None] * stride_kc_tok
            + offs_d[None, :] * stride_kc_d
        )
        dst_v = (
            v_cache_ptr
            + (dst_base + offs_t)[:, None] * stride_vc_tok
            + offs_d[None, :] * stride_vc_d
        )
        tl.store(dst_k, k_vals, mask=mask)
        tl.store(dst_v, v_vals, mask=mask)

    @triton.jit
    def _cache_fill_blocks_kernel_from_kv_strided(
        k_ptr,
        v_ptr,
        row_idx_ptr,
        blk_idx_ptr,
        dst_base_ptr,
        k_cache_ptr,
        v_cache_ptr,
        num_kv_heads,
        kv_heads_shift,
        kv_heads_pow2,
        stride_k_b,
        stride_k_k,
        stride_k_h,
        stride_k_d,
        stride_v_b,
        stride_v_k,
        stride_v_h,
        stride_v_d,
        stride_kc_tok,
        stride_kc_d,
        stride_vc_tok,
        stride_vc_d,
        num_blocks,
        block_size,
        head_dim,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_blk = tl.program_id(0)
        if pid_blk >= num_blocks:
            return

        pid_t = tl.program_id(1)
        pid_d = tl.program_id(2)

        row = tl.load(row_idx_ptr + pid_blk).to(tl.int64)
        blk = tl.load(blk_idx_ptr + pid_blk).to(tl.int64)
        dst_base = tl.load(dst_base_ptr + pid_blk).to(tl.int64)
        if kv_heads_pow2 > 0:
            shift = tl.full([], kv_heads_shift, dtype=tl.int64)
            b = row >> shift
            h = row - (b << shift)
        else:
            kv_heads = tl.full([], num_kv_heads, dtype=tl.int64)
            b = row // kv_heads
            h = row - b * kv_heads

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        tl.multiple_of(offs_d, 8)
        offs_d = tl.max_contiguous(offs_d, BLOCK_D)
        tok = blk * block_size + offs_t
        t_mask = offs_t < block_size
        d_mask = offs_d < head_dim
        mask = t_mask[:, None] & d_mask[None, :]

        src_k = (
            k_ptr
            + b * stride_k_b
            + tok[:, None] * stride_k_k
            + h * stride_k_h
            + offs_d[None, :] * stride_k_d
        )
        src_v = (
            v_ptr
            + b * stride_v_b
            + tok[:, None] * stride_v_k
            + h * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_vals = tl.load(src_k, mask=mask, other=0.0)
        v_vals = tl.load(src_v, mask=mask, other=0.0)

        dst_k = (
            k_cache_ptr
            + (dst_base + offs_t)[:, None] * stride_kc_tok
            + offs_d[None, :] * stride_kc_d
        )
        dst_v = (
            v_cache_ptr
            + (dst_base + offs_t)[:, None] * stride_vc_tok
            + offs_d[None, :] * stride_vc_d
        )
        tl.store(dst_k, k_vals, mask=mask)
        tl.store(dst_v, v_vals, mask=mask)

    @triton.jit
    def _cache_fill_blocks_kernel_from_pos_rank_strided(
        k_ptr,
        v_ptr,
        pos_ptr,
        keep_prefix_rank_ptr,
        page_offsets_ptr,
        k_cache_ptr,
        v_cache_ptr,
        num_kv_heads,
        kv_heads_shift,
        kv_heads_pow2,
        page_block_size,
        stride_pos_n,
        stride_pos_c,
        stride_rank_row,
        stride_rank_blk,
        stride_k_b,
        stride_k_k,
        stride_k_h,
        stride_k_d,
        stride_v_b,
        stride_v_k,
        stride_v_h,
        stride_v_d,
        stride_kc_tok,
        stride_kc_d,
        stride_vc_tok,
        stride_vc_d,
        num_blocks,
        block_size,
        head_dim,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_blk = tl.program_id(0)
        if pid_blk >= num_blocks:
            return

        pid_t = tl.program_id(1)
        pid_d = tl.program_id(2)

        row = tl.load(pos_ptr + pid_blk * stride_pos_n + 0 * stride_pos_c).to(tl.int64)
        blk = tl.load(pos_ptr + pid_blk * stride_pos_n + 1 * stride_pos_c).to(tl.int64)
        local_rank = tl.load(
            keep_prefix_rank_ptr + row * stride_rank_row + blk * stride_rank_blk
        ).to(tl.int64)
        page_off = tl.load(page_offsets_ptr + row).to(tl.int64)
        dst_base = page_off * page_block_size + local_rank * block_size

        if kv_heads_pow2 > 0:
            shift = tl.full([], kv_heads_shift, dtype=tl.int64)
            b = row >> shift
            h = row - (b << shift)
        else:
            kv_heads = tl.full([], num_kv_heads, dtype=tl.int64)
            b = row // kv_heads
            h = row - b * kv_heads

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        tl.multiple_of(offs_d, 8)
        offs_d = tl.max_contiguous(offs_d, BLOCK_D)
        tok = blk * block_size + offs_t
        t_mask = offs_t < block_size
        d_mask = offs_d < head_dim
        mask = t_mask[:, None] & d_mask[None, :]

        src_k = (
            k_ptr
            + b * stride_k_b
            + tok[:, None] * stride_k_k
            + h * stride_k_h
            + offs_d[None, :] * stride_k_d
        )
        src_v = (
            v_ptr
            + b * stride_v_b
            + tok[:, None] * stride_v_k
            + h * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_vals = tl.load(src_k, mask=mask, other=0.0)
        v_vals = tl.load(src_v, mask=mask, other=0.0)

        dst_k = (
            k_cache_ptr
            + (dst_base + offs_t)[:, None] * stride_kc_tok
            + offs_d[None, :] * stride_kc_d
        )
        dst_v = (
            v_cache_ptr
            + (dst_base + offs_t)[:, None] * stride_vc_tok
            + offs_d[None, :] * stride_vc_d
        )
        tl.store(dst_k, k_vals, mask=mask)
        tl.store(dst_v, v_vals, mask=mask)


def _choose_tile_t(block_size: int) -> int:
    if block_size <= 16:
        return 16
    if block_size <= 32:
        return 32
    return 64


def _choose_tile_d(head_dim: int) -> int:
    if head_dim <= 32:
        return 32
    if head_dim <= 64:
        return 64
    return 128


def _choose_num_warps(block_t: int, block_d: int) -> int:
    area = block_t * block_d
    if area <= 512:
        return 2
    if area <= 2048:
        return 4
    return 8


def select_cache_fill_launch_config(
    num_sel_blocks: int,
    block_size: int,
    head_dim: int,
    dtype: torch.dtype,
) -> dict:
    # Tuned path for current Llama chunked-prefill target.
    if (
        block_size == 64
        and head_dim == 128
        and dtype in (torch.float16, torch.bfloat16)
    ):
        if num_sel_blocks < 4096:
            return {
                "block_t": 64,
                "block_d": 128,
                "num_warps": 4,
                "num_stages": 2,
                "variant_id": 1,
                "small_calls": 1.0,
                "medium_calls": 0.0,
                "large_calls": 0.0,
                "tuned_calls": 1.0,
            }
        if num_sel_blocks < 16384:
            return {
                "block_t": 64,
                "block_d": 128,
                "num_warps": 8,
                "num_stages": 2,
                "variant_id": 2,
                "small_calls": 0.0,
                "medium_calls": 1.0,
                "large_calls": 0.0,
                "tuned_calls": 1.0,
            }
        return {
            "block_t": 64,
            "block_d": 128,
            "num_warps": 8,
            "num_stages": 3,
            "variant_id": 3,
            "small_calls": 0.0,
            "medium_calls": 0.0,
            "large_calls": 1.0,
            "tuned_calls": 1.0,
        }

    # Conservative fallback heuristics.
    block_t = _choose_tile_t(block_size)
    block_d = _choose_tile_d(head_dim)
    num_warps = _choose_num_warps(block_t, block_d)
    return {
        "block_t": block_t,
        "block_d": block_d,
        "num_warps": num_warps,
        "num_stages": 2,
        "variant_id": 0,
        "small_calls": 0.0,
        "medium_calls": 0.0,
        "large_calls": 0.0,
        "tuned_calls": 0.0,
    }


def _kv_head_pow2_meta(num_kv_heads: int) -> tuple:
    if num_kv_heads > 0 and (num_kv_heads & (num_kv_heads - 1)) == 0 and num_kv_heads in (8, 16, 32):
        return int(math.log2(num_kv_heads)), 1
    return 0, 0


def can_use_triton_cache_fill(
    k_blocks: torch.Tensor,  # [rows, Kb, block_size, D]
    v_blocks: torch.Tensor,  # [rows, Kb, block_size, D]
    row_idx: torch.Tensor,  # [Nsel]
    blk_idx: torch.Tensor,  # [Nsel]
    dst_token_base: torch.Tensor,  # [Nsel]
    k_cache_flat: torch.Tensor,  # [total_pages * page_block_size, D]
    v_cache_flat: torch.Tensor,  # [total_pages * page_block_size, D]
    block_size: int,
) -> bool:
    if not triton_available():
        return False
    if block_size not in (16, 32, 64):
        return False
    if not (
        k_blocks.is_cuda
        and v_blocks.is_cuda
        and row_idx.is_cuda
        and blk_idx.is_cuda
        and dst_token_base.is_cuda
        and k_cache_flat.is_cuda
        and v_cache_flat.is_cuda
    ):
        return False
    if k_blocks.dtype not in (torch.float16, torch.bfloat16):
        return False
    if v_blocks.dtype != k_blocks.dtype or k_cache_flat.dtype != k_blocks.dtype or v_cache_flat.dtype != k_blocks.dtype:
        return False
    if k_blocks.ndim != 4 or v_blocks.ndim != 4 or k_cache_flat.ndim != 2 or v_cache_flat.ndim != 2:
        return False
    if not (
        k_blocks.is_contiguous()
        and v_blocks.is_contiguous()
        and row_idx.is_contiguous()
        and blk_idx.is_contiguous()
        and dst_token_base.is_contiguous()
        and k_cache_flat.is_contiguous()
        and v_cache_flat.is_contiguous()
    ):
        return False
    if row_idx.dtype != torch.long or blk_idx.dtype != torch.long or dst_token_base.dtype != torch.long:
        return False
    if row_idx.numel() != blk_idx.numel() or row_idx.numel() != dst_token_base.numel():
        return False
    if k_blocks.shape != v_blocks.shape:
        return False
    if int(k_blocks.shape[2]) != block_size:
        return False
    if int(k_blocks.shape[3]) != 128:
        return False
    if int(k_cache_flat.shape[1]) != 128 or int(v_cache_flat.shape[1]) != 128:
        return False
    return True


def cache_fill_blocks_to_paged_kv(
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    row_idx: torch.Tensor,
    blk_idx: torch.Tensor,
    dst_token_base: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
) -> None:
    if row_idx.numel() == 0:
        return

    block_t = _choose_tile_t(block_size)
    block_d = _choose_tile_d(int(k_blocks.shape[-1]))
    num_warps = _choose_num_warps(block_t, block_d)

    grid = (
        int(row_idx.numel()),
        triton.cdiv(block_size, block_t),
        triton.cdiv(int(k_blocks.shape[-1]), block_d),
    )
    _cache_fill_blocks_kernel[grid](
        k_blocks,
        v_blocks,
        row_idx,
        blk_idx,
        dst_token_base,
        k_cache_flat,
        v_cache_flat,
        k_blocks.stride(0),
        k_blocks.stride(1),
        k_blocks.stride(2),
        k_blocks.stride(3),
        v_blocks.stride(0),
        v_blocks.stride(1),
        v_blocks.stride(2),
        v_blocks.stride(3),
        k_cache_flat.stride(0),
        k_cache_flat.stride(1),
        v_cache_flat.stride(0),
        v_cache_flat.stride(1),
        int(row_idx.numel()),
        int(block_size),
        int(k_blocks.shape[-1]),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )


def can_use_triton_cache_fill_from_kv(
    k: torch.Tensor,  # [B, K, Hkv, D]
    v: torch.Tensor,  # [B, K, Hkv, D]
    row_idx: torch.Tensor,  # [Nsel]
    blk_idx: torch.Tensor,  # [Nsel]
    dst_token_base: torch.Tensor,  # [Nsel]
    k_cache_flat: torch.Tensor,  # [total_pages * page_block_size, D]
    v_cache_flat: torch.Tensor,  # [total_pages * page_block_size, D]
    block_size: int,
) -> bool:
    if not triton_available():
        return False
    if block_size not in (16, 32, 64):
        return False
    if not (
        k.is_cuda
        and v.is_cuda
        and row_idx.is_cuda
        and blk_idx.is_cuda
        and dst_token_base.is_cuda
        and k_cache_flat.is_cuda
        and v_cache_flat.is_cuda
    ):
        return False
    if k.dtype not in (torch.float16, torch.bfloat16):
        return False
    if v.dtype != k.dtype or k_cache_flat.dtype != k.dtype or v_cache_flat.dtype != k.dtype:
        return False
    if k.ndim != 4 or v.ndim != 4 or k_cache_flat.ndim != 2 or v_cache_flat.ndim != 2:
        return False
    if not (
        k.is_contiguous()
        and v.is_contiguous()
        and row_idx.is_contiguous()
        and blk_idx.is_contiguous()
        and dst_token_base.is_contiguous()
        and k_cache_flat.is_contiguous()
        and v_cache_flat.is_contiguous()
    ):
        return False
    if row_idx.dtype != torch.long or blk_idx.dtype != torch.long or dst_token_base.dtype != torch.long:
        return False
    if row_idx.numel() != blk_idx.numel() or row_idx.numel() != dst_token_base.numel():
        return False
    if k.shape != v.shape:
        return False
    if int(k.shape[-1]) != 128:
        return False
    if int(k_cache_flat.shape[1]) != 128 or int(v_cache_flat.shape[1]) != 128:
        return False
    return True


def cache_fill_blocks_to_paged_kv_from_kv_strided(
    k: torch.Tensor,  # [B, K, Hkv, D]
    v: torch.Tensor,  # [B, K, Hkv, D]
    row_idx: torch.Tensor,
    blk_idx: torch.Tensor,
    dst_token_base: torch.Tensor,
    k_cache_flat: torch.Tensor,
    v_cache_flat: torch.Tensor,
    block_size: int,
    launch_config: dict = None,
) -> None:
    if row_idx.numel() == 0:
        return

    config = launch_config or select_cache_fill_launch_config(
        num_sel_blocks=int(row_idx.numel()),
        block_size=int(block_size),
        head_dim=int(k.shape[-1]),
        dtype=k.dtype,
    )
    block_t = int(config["block_t"])
    block_d = int(config["block_d"])
    num_warps = int(config["num_warps"])
    num_stages = int(config["num_stages"])
    kv_heads_shift, kv_heads_pow2 = _kv_head_pow2_meta(int(k.shape[2]))

    grid = (
        int(row_idx.numel()),
        triton.cdiv(block_size, block_t),
        triton.cdiv(int(k.shape[-1]), block_d),
    )
    _cache_fill_blocks_kernel_from_kv_strided[grid](
        k,
        v,
        row_idx,
        blk_idx,
        dst_token_base,
        k_cache_flat,
        v_cache_flat,
        int(k.shape[2]),
        int(kv_heads_shift),
        int(kv_heads_pow2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        k_cache_flat.stride(0),
        k_cache_flat.stride(1),
        v_cache_flat.stride(0),
        v_cache_flat.stride(1),
        int(row_idx.numel()),
        int(block_size),
        int(k.shape[-1]),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def can_use_triton_cache_fill_from_pos_with_rank(
    k: torch.Tensor,  # [B, K, Hkv, D]
    v: torch.Tensor,  # [B, K, Hkv, D]
    pos: torch.Tensor,  # [Nsel, 2]
    keep_prefix_rank: torch.Tensor,  # [rows, Kb]
    page_offsets: torch.Tensor,  # [rows]
    k_cache_flat: torch.Tensor,  # [total_pages * page_block_size, D]
    v_cache_flat: torch.Tensor,  # [total_pages * page_block_size, D]
    block_size: int,
    page_block_size: int,
) -> bool:
    if not triton_available():
        return False
    if block_size not in (16, 32, 64):
        return False
    if page_block_size <= 0:
        return False
    if not (
        k.is_cuda
        and v.is_cuda
        and pos.is_cuda
        and keep_prefix_rank.is_cuda
        and page_offsets.is_cuda
        and k_cache_flat.is_cuda
        and v_cache_flat.is_cuda
    ):
        return False
    if k.dtype not in (torch.float16, torch.bfloat16):
        return False
    if v.dtype != k.dtype or k_cache_flat.dtype != k.dtype or v_cache_flat.dtype != k.dtype:
        return False
    if k.ndim != 4 or v.ndim != 4 or k_cache_flat.ndim != 2 or v_cache_flat.ndim != 2:
        return False
    if pos.ndim != 2 or int(pos.shape[1]) != 2:
        return False
    if keep_prefix_rank.ndim != 2 or page_offsets.ndim != 1:
        return False
    if not (
        k.is_contiguous()
        and v.is_contiguous()
        and pos.is_contiguous()
        and keep_prefix_rank.is_contiguous()
        and page_offsets.is_contiguous()
        and k_cache_flat.is_contiguous()
        and v_cache_flat.is_contiguous()
    ):
        return False
    if pos.dtype != torch.long:
        return False
    if keep_prefix_rank.dtype != torch.int32 or page_offsets.dtype != torch.int32:
        return False
    if k.shape != v.shape:
        return False
    if int(k.shape[-1]) != 128:
        return False
    if int(k_cache_flat.shape[1]) != 128 or int(v_cache_flat.shape[1]) != 128:
        return False
    return True


def cache_fill_blocks_to_paged_kv_from_pos_with_rank(
    k: torch.Tensor,  # [B, K, Hkv, D]
    v: torch.Tensor,  # [B, K, Hkv, D]
    pos: torch.Tensor,  # [Nsel, 2]
    keep_prefix_rank: torch.Tensor,  # [rows, Kb]
    page_offsets: torch.Tensor,  # [rows]
    k_cache_flat: torch.Tensor,  # [total_pages * page_block_size, D]
    v_cache_flat: torch.Tensor,  # [total_pages * page_block_size, D]
    block_size: int,
    page_block_size: int,
    launch_config: dict = None,
) -> None:
    if pos.numel() == 0:
        return

    config = launch_config or select_cache_fill_launch_config(
        num_sel_blocks=int(pos.shape[0]),
        block_size=int(block_size),
        head_dim=int(k.shape[-1]),
        dtype=k.dtype,
    )
    block_t = int(config["block_t"])
    block_d = int(config["block_d"])
    num_warps = int(config["num_warps"])
    num_stages = int(config["num_stages"])
    kv_heads_shift, kv_heads_pow2 = _kv_head_pow2_meta(int(k.shape[2]))

    grid = (
        int(pos.shape[0]),
        triton.cdiv(block_size, block_t),
        triton.cdiv(int(k.shape[-1]), block_d),
    )
    _cache_fill_blocks_kernel_from_pos_rank_strided[grid](
        k,
        v,
        pos,
        keep_prefix_rank,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        int(k.shape[2]),
        int(kv_heads_shift),
        int(kv_heads_pow2),
        int(page_block_size),
        pos.stride(0),
        pos.stride(1),
        keep_prefix_rank.stride(0),
        keep_prefix_rank.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        k_cache_flat.stride(0),
        k_cache_flat.stride(1),
        v_cache_flat.stride(0),
        v_cache_flat.stride(1),
        int(pos.shape[0]),
        int(block_size),
        int(k.shape[-1]),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
