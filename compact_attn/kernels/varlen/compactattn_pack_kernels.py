import torch

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
    def _gather_3d_to_cat_kernel(
        src_ptr,
        row_idx_ptr,
        tok_idx_ptr,
        dst_ptr,
        stride_src_row,
        stride_src_tok,
        stride_src_d,
        stride_dst_n,
        stride_dst_t,
        stride_dst_d,
        num_items,
        head_dim,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_items:
            return

        row = tl.load(row_idx_ptr + pid)
        tok = tl.load(tok_idx_ptr + pid)
        offs_d = tl.arange(0, BLOCK_D)
        dmask = offs_d < head_dim

        src = src_ptr + row * stride_src_row + tok * stride_src_tok + offs_d * stride_src_d
        vals = tl.load(src, mask=dmask, other=0.0)

        dst = dst_ptr + pid * stride_dst_n + offs_d * stride_dst_d
        tl.store(dst, vals, mask=dmask)


    @triton.jit
    def _scatter_cat_to_3d_kernel(
        src_ptr,
        row_idx_ptr,
        tok_idx_ptr,
        dst_ptr,
        stride_src_n,
        stride_src_t,
        stride_src_d,
        stride_dst_row,
        stride_dst_tok,
        stride_dst_d,
        num_items,
        head_dim,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_items:
            return

        row = tl.load(row_idx_ptr + pid)
        tok = tl.load(tok_idx_ptr + pid)
        offs_d = tl.arange(0, BLOCK_D)
        dmask = offs_d < head_dim

        src = src_ptr + pid * stride_src_n + offs_d * stride_src_d
        vals = tl.load(src, mask=dmask, other=0.0)

        dst = dst_ptr + row * stride_dst_row + tok * stride_dst_tok + offs_d * stride_dst_d
        tl.store(dst, vals, mask=dmask)


def _choose_block_d(head_dim: int) -> int:
    if head_dim <= 16:
        return 16
    if head_dim <= 32:
        return 32
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    if head_dim <= 256:
        return 256
    return 512


def _num_warps_for_block(block_d: int) -> int:
    if block_d <= 64:
        return 2
    if block_d <= 128:
        return 4
    return 8


def _is_supported_tensor_layout(src: torch.Tensor, row_idx: torch.Tensor, tok_idx: torch.Tensor) -> bool:
    if not triton_available():
        return False
    if (not src.is_cuda) or (not row_idx.is_cuda) or (not tok_idx.is_cuda):
        return False
    if src.dtype not in (torch.float16, torch.bfloat16):
        return False
    if row_idx.dtype != torch.long or tok_idx.dtype != torch.long:
        return False
    if src.ndim != 3:
        return False
    # Keep v1 conservative: contiguous source + contiguous indices.
    if (not src.is_contiguous()) or (not row_idx.is_contiguous()) or (not tok_idx.is_contiguous()):
        return False
    return True


def can_use_triton_pack(src: torch.Tensor, row_idx: torch.Tensor, tok_idx: torch.Tensor) -> bool:
    if not _is_supported_tensor_layout(src, row_idx, tok_idx):
        return False
    head_dim = int(src.shape[-1])
    return head_dim <= 512


def can_use_triton_scatter(
    src: torch.Tensor, row_idx: torch.Tensor, tok_idx: torch.Tensor, dst: torch.Tensor
) -> bool:
    if not can_use_triton_pack(src, row_idx, tok_idx):
        return False
    if (not dst.is_cuda) or (not dst.is_contiguous()):
        return False
    if dst.dtype != src.dtype or dst.ndim != 3:
        return False
    if int(dst.shape[-1]) != int(src.shape[-1]):
        return False
    return True


def gather_3d_to_cat(
    src: torch.Tensor,
    row_idx: torch.Tensor,
    tok_idx: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    if row_idx.numel() == 0:
        return
    block_d = _choose_block_d(int(src.shape[-1]))
    num_warps = _num_warps_for_block(block_d)
    grid = (int(row_idx.numel()),)
    _gather_3d_to_cat_kernel[grid](
        src,
        row_idx,
        tok_idx,
        dst,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        int(row_idx.numel()),
        int(src.shape[-1]),
        BLOCK_D=block_d,
        num_warps=num_warps,
    )


def scatter_cat_to_3d(
    src: torch.Tensor,
    row_idx: torch.Tensor,
    tok_idx: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    if row_idx.numel() == 0:
        return
    block_d = _choose_block_d(int(dst.shape[-1]))
    num_warps = _num_warps_for_block(block_d)
    grid = (int(row_idx.numel()),)
    _scatter_cat_to_3d_kernel[grid](
        src,
        row_idx,
        tok_idx,
        dst,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        int(row_idx.numel()),
        int(dst.shape[-1]),
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
