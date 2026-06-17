import torch
import triton
import triton.language as tl

from compact_attn.flashprefill_vendor.utils import autocast_custom_fwd, contiguous


def _score_configs():
    configs = []
    k_tiles = [64, 128, 256]
    warps = [4, 8]
    stages = [2, 3, 4, 5]
    for k_tile in k_tiles:
        for num_warps in warps:
            for num_stages in stages:
                if k_tile == 256 and num_warps == 4:
                    continue
                configs.append(
                    triton.Config({"K_TILE_SIZE": k_tile}, num_warps=num_warps, num_stages=num_stages)
                )
    return configs


def _attention_configs():
    configs = []
    tile_sizes = [(64, 64), (128, 64), (64, 128)]
    warps = [4, 8]
    stages = [2, 3, 4, 5]
    for q_tile, k_tile in tile_sizes:
        for num_warps in warps:
            for num_stages in stages:
                if q_tile == 128 and num_warps == 4:
                    continue
                configs.append(
                    triton.Config(
                        {"Q_TILE_SIZE": q_tile, "K_TILE_SIZE": k_tile},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=[triton.Config({}, num_warps=w, num_stages=s) for w in [4, 8] for s in [2, 3, 4, 5]],
    key=["seq_len", "num_q_heads", "BLOCK_SIZE"],
)
@triton.jit
def compute_mean_vector(
    q_ptr,
    mq_ptr,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_mqz,
    stride_mqm,
    stride_mqh,
    stride_mqd,
    num_q_heads,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    query_tile_index = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)

    offset_batch = offset_zh // num_q_heads
    offset_q_heads = offset_zh % num_q_heads

    offset_q_ptr = offset_batch * stride_qz + offset_q_heads * stride_qh
    offset_mq_ptr = offset_batch * stride_mqz + offset_q_heads * stride_mqh

    q_base_ptr = q_ptr + offset_q_ptr
    mq_base_ptr = mq_ptr + offset_mq_ptr

    offset_q = query_tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_dim = tl.arange(0, D_HEAD)

    q = tl.load(
        q_base_ptr + offset_q[:, None] * stride_qm + offset_dim[None, :] * stride_qd,
        mask=(offset_q[:, None] < seq_len) & (offset_dim[None, :] < D_HEAD),
        other=0.0,
    )

    q_sum = tl.sum(q, axis=0)
    q_mask = offset_q < seq_len
    num_ele = tl.sum(q_mask.to(tl.int32)).to(q.dtype)
    q_mean = tl.where(num_ele > 0, q_sum / num_ele, 0.0)

    tl.store(
        mq_base_ptr + query_tile_index * stride_mqm + offset_dim * stride_mqd,
        q_mean,
        mask=offset_dim < D_HEAD,
    )


@triton.autotune(
    configs=_score_configs(),
    key=["query_len", "sub_key_len", "num_q_heads", "num_k_heads"],
)
@triton.jit
def compute_block_score_chunked(
    q_ptr,
    k_ptr,
    scale,
    sc_ptr,
    mx_ptr,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_scz,
    stride_scmb,
    stride_scnb,
    stride_sch,
    stride_mxz,
    stride_mxmb,
    stride_mxnb,
    stride_mxh,
    num_q_heads,
    num_k_heads,
    query_len,
    sub_key_len,
    query_start_token,
    BLOCK_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    num_stride_per_block: tl.constexpr = BLOCK_SIZE // K_STRIDE
    num_block_per_tile: tl.constexpr = K_TILE_SIZE // num_stride_per_block

    query_tile_index = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)

    offset_batch = offset_zh // num_q_heads
    group_size = num_q_heads // num_k_heads
    offset_q_heads = offset_zh % num_q_heads
    offset_k_heads = offset_q_heads // group_size

    offset_q_ptr = offset_batch * stride_qz + offset_q_heads * stride_qh
    offset_k_ptr = offset_batch * stride_kz + offset_k_heads * stride_kh
    q_base_ptr = q_ptr + offset_q_ptr
    k_base_ptr = k_ptr + offset_k_ptr

    offset_sc_ptr = offset_batch * stride_scz + offset_q_heads * stride_sch
    offset_mx_ptr = offset_batch * stride_mxz + offset_q_heads * stride_mxh
    sc_base_ptr = sc_ptr + offset_sc_ptr
    mx_base_ptr = mx_ptr + offset_mx_ptr

    offset_q = query_tile_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_dim = tl.arange(0, D_HEAD)

    q_index_abs = query_start_token + offset_q
    q_index_max_abs = query_start_token + query_tile_index * BLOCK_SIZE + BLOCK_SIZE - 1

    q = tl.load(
        q_base_ptr + offset_q[:, None] * stride_qm + offset_dim[None, :] * stride_qd,
        mask=(offset_q[:, None] < query_len) & (offset_dim[None, :] < D_HEAD),
        other=0.0,
    )

    lo = 0
    hi = tl.minimum(tl.cdiv(q_index_max_abs + 1, K_STRIDE), sub_key_len)
    sm_scale = scale * 1.4426950408889634

    for j in range(lo, hi, K_TILE_SIZE):
        offset_k = j + tl.arange(0, K_TILE_SIZE)
        k_index_max = offset_k * K_STRIDE + K_STRIDE - 1
        k = tl.load(
            k_base_ptr + offset_k[:, None] * stride_kn + offset_dim[None, :] * stride_kd,
            mask=(offset_k[:, None] < sub_key_len) & (offset_dim[None, :] < D_HEAD),
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k))
        causal_mask = q_index_abs[:, None] >= k_index_max[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        qk *= sm_scale

        qk = tl.reshape(qk, (BLOCK_SIZE, num_block_per_tile, num_stride_per_block))
        m_i_block = tl.max(qk, axis=2)
        m_i_block = tl.max(m_i_block, axis=0)

        qk_block = qk - m_i_block[None, :, None]
        p_block = tl.exp2(qk_block)
        causal_mask_block = tl.reshape(causal_mask, (BLOCK_SIZE, num_block_per_tile, num_stride_per_block))
        p_block = tl.where(causal_mask_block, p_block, 0.0)
        p_block = tl.sum(p_block, 2)
        p_block = tl.sum(p_block, 0)

        offset_k_blocks = (j // K_TILE_SIZE) * num_block_per_tile + tl.arange(0, num_block_per_tile)
        tl.store(
            sc_base_ptr + query_tile_index * stride_scmb + offset_k_blocks * stride_scnb,
            p_block,
            mask=(query_tile_index < tl.cdiv(query_len, BLOCK_SIZE)) & (offset_k_blocks < sub_key_len),
        )
        tl.store(
            mx_base_ptr + query_tile_index * stride_mxmb + offset_k_blocks * stride_mxnb,
            m_i_block,
            mask=(query_tile_index < tl.cdiv(query_len, BLOCK_SIZE)) & (offset_k_blocks < sub_key_len),
        )


@triton.autotune(
    configs=_attention_configs(),
    key=["query_len", "key_len", "num_q_heads", "num_k_heads", "BLOCK_SIZE"],
    prune_configs_by={
        "early_config_prune": lambda configs, named_args, **kwargs: [
            c
            for c in configs
            if c.kwargs["Q_TILE_SIZE"] <= kwargs["BLOCK_SIZE"] and c.kwargs["K_TILE_SIZE"] <= kwargs["BLOCK_SIZE"]
        ]
    },
)
@triton.jit
def _flash_forward_chunked(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    index_ptr,
    valid_ptr,
    scale,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vz,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_oz,
    stride_om,
    stride_oh,
    stride_od,
    stride_indexz,
    stride_indexm,
    stride_indexn,
    stride_indexh,
    stride_validz,
    stride_validm,
    stride_validh,
    query_len,
    key_len,
    query_start_token,
    query_block_offset,
    num_q_heads,
    num_k_heads,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    query_tile_index = tl.program_id(0).to(tl.int64)
    offset_zh = tl.program_id(1).to(tl.int64)

    offset_batch = offset_zh // num_q_heads
    group_size = num_q_heads // num_k_heads
    index_group_size = BLOCK_SIZE // Q_TILE_SIZE
    offset_q_heads = offset_zh % num_q_heads
    offset_k_heads = offset_q_heads // group_size

    offset_q_ptr = offset_batch * stride_qz + offset_q_heads * stride_qh
    offset_k_ptr = offset_batch * stride_kz + offset_k_heads * stride_kh
    offset_v_ptr = offset_batch * stride_vz + offset_k_heads * stride_vh
    offset_o_ptr = offset_batch * stride_oz + offset_q_heads * stride_oh
    offset_index_ptr = (
        offset_batch * stride_indexz
        + offset_q_heads * stride_indexh
        + (query_tile_index // index_group_size) * stride_indexm
    )
    offset_valid_ptr = (
        offset_batch * stride_validz
        + offset_q_heads * stride_validh
        + (query_tile_index // index_group_size) * stride_validm
    )

    q_base_ptr = q_ptr + offset_q_ptr
    k_base_ptr = k_ptr + offset_k_ptr
    v_base_ptr = v_ptr + offset_v_ptr
    o_base_ptr = o_ptr + offset_o_ptr
    index_base_ptr = index_ptr + offset_index_ptr
    valid_base_ptr = valid_ptr + offset_valid_ptr

    offset_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    offset_dim = tl.arange(0, D_HEAD)
    q_abs = query_start_token + offset_q

    q = tl.load(
        q_base_ptr + offset_q[:, None] * stride_qm + offset_dim[None, :] * stride_qd,
        mask=(offset_q[:, None] < query_len) & (offset_dim[None, :] < D_HEAD),
        other=0.0,
    )

    sm_scale = scale * 1.4426950408889634
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((Q_TILE_SIZE,), 1.0, dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D_HEAD), dtype=tl.float32)

    lo = 0
    hi = tl.load(valid_base_ptr)
    block_index = tl.load(index_base_ptr + lo * stride_indexn, mask=lo < hi, other=0)
    query_block_index = query_block_offset + (query_tile_index // index_group_size)

    for i in range(lo, hi):
        next_block_index = tl.load(index_base_ptr + (i + 1) * stride_indexn, mask=(i + 1) < hi, other=0)
        key_start_index = block_index * BLOCK_SIZE
        key_end_index = (block_index + 1) * BLOCK_SIZE
        is_diagonal_block = block_index == query_block_index

        for j in range(key_start_index, key_end_index, K_TILE_SIZE):
            k_index = j + tl.arange(0, K_TILE_SIZE)
            k_j = tl.load(
                k_base_ptr + k_index[:, None] * stride_kn + offset_dim[None, :] * stride_kd,
                mask=(k_index[:, None] < key_len) & (offset_dim[None, :] < D_HEAD),
                other=0.0,
            )
            v_j = tl.load(
                v_base_ptr + k_index[:, None] * stride_vn + offset_dim[None, :] * stride_vd,
                mask=(k_index[:, None] < key_len) & (offset_dim[None, :] < D_HEAD),
                other=0.0,
            )

            qk = tl.dot(q, tl.trans(k_j))
            if is_diagonal_block:
                causal_mask = q_abs[:, None] >= k_index[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))
            qk *= sm_scale
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            qk -= m_i_new[:, None]
            p = tl.exp2(qk)
            lij = tl.sum(p, 1)
            alpha = tl.exp2(m_i - m_i_new)
            alpha_mask = alpha != alpha
            alpha = tl.where(alpha_mask, 1.0, alpha)
            acc *= alpha[:, None]
            l_i = l_i * alpha + lij
            p = p.to(v_j.dtype)
            acc += tl.dot(p, v_j)
            m_i = m_i_new
        block_index = next_block_index

    l_rec = 1 / l_i[:, None]
    acc = acc * l_rec

    o_offset_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    o_offset_d = tl.arange(0, D_HEAD)
    tl.store(
        o_base_ptr + o_offset_m[:, None] * stride_om + o_offset_d[None, :] * stride_od,
        acc.to(q.dtype),
        mask=(o_offset_m[:, None] < query_len) & (o_offset_d[None, :] < D_HEAD),
    )


def normalize_scores(output_score: torch.Tensor, output_max: torch.Tensor):
    mask = output_max != float("-inf")
    max_in_max = torch.max(output_max, dim=2, keepdim=True).values
    output_max = output_max - max_in_max
    output_max = torch.exp2(output_max)
    output_max = torch.where(mask, output_max, 1.0)
    output_score = torch.where(mask, output_score, 0.0)
    output_score = output_score * output_max
    output_score = output_score / (output_score.sum(dim=2, keepdim=True) + 1e-9)
    return output_score


def deal_output_score_chunked(
    score: torch.Tensor,
    attention_sink: int,
    window: int,
    alpha: float = 0.1,
    last_n_blocks_full: int = 2,
    min_budget: int = 0,
    q_block_offset: int = 0,
):
    bsz, num_query_blocks, num_key_blocks, num_heads = score.shape
    device = score.device
    k_ids = torch.arange(num_key_blocks, device=device).view(1, 1, num_key_blocks, 1)
    q_ids = torch.arange(
        q_block_offset,
        q_block_offset + num_query_blocks,
        device=device,
    ).view(1, num_query_blocks, 1, 1)

    if min_budget > 0:
        topk_vals, topk_indices = torch.topk(score, k=min_budget, dim=2)
        max_val = topk_vals[:, :, :1, :]
        mask_score = score >= (max_val * alpha)
        mask_score.scatter_(2, topk_indices, True)
    else:
        max_val = score.max(dim=2, keepdim=True).values
        mask_score = score >= (max_val * alpha)

    mask_sink = k_ids < attention_sink
    dist = q_ids - k_ids
    mask_window = (dist >= 0) & (dist < window)
    mask_last_full = q_ids >= (num_key_blocks - last_n_blocks_full)
    mask_causal = dist >= 0

    is_active = (mask_score | mask_sink | mask_window | mask_last_full) & mask_causal
    counts = is_active.sum(dim=2).int()
    indices = k_ids.expand(bsz, num_query_blocks, num_key_blocks, num_heads)
    indices_to_sort = indices.masked_fill(~is_active, num_key_blocks)
    compact_indices, _ = indices_to_sort.sort(dim=2)
    return compact_indices, counts


@torch.no_grad()
def flash_prefill_select_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block_full: int,
    min_budget: int,
):
    batch_size, query_len, num_q_heads, head_dim = q.shape
    key_len = k.shape[1]
    num_k_heads = k.shape[-2]
    query_start_token = key_len - query_len
    query_block_offset = triton.cdiv(query_start_token, block_size)

    mean_k = torch.empty(
        batch_size,
        triton.cdiv(key_len, block_size),
        num_k_heads,
        head_dim,
        dtype=k.dtype,
        device=k.device,
    )

    def grid_mean(meta):
        return (triton.cdiv(key_len, block_size), batch_size * num_k_heads, 1)

    compute_mean_vector[grid_mean](
        k,
        mean_k,
        *k.stride(),
        *mean_k.stride(),
        num_k_heads,
        key_len,
        block_size,
        head_dim,
    )

    num_query_blocks = triton.cdiv(query_len, block_size)
    num_key_blocks = triton.cdiv(key_len, block_size)
    output_score = torch.full(
        (batch_size, num_query_blocks, num_key_blocks, num_q_heads),
        fill_value=float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    output_max = torch.full_like(output_score, fill_value=float("-inf"))

    def grid_score(meta):
        return (num_query_blocks, batch_size * num_q_heads, 1)

    compute_block_score_chunked[grid_score](
        q,
        mean_k,
        1 / (head_dim ** 0.5),
        output_score,
        output_max,
        *q.stride(),
        *mean_k.stride(),
        *output_score.stride(),
        *output_max.stride(),
        num_q_heads,
        num_k_heads,
        query_len,
        num_key_blocks,
        query_start_token,
        block_size,
        K_STRIDE=block_size,
        D_HEAD=head_dim,
    )

    output_score = normalize_scores(output_score, output_max)
    block_index, counts = deal_output_score_chunked(
        output_score,
        attention_sink=attention_sink,
        window=window_size,
        alpha=alpha,
        last_n_blocks_full=last_n_block_full,
        min_budget=min_budget,
        q_block_offset=query_block_offset,
    )
    return output_score, block_index, counts


@torch.no_grad()
def flash_prefill_compute_mean_k(
    k: torch.Tensor,
    *,
    block_size: int,
):
    key_len = k.shape[1]
    num_k_heads = k.shape[-2]
    head_dim = k.shape[-1]
    batch_size = k.shape[0]

    mean_k = torch.empty(
        batch_size,
        triton.cdiv(key_len, block_size),
        num_k_heads,
        head_dim,
        dtype=k.dtype,
        device=k.device,
    )

    def grid_mean(meta):
        return (triton.cdiv(key_len, block_size), batch_size * num_k_heads, 1)

    compute_mean_vector[grid_mean](
        k,
        mean_k,
        *k.stride(),
        *mean_k.stride(),
        num_k_heads,
        key_len,
        block_size,
        head_dim,
    )
    return mean_k


@torch.no_grad()
def flash_prefill_score_chunked_from_mean_k(
    q: torch.Tensor,
    mean_k: torch.Tensor,
    *,
    key_len: int,
    block_size: int,
):
    batch_size, query_len, num_q_heads, head_dim = q.shape
    num_k_heads = mean_k.shape[-2]
    query_start_token = key_len - query_len

    num_query_blocks = triton.cdiv(query_len, block_size)
    num_key_blocks = mean_k.shape[1]
    output_score = torch.full(
        (batch_size, num_query_blocks, num_key_blocks, num_q_heads),
        fill_value=float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    output_max = torch.full_like(output_score, fill_value=float("-inf"))

    def grid_score(meta):
        return (num_query_blocks, batch_size * num_q_heads, 1)

    compute_block_score_chunked[grid_score](
        q,
        mean_k,
        1 / (head_dim ** 0.5),
        output_score,
        output_max,
        *q.stride(),
        *mean_k.stride(),
        *output_score.stride(),
        *output_max.stride(),
        num_q_heads,
        num_k_heads,
        query_len,
        num_key_blocks,
        query_start_token,
        block_size,
        K_STRIDE=block_size,
        D_HEAD=head_dim,
    )

    return normalize_scores(output_score, output_max)


@torch.no_grad()
def flash_prefill_score_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
):
    mean_k = flash_prefill_compute_mean_k(k, block_size=block_size)
    return flash_prefill_score_chunked_from_mean_k(
        q,
        mean_k,
        key_len=k.shape[1],
        block_size=block_size,
    )


@torch.no_grad()
def flash_prefill_select_chunked_from_mean_k(
    q: torch.Tensor,
    mean_k: torch.Tensor,
    *,
    key_len: int,
    block_size: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block_full: int,
    min_budget: int,
):
    output_score = flash_prefill_score_chunked_from_mean_k(
        q,
        mean_k,
        key_len=key_len,
        block_size=block_size,
    )
    query_start_token = key_len - q.shape[1]
    query_block_offset = triton.cdiv(query_start_token, block_size)
    block_index, counts = deal_output_score_chunked(
        output_score,
        attention_sink=attention_sink,
        window=window_size,
        alpha=alpha,
        last_n_blocks_full=last_n_block_full,
        min_budget=min_budget,
        q_block_offset=query_block_offset,
    )
    return output_score, block_index, counts


class FlashPrefillChunked(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        out,
        block_size=128,
        attention_sink=2,
        window=4,
        alpha=0.05,
        last_n_blocks_full=2,
        min_budget=0,
    ):
        batch_size, query_len, num_q_heads, head_dim = q.shape
        key_len = k.shape[1]
        num_k_heads = k.shape[-2]
        query_start_token = key_len - query_len
        query_block_offset = triton.cdiv(query_start_token, block_size)

        mean_k = torch.empty(
            batch_size,
            triton.cdiv(key_len, block_size),
            num_k_heads,
            head_dim,
            dtype=k.dtype,
            device=k.device,
        )

        def grid_mean(meta):
            return (triton.cdiv(key_len, block_size), batch_size * num_k_heads, 1)

        compute_mean_vector[grid_mean](
            k,
            mean_k,
            *k.stride(),
            *mean_k.stride(),
            num_k_heads,
            key_len,
            block_size,
            head_dim,
        )

        num_query_blocks = triton.cdiv(query_len, block_size)
        num_key_blocks = triton.cdiv(key_len, block_size)
        output_score = torch.full(
            (batch_size, num_query_blocks, num_key_blocks, num_q_heads),
            fill_value=float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        output_max = torch.full_like(output_score, fill_value=float("-inf"))

        def grid_score(meta):
            return (num_query_blocks, batch_size * num_q_heads, 1)

        compute_block_score_chunked[grid_score](
            q,
            mean_k,
            1 / (head_dim ** 0.5),
            output_score,
            output_max,
            *q.stride(),
            *mean_k.stride(),
            *output_score.stride(),
            *output_max.stride(),
            num_q_heads,
            num_k_heads,
            query_len,
            num_key_blocks,
            query_start_token,
            block_size,
            K_STRIDE=block_size,
            D_HEAD=head_dim,
        )

        output_score = normalize_scores(output_score, output_max)
        block_index, counts = deal_output_score_chunked(
            output_score,
            attention_sink=attention_sink,
            window=window,
            alpha=alpha,
            last_n_blocks_full=last_n_blocks_full,
            min_budget=min_budget,
            q_block_offset=query_block_offset,
        )

        def grid(meta):
            return (triton.cdiv(query_len, meta["Q_TILE_SIZE"]), batch_size * num_q_heads, 1)

        _flash_forward_chunked[grid](
            q,
            k,
            v,
            out,
            block_index,
            counts,
            1 / (head_dim ** 0.5),
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            *block_index.stride(),
            *counts.stride(),
            query_len,
            key_len,
            query_start_token,
            query_block_offset,
            num_q_heads,
            num_k_heads,
            BLOCK_SIZE=block_size,
            D_HEAD=head_dim,
        )


@torch.no_grad()
def flash_prefill_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    block_size: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block_full: int,
    min_budget: int,
):
    FlashPrefillChunked.apply(
        q,
        k,
        v,
        out,
        block_size,
        attention_sink,
        window_size,
        alpha,
        last_n_block_full,
        min_budget,
    )


@torch.no_grad()
def flash_prefill_kernel_only_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    block_index: torch.Tensor,
    counts: torch.Tensor,
    block_size: int,
):
    batch_size, query_len, num_q_heads, head_dim = q.shape
    key_len = k.shape[1]
    num_k_heads = k.shape[-2]
    query_start_token = key_len - query_len
    query_block_offset = triton.cdiv(query_start_token, block_size)

    def grid(meta):
        return (triton.cdiv(query_len, meta["Q_TILE_SIZE"]), batch_size * num_q_heads, 1)

    _flash_forward_chunked[grid](
        q,
        k,
        v,
        out,
        block_index,
        counts,
        1 / (head_dim ** 0.5),
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        *block_index.stride(),
        *counts.stride(),
        query_len,
        key_len,
        query_start_token,
        query_block_offset,
        num_q_heads,
        num_k_heads,
        BLOCK_SIZE=block_size,
        D_HEAD=head_dim,
    )


@torch.no_grad()
def flash_prefill_chunked_from_mean_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    mean_k: torch.Tensor,
    block_size: int,
    attention_sink: int,
    window_size: int,
    alpha: float,
    last_n_block_full: int,
    min_budget: int,
):
    _, block_index, counts = flash_prefill_select_chunked_from_mean_k(
        q,
        mean_k,
        key_len=k.shape[1],
        block_size=block_size,
        attention_sink=attention_sink,
        window_size=window_size,
        alpha=alpha,
        last_n_block_full=last_n_block_full,
        min_budget=min_budget,
    )
    flash_prefill_kernel_only_chunked(q, k, v, out, block_index, counts, block_size)
