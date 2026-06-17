#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace {

template <typename scalar_t>
__global__ void cache_fill_from_pos_rank_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ pos,
    const int32_t* __restrict__ keep_prefix_rank,
    const int32_t* __restrict__ page_offsets,
    scalar_t* __restrict__ k_cache_flat,
    scalar_t* __restrict__ v_cache_flat,
    int64_t kv_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rank_stride,
    int64_t block_size,
    int64_t page_block_size,
    int64_t num_blocks) {
    int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_blocks) {
        return;
    }

    int64_t row = pos[block_idx * 2];
    int64_t blk = pos[block_idx * 2 + 1];
    int64_t local_rank = static_cast<int64_t>(keep_prefix_rank[row * rank_stride + blk]);
    int64_t dst_base = static_cast<int64_t>(page_offsets[row]) * page_block_size + local_rank * block_size;
    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;

    int64_t total = block_size * head_dim;
    for (int64_t linear = threadIdx.x; linear < total; linear += blockDim.x) {
        int64_t tok_off = linear / head_dim;
        int64_t d = linear - tok_off * head_dim;
        int64_t tok = blk * block_size + tok_off;
        int64_t src_idx = (((batch_idx * kv_len + tok) * num_kv_heads + kv_head_idx) * head_dim) + d;
        int64_t dst_idx = ((dst_base + tok_off) * head_dim) + d;
        k_cache_flat[dst_idx] = k[src_idx];
        v_cache_flat[dst_idx] = v[src_idx];
    }
}

template <typename scalar_t>
__global__ void cache_fill_from_row_blk_dst_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ row_idx,
    const int64_t* __restrict__ blk_idx,
    const int64_t* __restrict__ dst_token_base,
    scalar_t* __restrict__ k_cache_flat,
    scalar_t* __restrict__ v_cache_flat,
    int64_t kv_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t num_blocks) {
    int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_blocks) {
        return;
    }

    int64_t row = row_idx[block_idx];
    int64_t blk = blk_idx[block_idx];
    int64_t dst_base = dst_token_base[block_idx];
    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;

    int64_t total = block_size * head_dim;
    for (int64_t linear = threadIdx.x; linear < total; linear += blockDim.x) {
        int64_t tok_off = linear / head_dim;
        int64_t d = linear - tok_off * head_dim;
        int64_t tok = blk * block_size + tok_off;
        int64_t src_idx = (((batch_idx * kv_len + tok) * num_kv_heads + kv_head_idx) * head_dim) + d;
        int64_t dst_idx = ((dst_base + tok_off) * head_dim) + d;
        k_cache_flat[dst_idx] = k[src_idx];
        v_cache_flat[dst_idx] = v[src_idx];
    }
}

template <typename scalar_t>
__global__ void cache_fill_from_pos_local_rank_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ pos,
    const int32_t* __restrict__ local_rank,
    const int32_t* __restrict__ page_offsets,
    scalar_t* __restrict__ k_cache_flat,
    scalar_t* __restrict__ v_cache_flat,
    int64_t kv_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t page_block_size,
    int64_t num_blocks) {
    int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_blocks) {
        return;
    }

    int64_t row = pos[block_idx * 2];
    int64_t blk = pos[block_idx * 2 + 1];
    int64_t dst_base =
        static_cast<int64_t>(page_offsets[row]) * page_block_size +
        static_cast<int64_t>(local_rank[block_idx]) * block_size;
    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;

    int64_t total = block_size * head_dim;
    for (int64_t linear = threadIdx.x; linear < total; linear += blockDim.x) {
        int64_t tok_off = linear / head_dim;
        int64_t d = linear - tok_off * head_dim;
        int64_t tok = blk * block_size + tok_off;
        int64_t src_idx = (((batch_idx * kv_len + tok) * num_kv_heads + kv_head_idx) * head_dim) + d;
        int64_t dst_idx = ((dst_base + tok_off) * head_dim) + d;
        k_cache_flat[dst_idx] = k[src_idx];
        v_cache_flat[dst_idx] = v[src_idx];
    }
}

template <typename scalar_t>
__global__ void pack_q_for_indexed_prefill_kernel(
    const scalar_t* __restrict__ q,
    scalar_t* __restrict__ q_group,
    int64_t q_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_key_value_groups,
    int64_t head_dim,
    int64_t total_rows) {
    int64_t row_token = static_cast<int64_t>(blockIdx.x);
    if (row_token >= total_rows * q_len) {
        return;
    }

    int64_t row = row_token / q_len;
    int64_t tok = row_token - row * q_len;
    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;
    int64_t head_base = kv_head_idx * num_key_value_groups;
    int64_t vec_count =
        (num_key_value_groups * head_dim * static_cast<int64_t>(sizeof(scalar_t))) / static_cast<int64_t>(sizeof(uint4));

    const uint4* __restrict__ src_vec = reinterpret_cast<const uint4*>(
        q + (((batch_idx * q_len + tok) * num_q_heads + head_base) * head_dim));
    uint4* __restrict__ dst_vec = reinterpret_cast<uint4*>(
        q_group + (((row * q_len + tok) * num_key_value_groups) * head_dim));

    for (int64_t vec_idx = threadIdx.x; vec_idx < vec_count; vec_idx += blockDim.x) {
        dst_vec[vec_idx] = src_vec[vec_idx];
    }
}

__global__ void build_block_table_kernel(
    const int32_t* __restrict__ pages_per_row,
    int32_t* __restrict__ block_table,
    int32_t* __restrict__ page_offsets,
    int64_t rows,
    int64_t max_pages) {
    extern __shared__ int32_t shared_offsets[];

    if (threadIdx.x == 0) {
        int32_t offset = 0;
        for (int64_t row = 0; row < rows; ++row) {
            shared_offsets[row] = offset;
            page_offsets[row] = offset;
            offset += pages_per_row[row];
        }
    }
    __syncthreads();

    int64_t total = rows * max_pages;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t row = idx / max_pages;
        int64_t slot = idx - row * max_pages;
        int32_t pages = pages_per_row[row];
        block_table[idx] = slot < pages ? shared_offsets[row] + static_cast<int32_t>(slot) : 0;
    }
}

__global__ void compact_keep_blocks_kernel(
    const bool* __restrict__ keep_flat,
    const int32_t* __restrict__ sel_blocks,
    int64_t* __restrict__ pos,
    int32_t* __restrict__ local_rank,
    int64_t rows,
    int64_t kv_blocks) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows || threadIdx.x != 0) {
        return;
    }

    int32_t offset = 0;
    for (int64_t r = 0; r < row; ++r) {
        offset += sel_blocks[r];
    }

    int32_t rank = 0;
    const bool* row_ptr = keep_flat + row * kv_blocks;
    for (int64_t blk = 0; blk < kv_blocks; ++blk) {
        if (row_ptr[blk]) {
            int64_t out_idx = static_cast<int64_t>(offset + rank);
            pos[out_idx * 2] = row;
            pos[out_idx * 2 + 1] = blk;
            local_rank[out_idx] = rank;
            ++rank;
        }
    }
}

__global__ void compact_keep_blocks_and_build_table_kernel(
    const bool* __restrict__ keep_flat,
    const int32_t* __restrict__ sel_blocks,
    const int32_t* __restrict__ pages_per_row,
    int64_t* __restrict__ pos,
    int32_t* __restrict__ local_rank,
    int32_t* __restrict__ block_table,
    int32_t* __restrict__ page_offsets,
    int64_t rows,
    int64_t kv_blocks,
    int64_t max_pages) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    __shared__ int32_t selected_base;
    __shared__ int32_t page_base;
    if (threadIdx.x == 0) {
        int32_t selected_offset = 0;
        int32_t page_offset = 0;
        for (int64_t r = 0; r < row; ++r) {
            selected_offset += sel_blocks[r];
            page_offset += pages_per_row[r];
        }
        selected_base = selected_offset;
        page_base = page_offset;
        page_offsets[row] = page_offset;
    }
    __syncthreads();

    int32_t pages = pages_per_row[row];
    int64_t table_base = row * max_pages;
    for (int64_t slot = threadIdx.x; slot < max_pages; slot += blockDim.x) {
        block_table[table_base + slot] = slot < pages ? page_base + static_cast<int32_t>(slot) : 0;
    }

    if (threadIdx.x == 0) {
        int32_t rank = 0;
        const bool* row_ptr = keep_flat + row * kv_blocks;
        for (int64_t blk = 0; blk < kv_blocks; ++blk) {
            if (row_ptr[blk]) {
                int64_t out_idx = static_cast<int64_t>(selected_base + rank);
                pos[out_idx * 2] = row;
                pos[out_idx * 2 + 1] = blk;
                local_rank[out_idx] = rank;
                ++rank;
            }
        }
    }
}

template <typename scalar_t>
__global__ void build_keep_past_fast_kernel(
    const scalar_t* __restrict__ attn_gate_score,
    bool* __restrict__ keep_past,
    float threshold,
    int64_t outer,
    int64_t q_blocks,
    int64_t kv_blocks,
    int64_t past_k_blocks) {
    int64_t outer_idx = static_cast<int64_t>(blockIdx.x);
    if (outer_idx >= outer) {
        return;
    }

    const scalar_t* score_ptr = attn_gate_score + outer_idx * q_blocks * kv_blocks;
    bool* out_ptr = keep_past + outer_idx * past_k_blocks;
    for (int64_t col = threadIdx.x; col < past_k_blocks; col += blockDim.x) {
        float col_max = -FLT_MAX;
        for (int64_t q = 0; q < q_blocks; ++q) {
            float val = static_cast<float>(score_ptr[q * kv_blocks + col]);
            col_max = val > col_max ? val : col_max;
        }
        out_ptr[col] = col_max > threshold;
    }
}

template <typename scalar_t>
__global__ void build_keep_curr_fast_kernel(
    const scalar_t* __restrict__ attn_gate_score,
    bool* __restrict__ keep_curr,
    float threshold,
    int64_t outer,
    int64_t q_blocks,
    int64_t kv_blocks,
    int64_t past_k_blocks,
    int64_t curr_k_blocks) {
    int64_t outer_idx = static_cast<int64_t>(blockIdx.x);
    if (outer_idx >= outer) {
        return;
    }

    const scalar_t* score_ptr = attn_gate_score + outer_idx * q_blocks * kv_blocks;
    bool* out_ptr = keep_curr + outer_idx * curr_k_blocks;
    for (int64_t col = threadIdx.x; col < curr_k_blocks; col += blockDim.x) {
        float col_max = -FLT_MAX;
        int64_t q_start = col < q_blocks ? col : q_blocks;
        for (int64_t q = q_start; q < q_blocks; ++q) {
            float val = static_cast<float>(score_ptr[q * kv_blocks + (past_k_blocks + col)]);
            col_max = val > col_max ? val : col_max;
        }
        out_ptr[col] = col_max > threshold;
    }
}

__global__ void build_past_indices_and_metadata_from_keep_block_kernel(
    const bool* __restrict__ keep_block,
    int32_t* __restrict__ past_block_indices,
    int32_t* __restrict__ past_block_counts,
    int32_t* __restrict__ pages_per_row,
    int32_t* __restrict__ cache_seqlens,
    int64_t batch_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_key_value_groups,
    int64_t kv_blocks,
    int64_t past_block_stride,
    int64_t past_k_blocks,
    int64_t curr_k_blocks,
    int64_t block_size,
    int64_t page_block_size) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= batch_size * num_kv_heads) {
        return;
    }
    extern __shared__ int32_t shared_keep_flags[];

    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;
    int64_t q_head_base = kv_head_idx * num_key_value_groups;
    int64_t row_base = row * past_block_stride;

    for (int64_t blk = threadIdx.x; blk < past_k_blocks; blk += blockDim.x) {
        bool keep = false;
        int64_t keep_base = ((batch_idx * num_q_heads) + q_head_base) * kv_blocks + blk;
        #pragma unroll
        for (int64_t g = 0; g < num_key_value_groups; ++g) {
            keep = keep || keep_block[keep_base + g * kv_blocks];
        }
        shared_keep_flags[blk] = keep ? 1 : 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t count = 0;
        for (int64_t blk = 0; blk < past_k_blocks; ++blk) {
            if (shared_keep_flags[blk] != 0) {
                past_block_indices[row_base + count] = static_cast<int32_t>(blk);
                ++count;
            }
        }
        for (int64_t idx = count; idx < past_k_blocks; ++idx) {
            past_block_indices[row_base + idx] = -1;
        }

        past_block_counts[row] = count;
        int32_t total_blocks = count + static_cast<int32_t>(curr_k_blocks);
        int32_t total_tokens = total_blocks * static_cast<int32_t>(block_size);
        cache_seqlens[row] = total_tokens;
        pages_per_row[row] =
            (total_tokens + static_cast<int32_t>(page_block_size) - 1) / static_cast<int32_t>(page_block_size);
    }
}

__global__ void build_selected_indices_and_metadata_from_keep_block_kernel(
    const bool* __restrict__ keep_block,
    int32_t* __restrict__ selected_block_indices,
    int32_t* __restrict__ selected_block_counts,
    int32_t* __restrict__ pages_per_row,
    int32_t* __restrict__ cache_seqlens,
    int64_t batch_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_key_value_groups,
    int64_t kv_blocks,
    int64_t selected_block_stride,
    int64_t past_k_blocks,
    int64_t curr_k_blocks,
    int64_t block_size,
    int64_t page_block_size) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= batch_size * num_kv_heads) {
        return;
    }
    extern __shared__ int32_t shared_keep_flags[];

    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;
    int64_t q_head_base = kv_head_idx * num_key_value_groups;
    int64_t row_base = row * selected_block_stride;

    for (int64_t blk = threadIdx.x; blk < past_k_blocks; blk += blockDim.x) {
        bool keep = false;
        int64_t keep_base = ((batch_idx * num_q_heads) + q_head_base) * kv_blocks + blk;
        #pragma unroll
        for (int64_t g = 0; g < num_key_value_groups; ++g) {
            keep = keep || keep_block[keep_base + g * kv_blocks];
        }
        shared_keep_flags[blk] = keep ? 1 : 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t count = 0;
        for (int64_t blk = 0; blk < past_k_blocks; ++blk) {
            if (shared_keep_flags[blk] != 0) {
                selected_block_indices[row_base + count] = static_cast<int32_t>(blk);
                ++count;
            }
        }
        for (int64_t curr_blk = 0; curr_blk < curr_k_blocks; ++curr_blk) {
            selected_block_indices[row_base + count] = static_cast<int32_t>(past_k_blocks + curr_blk);
            ++count;
        }
        for (int64_t idx = count; idx < selected_block_stride; ++idx) {
            selected_block_indices[row_base + idx] = -1;
        }

        selected_block_counts[row] = count;
        int32_t total_tokens = count * static_cast<int32_t>(block_size);
        cache_seqlens[row] = total_tokens;
        pages_per_row[row] =
            (total_tokens + static_cast<int32_t>(page_block_size) - 1) / static_cast<int32_t>(page_block_size);
    }
}

__global__ void build_selected_indices_from_kv_keep_block_kernel(
    const bool* __restrict__ keep_block_kv,
    int32_t* __restrict__ selected_block_indices,
    int32_t* __restrict__ selected_block_counts,
    int64_t batch_size,
    int64_t num_kv_heads,
    int64_t kv_blocks,
    int64_t selected_block_stride) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= batch_size * num_kv_heads) {
        return;
    }
    extern __shared__ int32_t shared_keep_flags[];

    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;
    int64_t keep_base = ((batch_idx * num_kv_heads) + kv_head_idx) * kv_blocks;
    int64_t row_base = row * selected_block_stride;

    for (int64_t blk = threadIdx.x; blk < kv_blocks; blk += blockDim.x) {
        shared_keep_flags[blk] = keep_block_kv[keep_base + blk] ? 1 : 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t count = 0;
        for (int64_t blk = 0; blk < kv_blocks; ++blk) {
            if (shared_keep_flags[blk] != 0) {
                selected_block_indices[row_base + count] = static_cast<int32_t>(blk);
                ++count;
            }
        }
        selected_block_counts[row] = count;
    }
}

__global__ void build_flashinfer_kv_indices_kernel(
    const int32_t* __restrict__ selected_block_indices,
    const int32_t* __restrict__ selected_block_counts,
    const int32_t* __restrict__ kv_indptr,
    int32_t* __restrict__ kv_indices,
    int64_t rows,
    int64_t selected_block_stride,
    int64_t kv_blocks) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    int32_t count = selected_block_counts[row];
    int64_t row_base = row * selected_block_stride;
    int64_t out_base = static_cast<int64_t>(kv_indptr[row]);
    int32_t global_base = static_cast<int32_t>(row * kv_blocks);
    for (int64_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        kv_indices[out_base + idx] = global_base + selected_block_indices[row_base + idx];
    }
}

__global__ void build_flashinfer_kv_indices_per_query_kernel(
    const int32_t* __restrict__ selected_block_indices,
    const int32_t* __restrict__ selected_block_counts,
    const int32_t* __restrict__ kv_indptr,
    int32_t* __restrict__ kv_indices,
    int64_t rows,
    int64_t selected_block_stride,
    int64_t kv_blocks,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_key_value_groups) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    int64_t batch_idx = row / num_q_heads;
    int64_t q_head_idx = row - batch_idx * num_q_heads;
    int64_t kv_head_idx = q_head_idx / num_key_value_groups;
    int32_t count = selected_block_counts[row];
    int64_t row_base = row * selected_block_stride;
    int64_t out_base = static_cast<int64_t>(kv_indptr[row]);
    int32_t global_base = static_cast<int32_t>(
        (batch_idx * num_kv_heads + kv_head_idx) * kv_blocks);
    for (int64_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        kv_indices[out_base + idx] = global_base + selected_block_indices[row_base + idx];
    }
}

template <typename scalar_t>
__global__ void cache_fill_from_past_indices_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ past_block_indices,
    const int32_t* __restrict__ past_block_counts,
    const int32_t* __restrict__ page_offsets,
    scalar_t* __restrict__ k_cache_flat,
    scalar_t* __restrict__ v_cache_flat,
    int64_t kv_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t past_block_stride,
    int64_t block_size,
    int64_t page_block_size,
    int64_t active_past_k_blocks,
    int64_t rows) {
    constexpr int64_t kVecBytes = sizeof(uint4);
    constexpr int64_t kElemsPerVec = kVecBytes / sizeof(scalar_t);
    constexpr int64_t kVecsPerHead = 128 / kElemsPerVec;

    int64_t global_block = static_cast<int64_t>(blockIdx.x);
    int64_t row = global_block / active_past_k_blocks;
    int64_t local_rank = global_block - row * active_past_k_blocks;
    if (row >= rows || local_rank >= static_cast<int64_t>(past_block_counts[row])) {
        return;
    }

    int32_t blk = past_block_indices[row * past_block_stride + local_rank];
    if (blk < 0) {
        return;
    }

    int64_t dst_base =
        static_cast<int64_t>(page_offsets[row]) * page_block_size + local_rank * block_size;
    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;
    int64_t total_vec = block_size * kVecsPerHead;

    const uint4* k_vec = reinterpret_cast<const uint4*>(k);
    const uint4* v_vec = reinterpret_cast<const uint4*>(v);
    uint4* k_cache_vec = reinterpret_cast<uint4*>(k_cache_flat);
    uint4* v_cache_vec = reinterpret_cast<uint4*>(v_cache_flat);

    for (int64_t vec_idx = threadIdx.x; vec_idx < total_vec; vec_idx += blockDim.x) {
        int64_t tok_off = vec_idx / kVecsPerHead;
        int64_t head_vec = vec_idx - tok_off * kVecsPerHead;
        int64_t tok = static_cast<int64_t>(blk) * block_size + tok_off;
        int64_t src_elem_idx =
            (((batch_idx * kv_len + tok) * num_kv_heads + kv_head_idx) * head_dim)
            + head_vec * kElemsPerVec;
        int64_t dst_elem_idx = ((dst_base + tok_off) * head_dim) + head_vec * kElemsPerVec;
        int64_t src_vec_idx = src_elem_idx / kElemsPerVec;
        int64_t dst_vec_idx = dst_elem_idx / kElemsPerVec;
        k_cache_vec[dst_vec_idx] = k_vec[src_vec_idx];
        v_cache_vec[dst_vec_idx] = v_vec[src_vec_idx];
    }
}

template <typename scalar_t>
__global__ void cache_fill_from_past_indices_compact_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ past_block_indices,
    const int32_t* __restrict__ past_block_counts,
    const int32_t* __restrict__ selected_offsets,
    const int32_t* __restrict__ page_offsets,
    scalar_t* __restrict__ k_cache_flat,
    scalar_t* __restrict__ v_cache_flat,
    int64_t kv_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t past_block_stride,
    int64_t block_size,
    int64_t page_block_size,
    int64_t rows,
    int64_t total_selected) {
    constexpr int64_t kVecBytes = sizeof(uint4);
    constexpr int64_t kElemsPerVec = kVecBytes / sizeof(scalar_t);
    constexpr int64_t kVecsPerHead = 128 / kElemsPerVec;

    int64_t global_block = static_cast<int64_t>(blockIdx.x);
    if (global_block >= total_selected) {
        return;
    }

    int64_t row = 0;
    for (int64_t next_row = 1; next_row < rows; ++next_row) {
        if (global_block < static_cast<int64_t>(selected_offsets[next_row])) {
            break;
        }
        row = next_row;
    }
    int64_t local_rank = global_block - static_cast<int64_t>(selected_offsets[row]);
    if (local_rank >= static_cast<int64_t>(past_block_counts[row])) {
        return;
    }

    int32_t blk = past_block_indices[row * past_block_stride + local_rank];
    if (blk < 0) {
        return;
    }

    int64_t dst_base =
        static_cast<int64_t>(page_offsets[row]) * page_block_size + local_rank * block_size;
    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;
    int64_t total_vec = block_size * kVecsPerHead;

    const uint4* k_vec = reinterpret_cast<const uint4*>(k);
    const uint4* v_vec = reinterpret_cast<const uint4*>(v);
    uint4* k_cache_vec = reinterpret_cast<uint4*>(k_cache_flat);
    uint4* v_cache_vec = reinterpret_cast<uint4*>(v_cache_flat);

    for (int64_t vec_idx = threadIdx.x; vec_idx < total_vec; vec_idx += blockDim.x) {
        int64_t tok_off = vec_idx / kVecsPerHead;
        int64_t head_vec = vec_idx - tok_off * kVecsPerHead;
        int64_t tok = static_cast<int64_t>(blk) * block_size + tok_off;
        int64_t src_elem_idx =
            (((batch_idx * kv_len + tok) * num_kv_heads + kv_head_idx) * head_dim)
            + head_vec * kElemsPerVec;
        int64_t dst_elem_idx = ((dst_base + tok_off) * head_dim) + head_vec * kElemsPerVec;
        int64_t src_vec_idx = src_elem_idx / kElemsPerVec;
        int64_t dst_vec_idx = dst_elem_idx / kElemsPerVec;
        k_cache_vec[dst_vec_idx] = k_vec[src_vec_idx];
        v_cache_vec[dst_vec_idx] = v_vec[src_vec_idx];
    }
}

template <typename scalar_t>
__global__ void cache_fill_current_tail_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ past_block_counts,
    const int32_t* __restrict__ page_offsets,
    scalar_t* __restrict__ k_cache_flat,
    scalar_t* __restrict__ v_cache_flat,
    int64_t kv_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t curr_k_blocks,
    int64_t past_k_blocks,
    int64_t block_size,
    int64_t page_block_size,
    int64_t rows) {
    constexpr int64_t kVecBytes = sizeof(uint4);
    constexpr int64_t kElemsPerVec = kVecBytes / sizeof(scalar_t);
    constexpr int64_t kVecsPerHead = 128 / kElemsPerVec;

    int64_t global_block = static_cast<int64_t>(blockIdx.x);
    int64_t row = global_block / curr_k_blocks;
    int64_t curr_rank = global_block - row * curr_k_blocks;
    if (row >= rows) {
        return;
    }

    int64_t dst_base =
        static_cast<int64_t>(page_offsets[row]) * page_block_size
        + static_cast<int64_t>(past_block_counts[row]) * block_size
        + curr_rank * block_size;
    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;
    int64_t blk = past_k_blocks + curr_rank;
    int64_t total_vec = block_size * kVecsPerHead;

    const uint4* k_vec = reinterpret_cast<const uint4*>(k);
    const uint4* v_vec = reinterpret_cast<const uint4*>(v);
    uint4* k_cache_vec = reinterpret_cast<uint4*>(k_cache_flat);
    uint4* v_cache_vec = reinterpret_cast<uint4*>(v_cache_flat);

    for (int64_t vec_idx = threadIdx.x; vec_idx < total_vec; vec_idx += blockDim.x) {
        int64_t tok_off = vec_idx / kVecsPerHead;
        int64_t head_vec = vec_idx - tok_off * kVecsPerHead;
        int64_t tok = blk * block_size + tok_off;
        int64_t src_elem_idx =
            (((batch_idx * kv_len + tok) * num_kv_heads + kv_head_idx) * head_dim)
            + head_vec * kElemsPerVec;
        int64_t dst_elem_idx = ((dst_base + tok_off) * head_dim) + head_vec * kElemsPerVec;
        int64_t src_vec_idx = src_elem_idx / kElemsPerVec;
        int64_t dst_vec_idx = dst_elem_idx / kElemsPerVec;
        k_cache_vec[dst_vec_idx] = k_vec[src_vec_idx];
        v_cache_vec[dst_vec_idx] = v_vec[src_vec_idx];
    }
}

template <typename scalar_t>
__global__ void cache_fill_from_selected_indices_row_tiled_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ selected_block_indices,
    const int32_t* __restrict__ selected_block_counts,
    const int32_t* __restrict__ page_offsets,
    scalar_t* __restrict__ k_cache_flat,
    scalar_t* __restrict__ v_cache_flat,
    int64_t kv_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t selected_block_stride,
    int64_t blocks_per_tile,
    int64_t block_size,
    int64_t page_block_size,
    int64_t rows) {
    constexpr int64_t kVecBytes = sizeof(uint4);
    constexpr int64_t kElemsPerVec = kVecBytes / sizeof(scalar_t);
    constexpr int64_t kVecsPerHead = 128 / kElemsPerVec;

    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    int32_t selected_count = selected_block_counts[row];
    if (selected_count <= 0) {
        return;
    }
    int64_t groups_per_tile = blocks_per_tile > 0 ? blocks_per_tile : 1;
    int64_t threads_per_group = (static_cast<int64_t>(blockDim.x) / groups_per_tile);
    threads_per_group = threads_per_group > 0 ? threads_per_group : 1;
    int64_t group_idx = static_cast<int64_t>(threadIdx.x) / threads_per_group;
    if (group_idx >= groups_per_tile) {
        return;
    }
    int64_t worker_idx = static_cast<int64_t>(blockIdx.y);
    int64_t thread_in_group = static_cast<int64_t>(threadIdx.x) - group_idx * threads_per_group;

    int64_t batch_idx = row / num_kv_heads;
    int64_t kv_head_idx = row - batch_idx * num_kv_heads;
    int64_t total_vec = block_size * kVecsPerHead;
    int64_t row_base = row * selected_block_stride;
    int64_t page_base = static_cast<int64_t>(page_offsets[row]) * page_block_size;

    const uint4* k_vec = reinterpret_cast<const uint4*>(k);
    const uint4* v_vec = reinterpret_cast<const uint4*>(v);
    uint4* k_cache_vec = reinterpret_cast<uint4*>(k_cache_flat);
    uint4* v_cache_vec = reinterpret_cast<uint4*>(v_cache_flat);

    int64_t rank_stride = static_cast<int64_t>(gridDim.y) * groups_per_tile;
    for (int64_t selected_rank = worker_idx * groups_per_tile + group_idx;
         selected_rank < static_cast<int64_t>(selected_count);
         selected_rank += rank_stride) {
        int32_t blk = selected_block_indices[row_base + selected_rank];
        if (blk < 0) {
            continue;
        }
        int64_t dst_base = page_base + selected_rank * block_size;

        for (int64_t vec_idx = thread_in_group; vec_idx < total_vec; vec_idx += threads_per_group) {
            int64_t tok_off = vec_idx / kVecsPerHead;
            int64_t head_vec = vec_idx - tok_off * kVecsPerHead;
            int64_t tok = static_cast<int64_t>(blk) * block_size + tok_off;
            int64_t src_elem_idx =
                (((batch_idx * kv_len + tok) * num_kv_heads + kv_head_idx) * head_dim)
                + head_vec * kElemsPerVec;
            int64_t dst_elem_idx = ((dst_base + tok_off) * head_dim) + head_vec * kElemsPerVec;
            int64_t src_vec_idx = src_elem_idx / kElemsPerVec;
            int64_t dst_vec_idx = dst_elem_idx / kElemsPerVec;
            k_cache_vec[dst_vec_idx] = k_vec[src_vec_idx];
            v_cache_vec[dst_vec_idx] = v_vec[src_vec_idx];
        }
    }
}

}  // namespace

void cache_fill_from_pos_rank_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& pos,
    const torch::Tensor& keep_prefix_rank,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size,
    int64_t page_block_size) {
    const at::cuda::CUDAGuard device_guard(k.device());
    int64_t num_blocks = pos.size(0);
    if (num_blocks <= 0) {
        return;
    }

    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(num_blocks));
    auto stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k.scalar_type(),
        "cache_fill_from_pos_rank_cuda",
        [&] {
            cache_fill_from_pos_rank_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                pos.data_ptr<int64_t>(),
                keep_prefix_rank.data_ptr<int32_t>(),
                page_offsets.data_ptr<int32_t>(),
                k_cache_flat.data_ptr<scalar_t>(),
                v_cache_flat.data_ptr<scalar_t>(),
                k.size(1),
                k.size(2),
                k.size(3),
                keep_prefix_rank.size(1),
                block_size,
                page_block_size,
                num_blocks);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cache_fill_from_row_blk_dst_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& row_idx,
    const torch::Tensor& blk_idx,
    const torch::Tensor& dst_token_base,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size) {
    const at::cuda::CUDAGuard device_guard(k.device());
    int64_t num_blocks = row_idx.size(0);
    if (num_blocks <= 0) {
        return;
    }

    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(num_blocks));
    auto stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k.scalar_type(),
        "cache_fill_from_row_blk_dst_cuda",
        [&] {
            cache_fill_from_row_blk_dst_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                row_idx.data_ptr<int64_t>(),
                blk_idx.data_ptr<int64_t>(),
                dst_token_base.data_ptr<int64_t>(),
                k_cache_flat.data_ptr<scalar_t>(),
                v_cache_flat.data_ptr<scalar_t>(),
                k.size(1),
                k.size(2),
                k.size(3),
                block_size,
                num_blocks);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cache_fill_from_pos_local_rank_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& pos,
    const torch::Tensor& local_rank,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size,
    int64_t page_block_size) {
    const at::cuda::CUDAGuard device_guard(k.device());
    int64_t num_blocks = pos.size(0);
    if (num_blocks <= 0) {
        return;
    }

    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(num_blocks));
    auto stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k.scalar_type(),
        "cache_fill_from_pos_local_rank_cuda",
        [&] {
            cache_fill_from_pos_local_rank_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                pos.data_ptr<int64_t>(),
                local_rank.data_ptr<int32_t>(),
                page_offsets.data_ptr<int32_t>(),
                k_cache_flat.data_ptr<scalar_t>(),
                v_cache_flat.data_ptr<scalar_t>(),
                k.size(1),
                k.size(2),
                k.size(3),
                block_size,
                page_block_size,
                num_blocks);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_q_for_indexed_prefill_cuda(
    const torch::Tensor& q,
    torch::Tensor& q_group,
    int64_t num_kv_heads,
    int64_t num_key_value_groups) {
    const at::cuda::CUDAGuard device_guard(q.device());
    int64_t total_rows = q.size(0) * num_kv_heads;
    int64_t num_row_tokens = total_rows * q.size(1);
    if (num_row_tokens <= 0) {
        return;
    }

    int threads = static_cast<int>(num_key_value_groups * 16);
    threads = std::max(32, std::min(256, threads));
    const dim3 blocks(static_cast<unsigned int>(num_row_tokens));
    auto stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "pack_q_for_indexed_prefill_cuda",
        [&] {
            pack_q_for_indexed_prefill_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                q.data_ptr<scalar_t>(),
                q_group.data_ptr<scalar_t>(),
                q.size(1),
                q.size(2),
                num_kv_heads,
                num_key_value_groups,
                q.size(3),
                total_rows);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_block_table_cuda(
    const torch::Tensor& pages_per_row,
    torch::Tensor& block_table,
    torch::Tensor& page_offsets) {
    const at::cuda::CUDAGuard device_guard(pages_per_row.device());
    int64_t rows = pages_per_row.size(0);
    int64_t max_pages = block_table.size(1);
    if (rows <= 0 || max_pages <= 0) {
        return;
    }

    const int threads = 256;
    int64_t total = rows * max_pages;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks < 1) {
        blocks = 1;
    }
    auto stream = at::cuda::getDefaultCUDAStream();
    build_block_table_kernel<<<blocks, threads, rows * sizeof(int32_t), stream>>>(
        pages_per_row.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        page_offsets.data_ptr<int32_t>(),
        rows,
        max_pages);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> compact_keep_blocks_cuda(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks) {
    const at::cuda::CUDAGuard device_guard(keep_flat.device());
    int64_t rows = keep_flat.size(0);
    int64_t kv_blocks = keep_flat.size(1);
    int64_t num_selected = sel_blocks.sum().item<int64_t>();

    auto pos = torch::empty({num_selected, 2}, keep_flat.options().dtype(torch::kInt64));
    auto local_rank = torch::empty({num_selected}, sel_blocks.options());
    if (num_selected <= 0) {
        return {pos, local_rank};
    }

    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(rows));
    auto stream = at::cuda::getDefaultCUDAStream();
    compact_keep_blocks_kernel<<<blocks, threads, 0, stream>>>(
        keep_flat.data_ptr<bool>(),
        sel_blocks.data_ptr<int32_t>(),
        pos.data_ptr<int64_t>(),
        local_rank.data_ptr<int32_t>(),
        rows,
        kv_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pos, local_rank};
}

int64_t compact_keep_blocks_out_cuda(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks,
    torch::Tensor& pos,
    torch::Tensor& local_rank) {
    const at::cuda::CUDAGuard device_guard(keep_flat.device());
    int64_t rows = keep_flat.size(0);
    int64_t kv_blocks = keep_flat.size(1);
    int64_t num_selected = sel_blocks.sum().item<int64_t>();
    TORCH_CHECK(pos.size(0) >= num_selected, "pos buffer is too small");
    TORCH_CHECK(local_rank.size(0) >= num_selected, "local_rank buffer is too small");
    if (num_selected <= 0) {
        return 0;
    }

    const int threads = 32;
    const dim3 blocks(static_cast<unsigned int>(rows));
    auto stream = at::cuda::getDefaultCUDAStream();
    compact_keep_blocks_kernel<<<blocks, threads, 0, stream>>>(
        keep_flat.data_ptr<bool>(),
        sel_blocks.data_ptr<int32_t>(),
        pos.data_ptr<int64_t>(),
        local_rank.data_ptr<int32_t>(),
        rows,
        kv_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return num_selected;
}

std::vector<torch::Tensor> compact_keep_blocks_and_build_table_cuda(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks,
    const torch::Tensor& pages_per_row,
    torch::Tensor& block_table,
    torch::Tensor& page_offsets) {
    const at::cuda::CUDAGuard device_guard(keep_flat.device());
    int64_t rows = keep_flat.size(0);
    int64_t kv_blocks = keep_flat.size(1);
    int64_t max_pages = block_table.size(1);
    int64_t num_selected = sel_blocks.sum().item<int64_t>();

    auto pos = torch::empty({num_selected, 2}, keep_flat.options().dtype(torch::kInt64));
    auto local_rank = torch::empty({num_selected}, sel_blocks.options());
    if (rows <= 0) {
        return {pos, local_rank};
    }

    const int threads = 128;
    const dim3 blocks(static_cast<unsigned int>(rows));
    auto stream = at::cuda::getDefaultCUDAStream();
    compact_keep_blocks_and_build_table_kernel<<<blocks, threads, 0, stream>>>(
        keep_flat.data_ptr<bool>(),
        sel_blocks.data_ptr<int32_t>(),
        pages_per_row.data_ptr<int32_t>(),
        pos.data_ptr<int64_t>(),
        local_rank.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        page_offsets.data_ptr<int32_t>(),
        rows,
        kv_blocks,
        max_pages);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pos, local_rank};
}

int64_t compact_keep_blocks_and_build_table_out_cuda(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks,
    const torch::Tensor& pages_per_row,
    torch::Tensor& pos,
    torch::Tensor& local_rank,
    torch::Tensor& block_table,
    torch::Tensor& page_offsets) {
    const at::cuda::CUDAGuard device_guard(keep_flat.device());
    int64_t rows = keep_flat.size(0);
    int64_t kv_blocks = keep_flat.size(1);
    int64_t max_pages = block_table.size(1);
    int64_t num_selected = sel_blocks.sum().item<int64_t>();
    TORCH_CHECK(pos.size(0) >= num_selected, "pos buffer is too small");
    TORCH_CHECK(local_rank.size(0) >= num_selected, "local_rank buffer is too small");
    if (rows <= 0) {
        return 0;
    }

    const int threads = 128;
    const dim3 blocks(static_cast<unsigned int>(rows));
    auto stream = at::cuda::getDefaultCUDAStream();
    compact_keep_blocks_and_build_table_kernel<<<blocks, threads, 0, stream>>>(
        keep_flat.data_ptr<bool>(),
        sel_blocks.data_ptr<int32_t>(),
        pages_per_row.data_ptr<int32_t>(),
        pos.data_ptr<int64_t>(),
        local_rank.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        page_offsets.data_ptr<int32_t>(),
        rows,
        kv_blocks,
        max_pages);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return num_selected;
}

torch::Tensor build_keep_past_fast_cuda(
    const torch::Tensor& attn_gate_score,
    double threshold,
    int64_t past_k_blocks) {
    int64_t outer = attn_gate_score.size(0) * attn_gate_score.size(1);
    auto out = torch::empty(
        {attn_gate_score.size(0), attn_gate_score.size(1), past_k_blocks},
        attn_gate_score.options().dtype(torch::kBool));
    if (past_k_blocks <= 0 || outer <= 0) {
        return out;
    }

    const int threads = 256;
    auto stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        attn_gate_score.scalar_type(),
        "build_keep_past_fast_cuda",
        [&] {
            build_keep_past_fast_kernel<scalar_t><<<static_cast<unsigned int>(outer), threads, 0, stream>>>(
                attn_gate_score.data_ptr<scalar_t>(),
                out.data_ptr<bool>(),
                static_cast<float>(threshold),
                outer,
                attn_gate_score.size(2),
                attn_gate_score.size(3),
                past_k_blocks);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor build_keep_curr_fast_cuda(
    const torch::Tensor& attn_gate_score,
    double threshold,
    int64_t past_k_blocks,
    int64_t curr_k_blocks) {
    int64_t outer = attn_gate_score.size(0) * attn_gate_score.size(1);
    auto out = torch::empty(
        {attn_gate_score.size(0), attn_gate_score.size(1), curr_k_blocks},
        attn_gate_score.options().dtype(torch::kBool));
    if (curr_k_blocks <= 0 || outer <= 0) {
        return out;
    }

    const int threads = 64;
    auto stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        attn_gate_score.scalar_type(),
        "build_keep_curr_fast_cuda",
        [&] {
            build_keep_curr_fast_kernel<scalar_t><<<static_cast<unsigned int>(outer), threads, 0, stream>>>(
                attn_gate_score.data_ptr<scalar_t>(),
                out.data_ptr<bool>(),
                static_cast<float>(threshold),
                outer,
                attn_gate_score.size(2),
                attn_gate_score.size(3),
                past_k_blocks,
                curr_k_blocks);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void build_past_indices_and_metadata_from_keep_block_cuda(
    const torch::Tensor& keep_block,
    int64_t num_key_value_groups,
    int64_t past_k_blocks,
    int64_t curr_k_blocks,
    int64_t block_size,
    int64_t page_block_size,
    torch::Tensor& past_block_indices,
    torch::Tensor& past_block_counts,
    torch::Tensor& pages_per_row,
    torch::Tensor& cache_seqlens) {
    const at::cuda::CUDAGuard device_guard(keep_block.device());
    int64_t batch_size = keep_block.size(0);
    int64_t num_q_heads = keep_block.size(1);
    int64_t num_kv_heads = num_q_heads / num_key_value_groups;
    int64_t rows = batch_size * num_kv_heads;
    if (rows <= 0) {
        return;
    }

    const int threads = 32;
    const dim3 blocks(static_cast<unsigned int>(rows));
    auto stream = at::cuda::getDefaultCUDAStream();
    build_past_indices_and_metadata_from_keep_block_kernel<<<
        blocks,
        threads,
        static_cast<size_t>(std::max<int64_t>(past_k_blocks, 1)) * sizeof(int32_t),
        stream>>>(
        keep_block.data_ptr<bool>(),
        past_block_indices.data_ptr<int32_t>(),
        past_block_counts.data_ptr<int32_t>(),
        pages_per_row.data_ptr<int32_t>(),
        cache_seqlens.data_ptr<int32_t>(),
        batch_size,
        num_q_heads,
        num_kv_heads,
        num_key_value_groups,
        keep_block.size(2),
        past_block_indices.size(1),
        past_k_blocks,
        curr_k_blocks,
        block_size,
        page_block_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_selected_indices_and_metadata_from_keep_block_cuda(
    const torch::Tensor& keep_block,
    int64_t num_key_value_groups,
    int64_t past_k_blocks,
    int64_t curr_k_blocks,
    int64_t block_size,
    int64_t page_block_size,
    torch::Tensor& selected_block_indices,
    torch::Tensor& selected_block_counts,
    torch::Tensor& pages_per_row,
    torch::Tensor& cache_seqlens) {
    const at::cuda::CUDAGuard device_guard(keep_block.device());
    int64_t batch_size = keep_block.size(0);
    int64_t num_q_heads = keep_block.size(1);
    int64_t num_kv_heads = num_q_heads / num_key_value_groups;
    int64_t rows = batch_size * num_kv_heads;
    if (rows <= 0) {
        return;
    }

    const int threads = 32;
    const dim3 blocks(static_cast<unsigned int>(rows));
    auto stream = at::cuda::getDefaultCUDAStream();
    build_selected_indices_and_metadata_from_keep_block_kernel<<<
        blocks,
        threads,
        static_cast<size_t>(std::max<int64_t>(past_k_blocks, 1)) * sizeof(int32_t),
        stream>>>(
        keep_block.data_ptr<bool>(),
        selected_block_indices.data_ptr<int32_t>(),
        selected_block_counts.data_ptr<int32_t>(),
        pages_per_row.data_ptr<int32_t>(),
        cache_seqlens.data_ptr<int32_t>(),
        batch_size,
        num_q_heads,
        num_kv_heads,
        num_key_value_groups,
        keep_block.size(2),
        selected_block_indices.size(1),
        past_k_blocks,
        curr_k_blocks,
        block_size,
        page_block_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_selected_indices_from_kv_keep_block_cuda(
    const torch::Tensor& keep_block_kv,
    torch::Tensor& selected_block_indices,
    torch::Tensor& selected_block_counts) {
    const at::cuda::CUDAGuard device_guard(keep_block_kv.device());
    int64_t batch_size = keep_block_kv.size(0);
    int64_t num_kv_heads = keep_block_kv.size(1);
    int64_t rows = batch_size * num_kv_heads;
    int64_t kv_blocks = keep_block_kv.size(2);
    if (rows <= 0 || kv_blocks <= 0) {
        return;
    }

    const int threads = 64;
    const dim3 blocks(static_cast<unsigned int>(rows));
    auto stream = at::cuda::getDefaultCUDAStream();
    build_selected_indices_from_kv_keep_block_kernel<<<
        blocks,
        threads,
        static_cast<size_t>(kv_blocks) * sizeof(int32_t),
        stream>>>(
        keep_block_kv.data_ptr<bool>(),
        selected_block_indices.data_ptr<int32_t>(),
        selected_block_counts.data_ptr<int32_t>(),
        batch_size,
        num_kv_heads,
        kv_blocks,
        selected_block_indices.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_flashinfer_kv_indices_cuda(
    const torch::Tensor& selected_block_indices,
    const torch::Tensor& selected_block_counts,
    const torch::Tensor& kv_indptr,
    torch::Tensor& kv_indices,
    int64_t kv_blocks) {
    const at::cuda::CUDAGuard device_guard(selected_block_indices.device());
    int64_t rows = selected_block_counts.size(0);
    if (rows <= 0) {
        return;
    }

    const int threads = 128;
    const dim3 blocks(static_cast<unsigned int>(rows));
    auto stream = at::cuda::getDefaultCUDAStream();
    build_flashinfer_kv_indices_kernel<<<blocks, threads, 0, stream>>>(
        selected_block_indices.data_ptr<int32_t>(),
        selected_block_counts.data_ptr<int32_t>(),
        kv_indptr.data_ptr<int32_t>(),
        kv_indices.data_ptr<int32_t>(),
        rows,
        selected_block_indices.size(1),
        kv_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_flashinfer_kv_indices_per_query_cuda(
    const torch::Tensor& selected_block_indices,
    const torch::Tensor& selected_block_counts,
    const torch::Tensor& kv_indptr,
    torch::Tensor& kv_indices,
    int64_t kv_blocks,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_key_value_groups) {
    const at::cuda::CUDAGuard device_guard(selected_block_indices.device());
    int64_t rows = selected_block_counts.size(0);
    if (rows <= 0) {
        return;
    }

    const int threads = 128;
    const dim3 blocks(static_cast<unsigned int>(rows));
    auto stream = at::cuda::getDefaultCUDAStream();
    build_flashinfer_kv_indices_per_query_kernel<<<blocks, threads, 0, stream>>>(
        selected_block_indices.data_ptr<int32_t>(),
        selected_block_counts.data_ptr<int32_t>(),
        kv_indptr.data_ptr<int32_t>(),
        kv_indices.data_ptr<int32_t>(),
        rows,
        selected_block_indices.size(1),
        kv_blocks,
        num_q_heads,
        num_kv_heads,
        num_key_value_groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cache_fill_from_past_indices_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& past_block_indices,
    const torch::Tensor& past_block_counts,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t active_past_k_blocks,
    int64_t block_size,
    int64_t page_block_size) {
    const at::cuda::CUDAGuard device_guard(k.device());
    int64_t rows = past_block_counts.size(0);
    int64_t past_block_stride = past_block_indices.size(1);
    if (rows <= 0 || active_past_k_blocks <= 0) {
        return;
    }

    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(rows * active_past_k_blocks));
    auto stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k.scalar_type(),
        "cache_fill_from_past_indices_cuda",
        [&] {
            cache_fill_from_past_indices_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                past_block_indices.data_ptr<int32_t>(),
                past_block_counts.data_ptr<int32_t>(),
                page_offsets.data_ptr<int32_t>(),
                k_cache_flat.data_ptr<scalar_t>(),
                v_cache_flat.data_ptr<scalar_t>(),
                k.size(1),
                k.size(2),
                k.size(3),
                past_block_stride,
                block_size,
                page_block_size,
                active_past_k_blocks,
                rows);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cache_fill_from_past_indices_compact_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& past_block_indices,
    const torch::Tensor& past_block_counts,
    const torch::Tensor& selected_offsets,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t total_selected,
    int64_t block_size,
    int64_t page_block_size) {
    const at::cuda::CUDAGuard device_guard(k.device());
    int64_t rows = past_block_counts.size(0);
    int64_t past_block_stride = past_block_indices.size(1);
    if (rows <= 0 || total_selected <= 0) {
        return;
    }

    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(total_selected));
    auto stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k.scalar_type(),
        "cache_fill_from_past_indices_compact_cuda",
        [&] {
            cache_fill_from_past_indices_compact_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                past_block_indices.data_ptr<int32_t>(),
                past_block_counts.data_ptr<int32_t>(),
                selected_offsets.data_ptr<int32_t>(),
                page_offsets.data_ptr<int32_t>(),
                k_cache_flat.data_ptr<scalar_t>(),
                v_cache_flat.data_ptr<scalar_t>(),
                k.size(1),
                k.size(2),
                k.size(3),
                past_block_stride,
                block_size,
                page_block_size,
                rows,
                total_selected);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cache_fill_current_tail_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& past_block_counts,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t past_k_blocks,
    int64_t curr_k_blocks,
    int64_t block_size,
    int64_t page_block_size) {
    const at::cuda::CUDAGuard device_guard(k.device());
    int64_t rows = past_block_counts.size(0);
    if (rows <= 0 || curr_k_blocks <= 0) {
        return;
    }

    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(rows * curr_k_blocks));
    auto stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k.scalar_type(),
        "cache_fill_current_tail_cuda",
        [&] {
            cache_fill_current_tail_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                past_block_counts.data_ptr<int32_t>(),
                page_offsets.data_ptr<int32_t>(),
                k_cache_flat.data_ptr<scalar_t>(),
                v_cache_flat.data_ptr<scalar_t>(),
                k.size(1),
                k.size(2),
                k.size(3),
                curr_k_blocks,
                past_k_blocks,
                block_size,
                page_block_size,
                rows);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cache_fill_from_selected_indices_row_tiled_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& selected_block_indices,
    const torch::Tensor& selected_block_counts,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size,
    int64_t page_block_size) {
    const at::cuda::CUDAGuard device_guard(k.device());
    int64_t rows = selected_block_counts.size(0);
    if (rows <= 0) {
        return;
    }

    constexpr int64_t kBlocksPerTile = 4;
    int64_t workers_per_row = 4;
    int64_t selected_block_stride = selected_block_indices.size(1);
    if (selected_block_stride >= 1024) {
        workers_per_row = 16;
    } else if (selected_block_stride >= 512) {
        workers_per_row = 8;
    }
    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(rows), static_cast<unsigned int>(workers_per_row));
    auto stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        k.scalar_type(),
        "cache_fill_from_selected_indices_row_tiled_cuda",
        [&] {
            cache_fill_from_selected_indices_row_tiled_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                selected_block_indices.data_ptr<int32_t>(),
                selected_block_counts.data_ptr<int32_t>(),
                page_offsets.data_ptr<int32_t>(),
                k_cache_flat.data_ptr<scalar_t>(),
                v_cache_flat.data_ptr<scalar_t>(),
                k.size(1),
                k.size(2),
                k.size(3),
                selected_block_stride,
                kBlocksPerTile,
                block_size,
                page_block_size,
                rows);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
