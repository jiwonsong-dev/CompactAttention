#include <torch/extension.h>

#include <vector>

void cache_fill_from_pos_rank_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& pos,
    const torch::Tensor& keep_prefix_rank,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size,
    int64_t page_block_size);

void cache_fill_from_row_blk_dst_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& row_idx,
    const torch::Tensor& blk_idx,
    const torch::Tensor& dst_token_base,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size);

void cache_fill_from_pos_local_rank_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& pos,
    const torch::Tensor& local_rank,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size,
    int64_t page_block_size);

void pack_q_for_indexed_prefill_cuda(
    const torch::Tensor& q,
    torch::Tensor& q_group,
    int64_t num_kv_heads,
    int64_t num_key_value_groups);

void build_block_table_cuda(
    const torch::Tensor& pages_per_row,
    torch::Tensor& block_table,
    torch::Tensor& page_offsets);

std::vector<torch::Tensor> compact_keep_blocks_cuda(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks);

std::vector<torch::Tensor> compact_keep_blocks_and_build_table_cuda(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks,
    const torch::Tensor& pages_per_row,
    torch::Tensor& block_table,
    torch::Tensor& page_offsets);

int64_t compact_keep_blocks_out_cuda(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks,
    torch::Tensor& pos,
    torch::Tensor& local_rank);

int64_t compact_keep_blocks_and_build_table_out_cuda(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks,
    const torch::Tensor& pages_per_row,
    torch::Tensor& pos,
    torch::Tensor& local_rank,
    torch::Tensor& block_table,
    torch::Tensor& page_offsets);

torch::Tensor build_keep_past_fast_cuda(
    const torch::Tensor& attn_gate_score,
    double threshold,
    int64_t past_k_blocks);

torch::Tensor build_keep_curr_fast_cuda(
    const torch::Tensor& attn_gate_score,
    double threshold,
    int64_t past_k_blocks,
    int64_t curr_k_blocks);

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
    torch::Tensor& cache_seqlens);

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
    torch::Tensor& cache_seqlens);

void build_selected_indices_from_kv_keep_block_cuda(
    const torch::Tensor& keep_block_kv,
    torch::Tensor& selected_block_indices,
    torch::Tensor& selected_block_counts);

void build_flashinfer_kv_indices_cuda(
    const torch::Tensor& selected_block_indices,
    const torch::Tensor& selected_block_counts,
    const torch::Tensor& kv_indptr,
    torch::Tensor& kv_indices,
    int64_t kv_blocks);

void build_flashinfer_kv_indices_per_query_cuda(
    const torch::Tensor& selected_block_indices,
    const torch::Tensor& selected_block_counts,
    const torch::Tensor& kv_indptr,
    torch::Tensor& kv_indices,
    int64_t kv_blocks,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_key_value_groups);

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
    int64_t page_block_size);

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
    int64_t page_block_size);

void cache_fill_from_selected_indices_row_tiled_cuda(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& selected_block_indices,
    const torch::Tensor& selected_block_counts,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size,
    int64_t page_block_size);

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
    int64_t page_block_size);

namespace {

void check_common_inputs(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& k_cache_flat,
    const torch::Tensor& v_cache_flat) {
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(k_cache_flat.is_cuda(), "k_cache_flat must be CUDA");
    TORCH_CHECK(v_cache_flat.is_cuda(), "v_cache_flat must be CUDA");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(k_cache_flat.is_contiguous(), "k_cache_flat must be contiguous");
    TORCH_CHECK(v_cache_flat.is_contiguous(), "v_cache_flat must be contiguous");
    TORCH_CHECK(k.scalar_type() == v.scalar_type(), "k/v dtype mismatch");
    TORCH_CHECK(k.scalar_type() == k_cache_flat.scalar_type(), "k/k_cache_flat dtype mismatch");
    TORCH_CHECK(v.scalar_type() == v_cache_flat.scalar_type(), "v/v_cache_flat dtype mismatch");
    TORCH_CHECK(k.dim() == 4, "k must be [B, K, Hkv, D]");
    TORCH_CHECK(v.dim() == 4, "v must be [B, K, Hkv, D]");
    TORCH_CHECK(k_cache_flat.dim() == 2, "k_cache_flat must be [tokens, D]");
    TORCH_CHECK(v_cache_flat.dim() == 2, "v_cache_flat must be [tokens, D]");
    TORCH_CHECK(k.size(3) == 128, "head_dim must be 128");
    TORCH_CHECK(v.size(3) == 128, "head_dim must be 128");
    TORCH_CHECK(k_cache_flat.size(1) == 128, "k_cache_flat head_dim must be 128");
    TORCH_CHECK(v_cache_flat.size(1) == 128, "v_cache_flat head_dim must be 128");
}

}  // namespace

void cache_fill_from_pos_rank(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& pos,
    const torch::Tensor& keep_prefix_rank,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size,
    int64_t page_block_size) {
    check_common_inputs(k, v, k_cache_flat, v_cache_flat);
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous(), "pos must be contiguous CUDA");
    TORCH_CHECK(keep_prefix_rank.is_cuda() && keep_prefix_rank.is_contiguous(), "keep_prefix_rank must be contiguous CUDA");
    TORCH_CHECK(page_offsets.is_cuda() && page_offsets.is_contiguous(), "page_offsets must be contiguous CUDA");
    TORCH_CHECK(pos.scalar_type() == torch::kInt64, "pos must be int64");
    TORCH_CHECK(keep_prefix_rank.scalar_type() == torch::kInt32, "keep_prefix_rank must be int32");
    TORCH_CHECK(page_offsets.scalar_type() == torch::kInt32, "page_offsets must be int32");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 2, "pos must be [Nsel, 2]");
    TORCH_CHECK(keep_prefix_rank.dim() == 2, "keep_prefix_rank must be [rows, Kb]");
    TORCH_CHECK(page_offsets.dim() == 1, "page_offsets must be [rows]");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(page_block_size > 0, "page_block_size must be positive");

    cache_fill_from_pos_rank_cuda(
        k,
        v,
        pos,
        keep_prefix_rank,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        block_size,
        page_block_size);
}

void cache_fill_from_row_blk_dst(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& row_idx,
    const torch::Tensor& blk_idx,
    const torch::Tensor& dst_token_base,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size) {
    check_common_inputs(k, v, k_cache_flat, v_cache_flat);
    TORCH_CHECK(row_idx.is_cuda() && row_idx.is_contiguous(), "row_idx must be contiguous CUDA");
    TORCH_CHECK(blk_idx.is_cuda() && blk_idx.is_contiguous(), "blk_idx must be contiguous CUDA");
    TORCH_CHECK(dst_token_base.is_cuda() && dst_token_base.is_contiguous(), "dst_token_base must be contiguous CUDA");
    TORCH_CHECK(row_idx.scalar_type() == torch::kInt64, "row_idx must be int64");
    TORCH_CHECK(blk_idx.scalar_type() == torch::kInt64, "blk_idx must be int64");
    TORCH_CHECK(dst_token_base.scalar_type() == torch::kInt64, "dst_token_base must be int64");
    TORCH_CHECK(row_idx.dim() == 1, "row_idx must be [Nsel]");
    TORCH_CHECK(blk_idx.dim() == 1, "blk_idx must be [Nsel]");
    TORCH_CHECK(dst_token_base.dim() == 1, "dst_token_base must be [Nsel]");
    TORCH_CHECK(row_idx.numel() == blk_idx.numel(), "row_idx/blk_idx size mismatch");
    TORCH_CHECK(row_idx.numel() == dst_token_base.numel(), "row_idx/dst_token_base size mismatch");
    TORCH_CHECK(block_size > 0, "block_size must be positive");

    cache_fill_from_row_blk_dst_cuda(
        k,
        v,
        row_idx,
        blk_idx,
        dst_token_base,
        k_cache_flat,
        v_cache_flat,
        block_size);
}

void cache_fill_from_pos_local_rank(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& pos,
    const torch::Tensor& local_rank,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size,
    int64_t page_block_size) {
    check_common_inputs(k, v, k_cache_flat, v_cache_flat);
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous(), "pos must be contiguous CUDA");
    TORCH_CHECK(local_rank.is_cuda() && local_rank.is_contiguous(), "local_rank must be contiguous CUDA");
    TORCH_CHECK(page_offsets.is_cuda() && page_offsets.is_contiguous(), "page_offsets must be contiguous CUDA");
    TORCH_CHECK(pos.scalar_type() == torch::kInt64, "pos must be int64");
    TORCH_CHECK(local_rank.scalar_type() == torch::kInt32, "local_rank must be int32");
    TORCH_CHECK(page_offsets.scalar_type() == torch::kInt32, "page_offsets must be int32");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 2, "pos must be [Nsel, 2]");
    TORCH_CHECK(local_rank.dim() == 1, "local_rank must be [Nsel]");
    TORCH_CHECK(page_offsets.dim() == 1, "page_offsets must be [rows]");
    TORCH_CHECK(pos.size(0) == local_rank.size(0), "pos/local_rank size mismatch");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(page_block_size > 0, "page_block_size must be positive");

    cache_fill_from_pos_local_rank_cuda(
        k,
        v,
        pos,
        local_rank,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        block_size,
        page_block_size);
}

void pack_q_for_indexed_prefill(
    const torch::Tensor& q,
    torch::Tensor& q_group,
    int64_t num_kv_heads,
    int64_t num_key_value_groups) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q_group.is_cuda(), "q_group must be CUDA");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q_group.is_contiguous(), "q_group must be contiguous");
    TORCH_CHECK(q.scalar_type() == q_group.scalar_type(), "q/q_group dtype mismatch");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16, "q must be fp16/bf16");
    TORCH_CHECK(q.dim() == 4, "q must be [B, Q, Hq, D]");
    TORCH_CHECK(q_group.dim() == 5, "q_group must be [B, Hkv, Q, G, D]");
    TORCH_CHECK(q.size(3) == 128, "head_dim must be 128");
    TORCH_CHECK(q_group.size(4) == 128, "q_group head_dim must be 128");
    TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be positive");
    TORCH_CHECK(num_key_value_groups > 0, "num_key_value_groups must be positive");
    TORCH_CHECK(q.size(2) == num_kv_heads * num_key_value_groups, "q head count mismatch");
    TORCH_CHECK(q_group.size(0) == q.size(0), "batch mismatch");
    TORCH_CHECK(q_group.size(1) == num_kv_heads, "q_group kv head mismatch");
    TORCH_CHECK(q_group.size(2) == q.size(1), "q_group q_len mismatch");
    TORCH_CHECK(q_group.size(3) == num_key_value_groups, "q_group group mismatch");

    pack_q_for_indexed_prefill_cuda(q, q_group, num_kv_heads, num_key_value_groups);
}

void build_block_table(
    const torch::Tensor& pages_per_row,
    torch::Tensor& block_table,
    torch::Tensor& page_offsets) {
    TORCH_CHECK(pages_per_row.is_cuda() && pages_per_row.is_contiguous(), "pages_per_row must be contiguous CUDA");
    TORCH_CHECK(block_table.is_cuda() && block_table.is_contiguous(), "block_table must be contiguous CUDA");
    TORCH_CHECK(page_offsets.is_cuda() && page_offsets.is_contiguous(), "page_offsets must be contiguous CUDA");
    TORCH_CHECK(pages_per_row.scalar_type() == torch::kInt32, "pages_per_row must be int32");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(page_offsets.scalar_type() == torch::kInt32, "page_offsets must be int32");
    TORCH_CHECK(pages_per_row.dim() == 1, "pages_per_row must be [rows]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [rows, max_pages]");
    TORCH_CHECK(page_offsets.dim() == 1, "page_offsets must be [rows]");
    TORCH_CHECK(block_table.size(0) == pages_per_row.size(0), "row count mismatch");
    TORCH_CHECK(page_offsets.size(0) == pages_per_row.size(0), "page_offsets row count mismatch");
    build_block_table_cuda(pages_per_row, block_table, page_offsets);
}

std::vector<torch::Tensor> compact_keep_blocks(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks) {
    TORCH_CHECK(keep_flat.is_cuda() && keep_flat.is_contiguous(), "keep_flat must be contiguous CUDA");
    TORCH_CHECK(sel_blocks.is_cuda() && sel_blocks.is_contiguous(), "sel_blocks must be contiguous CUDA");
    TORCH_CHECK(keep_flat.scalar_type() == torch::kBool, "keep_flat must be bool");
    TORCH_CHECK(sel_blocks.scalar_type() == torch::kInt32, "sel_blocks must be int32");
    TORCH_CHECK(keep_flat.dim() == 2, "keep_flat must be [rows, kv_blocks]");
    TORCH_CHECK(sel_blocks.dim() == 1, "sel_blocks must be [rows]");
    TORCH_CHECK(keep_flat.size(0) == sel_blocks.size(0), "row count mismatch");
    return compact_keep_blocks_cuda(keep_flat, sel_blocks);
}

int64_t compact_keep_blocks_out(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks,
    torch::Tensor& pos,
    torch::Tensor& local_rank) {
    TORCH_CHECK(keep_flat.is_cuda() && keep_flat.is_contiguous(), "keep_flat must be contiguous CUDA");
    TORCH_CHECK(sel_blocks.is_cuda() && sel_blocks.is_contiguous(), "sel_blocks must be contiguous CUDA");
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous(), "pos must be contiguous CUDA");
    TORCH_CHECK(local_rank.is_cuda() && local_rank.is_contiguous(), "local_rank must be contiguous CUDA");
    TORCH_CHECK(keep_flat.scalar_type() == torch::kBool, "keep_flat must be bool");
    TORCH_CHECK(sel_blocks.scalar_type() == torch::kInt32, "sel_blocks must be int32");
    TORCH_CHECK(pos.scalar_type() == torch::kInt64, "pos must be int64");
    TORCH_CHECK(local_rank.scalar_type() == torch::kInt32, "local_rank must be int32");
    TORCH_CHECK(keep_flat.dim() == 2, "keep_flat must be [rows, kv_blocks]");
    TORCH_CHECK(sel_blocks.dim() == 1, "sel_blocks must be [rows]");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 2, "pos must be [N, 2]");
    TORCH_CHECK(local_rank.dim() == 1, "local_rank must be [N]");
    TORCH_CHECK(keep_flat.size(0) == sel_blocks.size(0), "row count mismatch");
    return compact_keep_blocks_out_cuda(keep_flat, sel_blocks, pos, local_rank);
}

std::vector<torch::Tensor> compact_keep_blocks_and_build_table(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks,
    const torch::Tensor& pages_per_row,
    torch::Tensor& block_table,
    torch::Tensor& page_offsets) {
    TORCH_CHECK(keep_flat.is_cuda() && keep_flat.is_contiguous(), "keep_flat must be contiguous CUDA");
    TORCH_CHECK(sel_blocks.is_cuda() && sel_blocks.is_contiguous(), "sel_blocks must be contiguous CUDA");
    TORCH_CHECK(pages_per_row.is_cuda() && pages_per_row.is_contiguous(), "pages_per_row must be contiguous CUDA");
    TORCH_CHECK(block_table.is_cuda() && block_table.is_contiguous(), "block_table must be contiguous CUDA");
    TORCH_CHECK(page_offsets.is_cuda() && page_offsets.is_contiguous(), "page_offsets must be contiguous CUDA");
    TORCH_CHECK(keep_flat.scalar_type() == torch::kBool, "keep_flat must be bool");
    TORCH_CHECK(sel_blocks.scalar_type() == torch::kInt32, "sel_blocks must be int32");
    TORCH_CHECK(pages_per_row.scalar_type() == torch::kInt32, "pages_per_row must be int32");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(page_offsets.scalar_type() == torch::kInt32, "page_offsets must be int32");
    TORCH_CHECK(keep_flat.dim() == 2, "keep_flat must be [rows, kv_blocks]");
    TORCH_CHECK(sel_blocks.dim() == 1, "sel_blocks must be [rows]");
    TORCH_CHECK(pages_per_row.dim() == 1, "pages_per_row must be [rows]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [rows, max_pages]");
    TORCH_CHECK(page_offsets.dim() == 1, "page_offsets must be [rows]");
    TORCH_CHECK(keep_flat.size(0) == sel_blocks.size(0), "row count mismatch");
    TORCH_CHECK(keep_flat.size(0) == pages_per_row.size(0), "row count mismatch");
    TORCH_CHECK(block_table.size(0) == keep_flat.size(0), "block_table row count mismatch");
    TORCH_CHECK(page_offsets.size(0) == keep_flat.size(0), "page_offsets row count mismatch");
    return compact_keep_blocks_and_build_table_cuda(
        keep_flat,
        sel_blocks,
        pages_per_row,
        block_table,
        page_offsets);
}

int64_t compact_keep_blocks_and_build_table_out(
    const torch::Tensor& keep_flat,
    const torch::Tensor& sel_blocks,
    const torch::Tensor& pages_per_row,
    torch::Tensor& pos,
    torch::Tensor& local_rank,
    torch::Tensor& block_table,
    torch::Tensor& page_offsets) {
    TORCH_CHECK(keep_flat.is_cuda() && keep_flat.is_contiguous(), "keep_flat must be contiguous CUDA");
    TORCH_CHECK(sel_blocks.is_cuda() && sel_blocks.is_contiguous(), "sel_blocks must be contiguous CUDA");
    TORCH_CHECK(pages_per_row.is_cuda() && pages_per_row.is_contiguous(), "pages_per_row must be contiguous CUDA");
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous(), "pos must be contiguous CUDA");
    TORCH_CHECK(local_rank.is_cuda() && local_rank.is_contiguous(), "local_rank must be contiguous CUDA");
    TORCH_CHECK(block_table.is_cuda() && block_table.is_contiguous(), "block_table must be contiguous CUDA");
    TORCH_CHECK(page_offsets.is_cuda() && page_offsets.is_contiguous(), "page_offsets must be contiguous CUDA");
    TORCH_CHECK(keep_flat.scalar_type() == torch::kBool, "keep_flat must be bool");
    TORCH_CHECK(sel_blocks.scalar_type() == torch::kInt32, "sel_blocks must be int32");
    TORCH_CHECK(pages_per_row.scalar_type() == torch::kInt32, "pages_per_row must be int32");
    TORCH_CHECK(pos.scalar_type() == torch::kInt64, "pos must be int64");
    TORCH_CHECK(local_rank.scalar_type() == torch::kInt32, "local_rank must be int32");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(page_offsets.scalar_type() == torch::kInt32, "page_offsets must be int32");
    TORCH_CHECK(keep_flat.dim() == 2, "keep_flat must be [rows, kv_blocks]");
    TORCH_CHECK(sel_blocks.dim() == 1, "sel_blocks must be [rows]");
    TORCH_CHECK(pages_per_row.dim() == 1, "pages_per_row must be [rows]");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 2, "pos must be [N, 2]");
    TORCH_CHECK(local_rank.dim() == 1, "local_rank must be [N]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be [rows, max_pages]");
    TORCH_CHECK(page_offsets.dim() == 1, "page_offsets must be [rows]");
    TORCH_CHECK(keep_flat.size(0) == sel_blocks.size(0), "row count mismatch");
    TORCH_CHECK(keep_flat.size(0) == pages_per_row.size(0), "row count mismatch");
    TORCH_CHECK(block_table.size(0) == keep_flat.size(0), "block_table row count mismatch");
    TORCH_CHECK(page_offsets.size(0) == keep_flat.size(0), "page_offsets row count mismatch");
    return compact_keep_blocks_and_build_table_out_cuda(
        keep_flat,
        sel_blocks,
        pages_per_row,
        pos,
        local_rank,
        block_table,
        page_offsets);
}

torch::Tensor build_keep_past_fast(
    const torch::Tensor& attn_gate_score,
    double threshold,
    int64_t past_k_blocks) {
    TORCH_CHECK(attn_gate_score.is_cuda() && attn_gate_score.is_contiguous(), "attn_gate_score must be contiguous CUDA");
    TORCH_CHECK(attn_gate_score.dim() == 4, "attn_gate_score must be [B, H, Qb, Kb]");
    TORCH_CHECK(attn_gate_score.scalar_type() == torch::kFloat || attn_gate_score.scalar_type() == torch::kHalf ||
                    attn_gate_score.scalar_type() == torch::kBFloat16,
                "attn_gate_score must be float/half/bfloat16");
    TORCH_CHECK(past_k_blocks >= 0 && past_k_blocks <= attn_gate_score.size(3), "invalid past_k_blocks");
    return build_keep_past_fast_cuda(attn_gate_score, threshold, past_k_blocks);
}

torch::Tensor build_keep_curr_fast(
    const torch::Tensor& attn_gate_score,
    double threshold,
    int64_t past_k_blocks,
    int64_t curr_k_blocks) {
    TORCH_CHECK(attn_gate_score.is_cuda() && attn_gate_score.is_contiguous(), "attn_gate_score must be contiguous CUDA");
    TORCH_CHECK(attn_gate_score.dim() == 4, "attn_gate_score must be [B, H, Qb, Kb]");
    TORCH_CHECK(attn_gate_score.scalar_type() == torch::kFloat || attn_gate_score.scalar_type() == torch::kHalf ||
                    attn_gate_score.scalar_type() == torch::kBFloat16,
                "attn_gate_score must be float/half/bfloat16");
    TORCH_CHECK(past_k_blocks >= 0 && past_k_blocks <= attn_gate_score.size(3), "invalid past_k_blocks");
    TORCH_CHECK(curr_k_blocks >= 0 && past_k_blocks + curr_k_blocks <= attn_gate_score.size(3), "invalid curr_k_blocks");
    return build_keep_curr_fast_cuda(attn_gate_score, threshold, past_k_blocks, curr_k_blocks);
}

std::vector<torch::Tensor> build_past_indices_and_metadata_from_keep_block(
    const torch::Tensor& keep_block,
    int64_t num_key_value_groups,
    int64_t past_k_blocks,
    int64_t curr_k_blocks,
    int64_t block_size,
    int64_t page_block_size) {
    TORCH_CHECK(keep_block.is_cuda() && keep_block.is_contiguous(), "keep_block must be contiguous CUDA");
    TORCH_CHECK(keep_block.scalar_type() == torch::kBool, "keep_block must be bool");
    TORCH_CHECK(keep_block.dim() == 3, "keep_block must be [B, Hq, Kb]");
    TORCH_CHECK(num_key_value_groups > 0, "num_key_value_groups must be positive");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(page_block_size > 0, "page_block_size must be positive");
    TORCH_CHECK((page_block_size % block_size) == 0, "page_block_size must be divisible by block_size");
    TORCH_CHECK(past_k_blocks >= 0, "past_k_blocks must be non-negative");
    TORCH_CHECK(curr_k_blocks >= 0, "curr_k_blocks must be non-negative");
    TORCH_CHECK(
        past_k_blocks + curr_k_blocks <= keep_block.size(2),
        "past_k_blocks + curr_k_blocks exceeds keep_block width");
    TORCH_CHECK(
        (keep_block.size(1) % num_key_value_groups) == 0,
        "num_query_heads must be divisible by num_key_value_groups");

    const auto rows = keep_block.size(0) * (keep_block.size(1) / num_key_value_groups);
    auto meta_opts = keep_block.options().dtype(torch::kInt32);
    auto past_block_indices = torch::full({rows, past_k_blocks}, -1, meta_opts);
    auto past_block_counts = torch::zeros({rows}, meta_opts);
    auto pages_per_row = torch::zeros({rows}, meta_opts);
    auto cache_seqlens = torch::zeros({rows}, meta_opts);

    build_past_indices_and_metadata_from_keep_block_cuda(
        keep_block,
        num_key_value_groups,
        past_k_blocks,
        curr_k_blocks,
        block_size,
        page_block_size,
        past_block_indices,
        past_block_counts,
        pages_per_row,
        cache_seqlens);
    return {past_block_indices, past_block_counts, pages_per_row, cache_seqlens};
}

void build_past_indices_and_metadata_from_keep_block_out(
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
    TORCH_CHECK(keep_block.is_cuda() && keep_block.is_contiguous(), "keep_block must be contiguous CUDA");
    TORCH_CHECK(keep_block.scalar_type() == torch::kBool, "keep_block must be bool");
    TORCH_CHECK(keep_block.dim() == 3, "keep_block must be [B, Hq, Kb]");
    TORCH_CHECK(num_key_value_groups > 0, "num_key_value_groups must be positive");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(page_block_size > 0, "page_block_size must be positive");
    TORCH_CHECK((page_block_size % block_size) == 0, "page_block_size must be divisible by block_size");
    TORCH_CHECK(past_k_blocks >= 0, "past_k_blocks must be non-negative");
    TORCH_CHECK(curr_k_blocks >= 0, "curr_k_blocks must be non-negative");
    TORCH_CHECK(
        past_k_blocks + curr_k_blocks <= keep_block.size(2),
        "past_k_blocks + curr_k_blocks exceeds keep_block width");
    TORCH_CHECK(
        (keep_block.size(1) % num_key_value_groups) == 0,
        "num_query_heads must be divisible by num_key_value_groups");

    const auto rows = keep_block.size(0) * (keep_block.size(1) / num_key_value_groups);
    TORCH_CHECK(
        past_block_indices.is_cuda() && past_block_indices.is_contiguous(),
        "past_block_indices must be contiguous CUDA");
    TORCH_CHECK(
        past_block_counts.is_cuda() && past_block_counts.is_contiguous(),
        "past_block_counts must be contiguous CUDA");
    TORCH_CHECK(
        pages_per_row.is_cuda() && pages_per_row.is_contiguous(),
        "pages_per_row must be contiguous CUDA");
    TORCH_CHECK(
        cache_seqlens.is_cuda() && cache_seqlens.is_contiguous(),
        "cache_seqlens must be contiguous CUDA");
    TORCH_CHECK(past_block_indices.scalar_type() == torch::kInt32, "past_block_indices must be int32");
    TORCH_CHECK(past_block_counts.scalar_type() == torch::kInt32, "past_block_counts must be int32");
    TORCH_CHECK(pages_per_row.scalar_type() == torch::kInt32, "pages_per_row must be int32");
    TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32, "cache_seqlens must be int32");
    TORCH_CHECK(past_block_indices.dim() == 2, "past_block_indices must be [rows, capacity]");
    TORCH_CHECK(past_block_counts.dim() == 1, "past_block_counts must be [rows]");
    TORCH_CHECK(pages_per_row.dim() == 1, "pages_per_row must be [rows]");
    TORCH_CHECK(cache_seqlens.dim() == 1, "cache_seqlens must be [rows]");
    TORCH_CHECK(past_block_indices.size(0) == rows, "past_block_indices row count mismatch");
    TORCH_CHECK(past_block_indices.size(1) >= past_k_blocks, "past_block_indices capacity too small");
    TORCH_CHECK(past_block_counts.size(0) == rows, "past_block_counts row count mismatch");
    TORCH_CHECK(pages_per_row.size(0) == rows, "pages_per_row row count mismatch");
    TORCH_CHECK(cache_seqlens.size(0) == rows, "cache_seqlens row count mismatch");

    build_past_indices_and_metadata_from_keep_block_cuda(
        keep_block,
        num_key_value_groups,
        past_k_blocks,
        curr_k_blocks,
        block_size,
        page_block_size,
        past_block_indices,
        past_block_counts,
        pages_per_row,
        cache_seqlens);
}

void build_selected_indices_and_metadata_from_keep_block_out(
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
    TORCH_CHECK(keep_block.is_cuda() && keep_block.is_contiguous(), "keep_block must be contiguous CUDA");
    TORCH_CHECK(keep_block.scalar_type() == torch::kBool, "keep_block must be bool");
    TORCH_CHECK(keep_block.dim() == 3, "keep_block must be [B, Hq, Kb]");
    TORCH_CHECK(num_key_value_groups > 0, "num_key_value_groups must be positive");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(page_block_size > 0, "page_block_size must be positive");
    TORCH_CHECK((page_block_size % block_size) == 0, "page_block_size must be divisible by block_size");
    TORCH_CHECK(past_k_blocks >= 0, "past_k_blocks must be non-negative");
    TORCH_CHECK(curr_k_blocks >= 0, "curr_k_blocks must be non-negative");
    TORCH_CHECK(
        past_k_blocks + curr_k_blocks <= keep_block.size(2),
        "past_k_blocks + curr_k_blocks exceeds keep_block width");
    TORCH_CHECK(
        (keep_block.size(1) % num_key_value_groups) == 0,
        "num_query_heads must be divisible by num_key_value_groups");

    const auto rows = keep_block.size(0) * (keep_block.size(1) / num_key_value_groups);
    TORCH_CHECK(
        selected_block_indices.is_cuda() && selected_block_indices.is_contiguous(),
        "selected_block_indices must be contiguous CUDA");
    TORCH_CHECK(
        selected_block_counts.is_cuda() && selected_block_counts.is_contiguous(),
        "selected_block_counts must be contiguous CUDA");
    TORCH_CHECK(
        pages_per_row.is_cuda() && pages_per_row.is_contiguous(),
        "pages_per_row must be contiguous CUDA");
    TORCH_CHECK(
        cache_seqlens.is_cuda() && cache_seqlens.is_contiguous(),
        "cache_seqlens must be contiguous CUDA");
    TORCH_CHECK(selected_block_indices.scalar_type() == torch::kInt32, "selected_block_indices must be int32");
    TORCH_CHECK(selected_block_counts.scalar_type() == torch::kInt32, "selected_block_counts must be int32");
    TORCH_CHECK(pages_per_row.scalar_type() == torch::kInt32, "pages_per_row must be int32");
    TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32, "cache_seqlens must be int32");
    TORCH_CHECK(selected_block_indices.dim() == 2, "selected_block_indices must be [rows, capacity]");
    TORCH_CHECK(selected_block_counts.dim() == 1, "selected_block_counts must be [rows]");
    TORCH_CHECK(pages_per_row.dim() == 1, "pages_per_row must be [rows]");
    TORCH_CHECK(cache_seqlens.dim() == 1, "cache_seqlens must be [rows]");
    TORCH_CHECK(selected_block_indices.size(0) == rows, "selected_block_indices row count mismatch");
    TORCH_CHECK(
        selected_block_indices.size(1) >= past_k_blocks + curr_k_blocks,
        "selected_block_indices capacity too small");
    TORCH_CHECK(selected_block_counts.size(0) == rows, "selected_block_counts row count mismatch");
    TORCH_CHECK(pages_per_row.size(0) == rows, "pages_per_row row count mismatch");
    TORCH_CHECK(cache_seqlens.size(0) == rows, "cache_seqlens row count mismatch");

    build_selected_indices_and_metadata_from_keep_block_cuda(
        keep_block,
        num_key_value_groups,
        past_k_blocks,
        curr_k_blocks,
        block_size,
        page_block_size,
        selected_block_indices,
        selected_block_counts,
        pages_per_row,
        cache_seqlens);
}

void build_selected_indices_from_kv_keep_block_out(
    const torch::Tensor& keep_block_kv,
    torch::Tensor& selected_block_indices,
    torch::Tensor& selected_block_counts) {
    TORCH_CHECK(
        keep_block_kv.is_cuda() && keep_block_kv.is_contiguous(),
        "keep_block_kv must be contiguous CUDA");
    TORCH_CHECK(keep_block_kv.scalar_type() == torch::kBool, "keep_block_kv must be bool");
    TORCH_CHECK(keep_block_kv.dim() == 3, "keep_block_kv must be [B, Hkv, Kb]");

    const auto rows = keep_block_kv.size(0) * keep_block_kv.size(1);
    TORCH_CHECK(
        selected_block_indices.is_cuda() && selected_block_indices.is_contiguous(),
        "selected_block_indices must be contiguous CUDA");
    TORCH_CHECK(
        selected_block_counts.is_cuda() && selected_block_counts.is_contiguous(),
        "selected_block_counts must be contiguous CUDA");
    TORCH_CHECK(selected_block_indices.scalar_type() == torch::kInt32, "selected_block_indices must be int32");
    TORCH_CHECK(selected_block_counts.scalar_type() == torch::kInt32, "selected_block_counts must be int32");
    TORCH_CHECK(selected_block_indices.dim() == 2, "selected_block_indices must be [rows, capacity]");
    TORCH_CHECK(selected_block_counts.dim() == 1, "selected_block_counts must be [rows]");
    TORCH_CHECK(selected_block_indices.size(0) == rows, "selected_block_indices row count mismatch");
    TORCH_CHECK(
        selected_block_indices.size(1) >= keep_block_kv.size(2),
        "selected_block_indices capacity too small");
    TORCH_CHECK(selected_block_counts.size(0) == rows, "selected_block_counts row count mismatch");

    build_selected_indices_from_kv_keep_block_cuda(
        keep_block_kv,
        selected_block_indices,
        selected_block_counts);
}

void build_flashinfer_kv_indices(
    const torch::Tensor& selected_block_indices,
    const torch::Tensor& selected_block_counts,
    const torch::Tensor& kv_indptr,
    torch::Tensor& kv_indices,
    int64_t kv_blocks) {
    TORCH_CHECK(
        selected_block_indices.is_cuda() && selected_block_indices.is_contiguous(),
        "selected_block_indices must be contiguous CUDA");
    TORCH_CHECK(
        selected_block_counts.is_cuda() && selected_block_counts.is_contiguous(),
        "selected_block_counts must be contiguous CUDA");
    TORCH_CHECK(kv_indptr.is_cuda() && kv_indptr.is_contiguous(), "kv_indptr must be contiguous CUDA");
    TORCH_CHECK(kv_indices.is_cuda() && kv_indices.is_contiguous(), "kv_indices must be contiguous CUDA");
    TORCH_CHECK(selected_block_indices.scalar_type() == torch::kInt32, "selected_block_indices must be int32");
    TORCH_CHECK(selected_block_counts.scalar_type() == torch::kInt32, "selected_block_counts must be int32");
    TORCH_CHECK(kv_indptr.scalar_type() == torch::kInt32, "kv_indptr must be int32");
    TORCH_CHECK(kv_indices.scalar_type() == torch::kInt32, "kv_indices must be int32");
    TORCH_CHECK(selected_block_indices.dim() == 2, "selected_block_indices must be [rows, capacity]");
    TORCH_CHECK(selected_block_counts.dim() == 1, "selected_block_counts must be [rows]");
    TORCH_CHECK(kv_indptr.dim() == 1, "kv_indptr must be [rows + 1]");
    TORCH_CHECK(kv_indices.dim() == 1, "kv_indices must be [capacity]");
    TORCH_CHECK(selected_block_indices.size(0) == selected_block_counts.size(0), "row count mismatch");
    TORCH_CHECK(kv_indptr.size(0) == selected_block_counts.size(0) + 1, "kv_indptr row count mismatch");
    TORCH_CHECK(kv_blocks > 0, "kv_blocks must be positive");

    build_flashinfer_kv_indices_cuda(
        selected_block_indices,
        selected_block_counts,
        kv_indptr,
        kv_indices,
        kv_blocks);
}

void build_flashinfer_kv_indices_per_query(
    const torch::Tensor& selected_block_indices,
    const torch::Tensor& selected_block_counts,
    const torch::Tensor& kv_indptr,
    torch::Tensor& kv_indices,
    int64_t kv_blocks,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t num_key_value_groups) {
    TORCH_CHECK(
        selected_block_indices.is_cuda() && selected_block_indices.is_contiguous(),
        "selected_block_indices must be contiguous CUDA");
    TORCH_CHECK(
        selected_block_counts.is_cuda() && selected_block_counts.is_contiguous(),
        "selected_block_counts must be contiguous CUDA");
    TORCH_CHECK(kv_indptr.is_cuda() && kv_indptr.is_contiguous(), "kv_indptr must be contiguous CUDA");
    TORCH_CHECK(kv_indices.is_cuda() && kv_indices.is_contiguous(), "kv_indices must be contiguous CUDA");
    TORCH_CHECK(selected_block_indices.scalar_type() == torch::kInt32, "selected_block_indices must be int32");
    TORCH_CHECK(selected_block_counts.scalar_type() == torch::kInt32, "selected_block_counts must be int32");
    TORCH_CHECK(kv_indptr.scalar_type() == torch::kInt32, "kv_indptr must be int32");
    TORCH_CHECK(kv_indices.scalar_type() == torch::kInt32, "kv_indices must be int32");
    TORCH_CHECK(selected_block_indices.dim() == 2, "selected_block_indices must be [rows, capacity]");
    TORCH_CHECK(selected_block_counts.dim() == 1, "selected_block_counts must be [rows]");
    TORCH_CHECK(kv_indptr.dim() == 1, "kv_indptr must be [rows + 1]");
    TORCH_CHECK(kv_indices.dim() == 1, "kv_indices must be [capacity]");
    TORCH_CHECK(selected_block_indices.size(0) == selected_block_counts.size(0), "row count mismatch");
    TORCH_CHECK(kv_indptr.size(0) == selected_block_counts.size(0) + 1, "kv_indptr row count mismatch");
    TORCH_CHECK(kv_blocks > 0, "kv_blocks must be positive");
    TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && num_key_value_groups > 0, "head counts must be positive");
    TORCH_CHECK(num_q_heads == num_kv_heads * num_key_value_groups, "Hq must equal Hkv * G");
    TORCH_CHECK(
        selected_block_counts.size(0) % num_q_heads == 0,
        "row count must be divisible by num_q_heads");

    build_flashinfer_kv_indices_per_query_cuda(
        selected_block_indices,
        selected_block_counts,
        kv_indptr,
        kv_indices,
        kv_blocks,
        num_q_heads,
        num_kv_heads,
        num_key_value_groups);
}

void cache_fill_from_past_indices(
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
    check_common_inputs(k, v, k_cache_flat, v_cache_flat);
    TORCH_CHECK(past_block_indices.is_cuda() && past_block_indices.is_contiguous(), "past_block_indices must be contiguous CUDA");
    TORCH_CHECK(past_block_counts.is_cuda() && past_block_counts.is_contiguous(), "past_block_counts must be contiguous CUDA");
    TORCH_CHECK(page_offsets.is_cuda() && page_offsets.is_contiguous(), "page_offsets must be contiguous CUDA");
    TORCH_CHECK(past_block_indices.scalar_type() == torch::kInt32, "past_block_indices must be int32");
    TORCH_CHECK(past_block_counts.scalar_type() == torch::kInt32, "past_block_counts must be int32");
    TORCH_CHECK(page_offsets.scalar_type() == torch::kInt32, "page_offsets must be int32");
    TORCH_CHECK(past_block_indices.dim() == 2, "past_block_indices must be [rows, past_k_blocks]");
    TORCH_CHECK(past_block_counts.dim() == 1, "past_block_counts must be [rows]");
    TORCH_CHECK(page_offsets.dim() == 1, "page_offsets must be [rows]");
    TORCH_CHECK(past_block_indices.size(0) == past_block_counts.size(0), "row count mismatch");
    TORCH_CHECK(page_offsets.size(0) == past_block_counts.size(0), "row count mismatch");
    TORCH_CHECK(active_past_k_blocks >= 0, "active_past_k_blocks must be non-negative");
    TORCH_CHECK(
        past_block_indices.size(1) >= active_past_k_blocks,
        "past_block_indices capacity smaller than active_past_k_blocks");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(page_block_size > 0, "page_block_size must be positive");
    TORCH_CHECK((page_block_size % block_size) == 0, "page_block_size must be divisible by block_size");

    cache_fill_from_past_indices_cuda(
        k,
        v,
        past_block_indices,
        past_block_counts,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        active_past_k_blocks,
        block_size,
        page_block_size);
}

void cache_fill_from_past_indices_compact(
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
    check_common_inputs(k, v, k_cache_flat, v_cache_flat);
    TORCH_CHECK(past_block_indices.is_cuda() && past_block_indices.is_contiguous(), "past_block_indices must be contiguous CUDA");
    TORCH_CHECK(past_block_counts.is_cuda() && past_block_counts.is_contiguous(), "past_block_counts must be contiguous CUDA");
    TORCH_CHECK(selected_offsets.is_cuda() && selected_offsets.is_contiguous(), "selected_offsets must be contiguous CUDA");
    TORCH_CHECK(page_offsets.is_cuda() && page_offsets.is_contiguous(), "page_offsets must be contiguous CUDA");
    TORCH_CHECK(past_block_indices.scalar_type() == torch::kInt32, "past_block_indices must be int32");
    TORCH_CHECK(past_block_counts.scalar_type() == torch::kInt32, "past_block_counts must be int32");
    TORCH_CHECK(selected_offsets.scalar_type() == torch::kInt32, "selected_offsets must be int32");
    TORCH_CHECK(page_offsets.scalar_type() == torch::kInt32, "page_offsets must be int32");
    TORCH_CHECK(past_block_indices.dim() == 2, "past_block_indices must be [rows, past_k_blocks]");
    TORCH_CHECK(past_block_counts.dim() == 1, "past_block_counts must be [rows]");
    TORCH_CHECK(selected_offsets.dim() == 1, "selected_offsets must be [rows]");
    TORCH_CHECK(page_offsets.dim() == 1, "page_offsets must be [rows]");
    TORCH_CHECK(past_block_indices.size(0) == past_block_counts.size(0), "row count mismatch");
    TORCH_CHECK(selected_offsets.size(0) == past_block_counts.size(0), "row count mismatch");
    TORCH_CHECK(page_offsets.size(0) == past_block_counts.size(0), "row count mismatch");
    TORCH_CHECK(total_selected >= 0, "total_selected must be non-negative");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(page_block_size > 0, "page_block_size must be positive");
    TORCH_CHECK((page_block_size % block_size) == 0, "page_block_size must be divisible by block_size");

    cache_fill_from_past_indices_compact_cuda(
        k,
        v,
        past_block_indices,
        past_block_counts,
        selected_offsets,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        total_selected,
        block_size,
        page_block_size);
}

void cache_fill_from_selected_indices_row_tiled(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& selected_block_indices,
    const torch::Tensor& selected_block_counts,
    const torch::Tensor& page_offsets,
    torch::Tensor& k_cache_flat,
    torch::Tensor& v_cache_flat,
    int64_t block_size,
    int64_t page_block_size) {
    TORCH_CHECK(k.is_cuda() && v.is_cuda(), "k/v must be CUDA");
    TORCH_CHECK(
        selected_block_indices.is_cuda() && selected_block_counts.is_cuda() && page_offsets.is_cuda(),
        "selected_block_indices/selected_block_counts/page_offsets must be CUDA");
    TORCH_CHECK(k_cache_flat.is_cuda() && v_cache_flat.is_cuda(), "k_cache_flat/v_cache_flat must be CUDA");
    TORCH_CHECK(k.is_contiguous() && v.is_contiguous(), "k/v must be contiguous");
    TORCH_CHECK(
        selected_block_indices.is_contiguous() && selected_block_counts.is_contiguous() && page_offsets.is_contiguous(),
        "selected_block_indices/selected_block_counts/page_offsets must be contiguous");
    TORCH_CHECK(k_cache_flat.is_contiguous() && v_cache_flat.is_contiguous(), "k_cache_flat/v_cache_flat must be contiguous");
    TORCH_CHECK(k.scalar_type() == torch::kFloat16 || k.scalar_type() == torch::kBFloat16, "k must be fp16/bf16");
    TORCH_CHECK(v.scalar_type() == k.scalar_type(), "v dtype must match k");
    TORCH_CHECK(k_cache_flat.scalar_type() == k.scalar_type(), "k_cache_flat dtype must match k");
    TORCH_CHECK(v_cache_flat.scalar_type() == k.scalar_type(), "v_cache_flat dtype must match k");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be [B, K, Hkv, D]");
    TORCH_CHECK(k.sizes() == v.sizes(), "k/v shape mismatch");
    TORCH_CHECK(k.size(3) == 128, "head_dim must be 128");
    TORCH_CHECK(selected_block_indices.dim() == 2, "selected_block_indices must be [rows, stride]");
    TORCH_CHECK(selected_block_counts.dim() == 1, "selected_block_counts must be [rows]");
    TORCH_CHECK(page_offsets.dim() == 1, "page_offsets must be [rows]");
    TORCH_CHECK(selected_block_indices.scalar_type() == torch::kInt32, "selected_block_indices must be int32");
    TORCH_CHECK(selected_block_counts.scalar_type() == torch::kInt32, "selected_block_counts must be int32");
    TORCH_CHECK(page_offsets.scalar_type() == torch::kInt32, "page_offsets must be int32");
    TORCH_CHECK(selected_block_indices.size(0) == selected_block_counts.size(0), "row count mismatch");
    TORCH_CHECK(selected_block_indices.size(0) == page_offsets.size(0), "page_offsets row count mismatch");
    TORCH_CHECK(k_cache_flat.dim() == 2 && v_cache_flat.dim() == 2, "k_cache_flat/v_cache_flat must be [N, D]");
    TORCH_CHECK(k_cache_flat.size(1) == 128 && v_cache_flat.size(1) == 128, "cache head_dim must be 128");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(page_block_size > 0 && (page_block_size % block_size) == 0, "invalid page_block_size");

    cache_fill_from_selected_indices_row_tiled_cuda(
        k,
        v,
        selected_block_indices,
        selected_block_counts,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        block_size,
        page_block_size);
}

void cache_fill_current_tail(
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
    check_common_inputs(k, v, k_cache_flat, v_cache_flat);
    TORCH_CHECK(past_block_counts.is_cuda() && past_block_counts.is_contiguous(), "past_block_counts must be contiguous CUDA");
    TORCH_CHECK(page_offsets.is_cuda() && page_offsets.is_contiguous(), "page_offsets must be contiguous CUDA");
    TORCH_CHECK(past_block_counts.scalar_type() == torch::kInt32, "past_block_counts must be int32");
    TORCH_CHECK(page_offsets.scalar_type() == torch::kInt32, "page_offsets must be int32");
    TORCH_CHECK(past_block_counts.dim() == 1, "past_block_counts must be [rows]");
    TORCH_CHECK(page_offsets.dim() == 1, "page_offsets must be [rows]");
    TORCH_CHECK(page_offsets.size(0) == past_block_counts.size(0), "row count mismatch");
    TORCH_CHECK(past_k_blocks >= 0, "past_k_blocks must be non-negative");
    TORCH_CHECK(curr_k_blocks >= 0, "curr_k_blocks must be non-negative");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(page_block_size > 0, "page_block_size must be positive");
    TORCH_CHECK((page_block_size % block_size) == 0, "page_block_size must be divisible by block_size");

    cache_fill_current_tail_cuda(
        k,
        v,
        past_block_counts,
        page_offsets,
        k_cache_flat,
        v_cache_flat,
        past_k_blocks,
        curr_k_blocks,
        block_size,
        page_block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cache_fill_from_pos_rank", &cache_fill_from_pos_rank, "cache_fill_from_pos_rank");
    m.def("cache_fill_from_row_blk_dst", &cache_fill_from_row_blk_dst, "cache_fill_from_row_blk_dst");
    m.def("cache_fill_from_pos_local_rank", &cache_fill_from_pos_local_rank, "cache_fill_from_pos_local_rank");
    m.def("pack_q_for_indexed_prefill", &pack_q_for_indexed_prefill, "pack_q_for_indexed_prefill");
    m.def("build_block_table", &build_block_table, "build_block_table");
    m.def("compact_keep_blocks", &compact_keep_blocks, "compact_keep_blocks");
    m.def("compact_keep_blocks_out", &compact_keep_blocks_out, "compact_keep_blocks_out");
    m.def(
        "compact_keep_blocks_and_build_table",
        &compact_keep_blocks_and_build_table,
        "compact_keep_blocks_and_build_table");
    m.def(
        "compact_keep_blocks_and_build_table_out",
        &compact_keep_blocks_and_build_table_out,
        "compact_keep_blocks_and_build_table_out");
    m.def("build_keep_past_fast", &build_keep_past_fast, "build_keep_past_fast");
    m.def("build_keep_curr_fast", &build_keep_curr_fast, "build_keep_curr_fast");
    m.def(
        "build_past_indices_and_metadata_from_keep_block",
        &build_past_indices_and_metadata_from_keep_block,
        "build_past_indices_and_metadata_from_keep_block");
    m.def(
        "build_past_indices_and_metadata_from_keep_block_out",
        &build_past_indices_and_metadata_from_keep_block_out,
        "build_past_indices_and_metadata_from_keep_block_out");
    m.def(
        "build_selected_indices_and_metadata_from_keep_block_out",
        &build_selected_indices_and_metadata_from_keep_block_out,
        "build_selected_indices_and_metadata_from_keep_block_out");
    m.def(
        "build_selected_indices_from_kv_keep_block_out",
        &build_selected_indices_from_kv_keep_block_out,
        "build_selected_indices_from_kv_keep_block_out");
    m.def(
        "build_flashinfer_kv_indices",
        &build_flashinfer_kv_indices,
        "build_flashinfer_kv_indices");
    m.def(
        "build_flashinfer_kv_indices_per_query",
        &build_flashinfer_kv_indices_per_query,
        "build_flashinfer_kv_indices_per_query");
    m.def("cache_fill_from_past_indices", &cache_fill_from_past_indices, "cache_fill_from_past_indices");
    m.def(
        "cache_fill_from_past_indices_compact",
        &cache_fill_from_past_indices_compact,
        "cache_fill_from_past_indices_compact");
    m.def(
        "cache_fill_from_selected_indices_row_tiled",
        &cache_fill_from_selected_indices_row_tiled,
        "cache_fill_from_selected_indices_row_tiled");
    m.def("cache_fill_current_tail", &cache_fill_current_tail, "cache_fill_current_tail");
}
