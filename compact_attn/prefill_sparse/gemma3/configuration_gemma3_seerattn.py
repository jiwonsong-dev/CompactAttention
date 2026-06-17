# coding=utf-8
"""Gemma3 text configuration extended with SeerAttention training fields."""

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


class SeerAttnGemma3Config(Gemma3TextConfig):
    model_type = "gemma3_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        seerattn_sparsity_method="threshold",
        seerattn_threshold=0.0,
        seerattn_nz_ratio=1.0,
        seerattn_gate_type="Qavg_Kmaxminavg",
        seerattn_gate_block_size=64,
        seerattn_gate_hidden_size=128,
        seerattn_last_block_dense=True,
        seerattn_gate_force_double=False,
        seerattn_use_chunked_gate_cache=True,
        seerattn_chunked_prefill_force_dense=False,
        seerattn_chunked_prefill_final_dense_tail_blocks=0,
        seerattn_dense_backend="flashinfer",
        seerattn_gemma3_execution_mode="block_sparse",
        seerattn_compactattn_kv_group_aware_gate=False,
        seerattn_compactattn_gate_layout=None,
        seerattn_compactattn_threshold=5e-4,
        seerattn_compactattn_threshold_schedule=None,
        seerattn_compactattn_keep_recent_blocks=2,
        seerattn_compactattn_disable_first_chunk_dense=False,
        seerattn_compactattn_chunked_gate_head_pool="none",
        seerattn_compactattn_pack_impl="indexed_dense",
        seerattn_compactattn_indexed_impl="fi_zero_copy",
        seerattn_compactattn_cache_fill_backend="auto",
        seerattn_compactattn_adjacent_align_lambda=1e-3,
        seerattn_compactattn_reinit_q_branch_on_load=False,
        fused_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.seerattn_sparsity_method = seerattn_sparsity_method
        self.seerattn_threshold = seerattn_threshold
        self.seerattn_nz_ratio = seerattn_nz_ratio
        self.seerattn_gate_type = seerattn_gate_type
        self.seerattn_gate_block_size = seerattn_gate_block_size
        self.seerattn_gate_hidden_size = seerattn_gate_hidden_size
        self.seerattn_last_block_dense = seerattn_last_block_dense
        self.seerattn_gate_force_double = seerattn_gate_force_double
        self.seerattn_use_chunked_gate_cache = seerattn_use_chunked_gate_cache
        self.seerattn_chunked_prefill_force_dense = seerattn_chunked_prefill_force_dense
        self.seerattn_chunked_prefill_final_dense_tail_blocks = seerattn_chunked_prefill_final_dense_tail_blocks
        self.seerattn_dense_backend = seerattn_dense_backend
        self.seerattn_gemma3_execution_mode = seerattn_gemma3_execution_mode
        self.seerattn_compactattn_kv_group_aware_gate = seerattn_compactattn_kv_group_aware_gate
        self.seerattn_compactattn_gate_layout = (
            "gqa_aware"
            if seerattn_compactattn_gate_layout is None and self.seerattn_compactattn_kv_group_aware_gate
            else "old_union_qhead"
            if seerattn_compactattn_gate_layout is None
            else str(seerattn_compactattn_gate_layout)
        )
        self.seerattn_compactattn_threshold = seerattn_compactattn_threshold
        self.seerattn_compactattn_threshold_schedule = seerattn_compactattn_threshold_schedule
        self.seerattn_compactattn_keep_recent_blocks = seerattn_compactattn_keep_recent_blocks
        self.seerattn_compactattn_disable_first_chunk_dense = seerattn_compactattn_disable_first_chunk_dense
        self.seerattn_compactattn_chunked_gate_head_pool = seerattn_compactattn_chunked_gate_head_pool
        self.seerattn_compactattn_pack_impl = seerattn_compactattn_pack_impl
        self.seerattn_compactattn_indexed_impl = seerattn_compactattn_indexed_impl
        self.seerattn_compactattn_cache_fill_backend = seerattn_compactattn_cache_fill_backend
        self.seerattn_compactattn_adjacent_align_lambda = seerattn_compactattn_adjacent_align_lambda
        self.seerattn_compactattn_reinit_q_branch_on_load = seerattn_compactattn_reinit_q_branch_on_load
        self.fused_norm = fused_norm

        assert self.seerattn_sparsity_method in ["threshold", "nz_ratio"]


__all__ = ["SeerAttnGemma3Config"]
