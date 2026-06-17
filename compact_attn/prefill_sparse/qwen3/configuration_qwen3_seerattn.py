# coding=utf-8
"""Qwen3 configuration extended with SeerAttention prefill training fields."""

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class SeerAttnQwen3Config(Qwen3Config):
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        use_flash_rope=False,
        seerattn_sparsity_method="threshold",
        seerattn_threshold=0.0,
        seerattn_nz_ratio=1.0,
        seerattn_gate_type="Qavg_Kmaxminavg",
        seerattn_gate_block_size=64,
        seerattn_gate_hidden_size=128,
        seerattn_last_block_dense=True,
        seerattn_gate_force_double=False,
        seerattn_compactattn_kv_group_aware_gate=False,
        seerattn_compactattn_adjacent_align_lambda=1e-3,
        seerattn_compactattn_reinit_q_branch_on_load=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_flash_rope = use_flash_rope
        self.seerattn_sparsity_method = seerattn_sparsity_method
        self.seerattn_threshold = seerattn_threshold
        self.seerattn_nz_ratio = seerattn_nz_ratio
        self.seerattn_gate_type = seerattn_gate_type
        self.seerattn_gate_block_size = seerattn_gate_block_size
        self.seerattn_gate_hidden_size = seerattn_gate_hidden_size
        self.seerattn_last_block_dense = seerattn_last_block_dense
        self.seerattn_gate_force_double = seerattn_gate_force_double
        self.seerattn_compactattn_kv_group_aware_gate = seerattn_compactattn_kv_group_aware_gate
        self.seerattn_compactattn_adjacent_align_lambda = seerattn_compactattn_adjacent_align_lambda
        self.seerattn_compactattn_reinit_q_branch_on_load = seerattn_compactattn_reinit_q_branch_on_load

        assert self.seerattn_sparsity_method in ["threshold", "nz_ratio"]


__all__ = ["SeerAttnQwen3Config"]
