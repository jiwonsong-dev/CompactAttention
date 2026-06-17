# coding=utf-8
from typing import Optional, Tuple, Union, List

import copy
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.utils.deprecation import deprecate_kwarg

from compact_attn.modules.attention_distill import attention_distill_forward
from compact_attn.modules.attention_forward import sparse_flash_attention_forward
from compact_attn.modules.dense_prefill import dense_prefill_full_kv
from compact_attn.prefill_sparse.attn_gate import ATTNGATE_CLASSES, MultiHeadLinear
from compact_attn.prefill_sparse.qwen.modeling_qwen2_seerattn import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    SeerAttnQwen2ForCausalLM,
    SeerAttnQwen2Model,
    SeerAttnQwen2PreTrainedModel,
    _resolve_qwen_seer_device_map,
)
from compact_attn.utils import BaseModelOutputWithPastAndSeer, CausalLMOutputWithPastAndSeer
from compact_attn.modules.common import apply_rotary_pos_emb
from einops import rearrange
from flash_attn.layers.rotary import apply_rotary_emb_func
from huggingface_hub import hf_hub_download

from .configuration_qwen3_seerattn import SeerAttnQwen3Config


def _sync_qwen3_base_vocab_size(config: SeerAttnQwen3Config, base_model: str, kwargs) -> None:
    config_kwargs = {}
    for key in ("cache_dir", "force_download", "local_files_only", "revision", "token", "trust_remote_code"):
        if key in kwargs:
            config_kwargs[key] = kwargs[key]
    base_config = AutoConfig.from_pretrained(base_model, **config_kwargs)
    config.vocab_size = base_config.vocab_size


class Qwen3RMSNorm(Qwen2RMSNorm):
    pass


class Qwen3RotaryEmbedding(Qwen2RotaryEmbedding):
    pass


class Qwen3MLP(Qwen2MLP):
    pass


class SeerAttnQwen3Attention(nn.Module):
    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_gate = ATTNGATE_CLASSES[config.seerattn_gate_type](
            config.seerattn_gate_block_size,
            self.head_dim,
            config.seerattn_gate_hidden_size,
            num_k_head=config.num_key_value_heads,
            num_q_head=config.num_attention_heads,
            force_double=config.seerattn_gate_force_double,
            use_flash_rope=config.use_flash_rope,
            kv_group_aware_query=bool(
                getattr(config, "seerattn_compactattn_kv_group_aware_gate", False)
            ),
        )

        self.mask_loss_func = torch.nn.KLDivLoss()
        self.profile_file = os.environ.get("PROFILE_FILE", None)

    def _chunked_gate_cache_store(self, past_key_value: Optional[Cache]):
        if past_key_value is None:
            return None
        store = getattr(past_key_value, "_seer_chunked_gate_k_cache", None)
        if store is None:
            store = {}
            setattr(past_key_value, "_seer_chunked_gate_k_cache", store)
        return store

    def _can_use_chunked_gate_cache(
        self,
        key_states_nope: torch.Tensor,
        cache_position: Optional[torch.LongTensor],
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> bool:
        if cache_position is None or block_position_embeddings is None:
            return False
        block_size = int(self.config.seerattn_gate_block_size)
        if block_size <= 0:
            return False
        chunk_start = int(cache_position[0].item())
        q_len = int(key_states_nope.shape[1])
        if (chunk_start % block_size) != 0:
            return False
        if q_len < block_size:
            return False
        return True

    def _build_chunked_gate_blocks(
        self,
        query_states_nope: torch.Tensor,
        key_states_nope: torch.Tensor,
        block_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_blocks = self.attn_gate.compress_query_blocks(query_states_nope)
        k_blocks = self.attn_gate.compress_key_blocks(key_states_nope)
        q_blocks, k_blocks = self.attn_gate.apply_block_position_embeddings(
            q=q_blocks,
            k=k_blocks,
            position_embeddings=block_position_embeddings,
        )
        return q_blocks, k_blocks

    def _append_chunked_gate_key_cache(
        self,
        key_states_nope: torch.Tensor,
        past_key_value: Optional[Cache],
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache_position: Optional[torch.LongTensor],
    ) -> bool:
        if not self._can_use_chunked_gate_cache(
            key_states_nope=key_states_nope,
            cache_position=cache_position,
            block_position_embeddings=block_position_embeddings,
        ):
            return False
        store = self._chunked_gate_cache_store(past_key_value)
        if store is None:
            return False

        k_blocks = self.attn_gate.compress_key_blocks(key_states_nope)
        _, k_blocks = self.attn_gate.apply_block_position_embeddings(
            q=None,
            k=k_blocks,
            position_embeddings=block_position_embeddings,
        )
        k_blocks = k_blocks.detach().contiguous()
        prev = store.get(self.layer_idx, None)
        if prev is None:
            store[self.layer_idx] = k_blocks
        else:
            store[self.layer_idx] = torch.cat((prev, k_blocks), dim=1).contiguous()
        return True

    def _compute_chunked_gate_from_cache(
        self,
        query_states_nope: torch.Tensor,
        key_states_nope: torch.Tensor,
        past_key_value: Optional[Cache],
        block_attention_mask: Optional[torch.Tensor],
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache_position: Optional[torch.LongTensor],
        use_softmax: bool,
    ) -> Optional[torch.Tensor]:
        if block_attention_mask is None:
            return None
        if not self._can_use_chunked_gate_cache(
            key_states_nope=key_states_nope,
            cache_position=cache_position,
            block_position_embeddings=block_position_embeddings,
        ):
            return None
        store = self._chunked_gate_cache_store(past_key_value)
        if store is None:
            return None
        prev_k_blocks = store.get(self.layer_idx, None)
        if prev_k_blocks is None:
            return None

        q_blocks, current_k_blocks = self._build_chunked_gate_blocks(
            query_states_nope=query_states_nope,
            key_states_nope=key_states_nope,
            block_position_embeddings=block_position_embeddings,
        )
        full_k_blocks = torch.cat((prev_k_blocks, current_k_blocks.detach()), dim=1).contiguous()

        if block_attention_mask.shape[-2] != q_blocks.shape[1]:
            return None
        if block_attention_mask.shape[-1] != full_k_blocks.shape[1]:
            return None

        attn_gate_output = self.attn_gate.score_compressed_blocks(
            q=q_blocks,
            k=full_k_blocks,
            attention_mask=block_attention_mask,
            use_softmax=use_softmax,
        )
        store[self.layer_idx] = full_k_blocks
        return attn_gate_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        block_position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        q_len = hidden_states.shape[1]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = rearrange(query_states, "... (h d) -> ... h d", d=self.head_dim)
        key_states = rearrange(key_states, "... (h d) -> ... h d", d=self.head_dim)
        value_states = rearrange(value_states, "... (h d) -> ... h d", d=self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if self.config.use_flash_rope:
            query_states_nope = query_states.clone()
            key_states_nope = key_states.clone()
        else:
            query_states_nope = query_states
            key_states_nope = key_states

        cos, sin = position_embeddings
        if self.config.use_flash_rope:
            query_states = apply_rotary_emb_func(
                query_states, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len
            )
            key_states = apply_rotary_emb_func(
                key_states, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=2
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states.flatten(-2, -1),
                value_states.flatten(-2, -1),
                self.layer_idx,
                cache_kwargs,
            )
            key_states = rearrange(key_states, "... (h d) -> ... h d", d=self.head_dim)
            value_states = rearrange(value_states, "... (h d) -> ... h d", d=self.head_dim)

        chunked_prefill = (not self.training) and (q_len > 1) and (key_states.shape[1] > q_len)
        force_dense_prefill = (
            (not self.training)
            and (q_len > 1)
            and bool(getattr(self.config, "seerattn_chunked_prefill_force_dense", False))
        )
        if force_dense_prefill:
            attn_gate_output = None
            self._append_chunked_gate_key_cache(
                key_states_nope=key_states_nope,
                past_key_value=past_key_value,
                block_position_embeddings=block_position_embeddings,
                cache_position=cache_position,
            )
        elif chunked_prefill:
            attn_gate_output = self._compute_chunked_gate_from_cache(
                query_states_nope=query_states_nope,
                key_states_nope=key_states_nope,
                past_key_value=past_key_value,
                block_attention_mask=block_attention_mask,
                block_position_embeddings=block_position_embeddings,
                cache_position=cache_position,
                use_softmax=self.config.seerattn_sparsity_method == "threshold",
            )
            if attn_gate_output is None:
                attn_gate_output = self.attn_gate(
                    query_states,
                    key_states,
                    block_attention_mask,
                    None,
                    use_softmax=self.config.seerattn_sparsity_method == "threshold",
                )
                self._append_chunked_gate_key_cache(
                    key_states_nope=key_states_nope,
                    past_key_value=past_key_value,
                    block_position_embeddings=block_position_embeddings,
                    cache_position=cache_position,
                )
        else:
            attn_gate_output = self.attn_gate(
                query_states_nope,
                key_states_nope,
                block_attention_mask,
                block_position_embeddings,
                use_softmax=not self.training and self.config.seerattn_sparsity_method == "threshold",
            )
            if q_len > 1:
                self._append_chunked_gate_key_cache(
                    key_states_nope=key_states_nope,
                    past_key_value=past_key_value,
                    block_position_embeddings=block_position_embeddings,
                    cache_position=cache_position,
                )

        if self.training:
            attn_output, ground_truth_mask = attention_distill_forward(
                query_states,
                key_states,
                value_states,
                softmax_scale=self.scaling,
                block_size=self.config.seerattn_gate_block_size,
                num_key_value_groups=self.num_key_value_groups,
            )
        else:
            if force_dense_prefill:
                attn_output, _ = dense_prefill_full_kv(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    attention_mask=attention_mask,
                    softmax_scale=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    fallback_used=0.0,
                    measure_timing=False,
                    attn_module=self,
                )
            else:
                attn_output = sparse_flash_attention_forward(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    query_length=q_len,
                    softmax_scale=self.scaling,
                    attn_gate_score=attn_gate_output,
                    sparsity_method=self.config.seerattn_sparsity_method,
                    threshold=self.config.seerattn_threshold,
                    nz_ratio=self.config.seerattn_nz_ratio,
                    last_block_dense=self.config.seerattn_last_block_dense,
                    block_size=self.config.seerattn_gate_block_size,
                    num_key_value_groups=self.num_key_value_groups,
                    profile_file=self.profile_file,
                    block_attention_mask=block_attention_mask,
                )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if self.training:
            ground_truth_mask = ground_truth_mask[:, :, ground_truth_mask.shape[2] // 4 :].to(torch.float32)
            attn_gate_output = attn_gate_output[:, :, attn_gate_output.shape[2] // 4 :].to(torch.float32)
            attn_gate_output = F.log_softmax(attn_gate_output, dim=-1)
            mask_loss = self.mask_loss_func(attn_gate_output, ground_truth_mask)
        else:
            mask_loss = 0.0
            attn_gate_output = None
            ground_truth_mask = None

        if not kwargs.get("output_attentions", False):
            attn_gate_output = None
            ground_truth_mask = None
        return attn_output, mask_loss, None, attn_gate_output, ground_truth_mask


class SeerAttnQwen3DecoderLayer(nn.Module):
    def __init__(self, config: SeerAttnQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SeerAttnQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, seerattn_mask_loss, self_attn_weights, mask_gate_prediction, mask_ground_truth = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            block_position_embeddings=block_position_embeddings,
            block_attention_mask=block_attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, seerattn_mask_loss)
        if output_attentions:
            outputs += (self_attn_weights, mask_gate_prediction, mask_ground_truth)
        return outputs


class SeerAttnQwen3PreTrainedModel(SeerAttnQwen2PreTrainedModel):
    config_class = SeerAttnQwen3Config
    _no_split_modules = ["SeerAttnQwen3DecoderLayer"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)


class SeerAttnQwen3Model(SeerAttnQwen2Model):
    def __init__(self, config: SeerAttnQwen3Config):
        SeerAttnQwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SeerAttnQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        block_config = copy.deepcopy(config)
        block_config.hidden_size = config.seerattn_gate_hidden_size * config.num_attention_heads
        self.block_rotary_emb = Qwen3RotaryEmbedding(config=block_config)
        self.gradient_checkpointing = False
        self.post_init()


class SeerAttnQwen3ForCausalLM(SeerAttnQwen2ForCausalLM):
    config_class = SeerAttnQwen3Config

    def __init__(self, config):
        SeerAttnQwen3PreTrainedModel.__init__(self, config)
        self.model = SeerAttnQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, load_gate=True, *model_args, **kwargs):
        force_dense_prefill = kwargs.pop("seerattn_chunked_prefill_force_dense", False)
        final_dense_tail_blocks = kwargs.pop("seerattn_chunked_prefill_final_dense_tail_blocks", 0)
        input_config = kwargs.pop("config", None)
        gate_type = kwargs.get("seerattn_gate_type", "Qavg_Kmaxminavg")
        gate_hidden_size = kwargs.get("seerattn_gate_hidden_size", 128)
        gate_force_double = kwargs.get("seerattn_gate_force_double", False)

        def _coerce_qwen3_config(base_config, *, base_model_name):
            if isinstance(base_config, SeerAttnQwen3Config):
                config = base_config
            else:
                config = SeerAttnQwen3Config(**base_config.to_dict())
            _sync_qwen3_base_vocab_size(config, base_model_name, kwargs)
            config.seerattn_gate_type = gate_type
            config.seerattn_gate_hidden_size = gate_hidden_size
            config.seerattn_gate_force_double = gate_force_double
            setattr(config, "seerattn_chunked_prefill_force_dense", bool(force_dense_prefill))
            setattr(config, "seerattn_chunked_prefill_final_dense_tail_blocks", int(final_dense_tail_blocks))
            return config

        if load_gate:
            config = input_config
            if config is None:
                config = SeerAttnQwen3Config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            base_model = getattr(config, "base_model", pretrained_model_name_or_path)
            config = _coerce_qwen3_config(config, base_model_name=base_model)
            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))
            kwargs["device_map"] = _resolve_qwen_seer_device_map(config, kwargs.get("device_map"))
            model = super(SeerAttnQwen2ForCausalLM, cls).from_pretrained(
                base_model,
                config=config,
                *model_args,
                **kwargs,
            )

            if os.path.exists(pretrained_model_name_or_path):
                gate_weights = torch.load(os.path.join(pretrained_model_name_or_path, "attn_gate_weights.pth"))
            else:
                gate_weights = torch.load(
                    hf_hub_download(repo_id=pretrained_model_name_or_path, filename="attn_gate_weights.pth")
                )
            model.load_state_dict(gate_weights, strict=False)
            print("Attention gate weights loaded successfully.")
        else:
            config = input_config
            if config is None:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            base_model = getattr(config, "base_model", pretrained_model_name_or_path)
            config = _coerce_qwen3_config(config, base_model_name=base_model)
            for key in list(kwargs.keys()):
                if hasattr(config, key) and key != "torch_dtype":
                    setattr(config, key, kwargs.pop(key))
            kwargs["device_map"] = _resolve_qwen_seer_device_map(config, kwargs.get("device_map"))
            model = super(SeerAttnQwen2ForCausalLM, cls).from_pretrained(
                base_model,
                config=config,
                *model_args,
                **kwargs,
            )
            setattr(model.config, "seerattn_chunked_prefill_force_dense", bool(force_dense_prefill))
            setattr(model.config, "seerattn_chunked_prefill_final_dense_tail_blocks", int(final_dense_tail_blocks))
        return model


__all__ = [
    "SeerAttnQwen3Config",
    "SeerAttnQwen3ForCausalLM",
]
