import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import combinations
from compact_attn.modules.common import apply_rotary_pos_emb, repeat_kv, repeat_kv_varlen
from flash_attn.layers.rotary import apply_rotary_emb_func




def min_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return -F.max_pool3d(-input, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)



class MultiHeadLinear(nn.Module):
    def __init__(self, in_channel_size, hidden_size, num_head):
        super(MultiHeadLinear, self).__init__()
        self.in_channel = in_channel_size
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.weight = nn.Parameter(torch.Tensor(self.num_head, self.in_channel, self.hidden_size))
    

    def forward(self, x): # x shape (batch_size, seq_length, head, channel_size)
        if x.shape[2] < self.num_head:
            x = repeat_kv_varlen(x, self.num_head // x.shape[2])
        # print(f"x.shape: {x.shape}, self.weight.shape: {self.weight.shape}")
        return torch.einsum('bshi, hio->bsho', x, self.weight) # torch.matmul(x, self.weight)
        # return torch.matmul(x, self.weight) # torch.einsum('bhsi,hio->bhso', x, self.weight)


class AttnGate(nn.Module):
    def __init__(
        self,
        block_size,
        in_channel_size,
        hidden_size,
        num_k_head,
        num_q_head,
        q_pooling_funcs,
        k_pooling_funcs,
        force_double=False,
        use_flash_rope=False,
        kv_group_aware_query=False,
    ):
        super(AttnGate, self).__init__()
        self.block_size = block_size
        self.in_channel = in_channel_size
        self.hidden_size = hidden_size
        self.num_k_head = num_k_head
        self.num_q_head = num_q_head
        if self.num_q_head % self.num_k_head != 0:
            raise ValueError(
                f"num_q_head ({self.num_q_head}) must be divisible by num_k_head ({self.num_k_head})."
            )
        self.num_key_value_groups = self.num_q_head // self.num_k_head
        self.use_flash_rope = use_flash_rope
        self.kv_group_aware_query = bool(kv_group_aware_query)
        self.q_pooling_funcs = q_pooling_funcs
        self.k_pooling_funcs = k_pooling_funcs
        self.scale = self.hidden_size ** -0.5

        self.q_dup_size = len(q_pooling_funcs)
        self.k_dup_size = len(k_pooling_funcs)

        q_in_channel_size = in_channel_size * self.q_dup_size
        k_in_channel_size = in_channel_size * self.k_dup_size
        
        
        if self.kv_group_aware_query:
            self.mask_linear_q = MultiHeadLinear(
                self.num_key_value_groups * q_in_channel_size,
                self.hidden_size,
                self.num_k_head,
            )
            self.mask_linear_k = MultiHeadLinear(k_in_channel_size, self.hidden_size, self.num_k_head)
        elif self.q_dup_size > 1 or self.hidden_size != in_channel_size or force_double:
            self.mask_linear_q = MultiHeadLinear(q_in_channel_size, self.hidden_size, self.num_q_head)
            self.mask_linear_k = MultiHeadLinear(k_in_channel_size, self.hidden_size, self.num_k_head)
        else: # Can use a single linear layer if hidden_size = in_channel_size
            self.mask_linear_q = None
            self.mask_linear_k = MultiHeadLinear(k_in_channel_size, self.hidden_size, self.num_q_head)

    def reset_query_branch_parameters(self, initializer_range: float) -> None:
        if self.mask_linear_q is None:
            return
        with torch.no_grad():
            self.mask_linear_q.weight.data.normal_(mean=0.0, std=initializer_range)

    def _reshape_query_groups(self, q_compressed: torch.Tensor) -> torch.Tensor:
        if not self.kv_group_aware_query:
            return q_compressed
        bsz, q_blocks, _, channels = q_compressed.shape
        q_grouped = q_compressed.contiguous().view(
            bsz,
            q_blocks,
            self.num_k_head,
            self.num_key_value_groups,
            channels,
        )
        return q_grouped.reshape(
            bsz,
            q_blocks,
            self.num_k_head,
            self.num_key_value_groups * channels,
        )

    def compress_query_blocks(self, q: torch.Tensor) -> torch.Tensor:
        q_pooled = [
            pool_func(q, kernel_size=[self.block_size, 1, 1], stride=[self.block_size, 1, 1], ceil_mode=True)
            for pool_func in self.q_pooling_funcs
        ]
        q_compressed = torch.cat(q_pooled, dim=-1)
        q_compressed = self._reshape_query_groups(q_compressed)
        if self.mask_linear_q is not None:
            q_compressed = self.mask_linear_q(q_compressed)
        return q_compressed

    def compress_key_blocks(self, k: torch.Tensor) -> torch.Tensor:
        k_pooled = [
            pool_func(k, kernel_size=[self.block_size, 1, 1], stride=[self.block_size, 1, 1], ceil_mode=True)
            for pool_func in self.k_pooling_funcs
        ]
        k_compressed = torch.cat(k_pooled, dim=-1)
        return self.mask_linear_k(k_compressed)

    def apply_block_position_embeddings(
            self,
            q: Optional[torch.Tensor] = None,
            k: Optional[torch.Tensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if position_embeddings is None:
            return q, k

        cos, sin = position_embeddings
        use_flash_rope = self.use_flash_rope and cos.dim() == 2
        if q is not None:
            if use_flash_rope:
                q = apply_rotary_emb_func(
                    q,
                    cos,
                    sin,
                    False,
                    True,
                    cu_seqlens=None,
                    max_seqlen=q.shape[1],
                )
            else:
                q, _ = apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=2)

        if k is not None:
            if use_flash_rope:
                k = apply_rotary_emb_func(
                    k,
                    cos,
                    sin,
                    False,
                    True,
                    cu_seqlens=None,
                    max_seqlen=k.shape[1],
                )
            else:
                _, k = apply_rotary_pos_emb(k, k, cos, sin, unsqueeze_dim=2)

        return q, k

    def score_compressed_blocks(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            attention_mask: torch.Tensor,
            use_softmax: bool = True
        ) -> torch.Tensor:
        force_fp32 = os.environ.get("SEERATTN_DEBUG_GATE_SCORE_FP32", "0") == "1"
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if (not self.kv_group_aware_query) and k.shape[1] < self.num_q_head:
            k = repeat_kv(k, self.num_q_head // k.shape[1])

        if force_fp32:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None and attention_mask.shape[-2:] != attn.shape[-2:]:
            attention_mask = attention_mask[..., -attn.shape[-2]:, -attn.shape[-1]:]
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        if attention_mask is not None and attention_mask.shape[1] == 1 and attn.shape[1] != 1:
            attention_mask = attention_mask.expand(-1, attn.shape[1], -1, -1)
        if attention_mask is None:
            pass
        elif attention_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attention_mask, torch.finfo(attn.dtype).min)
        else:
            attn = attn + attention_mask
        if use_softmax:
            attn = F.softmax(attn, dim=-1)
        return attn

    
    def forward(
            self, 
            q, # [batch_size, seq_length, num_q_head, channel_size]
            k, # [batch_size, seq_length, num_k_head, channel_size]
            attention_mask, 
            position_embeddings=None, 
            use_softmax=True
        ):  
        q_len = q.shape[1]
        if q_len == 1:
            return None

        q = self.compress_query_blocks(q)
        k = self.compress_key_blocks(k)
        q, k = self.apply_block_position_embeddings(q=q, k=k, position_embeddings=position_embeddings)
        return self.score_compressed_blocks(q=q, k=k, attention_mask=attention_mask, use_softmax=use_softmax)


POOL_FUNCS = {
    'max': F.max_pool3d,
    'min': min_pool3d,
    'avg': F.avg_pool3d
}


def _create_generic_attngate_class(base_class, suffix, q_pooling_names, k_pooling_names):
    q_pooling_funcs = [POOL_FUNCS[name] for name in q_pooling_names]
    k_pooling_funcs = [POOL_FUNCS[name] for name in k_pooling_names]
    class_name = f"Q{''.join(q_pooling_names)}_K{''.join(k_pooling_names)}{suffix}"

    class NewAttnGate(base_class):
        def __init__(
            self,
            block_size,
            in_channel_size,
            hidden_size,
            num_k_head,
            num_q_head,
            force_double=False,
            use_flash_rope=False,
            kv_group_aware_query=False,
        ):
            super(NewAttnGate, self).__init__(
                block_size=block_size,
                in_channel_size=in_channel_size,
                hidden_size=hidden_size,
                num_k_head=num_k_head,
                num_q_head=num_q_head,
                q_pooling_funcs=q_pooling_funcs,
                k_pooling_funcs=k_pooling_funcs,
                force_double=force_double,
                use_flash_rope=use_flash_rope,
                kv_group_aware_query=kv_group_aware_query,
            )
    NewAttnGate.__name__ = class_name
    return class_name, NewAttnGate


def generate_combinations():
    new_classes = {}
    pool_types = ['max', 'min', 'avg']

    for q_comb in range(1, 4):
        for k_comb in range(1, 4):
            for q_pooling_comb in combinations(pool_types, q_comb):
                for k_pooling_comb in combinations(pool_types, k_comb):
                    class_name, new_class = _create_generic_attngate_class(AttnGate, '', q_pooling_comb, k_pooling_comb)
                    new_classes[class_name] = new_class
    return new_classes


ATTNGATE_CLASSES = generate_combinations()
