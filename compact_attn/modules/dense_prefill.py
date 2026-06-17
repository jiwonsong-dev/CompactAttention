from typing import Dict, Optional, Tuple

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from transformers.integrations.flash_attention import flash_attention_forward as hf_flash_attention_forward

from compact_attn.modules.common import _upad_input, pad_input, repeat_kv
from compact_attn.modules.dense_backend import (
    dense_prefill_flashinfer,
    resolve_dense_backend,
)

_FLASH_ATTN_INTERFACE = None


def _require_flash_attn_interface():
    global _FLASH_ATTN_INTERFACE
    if _FLASH_ATTN_INTERFACE is not None:
        return _FLASH_ATTN_INTERFACE
    try:
        import flash_attn_interface  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "fa3_direct dense backend selected, but `flash_attn_interface` is not importable. "
            "Install FlashAttention-3 from the flash-attention/hopper package first."
        ) from exc
    _FLASH_ATTN_INTERFACE = flash_attn_interface
    return flash_attn_interface


def _cuda_elapsed_ms(fn, enabled: bool = True):
    if not enabled:
        return fn(), 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    return out, float(start.elapsed_time(end))


def _normalize_hook_execution_device(execution_device) -> Optional[torch.device]:
    if execution_device is None:
        return None
    if isinstance(execution_device, torch.device):
        return execution_device
    if isinstance(execution_device, int):
        return torch.device("cuda", execution_device)
    try:
        return torch.device(execution_device)
    except (TypeError, RuntimeError, ValueError):
        return None


def _can_use_hf_fa2_on_local_shard(
    attn_module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask_fa2: Optional[torch.Tensor],
) -> bool:
    hook = getattr(attn_module, "_hf_hook", None)
    if hook is None:
        return False
    if attention_mask_fa2 is not None:
        return False
    if getattr(hook, "offload", False):
        return False
    if query_states.device != key_states.device or query_states.device != value_states.device:
        return False

    execution_device = _normalize_hook_execution_device(getattr(hook, "execution_device", None))
    if execution_device is None:
        return False
    return execution_device == query_states.device


def _normalize_dense_padding_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones((batch_size, kv_len), dtype=torch.long, device=device)

    mask = attention_mask.to(device=device)

    if mask.dim() == 2 and tuple(mask.shape) == (batch_size, kv_len):
        if mask.dtype == torch.bool:
            return mask.to(torch.long)
        if mask.dtype.is_floating_point:
            if torch.all((mask == 0) | (mask == 1)):
                return mask.to(torch.long)
            return torch.ones((batch_size, kv_len), dtype=torch.long, device=device)
        if mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            if torch.all((mask == 0) | (mask == 1)):
                return mask.to(torch.long)
            return torch.ones((batch_size, kv_len), dtype=torch.long, device=device)
        return torch.ones((batch_size, kv_len), dtype=torch.long, device=device)

    if mask.dim() >= 3 and mask.shape[-1] == kv_len:
        if mask.dtype == torch.bool:
            key_visible = mask
        else:
            key_visible = mask == 0
        reduce_dims = tuple(range(1, key_visible.dim() - 1))
        if reduce_dims:
            key_visible = key_visible.any(dim=reduce_dims)
        return key_visible.to(torch.long)

    return torch.ones((batch_size, kv_len), dtype=torch.long, device=device)


def dense_prefill_full_kv(
    query_states: torch.Tensor,  # [B, Q, Hq, D]
    key_states: torch.Tensor,  # [B, K, Hkv, D]
    value_states: torch.Tensor,  # [B, K, Hkv, D]
    attention_mask: Optional[torch.Tensor],  # [B, K]
    softmax_scale: float,
    num_key_value_groups: int,
    fallback_used: float = 1.0,
    measure_timing: bool = True,
    attn_module=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz, query_len, _, _ = query_states.shape
    kv_len = key_states.shape[1]
    device = query_states.device
    backend = resolve_dense_backend(attn_module=attn_module)

    if backend == "flashinfer":
        if attention_mask is not None:
            attention_mask = _normalize_dense_padding_mask(
                attention_mask,
                batch_size=bsz,
                kv_len=kv_len,
                device=device,
            )
        return dense_prefill_flashinfer(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            softmax_scale=softmax_scale,
            fallback_used=fallback_used,
            measure_timing=measure_timing,
        )

    if backend == "fa3_direct":
        flash_attn_interface = _require_flash_attn_interface()
        if attention_mask is None:
            has_padding = False
            attention_mask_fa3 = None
        else:
            attention_mask_fa3 = _normalize_dense_padding_mask(
                attention_mask,
                batch_size=bsz,
                kv_len=kv_len,
                device=device,
            )
            has_padding = bool((attention_mask_fa3 == 0).any().item())

        pack_gqa = None if num_key_value_groups == 1 else True
        if not has_padding:
            def _run_dense_fa3():
                return flash_attn_interface.flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    softmax_scale=softmax_scale,
                    causal=True,
                    pack_gqa=pack_gqa,
                )

            out, dense_kernel_ms = _cuda_elapsed_ms(_run_dense_fa3, enabled=measure_timing)
            stats = {
                "repeat_kv_ms": 0.0,
                "upad_input_ms": 0.0,
                "pad_output_ms": 0.0,
                "dense_kernel_ms": dense_kernel_ms,
                "gather_pack_ms": 0.0,
                "fallback_used": float(fallback_used),
            }
            return out, stats

        def _run_upad_fa3():
            return _upad_input(
                query_states, key_states, value_states, attention_mask_fa3, query_len
            )

        upad_out, upad_input_ms = _cuda_elapsed_ms(_run_upad_fa3, enabled=measure_timing)
        query_unpad, key_unpad, value_unpad, indices_q, cu_seq_lens, max_seq_lens = upad_out
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_q, max_seqlen_k = max_seq_lens

        def _run_dense_fa3_varlen():
            return flash_attn_interface.flash_attn_varlen_func(
                query_unpad,
                key_unpad,
                value_unpad,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                pack_gqa=pack_gqa,
            )

        out_unpad, dense_kernel_ms = _cuda_elapsed_ms(_run_dense_fa3_varlen, enabled=measure_timing)
        out, pad_output_ms = _cuda_elapsed_ms(
            lambda: pad_input(out_unpad, indices_q, bsz, query_len), enabled=measure_timing
        )
        stats = {
            "repeat_kv_ms": 0.0,
            "upad_input_ms": upad_input_ms,
            "pad_output_ms": pad_output_ms,
            "dense_kernel_ms": dense_kernel_ms,
            "gather_pack_ms": 0.0,
            "fallback_used": float(fallback_used),
        }
        return out, stats

    # device_map="auto" attaches AlignDevicesHook even on purely local CUDA shards.
    # The local varlen fallback can fail on short dense-tail segments with huge KV
    # lengths, so allow HF FA2 whenever the current shard is already local and the
    # segment has no padding mask.
    attention_mask_fa2 = attention_mask
    if attention_mask_fa2 is not None:
        attention_mask_fa2 = attention_mask_fa2.to(device=device)
        if not (attention_mask_fa2 == 0).any():
            attention_mask_fa2 = None

    is_sharded_attn_module = bool(getattr(attn_module, "_hf_hook", None) is not None) if attn_module is not None else False
    use_hf_fa2 = (
        attn_module is not None
        and query_states.is_cuda
        and query_states.dtype in (torch.float16, torch.bfloat16)
        and (
            (not is_sharded_attn_module)
            or _can_use_hf_fa2_on_local_shard(
                attn_module,
                query_states,
                key_states,
                value_states,
                attention_mask_fa2,
            )
        )
    )

    if use_hf_fa2:
        def _run_dense_fa2():
            attn_out, _ = hf_flash_attention_forward(
                attn_module,
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attention_mask_fa2,
                dropout=0.0,
                scaling=softmax_scale,
            )
            return attn_out

        try:
            out, dense_kernel_ms = _cuda_elapsed_ms(_run_dense_fa2, enabled=measure_timing)
            stats = {
                "repeat_kv_ms": 0.0,
                "upad_input_ms": 0.0,
                "pad_output_ms": 0.0,
                "dense_kernel_ms": dense_kernel_ms,
                "gather_pack_ms": 0.0,
                "fallback_used": float(fallback_used),
            }
            return out, stats
        except RuntimeError:
            if not is_sharded_attn_module:
                raise

    attention_mask = _normalize_dense_padding_mask(
        attention_mask,
        batch_size=bsz,
        kv_len=kv_len,
        device=device,
    )
    has_padding = bool((attention_mask == 0).any().item())

    def _run_repeat_kv():
        key_states_full = repeat_kv(
            key_states.transpose(1, 2).contiguous(), num_key_value_groups
        ).transpose(1, 2).contiguous()
        value_states_full = repeat_kv(
            value_states.transpose(1, 2).contiguous(), num_key_value_groups
        ).transpose(1, 2).contiguous()
        return key_states_full, value_states_full

    (key_states_full, value_states_full), repeat_kv_ms = _cuda_elapsed_ms(
        _run_repeat_kv, enabled=measure_timing
    )

    if not has_padding:
        def _run_dense_direct():
            return flash_attn_func(
                query_states,
                key_states_full,
                value_states_full,
                softmax_scale=softmax_scale,
                causal=True,
            )

        out, dense_kernel_ms = _cuda_elapsed_ms(_run_dense_direct, enabled=measure_timing)
        stats = {
            "repeat_kv_ms": repeat_kv_ms,
            "upad_input_ms": 0.0,
            "pad_output_ms": 0.0,
            "dense_kernel_ms": dense_kernel_ms,
            "gather_pack_ms": 0.0,
            "fallback_used": float(fallback_used),
        }
        return out, stats

    def _run_upad():
        return _upad_input(
            query_states, key_states_full, value_states_full, attention_mask, query_len
        )

    upad_out, upad_input_ms = _cuda_elapsed_ms(_run_upad, enabled=measure_timing)
    query_unpad, key_unpad, value_unpad, indices_q, cu_seq_lens, max_seq_lens = upad_out
    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_q, max_seqlen_k = max_seq_lens

    def _run_dense():
        return flash_attn_varlen_func(
            query_unpad,
            key_unpad,
            value_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
        )

    out_unpad, dense_kernel_ms = _cuda_elapsed_ms(_run_dense, enabled=measure_timing)
    out, pad_output_ms = _cuda_elapsed_ms(
        lambda: pad_input(out_unpad, indices_q, bsz, query_len), enabled=measure_timing
    )

    stats = {
        "repeat_kv_ms": repeat_kv_ms,
        "upad_input_ms": upad_input_ms,
        "pad_output_ms": pad_output_ms,
        "dense_kernel_ms": dense_kernel_ms,
        "gather_pack_ms": 0.0,
        "fallback_used": float(fallback_used),
    }
    return out, stats
