from __future__ import annotations

import os
import shutil
import sys
import inspect
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from compact_attn.kernels.varlen.flash_decode_varlen_left_pad_max_v2 import flash_decode_leftpad
from compact_attn.modules.common import _upad_input, pad_input

_VALID_DENSE_BACKENDS = {"flash_attn", "flashinfer", "fa3_direct"}
_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024
_FLASHINFER_WORKSPACES: Dict[tuple[int, str], torch.Tensor] = {}
_FLASHINFER_RAGGED_PREFILL_WRAPPERS: Dict[tuple[int, str, int, int, int], dict] = {}
_FLASHINFER_PAGED_PREFILL_WRAPPERS: Dict[tuple[int, str, int, int, int, int], dict] = {}
_FLASHINFER_PAGED_DECODE_WRAPPERS: Dict[tuple[int, str, int, int, int, int, int], dict] = {}
_FLASHINFER_FULL_CU_SEQLENS: Dict[tuple[int, str, int, int], torch.Tensor] = {}
_FLASHINFER_FULL_PAGED_METADATA: Dict[tuple[int, str, int, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_FLASHINFER_PREFILL_OUTPUTS: Dict[tuple[int, str, str, int, int, int], torch.Tensor] = {}
_FLASHINFER_DENSE_DEBUG_CALLS = 0
_FLASHINFER_SIGNATURE_KWARGS: Dict[tuple[int, str], bool] = {}
_FLASHINFER_ENV_CONFIGURED = False
_FLASHINFER_MODULE = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _flashinfer_attention_backend() -> str:
    override = os.environ.get("SEER_FLASHINFER_ATTENTION_BACKEND", "").strip()
    if override:
        return override
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
            if major == 9:
                return "fa3"
        except Exception:
            pass
    return "fa2"


def _flashinfer_supports_kwarg(fn, name: str) -> bool:
    key = (id(fn), name)
    cached = _FLASHINFER_SIGNATURE_KWARGS.get(key)
    if cached is not None:
        return cached
    try:
        supported = name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        supported = False
    _FLASHINFER_SIGNATURE_KWARGS[key] = supported
    return supported


def _flashinfer_single_attention_kwargs(fn, *, sm_scale: Optional[float]) -> dict:
    kwargs = {"kv_layout": "NHD", "sm_scale": sm_scale}
    if _flashinfer_supports_kwarg(fn, "backend"):
        kwargs["backend"] = _flashinfer_attention_backend()
    return kwargs


def _debug_flashinfer_dense() -> bool:
    return os.environ.get("SEER_DEBUG_DENSE_FLASHINFER", "0").strip() == "1"


def _debug_flashinfer_dense_limit() -> int:
    value = os.environ.get("SEER_DEBUG_DENSE_FLASHINFER_LIMIT", "8").strip()
    try:
        return max(int(value), 0)
    except ValueError:
        return 8


def _dense_flashinfer_full_prefill_mode() -> str:
    return os.environ.get("SEER_DENSE_FLASHINFER_FULL_PREFILL", "paged").strip().lower()


def _dense_flashinfer_page_size(query_len: int, kv_len: int) -> int:
    value = os.environ.get("SEER_DENSE_FLASHINFER_PAGE_SIZE", "64").strip()
    try:
        page_size = int(value)
    except ValueError:
        page_size = 64
    if page_size <= 0 or int(kv_len) % page_size != 0:
        page_size = int(query_len)
    return page_size


def _dense_flashinfer_cache_plans() -> bool:
    return _env_flag("SEER_DENSE_FLASHINFER_CACHE_PLANS", True)


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


def resolve_dense_backend(*, attn_module=None, config=None) -> str:
    if config is None and attn_module is not None:
        config = getattr(attn_module, "config", None)
    backend = str(getattr(config, "seerattn_dense_backend", "flash_attn"))
    if backend not in _VALID_DENSE_BACKENDS:
        raise ValueError(
            f"Unsupported dense backend '{backend}'. Expected one of {sorted(_VALID_DENSE_BACKENDS)}."
        )
    return backend


def _configure_flashinfer_env() -> None:
    global _FLASHINFER_ENV_CONFIGURED
    if _FLASHINFER_ENV_CONFIGURED:
        return
    torch_cuda = getattr(torch.version, "cuda", None)
    if torch_cuda and torch_cuda.startswith("13."):
        env_root = Path(sys.executable).resolve().parents[1]
        for nvcc in env_root.glob("lib/python*/site-packages/nvidia/cu13/bin/nvcc"):
            cuda_root = nvcc.parents[1]
            os.environ["CUDA_HOME"] = str(cuda_root)
            os.environ["CUDA_PATH"] = str(cuda_root)
            os.environ["FLASHINFER_NVCC"] = str(nvcc)
            break

    system_cuda_root = Path("/usr/local/cuda")
    system_nvcc = system_cuda_root / "bin" / "nvcc"
    if system_nvcc.exists():
        os.environ.setdefault("CUDA_HOME", str(system_cuda_root))
        os.environ.setdefault("CUDA_PATH", str(system_cuda_root))
        os.environ.setdefault("FLASHINFER_NVCC", str(system_nvcc))

    if shutil.which("ninja") is None:
        local_bin = Path(sys.executable).resolve().parent
        local_ninja = local_bin / "ninja"
        if local_ninja.exists():
            current_path = os.environ.get("PATH", "")
            path_entries = [p for p in current_path.split(os.pathsep) if p]
            if str(local_bin) not in path_entries:
                path_entries.append(str(local_bin))
                os.environ["PATH"] = os.pathsep.join(path_entries)

    if "FLASHINFER_CUDA_ARCH_LIST" in os.environ:
        _FLASHINFER_ENV_CONFIGURED = True
        return
    if not torch.cuda.is_available():
        _FLASHINFER_ENV_CONFIGURED = True
        return
    try:
        major_versions = {
            torch.cuda.get_device_capability(device_idx)[0]
            for device_idx in range(torch.cuda.device_count())
        }
    except Exception:
        _FLASHINFER_ENV_CONFIGURED = True
        return
    if major_versions and major_versions == {12}:
        # Blackwell systems in this environment can expose a CUDA 12.8 nvcc via
        # the conda env even when the runtime stack is CUDA 13.0. Pinning the
        # JIT target to SM120f avoids the import-time arch normalization failure
        # and keeps builds scoped to the requested architecture only.
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "12.0f"
    _FLASHINFER_ENV_CONFIGURED = True


def _require_flashinfer():
    global _FLASHINFER_MODULE
    if _FLASHINFER_MODULE is not None:
        return _FLASHINFER_MODULE
    _configure_flashinfer_env()
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "flashinfer backend selected, but `ninja` is not available in PATH. "
            "Install `ninja` in the active environment first."
        )
    try:
        import flashinfer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "flashinfer backend selected, but `flashinfer` is not importable in the active environment."
        ) from exc
    _FLASHINFER_MODULE = flashinfer
    return flashinfer


def _flashinfer_error_context(exc: Exception, *, op_name: str) -> RuntimeError:
    return RuntimeError(
        f"flashinfer backend selected, but {op_name} failed to initialize or execute: {exc}"
    )


def _get_workspace(device: torch.device) -> torch.Tensor:
    key = (device.index if device.index is not None else -1, device.type)
    workspace = _FLASHINFER_WORKSPACES.get(key)
    if workspace is None or workspace.device != device:
        workspace = torch.zeros(
            (_FLASHINFER_WORKSPACE_BYTES,), dtype=torch.uint8, device=device
        )
        _FLASHINFER_WORKSPACES[key] = workspace
    return workspace


def _get_full_cu_seqlens(device: torch.device, batch_size: int, seq_len: int) -> torch.Tensor:
    key = (device.index if device.index is not None else -1, device.type, int(batch_size), int(seq_len))
    cu = _FLASHINFER_FULL_CU_SEQLENS.get(key)
    if cu is None or cu.device != device:
        cu = torch.arange(
            0,
            (int(batch_size) + 1) * int(seq_len),
            int(seq_len),
            dtype=torch.int32,
            device=device,
        )
        _FLASHINFER_FULL_CU_SEQLENS[key] = cu
    return cu


def _get_full_paged_metadata(
    device: torch.device,
    batch_size: int,
    query_len: int,
    kv_len: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pages = int(kv_len) // int(page_size)
    key = (
        device.index if device.index is not None else -1,
        device.type,
        int(batch_size),
        int(query_len),
        int(kv_len),
        int(page_size),
    )
    cached = _FLASHINFER_FULL_PAGED_METADATA.get(key)
    if cached is not None and cached[0].device == device:
        return cached
    qo_indptr = torch.arange(
        0,
        (int(batch_size) + 1) * int(query_len),
        int(query_len),
        dtype=torch.int32,
        device=device,
    )
    paged_kv_indptr = torch.arange(
        0,
        (int(batch_size) + 1) * n_pages,
        n_pages,
        dtype=torch.int32,
        device=device,
    )
    paged_kv_indices = torch.arange(
        int(batch_size) * n_pages,
        dtype=torch.int32,
        device=device,
    )
    paged_kv_last_page_len = torch.full(
        (int(batch_size),),
        int(page_size),
        dtype=torch.int32,
        device=device,
    )
    cached = (qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len)
    _FLASHINFER_FULL_PAGED_METADATA[key] = cached
    return cached


def _get_prefill_output_workspace(
    *,
    device: torch.device,
    dtype: torch.dtype,
    total_tokens: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    key = (
        device.index if device.index is not None else -1,
        device.type,
        str(dtype),
        int(total_tokens),
        int(num_heads),
        int(head_dim),
    )
    out = _FLASHINFER_PREFILL_OUTPUTS.get(key)
    expected_shape = (int(total_tokens), int(num_heads), int(head_dim))
    if out is None or out.device != device or out.shape != expected_shape:
        out = torch.empty(expected_shape, dtype=dtype, device=device)
        _FLASHINFER_PREFILL_OUTPUTS[key] = out
    return out


def _get_ragged_prefill_entry(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    plan_cache_key=None,
):
    flashinfer = _require_flashinfer()
    key = (
        device.index if device.index is not None else -1,
        str(q_dtype),
        int(num_q_heads),
        int(num_kv_heads),
        int(head_dim),
    )
    if plan_cache_key is not None:
        key = (*key, plan_cache_key)
    entry = _FLASHINFER_RAGGED_PREFILL_WRAPPERS.get(key)
    if entry is None:
        try:
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                _get_workspace(device), "NHD", backend=_flashinfer_attention_backend()
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise _flashinfer_error_context(exc, op_name="BatchPrefillWithRaggedKVCacheWrapper")
        entry = {
            "wrapper": wrapper,
            "plan_signature": None,
        }
        _FLASHINFER_RAGGED_PREFILL_WRAPPERS[key] = entry
    return entry


def _get_paged_prefill_entry(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    plan_cache_key=None,
):
    flashinfer = _require_flashinfer()
    key = (
        device.index if device.index is not None else -1,
        str(q_dtype),
        int(num_q_heads),
        int(num_kv_heads),
        int(head_dim),
        int(page_size),
    )
    if plan_cache_key is not None:
        key = (*key, plan_cache_key)
    entry = _FLASHINFER_PAGED_PREFILL_WRAPPERS.get(key)
    if entry is None:
        try:
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                _get_workspace(device), "NHD", backend=_flashinfer_attention_backend()
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise _flashinfer_error_context(exc, op_name="BatchPrefillWithPagedKVCacheWrapper")
        entry = {
            "wrapper": wrapper,
            "plan_signature": None,
        }
        _FLASHINFER_PAGED_PREFILL_WRAPPERS[key] = entry
    return entry


def _get_paged_decode_entry(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    batch_size: int,
):
    flashinfer = _require_flashinfer()
    key = (
        device.index if device.index is not None else -1,
        str(q_dtype),
        int(num_q_heads),
        int(num_kv_heads),
        int(head_dim),
        int(page_size),
        int(batch_size),
    )
    entry = _FLASHINFER_PAGED_DECODE_WRAPPERS.get(key)
    if entry is None:
        try:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                _get_workspace(device), "NHD", backend=_flashinfer_attention_backend()
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise _flashinfer_error_context(exc, op_name="BatchDecodeWithPagedKVCacheWrapper")
        page_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        page_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        entry = {
            "wrapper": wrapper,
            "indices": page_indices,
            "indptr": page_indptr,
            "plan_signature": None,
        }
        _FLASHINFER_PAGED_DECODE_WRAPPERS[key] = entry
    return entry


def dense_decode_with_backend(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: Optional[float],
    backend: Optional[str] = None,
    attn_module=None,
    measure_timing: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    backend = backend or resolve_dense_backend(attn_module=attn_module)
    if backend == "flash_attn":
        out, dense_kernel_ms = _cuda_elapsed_ms(
            lambda: flash_decode_leftpad(
                query_states,
                key_states,
                value_states,
                cache_seqlens=cache_seqlens,
                sm_scale=softmax_scale,
            ),
            enabled=measure_timing,
        )
        return out, {"plan_ms": 0.0, "dense_kernel_ms": dense_kernel_ms}
    if backend != "flashinfer":
        raise ValueError(f"Unsupported dense backend '{backend}'.")
    return _dense_decode_flashinfer(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        measure_timing=measure_timing,
    )


def _dense_decode_flashinfer(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: Optional[float],
    measure_timing: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    flashinfer = _require_flashinfer()
    if not query_states.is_cuda:
        raise RuntimeError("flashinfer backend requires CUDA tensors.")

    q = query_states.squeeze(1) if query_states.dim() == 4 else query_states
    if q.dim() != 3:
        raise ValueError(f"Expected decode query shape [B, H, D], got {tuple(q.shape)}")

    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = int(key_states.shape[2])
    page_size = int(key_states.shape[1])
    cache_seqlens_i32 = (
        cache_seqlens
        if cache_seqlens.dtype == torch.int32 and cache_seqlens.is_contiguous()
        else cache_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    ).view(batch_size)

    if batch_size == 1:
        def _run_single():
            try:
                fn = flashinfer.single_decode_with_kv_cache
                out = flashinfer.single_decode_with_kv_cache(
                    q[0].contiguous(),
                    key_states[0].contiguous(),
                    value_states[0].contiguous(),
                    **_flashinfer_single_attention_kwargs(fn, sm_scale=softmax_scale),
                )
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise _flashinfer_error_context(exc, op_name="single_decode_with_kv_cache")
            return out.unsqueeze(0)

        out, dense_kernel_ms = _cuda_elapsed_ms(_run_single, enabled=measure_timing)
        return out, {"plan_ms": 0.0, "dense_kernel_ms": dense_kernel_ms}

    entry = _get_paged_decode_entry(
        device=q.device,
        q_dtype=q.dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        batch_size=batch_size,
    )
    wrapper = entry["wrapper"]
    plan_signature = (
        int(page_size),
        int(batch_size),
        int(cache_seqlens_i32.data_ptr()),
        tuple(cache_seqlens_i32.shape),
    )
    plan_ms = 0.0
    if entry["plan_signature"] != plan_signature:
        def _run_plan():
            try:
                wrapper.plan(
                    entry["indptr"],
                    entry["indices"],
                    cache_seqlens_i32,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    page_size,
                    q_data_type=q.dtype,
                    kv_data_type=key_states.dtype,
                    o_data_type=q.dtype,
                    data_type=q.dtype,
                    sm_scale=softmax_scale,
                )
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise _flashinfer_error_context(exc, op_name="BatchDecodeWithPagedKVCacheWrapper.plan")

        _, plan_ms = _cuda_elapsed_ms(_run_plan, enabled=measure_timing)
        entry["plan_signature"] = plan_signature

    def _run_decode():
        try:
            return wrapper.run(q.contiguous(), (key_states, value_states))
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise _flashinfer_error_context(exc, op_name="BatchDecodeWithPagedKVCacheWrapper.run")

    out, dense_kernel_ms = _cuda_elapsed_ms(_run_decode, enabled=measure_timing)
    return out, {"plan_ms": plan_ms, "dense_kernel_ms": dense_kernel_ms}


def dense_prefill_flashinfer(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    softmax_scale: float,
    fallback_used: float = 1.0,
    measure_timing: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    global _FLASHINFER_DENSE_DEBUG_CALLS
    debug_dense = _debug_flashinfer_dense()
    if debug_dense:
        measure_timing = True
    debug_start_s = time.perf_counter() if debug_dense else 0.0

    if not query_states.is_cuda:
        raise RuntimeError("flashinfer backend requires CUDA tensors.")

    bsz, query_len, num_q_heads, head_dim = query_states.shape
    kv_len = key_states.shape[1]
    has_padding = False if attention_mask is None else bool((attention_mask == 0).any().item())
    num_kv_heads = int(key_states.shape[2])

    if query_len == 1:
        cache_seqlens = (
            torch.full((bsz,), kv_len, device=query_states.device, dtype=torch.int32)
            if attention_mask is None
            else attention_mask.sum(dim=-1, dtype=torch.int32)
        )
        out, decode_stats = dense_decode_with_backend(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            backend="flashinfer",
            measure_timing=measure_timing,
        )
        stats = {
            "repeat_kv_ms": 0.0,
            "upad_input_ms": 0.0,
            "pad_output_ms": 0.0,
            "dense_kernel_ms": float(decode_stats.get("dense_kernel_ms", 0.0)),
            "gather_pack_ms": float(decode_stats.get("plan_ms", 0.0)),
            "fallback_used": float(fallback_used),
        }
        if debug_dense and _FLASHINFER_DENSE_DEBUG_CALLS < _debug_flashinfer_dense_limit():
            _FLASHINFER_DENSE_DEBUG_CALLS += 1
            rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "?"))
            print(
                "[dense_flashinfer_debug]"
                f" rank={rank}"
                f" call={_FLASHINFER_DENSE_DEBUG_CALLS}"
                f" q={tuple(query_states.shape)} k={tuple(key_states.shape)}"
                f" has_padding={has_padding}"
                f" backend={_flashinfer_attention_backend()}"
                f" wall_ms={(time.perf_counter() - debug_start_s) * 1000.0:.3f}"
                f" stats={stats}",
                flush=True,
            )
        return out, stats

    flashinfer = _require_flashinfer()
    if bsz == 1 and not has_padding:
        def _run_single():
            try:
                fn = flashinfer.single_prefill_with_kv_cache
                out = flashinfer.single_prefill_with_kv_cache(
                    query_states[0].contiguous(),
                    key_states[0].contiguous(),
                    value_states[0].contiguous(),
                    causal=True,
                    **_flashinfer_single_attention_kwargs(fn, sm_scale=softmax_scale),
                )
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise _flashinfer_error_context(exc, op_name="single_prefill_with_kv_cache")
            return out.unsqueeze(0)

        out, dense_kernel_ms = _cuda_elapsed_ms(_run_single, enabled=measure_timing)
        return out, {
            "repeat_kv_ms": 0.0,
            "upad_input_ms": 0.0,
            "pad_output_ms": 0.0,
            "dense_kernel_ms": dense_kernel_ms,
            "gather_pack_ms": 0.0,
            "fallback_used": float(fallback_used),
        }

    full_prefill_mode = _dense_flashinfer_full_prefill_mode()
    can_use_paged_full = (
        not has_padding
        and full_prefill_mode in {"paged", "paged_heads"}
        and query_len > 1
        and kv_len % _dense_flashinfer_page_size(query_len, kv_len) == 0
    )
    if can_use_paged_full and full_prefill_mode == "paged_heads" and num_q_heads % num_kv_heads == 0:
        page_size = _dense_flashinfer_page_size(query_len, kv_len)
        n_pages = int(kv_len) // page_size
        groups = int(num_q_heads) // int(num_kv_heads)
        rows = int(bsz) * int(num_kv_heads)
        query_was_contiguous = query_states.is_contiguous()
        key_was_contiguous = key_states.is_contiguous()
        value_was_contiguous = value_states.is_contiguous()

        def _make_full_paged_head_views():
            q_fi = (
                query_states.view(bsz, query_len, num_kv_heads, groups, head_dim)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(rows * query_len, groups, head_dim)
            )
            k_pages = (
                key_states.permute(0, 2, 1, 3)
                .contiguous()
                .view(rows * n_pages, page_size, 1, head_dim)
            )
            v_pages = (
                value_states.permute(0, 2, 1, 3)
                .contiguous()
                .view(rows * n_pages, page_size, 1, head_dim)
            )
            return q_fi, k_pages, v_pages

        (query_unpad, key_pages, value_pages), contiguous_ms = _cuda_elapsed_ms(
            _make_full_paged_head_views,
            enabled=measure_timing,
        )
        (
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = _get_full_paged_metadata(query_states.device, rows, query_len, kv_len, page_size)
        plan_signature = (
            "full_paged_heads",
            int(rows),
            int(query_len),
            int(kv_len),
        )
        entry = _get_paged_prefill_entry(
            device=query_states.device,
            q_dtype=query_states.dtype,
            num_q_heads=groups,
            num_kv_heads=1,
            head_dim=head_dim,
            page_size=page_size,
            plan_cache_key=plan_signature if _dense_flashinfer_cache_plans() else None,
        )
        wrapper = entry["wrapper"]
        plan_ms = 0.0
        if entry["plan_signature"] != plan_signature:
            def _run_plan():
                try:
                    wrapper.plan(
                        qo_indptr=qo_indptr,
                        paged_kv_indptr=paged_kv_indptr,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len=paged_kv_last_page_len,
                        num_qo_heads=groups,
                        num_kv_heads=1,
                        head_dim_qk=head_dim,
                        page_size=page_size,
                        causal=True,
                        sm_scale=softmax_scale,
                        q_data_type=query_states.dtype,
                        kv_data_type=key_states.dtype,
                        o_data_type=query_states.dtype,
                    )
                except Exception as exc:  # pragma: no cover - depends on runtime env
                    raise _flashinfer_error_context(exc, op_name="BatchPrefillWithPagedKVCacheWrapper.plan")

            _, plan_ms = _cuda_elapsed_ms(_run_plan, enabled=measure_timing)
            entry["plan_signature"] = plan_signature

        def _run_prefill():
            try:
                out_buf = _get_prefill_output_workspace(
                    device=query_states.device,
                    dtype=query_states.dtype,
                    total_tokens=rows * query_len,
                    num_heads=groups,
                    head_dim=head_dim,
                )
                return wrapper.run(query_unpad, (key_pages, value_pages), out=out_buf)
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise _flashinfer_error_context(exc, op_name="BatchPrefillWithPagedKVCacheWrapper.run")

        out_unpad, dense_kernel_ms = _cuda_elapsed_ms(_run_prefill, enabled=measure_timing)
        out = (
            out_unpad.view(bsz, num_kv_heads, query_len, groups, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(bsz, query_len, num_q_heads, head_dim)
        )
        stats = {
            "repeat_kv_ms": 0.0,
            "upad_input_ms": 0.0,
            "pad_output_ms": 0.0,
            "dense_kernel_ms": dense_kernel_ms,
            "gather_pack_ms": plan_ms,
            "contiguous_ms": contiguous_ms,
            "fallback_used": float(fallback_used),
        }
        if debug_dense and _FLASHINFER_DENSE_DEBUG_CALLS < _debug_flashinfer_dense_limit():
            _FLASHINFER_DENSE_DEBUG_CALLS += 1
            rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "?"))
            print(
                "[dense_flashinfer_debug]"
                f" rank={rank}"
                f" call={_FLASHINFER_DENSE_DEBUG_CALLS}"
                f" path=full_paged_heads"
                f" pages={n_pages}"
                f" rows={rows}"
                f" q={tuple(query_states.shape)} k={tuple(key_states.shape)}"
                f" contig=({int(query_was_contiguous)},{int(key_was_contiguous)},{int(value_was_contiguous)})"
                f" backend={_flashinfer_attention_backend()}"
                f" wall_ms={(time.perf_counter() - debug_start_s) * 1000.0:.3f}"
                f" stats={stats}",
                flush=True,
            )
        return out, stats

    if can_use_paged_full:
        page_size = _dense_flashinfer_page_size(query_len, kv_len)
        n_pages = int(kv_len) // page_size
        query_was_contiguous = query_states.is_contiguous()
        key_was_contiguous = key_states.is_contiguous()
        value_was_contiguous = value_states.is_contiguous()

        def _make_full_paged_views():
            return (
                query_states.contiguous().view(bsz * query_len, num_q_heads, head_dim),
                key_states.contiguous().view(bsz * n_pages, page_size, num_kv_heads, head_dim),
                value_states.contiguous().view(bsz * n_pages, page_size, num_kv_heads, head_dim),
            )

        (query_unpad, key_pages, value_pages), contiguous_ms = _cuda_elapsed_ms(
            _make_full_paged_views,
            enabled=measure_timing,
        )
        (
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = _get_full_paged_metadata(query_states.device, bsz, query_len, kv_len, page_size)
        plan_signature = (
            "full_paged",
            int(bsz),
            int(query_len),
            int(kv_len),
        )
        entry = _get_paged_prefill_entry(
            device=query_states.device,
            q_dtype=query_states.dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            plan_cache_key=plan_signature if _dense_flashinfer_cache_plans() else None,
        )
        wrapper = entry["wrapper"]
        plan_ms = 0.0
        if entry["plan_signature"] != plan_signature:
            def _run_plan():
                try:
                    wrapper.plan(
                        qo_indptr=qo_indptr,
                        paged_kv_indptr=paged_kv_indptr,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len=paged_kv_last_page_len,
                        num_qo_heads=num_q_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim_qk=head_dim,
                        page_size=page_size,
                        causal=True,
                        sm_scale=softmax_scale,
                        q_data_type=query_states.dtype,
                        kv_data_type=key_states.dtype,
                        o_data_type=query_states.dtype,
                    )
                except Exception as exc:  # pragma: no cover - depends on runtime env
                    raise _flashinfer_error_context(exc, op_name="BatchPrefillWithPagedKVCacheWrapper.plan")

            _, plan_ms = _cuda_elapsed_ms(_run_plan, enabled=measure_timing)
            entry["plan_signature"] = plan_signature

        def _run_prefill():
            try:
                out_buf = _get_prefill_output_workspace(
                    device=query_states.device,
                    dtype=query_states.dtype,
                    total_tokens=bsz * query_len,
                    num_heads=num_q_heads,
                    head_dim=head_dim,
                )
                return wrapper.run(query_unpad, (key_pages, value_pages), out=out_buf)
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise _flashinfer_error_context(exc, op_name="BatchPrefillWithPagedKVCacheWrapper.run")

        out_unpad, dense_kernel_ms = _cuda_elapsed_ms(_run_prefill, enabled=measure_timing)
        out = out_unpad.view(bsz, query_len, num_q_heads, head_dim)
        stats = {
            "repeat_kv_ms": 0.0,
            "upad_input_ms": 0.0,
            "pad_output_ms": 0.0,
            "dense_kernel_ms": dense_kernel_ms,
            "gather_pack_ms": plan_ms,
            "contiguous_ms": contiguous_ms,
            "fallback_used": float(fallback_used),
        }
        if debug_dense and _FLASHINFER_DENSE_DEBUG_CALLS < _debug_flashinfer_dense_limit():
            _FLASHINFER_DENSE_DEBUG_CALLS += 1
            rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "?"))
            print(
                "[dense_flashinfer_debug]"
                f" rank={rank}"
                f" call={_FLASHINFER_DENSE_DEBUG_CALLS}"
                f" path=full_paged"
                f" pages={n_pages}"
                f" q={tuple(query_states.shape)} k={tuple(key_states.shape)}"
                f" contig=({int(query_was_contiguous)},{int(key_was_contiguous)},{int(value_was_contiguous)})"
                f" backend={_flashinfer_attention_backend()}"
                f" wall_ms={(time.perf_counter() - debug_start_s) * 1000.0:.3f}"
                f" stats={stats}",
                flush=True,
            )
        return out, stats

    if not has_padding:
        query_was_contiguous = query_states.is_contiguous()
        key_was_contiguous = key_states.is_contiguous()
        value_was_contiguous = value_states.is_contiguous()

        def _make_full_unpad_views():
            return (
                query_states.contiguous().view(bsz * query_len, num_q_heads, head_dim),
                key_states.contiguous().view(bsz * kv_len, num_kv_heads, head_dim),
                value_states.contiguous().view(bsz * kv_len, num_kv_heads, head_dim),
            )

        (query_unpad, key_unpad, value_unpad), contiguous_ms = _cuda_elapsed_ms(
            _make_full_unpad_views,
            enabled=measure_timing,
        )
        cu_seqlens_q = _get_full_cu_seqlens(query_states.device, bsz, query_len)
        cu_seqlens_k = _get_full_cu_seqlens(query_states.device, bsz, kv_len)
        plan_signature = (
            "full",
            int(bsz),
            int(query_len),
            int(kv_len),
        )
        entry = _get_ragged_prefill_entry(
            device=query_states.device,
            q_dtype=query_states.dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            plan_cache_key=plan_signature if _dense_flashinfer_cache_plans() else None,
        )
        wrapper = entry["wrapper"]
        plan_ms = 0.0
        if entry["plan_signature"] != plan_signature:
            def _run_plan():
                try:
                    wrapper.plan(
                        cu_seqlens_q,
                        cu_seqlens_k,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        causal=True,
                        sm_scale=softmax_scale,
                        q_data_type=query_states.dtype,
                        kv_data_type=key_states.dtype,
                        o_data_type=query_states.dtype,
                    )
                except Exception as exc:  # pragma: no cover - depends on runtime env
                    raise _flashinfer_error_context(exc, op_name="BatchPrefillWithRaggedKVCacheWrapper.plan")

            _, plan_ms = _cuda_elapsed_ms(_run_plan, enabled=measure_timing)
            entry["plan_signature"] = plan_signature

        def _run_prefill():
            try:
                out_buf = _get_prefill_output_workspace(
                    device=query_states.device,
                    dtype=query_states.dtype,
                    total_tokens=bsz * query_len,
                    num_heads=num_q_heads,
                    head_dim=head_dim,
                )
                return wrapper.run(query_unpad, key_unpad, value_unpad, out=out_buf)
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise _flashinfer_error_context(exc, op_name="BatchPrefillWithRaggedKVCacheWrapper.run")

        out_unpad, dense_kernel_ms = _cuda_elapsed_ms(_run_prefill, enabled=measure_timing)
        out = out_unpad.view(bsz, query_len, num_q_heads, head_dim)
        stats = {
            "repeat_kv_ms": 0.0,
            "upad_input_ms": 0.0,
            "pad_output_ms": 0.0,
            "dense_kernel_ms": dense_kernel_ms,
            "gather_pack_ms": plan_ms,
            "contiguous_ms": contiguous_ms,
            "fallback_used": float(fallback_used),
        }
        if debug_dense and _FLASHINFER_DENSE_DEBUG_CALLS < _debug_flashinfer_dense_limit():
            _FLASHINFER_DENSE_DEBUG_CALLS += 1
            rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "?"))
            print(
                "[dense_flashinfer_debug]"
                f" rank={rank}"
                f" call={_FLASHINFER_DENSE_DEBUG_CALLS}"
                f" path=full"
                f" q={tuple(query_states.shape)} k={tuple(key_states.shape)}"
                f" contig=({int(query_was_contiguous)},{int(key_was_contiguous)},{int(value_was_contiguous)})"
                f" backend={_flashinfer_attention_backend()}"
                f" wall_ms={(time.perf_counter() - debug_start_s) * 1000.0:.3f}"
                f" stats={stats}",
                flush=True,
            )
        return out, stats

    upad_out, upad_input_ms = _cuda_elapsed_ms(
        lambda: _upad_input(query_states, key_states, value_states, attention_mask, query_len),
        enabled=measure_timing,
    )
    query_unpad, key_unpad, value_unpad, indices_q, cu_seq_lens, _ = upad_out
    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    entry = _get_ragged_prefill_entry(
        device=query_states.device,
        q_dtype=query_states.dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    wrapper = entry["wrapper"]
    plan_signature = (
        tuple(attention_mask.shape),
        int(attention_mask.data_ptr()),
        int(query_len),
        int(kv_len),
    )
    plan_ms = 0.0
    if entry["plan_signature"] != plan_signature:
        def _run_plan():
            try:
                wrapper.plan(
                    cu_seqlens_q,
                    cu_seqlens_k,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    causal=True,
                    sm_scale=softmax_scale,
                    q_data_type=query_states.dtype,
                    kv_data_type=key_states.dtype,
                    o_data_type=query_states.dtype,
                )
            except Exception as exc:  # pragma: no cover - depends on runtime env
                raise _flashinfer_error_context(exc, op_name="BatchPrefillWithRaggedKVCacheWrapper.plan")

        _, plan_ms = _cuda_elapsed_ms(_run_plan, enabled=measure_timing)
        entry["plan_signature"] = plan_signature

    def _run_prefill():
        try:
            return wrapper.run(
                query_unpad.contiguous(),
                key_unpad.contiguous(),
                value_unpad.contiguous(),
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise _flashinfer_error_context(exc, op_name="BatchPrefillWithRaggedKVCacheWrapper.run")

    out_unpad, dense_kernel_ms = _cuda_elapsed_ms(_run_prefill, enabled=measure_timing)
    out, pad_output_ms = _cuda_elapsed_ms(
        lambda: pad_input(out_unpad, indices_q, bsz, query_len),
        enabled=measure_timing,
    )
    stats = {
        "repeat_kv_ms": 0.0,
        "upad_input_ms": upad_input_ms,
        "pad_output_ms": pad_output_ms,
        "dense_kernel_ms": dense_kernel_ms,
        "gather_pack_ms": plan_ms,
        "fallback_used": float(fallback_used),
    }
    if debug_dense and _FLASHINFER_DENSE_DEBUG_CALLS < _debug_flashinfer_dense_limit():
        valid_lens = attention_mask.sum(dim=-1).detach().to("cpu").tolist()
        _FLASHINFER_DENSE_DEBUG_CALLS += 1
        rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "?"))
        print(
            "[dense_flashinfer_debug]"
            f" rank={rank}"
            f" call={_FLASHINFER_DENSE_DEBUG_CALLS}"
            f" q={tuple(query_states.shape)} k={tuple(key_states.shape)}"
            f" has_padding={has_padding}"
            f" valid_lens_min={min(valid_lens) if valid_lens else None}"
            f" valid_lens_max={max(valid_lens) if valid_lens else None}"
            f" backend={_flashinfer_attention_backend()}"
            f" wall_ms={(time.perf_counter() - debug_start_s) * 1000.0:.3f}"
            f" stats={stats}",
            flush=True,
        )
    return out, stats
