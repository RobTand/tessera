"""Skip the zero-width RoPE query concatenation in stock sparse MLA.

GLM-5.3 attention has ``qk_rope_head_dim == 0``. vLLM's MLA layer still passes
the sparse backend a ``(q_nope, q_pe)`` tuple, ``q_pe`` zero wide, and
``FlashInferMLASparseSM120Impl.forward_mqa`` joins the two with ``torch.cat``.
That copy is all of ``q_nope``. On a 2048-token prefill chunk at TP2 it moves a
``[2048, 32, 512]`` bf16 tensor (67 MB) per MLA layer, 0.56 ms each on GB10,
across 11 MLA layers (A8SESHMN served trace, 2026-10-01).

When the second part is zero wide, ``torch.cat`` returns a new contiguous
tensor that holds exactly ``q_nope``'s bytes, with ``q_nope``'s shape and
strides. When ``q_nope`` is itself contiguous and allocator-aligned, handing it
to the kernel directly gives the kernel the same bytes, shape, strides and
alignment at another address, so the kernel's output is bitwise equal.
``experiments/t8r_speed/empty_rope_cat_check.py`` checks this on the GPU at the
served shapes.

The change is opt-in: set ``TESSERA_GLM53_SKIP_EMPTY_ROPE_CAT=1``.
``TesseraConfig.get_quant_method`` installs it. If the stock source's sha256
isn't one this module was inspected against, the serve keeps the stock
``forward_mqa`` and logs one warning. Any query that is not exactly the
zero-width form described above goes to the stock code unchanged.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import logging
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from .flags import latched_bool

__all__ = ["FLAG", "install_for_current_config", "query_without_empty_rope"]

_log = logging.getLogger(__name__)

FLAG = "TESSERA_GLM53_SKIP_EMPTY_ROPE_CAT"

_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120"
_CLASS = "FlashInferMLASparseSM120Impl"

#: sha256 of ``vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py``
#: as inspected: image
#: localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a
#: (vLLM 0.30.1rc1.dev336+gaf5b4857e.d20260929). Its ``forward_mqa`` begins
#: ``if isinstance(q, tuple): q = torch.cat(q, dim=-1)`` and only reads ``q``.
_INSPECTED_SHA256 = frozenset({
    "102ca08793d567f95598eefeb34c3f6ec50b3b9d704f162d1402b97b12b5777b",
})

_SIGNATURE = ("self", "q", "kv_c_and_k_pe_cache", "attn_metadata", "layer")

#: The CUDA caching allocator starts every block on a 512-byte boundary, so a
#: fresh ``torch.cat`` result is 512-byte aligned. Requiring the same of
#: ``q_nope`` keeps every input property the kernel can observe identical.
_ALLOCATOR_ALIGNMENT = 512

_MARK = "_tessera_skip_empty_rope_cat"
_LOCK = RLock()
_DECLINED: list[str] = []


def query_without_empty_rope(q: Any) -> Any:
    """Return ``q_nope`` when ``torch.cat(q, dim=-1)`` would only copy it.

    ``q`` must be a ``(q_nope, q_pe)`` pair of tensors where ``q_pe`` is zero
    wide in the last dimension, both parts agree on the leading shape, dtype
    and device, and ``q_nope`` is contiguous and 512-byte aligned. Any other
    ``q`` is returned unchanged, so the caller's stock code handles it.
    """
    if not isinstance(q, tuple) or len(q) != 2:
        return q
    nope, pe = q
    if not isinstance(nope, torch.Tensor) or not isinstance(pe, torch.Tensor):
        return q
    if nope.dim() == 0 or pe.dim() != nope.dim():
        return q
    if pe.shape[-1] != 0 or pe.shape[:-1] != nope.shape[:-1]:
        return q
    if pe.dtype != nope.dtype or pe.device != nope.device:
        return q
    if not nope.is_contiguous() or nope.data_ptr() % _ALLOCATOR_ALIGNMENT:
        return q
    return nope


def _decline_reason(module: Any) -> str | None:
    path = getattr(module, "__file__", None)
    if not path:
        return f"{_MODULE} has no source file to inspect"
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest not in _INSPECTED_SHA256:
        return f"{_MODULE} sha256 {digest} is not an inspected source"
    impl = getattr(module, _CLASS, None)
    if impl is None:
        return f"{_MODULE} has no {_CLASS}"
    params = tuple(inspect.signature(impl.forward_mqa).parameters)
    if params != _SIGNATURE:
        return f"{_CLASS}.forward_mqa has parameters {params}, expected {_SIGNATURE}"
    return None


def _wrap(original: Any) -> Any:
    @functools.wraps(original)
    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        return original(self, query_without_empty_rope(q), kv_c_and_k_pe_cache,
                        attn_metadata, layer)

    setattr(forward_mqa, _MARK, True)
    forward_mqa.__wrapped_stock__ = original
    return forward_mqa


def install_for_current_config() -> bool:
    """Rebind the stock ``forward_mqa`` when the flag is on; True if installed.

    Idempotent. Declines to the stock path, warning once, when the source
    differs from an inspected one or the module can't be imported.
    """
    if not latched_bool(FLAG, meaning="skipping the zero-width RoPE query cat"):
        return False
    with _LOCK:
        try:
            module = importlib.import_module(_MODULE)
        except ImportError as exc:
            reason = f"{_MODULE} is not importable ({exc})"
            module = None
        else:
            impl = getattr(module, _CLASS, None)
            if impl is not None and getattr(impl.forward_mqa, _MARK, False):
                return True
            reason = _decline_reason(module)
        if reason is not None:
            if not _DECLINED:
                _DECLINED.append(reason)
                _log.warning("%s=1 declined; the stock query cat stays: %s", FLAG, reason)
            return False
        impl = getattr(module, _CLASS)
        impl.forward_mqa = _wrap(impl.forward_mqa)
        _log.info("%s=1: %s.forward_mqa passes a zero-width-RoPE query without cat",
                  FLAG, _CLASS)
        return True
