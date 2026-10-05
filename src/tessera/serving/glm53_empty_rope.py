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
``forward_mqa``. Each process logs exactly one install line, ``installed``,
``declined`` (with the reason) or ``off``, and one line for the first tuple
query it sees (``cat skipped`` or ``cat kept`` with the reason). Any query that
is not exactly the zero-width form described above goes to the stock code
unchanged.
"""
from __future__ import annotations

import functools
import importlib
import logging
import os
from threading import RLock
from typing import Any

import torch

from .flags import latched_bool
from .stock_interface import source_digest as _source_digest, signature_parameters, stock_attribute

__all__ = ["FLAG", "install_for_current_config", "query_without_empty_rope", "skip_reason"]

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
#: The one install line this process logged ("installed", "declined" or "off").
_REPORTED: list[str] = []
#: Whether the first tuple query has been reported (one line per process).
_FIRST_QUERY: list[bool] = []

_LEVER = "tessera.glm53_empty_rope: query cat skip"


def skip_reason(q: Any) -> str | None:
    """None when ``torch.cat(q, dim=-1)`` would only copy ``q[0]``; else why not.

    ``q`` must be a ``(q_nope, q_pe)`` pair of tensors where ``q_pe`` is zero
    wide in the last dimension, both parts agree on the leading shape, dtype
    and device, and ``q_nope`` is contiguous and 512-byte aligned.
    """
    if not isinstance(q, tuple) or len(q) != 2:
        return "query is not a 2-tuple"
    nope, pe = q
    if not isinstance(nope, torch.Tensor) or not isinstance(pe, torch.Tensor):
        return "query parts are not tensors"
    if nope.dim() == 0 or pe.dim() != nope.dim():
        return f"rank mismatch ({nope.dim()} vs {pe.dim()})"
    if pe.shape[-1] != 0:
        return f"RoPE part is {pe.shape[-1]} wide"
    if pe.shape[:-1] != nope.shape[:-1]:
        return f"leading shapes differ ({tuple(nope.shape)} vs {tuple(pe.shape)})"
    if pe.dtype != nope.dtype or pe.device != nope.device:
        return f"dtype/device differ ({nope.dtype}/{nope.device} vs {pe.dtype}/{pe.device})"
    if not nope.is_contiguous():
        return "q_nope is not contiguous"
    if nope.data_ptr() % _ALLOCATOR_ALIGNMENT:
        return f"q_nope is not {_ALLOCATOR_ALIGNMENT}-byte aligned"
    return None


def query_without_empty_rope(q: Any) -> Any:
    """Return ``q_nope`` when ``torch.cat(q, dim=-1)`` would only copy it.

    Any other ``q`` is returned unchanged, so the caller's stock code handles
    it. ``skip_reason`` states the conditions.
    """
    return q[0] if skip_reason(q) is None else q


def _report(line: str) -> None:
    if not _REPORTED:
        _REPORTED.append(line)
        _log.warning(line)


def _inspect(module: Any):
    """Return the inspected class/method/digest with the existing decline policy."""
    digest=_source_digest(module)
    if digest is None:return None,None,digest,f"{_MODULE} has no source file to inspect"
    if digest not in _INSPECTED_SHA256:return None,None,digest,f"{_MODULE} sha256 {digest[:12]} is not an inspected source"
    impl=stock_attribute(module,_CLASS)
    if impl is None:return None,None,digest,f"{_MODULE} has no {_CLASS}"
    params=signature_parameters(impl,"forward_mqa")
    if params!=_SIGNATURE:return None,None,digest,f"{_CLASS}.forward_mqa has parameters {params}, expected {_SIGNATURE}"
    return impl,stock_attribute(impl,"forward_mqa"),digest,None


def _decline_reason(module: Any) -> str | None:
    return _inspect(module)[3]


def _wrap(original: Any) -> Any:
    @functools.wraps(original)
    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        reason = skip_reason(q)
        if not _FIRST_QUERY and isinstance(q, tuple):
            _FIRST_QUERY.append(True)
            _log.warning("%s: first tuple query %s: %s", _LEVER,
                         [tuple(t.shape) for t in q if isinstance(t, torch.Tensor)],
                         "cat skipped" if reason is None else f"cat kept ({reason})")
        return original(self, q if reason is not None else q[0], kv_c_and_k_pe_cache,
                        attn_metadata, layer)

    setattr(forward_mqa, _MARK, True)
    forward_mqa.__wrapped_stock__ = original
    return forward_mqa


def install_for_current_config() -> bool:
    """Rebind the stock ``forward_mqa`` when the flag is on; True if installed.

    Idempotent. Logs exactly one line per process, at the first call:
    ``installed (...)``, ``declined, stock ...: reason`` or ``off (...)``.
    Declines to the stock path when the source differs from an inspected one or
    the module can't be imported.
    """
    if not latched_bool(FLAG, meaning="skipping the zero-width RoPE query cat"):
        _report(f"{_LEVER} off ({FLAG} unset or 0)")
        return False
    with _LOCK:
        try:
            module = importlib.import_module(_MODULE)
        except Exception as exc:
            reason = f"{_MODULE} is not importable ({exc})"
            module = None
        else:
            impl = stock_attribute(module,_CLASS)
            if impl is not None and stock_attribute(stock_attribute(impl,"forward_mqa"),_MARK,False):
                return True
            impl,original,digest,reason = _inspect(module)
        if reason is not None:
            _report(f"{_LEVER} declined, stock {_CLASS}.forward_mqa: {reason}")
            return False
        impl.forward_mqa = _wrap(original)
        image = os.environ.get("TESSERA_CENSUS_RUNTIME_IMAGE", "").rpartition("@sha256:")[2][:12]
        _report(f"{_LEVER} installed (stock source sha256 {digest[:12]}, "
                f"image sha {image or 'unstated'}): {_CLASS}.forward_mqa passes a "
                "zero-width-RoPE query without torch.cat")
        return True
