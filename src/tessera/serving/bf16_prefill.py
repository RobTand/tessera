"""Decode the packed BF16 WINDOW wire into one shared scratch before each GEMM.

Packed bundles remain the module storage. The pool owns one BF16 matrix for each
shape and device. Each full lane call includes direct decode and cuBLAS BF16 GEMM.
No module holds a private materialized matrix or a second row-scale tensor.

Ordinary eager calls use a host lock and a CUDA event through the GEMM completion.
CUDA graph capture omits this event protocol. Graph replay does not execute the
host lock or update the event. Graph users must externally serialize every replay
and every eager user of the same scratch. Serialized benchmark replay is the scope.
This module does not qualify concurrent graph replay.
Retain each prepared workspace for the full graph lifetime.
"""
from __future__ import annotations

import dataclasses
import os
import threading
import weakref

import torch

from ..window_gemm import PreparedWindowGemm, decode_window_bf16_into

__all__ = ["Bf16Prefill", "FLAG", "MIN_M_FLAG", "configured_min_m", "decode_into",
           "enabled", "prefill_apply", "prepare_bf16_prefill", "scratch_pool_footprint"]

FLAG = "TESSERA_BF16_DECODE_ONCE"
MIN_M_FLAG = "TESSERA_BF16_DECODE_ONCE_MIN_M"


def enabled() -> bool:
    """Return the strict, process-stable opt-in flag. The default is off."""
    from .flags import latched_bool

    return latched_bool(FLAG, meaning="the shared BF16 prefill lane")


def configured_min_m() -> int:
    """Read an explicit measurement-selected threshold. No default exists."""
    from .flags import _latch

    current = os.environ.get(MIN_M_FLAG, "").strip()
    if not current.isascii() or not current.isdecimal() or int(current) < 1:
        raise ValueError(f"{MIN_M_FLAG} must specify a positive measured admission threshold")
    return int(_latch(MIN_M_FLAG, current))


class _Scratch:
    """One physical matrix and the eager stream order for a shape and device."""

    def __init__(self, rows: int, cols: int, device: torch.device):
        self.weight = torch.empty(rows, cols, dtype=torch.bfloat16, device=device)
        self.weight_t = self.weight.t()
        self.lock = threading.Lock()
        self.done = torch.cuda.Event()
        # Allocate the CUDA event before any graph capture.
        self.done.record(torch.cuda.current_stream(device))


_POOL_LOCK = threading.Lock()
_POOL: weakref.WeakValueDictionary[tuple[str, int, int], _Scratch] = weakref.WeakValueDictionary()


@dataclasses.dataclass(frozen=True)
class Bf16Prefill:
    """Frozen packed roles and a reference to their shared scratch. No decoded copy."""

    role_bundles: tuple[PreparedWindowGemm, ...]
    _workspace: _Scratch = dataclasses.field(repr=False, compare=False)

    @property
    def weight(self) -> torch.Tensor:
        return self._workspace.weight

    @property
    def nbytes(self) -> int:
        """Return the shared physical scratch size. Do not sum it across modules."""
        return self.weight.numel() * self.weight.element_size()


def scratch_pool_footprint() -> dict[tuple[str, int, int], int]:
    """Return each shared shape once. The sum is the pool's physical scratch size."""
    with _POOL_LOCK:
        return {key: item.weight.numel() * item.weight.element_size()
                for key, item in _POOL.items()}


def prepare_bf16_prefill(module) -> Bf16Prefill:
    """Attach no decoded weights. Freeze the roles and obtain their shared scratch."""
    bundles = module.role_bundles
    if module.family != "value" or module.arithmetic != "folded":
        raise ValueError("BF16 prefill requires the folded value family")
    if not bundles or module.device.type != "cuda":
        raise ValueError("BF16 prefill requires packed CUDA role bundles")
    if (sum(bundle.rows for bundle in bundles) != module.rows
            or any(bundle.cols != module.columns or bundle.device != module.device
                   or bundle.family != "value" or bundle.arithmetic != "folded"
                   for bundle in bundles)):
        raise ValueError("BF16 prefill role bundles do not match the module")
    device = bundles[0].device
    key = (str(device), module.rows, module.columns)
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare BF16 prefill before CUDA graph capture")
        with _POOL_LOCK:
            workspace = _POOL.get(key)
            if workspace is None:
                workspace = _Scratch(module.rows, module.columns, device)
                _POOL[key] = workspace
        # Order this module's load work against prior eager scratch users.
        with workspace.lock:
            stream = torch.cuda.current_stream(device)
            stream.wait_event(workspace.done)
            workspace.done.record(stream)
    return Bf16Prefill(role_bundles=tuple(bundles), _workspace=workspace)


def _decode(prepared: Bf16Prefill) -> None:
    offset = 0
    for bundle in prepared.role_bundles:
        decode_window_bf16_into(bundle, prepared.weight, row_offset=offset)
        offset += bundle.rows


def decode_into(prepared: Bf16Prefill) -> torch.Tensor:
    """Decode into the shared scratch for diagnostic time, without a GEMM.

    The returned tensor is shared mutable scratch. A later same-shape call overwrites
    it. Use prefill_apply for an atomic eager decode and GEMM sequence.
    """
    workspace = prepared._workspace
    with torch.cuda.device(prepared.weight.device), workspace.lock:
        capture = torch.cuda.is_current_stream_capturing()
        stream = torch.cuda.current_stream(prepared.weight.device)
        if not capture:
            stream.wait_event(workspace.done)
            workspace.weight.record_stream(stream)
        try:
            _decode(prepared)
        finally:
            if not capture:
                workspace.done.record(stream)
    return prepared.weight


def prefill_apply(prepared: Bf16Prefill, x: torch.Tensor) -> torch.Tensor:
    """Decode every call, then return a fresh BF16 output from cuBLAS GEMM."""
    if torch.compiler.is_compiling():
        raise RuntimeError(f"{FLAG}=1 serves an eager-only lane")
    if (x.ndim != 2 or x.shape[1] != prepared.weight.shape[1]
            or x.device != prepared.weight.device or x.dtype != torch.bfloat16
            or not x.is_contiguous()):
        raise ValueError("BF16 prefill requires contiguous BF16 x on the scratch device")
    workspace = prepared._workspace
    with torch.cuda.device(prepared.weight.device), workspace.lock:
        capture = torch.cuda.is_current_stream_capturing()
        stream = torch.cuda.current_stream(prepared.weight.device)
        if not capture:
            stream.wait_event(workspace.done)
            workspace.weight.record_stream(stream)
        try:
            _decode(prepared)
            return torch.mm(x, workspace.weight_t)
        finally:
            if not capture:
                workspace.done.record(stream)
