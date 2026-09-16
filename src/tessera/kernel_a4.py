"""Native E2M1 span-2 (A4) packed compute: fused decode + block-scaled FP4 MMA.

One family of kernels consumes the packing `lane_planes.prepare_span2_planes`
publishes -- select/label/point planes, the LUT scale nibbles, the 16-entry
E4M3 table, the label LUT and the subset nibbles -- and executes the
``e2m1_group16_ue4m3_static`` contract directly:

    y[m,n] = sum_k A_code[m,k]*a_sf[m,k//16] * W_code[n,k]*w_sf[n,k//16]
    out    = y * (unit.global_scale / input_global_scale)

The weight codes and block scales are decoded **inside** the kernel from the
packed planes; no decoded tile is written to global memory, at load or in a
forward.  The multiply is the hardware block-scaled FP4 MMA
(``mma.sync...kind::mxf4nvf4``), the same instruction class the stock NVFP4
lane executes on this part; ``require_native_fp4_mma`` proves the instruction is
emitted by the installed Triton for this device and refuses otherwise -- a
lowering that silently fell back to bf16 emulation is a refusal here, never a
slow serve.

WHY THE DECODE IS CHEAP ENOUGH TO FUSE.  The span-2 pair is the decode unit:
for one column ``k`` and one pair of codes (rows ``4p..4p+3`` at arity 2) the
kernel reads the pair's seven-bit select window (``label_lut`` gives the
super-label), the stored label of position 1, the pair's two point fields (one
aligned 16-bit read at this wire's R=7), and turns each (label, point) into its
code's two E2M1 nibbles with one byte lookup into ``code_nibbles`` --
``build_code_nibbles``' table over the unit's own ``subset_nibbles``, the same
codes the reference decoder writes, checked entry by entry in the tests.  At
R=7 the pair is 64x64 possible (point, point) cells per label pair, so one
fused pair table would be 64 KiB of working set per unit; two 256-entry code
lookups are the L1-resident spelling of the same table.

The four nibbles a pair decode returns are rows ``4p..4p+3`` of ONE column,
while the MMA operand is packed along K (column pairs, both nibbles for one
row in one byte).  ``tl.interleave`` rearranges the four row-bytes into
``[K/2, N]`` without a memory round trip.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Mapping

import torch

from .errors import GrammarError

__all__ = [
    "A4Unit",
    "A4UnitStack",
    "a4_quantize_activation",
    "a4_span2_gemm",
    "a4_span2_gemv",
    "a4_span2_grouped_gemm",
    "a4_decode_span2_tile",
    "build_code_nibbles",
    "native_fp4_backend",
    "native_fp4_mma_ptx_tokens",
    "require_native_fp4_mma",
]

#: Triton builds that can lower block-scaled FP4 MMA for sm_121a, in preference
#: order.  ``triton`` itself is listed last: on the pinned serve image its
#: ``make_ttgir`` refuses the operation outright, and a build that cannot lower
#: it is refused by ``require_native_fp4_mma`` rather than silently emulated.
_FP4_TRITON_PREFERENCE = ("tokenspeed_triton", "triton")


def _load_triton():
    """``(triton, tl, name)`` of the first importable build, or ``None``."""
    for name in _FP4_TRITON_PREFERENCE:
        try:
            triton = importlib.import_module(name)
            tl = importlib.import_module(f"{name}.language")
        except Exception:  # noqa: BLE001 -- absence is the only signal
            continue
        return triton, tl, name
    return None


_LOADED = _load_triton()
_TRITON, _TL, _TRITON_NAME = _LOADED if _LOADED is not None else (None, None, None)

#: ``{"error": str}`` after a refused probe, ``{"ok": True, ...}`` after a pass.
_probe_cache: "dict" = {}


def native_fp4_backend() -> "str | None":
    """The Triton build these kernels would use, or ``None`` if none imports."""
    return _TRITON_NAME


def native_fp4_mma_ptx_tokens() -> "list[str]":
    """The FP4-relevant instructions the probe kernel's PTX carries."""
    require_native_fp4_mma("native_fp4_mma_ptx_tokens")
    return list(_probe_cache["tokens"])


def _require_kernels_defined(context: str):
    if _TRITON is None:
        raise GrammarError(
            f"{context}: no Triton is importable; the A4 span-2 kernels need a "
            f"build that lowers block-scaled FP4 MMA (tried "
            f"{list(_FP4_TRITON_PREFERENCE)})")


def require_native_fp4_mma(context: str) -> None:
    """Refuse, by name, a launch whose backend would not execute FP4 MMA."""
    _require_kernels_defined(context)
    if _probe_cache.get("error") is not None:
        raise GrammarError(f"{context}: {_probe_cache['error']}")
    if _probe_cache.get("ok"):
        return
    # The capability probe is deliberately the simplest possible known
    # answer: every E2M1 code is 1.0 (packed 0x22), every E4M3 scale is 1.0,
    # so a correct m16n8k64 e2m1 x e2m1 block-scaled MMA returns exactly K on
    # every output element.  No reference packing is involved, so a failure
    # cannot be a harness-shaped bug reading as a hardware one.
    try:
        dev = torch.device("cuda")
        bk = 64
        a = torch.full((16, bk // 2), 0x22, dtype=torch.uint8, device=dev)
        b = torch.full((bk // 2, 16), 0x22, dtype=torch.uint8, device=dev)
        ones = torch.full((16, bk // 16), 0x38, dtype=torch.uint8,
                          device=dev).view(torch.float8_e4m3fn)
        out = torch.empty((16, 16), dtype=torch.float32, device=dev)
        _fp4_mma_probe_kernel[(1, 1)](a, ones, b, ones, out, BM=16, BN=16, BK=bk)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 -- the capability refusal
        message = (
            f"{_TRITON_NAME} cannot compile or run block-scaled FP4 MMA for this "
            f"device ({type(exc).__name__}: {exc}); the A4 span-2 native kernels "
            "refuse rather than emulating the multiply in bf16")
        _probe_cache["error"] = message
        raise GrammarError(f"{context}: {message}") from exc
    if not bool(torch.allclose(out, torch.full_like(out, float(bk)),
                               rtol=0.0, atol=1e-4)):
        message = (
            f"{_TRITON_NAME}'s block-scaled FP4 MMA failed its known-answer gate: "
            f"ones x ones over K={bk} must return {bk} everywhere, got "
            f"[{float(out.min())}, {float(out.max())}]")
        _probe_cache["error"] = message
        raise GrammarError(f"{context}: {message}")
    try:
        cached = _fp4_mma_probe_kernel.device_caches[torch.cuda.current_device()][0]
        compiled = list(cached.values())[-1]
        ptx = compiled.asm["ptx"]
    except Exception as exc:  # noqa: BLE001 -- the PTX read is the gate's own step
        message = (
            f"{_TRITON_NAME} ran the FP4 MMA probe but its PTX could not be read "
            f"({type(exc).__name__}: {exc}); the A4 lane holds hardware execution "
            "to the emitted instruction")
        _probe_cache["error"] = message
        raise GrammarError(f"{context}: {message}") from exc
    tokens = [token for token in ("mxf4nvf4", "mma.sync", "fma.rn.f32") if token in ptx]
    if "mxf4nvf4" not in tokens:
        message = (
            f"{_TRITON_NAME} answered the known-answer probe but its PTX carries "
            f"no block-scaled FP4 MMA instruction (tokens: {tokens}); the A4 lane "
            "refuses an emulated multiply")
        _probe_cache["error"] = message
        raise GrammarError(f"{context}: {message}")
    _probe_cache.update(ok=True, tokens=tokens, known_answer=float(bk))
    return None


def _unit_dtype(tensor: torch.Tensor, context: str) -> torch.Tensor:
    """The packed/scale tensors as the uint8/e4m3 bytes the kernels address."""
    if tensor.dtype in (torch.uint8, torch.float8_e4m3fn):
        return tensor
    if tensor.dtype == torch.int32:
        return tensor.view(torch.uint8)
    raise GrammarError(f"{context}: expected uint8/e4m3 bytes, got {tensor.dtype}")


# ---------------------------------------------------------------------------
# The plane bundle the kernels take
# ---------------------------------------------------------------------------


def build_code_nibbles(subset_nibbles: torch.Tensor, points: int, arity: int) -> torch.Tensor:
    """``(label, point) -> uint8``: a code's two E2M1 nibbles, low row first.

    ``subset_nibbles`` is ``build_subset_nibbles``' table in subset order:
    ``(label * points + point) * arity + a`` holds the code of the ``a``-th
    row of that (label, point).  A code's two rows are adjacent, so they are
    one byte here: ``nibble(a=0) | nibble(a=1) << 4``.

    Derived from ``subset_nibbles`` rather than rebuilt from the forest so the
    kernel and the reference decode cannot disagree about a code -- including
    the two spellings of zero E2M1 has (``build_subset_nibbles`` reads the
    anchor's CODE, which is where +0.0 vs -0.0 is decided).

    The table is ``4 * points`` bytes: 256 B at the shipping R=7 A4 wire, which
    is why the pair's two codes are two lookups rather than one table over
    ``(lab0, lab1, pt0, pt1)`` (64 KiB at R=7 -- a working set no L1 wants).
    """
    subset_nibbles = torch.as_tensor(subset_nibbles).to(torch.int64)
    arity = int(arity)
    points = int(points)
    if arity != 2:
        raise GrammarError(
            f"a span-2 code table is defined for arity 2 (the shipping E2M1x2 "
            f"grid); this unit has arity {arity}")
    if subset_nibbles.numel() != 4 * points * arity:
        raise GrammarError(
            f"a label has four values, so the subset table holds 4*{points}*{arity}"
            f" = {4 * points * arity} entries; got {subset_nibbles.numel()}")
    sub = subset_nibbles.reshape(4, points, arity)
    return (sub[:, :, 0] | (sub[:, :, 1] << 4)).reshape(-1).to(torch.uint8).contiguous()


@dataclass(frozen=True)
class A4UnitStack:
    """``E`` same-geometry span-2 bundles for the grouped operator.

    One allocation per plane kind on the expert axis (the shape
    ``moe_route._RankLocalPackedIntake`` already builds), so a grouped launch
    reads every expert's plane through the same base pointer plus ``e *
    stride``.  Heterogeneous geometry refuses: the grouped kernel takes one
    ``rows/cols/rate/arity/memory/half`` for the whole expert axis.
    """

    select: torch.Tensor
    label: torch.Tensor
    point: torch.Tensor
    nibbles: torch.Tensor
    lut_bytes: torch.Tensor
    label_lut: torch.Tensor
    code_nibbles: torch.Tensor
    globals: torch.Tensor
    rows: int
    cols: int
    rate: int
    arity: int
    memory: int
    half: int

    @classmethod
    def stack(cls, units) -> "A4UnitStack":
        units = list(units)
        if not units:
            raise GrammarError("an expert stack needs at least one unit")
        first = units[0]
        for unit in units:
            if (unit.rows, unit.cols, unit.rate, unit.arity, unit.memory,
                    unit.half) != (first.rows, first.cols, first.rate, first.arity,
                                   first.memory, first.half):
                raise GrammarError(
                    "a grouped launch takes one geometry for the expert axis; "
                    f"{first.rows}x{first.cols} R{first.rate} and "
                    f"{unit.rows}x{unit.cols} R{unit.rate} are different units")
        fields = ("select", "label", "point", "nibbles", "lut_bytes", "label_lut",
                  "code_nibbles")
        return cls(
            **{name: torch.stack([getattr(unit, name) for unit in units])
               for name in fields},
            globals=torch.tensor([unit.global_scale for unit in units],
                                 dtype=torch.float32, device=first.select.device),
            rows=first.rows, cols=first.cols, rate=first.rate, arity=first.arity,
            memory=first.memory, half=first.half,
        )

    @property
    def experts(self) -> int:
        return int(self.globals.numel())

    def _check(self, context: str) -> None:
        if self.arity != 2:
            raise GrammarError(f"{context}: the span-2 pair decode needs arity 2")
        if self.half != 16:
            raise GrammarError(f"{context}: the LUT plane is one nibble per 16 columns")
        if self.code_nibbles.shape[1] != 4 * (1 << (self.rate - 1)):
            raise GrammarError(
                f"{context}: the code table holds 4*points = "
                f"{4 * (1 << (self.rate - 1))} entries")
        field = self.rate - 1
        stride = (self.rows // self.arity) * field
        divisor = math.gcd(math.gcd(stride, 2 * field), 8)
        worst = 0 if divisor >= 8 else 8 - divisor
        if worst + 2 * field > 16:
            raise GrammarError(
                f"{context}: the point fields can start at bit {worst} mod 8 and "
                f"two {field}-bit points need {worst + 2 * field} bits; the "
                "span-2 decode reads 16 and does not widen silently")


@dataclass(frozen=True)
class A4Unit:
    """One rank-local span-2 tile's residency bundle, ready for the kernels.

    Constructed from ``lane_planes.prepare_span2_planes``' dict, which is the
    single plane-layout authority: no byte here is re-derived, only
    reinterpreted (the scale table as E4M3 bytes) or compiled (the pair table).
    ``global_scale`` is the shared/joined LUT global the loader's
    ``shared_lut_global`` move produced; the caller passes it explicitly when
    the tile is a member of a fused module whose global was moved.
    """

    select: torch.Tensor
    label: torch.Tensor
    point: torch.Tensor
    nibbles: torch.Tensor
    lut_bytes: torch.Tensor
    label_lut: torch.Tensor
    subset_nibbles: torch.Tensor
    code_nibbles: torch.Tensor
    rows: int
    cols: int
    rate: int
    arity: int
    memory: int
    half: int
    global_scale: float

    @classmethod
    def from_prepared(cls, prepared: Mapping, global_scale: "float | None" = None) -> "A4Unit":
        required = ("select", "label", "point", "nibbles", "label_lut",
                    "subset_nibbles", "global_scale", "rows", "cols", "rate",
                    "arity", "memory", "half")
        missing = [name for name in required if name not in prepared]
        if missing:
            raise GrammarError(
                f"an A4 unit needs the prepared span-2 bundle; missing {missing}")
        rate = int(prepared["rate"])
        arity = int(prepared["arity"])
        rows = int(prepared["rows"])
        if rows % (2 * arity):
            raise GrammarError(
                f"a span-2 tile holds whole pairs of codes; {rows} rows at arity "
                f"{arity} is not a multiple of {2 * arity}")
        return cls(
            select=_unit_dtype(prepared["select"], "select plane"),
            label=_unit_dtype(prepared["label"], "label plane"),
            point=_unit_dtype(prepared["point"], "point plane"),
            nibbles=_unit_dtype(prepared["nibbles"], "LUT scale nibbles"),
            lut_bytes=torch.as_tensor(prepared["lut_bytes"]).view(
                torch.float8_e4m3fn).contiguous(),
            label_lut=torch.as_tensor(prepared["label_lut"]).to(torch.int32).contiguous(),
            subset_nibbles=_unit_dtype(prepared["subset_nibbles"], "subset nibbles"),
            code_nibbles=build_code_nibbles(prepared["subset_nibbles"], 1 << (rate - 1), arity),
            rows=rows,
            cols=int(prepared["cols"]),
            rate=rate,
            arity=arity,
            memory=int(prepared["memory"]),
            half=int(prepared["half"]),
            global_scale=float(prepared["global_scale"] if global_scale is None
                               else global_scale),
        )

    def epilogue_for(self, input_global_scale: torch.Tensor) -> torch.Tensor:
        """``[1]`` fp32 device tensor: ``global_scale / input_global_scale``.

        Computed once at preparation.  The kernels read the epilogue from the
        device, so a forward never reads a tensor's contents on the host and
        the call stays inside a CUDA graph.
        """
        gscale = torch.as_tensor(input_global_scale)
        if gscale.numel() != 1:
            raise GrammarError(
                "the A-side static global is one scalar per GEMM; got "
                f"{gscale.numel()} values")
        return (torch.full((1,), float(self.global_scale), dtype=torch.float32,
                           device=gscale.device) / gscale.to(torch.float32).reshape(1))

    def to(self, device) -> "A4Unit":
        """The bundle on ``device`` (every plane and table moves with it)."""
        fields = ("select", "label", "point", "nibbles", "lut_bytes", "label_lut",
                  "subset_nibbles", "code_nibbles")
        return A4Unit(**{name: getattr(self, name).to(device) for name in fields},
                      rows=self.rows, cols=self.cols, rate=self.rate, arity=self.arity,
                      memory=self.memory, half=self.half, global_scale=self.global_scale)

    def _check(self, context: str) -> None:
        if self.arity != 2:
            raise GrammarError(
                f"{context}: the span-2 pair decode is defined for arity 2 (the "
                f"shipping E2M1x2 grid); this unit is arity {self.arity}")
        if self.half != 16:
            raise GrammarError(
                f"{context}: the LUT scale plane is one nibble per 16 columns; "
                f"this unit's group is {self.half}")
        if self.cols % self.half:
            raise GrammarError(f"{context}: {self.cols} columns is not whole groups")
        if self.code_nibbles.numel() != (4 * (1 << (self.rate - 1))):
            raise GrammarError(
                f"{context}: the code table holds 4*points = "
                f"{4 * (1 << (self.rate - 1))} entries; got {self.code_nibbles.numel()}")
        # The pair's two point fields are one 16-bit read in the kernel.  The
        # fields start at absolute bit t and t + (rate-1) inside the column; the
        # read opens on t's byte, so the pair must fit behind that byte's
        # offset.  Both fields' shifts are absolute (never taken mod 8): a
        # wrapped shift drops the carry of the second field.
        field = self.rate - 1
        stride = (self.rows // self.arity) * field
        divisor = math.gcd(math.gcd(stride, 2 * field), 8)
        worst = 0 if divisor >= 8 else 8 - divisor
        if worst + 2 * field > 16:
            raise GrammarError(
                f"{context}: this column's point fields can start at bit {worst} "
                f"mod 8 and two {field}-bit points need "
                f"{worst + 2 * field} bits; the span-2 decode reads 16 and does "
                "not widen silently")
        if self.label_lut.numel() != (1 << (self.memory + 1)):
            raise GrammarError(
                f"{context}: the label LUT holds 2^(memory+1) = "
                f"{1 << (self.memory + 1)} windows; got {self.label_lut.numel()}")
        span = self.cols * max(self.rows // self.arity * 2,
                               self.rows // self.arity + 8)
        if span >= (1 << 31) - 8:
            raise GrammarError(
                f"{context}: this tile's largest plane bit index is {span}, which "
                "the int32 addressing these kernels use cannot hold")


# ---------------------------------------------------------------------------
# A-side: the static-global group-16 E2M1 quantizer (vLLM's own op)
# ---------------------------------------------------------------------------


def a4_quantize_activation(x: torch.Tensor, input_global_scale: torch.Tensor):
    """``(packed [M, K/2] uint8, scales [M, K//16] e4m3)`` in LINEAR layout.

    This calls the runtime's registered ``scaled_fp4_quant`` -- the same
    operator the stock NVFP4 route executes -- with the scale factors in the
    row-major layout the kernels here address.  The static global is the
    caller's checkpoint fact (``capacity / amax``); nothing is re-derived.
    """
    if x.device.type != "cuda" or x.dim() != 2:
        raise GrammarError("A4 activation quantization needs a 2-D CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise GrammarError(
            f"A4 activation quantization takes bf16/fp16 activations, got {x.dtype}")
    input_global_scale = torch.as_tensor(input_global_scale)
    if input_global_scale.numel() != 1 or input_global_scale.device != x.device:
        raise GrammarError(
            "A4 activation quantization needs one global scale on the input device")
    if x.shape[1] % 16:
        raise GrammarError(
            f"A4 activation quantization needs K divisible by 16; got {x.shape[1]}")
    if not callable(getattr(torch.ops._C, "scaled_fp4_quant", None)):
        try:
            import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
        except Exception as exc:  # noqa: BLE001 -- one diagnosis for every cause
            raise GrammarError(
                "the A4 route's A side is the runtime's registered "
                f"scaled_fp4_quant; this build cannot register it ({type(exc).__name__}: "
                f"{exc})") from exc
        if not callable(getattr(torch.ops._C, "scaled_fp4_quant", None)):
            raise GrammarError(
                "the A4 route's A side is the runtime's registered scaled_fp4_quant; "
                "this build does not register it")
    packed, scale = torch.ops._C.scaled_fp4_quant(
        x.contiguous(), input_global_scale.to(torch.float32), False)
    return packed.view(torch.uint8), scale.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Kernels (defined only where a Triton is importable)
# ---------------------------------------------------------------------------

if _TL is not None:

    @_TRITON.jit
    def _fp4_mma_probe_kernel(a_ptr, sa_ptr, b_ptr, sb_ptr, out_ptr,
                              BM: _TL.constexpr, BN: _TL.constexpr, BK: _TL.constexpr):
        """One tiny e2m1 x e2m1 block-scaled dot; its PTX is the capability gate."""
        offs_m = _TL.arange(0, BM)
        offs_n = _TL.arange(0, BN)
        kp = _TL.arange(0, BK // 2)
        kq = _TL.arange(0, BK // 16)
        a = _TL.load(a_ptr + offs_m[:, None] * (BK // 2) + kp[None, :])
        b = _TL.load(b_ptr + kp[:, None] * BN + offs_n[None, :])
        sa = _TL.load(sa_ptr + offs_m[:, None] * (BK // 16) + kq[None, :])
        sb = _TL.load(sb_ptr + offs_n[:, None] * (BK // 16) + kq[None, :])
        acc = _TL.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
        _TL.store(out_ptr + offs_m[:, None] * BN + offs_n[None, :], acc)

    @_TRITON.jit
    def _decode_pair_states(select_ptr, label_ptr, point_ptr, code_ptr, label_lut_ptr,
                            k, p, pairs, steps,
                            MEMORY: _TL.constexpr, PAD: _TL.constexpr,
                            RATE: _TL.constexpr, POINTS: _TL.constexpr):
        """Every intermediate of one (column, row-pair) span-2 decode.

        ``k`` is the column (broadcast against ``p``, the pair index).  The
        select window is the ``memory+1`` bits ending at the pair's own select
        bit, read as a big-endian 16-bit field; ``label_lut`` maps it to the
        pair's super-label; position 1's label is stored, position 0's is the
        super-label minus it mod 4.  The pair's two ``RATE-1``-bit points are
        one 16-bit read, both shifts absolute within that read (the caller's
        admission holds their span behind the byte offset).  Returns
        ``(window, ell, stored, lab0, lab1, pt0, pt1, code0, code1)`` -- the
        last two are the row-pair's two code bytes, rows ``4p, 4p+1`` and
        ``4p+2, 4p+3``.
        """
        # The window's anchor is the OLDEST of its memory+1 bits:
        # select_base = k*(pairs+PAD) + PAD - memory, then the pair's own bit
        # is the newest -- the order lane_planes._thread_start_state writes the
        # pad in and build_span2_luts reads a window in (newest bit on top).
        q = k * (pairs + PAD) + PAD - MEMORY + p
        byte = q // 8
        b0 = _TL.load(select_ptr + byte).to(_TL.int32)
        b1 = _TL.load(select_ptr + byte + 1).to(_TL.int32)
        window = (((b0 << 8) | b1) >> (15 - MEMORY - (q % 8))) & ((1 << (MEMORY + 1)) - 1)
        ell = _TL.load(label_lut_ptr + window).to(_TL.int32)
        lab_byte = _TL.load(label_ptr + (k * (pairs * 2) + p * 2) // 8).to(_TL.int32)
        stored = (lab_byte >> (6 - (p * 2) % 8)) & 3
        lab0 = (ell - stored) & 3
        field = RATE - 1
        t = k * (steps * field) + p * (2 * field)
        p0 = _TL.load(point_ptr + t // 8).to(_TL.int32)
        p1 = _TL.load(point_ptr + t // 8 + 1).to(_TL.int32)
        u = (p0 << 8) | p1
        pt0 = (u >> (16 - field - (t % 8))) & (POINTS - 1)
        pt1 = (u >> (16 - 2 * field - (t % 8))) & (POINTS - 1)
        lab1 = stored
        code0 = _TL.load(code_ptr + lab0 * POINTS + pt0)
        code1 = _TL.load(code_ptr + lab1 * POINTS + pt1)
        return window, ell, stored, lab0, lab1, pt0, pt1, code0, code1

    @_TRITON.jit
    def _decode_pair_column(select_ptr, label_ptr, point_ptr, code_ptr, label_lut_ptr,
                            k, p, pairs, steps,
                            MEMORY: _TL.constexpr, PAD: _TL.constexpr,
                            RATE: _TL.constexpr, POINTS: _TL.constexpr):
        """``([.., P] uint8, [.., P] uint8)``: a column pair's two code bytes."""
        _w, _e, _s, _l0, _l1, _p0, _p1, code0, code1 = _decode_pair_states(
            select_ptr, label_ptr, point_ptr, code_ptr, label_lut_ptr, k, p, pairs,
            steps, MEMORY=MEMORY, PAD=PAD, RATE=RATE, POINTS=POINTS)
        return code0, code1

    @_TRITON.jit
    def _a4_span2_gemm_kernel(
        a_ptr, a_scale_ptr, select_ptr, label_ptr, point_ptr, nibbles_ptr, lut_ptr,
        label_lut_ptr, code_ptr, out_ptr, epilogue_ptr,
        M, rows, cols,
        MEMORY: _TL.constexpr, PAD: _TL.constexpr, RATE: _TL.constexpr,
        BM: _TL.constexpr, BN: _TL.constexpr, BK: _TL.constexpr,
    ):
        """``x @ W.T``, W4A4, decoding the span-2 planes in the mainloop."""
        POINTS: _TL.constexpr = 1 << (RATE - 1)
        steps = rows // 2
        pairs = steps // 2
        pid_m = _TL.program_id(0)
        pid_n = _TL.program_id(1)
        offs_m = pid_m * BM + _TL.arange(0, BM)
        offs_n = pid_n * BN + _TL.arange(0, BN)
        live_m = offs_m < M
        i = _TL.arange(0, BK // 2)
        p = _TL.arange(0, BN // 4)
        acc = _TL.zeros((BM, BN), dtype=_TL.float32)
        for k0 in range(0, cols, BK):
            kA = (k0 + 2 * i)[:, None]
            pv = (pid_n * (BN // 4) + p)[None, :]
            na0, na1 = _decode_pair_column(select_ptr, label_ptr, point_ptr, code_ptr,
                                           label_lut_ptr, kA, pv, pairs, steps,
                                           MEMORY=MEMORY, PAD=PAD, RATE=RATE, POINTS=POINTS)
            nb0, nb1 = _decode_pair_column(select_ptr, label_ptr, point_ptr, code_ptr,
                                           label_lut_ptr, kA + 1, pv, pairs, steps,
                                           MEMORY=MEMORY, PAD=PAD, RATE=RATE, POINTS=POINTS)
            r0 = ((na0 & 0xF) | ((nb0 & 0xF) << 4)).to(_TL.uint8)
            r1 = ((na0 >> 4) | (nb0 & 0xF0)).to(_TL.uint8)
            r2 = ((na1 & 0xF) | ((nb1 & 0xF) << 4)).to(_TL.uint8)
            r3 = ((na1 >> 4) | (nb1 & 0xF0)).to(_TL.uint8)
            w = _TL.interleave(_TL.interleave(r0, r2), _TL.interleave(r1, r3))

            g = _TL.arange(0, BK // 16)
            gn = (k0 // 16 + g)[None, :] * rows + offs_n[:, None]
            nb = _TL.load(nibbles_ptr + gn // 2).to(_TL.int32)
            nib = (nb >> (4 * (1 - (gn & 1)))).to(_TL.int32) & 0xF
            sw = _TL.load(lut_ptr + nib).to(_TL.float8e4nv, bitcast=True)

            kp = _TL.arange(0, BK // 2)
            a = _TL.load(a_ptr + offs_m[:, None] * (cols // 2) + (k0 // 2 + kp)[None, :],
                         mask=live_m[:, None], other=0)
            # The scale is loaded as its byte and bitcast: a masked fp8 load
            # cannot take an integer `other`, and a masked-out row must be a
            # zero scale, which byte 0 is (e4m3 +0.0).
            sa = _TL.load(a_scale_ptr + offs_m[:, None] * (cols // 16)
                          + (k0 // 16 + g)[None, :], mask=live_m[:, None],
                          other=0).to(_TL.float8e4nv, bitcast=True)
            acc = _TL.dot_scaled(a, sa, "e2m1", w, sw, "e2m1", acc)
        epilogue = _TL.load(epilogue_ptr)
        _TL.store(out_ptr + offs_m[:, None] * rows + offs_n[None, :],
                  acc * epilogue, mask=live_m[:, None])

    @_TRITON.jit
    def _a4_span2_grouped_kernel(
        a_ptr, a_scale_ptr, token_ids_ptr, expert_start_ptr,
        select_ptr, label_ptr, point_ptr, nibbles_ptr, lut_ptr, label_lut_ptr,
        code_ptr, epilogue_ptr, partial_ptr,
        rows, cols, num_routes,
        SELECT_STRIDE, LABEL_STRIDE, POINT_STRIDE, NIB_STRIDE, CODE_STRIDE,
        LUT_STRIDE, LUTLUT_STRIDE,
        MEMORY: _TL.constexpr, PAD: _TL.constexpr, RATE: _TL.constexpr,
        BM: _TL.constexpr, BN: _TL.constexpr, BK: _TL.constexpr,
    ):
        """Grouped W4A4: one launch over the expert axis, rows gathered by token.

        ``expert_start`` is the CSR offset of each expert's run in
        ``token_ids`` (device-built by the runtime's dispatch, as
        ``moe_align_block_size`` already does); every program whose row block
        starts past its expert's count exits.  Output rows are the ROUTING
        rows in dispatch order, one expert-scoped GEMM each: the caller keeps
        them per route through the gate/up activation and the down projection,
        and applies routing weights only where the runtime contract says so
        (routed-MoE combines after the down projection, not before).
        """
        POINTS: _TL.constexpr = 1 << (RATE - 1)
        steps = rows // 2
        pairs = steps // 2
        e = _TL.program_id(0)
        pid_m = _TL.program_id(1)
        pid_n = _TL.program_id(2)
        start = _TL.load(expert_start_ptr + e)
        count = _TL.load(expert_start_ptr + e + 1) - start
        m0 = pid_m * BM
        if m0 >= count:
            return
        row_ids = m0 + _TL.arange(0, BM)
        live_m = row_ids < count
        tokens = _TL.load(token_ids_ptr + start + row_ids, mask=live_m, other=0)
        epilogue = _TL.load(epilogue_ptr + e)

        sel = select_ptr + e * SELECT_STRIDE
        lab_p = label_ptr + e * LABEL_STRIDE
        pt_p = point_ptr + e * POINT_STRIDE
        code_p = code_ptr + e * CODE_STRIDE
        lut_lut = label_lut_ptr + e * LUTLUT_STRIDE
        nib_p = nibbles_ptr + e * NIB_STRIDE
        lut_p = lut_ptr + e * LUT_STRIDE

        offs_n = pid_n * BN + _TL.arange(0, BN)
        i = _TL.arange(0, BK // 2)
        p = _TL.arange(0, BN // 4)
        acc = _TL.zeros((BM, BN), dtype=_TL.float32)
        for k0 in range(0, cols, BK):
            kA = (k0 + 2 * i)[:, None]
            pv = (pid_n * (BN // 4) + p)[None, :]
            na0, na1 = _decode_pair_column(sel, lab_p, pt_p, code_p, lut_lut, kA, pv,
                                           pairs, steps, MEMORY=MEMORY, PAD=PAD,
                                           RATE=RATE, POINTS=POINTS)
            nb0, nb1 = _decode_pair_column(sel, lab_p, pt_p, code_p, lut_lut, kA + 1, pv,
                                           pairs, steps, MEMORY=MEMORY, PAD=PAD,
                                           RATE=RATE, POINTS=POINTS)
            r0 = ((na0 & 0xF) | ((nb0 & 0xF) << 4)).to(_TL.uint8)
            r1 = ((na0 >> 4) | (nb0 & 0xF0)).to(_TL.uint8)
            r2 = ((na1 & 0xF) | ((nb1 & 0xF) << 4)).to(_TL.uint8)
            r3 = ((na1 >> 4) | (nb1 & 0xF0)).to(_TL.uint8)
            w_tile = _TL.interleave(_TL.interleave(r0, r2), _TL.interleave(r1, r3))

            g = _TL.arange(0, BK // 16)
            gn = (k0 // 16 + g)[None, :] * rows + offs_n[:, None]
            nb = _TL.load(nib_p + gn // 2).to(_TL.int32)
            nib = (nb >> (4 * (1 - (gn & 1)))).to(_TL.int32) & 0xF
            sw = _TL.load(lut_p + nib).to(_TL.float8e4nv, bitcast=True)

            kp = _TL.arange(0, BK // 2)
            a = _TL.load(a_ptr + tokens[:, None] * (cols // 2) + (k0 // 2 + kp)[None, :],
                         mask=live_m[:, None], other=0)
            sa = _TL.load(a_scale_ptr + tokens[:, None] * (cols // 16)
                          + (k0 // 16 + g)[None, :], mask=live_m[:, None],
                          other=0).to(_TL.float8e4nv, bitcast=True)
            acc = _TL.dot_scaled(a, sa, "e2m1", w_tile, sw, "e2m1", acc)
        _TL.store(partial_ptr + (start + row_ids)[:, None] * rows + offs_n[None, :],
                  acc * epilogue, mask=live_m[:, None])

    @_TRITON.jit
    def _a4_state_dump_kernel(select_ptr, label_ptr, point_ptr, code_ptr, label_lut_ptr,
                              k_ptr, p_ptr, pairs, steps,
                              window_ptr, ell_ptr, stored_ptr, lab0_ptr, lab1_ptr,
                              pt0_ptr, pt1_ptr, code0_ptr, code1_ptr, n,
                              MEMORY: _TL.constexpr, PAD: _TL.constexpr,
                              RATE: _TL.constexpr, POINTS: _TL.constexpr,
                              BLOCK: _TL.constexpr):
        """Bounded diagnostic: the pair decode's intermediates at listed indices."""
        idx = _TL.program_id(0) * BLOCK + _TL.arange(0, BLOCK)
        live = idx < n
        k = _TL.load(k_ptr + idx, mask=live, other=0)
        p = _TL.load(p_ptr + idx, mask=live, other=0)
        window, ell, stored, lab0, lab1, pt0, pt1, code0, code1 = _decode_pair_states(
            select_ptr, label_ptr, point_ptr, code_ptr, label_lut_ptr, k, p, pairs,
            steps, MEMORY=MEMORY, PAD=PAD, RATE=RATE, POINTS=POINTS)
        _TL.store(window_ptr + idx, window.to(_TL.int32), mask=live)
        _TL.store(ell_ptr + idx, ell, mask=live)
        _TL.store(stored_ptr + idx, stored, mask=live)
        _TL.store(lab0_ptr + idx, lab0, mask=live)
        _TL.store(lab1_ptr + idx, lab1, mask=live)
        _TL.store(pt0_ptr + idx, pt0, mask=live)
        _TL.store(pt1_ptr + idx, pt1, mask=live)
        _TL.store(code0_ptr + idx, code0.to(_TL.int32), mask=live)
        _TL.store(code1_ptr + idx, code1.to(_TL.int32), mask=live)

    @_TRITON.jit
    def _a4_decode_dump_kernel(
        select_ptr, label_ptr, point_ptr, nibbles_ptr, lut_ptr, label_lut_ptr, code_ptr,
        va_ptr, vb_ptr, scale_ptr, rows, cols,
        MEMORY: _TL.constexpr, PAD: _TL.constexpr, RATE: _TL.constexpr,
        KP: _TL.constexpr, PP: _TL.constexpr,
    ):
        """Test oracle: the pair decode as columns of nibble pairs, plus scales.

        ``va``/``vb`` are ``[cols // 2, pairs]`` uint16 (columns 2i and 2i+1);
        ``scale`` is ``[rows, cols // 16]`` E4M3 bytes, the zero-padded stock
        layout ``materialize_stock`` writes.
        """
        POINTS: _TL.constexpr = 1 << (RATE - 1)
        steps = rows // 2
        pairs = steps // 2
        pid_k = _TL.program_id(0)
        pid_p = _TL.program_id(1)
        i = pid_k * KP + _TL.arange(0, KP)
        p = pid_p * PP + _TL.arange(0, PP)
        live_i = i < (cols // 2)
        live_p = p < pairs
        kA = (2 * i)[:, None]
        pv = p[None, :]
        mask = live_i[:, None] & live_p[None, :]
        na0, na1 = _decode_pair_column(select_ptr, label_ptr, point_ptr, code_ptr,
                                       label_lut_ptr, kA, pv, pairs, steps,
                                       MEMORY=MEMORY, PAD=PAD, RATE=RATE, POINTS=POINTS)
        nb0, nb1 = _decode_pair_column(select_ptr, label_ptr, point_ptr, code_ptr,
                                       label_lut_ptr, kA + 1, pv, pairs, steps,
                                       MEMORY=MEMORY, PAD=PAD, RATE=RATE, POINTS=POINTS)
        va = na0.to(_TL.uint16) | (na1.to(_TL.uint16) << 8)
        vb = nb0.to(_TL.uint16) | (nb1.to(_TL.uint16) << 8)
        _TL.store(va_ptr + i[:, None] * pairs + p[None, :], va, mask=mask)
        _TL.store(vb_ptr + i[:, None] * pairs + p[None, :], vb, mask=mask)
        groups = cols // 16
        pid_g = _TL.program_id(2)
        n = pid_g * KP + _TL.arange(0, KP)
        live_n = n < rows
        # The group axis is walked in PP-wide steps: the grid's third axis
        # covers ROWS, not groups, and a unit has more groups than PP.
        for g0 in range(0, groups, PP):
            g = g0 + _TL.arange(0, PP)
            live = live_n[:, None] & (g[None, :] < groups)
            gn = g[None, :] * rows + n[:, None]
            nb = _TL.load(nibbles_ptr + gn // 2, mask=live, other=0).to(_TL.int32)
            nib = (nb >> (4 * (1 - (gn & 1)))).to(_TL.int32) & 0xF
            sc = _TL.load(lut_ptr + nib, mask=live, other=0).to(_TL.float8e4nv, bitcast=True)
            _TL.store(scale_ptr + n[:, None] * groups + g[None, :], sc, mask=live)


def _require_unit_tensor(unit: A4Unit, context: str) -> None:
    for name in ("select", "label", "point", "nibbles", "lut_bytes", "label_lut",
                 "subset_nibbles", "code_nibbles"):
        tensor = getattr(unit, name)
        if not tensor.is_cuda:
            raise GrammarError(f"{context}: the bundle's {name} is not on a CUDA device")
        if not tensor.is_contiguous():
            raise GrammarError(f"{context}: the bundle's {name} is not contiguous")


def a4_span2_gemm(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    unit: A4Unit,
    epilogue: "torch.Tensor | float",
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 128,
    out_dtype: torch.dtype = torch.float32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """``[M, rows]``: W4A4 over the unit's packed span-2 planes.

    ``a_packed``/``a_scale`` are ``a4_quantize_activation``'s output (packed
    along K, low nibble = even column; one E4M3 scale per 16 columns, linear
    layout).  ``epilogue`` is the ``[1]`` fp32 device tensor
    ``A4Unit.epilogue_for(input_global_scale)`` computed once at preparation;
    a float is accepted for development callers and is frozen here, but a
    captured forward must pass the tensor so no host read sits in the call.
    """
    require_native_fp4_mma("a4_span2_gemm")
    unit._check("a4_span2_gemm")
    _require_unit_tensor(unit, "a4_span2_gemm")
    if a_packed.dim() != 2 or a_packed.shape[1] * 2 != unit.cols:
        raise GrammarError(
            f"a4_span2_gemm: activation is {tuple(a_packed.shape)}, which is not "
            f"[M, {unit.cols // 2}] packed codes for {unit.cols} columns")
    if a_scale.shape != (a_packed.shape[0], unit.cols // 16):
        raise GrammarError(
            f"a4_span2_gemm: activation scales are {tuple(a_scale.shape)}, not "
            f"[{a_packed.shape[0]}, {unit.cols // 16}]")
    if unit.rows % block_n or unit.cols % block_k:
        raise GrammarError(
            f"a4_span2_gemm: the tile is {unit.rows}x{unit.cols} and the launch "
            f"takes {block_n} row / {block_k} column blocks; a remainder needs a "
            "masked variant this kernel does not have (fail closed)")
    a_packed = _unit_dtype(a_packed, "a4_span2_gemm activation codes").contiguous()
    # The kernel loads the scale planes as bytes and bitcasts in-kernel; a
    # masked-out scale must be a zero byte, which is e4m3 +0.0.
    a_scale = _unit_dtype(a_scale, "a4_span2_gemm activation scales")
    a_scale = a_scale.view(torch.uint8).contiguous()
    M = a_packed.shape[0]
    if not isinstance(epilogue, torch.Tensor):
        epilogue = unit.epilogue_for(
            torch.tensor([float(epilogue)], dtype=torch.float32, device=a_packed.device))
    if epilogue.numel() != 1 or epilogue.device != a_packed.device:
        raise GrammarError(
            "a4_span2_gemm: the epilogue is one fp32 value on the activation device")
    epilogue = epilogue.to(torch.float32).reshape(1).contiguous()
    out = torch.empty((M, unit.rows), dtype=out_dtype, device=a_packed.device)
    grid = (max(1, -(-M // block_m)), unit.rows // block_n)
    _a4_span2_gemm_kernel[grid](
        a_packed, a_scale, unit.select, unit.label, unit.point, unit.nibbles,
        unit.lut_bytes.view(torch.uint8), unit.label_lut, unit.code_nibbles, out, epilogue,
        M, unit.rows, unit.cols,
        MEMORY=unit.memory, PAD=8, RATE=unit.rate,
        BM=block_m, BN=block_n, BK=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def a4_span2_grouped_gemm(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    stack: A4UnitStack,
    epilogues: torch.Tensor,
    *,
    expert_offsets: torch.Tensor,
    token_ids: torch.Tensor,
    num_routes: "int | None" = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 128,
    out_dtype: torch.dtype = torch.float32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """``[num_routes, rows]``: one expert-scoped W4A4 GEMM per routing row.

    ``expert_offsets``/``token_ids`` are the device-built dispatch: a CSR run
    per expert over the routing rows, each row naming the token it belongs to.
    The kernel launches once over the expert axis (a program whose row block
    starts past its expert's run exits) and writes the routing rows in
    dispatch order.  It does NOT reduce onto tokens and does NOT apply routing
    weights: routed-MoE keeps each route's gate/up output through the
    activation and the down projection, and applies weights only where its
    contract says so (after the down projection).  ``epilogues`` is the
    ``[E]`` fp32 tensor of per-expert ``weight_global / input_global_scale``
    quotients, computed once at preparation -- per-expert activation globals
    keep their own value, as the stock MoE's per-expert input scales do.

    ``num_routes`` is the row-axis extent the caller's dispatch actually uses
    (repeated token ids admitted); it defaults to ``token_ids.numel()``.
    """
    require_native_fp4_mma("a4_span2_grouped_gemm")
    stack._check("a4_span2_grouped_gemm")
    for name in ("select", "label", "point", "nibbles", "lut_bytes", "label_lut",
                 "code_nibbles", "globals"):
        tensor = getattr(stack, name)
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise GrammarError(
                f"a4_span2_grouped_gemm: the stack's {name} must be contiguous CUDA")
    if a_packed.dim() != 2 or a_packed.shape[1] * 2 != stack.cols:
        raise GrammarError(
            f"a4_span2_grouped_gemm: activation is {tuple(a_packed.shape)}, not "
            f"[T, {stack.cols // 2}]")
    if a_scale.shape != (a_packed.shape[0], stack.cols // 16):
        raise GrammarError(
            f"a4_span2_grouped_gemm: activation scales are {tuple(a_scale.shape)}")
    if stack.rows % block_n or stack.cols % block_k:
        raise GrammarError(
            f"a4_span2_grouped_gemm: the tile is {stack.rows}x{stack.cols} and the "
            f"launch takes {block_n} row / {block_k} column blocks; fail closed")
    if expert_offsets.numel() != stack.experts + 1:
        raise GrammarError(
            f"a4_span2_grouped_gemm: expert_offsets holds {expert_offsets.numel()} "
            f"entries for {stack.experts} experts + 1")
    if num_routes is None:
        num_routes = int(token_ids.numel())
    if int(num_routes) < int(token_ids.numel()):
        raise GrammarError(
            f"a4_span2_grouped_gemm: the route axis is {num_routes} rows but the "
            f"dispatch names {token_ids.numel()}")
    if epilogues.numel() != stack.experts or epilogues.device != a_packed.device:
        raise GrammarError(
            f"a4_span2_grouped_gemm: one epilogue per expert on the activation "
            f"device ({stack.experts} experts, got {epilogues.numel()})")
    epilogues = epilogues.to(torch.float32).reshape(-1).contiguous()
    device = a_packed.device
    expert_offsets = expert_offsets.to(torch.int32).to(device).contiguous()
    token_ids = token_ids.to(torch.int32).to(device).contiguous()
    a_packed = _unit_dtype(a_packed, "a4_span2_grouped_gemm activation codes").contiguous()
    a_scale = _unit_dtype(a_scale, "a4_span2_grouped_gemm activation scales")
    a_scale = a_scale.view(torch.uint8).contiguous()
    partial = torch.empty((int(num_routes), stack.rows), dtype=torch.float32,
                          device=device)
    grid = (stack.experts, max(1, -(-int(num_routes) // block_m)), stack.rows // block_n)
    _a4_span2_grouped_kernel[grid](
        a_packed, a_scale, token_ids, expert_offsets,
        stack.select, stack.label, stack.point, stack.nibbles,
        stack.lut_bytes.view(torch.uint8), stack.label_lut, stack.code_nibbles,
        epilogues, partial,
        stack.rows, stack.cols, int(num_routes),
        int(stack.select.shape[1]), int(stack.label.shape[1]), int(stack.point.shape[1]),
        int(stack.nibbles.shape[1]), int(stack.code_nibbles.shape[1]),
        int(stack.lut_bytes.shape[1]), int(stack.label_lut.shape[1]),
        MEMORY=stack.memory, PAD=8, RATE=stack.rate,
        BM=block_m, BN=block_n, BK=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    if out_dtype != torch.float32:
        partial = partial.to(out_dtype)
    return partial


def a4_span2_gemv(x: torch.Tensor, unit: A4Unit, input_global_scale, *,
                  out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[rows]`` for one token: quantize the vector, then one M=1 GEMM."""
    if x.dim() != 1 or x.shape[0] != unit.cols:
        raise GrammarError(
            f"a4_span2_gemv: activation is {tuple(x.shape)} for a reduction over "
            f"{unit.cols} columns")
    gscale = torch.as_tensor(input_global_scale, dtype=torch.float32, device=x.device)
    packed, scale = a4_quantize_activation(
        x.to(torch.bfloat16).reshape(1, unit.cols), gscale)
    return a4_span2_gemm(packed, scale, unit, unit.epilogue_for(gscale),
                         out_dtype=out_dtype)[0]


def a4_decode_states_at(unit: A4Unit, ks: torch.Tensor, ps: torch.Tensor):
    """Diagnostic: every decode intermediate at the given ``(k, p)`` indices.

    Bounded, explicit-index companion to ``a4_decode_span2_tile``: returns nine
    int32 tensors ``(window, ell, stored, lab0, lab1, pt0, pt1, code0, code1)``
    for the listed column/pair pairs, so a mismatch can name which field read
    wrong instead of only that the tile differs.
    """
    require_native_fp4_mma("a4_decode_states_at")
    unit._check("a4_decode_states_at")
    _require_unit_tensor(unit, "a4_decode_states_at")
    if ks.numel() != ps.numel():
        raise GrammarError("a4_decode_states_at takes one pair index per column index")
    ks = ks.to(torch.int32).contiguous()
    ps = ps.to(torch.int32).contiguous()
    n = int(ks.numel())
    out = [torch.empty(n, dtype=torch.int32, device=unit.select.device) for _ in range(9)]
    _a4_state_dump_kernel[(triton_cdiv(n, 128),)](
        unit.select, unit.label, unit.point, unit.code_nibbles, unit.label_lut,
        ks, ps, int(unit.rows // unit.arity // 2), int(unit.rows // unit.arity),
        *out, n,
        MEMORY=unit.memory, PAD=8, RATE=unit.rate, POINTS=1 << (unit.rate - 1),
        BLOCK=128,
    )
    return tuple(out)


def triton_cdiv(a: int, b: int) -> int:
    return max(1, -(-a // b))


def a4_decode_span2_tile(unit: A4Unit, *, block: int = 64):
    """Test oracle: the kernel's decode as ``(va, vb, scale)`` tensors.

    ``va[i, p]``/``vb[i, p]`` are the four nibbles (little-endian, rows
    ``4p..4p+3``) of columns ``2i``/``2i+1``; ``scale`` is the ``[rows,
    cols//16]`` E4M3 byte tile.  The tests rebuild ``materialize_stock``'s
    ``[rows, cols/2]`` code tile from these and hold them byte for byte.
    """
    require_native_fp4_mma("a4_decode_span2_tile")
    unit._check("a4_decode_span2_tile")
    _require_unit_tensor(unit, "a4_decode_span2_tile")
    pairs = unit.rows // unit.arity // 2
    half = unit.cols // 2
    va = torch.empty((half, pairs), dtype=torch.uint16, device=unit.select.device)
    vb = torch.empty_like(va)
    scale = torch.empty((unit.rows, unit.cols // unit.half),
                        dtype=torch.float8_e4m3fn, device=unit.select.device)
    grid = (-(-half // block), -(-pairs // block), -(-unit.rows // block))
    _a4_decode_dump_kernel[grid](
        unit.select, unit.label, unit.point, unit.nibbles,
        unit.lut_bytes.view(torch.uint8),
        unit.label_lut, unit.code_nibbles, va, vb, scale,
        unit.rows, unit.cols,
        MEMORY=unit.memory, PAD=8, RATE=unit.rate, KP=block, PP=block,
        num_warps=4,
    )
    return va, vb, scale
