"""The window body's prefill GEMM: the ~4 bits/weight wire read directly,
decoded into registers, and fed to a hardware ``tl.dot``.

WHAT IT SERVES.  ``WindowGemvUnit`` in either window family:

* ``family == "value"`` (BF16) -- the table holds bf16 values and the
  per-row fp32 ``scale`` is applied in one of two places, named once at
  prepare time by ``arithmetic`` and never substituted silently:

  - ``"epilogue"`` (the default of this module): ``tl.dot`` on the raw table
    values, the fp32 accumulator multiplied by the row scale once per output;
  - ``"folded"``: one bf16 rounding of ``value * row_scale`` per weight, in
    registers, before ``tl.dot``, and no scale in the epilogue -- exactly
    ``decode.materialize_bf16_folded``'s ``(values.float() * scale[:, None])
    .to(torch.bfloat16)``, and the arithmetic ``window_gemm_grouped``'s
    ``arithmetic="folded"`` runs for an expert stack.  The dense BF16 route
    serves this one (tessera#614).

  The two are different numerical functions of the same wire, so a bundle
  carries its arithmetic as a field and the route stamps a decoder per
  arithmetic.
* ``family == "e4m3"`` (FP8) -- the wire decodes to E4M3 bytes through the
  unit's own ``codes_of_state`` ([2^L] grid codes) and ``native`` ([256]
  code -> byte) tables, the activation is quantized per token by **vLLM's
  native CUDA quantizer** (``serving.native_ops.native_fp8_quant``, the
  ``fp8_per_token_dynamic`` contract), and the epilogue is
  ``y = a_scale[m] * w_scale[n] * acc``.  A bare bf16 activation multiplied
  without that quantizer would be a different contract and is not offered.

PREPARED ONCE, CALLED HOT.  ``prepare_window_gemm`` performs every
tensor-content validation (family fields, table exactness, ``initial_state``
bounds) and freezes the constants -- table/codes/native, permuted
``initial_state``, row scale, geometry.  ``PreparedWindowGemm.__call__`` only
checks cheap metadata (rank, width, dtype, device) and launches: no
GPU-to-host synchronisation, no dtype casts, no reallocation of the constant
bundles, so the call is capturable in a CUDA graph.  ``window_gemm`` remains
a one-shot convenience that prepares and calls; serving holds the prepared
object instead.

THE DECODE.  ``reference_states`` defines the state of a column at position
``q`` as the **last ``L`` bits of the MSB-first code stream that end at
``q = (row + 1) * rate``**, zero-padded before the stream starts.  So no
sequential walk is needed: the state of row ``n`` is a windowed bit gather,
and the whole ``[BLOCK_K, BLOCK_N]`` tile is read straight from the repacked
words.  The first tile's sub-L-bit rows take their high bits from the zero
pad, or from ``initial_state`` when a TP row cut carries one.

RATE RUNS.  ``rep.runs`` fixes the layout: rate runs are contiguous in
permuted column order, so the K loop walks runs and cuts each into ``BLOCK_K``
segments; a short segment is masked, never reordered.  ``rep.perm`` maps the
permuted columns back to ``x``'s original ones.

WHAT IS NOT HERE.  Grouped/MoE stacking (the next milestone; its API is
published in the task interface).  This file is new and edits no shared
module; the loader owns the bundle/dataclass extension.
"""
from __future__ import annotations

import dataclasses

import torch
import triton
import triton.language as tl

from .errors import GrammarError
from .kernel_window_gemv import TILE_ROWS, WindowGemvUnit

__all__ = ["window_gemm", "prepare_window_gemm", "PreparedWindowGemm", "MIN_BLOCK",
           "ARITHMETICS"]

#: ``tl.dot`` needs a 16-row minimum operand; any M is served by masking.
MIN_BLOCK = 16

#: The value family's two weight arithmetics (see the module docstring).
ARITHMETICS = ("epilogue", "folded")


@triton.jit
def _window_gemm_kernel(
    words_ptr, table_ptr, codes_ptr, native_ptr, x_ptr, out_ptr,
    scale_ptr, a_scale_ptr, runs_ptr, init_ptr,
    n_runs, tile_words, total_words, M, rows, cols,
    L: tl.constexpr, TILE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_INIT: tl.constexpr, FP8: tl.constexpr, FOLDED: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n0 = pid_n * BN
    g = n0 // TILE                     # BN divides TILE: one block is one tile
    t = n0 - g * TILE                  # local first row inside the tile

    offs_n = n0 + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    live_n = offs_n < rows
    live_m = offs_m < M

    rows_v = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if FOLDED:
        # the folded value family multiplies every decoded weight by its row's
        # scale, so the scale is read once, before the K loop
        wscale = tl.load(scale_ptr + offs_n, mask=live_n, other=0.0)

    for r in range(n_runs):
        rate = tl.load(runs_ptr + r * 4 + 0)
        col0 = tl.load(runs_ptr + r * 4 + 1)
        ncols = tl.load(runs_ptr + r * 4 + 2)
        word0 = tl.load(runs_ptr + r * 4 + 3)
        CHUNK = 16 * rate              # words per column chunk (512 rows)
        tile_base = g * tile_words + word0

        for c0 in range(0, ncols, BK):
            offs_k = c0 + tl.arange(0, BK)
            live_k = offs_k < ncols
            live_k2 = live_k[:, None]
            kglob = col0 + offs_k                        # permuted column index [BK]
            base = tile_base + offs_k * CHUNK            # [BK]

            qq = (t + 1 + rows_v) * rate                 # [BN], ends of the windows
            if HAS_INIT:
                # history above row 0 keeps every window full
                length = tl.full((BN,), L, tl.int32)
            else:
                # the first tile's opening is zero-padded; later tiles are not
                length = tl.minimum(qq + g * TILE * rate, L)
            qb = qq - L
            neg = qb < 0
            # stream word holding the first field bit, relative to `base`:
            # -1 means the previous tile's last word of the same run, which is
            # the 32 bits immediately before this tile's first word
            wi = tl.where(neg, -1, qb >> 5)
            d1 = wi + 1                                  # current chunk's first word when neg
            # combined bit of the field's last bit (bit 64-b.L of the pair)
            shift = 64 - qq + 32 * wi

            prev_off = -tile_words + CHUNK - 1
            idx_prev = base[:, None] + prev_off
            idx_norm = base[:, None] + wi[None, :]
            idx1 = base[:, None] + d1[None, :]
            live_prev = live_k2 & (g > 0) & (idx_prev >= 0) & (idx_prev < total_words)
            live_norm = live_k2 & (idx_norm >= 0) & (idx_norm < total_words)
            live1 = live_k2 & (idx1 >= 0) & (idx1 < total_words)
            w0_prev = tl.load(words_ptr + idx_prev, mask=live_prev, other=0).to(tl.int64) & 0xFFFFFFFF
            w0_norm = tl.load(words_ptr + idx_norm, mask=live_norm, other=0).to(tl.int64) & 0xFFFFFFFF
            if HAS_INIT:
                w0_init = tl.load(init_ptr + kglob, mask=live_k, other=0).to(tl.int64) & 0xFFFFFFFF
                w0 = tl.where(neg[None, :] & (g == 0), w0_init[:, None],
                              tl.where(neg[None, :], w0_prev, w0_norm))
            else:
                w0 = tl.where(neg[None, :], w0_prev, w0_norm)
            w1 = tl.load(words_ptr + idx1, mask=live1, other=0).to(tl.int64) & 0xFFFFFFFF

            combined = (w0 << 32) | w1
            state = (combined >> shift[None, :]) & ((1 << length) - 1)[None, :]

            if FP8:
                code = tl.load(codes_ptr + state, mask=live_k2, other=0)
                byte = tl.load(native_ptr + code.to(tl.int32), mask=live_k2, other=0)
                val = byte.to(tl.float8e4nv, bitcast=True)
            else:
                val = tl.load(table_ptr + state, mask=live_k2, other=0.0)
                if FOLDED:
                    # one bf16 rounding of (value * row scale) in registers,
                    # before the dot: materialize_bf16_folded's tile
                    val = (val.to(tl.float32) * wscale[None, :]).to(tl.bfloat16)

            xk = tl.load(
                x_ptr + offs_m[:, None] * cols + kglob[None, :],
                mask=live_m[:, None] & live_k[None, :], other=0.0
            )
            acc += tl.dot(xk, val, out_dtype=tl.float32)

    if FOLDED:
        # the scale is already inside every weight: no epilogue factor
        y = acc
    else:
        scale = tl.load(scale_ptr + offs_n, mask=live_n, other=0.0)
        if FP8:
            a_scale = tl.load(a_scale_ptr + offs_m, mask=live_m, other=0.0)
            y = (acc * a_scale[:, None]) * scale[None, :]
        else:
            y = acc * scale[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * rows + offs_n[None, :],
        y.to(tl.bfloat16), mask=live_m[:, None] & live_n[None, :],
    )


@dataclasses.dataclass(frozen=True)
class PreparedWindowGemm:
    """The frozen bundle: constants, geometry, launch shape.  ``__call__``
    is the hot path and performs no tensor-content validation."""

    words: torch.Tensor
    table: torch.Tensor
    codes: torch.Tensor
    native: torch.Tensor
    scale: torch.Tensor
    runs: torch.Tensor
    init_perm: torch.Tensor
    perm: torch.Tensor
    tile_words: int
    total_words: int
    rows: int
    cols: int
    window_bits: int
    family: str
    has_init: bool
    block_m: int
    block_n: int
    block_k: int
    quantizer: str = "native"
    #: Where the value family applies the row scale: ``"epilogue"`` (on the
    #: fp32 accumulator) or ``"folded"`` (into each decoded weight, one bf16
    #: rounding, before the dot).  The E4M3 family is always ``"epilogue"``.
    arithmetic: str = "epilogue"

    def __post_init__(self):
        # Metadata only -- no tensor is read -- so the custom op that rebuilds
        # this bundle per call pays two string comparisons, not a sync.
        if self.arithmetic not in ARITHMETICS:
            raise GrammarError(f"unknown weight arithmetic {self.arithmetic!r}")
        if self.arithmetic == "folded" and self.family != "value":
            raise GrammarError(
                "the folded weight arithmetic is the BF16 (value family) contract; the "
                "E4M3 family keeps the per-token A quant and the row-scale epilogue")

    @property
    def device(self) -> torch.device:
        return self.words.device

    def __call__(self, x: torch.Tensor, a_scale: "torch.Tensor | None" = None,
                 out: "torch.Tensor | None" = None) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.cols or x.device != self.device:
            raise GrammarError(
                f"x must be a [M, {self.cols}] tensor on {self.device}, got "
                f"{tuple(x.shape)} on {x.device}"
            )
        m = int(x.shape[0])
        if m == 0:
            return torch.empty(0, self.rows, dtype=torch.bfloat16, device=self.device)
        if self.family == "value":
            if x.dtype != torch.bfloat16 or a_scale is not None:
                raise GrammarError(
                    "the value family takes a bf16 x and no activation scale; the E4M3 "
                    "family is the one with a per-token FP8 activation"
                )
            x_perm = x.index_select(1, self.perm).contiguous()
            dtype = torch.bfloat16
            a = self.scale.new_zeros(1)
            fp8 = False
        else:
            if x.dtype == torch.float8_e4m3fn:
                if a_scale is None:
                    raise GrammarError(
                        "an fp8 x must carry its per-token activation scale; a bare "
                        "fp8 or bf16 activation would change the contract"
                    )
                a = a_scale.reshape(-1)
                if a.numel() != m or a.dtype != torch.float32 or a.device != self.device:
                    raise GrammarError(
                        f"a_scale must be fp32 [{m}] on {self.device}"
                    )
                if not a.is_contiguous():
                    raise GrammarError(
                        "a_scale must be contiguous after flattening; the kernel reads it "
                        "with one stride"
                    )
                x_perm = x.index_select(1, self.perm).contiguous()
            else:
                if x.dtype != torch.bfloat16:
                    raise GrammarError(f"the E4M3 family takes bf16 or fp8 x, got {x.dtype}")
                if self.quantizer != "native":
                    raise GrammarError(
                        "this bundle was prepared without a quantizer; pass the "
                        "prequantized fp8 activation together with its per-token scale"
                    )
                from .serving.native_ops import native_fp8_quant
                x_perm_raw = x.index_select(1, self.perm).contiguous()
                x_perm, s = native_fp8_quant(x_perm_raw)
                a = s.reshape(-1)
            dtype = torch.float8_e4m3fn
            fp8 = True
        y = (torch.empty(m, self.rows, dtype=torch.bfloat16, device=self.device)
             if out is None else out)
        if y.shape != (m, self.rows) or y.dtype != torch.bfloat16 or y.device != self.device:
            raise GrammarError(f"out must be bf16 [{m}, {self.rows}] on {self.device}")
        if not y.is_contiguous():
            raise GrammarError("out must be contiguous; the kernel writes row-major")
        grid = (triton.cdiv(self.rows, self.block_n), triton.cdiv(m, self.block_m))
        _window_gemm_kernel[grid](
            self.words, self.table, self.codes, self.native, x_perm, y,
            self.scale, a, self.runs, self.init_perm,
            int(self.runs.shape[0]), self.tile_words, self.total_words,
            m, self.rows, self.cols,
            L=self.window_bits, TILE=TILE_ROWS,
            BM=self.block_m, BN=self.block_n, BK=self.block_k,
            HAS_INIT=self.has_init, FP8=fp8, FOLDED=self.arithmetic == "folded",
            num_warps=8,
        )
        return y


def _resolve_initial_state(unit: WindowGemvUnit,
                           initial_state: "torch.Tensor | None") -> "torch.Tensor | None":
    if initial_state is None:
        initial_state = getattr(unit, "initial_state", None)
    if initial_state is None:
        initial_state = getattr(unit.rep, "initial_state", None)
    return initial_state


def prepare_window_gemm(
    unit: WindowGemvUnit,
    *,
    initial_state: "torch.Tensor | None" = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    quantizer: "str | None" = "native",
    arithmetic: str = "epilogue",
) -> PreparedWindowGemm:
    """Validate the unit once and freeze every constant the call needs.

    Every tensor-content check (table exactness, family tables, the
    ``initial_state`` bounds) happens here; the returned object's ``__call__``
    may not synchronise.  For the E4M3 family, ``quantizer="native"`` attests
    vLLM's per-token FP8 quantizer once, here; ``quantizer=None`` prepares a
    compute-only bundle that takes prequantized activations.

    ``arithmetic`` names the value family's weight-side contract, once and
    explicitly: ``"epilogue"`` multiplies the fp32 accumulator by the row
    scale after the dot; ``"folded"`` rounds ``value * row_scale`` to bf16 per
    weight before the dot and applies no scale after it
    (``decode.materialize_bf16_folded``).  The E4M3 family has no folded form
    -- its per-token A quant and row-scale epilogue are the published contract
    -- so ``"folded"`` is refused there.
    """
    if arithmetic not in ARITHMETICS:
        raise GrammarError(f"unknown weight arithmetic {arithmetic!r}")
    if arithmetic == "folded" and unit.family != "value":
        raise GrammarError(
            "the folded weight arithmetic is the BF16 (value family) contract; the E4M3 "
            "family keeps the per-token A quant and the row-scale epilogue")
    if unit.family not in ("value", "e4m3"):
        raise GrammarError(f"window_gemm serves the value and e4m3 families, got {unit.family!r}")
    for name, v in (("block_m", block_m), ("block_n", block_n), ("block_k", block_k)):
        if v < MIN_BLOCK or v & (v - 1):
            raise GrammarError(f"{name}={v} must be a power of two >= {MIN_BLOCK}")
    if TILE_ROWS % block_n:
        raise GrammarError(
            f"block_n={block_n} does not divide the {TILE_ROWS}-row wire tile; "
            "one N block must stay inside one tile (its lookback would be wrong)"
        )
    device = unit.rep.words.device
    scale = unit.scale
    if scale.numel() != unit.rows:
        raise GrammarError(f"the unit's scale has {scale.numel()} entries for {unit.rows} rows")
    # constants are frozen contiguous here so the hot path never re-strides
    scale = scale.to(torch.float32).contiguous()
    words = unit.rep.words.contiguous()
    runs = unit.rep.runs.contiguous()
    perm = unit.rep.perm.contiguous()

    table = unit.table
    codes = unit.codes_of_state
    native = unit.native
    if unit.family == "value":
        if table is None:
            raise GrammarError("the value family needs its table")
        if table.dtype == torch.float32:
            rounded = table.to(torch.bfloat16)
            if not torch.equal(rounded.float(), table):
                raise GrammarError(
                    "the value family's table must be exactly representable in bf16; "
                    "an fp32 table with rounded bits would silently change the values"
                )
            table = rounded
        elif table.dtype != torch.bfloat16:
            raise GrammarError(f"the value family reads a bf16 table; got {table.dtype}")
        table = table.contiguous()
        codes = native = torch.zeros(0, dtype=torch.uint8, device=device)
    else:
        if codes is None or native is None:
            raise GrammarError(
                "the E4M3 family needs its codes_of_state and native byte tables; "
                "a unit without them has no E4M3 bytes to decode"
            )
        if codes.dtype != torch.uint8 or codes.numel() != 1 << int(unit.window_bits):
            raise GrammarError(
                f"codes_of_state must be uint8 [{1 << int(unit.window_bits)}], got "
                f"{codes.dtype} [{codes.numel()}]"
            )
        if native.dtype != torch.uint8 or native.numel() != 256:
            raise GrammarError(f"native must be uint8 [256], got {native.dtype} [{native.numel()}]")
        codes = codes.contiguous()
        native = native.contiguous()
        table = torch.zeros(0, dtype=torch.bfloat16, device=device)
        if quantizer == "native":
            from .serving.native_ops import require_native_fp8_quant
            require_native_fp8_quant("window_gemm: per-token FP8 quantizer")
        elif quantizer is not None:
            raise GrammarError(f"unknown quantizer {quantizer!r}; use 'native' or None")

    resolved = _resolve_initial_state(unit, initial_state)
    limit = 1 << int(unit.window_bits)
    if resolved is None:
        init_perm = torch.zeros(unit.cols, dtype=torch.int32, device=device)
        has_init = False
    else:
        if resolved.numel() != unit.cols:
            raise GrammarError(
                f"initial_state has {resolved.numel()} columns, the unit {unit.cols}"
            )
        bad = (resolved < 0) | (resolved >= limit)
        if bool(bad.any()):
            raise GrammarError(
                f"initial_state must hold {unit.window_bits}-bit window states "
                f"in [0, {limit})"
            )
        init_perm = resolved.to(device=device, dtype=torch.int32)
        init_perm = init_perm.index_select(0, unit.rep.perm.long()).contiguous()
        has_init = True

    return PreparedWindowGemm(
        words=words,
        table=table,
        codes=codes,
        native=native,
        scale=scale,
        runs=runs,
        init_perm=init_perm,
        perm=perm,
        tile_words=int(unit.rep.tile_words),
        total_words=int(unit.rep.words.numel()),
        rows=int(unit.rows),
        cols=int(unit.cols),
        window_bits=int(unit.window_bits),
        family=unit.family,
        has_init=has_init,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        quantizer=quantizer if unit.family == "e4m3" else "native",
        arithmetic=arithmetic,
    )


def window_gemm(
    unit: WindowGemvUnit,
    x: torch.Tensor,
    *,
    initial_state: "torch.Tensor | None" = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    out: "torch.Tensor | None" = None,
    arithmetic: str = "epilogue",
) -> torch.Tensor:
    """One-shot ``prepare_window_gemm`` + call.  Convenience for tests and
    single uses; a serving path prepares once and holds the object."""
    prepared = prepare_window_gemm(
        unit, initial_state=initial_state,
        block_m=block_m, block_n=block_n, block_k=block_k, arithmetic=arithmetic,
    )
    return prepared(x, out=out)
