"""Segment 2a: rank-1 channel diagonals, and the S5 rotation states.

S5 lists segment 2a as "rank-1 channel diagonals `su`/`sv` -- cheapest bits
measured; vendor-neutral", and S7 records the prior art it comes from: EXL3
"ships `su`/`sv` channel diagonals around its GEMM"
(`quantize.py:1208-1212`, `weight /= su`).  Vendor-neutral is the operative
word -- the diagonals are applied *outside* the MMA, so they cost no kernel
and survive into a stock NVFP4 serving path.

**What they buy.**  The group scale is one number per 32 weights along a row.
If one input channel is systematically hotter than its neighbours it dominates
every group it touches, and every other weight in those groups is quantised
against a scale set by an outlier it has nothing to do with.  Rank-1 balancing
removes exactly the part of that imbalance that factorises as
`W[i,j] ~ sv[i] * su[j]`, which is the part a per-32 block scale cannot see.

**What they cost.**  One FP16 per row plus one per column:
`16*(rows+cols) / (rows*cols)` bpp -- 0.011 bpp on a 2048x5120 Linear.  S5's
"cheapest bits measured" is an understatement at that price.

**Rotation states.**  ``RotationState`` offers ``NONE`` and ``R_IN_ONLY`` and
deliberately not two-sided: S7 and the enum's own docstring make two-sided a
*weight-space measurement state*, because its output basis needs an ``R_out^T``
inverse or proved propagation through every consumer, which is a model-level
contract and not a per-unit branch.  Offering it as a serving branch here would
manufacture an artifact no runtime can honour, which principle 9 forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .errors import GrammarError
from .manifest import RotationState

__all__ = [
    "Diagonals",
    "fit_diagonals",
    "apply_diagonals",
    "undo_diagonals",
    "require_invertible_diagonals",
    "hadamard_block",
    "apply_rotation",
    "undo_rotation",
    "rotation_block_for",
    "transport_metric",
    "diagonal_bits",
]


@dataclass(frozen=True)
class Diagonals:
    """``sv`` per output channel (rows), ``su`` per input channel (columns)."""

    sv: torch.Tensor
    su: torch.Tensor

    def bits(self) -> int:
        return 16 * (self.sv.numel() + self.su.numel())


def diagonal_bits(rows: int, columns: int) -> int:
    """The segment-2a plane cost, matching ``NORMATIVE_ELEMENT_BITS``."""
    return 16 * (rows + columns)


def fit_diagonals(
    weights: torch.Tensor, iterations: int = 8, eps: float = 1e-12
) -> Diagonals:
    """Fit the rank-1 diagonals by alternating RMS balancing (Sinkhorn).

    Alternately normalising rows and columns to unit RMS converges to the
    unique rank-1 factor of the magnitude field, which is precisely the
    component a per-block scale cannot represent.  Eight sweeps is well past
    the knee for weight matrices; the fit is deterministic, so an artifact
    stays a pure function of its input.

    Three representation rules bound the fit at FP16, the wire's precision
    (tessera#229) -- each derived from the dtype or the transform's own
    algebra, none a tuning:

    * **A zero row or column takes the identity factor 1.0.**  Its balanced
      values are zero under any factor, so the factor is pure gauge; 1.0 is
      invertible, exact at every precision, and keeps a degenerate row from
      dragging the range repair below toward zero.
    * **The rank-1 gauge is spent on representability.**  ``(sv * c, su / c)``
      balances the same matrix for any scalar ``c > 0``, so when the direct
      FP16 cast would round a factor to zero or overflow it to infinity, the
      one free scalar is chosen to land both factors inside FP16's normal
      range -- at the geometric midpoint of the feasible interval, the
      maximal joint log-margin.  When the direct cast is already invertible
      the fit is byte for byte what it always was: re-gauging a healthy fit
      would move stored planes for nothing.
    * **A spread no gauge can represent is refused, by field name.**  One
      scalar cannot fix a ratio: factors spanning more than FP16's normal
      range (``max / tiny`` ~ 1.07e9) cannot all be stored invertibly, and
      ``undo_diagonals`` multiplies the stored words back, so writing them
      anyway decodes finite weights to zero or NaN (the P0 this rule closes).
      The caller's out is to encode without segment-2a diagonals.
    """
    if weights.ndim != 2:
        raise GrammarError(f"expected a 2-D weight, got {tuple(weights.shape)}")
    work = weights.to(torch.float32)
    rows, cols = work.shape
    zero_rows = (work == 0).all(dim=1)
    zero_cols = (work == 0).all(dim=0)
    sv = torch.ones(rows, device=work.device, dtype=torch.float32)
    su = torch.ones(cols, device=work.device, dtype=torch.float32)
    for _ in range(iterations):
        row_rms = work.pow(2).mean(dim=1).sqrt().clamp_min(eps)
        work = work / row_rms.unsqueeze(1)
        sv = sv * row_rms
        col_rms = work.pow(2).mean(dim=0).sqrt().clamp_min(eps)
        work = work / col_rms.unsqueeze(0)
        su = su * col_rms
    # The zero-row/column policy, before the gauge reads the ranges: the
    # Sinkhorn loop accumulated eps clamps on these, which are not factors of
    # anything.
    sv = torch.where(zero_rows, torch.ones_like(sv), sv)
    su = torch.where(zero_cols, torch.ones_like(su), su)
    sv, su = _land_in_fp16(sv, su)
    # Store at the wire precision, and re-derive from the stored value so the
    # encoder quantises against exactly what the decoder will reconstruct.
    return require_invertible_diagonals(Diagonals(sv=sv, su=su))


def _land_in_fp16(sv: torch.Tensor, su: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """Spend the rank-1 gauge landing both fp32 factors in FP16, or refuse.

    Feasibility is exact: ``sv * c`` fits ``[tiny, max]`` iff ``c`` is in
    ``[tiny / min(sv), max / max(sv)]``, and ``su / c`` iff ``c`` is in
    ``[max(su) / max, min(su) / tiny]``.  An empty intersection is a spread
    one scalar cannot fix, and is refused naming the field that overflows.
    The band is FP16's *normal* range: subnormal factors are representable
    but carry as little as one significand bit, and a bound derived from the
    dtype's precision is the rule (working rule 2), not a wider one that
    happens to pass.
    """
    cast = Diagonals(sv=sv.to(torch.float16), su=su.to(torch.float16))
    if _invertible(cast.sv) and _invertible(cast.su):
        return cast.sv, cast.su
    finfo = torch.finfo(torch.float16)
    a_lo, a_hi = float(sv.min()), float(sv.max())
    b_lo, b_hi = float(su.min()), float(su.max())
    lo = max(finfo.tiny / a_lo if a_lo > 0 else float("inf"), b_hi / finfo.max)
    hi = min(finfo.max / a_hi, (b_lo / finfo.tiny) if b_lo > 0 else 0.0)
    if lo <= hi:
        c = (lo ** 0.5) * (hi ** 0.5)
        sv16 = (sv * c).to(torch.float16)
        su16 = (su / c).to(torch.float16)
        if _invertible(sv16) and _invertible(su16):
            return sv16, su16
    name = "DIAG_SV" if not _invertible(cast.sv) else "DIAG_SU"
    raise GrammarError(
        f"the fitted channel diagonals do not fit FP16, the {name} plane's "
        f"element width: sv spans [{a_lo:.3e}, {a_hi:.3e}] and su spans "
        f"[{b_lo:.3e}, {b_hi:.3e}], and no rank-1 gauge lands both inside "
        f"[{finfo.tiny:.3e}, {finfo.max:.3e}]. Stored anyway they would round "
        "to zero or infinity and decode finite weights to zero or NaN "
        "(tessera#229); encode this unit without segment-2a diagonals"
    )


def _invertible(factor: torch.Tensor) -> bool:
    """Finite and strictly positive at the stored precision -- the property
    that makes ``apply_diagonals`` and ``undo_diagonals`` inverses."""
    f = factor.float()
    return bool(torch.isfinite(f).all()) and bool((f > 0).all())


def require_invertible_diagonals(diagonals: Diagonals) -> Diagonals:
    """Refuse a segment-2a pair that is not invertible at its stored words.

    ``undo_diagonals`` multiplies the stored FP16 factors back, so a zero,
    negative or non-finite factor decodes every weight it touches to zero,
    a flipped sign, or NaN -- silently, because the artifact stays well-formed
    (tessera#229).  One rule, one home: the fit, both transform directions,
    the writer and the reader all call this instead of each clamping or
    trusting their own side.
    """
    for name, factor in (("DIAG_SV", diagonals.sv), ("DIAG_SU", diagonals.su)):
        f = factor.float()
        bad = ~torch.isfinite(f) | (f <= 0)
        if bool(bad.any()):
            raise GrammarError(
                f"{name} holds {int(bad.sum())} of {factor.numel()} factor(s) "
                "that are zero, negative or non-finite: the pair is not "
                "invertible, so balancing through it encodes finite weights "
                "as zero or NaN (tessera#229)"
            )
    return diagonals


def apply_diagonals(weights: torch.Tensor, diagonals: Diagonals) -> torch.Tensor:
    """``diag(1/sv) @ W @ diag(1/su)`` -- the balanced matrix the body codes.

    Refuses a non-invertible pair instead of clamping: the old one-sided
    ``clamp_min(1e-12)`` made the forward divide finite while
    ``undo_diagonals`` multiplied the stored zero back, so the two directions
    were silently not inverses (tessera#229).
    """
    require_invertible_diagonals(diagonals)
    sv = diagonals.sv.to(torch.float32)
    su = diagonals.su.to(torch.float32)
    return weights.to(torch.float32) / sv.unsqueeze(1) / su.unsqueeze(0)


def undo_diagonals(balanced: torch.Tensor, diagonals: Diagonals) -> torch.Tensor:
    """``diag(sv) @ W' @ diag(su)`` -- what the serving path applies."""
    require_invertible_diagonals(diagonals)
    sv = diagonals.sv.to(torch.float32)
    su = diagonals.su.to(torch.float32)
    return balanced * sv.unsqueeze(1) * su.unsqueeze(0)


def hadamard_block(size: int, device=None) -> torch.Tensor:
    """A normalised Sylvester Hadamard of ``size``, which must be a power of 2."""
    if size < 1 or size & (size - 1):
        raise GrammarError(f"Hadamard block {size} is not a power of two")
    matrix = torch.ones(1, 1, device=device, dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix / (size ** 0.5)


def _block_size(columns: int, cap: int = 128) -> int:
    """Largest power of two that divides ``columns``, capped.

    Real hidden sizes are rarely powers of two (5120 = 2^10 x 5), so a full
    Hadamard does not fit and the transform is applied blockwise.  The cap
    keeps the block small enough that the rotation stays cheap and local.
    """
    size = 1
    while size * 2 <= cap and columns % (size * 2) == 0:
        size *= 2
    return size


def rotation_block_for(state: RotationState, columns: int) -> int:
    """The rotation block a ``columns``-wide artifact carries: derived, never
    stored.

    The wire has no rotation-block field, so writer and reader must agree on
    it as a pure function of what the wire does carry -- the rotation state
    and the geometry's width.  This is ``apply_rotation``'s own default
    (``_block_size``: the largest power of two dividing the width, capped at
    128) stated once, read by ``unit_artifact``'s writer to refuse any unit
    rotated at a block the wire cannot represent, and by both body parsers to
    rebuild the block instead of assuming 128 (tessera#210 -- below 128
    columns the assumption was a reshape error in the reader; elsewhere it
    substituted a rotation the encoder never applied).
    """
    if state is RotationState.NONE:
        return 1
    return _block_size(columns)


def apply_rotation(
    weights: torch.Tensor, state: RotationState, block: int | None = None
) -> "tuple[torch.Tensor, int]":
    """Rotate the input-channel axis.  Returns ``(rotated, block_size)``.

    ``R_IN_ONLY`` is the sole algebraically local state: rotating the input
    axis is undone by rotating the activation with the same orthogonal matrix,
    which the consumer can fold into the preceding op.  ``NONE`` is identity.
    """
    if state is RotationState.NONE:
        return weights.to(torch.float32), 1
    if state is not RotationState.R_IN_ONLY:
        raise GrammarError(
            f"{state!r} is not a serving rotation state. Two-sided rotation is "
            "a weight-space measurement state only (doc S7): its output basis "
            "needs an R_out^T inverse or proved propagation through every "
            "consumer, which is a model-level contract, not a per-unit branch."
        )
    cols = weights.shape[1]
    size = block or _block_size(cols)
    if size == 1:
        return weights.to(torch.float32), 1
    if cols % size:
        raise GrammarError(f"{cols} columns is not a multiple of block {size}")
    work = weights.to(torch.float32).reshape(-1, cols // size, size)
    matrix = hadamard_block(size, weights.device)
    return (work @ matrix).reshape(weights.shape), size


def undo_rotation(
    rotated: torch.Tensor, state: RotationState, block: int
) -> torch.Tensor:
    """Inverse of :func:`apply_rotation`.  Hadamard is its own inverse."""
    if state is RotationState.NONE or block == 1:
        return rotated
    cols = rotated.shape[1]
    work = rotated.reshape(-1, cols // block, block)
    matrix = hadamard_block(block, rotated.device)
    return (work @ matrix.T).reshape(rotated.shape)


def transport_metric(
    metric: torch.Tensor,
    state: RotationState,
    block: "int | None" = None,
    diagonals: "Diagonals | None" = None,
) -> torch.Tensor:
    """Carry an input-activation metric into the encoder's working basis.

    The transport rule (tessera#231), stated once where the basis change
    lives.  The encoder quantises ``Wwork = Dv^-1 (W R) Du^-1``
    (:func:`apply_rotation` then :func:`apply_diagonals`), so
    ``W = Dv Wwork Du R^T`` and the activations its working rows meet are
    ``xwork = Du R^T x``.  A source second-moment ``H = E[x x^T]`` therefore
    becomes

        ``H' = Du R^T H R Du``

    and the source output loss of a working-coordinate error ``E`` is
    ``sum_r sv_r^2 E_r H' E_r^T`` -- the row factors ride separately because
    they weight rows, not columns (``encode_unit`` threads them into the
    shared LUT-table fit, the one cross-row decision).  Deriving an LDL
    factor or a diagonal-power objective from the *untransported* H prices a
    different quadratic than the encode minimises, which is the mispricing
    tessera#231 measured.

    A 1-D (diagonal) metric stays 1-D under ``NONE`` rotation -- ``h' = su^2
    h`` -- and is densified first under a real one, because a diagonal is not
    diagonal in a rotated basis.  Regularisation is applied by the caller
    AFTER this transport, in the basis the factorisation runs in: under the
    orthogonal ``R`` alone the two orders agree, but ``Du`` rescales a
    source-basis ridge into a column-dependent one, which is not the
    stabiliser the solve asked for.
    """
    if metric.ndim not in (1, 2):
        raise GrammarError(
            f"a refit metric is per-column [cols] or a full [cols, cols] "
            f"Hessian, got shape {tuple(metric.shape)}"
        )
    work = metric.to(torch.float32)
    if state is not RotationState.NONE:
        cols = work.shape[-1]
        size = block if block is not None else _block_size(cols)
        if size > 1:
            if cols % size:
                raise GrammarError(
                    f"{cols} metric columns is not a multiple of rotation "
                    f"block {size}"
                )
            if work.ndim == 1:
                work = torch.diag(work)
            matrix = hadamard_block(size, work.device)
            # ``R^T H R`` for the block-diagonal R, one axis at a time.
            work = (work.reshape(cols, -1, size) @ matrix).reshape(cols, cols)
            work = (work.T.reshape(cols, -1, size) @ matrix).reshape(cols, cols).T
    if diagonals is not None:
        require_invertible_diagonals(diagonals)
        su = diagonals.su.to(torch.float32)
        if work.ndim == 1:
            work = work * su * su
        else:
            work = work * su.unsqueeze(0) * su.unsqueeze(1)
    return work
