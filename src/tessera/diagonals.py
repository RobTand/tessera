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
    "hadamard_block",
    "apply_rotation",
    "undo_rotation",
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
    """
    if weights.ndim != 2:
        raise GrammarError(f"expected a 2-D weight, got {tuple(weights.shape)}")
    work = weights.to(torch.float32)
    rows, cols = work.shape
    sv = torch.ones(rows, device=work.device, dtype=torch.float32)
    su = torch.ones(cols, device=work.device, dtype=torch.float32)
    for _ in range(iterations):
        row_rms = work.pow(2).mean(dim=1).sqrt().clamp_min(eps)
        work = work / row_rms.unsqueeze(1)
        sv = sv * row_rms
        col_rms = work.pow(2).mean(dim=0).sqrt().clamp_min(eps)
        work = work / col_rms.unsqueeze(0)
        su = su * col_rms
    # Store at the wire precision, and re-derive from the stored value so the
    # encoder quantises against exactly what the decoder will reconstruct.
    return Diagonals(sv=sv.to(torch.float16), su=su.to(torch.float16))


def apply_diagonals(weights: torch.Tensor, diagonals: Diagonals) -> torch.Tensor:
    """``diag(1/sv) @ W @ diag(1/su)`` -- the balanced matrix the body codes."""
    sv = diagonals.sv.to(torch.float32).clamp_min(1e-12)
    su = diagonals.su.to(torch.float32).clamp_min(1e-12)
    return weights.to(torch.float32) / sv.unsqueeze(1) / su.unsqueeze(0)


def undo_diagonals(balanced: torch.Tensor, diagonals: Diagonals) -> torch.Tensor:
    """``diag(sv) @ W' @ diag(su)`` -- what the serving path applies."""
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
