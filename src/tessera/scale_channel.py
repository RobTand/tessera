"""Segment 2b per output channel: the scale plane an FP8 tensor core consumes.

S6b and the LUT plane set one scale per half of sixteen weights along a row:
the layout NVFP4's block-scaled MMA takes, and the plane that carries the
column structure an E2M1 tile cannot express by itself
(``tessera-scale-plane-buys-column-structure``).  An E4M3 tile carries its
own exponent, and the FP8 MMA that executes it takes **one scale per output
channel** -- ``compressed-tensors``' ``strategy: channel`` -- so on that grid
the block plane is redundant twice over: it spends a quarter-bit per weight
the tile does not need, and it is not the scale layout the kernel reads.
Measured on six GLM experts at 4.0 bpp, the window body over a per-channel
plane is 1.07x better than over the LUT plane at the same bytes
(``docs/measurements/tessera-window-body-2026-09-02.md``).

The CHANNEL plane (``ScalePlaneKind.CHANNEL``, schema minor 3) is spelled
with elements the wire already has: the row scale rides the **DIAG_SV**
plane -- one fp16 per output row, the field segment 2a already declares --
over the unit's fp32 ``global_scale``, and SCALE_BASE, SCALE_REFINE and
DIAG_SU are absent.  A weight decodes as ``grid_value(code) * global *
sv[row]``.  No plane kind, element width or order changes, which is why it
is a minor.

The fit is the same discipline as the block planes': an initial scale from
the row's RMS against the modelled source spread, then least-squares refits
to the codes the trellis chose, every scale **landed on the stored fp16
word** before the trellis sees it again, so the encoder quantises against
exactly what the decoder reconstructs.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from .alphabet import GAUSSIAN_SOURCE, PayloadGrid
from .errors import GrammarError

__all__ = [
    "default_channel_sigma",
    "channel_global",
    "check_refit_metric",
    "land_channel_scale",
    "initial_channel_scale",
    "refit_channel_scale",
    "land_at_least",
    "channel_scale_field",
]

#: fp16's largest finite value; a stored row word may not exceed it.
_FP16_MAX = 65504.0


@lru_cache(maxsize=16)
def _default_sigma(name: str, values: "tuple[float, ...]", arity: int) -> float:
    """The RTN-optimal Gaussian spread on the grid's scalar values, in grid units.

    Derived from the objective rather than chosen: over a dyadic ladder of
    spreads below the grid's peak, the one whose nearest-value error on a
    unit Gaussian is smallest, relative to the variance quantised.  On E2M1
    this lands near ``peak / 2.7``; on E4M3, whose values are log-spaced, the
    error is flat over a wide band and the ladder picks one point in it.
    Deterministic: the source is ``GAUSSIAN_SOURCE``'s fixed quantile sample.
    """
    scalar = torch.tensor(sorted(set(values)), dtype=torch.float64)
    peak = float(scalar.abs().max())
    sample = torch.tensor(GAUSSIAN_SOURCE(1 << 12, 1.0), dtype=torch.float64)
    best, best_err = None, None
    for k in range(0, 40):
        sigma = peak * 2.0 ** (-k / 4)
        # Nearest grid value to each sample expressed in grid units.
        scaled = sample * sigma
        err = (scaled.unsqueeze(1) - scalar.unsqueeze(0)).abs().min(dim=1).values
        rel = float((err * err).mean() / (sigma * sigma))
        if best_err is None or rel < best_err:
            best, best_err = sigma, rel
    return float(best)


def default_channel_sigma(grid: PayloadGrid) -> float:
    """The per-channel plane's modelled source spread for ``grid``, in grid units.

    A row is first scaled so its RMS sits at this spread; the window table
    (and a TCQ forest under this plane) models the same Gaussian.  It is the
    starting point the refit moves from, not a constraint on the fit.
    """
    return _default_sigma(grid.name, tuple(grid.values), grid.arity)


def check_refit_metric(metric: torch.Tensor, cols: int) -> torch.Tensor:
    """Refuse a refit metric that is not this unit's, and say which it is.

    Every refit that takes a metric reads it as ``[cols]`` (the diagonal
    objective) or ``[cols, cols]`` (the full Hessian).  Only the *rank* was
    checked, so a metric of the wrong width reached the arithmetic and came
    back as a raw ``RuntimeError`` about broadcasting or matmul shapes -- and a
    length-1 metric broadcast silently, weighting every column equally under a
    name that says it does not.  A metric belongs to a unit's columns; one that
    does not fit them is a caller error, and it is refused by name.
    """
    if metric.ndim == 1:
        if metric.numel() != cols:
            raise GrammarError(
                f"a per-column refit metric holds one weight per input column: "
                f"{metric.numel()} for {cols} columns"
            )
    elif metric.ndim == 2:
        if tuple(metric.shape) != (cols, cols):
            raise GrammarError(
                f"a full-Hessian refit metric is [cols, cols]: "
                f"{tuple(metric.shape)} for {cols} columns"
            )
    else:
        raise GrammarError(
            f"a refit metric is per-column [cols] or a full [cols, cols] Hessian, "
            f"got shape {tuple(metric.shape)}"
        )
    return metric


def channel_global(scale: torch.Tensor) -> float:
    """A power of two that puts the median row scale near one.

    fp16 holds five decades; a global keeps every row's stored word normal
    whatever the tensor's magnitude, and a power of two keeps the product
    ``word * global`` exact in fp32.
    """
    positive = scale[torch.isfinite(scale) & (scale > 0)]
    if positive.numel() == 0:
        return 1.0
    return float(2.0 ** math.floor(math.log2(float(positive.median()))))


def land_channel_scale(
    scale: torch.Tensor, global_scale: float
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Store ``scale`` as fp16 words over ``global_scale``.

    Returns ``(stored fp16 [rows], effective fp32 [rows])`` where the
    effective scale is re-derived from the stored word -- the value the
    decoder will reconstruct and therefore the one the trellis quantises
    against.  A zero or non-finite row lands on the floor the clamp applies to
    every row, so that no target is ever divided by zero; its codes decode to
    nothing either way.  That floor is ``2^-14``, the smallest positive
    **normal** fp16 word -- not the smallest positive one, since fp16's
    subnormals run down to ``2^-24``.  Staying above them is what keeps every
    stored row word carrying its full mantissa.
    """
    if global_scale <= 0 or not math.isfinite(global_scale):
        raise GrammarError(f"the CHANNEL global scale must be positive and finite, got {global_scale}")
    word = scale.float() / global_scale
    word = torch.where(torch.isfinite(word) & (word > 0), word, torch.full_like(word, 2.0 ** -14))
    stored = word.clamp(2.0 ** -14, _FP16_MAX).to(torch.float16)
    return stored, stored.float() * global_scale


def initial_channel_scale(
    work: torch.Tensor, sigma: float, reach: "float | None" = None
) -> "tuple[torch.Tensor, torch.Tensor, float]":
    """The initial plane: every row's RMS lands on ``sigma`` grid units, and
    every row's largest weight lands within one fp16 ulp of the body's
    ``reach``.

    ``reach`` is the largest magnitude, in grid units, the body can emit: the
    window table's extreme entry, or the largest anchor a TCQ forest reaches.
    A row whose largest weight would land beyond it starts lower instead, at
    the sigma that puts that weight exactly on the reach, because nothing
    downstream recovers a clipped weight -- the trellis can only emit the
    reach for it, and the refit is a least-squares step on the codes the
    trellis chose, which re-inflates a globally lowered sigma back to the
    clip.  Rows inside the reach are untouched, so on a light-tailed source
    this is the plain RMS start byte for byte; ``None`` is that start for
    every row.

    **"Within one ulp", not "inside".**  ``scale = amax / reach`` is a *lower*
    bound -- an effective scale below it puts ``amax / effective`` back past the
    reach -- and ``land_channel_scale`` rounds to **nearest**, so about a third
    of the raised rows land one fp16 ulp under their own bound and the trellis
    clips them to the table's extreme entry anyway.  Measured (issue #87,
    ``experiments/reach_land_census.py``): 29609 of the 87870 raised rows over
    196 dense Qwen3-0.6B Linears at E4M3/L=14, 16-44% per unit, worst shortfall
    3.2e-4 relative -- one ulp, as the mechanism predicts.  The clipped energy
    is 9.1e-10 of the h-weighted budget on the worst unit, which is why this is
    stated rather than fixed: closing it would move a row word by one ulp on
    every CHANNEL artifact with rows past the reach, and that is a re-cut of the
    E4M3 wire under an unchanged ``encoder_profile_id``.

    The BF16 wire has no shortfall at all, and not by luck: its reach is exactly
    4.0 and its source words are bf16, so ``amax / 4`` over a power-of-two
    global is exact in fp16.  0 of 100575 raised rows landed short.

    ``refit_channel_scale``'s ``floor`` lands the *same* bound with
    ``land_at_least``, which rounds **up**.  The two halves of one rule
    therefore differ by design and not by oversight: the refit re-imposes the
    bound exactly, the start honours it to a rounding.  Anything that reads
    this start as an exact guarantee -- as the refit's ``floor`` docstring
    does when it says "which no later step recovers" -- is reading one ulp too
    much into it.

    Why per row and not per unit: the source model the alphabet and the
    table were fit to is a Gaussian, whose rows exceed the E4M3 table's
    4.08-sigma reach 4.5% of the time at width 1024.  Qwen3-0.6B's rows do
    so 25% of the time (59% of ``down_proj`` rows, up to 30 sigma), and the
    clipped entries sat in the Hessian-dominant columns.  Measured there
    (``experiments/dense_spread_fix.py``, ``dense_spread_census.py``): the
    worst unit's H-weighted error 0.410 -> 0.058 against production NVFP4
    GPTQ+JSO at 0.053, the 196-tensor geomean 0.0872 -> 0.0765 (NVFP4
    0.0955) with plain error 0.0742 -> 0.0704; a global lower sigma moved
    nothing.

    Returns ``(stored fp16 [rows], effective fp32 [rows], global)``.
    """
    if not sigma > 0:
        raise GrammarError(f"the channel source sigma must be positive, got {sigma}")
    w = work.float()
    rms = w.pow(2).mean(dim=1).sqrt()
    row_sigma = torch.full_like(rms, float(sigma))
    if reach is not None:
        if not reach > 0:
            raise GrammarError(f"the body's reach must be positive grid units, got {reach}")
        amax = w.abs().amax(dim=1)
        # A row's largest weight in grid units is ``amax / rms * sigma``; past
        # the reach the row starts at ``reach * rms / amax`` instead.
        over = amax * float(sigma) > float(reach) * rms
        row_sigma = torch.where(over, float(reach) * rms / amax.clamp_min(1e-30), row_sigma)
    scale = rms / row_sigma
    global_scale = channel_global(scale)
    stored, effective = land_channel_scale(scale, global_scale)
    return stored, effective, global_scale


def land_at_least(
    floor: torch.Tensor, global_scale: float
) -> "tuple[torch.Tensor, torch.Tensor]":
    """The smallest fp16 word over ``global_scale`` that is at least ``floor``.

    ``land_channel_scale`` rounds to nearest, which may land a hair *below* a
    scale that was computed as a lower bound.  A reach floor that is missed by
    one ulp clips the very weight it was raised for, so this rounds up: cast,
    then step to the next fp16 word up where the cast fell short.

    The step is taken on the **bit pattern**, not by multiplying by
    ``1 + 2^-10``.  Every word here is positive and finite, so incrementing the
    stored bits is exactly ``nextafter(w, +inf)`` and the result is the
    smallest word that clears the floor -- multiply-then-round is not: it
    overshoots by two ulps on 263 of 1500 sampled floors, and at the top of the
    range ``65504 * (1 + 2^-10)`` rounds to **infinity**, whose ``0 * inf`` in
    the refit's ``A s^2 - 2 B s`` is a NaN that then decides the accept test by
    comparing false.  A floor no fp16 word over this global can carry is
    refused here instead: saturating would silently reinstate the very clip the
    floor exists to prevent.
    """
    stored, effective = land_channel_scale(floor, global_scale)
    short = effective < floor
    if bool(short.any()):
        bits = stored.contiguous().view(torch.int16)
        bumped = (bits + 1).view(torch.float16)
        stored = torch.where(short, bumped, stored)
        effective = stored.float() * global_scale
        over = ~torch.isfinite(effective)
        if bool(over.any()):
            raise GrammarError(
                f"{int(over.sum())} of {effective.numel()} row scale floors are above "
                f"fp16's range over a global of {global_scale}; the largest is "
                f"{float(floor.float().max())} and the largest word is "
                f"{_FP16_MAX * global_scale}"
            )
        if bool((effective < floor).any()):
            raise GrammarError(
                "landing a row scale upward left it below its floor; one ulp above "
                "a round-to-nearest cast should always clear it"
            )
    return stored, effective


def refit_channel_scale(
    work: torch.Tensor,
    units: torch.Tensor,
    stored: torch.Tensor,
    global_scale: float,
    metric: "torch.Tensor | None" = None,
    floor: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """One least-squares step on the row scales, landed on fp16 words.

    With the codes fixed a row's error is ``A s^2 - 2 B s + C`` in its scale,
    ``A = <u, u>`` and ``B = <w, u>`` over the row's unscaled grid values
    ``u``; the optimum ``B / A`` is landed on a word and kept only where the
    landed word lowers the row's error, so no row ends worse than it began and
    the *step* is monotone in squared error.

    The *alternation* with the trellis is monotone only where the trellis
    descends that same error: ``trellis_weighting="scale"`` (the exporter's
    default -- ``encode_unit``'s signature default ``"none"`` minimises the
    normalised error instead), no ``ldl`` (which points the trellis at the
    compensated target while this step refits against ``work``), and
    ``metric=None``, since the Viterbi's branch metric never sees ``metric``.
    Elsewhere the step is monotone in its own error and the pair is not.

    **A row with ``B <= 0`` keeps the scale it has**, which is the hold the LUT
    plane's refit (``encode._refit_scales_lut``) and S6b's both carry.  Such a
    row's codes anti-correlate with its weights, so the parabola has no
    positive minimiser: it falls all the way to ``s -> 0``, the word floors on
    fp16's smallest positive value, and the accept test *prefers* it because
    ``A s^2 - 2 B s`` really is smaller there.  The row then decodes to about
    nothing, and the next trellis pass sees ``work / scale`` at four orders of
    magnitude, which is the residual it has to spend the column's shared path
    on.  A scale that is optimal for frozen codes and ruinous for the codes
    that follow is not a step this alternation may take.

    ``metric`` changes which error that is, and nothing else.  ``None`` is the
    plain squared error above.  A 1-D ``[cols]`` tensor is a per-input-column
    weight (the diagonal-Hessian objective ``sum_j h_j e_ij^2``, or any power
    of it).  A 2-D ``[cols, cols]`` tensor is the full Hessian, for which the
    row's true proxy loss ``(w_r - s u_r) H (w_r - s u_r)^T`` is the same
    scalar quadratic with ``A = u_r H u_r^T`` and ``B = w_r H u_r^T`` -- the
    exact form, with no exponent to choose.  The monotone guard is evaluated
    under whichever metric was asked for, so the *step* stays monotone in
    *that* error -- the alternation does not, per the paragraph above.

    ``floor`` is a per-row lower bound on the effective scale, landed upward.
    It is how the body's reach survives a refit: a least-squares step is
    amax-blind, and a row whose target exceeds ``reach * s`` clips at the
    trellis, which no later step recovers (``initial_channel_scale``).  The
    bound is applied after the guard, so a row is never left both clipping and
    "improved".
    """
    W = work.float()
    U = units.float()
    if metric is None:
        A = (U * U).sum(dim=1)
        B = (W * U).sum(dim=1)
    else:
        check_refit_metric(metric, W.shape[1])
        if metric.ndim == 1:
            m = metric.to(W.dtype).to(W.device).reshape(1, -1)
            A = (U * U * m).sum(dim=1)
            B = (W * U * m).sum(dim=1)
        else:
            H = metric.to(W.dtype).to(W.device)
            UH = U @ H
            A = (UH * U).sum(dim=1)
            B = (UH * W).sum(dim=1)
    old_eff = stored.float() * global_scale
    valid = (A > 0) & (B > 0)
    candidate = torch.where(valid, B / A.clamp_min(1e-30), old_eff)
    new_stored, new_eff = land_channel_scale(candidate, global_scale)
    err_old = A * old_eff * old_eff - 2 * B * old_eff
    err_new = A * new_eff * new_eff - 2 * B * new_eff
    keep_new = valid & (err_new < err_old)
    stored_out = torch.where(keep_new, new_stored, stored)
    if floor is not None:
        floor_stored, floor_eff = land_at_least(floor.float(), global_scale)
        below = stored_out.float() * global_scale < floor_eff
        stored_out = torch.where(below, floor_stored, stored_out)
    return stored_out, stored_out.float() * global_scale


def channel_scale_field(
    stored: torch.Tensor, global_scale: float, rows: int, cols: int
) -> torch.Tensor:
    """The per-position scale a CHANNEL plane decodes to: ``[rows, cols]`` fp32."""
    if stored.numel() != rows:
        raise GrammarError(
            f"a CHANNEL plane holds one word per output row: {stored.numel()} words for {rows} rows"
        )
    return (stored.float() * float(global_scale)).view(rows, 1).expand(rows, cols)
