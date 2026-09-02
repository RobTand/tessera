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
    "land_channel_scale",
    "initial_channel_scale",
    "refit_channel_scale",
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
    against.  A zero or non-finite row lands on the smallest positive word so
    that no target is ever divided by zero; its codes decode to nothing
    either way.
    """
    if global_scale <= 0 or not math.isfinite(global_scale):
        raise GrammarError(f"the CHANNEL global scale must be positive and finite, got {global_scale}")
    word = scale.float() / global_scale
    word = torch.where(torch.isfinite(word) & (word > 0), word, torch.full_like(word, 2.0 ** -14))
    stored = word.clamp(2.0 ** -14, _FP16_MAX).to(torch.float16)
    return stored, stored.float() * global_scale


def initial_channel_scale(
    work: torch.Tensor, sigma: float
) -> "tuple[torch.Tensor, torch.Tensor, float]":
    """The amax-free initial plane: every row's RMS lands on ``sigma`` grid units.

    Returns ``(stored fp16 [rows], effective fp32 [rows], global)``.
    """
    if not sigma > 0:
        raise GrammarError(f"the channel source sigma must be positive, got {sigma}")
    rms = work.float().pow(2).mean(dim=1).sqrt()
    scale = rms / float(sigma)
    global_scale = channel_global(scale)
    stored, effective = land_channel_scale(scale, global_scale)
    return stored, effective, global_scale


def refit_channel_scale(
    work: torch.Tensor,
    units: torch.Tensor,
    stored: torch.Tensor,
    global_scale: float,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """One least-squares step on the row scales, landed on fp16 words.

    With the codes fixed a row's error is ``A s^2 - 2 B s + C`` in its scale,
    ``A = <u, u>`` and ``B = <w, u>`` over the row's unscaled grid values
    ``u``; the optimum ``B / A`` is landed on a word and kept only where the
    landed word lowers the row's error, so no row ends worse than it began
    and the alternation with the trellis is monotone in squared error.
    """
    W = work.float()
    U = units.float()
    A = (U * U).sum(dim=1)
    B = (W * U).sum(dim=1)
    old_eff = stored.float() * global_scale
    candidate = torch.where(A > 0, B / A.clamp_min(1e-30), old_eff)
    new_stored, new_eff = land_channel_scale(candidate, global_scale)
    err_old = A * old_eff * old_eff - 2 * B * old_eff
    err_new = A * new_eff * new_eff - 2 * B * new_eff
    keep_new = err_new < err_old
    stored_out = torch.where(keep_new, new_stored, stored)
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
