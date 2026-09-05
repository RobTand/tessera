"""The learned tree codebook: distance stability and the split's fixed point.

``_hoist``'s tie semantics are pinned in ``test_audit_container_accounting``;
this file owns the numerics of the fit itself -- that the distance evaluation
does not cancel on translated data (tessera#227) and that the Lloyd split ends
at its own fixed point rather than at a pass count (tessera#228).
"""

from __future__ import annotations

import pytest
import torch

from tessera.codebook import _two_means, learn_tree_codebook
from tessera.errors import GrammarError

#: The tessera#227 witness: 64 distinct finite values, then the same cloud
#: translated far from the origin.  Every shifted input is exactly
#: representable up to half an ulp at the shifted magnitude, so a Euclidean
#: fit of the shifted cloud is the translated fit up to that input rounding
#: plus the accumulation error of the means -- nothing entitles it to merge
#: genuinely distinct centroids.
SHIFT = 10000.0


def _witness() -> torch.Tensor:
    return torch.linspace(-1, 1, 64).view(-1, 1)


def _translation_atol(shift: float, amax: float) -> float:
    """Derived from float32, not chosen: rounding each shifted input costs at
    most one ulp at ``shift + amax``; a mean over up to 64 such values adds at
    most ``log2(64) = 6`` ulps of pairwise-summation error plus one for the
    divide.  Eight ulps at the working magnitude bounds the lot."""
    return 8 * torch.finfo(torch.float32).eps * (shift + amax)


def test_two_means_translates_with_its_data():
    """tessera#227: the GEMM distance expansion ||x||^2 - 2<x,c> + ||c||^2
    subtracts large nearly-equal terms on uncentred data, and the small
    squared separations cancel to zero -- at SHIFT the two root centroids
    came back as (0, +spread) relative to the shift, with the lower half of
    the cloud unrepresented.  Distances taken as direct squared differences
    subtract first, so the translated fit is the fit, translated."""
    base = _witness()
    reference = _two_means(base)
    shifted = _two_means(base + SHIFT)
    atol = _translation_atol(SHIFT, 1.0)
    assert torch.allclose(shifted - SHIFT, reference, atol=atol, rtol=0.0)


def test_tree_codebook_translates_without_duplicate_leaves():
    """tessera#227's full collapse: at SHIFT the depth-2 grid came back as
    one lower value and three duplicates -- empty cells manufactured by a
    wrong distance, not by the sanctioned too-small-to-split fallback.  The
    translated grid must keep every leaf distinct and sit on the untranslated
    grid up to the derived input rounding."""
    base = _witness()
    reference = learn_tree_codebook(base, depth=2)
    shifted = learn_tree_codebook(base + SHIFT, depth=2)
    assert len(set(shifted.values)) == len(set(reference.values)) == 4
    atol = _translation_atol(SHIFT, 1.0)
    assert torch.allclose(
        torch.tensor(sorted(shifted.values)) - SHIFT,
        torch.tensor(sorted(reference.values)),
        atol=atol, rtol=0.0,
    )


def test_fitting_is_deterministic():
    """The docstring's promise: same tensor, same depth, same grid."""
    samples = torch.randn(256, 2, generator=torch.Generator().manual_seed(7))
    assert (learn_tree_codebook(samples, depth=3).values
            == learn_tree_codebook(samples, depth=3).values)

def test_two_means_ends_at_its_own_fixed_point():
    """tessera#228: a 12-pass cap stopped a descent that was still moving
    (the witness's centroids change on the thirteenth pass).  The split must
    return only at its fixed point: one more Lloyd pass -- the same
    assignment rule, the same update -- reproduces the returned centroids
    exactly.  Exact equality, because a fixed point of a deterministic map
    needs no tolerance."""
    points = torch.randn(256, 2, generator=torch.Generator().manual_seed(19))
    centroids = _two_means(points)
    assign = (points.unsqueeze(1) - centroids.unsqueeze(0)).square().sum(2).argmin(1)
    again = centroids.clone()
    for side in (0, 1):
        if int((assign == side).sum()):
            again[side] = points[assign == side].mean(0)
    assert torch.equal(again, centroids)


def test_two_means_backstop_refuses_rather_than_truncates():
    """The pass budget is a backstop, not an answer: a budget that binds is
    an unfinished descent, and it is refused by name rather than returned as
    if it had converged."""
    points = torch.randn(256, 2, generator=torch.Generator().manual_seed(19))
    with pytest.raises(GrammarError, match="fixed point"):
        _two_means(points, iterations=3)
