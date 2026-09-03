"""Issue #79: ``_fit_lut``'s stop test must come from the cost dtype, not a literal.

``encode._fit_lut`` runs its swap refinement to convergence -- the ``swaps``
cap is only the backstop -- so the thing that actually stops it is the
relative-improvement accept test. The rule is the house rule: below one ulp
of the running cost, in the dtype the cost is accumulated in, an
"improvement" is rounding noise rather than a descent step, so the threshold
is ``torch.finfo(cost.dtype).eps`` scaled by the current cost.

These tests drive the real ``_fit_lut`` swap loop with a scripted cost, so the
accept/reject decision is exact instead of data-dependent: a sub-ulp
improvement must leave the table exactly as the no-refinement run left it,
while a multi-ulp one must move it -- in float32 *and* float64, which is what
pins the threshold to the dtype rather than to any one constant. The old
``1e-9`` literal sits between the two epsilons (far below float32's,
far above float64's), so each dtype discriminates it from a different side.
"""

import itertools

import torch

import tessera.encode as encode
from tessera.encode import _fit_lut


def _witness():
    gen = torch.Generator().manual_seed(3)
    targets = torch.exp(torch.randn(256, generator=gen) + 0.5).clamp(0.02, 7.0)
    weights = torch.exp(torch.randn(256, generator=gen) * 1.5)
    return targets, weights, float(2.0 ** -6)


def _scripted_cost(monkeypatch, base, trial, dtype):
    """Serve ``base`` once, ``trial`` once, then a strictly worse cost forever.

    The first ``_lut_cost`` call of a swap pass reads the running cost; the
    next call is the first candidate swap, which is the decision under test.
    Everything after that is worse, so at most one swap is ever taken and the
    run ends on the pass after it.
    """
    worse = base + abs(base)
    values = itertools.chain([base, trial], itertools.repeat(worse))
    monkeypatch.setattr(
        encode, "_lut_cost", lambda *args: torch.tensor(next(values), dtype=dtype)
    )


def _tables(targets, weights, global_scale, monkeypatch, base, trial, dtype):
    _scripted_cost(monkeypatch, base, trial, dtype)
    _, refined = _fit_lut(targets, weights, global_scale, 16, swaps=32)
    _, unrefined = _fit_lut(targets, weights, global_scale, 16, swaps=0)
    return refined, unrefined


def test_sub_ulp_improvement_is_not_a_step_float32(monkeypatch):
    """Half an ulp of a float32 running cost must not move the table."""
    targets, weights, global_scale = _witness()
    eps = torch.finfo(torch.float32).eps
    refined, unrefined = _tables(
        targets, weights, global_scale, monkeypatch,
        base=100.0, trial=100.0 * (1.0 - eps / 2.0), dtype=torch.float32,
    )
    assert torch.equal(refined, unrefined)


def test_two_ulp_improvement_is_a_step_float64(monkeypatch):
    """Two ulps of a float64 running cost must move the table.

    Two ulps in float64 is ~4e-14 relative -- four orders below any constant a
    float32-calibrated literal would accept -- so only a dtype-derived
    threshold takes this swap.
    """
    targets, weights, global_scale = _witness()
    eps = torch.finfo(torch.float64).eps
    refined, unrefined = _tables(
        targets, weights, global_scale, monkeypatch,
        base=100.0, trial=100.0 * (1.0 - 2.0 * eps), dtype=torch.float64,
    )
    assert not torch.equal(refined, unrefined)


def test_clear_improvement_is_still_taken_float32(monkeypatch):
    """Ten ulps of a float32 running cost must move the table (both regimes)."""
    targets, weights, global_scale = _witness()
    eps = torch.finfo(torch.float32).eps
    refined, unrefined = _tables(
        targets, weights, global_scale, monkeypatch,
        base=100.0, trial=100.0 * (1.0 - 10.0 * eps), dtype=torch.float32,
    )
    assert not torch.equal(refined, unrefined)
