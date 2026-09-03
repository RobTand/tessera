"""The LUT table fit, solved exactly -- issue #50.

``_fit_lut`` minimises ``sum_b w_b (s_b - nearest(table, s_b))^2`` over sixteen
E4M3 values by greedy backward elimination plus swap passes.  That objective is
a one-dimensional weighted k-median over a sorted finite candidate set, which
has an exact dynamic program, so the greedy is a heuristic where an explicit
exists.  These tests pin the explicit against brute force on instances small
enough to enumerate, and against the greedy on instances that are not.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from tessera.encode import (
    E4M3_NORMAL_BYTES,
    _fit_lut,
    _fit_lut_exact,
    _lut_cost,
    e4m3_positive_values,
)


def _cost64(targets, weights, table):
    """``_lut_cost`` in float64 -- the fit's objective, evaluated exactly.

    The comparisons below are between two tables whose costs differ in the
    fourth or fifth digit, and float32 leaves that below its own noise.
    """
    t = targets.double()
    return float(_lut_cost(t, weights.double(), table.double()))


def _brute(targets, weights, candidates, entries):
    """The optimum by enumeration.  Only tractable for tiny ``candidates``."""
    best, arg = float("inf"), None
    for combo in itertools.combinations(range(candidates.numel()), entries):
        table = candidates[list(combo)]
        c = _cost64(targets, weights, table)
        if c < best:
            best, arg = c, combo
    return best, arg


@pytest.mark.parametrize("seed", range(8))
def test_exact_matches_brute_force_on_a_small_instance(seed):
    """Three entries out of a seven-value bracket, enumerated: the DP is optimal.

    ``_fit_lut_exact`` picks its candidates from the same bracket ``_fit_lut``
    does -- the grid values that straddle the targets -- so the brute force is
    run over exactly that bracket and the two answers must agree in cost.
    """
    g = torch.Generator().manual_seed(seed)
    grid = e4m3_positive_values()
    # A bracket of seven consecutive E4M3 values, and targets strictly inside
    # it so the fit's own bracketing picks the same seven.
    first = 40 + seed
    bracket = grid[first:first + 7]
    lo, hi = float(bracket[1]), float(bracket[5])
    targets = lo + (hi - lo) * torch.rand(64, generator=g)
    weights = torch.rand(64, generator=g) + 0.05

    bytes_, table = _fit_lut_exact(targets, weights, 1.0, entries=3)
    got = _cost64(targets, weights, table)
    best, _ = _brute(targets, weights, bracket, 3)
    assert got <= best * (1 + 1e-12), (got, best)
    # And it is a legal table: strictly ascending distinct E4M3 bytes.
    assert bytes_.numel() == 3
    assert torch.equal(bytes_.sort().values, bytes_)
    assert bytes_.unique().numel() == 3
    assert torch.allclose(
        bytes_.view(torch.float8_e4m3fn).float(), table, rtol=0, atol=0)


@pytest.mark.parametrize("seed", range(6))
def test_exact_is_never_worse_than_the_greedy(seed):
    """On sixteen entries the DP is optimal, so it cannot lose to the greedy.

    This is the claim issue #50's arm rests on: any gap between the two is the
    greedy's, and the DP's number is the best a sixteen-entry table can do.
    """
    g = torch.Generator().manual_seed(1000 + seed)
    n = 4096
    # Log-uniform over three binades: the shape a per-block amax field has.
    targets = torch.exp(torch.rand(n, generator=g) * 3.0 * 0.6931 - 4.0)
    weights = torch.rand(n, generator=g).pow(2) + 1e-3

    gb, gt = _fit_lut(targets, weights, 1.0)
    eb, et = _fit_lut_exact(targets, weights, 1.0)
    greedy, exact = _cost64(targets, weights, gt), _cost64(targets, weights, et)
    assert exact <= greedy * (1 + 1e-12), (exact, greedy)
    assert eb.numel() == gb.numel() == 16
    assert torch.equal(eb.sort().values, eb)
    assert eb.unique().numel() == 16


def test_dispatch_and_the_all_dead_unit():
    """``exact=True`` routes to the DP; a unit with no live weight is legal."""
    g = torch.Generator().manual_seed(7)
    targets = torch.rand(256, generator=g) + 0.5
    weights = torch.rand(256, generator=g)
    a = _fit_lut(targets, weights, 1.0, exact=True)
    b = _fit_lut_exact(targets, weights, 1.0)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    dead = torch.zeros(32)
    bytes_, table = _fit_lut_exact(torch.ones(32), dead, 1.0)
    assert bytes_.numel() == 16
    assert int(bytes_[0]) == E4M3_NORMAL_BYTES[0]


def test_a_degenerate_bracket_still_returns_sixteen_entries():
    """Targets spanning less than the table's width: the bracket widens.

    ``_fit_lut``'s bracket is one grid step wider than the targets on each
    side, which can hold fewer than sixteen values; both fits widen it until
    it holds sixteen, and the DP must return sixteen distinct ascending bytes
    either way.
    """
    grid = e4m3_positive_values()
    lo = float(grid[60])
    targets = torch.full((128,), lo) + torch.linspace(0, float(grid[61]) - lo, 128)
    weights = torch.ones(128)
    bytes_, table = _fit_lut_exact(targets, weights, 1.0)
    assert bytes_.numel() == 16
    assert bytes_.unique().numel() == 16
    assert torch.equal(bytes_.sort().values, bytes_)
    greedy = _fit_lut(targets, weights, 1.0)
    assert _cost64(targets, weights, table) <= _cost64(targets, weights, greedy[1]) * (1 + 1e-12)
