"""E4M3 ``channel_sigma``: the ceiling is a dtype constant, and below it the
spread is a gauge (issue #36).

The E4M3 recipe builds its window table at ``sigma = channel_sigma`` (no
``window_sigma``), so the table's largest entry is ``snap(z_max(L) * sigma)``
capped at the format's peak, 448.  Above ``448 / z_max(L)`` the top of the
table pins at the peak and the body's reach in row-RMS units, ``448 /
sigma``, falls with every further increase -- that is the steep side of the
cliff the issue measured.  Below it E4M3's log spacing makes a x2 shift of
``sigma`` an exact gauge on every entry in the normal range: the same codes
decode to values exactly twice as large, the rows are scaled by exactly half,
and nothing about the encode changes but the handful of near-zero entries
that cross the subnormal floor -- that is the flat side.  The dyadic ladder
that picks the default walks that gauge, which is why it cannot see the axis.

These tests hold the derived facts so the default cannot drift onto the
cliff unnoticed (a wider table, or a ladder change, moves ``z_max(L) * sigma``
towards 448) and record why the ladder's pick is a gauge choice, not an
optimum: measured on the receipt
``docs/measurements/tessera-e4m3-reach-cliff-2026-09-03.md``.
"""
import math

import pytest
import torch

from tessera.alphabet import E4M3_GRID, GAUSSIAN_SOURCE
from tessera.encode import grid_vector_table, window_table
from tessera.export import E4M3_RECIPE, E4M3_WINDOW_BITS
from tessera.scale_channel import _ladder_relative_error, default_channel_sigma

PEAK = max(abs(v) for v in E4M3_GRID.values)          # 448: E4M3FN's 0x7E
MIN_NORMAL = 2.0 ** -6                                 # below it the grid is subnormal


def _z_max(window_bits: int) -> float:
    return max(abs(z) for z in GAUSSIAN_SOURCE(1 << window_bits, 1.0))


def _table_values(sigma: float, window_bits: int = E4M3_WINDOW_BITS) -> torch.Tensor:
    codes = window_table(E4M3_GRID, window_bits, sigma=sigma, seed=0, half=16, device="cpu")
    return grid_vector_table(E4M3_GRID, "cpu")[codes.long()]


def test_the_e4m3_recipe_pins_the_table_to_the_row_spread():
    """The wire's E4M3 recipe sets no ``window_sigma``, so the table is built
    at ``channel_sigma`` and its reach in row-RMS is ``z_max(L)``, snapped."""
    assert E4M3_RECIPE.window_sigma is None
    assert E4M3_RECIPE.channel_sigma is None
    assert E4M3_RECIPE.window_bits == E4M3_WINDOW_BITS == 14
    assert PEAK == 448.0


def test_the_default_spread_sits_below_the_format_ceiling():
    """``z_max(L) * default_channel_sigma < 448`` at the recipe's width: no
    table entry is clipped and none sits on the peak.  The margin is a fact
    about the ladder's pick (0.247 binades at L=14, 0.042 at L=18), not a
    designed quantity, which is what this test exists to notice."""
    sigma0 = default_channel_sigma(E4M3_GRID)
    ceiling = PEAK / _z_max(E4M3_WINDOW_BITS)
    assert sigma0 < ceiling, (sigma0, ceiling)
    values = _table_values(sigma0).abs()
    assert float(values.max()) < PEAK
    assert int((values >= PEAK).sum()) == 0
    # The bound as a function of width: the widest table the pick admits
    # without clipping is L=18 (435 < 448); L=20 would clip.  Recipe width 14.
    widest = max(L for L in range(8, 21) if _z_max(L) * sigma0 <= PEAK)
    assert widest == 18
    assert E4M3_WINDOW_BITS <= widest


def test_above_the_ceiling_the_reach_falls_as_448_over_sigma():
    """Every spread past the ceiling pins the table's top at the peak, so the
    reach in row-RMS units is exactly ``448 / sigma`` -- the coordinate the
    issue's x2 / x4 arms actually moved."""
    sigma0 = default_channel_sigma(E4M3_GRID)
    for m in (1.5, 2.0, 4.0):
        values = _table_values(sigma0 * m).abs()
        assert float(values.max()) == PEAK
        assert float(values.max()) / (sigma0 * m) == pytest.approx(PEAK / (sigma0 * m))
        # and the entries that would have lain past the peak collapse onto it
        assert float((values >= PEAK).float().mean()) > 0.0


def test_below_the_ceiling_a_dyadic_shift_is_an_exact_gauge():
    """``table(sigma / 2) * 2 == table(sigma)`` on every entry that stays in
    E4M3's normal range: the codes differ, the decoded values are the same
    tensor up to the power of two the row scale absorbs.  Only entries that
    fall below ``2^-6`` (a few parts in a hundred thousand of the variance)
    can move, which is the whole content of the issue's 'free to five digits'."""
    sigma0 = default_channel_sigma(E4M3_GRID)
    hi = _table_values(sigma0)
    for k in (1, 2, 3):
        lo = _table_values(sigma0 / 2 ** k)
        normal = lo.abs() >= MIN_NORMAL
        assert bool(normal.float().mean() > 0.999)
        assert torch.equal(lo[normal] * 2 ** k, hi[normal])
        moved = (lo * 2 ** k != hi)
        assert float((hi[moved] ** 2).sum() / (hi ** 2).sum()) < 1e-5


def test_the_ladder_criterion_is_periodic_in_the_binade_on_e4m3():
    """The ladder's relative RTN error is the same in every normal binade to
    floating-point noise -- ``rel(k) == rel(k + 4)`` for every rung under the
    ceiling -- so which binade it picks is decided by iteration order, and
    the 1.1% ripple inside a binade is the mantissa phase of a 4096-sample
    Gaussian.  The pick is a gauge choice; the test records that it is."""
    scalar = torch.tensor(sorted(set(E4M3_GRID.values)), dtype=torch.float64)
    sample = torch.tensor(GAUSSIAN_SOURCE(1 << 12, 1.0), dtype=torch.float64)
    rel = [_ladder_relative_error(scalar, sample, PEAK * 2.0 ** (-k / 4)) for k in range(40)]
    best = min(rel)
    # Equal to floating-point noise while the sample's smallest quantiles
    # stay on the normal grid (k <= 24: sigma >= 7, smallest |z| sigma > 2^-9);
    # below that the subnormal grid bends the period by 1e-7..1e-5 relative,
    # and the band bound (a 1.1% ripple) holds all the way down.
    for k in range(8, 25):
        assert rel[k] == pytest.approx(rel[k + 4], rel=1e-6), k
    for k in range(8, 40):
        assert rel[k] / best < 1.012, (k, rel[k] / best)
    # The ladder's own cliff: the 4096-sample Gaussian reaches 3.67 sigma, so
    # its error climbs once 3.67 * sigma passes 448 -- one rung ABOVE the
    # table's ceiling, because the table's z_max(14) = 4.01 is larger.
    assert rel[7] / best > 1.05 and rel[6] / best > 1.5
    assert math.isclose(default_channel_sigma(E4M3_GRID), PEAK * 2.0 ** (-9 / 4))
    assert _z_max(12) < _z_max(14)
