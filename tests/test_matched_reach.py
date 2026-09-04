"""A matched-reach arm is matched or it does not exist (issue #18).

``experiments/matched_reach.py`` builds the arms that separate a window
table's two entangled axes -- how many entries it has, and how far its
outermost entry reaches -- by encoding one width at another width's reach.
The whole construction rests on one property: the ratio it returns must make
the table land on *exactly* the target reach, because a table that lands one
grid step away is not a control, it is a second treatment.

The tests below pin that property and the reason it needs a search:

* the diagonal is ratio 1.0 -- a width at its own reach is the shipped arm;
* every off-diagonal ratio realises the target exactly, on the built table;
* the naive ``target / own_reach`` division misses on at least one cell of
  the shipped BF16 grid, which is why the module bisects instead of dividing;
* a target no step of a width lands on raises rather than returning a near
  miss;
* and the table the helper reads is the one ``encode_unit`` builds -- the
  sigma, the seed, the half and the grid are the recipe's own, so a matched
  arm is matched in the encode and not only in the report.

Needs torch (the table build lives in ``tessera.encode``), so it is one of
the modules ``tests/conftest.py`` drops from the pure lane.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "matched_reach", ROOT / "experiments" / "matched_reach.py")
matched_reach = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matched_reach)

from tessera.alphabet import BF16_GRID  # noqa: E402
from tessera.export import BF16_CHANNEL_SIGMA as SIGMA  # noqa: E402

WIDTHS = [12, 14, 16]


@pytest.fixture(scope="module")
def grid():
    return matched_reach.reach_grid(BF16_GRID, WIDTHS, base_sigma=SIGMA)


def test_own_reaches_are_the_shipped_ones(grid):
    """The reaches the #18 receipt names, from the code rather than the doc."""
    assert grid["own"] == {12: 3.671875, 14: 4.0, 16: 4.3125}


def test_diagonal_is_the_shipped_ratio(grid):
    """A width asked for its own reach must accept ratio 1.0.

    Not "returns something near 1.0": the interval of ratios delivering that
    reach has to contain the value the recipe actually stores, or the arm the
    sweep runs at ratio 1.0 is not the diagonal of this factorial.
    """
    for L in WIDTHS:
        lo, hi = grid["cells"][(L, L)]["interval"]
        assert lo <= 1.0 <= hi, (L, lo, hi)


def test_every_cell_lands_on_its_target_exactly(grid):
    """The built table's realised reach equals the target, to the bit."""
    for (L, target_L), cell in grid["cells"].items():
        got = matched_reach.realised_reach(BF16_GRID, L, cell["ratio"] * SIGMA)
        assert got == grid["own"][target_L], (L, target_L, got)
        assert cell["realised"] == grid["own"][target_L]


def test_the_naive_division_misses(grid):
    """Why this module exists: ``target / own`` is not the matched ratio.

    The reach is the *snapped* outermost quantile, so it is a step function
    of the spread and the linear guess lands on the neighbouring step.  At
    least one cell of the shipped grid must miss, or a division would do and
    the search would be ceremony.
    """
    misses = []
    for L in WIDTHS:
        for target_L in WIDTHS:
            naive = grid["own"][target_L] / grid["own"][L]
            got = matched_reach.realised_reach(BF16_GRID, L, naive * SIGMA)
            if got != grid["own"][target_L]:
                misses.append((L, target_L, naive, got))
    assert misses, "the naive division hits every cell; the bisection is ceremony"


def test_unreachable_target_raises():
    """A near miss is refused, not returned."""
    with pytest.raises(ValueError):
        matched_reach.matched_ratio(BF16_GRID, 14, 3.6800001,
                                    base_sigma=SIGMA)
    with pytest.raises(ValueError):
        matched_reach.matched_ratio(BF16_GRID, 14, 4.0, base_sigma=SIGMA,
                                    lo=2.0, hi=8.0)


def test_interval_is_narrow_enough_to_retype(grid):
    """The reported interval is what says how much precision the arm needs."""
    for cell in grid["cells"].values():
        assert cell["width"] > 0
        assert cell["interval"][0] <= cell["ratio"] <= cell["interval"][1]


def test_the_diagnostic_reads_the_table_the_encoder_builds():
    """The reach helper and the BF16 encode share every table argument.

    ``rows_over_reach`` is computed from the same helper on both sides of a
    comparison, so it can only ever confirm the *diagnostic*'s arithmetic --
    it is silent about whether the encoder built that table.  What makes the
    matched arm an arm is this identity instead, and it is four constants:

    * the sigma.  ``bf16_l_sigma_sweep`` passes ``window_sigma = ratio *
      BF16_CHANNEL_SIGMA`` (``None`` at ratio 1.0), and ``encode_unit``'s
      CHANNEL branch sets ``table_sigma = window_sigma``, falling back to
      ``channel_sigma`` only when it is ``None`` -- the two spellings of the
      same number (``encode.py`` `table_sigma`).
    * the seed.  ``BF16_RECIPE.window_seed`` is ``DEFAULT_WINDOW_SEED`` is 0,
      which is ``matched_reach``'s default.
    * the half.  Read only on the ``sigma is None`` branch of
      ``_window_points_cpu``; the BF16 CHANNEL path never takes it, so the
      argument cannot move the table there.  Asserted rather than argued.
    * the grid.  ``BF16_GRID``, the recipe's own.
    """
    from tessera import encode as enc
    from tessera.export import BF16_RECIPE, DEFAULT_WINDOW_SEED

    assert BF16_RECIPE.window_seed == DEFAULT_WINDOW_SEED == 0
    assert BF16_RECIPE.window_sigma is None          # ratio 1.0 is the shipped arm
    assert BF16_RECIPE.channel_sigma == SIGMA
    assert BF16_RECIPE.scale_plane is enc.ScalePlaneKind.CHANNEL
    assert BF16_RECIPE.body is enc.BodyKind.WINDOW

    # ``half`` is inert once a sigma is given, so a matched arm cannot be
    # reading one table while the encoder builds another through it.
    for L in (12, 14):
        for sigma in (SIGMA, 1.25 * SIGMA):
            reaches = {
                enc.window_table_reach(BF16_GRID, L, sigma=sigma, seed=0,
                                       half=h).realised
                for h in (8, 16, 32)
            }
            assert len(reaches) == 1, (L, sigma, reaches)
