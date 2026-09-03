"""The rate axis is not monotone, and the menu says which rungs that costs (#43).

Two claims are pinned here, in the order they have to be made:

1. **The accountant is exact.**  ``unit_wire_bits`` must equal
   ``encode_linear(...).exact_bytes * 8`` on both bodies and both arities.  It
   did not until 2026-09-02: a TCQ body's ALPHABET and DESCENDANT planes were
   priced at zero, so every arity-1 E2M1 unit came out 20-44 B light and every
   E2M1x2 unit at the coset cap 512 B light, while every window unit was
   exact.  A count of dominated rungs taken with that accountant is a count of
   an artifact of the accountant.

2. **Domination is real, and it is a shape effect.**  With the accountant
   exact, 87 of 385 legal rungs at 96x320 and 160 of 769 at 64x512 are matched
   or beaten by a higher rung, against 0 at 512x2048 and 1024x3072.

The "no worse" leg is measured too, not assumed:
``test_a_dominated_rung_is_worse_on_both_axes`` decodes both arms.

CPU only; the encodes here are seconds each by design.
"""

from fractions import Fraction

import pytest
import torch

from tessera.calculator import terminal_rate
from tessera.control import GRID_NAMES, grid_for_name, rate_menu, unit_wire_bits
from tessera.errors import GrammarError
from tessera.export import encode_linear
from tessera.grammar import forest_plane_bytes
from tessera.unit_artifact import read_unit_artifact

#: The counts measured by ``experiments/tessera_dominated_rungs.py``:
#: ``(grid, rows, columns) -> (legal rungs, dominated rungs)``.
DOMINATED = {
    ("E2M1x2", 96, 320): (385, 87),
    ("E2M1x2", 64, 512): (769, 160),
    ("E2M1x2", 64, 640): (769, 115),
    ("E2M1x2", 96, 768): (769, 35),
    ("E2M1x2", 512, 2048): (769, 0),
    ("E2M1x2", 1024, 3072): (769, 0),
    # A second, independent mechanism, and one the issue does not report: an
    # arity-1 schedule that spans two distinct rates carries two forests, so
    # R511 outweighs the uniform R512 above it.  Two rungs, 16 bytes, real on
    # exported bytes.
    ("E2M1", 64, 512): (513, 2),
    ("E2M1", 1024, 3072): (513, 0),
    # Window bodies at every rung: no forest, one table width, no domination.
    ("E4M3", 96, 320): (449, 0),
    ("BF16", 96, 320): (961, 0),
}


@pytest.mark.parametrize("key,expected", sorted(DOMINATED.items()))
def test_dominated_rungs_are_a_shape_effect(key, expected):
    name, rows, columns = key
    menu = rate_menu(name, rows, columns)
    assert (len(menu.prices), len(menu.dominated)) == expected


@pytest.mark.parametrize("name", GRID_NAMES)
@pytest.mark.parametrize("shape", [(96, 320), (64, 512), (96, 768), (1024, 3072)])
def test_the_offered_menu_is_strictly_increasing_in_bits(name, shape):
    """The property a bisection or a monotone DP over the axis needs.

    It is false of the raw axis on a small unit, which is the defect; it is
    true of what :func:`rate_menu` offers, which is the fix.
    """
    menu = rate_menu(name, *shape)
    offered = menu.offered
    assert offered, (name, shape)
    for lower, higher in zip(offered, offered[1:]):
        assert higher.q256 > lower.q256
        assert higher.bits > lower.bits, (name, shape, lower.q256, higher.q256)


def test_the_top_rung_is_always_offered_and_names_its_dependents():
    """Nothing is above the ceiling, so the ceiling can never be dominated."""
    menu = rate_menu("E2M1x2", 64, 512)
    assert menu.prices[-1].q256 == 896 and menu.prices[-1].is_offered
    # every dominated rung here is dominated by the cap rung, which is the
    # single 4096-byte window table the whole effect is made of
    assert {price.dominated_by for price in menu.dominated} == {896}
    assert min(price.q256 for price in menu.dominated) == 736


def test_the_raw_axis_really_is_non_monotone_at_the_coset_cap():
    """R896 weighs less than R895 on a small unit: 0.1350 bpp at 96x768.

    The issue reported 0.1905 from PrismaQuant's accountant, which does not
    charge the cap rung's forest (RobTand/prismaquant#126).  The exporter's
    own plane ranges say 0.1350, and so does this one now.
    """
    rows, columns = 96, 768
    below = unit_wire_bits("E2M1x2", 895, rows, columns)
    at_cap = unit_wire_bits("E2M1x2", 896, rows, columns)
    assert at_cap < below
    assert float(Fraction(below - at_cap, rows * columns)) == pytest.approx(
        0.13498, abs=1e-5
    )


def test_a_menu_is_pruned_at_one_shape_and_is_wrong_at_another():
    """The same rung is dominated on a small unit and offered on a large one."""
    assert not rate_menu("E2M1x2", 64, 512).price(736).is_offered
    assert rate_menu("E2M1x2", 1024, 3072).price(736).is_offered


def test_rate_menu_refuses_a_rung_the_grammar_does_not_admit():
    menu = rate_menu("E2M1x2", 64, 512)
    with pytest.raises(GrammarError, match="not a legal rung"):
        menu.bpp(1024)
    assert menu.bpp(896) == Fraction(unit_wire_bits("E2M1x2", 896, 64, 512), 64 * 512)


def test_the_json_records_what_was_pruned_and_why():
    block = rate_menu("E2M1", 64, 512).to_json()
    assert block["grid"] == "E2M1" and block["shape"] == [64, 512]
    assert block["dominated"] == {"511": 512, "767": 768}
    assert "tessera#43" in block["reason"]


# ------------------------------------------------- the accountant is exact

#: ``(grid, q256, rows, columns)``: both bodies, both arities, both sides of
#: the coset cap, and the arity-1 pair whose two-rate schedule carries two
#: forests against the uniform rung above it.
EXACT_CASES = (
    ("E2M1", 511, 32, 256),
    ("E2M1", 512, 32, 256),
    ("E2M1x2", 895, 32, 384),
    ("E2M1x2", 896, 32, 384),
)


@pytest.mark.parametrize("name,q256,rows,columns", EXACT_CASES)
def test_the_accountant_prices_what_the_exporter_writes(name, q256, rows, columns):
    grid = grid_for_name(name)
    torch.manual_seed(11)
    unit = encode_linear(torch.randn(rows, columns), grid=grid, q256=q256)
    assert unit.exact_bytes * 8 == int(unit_wire_bits(grid, q256, rows, columns))


def test_the_forest_charge_is_opt_in_so_the_published_figures_still_mean_what_they_meant():
    """``terminal_rate`` prices position planes by default, the wire on request.

    The calculator's published figures are position-domain rates derived
    against empty forest blobs, and ``tests/test_calculator.py`` pins them to
    exact fractions.  ``with_forest`` is what a caller pricing a *unit* passes;
    the difference between the two is the forest and nothing else.
    """
    rows, columns, q256 = 64, 512, 512
    plain = terminal_rate(q256, rows, columns, cap=3)
    charged = terminal_rate(q256, rows, columns, cap=3, with_forest=True)
    assert charged - plain == Fraction(8 * sum(forest_plane_bytes((2,), 3)), rows * columns)
    # a window body has no forest, so the flag cannot move it
    window = dict(window_bits=12, with_scale_base=False, with_scale_refine=True, cap=7)
    assert terminal_rate(q256, rows, columns, **window) == terminal_rate(
        q256, rows, columns, with_forest=True, **window
    )


def test_forest_plane_bytes_is_arithmetic_in_the_schedule():
    """One descendant block per distinct rate, ``2^(cap+1)`` bytes each."""
    assert forest_plane_bytes((7,), 7) == (256, 256)
    assert forest_plane_bytes((6, 7), 7) == (128 + 256, 256 + 256)
    assert forest_plane_bytes((2,), 3) == (8, 16)
    assert forest_plane_bytes((1, 2), 3) == (4 + 8, 16 + 16)


# --------------------------------------------------------- the other axis


def test_a_dominated_rung_is_worse_on_both_axes():
    """Dominated has to mean *no more bytes and no worse*, so both are measured.

    At 32x384 on E2M1x2 the cap rung R896 weighs 6656 B against R363's 6658 --
    two bytes fewer -- and its decoded unit is 20x closer to the source.  One
    Gaussian unit, weight space: the scope of the claim, and enough to settle
    that the domination is not a byte artifact pointing at a quality gain.
    """
    grid = grid_for_name("E2M1x2")
    rows, columns = 32, 384
    torch.manual_seed(7)
    weight = torch.randn(rows, columns)
    denom = float((weight.double() ** 2).sum())

    def arm(q256):
        unit = encode_linear(weight, grid=grid, q256=q256)
        decoded = read_unit_artifact(unit.blob).to(torch.float64)
        return unit.exact_bytes, float(((decoded - weight.double()) ** 2).sum()) / denom

    menu = rate_menu(grid, rows, columns)
    worst = min(price.q256 for price in menu.dominated)
    assert worst == 363 and menu.price(363).dominated_by == 896

    dominated_bytes, dominated_sse = arm(worst)
    cap_bytes, cap_sse = arm(896)
    assert cap_bytes <= dominated_bytes
    assert cap_sse < dominated_sse


def test_the_uniform_control_reports_a_dominated_rung_rather_than_correcting_it():
    """A control on a dominated rung is a handicapped uniform arm, and says so.

    ``uniform_control`` still picks by bits -- matching bytes is its contract,
    and the dominating rung is in general *further* from the candidate -- so
    the finding is reported, not silently repaired.
    """
    from tessera.control import PlannedUnit, uniform_control

    # a small-unit plan whose candidate sits on a dominated rung
    units = [PlannedUnit(f"m{i}.weight", "E2M1x2", 800, 64, 512) for i in range(4)]
    control = uniform_control(units)
    assert control.q256 == 800 and control.dominated_by == 896
    assert control.to_json()["dominated_by"] == 896

    # and a production-shaped plan, where there is nothing to report
    big = [PlannedUnit(f"m{i}.weight", "E2M1x2", 800, 1024, 3072) for i in range(4)]
    assert uniform_control(big).dominated_by is None
