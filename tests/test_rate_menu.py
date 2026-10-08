"""The accountant prices the served wire and records byte domination.

These tests compare actual encoded plane bytes with the unit price.
Scalar TCQ still has two-rate forest costs. Paired serving uses WINDOW L14.
"""

from fractions import Fraction

import pytest
import torch

from tessera.calculator import terminal_rate
from tessera.control import GRID_NAMES, grid_for_name, rate_menu, unit_wire_bits
from tessera.errors import GrammarError
from tessera.export import encode_linear, rung_ceiling, served_recipe
from tessera.grammar import forest_plane_bytes

#: The counts measured by ``experiments/tessera_dominated_rungs.py``:
#: ``(grid, rows, columns) -> (legal rungs, dominated rungs)``.
DOMINATED = {
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


@pytest.mark.parametrize("name", GRID_NAMES)
def test_the_top_rung_is_always_offered(name):
    grid = grid_for_name(name)
    menu = rate_menu(grid, 64, 512)
    from tessera.manifest import body_rate_cap
    recipe = served_recipe(grid, rung_ceiling(grid))
    ceiling = body_rate_cap(recipe.body, grid) * 256 // grid.arity
    assert menu.price(ceiling).is_offered


def test_rate_menu_refuses_a_rung_above_the_grid_ceiling():
    grid = grid_for_name("E2M1x2")
    menu = rate_menu(grid, 64, 512)
    with pytest.raises(GrammarError, match="not a legal rung"):
        menu.bpp(rung_ceiling(grid) + 1)
    assert menu.bpp(1024) == Fraction(unit_wire_bits(grid, 1024, 64, 512), 64 * 512)


def test_the_json_records_what_was_pruned_and_why():
    block = rate_menu("E2M1", 64, 512).to_json()
    assert block["grid"] == "E2M1" and block["shape"] == [64, 512]
    assert block["dominated"] == {"511": 512, "767": 768}


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
    recipe = served_recipe(grid, q256)
    unit = encode_linear(torch.randn(rows, columns), grid=grid, q256=q256,
        body=recipe.body, span=recipe.span, scale_plane=recipe.scale_plane,
        window_bits=recipe.window_bits, window_seed=recipe.window_seed)
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


