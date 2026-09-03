"""The plan memo is sized by the key space it is keyed over.

``export._plan_for`` builds the rate schedule and the anchor forests for one
``(grid, rung, width, body, sigma)``, and the forests are an exhaustive
per-rate optimisation -- which is why the memo exists.  It was sized ``256``,
a literal against a space nobody had counted, and stated in the wrong unit
besides (its docstring justified itself in *units sharing a plan*, which is
not what it is keyed on).  A campaign asks tens of rungs per shape across
tens of shapes, so a shape revisited after 256 intervening keys rebuilt its
forests.

Three of the five axes are countable and are counted
(:func:`export.plan_cache_bound`); the two that are not -- a width is whatever
a checkpoint presents, a sigma is a float -- key one memo per shape instead of
being folded in with a guessed factor.  These tests pin that rule: a second
pass over the whole countable space, at many shapes, must recompute nothing,
and the bound must move when the space moves.
"""
from __future__ import annotations

import pytest

from tessera import export
from tessera.alphabet import E2M1_GRID, E4M3_GRID, SERIALISABLE_GRIDS, tuple_grid
from tessera.errors import GrammarError
from tessera.export import _plan_for, rung_ceiling, tcq_cap_q256
from tessera.grammar import Q256_UNIT
from tessera.manifest import BodyKind

#: 256 columns realises every rung's root exactly (the Bresenham quota is
#: ``n_columns * frac``, and every root's denominator divides 256), so the
#: whole rung space is enumerable at one width.
COLUMNS = 256


@pytest.fixture(autouse=True)
def _empty_memo():
    _plan_for.cache_clear()
    yield
    _plan_for.cache_clear()


def _key_space():
    """Every ``(grid, rung, body)`` an artifact-writable grid can present."""
    for grid in SERIALISABLE_GRIDS.values():
        lo, remainder = divmod(Q256_UNIT, grid.arity)
        if remainder:
            continue
        for body, ceiling in ((BodyKind.TCQ, tcq_cap_q256(grid)),
                              (BodyKind.WINDOW, rung_ceiling(grid))):
            for q256 in range(lo, ceiling + 1):
                yield grid, q256, body


@pytest.fixture()
def counted_forests(monkeypatch):
    """``build_forest`` replaced by a counter.

    The memo's rule is what is under test, not the forest search: the real
    search is ~0.1 s per rate, which over the whole rung space is tens of
    minutes and would price this test out of the gate set.
    """
    calls = []

    def build_forest(rate, samples=None, grid=None):
        calls.append((rate, grid))
        return ("forest", rate, grid)

    monkeypatch.setattr(export, "build_forest", build_forest)
    return calls


def test_the_counted_space_is_far_larger_than_the_literal_it_replaced():
    space = sum(1 for _ in _key_space())
    assert space == export.plan_cache_bound()
    assert space > 256, (
        "the retired literal was 256; a bound below the space it is keyed over "
        "evicts entries a single pass is still filling"
    )


def test_the_bound_moves_with_the_grid_space(monkeypatch):
    """Derived, not listed: add a grid, the bound grows by that grid's interval."""
    extra = tuple_grid(E4M3_GRID, 2)
    assert extra not in SERIALISABLE_GRIDS.values()
    before = export.plan_cache_bound()
    monkeypatch.setattr(
        export, "SERIALISABLE_GRIDS", {**SERIALISABLE_GRIDS, "extra": extra})
    lo = Q256_UNIT // extra.arity
    grew = (tcq_cap_q256(extra) - lo + 1) + (rung_ceiling(extra) - lo + 1)
    assert export.plan_cache_bound() == before + grew


def test_an_empty_grid_space_is_refused_rather_than_sized_zero(monkeypatch):
    monkeypatch.setattr(export, "SERIALISABLE_GRIDS", {})
    with pytest.raises(GrammarError, match="counted 0 keys"):
        _plan_for(E2M1_GRID, 768, COLUMNS)


def test_a_second_pass_over_the_whole_key_space_recomputes_nothing(counted_forests):
    space = list(_key_space())
    for grid, q256, body in space:
        _plan_for(grid, q256, COLUMNS, body)
    first = _plan_for.cache_info()
    assert first.misses == len(space), "a key was answered from a stale entry"
    built = len(counted_forests)

    for grid, q256, body in space:
        _plan_for(grid, q256, COLUMNS, body)
    second = _plan_for.cache_info()
    assert second.misses == first.misses, (
        f"the second pass recomputed {second.misses - first.misses} of {len(space)} plans; "
        "the memo is smaller than the space it is keyed over"
    )
    assert second.hits == first.hits + len(space)
    assert len(counted_forests) == built, "a forest was rebuilt on the second pass"


def test_a_second_pass_over_many_shapes_recomputes_nothing(counted_forests):
    """The axis the literal actually broke: one rung set, many Linear widths.

    Widths and sigmas are the axes with no bound before a checkpoint is
    opened, so they must not share one fixed-size memo with the rung axis --
    which is what made a revisited shape rebuild its forests.
    """
    grid = E4M3_GRID
    rungs = range(Q256_UNIT, tcq_cap_q256(grid) + 1, 64)
    shapes = [(columns, sigma)
              for columns in (256, 512, 1024, 2560, 4096, 12288, 19456)
              for sigma in (None, 1.0, 20.5)]
    pass_ = [(grid, q256, columns, sigma)
             for columns, sigma in shapes for q256 in rungs]
    for grid_, q256, columns, sigma in pass_:
        _plan_for(grid_, q256, columns, BodyKind.TCQ, sigma)
    first = _plan_for.cache_info()
    assert first.misses == len(pass_) > 256
    assert first.shapes == len(shapes)

    for grid_, q256, columns, sigma in pass_:
        _plan_for(grid_, q256, columns, BodyKind.TCQ, sigma)
    second = _plan_for.cache_info()
    assert second.misses == first.misses, (
        f"the second pass recomputed {second.misses - first.misses} of {len(pass_)} plans "
        f"across {len(shapes)} shapes"
    )


def test_the_body_is_normalised_before_the_key():
    """``BodyKind.TCQ`` and the int it equals are one key, not two."""
    _plan_for(E4M3_GRID, Q256_UNIT, COLUMNS, BodyKind.WINDOW)
    _plan_for(E4M3_GRID, Q256_UNIT, COLUMNS, int(BodyKind.WINDOW))
    info = _plan_for.cache_info()
    assert (info.misses, info.hits) == (1, 1)
