"""Which grids have a TCQ forest body, and where that is refused.

A forest over a **registered** grid whose ALPHABET/DESCENDANT planes cannot
carry its codes is an object for nobody: no recipe writes a TCQ body on it,
and `AnchorForest._refuse_unserialisable` would refuse its planes anyway.  On
BF16 that object costs 65.7 GiB at R=11 before the refusal it was always going
to get (tessera#285, `docs/measurements/build-forest-memory-2026-09-05.md`).
So `build_forest` refuses first, by name.

The roster is not restated here.  The membership test is
`SERIALISABLE_GRIDS` and the width test is the forest planes' own element
width; this file's job is to prove that the two, taken together, pick out
exactly the grids the *recipe table* already gives a window body at every
rung -- the other home of the same fact, in `export`.
"""

from __future__ import annotations

import pytest

from tessera.alphabet import (
    BF16_GRID,
    E2M1_GRID,
    E4M3_GRID,
    SERIALISABLE_GRIDS,
    build_forest,
    lloyd_max_grid,
    tuple_grid,
)
from tessera.errors import GrammarError


def test_build_forest_refuses_bf16_by_name_before_it_allocates():
    """R=1 is the cheapest BF16 cell (108 MiB) and is refused like every other.

    The rate is not the reason -- the grid is -- so the refusal is stated once,
    at the top, and the message names the grid, its code count and the plane
    that cannot carry it.
    """
    with pytest.raises(GrammarError) as excinfo:
        build_forest(1, grid=BF16_GRID)
    message = str(excinfo.value)
    assert "BF16" in message
    assert "65536" in message
    assert "window" in message


def test_the_refused_set_is_what_the_recipe_table_calls_window_only():
    """Rule 3: the expected set comes from `export`, not from a list here.

    A registered grid has a TCQ forest body exactly when its recipe table
    holds a TCQ row at some rung.  `build_forest` decides the same thing from
    the registry and the plane width, without importing the exporter; this
    pins the two spellings together so neither can drift alone.

    That asymmetry is why this body skips rather than fails without torch:
    the refusal is torch-free and so is this module's import, so the `pure`
    job collects it -- and reaching the *other* home means reaching the
    exporter, which imports torch (tessera#309's convention).
    """
    pytest.importorskip("torch")
    from tessera.export import recipe_table
    from tessera.manifest import BodyKind

    window_only = set()
    for grid in SERIALISABLE_GRIDS.values():
        bodies = {BodyKind(row.recipe.body) for row in recipe_table(grid)}
        if BodyKind.TCQ not in bodies:
            window_only.add(grid.name)

    refused = set()
    for grid in SERIALISABLE_GRIDS.values():
        try:
            build_forest(1, grid=grid)
        except GrammarError:
            refused.add(grid.name)

    # E4M3's shipping recipe is the window body at every rung too, but its
    # forest is 256 codes and the planes carry it, so the TCQ body stays
    # reachable there by override and its forest is still built.
    assert refused == {"BF16"}
    assert refused <= window_only
    assert window_only - refused == {"E4M3"}


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param(E2M1_GRID, id="E2M1"),
        pytest.param(E4M3_GRID, id="E4M3"),
        pytest.param(tuple_grid(E2M1_GRID, 2), id="E2M1x2"),
        pytest.param(lloyd_max_grid(16), id="free-scalar-16"),
        pytest.param(tuple_grid(lloyd_max_grid(32), 2), id="free-tuple-1024"),
    ],
)
def test_every_other_grid_still_builds(grid):
    """Measuring on an unregistered grid stays open, however wide it is.

    `AnchorForest._refuse_unserialisable`'s promise -- "encoding, decoding and
    measuring on any grid stay open" -- is what keeps the free-tuple grid at
    1024 codes buildable: it is not a wire commitment, so nothing about it was
    decided by a recipe, and refusing it here would narrow the research
    surface rather than close tessera#285.
    """
    forest = build_forest(1, grid=grid)
    assert len(forest.blocks) == 4


def test_the_tcq_override_on_bf16_is_refused_at_plan_time():
    """The one production reach into these cells, closed.

    `export._resolve_recipe` honours `body=BodyKind.TCQ` on any grid, so a
    caller could ask the exporter for a coset trellis over BF16 and
    `_build_plan` would build one forest per scheduled rate.  It now refuses
    instead, before the first allocation.
    """
    pytest.importorskip("torch")
    from tessera.export import _plan_for
    from tessera.manifest import BodyKind

    # q256 == 256 is one payload bit per position, so every scheduled rate is
    # 1 -- the cheapest cell there is, and pre-fix this call built it.
    with pytest.raises(GrammarError, match="BF16"):
        _plan_for(BF16_GRID, 256, 64, body=BodyKind.TCQ)

    # The window body, which is what every BF16 rung actually resolves to,
    # is untouched: it returns the grid in the forests' place.
    rates, forests = _plan_for(BF16_GRID, 256, 64, body=BodyKind.WINDOW)
    assert forests is BF16_GRID
