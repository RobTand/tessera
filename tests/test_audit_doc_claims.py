"""The claims the Tier C doc pass now makes (math audit 2026-09-02, issue #25).

Tier C was documentation: a claim this repo cannot cheaply re-establish gets
deleted, and a claim it keeps gets stated exactly.  A docstring nothing
executes drifts back the moment the code moves, so the two claims with
arithmetic in them are pinned here.

* ``ConvCode(memory=0)`` -- the one behavioural change.  It **fails on
  master**: the dataclass constructed, and torch answers ``state >> -1`` with
  zero instead of raising, so a zero-memory code encoded every position from a
  constant select bit rather than faulting.
* ``EncodedUnit.sse`` and the forest-plane byte rule -- regression guards.
  They pass on both sides by construction; they exist so the field comment and
  ``docs/schema/prismaquant.tessera.v1.md`` stop being prose nothing checks.
"""

import pytest
import torch

from tessera.alphabet import (
    E2M1_GRID,
    E4M3_GRID,
    SERIALISABLE_GRIDS,
    build_forest,
    tuple_grid,
)
from tessera.encode import encode_unit, grid_vector_table, viterbi_window, window_table
from tessera.errors import GrammarError
from tessera.trellis import ConvCode
from tessera.wire import scales_from_planes


# --- the behavioural one: a code with no shift register --------------------


def test_a_convolutional_code_needs_at_least_one_memory_element():
    """Explicit generators used to walk past the default-generator lookup, and
    ``state >> -1`` is silently zero in torch rather than an error."""
    with pytest.raises(GrammarError, match="no shift register"):
        ConvCode(memory=0, generators=(0o1, 0o1))
    with pytest.raises(GrammarError, match="no shift register"):
        ConvCode(memory=-1, generators=(0o1, 0o1))
    assert ConvCode(memory=1, generators=(0o1, 0o3)).states == 2
    assert ConvCode().memory == 6                      # the default is untouched


def test_torch_right_shift_by_minus_one_is_zero_not_an_error():
    """The premise of the guard above, pinned: this is why memory=0 was silent
    and not a crash."""
    assert (torch.tensor([1, 2, 7]) >> -1).tolist() == [0, 0, 0]


# --- the forest-plane byte rule the schema doc now states ------------------


@pytest.mark.parametrize("rate,expected", [(1, 20), (2, 24), (3, 32)])
def test_e2m1_forest_planes_cost_two_to_the_rate_plus_one_plus_the_code_space(
    rate, expected
):
    """``2^(R+1)`` anchor bytes plus one flattened forest of ``2^(cap+1)``."""
    forest = build_forest(rate, grid=E2M1_GRID)
    assert len(forest.alphabet_plane()) == 1 << (rate + 1)
    assert len(forest.descendant_plane()) == 1 << (E2M1_GRID.rate_cap + 1)
    assert len(forest.alphabet_plane()) + len(forest.descendant_plane()) == expected


def test_the_forest_planes_are_512_bytes_at_the_e2m1x2_cap():
    """The only 8-bit-grid rung ``wire_recipe`` still writes a TCQ body for."""
    grid = tuple_grid(E2M1_GRID, 2)
    forest = build_forest(grid.rate_cap, grid=grid)
    assert len(forest.alphabet_plane()) + len(forest.descendant_plane()) == 512


def test_the_1420_byte_figure_is_one_schedule_on_one_grid():
    """The schema doc used to quote ~1.4 KB as a general per-unit figure "at
    64x512".  It is E4M3 over rates {1,2,6,7} and it has no shape in it."""
    total = sum(
        len(build_forest(r, grid=E4M3_GRID).alphabet_plane())
        + len(build_forest(r, grid=E4M3_GRID).descendant_plane())
        for r in (1, 2, 6, 7)
    )
    assert total == 1420


def test_the_forest_planes_are_bounded_by_2300_bytes():
    """The ceiling the schema doc quotes: every legal rate of an 8-bit grid."""
    for grid in SERIALISABLE_GRIDS.values():
        if grid.size > 256:                 # window-body only; no forest planes
            continue
        total = sum(
            len(build_forest(r, grid=grid).alphabet_plane())
            + len(build_forest(r, grid=grid).descendant_plane())
            for r in range(1, grid.rate_cap + 1)
        )
        assert total <= 2300


# --- EncodedUnit.sse: three meanings, and always pre-release ---------------


ROWS, COLS, RATE = 32, 64, 2


def _fixture():
    torch.manual_seed(3)
    return torch.randn(ROWS, COLS), {RATE: build_forest(RATE, grid=E2M1_GRID)}


def _plain_and_weighted(unit, w):
    vectors = grid_vector_table(E2M1_GRID, w.device)
    units = vectors[unit.codes].permute(0, 2, 1).reshape(ROWS, COLS)
    per_half = scales_from_planes(
        unit.scale_base, unit.scale_refine, unit.group, unit.half
    )
    scale = torch.repeat_interleave(per_half, unit.half).reshape(ROWS, COLS)
    residual = (w / scale - units) ** 2
    weight = (scale / scale.amax(dim=0, keepdim=True)) ** 2
    return float(residual.sum()), float((weight * residual).sum())


def _encode(w, forests, **kw):
    return encode_unit(w, forests, (RATE,) * COLS, **kw)


def test_sse_is_the_unweighted_target_error_wherever_a_refit_runs():
    w, forests = _fixture()
    for weighting in ("none", "scale"):
        unit = _encode(w, forests, scale_refit=4, trellis_weighting=weighting)
        plain, _ = _plain_and_weighted(unit, w)
        assert unit.sse == pytest.approx(plain, rel=1e-6)


def test_sse_is_the_viterbi_total_without_a_refit_and_the_weighting_decides():
    w, forests = _fixture()
    none = _encode(w, forests, scale_refit=0, trellis_weighting="none")
    scale = _encode(w, forests, scale_refit=0, trellis_weighting="scale")
    plain_none, _ = _plain_and_weighted(none, w)
    plain_scale, weighted_scale = _plain_and_weighted(scale, w)
    assert none.sse == pytest.approx(plain_none, rel=1e-6)
    assert scale.sse == pytest.approx(weighted_scale, rel=1e-6)
    # The weighted total is not the unweighted one: two different quantities in
    # one field, which is the whole reason the field carries a comment now.
    assert scale.sse != pytest.approx(plain_scale, rel=1e-3)


def test_sse_is_pre_release():
    """``released_positions`` overrides codes after ``sse`` is computed."""
    w, forests = _fixture()
    quiet = _encode(w, forests, scale_refit=4, trellis_weighting="scale")
    loud = _encode(
        w, forests, scale_refit=4, trellis_weighting="scale", released_positions=8
    )
    assert loud.release_index.numel() == 8
    assert loud.sse == quiet.sse                      # unchanged by the release
    after, _ = _plain_and_weighted(loud, w)
    assert after != pytest.approx(loud.sse, rel=1e-9)  # and no longer the truth


# --- the chunk contract on viterbi_window ---------------------------------


def test_the_window_viterbi_states_are_chunk_invariant_and_its_sse_is_not_exact():
    """Columns are independent, so the states are bit-identical at every chunk;
    ``sse`` is a per-chunk fp32 sum and is only equal to a tolerance."""
    torch.manual_seed(11)
    L, R = 8, 2
    codes = window_table(E2M1_GRID, L, sigma=None, seed=0, half=16, device="cpu")
    vectors = grid_vector_table(E2M1_GRID, "cpu")[codes.long()]
    targets = torch.randn(32, 96)
    base_states, base_sse = viterbi_window(targets, vectors, L, R, chunk=512,
                                           impl="reference")
    for chunk in (7, 16, 32, 96):
        states, sse = viterbi_window(targets, vectors, L, R, chunk=chunk,
                                     impl="reference")
        assert torch.equal(states, base_states)
        assert sse == pytest.approx(base_sse, rel=1e-6)
