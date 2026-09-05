"""The §4/§5 findings of the 2026-09-02 math audit, each pinned by its own case.

Every test here failed before the fix it names, on the tree at ``6c82ed4``.
The findings and their verifications are
``docs/handovers/math-audit-triage-2026-09-02.md`` and
``/home/rob/tmp/audit/report{4,5}.md``; the reproducing inputs below are the
verifier's own where it had one, because a fix that passes a case the verifier
did not run has not answered the verifier.

CPU only: nothing here touches a device.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid   # noqa: E402
from tessera.decode import replay_body, unit_scale_field        # noqa: E402
from tessera.encode import (                                    # noqa: E402
    _refit_scales_lut, grid_vector_table, viterbi_columns)
from tessera.errors import GrammarError                         # noqa: E402
from tessera.export import encode_linear, encode_linear_planes  # noqa: E402
from tessera.fused import shared_lut_global                     # noqa: E402
from tessera.manifest import BodyKind, ScalePlaneKind           # noqa: E402
from tessera.scale_channel import (                             # noqa: E402
    land_at_least, refit_channel_scale)
from tessera.trellis import ConvCode                            # noqa: E402
from tessera.unit_artifact import parse_unit_artifact           # noqa: E402


# --- A: the CHANNEL refit holds a row whose codes point the wrong way --------

def test_channel_refit_holds_a_row_with_a_non_positive_b():
    """§5 P0-3.  ``B <= 0`` has its exact minimiser at zero, and the parabola
    prefers it, so without the hold the row's scale collapses to fp16's
    smallest word and every weight in it decodes to about nothing.

    The verifier's own instance: one row, codes exactly anti-correlated with
    the weights.  Before the fix ``stored_out`` came back ``6.104e-05``.
    """
    work = torch.tensor([[1.0, -1.0]])
    units = torch.tensor([[-1.0, 1.0]])
    stored = torch.tensor([1.0], dtype=torch.float16)

    A = float((units * units).sum())
    B = float((work * units).sum())
    assert A > 0 and B <= 0, "the case must be the one the finding names"

    out_stored, out_eff = refit_channel_scale(work, units, stored, 1.0)
    assert float(out_stored[0]) == 1.0
    assert float(out_eff[0]) == 1.0


def test_channel_refit_holds_a_non_positive_b_under_a_full_hessian():
    """The same hold under the 2-D metric, where ``B = u H w^T`` turns negative
    more readily than the diagonal case: H is the metric that ships on this
    plane (``export.DEFAULT_REFIT_OBJECTIVE`` maps CHANNEL to ``hessian``)."""
    work = torch.tensor([[2.0, 1.0]])
    units = torch.tensor([[1.0, -1.0]])
    H = torch.tensor([[1.0, 0.5], [0.5, 4.0]])          # PSD: det 3.75, a > 0
    stored = torch.tensor([1.0], dtype=torch.float16)

    # The plain metric would refit this row happily; H is what turns B over.
    assert float((work * units).sum()) > 0
    UH = units @ H
    assert float((UH * units).sum()) > 0
    assert float((UH * work).sum()) <= 0

    out_stored, _ = refit_channel_scale(work, units, stored, 1.0, metric=H)
    assert float(out_stored[0]) == 1.0


def test_channel_refit_still_takes_a_row_that_improves():
    """The hold is a hold, not a freeze: a row with ``B > 0`` still refits."""
    work = torch.tensor([[2.0, 2.0]])
    units = torch.tensor([[1.0, 1.0]])
    stored = torch.tensor([1.0], dtype=torch.float16)
    out_stored, out_eff = refit_channel_scale(work, units, stored, 1.0)
    assert float(out_eff[0]) == pytest.approx(2.0, rel=1e-3)


# --- B / P2-9: land_at_least saturates nothing and overshoots nothing --------

def test_land_at_least_refuses_a_floor_above_fp16():
    """§5 P0-4.  ``65504 * (1 + 2^-10)`` rounds to infinity in fp16; the old
    code stored it, and ``0 * inf`` then made the refit's accept test compare
    NaN.  A floor no word can carry is a refusal, not an infinity."""
    with pytest.raises(GrammarError, match="above fp16's range"):
        land_at_least(torch.tensor([65505.0]), 1.0)


def test_land_at_least_lands_on_the_minimal_word():
    """§5 P2-9.  The old bump multiplied by ``1 + 2^-10`` and re-rounded, which
    overshot the smallest word clearing the floor by two ulps on 263 of 1500
    sampled floors.  A bit increment is ``nextafter`` and cannot."""
    gs = 0.37
    floors = torch.linspace(0.05, 20.0, 1500)
    stored, effective = land_at_least(floors, gs)
    assert bool((effective >= floors).all()), "a floor is a floor"

    # The minimal word: walk down from what we stored while the floor still
    # clears.  A minimal landing cannot take a single step down.
    lower = (stored.contiguous().view(torch.int16) - 1).view(torch.float16)
    assert bool(((lower.float() * gs) < floors).all()), "not the minimal word"


def test_land_at_least_leaves_a_word_that_already_clears():
    exact = torch.tensor([0.5, 1.0, 2.0])
    stored, effective = land_at_least(exact, 1.0)
    assert torch.equal(effective, exact)


# --- C: the fused group refuses a scale byte its kernel misreads ------------

def _bytes(*values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.uint8)


def test_shared_lut_global_refuses_a_moved_subnormal():
    """§5 P0-5.  The verifier's reproducing entry, not the audit's: ``0x18`` at
    global 32 moves to ``0x01`` at global 1024 and round-trips exactly, so the
    finite-and-equal test passes -- but ``0x01`` has a zero exponent field and
    the kernel reads a scale byte as ``2^(e-7)(1+m/8)``, which is not its
    value.  (The audit's own ``0x08 -> 2^-11`` instance does NOT reproduce: it
    flushes to ``0x00`` and was already refused.)"""
    with pytest.raises(GrammarError, match="E4M3 normals"):
        shared_lut_global(
            [_bytes(0x18, 0x7E), _bytes(0x7E)], [32.0, 1024.0], ["a", "b"])


@pytest.mark.parametrize("byte", [0x20, 0x22, 0x24, 0x26, 0x28, 0x2A, 0x2C, 0x2E])
def test_shared_lut_global_refuses_every_neighbouring_subnormal(byte):
    """The seven neighbours the verifier listed behave identically."""
    with pytest.raises(GrammarError, match="E4M3 normals"):
        shared_lut_global(
            [_bytes(byte, 0x7E), _bytes(0x7E)], [32.0, 1024.0], ["a", "b"])


def test_shared_lut_global_refuses_a_subnormal_it_was_handed():
    """The range holds on the tables that come back untouched too: one global
    for the whole group is the path that returns the caller's own bytes."""
    with pytest.raises(GrammarError, match="normal range"):
        shared_lut_global([_bytes(0x01, 0x7E)], [32.0], ["a"])


def test_shared_lut_global_still_carries_a_normal_group():
    """A group that stays in normals is unaffected -- and none of the fourteen
    LUT-plane artifacts on this box has a byte outside the range."""
    shared, moved = shared_lut_global(
        [_bytes(0x40, 0x44), _bytes(0x38)], [32.0, 64.0], ["a", "b"])
    assert shared == 32.0
    a = moved[0].view(torch.float8_e4m3fn).float() * shared
    b = moved[1].view(torch.float8_e4m3fn).float() * shared
    assert torch.equal(a, _bytes(0x40, 0x44).view(torch.float8_e4m3fn).float() * 32.0)
    assert torch.equal(b, _bytes(0x38).view(torch.float8_e4m3fn).float() * 64.0)


# --- D: the fused serving lane refuses the CHANNEL plane by name ------------

def test_prepare_tessera_module_refuses_a_channel_plane_unit():
    """§5 P2-10d.  A CHANNEL+TCQ unit is a supported export (``export.py``'s
    resolver gives one to a caller that names ``body=TCQ`` over the E4M3
    recipe) and carries no ``scale_lut``, so it used to reach
    ``shared_lut_global`` as ``None`` and die on ``AttributeError``."""
    ops = pytest.importorskip("tessera.serving.ops")
    blob = encode_linear(
        torch.randn(32, 256), grid=E4M3_GRID, q256=1024,
        body=BodyKind.TCQ, verify=False,
    )
    parsed = parse_unit_artifact(blob.blob)
    assert ScalePlaneKind(parsed.unit.scale_plane) is ScalePlaneKind.CHANNEL
    assert parsed.unit.scale_lut is None
    with pytest.raises(ValueError, match="CHANNEL"):
        ops.prepare_tessera_module([("weight", parsed)], device="cpu")


# --- E: the completion argmin scores under the trellis's own metric ---------

def test_completion_is_optimal_in_weight_space_at_arity_two():
    """§4 P0.  ``sub_w`` was computed and never used, so the completion bits
    were chosen unweighted while the anchor they refine was chosen weighted.
    At arity 1 a positive scalar cancels in an argmin; at arity 2 the tuple's
    two rows sit in different halves at different scales and the pick moves.

    The property, stated where it can be checked from the unit alone: no
    position's completion bits can be improved by another reachable
    descendant, under the true weight-space error.  Before the fix, 4 to 8 of
    256 positions failed it on every seed tried.
    """
    grid = tuple_grid(E2M1_GRID, 2)
    rows, cols = 8, 64
    torch.manual_seed(3)
    weight = torch.randn(rows, cols)
    weight[1::2] *= 40.0        # the two rows of a tuple live decades apart

    _, unit, forests = encode_linear_planes(
        weight, grid=grid, q256=512, name="u", body=BodyKind.TCQ,
        completion=None, trellis_weighting="scale", scale_refit=0, verify=False,
    )
    vectors = grid_vector_table(grid)
    scale = unit_scale_field(unit, rows, cols)
    forest = forests[unit.rates[0]]
    reachable = torch.tensor(forest.blocks, dtype=torch.long)   # level == depth
    steps = rows // grid.arity

    candidates = vectors[reachable[unit.anchors]]               # [steps, cols, D, k]
    w = weight.reshape(steps, grid.arity, cols).permute(0, 2, 1).unsqueeze(2)
    s = scale.reshape(steps, grid.arity, cols).permute(0, 2, 1).unsqueeze(2)
    best = ((w - s * candidates) ** 2).sum(dim=3).argmin(dim=2)
    assert torch.equal(best, unit.completion_bits)


# --- F: memory 0, and a refit metric of the wrong shape ---------------------

def test_the_encoder_refuses_a_code_with_no_memory():
    """§4 P2.  ``state >> (memory - 1)`` is a shift by -1, which torch
    evaluates to zero and raises nothing: the vectorised trellis then reported
    anchors it had not emitted."""
    from tessera.alphabet import build_forest

    forest = build_forest(2, grid=E2M1_GRID)
    with pytest.raises(GrammarError, match="at least one memory element"):
        viterbi_columns(
            torch.randn(4, 8), forest, ConvCode(memory=0, generators=(0o1, 0o1)), 0)


def test_the_decoder_refuses_a_code_with_no_memory():
    """The same code raised a bare ``ValueError: negative shift count`` from
    ``1 << (memory - 1)`` one layer down.  Both ends refuse it by name."""
    from tessera.alphabet import build_forest

    forest = build_forest(2, grid=E2M1_GRID)
    with pytest.raises(GrammarError, match="at least one memory element"):
        replay_body(
            torch.zeros(4, 8, dtype=torch.uint8), forest,
            ConvCode(memory=0, generators=(0o1, 0o1)))


@pytest.mark.parametrize("bad", [
    torch.ones(7), torch.ones(1), torch.ones(8, 7), torch.ones(7, 8),
])
def test_the_channel_refit_refuses_a_metric_of_the_wrong_shape(bad):
    """§5 P2-8.  A wrong-width metric reached the arithmetic and came back as a
    raw ``RuntimeError`` about broadcasting; a length-1 metric broadcast in
    silence, weighting every column equally under a name that says otherwise."""
    work = torch.randn(4, 8)
    units = torch.randn(4, 8)
    stored = torch.ones(4, dtype=torch.float16)
    with pytest.raises(GrammarError, match="refit metric"):
        refit_channel_scale(work, units, stored, 1.0, metric=bad)


@pytest.mark.parametrize("bad", [torch.ones(15), torch.ones(1), torch.ones(16, 15)])
def test_the_lut_refit_refuses_a_metric_of_the_wrong_shape(bad):
    """The LUT plane's metric path is held to the same check."""
    rows, cols, half = 2, 32, 16
    table = torch.arange(16, dtype=torch.uint8) + 0x40
    with pytest.raises(GrammarError, match="refit metric"):
        _refit_scales_lut(
            torch.randn(rows, cols), torch.randn(rows, cols), half, table,
            torch.zeros(rows * cols // half, dtype=torch.uint8),
            torch.ones(rows * cols // half), 1.0, metric=bad,
        )


def test_check_refit_metric_accepts_the_two_shapes_it_documents():
    from tessera.scale_channel import check_refit_metric

    check_refit_metric(torch.ones(8), 8)
    check_refit_metric(torch.ones(8, 8), 8)


# --- G: sse means one thing, and it is the unit's own -----------------------

def _reconstruction_sse(weight, unit, rows, cols, grid):
    vectors = grid_vector_table(grid)
    scale = unit_scale_field(unit, rows, cols)
    units = vectors[unit.codes].permute(0, 2, 1).reshape(rows, cols)
    return float(((weight - units * scale) ** 2).sum())


@pytest.mark.parametrize("weighting", ["none", "scale"])
@pytest.mark.parametrize("refit", [0, 4])
def test_sse_is_the_units_own_weight_space_error(weighting, refit):
    """§4 P1.  ``sse`` used to be the WEIGHTED trellis total under
    ``trellis_weighting='scale'`` (the shipping setting), an unweighted
    *normalised* sum after a refit, and in every case the value from before
    release mutated the codes.  One definition now: the unweighted squared
    error of the unit's own reconstruction, in the weight's units."""
    rows, cols = 16, 128
    torch.manual_seed(7)
    weight = torch.randn(rows, cols)
    _, unit, _ = encode_linear_planes(
        weight, grid=E2M1_GRID, q256=256, name="u",
        trellis_weighting=weighting, scale_refit=refit, verify=False,
    )
    expected = _reconstruction_sse(weight, unit, rows, cols, E2M1_GRID)
    assert unit.sse == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("rotation_on", [False, True])
@pytest.mark.parametrize("supplied", [False, True])
def test_sse_is_in_the_source_weights_units_under_diagonals(rotation_on, supplied):
    """tessera#230.  With segment-2a diagonals the encoder codes ``work =
    Dv^-1 W R Du^-1``, and ``sse`` measured ``||work - reconstruction||^2``
    in those *balanced* coordinates: for a balanced error E the source-weight
    error is ``||Dv E Du||^2`` (the remaining rotation is orthogonal), so the
    reported number depended on the balancing gauge -- the issue's witness at
    sv=2, su=1 read exactly 4x low.  The reference is ``reconstruct_unit``,
    the full stored inverse path, never a helper that repeats the
    implementation's omissions."""
    from tessera.alphabet import build_forest
    from tessera.decode import reconstruct_unit
    from tessera.diagonals import Diagonals
    from tessera.encode import encode_unit
    from tessera.manifest import RotationState

    weight = torch.randn(16, 32, generator=torch.Generator().manual_seed(1)) * .02
    weight[5, :] *= 6.0             # a fitted sv is then genuinely nonuniform
    forest = build_forest(3, grid=E2M1_GRID)
    code = ConvCode(memory=6)
    kwargs: dict = dict(
        rotation=RotationState.R_IN_ONLY if rotation_on else RotationState.NONE,
        scale_refit=1,
    )
    if supplied:
        kwargs["diagonals"] = Diagonals(
            torch.full((16,), 2., dtype=torch.float16),
            torch.ones(32, dtype=torch.float16))
    else:
        kwargs["with_diagonals"] = True
    unit = encode_unit(weight, forest, (3,) * 32, code, **kwargs)
    source_sse = float(((weight - reconstruct_unit(unit, forest, code)) ** 2).sum())
    assert unit.sse == pytest.approx(source_sse, rel=1e-4)


def test_sse_does_not_depend_on_the_trellis_weighting_convention():
    """Two encodes that land on the same codes report the same ``sse``: the
    number is a property of the encoding, not of the objective that found it.
    Before the fix the same weights reported 444.11 and 138.08."""
    rows, cols = 16, 128
    torch.manual_seed(11)
    weight = torch.randn(rows, cols)
    units = []
    for weighting in ("none", "scale"):
        _, unit, _ = encode_linear_planes(
            weight, grid=E2M1_GRID, q256=256, name="u",
            trellis_weighting=weighting, scale_refit=0, verify=False,
        )
        units.append(unit)
    if torch.equal(units[0].codes, units[1].codes):
        assert units[0].sse == pytest.approx(units[1].sse, rel=1e-6)
    for unit in units:
        assert unit.sse == pytest.approx(
            _reconstruction_sse(weight, unit, rows, cols, E2M1_GRID), rel=1e-5)
    assert math.isfinite(units[0].sse)
