"""Segment 2a diagonals and the S5 rotation states."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tessera.diagonals import (  # noqa: E402
    apply_diagonals,
    apply_rotation,
    diagonal_bits,
    fit_diagonals,
    hadamard_block,
    undo_diagonals,
    undo_rotation,
)
from tessera.errors import GrammarError  # noqa: E402
from tessera.manifest import RotationState  # noqa: E402


def _weights(rows=64, cols=256, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator) * 0.02


def test_diagonals_invert_exactly():
    weights = _weights()
    fitted = fit_diagonals(weights)
    restored = undo_diagonals(apply_diagonals(weights, fitted), fitted)
    assert torch.allclose(restored, weights, atol=1e-5, rtol=1e-3)


def test_diagonals_remove_a_planted_rank_one_field():
    """The rank-1 magnitude field is exactly what a per-block scale cannot see."""
    weights = _weights()
    weights[:, 7] *= 9.0
    weights[13, :] *= 5.0
    before_rows = weights.pow(2).mean(1).sqrt()
    balanced = apply_diagonals(weights, fit_diagonals(weights))
    after_rows = balanced.pow(2).mean(1).sqrt()
    spread = lambda t: (t.std() / t.mean()).item()  # noqa: E731
    assert spread(after_rows) < spread(before_rows) / 100
    after_cols = balanced.pow(2).mean(0).sqrt()
    assert spread(after_cols) < 0.01


def test_diagonal_cost_is_the_declared_plane_width():
    """16 bits per row plus 16 per column -- the DIAG_SU/SV element width."""
    assert diagonal_bits(2048, 5120) == 16 * (2048 + 5120)
    assert diagonal_bits(2048, 5120) / (2048 * 5120) < 0.011


def test_fit_is_deterministic():
    weights = _weights()
    first, second = fit_diagonals(weights), fit_diagonals(weights)
    assert torch.equal(first.su, second.su) and torch.equal(first.sv, second.sv)


def test_hadamard_is_orthonormal():
    matrix = hadamard_block(64)
    assert torch.allclose(matrix @ matrix.T, torch.eye(64), atol=1e-5)
    with pytest.raises(GrammarError, match="not a power of two"):
        hadamard_block(48)


def test_rotation_inverts_and_none_is_identity():
    weights = _weights()
    rotated, block = apply_rotation(weights, RotationState.R_IN_ONLY)
    assert block > 1
    assert torch.allclose(
        undo_rotation(rotated, RotationState.R_IN_ONLY, block), weights, atol=1e-5
    )
    same, block_none = apply_rotation(weights, RotationState.NONE)
    assert block_none == 1 and torch.allclose(same, weights)


def test_rotation_suppresses_outliers():
    """Incoherence processing: the point of rotating is to thin the tail."""
    weights = _weights()
    weights[:, 3] *= 30.0
    kurt = lambda t: ((t - t.mean()) ** 4).mean() / t.var() ** 2  # noqa: E731
    rotated, _ = apply_rotation(weights, RotationState.R_IN_ONLY)
    assert kurt(rotated) < kurt(weights)


def test_fit_lands_extreme_magnitudes_inside_fp16_range():
    """tessera#229: a finite weight must never fit factors that store as zero
    or infinity.  ``undo_diagonals`` multiplies the stored FP16 words back, so
    a factor outside FP16's range decodes every weight it touches to zero or
    NaN.  The rank-1 gauge is free -- ``(sv * c, su / c)`` balances the same
    matrix -- so the fit must spend it landing both factors in range."""
    finfo = torch.finfo(torch.float16)
    for magnitude in (1e-8, 1e5):
        weights = torch.full((16, 32), magnitude)
        fitted = fit_diagonals(weights)
        for factor in (fitted.sv, fitted.su):
            assert torch.isfinite(factor).all(), magnitude
            assert (factor > 0).all(), magnitude
            assert float(factor.abs().max()) <= finfo.max
        restored = undo_diagonals(apply_diagonals(weights, fitted), fitted)
        assert torch.allclose(restored, weights, rtol=1e-2)


def test_extreme_magnitudes_encode_and_reconstruct_finite(tmp_path):
    """The end-to-end witness from tessera#229: encode/write/read a constant
    finite source at 1e-8 and 1e5 with fitted diagonals, and require the
    reconstruction to be finite and near the source -- not silently zero or
    NaN with a healthy-looking artifact."""
    from tessera.alphabet import E2M1_GRID, build_forest
    from tessera.encode import encode_unit
    from tessera.decode import reconstruct_unit
    from tessera.trellis import ConvCode

    forest = build_forest(3, grid=E2M1_GRID)
    code = ConvCode(memory=6)
    for magnitude in (1e-8, 1e5):
        weight = torch.full((16, 32), magnitude)
        unit = encode_unit(weight, forest, (3,) * 32, code,
                           with_diagonals=True, scale_refit=1)
        output = reconstruct_unit(unit, forest, code)
        assert torch.isfinite(output).all(), magnitude
        assert torch.allclose(output, weight, rtol=0.25), magnitude


def test_fit_gives_zero_rows_and_columns_an_invertible_factor():
    """A zero row's balanced values are zero under ANY factor, so the fit's
    deliberate policy is the identity factor 1.0 -- invertible, exact, and it
    does not drag the representable-range gauge toward zero."""
    weights = _weights(rows=16, cols=32)
    weights[3, :] = 0.0
    weights[:, 5] = 0.0
    fitted = fit_diagonals(weights)
    assert float(fitted.sv[3]) == 1.0
    assert float(fitted.su[5]) == 1.0
    assert torch.isfinite(fitted.sv).all() and (fitted.sv > 0).all()
    assert torch.isfinite(fitted.su).all() and (fitted.su > 0).all()
    restored = undo_diagonals(apply_diagonals(weights, fitted), fitted)
    assert torch.allclose(restored, weights, atol=1e-6, rtol=1e-2)
    assert torch.equal(restored[3, :], torch.zeros(32))


def test_a_fit_no_gauge_can_represent_is_refused_by_name():
    """Rows at 1e-8 next to rows at 1e8 need an sv spread of 1e16; FP16 holds
    a factor of ~1.07e9 between its smallest normal and its largest value, and
    one global gauge cannot fix a spread.  Refuse at the fit, by field name,
    rather than write factors that decode to zero or NaN."""
    weights = torch.ones(16, 32)
    weights[:8] *= 1e-8
    weights[8:] *= 1e8
    with pytest.raises(GrammarError, match="DIAG_SV"):
        fit_diagonals(weights)


def test_supplied_non_invertible_factors_are_refused():
    """tessera#229: ``apply_diagonals`` used to clamp a zero factor to 1e-12
    for the forward divide while ``undo_diagonals`` multiplied the stored zero
    back -- the transform pair was silently not a pair.  Both directions now
    refuse a zero, negative or non-finite factor by name."""
    from tessera.diagonals import Diagonals

    good = torch.ones(16, dtype=torch.float16)
    weights = _weights(rows=16, cols=16)
    for sv, su in (
        (torch.zeros(16, dtype=torch.float16), good),
        (good, torch.full((16,), float("inf"), dtype=torch.float16)),
        (torch.full((16,), float("nan"), dtype=torch.float16), good),
        (good, -torch.ones(16, dtype=torch.float16)),
    ):
        bad = Diagonals(sv=sv, su=su)
        with pytest.raises(GrammarError, match="DIAG_S"):
            apply_diagonals(weights, bad)
        with pytest.raises(GrammarError, match="DIAG_S"):
            undo_diagonals(weights, bad)


def test_the_writer_refuses_non_invertible_diagonals():
    """Refuse where the bytes are decided: a unit carrying factors the decoder
    cannot invert must not serialise, whatever produced the unit."""
    import dataclasses

    from tessera.alphabet import E2M1_GRID, build_forest
    from tessera.diagonals import Diagonals
    from tessera.encode import encode_unit
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import build_unit_artifact

    forest = build_forest(3, grid=E2M1_GRID)
    code = ConvCode(memory=6)
    weight = _weights(rows=16, cols=32)
    unit = encode_unit(weight, forest, (3,) * 32, code,
                       with_diagonals=True, scale_refit=1)
    bad = dataclasses.replace(
        unit,
        diagonals=Diagonals(sv=torch.zeros(16, dtype=torch.float16),
                            su=torch.ones(32, dtype=torch.float16)),
    )
    with pytest.raises(GrammarError, match="DIAG_SV"):
        build_unit_artifact(bad, "u", {3: forest}, 768, code, fixture_id=None)


#: The three ways a word wider than FP16 changes meaning on the way to the
#: DIAG_SU/SV bytes: it overflows to infinity, underflows to zero, or merely
#: rounds -- the last one is the quiet case, where the artifact reads back
#: fine and serves something other than what was priced (tessera#286).
_WIDER_THAN_FP16 = (torch.float32, torch.float64)
_FP16_WITNESSES = (("overflows", 1e5), ("underflows", 1e-8), ("rounds", 1.0004))


@pytest.fixture(scope="module")
def _fitted_unit():
    """One healthy fitted-diagonals unit, for the writer refusals to replace
    the pair on."""
    from tessera.alphabet import E2M1_GRID, build_forest
    from tessera.encode import encode_unit
    from tessera.trellis import ConvCode

    forest = build_forest(3, grid=E2M1_GRID)
    code = ConvCode(memory=6)
    weight = _weights(rows=16, cols=32, seed=7)
    unit = encode_unit(weight, forest, (3,) * 32, code,
                       with_diagonals=True, scale_refit=1)
    return weight, forest, code, unit


@pytest.mark.parametrize("field", ["DIAG_SV", "DIAG_SU"])
@pytest.mark.parametrize("dtype", _WIDER_THAN_FP16, ids=str)
@pytest.mark.parametrize("case,value", _FP16_WITNESSES, ids=[c for c, _ in _FP16_WITNESSES])
def test_supplied_factors_wider_than_fp16_are_refused_by_field_name(
    _fitted_unit, field, dtype, case, value
):
    """tessera#286: the guard validated ``factor.float()``, so a positive
    finite FP32/FP64 pair passed encode and write while ``pack_fp16`` cast it
    on the way to the bytes -- sv=1e5 stored infinity and sv=1e-8 stored
    zero (an artifact the writer's own reader refuses), and sv=1.0004 stored
    1.0 (readable, and serving weights other than the ones priced).  The
    pair is FP16 words from the moment it exists: every consumer -- both
    transform directions, the metric transport, the encoder and the writer
    -- refuses a wider dtype by field name, before any value is used."""
    import dataclasses

    from tessera.diagonals import Diagonals, transport_metric
    from tessera.encode import encode_unit
    from tessera.unit_artifact import build_unit_artifact

    weight, forest, code, unit = _fitted_unit
    rows, cols = weight.shape
    sv = torch.ones(rows, dtype=torch.float16)
    su = torch.ones(cols, dtype=torch.float16)
    if field == "DIAG_SV":
        sv = torch.full((rows,), value, dtype=dtype)
    else:
        su = torch.full((cols,), value, dtype=dtype)
    supplied = Diagonals(sv=sv, su=su)
    with pytest.raises(GrammarError, match=field):
        apply_diagonals(weight, supplied)
    with pytest.raises(GrammarError, match=field):
        undo_diagonals(weight, supplied)
    with pytest.raises(GrammarError, match=field):
        transport_metric(torch.ones(cols), RotationState.NONE, 1, supplied)
    with pytest.raises(GrammarError, match=field):
        encode_unit(weight, forest, (3,) * cols, code,
                    diagonals=supplied, scale_refit=1)
    # Where the bytes are decided: a unit carrying the pair, however it got
    # there, must not serialise -- before this the writer returned 938 bytes.
    bad = dataclasses.replace(unit, diagonals=supplied)
    with pytest.raises(GrammarError, match=field):
        build_unit_artifact(bad, "u", {3: forest}, 768, code, fixture_id=None)


@pytest.mark.parametrize("dtype", [torch.float16, *_WIDER_THAN_FP16], ids=str)
def test_an_accepted_supplied_pair_reconstructs_the_same_before_and_after_the_wire(dtype):
    """The rule, not the roster: whatever the encoder accepts as a supplied
    pair, the wire must hand back unchanged -- the reconstruction of the
    parsed unit equals the reconstruction of the encoded one, exactly, and
    the parsed factors are the encoded ones word for word.  1.0004 and 0.7503
    are where FP16 and a wider dtype disagree by one rounding: before the
    fix an FP32 pair was accepted and the artifact served 2.8e-05 off what
    was priced.  A refusal is the other legal answer, and only for a dtype
    wider than the wire's: FP16 words are the canonical form and are never
    refused for their dtype."""
    from tessera.alphabet import E2M1_GRID, build_forest
    from tessera.decode import reconstruct_unit
    from tessera.diagonals import Diagonals
    from tessera.encode import encode_unit
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    forest = build_forest(3, grid=E2M1_GRID)
    code = ConvCode(memory=6)
    weight = _weights(rows=16, cols=32, seed=7)
    supplied = Diagonals(sv=torch.full((16,), 1.0004, dtype=dtype),
                         su=torch.full((32,), 0.7503, dtype=dtype))
    try:
        unit = encode_unit(weight, forest, (3,) * 32, code,
                           diagonals=supplied, scale_refit=1)
    except GrammarError as refused:
        assert dtype is not torch.float16, refused
        assert "DIAG_SV" in str(refused)
        return
    before = reconstruct_unit(unit, forest, code)
    _, _, blob = build_unit_artifact(unit, "u", {3: forest}, 768, code, fixture_id=None)
    parsed = parse_unit_artifact(blob).unit
    assert torch.equal(parsed.diagonals.sv, unit.diagonals.sv)
    assert torch.equal(parsed.diagonals.su, unit.diagonals.su)
    after = reconstruct_unit(parsed, forest, code)
    assert torch.equal(before, after)


def test_the_fit_returns_the_canonical_words_the_guard_accepts_unchanged():
    """The fitted path and the supplied path meet at one rule: ``fit_diagonals``
    lands its factors as FP16 words, and ``require_invertible_diagonals``
    hands those back as they are -- the object IS the canonical form, so no
    consumer can drift by reading the caller's copy instead of a return
    value.  (Positive control for the refusals above.)"""
    from tessera.diagonals import require_invertible_diagonals

    fitted = fit_diagonals(_weights(rows=16, cols=32))
    assert fitted.sv.dtype is torch.float16 and fitted.su.dtype is torch.float16
    assert require_invertible_diagonals(fitted) is fitted


def test_transport_metric_carries_the_quadratic_into_the_encoded_basis():
    """tessera#231: the encoder quantises ``Wwork = Dv^-1 W R Du^-1``, so the
    activations its rows meet are ``xwork = Du R^T x`` and the metric in the
    encoded basis is ``H' = Du R^T H R Du``.  The invariant that defines the
    transport: for any working-coordinate error row ``e``, the source-output
    quadratic of the row it came from is ``sv_r^2 * (e H' e^T)``."""
    from tessera.diagonals import Diagonals, transport_metric

    g = torch.Generator().manual_seed(5)
    cols, rows = 64, 8
    x = torch.randn(4 * cols, cols, generator=g)
    H = (x.T @ x) / x.shape[0]
    sv = (torch.rand(rows, generator=g) * 3 + 0.5).to(torch.float16)
    su = (torch.rand(cols, generator=g) * 3 + 0.5).to(torch.float16)
    fitted = Diagonals(sv=sv, su=su)
    E_work = torch.randn(rows, cols, generator=g)
    R = hadamard_block(64)

    got = transport_metric(H, RotationState.R_IN_ONLY, 64, fitted)
    # The source-coordinate error of a working-coordinate error E is
    # Dv E Du R^T; its H-quadratic must equal the sv^2-weighted H'-quadratic.
    E_src = (sv.float()[:, None] * E_work * su.float()[None, :]) @ R.T
    want = ((E_src @ H) * E_src).sum()
    have = ((E_work @ got) * E_work * sv.float().pow(2)[:, None]).sum()
    assert torch.isclose(want, have, rtol=1e-4)
    # Distinguishable from the source-basis metric wherever the transform is
    # nontrivial -- the mispricing tessera#231 is about.
    assert not torch.allclose(got, H)


def test_transport_metric_keeps_a_diagonal_metric_diagonal_without_rotation():
    """Under NONE rotation a diagonal metric transports as ``su^2 h`` and must
    stay 1-D: the separable refit paths key on the metric's rank."""
    from tessera.diagonals import Diagonals, transport_metric

    h = torch.arange(1.0, 17.0)
    su = torch.full((16,), 2.0, dtype=torch.float16)
    fitted = Diagonals(sv=torch.ones(4, dtype=torch.float16), su=su)
    got = transport_metric(h, RotationState.NONE, 1, fitted)
    assert got.ndim == 1
    assert torch.allclose(got, h * 4.0)
    # And the identity transport is the metric itself.
    same = transport_metric(h, RotationState.NONE, 1, None)
    assert torch.allclose(same, h)


def test_transport_metric_densifies_a_diagonal_metric_under_rotation():
    """A diagonal source H is dense in the rotated basis; forwarding the
    diagonal power as if nothing moved is the mispricing tessera#231 shows."""
    from tessera.diagonals import transport_metric

    h = torch.arange(1.0, 33.0)
    got = transport_metric(h, RotationState.R_IN_ONLY, 32, None)
    R = hadamard_block(32)
    want = R.T @ torch.diag(h) @ R
    assert got.ndim == 2
    assert torch.allclose(got, want, atol=1e-5)


def test_two_sided_rotation_is_refused_as_a_serving_branch():
    """S7: two-sided is a weight-space measurement state, not a branch.

    Offering it would manufacture an artifact whose output basis no runtime can
    honour without an R_out^T inverse -- a model-level contract, not a per-unit
    one. Principle 9 makes that a refusal, not a warning.
    """

    class _Fake:
        pass

    with pytest.raises(GrammarError, match="measurement state only"):
        apply_rotation(_weights(), _Fake())
