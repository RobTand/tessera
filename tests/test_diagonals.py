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
