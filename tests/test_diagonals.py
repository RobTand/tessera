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
