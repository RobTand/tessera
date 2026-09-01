"""End-to-end: every S5 segment together, across a mixed-rate schedule."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tessera.alphabet import build_forest  # noqa: E402
from tessera.decode import decode_codes, reconstruct_unit, replay_body  # noqa: E402
from tessera.diagonals import apply_diagonals, apply_rotation  # noqa: E402
from tessera.encode import _pack_scales, encode_unit  # noqa: E402
from tessera.errors import GrammarError  # noqa: E402
from tessera.grammar import bresenham_rate_schedule, root_from_q256  # noqa: E402
from tessera.manifest import RotationState  # noqa: E402
from tessera.trellis import ConvCode  # noqa: E402

CODE = ConvCode(memory=3)
FORESTS = {rate: build_forest(rate) for rate in (1, 2, 3)}


def _weights(rows=64, cols=256, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator) * 0.02


def _scale_for(unit, weights):
    rotated, _ = apply_rotation(weights, unit.rotation)
    base = apply_diagonals(rotated, unit.diagonals) if unit.diagonals else rotated
    _, _, effective = _pack_scales(base, 32, 16)
    return torch.repeat_interleave(effective, 16).reshape(weights.shape)


@pytest.mark.parametrize("q256", [256, 384, 512, 640, 768])
def test_mixed_rate_schedules_encode_and_replay(q256):
    """A fractional root mixes rates per column; columns are independent."""
    weights = _weights()
    rates = bresenham_rate_schedule(root_from_q256(q256), weights.shape[1])
    unit = encode_unit(weights, FORESTS, rates, CODE)
    assert set(unit.rates) == set(rates)
    # Replay each rate group from its own body bits alone.
    lanes = torch.tensor(rates)
    for rate in sorted(set(rates)):
        which = torch.nonzero(lanes == rate).squeeze(1)
        anchors = replay_body(
            unit.body_bits[:, which].contiguous(), FORESTS[rate], CODE
        )
        assert torch.equal(anchors, unit.anchors[:, which])


def test_missing_forest_for_a_used_rate_is_refused():
    weights = _weights()
    rates = bresenham_rate_schedule(root_from_q256(640), weights.shape[1])
    with pytest.raises(GrammarError, match="no forest was supplied"):
        encode_unit(weights, {3: FORESTS[3]}, rates, CODE)


@pytest.mark.parametrize(
    "rotation,diagonals",
    [
        (RotationState.NONE, False),
        (RotationState.NONE, True),
        (RotationState.R_IN_ONLY, False),
        (RotationState.R_IN_ONLY, True),
    ],
)
def test_full_inverse_path_round_trips(rotation, diagonals):
    """Body -> codes -> weights -> 2a -> rotation, undone in reverse order.

    An inverse applied out of order is wrong by a rank-1 factor or an
    orthogonal transform; both look plausible and neither survives this.
    """
    weights = _weights()
    rates = (3,) * weights.shape[1]
    # ``_scale_for`` rebuilds the amax plane, so this encode runs without the
    # scale refit (whose plane is not derivable from the weights alone).
    unit = encode_unit(
        weights, FORESTS[3], rates, CODE, rotation=rotation,
        with_diagonals=diagonals, scale_refit=0,
    )
    scale = _scale_for(unit, weights)
    reconstructed = reconstruct_unit(unit, FORESTS[3], CODE, scale)
    assert reconstructed.shape == weights.shape
    assert torch.isfinite(reconstructed).all()
    # The transforms are lossless, so reconstruction error is the body's alone.
    error = (weights - reconstructed).norm() / weights.norm()
    assert error < 0.30


def test_release_overrides_survive_the_inverse_path():
    weights = _weights()
    released = 512
    unit = encode_unit(
        weights, FORESTS[3], (3,) * weights.shape[1], CODE, released_positions=released
    )
    assert unit.released_positions == released
    codes = decode_codes(unit, FORESTS[3], CODE)
    assert torch.equal(codes.reshape(-1)[unit.release_index], unit.release_code)


def test_every_code_is_a_legal_e2m1_nibble():
    """Materialisation pads into legal nibbles; it never truncates (S6)."""
    weights = _weights()
    unit = encode_unit(weights, FORESTS, bresenham_rate_schedule(root_from_q256(512), weights.shape[1]), CODE)
    assert int(unit.codes.min()) >= 0 and int(unit.codes.max()) <= 15
