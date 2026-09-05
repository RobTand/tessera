"""End-to-end: every S5 segment together, across a mixed-rate schedule."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tessera.alphabet import build_forest  # noqa: E402
from tessera.decode import (  # noqa: E402
    decode_codes,
    decode_codes_mixed,
    reconstruct_unit,
    replay_body,
)
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


@pytest.mark.parametrize("released", [0, 8])
def test_single_forest_decode_reads_the_written_completion_depth(released):
    """The two public TCQ decoders agree on a unit written below capacity.

    A rate-1 column may spend up to ``cap - rate = 2`` completion bits, so
    ``completion=1`` is a real intermediate rung: the stored word indexes the
    *written*-depth descendant table, not the full-capacity one.  The
    single-forest wrapper used to read it at full capacity anyway, which
    turned a valid unit into plausible wrong codes -- and an explicitly
    shallower read into an IndexError, because the stored word was never
    narrowed with the table.  Both reads must match the encoder's own codes
    and the mixed decoder, release overrides included.
    """
    weights = _weights(rows=8, cols=32)
    forest = FORESTS[1]
    depth = forest.cap - forest.rate
    unit = encode_unit(
        weights, forest, (1,) * weights.shape[1], CODE,
        completion=1, scale_refit=0, released_positions=released,
    )
    assert 0 < unit.completion_limit < depth, (
        "the trigger is a written depth strictly between zero and capacity"
    )
    codes = decode_codes(unit, forest, CODE)
    assert torch.equal(codes.long(), unit.codes.long()), (
        f"default read disagrees with the encoder on "
        f"{int((codes.long() != unit.codes.long()).sum())} of "
        f"{unit.codes.numel()} codes"
    )
    assert torch.equal(codes, decode_codes_mixed(unit, forest, CODE))
    if released:
        assert torch.equal(
            codes.reshape(-1)[unit.release_index], unit.release_code.to(codes.dtype)
        )
    # A truncating read narrows the word with the table; the wrappers agree.
    shallow = decode_codes(unit, forest, CODE, completion=0)
    assert torch.equal(shallow, decode_codes_mixed(unit, forest, CODE, completion=0))


def test_single_forest_decode_refuses_a_rate_the_forest_does_not_hold():
    """``decode_codes`` is the uniform-rate spelling; a unit whose schedule
    names another rate needs a forest per rate and is refused by name."""
    weights = _weights()
    rates = bresenham_rate_schedule(root_from_q256(640), weights.shape[1])
    unit = encode_unit(weights, FORESTS, rates, CODE)
    with pytest.raises(GrammarError, match="decode_codes_mixed"):
        decode_codes(unit, FORESTS[3], CODE)


def test_every_code_is_a_legal_e2m1_nibble():
    """Materialisation pads into legal nibbles; it never truncates (S6)."""
    weights = _weights()
    unit = encode_unit(weights, FORESTS, bresenham_rate_schedule(root_from_q256(512), weights.shape[1]), CODE)
    assert int(unit.codes.min()) >= 0 and int(unit.codes.max()) <= 15
