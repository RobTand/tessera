"""A parsed window unit holds no anchor, code or completion plane (tessera#502).

A window body decodes from BODY and its table; the TCQ-only fields a unit
carries -- ``anchors``, ``codes``, ``completion_bits`` -- are zeros on it.
The reader used to allocate three int64 ``(steps, cols)`` planes of those
zeros per unit, 24 bytes per weight, more than the wire's own 4-bit body
occupied, on every role of every routed expert of a serving load.  Now one
shared zero-stride view stands in for all three.  Pinned here:

- the three fields are one element of storage, read as zeros at the unit's
  shape, and refuse an in-place write instead of aliasing it;
- a shard cut by ``slice_unit`` keeps them that way (the sharded load cuts
  every role);
- the unit decodes and rewrites to the bytes it was read from, whole and cut.
"""

import pytest
import torch

from tessera.alphabet import E4M3_GRID
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.slicing import slice_unit
from tessera.trellis import ConvCode
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact, read_unit_artifact

CODE = ConvCode(memory=6)
ROWS, COLS, Q256 = 16, 256, 1024


@pytest.fixture(scope="module")
def window_blob():
    g = torch.Generator().manual_seed(7)
    w = torch.randn(ROWS, COLS, generator=g) * 0.02
    unit = encode_unit(w, E4M3_GRID, (4,) * COLS, CODE, body=BodyKind.WINDOW,
                       window_bits=8, scale_plane=ScalePlaneKind.CHANNEL, scale_refit=1)
    return build_unit_artifact(unit, "u", E4M3_GRID, Q256, CODE, fixture_id=None)[2]


def _assert_zero_view(plane, shape):
    assert tuple(plane.shape) == shape and plane.dtype == torch.long
    assert plane.untyped_storage().nbytes() <= 8, plane.untyped_storage().nbytes()
    assert not bool(plane.any())


def test_a_parsed_window_unit_allocates_no_step_planes(window_blob):
    unit = parse_unit_artifact(window_blob).unit
    steps, cols = unit.body_bits.shape
    for name in ("anchors", "codes", "completion_bits"):
        _assert_zero_view(getattr(unit, name), (steps, cols))
    # A write across elements that share the one word raises rather than
    # aliasing; nothing in the tree writes these fields at all.
    with pytest.raises(RuntimeError):
        unit.anchors[:, 0] = 1


def test_a_shard_keeps_the_view(window_blob):
    parsed = parse_unit_artifact(window_blob)
    shard = slice_unit(parsed, rows=(ROWS // 2, ROWS), cols=(COLS // 4, COLS))
    steps, cols = shard.body_bits.shape
    for name in ("anchors", "codes", "completion_bits"):
        _assert_zero_view(getattr(shard, name), (steps, cols))


def test_the_unit_decodes_and_rewrites_to_its_own_bytes(window_blob):
    parsed = parse_unit_artifact(window_blob)
    assert torch.equal(reconstruct_unit(parsed.unit, parsed.forests, None),
                       read_unit_artifact(window_blob))
    rebuilt = build_unit_artifact(parsed.unit, "u", parsed.forests, Q256, CODE,
                                  fixture_id=None)[2]
    assert rebuilt == window_blob
    whole = build_unit_artifact(slice_unit(parsed), "u", parsed.forests, Q256, CODE,
                                fixture_id=None)[2]
    assert whole == window_blob
