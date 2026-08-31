"""The seam tests: bits to bytes to bits, and bytes to weights.

These are the only tests in the suite that exercise a *serialised* artifact.
Everything else round-trips tensors, which cannot see bit order, per-superblock
counts, sub-byte padding, or whether the artifact is self-describing at all.
"""

import pytest
import torch

from tessera.alphabet import build_forest
from tessera.container import parse
from tessera.decode import materialize_nvfp4, reconstruct_unit
from tessera.encode import _pack_scales, encode_unit
from tessera.errors import GrammarError
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.trellis import ConvCode
from tessera.unit_artifact import (
    build_unit_artifact,
    encoder_profile_id,
    read_unit_artifact,
)
from tessera.wire import (
    nvfp4_scale_bytes,
    pack_body,
    pack_uniform,
    scales_from_planes,
    unpack_body,
    unpack_uniform,
)

CODE = ConvCode(memory=6)
FORESTS = {rate: build_forest(rate) for rate in (1, 2, 3)}


def _unit(rows=64, cols=512, q256=640, released=0, seed=0, diagonals=True):
    torch.manual_seed(seed)
    weights = torch.randn(rows, cols) * 0.02
    rates = bresenham_rate_schedule(root_from_q256(q256), cols)
    unit = encode_unit(
        weights, FORESTS, rates, CODE,
        with_diagonals=diagonals, released_positions=released,
    )
    return weights, unit


# ---------------------------------------------------------------- bit packing


@pytest.mark.parametrize("width", [1, 2, 3, 4, 8, 16])
def test_uniform_pack_round_trips(width):
    values = torch.randint(0, 1 << width, (257,))
    blob = pack_uniform(values, width)
    assert len(blob) == (257 * width + 7) // 8
    assert torch.equal(unpack_uniform(blob, 257, width), values)


def test_pack_is_msb_first():
    """MSB-first is not a preference: verify_plane_region's pad-bit check
    assumes the slack lands in the *low* bits of the final content byte."""
    assert pack_uniform(torch.tensor([1]), 4) == b"\x10"
    assert pack_uniform(torch.tensor([0b101]), 3) == b"\xa0"


def test_uniform_pack_refuses_overwide_values():
    with pytest.raises(GrammarError, match="out of range"):
        pack_uniform(torch.tensor([16]), 4)


def test_body_pack_round_trips_mixed_rates():
    rates = bresenham_rate_schedule(root_from_q256(640), 512)
    body = torch.stack([
        torch.randint(0, 1 << r, (64,)) for r in rates
    ], dim=1)
    blob = pack_body(body, rates)
    assert len(blob) == (sum(rates) * 64 + 7) // 8
    assert torch.equal(unpack_body(blob, rates, 64), body)


def test_body_pack_handles_zero_width_columns():
    """At R=3, c = 3-R = 0, so a full COMPLETION plane is entirely zero-width.
    That is the commonest case in the format, not an edge case."""
    rates = (0,) * 16
    body = torch.zeros(8, 16, dtype=torch.long)
    assert pack_body(body, rates) == b""
    assert torch.equal(unpack_body(b"", rates, 8), body)


# ------------------------------------------------------------------ §6b codec


def test_scales_from_planes_is_bit_exact():
    """S6b's round trip is what the doc says T-nvfp4-class is conjectural
    without.  Not 'close': equal."""
    weights, unit = _unit(diagonals=False)
    _, _, effective = _pack_scales(weights.float(), 32, 16)
    assert torch.equal(
        scales_from_planes(unit.scale_base, unit.scale_refine), effective
    )


def test_scales_from_planes_checks_plane_agreement():
    _, unit = _unit()
    with pytest.raises(GrammarError, match="do not match"):
        scales_from_planes(unit.scale_base, unit.scale_refine[:-3])


def test_nvfp4_scale_plane_is_an_exact_relabelling():
    """The E4M3 block scale times the po2 global scale reproduces the S6b
    number exactly -- so the Tessera decode and the compressed-tensors artifact
    hold the identical value, which is principle 8's rendering identity."""
    _, unit = _unit()
    effective = scales_from_planes(unit.scale_base, unit.scale_refine)
    packed, global_scale = nvfp4_scale_bytes(unit.scale_base, unit.scale_refine)
    as_float = packed.view(torch.float8_e4m3fn).to(torch.float32) * global_scale
    assert torch.equal(as_float, effective.float())


def test_global_scale_is_a_power_of_two():
    """A non-po2 global scale (the usual amax/448) would reintroduce rounding
    and break the exactness above."""
    _, unit = _unit()
    _, global_scale = nvfp4_scale_bytes(unit.scale_base, unit.scale_refine)
    mantissa, _ = torch.frexp(torch.tensor(global_scale))
    assert float(mantissa) == 0.5


def test_materialized_nibbles_are_all_legal():
    _, unit = _unit()
    packed, scales, _ = materialize_nvfp4(unit.codes, unit.scale_base, unit.scale_refine)
    assert packed.dtype is torch.uint8 and scales.dtype is torch.uint8
    assert packed.shape == (64, 256) and scales.shape == (64, 32)


# -------------------------------------------------------------- the full seam


@pytest.mark.parametrize(
    "q256,released", [(256, 0), (640, 0), (640, 3000), (768, 5000), (768, 0)]
)
def test_wire_round_trip_is_exact(q256, released):
    """Encode -> pack -> serialise -> parse -> unpack -> decode, and the
    weights that come back off the bytes equal the ones the tensor-level
    inverse produces.  This is the test that makes 1a/1b and the encoder one
    artifact instead of two."""
    _, unit = _unit(q256=q256, released=released)
    reference = reconstruct_unit(unit, FORESTS, CODE)
    _, _, blob = build_unit_artifact(unit, "unit0", FORESTS, q256, CODE)
    assert torch.equal(read_unit_artifact(blob), reference)


def test_reader_takes_bytes_and_nothing_else():
    """No forests, no scale tensor, no ConvCode passed alongside.  If the
    reader needed the encoder's context the format would not be a format."""
    _, unit = _unit(q256=640, released=1000)
    _, _, blob = build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)
    assert read_unit_artifact(blob).shape == (64, 512)


def test_artifact_parses_as_a_declared_terminal():
    _, unit = _unit()
    manifest, region, blob = build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)
    art = parse(blob)
    assert art.terminal.slot_id == "t-nvfp4"
    assert art.terminal.exact_bytes == len(region)


def test_release_placement_is_recovered_not_stored():
    """The RELEASE plane stores 4 bits of *code* per position and no index --
    that is where its rate advantage comes from.  The reader has to rebuild
    S9's order from the pre-release decode, so a wrong order would corrupt
    exactly the released positions and nothing else."""
    _, unit = _unit(q256=640, released=2000)
    _, _, blob = build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)
    recovered = read_unit_artifact(blob)
    assert torch.equal(recovered, reconstruct_unit(unit, FORESTS, CODE))


def test_decoder_derives_its_own_scale():
    """reconstruct_unit with no scale argument must equal the encoder's."""
    weights, unit = _unit(diagonals=False)
    _, _, effective = _pack_scales(weights.float(), 32, 16)
    explicit = torch.repeat_interleave(effective, 16).reshape(64, 512)
    assert torch.equal(
        reconstruct_unit(unit, FORESTS, CODE),
        reconstruct_unit(unit, FORESTS, CODE, explicit),
    )


# ------------------------------------------------------- the code is wire


def test_encoder_profile_commits_to_the_convolutional_code():
    """Two encoders disagreeing on memory or generators emit streams that
    decode to each other's nonsense, silently.  trellis.py claims the profile
    id covers them; this is what makes that true."""
    rates = (3,) * 8
    ids = {encoder_profile_id(ConvCode(memory=m), rates) for m in (3, 4, 5, 6, 8)}
    assert len(ids) == 5
    custom = ConvCode(memory=3, generators=(0o7, 0o5))
    assert encoder_profile_id(custom, rates) != encoder_profile_id(ConvCode(3), rates)


def test_reader_fails_closed_on_an_unknown_trellis():
    _, unit = _unit()
    manifest, region, blob = build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)
    from dataclasses import replace as dc_replace
    from tessera.container import serialize

    forged = dc_replace(manifest, encoder_profile_id=bytes(32))
    with pytest.raises(GrammarError, match="matches no convolutional code"):
        read_unit_artifact(serialize(forged, region))
