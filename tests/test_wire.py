"""The seam tests: bits to bytes to bits, and bytes to weights.

These are the only tests in the suite that exercise a *serialised* artifact.
Everything else round-trips tensors, which cannot see bit order, per-superblock
counts, sub-byte padding, or whether the artifact is self-describing at all.
"""

import os
import struct
import warnings
from unittest import mock

import pytest
import torch

from tessera.alphabet import build_forest
from tessera.container import parse
from tessera.decode import materialize_nvfp4, reconstruct_unit
from tessera.encode import _pack_scales, encode_unit
from tessera.errors import GrammarError
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.trellis import ConvCode
from tessera.grammar import release_quota, superblock_widths
from tessera.unit_artifact import (
    build_unit_artifact,
    encoder_profile_id,
    parse_unit_artifact,
    read_unit_artifact,
)
from tessera.wire import (
    nvfp4_scale_bytes,
    pack_body,
    pack_fp16,
    pack_uniform,
    scales_from_planes,
    unpack_body,
    unpack_fp16,
    unpack_uniform,
)

CODE = ConvCode(memory=6)
FORESTS = {rate: build_forest(rate) for rate in (1, 2, 3)}


def _unit(rows=64, cols=512, q256=640, released=0, seed=0, diagonals=True):
    torch.manual_seed(seed)
    weights = torch.randn(rows, cols) * 0.02
    rates = bresenham_rate_schedule(root_from_q256(q256), cols)
    # These tests re-derive the amax plane from the weights with
    # ``_pack_scales`` and hold the unit to it, so they run the encoder without
    # its scale refit; ``test_scale_refit.py`` covers the refit plane.
    unit = encode_unit(
        weights, FORESTS, rates, CODE,
        with_diagonals=diagonals, released_positions=released, scale_refit=0,
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


def test_release_reaches_a_trailing_partial_superblock():
    """A 640-column unit has three superblocks -- 256, 256, 128 -- and the
    layout gives the last one a granule.  The release quota used to run over
    ``cols // superblock`` blocks, a floor, so positions 512..639 were
    unreachable: the unit paid for a granule release could never populate.
    Encoder and decoder floored alike, so the round trip never noticed; the
    only way to see it is to look at *which* superblocks were placed in.
    """
    cols, superblock = 640, 256
    _, unit = _unit(cols=cols, q256=640, released=8)
    blocks = sorted(set(((unit.release_index % cols) // superblock).tolist()))
    assert blocks == [0, 1, 2]
    _, _, blob = build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)
    assert torch.equal(read_unit_artifact(blob), reconstruct_unit(unit, FORESTS, CODE))


@pytest.mark.parametrize("cols", [320, 384, 512, 640, 768])
@pytest.mark.parametrize("released", [0, 1, 64, 3000])
def test_release_placement_survives_the_bytes_at_every_width(cols, released):
    """The writer's quota and the reader's respread are the same function.

    The reader regenerates a whole unit's placement from the *total* alone
    (``unit_artifact._release_placement``), so the two sides agree only if they
    apportion that total the same way.  Element for element, not as a set and
    not merely as a decode: a placement that recovered the same positions in a
    different order would still be a drift, because the RELEASE plane's codes
    are stored in placement order and nothing else names them.
    """
    rows = 64
    _, unit = _unit(rows=rows, cols=cols, q256=640, released=released)
    assert unit.released_positions == released
    assert unit.release_index.numel() == released
    _, _, blob = build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)
    recovered = parse_unit_artifact(blob).unit
    assert torch.equal(recovered.release_index, unit.release_index)
    assert torch.equal(recovered.release_code, unit.release_code)
    assert torch.equal(read_unit_artifact(blob), reconstruct_unit(unit, FORESTS, CODE))
    # ...and the placement is the quota's, superblock for superblock.
    superblock = 256
    counts = release_quota(released, cols, superblock)
    placed = ((unit.release_index % cols) // superblock).tolist()
    assert tuple(placed.count(b) for b in range(len(counts))) == counts


@pytest.mark.parametrize("cols", [320, 640])
def test_release_fills_the_unit_where_an_equal_count_quota_refused(cols):
    """Issue #27, fail-before.  An equal *count* per superblock asks a narrow
    trailing block for the same number of releases as a full one -- up to
    ``superblock_columns`` times the density -- and so overruns it long before
    the unit is full: on 8x640 the equal-count quota refused at a total of
    4000 of 5120 positions, and on 64x320 at 12000 of 20480.  The
    width-proportional quota holds the *density* equal instead, so every total
    the unit has positions for is placeable, up to and including all of them.
    """
    rows = 8
    positions = rows * cols
    for released in (positions // 2, positions - 1, positions):
        _, unit = _unit(rows=rows, cols=cols, q256=640, released=released)
        assert unit.release_index.numel() == released
        assert len(set(unit.release_index.tolist())) == released
        _, _, blob = build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)
        assert torch.equal(
            parse_unit_artifact(blob).unit.release_index, unit.release_index
        )


def test_release_refuses_more_positions_than_the_unit_has():
    """What the overrun guard is *for*, once the quota is width-proportional.

    At a legal total the guard cannot fire -- a superblock's share of
    ``total <= rows * cols`` is at most its ``rows * width`` positions, which
    ``tests/test_grammar.py`` proves directly over every width.  What is left
    for it to catch is an *illegal* total: asking for more releases than the
    unit has positions, which it refuses at the first superblock whose share
    overruns, before the layout's own range check ever sees the count.
    """
    with pytest.raises(GrammarError, match="superblock 0 releases 2049 of 2048"):
        _unit(rows=8, cols=640, q256=640, released=8 * 640 + 1)


def test_release_density_is_equal_across_a_partial_superblock():
    """The principle ``layout._superblock_counts`` already applies to BODY and
    COMPLETION -- "a granule's count has to be the bits that granule's columns
    actually carry" -- read for RELEASE, whose per-position quantity is
    positions.  A 640-column unit's last superblock is half-width and takes
    half the releases, not the same number.
    """
    rows, cols, superblock = 8, 640, 256
    _, unit = _unit(rows=rows, cols=cols, q256=640, released=4000)
    widths = superblock_widths(cols, superblock)
    assert widths == (256, 256, 128)
    placed = ((unit.release_index % cols) // superblock).tolist()
    counts = [placed.count(b) for b in range(len(widths))]
    assert counts == [1600, 1600, 800]
    density = {c / (rows * w) for c, w in zip(counts, widths)}
    assert density == {4000 / (rows * cols)}


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
    with pytest.raises(GrammarError, match=r"matches no \(convolutional code, payload grid\) pair"):
        read_unit_artifact(serialize(forged, region))


@pytest.mark.parametrize("q256,released", [(640, 0), (768, 2000), (256, 500)])
def test_wire_round_trip_without_diagonals(q256, released):
    """The *recommended* recipe has segment 2a off, so it is the configuration
    that most needs a wire path.  It had none: build_unit_artifact refused a
    unit with no diagonals, and every other test here passes diagonals=True."""
    _, unit = _unit(q256=q256, released=released, diagonals=False)
    reference = reconstruct_unit(unit, FORESTS, CODE)
    _, region, blob = build_unit_artifact(unit, "unit0", FORESTS, q256, CODE)
    assert torch.equal(read_unit_artifact(blob), reference)


def test_absent_diagonals_shrink_the_artifact_by_their_exact_size():
    """Absent is not the same as truncated away: the planes must not be
    declared at all, or every offset after DIAG_SU is wrong."""
    _, with_d = _unit(q256=640, diagonals=True)
    _, without = _unit(q256=640, diagonals=False)
    _, region_a, _ = build_unit_artifact(with_d, "u", FORESTS, 640, CODE)
    _, region_b, _ = build_unit_artifact(without, "u", FORESTS, 640, CODE)
    assert len(region_a) - len(region_b) == 2 * (64 + 512)  # 16 bits per channel


@pytest.mark.parametrize("memory", [3, 4, 5, 6, 8])
def test_fused_replay_equals_the_eager_path_bit_for_bit(memory):
    """The fused path is an optimisation, so it owes an exact match.

    ``TESSERA_FUSED_REPLAY=0`` selects the eager chain; the compiled one must
    agree on every position, or the decoder's output depends on whether
    inductor was available -- which is the same class of bug as a producer and
    consumer disagreeing about the wire.
    """
    from tessera.decode import replay_body

    device = "cuda" if torch.cuda.is_available() else "cpu"
    code = ConvCode(memory=memory)
    for rate in (1, 2, 3):
        forest = build_forest(rate)
        bits = torch.randint(0, 1 << rate, (129, 96), device=device, dtype=torch.uint8)
        # No ``cache_clear()`` around the toggle: the env var is read per call.
        with mock.patch.dict(os.environ, {"TESSERA_FUSED_REPLAY": "0"}):
            eager = replay_body(bits, forest, code)
        fused = replay_body(bits, forest, code)
        assert torch.equal(eager, fused)


def test_an_unwritable_global_scale_is_refused_where_the_field_has_a_name():
    """A scale the codec cannot write is refused by the plane, not by the codec.

    ``ScalePlane`` already refuses a global scale that is not exactly
    representable as a float.  That is a weaker condition than the wire's:
    ``Fraction(3.7e-5)`` IS float-exact and has a 68-bit denominator, so it
    passed construction and then failed deep inside ``canonical.Writer.ratio``
    with ``value exceeds 64-bit domain: 147573952589676412928`` -- a codec error
    naming a 21-digit integer, with no mention of a scale, a plane, or a unit.
    The first person to hit it reads that and looks in the codec (#33).

    Dyadic rationals of modest denominator -- what every shipped plane snaps to
    -- stay legal, including ones that are not powers of two.
    """
    from fractions import Fraction

    from tessera.errors import ManifestError
    from tessera.manifest import ScalePlane

    for good in (2.0**-10, 2.0**-12, 0.75, 1.0, 2.0**20):
        assert ScalePlane.channel(good).global_scale == Fraction(good)
        assert ScalePlane.lut(bytes([0x30, 0x38]), good).global_scale == Fraction(good)

    for maker, field in ((lambda g: ScalePlane.channel(g), "CHANNEL"),
                         (lambda g: ScalePlane.lut(bytes([0x30, 0x38]), g), "LUT")):
        with pytest.raises(ManifestError) as excinfo:
            maker(3.7e-5)
        message = str(excinfo.value)
        # The three things the codec's message could not say.
        assert f"{field} global scale" in message
        assert "3.7e-05" in message
        assert "denominator" in message


# ------------------------------------------------- the E4M3 plane's top binade
def _scale_plane(scale_base, scale_refine):
    return (
        torch.tensor(scale_base, dtype=torch.uint8),
        torch.tensor(scale_refine, dtype=torch.uint8),
    )


def test_a_fifteen_binade_span_is_legal_unless_the_top_word_is_nan():
    """E4M3FN holds exponents 1..15, and ``e=15, m=7`` is its NaN.  A unit
    whose scales span exactly fifteen binades after the power-of-two shift is
    therefore representable unless a half at the top binade carries mantissa
    7.  The gate that decides this ran a numpy keyword against a torch tensor
    and raised ``TypeError`` on every fifteen-binade unit, legal or not."""
    # Exponents 0 and 14 (base 120 is ``e=0``); halves per group = 2.
    base = [120, 134]
    plane, global_scale = nvfp4_scale_bytes(*_scale_plane(base, [0, 0, 0, 0]))
    assert global_scale == 0.5
    assert plane.tolist() == [1 << 3, 1 << 3, 15 << 3, 15 << 3]

    # Mantissa 7 below the top binade is a normal E4M3 number.
    plane, _ = nvfp4_scale_bytes(*_scale_plane(base, [7, 7, 0, 0]))
    assert plane.tolist() == [(1 << 3) | 7, (1 << 3) | 7, 15 << 3, 15 << 3]

    # Mantissa 7 at the top binade is 0x7F, which E4M3FN reads as NaN.
    with pytest.raises(GrammarError, match="span 15"):
        nvfp4_scale_bytes(*_scale_plane(base, [0, 0, 0, 7]))


def test_the_diagonal_planes_are_little_endian_by_the_format_not_by_the_host():
    """``pack_fp16``/``unpack_fp16`` document "little-endian" and wrote the
    host's order.  Every box this has run on is little-endian, so the pin is
    what the docstring already promised rather than a caught regression: on a
    big-endian host the old pair round-tripped against itself and against no
    other reader, which is the one thing a wire format may not do.
    ``unit_artifact``'s window-table reader states the same rule with the same
    ``<`` spelling."""
    values = torch.tensor([1.0, -2.5, 0.0, 65504.0, 6.103515625e-05], dtype=torch.float16)
    expected = struct.pack("<5e", *(float(v) for v in values))
    assert pack_fp16(values) == expected
    assert torch.equal(unpack_fp16(expected, len(values)), values)
    # A reader that took the host's order would decode the byte-swapped
    # stream, so the swap has to change the answer.
    swapped = struct.pack(">5e", *(float(v) for v in values))
    assert not torch.equal(unpack_fp16(swapped, len(values)), values)


def test_the_fusion_switch_is_read_on_every_call_not_once_per_process():
    """``TESSERA_FUSED_REPLAY`` was read *inside* an ``lru_cache(maxsize=1)``,
    so the first caller in a process decided for every later one: a serve or a
    test that set the variable afterwards silently got the earlier decision.
    Both existing togglers only passed because they called ``cache_clear()``
    by hand around the toggle, which is a workaround for this and not a
    property of the knob -- so this test deliberately does not clear anything.
    """
    from tessera import decode

    with mock.patch.dict(os.environ, {"TESSERA_FUSED_REPLAY": "1"}):
        first = decode._fused_replay()
    with mock.patch.dict(os.environ, {"TESSERA_FUSED_REPLAY": "0"}):
        assert decode._fused_replay() is None
        assert decode._fused_decode() is None
    with mock.patch.dict(os.environ, {"TESSERA_FUSED_REPLAY": "1"}):
        # And back: the compile is cached, the decision is not.
        assert decode._fused_replay() is first


def test_a_compiled_chain_that_falls_back_is_counted_and_says_so_once():
    """`except Exception: pass  # fall back, never fail closed` swallowed
    every exception from the compiled path.  Correct for the output -- the
    eager path is the same function -- but a permanently broken fusion and a
    working one were indistinguishable: no counter, no warning, and the
    fused-path speed claim silently stopped holding."""
    from tessera import decode

    def _always_raises(*_args):
        raise RuntimeError("inductor said no")

    decode._FUSION_FALLBACKS.pop("probe", None)
    decode._FUSION_LAST_ERROR.pop("probe", None)
    try:
        with pytest.warns(RuntimeWarning, match="fell back|answered instead"):
            assert decode._run_fused(_always_raises, (), "probe") is None
        # Only the first warns; every one counts.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert decode._run_fused(_always_raises, (), "probe") is None
        count, last = decode.fusion_fallbacks()["probe"]
        assert count == 2
        assert "inductor said no" in last
        # A chain that works is not counted.
        assert decode._run_fused(lambda: torch.zeros(1), (), "probe") is not None
        assert decode.fusion_fallbacks()["probe"][0] == 2
    finally:
        decode._FUSION_FALLBACKS.pop("probe", None)
        decode._FUSION_LAST_ERROR.pop("probe", None)
