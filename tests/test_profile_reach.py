"""Issue #77: ``encoder_profile_id`` must bind the reach terms.

Two units with the same profile id and different ``channel_sigma`` are
different bytes -- the reach-aware start moves the row scales, and the
window table moves with ``window_sigma``/``window_seed`` -- so any consumer
that treats id equality as byte equality (a cache resume key, a merge guard,
an A/B assuming one arm is a re-cut of the other) reads them as the same
object.

The rule, from the code that owns it: the digest binds a reach spelling
exactly where that spelling moves bytes -- ``window_seed``/``window_sigma``
under a WINDOW body, ``channel_sigma`` under a CHANNEL plane -- and binds
nothing elsewhere, so every default build keeps the digest (and the bytes
and the minor) it always had. ``None`` spells the grid-derived default, the
same convention the checkpoint config already uses.
"""
import pytest
import torch

from tessera.alphabet import E4M3_GRID, build_forest
from tessera.container import parse
from tessera.planes import PlaneLayout
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.errors import TesseraError
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.scale_channel import default_channel_sigma
from tessera.trellis import ConvCode
from tessera.unit_artifact import (
    build_unit_artifact,
    encoder_profile_id,
    parse_unit_artifact,
    read_unit_artifact,
)

CODE = ConvCode(memory=6)
CHANNEL = ScalePlaneKind.CHANNEL
WINDOW = BodyKind.WINDOW
RATES = (2,) * 64
Q256 = 2 * 256


def _weights(rows=32, cols=64, seed=0):
    torch.manual_seed(seed)
    return torch.randn(rows, cols) * 0.02


def _window_unit(w, window_bits=6, scale_plane=CHANNEL, **reach):
    return encode_unit(
        w, E4M3_GRID, RATES, CODE, body=WINDOW, window_bits=window_bits,
        scale_plane=scale_plane, scale_refit=1, completion=0, **reach)


def _built(unit):
    # These tests pin the reach record's own lowest minor.  Suppress the
    # independent encoder-identity envelope so that assertion stays literal.
    manifest, region, blob = build_unit_artifact(
        unit, "unit0", E4M3_GRID, Q256, CODE, fixture_id=None, layout=PlaneLayout.LEGACY
    )
    return manifest, region, blob


# ------------------------------------------------------- the defect, end to end


def test_two_channel_sigmas_are_two_profiles():
    """One unit encoded twice, at two ``channel_sigma`` values: the bytes move
    (the row scales provably do -- ``rms / sigma``), so the ids must."""
    w = _weights()
    default = default_channel_sigma(E4M3_GRID)
    # Halving the spread doubles every row's initial scale by construction, so
    # the DIAG_SV bytes cannot coincide: the premise, not the claim.
    other = default / 2
    assert other != default and other > 0
    u_plain = _window_unit(w, channel_sigma=None)
    u_other = _window_unit(w, channel_sigma=other)
    # Halving the spread doubles every row's initial scale, so the power of
    # two that puts the median row near one doubles too: the global rides the
    # manifest, so the bytes cannot coincide -- the premise, not the claim.
    # (The per-row words reconverge under the refit; the global does not.)
    assert u_plain.scale_global != u_other.scale_global
    m_plain, _, b_plain = _built(u_plain)
    m_other, _, b_other = _built(u_other)
    assert b_plain != b_other
    assert m_plain.encoder_profile_id != m_other.encoder_profile_id


def test_two_window_sigmas_are_two_profiles():
    """Same, for the table's own spread under a block plane: the table is the
    ALPHABET plane, so different tables are different bytes by construction."""
    w = _weights()
    plane = ScalePlaneKind.LUT
    u_plain = _window_unit(w, scale_plane=plane, window_sigma=None)
    u_shaped = _window_unit(w, scale_plane=plane, window_sigma=default_channel_sigma(E4M3_GRID))
    assert not torch.equal(u_plain.window_codes, u_shaped.window_codes)
    m_plain, _, b_plain = _built(u_plain)
    m_shaped, _, b_shaped = _built(u_shaped)
    assert b_plain != b_shaped
    assert m_plain.encoder_profile_id != m_shaped.encoder_profile_id


def test_two_window_seeds_are_two_profiles():
    """Same, for the table's seed: a different permutation is a different
    ALPHABET plane."""
    w = _weights()
    u0 = _window_unit(w, scale_plane=ScalePlaneKind.LUT, window_seed=0)
    u1 = _window_unit(w, scale_plane=ScalePlaneKind.LUT, window_seed=1)
    assert not torch.equal(u0.window_codes, u1.window_codes)
    m0, _, b0 = _built(u0)
    m1, _, b1 = _built(u1)
    assert b0 != b1
    assert m0.encoder_profile_id != m1.encoder_profile_id


def test_a_bound_reach_writes_minor_5_and_reads_from_bytes_alone():
    """A unit with a non-default reach spelling declares schema minor 5, and
    the reader -- which takes the record off the manifest and recomputes the
    digest with it -- needs nothing but the bytes."""
    w = _weights()
    unit = _window_unit(w, channel_sigma=default_channel_sigma(E4M3_GRID) / 2)
    manifest, _region, blob = _built(unit)
    assert blob[10] == 5, "schema minor 5"
    assert manifest.schema_minor == 5
    assert manifest.reach is not None
    assert float(manifest.reach.channel_sigma) == pytest.approx(
        default_channel_sigma(E4M3_GRID) / 2)
    assert manifest.reach.window_seed == 0
    assert manifest.reach.window_sigma is None
    assert torch.equal(read_unit_artifact(blob), reconstruct_unit(unit, E4M3_GRID, None))
    # A header too old to carry the record cannot occur: refused, not misread.
    stale = bytearray(blob)
    stale[10] = 4
    with pytest.raises(TesseraError):
        read_unit_artifact(bytes(stale))


# ------------------------------------------------- the rule, at the digest


def test_the_digest_binds_reach_spellings_where_they_move_bytes():
    """The conditional rule, pinned without an encoder: WINDOW binds seed and
    table spread, CHANNEL binds the row spread, and each ignores the other's
    slots -- while all-default spellings reproduce the old digest exactly."""
    rates = (2,) * 8
    base = encoder_profile_id(None, rates, E4M3_GRID, 1, CHANNEL, WINDOW, 6)
    assert encoder_profile_id(None, rates, E4M3_GRID, 1, CHANNEL, WINDOW, 6,
                              0, None, None) == base
    assert encoder_profile_id(None, rates, E4M3_GRID, 1, CHANNEL, WINDOW, 6,
                              1, None, None) != base
    assert encoder_profile_id(None, rates, E4M3_GRID, 1, CHANNEL, WINDOW, 6,
                              0, 1.0, None) != base
    assert encoder_profile_id(None, rates, E4M3_GRID, 1, CHANNEL, WINDOW, 6,
                              0, None, 1.0) != base
    # A block plane never reads channel_sigma: inert there, whatever it says.
    lut_plain = encoder_profile_id(None, rates, E4M3_GRID, 1, ScalePlaneKind.LUT, WINDOW, 6)
    assert encoder_profile_id(None, rates, E4M3_GRID, 1, ScalePlaneKind.LUT, WINDOW, 6,
                              0, None, 2.5) == lut_plain
    # TCQ never reads the window slots: inert there too.
    tcq_plain = encoder_profile_id(CODE, rates, E4M3_GRID, 1, CHANNEL, BodyKind.TCQ, 0)
    assert encoder_profile_id(CODE, rates, E4M3_GRID, 1, CHANNEL, BodyKind.TCQ, 0,
                              7, 2.5, None) == tcq_plain


# ------------------------------------------------- what did not move


def test_default_builds_keep_their_bytes_minor_and_digest():
    """Lowest-minor discipline: a default reach spelling says nothing, so no
    record is written, the minor stays where it was, and the digest is the
    old one -- for the born-against encoder requested by ``_built``.  A
    current default production build also carries the independent
    ``encoder_fixture_id`` envelope, whose minor is tested separately."""
    w = _weights()
    unit = _window_unit(w)
    manifest, _region, blob = _built(unit)
    assert blob[10] == 3 and manifest.schema_minor == 3
    assert manifest.reach is None
    assert manifest.encoder_profile_id == encoder_profile_id(
        None, RATES, E4M3_GRID, 1, CHANNEL, WINDOW, 6)
    assert torch.equal(read_unit_artifact(blob), reconstruct_unit(unit, E4M3_GRID, None))


def test_inert_reach_attrs_change_neither_bytes_nor_digest():
    """Reach attributes on a unit whose body/plane never reads them are
    normalised away: a TCQ+S6b unit carrying junk reach spellings writes the
    bytes (and the minor-0 digest) it always wrote."""
    from tessera.alphabet import build_forest

    torch.manual_seed(4)
    w = torch.randn(32, 64) * 0.02
    rates = (2,) * 64
    forests = {2: build_forest(2, grid=E4M3_GRID)}
    unit = encode_unit(w, forests, rates, CODE, scale_refit=1, completion=0)
    _m0, _r0, blob0 = build_unit_artifact(
        unit, "unit0", forests, Q256, CODE, fixture_id=None, layout=PlaneLayout.LEGACY
    )
    unit.window_seed = 7
    unit.window_sigma = 2.5
    unit.channel_sigma = 2.5
    manifest, _region, blob = build_unit_artifact(
        unit, "unit0", forests, Q256, CODE, fixture_id=None, layout=PlaneLayout.LEGACY
    )
    assert blob == blob0
    assert blob[10] == 0 and manifest.schema_minor == 0


def test_a_shard_keeps_its_parents_reach_profile():
    """Reach spellings ride the unit the way span and body already do: the
    identity slice of a minor-5 unit rebuilds to the parent's own bytes, and
    a row shard verifies under the parent's profile id."""
    from tessera.layout import slice_unit

    w = _weights()
    unit = _window_unit(w, channel_sigma=default_channel_sigma(E4M3_GRID) / 2)
    _m, _r, blob = _built(unit)
    parsed = parse_unit_artifact(blob)
    whole = slice_unit(parsed)
    # A byte-for-byte rebuild inherits the parsed artifact's identity *and*
    # its plane layout, both explicitly: ``_built`` asks for the minor-5
    # spelling, which is the LEGACY layout.
    _m2, _r2, again = build_unit_artifact(
        whole, parsed.manifest.branch.unit_id, parsed.forests,
        parsed.manifest.branch.root_q256, parsed.code or CODE,
        fixture_id=parsed.manifest.encoder_fixture_id,
        layout=parsed.manifest.layout)
    assert again == blob
    shard = slice_unit(parsed, rows=(0, 16))
    _m3, _r3, shard_blob = build_unit_artifact(
        shard, parsed.manifest.branch.unit_id, parsed.forests,
        parsed.manifest.branch.root_q256, parsed.code or CODE,
        fixture_id=parsed.manifest.encoder_fixture_id)
    assert parse(shard_blob).manifest.encoder_profile_id == parsed.manifest.encoder_profile_id
    assert torch.equal(
        read_unit_artifact(shard_blob),
        read_unit_artifact(blob)[:16, :])


def test_an_explicitly_resolved_default_spread_is_the_default_profile():
    """#81.  ``None`` means "resolve the grid's default"; a caller that
    resolves it itself has not changed encoder, and the bytes agree.  Every
    sweep harness resolves it -- and a per-unit spread rule would -- so an id
    that distinguished the two would refuse on nothing, over and over."""
    from tessera.scale_channel import default_channel_sigma

    kw = dict(
        grid=E4M3_GRID, span=1, scale_plane=ScalePlaneKind.CHANNEL,
        body=BodyKind.WINDOW, window_bits=14,
    )
    implicit = encoder_profile_id(None, (4,), channel_sigma=None, **kw)
    explicit = encoder_profile_id(
        None, (4,), channel_sigma=default_channel_sigma(E4M3_GRID), **kw)
    assert implicit == explicit

    # ...and the normalisation is to the default only, not to everything:
    moved = encoder_profile_id(
        None, (4,),
        channel_sigma=1.25 * default_channel_sigma(E4M3_GRID), **kw)
    assert moved != implicit


def test_a_resolved_channel_spread_spelled_as_window_sigma_is_the_default(
):
    """#90, #81's sibling one slot over.

    Under a CHANNEL plane the encoder resolves an unset ``window_sigma`` to the
    channel spread, for the table and for the reach both.  So ``None`` and that
    resolved number are one encoder, and must be one profile.  Before the fix
    the second spelling cost a different id, 20 extra bytes and a schema-minor
    bump that drops the artifact below every reader under minor 5 -- all for a
    byte-identical decoded tensor.
    """
    grid = E4M3_GRID
    resolved = default_channel_sigma(grid)
    common = dict(
        code=None, rates=(4,) * 8, grid=grid, span=1,
        scale_plane=ScalePlaneKind.CHANNEL, body=BodyKind.WINDOW,
        window_bits=6, window_seed=0,
    )
    assert (encoder_profile_id(window_sigma=None, **common)
            == encoder_profile_id(window_sigma=resolved, **common))
    # And the normalisation is not a blanket one: a spread that is NOT the
    # resolved default is a different encoder and must keep its own id, or
    # this test would pass against a function that erases the slot.
    assert (encoder_profile_id(window_sigma=None, **common)
            != encoder_profile_id(window_sigma=resolved * 1.25, **common))
    # Off a CHANNEL plane the encoder never resolves window_sigma from the
    # channel spread, so the equality must NOT hold there.
    s6b = dict(common, scale_plane=ScalePlaneKind.S6B)
    assert (encoder_profile_id(window_sigma=None, **s6b)
            != encoder_profile_id(window_sigma=resolved, **s6b))


def test_a_reach_ratio_no_float_holds_is_refused_and_a_float_ratio_round_trips():
    """#254.  The wire carries sigmas as exact ratios and the record promises
    to refuse a stored spread that is not exactly representable as its float
    value -- but ``decode`` rounded the ratio to float before construction,
    so the check could only ever see the rounded number.  ``1/3`` was
    accepted as ``0.3333333333333333`` and re-encoded as
    ``6004799503160661/18014398509481984``: an accepted canonical record
    whose bytes change on a read/write cycle, and two distinct on-wire
    spreads collapsing to one reconstructed profile input.  Exactness, not a
    tolerance: a ratio is legal exactly when ``Fraction(float(r)) == r``.
    Both sigma slots, both directions."""
    from fractions import Fraction

    from tessera.canonical import Reader, Writer
    from tessera.errors import ManifestError
    from tessera.manifest import ReachParams

    def record(position, ratio):
        writer = Writer().uint(7)  # seed
        for slot in range(2):
            if slot == position:
                writer.uint(1).ratio(ratio)
            else:
                writer.uint(0)
        return writer.bytes

    for position, name in ((0, "window_sigma"), (1, "channel_sigma")):
        raw = record(position, Fraction(3, 4))  # exactly a float
        decoded = ReachParams.decode(Reader(raw))
        assert getattr(decoded, name) == 0.75
        out = Writer()
        decoded.encode(out)
        assert out.bytes == raw, (
            f"an accepted canonical {name} record must round-trip unchanged"
        )
        with pytest.raises(ManifestError, match=f"{name}.*no float holds"):
            ReachParams.decode(Reader(record(position, Fraction(1, 3))))
