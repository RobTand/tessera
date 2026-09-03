"""The §2 findings of the 2026-09-02 math audit, as tests.

Every test here fails at 6c82ed4 and passes after the fix pass.  They are
grouped by the finding letter of the triage brief, so a regression names the
finding it resurrects.

The wire is a compatibility surface and twenty-two artifacts exist on this
box, so each byte-affecting fix is paired with the shape that proves it does
*not* move a conforming unit: finding A changes the granule arithmetic, and
the conforming case is here to say it changes it only where the old
arithmetic was wrong.
"""

import hashlib
from fractions import Fraction

import pytest
import torch

from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.bitio import BitReader, BitWriter
from tessera.canonical import CanonicalEncodingError, Reader, Writer, encode_uint
from tessera.errors import GrammarError, ManifestError, PlaneLayoutError
from tessera.export import encode_linear
from tessera.grammar import (
    GRID_CODES,
    bresenham_rate_schedule,
    q256_from_root,
    root_from_q256,
    validate_descendant_map,
)
from tessera.layout import (
    ZERO_DIGEST,
    TerminalSpec,
    _counts_for,
    build_planes,
    build_terminal,
)
from tessera.manifest import (
    ArrangementMode,
    BranchIdentity,
    ContainerClass,
    Geometry,
    Manifest,
    RotationState,
    TerminalRecord,
)
from tessera.planes import PlaneKind
from tessera.trellis import body_bits
from tessera.unit_artifact import parse_unit_artifact

from conftest import ALPHABET_BLOB, DESCENDANT_BLOB, make_geometry


# ---------------------------------------------------------------------------
# A -- partial trailing superblocks
# ---------------------------------------------------------------------------


def _true_body_counts(rates, steps, span, superblock):
    return tuple(
        sum(body_bits(rate, steps, span) for rate in rates[i : i + superblock])
        for i in range(0, len(rates), superblock)
    )


def test_a_partial_trailing_superblock_gets_its_own_granule():
    """640 columns is three superblocks of 256, not two.

    The restart table is the segment-local seek contract, so a table with two
    entries for three superblocks misdescribes the stream -- even though a
    whole-unit decode, which never seeks, is unaffected.
    """
    torch.manual_seed(0)
    exported = encode_linear(
        torch.randn(64, 640), grid=tuple_grid(E2M1_GRID, 2), q256=896
    )
    manifest = parse_unit_artifact(exported.blob).manifest
    body = manifest.plane(PlaneKind.BODY)
    assert body.counts == (61440, 61440, 30720)
    assert body.restart_offsets == (0, 61440, 122880)


def test_a_granule_counts_are_the_true_per_superblock_sums():
    """Not a spread of the total: the sum over each superblock's own columns."""
    geometry = Geometry(
        rows=4,
        columns=20,
        superblock_columns=8,
        group_weights=32,
        half_weights=16,
        quantizable_params=80,
    )
    # Heterogeneous on purpose: a uniform schedule cannot tell a true sum from
    # an even spread.
    rates = (3, 1) * 4 + (2, 2) * 4 + (3, 1) * 2
    spec = TerminalSpec("t", tuple(3 - rate for rate in rates), with_scale_base=False)
    planes = build_planes(geometry, rates, ALPHABET_BLOB, DESCENDANT_BLOB, spec=spec)
    by_kind = {plane.kind: plane for plane in planes}
    assert by_kind[PlaneKind.BODY].counts == _true_body_counts(rates, 4, 1, 8)
    assert by_kind[PlaneKind.COMPLETION].counts == (
        sum(3 - rate for rate in rates[0:8]) * 4,
        sum(3 - rate for rate in rates[8:16]) * 4,
        sum(3 - rate for rate in rates[16:20]) * 4,
    )


def test_a_conforming_shape_keeps_the_counts_it_had():
    """The fix may not move a unit whose column count is a whole number of
    superblocks -- which is every artifact on this box."""
    geometry = make_geometry(rows=8, columns=32, superblock_columns=8)
    rates = bresenham_rate_schedule(root_from_q256(512), 32)
    planes = build_planes(geometry, rates, ALPHABET_BLOB, DESCENDANT_BLOB)
    body = {plane.kind: plane for plane in planes}[PlaneKind.BODY]
    assert body.counts == (128, 128, 128, 128)


def test_a_release_counts_cover_the_partial_superblock_too():
    """A shard's per-superblock release vector is checked against every
    superblock, the trailing partial one included."""
    geometry = Geometry(
        rows=4,
        columns=20,
        superblock_columns=8,
        group_weights=32,
        half_weights=16,
        quantizable_params=80,
    )
    with pytest.raises(PlaneLayoutError, match="3 superblocks"):
        build_planes(
            geometry,
            rates=(2,) * 20,
            alphabet_blob=ALPHABET_BLOB,
            descendant_blob=DESCENDANT_BLOB,
            max_released=4,
            release_counts=(2, 2),
        )


# ---------------------------------------------------------------------------
# B -- Manifest.schedule on a real E4M3 checkpoint
# ---------------------------------------------------------------------------


def test_b_schedule_reads_a_rate_above_the_e2m1_cap():
    """A property that cannot read a valid artifact is broken.

    An E4M3 unit carries rates 4 and 5; the cap belongs to the payload grid,
    which the manifest resolves only through ``encoder_profile_id``, so the
    schedule defers the ceiling exactly as ``__post_init__`` already does.
    """
    torch.manual_seed(1)
    exported = encode_linear(torch.randn(32, 256), grid=E4M3_GRID, q256=1024)
    manifest = parse_unit_artifact(exported.blob).manifest
    assert max(manifest.rates) > 3
    schedule = manifest.schedule
    assert schedule.rates == manifest.rates
    assert schedule.root == manifest.branch.root


def test_b_descendant_map_validates_at_a_wider_grid():
    """At cap 7 the grid is 256 codes, not 16, and rate 5 is legal."""
    rate, cap = 5, 7
    anchors = 1 << (rate + 1)  # 64
    depth = 1 << (cap - rate)  # 4
    mapping = {
        anchor: tuple(range(anchor * depth, (anchor + 1) * depth))
        for anchor in range(anchors)
    }
    validate_descendant_map(rate, cap - rate, mapping, cap=cap)
    assert anchors * depth == 1 << (cap + 1) == 256


def test_b_descendant_map_still_partitions_the_16_grid_by_default():
    """The default is unchanged: cap 3, sixteen codes."""
    mapping = {anchor: (anchor * 2, anchor * 2 + 1) for anchor in range(8)}
    validate_descendant_map(2, 1, mapping)
    assert GRID_CODES == 16


# ---------------------------------------------------------------------------
# C -- padding canonicality
# ---------------------------------------------------------------------------


def test_c_padding_check_passes_on_a_canonical_last_byte():
    writer = BitWriter()
    writer.write(0b101, 3)
    reader = BitReader(writer.bytes, bit_length=3)
    assert reader.read(3) == 0b101
    reader.check_padding_zero()  # the five pad bits are zero


def test_c_padding_check_rejects_a_non_zero_pad_bit():
    reader = BitReader(bytes([0b10100001]), bit_length=3)
    assert reader.read(3) == 0b101
    with pytest.raises(PlaneLayoutError, match="padding"):
        reader.check_padding_zero()


def test_c_padding_check_is_a_no_op_on_a_byte_boundary():
    reader = BitReader(bytes([0xFF]), bit_length=8)
    assert reader.read(8) == 0xFF
    reader.check_padding_zero()


# ---------------------------------------------------------------------------
# D -- validation holes
# ---------------------------------------------------------------------------


def test_d1_indivisible_positions_are_refused_not_floored():
    """10 weights over 32-weight groups is zero scale entries, silently."""
    geometry = Geometry(
        rows=2,
        columns=5,
        superblock_columns=256,
        group_weights=32,
        half_weights=16,
        quantizable_params=10,
    )
    for kind in (PlaneKind.SCALE_BASE, PlaneKind.SCALE_REFINE):
        with pytest.raises(GrammarError, match="whole number of"):
            _counts_for(kind, geometry, (1,) * 5, None, 0, 0)


def test_d1_an_absent_scale_plane_is_still_allowed():
    """A CHANNEL unit declares group/half and carries no block plane, so the
    rule binds the plane that is present, not the geometry."""
    geometry = Geometry(
        rows=2,
        columns=5,
        superblock_columns=256,
        group_weights=32,
        half_weights=16,
        quantizable_params=10,
    )
    spec = TerminalSpec("t", (0,) * 5, with_scale_base=False, with_row_scale=True)
    assert _counts_for(PlaneKind.SCALE_BASE, geometry, (1,) * 5, spec, 0, 0) == 0
    assert _counts_for(PlaneKind.SCALE_REFINE, geometry, (1,) * 5, spec, 0, 0) == 0


def _manifest_with_terminal_body_count(count: int) -> Manifest:
    """A valid unit whose single terminal cuts BODY at ``count`` elements.

    BODY's granules are (40, 40, 40, 40), so the quota boundaries are
    0/40/80/120/160 and nothing else.
    """
    geometry = Geometry(
        rows=8,
        columns=20,
        superblock_columns=5,
        group_weights=32,
        half_weights=16,
        quantizable_params=160,
    )
    rates = (1,) * 20
    planes = build_planes(
        geometry, rates, ALPHABET_BLOB, DESCENDANT_BLOB, with_diagonals=False
    )
    by_kind = {plane.kind: plane for plane in planes}
    order = [PlaneKind.ALPHABET, PlaneKind.DESCENDANT, PlaneKind.BODY]
    elements = []
    total_bytes = 0
    for kind in geometry_plane_order():
        if kind is PlaneKind.BODY:
            value = count
        elif kind in order:
            value = by_kind[kind].element_count
        else:
            value = 0
        elements.append(value)
        total_bytes += by_kind[kind].byte_length(value)
    terminal = TerminalRecord(
        slot_id="t-cut",
        clip_exponent_code=0,
        plane_elements=tuple(elements),
        exact_bytes=total_bytes,
        exact_bpp=Fraction(8 * total_bytes, geometry.quantizable_params),
        payload_digest=bytes(32),
    )
    return Manifest(
        encoder_profile_id=hashlib.sha256(b"profile").digest(),
        branch=BranchIdentity(
            unit_id="u",
            root_q256=256,
            rotation=RotationState.NONE,
            container=ContainerClass.GRIDBOOK,
        ),
        geometry=geometry,
        arrangement=ArrangementMode.BRESENHAM,
        rates=rates,
        planes=planes,
        terminals=(terminal,),
        payload_digest=bytes(32),
    )


def geometry_plane_order():
    from tessera.planes import CANONICAL_PLANE_ORDER

    return CANONICAL_PLANE_ORDER


def test_d2_terminal_must_cut_at_a_superblock_quota_boundary():
    """``planes.py`` declares the last non-empty plane is cut at a
    per-superblock quota boundary; nothing enforced it."""
    with pytest.raises(ManifestError, match="quota boundary"):
        _manifest_with_terminal_body_count(50)


def test_d2_a_boundary_aligned_terminal_still_validates():
    manifest = _manifest_with_terminal_body_count(80)
    assert manifest.terminals[0].plane_elements[2] == 80


def test_d3_reader_blob_rejects_an_out_of_domain_length():
    with pytest.raises(CanonicalEncodingError, match="too large"):
        Reader(encode_uint(1 << 32)).blob()


def test_d3_reader_blob_still_reads_a_legal_one():
    assert Reader(Writer().blob(b"abc").bytes).blob() == b"abc"


def test_d4_quantizable_params_may_not_exceed_the_positions():
    with pytest.raises(ManifestError, match="quantizable_params"):
        Geometry(
            rows=2,
            columns=5,
            superblock_columns=256,
            group_weights=1,
            half_weights=1,
            quantizable_params=10**12,
        )


def test_d5_zero_width_write_may_not_carry_a_value():
    with pytest.raises(PlaneLayoutError, match="width"):
        BitWriter().write(1, 0)
    BitWriter().write(0, 0)  # the only legal zero-width write


def test_d6_q256_round_trip_is_symmetric_at_zero():
    with pytest.raises(GrammarError, match="positive"):
        q256_from_root(Fraction(0))
    with pytest.raises(GrammarError, match="positive"):
        root_from_q256(0)


def test_d7_a_terminal_without_a_region_carries_no_digest():
    """``sha256(zeros)`` is a look-valid digest of data that was never hashed."""
    geometry = make_geometry(rows=8, columns=32, superblock_columns=8)
    rates = bresenham_rate_schedule(root_from_q256(512), 32)
    spec = TerminalSpec("t", tuple(3 - rate for rate in rates))
    planes = build_planes(geometry, rates, ALPHABET_BLOB, DESCENDANT_BLOB)
    terminal = build_terminal(
        geometry, rates, spec, planes, len(ALPHABET_BLOB), len(DESCENDANT_BLOB)
    )
    assert terminal.payload_digest == ZERO_DIGEST
    assert (
        terminal.payload_digest
        != hashlib.sha256(bytes(terminal.exact_bytes)).digest()
    )


def test_d8_release_extent_and_terminal_slice_may_not_disagree():
    geometry = make_geometry(rows=8, columns=32, superblock_columns=8)
    rates = bresenham_rate_schedule(root_from_q256(512), 32)
    spec = TerminalSpec("t", tuple(3 - rate for rate in rates), released_positions=4)
    with pytest.raises(PlaneLayoutError, match="released"):
        build_planes(
            geometry,
            rates,
            ALPHABET_BLOB,
            DESCENDANT_BLOB,
            max_released=8,
            spec=spec,
        )
    # Agreement, and the one-sided calls, are unchanged.
    build_planes(
        geometry, rates, ALPHABET_BLOB, DESCENDANT_BLOB, max_released=4, spec=spec
    )
    build_planes(geometry, rates, ALPHABET_BLOB, DESCENDANT_BLOB, max_released=8)
