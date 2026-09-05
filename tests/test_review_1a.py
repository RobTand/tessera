"""Every fix from the item-1a review, each proved against its own defect.

A fix without a test that fails on the *unfixed* behaviour is self-certification.
Each test here names the finding it closes and exercises the exact input that
used to be accepted.
"""

import dataclasses
import hashlib

import pytest

from conftest import ALPHABET_BLOB, DESCENDANT_BLOB, make_artifact, make_geometry
from tessera.container import parse, serialize, verify_plane_region
from tessera.errors import ManifestError, PlaneLayoutError, SchemaError
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.layout import TerminalSpec, build_plane_region, build_planes, build_terminal
from tessera.manifest import Manifest
from tessera.planes import (
    NORMATIVE_ELEMENT_BITS,
    BitOrder,
    CountGranularity,
    IndexDomain,
    PayloadDtype,
    PlaneDescriptor,
    PlaneKind,
    Storage,
)


def _retype(record, **changes):
    fields = dict(
        slot_id=record.slot_id,
        clip_exponent_code=record.clip_exponent_code,
        plane_elements=record.plane_elements,
        exact_bytes=record.exact_bytes,
        exact_bpp=record.exact_bpp,
        payload_digest=record.payload_digest,
    )
    fields.update(changes)
    return type(record)(**fields)


# --- F1: per-plane content digests are verified ---------------------------


def test_a_flipped_byte_in_a_complete_plane_is_caught(artifact):
    """`content_digest` used to be unverifiable: it held sha256(b"")."""
    manifest, region, _ = artifact
    tampered = bytearray(region)
    tampered[0] ^= 0x01  # first byte of the ALPHABET plane
    full = max(manifest.terminals, key=lambda t: t.exact_bytes)
    with pytest.raises(SchemaError):
        verify_plane_region(manifest, full, bytes(tampered))


def test_plane_digests_cover_real_content_not_a_placeholder(artifact):
    manifest, _, _ = artifact
    alphabet = manifest.plane(PlaneKind.ALPHABET)
    assert alphabet.content_digest == hashlib.sha256(ALPHABET_BLOB).digest()
    assert alphabet.content_digest != hashlib.sha256(b"").digest()


# --- F9: a truncated artifact has integrity at all ------------------------


def test_a_legal_truncation_still_round_trips(artifact):
    manifest, region, blob = artifact
    short = min(manifest.terminals, key=lambda t: t.exact_bytes)
    head = blob[: len(blob) - len(region) + short.exact_bytes]
    parsed = parse(head)
    assert parsed.terminal.slot_id == short.slot_id


def test_a_flipped_byte_inside_a_truncation_is_caught(artifact):
    """The headline case: before this review, truncations were unverified.

    The whole-artifact `payload_digest` covers only the untruncated bytes, so a
    truncation -- the format's entire point -- carried no integrity check.
    """
    manifest, region, blob = artifact
    short = min(manifest.terminals, key=lambda t: t.exact_bytes)
    prefix_len = len(blob) - len(region) + short.exact_bytes
    head = bytearray(blob[:prefix_len])
    head[-1] ^= 0x80
    with pytest.raises(SchemaError, match="payload digest"):
        parse(bytes(head))


# --- F8: a terminal must be a genuine prefix ------------------------------


def test_a_non_prefix_terminal_is_refused(artifact):
    """(full, empty, full) prices to a real byte count and used to be accepted."""
    manifest, _, _ = artifact
    full = max(manifest.terminals, key=lambda t: t.exact_bytes)
    counts = list(full.plane_elements)
    body = list(PlaneKind).index(PlaneKind.BODY)
    gapped = counts[:]
    gapped[body] = counts[body] // 2  # leave BODY incomplete...
    assert any(gapped[body + 1 :]), "later planes must be non-empty for this test"
    with pytest.raises(ManifestError, match="not a prefix"):
        Manifest(
            encoder_profile_id=manifest.encoder_profile_id,
            branch=manifest.branch,
            geometry=manifest.geometry,
            arrangement=manifest.arrangement,
            rates=manifest.rates,
            planes=manifest.planes,
            terminals=(_retype(full, slot_id="gapped", plane_elements=tuple(gapped)),),
            payload_digest=manifest.payload_digest,
        )


def test_an_over_claiming_terminal_is_refused_at_construction(artifact):
    """F2: the bound moved from the accountant into the validator."""
    manifest, _, _ = artifact
    full = max(manifest.terminals, key=lambda t: t.exact_bytes)
    counts = list(full.plane_elements)
    counts[0] += 10**6
    with pytest.raises(ManifestError, match="declares only"):
        Manifest(
            encoder_profile_id=manifest.encoder_profile_id,
            branch=manifest.branch,
            geometry=manifest.geometry,
            arrangement=manifest.arrangement,
            rates=manifest.rates,
            planes=manifest.planes,
            terminals=(_retype(full, plane_elements=tuple(counts)),),
            payload_digest=manifest.payload_digest,
        )


# --- F3: element_bits is bound to the plane kind --------------------------


def _descriptor(**changes):
    fields = dict(
        kind=PlaneKind.BODY,
        index_domain=IndexDomain.POSITION,
        storage=Storage.INLINE,
        element_bits=NORMATIVE_ELEMENT_BITS[PlaneKind.BODY],
        bit_order=BitOrder.MSB_FIRST,
        alignment_bytes=1,
        count_granularity=CountGranularity.WHOLE_PLANE,
        counts=(8,),
        restart_offsets=(0,),
        payload_dtype=PayloadDtype.RAW_BITS,
        content_digest=bytes(32),
    )
    fields.update(changes)
    return PlaneDescriptor(**fields)


def test_a_contradicting_element_width_is_refused():
    with pytest.raises(PlaneLayoutError, match="normative width"):
        _descriptor(element_bits=7)


def test_every_plane_kind_has_a_normative_width_but_one():
    """One plane's width is not the schema's to fix, and only one.

    INITIAL_STATE (schema minor 4) carries the trellis state a shard starts
    from, and that is ``window_bits`` under a window body and the
    convolutional code's memory under the coset trellis -- a property of the
    encoder profile, not of the schema.  A normative entry here would have to
    be wrong for one of the two bodies.  So the exception is deliberate, and
    this asserts it is *exactly one*: any other kind arriving without a width
    would be an unpriced plane two conforming decoders could disagree on.

    The width is still bound, twice, just not here: ``Manifest`` refuses a
    descriptor whose ``element_bits`` differs from the ``state_bits`` its
    shard record declares (and, under a window body, from ``window_bits``),
    and ``parse_unit_artifact`` re-checks it against the body once the profile
    id has resolved one -- the deferred validation the rate cap already uses.
    ``tests/test_slice_unit.py`` holds both refusals.
    """
    assert set(NORMATIVE_ELEMENT_BITS) == set(PlaneKind) - {PlaneKind.INITIAL_STATE}
    # ...and the absence really does leave the width free, rather than
    # defaulting to something a descriptor could contradict silently.
    for bits in (6, 14):
        assert _descriptor(kind=PlaneKind.INITIAL_STATE, element_bits=bits).element_bits == bits


# --- F5 / F6: derivable metadata must agree with what it derives from -----


def test_restart_offsets_must_be_the_prefix_sums_of_counts():
    with pytest.raises(PlaneLayoutError, match="prefix sums"):
        _descriptor(
            count_granularity=CountGranularity.PER_SUPERBLOCK,
            counts=(4, 4),
            restart_offsets=(0, 3),
        )


def test_a_zero_count_granule_may_repeat_an_offset():
    """Ascent must stay non-strict: an empty granule is legal."""
    descriptor = _descriptor(
        count_granularity=CountGranularity.PER_SUPERBLOCK,
        counts=(4, 0, 4),
        restart_offsets=(0, 4, 4),
    )
    assert descriptor.element_count == 8


def test_whole_plane_granularity_carries_exactly_one_count():
    with pytest.raises(PlaneLayoutError, match="expected exactly 1"):
        _descriptor(counts=(4, 4), restart_offsets=(0, 4))


# --- F4: padding is zero, so the encoding is canonical --------------------


def _aligned_artifact(alignment_bytes=64):
    """A fully-present terminal on a padded layout."""
    geometry = make_geometry()
    rates = bresenham_rate_schedule(root_from_q256(512), geometry.columns)
    payloads = {
        PlaneKind.ALPHABET: ALPHABET_BLOB,
        PlaneKind.DESCENDANT: DESCENDANT_BLOB,
    }
    planes = build_planes(
        geometry,
        rates,
        ALPHABET_BLOB,
        DESCENDANT_BLOB,
        alignment_bytes=alignment_bytes,
        max_released=4,
        payloads=payloads,
    )
    region = build_plane_region(planes, payloads)
    spec = TerminalSpec(
        "t-full",
        tuple(3 - rate for rate in rates),
        released_positions=4,
        with_scale_base=True,
        with_scale_refine=True,
        with_diagonals=True,
    )
    terminal = build_terminal(
        geometry,
        rates,
        spec,
        planes,
        len(ALPHABET_BLOB),
        len(DESCENDANT_BLOB),
        plane_region=region,
    )
    return planes, terminal, region


def _reseal(planes, terminal, region):
    """Re-digest everything over `region`.

    This is what a *second encoder with a different padding convention* would
    produce: internally consistent, byte-different from ours for identical
    logical content.  Every digest matches, so only an explicit canonicality
    rule can refuse it -- which is the whole point of finding F4.
    """
    sealed, offset = [], 0
    for descriptor in planes:
        total = descriptor.byte_length()
        chunk = region[offset : offset + total]
        sealed.append(
            dataclasses.replace(
                descriptor, content_digest=hashlib.sha256(chunk).digest()
            )
        )
        offset += total
    resealed = _retype(
        terminal,
        payload_digest=hashlib.sha256(region[: terminal.exact_bytes]).digest(),
    )
    return tuple(sealed), resealed


def test_nonzero_alignment_padding_is_refused():
    """Canonicality: identical content must not admit two byte strings."""
    planes, terminal, region = _aligned_artifact()
    content = len(ALPHABET_BLOB)
    assert planes[0].byte_length() > content, "need real padding for this test"

    tampered = bytearray(region)
    tampered[content] = 0xFF  # pad the ALPHABET plane with 0xFF instead of 0x00
    sealed_planes, sealed_terminal = _reseal(planes, terminal, bytes(tampered))
    manifest = _manifest_for(sealed_planes, sealed_terminal)

    # every digest agrees with these bytes ...
    assert hashlib.sha256(bytes(tampered)).digest() == sealed_terminal.payload_digest
    # ... and it is still refused, because padding is not the encoder's to choose
    with pytest.raises(PlaneLayoutError, match="padding"):
        verify_plane_region(manifest, sealed_terminal, bytes(tampered))


def test_the_canonical_region_passes_the_same_check():
    planes, terminal, region = _aligned_artifact()
    verify_plane_region(_manifest_for(planes, terminal), terminal, region)


def _manifest_for(planes, terminal):
    geometry = make_geometry()
    rates = bresenham_rate_schedule(root_from_q256(512), geometry.columns)
    reference = parse(make_artifact()[2]).manifest
    return Manifest(
        encoder_profile_id=reference.encoder_profile_id,
        branch=reference.branch,
        geometry=geometry,
        arrangement=reference.arrangement,
        rates=rates,
        planes=planes,
        terminals=(terminal,),
        payload_digest=bytes(32),
    )


# --- F7: the write side is as fail-closed as the read side ----------------


def test_serialize_refuses_a_region_that_matches_no_terminal(artifact):
    manifest, region, _ = artifact
    padded = region + b"\x00"
    with pytest.raises(Exception):
        serialize(manifest, padded)


# --- F10: the reader's integer domain must equal the writer's -------------


def test_a_varint_outside_the_64_bit_domain_is_refused():
    """The decoder used to accept what the encoder cannot produce.

    Ten bytes -- nine continuations then a final group at shift 63 -- decode to
    about 2**70. `encode_uint` refuses that value, so the byte string is outside
    the image of every conforming encoder, and a content-addressed format must
    not accept it.
    """
    from tessera.canonical import decode_uint, encode_uint
    from tessera.errors import CanonicalEncodingError

    blob = bytes([0x80] * 9 + [0x7F])
    with pytest.raises(CanonicalEncodingError, match="64-bit domain"):
        decode_uint(blob)
    with pytest.raises(CanonicalEncodingError, match="64-bit domain"):
        encode_uint(1 << 64)


def test_the_largest_legal_uint_still_round_trips():
    from tessera.canonical import decode_uint, encode_uint

    largest = (1 << 64) - 1
    assert decode_uint(encode_uint(largest))[0] == largest


def test_signed_encoding_refuses_values_outside_its_domain():
    """`_zigzag`'s sign mask is only valid inside the signed 64-bit domain."""
    from tessera.canonical import CanonicalEncodingError, Writer

    writer = Writer()
    writer.sint(-(1 << 63))  # the boundary is legal
    with pytest.raises(CanonicalEncodingError, match="signed 64-bit domain"):
        writer.sint(-(1 << 70))


def test_the_parser_has_no_switch_that_turns_its_digests_off():
    """`parse(data, verify_payload_digest=False)` disabled every plane digest
    and the padding-canonicality check in one keyword, and no caller in the
    tree ever passed it.  A fail-closed parser with an unused off switch is a
    way for one reader to accept bytes no other reader would."""
    _manifest, _region, data = make_artifact()
    assert parse(data)  # the one way in still works
    with pytest.raises(TypeError):
        parse(data, verify_payload_digest=False)
