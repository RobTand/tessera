"""S9 wire contract: parse starts from serialized bytes only, and fails closed."""

import hashlib

import pytest

from tessera.container import HEADER_BYTES, MAGIC, parse, serialize
from tessera.errors import (
    CanonicalEncodingError,
    ManifestError,
    SchemaError,
    TruncationError,
)
from tessera.manifest import Manifest

from conftest import make_artifact


def test_round_trip_from_bytes_only(artifact):
    manifest, plane_region, data = artifact
    parsed = parse(data)
    assert parsed.manifest.encode() == manifest.encode()
    assert parsed.plane_region == plane_region
    assert len(parsed.terminal_id) == 32


def test_magic_is_not_valid_ascii():
    """Disjointness from the legacy text name grammar, by construction."""
    assert MAGIC[0] > 0x7F
    with pytest.raises(UnicodeDecodeError):
        MAGIC.decode("ascii")


def test_foreign_magic_is_rejected(artifact):
    _, _, data = artifact
    with pytest.raises(SchemaError, match="foreign magic"):
        parse(b"TCQ_XXXX" + data[8:])


def test_truncated_header_is_rejected(artifact):
    _, _, data = artifact
    with pytest.raises(SchemaError):
        parse(data[: HEADER_BYTES - 1])


def test_legal_truncation_at_a_declared_terminal(artifact):
    manifest, _, data = artifact
    side = HEADER_BYTES + len(manifest.encode())
    for terminal in manifest.terminals:
        truncated = data[: side + terminal.exact_bytes]
        parsed = parse(truncated)
        assert parsed.terminal.slot_id == terminal.slot_id
        assert len(parsed.plane_region) == terminal.exact_bytes


def test_arbitrary_byte_prefix_is_not_a_terminal(artifact):
    """The core S9 rule: only declared terminals are legal truncations."""
    manifest, _, data = artifact
    side = HEADER_BYTES + len(manifest.encode())
    legal = {t.exact_bytes for t in manifest.terminals}
    rejected = 0
    for length in range(0, max(legal) + 1):
        if length in legal:
            continue
        with pytest.raises(TruncationError):
            parse(data[: side + length])
        rejected += 1
    assert rejected > 0


def test_manifest_is_never_truncatable(artifact):
    manifest, _, data = artifact
    manifest_bytes = len(manifest.encode())
    with pytest.raises(SchemaError, match="truncated manifest"):
        parse(data[: HEADER_BYTES + manifest_bytes - 1])


def test_trailing_bytes_in_manifest_are_rejected(artifact):
    manifest, _, _ = artifact
    with pytest.raises(CanonicalEncodingError, match="trailing"):
        Manifest.decode(manifest.encode() + b"\x00")


def test_payload_digest_mismatch_is_rejected(artifact):
    manifest, plane_region, data = artifact
    corrupted = bytearray(data)
    corrupted[-1] ^= 0xFF
    with pytest.raises(SchemaError, match="payload digest"):
        parse(bytes(corrupted))


def test_serialize_refuses_a_mismatched_payload_digest(artifact):
    manifest, plane_region, _ = artifact
    with pytest.raises(SchemaError):
        serialize(manifest, plane_region + b"\x00")


def test_canonical_manifest_encoding_is_stable(artifact):
    manifest, _, _ = artifact
    once = manifest.encode()
    assert Manifest.decode(once).encode() == once
    assert manifest.manifest_digest() == Manifest.decode(once).manifest_digest()


def test_terminal_ids_are_distinct_and_content_addressed(artifact):
    manifest, _, _ = artifact
    ids = manifest.terminal_ids()
    assert len(set(ids.values())) == len(ids)


def test_terminal_id_binds_the_branch(artifact):
    """Two encodes with the same counts under different branches differ."""
    manifest, _, _ = artifact
    other, _, _ = make_artifact(q256=256)
    terminal = manifest.terminals[0]
    assert terminal.terminal_id(
        manifest.branch, manifest.encoder_profile_id
    ) != terminal.terminal_id(other.branch, other.encoder_profile_id)


def test_duplicate_terminal_sizes_are_rejected(artifact):
    """A truncation length must identify exactly one terminal."""
    manifest, _, _ = artifact
    clash = manifest.terminals[0]
    doubled = manifest.terminals + (
        type(clash)(
            slot_id="clash",
            clip_exponent_code=clash.clip_exponent_code,
            plane_elements=clash.plane_elements,
            exact_bytes=clash.exact_bytes,
            exact_bpp=clash.exact_bpp,
        ),
    )
    with pytest.raises(ManifestError, match="same exact_bytes"):
        type(manifest)(
            encoder_profile_id=manifest.encoder_profile_id,
            branch=manifest.branch,
            geometry=manifest.geometry,
            arrangement=manifest.arrangement,
            rates=manifest.rates,
            planes=manifest.planes,
            terminals=doubled,
            payload_digest=manifest.payload_digest,
        )
