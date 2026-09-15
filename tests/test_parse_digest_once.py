"""``container.parse`` hashes the plane region once (tessera#503).

The region is the bulk of every wire -- hundreds of megabytes per routed
expert role on GLM-5.3-Flash -- and ``parse`` used to sha256 it twice on a
complete artifact: once inside ``verify_plane_region`` against the terminal's
digest, and again against the manifest's.  Both comparisons stay, with both
refusal messages; only the second pass over the bytes goes.  The truncation
and digest-refusal cases themselves live in ``tests/test_review_1a.py``.
"""

import dataclasses
import hashlib

import pytest

from tessera import container
from tessera.container import HEADER_BYTES, MAGIC, SCHEMA_MAJOR, parse
from tessera.errors import SchemaError


_SHA256 = hashlib.sha256  # ``container.hashlib`` is this module: patch a copy


class _CountingSha256:
    """``hashlib.sha256`` that records the length of every whole input."""

    def __init__(self):
        self.lengths = []

    def __call__(self, data=b""):
        self.lengths.append(len(data))
        return _SHA256(data)


def test_a_complete_artifact_hashes_its_region_once(artifact, monkeypatch):
    manifest, region, blob = artifact
    counter = _CountingSha256()
    monkeypatch.setattr(container.hashlib, "sha256", counter)
    parsed = parse(blob)
    assert parsed.plane_region == region
    assert counter.lengths.count(len(region)) == 1, counter.lengths


def test_a_truncation_hashes_its_prefix_once(artifact, monkeypatch):
    manifest, region, blob = artifact
    short = min(manifest.terminals, key=lambda t: t.exact_bytes)
    head = blob[: len(blob) - len(region) + short.exact_bytes]
    counter = _CountingSha256()
    monkeypatch.setattr(container.hashlib, "sha256", counter)
    parse(head)
    assert counter.lengths.count(short.exact_bytes) == 1, counter.lengths


def test_the_manifest_digest_is_still_compared_on_a_complete_artifact(artifact):
    """A terminal digest that matches does not stand in for the manifest's:
    the reused digest must still be compared, with its own message."""
    manifest, region, _ = artifact
    forged = dataclasses.replace(manifest, payload_digest=bytes(32))
    minor = forged.schema_minor
    manifest_bytes = forged.encode(minor)
    header = container._HEADER.pack(
        MAGIC, SCHEMA_MAJOR, minor, HEADER_BYTES, len(manifest_bytes), len(region)
    )
    with pytest.raises(SchemaError, match="payload digest mismatch on a complete artifact"):
        parse(header + manifest_bytes + region)
