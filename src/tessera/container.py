"""The Tessera container: header, manifest, plane region (doc S9).

Layout, little-endian:

===== ====== =========================================================
off   size   field
===== ====== =========================================================
0     8      magic ``\\x89TESSERA``
8     2      schema major
10    2      schema minor
12    4      header bytes (== 24)
16    4      manifest bytes
20    4      plane-region bytes
24    ...    canonical manifest
...   ...    plane region
===== ====== =========================================================

Schema 1a decision D7 -- the magic's leading ``0x89`` (the PNG trick) has the
high bit set, so a Tessera artifact is never valid ASCII or UTF-8 at byte 0.
The legacy ``TCQ_*`` name grammar is a pure-ASCII text language, so the two
languages are disjoint *by construction* rather than by a lookup table, and the
legacy two-tuple parser cannot silently accept a Tessera artifact.

Truncation is fail-closed: a byte length that does not match a declared
terminal's plane-region size is rejected.  Per-superblock quota-boundary
truncations within a plane are legal and enumerate their own ``terminal_id``;
arbitrary interleaved byte-prefixes are not terminals.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from .errors import SchemaError, TruncationError
from .footprint import account_terminal, plane_region_bytes
from .manifest import Manifest, TerminalRecord

__all__ = [
    "MAGIC",
    "HEADER_BYTES",
    "SCHEMA_MAJOR",
    "SCHEMA_MINOR",
    "serialize",
    "parse",
    "ParsedArtifact",
]

MAGIC = b"\x89TESSERA"
HEADER_BYTES = 24
SCHEMA_MAJOR = 1
SCHEMA_MINOR = 0

_HEADER = struct.Struct("<8sHHIII")


def serialize(manifest: Manifest, plane_region: bytes) -> bytes:
    """Emit a full (untruncated) Tessera artifact."""
    manifest_bytes = manifest.encode()
    digest = hashlib.sha256(plane_region).digest()
    if digest != manifest.payload_digest:
        raise SchemaError(
            "payload_digest does not match the supplied plane region"
        )
    header = _HEADER.pack(
        MAGIC,
        SCHEMA_MAJOR,
        SCHEMA_MINOR,
        HEADER_BYTES,
        len(manifest_bytes),
        len(plane_region),
    )
    return header + manifest_bytes + plane_region


@dataclass(frozen=True)
class ParsedArtifact:
    manifest: Manifest
    terminal: TerminalRecord
    plane_region: bytes
    side_bytes: int

    @property
    def terminal_id(self) -> bytes:
        return self.terminal.terminal_id(
            self.manifest.branch, self.manifest.encoder_profile_id
        )


def parse(data: bytes, verify_payload_digest: bool = True) -> ParsedArtifact:
    """Parse an artifact from **bytes only**, fail-closed.

    `verify_payload_digest` applies to the untruncated artifact; a legal
    truncation necessarily has a different payload digest, so it is checked
    only when the plane region is complete.
    """
    if len(data) < HEADER_BYTES:
        raise SchemaError(f"artifact shorter than a {HEADER_BYTES}-byte header")
    magic, major, minor, header_bytes, manifest_bytes, region_bytes = _HEADER.unpack(
        data[:HEADER_BYTES]
    )
    if magic != MAGIC:
        raise SchemaError(f"foreign magic {magic!r}: not a Tessera artifact")
    if header_bytes != HEADER_BYTES:
        raise SchemaError(f"declared header size {header_bytes} != {HEADER_BYTES}")
    if (major, minor) != (SCHEMA_MAJOR, SCHEMA_MINOR):
        raise SchemaError(f"unsupported schema version {major}.{minor}")

    manifest_end = HEADER_BYTES + manifest_bytes
    if len(data) < manifest_end:
        raise SchemaError("truncated manifest: the manifest is never truncatable")
    manifest = Manifest.decode(data[HEADER_BYTES:manifest_end])

    plane_region = data[manifest_end:]
    if len(plane_region) != region_bytes:
        # A legal truncation shortens the region; the header still declares the
        # full size, so this is expected and is resolved against the terminals.
        if len(plane_region) > region_bytes:
            raise SchemaError("plane region longer than the header declares")

    matches = [
        terminal
        for terminal in manifest.terminals
        if terminal.exact_bytes == len(plane_region)
    ]
    if not matches:
        legal = sorted(t.exact_bytes for t in manifest.terminals)
        raise TruncationError(
            f"{len(plane_region)} plane-region bytes match no declared terminal; "
            f"legal terminal sizes are {legal}. Arbitrary byte-prefixes are not "
            "terminals (doc S9)."
        )
    terminal = matches[0]

    side_bytes = HEADER_BYTES + manifest_bytes
    account_terminal(
        manifest, terminal, side_bytes=side_bytes, physical_bytes=len(plane_region)
    )

    if verify_payload_digest and len(plane_region) == region_bytes:
        if hashlib.sha256(plane_region).digest() != manifest.payload_digest:
            raise SchemaError("payload digest mismatch on a complete artifact")

    return ParsedArtifact(
        manifest=manifest,
        terminal=terminal,
        plane_region=plane_region,
        side_bytes=side_bytes,
    )
