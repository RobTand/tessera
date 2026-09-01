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

from .errors import PlaneLayoutError, SchemaError, TruncationError
from .footprint import account_terminal, plane_region_bytes
from .exact import bits_to_bytes
from .manifest import Manifest, TerminalRecord
from .planes import CANONICAL_PLANE_ORDER, PlaneDescriptor, Storage

__all__ = [
    "MAGIC",
    "HEADER_BYTES",
    "SCHEMA_MAJOR",
    "SCHEMA_MINOR",
    "SCHEMA_MINORS_READ",
    "serialize",
    "parse",
    "ParsedArtifact",
    "plane_ranges",
    "verify_plane_region",
]

MAGIC = b"\x89TESSERA"
HEADER_BYTES = 24
SCHEMA_MAJOR = 1
#: Minor 1 (2026-09-01) appends the trellis span and the scale-plane record to
#: the manifest.  ``serialize`` writes the lowest minor a manifest needs, so a
#: span-1 S6b unit is still a minor-0 artifact byte for byte; ``parse`` reads
#: both.  A minor bump, not a major one, because the plane region's grammar is
#: unchanged and every minor-0 artifact means exactly what it meant.
SCHEMA_MINOR = 1
SCHEMA_MINORS_READ = (0, 1)

_HEADER = struct.Struct("<8sHHIII")


def serialize(manifest: Manifest, plane_region: bytes) -> bytes:
    """Emit a full (untruncated) Tessera artifact."""
    minor = manifest.schema_minor
    manifest_bytes = manifest.encode(minor)
    digest = hashlib.sha256(plane_region).digest()
    if digest != manifest.payload_digest:
        raise SchemaError(
            "payload_digest does not match the supplied plane region"
        )
    full = max(manifest.terminals, key=lambda terminal: terminal.exact_bytes)
    account_terminal(
        manifest, full, side_bytes=0, physical_bytes=len(plane_region)
    )
    verify_plane_region(manifest, full, plane_region)
    header = _HEADER.pack(
        MAGIC,
        SCHEMA_MAJOR,
        minor,
        HEADER_BYTES,
        len(manifest_bytes),
        len(plane_region),
    )
    return header + manifest_bytes + plane_region


def plane_ranges(
    manifest: Manifest, terminal: TerminalRecord
) -> "list[tuple[PlaneDescriptor, int, int, int]]":
    """`(descriptor, offset, content_bytes, total_bytes)` per plane, in order.

    The canonical plane order is the byte order, so a terminal's region is the
    concatenation of each plane's truncated extent.  `content_bytes` excludes
    alignment padding; `total_bytes` includes it.
    """
    order = {kind: index for index, kind in enumerate(CANONICAL_PLANE_ORDER)}
    ranges, offset = [], 0
    for descriptor in manifest.planes:
        if descriptor.storage is Storage.REFERENCE:
            continue
        count = terminal.plane_elements[order[descriptor.kind]]
        total = descriptor.byte_length(count)
        content = bits_to_bytes(count * descriptor.element_bits)
        ranges.append((descriptor, offset, content, total))
        offset += total
    return ranges


def verify_plane_region(
    manifest: Manifest, terminal: TerminalRecord, plane_region: bytes
) -> None:
    """Check every integrity claim the manifest makes about these bytes.

    Three claims, none of which was checked before this review:

    * the terminal's own `payload_digest` over its whole byte prefix, which is
      the only integrity a *truncated* artifact has (F9);
    * each fully-present plane's `content_digest` (F1);
    * that alignment padding is zero, so the encoding is canonical and the
      slack is not a covert channel (F4).
    """
    if hashlib.sha256(plane_region).digest() != terminal.payload_digest:
        raise SchemaError(
            f"terminal {terminal.slot_id!r}: plane-region bytes do not match "
            "the declared payload digest"
        )
    order = {kind: index for index, kind in enumerate(CANONICAL_PLANE_ORDER)}
    for descriptor, offset, content, total in plane_ranges(manifest, terminal):
        chunk = plane_region[offset : offset + total]
        if len(chunk) != total:
            raise TruncationError(
                f"{descriptor.kind.name}: region holds {len(chunk)} of {total} bytes"
            )
        if any(chunk[content:]):
            raise PlaneLayoutError(
                f"{descriptor.kind.name}: non-zero alignment padding; the "
                "encoding must be canonical"
            )
        # Sub-byte slack too: a 1-bit plane whose count is not a multiple of 8
        # leaves pad bits inside the final content byte.  MSB-first packing puts
        # them in the low bits.  Unconstrained, they are the same canonicality
        # hole as the alignment bytes above, one byte earlier.
        bits = count_bits = terminal.plane_elements[
            order[descriptor.kind]
        ] * descriptor.element_bits
        slack = (-bits) % 8
        if slack and content:
            if chunk[content - 1] & ((1 << slack) - 1):
                raise PlaneLayoutError(
                    f"{descriptor.kind.name}: non-zero pad bits in the final "
                    "content byte; the encoding must be canonical"
                )
        count = terminal.plane_elements[order[descriptor.kind]]
        if count == descriptor.element_count:
            if hashlib.sha256(chunk).digest() != descriptor.content_digest:
                raise SchemaError(
                    f"{descriptor.kind.name}: bytes do not match the plane's "
                    "declared content digest"
                )


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
    if major != SCHEMA_MAJOR or minor not in SCHEMA_MINORS_READ:
        raise SchemaError(f"unsupported schema version {major}.{minor}")

    manifest_end = HEADER_BYTES + manifest_bytes
    if len(data) < manifest_end:
        raise SchemaError("truncated manifest: the manifest is never truncatable")
    manifest = Manifest.decode(data[HEADER_BYTES:manifest_end], schema_minor=minor)
    if manifest.schema_minor > minor:
        raise SchemaError(
            f"header declares schema minor {minor} but the manifest needs "
            f"{manifest.schema_minor}"
        )

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

    if verify_payload_digest:
        verify_plane_region(manifest, terminal, plane_region)
        if len(plane_region) == region_bytes:
            if hashlib.sha256(plane_region).digest() != manifest.payload_digest:
                raise SchemaError("payload digest mismatch on a complete artifact")

    return ParsedArtifact(
        manifest=manifest,
        terminal=terminal,
        plane_region=plane_region,
        side_bytes=side_bytes,
    )
