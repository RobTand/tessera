"""Standard-library source framing; no serving, encoder or wire imports."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

PACKAGE_SOURCE_V2 = "tessera.package_source.v2"
ENCODER_SOURCE_V1 = "tessera.encoder_source.v1"


def source_profiles(records: Iterable[tuple[str, bytes]], *, legacy_profile: str,
                    legacy_prefix: bytes = b"") -> dict[str, str]:
    """Label the unchanged caller-ordered v1 and byte-name-ordered v2.

    v2 begins with ``tessera.package_source.v2\0``. Each record contains the
    unsigned eight-byte big-endian UTF-8 name length, name bytes, unsigned
    eight-byte big-endian content length and raw content. NUL is ordinary
    content, not a record separator. Selection and relative-name roots belong
    to the caller; neither is silently broadened here.
    """
    entries = list(records)
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError("source profiles require unique file names")
    legacy = hashlib.sha256(legacy_prefix)
    encoded = []
    for name, raw in entries:
        if not isinstance(name, str) or not isinstance(raw, bytes):
            raise TypeError("source profiles require UTF-8 names and byte contents")
        name_bytes = name.encode("utf-8")
        legacy.update(name_bytes + b"\0")
        legacy.update(raw)
        legacy.update(b"\0")
        encoded.append((name_bytes, raw))
    framed = hashlib.sha256(PACKAGE_SOURCE_V2.encode("ascii") + b"\0")
    for name_bytes, raw in sorted(encoded, key=lambda entry: entry[0]):
        framed.update(len(name_bytes).to_bytes(8, "big"))
        framed.update(name_bytes)
        framed.update(len(raw).to_bytes(8, "big"))
        framed.update(raw)
    return {legacy_profile: legacy.hexdigest(), PACKAGE_SOURCE_V2: framed.hexdigest()}
