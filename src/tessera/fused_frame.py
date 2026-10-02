"""Canonical fused framing, shared by native loaders and passive receipts."""
from __future__ import annotations
from dataclasses import dataclass
import struct
from .errors import GrammarError

FUSED_MAGIC = b"TSRFUSE1"
_VERSION = 1
_HEADER = struct.Struct("<8sBB")          # magic, version, member count
_MEMBER = struct.Struct("<HIQ")           # name length, rows, blob length

#: Fixed bytes of the container header.
HEADER_BYTES = _HEADER.size
#: Fixed bytes of one member row, before its name bytes.
MEMBER_ROW_BYTES = _MEMBER.size


def frame_bytes(member_names: "list[str]") -> int:
    """Framing bytes of a container over these roles, blobs excluded.

    ``len(pack_fused(members)) == frame_bytes(names) + sum(blob lengths)``.
    """
    return HEADER_BYTES + sum(
        MEMBER_ROW_BYTES + len(name.encode("utf-8")) for name in member_names
    )


@dataclass(frozen=True)
class FusedMember:
    name: str
    rows: int
    blob: bytes


def pack_fused(members: "list[tuple[str, int, bytes]]") -> bytes:
    """Frame role blobs, in the row order the fused module stacks them."""
    if not members:
        raise GrammarError("a fused container needs at least one member")
    if len(members) > 255:
        raise GrammarError("a fused container holds at most 255 members")
    names = [m[0] for m in members]
    if len(set(names)) != len(names):
        raise GrammarError(f"duplicate role names in a fused container: {names}")
    out = [_HEADER.pack(FUSED_MAGIC, _VERSION, len(members))]
    for name, rows, blob in members:
        raw = name.encode("utf-8")
        _check_member(name, len(raw), rows, len(blob))
        out.append(_MEMBER.pack(len(raw), int(rows), len(blob)))
        out.append(raw)
    for _name, _rows, blob in members:
        out.append(blob)
    return b"".join(out)


def _check_member(name: str, name_bytes: int, rows: int, blob_len: int) -> None:
    """The member domain, stated once for the writer and the reader.

    The reader used to check the member table and the blobs but not the
    header fields themselves, so a member the writer would have refused
    decoded and failed a step later in somebody else's words: a truncated
    name ran the cursor past the end and the framing check then reported
    ``"-12 trailing bytes"``.  A reader that cannot refuse what its writer
    cannot write is not fail-closed (AGENTS.md 4, 5).
    """
    if name_bytes > 0xFFFF:
        raise GrammarError(
            f"fused member {name!r}: name is {name_bytes} bytes, at most 65535"
        )
    if rows <= 0:
        raise GrammarError(f"fused member {name!r}: rows must be positive, got {rows}")
    if blob_len <= 0:
        raise GrammarError(f"fused member {name!r}: empty blob")


def parse_fused(data: bytes) -> "list[FusedMember]":
    """The members of a fused container, fail-closed on any framing error."""
    if len(data) < _HEADER.size:
        raise GrammarError("fused container shorter than its header")
    magic, version, count = _HEADER.unpack_from(data)
    if magic != FUSED_MAGIC or version != _VERSION:
        raise GrammarError("not a Tessera fused container (v1)")
    if count == 0:
        raise GrammarError("a fused container with no members")
    cursor = _HEADER.size
    heads = []
    for _ in range(count):
        if cursor + _MEMBER.size > len(data):
            raise GrammarError("truncated fused member table")
        name_len, rows, blob_len = _MEMBER.unpack_from(data, cursor)
        cursor += _MEMBER.size
        if cursor + name_len > len(data):
            raise GrammarError(
                f"truncated fused member name: {name_len} byte(s) declared, "
                f"{len(data) - cursor} left"
            )
        raw = data[cursor:cursor + name_len]
        try:
            name = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GrammarError(
                f"fused member name is not UTF-8: {raw!r}"
            ) from exc
        cursor += name_len
        _check_member(name, name_len, rows, blob_len)
        heads.append((name, rows, blob_len))
    members = []
    for name, rows, blob_len in heads:
        blob = data[cursor:cursor + blob_len]
        if len(blob) != blob_len:
            raise GrammarError(f"fused member {name!r}: truncated blob")
        cursor += blob_len
        members.append(FusedMember(name, rows, bytes(blob)))
    if cursor != len(data):
        raise GrammarError(f"{len(data) - cursor} trailing bytes after the last fused member")
    return members
