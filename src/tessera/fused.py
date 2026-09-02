"""One wire per vLLM-fused module: the fused container and the shared global.

vLLM merges q/k/v and gate/up into one Linear, and a serving lane that decodes
a unit into a tile wants one blob per *module*.  A Tessera unit is per-role
(each has its own LUT table, its own global, its own trellis), and three
wires are not a wire -- so the module's blob is a **container** of role blobs
with the row offsets, nothing more.  The units inside are the units the
encoder wrote, byte for byte; the container is framing.

The one thing the roles must agree on at decode time is the global scale,
because a stock NVFP4 tile carries one ``weight_global_scale`` per module and
the lane's epilogue is one scalar.  ``shared_lut_global`` moves every role's
16-entry E4M3 table onto one global by an exact binade shift -- the same rule
``stock.share_global`` applies to materialised scale bytes, applied to the
table instead: a role's table times ``own / shared`` is exact when every
entry re-snaps to itself as float8_e4m3fn and none became NaN.  Candidates
are tried smallest multiplier (largest divisor) first, the choice vLLM makes
when shards disagree, and a group no candidate carries exactly is refused
with the roles named.  Nothing is rewritten on disk: the shift is derived
at load from the units as they are, so the digest the reader verified is the
digest the lane decodes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import torch

from .errors import GrammarError

__all__ = ["FUSED_MAGIC", "FusedMember", "pack_fused", "parse_fused", "shared_lut_global"]

FUSED_MAGIC = b"TSRFUSE1"
_VERSION = 1
_HEADER = struct.Struct("<8sBB")          # magic, version, member count
_MEMBER = struct.Struct("<HIQ")           # name length, rows, blob length


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
        if len(raw) > 0xFFFF or rows <= 0 or not blob:
            raise GrammarError(f"fused member {name!r}: bad name, rows {rows} or empty blob")
        out.append(_MEMBER.pack(len(raw), int(rows), len(blob)))
        out.append(raw)
    for _name, _rows, blob in members:
        out.append(blob)
    return b"".join(out)


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
        name = data[cursor:cursor + name_len].decode("utf-8")
        cursor += name_len
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


def _po2_exponent(value: float, what: str) -> int:
    if not (value > 0) or not math.isfinite(value):
        raise GrammarError(f"{what} must be a positive finite number, got {value!r}")
    exponent = math.log2(value)
    if exponent != int(exponent):
        raise GrammarError(f"{what} is not a power of two: {value!r}")
    return int(exponent)


def shared_lut_global(
    tables: "list[torch.Tensor]", globals_: "list[float]", names: "list[str] | None" = None
) -> "tuple[float, list[torch.Tensor]]":
    """One global for a fused group of LUT-plane units.

    ``tables[i]`` is unit ``i``'s E4M3 table as **uint8 bytes** (≤16 entries),
    ``globals_[i]`` its multiplier (``scale_global``).  Returns the shared
    multiplier and each unit's table moved onto it, as uint8 E4M3 bytes; every
    ``bytes -> value * shared`` equals the original ``value * own`` exactly.
    A group already on one global comes back untouched.
    """
    if not tables or len(tables) != len(globals_):
        raise GrammarError("shared_lut_global needs one table per global")
    names = names or [f"member{i}" for i in range(len(tables))]
    exps = [_po2_exponent(g, f"{n}'s scale_global") for g, n in zip(globals_, names)]
    if len(set(exps)) == 1:
        return float(globals_[0]), [t.to(torch.uint8) for t in tables]
    failures = {}
    for e in range(min(exps), max(exps) + 1):          # smallest multiplier first
        shared = float(2.0 ** e)
        moved_tables = []
        for table, own, name in zip(tables, globals_, names):
            ratio = float(own) / shared
            values = table.to(torch.uint8).view(torch.float8_e4m3fn).float()
            moved = values * ratio
            snapped = moved.to(torch.float8_e4m3fn)
            back = snapped.float()
            if not bool(torch.isfinite(back).all()) or not torch.equal(back, moved):
                failures[shared] = name
                break
            moved_tables.append(snapped.view(torch.uint8))
        else:
            return shared, moved_tables
    raise GrammarError(
        "no single scale_global carries every role's LUT table exactly: "
        + ", ".join(f"{g:g} fails on {n}" for g, n in failures.items())
    )
