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

__all__ = ["FUSED_MAGIC", "FusedMember", "pack_fused", "parse_fused",
           "shared_input_global_scale", "shared_lut_global"]

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


def _po2_exponent(value: float, what: str) -> int:
    if not (value > 0) or not math.isfinite(value):
        raise GrammarError(f"{what} must be a positive finite number, got {value!r}")
    exponent = math.log2(value)
    if exponent != int(exponent):
        raise GrammarError(f"{what} is not a power of two: {value!r}")
    return int(exponent)


#: The E4M3FN bytes the fused decoder can read: positive **normals**.  The
#: kernel turns a scale byte into a scale by field arithmetic,
#: ``2^(e - 7) * (1 + m / 8)``, which is the value only when the exponent field
#: is non-zero -- a subnormal byte encodes ``2^-6 * (m / 8)`` and a byte with
#: ``e = 15`` is NaN.  ``0x00`` (zero) and ``0x7F`` (NaN) are outside the range
#: for the same reason, and every negative byte is: a scale is positive.
_LUT_BYTE_MIN, _LUT_BYTE_MAX = 0x08, 0x7E


def _check_lut_bytes(table: torch.Tensor, name: str, what: str) -> torch.Tensor:
    """Refuse a LUT scale byte the fused decoder's field arithmetic misreads."""
    byte = table.to(torch.uint8)
    bad = (byte < _LUT_BYTE_MIN) | (byte > _LUT_BYTE_MAX)
    if bool(bad.any()):
        offenders = ", ".join(f"0x{int(b):02x}" for b in byte[bad][:8].tolist())
        raise GrammarError(
            f"{name}'s {what} carries {int(bad.sum())} scale byte(s) outside the "
            f"E4M3 normal range 0x{_LUT_BYTE_MIN:02x}..0x{_LUT_BYTE_MAX:02x} "
            f"({offenders}); the fused decoder reads a scale byte as "
            f"2^(e-7)*(1+m/8), which is not its value when the exponent field "
            f"is zero"
        )
    return byte


def shared_lut_global(
    tables: "list[torch.Tensor]", globals_: "list[float]", names: "list[str] | None" = None
) -> "tuple[float, list[torch.Tensor]]":
    """One global for a fused group of LUT-plane units.

    ``tables[i]`` is unit ``i``'s E4M3 table as **uint8 bytes** (≤16 entries),
    ``globals_[i]`` its multiplier (``scale_global``).  Returns the shared
    multiplier and each unit's table moved onto it, as uint8 E4M3 bytes; every
    ``bytes -> value * shared`` equals the original ``value * own`` exactly.
    A group already on one global comes back untouched.

    Exactness is necessary and not sufficient.  A binade shift can move a
    normal byte onto a **subnormal** one and round-trip perfectly -- ``0x18``
    at global 32 lands on ``0x01`` at global 1024, and ``0x01``'s float value
    is exactly what it was -- but the fused decoder does not read the byte's
    float value, it reads the byte's fields.  So the range is checked on every
    table this returns, moved or not, and a group that cannot be carried in
    normals is refused rather than served wrong.  Counted over the twenty-two
    ``.tessera`` artifacts on this box: fourteen carry a LUT plane and none has
    a byte outside the range, so this refuses nothing that exists.
    """
    if not tables or len(tables) != len(globals_):
        raise GrammarError("shared_lut_global needs one table per global")
    names = names or [f"member{i}" for i in range(len(tables))]
    exps = [_po2_exponent(g, f"{n}'s scale_global") for g, n in zip(globals_, names)]
    if len(set(exps)) == 1:
        return float(globals_[0]), [
            _check_lut_bytes(t, n, "LUT table") for t, n in zip(tables, names)
        ]
    for t, n in zip(tables, names):
        _check_lut_bytes(t, n, "LUT table")
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
            byte = snapped.view(torch.uint8)
            out_of_range = bool(
                ((byte < _LUT_BYTE_MIN) | (byte > _LUT_BYTE_MAX)).any()
            )
            if (not bool(torch.isfinite(back).all())
                    or not torch.equal(back, moved)
                    or out_of_range):
                failures[shared] = name
                break
            moved_tables.append(byte)
        else:
            return shared, moved_tables
    raise GrammarError(
        "no single scale_global carries every role's LUT table exactly, in "
        "E4M3 normals: "
        + ", ".join(f"{g:g} fails on {n}" for g, n in failures.items())
    )


#: One bf16 ULP, relative -- the derived bound between "one calibration,
#: spelled twice" and "two calibrations".  The NVFP4 route casts every A
#: tensor to bf16 before the native quantiser sees it
#: (``serving.nvfp4_route``; ``native_ops.native_fp4_quant`` refuses anything
#: but BF16/FP16), so a calibrated amax is an observation of a bf16 tensor:
#: any arithmetic that faithfully read the same tensor lands within one step
#: of its lattice, and ``input_global_scale`` (capacity over amax) inherits
#: the same relative bound.  A wider spread cannot be an observation of one
#: tensor.  ``torch.finfo(torch.bfloat16).eps`` = 2^-7, the lattice's
#: relative step -- a dtype's precision, not a chosen tolerance.
FUSED_INPUT_SCALE_ULP = float(torch.finfo(torch.bfloat16).eps)


def shared_input_global_scale(scales: "list[float]", names: "list[str] | None" = None) -> float:
    """One A-side static scale for a fused module: the MIN member scale.

    ``input_global_scale`` is capacity over amax.  The NVFP4 route hands the
    value unmodified to vLLM's native quantiser, which stores each group-16
    block scale as ``e4m3(block_amax / 6 * scale)`` clamped at 448 -- so a
    value too LARGE for the tensor's true amax saturates the stored block
    scale and every activation above ``448 * 6 / scale`` clips silently,
    while a value too small only spends block-scale precision.  A fused
    module's one GEMM quantises ONE input tensor for every member, so the
    module carries the scale of the largest calibrated amax: the minimum
    member scale.  (vLLM's stock compressed-tensors scheme reduces member
    scales with ``.max()`` -- the clipping direction -- but only over
    checkpoints whose calibrators already unified the members, warning when
    they differ: a degenerate no-op over equal values, not a join rule.
    PrismaQuant's ``unify_fused_sibling_input_global_scales`` states the same
    min-scale / max-amax rule at calibration time.)

    Members that diverge beyond ``FUSED_INPUT_SCALE_ULP`` are refused rather
    than joined: they are two calibrations (mixed draws, mixed policies, or a
    group that was never calibrated jointly), and a joined value would serve
    a distribution nobody measured.  The fix is a joint recalibration -- one
    amax over the members' shared input, which is what both calibrators
    already emit -- not a wider tolerance here.
    """
    if not scales:
        raise GrammarError("shared_input_global_scale needs at least one member scale")
    names = names or [f"member{i}" for i in range(len(scales))]
    if len(names) != len(scales):
        raise GrammarError("shared_input_global_scale needs one name per scale")
    values = []
    for name, value in zip(names, scales):
        value = float(value)
        if not (value > 0.0 and math.isfinite(value)):
            raise GrammarError(
                f"{name}: input_global_scale must be a finite positive scalar "
                f"(the route's load gate refuses anything else), got {value!r}")
        values.append(value)
    low, high = min(values), max(values)
    if high > low * (1.0 + FUSED_INPUT_SCALE_ULP):
        spread = ", ".join(f"{n}={v:g}" for n, v in zip(names, values))
        raise GrammarError(
            "fused members' input_global_scale values diverge beyond one bf16 "
            f"ULP (max/min = {high / low:.6g} > 1 + 2^-7): {spread}. The members "
            "of a fused module quantise ONE bf16 input tensor, so scales from "
            "one calibration agree to within one step of its lattice; this "
            "spread is two calibrations, and a joined value would serve a "
            "distribution nobody measured. Recalibrate the group jointly -- "
            "one amax over the shared input, the minimum scale -- rather than "
            "widening this bound.")
    return low
