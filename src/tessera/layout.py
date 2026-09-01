"""Plane layout construction from declared parameters.

This is the serializer half of build item 1b.  It makes **no coding
decisions**: it takes a declared geometry and a terminal spec and computes
element counts, plane extents, and exact bytes.  Choosing anchors, completion
placement, or release positions is the encoder's job, and the encoder is gated
(arm 2's minimal measurement encoder is the first gated-work request).

Element units per plane, so that one uniform `element_bits` describes each
plane even though per-column rates differ:

===============  ============  ==========================================
plane            element        element_bits
===============  ============  ==========================================
ALPHABET          byte           8
DESCENDANT        byte           8
BODY              bit            1   (count = sum over columns of R * rows)
SCALE_BASE        group          8   (one E8M0 byte per 32 weights)
COMPLETION        bit            1   (count = sum over columns of c * rows)
DIAG_SU           in-channel    16
DIAG_SV           out-channel   16
SCALE_REFINE      half-block     4   (one nibble per 16 weights)
RELEASE           position       4   (a released position stores 16 codes)
===============  ============  ==========================================
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

from .errors import GrammarError, PlaneLayoutError
from .exact import bits_to_bytes
from .grammar import C_FULL_BITS, RELEASE_BITS, completion_capacity
from .manifest import Geometry, TerminalRecord
from .planes import (
    CANONICAL_PLANE_ORDER,
    NORMATIVE_ELEMENT_BITS,
    BitOrder,
    CountGranularity,
    IndexDomain,
    PayloadDtype,
    PlaneDescriptor,
    PlaneKind,
    Storage,
)

__all__ = ["TerminalSpec", "build_planes", "build_terminal", "ZERO_DIGEST"]

ZERO_DIGEST = bytes(32)


_INDEX_DOMAIN = {
    PlaneKind.ALPHABET: IndexDomain.WHOLE_UNIT,
    PlaneKind.DESCENDANT: IndexDomain.WHOLE_UNIT,
    PlaneKind.BODY: IndexDomain.POSITION,
    PlaneKind.SCALE_BASE: IndexDomain.BLOCK,
    PlaneKind.COMPLETION: IndexDomain.POSITION,
    PlaneKind.DIAG_SU: IndexDomain.AXIS_IN,
    PlaneKind.DIAG_SV: IndexDomain.AXIS_OUT,
    PlaneKind.SCALE_REFINE: IndexDomain.HALF_BLOCK,
    PlaneKind.RELEASE: IndexDomain.POSITION,
}

_DTYPE = {
    PlaneKind.ALPHABET: PayloadDtype.RAW_BITS,
    PlaneKind.DESCENDANT: PayloadDtype.RAW_BITS,
    PlaneKind.BODY: PayloadDtype.RAW_BITS,
    PlaneKind.SCALE_BASE: PayloadDtype.E8M0,
    PlaneKind.COMPLETION: PayloadDtype.RAW_BITS,
    PlaneKind.DIAG_SU: PayloadDtype.FP16,
    PlaneKind.DIAG_SV: PayloadDtype.FP16,
    PlaneKind.SCALE_REFINE: PayloadDtype.RAW_BITS,
    PlaneKind.RELEASE: PayloadDtype.UINT,
}


@dataclass(frozen=True)
class TerminalSpec:
    """A declared terminal slot: which planes, at what depth."""

    slot_id: str
    completion_bits: tuple[int, ...]  # per column, 0 <= c <= 3 - R
    released_positions: int = 0
    with_scale_base: bool = True
    with_scale_refine: bool = False
    with_diagonals: bool = False
    clip_exponent_code: int = 0


def _counts_for(
    kind: PlaneKind,
    geometry: Geometry,
    rates: tuple[int, ...],
    spec: TerminalSpec | None,
    alphabet_bytes: int,
    descendant_bytes: int,
    max_released: int = 0,
    cap: int = C_FULL_BITS,
    arity: int = 1,
) -> int:
    # ``geometry`` is declared in weight space.  BODY and COMPLETION are
    # per-CODE planes, and a code covers ``arity`` consecutive rows, so they are
    # sized in *steps*; the scale planes and DIAG_SV stay in weight space.
    # Sizing a per-code plane in weight space is not a rounding error -- it
    # over-declares the plane by exactly ``arity``, and then every plane offset
    # after it is wrong.
    if arity < 1 or geometry.rows % arity:
        raise GrammarError(
            f"geometry declares {geometry.rows} rows, not a whole number of "
            f"arity-{arity} tuples"
        )
    rows = geometry.rows
    steps = rows // arity
    positions = geometry.positions
    if kind is PlaneKind.ALPHABET:
        return alphabet_bytes
    if kind is PlaneKind.DESCENDANT:
        return descendant_bytes
    if kind is PlaneKind.BODY:
        return sum(rates) * steps
    if kind is PlaneKind.SCALE_BASE:
        if spec is not None and not spec.with_scale_base:
            return 0
        return positions // geometry.group_weights
    if kind is PlaneKind.COMPLETION:
        if spec is None:
            return sum(completion_capacity(rate, cap) for rate in rates) * steps
        return sum(spec.completion_bits) * steps
    if kind in (PlaneKind.DIAG_SU, PlaneKind.DIAG_SV):
        if spec is not None and not spec.with_diagonals:
            return 0
        return geometry.columns if kind is PlaneKind.DIAG_SU else rows
    if kind is PlaneKind.SCALE_REFINE:
        if spec is not None and not spec.with_scale_refine:
            return 0
        return positions // geometry.half_weights
    if kind is PlaneKind.RELEASE:
        return max_released if spec is None else spec.released_positions
    raise GrammarError(f"unhandled plane kind {kind}")


def content_byte_length(descriptor: PlaneDescriptor) -> int:
    """Bytes of real content in a plane, before alignment padding."""
    return bits_to_bytes(descriptor.element_count * descriptor.element_bits)


def build_plane_region(
    planes: tuple[PlaneDescriptor, ...],
    payloads: "dict[PlaneKind, bytes] | None" = None,
) -> bytes:
    """Lay payloads out into the exact plane region, in canonical order.

    This is the serializer half of build item 1b.  Padding is written as zero
    and is not the caller's to choose: unconstrained padding would make the
    same logical content admit many byte strings, and identity here is a
    function of content (review finding F4).
    """
    payloads = payloads or {}
    region = bytearray()
    for descriptor in planes:
        if descriptor.storage is Storage.REFERENCE:
            continue
        need = content_byte_length(descriptor)
        payload = payloads.get(descriptor.kind, bytes(need))
        if len(payload) != need:
            raise PlaneLayoutError(
                f"{descriptor.kind.name}: payload is {len(payload)} bytes, "
                f"the plane holds exactly {need}"
            )
        region += payload
        region += bytes(descriptor.byte_length() - need)
    return bytes(region)


def build_planes(
    geometry: Geometry,
    rates: tuple[int, ...],
    alphabet_blob: bytes,
    descendant_blob: bytes,
    alignment_bytes: int = 1,
    max_released: int = 0,
    payloads: "dict[PlaneKind, bytes] | None" = None,
    with_diagonals: bool = True,
    cap: int = C_FULL_BITS,
    arity: int = 1,
) -> tuple[PlaneDescriptor, ...]:
    """Full-extent descriptors, one per plane, in canonical order.

    Counts are per-superblock granules for position-domain planes, which is the
    granularity a legal truncation respects.  ``max_released`` declares the
    RELEASE plane's full extent: every terminal is a prefix of the declared
    extent, so a terminal may never claim more released positions than the
    plane declares.

    ``with_diagonals=False`` declares segment 2a **absent from the unit**, which
    is different from a terminal that merely truncates it away.  A terminal's
    byte range is the concatenation of each plane's truncated extent, so a unit
    that never fitted diagonals must not declare their full extent either --
    otherwise the region written and the ranges a terminal computes disagree by
    ``16 * (rows + columns)`` bits and every offset after DIAG_SU is wrong.
    """
    superblocks = max(1, len(rates) // geometry.superblock_columns)
    descriptors = []
    for kind in CANONICAL_PLANE_ORDER:
        total = _counts_for(
            kind,
            geometry,
            rates,
            None,
            len(alphabet_blob),
            len(descendant_blob),
            max_released,
            cap=cap,
            arity=arity,
        )
        if not with_diagonals and kind in (PlaneKind.DIAG_SU, PlaneKind.DIAG_SV):
            total = 0
        if kind in (PlaneKind.BODY, PlaneKind.COMPLETION):
            granularity = CountGranularity.PER_SUPERBLOCK
            per, remainder = divmod(total, superblocks)
            counts = tuple(
                per + (1 if index < remainder else 0) for index in range(superblocks)
            )
        else:
            granularity = CountGranularity.WHOLE_PLANE
            counts = (total,)
        offsets, running = [], 0
        for count in counts:
            offsets.append(running)
            running += count
        # The plane's digest covers its exact on-wire byte range -- content
        # plus the zero padding -- so `parse` can verify it against the bytes it
        # actually holds.  Digesting a placeholder (this was `sha256(b"")` for
        # every non-blob plane) made the field unverifiable by construction:
        # review finding F1.
        bits = NORMATIVE_ELEMENT_BITS[kind]
        need = bits_to_bytes(total * bits)
        default = (
            alphabet_blob
            if kind is PlaneKind.ALPHABET
            else descendant_blob if kind is PlaneKind.DESCENDANT else bytes(need)
        )
        blob = (payloads or {}).get(kind, default)
        if len(blob) != need:
            raise PlaneLayoutError(
                f"{kind.name}: payload is {len(blob)} bytes, the plane holds "
                f"exactly {need}"
            )
        raw = bits_to_bytes(total * bits)
        padded = raw + (
            0 if raw % alignment_bytes == 0 else alignment_bytes - raw % alignment_bytes
        )
        descriptors.append(
            PlaneDescriptor(
                kind=kind,
                index_domain=_INDEX_DOMAIN[kind],
                storage=Storage.INLINE,
                element_bits=NORMATIVE_ELEMENT_BITS[kind],
                bit_order=BitOrder.MSB_FIRST,
                alignment_bytes=alignment_bytes,
                count_granularity=granularity,
                counts=counts,
                restart_offsets=tuple(offsets),
                payload_dtype=_DTYPE[kind],
                content_digest=hashlib.sha256(blob + bytes(padded - raw)).digest(),
            )
        )
    return tuple(descriptors)


def build_terminal(
    geometry: Geometry,
    rates: tuple[int, ...],
    spec: TerminalSpec,
    planes: tuple[PlaneDescriptor, ...],
    alphabet_bytes: int,
    descendant_bytes: int,
    plane_region: bytes | None = None,
    cap: int = C_FULL_BITS,
    arity: int = 1,
) -> TerminalRecord:
    """Compute a terminal's exact per-plane counts, bytes, bpp, and digest.

    `plane_region` is the artifact's full region; the terminal's digest covers
    its own byte prefix of it.  Without a per-terminal digest, every legal
    truncation -- this format's headline case -- would carry no integrity check
    at all, because the whole-artifact digest only covers the untruncated
    bytes (review finding F9).
    """
    if len(spec.completion_bits) != len(rates):
        raise GrammarError(
            f"terminal {spec.slot_id!r}: completion vector covers "
            f"{len(spec.completion_bits)} columns, rates cover {len(rates)}"
        )
    for column, (rate, completion) in enumerate(zip(rates, spec.completion_bits)):
        if not 0 <= completion <= completion_capacity(rate, cap):
            raise GrammarError(
                f"terminal {spec.slot_id!r} column {column}: completion "
                f"{completion} exceeds capacity "
                f"{completion_capacity(rate, cap)} at rate {rate} (cap {cap})"
            )
    if not 0 <= spec.released_positions <= geometry.positions:
        raise GrammarError(f"terminal {spec.slot_id!r}: release count out of range")

    by_kind = {plane.kind: plane for plane in planes}
    elements, total_bytes = [], 0
    for kind in CANONICAL_PLANE_ORDER:
        count = _counts_for(
            kind, geometry, rates, spec, alphabet_bytes, descendant_bytes,
            cap=cap, arity=arity,
        )
        elements.append(count)
        total_bytes += by_kind[kind].byte_length(count)

    if plane_region is None:
        payload_digest = hashlib.sha256(bytes(total_bytes)).digest()
    else:
        if len(plane_region) < total_bytes:
            raise PlaneLayoutError(
                f"terminal {spec.slot_id!r}: needs {total_bytes} bytes, the "
                f"region holds {len(plane_region)}"
            )
        payload_digest = hashlib.sha256(plane_region[:total_bytes]).digest()

    return TerminalRecord(
        slot_id=spec.slot_id,
        clip_exponent_code=spec.clip_exponent_code,
        plane_elements=tuple(elements),
        exact_bytes=total_bytes,
        exact_bpp=Fraction(8 * total_bytes, geometry.quantizable_params),
        payload_digest=payload_digest,
    )
