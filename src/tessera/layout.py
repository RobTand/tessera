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

from .errors import GrammarError
from .grammar import RELEASE_BITS, completion_capacity
from .manifest import Geometry, TerminalRecord
from .planes import (
    CANONICAL_PLANE_ORDER,
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

_ELEMENT_BITS = {
    PlaneKind.ALPHABET: 8,
    PlaneKind.DESCENDANT: 8,
    PlaneKind.BODY: 1,
    PlaneKind.SCALE_BASE: 8,
    PlaneKind.COMPLETION: 1,
    PlaneKind.DIAG_SU: 16,
    PlaneKind.DIAG_SV: 16,
    PlaneKind.SCALE_REFINE: 4,
    PlaneKind.RELEASE: RELEASE_BITS,
}

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
) -> int:
    rows = geometry.rows
    positions = geometry.positions
    if kind is PlaneKind.ALPHABET:
        return alphabet_bytes
    if kind is PlaneKind.DESCENDANT:
        return descendant_bytes
    if kind is PlaneKind.BODY:
        return sum(rates) * rows
    if kind is PlaneKind.SCALE_BASE:
        if spec is not None and not spec.with_scale_base:
            return 0
        return positions // geometry.group_weights
    if kind is PlaneKind.COMPLETION:
        if spec is None:
            return sum(completion_capacity(rate) for rate in rates) * rows
        return sum(spec.completion_bits) * rows
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


def build_planes(
    geometry: Geometry,
    rates: tuple[int, ...],
    alphabet_blob: bytes,
    descendant_blob: bytes,
    alignment_bytes: int = 1,
    max_released: int = 0,
) -> tuple[PlaneDescriptor, ...]:
    """Full-extent descriptors, one per plane, in canonical order.

    Counts are per-superblock granules for position-domain planes, which is the
    granularity a legal truncation respects.  ``max_released`` declares the
    RELEASE plane's full extent: every terminal is a prefix of the declared
    extent, so a terminal may never claim more released positions than the
    plane declares.
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
        )
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
        blob = (
            alphabet_blob
            if kind is PlaneKind.ALPHABET
            else descendant_blob if kind is PlaneKind.DESCENDANT else b""
        )
        descriptors.append(
            PlaneDescriptor(
                kind=kind,
                index_domain=_INDEX_DOMAIN[kind],
                storage=Storage.INLINE,
                element_bits=_ELEMENT_BITS[kind],
                bit_order=BitOrder.MSB_FIRST,
                alignment_bytes=alignment_bytes,
                count_granularity=granularity,
                counts=counts,
                restart_offsets=tuple(offsets),
                payload_dtype=_DTYPE[kind],
                content_digest=hashlib.sha256(blob).digest(),
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
) -> TerminalRecord:
    """Compute a terminal's exact per-plane counts, bytes, and bpp."""
    if len(spec.completion_bits) != len(rates):
        raise GrammarError(
            f"terminal {spec.slot_id!r}: completion vector covers "
            f"{len(spec.completion_bits)} columns, rates cover {len(rates)}"
        )
    for column, (rate, completion) in enumerate(zip(rates, spec.completion_bits)):
        if not 0 <= completion <= completion_capacity(rate):
            raise GrammarError(
                f"terminal {spec.slot_id!r} column {column}: completion "
                f"{completion} exceeds capacity {completion_capacity(rate)} at "
                f"rate {rate}"
            )
    if not 0 <= spec.released_positions <= geometry.positions:
        raise GrammarError(f"terminal {spec.slot_id!r}: release count out of range")

    by_kind = {plane.kind: plane for plane in planes}
    elements, total_bytes = [], 0
    for kind in CANONICAL_PLANE_ORDER:
        count = _counts_for(
            kind, geometry, rates, spec, alphabet_bytes, descendant_bytes
        )
        elements.append(count)
        total_bytes += by_kind[kind].byte_length(count)

    return TerminalRecord(
        slot_id=spec.slot_id,
        clip_exponent_code=spec.clip_exponent_code,
        plane_elements=tuple(elements),
        exact_bytes=total_bytes,
        exact_bpp=Fraction(8 * total_bytes, geometry.quantizable_params),
    )
