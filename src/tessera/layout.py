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
from dataclasses import dataclass, replace
from fractions import Fraction
from math import gcd

import torch

from .encode import EncodedUnit
from .errors import GrammarError, PlaneLayoutError
from .exact import bits_to_bytes
from .grammar import C_FULL_BITS, RELEASE_BITS, completion_capacity
from .trellis import body_bits as _body_bits
from .manifest import (
    BodyKind,
    Geometry,
    RotationState,
    ScalePlaneKind,
    TerminalRecord,
)
from .planes import (
    CANONICAL_PLANE_ORDER,
    plane_order,
    NORMATIVE_ELEMENT_BITS,
    BitOrder,
    CountGranularity,
    IndexDomain,
    PayloadDtype,
    PlaneDescriptor,
    PlaneKind,
    Storage,
)

__all__ = [
    "TerminalSpec",
    "build_planes",
    "build_terminal",
    "ZERO_DIGEST",
    "SlicedUnit",
    "slice_unit",
    "shard_granularity",
    "can_shard",
]

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
    # One state per COLUMN, and a column is an input channel.
    PlaneKind.INITIAL_STATE: IndexDomain.AXIS_IN,
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
    PlaneKind.INITIAL_STATE: PayloadDtype.UINT,
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
    #: A CHANNEL scale plane (schema minor 3): the row scale rides DIAG_SV
    #: alone -- one fp16 per output row -- with DIAG_SU absent.  Distinct from
    #: ``with_diagonals``, which declares the rank-1 pair.
    with_row_scale: bool = False
    #: The INITIAL_STATE plane's element width (schema minor 4): the trellis
    #: state one column starts from.  Zero -- the default, and every whole
    #: unit -- declares the plane absent and the pinned zero start.
    state_bits: int = 0


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
    span: int = 1,
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
    if span < 1 or steps % span:
        raise GrammarError(
            f"{steps} trellis positions per column is not a whole number of "
            f"span-{span} super-symbols"
        )
    if kind is PlaneKind.ALPHABET:
        return alphabet_bytes
    if kind is PlaneKind.DESCENDANT:
        return descendant_bytes
    if kind is PlaneKind.BODY:
        # ``span * R + span - 1`` bits per super-symbol (``trellis.body_bits``);
        # at span 1 that is ``R * steps`` per column, the count it always was.
        return sum(_body_bits(rate, steps, span) for rate in rates)
    if kind is PlaneKind.SCALE_BASE:
        if spec is not None and not spec.with_scale_base:
            return 0
        if positions % geometry.group_weights:
            raise GrammarError(
                f"{positions} weight positions is not a whole number of "
                f"{geometry.group_weights}-weight groups; a floored count "
                "would silently leave the trailing weights scaleless"
            )
        return positions // geometry.group_weights
    if kind is PlaneKind.COMPLETION:
        if spec is None:
            return sum(completion_capacity(rate, cap) for rate in rates) * steps
        return sum(spec.completion_bits) * steps
    if kind is PlaneKind.DIAG_SU:
        if spec is not None and not spec.with_diagonals:
            return 0
        return geometry.columns
    if kind is PlaneKind.DIAG_SV:
        if spec is not None and not (spec.with_diagonals or spec.with_row_scale):
            return 0
        return rows
    if kind is PlaneKind.SCALE_REFINE:
        if spec is not None and not spec.with_scale_refine:
            return 0
        if positions % geometry.half_weights:
            raise GrammarError(
                f"{positions} weight positions is not a whole number of "
                f"{geometry.half_weights}-weight halves; a floored count "
                "would silently leave the trailing weights scaleless"
            )
        return positions // geometry.half_weights
    if kind is PlaneKind.RELEASE:
        return max_released if spec is None else spec.released_positions
    if kind is PlaneKind.INITIAL_STATE:
        # One state per column, or none: a shard cut at row 0 starts from the
        # pinned zero the decoder already assumes.  ``state_bits`` is the
        # element *width*, so the count does not depend on it.
        if spec is not None and not spec.state_bits:
            return 0
        return geometry.columns
    raise GrammarError(f"unhandled plane kind {kind}")


def _superblock_counts(
    kind: PlaneKind,
    geometry: Geometry,
    rates: tuple[int, ...],
    spec: "TerminalSpec | None",
    superblocks: int,
    cap: int,
    arity: int,
    span: int,
) -> "tuple[int, ...]":
    """Per-superblock element counts, summed over each superblock's own columns.

    The restart table is the segment-local seek contract -- the offsets a GPU
    consumer enters the stream at without a host parse -- so a granule's count
    has to be the bits that granule's columns actually carry.  Spreading the
    plane total evenly across the granules instead (``divmod``) coincides with
    that only when every superblock carries the same bits, which is true for a
    complete superblock under the rate quota and false for a trailing partial
    one.  A ``sum`` check at the call site binds the two together, so the
    granules can never describe a different plane from the one that was built.
    """
    steps = geometry.rows // arity
    superblock = geometry.superblock_columns
    counts = []
    for index in range(superblocks):
        window = slice(index * superblock, (index + 1) * superblock)
        if kind is PlaneKind.BODY:
            counts.append(sum(_body_bits(rate, steps, span) for rate in rates[window]))
        elif spec is None:
            counts.append(
                sum(completion_capacity(rate, cap) for rate in rates[window]) * steps
            )
        else:
            counts.append(sum(spec.completion_bits[window]) * steps)
    return tuple(counts)


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
    # No ``Storage.REFERENCE`` skip: ``PlaneDescriptor.__post_init__`` refuses
    # that storage at construction (``tests/test_audit_container_accounting.py::
    # test_reference_storage_is_refused_not_charged_zero``), so a skip here
    # could only ever advertise support for a plane no descriptor can hold.
    for descriptor in planes:
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
    spec: "TerminalSpec | None" = None,
    span: int = 1,
    with_row_scale: bool = False,
    state_bits: int = 0,
    release_counts: "tuple[int, ...] | None" = None,
) -> tuple[PlaneDescriptor, ...]:
    """Full-extent descriptors, one per plane, in canonical order.

    ``state_bits > 0`` declares an INITIAL_STATE plane (schema minor 4): one
    ``state_bits``-wide word per column, ahead of BODY in the wire order, and
    the descriptors come back in ``SHARD_PLANE_ORDER``.  Zero -- every whole
    unit -- leaves the plane and the order exactly as they were.

    ``release_counts`` puts the RELEASE plane's per-superblock counts *on the
    wire*.  A whole unit leaves it ``None``: its counts are the Bresenham
    spread of the total, which the reader regenerates.  A shard cannot, and
    must not -- its counts are the restriction of its parent's, which is a
    different vector.

    ``with_row_scale=True`` declares a CHANNEL scale plane's row field: the
    DIAG_SV plane is present at ``rows`` fp16 words with DIAG_SU absent.

    Counts are per-superblock granules for position-domain planes, which is the
    granularity a legal truncation respects, and a trailing partial superblock
    is a granule of its own.  ``max_released`` declares the RELEASE plane's full
    extent: every terminal is a prefix of the declared extent, so a terminal may
    never claim more released positions than the plane declares.  Passing both
    ``max_released`` and a ``spec`` that names a different count is refused --
    the extent and the terminal's slice of it are one decision, and one
    parameter silently winning over the other hid the disagreement.

    ``spec`` declares the terminal this unit is built for, and it is what makes
    the COMPLETION plane's extent follow the depth the encoder actually used
    rather than the depth the *rate* leaves room for.  Passing ``None`` sizes
    COMPLETION at full capacity, which is what a unit that used every
    completion bit gets anyway -- so the default is unchanged.  Without it a
    column encoded at ``completion=0`` still paid ``cap - rate`` bits for an
    all-zero plane, which is what made every rung of a family weigh the same.

    ``with_diagonals=False`` declares segment 2a **absent from the unit**, which
    is different from a terminal that merely truncates it away.  A terminal's
    byte range is the concatenation of each plane's truncated extent, so a unit
    that never fitted diagonals must not declare their full extent either --
    otherwise the region written and the ranges a terminal computes disagree by
    ``16 * (rows + columns)`` bits and every offset after DIAG_SU is wrong.
    """
    # Ceiling, not floor: a trailing partial superblock is a granule of its own.
    # ``superblock_quota_ok`` already declares such a superblock legal (it
    # constrains only *complete* ones), so flooring it away was the layout
    # refusing to describe a shape the grammar admits -- and the restart table
    # it wrote then had one entry fewer than the stream had segments.
    superblocks = max(1, -(-len(rates) // geometry.superblock_columns))
    if spec is not None and max_released and max_released != spec.released_positions:
        # One parameter cannot mean both the plane's full extent and the
        # terminal's slice of it.  When a caller declares both, they are the
        # same decision and must agree; silently preferring one hid the other.
        raise PlaneLayoutError(
            f"the RELEASE plane is declared at {max_released} released "
            f"positions and the terminal spec claims "
            f"{spec.released_positions}: the extent and the terminal's slice "
            "of it are one decision"
        )
    if release_counts is not None and len(release_counts) != superblocks:
        raise PlaneLayoutError(
            f"RELEASE declares {len(release_counts)} superblock counts, the "
            f"schedule has {superblocks} superblocks"
        )
    descriptors = []
    for kind in plane_order(state_bits > 0):
        total = _counts_for(
            kind,
            geometry,
            rates,
            spec,
            len(alphabet_blob),
            len(descendant_blob),
            max_released,
            cap=cap,
            arity=arity,
            span=span,
        )
        if kind is PlaneKind.DIAG_SU and not with_diagonals:
            total = 0
        if kind is PlaneKind.DIAG_SV and not (with_diagonals or with_row_scale):
            total = 0
        if kind is PlaneKind.RELEASE and release_counts is not None:
            if sum(release_counts) != total:
                raise PlaneLayoutError(
                    f"RELEASE counts sum to {sum(release_counts)}, the terminal "
                    f"declares {total} released positions"
                )
            granularity = CountGranularity.PER_SUPERBLOCK
            counts = tuple(release_counts)
        elif kind in (PlaneKind.BODY, PlaneKind.COMPLETION):
            granularity = CountGranularity.PER_SUPERBLOCK
            counts = _superblock_counts(
                kind, geometry, rates, spec, superblocks, cap, arity, span
            )
            if sum(counts) != total:
                raise PlaneLayoutError(
                    f"{kind.name}: the per-superblock counts sum to "
                    f"{sum(counts)}, the plane holds {total}"
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
        bits = NORMATIVE_ELEMENT_BITS.get(kind) or state_bits
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
                element_bits=bits,
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
    span: int = 1,
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
    # The terminal's count array is indexed by the *unit's* wire order, and a
    # shard's has one more entry than a whole unit's.  Taking it from the spec
    # rather than from a module constant is what keeps the two from drifting.
    for kind in plane_order(spec.state_bits > 0):
        count = _counts_for(
            kind, geometry, rates, spec, alphabet_bytes, descendant_bytes,
            cap=cap, arity=arity, span=span,
        )
        elements.append(count)
        total_bytes += by_kind[kind].byte_length(count)

    if plane_region is None:
        # No bytes were supplied, so no bytes were hashed.  ``sha256(zeros)``
        # is a well-formed 32-byte digest that verifies against a zero region,
        # which is a plausible-looking lie about data nobody hashed; the
        # all-zero sentinel is not a digest of anything and cannot be mistaken
        # for one.  Callers that only price a terminal (``calculator``) read
        # ``exact_bpp`` and never this field.
        payload_digest = ZERO_DIGEST
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


# ---------------------------------------------------------------------------
# Tensor parallelism: cutting a unit into the shard one rank loads
# ---------------------------------------------------------------------------
#
# A Tessera artifact is written once, by an exporter that never learns the TP
# degree, and every rank cuts its own shard out of those bytes at load.  That
# is the whole contract, and it rests on one fact about the wire: a column's
# body is a bit stream whose only carried state is the trellis register, so a
# stream can be *entered in the middle* provided the state at that point
# travels with it.  ``slice_unit`` does the cutting; ``INITIAL_STATE`` is the
# one plane it adds; ``shard_granularity`` says where the cuts may fall.
#
# Nothing here re-encodes.  Every code in the shard is the code the parent
# stored -- the same E4M3 or E2M1 nibble, against the same scale -- so a rank's
# shard decodes bit-for-bit to its window of the parent's decode.  The
# alternative, which this replaces, was encoding one artifact per TP degree.


@dataclass
class SlicedUnit(EncodedUnit):
    """An ``EncodedUnit`` that is a window of another one.

    The extra fields are the shard record's (``manifest.ShardOrigin``) plus
    the start state itself.  It is a subclass rather than new fields on
    ``EncodedUnit`` because the encoder never produces one: a shard is made by
    cutting, at load, and every decoder path reads the state through
    ``getattr(unit, "initial_state", None)`` -- so a whole unit takes exactly
    the path it always did.

    ``release_counts`` is the per-superblock release count vector.  A whole
    unit does not carry one (its counts are the Bresenham spread of the total,
    which the reader regenerates); a shard must, because its counts are the
    *restriction* of its parent's and no spread reproduces them.
    """

    row_offset: int = 0
    col_offset: int = 0
    #: ``[cols]`` int64: the trellis state column ``j`` starts from.  ``None``
    #: when ``row_offset == 0`` -- the pinned zero start the decoder assumes.
    initial_state: "torch.Tensor | None" = None
    parent_rows: int = 0
    parent_columns: int = 0
    parent_digest: bytes = b""
    release_counts: "tuple[int, ...]" = ()
    #: The INITIAL_STATE plane's element width: the window width under a
    #: WINDOW body, the convolutional code's memory under TCQ.  Zero exactly
    #: when there is no state plane.
    state_bits: int = 0


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _scale_columns_per_row(unit) -> "int | None":
    """The column stride of the block scale planes, or ``None`` if there are none."""
    kind = ScalePlaneKind(getattr(unit, "scale_plane", ScalePlaneKind.S6B))
    if kind is ScalePlaneKind.CHANNEL:
        return None
    return unit.group if kind is ScalePlaneKind.S6B else unit.half


def shard_granularity(unit, superblock: int = 256, arity: int = 1):
    """``(row_granularity, col_granularity)``: where a cut may legally fall.

    Both numbers are *derived* from the checks ``slice_unit`` applies, never
    asserted alongside them, so a granularity this reports is one that slices.

    **Rows.**  A cut has to land on a trellis step, and a step covers ``arity``
    weight rows; under the coset trellis it also has to land on a super-symbol
    boundary, because the stored labels of a span-L super-symbol are read
    together and position 0's label is derived from them -- so entering a
    super-symbol halfway carries a *label phase* no state can express.  Hence
    ``arity * span`` under TCQ and ``arity`` under the window body, whose span
    is always 1.  The block scale planes run along the row, so a row boundary
    is a block boundary whenever the unit's column count is a whole number of
    blocks; when it is not, the row granularity rises to make it one.

    **Columns.**  A block scale plane is indexed by ``(row * cols + col) //
    block``, so a column cut must fall on a block: 32 weights under S6b (which
    carries a base plane per 32 and a refinement per 16), 16 under LUT.  A
    CHANNEL plane has no column structure at all -- its scale is one word per
    output row -- so its column granularity is 1, which is what makes the
    served FP8 route the cheapest one to shard.  Two things raise it to the
    superblock: a RELEASE plane, whose placement is defined *within* a
    superblock (see ``decode.release_order``), and a mixed rate schedule,
    whose quota ``sum(rates) == root * columns`` is only exact on whole
    superblocks.
    """
    from .manifest import BodyKind, Manifest

    if isinstance(unit, Manifest):
        return _manifest_granularity(unit)
    unit, superblock, arity = _unwrap(unit, superblock, arity)
    steps, cols = unit.body_bits.shape
    span = int(getattr(unit, "span", 1))
    body = BodyKind(getattr(unit, "body", BodyKind.TCQ))
    row = arity * (span if body is BodyKind.TCQ else 1)
    block = _scale_columns_per_row(unit)
    if block is not None and cols % block:
        # A row is not a whole number of blocks, so a block straddles rows and
        # only a run of rows that closes one is cuttable.
        row = _lcm(row, block // gcd(block, cols))
    col = 1 if block is None else block
    mixed = len(set(unit.rates)) > 1
    if mixed or unit.released_positions:
        col = _lcm(col, superblock)
    return row, col


def _manifest_granularity(manifest):
    """``shard_granularity`` for a manifest -- what a reader holding bytes has."""
    from .manifest import BodyKind, ScalePlaneKind as _Kind

    geometry = manifest.geometry
    kind = manifest.scale_plane.kind
    block = (
        None
        if kind is _Kind.CHANNEL
        else geometry.group_weights if kind is _Kind.S6B else geometry.half_weights
    )
    arity = geometry.rows * geometry.columns // (geometry.columns * _steps_of(manifest))
    row = arity * (manifest.span if manifest.body is BodyKind.TCQ else 1)
    if block is not None and geometry.columns % block:
        row = _lcm(row, block // gcd(block, geometry.columns))
    col = 1 if block is None else block
    released = max(
        terminal.plane_elements[
            plane_order(manifest.shard is not None
                        and manifest.shard.has_initial_state).index(PlaneKind.RELEASE)
        ]
        for terminal in manifest.terminals
    )
    if len(set(manifest.rates)) > 1 or released:
        col = _lcm(col, geometry.superblock_columns)
    return row, col


def _steps_of(manifest) -> int:
    """Trellis steps per column, from the BODY plane's declared element count."""
    from .trellis import body_bits as _bits

    wire = plane_order(manifest.shard is not None and manifest.shard.has_initial_state)
    elements = max(
        terminal.plane_elements[wire.index(PlaneKind.BODY)]
        for terminal in manifest.terminals
    )
    # 1 and 2 are the arities a reader can meet: ``arity`` is the grid's tuple
    # order, and the only serialisable grids are E2M1, E2M1x2, E4M3 and BF16
    # (``alphabet.SERIALISABLE_GRIDS``).  A wider tuple is refused twice over --
    # by that registry, and by the 256-code ceiling on the byte-wide
    # ALPHABET/DESCENDANT planes, which E2M1^3's 4096 codes already break.  The
    # 4 and 8 this loop used to try could only mis-attribute a body-bit count
    # that 1 and 2 had already failed to explain; refusing is the honest answer.
    for arity in (1, 2):
        if manifest.geometry.rows % arity:
            continue
        steps = manifest.geometry.rows // arity
        if steps % manifest.span:
            continue
        if sum(_bits(rate, steps, manifest.span) for rate in manifest.rates) == elements:
            return steps
    raise GrammarError(
        f"the BODY plane declares {elements} bits, which no arity over this "
        "rate schedule produces"
    )


def can_shard(unit, tp: int, axis: str, superblock: int = 256, arity: int = 1) -> bool:
    """Can ``unit`` be cut into ``tp`` equal shards along ``axis``?

    ``axis`` is ``"row"`` for a column-parallel Linear (q/k/v, gate/up: vLLM
    splits the *output* features, which are this unit's rows) and ``"column"``
    for a row-parallel one (o_proj, down_proj: the *input* features, this
    unit's columns).  Expert parallelism moves whole units and asks nothing of
    this function.
    """
    if tp < 1:
        raise GrammarError(f"tp must be positive, got {tp}")
    if axis not in ("row", "column"):
        raise GrammarError(f"axis is 'row' or 'column', got {axis!r}")
    from .manifest import Manifest

    if isinstance(unit, Manifest):
        rows, cols = unit.geometry.rows, unit.geometry.columns
    else:
        unit, superblock, arity = _unwrap(unit, superblock, arity)
        steps, cols = unit.body_bits.shape
        rows = steps * arity
    row_gran, col_gran = shard_granularity(unit, superblock, arity)
    extent, granularity = (rows, row_gran) if axis == "row" else (cols, col_gran)
    return extent % tp == 0 and (extent // tp) % granularity == 0


def _unwrap(unit, superblock: int, arity: int):
    """A ``ParsedUnit`` carries its own superblock and arity; take them.

    A loader holds what ``parse_unit_artifact`` returned, not a bare
    ``EncodedUnit``, and the defaults here (superblock 256, arity 1) are wrong
    for a k-tuple grid -- so reading them off the parse is what keeps
    ``can_shard`` and ``slice_unit`` answering about the same unit.
    """
    from .unit_artifact import ParsedUnit

    if not isinstance(unit, ParsedUnit):
        return unit, superblock, arity
    return (
        unit.unit,
        unit.manifest.geometry.superblock_columns,
        unit.grid.arity if unit.grid is not None else arity,
    )


def _initial_state(unit, steps0: int, arity: int, code, parent_state):
    """The trellis state each column is at just before step ``steps0``.

    Computed with the **decoder's own** replay, not a second formula: the
    window body's state is ``decode.replay_window`` read at the last step above
    the cut, and the coset trellis's is one ``ConvCode`` step past
    ``decode._conv_state_stream``'s last row.  A shard of a shard therefore
    composes for free -- the parent's own start state is an input to both --
    and there is no separate derivation to drift from the decoder.

    The replay runs over a **bounded tail**, not over every row above the cut.
    Both registers are finite: the window body's state is the last ``L`` bits
    of the stream, which ``ceil(L / R)`` positions fill, and the coset
    trellis's is the last ``memory`` select bits, which ``memory + 1``
    super-symbols fill (one more, because the stream reports the register
    *before* each step it has bits for).  Past that depth the rows above
    contribute nothing -- including the parent's own start state, which is why
    the tail needs no init once it is full.  Running the whole prefix instead
    made a rank's cut cost O(rows above it): the last rank of a tp=8 cut
    replayed seven eighths of the unit to recover fourteen bits per column.
    """
    from .decode import _conv_state_stream, replay_window

    if steps0 == 0:
        # Nothing above the cut inside *this* unit -- but if this unit is
        # itself a shard, its own start state is what row 0 replays from, and
        # the sub-shard inherits it verbatim.  Returning ``None`` here made a
        # re-slice of any rank but the first decode from the pinned zero.
        return None if parent_state is None else parent_state.clone()
    body = unit.body_bits
    cols = body.shape[1]
    device = body.device
    rates = torch.tensor(unit.rates, device=device)
    state = torch.zeros(cols, dtype=torch.long, device=device)
    span = int(getattr(unit, "span", 1))
    window = BodyKind(getattr(unit, "body", BodyKind.TCQ)) is BodyKind.WINDOW
    for present in sorted(set(unit.rates)):
        which = torch.nonzero(rates == present).squeeze(1)
        if window:
            window_bits = int(unit.window_bits)
            taps = -(-window_bits // present)
            depth = min(steps0, taps)
            start = (
                None
                if depth == taps or parent_state is None
                else parent_state.to(device)[which]
            )
            state[which] = replay_window(
                body[steps0 - depth : steps0, which], window_bits, present, start
            )[-1]
            continue
        start = None if parent_state is None else parent_state.to(device)[which]
        # The coset trellis: one select bit per super-symbol.  The stream gives
        # the register *before* each super-symbol it has bits for, so the state
        # at the cut is one step past its last row -- ``ConvCode.step``'s
        # ``((bit << memory) | state) >> 1``, which is exact and costs a shift.
        supers = steps0 // span
        depth = min(supers, code.memory + 1)
        if depth == code.memory + 1:
            start = None
        # Row-slice first, then column-gather: gathering the whole prefix and
        # slicing it afterwards copies every row above the cut.
        tail = body[(supers - depth) * span : steps0, which]
        select = (
            (tail.long().reshape(depth, span, which.numel())[:, 0]) >> (present - 1)
        ) & 1
        stream = _conv_state_stream(select, code.memory, start)
        state[which] = (
            (select[depth - 1].long() << code.memory) | stream[depth - 1].long()
        ) >> 1
    return state


def slice_unit(unit, rows=None, cols=None, *, arity: int = 1, code=None,
               superblock: int = 256, parent_shape=None, parent_digest: bytes = b"",
               grid=None):
    """Cut ``unit`` down to ``rows = (r0, r1)`` by ``cols = (c0, c1)``.

    The result is a **standalone unit**: it decodes on its own, through the
    same ``tessera.decode`` entry points, to exactly ``decode(unit)[r0:r1,
    c0:c1]`` -- the same codes against the same scales, bit for bit, with no
    re-encoding anywhere.  That is what makes a Tessera artifact
    tensor-parallel by construction: the exporter writes one unit and never
    learns the TP degree, and each rank cuts its own shard at load.

    Everything is a restriction of what the parent already stored:

    * **BODY / COMPLETION** are per-column streams, so they slice on both axes.
    * **The block scale planes** are indexed ``(row * cols + col) // block``, so
      a column cut must fall on a block boundary; a row cut is free whenever a
      row is a whole number of blocks.  A **CHANNEL** plane slices along rows
      alone and constrains columns not at all.
    * **DIAG_SU** is per input channel and **DIAG_SV** per output channel, so
      each slices on its own axis.
    * **RELEASE** restricts by the threshold argument in
      ``decode.release_order``; the shard's per-superblock counts are the
      parent's counts restricted, and travel on the wire because no spread
      reproduces them.
    * **ALPHABET / DESCENDANT** (the forests, or the window table) and the LUT
      table are whole-unit and are carried across untouched.

    The one thing that is *not* a restriction is the trellis state.  A column's
    body is a bit stream entered at row 0 from a pinned zero state; entered at
    row ``r0`` it starts from whatever the rows above left in the register, so
    that register is stored -- one word per column, on the INITIAL_STATE plane
    (schema minor 4).  At ``r0 == 0`` there is nothing to store, the plane is
    absent, and the bytes are the parent's own: the identity slice of any unit
    is that unit, byte for byte.

    ``unit`` may be an ``EncodedUnit`` or a ``ParsedUnit``; a parsed one
    supplies its own ``code``, ``superblock``, ``arity`` and parent digest.
    ``rows``/``cols`` default to the full extent.  Rotation is refused: an
    ``R_in``-only unit's rotation blocks are a column structure a column cut
    would break silently.
    """
    from .trellis import ConvCode
    from .unit_artifact import ParsedUnit

    if isinstance(unit, ParsedUnit):
        manifest = unit.manifest
        grid = unit.grid
        code = code or unit.code
        superblock = manifest.geometry.superblock_columns
        parent_shape = parent_shape or (
            manifest.geometry.rows, manifest.geometry.columns
        )
        parent_digest = parent_digest or manifest.manifest_digest()
        unit = unit.unit
    if grid is not None:
        arity = grid.arity
    code = code or ConvCode()

    steps, columns = unit.body_bits.shape
    n_rows = steps * arity
    r0, r1 = (0, n_rows) if rows is None else (int(rows[0]), int(rows[1]))
    c0, c1 = (0, columns) if cols is None else (int(cols[0]), int(cols[1]))
    if not (0 <= r0 < r1 <= n_rows) or not (0 <= c0 < c1 <= columns):
        raise GrammarError(
            f"slice rows [{r0}, {r1}) x cols [{c0}, {c1}) is not inside a "
            f"{n_rows}x{columns} unit"
        )
    if unit.rotation is not RotationState.NONE:
        raise GrammarError(
            "refusing to slice a rotated unit: R_in-only rotation is a "
            f"{unit.rotation_block}-column block structure a column cut would "
            "break, and the pieces would decode to plausible wrong weights"
        )
    row_gran, col_gran = shard_granularity(unit, superblock, arity)
    for offset, name, granularity in ((r0, "row", row_gran), (c0, "column", col_gran)):
        if offset % granularity:
            raise GrammarError(
                f"{name} offset {offset} is not a multiple of this unit's "
                f"{name} granularity {granularity}"
            )
    if r1 % row_gran and r1 != n_rows:
        raise GrammarError(
            f"row {r1} is not a multiple of the row granularity {row_gran}"
        )
    if c1 % col_gran and c1 != columns:
        raise GrammarError(
            f"column {c1} is not a multiple of the column granularity {col_gran}"
        )

    rates = tuple(unit.rates[c0:c1])
    if len(set(unit.rates)) > 1:
        # The rate quota is exact per whole superblock, so a slice on
        # superblock boundaries keeps it -- but the check is the arithmetic,
        # never the boundary rule that is supposed to imply it.
        root = Fraction(sum(unit.rates), len(unit.rates))
        want = root * (c1 - c0)
        if want.denominator != 1 or sum(rates) != int(want):
            raise GrammarError(
                f"columns [{c0}, {c1}) carry {sum(rates)} rate bits; the root "
                f"{root} over {c1 - c0} columns requires {want}. This cut does "
                "not keep the rate quota exact"
            )

    s0, s1 = r0 // arity, r1 // arity
    span = int(getattr(unit, "span", 1))
    if r0 % arity or r1 % arity or s0 % span or (s1 - s0) % span:
        raise GrammarError(
            f"rows [{r0}, {r1}) is not a whole number of span-{span} "
            f"super-symbols at arity {arity}"
        )
    parent_state = getattr(unit, "initial_state", None)
    state = _initial_state(unit, s0, arity, code, parent_state)
    # The state is computed over the parent's columns, because the replay that
    # produces it is per column of the parent; the shard keeps its own.
    if state is not None:
        state = state[c0:c1].contiguous()

    kind = ScalePlaneKind(getattr(unit, "scale_plane", ScalePlaneKind.S6B))
    scale_base = _slice_block_plane(unit.scale_base, n_rows, columns, unit.group,
                                    r0, r1, c0, c1, "SCALE_BASE")
    scale_refine = _slice_block_plane(unit.scale_refine, n_rows, columns, unit.half,
                                      r0, r1, c0, c1, "SCALE_REFINE")
    scale_rows = None if unit.scale_rows is None else unit.scale_rows[r0:r1].clone()
    diagonals = None
    if unit.diagonals is not None:
        from .diagonals import Diagonals

        diagonals = Diagonals(su=unit.diagonals.su[c0:c1].clone(),
                              sv=unit.diagonals.sv[r0:r1].clone())

    index, release_code, counts = _slice_release(
        unit, n_rows, columns, r0, r1, c0, c1, superblock
    )
    body_kind = BodyKind(getattr(unit, "body", BodyKind.TCQ))
    state_bits = 0
    if state is not None:
        state_bits = (
            int(unit.window_bits) if body_kind is BodyKind.WINDOW else code.memory
        )
    parent_rows, parent_columns = parent_shape or (n_rows, columns)
    # The identity slice of a whole unit is that unit: it names no parent, so
    # it writes no shard record and its bytes are the bytes it came from.  A
    # slice of a *shard* keeps the shard record whatever its extent, because
    # the offsets it composes are still offsets into the original parent.
    inherited = int(getattr(unit, "parent_rows", 0))
    if not inherited and (r0, c0, r1, c1) == (0, 0, n_rows, columns):
        parent_rows = parent_columns = 0
        parent_digest = b""
    return SlicedUnit(
        rates=rates,
        anchors=_slice_step_plane(unit.anchors, s0, s1, c0, c1),
        codes=_slice_step_plane(unit.codes, s0, s1, c0, c1),
        body_bits=unit.body_bits[s0:s1, c0:c1].contiguous(),
        completion_bits=_slice_step_plane(unit.completion_bits, s0, s1, c0, c1),
        scale_base=scale_base,
        scale_refine=scale_refine,
        release_index=index,
        release_code=release_code,
        # The parent's summed squared error is not a property of the shard and
        # no restriction of it is; a shard that claimed one would be claiming a
        # measurement nobody made.
        sse=0.0,
        rotation=unit.rotation,
        rotation_block=unit.rotation_block,
        diagonals=diagonals,
        group=unit.group,
        half=unit.half,
        completion_limit=unit.completion_limit,
        scale_refit=unit.scale_refit,
        span=span,
        scale_plane=kind,
        scale_lut=unit.scale_lut,
        scale_global=unit.scale_global,
        body=body_kind,
        window_bits=int(getattr(unit, "window_bits", 0)),
        window_codes=unit.window_codes,
        scale_rows=scale_rows,
        row_offset=r0 + int(getattr(unit, "row_offset", 0)),
        col_offset=c0 + int(getattr(unit, "col_offset", 0)),
        initial_state=state,
        parent_rows=parent_rows,
        parent_columns=parent_columns,
        parent_digest=parent_digest or getattr(unit, "parent_digest", b""),
        release_counts=counts,
        state_bits=state_bits,
    )


def _slice_step_plane(plane, s0: int, s1: int, c0: int, c1: int):
    """Slice a per-step plane, tolerating the zero placeholders a reader makes."""
    if plane is None or plane.ndim != 2 or plane.numel() == 0:
        return plane
    return plane[s0:s1, c0:c1].contiguous()


def _slice_block_plane(plane, rows, columns, block, r0, r1, c0, c1, name):
    """Slice a block scale plane, whose index is ``(row * cols + col) // block``.

    The plane is one entry per ``block`` consecutive **columns** of one row, so
    it reshapes to ``[rows, cols // block]`` and slices on both axes -- a
    strided gather along the row, not a contiguous run, which is exactly why
    the column cut has to land on a block.
    """
    if plane is None or plane.numel() == 0:
        return plane
    if columns % block or c0 % block or (c1 - c0) % block:
        raise GrammarError(
            f"{name}: a {block}-weight block does not divide a cut at columns "
            f"[{c0}, {c1}) of {columns}"
        )
    if plane.numel() != rows * columns // block:
        raise GrammarError(
            f"{name} holds {plane.numel()} entries; a {rows}x{columns} unit at "
            f"block {block} needs {rows * columns // block}"
        )
    field = plane.reshape(rows, columns // block)
    return field[r0:r1, c0 // block : c1 // block].reshape(-1).contiguous()


def _slice_release(unit, rows, columns, r0, r1, c0, c1, superblock):
    """Restrict the RELEASE plane, and report the shard's per-superblock counts.

    The parent's ``release_index`` is already in S9 order -- superblock-major,
    then descending decoded ``|value|`` with a positional tie-break -- and the
    restriction preserves both keys: superblocks map monotonically onto the
    shard's (a cut on a superblock boundary is a refinement of the partition),
    and within one superblock the surviving entries keep their relative order.
    So the filter *is* the shard's order, and ``decode.release_order``
    reproduces it from the shard's own decode.  See that function for why the
    restriction of a top-n set is a top-k set.
    """
    device = unit.body_bits.device
    empty = torch.zeros(0, dtype=torch.long, device=device)
    if unit.release_index.numel() == 0:
        return empty, empty, ()
    width = c1 - c0
    if c0 % superblock or (width % superblock and width > superblock):
        raise GrammarError(
            f"a unit with {unit.release_index.numel()} released positions cuts "
            f"only on superblock boundaries: columns [{c0}, {c1}) is neither a "
            f"union of {superblock}-column superblocks nor inside one"
        )
    blocks = max(1, width // superblock)
    flat = unit.release_index.long()
    row = flat // columns
    col = flat % columns
    kept = torch.nonzero(
        (row >= r0) & (row < r1) & (col >= c0) & (col < c1)
    ).squeeze(1)
    index = (row[kept] - r0) * width + (col[kept] - c0)
    block_of = (col[kept] - c0) // superblock
    counts = tuple(int((block_of == b).sum()) for b in range(blocks))
    return index, unit.release_code[kept].clone(), counts
