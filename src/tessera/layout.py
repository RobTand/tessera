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
COMPLETION        bit            1   (count = sum over columns of c * rows;
                                      level-major since schema minor 7)
DIAG_SU           in-channel    16
DIAG_SV           out-channel   16
SCALE_REFINE      half-block     4   (one nibble per 16 weights)
RELEASE           position       4   (a released position stores 16 codes)
===============  ============  ==========================================

The tensor-parallel slicing half -- ``SlicedUnit``, ``slice_unit``,
``shard_granularity``, ``can_shard`` and their helpers -- lives in
:mod:`tessera.slicing`, not here: a shard subclasses ``encode.EncodedUnit``
and the cut replays the trellis, so it needs torch and this module must stay
importable without it.  The cutter names stay available as ``tessera.layout``
attributes through the lazy re-export below, so no caller moves.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction

from .errors import GrammarError, PlaneLayoutError
from .exact import bits_to_bytes
from .grammar import (
    C_FULL_BITS,
    RELEASE_BITS,
    completion_capacity,
    completion_level_counts,
    superblock_count,
)
from .trellis import body_bits as _body_bits
from .manifest import (
    Geometry,
    TerminalRecord,
)
from .planes import (
    CANONICAL_PLANE_ORDER,
    plane_order,
    layout_of,
    NORMATIVE_ELEMENT_BITS,
    BitOrder,
    CountGranularity,
    IndexDomain,
    PayloadDtype,
    PlaneDescriptor,
    PlaneKind,
    PlaneLayout,
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

# The slicing half (``tessera.slicing``) needs torch; this module must not, so
# the cutter names are re-exported lazily rather than imported.  The first
# attribute access imports ``tessera.slicing`` and its torch dependency;
# everything else in this module works without either.  The private helpers
# are included because the cutter's own tests reach them through this module
# (``layout._steps_of``, ``layout._slice_release``).
_SLICING_NAMES = frozenset({
    "SlicedUnit",
    "slice_unit",
    "shard_granularity",
    "can_shard",
    "unsliceable_reason",
    "_slicing_facts",
    "SLICEABLE_SCHEMA_MINOR",
    "tp_agnostic_at_minor",
    "_initial_state",
    "_slice_release",
    "_slice_step_plane",
    "_slice_block_plane",
    "_unwrap",
    "_manifest_granularity",
    "_steps_of",
    "_scale_columns_per_row",
    "_block_straddles_rows",
    "_unsliceable_reason",
    "_lcm",
})


def __getattr__(name: str):
    if name in _SLICING_NAMES:
        from . import slicing as _slicing

        return getattr(_slicing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | _SLICING_NAMES)

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
    #: How many leading halves of an S6b unit's SCALE_REFINE plane this
    #: terminal carries.  Schema D3 gives the plane prefix semantics -- the
    #: halves a terminal does not carry sit at their group's po2 base -- and
    #: this is the spelling of that rung.  ``None`` defers to
    #: ``with_scale_refine``: the whole plane, or none of it.
    scale_refine_halves: "int | None" = None


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
            if spec.scale_refine_halves:
                raise GrammarError(
                    f"terminal {spec.slot_id!r} carries "
                    f"{spec.scale_refine_halves} refinement halves but declares "
                    "no SCALE_REFINE plane"
                )
            return 0
        if positions % geometry.half_weights:
            raise GrammarError(
                f"{positions} weight positions is not a whole number of "
                f"{geometry.half_weights}-weight halves; a floored count "
                "would silently leave the trailing weights scaleless"
            )
        halves = positions // geometry.half_weights
        if spec is None or spec.scale_refine_halves is None:
            return halves
        if not 0 <= spec.scale_refine_halves <= halves:
            raise GrammarError(
                f"terminal {spec.slot_id!r} carries "
                f"{spec.scale_refine_halves} refinement halves of {halves}"
            )
        return spec.scale_refine_halves
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
    layout: PlaneLayout = PlaneLayout.LADDER,
) -> tuple[PlaneDescriptor, ...]:
    """Full-extent descriptors, one per plane, in the layout's wire order.

    ``layout`` selects the wire (``planes.PlaneLayout``).  The default is the
    current one, minor 7: COMPLETION after the scale planes and cut by depth
    level.  ``LEGACY`` reproduces a minor 0-6 artifact -- COMPLETION between
    SCALE_BASE and DIAG_SU, cut by superblock -- byte for byte, and exists so
    a reader can be held to those bytes by a test that builds them.

    ``state_bits > 0`` declares an INITIAL_STATE plane (schema minor 4): one
    ``state_bits``-wide word per column, ahead of BODY in the wire order, and
    the descriptors come back in ``SHARD_PLANE_ORDER``.  Zero -- every whole
    unit -- leaves the plane and the order exactly as they were.

    ``release_counts`` puts the RELEASE plane's per-superblock counts *on the
    wire*.  A whole unit leaves it ``None``: its counts are
    ``grammar.release_quota`` of the total -- the total at a uniform release
    density, awarded by largest remainder -- which the reader regenerates.  A
    shard cannot, and must not -- its counts are the restriction of its
    parent's, which is a different vector.

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
    # it wrote then had one entry fewer than the stream had segments.  The
    # count lives in ``grammar`` so the release quota cannot floor what the
    # granules ceiling.
    superblocks = superblock_count(len(rates), geometry.superblock_columns)
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
    steps = geometry.rows // arity
    for kind in plane_order(state_bits > 0, layout):
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
        elif kind is PlaneKind.COMPLETION and layout is PlaneLayout.LADDER:
            # One granule per depth level, so a terminal cut at level ``l``
            # is the first ``l`` granules and a byte prefix of the plane
            # (``wire.pack_levels``).  The widths are the terminal's when it
            # declares them and the rate ceiling otherwise -- the same rule
            # ``_counts_for`` sizes the total by, which the sum check binds.
            granularity = CountGranularity.PER_LEVEL
            widths = (
                tuple(completion_capacity(rate, cap) for rate in rates)
                if spec is None
                else tuple(spec.completion_bits)
            )
            counts = completion_level_counts(widths, steps)
            if sum(counts) != total:
                raise PlaneLayoutError(
                    f"{kind.name}: the per-level counts sum to {sum(counts)}, "
                    f"the plane holds {total}"
                )
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
    its own byte prefix of it.  Without a per-terminal digest, a legal
    truncation would carry no integrity check at all, because the
    whole-artifact digest only covers the untruncated bytes (review finding
    F9).

    "Legal truncation" is the layout's capability, not something an encoded
    artifact currently offers: ``unit_artifact.build_unit_artifact`` declares
    one terminal per unit, so every artifact this tree writes has exactly one
    legal length and the per-terminal digest is, for now, a second digest over
    the whole region.  Since minor 7 the wire *can* carry a shorter one --
    the plane order and the COMPLETION cut were the obstacles, and the
    schema's §3c records that history -- so what a writer declares is the
    writer's decision (tessera#144), and this function prices whatever it is
    asked for.  The layout is read off ``planes``: a full descriptor sequence
    is in exactly one wire order, and taking it as a second parameter would
    only let the two disagree.
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
    # and the descriptors rather than from a module constant is what keeps
    # the three from drifting.
    layout = layout_of((plane.kind for plane in planes), spec.state_bits > 0)
    for kind in plane_order(spec.state_bits > 0, layout):
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
# That half lived here and moved to ``tessera.slicing``: ``SlicedUnit``
# subclasses ``encode.EncodedUnit`` and the cut replays the trellis, so it
# needs torch and the byte-layout half must not.  The cutter names stay
# available as ``tessera.layout`` attributes through the lazy re-export above,
# so no caller moves.




