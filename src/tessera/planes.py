"""Typed plane descriptors (doc S9).

The wire is **separate typed planes**, not one interleaved event stream.  Codex
round-5 P0-1 settled why: an interleaved stream ordered by a positional key is
not uniquely decodable, because the key identifies a position but not an
event's type, width, family count, or restart offset -- and a block-level scale
event has no positional key at all.

Each plane therefore carries its own descriptor: index domain, count
granularity, integer width, endianness and bit order, alignment and padding,
offset/restart encoding, and payload dtype.  Canonical placement removes the
per-position mask; it does **not** remove the per-plane counts, which are
stored and charged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .canonical import DIGEST_BYTES, Reader, Writer
from .grammar import RELEASE_BITS
from .errors import PlaneLayoutError
from .exact import bits_to_bytes

__all__ = [
    "PlaneKind",
    "PlaneLayout",
    "NORMATIVE_ELEMENT_BITS",
    "SHARD_PLANE_ORDER",
    "LEGACY_PLANE_ORDER",
    "LEGACY_SHARD_PLANE_ORDER",
    "plane_order",
    "layout_of",
    "IndexDomain",
    "Storage",
    "BitOrder",
    "CountGranularity",
    "PayloadDtype",
    "PlaneDescriptor",
    "CANONICAL_PLANE_ORDER",
]


class PlaneKind(IntEnum):
    """The plane families of doc S5/S9."""

    ALPHABET = 0  # base alphabet blob (a charged plane)
    DESCENDANT = 1  # stage-C descendant map blob
    BODY = 2  # segment 0: the shaped trellis body
    SCALE_BASE = 3  # segment 2b plane 1: E8M0 per-32 base
    COMPLETION = 4  # stage C: alphabet-completion bits
    DIAG_SU = 5  # segment 2a: input-channel diagonal
    DIAG_SV = 6  # segment 2a: output-channel diagonal
    SCALE_REFINE = 7  # segment 2b plane 2: 4-bit per-16 refinement
    RELEASE = 8  # stage B: 4-bit constraint-release overrides
    INITIAL_STATE = 9  # shard: the trellis state each column starts from


class PlaneLayout(IntEnum):
    """Which wire order, and which COMPLETION cut, an artifact's planes use.

    Not a wire field: the container's schema minor implies it (``LEGACY`` for
    minors 0-6, ``LADDER`` from minor 7), and ``Manifest.decode`` sets it from
    the header.  It is a value rather than a boolean so a third layout, if one
    is ever needed, has a name and not a negation.
    """

    #: Minors 0-6: COMPLETION sits between SCALE_BASE and DIAG_SU and is cut
    #: by superblock of columns at full depth.  Read, never written, since
    #: minor 7 -- except by ``layout=PlaneLayout.LEGACY``, which reproduces a
    #: pre-minor-7 artifact byte for byte and exists so a test can build one.
    LEGACY = 0
    #: Minor 7 (2026-09-05): COMPLETION sits after SCALE_REFINE, ahead of
    #: RELEASE only, and is cut by depth level (``CountGranularity.PER_LEVEL``),
    #: so a shallower completion rung is a byte prefix of the plane.
    LADDER = 1


#: Wire order, which is also the truncation order (schema 1a decision D5, as
#: revised at minor 7 -- ``PlaneLayout.LADDER``).
#:
#: A terminal is a prefix of this sequence, with the last non-empty plane cut
#: at a granule boundary (a superblock of BODY, a depth level of COMPLETION).
#: The order is forced by what a truncated reading needs: everything a decode
#: at *any* completion depth consumes -- the two blob planes, the body, the
#: block scales (S6b base and its refinement, or the LUT plane's index nibble
#: on SCALE_REFINE), the diagonals, a CHANNEL plane's row scale on DIAG_SV --
#: precedes COMPLETION, so cutting the completion axis short keeps every one
#: of them.  RELEASE follows COMPLETION and nothing else could: §9 places
#: releases by ranking the pre-release decode at the *written* depth
#: (``unit_artifact._release_placement``), so a shallower COMPLETION reading
#: moves the positions the RELEASE codes land on, and a rung that shortens
#: COMPLETION cannot keep RELEASE.  The prefix rule then forces RELEASE last.
#:
#: The consequence for doc S6's terminal classes: T-po2 (body + po2 base and
#: nothing after) is still a prefix; "completion without refinement" no longer
#: is, because the refinement now leads the completion axis.  The ladder a
#: writer *may* declare runs po2 base -> block scales (+ diagonals) ->
#: completion depth 1..c -> release.  What a writer *does* declare is its own
#: business: ``unit_artifact.build_unit_artifact`` still declares exactly one
#: terminal, at the depth the encoder used (schema §3c, tessera#144).
#:
#: Every plane an artifact today's recipe table writes is empty at COMPLETION,
#: so the plane *region* of such a unit is byte-identical under either order;
#: only the manifest's descriptor order and the terminal's count array move.
CANONICAL_PLANE_ORDER: tuple[PlaneKind, ...] = (
    PlaneKind.ALPHABET,
    PlaneKind.DESCENDANT,
    PlaneKind.BODY,
    PlaneKind.SCALE_BASE,
    PlaneKind.DIAG_SU,
    PlaneKind.DIAG_SV,
    PlaneKind.SCALE_REFINE,
    PlaneKind.COMPLETION,
    PlaneKind.RELEASE,
)


#: The order a **shard** writes: the same sequence with INITIAL_STATE wedged
#: between the blob planes and the body (schema minor 4).
#:
#: The position is forced, not stylistic.  The order is also the truncation
#: order, and a terminal is a *prefix* of it -- so a plane placed after BODY
#: could be truncated away while the body it governs stayed, and the body
#: would then replay from the pinned zero start and decode to plausible wrong
#: weights.  Ahead of BODY, no legal truncation can separate the two.  It
#: sits after ALPHABET/DESCENDANT for the same reason those lead: nothing
#: decodes without them either, and the state is meaningless without the
#: table the body indexes.
#:
#: A whole unit never writes this plane, so ``CANONICAL_PLANE_ORDER`` is
#: unchanged by it and a whole unit's ``plane_elements`` count array stays
#: nine entries long.
SHARD_PLANE_ORDER: tuple[PlaneKind, ...] = (
    PlaneKind.ALPHABET,
    PlaneKind.DESCENDANT,
    PlaneKind.INITIAL_STATE,
    PlaneKind.BODY,
    PlaneKind.SCALE_BASE,
    PlaneKind.DIAG_SU,
    PlaneKind.DIAG_SV,
    PlaneKind.SCALE_REFINE,
    PlaneKind.COMPLETION,
    PlaneKind.RELEASE,
)


#: The orders every artifact written at minors 0-6 uses (``PlaneLayout.LEGACY``):
#: COMPLETION between SCALE_BASE and DIAG_SU.  Kept verbatim so those bytes
#: keep meaning what they meant, and so a test can write one.  The position
#: was doc S6's: T-po2 = body + po2 base + partial completion, T-C3 adds
#: C-full, T-nvfp4-class adds refinement and release -- a ladder in which a
#: LUT plane's index nibble sat *after* the completion axis, which is what
#: made every completion rung un-truncatable on the wire the recipe table
#: writes (the first obstacle of tessera#144).
LEGACY_PLANE_ORDER: tuple[PlaneKind, ...] = (
    PlaneKind.ALPHABET,
    PlaneKind.DESCENDANT,
    PlaneKind.BODY,
    PlaneKind.SCALE_BASE,
    PlaneKind.COMPLETION,
    PlaneKind.DIAG_SU,
    PlaneKind.DIAG_SV,
    PlaneKind.SCALE_REFINE,
    PlaneKind.RELEASE,
)

LEGACY_SHARD_PLANE_ORDER: tuple[PlaneKind, ...] = (
    PlaneKind.ALPHABET,
    PlaneKind.DESCENDANT,
    PlaneKind.INITIAL_STATE,
    PlaneKind.BODY,
    PlaneKind.SCALE_BASE,
    PlaneKind.COMPLETION,
    PlaneKind.DIAG_SU,
    PlaneKind.DIAG_SV,
    PlaneKind.SCALE_REFINE,
    PlaneKind.RELEASE,
)


def plane_order(
    has_initial_state: bool, layout: PlaneLayout
) -> "tuple[PlaneKind, ...]":
    """The wire/truncation order for a unit: by its state plane and its layout.

    Every consumer that indexes ``TerminalRecord.plane_elements`` positionally
    -- the container's ``plane_ranges`` and ``verify_plane_region``, the
    accountant, the layout builder, the reader -- takes its order from here
    (through ``Manifest.plane_order``, which knows both arguments), so the
    four orders cannot drift apart in one of them.  ``layout`` has no default
    on purpose: a caller that does not know which wire it is on must not be
    handed the current one.
    """
    if PlaneLayout(layout) is PlaneLayout.LADDER:
        return SHARD_PLANE_ORDER if has_initial_state else CANONICAL_PLANE_ORDER
    return LEGACY_SHARD_PLANE_ORDER if has_initial_state else LEGACY_PLANE_ORDER


def layout_of(kinds, has_initial_state: bool) -> PlaneLayout:
    """Which layout a full descriptor sequence is in, or a refusal.

    For a caller holding every plane's descriptor in wire order -- the layout
    builder's output -- the sequence itself says which wire it is on, so
    ``build_terminal`` derives the layout here rather than taking it as a
    second parameter that could disagree with the first.
    """
    kinds = tuple(kinds)
    for layout in PlaneLayout:
        if kinds == plane_order(has_initial_state, layout):
            return layout
    raise PlaneLayoutError(
        f"plane sequence {[kind.name for kind in kinds]} is neither the "
        f"{'shard' if has_initial_state else 'whole-unit'} wire order of "
        f"{[layout.name for layout in PlaneLayout]}"
    )


#: Normative per-plane element width (schema 1a, review finding F3).
#:
#: These widths are fixed by the doc: a stage-B release is 4 bits/position, a
#: C-full completion is 1 bit, an E8M0 base is a byte, a segment-2b refinement
#: word is a nibble.  The table lived in ``layout.py`` and was therefore
#: consulted only when *building* a descriptor, so a decoded or hand-built
#: manifest could declare any width and two conforming decoders would disagree
#: on bytes.  It binds every descriptor now, however constructed.
NORMATIVE_ELEMENT_BITS: "dict[PlaneKind, int]" = {}  # populated below


class IndexDomain(IntEnum):
    """What one element of the plane is indexed by."""

    POSITION = 0  # one weight position
    HALF_BLOCK = 1  # one 16-weight half
    BLOCK = 2  # one 32-weight group
    AXIS_IN = 3  # one input channel
    AXIS_OUT = 4  # one output channel
    WHOLE_UNIT = 5  # an opaque blob for the unit


class Storage(IntEnum):
    INLINE = 0  # bytes live in this artifact's plane region
    REFERENCE = 1  # content-addressed; bytes charged at bundle level


class BitOrder(IntEnum):
    MSB_FIRST = 0
    LSB_FIRST = 1


class CountGranularity(IntEnum):
    WHOLE_PLANE = 0
    PER_SUPERBLOCK = 1
    PER_BLOCK = 2
    #: Minor 7: one granule per completion depth level, the plane laid out
    #: level-major (``wire.pack_levels``).  A COMPLETION plane under
    #: ``PlaneLayout.LADDER`` declares this and nothing else does; a plane
    #: written at depth 0 has no levels and declares no granules.
    PER_LEVEL = 3


class PayloadDtype(IntEnum):
    RAW_BITS = 0
    UINT = 1
    E4M3FN = 2
    E8M0 = 3
    FP16 = 4


#: INITIAL_STATE is deliberately **absent** from this table: its width is the
#: body's state width (``window_bits`` under WINDOW, the convolutional code's
#: memory under TCQ), which is a property of the encoder profile and not of
#: the schema.  A fixed normative width here would either overcharge every
#: shard or be wrong for one of the two bodies.  It is bound instead by
#: ``Manifest.__post_init__``, which asserts the descriptor's ``element_bits``
#: equals the ``state_bits`` its shard record declares, and by
#: ``parse_unit_artifact``, which asserts that width against the body the
#: profile id resolved to -- the same deferred-validation pattern the rate cap
#: uses.
NORMATIVE_ELEMENT_BITS.update(
    {
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
)


@dataclass(frozen=True)
class PlaneDescriptor:
    """One plane's complete self-description.

    `counts` is the per-granule element-count vector -- the quantity canonical
    placement does *not* remove.  `restart_offsets` is the offset/restart table
    that makes segment-local random access possible on the GPU without a host
    parse.
    """

    kind: PlaneKind
    index_domain: IndexDomain
    storage: Storage
    element_bits: int
    bit_order: BitOrder
    alignment_bytes: int
    count_granularity: CountGranularity
    counts: tuple[int, ...]
    restart_offsets: tuple[int, ...]
    payload_dtype: PayloadDtype
    content_digest: bytes

    def __post_init__(self) -> None:
        # Two enum members are refused outright rather than half-supported.
        # Both are constructed nowhere -- by ``layout.py``, by the decoder's
        # callers, by any test -- and both are worse than absent while that
        # holds, because each has a consumer that would quietly agree with it.
        #
        # REFERENCE: ``byte_length`` returned 0 for it and the accountant
        # summed those zeros, so a referenced plane's content was charged
        # nowhere at all (the callerless ``footprint.reference_bundle_bytes``
        # that did the summing is gone with the skips, #24).  An artifact
        # that declared one would have understated its own size and no gate
        # would have noticed.  Fail closed here, at construction, so a hostile
        # or merely mistaken manifest is refused on decode instead of being
        # believed: when bundle-level accounting exists, this refusal is the
        # one line that has to move, and it is the line that says why.
        #
        # LSB_FIRST: ``wire.py`` packs MSB-first unconditionally and
        # ``container.verify_plane_region`` masks pad bits in the low bits, so
        # an LSB_FIRST descriptor names a layout no writer in this package can
        # produce and no verifier here would check correctly.  Honouring it in
        # the verifier alone would be half a feature -- a plane that passes
        # canonicality and decodes to noise.
        if self.storage is not Storage.INLINE:
            raise PlaneLayoutError(
                f"{self.kind.name}: {self.storage.name} storage is not charged "
                "by any accountant in this package; only INLINE planes may be "
                "declared until referenced bytes are counted at bundle level"
            )
        if self.bit_order is not BitOrder.MSB_FIRST:
            raise PlaneLayoutError(
                f"{self.kind.name}: {self.bit_order.name} is not produced by "
                "this package's packer and is not verified by its canonicality "
                "check; the wire is MSB-first"
            )
        if self.element_bits <= 0:
            raise PlaneLayoutError(f"{self.kind.name}: element_bits must be positive")
        normative = NORMATIVE_ELEMENT_BITS.get(self.kind)
        if normative is not None and self.element_bits != normative:
            raise PlaneLayoutError(
                f"{self.kind.name}: element_bits {self.element_bits} contradicts "
                f"the schema's normative width {normative}"
            )
        if self.alignment_bytes <= 0:
            raise PlaneLayoutError(f"{self.kind.name}: alignment must be positive")
        if self.alignment_bytes & (self.alignment_bytes - 1):
            raise PlaneLayoutError(
                f"{self.kind.name}: alignment {self.alignment_bytes} is not a "
                "power of two"
            )
        if any(count < 0 for count in self.counts):
            raise PlaneLayoutError(f"{self.kind.name}: negative element count")
        if len(self.content_digest) != DIGEST_BYTES:
            raise PlaneLayoutError(f"{self.kind.name}: malformed content digest")
        if self.count_granularity is CountGranularity.WHOLE_PLANE:
            if len(self.counts) != 1:
                raise PlaneLayoutError(
                    f"{self.kind.name}: WHOLE_PLANE granularity declares "
                    f"{len(self.counts)} counts, expected exactly 1"
                )
        elif not self.counts and self.count_granularity is not CountGranularity.PER_LEVEL:
            # A superblock-cut plane always spans at least one superblock, so
            # an empty count vector there describes nothing.  A level-cut
            # plane at depth 0 has no levels: the empty vector is its exact
            # description, and ``element_count`` is 0.
            raise PlaneLayoutError(
                f"{self.kind.name}: {self.count_granularity.name} granularity "
                "declares no granules"
            )
        if self.restart_offsets and len(self.restart_offsets) != len(self.counts):
            raise PlaneLayoutError(
                f"{self.kind.name}: restart table has {len(self.restart_offsets)} "
                f"entries for {len(self.counts)} granules"
            )
        # The table is the running prefix sum of `counts` by construction, so it
        # is a derivable that must agree with what it is derived from.  Ascent
        # alone is too weak (a zero-count granule makes two offsets equal, which
        # is legal) and does not bound the offsets at all -- and this table is
        # what a GPU consumer uses for segment-local random access without a
        # host parse (doc S9).  Review finding F5.
        if self.restart_offsets:
            expected, running = [], 0
            for count in self.counts:
                expected.append(running)
                running += count
            if list(self.restart_offsets) != expected:
                raise PlaneLayoutError(
                    f"{self.kind.name}: restart offsets {list(self.restart_offsets)} "
                    f"are not the prefix sums of counts ({expected})"
                )

    @property
    def element_count(self) -> int:
        return sum(self.counts)

    @property
    def payload_bits(self) -> int:
        return self.element_count * self.element_bits

    def byte_length(self, element_count: int | None = None) -> int:
        """Exact physical bytes for this plane, padding and alignment included.

        This is the exact-byte authority for the plane; the accountant and the
        serializer both call it, so they cannot drift.
        """
        count = self.element_count if element_count is None else element_count
        if count < 0:
            raise PlaneLayoutError(f"{self.kind.name}: negative element count")
        if self.storage is not Storage.INLINE:
            # Unreachable: __post_init__ refuses a non-INLINE descriptor.  The
            # branch stays because returning 0 here was the lie itself, and a
            # future bundle-level accountant must replace it with a charge,
            # not inherit a zero.
            raise PlaneLayoutError(
                f"{self.kind.name}: {self.storage.name} storage has no local "
                "byte length; its bytes are charged by the bundle"
            )
        raw = bits_to_bytes(count * self.element_bits)
        remainder = raw % self.alignment_bytes
        return raw if remainder == 0 else raw + (self.alignment_bytes - remainder)

    def encode(self, writer: Writer) -> None:
        (
            writer.uint(int(self.kind))
            .uint(int(self.index_domain))
            .uint(int(self.storage))
            .uint(self.element_bits)
            .uint(int(self.bit_order))
            .uint(self.alignment_bytes)
            .uint(int(self.count_granularity))
            .uint_seq(self.counts)
            .uint_seq(self.restart_offsets)
            .uint(int(self.payload_dtype))
            .digest32(self.content_digest)
        )

    @classmethod
    def decode(cls, reader: Reader) -> "PlaneDescriptor":
        return cls(
            kind=reader.enum(PlaneKind),
            index_domain=reader.enum(IndexDomain),
            storage=reader.enum(Storage),
            element_bits=reader.uint(),
            bit_order=reader.enum(BitOrder),
            alignment_bytes=reader.uint(),
            count_granularity=reader.enum(CountGranularity),
            counts=reader.uint_seq(),
            restart_offsets=reader.uint_seq(),
            payload_dtype=reader.enum(PayloadDtype),
            content_digest=reader.digest32(),
        )
