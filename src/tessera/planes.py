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
    "NORMATIVE_ELEMENT_BITS",
    "SHARD_PLANE_ORDER",
    "plane_order",
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


#: Wire order, which is also the truncation order (schema 1a decision D5).
#:
#: A terminal is a prefix of this sequence, with the last non-empty plane cut at
#: a per-superblock quota boundary.  The order is forced by the terminal classes
#: of doc S6: T-po2 is body + po2 base + partial completion, T-C3 adds C-full,
#: and T-nvfp4-class adds the refinement and release planes on top.  The two
#: blob planes lead because nothing else decodes without them.
CANONICAL_PLANE_ORDER: tuple[PlaneKind, ...] = (
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
#: unchanged and every artifact written before this schema minor is
#: byte-identical -- including its ``plane_elements`` count array, which stays
#: nine entries long.
SHARD_PLANE_ORDER: tuple[PlaneKind, ...] = (
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


def plane_order(has_initial_state: bool) -> "tuple[PlaneKind, ...]":
    """The wire/truncation order for a unit, by whether it carries a state.

    Every consumer that indexes ``TerminalRecord.plane_elements`` positionally
    -- the container's ``plane_ranges`` and ``verify_plane_region``, the
    accountant, the layout builder, the reader -- takes its order from here,
    so the two orders cannot drift apart in one of them.
    """
    return SHARD_PLANE_ORDER if has_initial_state else CANONICAL_PLANE_ORDER


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
        elif not self.counts:
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
