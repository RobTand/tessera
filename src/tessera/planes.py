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
        if self.storage is Storage.REFERENCE:
            return 0
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
            kind=PlaneKind(reader.uint()),
            index_domain=IndexDomain(reader.uint()),
            storage=Storage(reader.uint()),
            element_bits=reader.uint(),
            bit_order=BitOrder(reader.uint()),
            alignment_bytes=reader.uint(),
            count_granularity=CountGranularity(reader.uint()),
            counts=reader.uint_seq(),
            restart_offsets=reader.uint_seq(),
            payload_dtype=PayloadDtype(reader.uint()),
            content_digest=reader.digest32(),
        )
