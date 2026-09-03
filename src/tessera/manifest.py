"""Manifest, branch identity, and terminal records (doc S7, S9).

Terminal **classes** ({T-po2, T-C3, T-nvfp4-class}) are templates with free
epsilon parameters.  Every actual encoder or allocator candidate carries a
stable ``terminal_id`` whose manifest record holds the complete per-plane count
arrays, the clip-scalar code, the exact physical bytes, and the exact bpp
(Codex round-6 P0-1).  Two encodes with different count arrays are different
terminals, whatever their class.

Identity discipline (round-7 P1-3, round-8 P1-5):

* ``encoder_profile_id`` is **input-only**: it declares a finite ordered set of
  terminal *slots* and contains nothing an encode alone can produce.
* Pass-1 weights index only the stable input ``terminal_slot_id``.
* The post-encode **receipt** maps slot to realised terminal.
* The persisted assignment carries exactly **one** normative representation --
  this structured record -- with the digest computed over it as its hash
  domain.  There is no "or" alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction

from .canonical import DIGEST_BYTES, Reader, Writer, digest, fits_uint
from .errors import ManifestError
from .exact import Fraction as _Fraction  # re-export guard
from .grammar import (
    RateSchedule,
    bresenham_rate_schedule,
    root_from_q256,
    superblock_quota_ok,
    validate_rate_schedule,
)
from .planes import (
    CANONICAL_PLANE_ORDER,
    SHARD_PLANE_ORDER,
    CountGranularity,
    PlaneDescriptor,
    PlaneKind,
    Storage,
    plane_order,
)

__all__ = [
    "SCHEMA_ID",
    "RotationState",
    "ContainerClass",
    "ArrangementMode",
    "BranchIdentity",
    "Geometry",
    "ScalePlaneKind",
    "ScalePlane",
    "BodyKind",
    "WINDOW_BITS_MAX",
    "ShardOrigin",
    "TerminalRecord",
    "Manifest",
]

#: The stream schema, born at build item 1a.
SCHEMA_ID = "prismaquant.tessera.v1"

_DOMAIN_TERMINAL = "prismaquant.tessera.v1/terminal_id"
_DOMAIN_MANIFEST = "prismaquant.tessera.v1/manifest"
_DOMAIN_PROFILE = "prismaquant.tessera.v1/encoder_profile_id"


class RotationState(IntEnum):
    """Serving rotation states (doc S7, Codex round-5 P1-4).

    ``R_in``-only is the sole algebraically local state under the current
    output contract.  Two-sided rotation is a **weight-space measurement
    state**, not a serving branch: its output basis needs an ``R_out^T``
    inverse or proved propagation through every consumer, which is a
    model-level contract rather than a per-unit branch.
    """

    NONE = 0
    R_IN_ONLY = 1


class ContainerClass(IntEnum):
    CT_LEGAL = 0  # compressed-tensors: unrotated, diagonal-free or folded
    GRIDBOOK = 1  # legacy Gridbook-lane name; Tessera's own serving plugin reads these bytes now
    REQUANT_DERIVED = 2  # T-nvfp4-RQ, a separate derived artifact


class ArrangementMode(IntEnum):
    BRESENHAM = 0  # canonical, regenerable from (root, n_columns)
    STORED = 1  # importance-placed; the rate vector is on the wire


@dataclass(frozen=True)
class BranchIdentity:
    """Immutable branch coordinates (doc S7).

    Rotation changes the source tensor, so base bytes are never shared across
    rotation states; container-target classes differ in diagonal legality.
    """

    unit_id: str
    root_q256: int
    rotation: RotationState
    container: ContainerClass

    @property
    def root(self) -> Fraction:
        return root_from_q256(self.root_q256)

    def encode(self, writer: Writer) -> None:
        (
            writer.text(self.unit_id)
            .uint(self.root_q256)
            .uint(int(self.rotation))
            .uint(int(self.container))
        )

    @classmethod
    def decode(cls, reader: Reader) -> "BranchIdentity":
        return cls(
            unit_id=reader.text(),
            root_q256=reader.uint(),
            rotation=RotationState(reader.uint()),
            container=ContainerClass(reader.uint()),
        )


@dataclass(frozen=True)
class Geometry:
    """Declared physical shape. Nothing here is a guessed constant.

    ``superblock_columns``, ``group_weights`` and ``half_weights`` are schema
    parameters carried on the wire, because this package cannot verify a
    Gridbook-side constant from outside that repository.
    """

    rows: int
    columns: int
    superblock_columns: int
    group_weights: int
    half_weights: int
    quantizable_params: int

    def __post_init__(self) -> None:
        for name in ("rows", "columns", "superblock_columns", "group_weights",
                     "half_weights", "quantizable_params"):
            if getattr(self, name) <= 0:
                raise ManifestError(f"geometry.{name} must be positive")
        if self.group_weights % self.half_weights:
            raise ManifestError(
                f"group_weights {self.group_weights} is not a multiple of "
                f"half_weights {self.half_weights}"
            )
        # The denominator of every bpp figure this artifact quotes.  Unbounded
        # above, a wire value of 1e12 validated cleanly and understated the
        # rate by orders of magnitude; the ceiling is the unit's own position
        # count, because a unit cannot hold more quantizable parameters than it
        # holds weights.  Below it is legitimate -- a profile-pinned or
        # otherwise excluded sub-tensor is exactly what the convention exists
        # for (CLAUDE.md principle 12).
        if self.quantizable_params > self.rows * self.columns:
            raise ManifestError(
                f"geometry.quantizable_params {self.quantizable_params} exceeds "
                f"the {self.rows * self.columns} weight positions this unit "
                "holds; every bpp figure divides by it"
            )

    @property
    def positions(self) -> int:
        return self.rows * self.columns

    def encode(self, writer: Writer) -> None:
        (
            writer.uint(self.rows)
            .uint(self.columns)
            .uint(self.superblock_columns)
            .uint(self.group_weights)
            .uint(self.half_weights)
            .uint(self.quantizable_params)
        )

    @classmethod
    def decode(cls, reader: Reader) -> "Geometry":
        return cls(
            rows=reader.uint(),
            columns=reader.uint(),
            superblock_columns=reader.uint(),
            group_weights=reader.uint(),
            half_weights=reader.uint(),
            quantizable_params=reader.uint(),
        )


class ScalePlaneKind(IntEnum):
    """How the segment-2b bytes turn into a per-half scale."""

    #: One E8M0 base byte per group (SCALE_BASE) plus a ``(d, m)`` nibble per
    #: half (SCALE_REFINE): ``2^(E-127+d) * (1 + m/8)``, ``scale_codec``'s S6b.
    S6B = 0
    #: No base plane.  The SCALE_REFINE nibble indexes a per-unit table of up
    #: to sixteen distinct E4M3FN bytes, times one fp32 global: the half's
    #: scale is ``e4m3(table[nibble]) * global``.  Same index granularity as
    #: S6b at half the bytes; the table is chosen per unit by the encoder.
    LUT = 1
    #: One scale per output channel (schema minor 3): the layout an FP8
    #: tensor core consumes.  No SCALE_BASE, no SCALE_REFINE and no DIAG_SU;
    #: the row scale is the DIAG_SV plane -- one fp16 per output row, the
    #: field segment 2a already declares -- times the unit's fp32 global.  A
    #: weight is ``grid_value(code) * global * sv[row]``.  The block planes
    #: carry column structure an E2M1 tile cannot express; an E4M3 tile has
    #: its own exponent and the FP8 MMA takes a per-channel scale, so on that
    #: grid this plane is both cheaper and the served layout
    #: (``scale_channel.py``).
    CHANNEL = 2


class BodyKind(IntEnum):
    """What the BODY plane's bits are, and how a code is recovered from them."""

    #: The shaped trellis: a rate-1/2 convolutional code over four subsets of
    #: the anchor alphabet, ``R`` bits per position (``R + 1`` at a stored
    #: label when ``span > 1``), anchors on the ALPHABET plane and the
    #: completion forest on DESCENDANT.  Every artifact before minor 2.
    TCQ = 0
    #: The window body (schema minor 2).  ``R`` bits per position, and the
    #: code at a position is a table lookup on the last ``window_bits`` bits
    #: of the column's stream: ``state_t = ((state_{t-1} << R) | bits_t) mod
    #: 2^window_bits``, ``state_{-1} = 0``, ``code_t = ALPHABET[state_t]``.
    #: The ALPHABET plane *is* the table (``2^window_bits`` grid codes);
    #: DESCENDANT and COMPLETION are empty; ``span`` is 1.  No convolutional
    #: code, no forest: the redundancy that shapes the reconstruction is the
    #: ``window_bits - R`` bits of history every position shares with its
    #: predecessors (Tseng et al.'s bitshift trellis, on the hardware tile).
    WINDOW = 1


#: The widest window a reader will allocate a table for: a 2^20-entry table
#: is 1 MiB per unit, and the ALPHABET plane is charged per unit, so nothing
#: above this is a rate anyone would ship.  A bound, not a tuning constant.
WINDOW_BITS_MAX = 20


#: The wire writes a global scale as an exact ``Fraction`` through
#: ``canonical.Writer.ratio`` -- a varint numerator and a varint denominator --
#: so both terms must fit the codec's unsigned domain.  This is NOT the same
#: constraint as "exactly representable as a float": ``Fraction(3.7e-5)`` is
#: float-exact and has a 68-bit denominator, so it passes that check and then
#: fails inside the codec with a 21-digit integer and no mention of a scale
#: (#33).  Refuse it here, where the field has a name.  The bound itself is
#: the codec's to state -- ``canonical.fits_uint`` -- because a second copy of
#: it here would be a second thing to forget to change.
def _require_wire_ratio(field: str, value: Fraction) -> None:
    if fits_uint(value.numerator) and fits_uint(value.denominator):
        return
    raise ManifestError(
        f"the {field} {float(value)!r} is not writable to the wire: it encodes "
        f"as the exact ratio {value.numerator}/{value.denominator}, whose "
        f"{'numerator' if not fits_uint(value.numerator) else 'denominator'} "
        f"needs {max(value.numerator, value.denominator).bit_length()} bits and "
        "the canonical codec's varints hold 64.  A scale reaches this state by "
        "being a float that is not a dyadic rational of modest denominator -- "
        "the shipped planes snap the global to a power of two, which encodes in "
        "a handful of bytes.  Snap it, or carry the residue in the per-row or "
        "per-entry scale instead of the global."
    )


@dataclass(frozen=True)
class ScalePlane:
    """The scale plane's kind and, for a LUT plane, its table and global.

    The table travels here rather than in a plane because it is a *unit-level*
    parameter with no positional index -- the same reason the group and half
    sizes live in ``Geometry`` -- and because a terminal must stay a prefix of
    the plane order.  It is charged: manifest bytes are side bytes, which the
    accountant reports in ``wire_bpp``.
    """

    kind: ScalePlaneKind
    table: bytes = b""
    global_scale: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        if self.kind is ScalePlaneKind.S6B:
            if self.table or self.global_scale != 1:
                raise ManifestError("an S6b scale plane carries no table or global")
            return
        if self.kind is ScalePlaneKind.CHANNEL:
            if self.table:
                raise ManifestError("a CHANNEL scale plane carries no table; its rows are the DIAG_SV plane")
            if self.global_scale <= 0:
                raise ManifestError("the CHANNEL global scale must be positive")
            if Fraction(float(self.global_scale)) != self.global_scale:
                raise ManifestError("the CHANNEL global scale must be exactly representable as a float")
            _require_wire_ratio("CHANNEL global scale", self.global_scale)
            return
        if not 2 <= len(self.table) <= 16:
            raise ManifestError(
                f"a LUT scale plane holds 2..16 entries, got {len(self.table)}"
            )
        # Positive NORMAL E4M3FN: bytes 0x08..0x7E.  Subnormals (0x01..0x07)
        # are excluded because the kernel lane decodes the materialised byte
        # as 2^(e-7)(1+m/8) -- the S6b relabelling's contract -- which is
        # wrong at exponent field 0.  Bytes are monotone in value over the
        # range, so strictly ascending bytes are strictly ascending, distinct
        # scales -- the canonical order.
        if any(not 0x08 <= byte <= 0x7E for byte in self.table):
            raise ManifestError(
                "LUT entries must be positive normal E4M3FN bytes (0x08..0x7E)"
            )
        if any(a >= b for a, b in zip(self.table, self.table[1:])):
            raise ManifestError("LUT entries must be strictly ascending")
        if self.global_scale <= 0:
            raise ManifestError("the LUT global scale must be positive")
        if float(self.global_scale) == 0.0 or Fraction(float(self.global_scale)) != self.global_scale:
            raise ManifestError(
                "the LUT global scale must be exactly representable as a float"
            )
        _require_wire_ratio("LUT global scale", self.global_scale)

    @classmethod
    def s6b(cls) -> "ScalePlane":
        return cls(ScalePlaneKind.S6B)

    @classmethod
    def lut(cls, table: bytes, global_scale: float) -> "ScalePlane":
        return cls(ScalePlaneKind.LUT, bytes(table), Fraction(float(global_scale)))

    @classmethod
    def channel(cls, global_scale: float) -> "ScalePlane":
        return cls(ScalePlaneKind.CHANNEL, b"", Fraction(float(global_scale)))

    def encode(self, writer: Writer) -> None:
        writer.uint(int(self.kind))
        if self.kind is ScalePlaneKind.LUT:
            writer.blob(self.table).ratio(self.global_scale)
        elif self.kind is ScalePlaneKind.CHANNEL:
            writer.ratio(self.global_scale)

    @classmethod
    def decode(cls, reader: Reader) -> "ScalePlane":
        raw = reader.uint()
        try:
            kind = ScalePlaneKind(raw)
        except ValueError:
            raise ManifestError(f"unknown scale-plane kind {raw}") from None
        if kind is ScalePlaneKind.S6B:
            return cls(kind)
        if kind is ScalePlaneKind.CHANNEL:
            return cls(kind, b"", reader.ratio())
        table = reader.blob()
        return cls(kind, bytes(table), reader.ratio())


@dataclass(frozen=True)
class ShardOrigin:
    """Where a unit sits inside the unit it was cut from (schema minor 4).

    A Tessera artifact is **tensor-parallel by construction**: the exporter
    writes one whole unit and never learns the TP degree, and every rank cuts
    its own shard out of those bytes at load (``layout.slice_unit``).  This
    record is what makes a shard a first-class unit rather than a fragment --
    it is a complete, self-describing artifact that decodes on its own, and
    this says which rows and columns of which parent it decodes *to*.

    ``state_bits`` is the width of one INITIAL_STATE element: the window width
    under a WINDOW body, the convolutional code's memory under TCQ.  It is
    zero exactly when ``row_offset`` is zero, because a column cut at row 0
    starts from the pinned zero state the decoder already assumes -- which is
    why a shard that only slices columns carries no state plane at all, and
    why the identity slice is byte-identical to its parent.

    ``parent_digest`` is the parent manifest's ``manifest_digest``.  It is
    provenance, not a decode input: two ranks holding two shards can prove
    they came from one artifact without either holding the other's bytes.
    """

    row_offset: int
    col_offset: int
    parent_rows: int
    parent_columns: int
    parent_digest: bytes
    state_bits: int = 0

    def __post_init__(self) -> None:
        for name in ("row_offset", "col_offset", "parent_rows", "parent_columns",
                     "state_bits"):
            if getattr(self, name) < 0:
                raise ManifestError(f"shard.{name} must not be negative")
        if self.parent_rows <= 0 or self.parent_columns <= 0:
            raise ManifestError("a shard names a parent with positive extent")
        if len(self.parent_digest) != DIGEST_BYTES:
            raise ManifestError("malformed parent digest")
        if bool(self.state_bits) != bool(self.row_offset):
            raise ManifestError(
                f"shard declares row_offset {self.row_offset} and state_bits "
                f"{self.state_bits}: a shard cut below row 0 carries its start "
                "state, and one cut at row 0 carries none"
            )

    @property
    def has_initial_state(self) -> bool:
        return self.row_offset > 0

    def encode(self, writer: Writer) -> None:
        (
            writer.uint(self.row_offset)
            .uint(self.col_offset)
            .uint(self.parent_rows)
            .uint(self.parent_columns)
            .digest32(self.parent_digest)
            .uint(self.state_bits)
        )

    @classmethod
    def decode(cls, reader: Reader) -> "ShardOrigin":
        return cls(
            row_offset=reader.uint(),
            col_offset=reader.uint(),
            parent_rows=reader.uint(),
            parent_columns=reader.uint(),
            parent_digest=reader.digest32(),
            state_bits=reader.uint(),
        )


@dataclass(frozen=True)
class TerminalRecord:
    """One concrete, exactly-priced terminal.

    `plane_elements` is the per-plane element count *in canonical plane order*
    -- the "complete per-plane count arrays" of Codex round-6 P0-1.  A plane
    absent from this terminal carries zero.
    """

    slot_id: str
    clip_exponent_code: int
    plane_elements: tuple[int, ...]
    exact_bytes: int
    exact_bpp: Fraction
    payload_digest: bytes

    def __post_init__(self) -> None:
        # Either wire order: nine entries for a whole unit, ten for a shard,
        # whose extra entry is the INITIAL_STATE plane.  Which of the two this
        # record means is not the record's to know -- ``Manifest`` owns that
        # and checks the length against its own order.
        if len(self.plane_elements) not in (
            len(CANONICAL_PLANE_ORDER), len(SHARD_PLANE_ORDER)
        ):
            raise ManifestError(
                f"terminal {self.slot_id!r}: plane_elements has "
                f"{len(self.plane_elements)} entries, expected "
                f"{len(CANONICAL_PLANE_ORDER)} or {len(SHARD_PLANE_ORDER)}"
            )
        if any(count < 0 for count in self.plane_elements):
            raise ManifestError(f"terminal {self.slot_id!r}: negative plane count")
        if self.exact_bytes < 0:
            raise ManifestError(f"terminal {self.slot_id!r}: negative byte count")
        if len(self.payload_digest) != DIGEST_BYTES:
            raise ManifestError(
                f"terminal {self.slot_id!r}: malformed payload digest"
            )
        if not 0 <= self.clip_exponent_code < 8:
            raise ManifestError(
                f"terminal {self.slot_id!r}: clip exponent code "
                f"{self.clip_exponent_code} outside the declared 3-bit domain"
            )

    def _identity_bytes(self, branch: BranchIdentity, profile_id: bytes) -> bytes:
        writer = Writer()
        writer.text(SCHEMA_ID).digest32(profile_id)
        branch.encode(writer)
        (
            writer.text(self.slot_id)
            .uint(self.clip_exponent_code)
            .uint_seq(self.plane_elements)
            .uint(self.exact_bytes)
            .ratio(self.exact_bpp)
            .digest32(self.payload_digest)
        )
        return writer.bytes

    def terminal_id(self, branch: BranchIdentity, profile_id: bytes) -> bytes:
        """Content-addressed terminal identity.

        Bound to the branch and the encoder profile, so the same count array
        under a different branch or profile is a different terminal.
        """
        return digest(_DOMAIN_TERMINAL, self._identity_bytes(branch, profile_id))

    def encode(self, writer: Writer) -> None:
        (
            writer.text(self.slot_id)
            .uint(self.clip_exponent_code)
            .uint_seq(self.plane_elements)
            .uint(self.exact_bytes)
            .ratio(self.exact_bpp)
            .digest32(self.payload_digest)
        )

    @classmethod
    def decode(cls, reader: Reader) -> "TerminalRecord":
        return cls(
            slot_id=reader.text(),
            clip_exponent_code=reader.uint(),
            plane_elements=reader.uint_seq(),
            exact_bytes=reader.uint(),
            exact_bpp=reader.ratio(),
            payload_digest=reader.digest32(),
        )


@dataclass(frozen=True)
class Manifest:
    """The closed manifest at the artifact boundary (doc S12)."""

    encoder_profile_id: bytes
    branch: BranchIdentity
    geometry: Geometry
    arrangement: ArrangementMode
    rates: tuple[int, ...]
    planes: tuple[PlaneDescriptor, ...]
    terminals: tuple[TerminalRecord, ...]
    payload_digest: bytes
    # Schema minor 1 (2026-09-01).  ``span`` is the trellis super-symbol
    # length; ``scale_plane`` says how segment 2b decodes.  A minor-0 artifact
    # carries neither and means ``(1, S6B)``, which is what these default to,
    # so every artifact written before the fields existed reads back unchanged
    # -- and ``encode`` writes a minor-0 manifest whenever that is all there is
    # to say, so re-serialising one is byte-identical too.
    span: int = 1
    scale_plane: ScalePlane = ScalePlane(ScalePlaneKind.S6B)
    # Schema minor 2 (2026-09-02).  ``body`` says what the BODY bits are and
    # ``window_bits`` is the window body's state width (0 under TCQ).  Same
    # discipline as minor 1: a TCQ manifest carries neither field and writes
    # at the minor it needed before they existed.
    body: BodyKind = BodyKind.TCQ
    window_bits: int = 0
    # Schema minor 4 (2026-09-02).  ``shard`` is present only on a unit cut
    # out of another with ``layout.slice_unit``; a whole unit carries None and
    # writes at the minor it needed before the field existed, so every
    # artifact ever written is byte-identical across this bump.
    shard: "ShardOrigin | None" = None

    @property
    def plane_order(self) -> "tuple[PlaneKind, ...]":
        """This manifest's wire order -- the order its counts are indexed by."""
        return plane_order(self.shard is not None and self.shard.has_initial_state)

    @property
    def schema_minor(self) -> int:
        """The lowest schema minor that expresses this manifest.

        Minor 3 (2026-09-02) adds no field: it is the ``CHANNEL`` value of
        the minor-1 scale-plane record, which a minor-1 or minor-2 reader
        cannot resolve, so a manifest carrying it declares the minor that
        can.  Minor 4 (2026-09-02) appends the shard record, and a shard cut
        below row 0 also changes the plane order, so an earlier reader must
        not try.
        """
        if self.shard is not None:
            return 4
        if self.scale_plane.kind is ScalePlaneKind.CHANNEL:
            return 3
        if self.body is BodyKind.WINDOW:
            return 2
        legacy = self.span == 1 and self.scale_plane.kind is ScalePlaneKind.S6B
        return 0 if legacy else 1

    def __post_init__(self) -> None:
        if len(self.encoder_profile_id) != DIGEST_BYTES:
            raise ManifestError("malformed encoder_profile_id")
        if self.span < 1:
            raise ManifestError(f"span must be positive, got {self.span}")
        if self.body is BodyKind.WINDOW:
            if self.span != 1:
                raise ManifestError(
                    f"a window body has no super-symbols; span must be 1, got {self.span}"
                )
            if not 1 <= self.window_bits <= WINDOW_BITS_MAX:
                raise ManifestError(
                    f"window_bits {self.window_bits} outside 1..{WINDOW_BITS_MAX}"
                )
        elif self.window_bits:
            raise ManifestError("window_bits is only meaningful under a window body")
        if len(self.payload_digest) != DIGEST_BYTES:
            raise ManifestError("malformed payload_digest")
        if not self.terminals:
            raise ManifestError("a manifest declares at least one terminal")

        kinds = [plane.kind for plane in self.planes]
        if len(set(kinds)) != len(kinds):
            raise ManifestError("duplicate plane kind in manifest")
        wire = self.plane_order
        order = {kind: index for index, kind in enumerate(wire)}
        if any(kind not in order for kind in kinds):
            stray = [kind.name for kind in kinds if kind not in order]
            raise ManifestError(
                f"plane {stray} has no place in this unit's wire order; an "
                "INITIAL_STATE plane belongs to a shard cut below row 0 and "
                "to nothing else"
            )
        if [order[kind] for kind in kinds] != sorted(order[kind] for kind in kinds):
            raise ManifestError("planes are not in canonical plane order")
        self._validate_shard(wire)

        if len(self.rates) != self.geometry.columns:
            raise ManifestError(
                f"rate schedule covers {len(self.rates)} columns, geometry "
                f"declares {self.geometry.columns}"
            )
        # ``cap=None``: the rate ceiling belongs to the payload grid, which is
        # committed in ``encoder_profile_id`` and resolved only after this
        # manifest validates.  Asserting TESSERA-4's cap here would refuse
        # every legal rung of every other family; carrying the cap as a second
        # wire field would let it disagree with the grid.  The bound is applied
        # against the real grid at forest rebuild, before any decode.
        validate_rate_schedule(self.rates, self.branch.root, cap=None)
        if self.body is BodyKind.WINDOW and self.window_bits < max(self.rates):
            raise ManifestError(
                f"window_bits {self.window_bits} cannot hold a rate-"
                f"{max(self.rates)} position's bits"
            )
        if not superblock_quota_ok(
            self.rates, self.geometry.superblock_columns, self.branch.root
        ):
            raise ManifestError("a complete superblock violates the rate quota")
        if self.arrangement is ArrangementMode.BRESENHAM:
            canonical = bresenham_rate_schedule(
                self.branch.root, self.geometry.columns, cap=None
            )
            if self.rates != canonical:
                raise ManifestError(
                    "arrangement declares BRESENHAM but the rate vector is not "
                    "the canonical schedule"
                )

        slots = [terminal.slot_id for terminal in self.terminals]
        if len(set(slots)) != len(slots):
            raise ManifestError("duplicate terminal_slot_id")
        sizes = [terminal.exact_bytes for terminal in self.terminals]
        if len(set(sizes)) != len(sizes):
            raise ManifestError(
                "two terminals declare the same exact_bytes: a truncation length "
                "must identify exactly one terminal"
            )
        self._validate_terminal_prefixes()

    def _validate_shard(self, wire: "tuple[PlaneKind, ...]") -> None:
        """A shard's geometry, its state plane and its parent must agree.

        The state plane's *width* is checked here against the shard record;
        that the width is the right one for the **body** is checked in
        ``parse_unit_artifact``, after the profile id has resolved the
        convolutional code -- the manifest cannot know the code's memory order,
        for the same reason it defers the rate cap to the payload grid.
        """
        state = self.plane(PlaneKind.INITIAL_STATE)
        if self.shard is None:
            if state is not None:
                raise ManifestError(
                    "an INITIAL_STATE plane without a shard record: nothing "
                    "says which rows of what this state starts"
                )
            return
        shard = self.shard
        if shard.row_offset + self.geometry.rows > shard.parent_rows:
            raise ManifestError(
                f"shard rows [{shard.row_offset}, "
                f"{shard.row_offset + self.geometry.rows}) run past a parent of "
                f"{shard.parent_rows} rows"
            )
        if shard.col_offset + self.geometry.columns > shard.parent_columns:
            raise ManifestError(
                f"shard columns [{shard.col_offset}, "
                f"{shard.col_offset + self.geometry.columns}) run past a parent "
                f"of {shard.parent_columns} columns"
            )
        if not shard.has_initial_state:
            return
        if state is None:
            raise ManifestError(
                f"shard starts at row {shard.row_offset} but declares no "
                "INITIAL_STATE plane; a body replayed from the pinned zero "
                "start would decode to plausible wrong weights"
            )
        if state.element_bits != shard.state_bits:
            raise ManifestError(
                f"the INITIAL_STATE plane is {state.element_bits} bits wide, "
                f"the shard record declares {shard.state_bits}"
            )
        if state.element_count != self.geometry.columns:
            raise ManifestError(
                f"the INITIAL_STATE plane holds {state.element_count} entries "
                f"for {self.geometry.columns} columns: one state per column"
            )
        if self.body is BodyKind.WINDOW and shard.state_bits != self.window_bits:
            raise ManifestError(
                f"a window body's start state is its {self.window_bits}-bit "
                f"window; the shard declares {shard.state_bits}"
            )

    def _validate_terminal_prefixes(self) -> None:
        """Every terminal must be a genuine **prefix** of the plane region.

        Review findings F2 and F8.  The canonical plane order is also the
        truncation order, and `container.parse` hands a matched byte length back
        as that terminal's whole plane region -- so a terminal's declared counts
        are only meaningful if the bytes they describe really are the leading
        bytes of the artifact.  That requires, in canonical order: full planes,
        then at most one partially-present plane, then nothing.

        A terminal shaped (full, empty, full) prices to a real byte count and
        would match a real truncation length, but the bytes at that length are
        not the bytes it describes.  The accountant catches an *over*-claim
        (`footprint.plane_region_bytes`); nothing caught the shape.
        """
        wire = self.plane_order
        extents = [0] * len(wire)
        order = {kind: index for index, kind in enumerate(wire)}
        # The quota boundaries a granular plane may be cut at: its running
        # prefix sums, 0 and the full extent included.  ``planes.py`` states
        # the rule -- "the last non-empty plane [is] cut at a per-superblock
        # quota boundary" -- and nothing enforced it, so a terminal could name
        # a count that falls in the middle of a granule and price it exactly.
        # A WHOLE_PLANE plane has no granule structure and is deliberately not
        # bound: a whole unit's RELEASE counts are respread by the reader.
        boundaries: "dict[int, set[int]]" = {}
        for descriptor in self.planes:
            extents[order[descriptor.kind]] = descriptor.element_count
            if descriptor.count_granularity in (
                CountGranularity.PER_SUPERBLOCK,
                CountGranularity.PER_BLOCK,
            ):
                allowed, running = {0}, 0
                for count in descriptor.counts:
                    running += count
                    allowed.add(running)
                boundaries[order[descriptor.kind]] = allowed

        for terminal in self.terminals:
            if len(terminal.plane_elements) != len(wire):
                raise ManifestError(
                    f"terminal {terminal.slot_id!r} counts "
                    f"{len(terminal.plane_elements)} planes, this unit's wire "
                    f"order has {len(wire)}"
                )
            truncated = False
            for index, kind in enumerate(wire):
                count = terminal.plane_elements[index]
                extent = extents[index]
                if count > extent:
                    raise ManifestError(
                        f"terminal {terminal.slot_id!r} claims {count} elements "
                        f"of {kind.name}, which declares only {extent}"
                    )
                if truncated and count:
                    raise ManifestError(
                        f"terminal {terminal.slot_id!r} is not a prefix: "
                        f"{kind.name} carries {count} elements after an earlier "
                        "plane was left incomplete"
                    )
                allowed = boundaries.get(index)
                if allowed is not None and count not in allowed:
                    raise ManifestError(
                        f"terminal {terminal.slot_id!r} cuts {kind.name} at "
                        f"{count} elements, which is not a per-superblock quota "
                        f"boundary of {sorted(allowed)}"
                    )
                if count < extent:
                    truncated = True

    @property
    def schedule(self) -> RateSchedule:
        """This unit's rate schedule, at the cap the manifest is entitled to.

        ``cap=None`` -- the same deferral ``__post_init__`` applies when it
        calls ``validate_rate_schedule``.  The ceiling belongs to the payload
        grid, which is committed in ``encoder_profile_id`` and resolved only
        after this manifest validates; asserting E2M1's cap of 3 here made the
        property raise on every real E4M3 unit, whose rates run to 5.
        """
        return RateSchedule(rates=self.rates, root=self.branch.root, cap=None)

    def plane(self, kind: PlaneKind) -> PlaneDescriptor | None:
        for descriptor in self.planes:
            if descriptor.kind is kind:
                return descriptor
        return None

    def terminal_ids(self) -> dict[str, bytes]:
        return {
            terminal.slot_id: terminal.terminal_id(
                self.branch, self.encoder_profile_id
            )
            for terminal in self.terminals
        }

    def encode(self, schema_minor: "int | None" = None) -> bytes:
        """Canonical bytes.  ``schema_minor`` defaults to the lowest that fits.

        Asking for minor 0 on a manifest that needs minor 1 is refused rather
        than silently dropping the fields: a reader given those bytes would
        decode a span-2 body as span 1 and produce plausible garbage.
        """
        minor = self.schema_minor if schema_minor is None else schema_minor
        if minor < self.schema_minor:
            raise ManifestError(
                f"schema minor {minor} cannot express a {self.body.name} body at "
                f"span {self.span} with a {self.scale_plane.kind.name} scale "
                f"plane"
                + (" on a shard" if self.shard is not None else "")
                + f"; needs minor {self.schema_minor}"
            )
        writer = Writer()
        writer.text(SCHEMA_ID).digest32(self.encoder_profile_id)
        self.branch.encode(writer)
        self.geometry.encode(writer)
        writer.uint(int(self.arrangement))
        # A Bresenham arrangement is regenerable, so it is not stored twice.
        if self.arrangement is ArrangementMode.STORED:
            writer.uint_seq(self.rates)
        writer.uint(len(self.planes))
        for descriptor in self.planes:
            descriptor.encode(writer)
        writer.uint(len(self.terminals))
        for terminal in self.terminals:
            terminal.encode(writer)
        writer.digest32(self.payload_digest)
        if minor >= 1:
            writer.uint(self.span)
            self.scale_plane.encode(writer)
        if minor >= 2:
            writer.uint(int(self.body)).uint(self.window_bits)
        if minor >= 4:
            writer.uint(1 if self.shard is not None else 0)
            if self.shard is not None:
                self.shard.encode(writer)
        return writer.bytes

    @classmethod
    def decode(cls, data: bytes, schema_minor: int = 0) -> "Manifest":
        """Parse canonical bytes written at ``schema_minor``.

        The minor is the container header's, not something the manifest can
        discover about itself: the minor-1 fields follow the payload digest
        and the minor-2 fields follow those, so a reader stopping early would
        leave trailing bytes, which ``finish`` refuses.  The default is 0
        because that is what every caller that predates the field means.
        """
        reader = Reader(data)
        schema = reader.text()
        if schema != SCHEMA_ID:
            raise ManifestError(f"foreign schema id {schema!r}")
        profile_id = reader.digest32()
        branch = BranchIdentity.decode(reader)
        geometry = Geometry.decode(reader)
        arrangement = ArrangementMode(reader.uint())
        if arrangement is ArrangementMode.STORED:
            rates = reader.uint_seq()
        else:
            # cap=None for the same reason as in __post_init__: the rate
            # ceiling lives on the payload grid, which this parser has not yet
            # resolved.  The schedule itself is cap-independent -- Bresenham
            # only ever mixes the two rates bracketing the root -- so deferring
            # the bound changes which artifacts are *accepted here*, never
            # which schedule is reconstructed.
            rates = bresenham_rate_schedule(
                branch.root, geometry.columns, cap=None
            )
        planes = tuple(
            PlaneDescriptor.decode(reader) for _ in range(reader.uint())
        )
        terminals = tuple(
            TerminalRecord.decode(reader) for _ in range(reader.uint())
        )
        payload_digest = reader.digest32()
        span, scale_plane = 1, ScalePlane(ScalePlaneKind.S6B)
        body, window_bits = BodyKind.TCQ, 0
        if schema_minor >= 1:
            span = reader.uint()
            scale_plane = ScalePlane.decode(reader)
            if scale_plane.kind is ScalePlaneKind.CHANNEL and schema_minor < 3:
                raise ManifestError(
                    f"a CHANNEL scale plane needs schema minor 3; the header says {schema_minor}"
                )
        if schema_minor >= 2:
            body = BodyKind(reader.uint())
            window_bits = reader.uint()
        shard = None
        if schema_minor >= 4:
            if reader.uint():
                shard = ShardOrigin.decode(reader)
        reader.finish()
        return cls(
            encoder_profile_id=profile_id,
            branch=branch,
            geometry=geometry,
            arrangement=arrangement,
            rates=rates,
            planes=planes,
            terminals=terminals,
            payload_digest=payload_digest,
            span=span,
            scale_plane=scale_plane,
            body=body,
            window_bits=window_bits,
            shard=shard,
        )

    def manifest_digest(self) -> bytes:
        return digest(_DOMAIN_MANIFEST, self.encode())
