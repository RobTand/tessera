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

from .canonical import DIGEST_BYTES, Reader, Writer, digest
from .errors import ManifestError
from .exact import Fraction as _Fraction  # re-export guard
from .grammar import (
    RateSchedule,
    bresenham_rate_schedule,
    root_from_q256,
    superblock_quota_ok,
    validate_rate_schedule,
)
from .planes import CANONICAL_PLANE_ORDER, PlaneDescriptor, PlaneKind, Storage

__all__ = [
    "SCHEMA_ID",
    "RotationState",
    "ContainerClass",
    "ArrangementMode",
    "BranchIdentity",
    "Geometry",
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
    GRIDBOOK = 1  # the only consumer of Tessera bytes
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
        if len(self.plane_elements) != len(CANONICAL_PLANE_ORDER):
            raise ManifestError(
                f"terminal {self.slot_id!r}: plane_elements has "
                f"{len(self.plane_elements)} entries, expected "
                f"{len(CANONICAL_PLANE_ORDER)}"
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

    def __post_init__(self) -> None:
        if len(self.encoder_profile_id) != DIGEST_BYTES:
            raise ManifestError("malformed encoder_profile_id")
        if len(self.payload_digest) != DIGEST_BYTES:
            raise ManifestError("malformed payload_digest")
        if not self.terminals:
            raise ManifestError("a manifest declares at least one terminal")

        kinds = [plane.kind for plane in self.planes]
        if len(set(kinds)) != len(kinds):
            raise ManifestError("duplicate plane kind in manifest")
        order = {kind: index for index, kind in enumerate(CANONICAL_PLANE_ORDER)}
        if [order[kind] for kind in kinds] != sorted(order[kind] for kind in kinds):
            raise ManifestError("planes are not in canonical plane order")

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
        extents = [0] * len(CANONICAL_PLANE_ORDER)
        order = {kind: index for index, kind in enumerate(CANONICAL_PLANE_ORDER)}
        for descriptor in self.planes:
            extents[order[descriptor.kind]] = descriptor.element_count

        for terminal in self.terminals:
            truncated = False
            for index, kind in enumerate(CANONICAL_PLANE_ORDER):
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
                if count < extent:
                    truncated = True

    @property
    def schedule(self) -> RateSchedule:
        return RateSchedule(rates=self.rates, root=self.branch.root)

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

    def encode(self) -> bytes:
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
        return writer.bytes

    @classmethod
    def decode(cls, data: bytes) -> "Manifest":
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
        )

    def manifest_digest(self) -> bytes:
        return digest(_DOMAIN_MANIFEST, self.encode())
