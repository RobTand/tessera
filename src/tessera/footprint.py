"""The exact-byte accountant and the four byte quantities (doc S7).

Doc S6 draws the line this module enforces: "Provisional arithmetic is
admissible for headers and padding, not for body bits."  Padding and alignment
are still *counted* exactly here -- they are physical bytes -- but nothing is
estimated.  The accountant is the single authority; the serializer and the
parser both defer to it, so a disagreement is a defect rather than a rounding
difference.

Four byte quantities are named and charged separately, because conflating them
is how a bundle figure gets quoted as a checkpoint rate:

1. **canonical bundle** -- every branch of a unit. Never sub-4, never a
   checkpoint rate.
2. **selected prefix** -- the serving artifact the DP emits. *The only
   quantity a sub-4 claim ever attaches to.*
3. **encoded resident** -- wire bytes resident in the Gridbook lane.
4. **expanded resident** -- materialised tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from .errors import FootprintDisagreementError
from .manifest import Manifest, TerminalRecord

__all__ = [
    "ByteQuantity",
    "FootprintReport",
    "plane_region_bytes",
    "terminal_payload_bpp",
    "account_terminal",
    "BppClaim",
]


class ByteQuantity(Enum):
    """The four quantities of doc S7.4, kept apart by construction."""

    CANONICAL_BUNDLE = "canonical_bundle"
    SELECTED_PREFIX = "selected_prefix"
    ENCODED_RESIDENT = "encoded_resident"
    EXPANDED_RESIDENT = "expanded_resident"

    @property
    def may_carry_sub4_claim(self) -> bool:
        """Only the selected prefix may carry a sub-4 bpp claim."""
        return self is ByteQuantity.SELECTED_PREFIX


@dataclass(frozen=True)
class BppClaim:
    """A bits-per-parameter figure that knows which quantity it describes.

    Refuses cross-quantity comparison, and refuses to be read as a sub-4 claim
    unless it is the selected prefix.  bpp is over **quantizable parameters
    only** (doc S14).
    """

    quantity: ByteQuantity
    value: Fraction
    quantizable_params: int

    def as_sub4_claim(self) -> Fraction:
        if not self.quantity.may_carry_sub4_claim:
            raise FootprintDisagreementError(
                f"{self.quantity.value} may not carry a sub-4 bpp claim; only "
                "the selected serving artifact may (doc S7.4)"
            )
        return self.value

    def compare_to(self, other: "BppClaim") -> Fraction:
        if self.quantity is not other.quantity:
            raise FootprintDisagreementError(
                f"refusing to compare {self.quantity.value} against "
                f"{other.quantity.value}: different byte quantities"
            )
        if self.quantizable_params != other.quantizable_params:
            raise FootprintDisagreementError(
                "refusing to compare bpp figures over different quantizable "
                "parameter counts"
            )
        return self.value - other.value


def plane_region_bytes(manifest: Manifest, terminal: TerminalRecord) -> int:
    """Exact plane-region bytes for one terminal.

    A terminal is a prefix of the canonical plane order; a plane absent from
    this terminal carries a zero count and contributes nothing.
    """
    order = {kind: index for index, kind in enumerate(manifest.plane_order)}
    total = 0
    for descriptor in manifest.planes:
        count = terminal.plane_elements[order[descriptor.kind]]
        if count > descriptor.element_count:
            raise FootprintDisagreementError(
                f"terminal {terminal.slot_id!r} claims {count} elements of "
                f"{descriptor.kind.name}, which declares {descriptor.element_count}"
            )
        total += descriptor.byte_length(count)
    return total


def terminal_payload_bpp(manifest: Manifest, terminal: TerminalRecord) -> Fraction:
    """Exact payload bits per quantizable parameter for this terminal.

    Schema 1a decision D6: ``TerminalRecord.exact_bpp`` is the **plane-region**
    rate.  Header and manifest side bytes are real and are reported by
    :func:`account_terminal` as ``wire_bpp``; they are deliberately not folded
    into the stored figure, because the manifest's own size depends on the
    terminal records it contains and a self-referential figure could not be
    computed.
    """
    return Fraction(8 * terminal.exact_bytes, manifest.geometry.quantizable_params)


@dataclass(frozen=True)
class FootprintReport:
    """Three-way agreement between declared, recomputed, and physical bytes."""

    slot_id: str
    declared_bytes: int
    recomputed_bytes: int
    physical_bytes: int | None
    side_bytes: int
    payload_bpp: Fraction
    wire_bpp: Fraction

    @property
    def agrees(self) -> bool:
        if self.declared_bytes != self.recomputed_bytes:
            return False
        return self.physical_bytes in (None, self.recomputed_bytes)


def account_terminal(
    manifest: Manifest,
    terminal: TerminalRecord,
    side_bytes: int,
    physical_bytes: int | None = None,
) -> FootprintReport:
    """Recompute a terminal's exact bytes and check every declared figure.

    Raises :class:`FootprintDisagreementError` on any disagreement.  This is the
    "exact agreement between physical bytes and the footprint accountant" that
    build item 1b owes.
    """
    recomputed = plane_region_bytes(manifest, terminal)
    if recomputed != terminal.exact_bytes:
        raise FootprintDisagreementError(
            f"terminal {terminal.slot_id!r}: declared {terminal.exact_bytes} "
            f"plane-region bytes, accountant computes {recomputed}"
        )
    if physical_bytes is not None and physical_bytes != recomputed:
        raise FootprintDisagreementError(
            f"terminal {terminal.slot_id!r}: {physical_bytes} physical bytes, "
            f"accountant computes {recomputed}"
        )

    payload_bpp = terminal_payload_bpp(manifest, terminal)
    if payload_bpp != terminal.exact_bpp:
        raise FootprintDisagreementError(
            f"terminal {terminal.slot_id!r}: declared bpp {terminal.exact_bpp}, "
            f"accountant computes {payload_bpp}"
        )
    wire_bpp = Fraction(
        8 * (recomputed + side_bytes), manifest.geometry.quantizable_params
    )
    return FootprintReport(
        slot_id=terminal.slot_id,
        declared_bytes=terminal.exact_bytes,
        recomputed_bytes=recomputed,
        physical_bytes=physical_bytes,
        side_bytes=side_bytes,
        payload_bpp=payload_bpp,
        wire_bpp=wire_bpp,
    )
