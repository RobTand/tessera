"""Persisted identity, the disjoint parser, and name novelty (doc S9).

Round-7 P1-7 and round-8 P1-5 set the rules this module implements:

* ``TESSERA_E2M1_R{q256}`` / ``TESSERA_E4M3_R{q256}`` is a **human-readable
  family/root descriptor only**.  It is not an assignment, and it never
  stands in for a terminal.
* The persisted assignment carries **one** normative representation: the
  structured record of schema, ``encoder_profile_id``, ``terminal_id``, branch
  identity, and payload digest, with a digest over that record as its hash
  domain.  There is no "or" alternative.
* The Tessera record type is **disjoint** from the legacy one and fails closed
  wherever a concrete terminal record is required.
* Legacy ``TCQ_*`` names and their parser stay immutable; the legacy two-tuple
  parser never silently accepts a Tessera artifact.

The legacy grammar is mirrored here **only** so the collision tests have
something to run against.  The real legacy parser lives in PrismaQuant and is
not touched; this mirror is non-normative and deliberately accepts exactly the
documented ``TCQ_*`` two-tuple language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .canonical import DIGEST_BYTES, Writer, digest
from .errors import IdentityError
from .manifest import SCHEMA_ID, BranchIdentity

__all__ = [
    "FamilyDescriptor",
    "parse_family_descriptor",
    "legacy_tcq_parse",
    "AssignmentRecord",
    "require_terminal_record",
    "check_name_novelty",
]

_TESSERA_DESCRIPTOR = re.compile(r"\ATESSERA_(E2M1|E4M3)_R(\d+)\Z")
_LEGACY_TCQ = re.compile(r"\ATCQ_([A-Z0-9]+)_R(\d+)\Z")

_DOMAIN_ASSIGNMENT = "prismaquant.tessera.v1/assignment"


@dataclass(frozen=True)
class FamilyDescriptor:
    """A human-readable family/root label. Carries no normative weight."""

    payload: str  # "E2M1" (TESSERA-4) or "E4M3" (TESSERA-8)
    q256: int

    def __str__(self) -> str:
        return f"TESSERA_{self.payload}_R{self.q256}"

    @property
    def is_normative(self) -> bool:
        """Always False. A descriptor is never a persisted identity."""
        return False


def parse_family_descriptor(text: str) -> FamilyDescriptor:
    """Parse a Tessera family descriptor. Rejects every legacy name.

    This parser's accepted language is disjoint from the legacy ``TCQ_*``
    language by construction: the two prefixes cannot both match.
    """
    match = _TESSERA_DESCRIPTOR.match(text)
    if not match:
        raise IdentityError(f"not a Tessera family descriptor: {text!r}")
    return FamilyDescriptor(payload=match.group(1), q256=int(match.group(2)))


def legacy_tcq_parse(text: str) -> tuple[str, int]:
    """Non-normative mirror of the documented legacy ``TCQ_*`` grammar.

    Present only so the disjointness tests have a counterparty.  Returns the
    legacy two-tuple; raises on anything else, Tessera names included.
    """
    match = _LEGACY_TCQ.match(text)
    if not match:
        raise IdentityError(f"not a legacy TCQ name: {text!r}")
    return (match.group(1), int(match.group(2)))


@dataclass(frozen=True)
class AssignmentRecord:
    """The one normative persisted representation (round-8 P1-5).

    A `FamilyDescriptor` may accompany this record as a human label, but it is
    never a substitute for it.
    """

    encoder_profile_id: bytes
    terminal_id: bytes
    branch: BranchIdentity
    payload_digest: bytes
    schema_id: str = SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != SCHEMA_ID:
            raise IdentityError(f"foreign schema id {self.schema_id!r}")
        for name in ("encoder_profile_id", "terminal_id", "payload_digest"):
            if len(getattr(self, name)) != DIGEST_BYTES:
                raise IdentityError(f"malformed {name}")

    def record_digest(self) -> bytes:
        """Digest over this record, which is its hash domain."""
        writer = Writer()
        writer.text(self.schema_id).digest32(self.encoder_profile_id).digest32(
            self.terminal_id
        )
        self.branch.encode(writer)
        writer.digest32(self.payload_digest)
        return digest(_DOMAIN_ASSIGNMENT, writer.bytes)


def require_terminal_record(candidate: object) -> AssignmentRecord:
    """Fail closed where a concrete terminal record is required.

    A `FamilyDescriptor` -- or a bare string that merely *looks* like one -- is
    rejected here, which is the whole point of the disjoint record type.
    """
    if isinstance(candidate, AssignmentRecord):
        return candidate
    if isinstance(candidate, FamilyDescriptor):
        raise IdentityError(
            f"{candidate} is a human-readable family descriptor, not a terminal "
            "record; a concrete AssignmentRecord is required here (round-7 P1-7)"
        )
    if isinstance(candidate, str):
        raise IdentityError(
            f"{candidate!r} is a name, not a terminal record; names never carry "
            "normative identity"
        )
    raise IdentityError(f"not a Tessera assignment record: {type(candidate).__name__}")


def check_name_novelty(existing_names) -> None:
    """Reject a collision between the Tessera family prefix and a known name.

    The name-novelty check that lands with the 1a sweep.  Supply the registry's
    current format names; any name that would be captured by the Tessera
    descriptor grammar is a collision.
    """
    collisions = []
    for name in existing_names:
        try:
            parse_family_descriptor(name)
        except IdentityError:
            continue
        collisions.append(name)
    if collisions:
        raise IdentityError(
            f"name-novelty check failed: {sorted(collisions)} already occupy the "
            "Tessera descriptor grammar"
        )
