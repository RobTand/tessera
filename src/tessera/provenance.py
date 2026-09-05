"""Input-closure provenance: ancestry and denylist (doc S14, build item 11).

The ledger invariant is absolute: *no prohibited-source artifact participates
in Tessera measurement, fitting, allocation, encoding, export, or validation --
in inputs or in citations.  Retaining one as a "prior", a "historical note", or
a "removed row" is still use.*

Round-8 P0-1.6 requires a **content-addressed ancestry check**, because a text
denylist alone is insufficient: relabelling or copying does not change
identity, so an identity must be matched by content digest through the whole
transitive input closure, not by name.

**This module ships the mechanism, not the list.**  The prohibited identities
are recorded in the round-7 review file, which is outside this project's
declared input scope.  The denylist is therefore loaded from an
owner-supplied file and is **empty by default** -- and
:func:`assert_denylist_populated` exists so a caller can refuse to certify a
closure that was checked against nothing.  An empty denylist silently passing
everything is exactly the failure this check exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ProvenanceError

__all__ = [
    "DIGEST_ALPHABET",
    "DIGEST_LENGTH",
    "InputNode",
    "Denylist",
    "load_denylist",
    "check_closure",
    "assert_denylist_populated",
    "require_digest",
]

#: The identity is a SHA-256 content digest (:func:`content_digest`), so the
#: shape it has to have is the hash's own and not a number anyone chose: two
#: hex characters per digest byte (AGENTS.md rule 2).
DIGEST_LENGTH = hashlib.sha256().digest_size * 2
DIGEST_ALPHABET = "0123456789abcdef"


def require_digest(digest, *, field: str = "digest") -> str:
    """The one shape a content identity has, wherever a digest enters.

    Every digest in this module is compared to every other by equality, so a
    string that is not a possible :func:`content_digest` output can never
    match anything -- and a check against an identity nothing can carry is not
    a check.  It is refused where it enters instead: :class:`InputNode` did
    this for its own digest and :func:`load_denylist` did not, so a configured
    entry of sixty-four ``z``\\ s certified a denylist as *populated* while
    matching no possible input (tessera#226).  One rule, one home (AGENTS.md
    rule 4).
    """
    if (
        not isinstance(digest, str)
        or len(digest) != DIGEST_LENGTH
        or any(character not in DIGEST_ALPHABET for character in digest)
    ):
        raise ProvenanceError(
            f"{field} must be {DIGEST_LENGTH} lowercase hex characters: {digest!r}"
        )
    return digest


@dataclass(frozen=True)
class InputNode:
    """One node of the transitive input closure.

    `parents` are the digests this artifact was derived from, which is what
    makes the check transitive: a clean-looking artifact with a prohibited
    ancestor is still prohibited.  A parent is an identity by the same rule
    the node's own digest is, and is validated by the same function: a parent
    nothing can carry names no node, so the closure it claims cannot close.
    """

    digest: str
    kind: str
    label: str = ""
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_digest(self.digest)
        for parent in self.parents:
            require_digest(parent, field="parent digest")


@dataclass
class Denylist:
    """Content-addressed prohibited identities."""

    digests: set[str] = field(default_factory=set)
    source: str = "<empty>"

    def __len__(self) -> int:
        return len(self.digests)


def load_denylist(path: str | Path) -> Denylist:
    """Load an owner-supplied denylist of SHA-256 content digests.

    Case is normalised -- an owner may write a digest in either case -- and
    then every entry is held to :func:`require_digest`, the same identity
    :class:`InputNode` is held to.  Checking only the *length*, as this did
    until tessera#226, let a 64-character non-hex entry through: it can match
    no possible input, yet it satisfied :func:`assert_denylist_populated`,
    which exists precisely to refuse a closure checked against an ineffective
    list.

    One malformed entry refuses the whole file rather than being dropped: the
    owner's record is not this module's to edit, and a list that lost a row
    silently is a list nobody checked against.
    """
    path = Path(path)
    if not path.exists():
        raise ProvenanceError(f"denylist file not found: {path}")
    payload = json.loads(path.read_text())
    digests = set()
    for entry in payload.get("prohibited_digests", []):
        normalised = entry.lower() if isinstance(entry, str) else entry
        digests.add(require_digest(normalised, field="denylist digest"))
    return Denylist(digests=digests, source=str(path))


def assert_denylist_populated(denylist: Denylist) -> None:
    """Refuse to certify a closure checked against an empty denylist."""
    if not denylist.digests:
        raise ProvenanceError(
            "denylist is empty: a closure checked against no prohibited "
            "identities is not a passing check. Populate it from the owner's "
            "record before certifying any Tessera input closure."
        )


def check_closure(nodes, denylist: Denylist) -> None:
    """Walk the transitive input closure and reject any prohibited ancestor.

    Every node must be reachable and every parent must be present: a dangling
    parent means the closure is not closed, and an unverifiable closure fails
    rather than passes.
    """
    index = {node.digest: node for node in nodes}
    if len(index) != len(list(nodes)):
        raise ProvenanceError("duplicate digest in the input closure")

    for node in index.values():
        for parent in node.parents:
            if parent not in index:
                raise ProvenanceError(
                    f"input closure is not closed: {node.kind} {node.digest[:12]} "
                    f"cites absent parent {parent[:12]}"
                )

    for node in index.values():
        if node.digest in denylist.digests:
            raise ProvenanceError(
                f"prohibited source in closure: {node.kind} "
                f"{node.label or node.digest[:12]} matches a denylisted identity"
            )

    # Transitive reachability: a clean leaf with a prohibited ancestor is
    # prohibited. Relabelling or copying does not change identity.
    for node in index.values():
        seen: set[str] = set()
        stack = list(node.parents)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in denylist.digests:
                raise ProvenanceError(
                    f"prohibited ancestor of {node.kind} "
                    f"{node.label or node.digest[:12]}: {current[:12]}"
                )
            stack.extend(index[current].parents)


def content_digest(data: bytes) -> str:
    """SHA-256 hex of raw bytes -- the identity used throughout this module."""
    return hashlib.sha256(data).hexdigest()
