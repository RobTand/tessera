"""The refinement grammar (doc S6), decoder-level and exact.

A root ``r0`` fixes a per-column schedule over integer rates ``R in {1,2,3}``
under an exact Bresenham quota.  Each column's base alphabet ``A_R`` has
``|A_R| = 2**(R+1)`` codes.  Stage C (completion) spends ``c <= 3 - R`` bits
per column against a stored descendant map; at ``c = 3 - R`` the descendant
sets **partition** the 16-code grid.  Stage B (release) replaces a position's
code with any of 16 at a cost of 4 bits.

Two facts this module proves rather than asserts:

* C-full costs ``R + (3 - R) = 3`` bits per column from *every* root, so
  completion equalises all roots at the joint-16 wire.
* The partition property is forced by cardinality:
  ``|A_R| * 2**(3-R) == 2**(R+1) * 2**(3-R) == 16`` for every legal ``R``.

What this module deliberately does **not** do: define the rate-1/rate-2
set-partitioning alphabet convention.  That is build item 2 and is explicitly
owed; alphabets and descendant maps are supplied as stored blobs and validated
structurally only.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .errors import GrammarError

__all__ = [
    "NATIVE_CODE_BITS",
    "GRID_CODES",
    "LEGAL_RATES",
    "RELEASE_BITS",
    "C_FULL_BITS",
    "alphabet_size",
    "completion_capacity",
    "descendant_set_size",
    "bresenham_rate_schedule",
    "validate_rate_schedule",
    "superblock_quota_ok",
    "bits_per_position",
    "prefix_cardinality",
    "validate_descendant_map",
    "root_from_q256",
    "q256_from_root",
    "RateSchedule",
]

#: E2M1 is a 4-bit native code: 16 codes on the terminal grid.
NATIVE_CODE_BITS = 4
GRID_CODES = 1 << NATIVE_CODE_BITS  # 16

#: Shaped trellis rates. ``max_trellis_rate = native - 1`` (doc S15,
#: `trellis.py:112-113,182`), so the shaped body tops out at 3.
LEGAL_RATES = (1, 2, 3)

#: A released position stores a full 16-code override (doc S6).
RELEASE_BITS = 4

#: Payload bits per column at C-full, from any root.
C_FULL_BITS = 3

#: The q256 parameterisation: r0 = q256 / 256 (doc S5).
Q256_UNIT = 256


def alphabet_size(rate: int, cap: int = C_FULL_BITS) -> int:
    """``|A_R| = 2**(R+1)`` exhaustively optimised codes."""
    _check_rate(rate, cap)
    return 1 << (rate + 1)


def completion_capacity(rate: int, cap: int = C_FULL_BITS) -> int:
    """Maximum completion level ``c = cap - R`` for a column at this rate.

    ``cap`` is the payload grid's own width minus one: 3 over E2M1's 16 codes,
    7 over E4M3's 256.  It is a parameter rather than a constant because
    TESSERA-4 and TESSERA-8 are the *same* construction at two grid widths --
    ``2^(R+1) * 2^(cap-R) = 2^(cap+1)`` closes at every rate either way.
    """
    _check_rate(rate, cap)
    return cap - rate


def completion_widths(
    rates: "tuple[int, ...]", cap: int = C_FULL_BITS, limit: "int | None" = None
) -> "tuple[int, ...]":
    """Bits the COMPLETION plane spends per column at an encoded depth.

    ``completion_capacity`` is a *ceiling* -- what the rate leaves room for.
    The encoder spends ``min(limit, capacity)`` (``encode_unit``), so these are
    two different numbers whenever the unit was encoded shallower than its rate
    allows, and the plane must be sized by the one that was actually written.
    ``limit=None`` means "as deep as each rate allows" and reproduces the
    ceiling exactly, which is why every full-depth artifact is unaffected.
    """
    caps = tuple(completion_capacity(rate, cap) for rate in rates)
    if limit is None:
        return caps
    if limit < 0:
        raise GrammarError(f"completion limit {limit} is negative")
    return tuple(min(limit, c) for c in caps)


def completion_limit_from_elements(
    elements: int, rates: "tuple[int, ...]", steps: int, cap: int = C_FULL_BITS
) -> "int | None":
    """Recover the depth a unit was written at from its declared plane size.

    The COMPLETION plane's element count is already on the wire -- the terminal
    records one per plane -- and ``sum(min(limit, cap - R))`` is monotone in
    ``limit``, so the depth is recoverable without a new schema field.  A reader
    that instead assumed the ceiling would mis-slice every shallow unit.
    """
    caps = [completion_capacity(rate, cap) for rate in rates]
    ceiling = max(caps, default=0)
    if steps <= 0 or not caps:
        return None if elements == 0 else _unrecoverable(elements, steps)
    for limit in range(ceiling + 1):
        if sum(min(limit, c) for c in caps) * steps == elements:
            return None if limit == ceiling else limit
    return _unrecoverable(elements, steps)


def _unrecoverable(elements: int, steps: int):
    raise GrammarError(
        f"COMPLETION declares {elements} elements over {steps} steps, which no "
        f"completion depth over this rate schedule produces. The plane, the "
        f"terminal and the rate schedule disagree; refusing to guess a depth "
        f"and mis-slice the plane."
    )


def descendant_set_size(completion: int) -> int:
    """``|D(a)| = 2**c`` -- the per-position reachable set after completion."""
    if completion < 0:
        raise GrammarError(f"negative completion level: {completion}")
    return 1 << completion


def _check_rate(rate: int, cap: "int | None" = C_FULL_BITS) -> None:
    """Bound a rate.  ``cap=None`` defers the *upper* bound, deliberately.

    The upper bound is a property of the payload grid (``rate_cap =
    payload_bits - 1``), and one caller legitimately does not have the grid
    yet: the manifest parser.  A grid is committed in ``encoder_profile_id``
    and recovered *after* the manifest validates, so requiring a cap there
    would mean either inventing one (silently refusing every rung above 3) or
    duplicating the grid's cap as a second wire field that could disagree with
    it.  Deferring is not dropping: ``AnchorForest.__post_init__`` applies the
    real cap the moment the grid is resolved, and it does so before any code is
    decoded, so nothing reaches a weight on an unbounded rate.
    """
    if rate < 1:
        raise GrammarError(f"rate {rate} is below the shaped domain (min 1)")
    if cap is None:
        return
    legal = LEGAL_RATES if cap == C_FULL_BITS else tuple(range(1, cap + 1))
    if rate not in legal:
        raise GrammarError(
            f"rate {rate} outside the shaped domain {legal} "
            "(max_trellis_rate = native - 1)"
        )


def root_from_q256(q256: int) -> Fraction:
    """Root rate ``r0`` for a q256 parameter."""
    if q256 <= 0:
        raise GrammarError(f"q256 must be positive: {q256}")
    return Fraction(q256, Q256_UNIT)


def q256_from_root(root: Fraction) -> int:
    """Inverse of :func:`root_from_q256`; raises if not an integral q256.

    The domain is the same one ``root_from_q256`` accepts.  It was not: this
    direction mapped a zero root to ``0``, which the other direction rejects,
    so the round trip broke at exactly the value a mis-derived root lands on.
    """
    scaled = root * Q256_UNIT
    if scaled.denominator != 1:
        raise GrammarError(f"root {root} does not land on an integral q256")
    if scaled <= 0:
        raise GrammarError(f"root {root} is not positive: q256 must be positive")
    return int(scaled)


@dataclass(frozen=True)
class RateSchedule:
    """A per-column rate assignment realising a root exactly.

    ``cap=None`` defers the upper bound to whoever resolves the payload grid,
    the same deferral ``_check_rate`` documents.  A manifest holds a schedule
    it cannot yet bound -- its grid is committed in ``encoder_profile_id`` and
    resolved after validation -- so a schedule built at the E2M1 cap could not
    describe a real E4M3 unit at all.
    """

    rates: tuple[int, ...]
    root: Fraction
    cap: "int | None" = C_FULL_BITS

    @property
    def total_body_bits_per_row(self) -> int:
        """Body bits contributed by one row across all columns."""
        return sum(self.rates)

    def __post_init__(self) -> None:
        validate_rate_schedule(self.rates, self.root, self.cap)


def bresenham_rate_schedule(
    root: Fraction, n_columns: int, cap: "int | None" = C_FULL_BITS
) -> tuple[int, ...]:
    """Canonical exact quota for ``root`` over ``n_columns`` columns.

    The schedule mixes only the two rates bracketing the root, and the count at
    the upper rate is exactly ``n_columns * (root - floor(root))`` -- which must
    be an integer, or the root is not realisable at this column count.
    Placement is Bresenham (evenly distributed, deterministic).

    Importance-placed arrangements are also legal provided every complete
    superblock keeps the quota (doc S6); see :func:`superblock_quota_ok`.

    ``cap`` is the family's rate cap -- ``payload_bits - 1``.  It defaults to
    ``C_FULL_BITS`` because that is TESSERA-4's, and every artifact built
    before families existed was TESSERA-4.  It is a real parameter, not a
    formality: a root above 3 is *ordinary* on E4M3 (cap 7) and on any k=2
    grid (cap 7), and defaulting it silently refused every rung those families
    exist to reach.
    """
    if n_columns <= 0:
        raise GrammarError(f"n_columns must be positive: {n_columns}")

    lower = int(root) if root.denominator == 1 else root.numerator // root.denominator
    upper = lower if root.denominator == 1 else lower + 1
    _check_rate(lower, cap)
    if upper != lower:
        _check_rate(upper, cap)

    exact_upper_count = (root - lower) * n_columns
    if exact_upper_count.denominator != 1:
        raise GrammarError(
            f"root {root} is not realisable over {n_columns} columns: "
            f"it needs {exact_upper_count} columns at rate {upper}"
        )
    n_upper = int(exact_upper_count)

    # Bresenham: column i takes the upper rate when the accumulated ideal count
    # crosses an integer boundary. Deterministic and evenly spread.
    schedule = []
    accumulator = 0
    for _ in range(n_columns):
        accumulator += n_upper
        if accumulator >= n_columns:
            accumulator -= n_columns
            schedule.append(upper)
        else:
            schedule.append(lower)
    return tuple(schedule)


def validate_rate_schedule(
    rates: tuple[int, ...], root: Fraction, cap: "int | None" = C_FULL_BITS
) -> None:
    """Raise unless every rate is legal for ``cap`` and the quota is exact."""
    if not rates:
        raise GrammarError("empty rate schedule")
    for rate in rates:
        _check_rate(rate, cap)
    total = sum(rates)
    exact = root * len(rates)
    if exact.denominator != 1 or total != int(exact):
        raise GrammarError(
            f"inexact quota: schedule sums to {total} bits, "
            f"root {root} over {len(rates)} columns requires {exact}"
        )


def superblock_quota_ok(
    rates: tuple[int, ...], superblock_columns: int, root: Fraction
) -> bool:
    """True iff every *complete* superblock keeps the quota (doc S6).

    A trailing partial superblock is not required to keep it; only complete
    superblocks are constrained, which is what makes importance placement legal.
    """
    if superblock_columns <= 0:
        raise GrammarError(f"superblock_columns must be positive: {superblock_columns}")
    per_superblock = root * superblock_columns
    if per_superblock.denominator != 1:
        raise GrammarError(
            f"root {root} does not yield an integral quota over "
            f"{superblock_columns} columns"
        )
    target = int(per_superblock)
    n_complete = len(rates) // superblock_columns
    for index in range(n_complete):
        block = rates[index * superblock_columns : (index + 1) * superblock_columns]
        if sum(block) != target:
            return False
    return True


def bits_per_position(
    rate: int, completion: int, released: bool = False, cap: int = C_FULL_BITS
) -> int:
    """Payload bits for one position: ``R + c`` plus 4 if released.

    Release-everywhere costs ``3 + 4 = 7`` bits per column, which is never
    byte-competitive with scalar 4.5 -- so scalar rate-4 is not a Tessera
    endpoint (doc S6).
    """
    _check_rate(rate, cap)
    if not 0 <= completion <= completion_capacity(rate, cap):
        raise GrammarError(
            f"completion {completion} exceeds capacity "
            f"{completion_capacity(rate, cap)} at rate {rate} (cap {cap})"
        )
    return rate + completion + (RELEASE_BITS if released else 0)


def prefix_cardinality(
    rate: int, completion: int, cap: int = C_FULL_BITS
) -> int:
    """Per-position reachable-set size after ``completion`` bits.

    Nesting: this is ``2**c`` at every prefix, and reaches the full 16-code
    grid jointly (not per position) exactly at ``c = 3 - R``.
    """
    _check_rate(rate, cap)
    if not 0 <= completion <= completion_capacity(rate, cap):
        raise GrammarError(
            f"completion {completion} exceeds capacity "
            f"{completion_capacity(rate, cap)} at rate {rate} (cap {cap})"
        )
    return descendant_set_size(completion)


def validate_descendant_map(
    rate: int,
    completion: int,
    descendant_map: dict[int, tuple[int, ...]],
    cap: int = C_FULL_BITS,
) -> None:
    """Validate a stored descendant map structurally (doc S6).

    Checks, in order: the map is keyed by exactly the alphabet's anchors; every
    descendant set has size ``2**c``; every descendant is a legal grid code;
    and, at ``c = cap - R`` only, the descendant sets **partition** the grid --
    every code is a descendant of exactly one anchor.

    ``cap`` is the payload grid's width minus one, the same parameter
    ``completion_capacity`` takes: 3 over E2M1's 16 codes, 7 over E4M3's 256.
    The grid's size is **derived** from it rather than passed alongside it --
    ``2^(R+1) * 2^(cap-R) == 2^(cap+1)`` is the cardinality identity this
    module proves, so a second parameter could only disagree with the first.
    Hardcoding cap 3 and 16 codes here refused every legal rate of TESSERA-8.

    The alphabet's *content* is not validated: the rate-1/rate-2
    set-partitioning convention is build item 2 and is not defined here.
    """
    _check_rate(rate, cap)
    capacity = completion_capacity(rate, cap)
    grid_codes = 1 << (cap + 1)
    if not 0 <= completion <= capacity:
        raise GrammarError(
            f"completion {completion} exceeds capacity {capacity} at rate {rate}"
        )

    expected_anchors = alphabet_size(rate, cap)
    if len(descendant_map) != expected_anchors:
        raise GrammarError(
            f"descendant map has {len(descendant_map)} anchors, "
            f"rate {rate} requires {expected_anchors}"
        )
    if set(descendant_map) != set(range(expected_anchors)):
        raise GrammarError("descendant map anchors must be exactly 0..|A_R|-1")

    expected_size = descendant_set_size(completion)
    seen: dict[int, int] = {}
    for anchor, descendants in sorted(descendant_map.items()):
        if len(descendants) != expected_size:
            raise GrammarError(
                f"anchor {anchor}: |D(a)| = {len(descendants)}, expected "
                f"{expected_size} at completion {completion}"
            )
        if len(set(descendants)) != len(descendants):
            raise GrammarError(f"anchor {anchor}: duplicate descendants")
        for code in descendants:
            if not 0 <= code < grid_codes:
                raise GrammarError(
                    f"anchor {anchor}: code {code} off the "
                    f"{grid_codes}-code grid"
                )
            if code in seen:
                raise GrammarError(
                    f"code {code} is a descendant of both anchor {seen[code]} "
                    f"and anchor {anchor}: descendant sets must be disjoint"
                )
            seen[code] = anchor

    if completion == capacity and len(seen) != grid_codes:
        raise GrammarError(
            f"at c = cap - R = {capacity} the descendant sets must partition "
            f"the {grid_codes}-code grid; they cover {len(seen)}"
        )
