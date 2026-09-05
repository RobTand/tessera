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
from typing import TYPE_CHECKING

from .errors import GrammarError

if TYPE_CHECKING:  # ``alphabet`` imports this module, so the grid type is a
    from .alphabet import PayloadGrid  # noqa: F401  # name here, not an import.

__all__ = [
    "NATIVE_CODE_BITS",
    "GRID_CODES",
    "LEGAL_RATES",
    "RELEASE_BITS",
    "C_FULL_BITS",
    "alphabet_size",
    "completion_capacity",
    "completion_level_counts",
    "forest_plane_bytes",
    "descendant_set_size",
    "bresenham_rate_schedule",
    "rate_set",
    "validate_rate_schedule",
    "superblock_quota_ok",
    "require_column_groups",
    "superblock_count",
    "superblock_widths",
    "release_quota",
    "release_defined_on",
    "require_release_defined",
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


def forest_plane_bytes(
    rates: "tuple[int, ...]", cap: int = C_FULL_BITS
) -> "tuple[int, int]":
    """``(ALPHABET, DESCENDANT)`` bytes a TCQ body's forest costs on the wire.

    ``unit_artifact._forest_planes`` concatenates one alphabet and one
    descendant block per *distinct* rate in the schedule, so the two plane
    lengths are ``sum 2^(R+1)`` and ``sum 2^(R+1) * 2^(cap-R)`` -- the second
    being ``2^(cap+1)`` per distinct rate, whatever the rate is.  Both are a
    function of ``(rates, cap)`` alone, which is the point: the forest's
    *contents* are an exhaustive search, but its *size* is arithmetic, and an
    accountant that skipped it because the contents are not derivable was
    charging a TCQ unit less than the wire does.  Measured: exactly the gap
    between :func:`tessera.control.unit_wire_bits` and
    ``encode_linear(...).exact_bytes`` on every TCQ rung tested -- 512 B at the
    E2M1x2 coset cap, 20-44 B on arity-1 E2M1.

    A window body has no forest and is not priced here: its table is the
    ALPHABET plane and ``terminal_rate`` already charges it.
    """
    present = sorted(set(int(rate) for rate in rates))
    alphabet = sum(alphabet_size(rate, cap) for rate in present)
    descendant = sum(
        alphabet_size(rate, cap) << completion_capacity(rate, cap) for rate in present
    )
    return alphabet, descendant


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


def completion_level_counts(
    widths: "tuple[int, ...]", steps: int
) -> "tuple[int, ...]":
    """Per-level element counts of a level-major COMPLETION plane (minor 7).

    Level ``l`` holds one bit -- bit ``l`` of the position's completion word,
    counted from the most significant -- for every column whose width reaches
    ``l``, at every trellis step.  The running prefix sum through level ``l``
    is therefore ``sum(min(l, c_j)) * steps``: exactly the count
    :func:`completion_limit_from_elements` already inverts, which is what lets
    a terminal cut at a level boundary declare its depth with no new field.
    A plane written at depth 0 has no levels and declares no granules.
    """
    if steps < 0:
        raise GrammarError(f"negative step count: {steps}")
    if any(width < 0 for width in widths):
        raise GrammarError(f"negative completion width in {widths}")
    depth = max(widths, default=0)
    return tuple(
        steps * sum(1 for width in widths if width >= level)
        for level in range(1, depth + 1)
    )


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


def rate_set(root: Fraction, cap: "int | None" = C_FULL_BITS) -> tuple[int, ...]:
    """The distinct rates :func:`bresenham_rate_schedule` emits for ``root``.

    Ascending, and independent of the column count: a schedule mixes only the
    two rates bracketing the root, and both appear whenever the root is not an
    integer (the fractional part is in ``(0, 1)``, so neither count can be
    zero at any column count the root is realisable over).  An integral root
    is one rate for every column.

    It exists because the rate axis is a PLAN-TIME fact and the column count
    is not.  A gate that must answer "can this lane read a unit at this rung?"
    is asked before the checkpoint's shapes are read -- and a lane whose
    kernel reads a fixed set of column widths (``kernel_window_gemv.
    SUPPORTED_RATES``) is a constraint on exactly this set.  Deriving it here
    rather than in the gate keeps one authority: ``tests/test_rate_set.py``
    checks it against ``bresenham_rate_schedule`` itself over every root the
    schedule realises.
    """
    lower = root.numerator // root.denominator
    _check_rate(lower, cap)
    if root.denominator == 1:
        return (lower,)
    _check_rate(lower + 1, cap)
    return (lower, lower + 1)


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


def require_column_groups(cols: int, half: int) -> None:
    """``cols`` must be a whole number of ``half``-sized scale groups.

    The block scale plane is ``[cols // half, rows]``, so the remainder has no
    group of its own: a GEMV walking ``cols // half`` groups never reaches
    those columns, and the GEMM, which indexes by ``k // half``, reaches one
    group past the plane.  Neither raises on its own.

    This lives here, not beside the kernel that first needed it, because it is
    a fact about the **wire** and every stage that touches the wire owes it:
    the writer (``unit_artifact.build_unit_artifact``), the materialiser
    (``decode.materialize_nvfp4``) and the kernel lane all refuse the same
    widths with the same words, because there is one rule and one place it is
    written (RobTand/tessera#56).

    A ``CHANNEL`` plane is exempt by construction, not by tolerance: its scale
    is one word per output row and there is no column group to be whole, so
    the caller checks the plane kind before calling this.
    """
    if half <= 0 or cols % half:
        raise GrammarError(
            f"{cols} columns is not a whole number of {half}-groups; the scale "
            "plane has no group for the remainder, so the last "
            f"{cols % half if half > 0 else cols} columns would leave the dot "
            "product (GEMV) or index one group past the plane (GEMM)"
        )


def superblock_count(n_columns: int, superblock_columns: int) -> int:
    """How many superblocks a unit of ``n_columns`` columns has.

    A **ceiling**, and it is the only answer: ``block_of = column //
    superblock_columns`` maps the trailing partial columns to a block index of
    their own, so a floor names fewer blocks than the position map produces.
    ``superblock_quota_ok`` already declares that trailing partial superblock
    legal (it constrains only *complete* ones), and ``layout.build_planes``
    already gives it a granule.  Everything that partitions a unit by
    superblock -- the plane granules, the release quota, the restart table --
    counts them here, so no two of them can ever disagree about how many
    there are.

    ``max(1, ...)`` covers the shard case ``n_columns < superblock_columns``,
    where the ceiling is 1 anyway; it survives only so a zero-column argument
    cannot silently produce an empty partition.
    """
    if superblock_columns <= 0:
        raise GrammarError(f"superblock_columns must be positive: {superblock_columns}")
    return max(1, -(-n_columns // superblock_columns))


def superblock_widths(n_columns: int, superblock_columns: int) -> tuple[int, ...]:
    """How many columns each superblock actually holds.

    The companion to :func:`superblock_count`: the count says how many
    granules the partition has, this says how big each one is.  They are one
    fact, so they live together -- a caller that ceilings the count and then
    assumes every granule is ``superblock_columns`` wide has re-floored the
    partition by another route.

    Every superblock spans the same rows, so a superblock's *width* is the
    only thing that varies across the partition: its share of any per-position
    quantity -- positions, release slots -- is its share of the columns.
    """
    if n_columns <= 0:
        raise GrammarError(f"n_columns must be positive: {n_columns}")
    blocks = superblock_count(n_columns, superblock_columns)
    return tuple(
        min(superblock_columns, n_columns - index * superblock_columns)
        for index in range(blocks)
    )


def release_quota(
    total: int, n_columns: int, superblock_columns: int
) -> tuple[int, ...]:
    """Per-superblock release counts: ``total`` at a uniform release density.

    A release is a per-*position* object, and a superblock's positions are
    ``rows * width`` -- the rows are common to every superblock, so the exact
    share of a superblock is ``total * width / n_columns``.  That exact share
    is the objective; this returns the integer vector nearest to it that still
    sums to ``total``, which is the largest-remainder award: floor every share,
    then hand the leftover to the largest fractional parts, lowest superblock
    index first.  No superblock is more than one release from its own exact
    share, which is the width-proportional reading of the promise the equal
    count spread used to make against every *other* superblock.

    Why width-proportional and not equal-count.  ``layout._superblock_counts``
    already argues this for BODY and COMPLETION -- "a granule's count has to be
    the bits that granule's columns actually carry" -- and RELEASE is the same
    partition.  Equal-count is that argument's special case, correct exactly
    when every superblock is complete, and on a trailing partial one it asks a
    narrow block for a release *density* up to ``superblock_columns`` times the
    rest of the unit.  It can therefore ask for more releases than the block
    has positions: on a 64x257 unit an equal-count quota overran at a total of
    130 of 16448 positions, capping the whole unit at 0.79% released.  Under
    this quota that overrun cannot happen at all: ``total <= positions``
    implies ``count <= rows * width`` for every superblock, because
    ``floor(total * width / n_columns) <= rows * width`` with equality only
    when ``total == positions``, where the leftover is zero.

    Why largest-remainder and not a cumulative-floor Bresenham.  Both are
    exact and both are width-proportional; they differ in *which* blocks take
    the leftover, and only one of them reduces to the spread already on the
    wire.  At equal widths every share is ``total / blocks`` and every
    fractional part is equal, so the tie-break awards the leftover to the
    lowest indices -- the ``divmod`` front-loading this replaced, element for
    element.  A cumulative-floor Bresenham back-loads instead (at 3 blocks and
    a remainder of 2 it awards blocks 1 and 2, not 0 and 1), which would have
    moved the released set of every unit whose column count *is* a whole
    number of superblocks -- a wire change with no reason behind it.
    """
    if total < 0:
        raise GrammarError(f"release total must not be negative: {total}")
    widths = superblock_widths(n_columns, superblock_columns)
    counts = [total * width // n_columns for width in widths]
    leftover = total - sum(counts)
    fractions = [total * width % n_columns for width in widths]
    order = sorted(range(len(widths)), key=lambda index: (-fractions[index], index))
    for index in order[:leftover]:
        counts[index] += 1
    return tuple(counts)


def release_defined_on(grid: "PayloadGrid") -> bool:
    """Whether a release can name a code on ``grid`` at all.

    A release replaces one position's code with **any** code of the grid, and
    the RELEASE plane stores that code in ``RELEASE_BITS`` bits whatever the
    grid (doc S6; the normative element width in doc S3 is this same
    constant).  So release is defined exactly where the grid's code space fits
    that width: ``grid.size <= 2**RELEASE_BITS``.

    Derived from ``RELEASE_BITS``, never from a roster of grid names -- the day
    the plane widens, this predicate is the one thing that moves, and a roster
    would still name E2M1 (rule 3).  Widening it is a wire change, not an
    encoder one.
    """
    return grid.size <= (1 << RELEASE_BITS)


def require_release_defined(grid: "PayloadGrid") -> None:
    """Refuse a released unit on a grid the RELEASE plane cannot name.

    One rule, one home (rule 4): the writer asks this before it does any work,
    and **both readers ask it before they place a release**, because the two
    questions are the same question and answering them differently is how an
    artifact this tree's own encoder calls undefined gets parsed anyway.  Such
    an artifact does not fail loudly on the read path -- the RELEASE plane is a
    legal 4-bit field on any grid, so its codes land on positions chosen by the
    reader's own ranking and decode to values no encoder chose (tessera#180,
    finding S5).  Refusing at read is the same doctrine ``canonical.Reader.enum``
    already applies to an ordinal no conforming encoder can produce.

    The message names the grid, the plane's width and the grid's code count, so
    a reader of it learns *why* release is undefined here rather than that it
    is; without that the refusal arrives from ``wire.pack_uniform`` at write as
    "value out of range for a 4-bit field", naming neither release nor the grid.
    """
    if release_defined_on(grid):
        return
    raise GrammarError(
        f"release is not defined over grid {grid.name}: the RELEASE plane "
        f"stores {RELEASE_BITS} bits per released position and the grid has "
        f"{grid.size} codes, so an override cannot name most of them. "
        f"Release is a {1 << RELEASE_BITS}-code grid's dial."
    )


def superblock_quota_ok(
    rates: tuple[int, ...], superblock_columns: int, root: Fraction
) -> bool:
    """True iff every *complete* superblock keeps the quota (doc S6).

    A trailing partial superblock is not required to keep it; only complete
    superblocks are constrained, which is what makes importance placement legal.

    Read the boundary case literally: a unit narrower than one superblock has
    no complete superblock, so this returns True for **any** schedule over it.
    That is the semantic, not a hole, and the caller is not left unguarded by
    it: ``Manifest.__post_init__`` runs ``validate_rate_schedule`` -- the
    whole-unit quota, which is exact at every width -- a few lines above the
    call (``manifest.py``: ``validate_rate_schedule``, then the ``window_bits``
    check, then this).
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
