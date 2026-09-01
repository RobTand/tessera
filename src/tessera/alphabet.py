"""Stage-C anchor forests: the alphabet and descendant planes (doc S6).

S6 stores both objects in the artifact -- "the alphabet blob (a charged
plane)" and "a stored descendant map per alphabet" -- so Tessera fixes no
alphabet constant anywhere.  The encoder optimizes them per artifact and
writes them to the wire.  This module builds them.

**The structure is forced, not chosen.**  S6 gives an alphabet of
``|A_R| = 2^(R+1)`` anchors, a completion level ``c <= 3 - R`` with
``|D(a)| = 2^c`` descendants per anchor, and the requirement that at
``c = 3 - R`` "the descendant sets **partition** the 16-code grid: every
code is a descendant of exactly one anchor."  Multiply the two cardinalities:

    2^(R+1) anchors  x  2^(3-R) descendants  =  2^4  =  16   for every R.

The partition is therefore exact at every rate, with no slack to distribute.
Together with build item 1's nesting obligation -- "every completion prefix is
a valid partial map" -- each anchor's descendants form a **binary tree of
depth 3-R**, and the c completion bits are the path from the root down it.
That is the whole of Stage C:

===  =======  ==========  ================================
R    anchors  tree depth  S6's own name for it
===  =======  ==========  ================================
3    16       0           "R=3, c=0: per-position 16 already"
2    8        1           "R=2, c=1: partner bijection"
1    4        2           "R=1, c=2: four-way map"
===  =======  ==========  ================================

**What is left to choose** is which codes group together and which member of
each node represents it.  Grouping: the codes are value-ordered and split
dyadically into contiguous blocks, because for a one-dimensional source the
minimum-SSE partition into contiguous-in-value cells is contiguous, and any
non-contiguous grouping is dominated.  Representatives: chosen by exhaustive
search against the source, which is cheap -- a node has at most 8 candidates.

Defining this is **build item 2**, which S6 says "gates the sub-3 ladder, the
matched-bpw fusing trade, and low-rate rotation behavior".

**Relation to the reviewed rate-2 fixture.**  S6 records ``(15, 13, 11, 9, 8,
2, 4, 7)`` sitting at value-order indices 0,2,4,6,8,10,12,15 -- "stride-2
Ungerboeck partitioning with the final slot snapped" -- and says plainly that
"one fixture is not a convention".  This construction does **not** reproduce
it, and the difference is the snap: the fixture takes the outer member of each
value-ordered pair, preserving |6.0| at c=0; SSE-optimal selection takes the
inner member, topping out at |4.0|.  Measured against a Gaussian source with
**each anchor set given its own optimal group scale** -- the only fair
comparison, since the scale is a free per-group parameter and a shorter
alphabet simply asks for a larger one -- the derived set wins by **29.2% SSE**
(RMSE 0.2097 vs 0.2492).  Preserving the extremes is a heuristic; the house
rule is to derive the decision from the objective, so the objective wins and
the fixture's snap is recorded as the ablation it is.

That number is a **synthetic screen at c=0**, not a served result: one
source distribution, one completion level, no end-to-end KL.  It justifies the
construction; it does not promote it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import GrammarError
from .grammar import alphabet_size, completion_capacity

#: The rate-1/2 convolutional code emits two bits, so it selects one of four
#: subsets.  Imported by ``trellis`` rather than the other way round: the grid
#: has to know the count to refuse a code space that cannot be split evenly.
#: ``PayloadGrid.partition`` for a grid whose code order is a tree traversal.
#: ``build_forest`` reads it to skip both the k-d bisection and the scalar
#: routing scan: a tree grid has already decided its blocks and their
#: representatives, at fit time, with the data in hand.
TREE_PARTITION = "tree"

SUBSET_COUNT = 4

__all__ = [
    "E2M1_VALUES",
    "E4M3_VALUES",
    "PayloadGrid",
    "E2M1_GRID",
    "E4M3_GRID",
    "value_order",
    "AnchorForest",
    "build_forest",
    "GAUSSIAN_SOURCE",
    "tuple_grid",
    "lloyd_max_grid",
    "grid_digest",
    "SERIALISABLE_GRIDS",
]

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: The 16 NVFP4 E2M1 codes indexed by nibble: bit 3 sign, bits 2..0 magnitude.
E2M1_VALUES: tuple[float, ...] = tuple(
    (-1.0 if code >> 3 else 1.0) * _E2M1_MAGNITUDES[code & 7] for code in range(16)
)


def _e4m3_value(byte: int) -> float:
    """One E4M3FN byte -> its value.  Sign 1, exponent 4, mantissa 3, bias 7."""
    sign = -1.0 if byte >> 7 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0:                       # subnormal
        return sign * (mantissa / 8.0) * 2.0 ** -6
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


#: The 256 E4M3FN byte patterns by value.  ``0x7F``/``0xFF`` are NaN in FN and
#: are given the value of their neighbour instead -- see ``E4M3_GRID``.
E4M3_VALUES: tuple[float, ...] = tuple(
    _e4m3_value(0x7E if byte == 0x7F else 0xFE if byte == 0xFF else byte)
    for byte in range(256)
)


@dataclass(frozen=True)
class PayloadGrid:
    """The reconstruction grid a trellis quantises onto.

    Tessera's grammar is stated over a code *space* of ``2^payload_bits`` slots
    -- ``|A_R| * |D(a)| = 2^(R+1) * 2^(cap-R)`` has to close exactly at every
    rate -- so what varies between families is the width of that space and what
    each slot decodes to.  TESSERA-4 is this construction over E2M1's 16
    nibbles; TESSERA-8 is the identical construction over E4M3's 256 bytes.

    ``native`` exists because E4M3FN is **not** a clean power of two: two of its
    256 patterns are NaN.  Dropping them would leave 254 slots and no exact
    dyadic partition, so instead those slots carry a neighbour's value and
    ``native`` maps them back to the legal byte at materialisation.  Four slots
    of 256 are then duplicates -- the two signed zeros, which E2M1 also has, and
    the two former NaNs.  A duplicate is never *preferred*: ties break to the
    lower code, and the lower code is always the legal one.
    """

    name: str
    values: tuple[float, ...]
    native: "tuple[int, ...] | None" = None
    arity: int = 1
    keys: "tuple[tuple[int, ...], ...] | None" = None
    partition: str = "stride"

    @property
    def size(self) -> int:
        """Number of codes.  ``values`` is flat, ``arity`` floats per code."""
        return len(self.values) // self.arity

    @property
    def payload_bits(self) -> int:
        """``log2(size)`` -- the width of one code, whatever it reconstructs."""
        return self.size.bit_length() - 1

    @property
    def rate_cap(self) -> int:
        """Highest trellis rate: one bit of the payload is the code's redundancy."""
        return self.payload_bits - 1

    @property
    def bits_per_position(self) -> float:
        """What a full-payload code costs per reconstructed weight.

        This is the whole point of ``arity``.  A code is charged once and pays
        for ``arity`` positions, so a k-tuple grid over a G-code base spends
        ``(k*log2(G) - 1)/k`` payload bits per weight at ``R = cap``.  At k=1
        over E2M1 that is the familiar 3.0; at k=2 it is 3.5; the rate quantum
        halves with every doubling of k, which is the ladder the scalar
        grammar cannot express at all.
        """
        return self.payload_bits / self.arity

    def vector(self, code: int) -> "tuple[float, ...]":
        """The ``arity`` values this code reconstructs, in row order."""
        return self.values[code * self.arity : (code + 1) * self.arity]

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise GrammarError(f"grid {self.name}: arity must be >= 1")
        if len(self.values) % self.arity:
            raise GrammarError(
                f"grid {self.name}: {len(self.values)} values is not a whole "
                f"number of arity-{self.arity} codes"
            )
        if self.size & (self.size - 1):
            raise GrammarError(
                f"grid {self.name} has {self.size} slots, which is not a power of "
                "two; the anchor/descendant partition cannot close"
            )
        if self.size % SUBSET_COUNT:
            raise GrammarError(
                f"grid {self.name} has {self.size} codes, which is not divisible "
                f"by the {SUBSET_COUNT} trellis subsets"
            )
        if self.native is not None and len(self.native) != self.size:
            raise GrammarError(f"grid {self.name}: native map is not {self.size} long")
        if self.partition not in ("stride", "coset", TREE_PARTITION):
            raise GrammarError(
                f"grid {self.name}: partition must be 'stride', 'coset' or "
                f"{TREE_PARTITION!r}, got {self.partition!r}"
            )
        if self.keys is None:
            # A scalar grid's key is its rank in value order, which is what the
            # stride rule has always used.  Deriving it here rather than at each
            # call site is what lets one formula cover both arities.
            rank = {
                code: position
                for position, code in enumerate(
                    sorted(range(self.size), key=lambda c: (self.values[c], c))
                )
            }
            object.__setattr__(
                self, "keys", tuple((rank[c],) for c in range(self.size))
            )
        elif len(self.keys) != self.size:
            raise GrammarError(f"grid {self.name}: keys map is not {self.size} long")


def tuple_grid(base: PayloadGrid, k: int, partition: str = "coset") -> PayloadGrid:
    """``k`` consecutive positions quantised as one code over ``base**k``.

    **This is the k-tuple trellis, and it needs no new trellis.**  ``|A_R| =
    2^(R+1)`` caps a *scalar* trellis at R = log2(G) - 1 over a G-code grid --
    R=3 on E2M1, because R=4 would need 32 reconstruction levels and E2M1 has
    16.  A pair of positions has G^2 joint codes, so the identical construction
    one level up spends R = 2*log2(G) - 1 bits per *pair*, which is
    ``(2*log2(G) - 1)/2`` bits per position with the redundancy bit intact.
    Everything downstream -- the anchor/descendant partition, the completion
    grammar, the Viterbi, the replay -- operates on codes and never asked how
    many weights a code stands for.  So the whole change is here.

    Codes are ordered ``c_1`` slowest: code ``i`` reconstructs base codes
    ``i // G^(k-1), ..., i % G``, mapped onto ``k`` **consecutive rows**.  Rows
    rather than columns because the trellis runs down columns, so a tuple must
    be a contiguous run along the trellis axis for its positions to share one
    branch decision.

    ``partition="coset"`` is the default at k>1: subsets are the level curves
    of ``sum of base ranks mod 4``, the standard multidimensional Ungerboeck
    partition, which reduces exactly to the scalar stride-4 rule at k=1.
    """
    if k < 1:
        raise GrammarError(f"tuple arity must be >= 1, got {k}")
    if base.arity != 1:
        raise GrammarError(
            f"tuple_grid needs a scalar base grid; {base.name} has arity "
            f"{base.arity}. Build the k-tuple in one step, not by nesting."
        )
    if k == 1:
        return base
    size = base.size**k
    if size > 1 << 16:
        raise GrammarError(
            f"{base.name}^{k} is {size} codes; the Viterbi scores every anchor "
            "at every step, so this is a cost refusal, not a grammar one"
        )
    base_rank = {code: rank for rank, code in enumerate(value_order(base))}
    values: list[float] = []
    keys: list[tuple[int, ...]] = []
    for code in range(size):
        digits = []
        rest = code
        for _ in range(k):
            digits.append(rest % base.size)
            rest //= base.size
        digits.reverse()
        for digit in digits:
            values.append(base.values[digit])
        keys.append(tuple(base_rank[digit] for digit in digits))
    return PayloadGrid(
        name=f"{base.name}x{k}",
        values=tuple(values),
        native=None,                 # a tuple code is not a hardware byte
        arity=k,
        keys=tuple(keys),
        partition=partition,
    )


def lloyd_max_grid(
    size: int,
    sigma: float = 1.0,
    iterations: int = 80,
    samples: int = 1 << 15,
    name: "str | None" = None,
) -> PayloadGrid:
    """The SSE-optimal scalar levels for a Gaussian source, as a grid.

    Promoted out of the measurement scripts because **a grid is wire**: a
    decoder that reconstructs on different levels than the encoder chose
    produces plausible, wrong weights rather than an error.  So the
    construction has to be deterministic and versioned, which means the
    iteration count and sample count are parameters of the artifact and not
    of whoever ran the script.  ``GAUSSIAN_SOURCE`` is an inverse-CDF sample,
    so there is no seed anywhere in this.

    These levels are **not** materialisable into any hardware format -- see
    ``native=None`` -- so a grid built here is kernel-lane only.
    """
    levels = _lloyd_levels(GAUSSIAN_SOURCE(samples, sigma), size, iterations)
    return PayloadGrid(name or f"LM{size}", tuple(levels))


def grid_digest(grid: PayloadGrid) -> str:
    """A stable identity for a grid, for the wire.

    The ALPHABET and DESCENDANT planes carry **codes**; code -> value comes
    from the grid, which no plane records.  Two artifacts over different grids
    are therefore byte-indistinguishable today, and the wrong one decodes to
    plausible wrong weights -- silent corruption, not a load error.  This is
    the value an ``encoder_profile_id`` has to absorb before anything but
    implicit-E2M1 is allowed to serialise.

    Values are digested at their exact float64 bit patterns, because a grid
    that round-trips through a lower precision is a *different* grid and must
    say so.
    """
    import hashlib
    import struct

    hasher = hashlib.sha256()
    hasher.update(f"tessera-grid-v1|{grid.name}|{grid.arity}|{grid.size}|".encode())
    hasher.update(grid.partition.encode())
    for value in grid.values:
        hasher.update(struct.pack("<d", value))
    hasher.update(b"|native|")
    if grid.native is None:
        hasher.update(b"none")
    else:
        for code in grid.native:
            hasher.update(struct.pack("<I", code))
    hasher.update(b"|keys|")
    for key in grid.keys or ():
        for entry in key:
            hasher.update(struct.pack("<I", entry))
    return hasher.hexdigest()


E2M1_GRID = PayloadGrid("E2M1", E2M1_VALUES, tuple(range(16)))
E4M3_GRID = PayloadGrid(
    "E4M3",
    E4M3_VALUES,
    tuple(0x7E if b == 0x7F else 0xFE if b == 0xFF else b for b in range(256)),
)


def value_order(grid: PayloadGrid = E2M1_GRID) -> tuple[int, ...]:
    """The grid's codes ascending by decoded value, ties broken by code.

    For a k-tuple grid there is no scalar value to sort on, so the order is
    over the code's **rank vector**: total rank first, then the vector itself.
    At k=1 the rank vector is ``(rank,)`` and total rank *is* the rank, so this
    reproduces the scalar value order exactly -- one formula, both arities.

    ``-0.0 == 0.0`` in IEEE arithmetic, so the two signed zeros tie and the
    tie-break places ``+0`` before ``-0``.  That is not cosmetic: it fixes which
    zero is an anchor at rates below the cap, and the reviewed rate-2 fixture
    agrees with this placement.  The same rule sends E4M3's two NaN-slot
    duplicates behind the legal bytes they copy, so neither is ever chosen as a
    representative over the byte it duplicates.
    """
    keys = grid.keys or ()
    return tuple(
        sorted(range(grid.size), key=lambda code: (sum(keys[code]), keys[code], code))
    )


def GAUSSIAN_SOURCE(count: int = 1 << 14, sigma: float = 1.0) -> tuple[float, ...]:
    """A deterministic standard-normal sample by inverse-CDF, on the E2M1 scale.

    Weights reach the alphabet already divided by their group scale, and the
    post-rotation residual the doc reports is white (S5: "the post-trellis
    additive residual is white ... rank-4 energy 0.73-0.91%"), so a Gaussian
    is the honest default source.  Deterministic, because an alphabet that
    changed run to run would make artifacts irreproducible.
    """
    from math import erf, sqrt

    # Inverse CDF by bisection: no scipy, and exactness beats speed at 16k.
    def ppf(p: float) -> float:
        lo, hi = -8.0, 8.0
        for _ in range(64):
            mid = (lo + hi) / 2
            if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return tuple(sigma * ppf((index + 0.5) / count) for index in range(count))


#: The grids a Tessera artifact may be serialised over.  **Closed by
#: construction**: every entry is a permanent wire commitment, because the
#: ALPHABET/DESCENDANT planes carry codes and the grid is what turns a code
#: into a value.  Membership is what ``encoder_profile_id`` binds and what
#: ``read_unit_artifact`` searches, so a grid that is not here cannot be
#: written *or* read -- both directions fail closed rather than guessing.
#:
#: Three entries today, all derivable from a name and an arity, which is why
#: the reader can rebuild them without the values on the wire:
#:   * ``E2M1`` -- arity 1, 16 codes, cap 3.  Every artifact built before the
#:     grid was bound into the profile id used this one implicitly.
#:   * ``E2M1^2`` -- arity 2, 256 codes, cap 7.  The stock-lane rung: a code
#:     covers two consecutive rows.  cap 7 over arity 2 is **3.5 payload bits
#:     per weight**; the 0.5 bpp scale plane brings the artifact to 4.0 bpp.
#:     Those two numbers get confused constantly -- 4.0 is the SIZE, 3.5 is the
#:     payload half of it, and EXL3's 4.0117 bpw is 4.0 payload + 0.0117.
#:   * ``E4M3`` -- arity 1, 256 codes, cap 7: **the 8-bit ladder**, 1.0 to 7.0
#:     payload bits per weight.  It was absent for no reason the criterion
#:     above supports -- its values come from the byte pattern
#:     (``_e4m3_value``), so a reader rebuilds them from the name exactly as it
#:     does E2M1's, which is precisely what the Lloyd-Max exclusion below turns
#:     on.  Its absence left the menu with **nothing between Tessera-4's 4.0
#:     bpp ceiling and FP8's 8.0**, so an allocator wanting 5 or 6 bits had to
#:     buy 8.  Round-trips at every rung (``test_e4m3_ladder_serialises``).
#:
#: ``E4M3^2`` is **not** here and its absence is structural, not an oversight:
#: 65536 codes, and the ALPHABET/DESCENDANT planes are one byte per code.  256
#: codes is the wire's ceiling, which is why the serialisable set is exactly
#: the three grids that fit in a byte.
#:
#: **Free (Lloyd-Max) grids are deliberately absent.**  Their values are fitted
#: to the tensor and are not reproducible by a reader from any identifier, so
#: admitting one needs the values themselves on the wire -- a VALUES plane, a
#: second schema change.  That is a deferral, not an oversight.
SERIALISABLE_GRIDS: "dict[str, PayloadGrid]" = {
    grid_digest(grid): grid
    for grid in (E2M1_GRID, tuple_grid(E2M1_GRID, 2), E4M3_GRID)
}


@dataclass(frozen=True)
class AnchorForest:
    """One rate's alphabet plus its nested descendant map.

    ``blocks[i]`` holds the ``2^(3-R)`` codes of anchor ``i``'s tree in
    **completion order**: index ``j`` is reached by the ``3-R`` completion bits
    of ``j`` read most-significant-first, so every prefix of those bits selects
    a valid shallower node (build item 1's nesting obligation).  ``blocks[i][0]``
    is the anchor, which is what ``c = 0`` decodes to.
    """

    rate: int
    blocks: tuple[tuple[int, ...], ...]
    grid: PayloadGrid = E2M1_GRID

    @property
    def cap(self) -> int:
        return self.grid.rate_cap

    def __post_init__(self) -> None:
        expected_anchors = alphabet_size(self.rate, self.cap)
        depth = completion_capacity(self.rate, self.cap)
        width = 1 << depth
        if len(self.blocks) != expected_anchors:
            raise GrammarError(
                f"rate {self.rate} needs {expected_anchors} anchors, "
                f"got {len(self.blocks)}"
            )
        seen: set[int] = set()
        for index, block in enumerate(self.blocks):
            if len(block) != width:
                raise GrammarError(
                    f"anchor {index}: |D(a)| must be 2^{depth} = {width}, "
                    f"got {len(block)}"
                )
            for code in block:
                if not 0 <= code < self.grid.size:
                    raise GrammarError(
                        f"code {code} is outside the {self.grid.name} grid"
                    )
                if code in seen:
                    raise GrammarError(
                        f"code {code} appears under two anchors; at c = cap - R "
                        f"the descendant sets must partition the "
                        f"{self.grid.size}-code grid"
                    )
                seen.add(code)
        if len(seen) != self.grid.size:
            missing = sorted(set(range(self.grid.size)) - seen)
            raise GrammarError(
                f"descendant sets do not cover the grid; missing {missing}"
            )

    @property
    def anchors(self) -> tuple[int, ...]:
        """The ``c = 0`` reachable set: one code per anchor."""
        return tuple(block[0] for block in self.blocks)

    def reachable(self, anchor: int, completion: int) -> tuple[int, ...]:
        """The ``2^c`` codes reachable from ``anchor`` at completion level c."""
        depth = completion_capacity(self.rate, self.cap)
        if not 0 <= completion <= depth:
            raise GrammarError(
                f"completion level {completion} exceeds cap - R = {depth}"
            )
        stride = 1 << (depth - completion)
        return self.blocks[anchor][::stride]

    def decode(self, anchor: int, bits: tuple[int, ...]) -> int:
        """Walk ``len(bits)`` completion bits down the tree from ``anchor``."""
        depth = completion_capacity(self.rate, self.cap)
        if len(bits) > depth:
            raise GrammarError(f"{len(bits)} completion bits exceed cap - R = {depth}")
        index = 0
        for bit in bits:
            index = (index << 1) | bit
        return self.blocks[anchor][index << (depth - len(bits))]

    def _refuse_unserialisable(self) -> None:
        """The one hard line: only a grid the wire commits to may be written.

        These planes carry **codes**.  Code -> value comes from the grid, so
        two artifacts over different grids would be byte-indistinguishable and
        the wrong one would decode to plausible wrong weights rather than to an
        error.  That ambiguity is now closed at its root: ``encoder_profile_id``
        absorbs ``grid_digest``, so the grid *is* on the wire, and a reader
        recovers it by searching :data:`SERIALISABLE_GRIDS` for a digest match
        exactly as it recovers the ConvCode.  What remains is the membership
        test -- a grid outside that registry has no identity a reader can
        resolve, so it is refused here, at the serialisation boundary.
        Encoding, decoding and measuring on any grid stay open.
        """
        digest = grid_digest(self.grid)
        if digest not in SERIALISABLE_GRIDS:
            raise GrammarError(
                f"grid {self.grid.name} (arity {self.grid.arity}, "
                f"{self.grid.size} codes, digest {digest[:16]}) is not in "
                "SERIALISABLE_GRIDS, so no reader can resolve its digest back "
                "to a code->value map and the artifact would decode to "
                "plausible wrong weights. A fitted/free grid needs its values "
                "on the wire (a VALUES plane) before it can serialise; a "
                "derivable one needs adding to the registry, which is a "
                "permanent wire commitment."
            )
        if self.grid.size > 256:
            raise GrammarError(
                f"grid {self.grid.name} has {self.grid.size} codes: the "
                "ALPHABET/DESCENDANT planes are one byte per code and cannot "
                "carry it. A wider code space is a schema change (wider plane "
                "element), not a cast."
            )

    def alphabet_plane(self) -> bytes:
        """The ALPHABET plane: one byte per anchor, in anchor order."""
        self._refuse_unserialisable()
        return bytes(self.anchors)

    def descendant_plane(self) -> bytes:
        """The DESCENDANT plane: the forest flattened, one byte per grid code."""
        self._refuse_unserialisable()
        return bytes(code for block in self.blocks for code in block)


def _best_representative(
    candidates: tuple[int, ...], assigned: "list[float]",
    grid: PayloadGrid = E2M1_GRID,
) -> int:
    """The candidate minimizing SSE over the source mass routed to this node."""
    if not assigned:
        # No mass: prefer the candidate nearest zero, deterministically.
        return min(candidates, key=lambda code: (abs(grid.values[code]), code))
    best, best_cost = candidates[0], None
    for code in candidates:
        value = grid.values[code]
        cost = sum((sample - value) ** 2 for sample in assigned)
        if best_cost is None or cost < best_cost:
            best, best_cost = code, cost
    return best


def build_forest(
    rate: int,
    samples: "tuple[float, ...] | None" = None,
    grid: PayloadGrid = E2M1_GRID,
) -> AnchorForest:
    """Build the optimized anchor forest for ``rate``.

    Contiguous dyadic blocks over the value order, with every node's
    representative chosen by exhaustive search against ``samples``.  Because
    representatives are chosen per node and the tree shape is fixed, nesting
    holds by construction: truncating completion bits lands on an ancestor,
    which is a legal partial map.
    """
    order = value_order(grid)
    depth = completion_capacity(rate, grid.rate_cap)
    width = 1 << depth
    anchors = alphabet_size(rate, grid.rate_cap)
    if grid.partition == TREE_PARTITION:
        return _build_forest_tree(rate, grid, width, anchors)
    if depth and grid.arity > 1:
        return _build_forest_kd(rate, grid, depth, width, anchors)
    if anchors * width != grid.size:
        raise GrammarError(
            f"rate {rate}: {anchors} anchors x {width} descendants "
            f"!= {grid.size} ({grid.name})"
        )
    if samples is None:
        # In grid units.  Weights reach the alphabet divided by their group
        # scale, and S6b sets that scale so the group's amax lands on the
        # grid's peak -- 6.0 on E2M1, 448.0 on E4M3.  So the source's spread is
        # a property of the *grid*, not a constant: the E2M1 default of
        # sigma=1.0 against a peak of 6.0 fixes the ratio, and every other grid
        # inherits it.  Optimising a 256-anchor E4M3 forest against a sigma-1
        # Gaussian instead puts every anchor in the bottom 1% of the range and
        # costs 4.4x the error at 3.5 bpp -- worse than the 16-code grid.
        peak = max(abs(value) for value in grid.values)
        samples = GROUP_SCALED_SOURCE(peak)

    # Contiguous blocks in value order, then route each sample to the block
    # whose value span is nearest -- a nearest-code assignment, since blocks
    # are contiguous.
    raw_blocks = [order[i * width : (i + 1) * width] for i in range(anchors)]
    if depth == 0:
        # Every block is one code, so there is no representative to choose and
        # no mass to route.  Skipping the routing scan is not just a saving:
        # it is what lets a k-tuple grid through, since routing reads
        # ``grid.values[code]`` as a scalar and a tuple code has no scalar.
        return AnchorForest(
            rate=rate, blocks=tuple((code,) for code in order), grid=grid
        )
    # Two candidate partitions, and the SOURCE picks between them.
    #
    # The contiguous rule is right whenever the grid's codes are spread like the
    # source, and it is what every E2M1 artifact was built with.  The
    # mass-balanced rule is right when they are not -- on E4M3 the contiguous
    # split wastes ten of sixteen anchors.  Neither dominates, so neither is
    # asserted: both are built, both are scored on the same Gaussian at the
    # ``c = 0`` the pipeline actually decodes at, and the cheaper one is used.
    # The choice is by measurement against the objective, which is what
    # principle 2 asks for -- not a rule about which grids are "log-spaced".
    balanced, balanced_reps = _mass_balanced_blocks(grid, samples, anchors, width)
    scored = [
        (_partition_cost(grid, samples, raw_blocks), raw_blocks),
        (_partition_cost(grid, samples, balanced, seed=balanced_reps), balanced),
    ]
    (_, routed), raw_blocks = min(scored, key=lambda entry: entry[0][0])

    blocks: list[tuple[int, ...]] = []
    for index, block in enumerate(raw_blocks):
        blocks.append(_order_block(block, routed[index], depth, grid))
    return AnchorForest(rate=rate, blocks=tuple(blocks), grid=grid)


def _lloyd_levels(
    source: "tuple[float, ...]", size: int, iterations: int = 40,
) -> "list[float]":
    """Lloyd-Max levels for an arbitrary SORTED source.

    Assignment is by bisection on the midpoints rather than a scan over levels,
    which is what makes this affordable to call once per forest build.
    """
    from bisect import bisect

    lo, hi = source[0], source[-1]
    levels = [lo + (hi - lo) * index / (size - 1) for index in range(size)]
    for _ in range(iterations):
        cuts = [(levels[i] + levels[i + 1]) / 2.0 for i in range(size - 1)]
        buckets: "list[list[float]]" = [[] for _ in range(size)]
        for sample in source:
            buckets[bisect(cuts, sample)].append(sample)
        levels = [
            sum(bucket) / len(bucket) if bucket else levels[index]
            for index, bucket in enumerate(buckets)
        ]
        levels.sort()
    return levels


def GROUP_SCALED_SOURCE(
    peak: float, group: int = 16, count: int = 1 << 14,
) -> "tuple[float, ...]":
    """The source the ALPHABET actually sees -- bounded by ``peak``, not by sigma.

    S6b divides every group of ``group`` weights by ``amax/peak``, so the value
    reaching the grid is ``w / amax * peak``: a Gaussian normalised by its OWN
    group maximum.  That distribution is **bounded** -- exactly one value per
    group lands on ``peak`` -- and it is not a Gaussian of any sigma.

    Modelling it as ``GAUSSIAN_SOURCE(sigma=peak/6)`` is wrong twice over.  The
    measured spread after scaling is ``peak/2.05``, not ``peak/6``; and no
    Gaussian is right at any sigma, because at ``sigma = peak/2.05`` Lloyd-Max's
    top level sits at ``1.36 x peak``, outside the grid entirely.  At the cap
    neither error matters -- every code is an anchor and the top level IS the
    peak by construction.  Below the cap the source decides WHICH codes become
    anchors, and a mis-modelled tail spends anchors on values the data never
    reaches while clipping the ones it does.  That is the whole of TESSERA-8's
    sub-cap collapse.

    Deterministic, because an alphabet that changed run to run would make
    artifacts irreproducible: a fixed LCG permutes the inverse-CDF sample so
    that groups are representative rather than sorted runs, and nothing here
    reads a seed from the environment or the clock.
    """
    base = list(GAUSSIAN_SOURCE(count))
    state = 0x2545F491
    for index in range(len(base) - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        swap = state % (index + 1)
        base[index], base[swap] = base[swap], base[index]
    out: "list[float]" = []
    for start in range(0, len(base) - group + 1, group):
        chunk = base[start : start + group]
        amax = max(abs(value) for value in chunk)
        if amax == 0.0:                         # pragma: no cover - measure zero
            continue
        out.extend(value * peak / amax for value in chunk)
    out.sort()
    return tuple(out)


def _partition_cost(
    grid: PayloadGrid, samples: "tuple[float, ...]",
    blocks: "list[tuple[int, ...]]",
    seed: "list[int] | None" = None, iterations: int = 12,
) -> "tuple[float, list[list[float]]]":
    """Expected SSE of a partition at ``c = 0``, and the routing that gives it.

    At ``c = 0`` a block reconstructs on exactly one member, so a partition is
    worth precisely what its ``anchors`` representatives are worth -- and the
    encoder sends a weight to the nearest **representative**, not to the block
    holding the nearest code.  Scoring by nearest-code flatters any partition
    whose blocks are value-contiguous, because every sample then lands in a
    block that contains something near it whether or not that block's single
    reachable value is near it.  That is the difference between a partition
    that looks balanced and one that reconstructs well.

    So: Lloyd descent under the block constraint.  Route to the nearest rep,
    re-pick each block's rep from its own members, repeat.  Both steps are
    non-increasing in SSE, so this converges, and it evaluates the quantity the
    decoder will actually pay.
    """
    from bisect import bisect

    values = grid.values
    # Lloyd runs on a deterministic stride of the source; the final routing is
    # over all of it.  Choosing between two partitions does not need 16k points.
    coarse = samples[::4] or samples
    # Lloyd is a descent, so the seed decides which optimum it finds.  A
    # mass-balanced block holds its anchor plus whatever fillers were nearest
    # it, so its mean is nowhere near its anchor -- seeding on the mean starts
    # that partition on a filler and it never recovers.  A construction that
    # knows its own anchors says so.
    reps = list(seed) if seed is not None else [
        min(block, key=lambda c: abs(values[c] - sum(values[m] for m in block) / len(block)))
        for block in blocks
    ]

    def route(source: "tuple[float, ...]") -> "list[list[float]]":
        order = sorted(range(len(reps)), key=lambda b: values[reps[b]])
        ladder = [values[reps[b]] for b in order]
        cuts = [(ladder[i] + ladder[i + 1]) / 2.0 for i in range(len(ladder) - 1)]
        out: "list[list[float]]" = [[] for _ in blocks]
        for sample in source:
            out[order[bisect(cuts, sample)]].append(sample)
        return out

    routed = route(coarse)
    for _ in range(iterations):
        moved = False
        for index, block in enumerate(blocks):
            assigned = routed[index]
            if not assigned:
                continue
            count = len(assigned)
            first = sum(assigned)
            best = min(
                block,
                key=lambda c: count * values[c] ** 2 - 2.0 * values[c] * first,
            )
            if best != reps[index]:
                reps[index], moved = best, True
        if not moved:
            break
        routed = route(coarse)

    routed = route(samples)
    total = 0.0
    for index, block in enumerate(blocks):
        assigned = routed[index]
        if not assigned:
            continue
        count = len(assigned)
        first = sum(assigned)
        second = sum(sample * sample for sample in assigned)
        total += min(
            count * values[c] ** 2 - 2.0 * values[c] * first + second for c in block
        )
    return total, routed


def _mass_balanced_blocks(
    grid: PayloadGrid, samples: "tuple[float, ...]", anchors: int, width: int,
) -> "tuple[list[tuple[int, ...]], list[int]]":
    """``anchors`` blocks of ``width`` codes, grouped by MASS rather than count.

    The grammar needs ``anchors`` blocks of exactly ``width`` codes, because the
    completion field is a fixed ``depth`` bits wide.  It does **not** need them
    contiguous in value order -- the ALPHABET and DESCENDANT planes write the
    grouping out explicitly, so any partition is wire-expressible.

    Contiguity is only correct when the grid's codes are spread like the source.
    E4M3's 256 codes are log-spaced over ``2^-9 .. 448``, so sixteen equal-COUNT
    runs put ten of the sixteen anchors inside ``|x| < sigma/10`` -- a region
    holding about 1% of a Gaussian's mass, and a 16-level budget spending six.

    So the anchors are placed where the source is: Lloyd-Max levels for the same
    Gaussian the forest is optimised against, snapped to distinct grid codes,
    with the remaining codes filling each block out to ``width`` nearest-anchor
    first so a block stays a neighbourhood and completion bits still refine.
    """
    values = grid.values
    targets = _lloyd_levels(samples[::4] or samples, anchors)

    # Snap each target to a DISTINCT code, globally greedy on distance, so the
    # result does not depend on the order the targets are visited.
    pairs = sorted(
        (abs(targets[t] - values[c]), t, c)
        for t in range(anchors)
        for c in range(grid.size)
    )
    rep_of: "dict[int, int]" = {}
    taken: "set[int]" = set()
    for _, target, code in pairs:
        if target not in rep_of and code not in taken:
            rep_of[target] = code
            taken.add(code)
            if len(rep_of) == anchors:
                break

    members: "list[list[int]]" = [[rep_of[t]] for t in range(anchors)]
    ranked = {
        code: sorted((abs(values[code] - values[rep_of[t]]), t) for t in range(anchors))
        for code in range(grid.size)
        if code not in taken
    }
    # Most-contested code first: one whose nearest and second-nearest anchors
    # are far apart has the most to lose from being displaced, so it chooses
    # before the ambivalent ones.
    for code in sorted(ranked, key=lambda c: ranked[c][0][0] - ranked[c][min(1, anchors - 1)][0]):
        for _, target in ranked[code]:
            if len(members[target]) < width:
                members[target].append(code)
                break
        else:                                   # pragma: no cover - capacity is exact
            raise GrammarError(
                f"no block had room for code {code}; {anchors} x {width} "
                f"!= {grid.size} ({grid.name})"
            )
    blocks = [tuple(sorted(block, key=lambda c: (values[c], c))) for block in members]
    return blocks, [rep_of[t] for t in range(anchors)]


def _code_density(grid: PayloadGrid) -> "tuple[float, ...]":
    """Each code's source mass under a product Gaussian, in grid units.

    The scalar builder routes explicit samples to blocks and then picks the
    member nearest the routed mean.  Since ``sum_s (s - v)^2`` is minimised by
    the ``v`` nearest ``mean(s)``, that rule *is* "nearest the mass centroid" --
    so the k-dimensional generalisation needs the centroid, not the samples,
    and the density gives it in closed form with no sampling and no seed.

    The one honest difference: the scalar path weights by mass actually routed
    to the cell, this weights by density at the code.  They agree in the limit
    of a fine grid and differ slightly on a coarse one.
    """
    from math import exp

    sigma = max(abs(value) for value in grid.values) / 6.0
    out = []
    for code in range(grid.size):
        mass = 1.0
        for value in grid.vector(code):
            mass *= exp(-0.5 * (value / sigma) ** 2)
        out.append(mass)
    return tuple(out)


def _kd_bisect(
    codes: "tuple[int, ...]", grid: PayloadGrid, density: "tuple[float, ...]"
) -> "tuple[tuple[int, ...], tuple[int, ...]]":
    """Split a code set into two equal halves across its widest axis.

    A k-tuple code space has no value order to chop contiguously, so the
    scalar builder's "contiguous dyadic blocks" has to become something that
    means the same thing in k dimensions.  Splitting the widest axis at the
    median is the k-d tree construction: cells stay compact, the halves stay
    exactly equal -- which the dyadic tree requires -- and it is deterministic,
    which a k-means split with a seed would not be.  At arity 1 the widest axis
    is the only axis and the median split *is* the contiguous split, so this
    reduces to the scalar rule (``tests/test_ktuple.py`` asserts it).
    """
    vectors = {code: grid.vector(code) for code in codes}
    total = sum(density[code] for code in codes) or 1.0
    mean = [
        sum(density[c] * vectors[c][axis] for c in codes) / total
        for axis in range(grid.arity)
    ]
    spread = [
        sum(density[c] * (vectors[c][axis] - mean[axis]) ** 2 for c in codes)
        for axis in range(grid.arity)
    ]
    axis = max(range(grid.arity), key=lambda i: (spread[i], -i))
    order = tuple(sorted(codes, key=lambda c: (vectors[c][axis], vectors[c], c)))
    half = len(order) // 2
    return order[:half], order[half:]


def _representative(
    codes: "tuple[int, ...]", grid: PayloadGrid, density: "tuple[float, ...]"
) -> int:
    """The member nearest this node's mass centroid -- the scalar rule, in k-d."""
    total = sum(density[code] for code in codes) or 1.0
    centroid = [
        sum(density[c] * grid.vector(c)[axis] for c in codes) / total
        for axis in range(grid.arity)
    ]
    return min(
        codes,
        key=lambda c: (
            sum((v - m) ** 2 for v, m in zip(grid.vector(c), centroid)),
            c,
        ),
    )


def _order_block_kd(
    codes: "tuple[int, ...]", grid: PayloadGrid, density: "tuple[float, ...]",
    depth: int,
) -> "tuple[int, ...]":
    """One block into completion order, representative first, in k dimensions."""
    if depth == 0:
        return codes
    low, high = _kd_bisect(codes, grid, density)
    left = _order_block_kd(low, grid, density, depth - 1)
    right = _order_block_kd(high, grid, density, depth - 1)
    pick = _representative((left[0], right[0]), grid, density)
    return left + right if pick == left[0] else right + left


def _build_forest_tree(
    rate: int, grid: PayloadGrid, width: int, anchors: int
) -> "AnchorForest":
    """The forest for a grid whose code order IS a tree traversal.

    There is nothing to choose.  ``learn_tree_codebook`` emits leaves in tree
    order, so the contiguous dyadic block ``[k*width, (k+1)*width)`` is exactly
    the set of leaves under node ``k`` at level ``rate+1``; and it hoists each
    node's representative to its block's first slot, so index 0 is already the
    anchor.  Both of the things ``build_forest`` normally computes -- which
    codes group together, and which of them speaks for the group -- were decided
    against the real points rather than against a Gaussian stand-in.
    """
    if anchors * width != grid.size:
        raise GrammarError(
            f"rate {rate}: {anchors} anchors x {width} descendants "
            f"!= {grid.size} ({grid.name})"
        )
    return AnchorForest(
        rate=rate,
        blocks=tuple(
            tuple(range(k * width, (k + 1) * width)) for k in range(anchors)
        ),
        grid=grid,
    )


def _build_forest_kd(
    rate: int, grid: PayloadGrid, depth: int, width: int, anchors: int
) -> "AnchorForest":
    """The forest for a k-tuple grid below its rate cap.

    Recursive balanced bisection down to ``anchors`` blocks, then each block
    into completion order.  This is the scalar construction with "contiguous in
    value order" replaced by "compact under k-d bisection", which is the only
    part of it that assumed one dimension.
    """
    density = _code_density(grid)
    blocks: "list[tuple[int, ...]]" = [tuple(range(grid.size))]
    while len(blocks) < anchors:
        blocks = [
            half for block in blocks for half in _kd_bisect(block, grid, density)
        ]
    ordered = [_order_block_kd(block, grid, density, depth) for block in blocks]
    # Anchor order follows the blocks' own bisection traversal, which keeps
    # neighbouring anchors adjacent -- that is what makes the stride subset
    # rule separate them.
    ordered.sort(key=lambda block: (sum(grid.keys[block[0]]), grid.keys[block[0]]))
    if width != len(ordered[0]):
        raise GrammarError(
            f"k-d bisection produced blocks of {len(ordered[0])}, need {width}"
        )
    return AnchorForest(rate=rate, blocks=tuple(ordered), grid=grid)


def _order_block(
    block: tuple[int, ...], assigned: "list[float]", depth: int,
    grid: PayloadGrid = E2M1_GRID,
) -> tuple[int, ...]:
    """Arrange one block into completion order by recursive dyadic refinement.

    The node's representative is placed first so that a ``c``-bit prefix reads
    it directly; the recursion then fills the two halves, so index ``j`` is
    reached by ``j``'s bits read most-significant-first.
    """
    if depth == 0:
        return block
    half = len(block) // 2
    low, high = block[:half], block[half:]
    split = (grid.values[low[-1]] + grid.values[high[0]]) / 2
    low_mass = [sample for sample in assigned if sample < split]
    high_mass = [sample for sample in assigned if sample >= split]
    left = _order_block(low, low_mass, depth - 1, grid)
    right = _order_block(high, high_mass, depth - 1, grid)
    # The representative of the whole node is the better of its two children's
    # representatives, and it must sit at index 0.  Swapping the halves is the
    # only reordering that achieves that while keeping the tree dyadic.
    pick = _best_representative((left[0], right[0]), assigned, grid)
    return left + right if pick == left[0] else right + left
