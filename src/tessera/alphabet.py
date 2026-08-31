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
    native: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def payload_bits(self) -> int:
        """``log2(size)`` -- the scalar format's own width."""
        return self.size.bit_length() - 1

    @property
    def rate_cap(self) -> int:
        """Highest trellis rate: one bit of the payload is the code's redundancy."""
        return self.payload_bits - 1

    def __post_init__(self) -> None:
        if self.size & (self.size - 1):
            raise GrammarError(
                f"grid {self.name} has {self.size} slots, which is not a power of "
                "two; the anchor/descendant partition cannot close"
            )
        if len(self.native) != self.size:
            raise GrammarError(f"grid {self.name}: native map is not {self.size} long")


E2M1_GRID = PayloadGrid("E2M1", E2M1_VALUES, tuple(range(16)))
E4M3_GRID = PayloadGrid(
    "E4M3",
    E4M3_VALUES,
    tuple(0x7E if b == 0x7F else 0xFE if b == 0xFF else b for b in range(256)),
)


def value_order(grid: PayloadGrid = E2M1_GRID) -> tuple[int, ...]:
    """The grid's codes ascending by decoded value, ties broken by code.

    ``-0.0 == 0.0`` in IEEE arithmetic, so the two signed zeros tie and the
    tie-break places ``+0`` before ``-0``.  That is not cosmetic: it fixes which
    zero is an anchor at rates below the cap, and the reviewed rate-2 fixture
    agrees with this placement.  The same rule sends E4M3's two NaN-slot
    duplicates behind the legal bytes they copy, so neither is ever chosen as a
    representative over the byte it duplicates.
    """
    return tuple(
        sorted(range(grid.size), key=lambda code: (grid.values[code], code))
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

    def alphabet_plane(self) -> bytes:
        """The ALPHABET plane: one byte per anchor, in anchor order."""
        return bytes(self.anchors)

    def descendant_plane(self) -> bytes:
        """The DESCENDANT plane: the forest flattened, one byte per grid code."""
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
        samples = GAUSSIAN_SOURCE(sigma=peak / 6.0)

    # Contiguous blocks in value order, then route each sample to the block
    # whose value span is nearest -- a nearest-code assignment, since blocks
    # are contiguous.
    raw_blocks = [order[i * width : (i + 1) * width] for i in range(anchors)]
    # Blocks are contiguous in value order, so routing to the nearest block is
    # routing to the nearest code, and the block's bounds decide it: a sample
    # below the first value belongs to block 0, above the last to block -1, and
    # otherwise to whichever adjacent block is closer.  Written as a scan over
    # every (sample, code) pair this is O(samples * grid), which is 4.2M inner
    # steps on E4M3 -- the same answer for a few hundred times the work.
    edges = [grid.values[block[-1]] for block in raw_blocks[:-1]]
    routed: list[list[float]] = [[] for _ in range(anchors)]
    for sample in samples:
        low, high = 0, len(edges)
        while low < high:                       # first block whose top >= sample
            mid = (low + high) // 2
            if edges[mid] < sample:
                low = mid + 1
            else:
                high = mid
        index = low
        if index and index <= len(edges):
            below = grid.values[raw_blocks[index - 1][-1]]
            above = grid.values[raw_blocks[index][0]]
            if abs(sample - below) < abs(sample - above):
                index -= 1
        routed[index].append(sample)

    blocks: list[tuple[int, ...]] = []
    for index, block in enumerate(raw_blocks):
        blocks.append(_order_block(block, routed[index], depth, grid))
    return AnchorForest(rate=rate, blocks=tuple(blocks), grid=grid)


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
