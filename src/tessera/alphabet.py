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


def value_order() -> tuple[int, ...]:
    """The 16 nibbles ascending by decoded value, ties broken by nibble.

    ``-0.0 == 0.0`` in IEEE arithmetic, so the two signed zeros tie and the
    tie-break places ``+0`` (nibble 0) before ``-0`` (nibble 8).  That is not
    cosmetic: it fixes which zero is an anchor at rates below 3, and the
    reviewed rate-2 fixture agrees with this placement.
    """
    return tuple(sorted(range(16), key=lambda code: (E2M1_VALUES[code], code)))


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

    def __post_init__(self) -> None:
        expected_anchors = alphabet_size(self.rate)
        depth = completion_capacity(self.rate)
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
                if not 0 <= code < 16:
                    raise GrammarError(f"code {code} is outside the E2M1 grid")
                if code in seen:
                    raise GrammarError(
                        f"code {code} appears under two anchors; at c = 3 - R "
                        "the descendant sets must partition the 16-code grid"
                    )
                seen.add(code)
        if len(seen) != 16:
            missing = sorted(set(range(16)) - seen)
            raise GrammarError(
                f"descendant sets do not cover the grid; missing {missing}"
            )

    @property
    def anchors(self) -> tuple[int, ...]:
        """The ``c = 0`` reachable set: one code per anchor."""
        return tuple(block[0] for block in self.blocks)

    def reachable(self, anchor: int, completion: int) -> tuple[int, ...]:
        """The ``2^c`` codes reachable from ``anchor`` at completion level c."""
        depth = completion_capacity(self.rate)
        if not 0 <= completion <= depth:
            raise GrammarError(
                f"completion level {completion} exceeds 3 - R = {depth}"
            )
        stride = 1 << (depth - completion)
        return self.blocks[anchor][::stride]

    def decode(self, anchor: int, bits: tuple[int, ...]) -> int:
        """Walk ``len(bits)`` completion bits down the tree from ``anchor``."""
        depth = completion_capacity(self.rate)
        if len(bits) > depth:
            raise GrammarError(f"{len(bits)} completion bits exceed 3 - R = {depth}")
        index = 0
        for bit in bits:
            index = (index << 1) | bit
        return self.blocks[anchor][index << (depth - len(bits))]

    def alphabet_plane(self) -> bytes:
        """The ALPHABET plane: one byte per anchor, in anchor order."""
        return bytes(self.anchors)

    def descendant_plane(self) -> bytes:
        """The DESCENDANT plane: the forest flattened, 16 bytes at every rate."""
        return bytes(code for block in self.blocks for code in block)


def _best_representative(
    candidates: tuple[int, ...], assigned: "list[float]"
) -> int:
    """The candidate minimizing SSE over the source mass routed to this node."""
    if not assigned:
        # No mass: prefer the candidate nearest zero, deterministically.
        return min(candidates, key=lambda code: (abs(E2M1_VALUES[code]), code))
    best, best_cost = candidates[0], None
    for code in candidates:
        value = E2M1_VALUES[code]
        cost = sum((sample - value) ** 2 for sample in assigned)
        if best_cost is None or cost < best_cost:
            best, best_cost = code, cost
    return best


def build_forest(rate: int, samples: "tuple[float, ...] | None" = None) -> AnchorForest:
    """Build the optimized anchor forest for ``rate``.

    Contiguous dyadic blocks over the value order, with every node's
    representative chosen by exhaustive search against ``samples``.  Because
    representatives are chosen per node and the tree shape is fixed, nesting
    holds by construction: truncating completion bits lands on an ancestor,
    which is a legal partial map.
    """
    order = value_order()
    depth = completion_capacity(rate)
    width = 1 << depth
    anchors = alphabet_size(rate)
    if anchors * width != 16:
        raise GrammarError(
            f"rate {rate}: {anchors} anchors x {width} descendants != 16"
        )
    if samples is None:
        samples = GAUSSIAN_SOURCE()

    # Contiguous blocks in value order, then route each sample to the block
    # whose value span is nearest -- a nearest-code assignment, since blocks
    # are contiguous.
    raw_blocks = [order[i * width : (i + 1) * width] for i in range(anchors)]
    routed: list[list[float]] = [[] for _ in range(anchors)]
    for sample in samples:
        best, best_distance = 0, None
        for index, block in enumerate(raw_blocks):
            distance = min(abs(sample - E2M1_VALUES[code]) for code in block)
            if best_distance is None or distance < best_distance:
                best, best_distance = index, distance
        routed[best].append(sample)

    blocks: list[tuple[int, ...]] = []
    for index, block in enumerate(raw_blocks):
        blocks.append(_order_block(block, routed[index], depth))
    return AnchorForest(rate=rate, blocks=tuple(blocks))


def _order_block(
    block: tuple[int, ...], assigned: "list[float]", depth: int
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
    split = (E2M1_VALUES[low[-1]] + E2M1_VALUES[high[0]]) / 2
    low_mass = [sample for sample in assigned if sample < split]
    high_mass = [sample for sample in assigned if sample >= split]
    left = _order_block(low, low_mass, depth - 1)
    right = _order_block(high, high_mass, depth - 1)
    # The representative of the whole node is the better of its two children's
    # representatives, and it must sit at index 0.  Swapping the halves is the
    # only reordering that achieves that while keeping the tree dyadic.
    pick = _best_representative((left[0], right[0]), assigned)
    return left + right if pick == left[0] else right + left
