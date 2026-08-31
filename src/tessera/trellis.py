"""Segment-0 body: Tessera's trellis over the Stage-C anchors (doc S5, S6).

S5 makes segment 0 a "TCQ trellis body ... per-column integer rates R in
{1,2,3}"; S6 gives the trellis its alphabet: ``|A_R| = 2^(R+1)`` anchors, and
a position's code is the anchor the trellis lands on, refined by Stage C's c
completion bits.  So the body is a rate-R trellis quantiser over ``2^(R+1)``
reconstruction levels -- one level of redundancy over the R bits spent, which
is what buys the coding gain.

That cardinality fixes the machine.  With ``2^(R+1)`` anchors and R bits per
position, one bit must drive a state machine and the rest select within its
reachable set.  Splitting the value-ordered anchors by stride into four
subsets and letting a rate-1/2 convolutional code choose the subset gives
subsets of ``2^(R+1)/4 = 2^(R-1)`` anchors, so ``R = 1 + (R-1)`` closes at
every legal rate:

===  ===========  ==========  ===============
R    ``|A_R|``    subset      bits/position
===  ===========  ==========  ===============
1    4            1           1 = 1 + 0
2    8            2           2 = 1 + 1
3    16          4           3 = 1 + 2
===  ===========  ==========  ===============

**Anticipated-completion metric.**  S9 requires "Base Viterbi with an
anticipated-completion metric: anchor choices scored at the intended
post-completion values".  ``encode`` therefore scores a branch by the error of
the *best reachable descendant* at the terminal's completion level, not by the
anchor's own value.  Scoring anchors directly would optimise a quantity the
artifact never decodes.

**Declared, not assumed.**  The convolutional code's memory order and
generators are wire: two encoders that disagree on them produce streams that
do not decode to each other.  They are parameters of ``ConvCode``, they are
covered by the encoder profile id, and the default is the published
maximum-free-distance rate-1/2 code at that memory order (Lin & Costello,
table of ODS codes) -- a textbook object, chosen so the choice is checkable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .alphabet import E2M1_VALUES, AnchorForest, value_order
from .errors import GrammarError

__all__ = ["ConvCode", "TCQ", "SUBSET_COUNT"]

#: The convolutional code emits two bits, so it selects one of four subsets.
#: This is a property of a rate-1/2 code, not a tunable.
SUBSET_COUNT = 4

#: Published maximum-free-distance rate-1/2 generators, by memory order.
_ODS_GENERATORS = {
    3: (0o5, 0o7),
    4: (0o23, 0o35),
    5: (0o53, 0o75),
    6: (0o133, 0o171),
    8: (0o561, 0o753),
}


@dataclass(frozen=True)
class ConvCode:
    """The rate-1/2 convolutional code that drives subset selection."""

    memory: int = 6
    generators: "tuple[int, int] | None" = None

    def __post_init__(self) -> None:
        if self.generators is None:
            if self.memory not in _ODS_GENERATORS:
                raise GrammarError(
                    f"no default generators for memory {self.memory}; "
                    f"have {sorted(_ODS_GENERATORS)}. Pass them explicitly -- "
                    "they are wire, not an implementation detail."
                )
            object.__setattr__(self, "generators", _ODS_GENERATORS[self.memory])

    @property
    def states(self) -> int:
        return 1 << self.memory

    def step(self, state: int, bit: int) -> "tuple[int, int]":
        """One input bit -> ``(next_state, subset)``."""
        register = (bit << self.memory) | state
        out = 0
        for index, tap in enumerate(self.generators):
            out |= (bin(register & tap).count("1") & 1) << index
        return register >> 1, out


@dataclass(frozen=True)
class TCQ:
    """A rate-R trellis quantiser over one anchor forest.

    This is the **reference** implementation: exact, scalar, and slow.  It is
    the oracle the vectorised encoder is tested against, not the production
    path -- principle 7 puts the production encoder on the GPU.
    """

    forest: AnchorForest
    code: ConvCode = ConvCode()

    @property
    def rate(self) -> int:
        return self.forest.rate

    @property
    def subsets(self) -> "tuple[tuple[int, ...], ...]":
        """The anchors, value-ordered and split by stride into four subsets."""
        anchors = self.forest.anchors
        order = [a for a in value_order() if a in set(anchors)]
        index = {anchor: position for position, anchor in enumerate(anchors)}
        ordered = [index[a] for a in order]
        return tuple(tuple(ordered[offset::SUBSET_COUNT]) for offset in range(SUBSET_COUNT))

    @property
    def point_bits(self) -> int:
        """Bits spent selecting within the subset the code chose."""
        return self.rate - 1

    def _reachable_value_error(self, anchor: int, target: float, completion: int) -> float:
        """Squared error at the best descendant reachable at level ``completion``."""
        codes = self.forest.reachable(anchor, completion)
        return min((target - E2M1_VALUES[code]) ** 2 for code in codes)

    def decode(self, bits: "list[int]", length: int) -> "list[int]":
        """Replay the code: input bits -> anchor indices. Exact, no search."""
        expected = length * self.rate
        if len(bits) != expected:
            raise GrammarError(
                f"decode needs {expected} bits for {length} positions, got {len(bits)}"
            )
        subsets = self.subsets
        state, out, cursor = 0, [], 0
        for _ in range(length):
            select = bits[cursor]
            cursor += 1
            point = 0
            for _ in range(self.point_bits):
                point = (point << 1) | bits[cursor]
                cursor += 1
            state, subset = self.code.step(state, select)
            out.append(subsets[subset][point])
        return out

    def encode(self, targets, completion: int = 0):
        """Exact Viterbi. Returns ``(bits, anchor_indices, sse)``.

        A true minimum-cost path, not a greedy per-position pick: the coding
        gain is the entire reason segment 0 is a trellis, and a greedy encoder
        would forfeit it while still emitting a decodable stream.
        """
        subsets = self.subsets
        n_states, inf = self.code.states, float("inf")
        cost = [0.0] + [inf] * (n_states - 1)
        back: "list[list[tuple[int, int, int]]]" = []

        for target in targets:
            new_cost = [inf] * n_states
            step: "list[tuple[int, int, int]]" = [(-1, 0, 0)] * n_states
            for state in range(n_states):
                here = cost[state]
                if here == inf:
                    continue
                for select in (0, 1):
                    nxt, subset = self.code.step(state, select)
                    best_point, best_err = 0, inf
                    for point, anchor in enumerate(subsets[subset]):
                        err = self._reachable_value_error(anchor, target, completion)
                        if err < best_err:
                            best_point, best_err = point, err
                    total = here + best_err
                    if total < new_cost[nxt]:
                        new_cost[nxt] = total
                        step[nxt] = (state, select, best_point)
            cost = new_cost
            back.append(step)

        end = min(range(n_states), key=lambda s: cost[s])
        sse, state = cost[end], end
        bits_rev, anchors_rev = [], []
        for step in reversed(back):
            prev, select, point = step[state]
            _, subset = self.code.step(prev, select)
            anchors_rev.append(subsets[subset][point])
            for shift in range(self.point_bits):
                bits_rev.append((point >> shift) & 1)
            bits_rev.append(select)
            state = prev
        return bits_rev[::-1], anchors_rev[::-1], sse
