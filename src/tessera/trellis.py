"""Segment-0 TCQ body: the trellis code, its alphabet, and exact Viterbi.

§5 calls segment 0 a "TCQ trellis body; root rate r0 ... per-column integer
rates R in {1,2,3}", and §6 fixes the alphabet cardinality at
``|A_R| = 2^(R+1)``.  Those two facts pin the construction to classical
Ungerboeck-partitioned trellis-coded quantisation (Marcellin & Fischer, §18):

* the alphabet holds ``2^(R+1)`` reconstruction levels -- one bit of redundancy
  over the ``R`` bits actually spent;
* the levels are value-ordered and split by **stride into four subsets**, which
  maximises the intra-subset spacing;
* per position, one input bit drives a rate-1/2 convolutional code whose two
  output bits choose the subset, and the remaining ``R-1`` bits choose a point
  inside it.  Subsets hold ``2^(R+1)/4 = 2^(R-1)`` points, so the arithmetic
  closes for every legal R:

  =====  ==========  ==========  ================
  R      ``|A_R|``   subset      bits/position
  =====  ==========  ==========  ================
  1      4           1           1 = 1 + 0
  2      8           2           2 = 1 + 1
  3      16          4           3 = 1 + 2
  =====  ==========  ==========  ================

The doc's own rate-2 fixture is the corroboration: §6 records that gridbook's
reviewed ``(15, 13, 11, 9, 8, 2, 4, 7)`` "sits at indices 0,2,4,6,8,10,12,15 of
the value-ordered alphabet: stride-2 Ungerboeck partitioning with the final slot
snapped."  Stride partitioning of a value-ordered alphabet is exactly what this
module does.

**Scope.** At ``R = 3`` the alphabet is the whole 16-code E2M1 grid, so no
choice is made and nothing here is a convention.  That is why §6 calls R=3
defined and leaves rate-1/rate-2 to build item 2: at those rates the alphabet is
a *subset* of the grid, and which subset is a decision this module does not get
to make.  ``select_alphabet`` therefore refuses R<3 rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import GrammarError
from .grammar import alphabet_size

__all__ = [
    "E2M1_VALUES",
    "E2M1_VALUE_ORDER",
    "ConvCode",
    "TCQ",
    "select_alphabet",
    "SUBSETS",
]

#: The 16 NVFP4 E2M1 codes, indexed by their on-wire nibble: bit 3 is the sign,
#: bits 2..0 index the magnitude ladder.  This is the grid §6's Stage-C
#: descendant sets partition at ``c = 3 - R``.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_VALUES: tuple[float, ...] = tuple(
    (-1.0 if code >> 3 else 1.0) * _E2M1_MAGNITUDES[code & 7] for code in range(16)
)

#: Nibbles sorted by reconstruction value -- the order set partitioning walks.
#: Note -0.0 (nibble 8) and +0.0 (nibble 0) both sit at zero; the sort is stable
#: so the positive zero leads and the pair stays adjacent, which keeps the
#: stride partition balanced.
E2M1_VALUE_ORDER: tuple[int, ...] = tuple(
    sorted(range(16), key=lambda c: (E2M1_VALUES[c], c))
)

#: Four subsets by stride over the value order (Ungerboeck).
SUBSETS: tuple[tuple[int, ...], ...] = tuple(
    tuple(E2M1_VALUE_ORDER[offset::4]) for offset in range(4)
)


@dataclass(frozen=True)
class ConvCode:
    """A rate-1/2 convolutional code driving subset selection.

    ``generators`` are octal taps in the usual TCQ presentation; the default
    (5, 7) is the standard 8-state code.  ``states`` is ``2 ** memory``.
    """

    memory: int = 3
    generators: tuple[int, int] = (0o5, 0o7)

    @property
    def states(self) -> int:
        return 1 << self.memory

    def step(self, state: int, bit: int) -> tuple[int, int]:
        """Return ``(next_state, subset)`` for one input bit."""
        register = (bit << self.memory) | state
        out = 0
        for index, tap in enumerate(self.generators):
            parity = bin(register & tap).count("1") & 1
            out |= parity << index
        return register >> 1, out


def select_alphabet(rate: int) -> tuple[int, ...]:
    """The value-ordered nibble alphabet for ``rate``.

    Refuses R<3.  §6: "defining the rate-1/rate-2 convention is build item 2,
    and it gates the sub-3 ladder."  A plausible-looking subset chosen here
    would be a convention smuggled in as an implementation detail, and the one
    in-tree rate-2 fixture is explicitly "one fixture, not a convention".
    """
    size = alphabet_size(rate)
    if rate < 3:
        raise GrammarError(
            f"rate {rate} needs a {size}-code alphabet, which is build item 2 "
            "(the rate-1/rate-2 convention). Only rate 3 is defined: its "
            "alphabet is the whole 16-code E2M1 grid, so nothing is chosen."
        )
    if size != 16:
        raise GrammarError(f"rate {rate} is outside the legal set {{1, 2, 3}}")
    return E2M1_VALUE_ORDER


@dataclass(frozen=True)
class TCQ:
    """A rate-R trellis quantiser over the E2M1 grid."""

    rate: int
    code: ConvCode = ConvCode()

    def __post_init__(self) -> None:
        select_alphabet(self.rate)  # refuses undefined rates at construction

    @property
    def subset_bits(self) -> int:
        """Bits spent choosing a point inside the selected subset."""
        return self.rate - 1

    @property
    def points_per_subset(self) -> int:
        return alphabet_size(self.rate) // 4

    def subset_points(self, subset: int) -> tuple[int, ...]:
        return SUBSETS[subset]

    # ---- decode ---------------------------------------------------------

    def decode(self, bits: list[int], length: int) -> list[int]:
        """Replay the code: input bits -> E2M1 nibbles. The decoder is exact."""
        expected = length * self.rate
        if len(bits) != expected:
            raise GrammarError(
                f"decode needs {expected} bits for {length} positions, got {len(bits)}"
            )
        state, out, cursor = 0, [], 0
        for _ in range(length):
            select = bits[cursor]
            cursor += 1
            point = 0
            for _ in range(self.subset_bits):
                point = (point << 1) | bits[cursor]
                cursor += 1
            state, subset = self.code.step(state, select)
            out.append(self.subset_points(subset)[point])
        return out

    # ---- encode ---------------------------------------------------------

    def encode(self, targets) -> tuple[list[int], list[int], float]:
        """Exact Viterbi over the whole sequence.

        Returns ``(bits, nibbles, sse)``.  This is a true minimum-squared-error
        path through the trellis, not a greedy per-position pick -- that is the
        entire coding gain of TCQ, and a greedy encoder would silently forfeit
        it while still producing a decodable stream.
        """
        n_states = self.code.states
        inf = float("inf")
        cost = [0.0] + [inf] * (n_states - 1)
        back: list[list[tuple[int, int, int]]] = []

        for target in targets:
            new_cost = [inf] * n_states
            step_back: list[tuple[int, int, int]] = [(-1, 0, 0)] * n_states
            for state in range(n_states):
                here = cost[state]
                if here == inf:
                    continue
                for select in (0, 1):
                    nxt, subset = self.code.step(state, select)
                    best_point, best_err = 0, inf
                    for point, nibble in enumerate(self.subset_points(subset)):
                        err = (target - E2M1_VALUES[nibble]) ** 2
                        if err < best_err:
                            best_point, best_err = point, err
                    total = here + best_err
                    if total < new_cost[nxt]:
                        new_cost[nxt] = total
                        step_back[nxt] = (state, select, best_point)
            cost, sb = new_cost, step_back
            back.append(sb)

        end = min(range(n_states), key=lambda s: cost[s])
        sse = cost[end]
        bits_rev, nibbles_rev, state = [], [], end
        for step in reversed(back):
            prev, select, point = step[state]
            _, subset = self.code.step(prev, select)
            nibbles_rev.append(self.subset_points(subset)[point])
            for shift in range(self.subset_bits):
                bits_rev.append((point >> shift) & 1)
            bits_rev.append(select)
            state = prev
        return bits_rev[::-1], nibbles_rev[::-1], sse
