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

**Multidimensional partition (Wei 1987).**  ``span = L > 1`` groups L
consecutive positions of a column into one super-symbol and labels it by the
*sum* of its positions' subset labels mod 4.  One convolutional-code bit then
picks the super-label, ``2(L-1)`` bits pick which member of that label class
the positions take, and ``L(R-1)`` bits pick points inside the chosen
per-position subsets: ``LR + L - 1`` bits per super-symbol, ``R + (L-1)/L``
per position.  Position 0 of a super-symbol stores ``[select | point]`` (R
bits); positions ``1..L-1`` store ``[label | point]`` (R+1 bits); position 0's
label is *derived*: ``(super-label - sum of the stored labels) mod 4``.  At
``L = 1`` there are no stored labels and the layout is byte-identical to the
per-position trellis.  A rate-k/(k+1) code can only ever spend a whole bit per
position; the fractional redundancy comes from the partition, not the code.

**Declared, not assumed.**  The convolutional code's memory order and
generators are wire: two encoders that disagree on them produce streams that
do not decode to each other.  They are parameters of ``ConvCode``, they are
covered by the encoder profile id, and the default is the published
maximum-free-distance rate-1/2 code at that memory order (Lin & Costello,
table of ODS codes) -- a textbook object, chosen so the choice is checkable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .alphabet import SUBSET_COUNT, AnchorForest, value_order
from .errors import GrammarError

__all__ = ["ConvCode", "TCQ", "SUBSET_COUNT", "body_bits"]

# ``SUBSET_COUNT`` is defined in ``alphabet`` -- the grid must refuse a code
# space it cannot split -- and re-exported here, where it reads naturally.

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
        # A memory order below 1 has no register to select on.  ``encode.py``
        # reads the select bit as ``state >> (memory - 1)``, and torch answers
        # a negative shift with **zero** rather than raising, so a zero-memory
        # code would not fault -- it would encode every position from a
        # silently-constant select bit.  The default-generator lookup below
        # would have caught it; passing generators explicitly walked straight
        # past.  Refuse it here, where both routes meet.
        if self.memory < 1:
            raise GrammarError(
                f"memory order {self.memory} has no shift register: a "
                "convolutional code needs at least one memory element, and "
                "the select bit is read as state >> (memory - 1)"
            )
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
        """The anchors, ordered and split into four subsets.

        Two rules, and they agree wherever both are defined at arity 1:

        - ``stride`` walks the value order and takes every fourth anchor.  This
          is Ungerboeck partitioning on a line and it is what every scalar
          Tessera artifact was built with, so it stays the arity-1 default
          verbatim -- including at ``R < cap``, where the anchors are a
          *subset* of the grid and their ranks are no longer contiguous.
        - ``coset`` groups by ``sum of rank vector mod 4``.  On a line that is
          the same partition as ``stride``; in ``k`` dimensions it is the
          standard multidimensional generalisation, and it is the only one of
          the two that is guaranteed balanced when the anchors are a lattice
          rather than an interval.

        Balance is not cosmetic: an unbalanced split gives subsets of different
        sizes, and the point field is a fixed ``R-1`` bits wide.  The ``coset``
        rule is balanced only where the anchors *are* that lattice, which on a
        k-tuple grid means the rate cap and nothing else; below the cap it is
        the stride rule that holds, and the fallback below is what makes the
        sub-cap rungs of a k-tuple family encodable at all.
        """
        anchors = self.forest.anchors
        grid = self.forest.grid
        if grid.arity > 1 and grid.partition == "coset":
            keys = grid.keys or ()
            groups: "list[list[int]]" = [[] for _ in range(SUBSET_COUNT)]
            for position, anchor in enumerate(anchors):
                groups[sum(keys[anchor]) % SUBSET_COUNT].append(position)
            width = len(anchors) // SUBSET_COUNT
            if all(len(group) == width for group in groups):
                return tuple(tuple(group) for group in groups)
            if len(anchors) % SUBSET_COUNT:
                raise GrammarError(
                    f"{grid.name} at rate {self.rate} has {len(anchors)} "
                    f"anchors, which no rule splits into {SUBSET_COUNT} equal "
                    f"subsets"
                )
            # Balance is structural, not a preference: the code emits one of
            # four subsets per step and the point field is a fixed R-1 bits, so
            # unequal subsets are not a worse partition, they are an unencodable
            # one.  The rank-sum coset rule is balanced only when the anchors
            # are a full lattice -- true at the rate cap, where every code is an
            # anchor, and false at every rate below it, where the anchors are
            # the representatives of k-d bisection blocks chosen for error and
            # not for their residues.  Rate 3 of E2M1x2 splits [3,5,4,4].
            #
            # So below the cap the partition falls back to the stride rule,
            # which is balanced by construction for any anchor count divisible
            # by four.  ``_build_forest_kd`` already sorts anchors by rank sum
            # then rank vector -- neighbouring anchors are near in value -- so
            # taking every fourth separates neighbours, which is what
            # Ungerboeck partitioning is for.
            #
            # This cannot move a byte of any artifact that exists: it is
            # reached only where the coset rule *raised*, so every rate it
            # newly admits was previously unencodable.  The rate cap keeps the
            # coset partition verbatim.
            return tuple(
                tuple(range(len(anchors)))[offset::SUBSET_COUNT]
                for offset in range(SUBSET_COUNT)
            )
        order = [a for a in value_order(grid) if a in set(anchors)]
        index = {anchor: position for position, anchor in enumerate(anchors)}
        ordered = [index[a] for a in order]
        return tuple(tuple(ordered[offset::SUBSET_COUNT]) for offset in range(SUBSET_COUNT))

    @property
    def point_bits(self) -> int:
        """Bits spent selecting within the subset the code chose."""
        return self.rate - 1

    def _reachable_value_error(self, anchor: int, target, completion: int) -> float:
        """Squared error at the best descendant reachable at level ``completion``.

        ``target`` is a scalar at arity 1 and a sequence of ``arity`` values
        otherwise; the metric is the squared Euclidean distance either way.
        """
        codes = self.forest.reachable(anchor, completion)
        grid = self.forest.grid
        if grid.arity == 1:
            return min((target - grid.values[code]) ** 2 for code in codes)
        return min(
            sum((t - v) ** 2 for t, v in zip(target, grid.vector(code)))
            for code in codes
        )

    def decode(
        self, bits: "list[int]", length: int, span: int = 1, initial_state: int = 0
    ) -> "list[int]":
        """Replay the code: input bits -> anchor indices. Exact, no search.

        ``span`` is the super-symbol length L.  Each super-symbol is read as
        ``[select | point_0]`` then ``[label_i | point_i]`` for ``i = 1..L-1``;
        the code's output for ``select`` is the super-label, and position 0's
        subset is the super-label minus the stored labels, mod 4.

        ``initial_state`` is the register the stream starts from.  Zero is the
        pinned start of a whole unit; a **shard** cut below row 0 carries its
        parent's register on the INITIAL_STATE plane (``layout.slice_unit``),
        and this scalar walk is the oracle the vectorised replay's closed-form
        correction is tested against.
        """
        if not 0 <= initial_state < self.code.states:
            raise GrammarError(
                f"initial state {initial_state} outside the code's "
                f"{self.code.states} states"
            )
        expected = body_bits(self.rate, length, span)
        if len(bits) != expected:
            raise GrammarError(
                f"decode needs {expected} bits for {length} positions at span "
                f"{span}, got {len(bits)}"
            )
        subsets = self.subsets
        state, out, cursor = initial_state, [], 0
        for _ in range(length // span):
            select = bits[cursor]
            cursor += 1
            fields = []
            for position in range(span):
                label = 0
                if position:
                    label = (bits[cursor] << 1) | bits[cursor + 1]
                    cursor += 2
                point = 0
                for _ in range(self.point_bits):
                    point = (point << 1) | bits[cursor]
                    cursor += 1
                fields.append((label, point))
            state, super_label = self.code.step(state, select)
            stored = sum(label for label, _ in fields[1:])
            for position, (label, point) in enumerate(fields):
                subset = (super_label - stored) % SUBSET_COUNT if position == 0 else label
                out.append(subsets[subset][point])
        return out

    def encode(self, targets, completion: int = 0, span: int = 1):
        """Exact Viterbi. Returns ``(bits, anchor_indices, sse)``.

        A true minimum-cost path, not a greedy per-position pick: the coding
        gain is the entire reason segment 0 is a trellis, and a greedy encoder
        would forfeit it while still emitting a decodable stream.

        At ``span = L > 1`` one trellis step covers L positions.  For a branch
        emitting super-label ``s`` the best member of the class is found by
        exhaustion over the ``4^(L-1)`` stored-label assignments -- exact and
        slow, which is what an oracle is for.  The vectorised encoder does the
        same minimisation as a min-plus fold over Z/4; the two must agree on
        the summed squared error to the digit.
        """
        targets = list(targets)
        if span < 1 or len(targets) % span:
            raise GrammarError(
                f"{len(targets)} positions is not a whole number of span-{span} "
                "super-symbols"
            )
        subsets = self.subsets
        n_states, inf = self.code.states, float("inf")
        cost = [0.0] + [inf] * (n_states - 1)
        back: "list[list[tuple[int, int, tuple[tuple[int, int], ...]]]]" = []

        # Per position, per subset: the best point and its error.  Independent
        # of the trellis state, so it is computed once per position.
        per_position = []
        for target in targets:
            best = []
            for subset in range(SUBSET_COUNT):
                best_point, best_err = 0, inf
                for point, anchor in enumerate(subsets[subset]):
                    err = self._reachable_value_error(anchor, target, completion)
                    if err < best_err:
                        best_point, best_err = point, err
                best.append((best_err, best_point))
            per_position.append(best)

        for start in range(0, len(targets), span):
            block = per_position[start : start + span]
            # For each super-label: the cheapest label assignment whose sum is
            # that label, as (error, ((label, point) per position)).
            by_label: "list[tuple[float, tuple]]" = [(inf, ())] * SUBSET_COUNT
            for stored in _label_assignments(span - 1):
                first = (-sum(stored)) % SUBSET_COUNT
                for super_label in range(SUBSET_COUNT):
                    label_0 = (super_label + first) % SUBSET_COUNT
                    labels = (label_0,) + stored
                    err = sum(block[i][labels[i]][0] for i in range(span))
                    if err < by_label[super_label][0]:
                        by_label[super_label] = (
                            err,
                            tuple((labels[i], block[i][labels[i]][1]) for i in range(span)),
                        )
            new_cost = [inf] * n_states
            step: "list[tuple[int, int, tuple]]" = [(-1, 0, ())] * n_states
            for state in range(n_states):
                here = cost[state]
                if here == inf:
                    continue
                for select in (0, 1):
                    nxt, super_label = self.code.step(state, select)
                    err, fields = by_label[super_label]
                    total = here + err
                    if total < new_cost[nxt]:
                        new_cost[nxt] = total
                        step[nxt] = (state, select, fields)
            cost = new_cost
            back.append(step)

        end = min(range(n_states), key=lambda s: cost[s])
        sse, state = cost[end], end
        bits_rev, anchors_rev = [], []
        for step in reversed(back):
            prev, select, fields = step[state]
            # Emit the super-symbol's bits in reverse, position by position.
            for position in range(span - 1, -1, -1):
                label, point = fields[position]
                anchors_rev.append(subsets[label][point])
                for shift in range(self.point_bits):
                    bits_rev.append((point >> shift) & 1)
                if position:
                    bits_rev.append(label & 1)
                    bits_rev.append(label >> 1)
                else:
                    bits_rev.append(select)
            state = prev
        return bits_rev[::-1], anchors_rev[::-1], sse


def body_bits(rate: int, positions: int, span: int = 1) -> int:
    """Bits the body spends on ``positions`` codes at ``rate`` and ``span``.

    ``span * rate + span - 1`` per super-symbol: one select bit, ``span - 1``
    two-bit stored labels, ``span`` point fields of ``rate - 1`` bits.  At
    ``span = 1`` this is ``rate * positions``, the per-position trellis.
    """
    if span < 1:
        raise GrammarError(f"span must be positive, got {span}")
    if positions % span:
        raise GrammarError(
            f"{positions} positions is not a whole number of span-{span} "
            "super-symbols"
        )
    return (span * rate + span - 1) * (positions // span)


def _label_assignments(count: int):
    """Every tuple of ``count`` subset labels, in lexicographic order."""
    if count == 0:
        yield ()
        return
    for head in range(SUBSET_COUNT):
        for tail in _label_assignments(count - 1):
            yield (head,) + tail
