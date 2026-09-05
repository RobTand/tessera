"""Build item 2: the rate-1/rate-2 alphabet convention, and its obligations.

Build item 1 owes "tiny exhaustive tests per R in {1,2,3} x c in {0..3-R}
proving nesting (every completion prefix is a valid partial map), unique
decode, legal truncation at quota boundaries, and code cardinality at every
prefix".  The first three are alphabet properties and are tested here.
"""

from __future__ import annotations

import itertools

import pytest

from tessera.alphabet import (
    E2M1_VALUES,
    GAUSSIAN_SOURCE,
    AnchorForest,
    build_forest,
    value_order,
)
from tessera.errors import GrammarError

RATES = (1, 2, 3)


def test_value_order_is_ascending_with_positive_zero_first():
    order = value_order()
    assert len(set(order)) == 16
    values = [E2M1_VALUES[code] for code in order]
    assert values == sorted(values)
    # The signed zeros tie; the tie-break must be deterministic and must put
    # +0 first, because it decides which zero is an anchor below rate 3.
    assert order.index(0) < order.index(8)


@pytest.mark.parametrize("rate", RATES)
def test_descendant_sets_partition_the_grid(rate):
    """S6: at c = 3 - R "every code is a descendant of exactly one anchor"."""
    forest = build_forest(rate)
    seen = [code for block in forest.blocks for code in block]
    assert sorted(seen) == list(range(16))


@pytest.mark.parametrize("rate", RATES)
def test_cardinality_at_every_prefix(rate):
    """|A_R| = 2^(R+1) anchors, and |D(a)| = 2^c at every completion level."""
    forest = build_forest(rate)
    assert len(forest.anchors) == 1 << (rate + 1)
    for completion in range(3 - rate + 1):
        for anchor in range(len(forest.blocks)):
            assert len(forest.reachable(anchor, completion)) == 1 << completion


@pytest.mark.parametrize("rate", RATES)
def test_nesting_every_prefix_is_a_valid_partial_map(rate):
    """A truncated completion word must land on an ancestor of the full word."""
    forest = build_forest(rate)
    depth = 3 - rate
    for anchor in range(len(forest.blocks)):
        for bits in itertools.product((0, 1), repeat=depth):
            full = forest.decode(anchor, bits)
            for prefix_len in range(depth + 1):
                shallow = forest.decode(anchor, bits[:prefix_len])
                assert shallow in forest.reachable(anchor, prefix_len)
            # c = 0 always decodes to the anchor itself.
            assert forest.decode(anchor, ()) == forest.anchors[anchor]
            assert full in forest.blocks[anchor]


@pytest.mark.parametrize("rate", RATES)
def test_unique_decode(rate):
    """Distinct (anchor, completion word) pairs decode to distinct codes."""
    forest = build_forest(rate)
    depth = 3 - rate
    decoded = [
        forest.decode(anchor, bits)
        for anchor in range(len(forest.blocks))
        for bits in itertools.product((0, 1), repeat=depth)
    ]
    assert len(decoded) == len(set(decoded)) == 16


@pytest.mark.parametrize("rate", RATES)
def test_planes_are_the_declared_widths(rate):
    forest = build_forest(rate)
    assert len(forest.alphabet_plane()) == 1 << (rate + 1)
    # The forest flattens to the whole grid at every rate, by the partition.
    assert len(forest.descendant_plane()) == 16


def test_forest_rejects_a_non_partition():
    with pytest.raises(GrammarError, match="partition|two anchors"):
        AnchorForest(rate=2, blocks=tuple(((0, 1),) * 8))


def test_forest_rejects_wrong_descendant_width():
    with pytest.raises(GrammarError, match=r"\|D\(a\)\|"):
        AnchorForest(rate=2, blocks=tuple((code,) for code in range(16))[:8])


def test_build_is_deterministic():
    """An alphabet that moved run to run would make artifacts irreproducible."""
    assert build_forest(2).blocks == build_forest(2).blocks


def test_forest_selection_prices_the_roots_it_emits():
    """The partition the selector picks must win on the forest as emitted.

    ``_partition_cost`` prices a partition by the best *member* of each block,
    but ``_order_block``'s recursion promotes each node's representative from
    its two children's, so the member the selector priced can be eliminated on
    the way up.  On this source (the tessera#223 witness) the mass-balanced
    candidate wins the member-min screen while its completed tree's actual
    ``c = 0`` anchors are ~7.6% worse than the contiguous candidate's.  The
    forest that comes back must be no worse, at ``c = 0`` on its own emitted
    roots, than any candidate partition completed the same way.
    """
    from bisect import bisect

    from tessera.alphabet import (
        E2M1_GRID,
        GROUP_SCALED_SOURCE,
        _mass_balanced_blocks,
        _order_block,
        _partition_cost,
    )
    from tessera.grammar import alphabet_size, completion_capacity

    grid = E2M1_GRID
    rate = 1
    samples = GROUP_SCALED_SOURCE(6.0, count=1024)
    depth = completion_capacity(rate, grid.rate_cap)
    width = 1 << depth
    anchors = alphabet_size(rate, cap=grid.rate_cap)
    order = value_order(grid)

    def emitted_sse(blocks):
        roots = sorted(grid.values[block[0]] for block in blocks)
        cuts = [(roots[i] + roots[i + 1]) / 2.0 for i in range(len(roots) - 1)]
        return sum((s - roots[bisect(cuts, s)]) ** 2 for s in samples)

    contiguous = [order[i * width : (i + 1) * width] for i in range(anchors)]
    balanced, reps = _mass_balanced_blocks(grid, samples, anchors, width)
    best = min(
        emitted_sse(
            [
                _order_block(block, routed[i], depth, grid)
                for i, block in enumerate(raw)
            ]
        )
        for raw, routed in (
            (contiguous, _partition_cost(grid, samples, contiguous)[1]),
            (balanced, _partition_cost(grid, samples, balanced, seed=reps)[1]),
        )
    )
    forest = build_forest(rate, samples=samples, grid=grid)
    assert emitted_sse(forest.blocks) <= best


def test_derived_anchors_beat_the_reviewed_fixture_at_matched_scale():
    """The recorded ablation: the fixture's extreme-snap costs SSE.

    S6's fixture ``(15, 13, 11, 9, 8, 2, 4, 7)`` keeps |6.0| reachable at c=0;
    SSE-optimal selection does not.  Compared at each set's own optimal group
    scale -- the fair comparison, since the scale is free per group -- the
    derived set wins.  A synthetic screen, not a served result.
    """
    samples = GAUSSIAN_SOURCE(1 << 12)
    fixture = (15, 13, 11, 9, 8, 2, 4, 7)
    derived = build_forest(2).anchors

    def sse_at(anchors, scale):
        values = [scale * E2M1_VALUES[a] for a in anchors]
        return sum(min((x - v) ** 2 for v in values) for x in samples)

    def best(anchors):
        lo, hi = 0.02, 3.0
        for _ in range(50):
            a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if sse_at(anchors, a) < sse_at(anchors, b):
                hi = b
            else:
                lo = a
        return sse_at(anchors, (lo + hi) / 2)

    assert best(derived) < best(fixture)
    # The fixture sits where S6 says it does, which is what makes it a
    # meaningful comparator rather than an arbitrary tuple.
    order = value_order()
    assert tuple(order.index(c) for c in fixture) == (0, 2, 4, 6, 8, 10, 12, 15)
