"""Segment-0 trellis: rate, round-trip, and the coding gain that justifies it."""

from __future__ import annotations

import random

import pytest

from tessera.alphabet import E2M1_VALUES, build_forest
from tessera.errors import GrammarError
from tessera.trellis import SUBSET_COUNT, ConvCode, TCQ

RATES = (1, 2, 3)
SMALL = ConvCode(memory=3)


def _targets(count=96, sigma=0.54, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(count)]


@pytest.mark.parametrize("rate", RATES)
def test_body_spends_exactly_rate_bits_per_position(rate):
    targets = _targets()
    bits, anchors, _ = TCQ(build_forest(rate), SMALL).encode(targets)
    assert len(bits) == rate * len(targets)
    assert len(anchors) == len(targets)


@pytest.mark.parametrize("rate", RATES)
def test_round_trip(rate):
    """Decode is a replay, so it must reproduce the encoder's anchors exactly."""
    targets = _targets()
    tcq = TCQ(build_forest(rate), SMALL)
    bits, anchors, _ = tcq.encode(targets, completion=3 - rate)
    assert tcq.decode(bits, len(targets)) == anchors


@pytest.mark.parametrize("rate", RATES)
def test_subsets_partition_the_anchors(rate):
    tcq = TCQ(build_forest(rate), SMALL)
    subsets = tcq.subsets
    assert len(subsets) == SUBSET_COUNT
    flat = sorted(index for subset in subsets for index in subset)
    assert flat == list(range(1 << (rate + 1)))
    # 2^(R+1)/4 anchors per subset, so R = 1 + (R-1) closes.
    assert all(len(subset) == (1 << (rate + 1)) // SUBSET_COUNT for subset in subsets)


@pytest.mark.parametrize("rate", RATES)
def test_viterbi_beats_greedy(rate):
    """The coding gain is the reason segment 0 is a trellis, not a rounder."""
    targets = _targets()
    tcq = TCQ(build_forest(rate), SMALL)
    completion = 3 - rate
    _, _, viterbi_sse = tcq.encode(targets, completion=completion)

    # Greedy: at each step take the locally best branch, ignoring the future.
    subsets, state, greedy_sse = tcq.subsets, 0, 0.0
    for target in targets:
        best = None
        for select in (0, 1):
            nxt, subset = tcq.code.step(state, select)
            for anchor in subsets[subset]:
                err = min(
                    (target - E2M1_VALUES[code]) ** 2
                    for code in tcq.forest.reachable(anchor, completion)
                )
                if best is None or err < best[0]:
                    best = (err, nxt)
        greedy_sse += best[0]
        state = best[1]
    assert viterbi_sse <= greedy_sse


def test_higher_rate_is_never_worse_at_c_full():
    """At C-full every root costs 3 bits/column; quality is not equalised.

    S6 equalises the roots on *cost* -- "C-full costs R + (3-R) = 3 bits per
    column from every root" -- and says they "differ only in anchor-tree
    depth".  That residual difference is real and large, which is why the low
    roots earn their place at low bpp rather than at 3.0.
    """
    targets = _targets()
    sse = [
        TCQ(build_forest(rate), SMALL).encode(targets, completion=3 - rate)[2]
        for rate in RATES
    ]
    assert sse[0] > sse[1] > sse[2]


def test_completion_level_is_honoured_by_the_metric():
    """Deeper completion can only reduce the anticipated error."""
    targets = _targets()
    tcq = TCQ(build_forest(1), SMALL)
    sse = [tcq.encode(targets, completion=c)[2] for c in (0, 1, 2)]
    assert sse[0] >= sse[1] >= sse[2]


def test_generators_are_declared_wire():
    """An unknown memory order must refuse rather than invent generators."""
    with pytest.raises(GrammarError, match="wire, not an implementation detail"):
        ConvCode(memory=7)
    assert ConvCode(memory=3).generators == (0o5, 0o7)


def test_decode_refuses_a_wrong_length_stream():
    tcq = TCQ(build_forest(3), SMALL)
    with pytest.raises(GrammarError, match="decode needs"):
        tcq.decode([0] * 10, 96)
