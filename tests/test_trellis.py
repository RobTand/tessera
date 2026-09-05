"""Segment-0 trellis: rate, round-trip, and the coding gain that justifies it."""

from __future__ import annotations

import random

import pytest

from tessera.alphabet import E2M1_VALUES, build_forest
from tessera.errors import GrammarError
from tessera.trellis import SUBSET_COUNT, ConvCode, TCQ, _ODS_GENERATORS

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
    """An unknown memory order must refuse rather than invent generators.

    The pinned pair is a roster pin on purpose: the generators are wire, so
    the roster IS the decision (0o15, 0o17 is the published memory-3 ODS pair;
    the superseded 0o5, 0o7 -- the *memory-2* pair -- stays replayable for
    artifacts written under it, but is no longer the default).
    """
    with pytest.raises(GrammarError, match="wire, not an implementation detail"):
        ConvCode(memory=7)
    assert ConvCode(memory=3).generators == (0o15, 0o17)


def test_every_default_code_reads_its_current_input():
    """The select bit must reach the emitted subset on the step it is paid.

    ``ConvCode.step`` places the current input at register bit ``memory``.  A
    generator pair in which neither mask taps that bit emits the same subset
    on both branches of every state: the machine is a delayed lower-memory
    code, and the last select bit of every non-flushed stream is a paid bit
    that cannot reach any reconstruction.  The output is XOR-linear in the
    register, so state 0 is a complete check -- flipping the input flips
    output bit ``i`` exactly when generator ``i`` taps bit ``memory``,
    identically at every state.
    """
    for memory in sorted(_ODS_GENERATORS):
        code = ConvCode(memory=memory)
        assert code.step(0, 0)[1] != code.step(0, 1)[1], (memory, code.generators)


def test_every_default_code_uses_its_whole_register():
    """Both ends of the register must be live at the declared memory order.

    If no generator taps bit ``memory`` (the current input) the code is the
    previous order's machine delayed one step; if none taps bit 0 (the oldest
    cell) it is the previous order's machine with double the states.  Either
    way the wire pays for a memory order the code does not have -- the
    pre-fix memory-3 entry (0o5, 0o7) was exactly the first case.
    """
    for memory in sorted(_ODS_GENERATORS):
        code = ConvCode(memory=memory)
        combined = code.generators[0] | code.generators[1]
        assert combined & (1 << memory), (memory, code.generators)
        assert combined & 1, (memory, code.generators)


def test_the_last_select_bit_of_a_stream_is_observable():
    """A one-position rate-1 stream spends exactly one select bit; the two
    values of that bit must decode to two different anchors, or the encoder
    charged a bit for a decision with one outcome (no flush is emitted, so
    the terminal step's output is the only place the bit can land)."""
    tcq = TCQ(build_forest(1), SMALL)
    assert tcq.decode([0], 1) != tcq.decode([1], 1)


def test_decode_refuses_a_wrong_length_stream():
    tcq = TCQ(build_forest(3), SMALL)
    with pytest.raises(GrammarError, match="decode needs"):
        tcq.decode([0] * 10, 96)
