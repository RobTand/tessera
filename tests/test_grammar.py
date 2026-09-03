"""S6 refinement-grammar obligations.

"Tiny exhaustive tests per R in {1,2,3} x c in {0..3-R} proving nesting (every
completion prefix is a valid partial map), unique decode, legal truncation at
quota boundaries, and code cardinality at every prefix."
"""

from fractions import Fraction
from itertools import product

import pytest

from tessera.errors import GrammarError
from tessera.grammar import (
    C_FULL_BITS,
    GRID_CODES,
    LEGAL_RATES,
    RELEASE_BITS,
    alphabet_size,
    bits_per_position,
    bresenham_rate_schedule,
    completion_capacity,
    descendant_set_size,
    prefix_cardinality,
    release_quota,
    root_from_q256,
    superblock_count,
    superblock_quota_ok,
    superblock_widths,
    validate_descendant_map,
    validate_rate_schedule,
)

ROOTS = (256, 384, 512, 640, 768)


def partition_map(rate, completion):
    """A structurally valid descendant map: contiguous blocks of the grid."""
    anchors = alphabet_size(rate)
    size = descendant_set_size(completion)
    return {
        anchor: tuple(range(anchor * size, anchor * size + size))
        for anchor in range(anchors)
    }


def test_alphabet_size_is_two_to_the_rate_plus_one():
    assert [alphabet_size(rate) for rate in LEGAL_RATES] == [4, 8, 16]


def test_c_full_costs_three_bits_from_every_root():
    """Completion equalises all roots at the joint-16 wire."""
    for rate in LEGAL_RATES:
        assert rate + completion_capacity(rate) == C_FULL_BITS
        assert bits_per_position(rate, completion_capacity(rate)) == C_FULL_BITS


def test_partition_cardinality_is_forced_for_every_rate():
    """|A_R| * 2**(3-R) == 16 exactly, for every legal R."""
    for rate in LEGAL_RATES:
        assert alphabet_size(rate) * descendant_set_size(
            completion_capacity(rate)
        ) == GRID_CODES


def test_cardinality_at_every_prefix_is_two_to_the_c():
    """Nesting: the reachable set is 2**c at every completion prefix."""
    for rate in LEGAL_RATES:
        for completion in range(completion_capacity(rate) + 1):
            assert prefix_cardinality(rate, completion) == 1 << completion


def test_every_completion_prefix_is_a_valid_partial_map():
    """Exhaustive over R in {1,2,3} x c in {0..3-R}."""
    for rate in LEGAL_RATES:
        for completion in range(completion_capacity(rate) + 1):
            validate_descendant_map(rate, completion, partition_map(rate, completion))


def test_full_completion_partitions_the_grid():
    for rate in LEGAL_RATES:
        capacity = completion_capacity(rate)
        mapping = partition_map(rate, capacity)
        covered = sorted(code for codes in mapping.values() for code in codes)
        assert covered == list(range(GRID_CODES))
        validate_descendant_map(rate, capacity, mapping)


def test_non_partitioning_map_is_rejected_at_c_full():
    """Overlapping descendant sets break unique decode and must fail closed."""
    rate, capacity = 2, completion_capacity(2)
    mapping = partition_map(rate, capacity)
    mapping[0] = mapping[1]  # collide two anchors
    with pytest.raises(GrammarError, match="disjoint"):
        validate_descendant_map(rate, capacity, mapping)


def test_completion_beyond_capacity_is_rejected():
    for rate in LEGAL_RATES:
        with pytest.raises(GrammarError):
            bits_per_position(rate, completion_capacity(rate) + 1)


def test_release_everywhere_is_not_an_endpoint():
    """3 + 4 = 7 bits/column is never byte-competitive with scalar 4.5."""
    for rate in LEGAL_RATES:
        released = bits_per_position(rate, completion_capacity(rate), released=True)
        assert released == C_FULL_BITS + RELEASE_BITS == 7
        assert released > Fraction(45, 10)


def test_bresenham_quota_is_exact_for_every_root():
    for q256 in ROOTS:
        root = root_from_q256(q256)
        for columns in (16, 32, 256, 4096):
            schedule = bresenham_rate_schedule(root, columns)
            assert len(schedule) == columns
            assert sum(schedule) == root * columns
            validate_rate_schedule(schedule, root)


def test_half_integer_roots_mix_exactly_the_bracketing_rates():
    for q256, expected in ((384, {1, 2}), (640, {2, 3})):
        schedule = bresenham_rate_schedule(root_from_q256(q256), 32)
        assert set(schedule) == expected
        assert schedule.count(min(expected)) == schedule.count(max(expected))


def test_superblock_quota_holds_for_bresenham_and_can_fail_for_others():
    root = root_from_q256(384)
    schedule = bresenham_rate_schedule(root, 32)
    assert superblock_quota_ok(schedule, 8, root)
    # An arrangement that front-loads the high rate breaks a complete superblock.
    front_loaded = tuple(sorted(schedule, reverse=True))
    assert not superblock_quota_ok(front_loaded, 8, root)


def test_unrealisable_root_is_rejected():
    with pytest.raises(GrammarError, match="not realisable"):
        bresenham_rate_schedule(root_from_q256(384), 3)


def test_rates_outside_the_shaped_domain_are_rejected():
    for bad in (0, 4, 5):
        with pytest.raises(GrammarError):
            alphabet_size(bad)


def test_inexact_quota_is_rejected():
    with pytest.raises(GrammarError, match="inexact quota"):
        validate_rate_schedule((1, 1, 1), Fraction(2))


# ------------------------------------------------------- the release quota
#
# Issue #27.  The quota over a partial trailing superblock used to be an equal
# *count* per superblock, which on a narrow tail is an equal count over unequal
# blocks: up to ``superblock_columns`` times the release density of the rest of
# the unit, and enough to ask a block for more releases than it has positions.
# The quota is width-proportional instead, and these are the three properties
# that makes it: it is exact, it never overruns, and at a complete width it is
# element for element what the equal-count spread was -- which is why the bytes
# of every artifact ever written are unchanged.


def _equal_count_spread(total, blocks):
    """The spread ``release_quota`` replaced, kept here as the oracle."""
    per, remainder = divmod(total, blocks)
    return tuple(per + (1 if index < remainder else 0) for index in range(blocks))


@pytest.mark.parametrize(
    "columns,widths",
    [
        (256, (256,)),
        (128, (128,)),
        (512, (256, 256)),
        (640, (256, 256, 128)),
        (257, (256, 1)),
        (320, (256, 64)),
        (1024, (256, 256, 256, 256)),
    ],
)
def test_superblock_widths_partition_the_columns(columns, widths):
    assert superblock_widths(columns, 256) == widths
    assert sum(widths) == columns
    assert len(widths) == superblock_count(columns, 256)


@pytest.mark.parametrize("blocks", range(1, 9))
def test_release_quota_is_the_equal_count_spread_at_a_complete_width(blocks):
    """The compatibility property, and the reason this change moved no bytes.

    At equal widths every exact share is ``total / blocks`` and every
    fractional part is equal, so the largest-remainder award falls to the
    lowest indices -- which is exactly what ``divmod`` front-loading did.  A
    cumulative-floor Bresenham would *not* have this property (at 3 blocks and
    a remainder of 2 it awards blocks 1 and 2, not 0 and 1), and would have
    moved the released set of every unit whose column count is a whole number
    of superblocks.
    """
    columns = blocks * 256
    totals = set(range(0, 400))
    totals |= {blocks * k + r for k in (1, 7, 133, 4001) for r in range(blocks)}
    for total in sorted(totals):
        assert release_quota(total, columns, 256) == _equal_count_spread(
            total, blocks
        ), (blocks, total)


@pytest.mark.parametrize("columns", [1, 3, 128, 255, 256, 257, 320, 384, 640, 897])
@pytest.mark.parametrize("rows", [1, 8, 64])
def test_release_quota_never_overruns_a_superblock(columns, rows):
    """The refusal ``encode._canonical_release_order`` guards is unreachable at
    any legal total: a superblock's count never exceeds its positions."""
    widths = superblock_widths(columns, 256)
    positions = rows * columns
    totals = set(range(0, min(positions, 300)))
    totals |= set(range(max(0, positions - 20), positions + 1))
    totals |= {positions // 2, positions // 7}
    for total in sorted(t for t in totals if 0 <= t <= positions):
        counts = release_quota(total, columns, 256)
        assert sum(counts) == total
        assert len(counts) == len(widths)
        for count, width in zip(counts, widths):
            assert count <= rows * width, (total, counts, widths)


@pytest.mark.parametrize("columns", [320, 384, 640, 768, 1024])
def test_release_quota_holds_the_density_equal_not_the_count(columns):
    """Every superblock is within one release of its exact share, which is the
    width-proportional reading of the promise the equal count made against
    every *other* superblock.  On a complete width the two coincide; on a
    partial one only this holds."""
    widths = superblock_widths(columns, 256)
    for total in (96, 1000, 3000, 4000):
        counts = release_quota(total, columns, 256)
        for count, width in zip(counts, widths):
            share = total * width / columns
            assert abs(count - share) < 1, (total, counts, widths)


def test_release_quota_refuses_a_negative_total():
    with pytest.raises(GrammarError, match="must not be negative"):
        release_quota(-1, 512, 256)


def test_superblock_widths_refuses_a_zero_column_unit():
    with pytest.raises(GrammarError, match="must be positive"):
        superblock_widths(0, 256)
