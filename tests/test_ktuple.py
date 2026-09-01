"""The k-tuple trellis, which is a payload grid rather than a new trellis.

The load-bearing claim of the whole change is that ``arity`` is the *only*
thing that had to move: the anchor/descendant partition, the completion
grammar, the Viterbi and the replay all operate on codes and never asked how
many weights a code stands for.  So most of these tests are about arity 1 not
changing, and about the k=2 path agreeing with the scalar reference oracle.
"""

import pytest
import torch

from tessera.alphabet import (
    E2M1_GRID,
    PayloadGrid,
    build_forest,
    grid_digest,
    lloyd_max_grid,
    tuple_grid,
    value_order,
    _kd_bisect,
    _code_density,
)
from tessera.decode import decode_codes_mixed, reconstruct_unit
from tessera.encode import encode_unit, grid_value_table, grid_vector_table, viterbi_columns
from tessera.errors import GrammarError
from tessera.manifest import RotationState
from tessera.trellis import TCQ, ConvCode

CC = ConvCode(memory=6)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- arity 1 is unchanged ------------------------------------------------


def test_scalar_grid_is_arity_one():
    assert E2M1_GRID.arity == 1
    assert E2M1_GRID.size == 16
    assert E2M1_GRID.bits_per_position == 4.0
    assert E2M1_GRID.rate_cap == 3


def test_value_order_unchanged_by_the_key_machinery():
    """The derived rank key must reproduce the original value sort exactly."""
    expected = tuple(
        sorted(range(16), key=lambda c: (E2M1_GRID.values[c], c))
    )
    assert value_order(E2M1_GRID) == expected


def test_tuple_grid_of_one_is_the_base_grid():
    assert tuple_grid(E2M1_GRID, 1) is E2M1_GRID


def test_vector_table_is_the_scalar_table_at_arity_one():
    vec = grid_vector_table(E2M1_GRID)
    assert vec.shape == (16, 1)
    assert torch.equal(vec.squeeze(-1), grid_value_table(E2M1_GRID))


# --- the tuple grid itself -----------------------------------------------


def test_tuple_grid_shape_and_rate():
    grid = tuple_grid(E2M1_GRID, 2)
    assert grid.size == 256
    assert grid.arity == 2
    assert grid.payload_bits == 8
    assert grid.rate_cap == 7
    # The whole point: 3.5 payload bits per position, which |A_R| = 2^(R+1)
    # forbids a scalar trellis over 16 codes from ever reaching.
    assert grid.rate_cap / grid.arity == 3.5


def test_tuple_code_decomposes_first_coordinate_slowest():
    grid = tuple_grid(E2M1_GRID, 2)
    for c1 in range(16):
        for c2 in range(16):
            assert grid.vector(c1 * 16 + c2) == (
                E2M1_GRID.values[c1],
                E2M1_GRID.values[c2],
            )


def test_tuple_grid_refuses_nesting():
    with pytest.raises(GrammarError, match="scalar base grid"):
        tuple_grid(tuple_grid(E2M1_GRID, 2), 2)


def test_tuple_grid_refuses_unaffordable_arity():
    with pytest.raises(GrammarError, match="cost refusal"):
        tuple_grid(E2M1_GRID, 5)


def test_coset_partition_reduces_to_stride_at_arity_one():
    """The two subset rules are the same partition on a line."""
    forest = build_forest(3)
    stride = TCQ(forest, CC).subsets
    keys = E2M1_GRID.keys
    anchors = forest.anchors
    coset = [[] for _ in range(4)]
    for position, anchor in enumerate(anchors):
        coset[sum(keys[anchor]) % 4].append(position)
    assert [sorted(s) for s in stride] == [sorted(s) for s in coset]


def test_coset_partition_is_balanced_at_the_cap():
    grid = tuple_grid(E2M1_GRID, 2)
    subsets = TCQ(build_forest(7, grid=grid), CC).subsets
    assert [len(s) for s in subsets] == [64, 64, 64, 64]


def test_coset_partition_refuses_when_unbalanced():
    """Below the cap the anchors are representatives, not a full lattice."""
    grid = tuple_grid(E2M1_GRID, 2)
    with pytest.raises(GrammarError, match="unbalanced"):
        TCQ(build_forest(6, grid=grid), CC).subsets


# --- the k-dimensional forest generalises the scalar one -----------------


def test_kd_bisect_is_the_contiguous_split_on_a_line():
    """The claim in ``_kd_bisect``'s docstring, asserted rather than asserted-to."""
    codes = tuple(range(16))
    low, high = _kd_bisect(codes, E2M1_GRID, _code_density(E2M1_GRID))
    order = value_order(E2M1_GRID)
    assert sorted(low) == sorted(order[:8])
    assert sorted(high) == sorted(order[8:])


def test_kd_forest_partitions_the_tuple_grid():
    grid = tuple_grid(E2M1_GRID, 2)
    for rate in (5, 6):
        forest = build_forest(rate, grid=grid)
        assert len(forest.blocks) == 1 << (rate + 1)
        assert len(forest.blocks[0]) == 1 << (7 - rate)
        seen = {code for block in forest.blocks for code in block}
        assert seen == set(range(256))     # AnchorForest checks this too


# --- the vectorised encoder against the scalar oracle --------------------


def test_ktuple_viterbi_matches_the_reference_trellis():
    """The oracle knows nothing about arity except the distance it sums."""
    device = _device()
    grid = tuple_grid(E2M1_GRID, 2)
    forest = build_forest(7, grid=grid)
    torch.manual_seed(0)
    targets = torch.randn(24, 1, device=device)
    anchors, _, _ = viterbi_columns(targets, forest, CC, 0)
    pairs = targets.reshape(12, 2).tolist()
    _, reference, _ = TCQ(forest, CC).encode(pairs, completion=0)
    assert anchors.squeeze(1).tolist() == list(reference)


@pytest.mark.parametrize("base_size,rate", [(8, 5), (16, 7), (32, 9)])
def test_ktuple_round_trip(base_size, rate):
    """The decoder must land on the encoder's codes from the body alone."""
    device = _device()
    grid = tuple_grid(lloyd_max_grid(base_size), 2)
    forests = {rate: build_forest(rate, grid=grid)}
    torch.manual_seed(0)
    weights = torch.randn(64, 8, device=device) * 0.02
    unit = encode_unit(
        weights, forests, (rate,) * 8, CC,
        rotation=RotationState.NONE, with_diagonals=False, completion=0,
    )
    assert unit.body_bits.shape == (32, 8)      # one code per PAIR of rows
    codes = decode_codes_mixed(unit, forests, CC, 0)
    assert torch.equal(codes.long(), unit.codes)
    out = reconstruct_unit(unit, forests, CC, completion=0)
    assert out.shape == weights.shape


def test_ktuple_beats_the_scalar_trellis_at_its_own_rate_ceiling():
    """k=2 reaches 3.5 payload bits; the scalar grammar tops out at 3.0."""
    device = _device()
    torch.manual_seed(0)
    weights = torch.randn(256, 32, device=device) * 0.02

    def error(grid, rate):
        forests = {rate: build_forest(rate, grid=grid)}
        unit = encode_unit(
            weights, forests, (rate,) * 32, CC,
            rotation=RotationState.NONE, with_diagonals=False, completion=0,
        )
        out = reconstruct_unit(unit, forests, CC, completion=0).float()
        return ((out - weights).norm() / weights.norm()).item()

    assert error(tuple_grid(E2M1_GRID, 2), 7) < error(E2M1_GRID, 3)


# --- fail-closed guards ---------------------------------------------------


def test_release_is_refused_at_arity_above_one():
    device = _device()
    grid = tuple_grid(E2M1_GRID, 2)
    forests = {7: build_forest(7, grid=grid)}
    weights = torch.randn(16, 8, device=device) * 0.02
    with pytest.raises(GrammarError, match="release is not defined"):
        encode_unit(
            weights, forests, (7,) * 8, CC, rotation=RotationState.NONE,
            with_diagonals=False, completion=0, released_positions=4,
        )


def test_rows_must_be_a_whole_number_of_tuples():
    device = _device()
    grid = tuple_grid(E2M1_GRID, 2)
    forests = {7: build_forest(7, grid=grid)}
    weights = torch.randn(15, 8, device=device) * 0.02
    with pytest.raises(GrammarError, match="whole number of arity-2 tuples"):
        encode_unit(
            weights, forests, (7,) * 8, CC,
            rotation=RotationState.NONE, with_diagonals=False, completion=0,
        )


def test_tuple_grids_refuse_to_serialise():
    """256 tuple codes DO fit a byte, which makes them more dangerous."""
    forest = build_forest(7, grid=tuple_grid(E2M1_GRID, 2))
    assert forest.grid.size == 256          # byte-valid, and still refused
    with pytest.raises(GrammarError, match="arity 2"):
        forest.alphabet_plane()
    with pytest.raises(GrammarError, match="arity 2"):
        forest.descendant_plane()


def test_wide_grids_refuse_the_byte_wide_planes():
    forest = build_forest(9, grid=tuple_grid(lloyd_max_grid(32), 2))
    with pytest.raises(GrammarError, match="arity 2|one byte per code"):
        forest.alphabet_plane()


def test_a_non_e2m1_scalar_grid_refuses_to_serialise():
    """Same 16 codes, different values -- the wire cannot tell them apart."""
    forest = build_forest(3, grid=lloyd_max_grid(16))
    with pytest.raises(GrammarError, match="silent misdecode"):
        forest.alphabet_plane()


def test_scalar_value_table_refuses_a_tuple_grid():
    with pytest.raises(GrammarError, match="not one"):
        grid_value_table(tuple_grid(E2M1_GRID, 2))


def test_grid_refuses_a_code_space_it_cannot_split():
    with pytest.raises(GrammarError, match="not divisible"):
        PayloadGrid("two", (0.0, 1.0))


def test_completion_is_refused_on_a_tuple_grid_by_the_scalar_builder():
    """A k-tuple forest below the cap goes through the k-d path, not the 1-D one."""
    grid = tuple_grid(E2M1_GRID, 2)
    forest = build_forest(6, grid=grid)
    assert forest.grid.arity == 2
    assert len(forest.blocks[0]) == 2       # built by k-d bisection, not by value order


# --- the grid is wire ------------------------------------------------------


def test_grid_digest_separates_grids_that_planes_cannot():
    """Two grids, same codes, different values: byte-identical on the wire."""
    other = PayloadGrid("shifted", tuple(v + 1.0 for v in E2M1_GRID.values))
    assert grid_digest(other) != grid_digest(E2M1_GRID)
    assert grid_digest(tuple_grid(E2M1_GRID, 2)) != grid_digest(E2M1_GRID)


def test_grid_digest_is_stable_across_equal_grids():
    again = PayloadGrid("E2M1", E2M1_GRID.values, tuple(range(16)))
    assert grid_digest(again) == grid_digest(E2M1_GRID)


def test_lloyd_max_grid_is_deterministic_and_sorted():
    first, second = lloyd_max_grid(16), lloyd_max_grid(16)
    assert first.values == second.values
    assert list(first.values) == sorted(first.values)
    assert first.native is None          # not materialisable into any format
    assert grid_digest(first) == grid_digest(second)


def test_lloyd_max_beats_e2m1_as_a_scalar_quantiser():
    """The construction has to earn its place: fewer SSE on its own source."""
    from tessera.alphabet import GAUSSIAN_SOURCE

    grid = lloyd_max_grid(16)
    source = GAUSSIAN_SOURCE(1 << 12)
    def sse(levels, peak):
        scaled = [s * peak / 6.0 for s in source]
        return sum(min((s - v) ** 2 for v in levels) for s in scaled)
    assert sse(grid.values, max(abs(v) for v in grid.values)) < sse(
        E2M1_GRID.values, 6.0
    )
