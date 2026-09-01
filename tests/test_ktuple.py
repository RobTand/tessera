"""The k-tuple trellis, which is a payload grid rather than a new trellis.

The load-bearing claim of the whole change is that ``arity`` is the *only*
thing that had to move: the anchor/descendant partition, the completion
grammar, the Viterbi and the replay all operate on codes and never asked how
many weights a code stands for.  So most of these tests are about arity 1 not
changing, and about the k=2 path agreeing with the scalar reference oracle.
"""

import dataclasses
from fractions import Fraction

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


def test_below_the_cap_the_partition_is_balanced_by_stride_instead():
    """Below the cap the anchors are representatives, not a full lattice.

    This used to raise.  Refusing was the right response to an unbalanced
    split -- the point field is a fixed ``R-1`` bits, so unequal subsets are
    unencodable, not merely worse -- but refusing at the *partition* left every
    sub-cap rung of every k-tuple family unreachable, which is why the family
    looked like a two-point set rather than a rate ladder.  The stride rule is
    balanced for any anchor count divisible by four, so the constraint is met
    without giving anything up, and only where the coset rule had raised.
    """
    grid = tuple_grid(E2M1_GRID, 2)
    for rate in range(1, 7):
        subsets = TCQ(build_forest(rate, grid=grid), CC).subsets
        width = (1 << (rate + 1)) // 4
        assert [len(s) for s in subsets] == [width] * 4, rate
        assert sorted(p for s in subsets for p in s) == list(range(1 << (rate + 1)))


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


def test_tuple_grids_now_serialise_because_the_wire_commits_to_the_grid():
    """256 tuple codes fit a byte, and the profile id now says *which* byte map.

    This test used to assert the opposite.  The refusal was never about arity
    as such -- it was that nothing on the wire recorded the grid, so an arity-2
    artifact and an arity-1 one were byte-indistinguishable.  Now that
    ``encoder_profile_id`` absorbs ``grid_digest``, that ambiguity is gone and
    the planes may be written.
    """
    forest = build_forest(7, grid=tuple_grid(E2M1_GRID, 2))
    assert forest.grid.size == 256
    assert len(forest.alphabet_plane()) == 256      # one byte per anchor
    assert forest.descendant_plane() is not None


def test_the_alphabet_plane_survives_code_255():
    """The top code of a 256-entry grid must round-trip through a uint8 plane.

    A byte plane holds 0..255 exactly, so code 255 is the boundary that a
    silent uint8 wrap would land on -- and a round-trip test over random
    weights can miss it, because nothing forces the top code to be selected.
    Probe it directly.
    """
    forest = build_forest(7, grid=tuple_grid(E2M1_GRID, 2))
    plane = forest.alphabet_plane()
    anchors = forest.anchors
    assert max(anchors) == 255, "R=7 over 256 codes must reach the top code"
    assert tuple(plane) == anchors
    assert all(0 <= b <= 255 for b in plane)


def test_wide_grids_refuse_the_byte_wide_planes():
    forest = build_forest(9, grid=tuple_grid(lloyd_max_grid(32), 2))
    with pytest.raises(GrammarError, match="SERIALISABLE_GRIDS|one byte per code"):
        forest.alphabet_plane()


def test_a_non_e2m1_scalar_grid_refuses_to_serialise():
    """Same 16 codes, different values -- no reader can resolve the digest."""
    forest = build_forest(3, grid=lloyd_max_grid(16))
    with pytest.raises(GrammarError, match="SERIALISABLE_GRIDS"):
        forest.alphabet_plane()


def test_a_free_grid_is_refused_by_name_not_by_accident():
    """The deferral is explicit: a fitted grid needs its values on the wire."""
    from tessera.alphabet import SERIALISABLE_GRIDS, grid_digest as _gd
    free = lloyd_max_grid(16)
    assert _gd(free) not in SERIALISABLE_GRIDS
    with pytest.raises(GrammarError, match="VALUES plane"):
        build_forest(3, grid=free).alphabet_plane()


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


# --------------------------------------------------------------------------
# The wire, at arity 2.  Everything above this line is about the encoder; the
# tests below are the ones that would have caught a schema change that got the
# *layout* right and the *identity* wrong (or vice versa).
# --------------------------------------------------------------------------


def _arity2_unit(rows=16, cols=8, rate=7):
    device = _device()
    grid = tuple_grid(E2M1_GRID, 2)
    forests = {rate: build_forest(rate, grid=grid)}
    torch.manual_seed(11)
    weights = torch.randn(rows, cols, device=device) * 0.02
    unit = encode_unit(
        weights, forests, (rate,) * cols, CC,
        rotation=RotationState.NONE, with_diagonals=False, completion=0,
        group=32, half=16,
    )
    return weights, unit, forests, grid


def test_arity2_artifact_round_trips_through_bytes():
    """build -> serialize -> parse -> decode, bit-exact against the direct decode.

    This is the claim the whole schema change exists to support: a k-tuple body
    survives the wire.  Comparing against ``reconstruct_unit`` on the *encoder's
    own* unit (rather than against the source weights) isolates the wire from
    the quantiser -- any difference here is a serialisation bug, not encoder
    error.
    """
    from tessera.unit_artifact import build_unit_artifact, read_unit_artifact

    weights, unit, forests, grid = _arity2_unit()
    direct = reconstruct_unit(unit, forests, CC)
    _manifest, _region, blob = build_unit_artifact(unit, "u0", forests, q256=7 * 256)
    off_wire = read_unit_artifact(blob, device=direct.device)
    assert off_wire.shape == weights.shape
    assert torch.equal(off_wire, direct)


def test_arity2_geometry_declares_weight_rows_not_trellis_steps():
    """The accountant divides by ``quantizable_params``; halving it inflates bpp.

    A code covers two rows, so every per-code plane has ``rows // 2`` entries.
    Recording that step count as ``geometry.rows`` would declare half the
    parameters and report double the true bits-per-parameter -- the one number
    this format exists to state honestly.
    """
    from tessera.footprint import terminal_payload_bpp
    from tessera.unit_artifact import build_unit_artifact

    _weights, unit, forests, _grid = _arity2_unit(rows=16, cols=8)
    assert unit.body_bits.shape == (8, 8)          # 16 rows / arity 2
    manifest, _region, _blob = build_unit_artifact(unit, "u0", forests, q256=7 * 256)
    assert manifest.geometry.rows == 16
    assert manifest.geometry.quantizable_params == 16 * 8

    # bpp needs a unit big enough for the body to dominate: the ALPHABET and
    # DESCENDANT planes are 256 bytes each *per unit* regardless of size, which
    # is 32 bpp of pure overhead on a 128-parameter toy and says nothing.
    _w2, big, big_forests, _g2 = _arity2_unit(rows=256, cols=256)
    manifest2, _r2, _b2 = build_unit_artifact(big, "u1", big_forests, q256=7 * 256)
    assert manifest2.geometry.quantizable_params == 256 * 256
    bpp = terminal_payload_bpp(manifest2, manifest2.terminals[0])
    assert bpp > Fraction(7, 2), "payload cannot be cheaper than the body itself"
    assert bpp < 5, f"arity-2 bpp {float(bpp)} looks like a halved denominator"


def test_both_accountants_agree_on_an_arity2_unit():
    """The standalone rate calculator and the built artifact must not drift.

    ``terminal_rate`` prices a family from integers alone; ``build_unit_artifact``
    lays out real bytes.  They are separate implementations of one schema, so
    they are exactly the pair that can disagree silently -- and at arity 2 an
    arity-blind calculator over-reports by the arity (7.5 bpp against a true
    4.03), which would land straight in a published table.

    They agree byte for byte once the forest planes are added back: the
    calculator is called with empty ALPHABET/DESCENDANT blobs by construction,
    so it prices the position-domain planes only.  Asserting the *identity*
    rather than equality states exactly what each accountant covers.
    """
    from tessera.calculator import terminal_rate
    from tessera.unit_artifact import build_unit_artifact

    rows, cols, rate = 256, 256, 7
    _w, unit, forests, grid = _arity2_unit(rows=rows, cols=cols, rate=rate)
    manifest, _region, _blob = build_unit_artifact(
        unit, "u0", forests, q256=rate * 256
    )
    terminal = manifest.terminals[0]
    body_rate = terminal_rate(
        rate * 256, rows, cols,
        with_scale_base=True, with_scale_refine=True, with_diagonals=False,
        completion=0, superblock_columns=256,
        cap=grid.rate_cap, arity=grid.arity,
    )
    # R=7 over a 2-row tuple: 3.5 body bits + 0.5 of segment-2b scale.
    assert body_rate == 4, f"the 4.0 bpp rung priced at {float(body_rate)}"
    forest_bytes = (
        len(forests[rate].alphabet_plane()) + len(forests[rate].descendant_plane())
    )
    assert (
        Fraction(terminal.exact_bytes) - body_rate * rows * cols / 8 == forest_bytes
    ), "the layout and the calculator disagree by something other than the forests"


def test_the_profile_id_separates_two_grids_at_the_same_shape():
    """Arity 1 and arity 2 must not collide -- that collision was the bug."""
    from tessera.unit_artifact import encoder_profile_id

    rates = (3,)
    scalar = encoder_profile_id(CC, rates, E2M1_GRID)
    tupled = encoder_profile_id(CC, rates, tuple_grid(E2M1_GRID, 2))
    assert scalar != tupled


def test_an_artifact_over_an_unknown_grid_fails_closed_on_read():
    """A profile id no reader can resolve must raise, not decode plausibly."""
    import hashlib

    from tessera.container import parse, serialize
    from tessera.errors import GrammarError as GE
    from tessera.unit_artifact import build_unit_artifact, read_unit_artifact

    _weights, unit, forests, _grid = _arity2_unit()
    manifest, region, _blob = build_unit_artifact(unit, "u0", forests, q256=7 * 256)
    forged = dataclasses.replace(
        manifest, encoder_profile_id=hashlib.sha256(b"some other grid").digest()
    )
    with pytest.raises(GE, match="payload grid"):
        read_unit_artifact(serialize(forged, region))
