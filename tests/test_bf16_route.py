"""The 16-bit route: the BF16 grid, its wire, and the tile it decodes to.

Three claims, each with its own failure mode:

1. **The grid is bf16 and nothing else.**  A code is a bf16 bit pattern, so a
   reader rebuilds every value from the name, the window table's snap *is*
   bf16 rounding, and the ALPHABET plane viewed as ``torch.bfloat16`` is the
   table a kernel gathers from.  Each of those is asserted rather than
   asserted-about: they are what entitles the kernel lane to take the view.
2. **The wire round-trips at its declared width.**  Two bytes an element on
   the code plane, priced by the accountant to the byte, and the reader
   resolves the grid off the profile id like any other.
3. **Three decode paths, one tensor.**  ``reconstruct_unit`` (fp32),
   ``materialize_bf16`` (the served tile) and the streamed decoder over the
   packed wire are bit-identical, which is what lets a serving lane hold the
   wire and a stock twin hold the tile and call them the same artifact.

Everything here is CPU: the route has no Triton path of its own, and the
window Viterbi's reference implementation is the definition.
"""
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import (  # noqa: E402
    BF16_GRID,
    E4M3_GRID,
    GAUSSIAN_SOURCE,
    PayloadGrid,
    SERIALISABLE_GRIDS,
    grid_digest,
)
from tessera.bf16_route import (  # noqa: E402
    BF16_FAMILY,
    prepare_bf16_unit,
    stream_bf16_tile,
    stream_bf16_unscaled,
    window_table_values,
)
from tessera.calculator import terminal_rate  # noqa: E402
from tessera.decode import (  # noqa: E402
    materialize_bf16, materialize_bf16_unscaled, materialize_fp8, reconstruct_unit)
from tessera.encode import grid_vector_table, window_table  # noqa: E402
from tessera.errors import GrammarError  # noqa: E402
from tessera.export import (  # noqa: E402
    BF16_CHANNEL_SIGMA,
    BF16_RECIPE,
    BF16_WINDOW_BITS,
    E4M3_RECIPE,
    encode_linear_planes,
    recipe_at,
    recipe_table,
    wire_recipe,
)
from tessera.manifest import BodyKind, ScalePlaneKind  # noqa: E402
from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact  # noqa: E402

#: Small enough that the reference Viterbi is quick, wide enough that a
#: Bresenham schedule really is mixed-rate.
SHAPE = (32, 96)
#: L=8 here, not the recipe's 14: the table's construction and the wire's
#: element width are what these tests are about, and 2^14 states per position
#: on CPU is minutes.  ``test_recipe_is_the_shipping_window`` pins the real L.
TEST_L = 8


def _weight(rows=SHAPE[0], cols=SHAPE[1], seed=0):
    torch.manual_seed(seed)
    return (torch.randn(rows, cols) * torch.linspace(0.2, 3.0, rows)[:, None]).float()


def _encode(q256=1024, window_bits=TEST_L, weight=None):
    w = _weight() if weight is None else weight
    exported, unit, forests = encode_linear_planes(
        w, grid=BF16_GRID, q256=q256, name="unit", window_bits=window_bits,
    )
    return w, exported, unit, forests


# --------------------------------------------------------------- the grid


def test_a_code_is_its_own_bf16_word():
    """65536 codes, code == bit pattern, ``payload_bits`` 16.

    This is the property the whole route rests on: it makes the ALPHABET
    plane the kernel's table, and it makes the grid reader-reconstructible
    from the name alone, which is `SERIALISABLE_GRIDS`' criterion.
    """
    assert BF16_GRID.size == 1 << 16
    assert BF16_GRID.payload_bits == 16 and BF16_GRID.arity == 1
    assert BF16_GRID.code_bytes == 2
    values = torch.tensor(BF16_GRID.values)
    patterns = (
        torch.arange(1 << 16, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    )
    finite = torch.isfinite(patterns)
    assert int(finite.sum()) == (1 << 16) - 256      # exponent 255 is Inf/NaN
    assert torch.equal(values[finite], patterns[finite].float())


def test_the_non_finite_slots_carry_their_legal_neighbour():
    """E4M3FN's two NaN bytes, at bf16's scale: 256 patterns (exponent 255
    over a 7-bit mantissa, both signs), mapped back by ``native`` so a
    materialised tile is always finite."""
    native = torch.tensor(BF16_GRID.native, dtype=torch.long)
    assert int((native != torch.arange(1 << 16)).sum()) == 256
    assert torch.isfinite(torch.tensor(BF16_GRID.values)).all()
    assert BF16_GRID.native[0x7F80] == 0x7F7F and BF16_GRID.native[0xFF80] == 0xFF7F


def test_the_grid_is_in_the_registry_and_is_deterministic():
    assert grid_digest(BF16_GRID) in SERIALISABLE_GRIDS
    from tessera.alphabet import _bf16_value

    rebuilt = PayloadGrid("BF16", tuple(_bf16_value(b) for b in range(1 << 16)),
                          BF16_GRID.native)
    assert grid_digest(rebuilt) == grid_digest(BF16_GRID)


def test_the_window_table_snap_is_bf16_rounding():
    """The table is ``2^L`` Gaussian quantiles snapped to the grid.  On BF16
    that snap is nearest-value over the whole format, which is bf16
    round-to-nearest -- exactly, at the shipping (L, sigma, seed).

    Ties are the one place the two rules can differ: the snap breaks to the
    lower code (toward zero), hardware breaks to even.  This asserts the
    count of entries where they disagree is zero *and reports it*, so a width
    or seed that lands on a midpoint is visible rather than silent.
    """
    table = window_table(BF16_GRID, BF16_WINDOW_BITS, sigma=BF16_CHANNEL_SIGMA,
                         seed=0, half=16)
    assert table.dtype is torch.int32 and table.numel() == 1 << BF16_WINDOW_BITS
    values = grid_vector_table(BF16_GRID)[table.long()].reshape(-1)
    quantiles = torch.tensor(GAUSSIAN_SOURCE(1 << BF16_WINDOW_BITS, BF16_CHANNEL_SIGMA))
    generator = torch.Generator().manual_seed(0)
    points = quantiles[torch.randperm(1 << BF16_WINDOW_BITS, generator=generator)]
    assert int((points.float().to(torch.bfloat16).float() != values).sum()) == 0
    # No state ever names a non-finite slot, so no decode can produce one.
    native = torch.tensor(BF16_GRID.native, dtype=torch.long)
    assert torch.equal(native[table.long()], table.long())


def test_the_window_table_is_deterministic_in_its_parameters():
    args = dict(sigma=BF16_CHANNEL_SIGMA, seed=0, half=16)
    first = window_table(BF16_GRID, TEST_L, **args)
    assert torch.equal(first, window_table(BF16_GRID, TEST_L, **args))
    assert not torch.equal(first, window_table(BF16_GRID, TEST_L, sigma=2.0,
                                               seed=0, half=16))
    assert not torch.equal(first, window_table(BF16_GRID, TEST_L, sigma=BF16_CHANNEL_SIGMA,
                                               seed=1, half=16))


def test_the_table_reach_is_what_the_row_start_uses():
    """The reach-aware per-row start reads the body's reach off the table
    (``encode_unit``'s CHANNEL branch), so on this grid the reach is the
    table's largest magnitude and nothing else -- 4.00 sigma at the shipping
    width, next to E4M3's 4.08."""
    table = window_table(BF16_GRID, BF16_WINDOW_BITS, sigma=BF16_CHANNEL_SIGMA,
                         seed=0, half=16)
    reach = float(grid_vector_table(BF16_GRID)[table.long()].abs().max())
    assert reach == pytest.approx(4.0, abs=0.05)


def test_a_high_row_is_started_inside_the_reach():
    """The dense-outlier fix, on this grid.

    ``encode_unit``'s CHANNEL branch computes the body's reach from **the
    table** -- ``|grid_value(table[state])|.max()`` -- and starts any row
    whose largest weight would land past it lower, so nothing is clipped to
    the table's extreme entry before the first trellis pass.  On BF16 the
    reach is the table's and not a constant, which is what this asserts:
    the same row started against a wider table gets a different word.
    """
    from tessera.scale_channel import initial_channel_scale

    w = _weight()
    w[0] *= 40.0                                     # one very wide row
    table = window_table(BF16_GRID, TEST_L, sigma=BF16_CHANNEL_SIGMA, seed=0, half=16)
    reach = float(grid_vector_table(BF16_GRID)[table.long()].abs().max())
    stored, effective, _ = initial_channel_scale(w, BF16_CHANNEL_SIGMA, reach=reach)
    amax = w.abs().amax(dim=1)
    # Every row's largest weight starts inside the reach, to the fp16 word.
    assert bool((amax <= reach * effective * (1 + 1e-3)).all())
    # ...and without it the wide row would have been clipped before the
    # first pass: the plain RMS start puts its largest weight past the table.
    _, plain_eff, _ = initial_channel_scale(w, BF16_CHANNEL_SIGMA, reach=None)
    assert float(effective[0]) > float(plain_eff[0])
    assert float(amax[0]) > reach * float(plain_eff[0])
    # The encoder uses it: the wide row's stored word dominates the rest.
    _, _, unit, _ = _encode(weight=w)
    rows = unit.scale_rows.float()
    assert rows[0] > 4 * rows[1:].median()


# --------------------------------------------------------------- the recipe


def test_recipe_is_the_shipping_window():
    recipe = wire_recipe(BF16_GRID)
    assert recipe == BF16_RECIPE
    assert recipe.body is BodyKind.WINDOW and recipe.span == 1
    assert recipe.scale_plane is ScalePlaneKind.CHANNEL
    assert recipe.window_bits == BF16_WINDOW_BITS == E4M3_RECIPE.window_bits
    # Stated, never searched: a dyadic ladder over a 65536-value grid is a
    # 4096 x 65536 float64 matrix forty times over, and it is choosing
    # between equals on a format with eight exponent bits.
    assert recipe.channel_sigma == BF16_CHANNEL_SIGMA is not None


def test_a_rung_above_the_table_widens_the_table():
    """``L >= R``: a window position's R new bits are the low R bits of the
    state.  Above R = 14 the recipe widens rather than claiming a width the
    encoder would refuse -- and the table it names is on the wire, so a
    checkpoint says which one it used."""
    assert wire_recipe(BF16_GRID, 14 * 256).window_bits == 14
    assert wire_recipe(BF16_GRID, 14 * 256 + 1).window_bits == 15
    assert wire_recipe(BF16_GRID, 16 * 256).window_bits == 16
    table = recipe_table(BF16_GRID)
    assert recipe_at(table, 1024) == BF16_RECIPE
    assert recipe_at(table, 16 * 256).window_bits == 16
    # A no-op on the grids that ship today: their cap is below the table.
    assert {r.recipe for r in recipe_table(E4M3_GRID)} == {E4M3_RECIPE}


def test_the_tcq_body_is_not_offered_on_this_grid():
    """65536 anchors scored per trellis step is the cost the encoder refuses;
    the window body never scores the grid, which is why one is reachable and
    the other is not."""
    with pytest.raises(GrammarError, match="one byte per code|SERIALISABLE"):
        encode_linear_planes(_weight(), grid=BF16_GRID, q256=1024, name="u",
                             body=BodyKind.TCQ, verify=False)


# ----------------------------------------------------------------- the wire


@pytest.mark.parametrize("q256", [1024, 1280, 1536, 1792, 2048])
def test_the_wire_round_trips_at_every_product_rung(q256):
    w, exported, unit, forests = _encode(q256=q256)
    recovered = read_unit_artifact(exported.blob)
    assert torch.equal(recovered, reconstruct_unit(unit, forests, None))
    parsed = parse_unit_artifact(exported.blob)
    assert parsed.grid is BF16_GRID and parsed.code is None
    assert parsed.unit.window_bits == TEST_L
    assert torch.equal(parsed.unit.window_codes.long(), unit.window_codes.long())


def test_the_code_plane_is_two_bytes_an_element():
    """The one thing the wire had to learn.  The plane's element stays a byte
    -- the count doubles, the grammar does not -- and the bytes are
    little-endian, which is what makes them a bf16 buffer."""
    import numpy as np

    _, exported, unit, _ = _encode()
    from tessera.container import parse, plane_ranges
    from tessera.planes import PlaneKind

    art = parse(exported.blob)
    chunk = None
    for descriptor, offset, content, _total in plane_ranges(art.manifest, art.terminal):
        if descriptor.kind is PlaneKind.ALPHABET:
            chunk = art.plane_region[offset : offset + content]
    assert len(chunk) == 2 * (1 << TEST_L)
    written = np.frombuffer(bytes(chunk), dtype="<u2").astype(np.int64)
    assert torch.equal(torch.from_numpy(written), unit.window_codes.long())
    # And viewed as bf16 it is the reconstruction table itself.
    view = torch.frombuffer(bytearray(chunk), dtype=torch.uint16).view(torch.bfloat16)
    assert torch.equal(view, window_table_values(unit.window_codes))


@pytest.mark.parametrize("q256", [1024, 1536, 2048])
def test_the_accountant_prices_the_wide_table_exactly(q256):
    """The failure this catches is under-pricing BF16 by half its table --
    the accountant and the wire must agree byte for byte, on this grid too."""
    w, exported, _, _ = _encode(q256=q256)
    predicted = terminal_rate(
        q256, w.shape[0], w.shape[1], with_scale_base=False, with_scale_refine=False,
        with_row_scale=True, window_bits=TEST_L, cap=BF16_GRID.payload_bits,
        arity=1, span=1, code_bytes=BF16_GRID.code_bytes,
    )
    assert predicted == exported.bpp
    # Priced at one byte an element it would be short by exactly the table.
    narrow = terminal_rate(
        q256, w.shape[0], w.shape[1], with_scale_base=False, with_scale_refine=False,
        with_row_scale=True, window_bits=TEST_L, cap=BF16_GRID.payload_bits,
        arity=1, span=1, code_bytes=1,
    )
    assert exported.bpp - narrow == Fraction((1 << TEST_L) * 8, w.shape[0] * w.shape[1])


# ---------------------------------------------------------------- the tile


def test_materialise_is_one_rounding_of_the_reconstruction():
    _, exported, _, _ = _encode()
    parsed = parse_unit_artifact(exported.blob)
    tile = materialize_bf16(parsed.unit, parsed.grid, parsed.code)
    assert tile.dtype is torch.bfloat16
    assert torch.equal(tile, reconstruct_unit(parsed.unit, parsed.grid, None)
                       .to(torch.bfloat16))


def test_materialise_refuses_another_grid():
    """Another grid's codes are not bf16 words, and a materialiser that
    accepted them would hand a runtime a plausible wrong tile."""
    from tessera.alphabet import E4M3_GRID

    w = _weight()
    _, unit, forests = encode_linear_planes(
        w, grid=E4M3_GRID, q256=1024, name="u", window_bits=TEST_L, verify=False,
    )
    with pytest.raises(GrammarError, match="needs the scalar BF16 grid"):
        materialize_bf16(unit, forests, None)
    # ...and the FP8 materialiser refuses the BF16 unit, symmetrically.
    _, bf_unit, bf_forests = _encode()[1:]
    with pytest.raises(GrammarError, match="256-code hardware grid"):
        materialize_fp8(bf_unit, bf_forests, None)


def test_streamed_decode_is_bit_identical_to_the_tile():
    """The product mode and the correctness path are the same artifact: the
    lane holds the wire at 4-8 bpp and decodes, the stock twin holds the
    tile, and the two tensors are equal bit for bit."""
    _, exported, _, _ = _encode(q256=1536)
    parsed = parse_unit_artifact(exported.blob)
    streamed = prepare_bf16_unit(parsed.unit)
    assert torch.equal(
        stream_bf16_tile(streamed),
        materialize_bf16(parsed.unit, parsed.grid, parsed.code),
    )
    assert streamed.resident_bytes < 16 * SHAPE[0] * SHAPE[1] / 8 * 2


def test_streamed_decode_refuses_what_it_does_not_apply():
    from tessera.alphabet import E4M3_GRID

    _, _, unit, _ = _encode()
    unit.scale_plane = ScalePlaneKind.LUT
    with pytest.raises(GrammarError, match="one scale per output row"):
        prepare_bf16_unit(unit)


def test_the_no_fold_pair_rounds_the_weight_nowhere():
    """A CHANNEL scale is an output-row factor, so it commutes with the matmul
    and a lane never has to fold it in.  Two claims, both exact:

    the code tile is already bf16 (every table entry is a bf16 value on this
    grid, so the cast rounds nothing), and folding is the *only* place a
    rounding enters -- ``bf16(code * s)`` is the folded tile, to the bit.
    """
    _, exported, _, _ = _encode(q256=1536)
    parsed = parse_unit_artifact(exported.blob)
    values, scale = materialize_bf16_unscaled(parsed.unit, parsed.grid, parsed.code)
    assert values.dtype is torch.bfloat16 and scale.shape == (SHAPE[0],)
    assert torch.equal(values.float(), values.float().to(torch.bfloat16).float())
    tile = materialize_bf16(parsed.unit, parsed.grid, parsed.code)
    assert torch.equal(tile, (values.float() * scale[:, None]).to(torch.bfloat16))
    # And the epilogue really is the more accurate arrangement.
    torch.manual_seed(7)
    x = torch.randn(64, SHAPE[1])
    exact = x @ (values.float() * scale[:, None]).T
    epilogue = (x @ values.float().T) * scale[None, :]
    folded = x @ tile.float().T
    assert float((epilogue - exact).norm()) < float((folded - exact).norm())


def test_the_streamed_no_fold_pair_is_the_materialised_one():
    _, exported, _, _ = _encode(q256=1536)
    parsed = parse_unit_artifact(exported.blob)
    got_values, got_scale = stream_bf16_unscaled(prepare_bf16_unit(parsed.unit))
    values, scale = materialize_bf16_unscaled(parsed.unit, parsed.grid, parsed.code)
    assert torch.equal(got_values, values) and torch.equal(got_scale, scale)


def test_the_no_fold_pair_refuses_a_block_plane():
    from tessera.alphabet import E4M3_GRID

    w = _weight()
    _, unit, forests = encode_linear_planes(
        w, grid=E4M3_GRID, q256=1024, name="u", window_bits=TEST_L, verify=False,
    )
    with pytest.raises(GrammarError, match="needs the scalar BF16 grid"):
        materialize_bf16_unscaled(unit, forests, None)


def test_a_checkpoint_config_naming_this_grid_replays_it():
    """``grid_from_config`` resolved grids from a closed two-name map.

    A BF16 checkpoint written by ``export_checkpoint`` would have failed to
    replay -- not misread, but refused as an unknown base, which is a hole in
    the library path rather than in the wire.  The digest check is the point:
    the name selects a grid and the digest proves it is the same grid.
    """
    from tessera.alphabet import grid_digest
    from tessera.export import grid_from_config

    config = {"grid": {"name": "BF16", "base": "BF16", "arity": 1,
                       "digest": grid_digest(BF16_GRID)}}
    assert grid_from_config(config) is BF16_GRID
    with pytest.raises(GrammarError, match="digest does not match"):
        grid_from_config({"grid": {**config["grid"], "digest": "0" * 64}})


def test_the_family_name_is_spelled_once():
    assert BF16_FAMILY == "TESSERA_BF16"
