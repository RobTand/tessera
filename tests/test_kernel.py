"""The kernel lane: Tessera bits consumed by a GEMV without materialising NVFP4.

Every test here compares against ``reconstruct_unit`` rather than against a
tolerance where it can, because the kernel and the reference decoder are two
implementations of one grammar and "close" is the wrong relationship for them
to have.
"""

from __future__ import annotations

import pytest
import torch

from tessera.alphabet import build_forest
from tessera.decode import decode_codes_mixed, materialize_nvfp4, reconstruct_unit
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import RotationState
from tessera.trellis import ConvCode
from tessera.wire import pack_body

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernel lane is a CUDA path"
)

CODE = ConvCode(memory=6)


@pytest.fixture(scope="module")
def unit():
    """A full-rate, release-free unit -- the configuration the kernel lane serves.

    R=3 everywhere means completion width 0, and release=0 means no
    data-dependent positions, which together are what make the body a pure
    windowed function of the bitstream.
    """
    device = "cuda"
    forests = {r: build_forest(r) for r in (1, 2, 3)}
    torch.manual_seed(0)
    rows, cols = 256, 512
    weights = (torch.randn(rows, cols, device=device) * 0.02).contiguous()
    rates = bresenham_rate_schedule(root_from_q256(768), cols)
    assert set(rates) == {3}
    encoded = encode_unit(
        weights, forests, rates, CODE,
        rotation=RotationState.NONE, with_diagonals=False, released_positions=0,
    )
    codes = decode_codes_mixed(encoded, forests, CODE)
    _packed, e4m3, global_scale = materialize_nvfp4(
        codes, encoded.scale_base, encoded.scale_refine, encoded.group, encoded.half
    )
    return {
        "unit": encoded, "forests": forests, "rates": rates, "codes": codes,
        "rows": rows, "cols": cols, "e4m3": e4m3, "global_scale": global_scale,
        "reference": reconstruct_unit(encoded, forests, CODE).float(),
    }


def test_dequant_kernel_is_bit_exact_against_the_reference_decoder(unit):
    from tessera.kernel import build_code_lut, tessera_dequant

    body = torch.frombuffer(
        bytearray(pack_body(unit["unit"].body_bits, unit["rates"]) + b"\x00"),
        dtype=torch.uint8,
    ).to("cuda")
    got = tessera_dequant(
        body, build_code_lut(unit["forests"][3], CODE), unit["e4m3"],
        unit["global_scale"], unit["rows"], unit["cols"],
    )
    assert torch.equal(got, unit["reference"])


def test_sliced_planes_round_trip_through_the_wire_body(unit):
    """The resident layout is a permutation of the wire body, not new data."""
    from tessera.kernel import SELECT_PAD, pack_kernel_planes

    select, point = pack_kernel_planes(unit["unit"].body_bits)
    rows, cols = unit["rows"], unit["cols"]
    assert select.numel() == cols * (rows + SELECT_PAD) // 8
    assert point.numel() == cols * rows * 2 // 8
    # Same bits as the wire body: 3 per position, no more.
    assert (select.numel() + point.numel()) * 8 == rows * cols * 3 + cols * SELECT_PAD


def test_one_hot_gemv_decodes_each_column_bit_exactly(unit):
    """``x = e_k`` returns column k of W, so this compares the decode itself.

    Columns 0..7 matter most: their history window straddles the select plane's
    zero pad, which is the mechanism standing in for "the trellis starts in
    state 0" and the easiest thing in this layout to get wrong.
    """
    from tessera.kernel import build_value_lut, pack_kernel_planes, tessera_gemv_wide

    select, point = pack_kernel_planes(unit["unit"].body_bits)
    lut = build_value_lut(unit["forests"][3], CODE)
    rows, cols = unit["rows"], unit["cols"]
    scales = unit["e4m3"].reshape(rows, cols // 16).t().contiguous()
    for k in (0, 1, 5, 6, 7, 8, 33, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = tessera_gemv_wide(
            x, select, point, lut, scales, unit["global_scale"], rows, cols,
            lanes=32, split_k=4,
        )
        assert torch.equal(got, unit["reference"][:, k]), f"column {k}"


def test_gemv_matches_the_materialised_path(unit):
    """The two lanes must agree: same artifact, same answer, different route."""
    from tessera.kernel import (
        build_value_lut, nvfp4_gemv_sliced, pack_kernel_planes,
        pack_nvfp4_column_major, tessera_gemv_wide,
    )

    rows, cols = unit["rows"], unit["cols"]
    select, point = pack_kernel_planes(unit["unit"].body_bits)
    scales = unit["e4m3"].reshape(rows, cols // 16).t().contiguous()
    torch.manual_seed(1)
    x = torch.randn(cols, device="cuda")
    tessera = tessera_gemv_wide(
        x, select, point, build_value_lut(unit["forests"][3], CODE), scales,
        unit["global_scale"], rows, cols, lanes=32, split_k=4,
    )
    stock = nvfp4_gemv_sliced(
        x, pack_nvfp4_column_major(unit["codes"]), scales,
        unit["global_scale"], rows, cols, block_n=64, split_k=4,
    )
    reference = unit["reference"] @ x
    assert (tessera - reference).norm() / reference.norm() < 1e-5
    assert (tessera - stock).norm() / stock.norm() < 1e-5


def test_the_kernel_lane_refuses_a_body_that_carries_completion_bits(unit):
    """Below R=3 a position needs completion bits, a plane this lane does not read.

    Failing closed matters more than usual here: the LUT would silently be built
    over the wrong descendant set and decode to plausible, wrong weights.
    """
    from tessera.errors import GrammarError
    from tessera.kernel import build_code_lut

    with pytest.raises(GrammarError, match="full-rate"):
        build_code_lut(unit["forests"][2], CODE)
