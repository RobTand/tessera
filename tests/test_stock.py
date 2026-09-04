"""The stock materialisation is the reader's reconstruction, in vLLM's arithmetic.

``stock_dequant`` is written from vLLM's decode -- its E2M1 table, its
inverted global, its per-row FP8 scale -- and the identity here is
``torch.equal`` against ``read_unit_artifact``, the bytes-only reader, for
every wire the exporter writes by default.  A close match would mean the
served weights are not the priced weights, which is the confound the whole
lane exists to exclude.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import box_artifacts

from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.errors import GrammarError
from tessera.export import DEFAULT_CODE, encode_linear_planes, wire_recipe
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.stock import (
    E2M1_MAGNITUDES, e2m1_nibbles, materialize_stock, share_global, stock_bytes,
    stock_dequant, stock_kind,
)
from tessera.unit_artifact import read_unit_artifact

K2 = tuple_grid(E2M1_GRID, 2)
QWEN = box_artifacts.path("models", "Qwen3-0.6B", "model.safetensors")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the encoder is a GPU job")


def _encode(weight, grid, q256, **overrides):
    return encode_linear_planes(weight, grid=grid, q256=q256, name="unit", **overrides)


@pytest.mark.parametrize("grid,q256,kind,resident_bpp", [
    (K2, 896, "nvfp4", 4.5),          # the coset trellis at its cap, LUT plane
    (K2, 768, "nvfp4", 4.5),          # the window body below the cap, LUT plane
    (E2M1_GRID, 768, "nvfp4", 4.5),   # K1, the coset trellis
    (E4M3_GRID, 1024, "fp8", 8.0),    # the window body over the CHANNEL plane, L=14
    (E4M3_GRID, 1280, "fp8", 8.0),
], ids=["k2-cap-tcq-lut", "k2-subcap-window-lut", "k1-tcq-lut", "e4m3-r4-window-channel",
        "e4m3-r5-window-channel"])
def test_stock_tensors_decode_to_the_reader_bit_for_bit(grid, q256, kind, resident_bpp):
    torch.manual_seed(3)
    # Real widths: the window body ships a 2^L-entry table per unit (16 KiB at
    # L=14), which on a toy tensor outweighs the body it shapes.
    rows, cols = 256, 1024
    weight = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    exported, unit, forests = _encode(weight, grid, q256)
    tensors = materialize_stock(unit, forests, DEFAULT_CODE)
    assert stock_kind(tensors) == kind
    served = stock_dequant(tensors)
    reader = read_unit_artifact(exported.blob, device="cuda").float()
    assert served.dtype is torch.float32 and served.shape == reader.shape
    assert torch.equal(served, reader)
    # The resident bytes are the stock format's, never the wire's: 4 bits +
    # one E4M3 per 16 + one fp32 global for NVFP4, 8 bits + one fp32 per row
    # for FP8 per-channel.
    expected = (rows * cols // 2 + rows * cols // 16 + 4) if kind == "nvfp4" else (rows * cols + 4 * rows)
    assert stock_bytes(tensors) == expected
    assert stock_bytes(tensors) * 8 / (rows * cols) == pytest.approx(resident_bpp, abs=0.15)
    assert float(exported.bpp) < resident_bpp


def test_the_runtime_e2m1_table_is_the_alphabet():
    codes = torch.arange(16)
    values = [(-1.0 if c >> 3 else 1.0) * E2M1_MAGNITUDES[c & 7] for c in codes.tolist()]
    assert tuple(E2M1_GRID.values) == tuple(values)


def test_tuple_codes_fan_to_consecutive_rows():
    codes = torch.tensor([[0x35, 0xA1], [0xF0, 0x07]])
    nibbles = e2m1_nibbles(codes, K2)
    assert nibbles.tolist() == [[0x3, 0xA], [0x5, 0x1], [0xF, 0x0], [0x0, 0x7]]
    with pytest.raises(GrammarError):
        e2m1_nibbles(codes, E4M3_GRID)


def test_a_channel_plane_on_an_e2m1_grid_has_no_stock_tensor():
    weight = (torch.randn(64, 128, device="cuda") * 0.02).contiguous()
    exported, unit, forests = _encode(
        weight, K2, 768, body=BodyKind.WINDOW, window_bits=8, scale_plane=ScalePlaneKind.CHANNEL,
    )
    with pytest.raises(GrammarError):
        materialize_stock(unit, forests, DEFAULT_CODE)


def _nvfp4(weight, q256=896):
    exported, unit, forests = _encode(weight, K2, q256)
    return exported, materialize_stock(unit, forests, DEFAULT_CODE)


def test_share_global_is_an_exact_binade_shift():
    torch.manual_seed(5)
    big = (torch.randn(64, 256, device="cuda") * 0.08).contiguous()
    small = (torch.randn(64, 256, device="cuda") * 0.005).contiguous()
    _, a = _nvfp4(big)
    _, b = _nvfp4(small)
    assert float(a["weight_global_scale"]) != float(b["weight_global_scale"])
    before = {"a": stock_dequant(a), "b": stock_dequant(b)}
    shared, divisor = share_global({"a": a, "b": b})
    assert float(shared["a"]["weight_global_scale"]) == divisor
    assert float(shared["b"]["weight_global_scale"]) == divisor
    assert torch.equal(stock_dequant(shared["a"]), before["a"])
    assert torch.equal(stock_dequant(shared["b"]), before["b"])
    # A power of two between the members' own: the largest that carries every
    # member exactly, which is vLLM's own choice (the max) only when the shift
    # it implies keeps the other members' scale bytes inside E4M3.
    own = sorted({float(a["weight_global_scale"]), float(b["weight_global_scale"])})
    assert own[0] <= divisor <= own[1] and divisor == 2.0 ** round(torch.log2(torch.tensor(divisor)).item())


def test_share_global_refuses_rather_than_rounds():
    torch.manual_seed(6)
    _, a = _nvfp4((torch.randn(32, 128, device="cuda") * 0.02).contiguous())
    b = {**a, "weight_global_scale": a["weight_global_scale"] * 2.0**30}
    with pytest.raises(GrammarError, match="no single weight_global_scale"):
        share_global({"a": a, "b": b})


@box_artifacts.require("models", "Qwen3-0.6B", "model.safetensors")
@pytest.mark.parametrize("layer", [0, 13, 27])
def test_qwen_fused_groups_share_one_global_exactly(layer):
    from safetensors import safe_open

    groups = {
        "qkv": [f"model.layers.{layer}.self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj")],
        "gate_up": [f"model.layers.{layer}.mlp.{n}.weight" for n in ("gate_proj", "up_proj")],
    }
    with safe_open(str(QWEN), framework="pt") as handle:
        for members in groups.values():
            group, reference = {}, {}
            for name in members:
                weight = handle.get_tensor(name).to("cuda", torch.float32).contiguous()
                _, tensors = _nvfp4(weight)
                group[name] = tensors
                reference[name] = stock_dequant(tensors)
            shared, _divisor = share_global(group)
            for name in members:
                assert torch.equal(stock_dequant(shared[name]), reference[name]), name
