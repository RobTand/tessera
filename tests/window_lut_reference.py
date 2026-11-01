"""Test-only serialized WINDOW fixtures and independent materializing preparation.

The production compact packer remains CUDA-only. This helper reconstructs
verified planes and uses the documented bitstream word packer, not a mock
forward or a production CPU fallback.
"""
from __future__ import annotations

from fractions import Fraction

import torch

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.compact_prep import WindowLutUnit
from tessera.encode import EncodedUnit
from tessera.grammar import bresenham_rate_schedule
from tessera.lane_planes import lut_scale_bytes, pack_scale_nibbles
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.slicing import slice_unit
from tessera.unit_artifact import _window_unit, build_unit_artifact

from window_pack_reference import pack_bitstream


def window_blob(q256, *, rows=512, cols=512, seed=0, global_scale=1.0,
                window_bits=14, name="weight", reverse_rates=False):
    """A real self-describing wire with nonuniform code and scale bytes."""
    grid = tuple_grid(E2M1_GRID, 2)
    rates = bresenham_rate_schedule(Fraction(int(q256) * 2, 256), cols, cap=8)
    if reverse_rates:
        rates = tuple(reversed(rates))
    generator = torch.Generator().manual_seed(seed)
    shape = (rows // 2, cols)
    zeros = torch.zeros(shape, dtype=torch.long)
    raw = torch.randint(0, 256, shape, generator=generator)
    body = (raw % torch.tensor([1 << rate for rate in rates])).to(torch.uint8)
    empty = torch.empty(0, dtype=torch.long)
    unit = EncodedUnit(
        rates=rates, anchors=zeros, codes=zeros, body_bits=body,
        completion_bits=zeros, scale_base=torch.empty(0, dtype=torch.uint8),
        scale_refine=torch.randint(0, 16, (rows * cols // 16,),
                                   generator=generator, dtype=torch.uint8),
        release_index=empty, release_code=empty, sse=0.0, completion_limit=0,
        span=1, scale_plane=ScalePlaneKind.LUT,
        scale_lut=torch.arange(0x20, 0x30, dtype=torch.uint8),
        scale_global=float(global_scale), body=BodyKind.WINDOW, window_bits=window_bits,
        window_codes=(torch.arange(1 << window_bits) % 256).to(torch.uint8))
    return build_unit_artifact(unit, name, grid, int(q256) * 2, fixture_id=None)[2]


def prepare_reference(wire, *, rows=None, cols=None, device="cpu", scratch=None):
    """Independent test substitute for hardware packing, not for validation."""
    assert torch.device(device).type == "cpu"
    parsed = _window_unit(wire.metadata, "cpu")
    unit = (slice_unit(parsed, rows=rows, cols=cols)
            if rows is not None or cols is not None else parsed.unit)
    rep = pack_bitstream(unit.body_bits, unit.rates)
    state = getattr(unit, "initial_state", None)
    if state is None:
        state = torch.zeros(rep.cols, dtype=torch.int32)
    local_rows, local_cols = int(unit.body_bits.shape[0]) * 2, int(unit.body_bits.shape[1])
    return WindowLutUnit(
        rep=rep, codes=unit.window_codes.to(torch.uint8),
        scale_plane=pack_scale_nibbles(unit.scale_refine, local_rows, local_cols, unit.half),
        scale_lut=lut_scale_bytes(unit.scale_lut, "cpu"), global_scale=float(unit.scale_global),
        window_bits=int(unit.window_bits), rows=local_rows, cols=local_cols,
        arity=2, half=int(unit.half), initial_state=state.to(torch.int32),
        row_offset=int(getattr(unit, "row_offset", 0)))
