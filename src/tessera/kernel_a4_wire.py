"""Opt-in geometry reader for actual mixed TCQ and LUT WINDOW E2M1 pairs.

Preparation lives in compact_prep and existing lane_planes owners. This module
is not imported by serving and changes no default. Packed BODY, rate and bit
start tables stay resident; weights are decoded only in the native FP4 mainloop.
The explicit diagnostic renderer is an oracle surface, never a compute fallback.
"""
from dataclasses import dataclass, fields

import torch

from .errors import GrammarError
from .kernel_a4 import _TRITON, _TL, require_native_fp4_mma, a4_quantize_activation


@dataclass(frozen=True)
class A4WireUnit:
    body: torch.Tensor
    starts: torch.Tensor
    rates: torch.Tensor
    code_starts: torch.Tensor
    codes: torch.Tensor
    labels: torch.Tensor
    initial: torch.Tensor
    nibbles: torch.Tensor
    lut_bytes: torch.Tensor
    rows: int
    cols: int
    memory: int
    window_bits: int
    body_kind: str
    global_scale: float
    layout: dict

    def named_tensors(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                yield field.name, value

    def epilogue_for(self, input_global_scale):
        return torch.full((1,), self.global_scale, device=self.body.device,
                          dtype=torch.float32) / input_global_scale.reshape(1)


def _cpu_field(body, offset, width):
    """Bounded MSB-first fields, including zero-width and pre-stream zeroes."""
    active = (offset >= 0) & (width > 0)
    byte = (offset.clamp(min=0) >> 3).long()
    bit = offset & 7
    value = torch.zeros_like(offset)
    for i in range(3):
        live = active & (bit + width > 8 * i) & (byte + i < body.numel())
        b = body[(byte + i).clamp(max=body.numel() - 1)].long()
        value |= torch.where(live, b, 0) << (16 - 8 * i)
    return (value >> (24 - bit - width).clamp(min=0)) & ((1 << width) - 1)


def decode_wire_codes(unit):
    """Diagnostic only: stock-layout code and scale bytes, on CPU or CUDA."""
    if unit.body.device.type == "cuda":
        codes = torch.empty((unit.rows, unit.cols // 2), dtype=torch.uint8,
                            device=unit.body.device)
        scales = torch.empty((unit.rows, unit.cols // 16), dtype=torch.uint8,
                             device=unit.body.device)
        _wire_dump_kernel[(_TRITON.cdiv(unit.rows * unit.cols // 2, 128),)](
            unit.body, unit.starts, unit.rates, unit.code_starts, unit.codes,
            unit.labels, unit.initial, unit.nibbles, unit.lut_bytes, codes, scales,
            unit.rows, unit.cols, unit.body.numel(), MEMORY=unit.memory,
            L=unit.window_bits, WINDOW=unit.body_kind == "window", BLOCK=128)
        return codes, scales
    r = torch.arange(unit.rows // 2)[:, None]
    rate = unit.rates.long()[None, :]
    start = unit.starts[None, :]
    if unit.body_kind == "window":
        offset = start + (r + 1) * rate
        state = _cpu_field(unit.body, offset, torch.full_like(offset, unit.window_bits))
        pair = unit.codes[state].long()
    else:
        p = r // 2
        per = 2 * rate + 1
        window = torch.zeros_like(start + p)
        for i in range(unit.memory + 1):
            previous = p - unit.memory + i
            bit = _cpu_field(unit.body, start + previous * per, torch.ones_like(previous + start))
            initial_bit = (unit.initial[None, :] >> i) & 1
            bit = torch.where(previous >= 0, bit, initial_bit)
            window = (window << 1) | bit
        ell = unit.labels[window].long()
        stored = _cpu_field(unit.body, start + p * per + rate, torch.full_like(start + p, 2))
        label = torch.where((r & 1) == 0, (ell - stored) & 3, stored)
        offset = start + p * per + torch.where((r & 1) == 0, 1, rate + 2)
        point = _cpu_field(unit.body, offset, (rate - 1).expand_as(offset))
        pair = unit.codes[unit.code_starts[None, :] + label * (1 << (rate - 1)) + point].long()
    nibbles = torch.stack((pair & 15, pair >> 4), dim=1).reshape(unit.rows, unit.cols)
    packed = (nibbles[:, ::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
    index = torch.arange(unit.rows)[..., None] + torch.arange(unit.cols // 16)[None, :] * unit.rows
    nib = (unit.nibbles[index // 2].long() >> (4 * (1 - (index & 1)))) & 15
    return packed, unit.lut_bytes[nib]


if _TL is not None:
    @_TRITON.jit
    def _field(body, offset, width, size, live):
        bit = offset & 7
        byte = offset >> 3
        active = live & (offset >= 0) & (width > 0)
        b0 = _TL.load(body + byte, mask=active & (byte < size), other=0).to(_TL.int32)
        b1 = _TL.load(body + byte + 1, mask=active & (bit + width > 8) & (byte + 1 < size), other=0).to(_TL.int32)
        b2 = _TL.load(body + byte + 2, mask=active & (bit + width > 16) & (byte + 2 < size), other=0).to(_TL.int32)
        return (((b0 << 16) | (b1 << 8) | b2) >> _TL.maximum(24 - bit - width, 0)) & ((1 << width) - 1)

    @_TRITON.jit
    def _pair(body, starts, rates, code_starts, codes, labels, initial,
              k, r, size, live, MEMORY: _TL.constexpr, L: _TL.constexpr,
              WINDOW: _TL.constexpr):
        rate = _TL.load(rates + k, mask=live, other=1).to(_TL.int32)
        start = _TL.load(starts + k, mask=live, other=0)
        if WINDOW:
            state = _field(body, start + (r + 1) * rate, L, size, live)
            result = _TL.load(codes + state, mask=live, other=0)
        else:
            p = r // 2
            per = 2 * rate + 1
            window = _TL.full(k.shape, 0, _TL.int32) + _TL.full(r.shape, 0, _TL.int32)
            init = _TL.load(initial + k, mask=live, other=0)
            for i in _TL.static_range(MEMORY + 1):
                previous = p - MEMORY + i
                bit = _field(body, start + previous * per, 1, size, live & (previous >= 0))
                bit = _TL.where(previous >= 0, bit, (init >> i) & 1)
                window = (window << 1) | bit
            ell = _TL.load(labels + window, mask=live, other=0)
            stored = _field(body, start + p * per + rate, 2, size, live)
            label = _TL.where((r & 1) == 0, (ell - stored) & 3, stored)
            offset = start + p * per + _TL.where((r & 1) == 0, 1, rate + 2)
            point = _field(body, offset, rate - 1, size, live)
            base = _TL.load(code_starts + k, mask=live, other=0)
            result = _TL.load(codes + base + label * (1 << (rate - 1)) + point, mask=live, other=0)
        return result.to(_TL.int32)

    @_TRITON.jit
    def _wire_dump_kernel(body, starts, rates, code_starts, codes, labels, initial,
                          nibbles, lut, out, scales, rows, cols, size,
                          MEMORY: _TL.constexpr, L: _TL.constexpr,
                          WINDOW: _TL.constexpr, BLOCK: _TL.constexpr):
        idx = _TL.program_id(0) * BLOCK + _TL.arange(0, BLOCK)
        live = idx < rows * (cols // 2)
        r = idx // (cols // 2)
        k = 2 * (idx % (cols // 2))
        a = _pair(body, starts, rates, code_starts, codes, labels, initial, k, r // 2, size, live, MEMORY, L, WINDOW)
        b = _pair(body, starts, rates, code_starts, codes, labels, initial, k + 1, r // 2, size, live, MEMORY, L, WINDOW)
        shift = 4 * (r & 1)
        _TL.store(out + idx, ((a >> shift) & 15) | (((b >> shift) & 15) << 4), mask=live)
        sidx = idx
        slive = sidx < rows * (cols // 16)
        sr = sidx // (cols // 16)
        sg = sidx % (cols // 16)
        ni = sg * rows + sr
        nb = _TL.load(nibbles + ni // 2, mask=slive, other=0)
        nib = (nb >> (4 * (1 - (ni & 1)))) & 15
        scale = _TL.load(lut + nib, mask=slive, other=0)
        _TL.store(scales + sidx, scale, mask=slive)

    @_TRITON.jit
    def _a4_wire_gemm_kernel(a, sa, body, starts, rates, code_starts, codes, labels,
                             initial, nibbles, lut, epilogue, tokens, offsets, out,
                             M, rows, cols, size, BS, CS, NS,
                             MEMORY: _TL.constexpr, L: _TL.constexpr,
                             WINDOW: _TL.constexpr, GROUPED: _TL.constexpr,
                             BM: _TL.constexpr, BN: _TL.constexpr, BK: _TL.constexpr):
        e = _TL.program_id(0)
        pm = _TL.program_id(1)
        pn = _TL.program_id(2)
        if GROUPED:
            first = _TL.load(offsets + e)
            count = _TL.load(offsets + e + 1) - first
        else:
            first = 0
            count = M
        if pm * BM >= count:
            return
        mi = pm * BM + _TL.arange(0, BM)
        live_m = mi < count
        if GROUPED:
            token = _TL.load(tokens + first + mi, mask=live_m, other=0)
        else:
            token = mi
        ni = pn * BN + _TL.arange(0, BN)
        pair_row = (pn * (BN // 2) + _TL.arange(0, BN // 2))[None, :]
        acc = _TL.zeros((BM, BN), _TL.float32)
        for k0 in range(0, cols, BK):
            k = (k0 + 2 * _TL.arange(0, BK // 2))[:, None]
            ca = _pair(body + e * BS, starts, rates, code_starts, codes + e * CS,
                       labels, initial + e * cols, k, pair_row, size, True, MEMORY, L, WINDOW)
            cb = _pair(body + e * BS, starts, rates, code_starts, codes + e * CS,
                       labels, initial + e * cols, k + 1, pair_row, size, True, MEMORY, L, WINDOW)
            even = ((ca & 15) | ((cb & 15) << 4)).to(_TL.uint8)
            odd = ((ca >> 4) | (cb & 240)).to(_TL.uint8)
            w = _TL.interleave(even, odd)
            g = _TL.arange(0, BK // 16)
            si = (k0 // 16 + g)[None, :] * rows + ni[:, None]
            nb = _TL.load(nibbles + e * NS + si // 2).to(_TL.int32)
            nib = (nb >> (4 * (1 - (si & 1)))) & 15
            sw = _TL.load(lut + e * 16 + nib).to(_TL.float8e4nv, bitcast=True)
            kp = k0 // 2 + _TL.arange(0, BK // 2)
            av = _TL.load(a + token[:, None] * (cols // 2) + kp[None, :], mask=live_m[:, None], other=0)
            ascale = _TL.load(sa + token[:, None] * (cols // 16) + (k0 // 16 + g)[None, :], mask=live_m[:, None], other=0).to(_TL.float8e4nv, bitcast=True)
            acc = _TL.dot_scaled(av, ascale, "e2m1", w, sw, "e2m1", acc)
        ratio = _TL.load(epilogue + e)
        _TL.store(out + (first + mi)[:, None] * rows + ni[None, :], acc * ratio, mask=live_m[:, None])


class PreparedA4Wire:
    """Freeze expert planes once; hot calls quantize and launch, never repack."""
    def __init__(self, units, input_global_scale):
        self.units = tuple(units)
        if not self.units:
            raise GrammarError("a packed A4 preparation needs at least one unit")
        self.unit = first = self.units[0]
        for unit in self.units[1:]:
            if (unit.rows, unit.cols, unit.body_kind, unit.memory, unit.window_bits, unit.layout) != (first.rows, first.cols, first.body_kind, first.memory, first.window_bits, first.layout):
                raise GrammarError("expert packed layouts differ")
        self.gs = input_global_scale
        self.body = torch.stack([u.body for u in self.units])
        self.codes = torch.stack([u.codes for u in self.units])
        self.initial = torch.stack([u.initial for u in self.units])
        self.nibbles = torch.stack([u.nibbles for u in self.units])
        self.lut = torch.stack([u.lut_bytes for u in self.units])
        self.epilogue = torch.cat([u.epilogue_for(input_global_scale) for u in self.units])

    def __call__(self, x, *, expert_offsets=None, route_ids=None, num_routes=None, out_dtype=torch.bfloat16):
        require_native_fp4_mma("packed A4 geometry")
        u = self.unit
        if u.rows % 64 or u.cols % 128:
            raise GrammarError("packed A4 native blocks require rows divisible by sixty four and columns by one hundred twenty eight")
        if x.device != u.body.device or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != u.cols:
            raise GrammarError("packed A4 activations must be BF16 [M, cols] on the unit device")
        grouped = expert_offsets is not None
        if grouped != (route_ids is not None):
            raise GrammarError("expert offsets and route identifiers must be supplied together")
        if not grouped and len(self.units) != 1:
            raise GrammarError("a dense packed A4 launch needs exactly one unit")
        if grouped:
            if expert_offsets.numel() != len(self.units) + 1 or route_ids.numel() != num_routes:
                raise GrammarError("dispatch shapes do not match the expert and route axes")
            if expert_offsets.dtype != torch.int32 or route_ids.dtype != torch.int32 or expert_offsets.device != x.device or route_ids.device != x.device:
                raise GrammarError("prepared dispatch must be int32 on the activation device")
            m = num_routes
            tokens, offsets = route_ids, expert_offsets
        else:
            m = x.shape[0]
            tokens = offsets = u.rates
        a, sa = a4_quantize_activation(x, self.gs)
        out = torch.empty((m, u.rows), device=x.device, dtype=torch.float32)
        _a4_wire_gemm_kernel[(len(self.units), max(1, _TRITON.cdiv(m, 64)), u.rows // 64)](
            a, sa.view(torch.uint8), self.body, u.starts, u.rates, u.code_starts,
            self.codes, u.labels, self.initial, self.nibbles, self.lut, self.epilogue,
            tokens, offsets, out, m, u.rows, u.cols, u.body.numel(),
            u.body.numel(), u.codes.numel(), u.nibbles.numel(),
            MEMORY=u.memory, L=u.window_bits, WINDOW=u.body_kind == "window",
            GROUPED=grouped, BM=64, BN=64, BK=128, num_warps=4, num_stages=2)
        return out if out_dtype == torch.float32 else out.to(out_dtype)
