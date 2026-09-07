"""CUDA field reconstruction at the existing wire unpacking seam.

Uses the window decoder's MSB-first word reader. Length and canonical padding
remain checked by wire.unpack_body before this module receives the plane.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .kernel_bits import _plane_words, _span_of


@triton.jit
def _unpack_body_kernel(words, schedule, out, ROWS: tl.constexpr,
                        COLS: tl.constexpr, SPAN: tl.constexpr,
                        BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = index < ROWS * COLS
    column = index % COLS
    row = index // COLS
    rate = tl.load(schedule + column, mask=mask, other=0)
    start = tl.load(schedule + COLS + column, mask=mask, other=0)
    position = row % SPAN
    width = rate + (position != 0)
    offset = start + (row // SPAN) * (SPAN * rate + SPAN - 1)
    offset += position * rate + tl.maximum(position - 1, 0)
    active = mask & (width > 0)
    packed = _span_of(words, offset // 8, active)
    # Width zero is masked to avoid LLVM's undefined shift by 64.
    shift = 64 - (offset % 8) - tl.maximum(width, 1)
    value = (packed >> shift) & ((1 << width) - 1)
    tl.store(out + index, value, mask=mask)


def unpack_body_cuda(data: bytes, rates: tuple[int, ...], rows: int,
                     device: torch.device, span: int) -> torch.Tensor:
    """Unpack byte-sized fields from a validated column-major plane."""
    cols = len(rates)
    out = torch.empty((rows, cols), dtype=torch.uint8, device=device)
    if not out.numel():
        return out
    offsets = []
    cursor = 0
    for rate in rates:
        offsets.append(cursor)
        cursor += (span * rate + span - 1) * (rows // span)
    schedule = torch.tensor((rates, tuple(offsets)), dtype=torch.int64, device=device)
    plane = (torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)
             if data else torch.empty(0, dtype=torch.uint8, device=device))
    words = _plane_words(plane)
    with torch.cuda.device(device):
        _unpack_body_kernel[(triton.cdiv(rows * cols, 256),)](
            words, schedule, out, rows, cols, span, 256)
    return out
