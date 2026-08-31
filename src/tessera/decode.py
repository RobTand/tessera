"""The Tessera decoder: replay to nibbles, then materialise native NVFP4.

The decoder is not a GEMM.  S6 says it plainly -- "A materialized
compressed-tensors artifact pads the <=3-bit information into legal 4-bit
nibbles (any code is legal), so CT bytes are 4.5-class while carrying EN
information" -- so decoding is a **materialisation** into the standard NVFP4
layout, and the matmul afterwards is stock ``cutlass_scaled_fp4_mm`` on native
tensor cores.  No custom kernel, which is the entire reason the artifact is
servable by a runtime that has never heard of Tessera.

That is also why the format's stored size and its resident size differ, and S7
names both: the *selected-prefix* bytes are what a sub-4 claim may attach to,
while *expanded-resident* is ~4.5.  Reporting one as the other in either
direction would be dishonest, so the two are separate quantities here too.
"""

from __future__ import annotations

import torch

from .alphabet import AnchorForest
from .encode import EncodedUnit, e2m1_value_table
from .errors import GrammarError
from .trellis import SUBSET_COUNT, ConvCode, TCQ, _ODS_GENERATORS  # noqa: F401
from .trellis import ConvCode as _ConvCode

__all__ = ["replay_body", "decode_codes", "dequantize", "materialize_nvfp4"]


def replay_body(
    body_bits: torch.Tensor, forest: AnchorForest, code: ConvCode
) -> torch.Tensor:
    """Replay the convolutional code from the body bits alone.

    This is the decoder's half of the round trip: it must land on the encoder's
    anchors using nothing but the stored bits, so the test that it equals
    ``EncodedUnit.anchors`` is the real proof the body is decodable.
    """
    from .encode import _subset_table, _transition_tables

    device = body_bits.device
    rows, cols = body_bits.shape
    tcq = TCQ(forest, code)
    subsets = _subset_table(tcq, device)
    points = subsets.shape[1]
    shift = points.bit_length() - 1
    mask = (1 << shift) - 1

    anchors = torch.zeros(rows, cols, dtype=torch.long, device=device)
    state = torch.zeros(cols, dtype=torch.long, device=device)
    # step() is cheap and pure, so build its table once and gather.
    table_next = torch.zeros(2, code.states, dtype=torch.long, device=device)
    table_sub = torch.zeros(2, code.states, dtype=torch.long, device=device)
    for value in range(code.states):
        for bit in (0, 1):
            nxt, sub = code.step(value, bit)
            table_next[bit, value] = nxt
            table_sub[bit, value] = sub

    for row in range(rows):
        word = body_bits[row]
        select = (word >> shift) & 1
        point = word & mask
        subset = table_sub[select, state]
        anchors[row] = subsets[subset, point]
        state = table_next[select, state]
    return anchors


def decode_codes(
    unit: EncodedUnit,
    forest: AnchorForest,
    code: ConvCode,
    completion: int | None = None,
) -> torch.Tensor:
    """Full decode from stored planes to E2M1 nibbles, in wire order."""
    depth = 3 - forest.rate
    completion = depth if completion is None else completion
    device = unit.body_bits.device
    anchors = replay_body(unit.body_bits, forest, code)
    blocks = torch.tensor(forest.blocks, device=device, dtype=torch.long)
    reachable = blocks[:, :: 1 << (depth - completion)]
    codes = reachable[anchors, unit.completion_bits]
    if unit.release_index.numel():
        codes.reshape(-1)[unit.release_index] = unit.release_code
    return codes


def dequantize(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Nibbles times their half-block scale: the weights the runtime will see."""
    return e2m1_value_table(codes.device)[codes] * scale


def materialize_nvfp4(
    codes: torch.Tensor, scale: torch.Tensor, half: int = 16
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Pack to the standard NVFP4 layout: 2 nibbles/byte + one E4M3 per 16.

    Returns ``(packed[rows, cols//2] uint8, scales[rows, cols//half] uint8)``.
    Every Tessera code is a legal E2M1 nibble by construction -- the alphabet
    *is* the grid -- so this pads information, never truncates it, and the
    result is an ordinary NVFP4 tensor that stock kernels consume.
    """
    rows, cols = codes.shape
    if cols % 2:
        raise GrammarError(f"{cols} columns cannot pack 2 nibbles to a byte")
    low = codes[:, 0::2].to(torch.uint8)
    high = codes[:, 1::2].to(torch.uint8)
    packed = (low & 0xF) | ((high & 0xF) << 4)

    per_half = scale[:, ::half].contiguous()
    exponent = torch.floor(torch.log2(per_half.clamp_min(1e-30)))
    mantissa = per_half / torch.exp2(exponent)
    biased = (exponent + 7).clamp(0, 15).to(torch.uint8)
    frac = ((mantissa - 1.0) * 8.0).round().clamp(0, 7).to(torch.uint8)
    e4m3 = (biased << 3) | frac
    return packed, e4m3
