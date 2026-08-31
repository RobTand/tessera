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

__all__ = [
    "replay_body",
    "decode_codes",
    "decode_codes_mixed",
    "dequantize",
    "materialize_nvfp4",
    "reconstruct_unit",
]


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
    codes: torch.Tensor,
    scale_base: torch.Tensor,
    scale_refine: torch.Tensor,
    group: int = 32,
    half: int = 16,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Pack to the standard NVFP4 layout: 2 nibbles/byte + one E4M3 per 16.

    Returns ``(packed[rows, cols//2] uint8, scales[rows, cols//half] uint8,
    global_scale float)`` -- NVFP4's two scale levels, both of them.
    Every Tessera code is a legal E2M1 nibble by construction -- the alphabet
    *is* the grid -- so this pads information, never truncates it, and the
    result is an ordinary NVFP4 tensor that stock kernels consume.
    """
    from .wire import nvfp4_scale_bytes

    rows, cols = codes.shape
    if cols % 2:
        raise GrammarError(f"{cols} columns cannot pack 2 nibbles to a byte")
    low = codes[:, 0::2].to(torch.uint8)
    high = codes[:, 1::2].to(torch.uint8)
    packed = (low & 0xF) | ((high & 0xF) << 4)
    e4m3, global_scale = nvfp4_scale_bytes(scale_base, scale_refine, group, half)
    return packed, e4m3.reshape(rows, cols // half), global_scale


def decode_codes_mixed(
    unit: "EncodedUnit",
    forest: "AnchorForest | dict[int, AnchorForest]",
    code: ConvCode,
    completion: int | None = None,
    apply_release: bool = True,
) -> torch.Tensor:
    """Body + completion -> E2M1 nibbles, over a mixed-rate schedule.

    ``apply_release=False`` returns the **pre-release** codes, which is not a
    debugging convenience: §9 orders released positions by descending decoded
    magnitude on the pre-release decode, so a reader has to reproduce that
    intermediate state to know *which* positions the RELEASE plane refers to.
    The plane stores codes, never indices -- that is where its rate advantage
    comes from, and it is only decodable because the order is derivable.
    """
    forests = forest if isinstance(forest, dict) else {forest.rate: forest}
    device = unit.body_bits.device
    rows, cols = unit.body_bits.shape
    rates = torch.tensor(unit.rates, device=device)
    codes = torch.zeros(rows, cols, dtype=torch.long, device=device)
    for present in sorted(set(unit.rates)):
        picked = forests[present]
        depth = 3 - picked.rate
        level = depth if completion is None else min(completion, depth)
        which = torch.nonzero(rates == present).squeeze(1)
        anchors = replay_body(unit.body_bits[:, which].contiguous(), picked, code)
        blocks = torch.tensor(picked.blocks, device=device, dtype=torch.long)
        reachable = blocks[:, :: 1 << (depth - level)]
        codes[:, which] = reachable[anchors, unit.completion_bits[:, which]]
    if apply_release and unit.release_index.numel():
        codes.reshape(-1)[unit.release_index] = unit.release_code
    return codes


def reconstruct_unit(
    unit: "EncodedUnit",
    forest: "AnchorForest | dict[int, AnchorForest]",
    code: ConvCode,
    scale: torch.Tensor | None = None,
    completion: int | None = None,
) -> torch.Tensor:
    """The whole inverse path: body -> codes -> weights -> 2a -> rotation.

    Undone in the exact reverse of the order the encoder applied them (S5's
    segment order read backwards).  Getting this order wrong produces weights
    that are wrong by a rank-1 factor or an orthogonal transform -- both of
    which look plausible and neither of which the round-trip test tolerates.

    ``scale`` defaults to the scale **derived from the unit's own stored
    segment-2b bytes** via S6b, which is the only scale a decoder reading an
    artifact can have.  Accepting the encoder's float tensor instead -- which
    this function used to require -- leaves the S6b codec untested, and S6b's
    round-trip is exactly what the doc says T-nvfp4-class is conjectural
    without.  The argument survives only so a test can pass a *different* scale
    and watch the reconstruction move.
    """
    from .decode import decode_codes as _decode
    from .diagonals import undo_diagonals, undo_rotation
    from .wire import scales_from_planes

    codes = decode_codes_mixed(unit, forest, code, completion)
    rows, cols = unit.body_bits.shape
    if scale is None:
        scale = torch.repeat_interleave(
            scales_from_planes(
                unit.scale_base, unit.scale_refine, unit.group, unit.half
            ),
            unit.half,
        ).reshape(rows, cols)
    out = dequantize(codes, scale)
    if unit.diagonals is not None:
        out = undo_diagonals(out, unit.diagonals)
    return undo_rotation(out, unit.rotation, unit.rotation_block)
