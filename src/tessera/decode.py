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

import functools

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


@functools.lru_cache(maxsize=32)
def _replay_tables(
    forest: AnchorForest, code: ConvCode, device: str
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Subset table + the code's transition tables, built once per trellis.

    These depend on nothing but ``(forest, code, device)``, and rebuilding them
    per call cost 264 small copies -- a third of the replay's GPU time and most
    of its launch gap -- on tables of 128 entries.  Caching is not a
    micro-optimisation here; it is the difference between paying O(model) and
    O(distinct trellises), of which there are three.
    """
    from .encode import _subset_table

    subsets = _subset_table(TCQ(forest, code), device)
    table_next = torch.zeros(2, code.states, dtype=torch.long, device=device)
    table_sub = torch.zeros(2, code.states, dtype=torch.long, device=device)
    for value in range(code.states):
        for bit in (0, 1):
            nxt, sub = code.step(value, bit)
            table_next[bit, value] = nxt
            table_sub[bit, value] = sub
    return subsets, table_next, table_sub


def replay_body(
    body_bits: torch.Tensor, forest: AnchorForest, code: ConvCode
) -> torch.Tensor:
    """Replay the convolutional code from the body bits alone.

    This is the decoder's half of the round trip: it must land on the encoder's
    anchors using nothing but the stored bits, so the test that it equals
    ``EncodedUnit.anchors`` is the real proof the body is decodable.
    """
    device = body_bits.device
    rows, cols = body_bits.shape
    subsets, table_next, table_sub = _replay_tables(forest, code, str(device))
    points = subsets.shape[1]
    shift = points.bit_length() - 1
    mask = (1 << shift) - 1

    select = (body_bits >> shift) & 1
    point = body_bits & mask

    # The recursion looks sequential -- state_r feeds state_{r+1} -- but
    # ConvCode.step is ``register >> 1`` over ``(bit << memory) | state``, a
    # pure shift register.  So state_r is nothing but the previous ``memory``
    # select bits, a *windowed function of the stored stream*, and the whole
    # replay has O(1) depth.  Walking it row by row instead costs one Python
    # iteration and ~6 kernel launches per trellis step: 561 ms on a single
    # 17k-row Linear, which extrapolates to 38 minutes of load-time decode on
    # a 355B body.  Principle 1 -- that is the measurement being wrong about
    # the problem, not a price serving has to pay.
    shifted = bool(
        torch.equal(
            table_next,
            (torch.arange(code.states, device=device) >> 1).expand(2, -1)
            | (torch.tensor([[0], [1 << (code.memory - 1)]], device=device)),
        )
    )
    if shifted:
        # Two things make the window cheap, and the profiler named both.  The
        # naive ladder ORs in one lagged bit at a time, so it costs ``memory``
        # passes over a full-size int64 tensor -- 10 GB of traffic on an 89M
        # Linear, 66% of the replay.  Doubling instead builds a 2k-bit window
        # from two k-bit ones, which is log2(memory) passes, and the window
        # itself only ever needs ``memory`` bits, so int16 carries it at an
        # eighth of the bytes.
        def lagged(value: torch.Tensor, rows_back: int) -> torch.Tensor:
            out = torch.zeros_like(value)
            out[rows_back:] = value[:-rows_back]
            return out

        window = {1: select.to(torch.int16)}
        while max(window) * 2 <= code.memory:
            width = max(window)
            window[width * 2] = (window[width] << width) | lagged(
                window[width], width
            )
        packed, held = None, 0
        for width in sorted(window, reverse=True):
            if not code.memory >> (width.bit_length() - 1) & 1:
                continue
            if packed is None:
                packed, held = window[width], width
            else:
                packed = (packed << width) | lagged(window[width], held)
                held += width
        state = lagged(packed, 1).to(torch.long)
        return subsets[table_sub[select, state], point]

    # A code whose step is not a shift register still has to decode, so the
    # sequential walk stays as the general path rather than an assumption.
    anchors = torch.zeros(rows, cols, dtype=torch.long, device=device)
    state = torch.zeros(cols, dtype=torch.long, device=device)
    for row in range(rows):
        subset = table_sub[select[row], state]
        anchors[row] = subsets[subset, point[row]]
        state = table_next[select[row], state]
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
