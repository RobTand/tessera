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
import os

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


def _replay_core(
    body: torch.Tensor,
    subsets_flat: torch.Tensor,
    table_sub_flat: torch.Tensor,
    shift: int,
    mask: int,
    memory: int,
    points: int,
) -> torch.Tensor:
    """Body bits -> anchor indices, as one fusable elementwise chain.

    Every tensor here is uint8 and every operation is elementwise or a lookup
    into a 128-entry table, so the whole replay is a single pass over the data
    that ``torch.compile`` fuses into one kernel.  Written unfused over int64 it
    was ~15 passes at eight bytes per three bits of payload, and that -- not the
    trellis -- was the load-time cost.

    The window is built by *doubling*, a 2k-bit window from two k-bit ones,
    rather than by ORing in one lagged bit at a time: depth log2(memory) instead
    of memory.  That matters less once fused, but it is what keeps the eager
    fallback tolerable on a machine with no inductor.
    """
    select = (body >> shift) & 1
    point = body & mask

    def lagged(value: torch.Tensor, rows_back: int) -> torch.Tensor:
        out = torch.zeros_like(value)
        out[rows_back:] = value[:-rows_back]
        return out

    window = {1: select}
    while max(window) * 2 <= memory:
        width = max(window)
        window[width * 2] = (window[width] << width) | lagged(window[width], width)
    packed, held = None, 0
    for width in sorted(window, reverse=True):
        if not memory >> (width.bit_length() - 1) & 1:
            continue
        if packed is None:
            packed, held = window[width], width
        else:
            packed = (packed << width) | lagged(window[width], held)
            held += width
    state = lagged(packed, 1)
    subset = table_sub_flat[(select.int() << memory) + state.int()]
    return subsets_flat[subset.int() * points + point.int()]


def _decode_core(
    body: torch.Tensor,
    subsets_flat: torch.Tensor,
    table_sub_flat: torch.Tensor,
    reach_flat: torch.Tensor,
    completion: torch.Tensor,
    shift: int,
    mask: int,
    memory: int,
    points: int,
    width: int,
) -> torch.Tensor:
    """Body bits + completion bits -> E2M1 nibbles, in one fused pass.

    Stopping at the anchor index and doing the completion lookup outside cost
    more than the replay did: the anchor plane is a full-size tensor that only
    exists to be indexed once.  Fused, it never materialises, and the decoder's
    whole output is a uint8 nibble per position.
    """
    anchor = _replay_core(body, subsets_flat, table_sub_flat, shift, mask, memory, points)
    return reach_flat[anchor.int() * width + completion.int()]


@functools.lru_cache(maxsize=1)
def _fused_decode():
    """``_decode_core`` fused, or ``None``.  See ``_fused_replay``."""
    if os.environ.get("TESSERA_FUSED_REPLAY", "1") == "0":
        return None
    try:
        return torch.compile(_decode_core, dynamic=True)
    except Exception:  # pragma: no cover
        return None


@functools.lru_cache(maxsize=1)
def _fused_replay():
    """``_replay_core`` fused into one kernel, or ``None`` if that is refused.

    ``dynamic=True`` because a model presents hundreds of Linear shapes and a
    recompile for each would cost more than the fusion saves.  Set
    ``TESSERA_FUSED_REPLAY=0`` to force the eager path: the two must agree
    bit-for-bit, and a test asserts they do.
    """
    if os.environ.get("TESSERA_FUSED_REPLAY", "1") == "0":
        return None
    try:
        return torch.compile(_replay_core, dynamic=True)
    except Exception:  # pragma: no cover - no inductor, no fusion, still correct
        return None


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

    # The recursion looks sequential -- state_r feeds state_{r+1} -- but
    # ConvCode.step is ``register >> 1`` over ``(bit << memory) | state``, a
    # pure shift register.  So state_r is nothing but the previous ``memory``
    # select bits, a *windowed function of the stored stream*, and the whole
    # replay has O(1) depth.  Walking it row by row instead cost ~6 kernel
    # launches per trellis step: 561 ms on a single 17k-row Linear, 38 minutes
    # of load-time decode on a 355B body.  Principle 1 -- that was the
    # measurement being wrong about the problem, not a price serving pays.
    #
    # The property is checked against the tables built from ``code.step``, never
    # assumed: a code that failed it would decode to garbage in silence.
    shifted = bool(
        torch.equal(
            table_next,
            (torch.arange(code.states, device=device) >> 1).expand(2, -1)
            | (torch.tensor([[0], [1 << (code.memory - 1)]], device=device)),
        )
    )
    if shifted:
        body = body_bits if body_bits.dtype == torch.uint8 else body_bits.to(torch.uint8)
        args = (
            body,
            subsets.reshape(-1).to(torch.uint8),
            table_sub.reshape(-1).to(torch.uint8),
            shift,
            mask,
            code.memory,
            points,
        )
        run = _fused_replay() if body.is_cuda else None
        if run is not None:
            try:
                return run(*args).long()
            except Exception:  # pragma: no cover - fall back, never fail closed
                pass
        return _replay_core(*args).long()

    # A code whose step is not a shift register still has to decode, so the
    # sequential walk stays as the general path rather than an assumption.
    select = (body_bits >> shift) & 1
    point = body_bits & mask
    anchors = torch.zeros(rows, cols, dtype=torch.long, device=device)
    state = torch.zeros(cols, dtype=torch.long, device=device)
    for row in range(rows):
        subset = table_sub[select[row].long(), state]
        anchors[row] = subsets[subset, point[row].long()]
        state = table_next[select[row].long(), state]
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
    # uint8 here too, so the single-rate and mixed-rate decoders agree on the
    # dtype of a nibble.  They did not, and the release scatter caught it.
    blocks = torch.tensor(forest.blocks, device=device, dtype=torch.uint8)
    reachable = blocks[:, :: 1 << (depth - completion)]
    codes = reachable[anchors, unit.completion_bits]
    if unit.release_index.numel():
        codes.reshape(-1)[unit.release_index] = unit.release_code.to(torch.uint8)
    return codes


def dequantize(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Nibbles times their half-block scale: the weights the runtime will see."""
    # ``.int()``: codes are uint8, and a uint8 index tensor is a boolean mask.
    return e2m1_value_table(codes.device)[codes.int()] * scale


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
    # uint8, not int64: a code is a 4-bit nibble.  Carrying the decoder's whole
    # output at eight bytes a nibble made the completion lookup cost more than
    # the trellis replay it followed.
    codes = torch.zeros(rows, cols, dtype=torch.uint8, device=device)
    for present in sorted(set(unit.rates)):
        picked = forests[present]
        depth = 3 - picked.rate
        level = depth if completion is None else min(completion, depth)
        which = torch.nonzero(rates == present).squeeze(1)
        body = unit.body_bits[:, which].contiguous()
        comp = unit.completion_bits[:, which].contiguous()
        subsets, table_next, table_sub = _replay_tables(picked, code, str(device))
        blocks = torch.tensor(picked.blocks, device=device, dtype=torch.uint8)
        reachable = blocks[:, :: 1 << (depth - level)].contiguous()
        points = subsets.shape[1]
        shift = points.bit_length() - 1
        run = _fused_decode() if body.is_cuda else None
        args = (
            body if body.dtype == torch.uint8 else body.to(torch.uint8),
            subsets.reshape(-1).to(torch.uint8),
            table_sub.reshape(-1).to(torch.uint8),
            reachable.reshape(-1),
            comp if comp.dtype == torch.uint8 else comp.to(torch.uint8),
            shift,
            (1 << shift) - 1,
            code.memory,
            points,
            reachable.shape[1],
        )
        if run is not None:
            try:
                codes[:, which] = run(*args)
                continue
            except Exception:  # pragma: no cover - fall back, never fail closed
                pass
        anchors = replay_body(body, picked, code)
        codes[:, which] = reachable.long()[anchors, comp.long()].to(torch.uint8)
    if apply_release and unit.release_index.numel():
        codes.reshape(-1)[unit.release_index] = unit.release_code.to(torch.uint8)
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
