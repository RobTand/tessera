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

from .alphabet import AnchorForest, PayloadGrid
from .encode import EncodedUnit, e2m1_value_table
from .errors import GrammarError
from .manifest import BodyKind, ScalePlaneKind
from .trellis import SUBSET_COUNT, ConvCode, TCQ, _ODS_GENERATORS  # noqa: F401
from .trellis import ConvCode as _ConvCode

__all__ = [
    "replay_body",
    "decode_codes",
    "decode_codes_mixed",
    "dequantize",
    "materialize_nvfp4",
    "reconstruct_unit",
    "unit_half_scales",
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
    subset = _window_subset(select, table_sub_flat, memory)
    return subsets_flat[subset.int() * points + point.int()]


def _window_subset(
    select: torch.Tensor, table_sub_flat: torch.Tensor, memory: int
) -> torch.Tensor:
    """Select bits ``[T, cols]`` -> the code's output subset per step.

    The state before step t is the previous ``memory`` select bits, so this is
    a windowed function of the stream with O(1) depth (see ``replay_body``).
    Shared by the per-position replay and the span-L replay, whose select
    stream is one bit per super-symbol.
    """

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
    return table_sub_flat[(select.int() << memory) + state.int()]


def _replay_span(
    body: torch.Tensor,
    subsets: torch.Tensor,
    table_sub: torch.Tensor,
    memory: int,
    span: int,
) -> torch.Tensor:
    """Span-L replay: one select bit per super-symbol, stored labels, points.

    Position 0 of a super-symbol carries ``[select | point]``; positions
    ``1..L-1`` carry ``[label | point]``.  The code's output for the select
    bit is the super-label; position 0's subset is the super-label minus the
    stored labels mod 4 (``trellis.py``).  Int64 throughout -- the fused uint8
    chain is the per-position path's; this one is correct first and the
    kernel lane is where the bandwidth question is answered.
    """
    steps, cols = body.shape
    points = subsets.shape[1]
    shift = points.bit_length() - 1
    mask = (1 << shift) - 1
    fields = body.long().reshape(steps // span, span, cols)
    select = (fields[:, 0] >> shift) & 1                        # [T, cols]
    stored = (fields[:, 1:] >> shift) & (SUBSET_COUNT - 1)     # [T, L-1, cols]
    point = fields & mask                                        # [T, L, cols]
    super_label = _window_subset(select, table_sub.reshape(-1), memory).long()
    first = (super_label - stored.sum(dim=1)) % SUBSET_COUNT
    labels = torch.cat([first.unsqueeze(1), stored], dim=1)     # [T, L, cols]
    return subsets[labels, point].reshape(steps, cols)


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
    body_bits: torch.Tensor, forest: AnchorForest, code: ConvCode, span: int = 1
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
    if span < 1 or rows % span:
        raise GrammarError(
            f"{rows} positions is not a whole number of span-{span} super-symbols"
        )

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
    if shifted and span > 1:
        return _replay_span(body_bits, subsets, table_sub, code.memory, span)
    if shifted:
        # Narrowing the body is a bandwidth choice too, and it is available
        # only while a code fits in a byte.  At R=9 this truncated the select
        # bit off every code and the replay diverged from row 1 on.
        body_fits = forest.rate <= 8
        body = (
            body_bits
            if body_bits.dtype == torch.uint8 or not body_fits
            else body_bits.to(torch.uint8)
        )
        # uint8 is a bandwidth choice, and it is only available while every
        # anchor index fits in one.  Above that it is a correctness bug.
        narrow = forest.grid.size <= 256 and points <= 256 and body_fits
        index_dtype = torch.uint8 if narrow else torch.int32
        args = (
            body,
            subsets.reshape(-1).to(index_dtype),
            table_sub.reshape(-1).to(torch.uint8),
            shift,
            mask,
            code.memory,
            points,
        )
        run = _fused_replay() if body.is_cuda and narrow else None
        if run is not None:
            try:
                return run(*args).long()
            except Exception:  # pragma: no cover - fall back, never fail closed
                pass
        return _replay_core(*args).long()

    # A code whose step is not a shift register still has to decode, so the
    # sequential walk stays as the general path rather than an assumption.
    fields = body_bits.long().reshape(rows // span, span, cols)
    select = (fields[:, 0] >> shift) & 1
    stored = (fields[:, 1:] >> shift) & (SUBSET_COUNT - 1)
    point = fields & mask
    anchors = torch.zeros(rows // span, span, cols, dtype=torch.long, device=device)
    state = torch.zeros(cols, dtype=torch.long, device=device)
    for sup in range(rows // span):
        super_label = table_sub[select[sup], state]
        first = (super_label - stored[sup].sum(dim=0)) % SUBSET_COUNT
        labels = torch.cat([first.unsqueeze(0), stored[sup]], dim=0)
        anchors[sup] = subsets[labels, point[sup]]
        state = table_next[select[sup], state]
    return anchors.reshape(rows, cols)


def replay_window(body_bits: torch.Tensor, window_bits: int, rate: int) -> torch.Tensor:
    """The window body's states from its bits alone (schema minor 2).

    ``state_t = ((state_{t-1} << R) | bits_t) mod 2^L`` from ``state_{-1} = 0``
    is a pure shift register, so a state is the last ``L`` bits of the
    column's stream: ``sum_j bits_{t-j} << jR`` over the ``ceil(L / R)``
    positions that reach back ``L`` bits, masked.  Closed form, one shifted
    add per tap, no walk down the column -- the same O(1)-depth property
    ``replay_body`` verifies for the convolutional code, here by construction.
    """
    if not 1 <= rate <= window_bits:
        raise GrammarError(f"rate {rate} does not fit a {window_bits}-bit window")
    bits = body_bits.long()
    state = bits.clone()
    taps = -(-window_bits // rate)
    for tap in range(1, taps):
        state[tap:] |= bits[:-tap] << (tap * rate)
    return state & ((1 << window_bits) - 1)


def _grid_and_forests(forest):
    """``(grid, forests)`` from a forest, a rate->forest map, or a bare grid.

    A window body has no forests -- its table is on the unit -- so a reader
    holding only bytes resolves the grid and passes that.  A TCQ unit still
    needs its forests, and the grid is theirs.
    """
    if isinstance(forest, PayloadGrid):
        return forest, {}
    forests = forest if isinstance(forest, dict) else {forest.rate: forest}
    return next(iter(forests.values())).grid, forests


def _decode_window(unit: "EncodedUnit", grid: PayloadGrid, code_dtype) -> torch.Tensor:
    device = unit.body_bits.device
    rows, cols = unit.body_bits.shape
    if unit.window_codes is None:
        raise GrammarError("a window body needs the unit's table")
    table = unit.window_codes.to(device).long()
    if table.numel() != 1 << unit.window_bits:
        raise GrammarError(
            f"the window table holds {table.numel()} entries, window_bits "
            f"{unit.window_bits} needs {1 << unit.window_bits}"
        )
    if table.numel() and int(table.max()) >= grid.size:
        raise GrammarError(
            f"the window table names code {int(table.max())} outside the "
            f"{grid.size}-code {grid.name} grid"
        )
    rates = torch.tensor(unit.rates, device=device)
    codes = torch.zeros(rows, cols, dtype=code_dtype, device=device)
    for present in sorted(set(unit.rates)):
        which = torch.nonzero(rates == present).squeeze(1)
        states = replay_window(unit.body_bits[:, which], unit.window_bits, present)
        codes[:, which] = table[states].to(code_dtype)
    return codes


def decode_codes(
    unit: EncodedUnit,
    forest: AnchorForest,
    code: ConvCode,
    completion: int | None = None,
) -> torch.Tensor:
    """Full decode from stored planes to E2M1 nibbles, in wire order."""
    if getattr(unit, "body", BodyKind.TCQ) is BodyKind.WINDOW:
        # This is the single-forest TCQ path; a window body has no forest and
        # replays through ``decode_codes_mixed``.  Refuse rather than replay a
        # window stream through the convolutional code, which would decode to
        # plausible wrong weights.
        raise GrammarError(
            "decode_codes is the TCQ path; a window body decodes through "
            "decode_codes_mixed / reconstruct_unit with its grid"
        )
    depth = forest.cap - forest.rate
    completion = depth if completion is None else completion
    device = unit.body_bits.device
    anchors = replay_body(unit.body_bits, forest, code, getattr(unit, "span", 1))
    # A code is a nibble only while the grid is E2M1.  Above 256 codes a uint8
    # table silently wraps, so the dtype follows the grid -- and it stays uint8
    # below that so the single-rate and mixed-rate decoders agree on the dtype
    # of a nibble.  They did not, and the release scatter caught it.
    code_dtype = torch.uint8 if forest.grid.size <= 256 else torch.int32
    blocks = torch.tensor(forest.blocks, device=device, dtype=code_dtype)
    reachable = blocks[:, :: 1 << (depth - completion)]
    codes = reachable[anchors, unit.completion_bits]
    if unit.release_index.numel():
        codes.reshape(-1)[unit.release_index] = unit.release_code.to(code_dtype)
    return codes


def dequantize(codes: torch.Tensor, scale: torch.Tensor, grid=None) -> torch.Tensor:
    """Codes times their half-block scale: the weights the runtime will see.

    ``codes`` is ``[steps, cols]``, one entry per *code*.  At arity 1 that is
    one per weight; above it, each code fans out to ``arity`` consecutive rows,
    which is the only place the tuple layout becomes visible outside the
    trellis.
    """
    from .alphabet import E2M1_GRID
    from .encode import grid_vector_table

    grid = grid or E2M1_GRID
    # ``.int()``: codes are uint8, and a uint8 index tensor is a boolean mask.
    out = grid_vector_table(grid, codes.device)[codes.int()]   # [steps, cols, k]
    if grid.arity == 1:
        return out.squeeze(-1) * scale
    steps, cols, arity = out.shape
    return out.permute(0, 2, 1).reshape(steps * arity, cols) * scale


def materialize_nvfp4(
    codes: torch.Tensor,
    scale_base: torch.Tensor,
    scale_refine: torch.Tensor,
    group: int = 32,
    half: int = 16,
    scale_lut: "torch.Tensor | None" = None,
    scale_global: float = 1.0,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Pack to the standard NVFP4 layout: 2 nibbles/byte + one E4M3 per 16.

    Returns ``(packed[rows, cols//2] uint8, scales[rows, cols//half] uint8,
    global_scale float)`` -- NVFP4's two scale levels, both of them.
    Every Tessera code is a legal E2M1 nibble by construction -- the alphabet
    *is* the grid -- so this pads information, never truncates it, and the
    result is an ordinary NVFP4 tensor that stock kernels consume.
    """
    from .wire import nvfp4_scale_bytes, nvfp4_scale_bytes_lut

    rows, cols = codes.shape
    if cols % 2:
        raise GrammarError(f"{cols} columns cannot pack 2 nibbles to a byte")
    low = codes[:, 0::2].to(torch.uint8)
    high = codes[:, 1::2].to(torch.uint8)
    packed = (low & 0xF) | ((high & 0xF) << 4)
    if scale_lut is not None:
        e4m3, global_scale = nvfp4_scale_bytes_lut(scale_refine, scale_lut, scale_global)
    else:
        e4m3, global_scale = nvfp4_scale_bytes(scale_base, scale_refine, group, half)
    return packed, e4m3.reshape(rows, cols // half), global_scale


def unit_half_scales(unit: "EncodedUnit") -> torch.Tensor:
    """The per-half scale a decoder derives from the unit's stored planes.

    Dispatches on the plane kind: S6b reads ``scale_base``/``scale_refine``
    through ``scales_from_planes``; a LUT plane reads the nibble through the
    unit's table and global.  Either way it is the only scale a reader has.
    """
    from .wire import scales_from_lut, scales_from_planes

    kind = getattr(unit, "scale_plane", ScalePlaneKind.S6B)
    if kind is ScalePlaneKind.CHANNEL:
        raise GrammarError(
            "a CHANNEL scale plane has no per-half scales: one word per output "
            "row; use unit_scale_field"
        )
    if kind is ScalePlaneKind.LUT:
        if unit.scale_lut is None:
            raise GrammarError("a LUT scale plane needs the unit's table")
        return scales_from_lut(unit.scale_refine, unit.scale_lut, unit.scale_global)
    return scales_from_planes(unit.scale_base, unit.scale_refine, unit.group, unit.half)


def unit_scale_field(unit: "EncodedUnit", rows: int, cols: int) -> torch.Tensor:
    """The per-position scale ``[rows, cols]`` a decoder derives from the planes.

    One dispatch for every plane kind: a block plane's halves repeated along
    the row (``unit_half_scales``), a CHANNEL plane's row word times the
    global broadcast along it.  ``rows`` is weight rows -- a caller holding a
    k-tuple unit passes ``steps * arity``.
    """
    if getattr(unit, "scale_plane", ScalePlaneKind.S6B) is ScalePlaneKind.CHANNEL:
        from .scale_channel import channel_scale_field

        if unit.scale_rows is None:
            raise GrammarError("a CHANNEL scale plane needs the unit's row words")
        return channel_scale_field(unit.scale_rows, unit.scale_global, rows, cols)
    return torch.repeat_interleave(unit_half_scales(unit), unit.half).reshape(rows, cols)


def materialize_fp8(
    unit: "EncodedUnit",
    forest: "AnchorForest | dict[int, AnchorForest] | PayloadGrid",
    code: "ConvCode | None",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """An E4M3 unit over a CHANNEL plane as the stock per-channel FP8 tensor.

    Returns ``(bytes uint8 [rows, cols], scale fp32 [rows])`` -- the
    ``weight`` / ``weight_scale`` pair a ``compressed-tensors`` FP8 checkpoint
    at ``strategy: channel`` carries, so a runtime that has never heard of
    Tessera serves the artifact W8A8 exactly as ``materialize_nvfp4`` lets
    one serve an E2M1 unit as NVFP4.  Codes map through the grid's ``native``
    bytes, so the two former-NaN slots land on their legal neighbour.
    """
    grid, forests = _grid_and_forests(forest)
    if grid.arity != 1 or grid.native is None or grid.size != 256:
        raise GrammarError(
            f"materialize_fp8 needs a scalar 256-code hardware grid, got {grid.name}"
        )
    if getattr(unit, "scale_plane", ScalePlaneKind.S6B) is not ScalePlaneKind.CHANNEL:
        raise GrammarError(
            "an FP8 per-channel tensor takes one scale per output row: the unit "
            f"carries a {unit.scale_plane.name} plane, which is a block layout"
        )
    codes = decode_codes_mixed(unit, forest, code)
    native = torch.tensor(grid.native, dtype=torch.long, device=codes.device)
    rows, cols = codes.shape
    return (
        native[codes.long()].to(torch.uint8),
        (unit.scale_rows.to(codes.device).float() * float(unit.scale_global)).reshape(rows),
    )


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
    grid, forests = _grid_and_forests(forest)
    device = unit.body_bits.device
    rows, cols = unit.body_bits.shape
    rates = torch.tensor(unit.rates, device=device)
    # uint8, not int64: a code is a 4-bit nibble.  Carrying the decoder's whole
    # output at eight bytes a nibble made the completion lookup cost more than
    # the trellis replay it followed.
    # A code is a nibble only on a 16-code grid.  ``uint8`` silently wraps at
    # 256, which a 1024-code k-tuple grid reaches -- and a wrapped code decodes
    # to a plausible wrong weight, never to an error.
    narrow = grid.size <= 256
    code_dtype = torch.uint8 if narrow else torch.int32
    if getattr(unit, "body", BodyKind.TCQ) is BodyKind.WINDOW:
        codes = _decode_window(unit, grid, code_dtype)
        if apply_release and unit.release_index.numel():
            codes.reshape(-1)[unit.release_index] = unit.release_code.to(code_dtype)
        return codes
    codes = torch.zeros(rows, cols, dtype=code_dtype, device=device)
    span = getattr(unit, "span", 1)
    # The depth the unit was WRITTEN at bounds the depth it can be read at.
    # ``completion_bits`` at level c is an index into ``reachable(anchor, c)``,
    # and the descendant order is a tree read most-significant-bit first, so the
    # same integer addresses a *different* node at a different level: reading a
    # level-1 index as a level-2 index lands in the wrong subtree.  This used to
    # default to the full capacity regardless of what the encoder spent, and it
    # was invisible -- ``encode_linear``'s round-trip check compares two decodes
    # that both made the assumption, so both were wrong identically.  Below the
    # rate cap it made every completion level decode as garbage, which is why
    # spending completion bits appeared to make the error *worse*.
    #
    # ``completion_limit`` is recovered from the artifact itself (the COMPLETION
    # plane's recorded element count), so a reader holding only bytes has it.
    # An explicit ``completion`` argument still truncates -- but it can only
    # truncate, never reach past what was written.
    for present in sorted(set(unit.rates)):
        picked = forests[present]
        depth = picked.cap - picked.rate
        limit = getattr(unit, "completion_limit", None)
        written = depth if limit is None else min(limit, depth)
        level = written if completion is None else min(completion, written)
        which = torch.nonzero(rates == present).squeeze(1)
        body = unit.body_bits[:, which].contiguous()
        comp = unit.completion_bits[:, which].contiguous()
        # A truncating read must narrow the completion WORD as well as the
        # descendant table.  ``reachable`` keeps 2^level columns, but the
        # stored word still holds the full 2^written-wide index, so indexing
        # one with the other runs off the end -- on CUDA as a device-side
        # assert, on CPU as an IndexError.  The descendant order is a tree read
        # most-significant-bit first, so the ancestor at ``level`` is the top
        # ``level`` bits: shift the low (written - level) bits away.  This is a
        # no-op at level == written, which is why every existing test missed it
        # -- they all decode at the level they encoded.
        if level < written:
            comp = comp >> (written - level)
        subsets, table_next, table_sub = _replay_tables(picked, code, str(device))
        blocks = torch.tensor(picked.blocks, device=device, dtype=code_dtype)
        reachable = blocks[:, :: 1 << (depth - level)].contiguous()
        points = subsets.shape[1]
        shift = points.bit_length() - 1
        # The fused chain carries anchors and points in uint8 for bandwidth.
        # Above 256 anchors that is not a slow path, it is a wrong one.
        run = (
            _fused_decode()
            if body.is_cuda and narrow and points <= 256 and picked.rate <= 8
            and span == 1
            else None
        )
        args = (
            body if run is None or body.dtype == torch.uint8 else body.to(torch.uint8),
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
        anchors = replay_body(body, picked, code, span)
        codes[:, which] = reachable.long()[anchors, comp.long()].to(code_dtype)
    if apply_release and unit.release_index.numel():
        codes.reshape(-1)[unit.release_index] = unit.release_code.to(code_dtype)
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

    codes = decode_codes_mixed(unit, forest, code, completion)
    grid, _forests = _grid_and_forests(forest)
    # ``body_bits`` is one row per CODE; the scale planes are per position.
    steps, cols = unit.body_bits.shape
    rows = steps * grid.arity
    if scale is None:
        scale = unit_scale_field(unit, rows, cols)
    out = dequantize(codes, scale, grid)
    if unit.diagonals is not None:
        out = undo_diagonals(out, unit.diagonals)
    return undo_rotation(out, unit.rotation, unit.rotation_block)
