"""Tessera's own decode kernel: compressed bits straight into the matmul.

The stock lane materialises Tessera's body into ordinary NVFP4 nibbles at load
time, so a runtime that has never heard of Tessera serves it.  That path is
attested and stays.  What it cannot do is *save memory*: the materialised
tensor is 4.5 bpp whatever the artifact cost on disk, because 4.5 bpp is what
NVFP4's layout weighs.

This module is the other lane.  It keeps the body compressed in VRAM and
decodes inside the kernel, so resident bytes equal stored bytes.  The whole
thing is possible because of one property established in
`docs/measurements/nvfp4-kernel-attestation.md`: ``ConvCode.step`` is a shift
register, so the trellis state at row *r* is nothing but the previous
``memory`` select bits.  A tile of the weight matrix therefore decodes from a
*local* span of the bitstream with a six-row halo -- no sequential dependence
down the column, which is what would otherwise make a trellis format
un-kernelable.

Every table here is generated from ``decode._replay_tables``, never re-derived,
so the kernel and the reference decoder cannot drift apart.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .alphabet import AnchorForest
from .decode import _replay_tables
from .errors import GrammarError
from .trellis import ConvCode

__all__ = ["build_code_lut", "build_tuple_index_lut", "build_anchor_values",
           "tessera_dequant", "tessera_gemv", "nvfp4_gemv"]


def build_code_lut(
    forest: AnchorForest, code: ConvCode, device: str = "cuda"
) -> torch.Tensor:
    """One flat table taking (state, select, point) straight to an E2M1 nibble.

    The reference decoder walks three lookups -- ``table_sub`` to a subset,
    ``subsets`` to an anchor, ``blocks`` to a code.  All three are pure
    functions of at most ``memory + R`` bits, so their composition is a single
    table of ``2^(memory + R)`` bytes: 512 for memory=6, R=3.  Composing them
    here rather than in the kernel means the kernel indexes the *same* objects
    the reference does, and a divergence is impossible rather than merely
    tested for.
    """
    if forest.rate != 3:
        raise GrammarError(
            f"the kernel lane is defined for a full-rate R=3 body; got R={forest.rate}. "
            "Lower rates carry completion bits, which is a second plane the kernel "
            "does not read yet."
        )
    subsets, _table_next, table_sub = _replay_tables(forest, code, device)
    blocks = torch.tensor(forest.blocks, device=device, dtype=torch.uint8)
    points = subsets.shape[1]
    states = code.states
    lut = torch.zeros(states * 2 * points, dtype=torch.uint8, device=device)
    for state in range(states):
        for select in range(2):
            subset = int(table_sub[select, state])
            for point in range(points):
                anchor = int(subsets[subset, point])
                lut[(state * 2 + select) * points + point] = blocks[anchor, 0]
    return lut


@triton.jit
def _decode_tile(
    body_ptr, lut_ptr, offs_n, offs_k, live, rows, col_stride_bits,
    memory: tl.constexpr, rate: tl.constexpr,
):
    """A [BLOCK_N, BLOCK_K] tile of E2M1 nibbles, decoded from the bitstream.

    Returns ``[BLOCK_K, BLOCK_N]`` -- row index *last*, which looks transposed
    and is not negotiable: the BODY plane is packed column-major, so a warp has
    to walk consecutive rows of one column or every lane lands in a different
    column ``3*rows`` bits apart.  Written the natural way round this kernel ran
    at 6% of the box's 246 GB/s; the NVFP4 comparator, whose plane *is*
    row-major, was unaffected, which is how the layout was identified as the
    cause rather than the decode.

    Shared verbatim by the dequant harness and the GEMV so the thing that is
    tested is the thing that runs.  The trellis runs down rows with one
    independent trellis per column, so this needs the six select bits
    immediately above the tile and nothing else -- that halo is the entire
    cross-tile dependency of a trellis format, and it is why the fused GEMM is
    possible at all.
    """
    pos = offs_k[:, None].to(tl.int64) * col_stride_bits + offs_n[None, :].to(tl.int64) * rate
    byte = pos // 8
    within = (pos % 8).to(tl.int32)
    lo = tl.load(body_ptr + byte, mask=live, other=0).to(tl.int32)
    hi = tl.load(body_ptr + byte + 1, mask=live, other=0).to(tl.int32)
    field = (((lo << 8) | hi) >> (16 - rate - within)) & ((1 << rate) - 1)
    select = field >> (rate - 1)
    point = field & ((1 << (rate - 1)) - 1)

    state = tl.zeros(select.shape, dtype=tl.int32)
    for j in tl.static_range(1, memory + 1):
        back = offs_n[None, :] - j
        ok = live & (back >= 0)
        p = offs_k[:, None].to(tl.int64) * col_stride_bits + back.to(tl.int64) * rate
        b = tl.load(body_ptr + p // 8, mask=ok, other=0).to(tl.int32)
        bit = (b >> (7 - (p % 8).to(tl.int32))) & 1
        state = state | (tl.where(ok, bit, 0) << (memory - j))

    return tl.load(
        lut_ptr + (state * 2 + select) * (1 << (rate - 1)) + point, mask=live, other=0
    ).to(tl.int32)


@triton.jit
def _apply_scale_kn(scale_ptr, offs_n, offs_k, live, cols, half: tl.constexpr):
    """Scales for a ``[BLOCK_K, BLOCK_N]`` tile.  See ``_apply_scale``."""
    group = (offs_n[None, :].to(tl.int64) * cols + offs_k[:, None].to(tl.int64)) // half
    e4m3 = tl.load(scale_ptr + group, mask=live, other=0).to(tl.int32)
    return tl.exp2(((e4m3 >> 3) & 0xF).to(tl.float32) - 7.0) * (
        1.0 + (e4m3 & 0x7).to(tl.float32) / 8.0
    )


@triton.jit
def _apply_scale(scale_ptr, offs_n, offs_k, live, cols, half: tl.constexpr):
    """NVFP4's first scale level: one E4M3 per `half` positions, row-major.

    Decoded by field arithmetic rather than a dtype cast, because the exponent
    and mantissa are exactly what segment 2b stored -- see `wire.nvfp4_scale_bytes`,
    which is the producer side of this same relabelling.
    """
    group = (offs_n[:, None].to(tl.int64) * cols + offs_k[None, :].to(tl.int64)) // half
    e4m3 = tl.load(scale_ptr + group, mask=live, other=0).to(tl.int32)
    return tl.exp2(((e4m3 >> 3) & 0xF).to(tl.float32) - 7.0) * (
        1.0 + (e4m3 & 0x7).to(tl.float32) / 8.0
    )


@triton.jit
def _dequant_kernel(
    body_ptr,          # uint8, the packed BODY plane
    lut_ptr,           # uint8, (state, select, point) -> nibble
    scale_ptr,         # uint8, one E4M3 per `half` positions, row-major
    value_ptr,         # fp32, the 16-entry E2M1 value table
    out_ptr,           # fp32 [rows, cols]
    global_scale,      # fp32, NVFP4's second scale level
    rows, cols,
    col_stride_bits,   # bits between the start of column k and column k+1
    memory: tl.constexpr,
    rate: tl.constexpr,
    half: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Decode a [BLOCK_N, BLOCK_K] tile of the weight matrix from the bitstream.

    The trellis runs down *rows* with one independent trellis per column, so a
    tile needs the six select bits immediately above it and nothing else.  That
    six-row halo is the entire cross-tile dependency of a trellis format.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    live = (offs_k[:, None] < cols) & (offs_n[None, :] < rows)

    nibble = _decode_tile(
        body_ptr, lut_ptr, offs_n, offs_k, live, rows, col_stride_bits, memory, rate
    )
    value = tl.load(value_ptr + nibble, mask=live, other=0.0)
    scale = _apply_scale_kn(scale_ptr, offs_n, offs_k, live, cols, half)
    tl.store(
        out_ptr + offs_n[None, :].to(tl.int64) * cols + offs_k[:, None],
        value * scale * global_scale,
        mask=live,
    )


@triton.jit
def _gemv_kernel(
    x_ptr, body_ptr, lut_ptr, scale_ptr, value_ptr, out_ptr,
    global_scale, rows, cols, col_stride_bits,
    memory: tl.constexpr, rate: tl.constexpr, half: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """y[n] = sum_k x[k] * W[n, k], with W never leaving its bitstream.

    This is the whole point of the lane.  The stock path must first expand the
    body into 4.5 bpp of NVFP4 and then read *that*; here the only weight bytes
    crossing the bus are the 3.5 bpp actually stored, and the decode is a
    handful of shifts and a 512-byte lookup on data already in registers.

    Split over K as well as N: a decode-phase GEMV on one Linear otherwise
    launches ``rows/BLOCK_N`` programs, which is tens of CTAs on a matrix that
    should saturate the machine.  The first version of this kernel was 34x off
    bandwidth for exactly that reason, and so was its NVFP4 comparator -- which
    is how we know it was the blocking and not the decode.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    span = tl.cdiv(cols, SPLIT_K)
    start = pid_k * span
    for k0 in range(start, tl.minimum(start + span, cols), BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        live = (offs_k[:, None] < cols) & (offs_n[None, :] < rows)
        nibble = _decode_tile(
            body_ptr, lut_ptr, offs_n, offs_k, live, rows, col_stride_bits, memory, rate
        )
        value = tl.load(value_ptr + nibble, mask=live, other=0.0)
        weight = value * _apply_scale_kn(scale_ptr, offs_n, offs_k, live, cols, half)
        xs = tl.load(x_ptr + offs_k, mask=offs_k < cols, other=0.0)
        acc += tl.sum(weight * xs[:, None], axis=0)

    tl.atomic_add(out_ptr + offs_n, acc * global_scale, mask=offs_n < rows)


@triton.jit
def _nvfp4_gemv_kernel(
    x_ptr, packed_ptr, scale_ptr, value_ptr, out_ptr,
    global_scale, rows, cols,
    half: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """The same GEMV over materialised NVFP4, written the same way.

    Not a baseline for NVFP4 performance -- CUTLASS is -- but the *controlled*
    comparison: identical blocking, identical accumulate, identical author.  The
    only difference left is how many weight bytes cross the bus, 4.5 bpp against
    3.5, which is the quantity the lane exists to change.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    span = tl.cdiv(cols, SPLIT_K)
    start = pid_k * span
    for k0 in range(start, tl.minimum(start + span, cols), BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        live = (offs_n[:, None] < rows) & (offs_k[None, :] < cols)
        flat = offs_n[:, None].to(tl.int64) * cols + offs_k[None, :].to(tl.int64)
        byte = tl.load(packed_ptr + flat // 2, mask=live, other=0).to(tl.int32)
        nibble = tl.where(flat % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
        value = tl.load(value_ptr + nibble, mask=live, other=0.0)
        weight = value * _apply_scale(scale_ptr, offs_n, offs_k, live, cols, half)
        xs = tl.load(x_ptr + offs_k, mask=offs_k < cols, other=0.0)
        acc += tl.sum(weight * xs[None, :], axis=1)

    tl.atomic_add(out_ptr + offs_n, acc * global_scale, mask=offs_n < rows)


def tessera_dequant(
    body: torch.Tensor,
    lut: torch.Tensor,
    e4m3: torch.Tensor,
    global_scale: float,
    rows: int,
    cols: int,
    rate: int = 3,
    memory: int = 6,
    half: int = 16,
) -> torch.Tensor:
    """Bits -> weights, without ever forming an NVFP4 tensor.

    This is the correctness harness for the kernel lane rather than its point:
    it proves the in-kernel decode reproduces the reference bit-for-bit, which
    is what a fused GEMM then gets to assume.
    """
    from .encode import e2m1_value_table

    out = torch.empty(rows, cols, dtype=torch.float32, device=body.device)
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_N"]),
        triton.cdiv(cols, meta["BLOCK_K"]),
    )
    _dequant_kernel[grid](
        body, lut, e4m3.reshape(-1), e2m1_value_table(body.device).float(), out,
        float(global_scale), rows, cols, rate * rows,
        memory=memory, rate=rate, half=half, BLOCK_N=64, BLOCK_K=64,
    )
    return out


def tessera_gemv(
    x: torch.Tensor,
    body: torch.Tensor,
    lut: torch.Tensor,
    e4m3: torch.Tensor,
    global_scale: float,
    rows: int,
    cols: int,
    rate: int = 3,
    memory: int = 6,
    half: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    split_k: int = 16,
) -> torch.Tensor:
    """``x @ W.T`` with W read straight from its bitstream at 3.5 bpp."""
    from .encode import e2m1_value_table

    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _gemv_kernel[(triton.cdiv(rows, block_n), split_k)](
        x.reshape(-1), body, lut, e4m3.reshape(-1),
        e2m1_value_table(x.device).float(), out,
        float(global_scale), rows, cols, rate * rows,
        memory=memory, rate=rate, half=half,
        BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
    )
    return out


def nvfp4_gemv(
    x: torch.Tensor,
    packed: torch.Tensor,
    e4m3: torch.Tensor,
    global_scale: float,
    rows: int,
    cols: int,
    half: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    split_k: int = 16,
) -> torch.Tensor:
    """The controlled 4.5 bpp comparator: same blocking, same author."""
    from .encode import e2m1_value_table

    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _nvfp4_gemv_kernel[(triton.cdiv(rows, block_n), split_k)](
        x.reshape(-1), packed.reshape(-1), e4m3.reshape(-1),
        e2m1_value_table(x.device).float(), out,
        float(global_scale), rows, cols,
        half=half, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
    )
    return out


# ---------------------------------------------------------------------------
# The kernel lane's resident layout
#
# The wire BODY plane interleaves each position's select bit with its point
# bits, which is right for a bitstream and wrong for a decoder: assembling the
# six-bit state then costs six separate byte loads, one per row of history, and
# the measured kernel spent its time on them rather than on weight bytes.
#
# Sliced into a select plane and a point plane, the state stops being six loads
# and becomes seven *adjacent bits*: rows n-6..n of one column are consecutive
# in the select plane, so one 16-bit window carries the whole history plus the
# current select bit.  Nothing about the artifact changes -- the same bits are
# permuted at load, exactly as the stock lane permutes them into NVFP4 nibbles
# -- so this costs no grammar and no stored bytes.
# ---------------------------------------------------------------------------

#: Zero bits prepended to each column's select plane so that row 0's history
#: window reads the encoder's initial state instead of the previous column.
#: Eight rather than six keeps every column byte-aligned.
SELECT_PAD = 8


def pack_kernel_planes(
    body_bits: torch.Tensor, rate: int = 3, memory: int = 6, span: int = 1
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Wire BODY -> (select plane, point plane), column-major, MSB-first.

    The select plane carries ``SELECT_PAD`` zero bits before each column, which
    is what lets a decoder read row 0's history without a boundary test: the pad
    *is* the initial state.

    ``span`` is the unit's trellis span.  The kernel lane reads one select bit
    and ``rate - 1`` point bits per position; a span-2 body carries a stored
    label at every second position and one select bit per pair, which these
    planes cannot express.  Refused here, at the seam, rather than decoded as
    if it were span 1 -- which would produce plausible weights from the wrong
    subsets.  The scale plane needs no such guard: a LUT plane materialises
    to the same per-16 E4M3 bytes the kernel already reads.
    """
    rows, cols = body_bits.shape
    device = body_bits.device
    if span != 1:
        raise GrammarError(
            f"the kernel lane decodes span-1 bodies; this unit is span {span}. "
            "The span-2 decode is the next kernel-lane item; use the reference "
            "decoder (materialize_nvfp4) until it lands."
        )
    if rows % 8:
        raise GrammarError(f"{rows} rows does not byte-align a column plane")
    body = body_bits.to(torch.int32)

    select = (body >> (rate - 1)) & 1
    padded = torch.zeros(rows + SELECT_PAD, cols, dtype=torch.int32, device=device)
    padded[SELECT_PAD:] = select
    select_plane = _pack_columns(padded, 1)

    point = body & ((1 << (rate - 1)) - 1)
    point_plane = _pack_columns(point, rate - 1)
    return select_plane, point_plane


def _pack_columns(values: torch.Tensor, width: int) -> torch.Tensor:
    """Pack ``[rows, cols]`` small integers column-major, MSB-first within byte."""
    rows, cols = values.shape
    bits = torch.zeros(cols, rows * width, dtype=torch.uint8, device=values.device)
    for position in range(width):
        bits[:, position::width] = (
            (values >> (width - 1 - position)) & 1
        ).t().to(torch.uint8)
    flat = bits.reshape(-1)
    weights = (1 << torch.arange(7, -1, -1, device=values.device, dtype=torch.uint8))
    return (flat.reshape(-1, 8) * weights).sum(1, dtype=torch.uint8)


def build_history_lut(
    forest: AnchorForest, code: ConvCode, device: str = "cuda"
) -> torch.Tensor:
    """``(history window, point) -> E2M1 nibble``, indexed by raw stream bits.

    The seven-bit window read out of the select plane is in *stream* order --
    oldest row first -- while ``ConvCode``'s state numbers the newest row
    highest.  Rather than reverse the bits in the kernel every position, the
    permutation is folded into the table, which costs nothing: the table is the
    same 512 bytes either way.  Built from ``_replay_tables`` so it cannot
    disagree with the reference decoder.
    """
    subsets, _table_next, table_sub = _replay_tables(forest, code, device)
    blocks = torch.tensor(forest.blocks, device=device, dtype=torch.uint8)
    points = subsets.shape[1]
    lut = torch.zeros((1 << (memory_bits := code.memory + 1)) * points,
                      dtype=torch.uint8, device=device)
    for window in range(1 << memory_bits):
        select = window & 1
        history = window >> 1
        # stream order: bit (memory-1-i) of `history` is row n-memory+i.
        state = 0
        for i in range(code.memory):
            bit = (history >> (code.memory - 1 - i)) & 1
            state |= bit << i
        subset = int(table_sub[select, state])
        for point in range(points):
            lut[window * points + point] = blocks[int(subsets[subset, point]), 0]
    return lut


@triton.jit
def _sliced_gemv_kernel(
    x_ptr, select_ptr, point_ptr, lut_ptr, scale_ptr, value_ptr, out_ptr,
    global_scale, rows, cols,
    memory: tl.constexpr, rate: tl.constexpr, half: tl.constexpr, pad: tl.constexpr,
    BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """GEMV over the sliced layout, one scale group of K per iteration.

    Two costs the interleaved version paid and this one does not: the six
    history loads collapse into one 16-bit window, and ``BLOCK_K == half`` makes
    the E4M3 scale constant across the iteration, so it is loaded once per output
    row instead of once per weight.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    live_n = offs_n < rows

    groups = cols // half
    for g in range(pid_k, groups, SPLIT_K):
        # One E4M3 per (row, group): loaded once, used `half` times.
        e4m3 = tl.load(scale_ptr + g * rows + offs_n, mask=live_n, other=0).to(tl.int32)
        scale = tl.exp2(((e4m3 >> 3) & 0xF).to(tl.float32) - 7.0) * (
            1.0 + (e4m3 & 0x7).to(tl.float32) / 8.0
        )
        for i in tl.static_range(half):
            k = g * half + i
            # Seven adjacent bits of the select plane: rows n-memory..n.
            p = k.to(tl.int64) * (rows + pad) + offs_n.to(tl.int64) + (pad - memory)
            lo = tl.load(select_ptr + p // 8, mask=live_n, other=0).to(tl.int32)
            hi = tl.load(select_ptr + p // 8 + 1, mask=live_n, other=0).to(tl.int32)
            window = (((lo << 8) | hi) >> (9 - (p % 8).to(tl.int32))) & 0x7F

            q = k.to(tl.int64) * rows * (rate - 1) + offs_n.to(tl.int64) * (rate - 1)
            byte = tl.load(point_ptr + q // 8, mask=live_n, other=0).to(tl.int32)
            pt = (byte >> (8 - (rate - 1) - (q % 8).to(tl.int32))) & ((1 << (rate - 1)) - 1)

            nib = tl.load(
                lut_ptr + window * (1 << (rate - 1)) + pt, mask=live_n, other=0
            ).to(tl.int32)
            acc += tl.load(value_ptr + nib, mask=live_n, other=0.0) * scale * tl.load(
                x_ptr + k
            )

    tl.atomic_add(out_ptr + offs_n, acc * global_scale, mask=live_n)


def tessera_gemv_sliced(
    x: torch.Tensor,
    select_plane: torch.Tensor,
    point_plane: torch.Tensor,
    lut: torch.Tensor,
    e4m3_t: torch.Tensor,
    global_scale: float,
    rows: int,
    cols: int,
    rate: int = 3,
    memory: int = 6,
    half: int = 16,
    block_n: int = 128,
    split_k: int = 8,
) -> torch.Tensor:
    """``W @ x`` from the sliced resident layout.  ``e4m3_t`` is ``[cols/half, rows]``."""
    from .encode import e2m1_value_table

    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _sliced_gemv_kernel[(triton.cdiv(rows, block_n), split_k)](
        x.reshape(-1), select_plane, point_plane, lut, e4m3_t.reshape(-1),
        e2m1_value_table(x.device).float(), out,
        float(global_scale), rows, cols,
        memory=memory, rate=rate, half=half, pad=SELECT_PAD,
        BLOCK_N=block_n, SPLIT_K=split_k,
    )
    return out


@triton.jit
def _nvfp4_sliced_gemv_kernel(
    x_ptr, packed_ptr, scale_ptr, value_ptr, out_ptr,
    global_scale, rows, cols,
    half: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """The 4.5 bpp comparator, in the same layout family as the sliced lane.

    Nibbles packed column-major so a warp walks consecutive output rows, scales
    transposed to ``[cols/half, rows]`` and hoisted out of the inner loop --
    every structural advantage the Tessera kernel gets, given to NVFP4 too.
    What is left between them is the thing under test: 4.5 bpp of weight bytes
    against 3.5, and one extra table lookup per position to decode them.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    live_n = offs_n < rows

    groups = cols // half
    for g in range(pid_k, groups, SPLIT_K):
        e4m3 = tl.load(scale_ptr + g * rows + offs_n, mask=live_n, other=0).to(tl.int32)
        scale = tl.exp2(((e4m3 >> 3) & 0xF).to(tl.float32) - 7.0) * (
            1.0 + (e4m3 & 0x7).to(tl.float32) / 8.0
        )
        for i in tl.static_range(half):
            k = g * half + i
            flat = k.to(tl.int64) * rows + offs_n.to(tl.int64)
            byte = tl.load(packed_ptr + flat // 2, mask=live_n, other=0).to(tl.int32)
            nib = tl.where(flat % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
            acc += tl.load(value_ptr + nib, mask=live_n, other=0.0) * scale * tl.load(
                x_ptr + k
            )

    tl.atomic_add(out_ptr + offs_n, acc * global_scale, mask=live_n)


def nvfp4_gemv_sliced(
    x: torch.Tensor,
    packed_t: torch.Tensor,
    e4m3_t: torch.Tensor,
    global_scale: float,
    rows: int,
    cols: int,
    half: int = 16,
    block_n: int = 256,
    split_k: int = 32,
) -> torch.Tensor:
    """``W @ x`` over column-major NVFP4 nibbles.  The controlled comparator."""
    from .encode import e2m1_value_table

    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _nvfp4_sliced_gemv_kernel[(triton.cdiv(rows, block_n), split_k)](
        x.reshape(-1), packed_t.reshape(-1), e4m3_t.reshape(-1),
        e2m1_value_table(x.device).float(), out,
        float(global_scale), rows, cols,
        half=half, BLOCK_N=block_n, SPLIT_K=split_k,
    )
    return out


def pack_nvfp4_column_major(codes: torch.Tensor) -> torch.Tensor:
    """Nibbles packed along the row axis, the comparator's resident layout."""
    rows, cols = codes.shape
    flat = codes.t().reshape(-1).to(torch.uint8)
    return (flat[0::2] & 0xF) | ((flat[1::2] & 0xF) << 4)


def build_value_lut(
    forest: AnchorForest, code: ConvCode, device: str = "cuda"
) -> torch.Tensor:
    """``(history window, point) -> the decoded E2M1 *value*``, as fp32.

    The nibble is never wanted for its own sake in a GEMV -- it is immediately
    used to index the 16-entry value table -- so the two lookups fold into one
    2 KB table and the inner loop loses a load per position.  Built by composing
    ``build_history_lut`` with the same value table the reference decoder uses.
    """
    from .encode import e2m1_value_table

    return e2m1_value_table(device).float()[build_history_lut(forest, code, device).long()]


@triton.jit
def _wide_gemv_kernel(
    x_ptr, select_ptr, point_ptr, lut_ptr, scale_ptr, out_ptr,
    global_scale, rows, cols,
    memory: tl.constexpr, rate: tl.constexpr, half: tl.constexpr, pad: tl.constexpr,
    LANES: tl.constexpr, VEC: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """GEMV where each lane decodes ``VEC`` consecutive output rows per load.

    The sliced layout removed the six history loads; what remained was one load
    per *position*, which left the kernel instruction-bound at a third of the
    bandwidth its byte count deserved -- while the NVFP4 comparator, needing one
    load per two positions, ran at 84% of peak.

    Consecutive rows of a column share almost all of their history: rows n and
    n+1 differ by one select bit.  So ``VEC`` rows need ``VEC + memory`` select
    bits -- fifteen for VEC=8 -- which is three bytes, and ``2*VEC`` point bits,
    which is two.  Five loads now serve eight positions instead of sixteen.

    Both planes land on constant shifts, which is why VEC is 8 and the pad is 8:
    with ``rows % 8 == 0`` the point plane is byte-aligned outright, and the
    select plane sits at a fixed offset of ``pad - memory`` bits into its byte.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    base = pid_n * LANES * VEC + tl.arange(0, LANES) * VEC
    vec = tl.arange(0, VEC)
    offs_n = base[:, None] + vec[None, :]
    live = offs_n < rows
    live_base = base < rows
    acc = tl.zeros((LANES, VEC), dtype=tl.float32)

    groups = cols // half
    for g in range(pid_k, groups, SPLIT_K):
        scale_byte = tl.load(
            scale_ptr + g * rows + offs_n, mask=live, other=0
        ).to(tl.int32)
        scale = tl.exp2(((scale_byte >> 3) & 0xF).to(tl.float32) - 7.0) * (
            1.0 + (scale_byte & 0x7).to(tl.float32) / 8.0
        )
        for i in tl.static_range(half):
            k = g * half + i
            # Select plane: VEC + memory bits at a constant sub-byte offset.
            p = k.to(tl.int64) * (rows + pad) + base.to(tl.int64) + (pad - memory)
            b0 = tl.load(select_ptr + p // 8, mask=live_base, other=0).to(tl.int32)
            b1 = tl.load(select_ptr + p // 8 + 1, mask=live_base, other=0).to(tl.int32)
            b2 = tl.load(select_ptr + p // 8 + 2, mask=live_base, other=0).to(tl.int32)
            wide = (b0 << 16) | (b1 << 8) | b2
            # `wide` is a 24-bit big-endian window from byte p//8.  Row base+v's
            # seven history bits end at bit offset (p%8) + v + memory, and
            # p%8 is the constant `pad - memory`, so the shift is 23 - pad - v.
            window = (wide[:, None] >> (23 - pad - vec[None, :])) & 0x7F

            # Point plane: 2*VEC bits, byte-aligned.
            q = k.to(tl.int64) * rows * (rate - 1) + base.to(tl.int64) * (rate - 1)
            c0 = tl.load(point_ptr + q // 8, mask=live_base, other=0).to(tl.int32)
            c1 = tl.load(point_ptr + q // 8 + 1, mask=live_base, other=0).to(tl.int32)
            pair = (c0 << 8) | c1
            pt = (pair[:, None] >> (16 - (rate - 1) * (vec[None, :] + 1))) & (
                (1 << (rate - 1)) - 1
            )

            value = tl.load(
                lut_ptr + window * (1 << (rate - 1)) + pt, mask=live, other=0.0
            )
            acc += value * scale * tl.load(x_ptr + k)

    tl.atomic_add(out_ptr + offs_n, acc * global_scale, mask=live)


def tessera_gemv_wide(
    x: torch.Tensor,
    select_plane: torch.Tensor,
    point_plane: torch.Tensor,
    value_lut: torch.Tensor,
    e4m3_t: torch.Tensor,
    global_scale: float,
    rows: int,
    cols: int,
    rate: int = 3,
    memory: int = 6,
    half: int = 16,
    lanes: int = 64,
    vec: int = 8,
    split_k: int = 32,
) -> torch.Tensor:
    """``W @ x`` at ``VEC`` output rows per lane per load.

    The split-K reduction lands through ``tl.atomic_add``, so the low bits of
    the output depend on the order the partial sums arrive and vary run to run.
    That is harmless for weights -- a one-hot probe returns a column exactly,
    because nothing is summed -- but it means this path must never be cited in a
    bit-identical-within-session reproducibility claim.
    """
    if vec != 8:
        raise GrammarError(
            f"vec={vec}: the constant-shift arithmetic in _wide_gemv_kernel is "
            "derived for VEC=8 against SELECT_PAD=8 and rows % 8 == 0. Another "
            "width needs the shifts re-derived, not just this check relaxed."
        )
    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _wide_gemv_kernel[(triton.cdiv(rows, lanes * vec), split_k)](
        x.reshape(-1), select_plane, point_plane, value_lut, e4m3_t.reshape(-1), out,
        float(global_scale), rows, cols,
        memory=memory, rate=rate, half=half, pad=SELECT_PAD,
        LANES=lanes, VEC=vec, SPLIT_K=split_k,
    )
    return out


def build_tuple_value_lut(
    forest: AnchorForest, code: ConvCode, device: str = "cuda"
) -> torch.Tensor:
    """``(history window, point) -> the code's ``arity`` decoded values``, fp32.

    The scalar ``build_value_lut`` folds three lookups -- replay, anchor,
    value -- into one table.  This folds the same three, and the k-tuple's fan
    out with them: a code decodes to ``arity`` consecutive output rows, so the
    table is ``arity`` floats wide and the kernel's inner loop still does one
    load per output row.

    Reading the reconstruction out of ``grid.vector`` rather than an E2M1 table
    is what makes the kernel lane grid-agnostic: a source-matched grid is just
    different numbers here, which is the whole reason it is reachable at all.

    **This fused form is the reference, not the serving path.**  It folds a
    shared structure -- which anchor a ``(window, point)`` lands on -- together
    with a per-unit meaning -- what that anchor reconstructs to.  While the grid
    is global that costs nothing, because every unit at a given rate shares one
    64 KB table.  Give each unit its own grid and the fused table becomes
    per-unit too: 37,694 units x 64 KB is 2.4 GB of resident lookup, which is
    1.6% of the body spent buying back bits the format just saved.

    ``build_tuple_index_lut`` and ``build_anchor_values`` are the same table
    split along that seam -- 16 KB shared plus 2 KB per unit, 32x less -- at the
    cost of one dependent load.  They compose back to exactly this tensor, which
    is why this function is now defined *as* their composition: two forms that
    must agree cannot drift if one is built from the other.
    """
    index = build_tuple_index_lut(forest, code, device)
    values = build_anchor_values(forest, device)
    arity = forest.grid.arity
    return values.reshape(-1, arity)[index.long()].reshape(-1)


def build_tuple_index_lut(
    forest: AnchorForest, code: ConvCode, device: str = "cuda"
) -> torch.Tensor:
    """``(history window, point) -> anchor index``.  The SHARED half.

    This half depends on the replay tables and the forest's *block layout* --
    never on what a block reconstructs to -- so every unit built at the same
    rate with the same code shares one copy of it, whatever grid it carries.
    """
    subsets, _table_next, table_sub = _replay_tables(forest, code, device)
    points = subsets.shape[1]
    subsets_cpu = subsets.tolist()
    sub_cpu = table_sub.tolist()
    if len(forest.blocks) > (1 << 15):
        raise GrammarError(
            f"{len(forest.blocks)} anchors does not fit the int16 index table; "
            "widen it deliberately rather than letting it wrap"
        )
    flat: "list[int]" = []
    for window in range(1 << (code.memory + 1)):
        select = window & 1
        history = window >> 1
        # The plane's window is in stream order -- oldest row first -- while
        # ConvCode numbers the newest bit highest.  Folding the reversal into
        # the table costs nothing and saves it per position.
        state = 0
        for index in range(code.memory):
            state |= ((history >> (code.memory - 1 - index)) & 1) << index
        row = subsets_cpu[sub_cpu[select][state]]
        flat.extend(int(row[point]) for point in range(points))
    return torch.tensor(flat, dtype=torch.int16, device=device)


def build_anchor_values(
    forest: AnchorForest, device: str = "cuda"
) -> torch.Tensor:
    """``anchor index -> the anchor's ``arity`` values``.  The PER-UNIT half.

    ``anchors * arity`` floats: 2 KB at the rate cap of a k=2 grid, against the
    fused table's 64 KB.  That ratio is the reason the split exists -- see the
    note on ``build_tuple_value_lut``.
    """
    grid = forest.grid
    flat: "list[float]" = []
    for block in forest.blocks:
        flat.extend(grid.vector(block[0]))
    return torch.tensor(flat, dtype=torch.float32, device=device)


@triton.jit
def _tuple_gemv_kernel(
    x_ptr, select_ptr, point_ptr, index_ptr, value_ptr, scale_ptr, out_ptr,
    global_scale, rows, steps, cols,
    memory: tl.constexpr, rate: tl.constexpr, arity: tl.constexpr,
    half: tl.constexpr, pad: tl.constexpr,
    LANES: tl.constexpr, VEC: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """GEMV over a k-tuple body: ``VEC`` codes, ``VEC * arity`` output rows.

    The scalar wide kernel reads one select bit and ``rate-1`` point bits per
    output row.  Here both are per *code*, so ``arity`` rows share them and the
    plane traffic per row falls by ``arity`` while the LUT traffic is unchanged.
    That is why a k-tuple body is not more expensive to decode than a scalar one
    despite spending more bits per code -- it spends fewer per weight.

    Two constant-shift facts hold this together, and both are asserted by the
    wrapper rather than assumed:

    - ``base`` is a multiple of ``VEC`` and ``steps % 8 == 0``, so the point
      field of code ``base`` starts on a byte and the ``VEC`` fields split into
      two halves of ``VEC/2 * (rate-1)`` bits, each fitting an int32.
    - the select plane's pad is ``SELECT_PAD`` and ``memory <= pad``, so code
      ``base``'s history begins at the fixed sub-byte offset ``pad - memory``.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    base = pid_n * LANES * VEC + tl.arange(0, LANES) * VEC       # first CODE
    j = tl.arange(0, VEC * arity)
    v = j // arity                        # which code within the lane
    a = j % arity                         # which position within the code
    offs_n = base[:, None] * arity + j[None, :]
    live = offs_n < rows
    live_base = base < steps
    acc = tl.zeros((LANES, VEC * arity), dtype=tl.float32)

    P: tl.constexpr = rate - 1
    POINTS: tl.constexpr = 1 << (rate - 1)
    HB: tl.constexpr = (VEC * (rate - 1)) // 16      # bytes per half-window
    HALF_CODES: tl.constexpr = VEC // 2

    groups = cols // half
    for g in range(pid_k, groups, SPLIT_K):
        scale_byte = tl.load(
            scale_ptr + g * rows + offs_n, mask=live, other=0
        ).to(tl.int32)
        scale = tl.exp2(((scale_byte >> 3) & 0xF).to(tl.float32) - 7.0) * (
            1.0 + (scale_byte & 0x7).to(tl.float32) / 8.0
        )
        for i in tl.static_range(half):
            k = g * half + i
            # Select plane: VEC + memory bits at the constant offset pad-memory.
            p = k.to(tl.int64) * (steps + pad) + base.to(tl.int64) + (pad - memory)
            b0 = tl.load(select_ptr + p // 8, mask=live_base, other=0).to(tl.int32)
            b1 = tl.load(select_ptr + p // 8 + 1, mask=live_base, other=0).to(tl.int32)
            b2 = tl.load(select_ptr + p // 8 + 2, mask=live_base, other=0).to(tl.int32)
            wide = (b0 << 16) | (b1 << 8) | b2
            window = (wide[:, None] >> (23 - pad - v[None, :])) & ((1 << (memory + 1)) - 1)

            # Point plane: VEC * P bits, byte-aligned, read as two int32 halves.
            q = k.to(tl.int64) * steps * P + base.to(tl.int64) * P
            lo = tl.zeros((LANES,), dtype=tl.int32)
            for t in tl.static_range(HB):
                lo = (lo << 8) | tl.load(
                    point_ptr + q // 8 + t, mask=live_base, other=0
                ).to(tl.int32)
            hi = tl.zeros((LANES,), dtype=tl.int32)
            for t in tl.static_range(HB):
                hi = (hi << 8) | tl.load(
                    point_ptr + q // 8 + HB + t, mask=live_base, other=0
                ).to(tl.int32)
            packed = tl.where(v[None, :] < HALF_CODES, lo[:, None], hi[:, None])
            shift = HALF_CODES * P - P * ((v % HALF_CODES) + 1)
            pt = (packed >> shift[None, :]) & (POINTS - 1)

            # Two loads, not one.  ``window * POINTS + pt`` does not depend on
            # ``a``, so the first is per CODE and the arity rows of a code hit
            # the same address; only the second is per row.  The working set
            # falls from a 64 KB fused table to 16 KB shared + 2 KB per unit.
            anchor = tl.load(
                index_ptr + window * POINTS + pt, mask=live, other=0
            ).to(tl.int32)
            value = tl.load(
                value_ptr + anchor * arity + a[None, :], mask=live, other=0.0
            )
            acc += value * scale * tl.load(x_ptr + k)

    tl.atomic_add(out_ptr + offs_n, acc * global_scale, mask=live)


def tessera_gemv_tuple(
    x: torch.Tensor,
    select_plane: torch.Tensor,
    point_plane: torch.Tensor,
    index_lut: torch.Tensor,
    value_table: torch.Tensor,
    e4m3_t: torch.Tensor,
    global_scale: float,
    rows: int,
    cols: int,
    rate: int,
    arity: int,
    memory: int = 6,
    half: int = 16,
    lanes: int = 64,
    vec: int = 8,
    split_k: int = 32,
) -> torch.Tensor:
    """``W @ x`` decoding a k-tuple body in the kernel.

    Takes the lookup in two pieces -- ``index_lut`` from
    ``build_tuple_index_lut``, shared by every unit at this rate, and
    ``value_table`` from ``build_anchor_values``, which is the only part a
    per-unit grid changes.  Passing the fused table instead would work
    arithmetically and cost 32x the resident memory; see
    ``build_tuple_value_lut``.

    See the split-K note on ``tessera_gemv_wide``: the reduction is an atomic
    add and its low bits are order-dependent."""
    steps = rows // arity
    if rows % arity or steps % 8:
        raise GrammarError(
            f"{rows} rows at arity {arity} gives {steps} codes; the column "
            "planes need a multiple of 8 codes to stay byte-aligned"
        )
    if vec != 8:
        raise GrammarError(
            f"vec={vec}: the two-int32-halves split of the point window is "
            "derived for VEC=8. Another width needs the shifts re-derived."
        )
    if (rate - 1) % 2 or (vec * (rate - 1)) % 16:
        raise GrammarError(
            f"rate {rate}: the point window is split into two equal byte "
            "halves, which needs an even (rate-1) and a whole number of bytes"
        )
    if memory > SELECT_PAD:
        raise GrammarError(f"memory {memory} exceeds the select pad {SELECT_PAD}")
    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _tuple_gemv_kernel[(triton.cdiv(steps, lanes * vec), split_k)](
        x.reshape(-1), select_plane, point_plane, index_lut, value_table,
        e4m3_t.reshape(-1), out,
        float(global_scale), rows, steps, cols,
        memory=memory, rate=rate, arity=arity, half=half, pad=SELECT_PAD,
        LANES=lanes, VEC=vec, SPLIT_K=split_k,
    )
    return out


@triton.jit
def _gemm_kernel(
    x_ptr, select_ptr, point_ptr, lut_ptr, scale_ptr, out_ptr,
    global_scale, M, rows, cols,
    memory: tl.constexpr, rate: tl.constexpr, half: tl.constexpr, pad: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BF16: tl.constexpr, GROUP_M: tl.constexpr, WIDE: tl.constexpr,
):
    """``x @ W.T`` for prefill: decode a weight tile, then feed it to ``tl.dot``.

    The GEMV pays one LUT gather per weight for one multiply-accumulate.  Here
    the decoded tile is reused ``BLOCK_M`` times, so the decode cost per useful
    FLOP falls as ``1/M`` -- which is the whole reason a prefill path is worth
    measuring separately rather than extrapolating from batch 1.

    **Why a tile is decodable at all.**  ``ConvCode.step`` is a shift register:
    the state after row ``n`` is the last ``memory`` select bits, so row ``n``
    decodes from bits ``n - memory .. n`` of its own column and nothing else.
    The trellis runs down ``rows`` (output features), so a ``[BLOCK_K,
    BLOCK_N]`` tile needs a six-row halo in N and has **no dependence along K
    whatsoever** -- and K is the reduction axis.  A prefill tile is therefore
    embarrassingly decodable, which is the property whose absence makes a
    sequential-dependence format lose here.

    Bits are read per element rather than by the GEMV's shared-window trick:
    the window trick amortises loads across ``VEC`` rows of ONE column, which
    is a batch-1 optimisation, and at prefill the decode is amortised over
    ``BLOCK_M`` anyway.  Correctness first; the shared window is available if a
    profile says the tile decode is the ceiling.
    """
    # Grouped program order: walk GROUP_M row-blocks before advancing the column,
    # so the tiles live at any instant share operands and hit in L2.  A plain 2D
    # program_id sweeps a whole matrix row before reusing anything.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(rows, BLOCK_N)
    width = GROUP_M * grid_n
    group = pid // width
    first_m = group * GROUP_M
    span = tl.minimum(grid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % width) % span)
    pid_n = (pid % width) // span
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    live_m = offs_m < M
    live_n = offs_n < rows
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    point_mask = (1 << (rate - 1)) - 1
    for k0 in range(0, cols, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        live_k = offs_k < cols
        k = offs_k[:, None]
        n = offs_n[None, :]
        live = live_k[:, None] & live_n[None, :]

        # Select plane: row n's `memory + 1` history bits start at bit b.
        #
        # The index arithmetic is int32 whenever the planes fit, and they nearly
        # always do: a 4096x4096 unit's largest select-plane bit index is
        # ~16.8M against int32's 2.1B headroom.  This is not a micro-optimisation
        # -- it is two 64-bit multiplies and two 64-bit divisions PER WEIGHT in
        # the inner loop of a decode-bound kernel, and the GEMV never paid it
        # because it addresses one column at a time.  `WIDE` restores int64 for
        # units that genuinely need it; the wrapper decides, nothing is assumed.
        kk = k.to(tl.int64) if WIDE else k
        nn = n.to(tl.int64) if WIDE else n
        b = kk * (rows + pad) + (pad - memory) + nn
        byte = b >> 3
        shift = 9 - (b & 7).to(tl.int32)
        s0 = tl.load(select_ptr + byte, mask=live, other=0).to(tl.int32)
        s1 = tl.load(select_ptr + byte + 1, mask=live, other=0).to(tl.int32)
        window = (((s0 << 8) | s1) >> shift) & 0x7F

        # Point plane: `rate - 1` bits, densely packed in row order.
        q = (kk * rows + nn) * (rate - 1)
        qbyte = q >> 3
        qshift = (16 - (rate - 1) - (q & 7)).to(tl.int32)
        p0 = tl.load(point_ptr + qbyte, mask=live, other=0).to(tl.int32)
        p1 = tl.load(point_ptr + qbyte + 1, mask=live, other=0).to(tl.int32)
        pt = (((p0 << 8) | p1) >> qshift) & point_mask

        value = tl.load(lut_ptr + window * (1 << (rate - 1)) + pt, mask=live, other=0.0)

        scale_byte = tl.load(
            scale_ptr + (k // half) * rows + n, mask=live, other=0
        ).to(tl.int32)
        scale = tl.exp2(((scale_byte >> 3) & 0xF).to(tl.float32) - 7.0) * (
            1.0 + (scale_byte & 0x7).to(tl.float32) / 8.0
        )
        w = value * scale                                   # [BLOCK_K, BLOCK_N]

        xt = tl.load(
            x_ptr + offs_m[:, None] * cols + offs_k[None, :],
            mask=live_m[:, None] & live_k[None, :], other=0.0,
        )
        # `tl.dot` on fp32 operands with tf32 disabled compiles to plain FMAs --
        # no tensor cores at all.  That is a correct GEMM and a useless one: it
        # measured this kernel 25x off cuBLAS and made the comparison a statement
        # about the schedule rather than about the format.  BF16 operands with an
        # fp32 accumulator is what a serving runtime actually executes, and it is
        # what the NVFP4 comparator must be given too or the arms are not matched.
        # The fp32 path stays because it is the one that reproduces a decoded
        # column bit-for-bit, which is how correctness is tested.
        if BF16:
            acc += tl.dot(xt.to(tl.bfloat16), w.to(tl.bfloat16), out_dtype=tl.float32)
        else:
            acc += tl.dot(xt, w, allow_tf32=False)

    tl.store(
        out_ptr + offs_m[:, None] * rows + offs_n[None, :], acc * global_scale,
        mask=live_m[:, None] & live_n[None, :],
    )


def tessera_gemm(
    x: torch.Tensor,
    select_plane: torch.Tensor,
    point_plane: torch.Tensor,
    value_lut: torch.Tensor,
    e4m3_t: torch.Tensor,
    global_scale: float,
    rows: int,
    cols: int,
    rate: int = 3,
    memory: int = 6,
    half: int = 16,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    bf16: bool = False,
    group_m: int = 8,
    wide: "bool | None" = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """``x @ W.T`` with ``W`` decoded from its planes inside the mainloop.

    Unlike :func:`tessera_gemv_wide` this reduces inside one program and stores
    rather than atomically adding, so the result is deterministic run to run and
    *may* be cited in a bit-identical claim.
    """
    if x.ndim != 2 or x.shape[1] != cols:
        raise GrammarError(
            f"x is {tuple(x.shape)}; a prefill GEMM needs [M, {cols}] because "
            f"the reduction runs over the {cols} columns the trellis does NOT "
            "run down"
        )
    if rows % 8:
        raise GrammarError(
            f"rows={rows} must be a multiple of 8: the select plane is written "
            f"with a {SELECT_PAD}-bit pad per column and the halo arithmetic "
            "assumes column starts land on byte boundaries"
        )
    M = x.shape[0]
    # Both planes are addressed in bits.  int32 holds the larger of the two up
    # to ~2.1e9 bits; past that the arithmetic must widen or it wraps silently,
    # which is the same class of bug that corrupted the body plane at rate 9.
    if wide is None:
        span = max(cols * (rows + SELECT_PAD), cols * rows * (rate - 1))
        wide = span >= (1 << 31) - 8
    out = torch.empty((M, rows), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(rows, block_n),)
    _gemm_kernel[grid](
        x, select_plane, point_plane, value_lut, e4m3_t.reshape(-1), out,
        float(global_scale), M, rows, cols,
        memory=memory, rate=rate, half=half, pad=SELECT_PAD,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        BF16=bf16, GROUP_M=group_m, WIDE=wide,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
