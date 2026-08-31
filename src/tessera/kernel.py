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

__all__ = ["build_code_lut", "tessera_dequant", "tessera_gemv", "nvfp4_gemv"]


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
    body_bits: torch.Tensor, rate: int = 3, memory: int = 6
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Wire BODY -> (select plane, point plane), column-major, MSB-first.

    The select plane carries ``SELECT_PAD`` zero bits before each column, which
    is what lets a decoder read row 0's history without a boundary test: the pad
    *is* the initial state.
    """
    rows, cols = body_bits.shape
    device = body_bits.device
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
