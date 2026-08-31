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
