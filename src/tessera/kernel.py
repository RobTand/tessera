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
from .grammar import require_column_groups as _require_column_groups
from .manifest import WINDOW_BITS_MAX, RotationState
from .trellis import ConvCode, SUBSET_COUNT
from .lane_planes import (  # noqa: F401 -- the packers live there, Triton-free
    SELECT_PAD, _pack_columns, build_anchor_values, build_history_lut, build_span2_luts,
    build_subset_values, build_window_values, lut_scale_table, pack_kernel_planes,
    pack_scale_nibbles, pack_unit_for_kernel, pack_window_planes, _pack_window_unit,
)

__all__ = ["build_code_lut", "build_tuple_index_lut", "build_anchor_values",
           "build_window_values", "pack_window_planes", "tessera_dequant",
           "tessera_gemv", "tessera_gemv_window", "nvfp4_gemv"]


# ---------------------------------------------------------------------------
# Launch-shape guards
#
# Every kernel below derives its addressing from two divisibilities the plane
# packers already enforce (``lane_planes.pack_kernel_planes`` refuses
# ``rows % 8``; ``pack_scale_nibbles`` cannot reshape a partial column group),
# and which the wrappers used to assume rather than state.  An artifact can
# therefore never carry these shapes -- but a hand-built call could, and the
# consequence is not an error: the GEMVs simply stop at ``(cols // half) *
# half`` columns and the rest never enter the dot product, while the GEMM
# reads scale group ``cols // half`` of a ``cols // half``-group plane.  A
# wrong weight is not an exception, so the shape is checked, not assumed --
# the same discipline ``tessera_gemv_window``'s reach check already applies.
# ---------------------------------------------------------------------------


def _require_byte_aligned_rows(rows: int) -> None:
    """``rows`` must byte-align a column plane.

    The select plane is written with a ``SELECT_PAD``-bit pad per column and
    the point plane packs ``rows * (rate - 1)`` bits per column, so a column
    starts on a byte -- and the constant sub-byte shifts every sliced/wide
    kernel is derived with are that byte offset.  With a remainder the shift
    varies with the column and the decoded weights are silently wrong.
    """
    if rows % 8:
        raise GrammarError(
            f"rows={rows} must be a multiple of 8: the select plane is written "
            f"with a {SELECT_PAD}-bit pad per column and the constant-shift "
            "arithmetic assumes column starts land on byte boundaries"
        )


def _require_history_fits_the_pad(memory: int) -> None:
    """``memory`` history bits have to come out of the select plane's pad.

    Every plane-reading kernel here takes a column's history as the window
    ending at the current row and starting ``memory`` bits earlier, so row
    0's window reaches ``memory`` bits into the ``SELECT_PAD`` zero bits the
    packer writes ahead of each column -- and a deeper code would read the
    previous column's last rows as its own initial state.

    It is the width bound of the *reads* too, which is why one function
    states it for all of them.  The sliced and prefill kernels take
    ``memory + 1`` bits out of a 16-bit window whose first bit lands up to
    seven bits into a byte, so ``memory + 1 + 7 <= 16``; the wide and tuple
    kernels take them out of a 24-bit window at the constant offset
    ``SELECT_PAD - memory``, which needs the same bound from the other side.
    Both come to ``SELECT_PAD``.
    """
    if memory > SELECT_PAD:
        raise GrammarError(f"memory {memory} exceeds the select pad {SELECT_PAD}")


def _require_point_window(rate: int, vec: int) -> None:
    """A lane's point field, as the two k-tuple kernels actually read it.

    Both read a lane's point bits as **two int32 halves**, ``vec // 2`` codes
    of ``rate - 1`` bits each, accumulated a byte at a time.  Two conditions
    follow from that extraction and neither is a preference:

    - a half is a whole number of bytes, so ``rate - 1`` is even and
      ``vec * (rate - 1)`` a multiple of 16;
    - a half *fits* an int32, so ``(vec // 2) * (rate - 1) <= 32``.  Past
      that the byte-at-a-time accumulation shifts the earliest bits out of
      the register: at rate 11 a half holds 40 bits, the first eight are
      gone, and half the codes in every lane decode to a different point
      with nothing to say so.  ``tuple_grid(lloyd_max_grid(64), 2)`` is a
      supported grid whose cap is exactly that rate.
    """
    if (rate - 1) % 2 or (vec * (rate - 1)) % 16:
        raise GrammarError(
            f"rate {rate}: the point window is split into two equal byte "
            "halves, which needs an even (rate-1) and a whole number of bytes"
        )
    bits = (vec // 2) * (rate - 1)
    if bits > 32:
        raise GrammarError(
            f"rate {rate}: a lane's {vec // 2} codes carry {bits} point bits "
            f"per half-window, which the kernel accumulates in an int32 -- "
            f"{bits - 32} bits would be shifted out before anything read them. "
            "Narrow the rate, or widen the accumulator deliberately"
        )


def _dense_activation(x: torch.Tensor, cols: int) -> torch.Tensor:
    """``x`` as every GEMV in this module actually addresses it.

    A kernel reads ``x_ptr + k``: a *storage* offset, stride 1, off whatever
    pointer the launch was handed.  A view with another stride keeps that
    stride through ``reshape(-1)`` -- reshaping a tensor to the shape it
    already has returns the tensor itself -- so ``backing[::2]`` arrives with
    the expected ``[cols]`` shape and the kernel reads ``backing[k]`` where
    the caller wrote ``backing[2k]``.  The dot product is then over different
    numbers, and nothing about the launch says so.

    The boundary therefore normalises, which is what the serving window lanes
    (``kernel_window``, ``kernel_window_gemv``) already do to their
    activations, and refuses a length that is not the reduction's: every
    inner load of ``x`` here is *unmasked* in ``k``, so a short activation is
    an out-of-bounds read and not a short dot product.
    """
    flat = x.reshape(-1)
    if flat.numel() != cols:
        raise GrammarError(
            f"the activation holds {flat.numel()} elements for a reduction over "
            f"{cols} columns: the GEMVs read x[k] for every column and mask "
            "nothing in k, so a shorter one reads past its own storage"
        )
    return flat.contiguous()


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

    # A split ends where the *split* ends, not where the matrix does.  ``span``
    # is a column count, not a whole number of ``BLOCK_K`` tiles, so the last
    # tile of a split generally overhangs it -- 32 columns of overhang at the
    # wrappers' own defaults (cols=512, SPLIT_K=16, BLOCK_K=64).  Masked
    # against ``cols`` alone, that overhang is columns the next program also
    # accumulates, and every one of them enters the dot product twice.
    span = tl.cdiv(cols, SPLIT_K)
    start = pid_k * span
    stop = tl.minimum(start + span, cols)
    for k0 in range(start, stop, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        live = (offs_k[:, None] < stop) & (offs_n[None, :] < rows)
        nibble = _decode_tile(
            body_ptr, lut_ptr, offs_n, offs_k, live, rows, col_stride_bits, memory, rate
        )
        value = tl.load(value_ptr + nibble, mask=live, other=0.0)
        weight = value * _apply_scale_kn(scale_ptr, offs_n, offs_k, live, cols, half)
        xs = tl.load(x_ptr + offs_k, mask=offs_k < stop, other=0.0)
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

    # The same split-end mask as ``_gemv_kernel``: the comparator partitions K
    # the same way, so it double-counts the same columns, and a comparator that
    # reports a doubled dot product as NVFP4's answer corrupts a measurement.
    span = tl.cdiv(cols, SPLIT_K)
    start = pid_k * span
    stop = tl.minimum(start + span, cols)
    for k0 in range(start, stop, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        live = (offs_n[:, None] < rows) & (offs_k[None, :] < stop)
        flat = offs_n[:, None].to(tl.int64) * cols + offs_k[None, :].to(tl.int64)
        byte = tl.load(packed_ptr + flat // 2, mask=live, other=0).to(tl.int32)
        nibble = tl.where(flat % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
        value = tl.load(value_ptr + nibble, mask=live, other=0.0)
        weight = value * _apply_scale(scale_ptr, offs_n, offs_k, live, cols, half)
        xs = tl.load(x_ptr + offs_k, mask=offs_k < stop, other=0.0)
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
        _dense_activation(x, cols), body, lut, e4m3.reshape(-1),
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
        _dense_activation(x, cols), packed.reshape(-1), e4m3.reshape(-1),
        e2m1_value_table(x.device).float(), out,
        float(global_scale), rows, cols,
        half=half, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
    )
    return out












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
            # ``memory + 1`` adjacent bits of the select plane: rows
            # n-memory..n.  The window is that wide because the history is
            # that deep -- seven bits at the shipping memory=6 -- so both the
            # shift and the mask are derived from ``memory`` and not written
            # for one code.  Its last bit sits at offset ``(p % 8) + memory``
            # of a 16-bit big-endian read, hence position 15 - that.
            p = k.to(tl.int64) * (rows + pad) + offs_n.to(tl.int64) + (pad - memory)
            lo = tl.load(select_ptr + p // 8, mask=live_n, other=0).to(tl.int32)
            hi = tl.load(select_ptr + p // 8 + 1, mask=live_n, other=0).to(tl.int32)
            window = (((lo << 8) | hi) >> (15 - memory - (p % 8).to(tl.int32))) & (
                (1 << (memory + 1)) - 1
            )

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

    _require_byte_aligned_rows(rows)
    _require_history_fits_the_pad(memory)
    _require_column_groups(cols, half)
    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _sliced_gemv_kernel[(triton.cdiv(rows, block_n), split_k)](
        _dense_activation(x, cols), select_plane, point_plane, lut, e4m3_t.reshape(-1),
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
    """``W @ x`` over column-major NVFP4 nibbles.  The controlled comparator.

    Guarded like the arm it is compared against: a dropped column here does
    not corrupt a weight, it corrupts a *measurement*, which is worse -- the
    comparator would simply report a smaller dot product as NVFP4's answer.
    """
    from .encode import e2m1_value_table

    _require_column_groups(cols, half)
    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _nvfp4_sliced_gemv_kernel[(triton.cdiv(rows, block_n), split_k)](
        _dense_activation(x, cols), packed_t.reshape(-1), e4m3_t.reshape(-1),
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
            # `memory + 1` history bits end at bit offset (p%8) + v + memory, and
            # p%8 is the constant `pad - memory`, so the shift is 23 - pad - v
            # whatever the memory -- but the mask is the window's own width.
            window = (wide[:, None] >> (23 - pad - vec[None, :])) & (
                (1 << (memory + 1)) - 1
            )

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
    # The other half of the same sentence, which used to live only in that
    # message: the derivation needs ``rows % 8 == 0`` too.
    _require_byte_aligned_rows(rows)
    _require_history_fits_the_pad(memory)
    _require_column_groups(cols, half)
    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _wide_gemv_kernel[(triton.cdiv(rows, lanes * vec), split_k)](
        _dense_activation(x, cols), select_plane, point_plane, value_lut, e4m3_t.reshape(-1), out,
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






@triton.jit
def _tuple_gemv_span2_kernel(
    x_ptr, select_ptr, label_ptr, point_ptr, nib_ptr, table_ptr,
    label_lut_ptr, value_ptr, out_ptr,
    global_scale, rows, steps, cols,
    memory: tl.constexpr, rate: tl.constexpr, arity: tl.constexpr,
    half: tl.constexpr, pad: tl.constexpr,
    LANES: tl.constexpr, VEC: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """GEMV over a span-2 k-tuple body with a LUT scale plane.

    The span-1 tuple kernel reads one select bit and ``rate-1`` point bits
    per code, and one E4M3 scale byte per output row per column group.  Here:

    - the select plane holds one bit per PAIR of codes, so a lane of ``VEC``
      codes reads ``VEC/2 + memory`` bits.  The pair index of a lane is
      ``base/2``, a multiple of ``VEC/2`` but not of 8, so the sub-byte offset
      of the window is not a constant: it is computed per lane (``pm``).
    - the label plane holds two bits per pair, MSB-first: a lane's four
      pairs are one byte, byte-aligned because ``base`` and ``steps`` are
      multiples of 8.
    - the point plane is the span-1 point plane, read exactly as before.
    - a pair's super-label comes from the window (``label_lut``); position 1
      takes the stored label, position 0 the super-label minus it mod 4.
      ``value_ptr`` is in SUBSET order (``build_subset_values``), so the
      value of ``(label, point)`` is at ``(label * POINTS + point) * arity``
      with no table between: the dependent-load depth per code is the
      span-1 kernel's -- window bytes, one small table, the value.
    - the scale is a nibble per (row, group) into a 16-entry fp32 table --
      the LUT plane at its wire size, not materialised to bytes.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    base = pid_n * LANES * VEC + tl.arange(0, LANES) * VEC       # first CODE
    j = tl.arange(0, VEC * arity)
    v = j // arity                        # which code within the lane
    a = j % arity                         # which position within the code
    sv = v // 2                           # which pair within the lane
    parity = v % 2                        # which position within the pair
    offs_n = base[:, None] * arity + j[None, :]
    live = offs_n < rows
    live_base = base < steps
    acc = tl.zeros((LANES, VEC * arity), dtype=tl.float32)

    P: tl.constexpr = rate - 1
    POINTS: tl.constexpr = 1 << (rate - 1)
    HB: tl.constexpr = (VEC * (rate - 1)) // 16      # bytes per half-window
    HALF_CODES: tl.constexpr = VEC // 2
    MASK: tl.constexpr = (1 << (memory + 1)) - 1

    pairs = steps // 2
    pbase = base // 2                                  # first PAIR of the lane
    groups = cols // half
    for g in range(pid_k, groups, SPLIT_K):
        # LUT scale plane: [groups, rows] nibbles, even row high.
        nidx = g.to(tl.int64) * rows + offs_n.to(tl.int64)
        nb = tl.load(nib_ptr + nidx // 2, mask=live, other=0).to(tl.int32)
        nib = (nb >> (4 * (1 - (offs_n & 1)))) & 0xF
        scale = tl.load(table_ptr + nib, mask=live, other=0.0)
        for i in tl.static_range(half):
            k = g * half + i
            # Select plane: one bit per pair; the window of pair ``pbase + sv``
            # is the memory+1 bits ending at that pair's bit.
            p = k.to(tl.int64) * (pairs + pad) + pbase.to(tl.int64) + (pad - memory)
            pm = (p % 8).to(tl.int32)
            b0 = tl.load(select_ptr + p // 8, mask=live_base, other=0).to(tl.int32)
            b1 = tl.load(select_ptr + p // 8 + 1, mask=live_base, other=0).to(tl.int32)
            b2 = tl.load(select_ptr + p // 8 + 2, mask=live_base, other=0).to(tl.int32)
            wide = (b0 << 16) | (b1 << 8) | b2
            window = (wide[:, None] >> (23 - pm[:, None] - memory - sv[None, :])) & MASK

            # Label plane: two bits per pair, the lane's four pairs in one byte.
            q = k.to(tl.int64) * steps + base.to(tl.int64)
            lb = tl.load(label_ptr + q // 8, mask=live_base, other=0).to(tl.int32)
            stored = (lb[:, None] >> (6 - 2 * sv[None, :])) & 3

            # Point plane: VEC * P bits, byte-aligned, read as two int32 halves.
            r = k.to(tl.int64) * steps * P + base.to(tl.int64) * P
            lo = tl.zeros((LANES,), dtype=tl.int32)
            for t in tl.static_range(HB):
                lo = (lo << 8) | tl.load(
                    point_ptr + r // 8 + t, mask=live_base, other=0
                ).to(tl.int32)
            hi = tl.zeros((LANES,), dtype=tl.int32)
            for t in tl.static_range(HB):
                hi = (hi << 8) | tl.load(
                    point_ptr + r // 8 + HB + t, mask=live_base, other=0
                ).to(tl.int32)
            packed = tl.where(v[None, :] < HALF_CODES, lo[:, None], hi[:, None])
            shift = HALF_CODES * P - P * ((v % HALF_CODES) + 1)
            pt = (packed >> shift[None, :]) & (POINTS - 1)

            ell = tl.load(label_lut_ptr + window, mask=live, other=0)
            lab = tl.where(parity[None, :] == 0, (ell - stored) & 3, stored)
            value = tl.load(
                value_ptr + (lab * POINTS + pt) * arity + a[None, :], mask=live, other=0.0
            )
            acc += value * scale * tl.load(x_ptr + k)

    tl.atomic_add(out_ptr + offs_n, acc * global_scale, mask=live)


def tessera_gemv_tuple_span2(
    x: torch.Tensor,
    select_plane: torch.Tensor,
    label_plane: torch.Tensor,
    point_plane: torch.Tensor,
    scale_nibbles: torch.Tensor,
    scale_table: torch.Tensor,
    label_lut: torch.Tensor,
    subset_values: torch.Tensor,
    global_scale: float,
    rows: int,
    cols: int,
    rate: int,
    arity: int,
    memory: int = 6,
    half: int = 16,
    lanes: int = 64,
    vec: int = 8,
    split_k: int = 128,
) -> torch.Tensor:
    """``W @ x`` decoding a span-2 k-tuple body with a LUT scale plane in the
    kernel.  Planes from ``pack_kernel_planes(span=2)`` and
    ``pack_scale_nibbles``; tables from ``build_span2_luts`` (the label
    half), ``build_subset_values`` (the values in subset order) and
    ``lut_scale_table`` -- or all of them from ``pack_unit_for_kernel``.  Split-K reduces by atomic add; see
    ``tessera_gemv_wide`` for what that means for the low bits.  The default
    launch shape is the measured one (``experiments/results/tessera_kernel_shape_sweep.json``,
    2048x4096 GLM expert): split-K 128 keeps 256 programs in flight, which is
    what a latency-bound decode needs on 48 SMs; the span-1 kernel's default
    of 32 leaves it at 64.

    ``rate`` must be **odd**, as in ``tessera_gemv_tuple``: the point window is
    split into two equal byte halves, so ``rate - 1`` is even and
    ``vec * (rate - 1)`` a whole number of bytes -- and each half must fit the
    int32 it is accumulated in, which caps ``(vec // 2) * (rate - 1)`` at 32.
    Both enforced below by ``_require_point_window``."""
    steps = rows // arity
    if rows % arity or steps % 16:
        raise GrammarError(
            f"{rows} rows at arity {arity} gives {steps} codes; a span-2 body "
            "needs a multiple of 16 codes (8 pairs) per column to stay byte-aligned"
        )
    if vec != 8:
        raise GrammarError(
            f"vec={vec}: the two-int32-halves split of the point window and the "
            "one-byte label read are derived for VEC=8"
        )
    _require_point_window(rate, vec)
    _require_history_fits_the_pad(memory)
    if scale_table.numel() != 16 or scale_table.dtype != torch.float32:
        raise GrammarError("the scale table is sixteen fp32 entries (lut_scale_table)")
    _require_column_groups(cols, half)
    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _tuple_gemv_span2_kernel[(triton.cdiv(steps, lanes * vec), split_k)](
        _dense_activation(x, cols), select_plane, label_plane, point_plane, scale_nibbles,
        scale_table, label_lut, subset_values, out,
        float(global_scale), rows, steps, cols,
        memory=memory, rate=rate, arity=arity, half=half, pad=SELECT_PAD,
        LANES=lanes, VEC=vec, SPLIT_K=split_k,
    )
    return out


# --- the window body (schema minor 2): a shift register, not a trellis -----
#
# ``state_t = ((state_{t-1} << R) | bits_t) mod 2^L`` from ``state_{-1} = 0``
# makes a state *literally* the last ``L`` bits of the column's stream, so the
# kernel needs no replay tables and no halo argument: it reads an L-bit field
# out of the plane and indexes the unit's table.  That is the same property the
# TCQ lane had to prove about ``ConvCode.step``; here it is the definition.






@triton.jit
def _window_gemv_kernel(
    x_ptr, plane_ptr, offset_ptr, rate_ptr, table_ptr, value_ptr,
    scale_ptr, scale_table_ptr, out_ptr,
    global_scale, rows, steps, cols,
    window: tl.constexpr, arity: tl.constexpr, half: tl.constexpr,
    lut_scale: tl.constexpr,
    LANES: tl.constexpr, VEC: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """GEMV over a window body: an L-bit field, the unit's table, a value.

    Per code the dependent-load depth is three -- the plane bytes, the unit's
    ``2^L`` table, the grid's value table -- against the span-2 trellis
    kernel's window bytes, label byte, point bytes, ``label_lut``, value.  It
    reads *fewer* planes than the trellis lane because the window body has
    fewer: no label plane, no separated point plane, no completion plane.

    Two things decide whether that structural advantage survives contact with
    the machine, and the first cut got both wrong -- it was 1.77x the span-2
    kernel at the E2M1x2 cap while reading *fewer* bytes:

    - **The block is ``[LANES, VEC, arity]``, not ``[LANES, VEC * arity]``.**
      A window, a state and a code are per *code*; only the value and the
      scale are per row.  Written flat, the arity rows of a code each redo the
      whole plane read, which at arity 2 doubles every load in the kernel for
      a result that is bit-identical across the axis.
    - **A lane reads its VEC windows out of two eight-byte spans**, not four
      bytes per code.  Consecutive codes are ``R`` bits apart, so ``VEC/2``
      of them span ``L + (VEC/2 - 1) R + 7 <= 64`` bits: one int64 per half
      covers them, and each code's window is a shift of it.  Per lane per
      column that is 16 byte loads instead of ``4 * VEC``.  The wrapper
      checks the inequality rather than assuming it.

    ``rate`` is a *runtime* scalar per column, not a ``constexpr``: a mixed
    schedule is the ordinary case for this body and the alternative -- one
    specialisation per rate, or a padded uniform stride -- would either
    recompile per unit or spend bits the wire does not spend.  The shift
    arithmetic is the same either way; only the multiplication is dynamic.

    ``lut_scale`` picks the plane: a LUT nibble per (row, group) through a
    16-entry fp32 table (the plane at its wire size, 0.25 b/wt), or the S6b
    plane materialised to one E4M3 byte per (row, group) and decoded by field
    arithmetic, which is what ``_apply_scale`` does for the span-1 lane.  Both
    already exist on this lane; neither is a new plane.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    base = pid_n * LANES * VEC + tl.arange(0, LANES) * VEC       # first CODE
    v = tl.arange(0, VEC)                 # which code within the lane
    a = tl.arange(0, arity)               # which position within the code
    codes = base[:, None] + v[None, :]                          # [LANES, VEC]
    hot = codes < steps
    offs_n = codes[:, :, None] * arity + a[None, None, :]       # [LANES, VEC, arity]
    live = offs_n < rows
    acc = tl.zeros((LANES, VEC, arity), dtype=tl.float32)

    MASK: tl.constexpr = (1 << window) - 1
    HALF_CODES: tl.constexpr = VEC // 2
    m = v % HALF_CODES                    # position of a code within its half
    first = base[:, None] + (v // HALF_CODES)[None, :] * HALF_CODES   # half's first code

    groups = cols // half
    for g in range(pid_k, groups, SPLIT_K):
        sidx = g.to(tl.int64) * rows + offs_n.to(tl.int64)
        if lut_scale:
            # LUT scale plane: [groups, rows] nibbles, even row high.
            nb = tl.load(scale_ptr + sidx // 2, mask=live, other=0).to(tl.int32)
            nib = (nb >> (4 * (1 - (offs_n & 1)))) & 0xF
            scale = tl.load(scale_table_ptr + nib, mask=live, other=0.0)
        else:
            # S6b materialised: one E4M3 byte per (row, group), field arithmetic.
            e4m3 = tl.load(scale_ptr + sidx, mask=live, other=0).to(tl.int32)
            scale = tl.exp2(((e4m3 >> 3) & 0xF).to(tl.float32) - 7.0) * (
                1.0 + (e4m3 & 0x7).to(tl.float32) / 8.0
            )
        for i in tl.static_range(half):
            k = g * half + i
            offset = tl.load(offset_ptr + k)
            rate = tl.load(rate_ptr + k).to(tl.int64)
            # The window of code t is the L bits at [offset + (t+1)*R, +L).
            # Read the half's eight bytes once; every code in it is a shift.
            anchor = offset + (first.to(tl.int64) + 1) * rate
            byte = anchor // 8
            span = tl.zeros((LANES, VEC), dtype=tl.int64)
            for t in tl.static_range(8):
                span = (span << 8) | tl.load(
                    plane_ptr + byte + t, mask=hot, other=0
                ).to(tl.int64)
            shift = 64 - (anchor % 8) - m[None, :].to(tl.int64) * rate - window
            # ``span``'s top byte can set the sign bit and ``>>`` is
            # arithmetic, but the sign fill lands at bit positions >= the
            # field width, which the mask removes.  One eight-byte read
            # therefore covers every legal window up to the wrapper's bound.
            state = (span >> shift) & MASK
            grid_code = tl.load(table_ptr + state, mask=hot, other=0).to(tl.int32)
            value = tl.load(
                value_ptr + grid_code[:, :, None] * arity + a[None, None, :],
                mask=live, other=0.0,
            )
            acc += value * scale * tl.load(x_ptr + k)

    tl.atomic_add(out_ptr + offs_n, acc * global_scale, mask=live)


def tessera_gemv_window(
    x: torch.Tensor,
    plane: torch.Tensor,
    offsets: torch.Tensor,
    rates: torch.Tensor,
    table: torch.Tensor,
    values: torch.Tensor,
    scale_plane: torch.Tensor,
    scale_table: "torch.Tensor | None",
    global_scale: float,
    rows: int,
    cols: int,
    window_bits: int,
    arity: int,
    half: int = 16,
    max_rate: "int | None" = None,
    lanes: int = 64,
    vec: int = 8,
    split_k: int = 128,
) -> torch.Tensor:
    """``W @ x`` decoding a window body (schema minor 2) in the kernel.

    Planes from ``pack_window_planes`` and either ``pack_scale_nibbles`` (LUT)
    or ``wire.nvfp4_scale_bytes`` transposed to ``[groups, rows]`` (S6b);
    ``table`` is the unit's own ``2^L`` ALPHABET plane and ``values`` the
    grid's, from ``build_window_values`` -- or all of them from
    ``pack_unit_for_kernel``.  ``scale_table`` is the 16-entry fp32 LUT table
    or ``None`` for the S6b plane.  Split-K reduces by atomic add, so this is
    not a bit-identical-across-runs path; a one-hot probe still returns a
    column exactly because nothing is summed.
    """
    steps = rows // arity
    if rows % arity or rows % 2:
        raise GrammarError(
            f"{rows} rows at arity {arity} does not give whole codes and paired "
            "scale nibbles"
        )
    _require_column_groups(cols, half)
    if table.numel() != 1 << window_bits:
        raise GrammarError(
            f"the window table holds {table.numel()} entries, window_bits "
            f"{window_bits} needs {1 << window_bits}"
        )
    if offsets.numel() != cols or rates.numel() != cols:
        raise GrammarError(
            f"{offsets.numel()} offsets and {rates.numel()} rates for {cols} columns"
        )
    if vec < 2 or vec & (vec - 1):
        raise GrammarError(f"vec={vec}: a lane's codes split into two halves")
    # A half's ``vec // 2`` windows are read out of one int64: the last one
    # ends at most ``window + (vec/2 - 1) * R + 7`` bits into it, counting the
    # sub-byte offset of the first.  Checked, not assumed -- the shifts are
    # silently wrong past it, and a wrong weight is not an error.
    #
    # ``max_rate`` is an argument because reading it off the device tensor is
    # a synchronisation, and a synchronisation on a launch-shape check that
    # cannot change between calls cost 2.6 ms against a ~0.1 ms kernel: 25x
    # the thing being measured, invisible to the profiler's kernel row and
    # visible only in ms/call.  ``pack_unit_for_kernel`` carries the value.
    if max_rate is None:
        max_rate = int(rates.max())
    reach = window_bits + (vec // 2 - 1) * int(max_rate) + 7
    if reach > 64:
        raise GrammarError(
            f"a window of {window_bits} bits at rate {int(rates.max())} reaches "
            f"{reach} bits across {vec // 2} codes, past the 64 the kernel reads "
            "in one span; lower vec or narrow the window"
        )
    lut = scale_table is not None
    if lut and (scale_table.numel() != 16 or scale_table.dtype != torch.float32):
        raise GrammarError("the scale table is sixteen fp32 entries (lut_scale_table)")
    # The kernel reads ``scale_table_ptr`` only under ``lut_scale``, but a
    # launch still needs a valid pointer for the argument, so the S6b branch
    # aliases the scale plane there.  Named rather than passed twice inline:
    # a repeated argument reads like a bug and this one is deliberate and
    # provably dead (`_window_gemv_kernel`'s `if lut_scale:` is its only use).
    scale_table_arg = scale_table if lut else scale_plane
    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _window_gemv_kernel[(triton.cdiv(steps, lanes * vec), split_k)](
        _dense_activation(x, cols), plane, offsets, rates, table, values,
        scale_plane, scale_table_arg, out,
        float(global_scale), rows, steps, cols,
        window=window_bits, arity=arity, half=half, lut_scale=lut,
        LANES=lanes, VEC=vec, SPLIT_K=split_k,
    )
    return out






def gemv_from_packed(x: torch.Tensor, packed: dict, **kw) -> torch.Tensor:
    """The lane's GEMV over a ``pack_unit_for_kernel`` result, either body."""
    if packed.get("kind", "span2") == "window":
        out = tessera_gemv_window(
            x, packed["plane"], packed["offsets"], packed["rates"],
            packed["table"], packed["values"], packed["scale_plane"],
            packed["scale_table"], packed["global_scale"], packed["rows"],
            packed["cols"], packed["window_bits"], packed["arity"],
            half=packed["half"], max_rate=packed["max_rate"], **kw,
        )
        row_scale = packed.get("row_scale")
        return out if row_scale is None else out * row_scale
    return tessera_gemv_tuple_span2(
        x, packed["select"], packed["label"], packed["point"], packed["nibbles"],
        packed["table"], packed["label_lut"], packed["values"],
        packed["global_scale"], packed["rows"], packed["cols"], packed["rate"],
        packed["arity"], memory=packed["memory"], half=packed["half"], **kw,
    )


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
    add and its low bits are order-dependent.

    ``rate`` must be **odd** -- the point window is split into two equal byte
    halves, which needs an even ``rate - 1`` and ``vec * (rate - 1)`` a whole
    number of bytes -- and narrow enough that a half fits the int32 it is
    accumulated in, which caps ``(vec // 2) * (rate - 1)`` at 32.  Enforced
    below; stated here because a caller choosing a schedule should not have to
    read the launch to find out."""
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
    _require_point_window(rate, vec)
    _require_history_fits_the_pad(memory)
    _require_column_groups(cols, half)
    out = torch.zeros(rows, dtype=torch.float32, device=x.device)
    _tuple_gemv_kernel[(triton.cdiv(steps, lanes * vec), split_k)](
        _dense_activation(x, cols), select_plane, point_plane, index_lut, value_table,
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
        # ``memory + 1`` bits ending at offset ``(b & 7) + memory`` of a
        # 16-bit big-endian read; both the shift and the mask follow from the
        # history's depth rather than from the shipping memory=6.
        shift = 15 - memory - (b & 7).to(tl.int32)
        s0 = tl.load(select_ptr + byte, mask=live, other=0).to(tl.int32)
        s1 = tl.load(select_ptr + byte + 1, mask=live, other=0).to(tl.int32)
        window = (((s0 << 8) | s1) >> shift) & ((1 << (memory + 1)) - 1)

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
    # ``_gemm_kernel`` addresses x as ``offs_m * cols + offs_k``, which is the
    # row-major storage of a [M, cols] tensor and not the strides of the view
    # it was handed: ``batch[:, ::2]`` and a transpose both pass the check
    # above and address a different matrix.  Normalised here for the same
    # reason ``_dense_activation`` normalises the GEMVs' -- the addressing is
    # the contract, so the boundary makes it true.
    x = x.contiguous()
    _require_byte_aligned_rows(rows)
    _require_history_fits_the_pad(memory)
    _require_column_groups(cols, half)
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
