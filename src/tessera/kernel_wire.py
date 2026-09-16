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


# ---------------------------------------------------------------------------
# Compact packed-to-packed repacks: wire bits -> the kernel plane layouts,
# with no expanded one-byte-per-position intermediate.
#
# Every kernel here reads the *packed* plane through ``_span_of`` (two
# aligned int64 loads per byte) and writes the destination plane's bytes
# directly.  The source bit offsets are affine arithmetic on the output byte's
# index, so a thread owns one output byte and races with nobody; the widest
# window a byte reads is 127 bits, which the host admits by rate (<= 8) and
# refuses above, by name, rather than truncating silently.
# ---------------------------------------------------------------------------


def _plane_u8(data: bytes, device: torch.device) -> torch.Tensor:
    """The wire plane's bytes on the device, or one zero byte for an empty
    plane (a plane with no content has no caller)."""
    if not data:
        return torch.zeros(1, dtype=torch.uint8, device=device)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)


@triton.jit
def _bit_window(words_ptr, byte, rel, mask):
    """Bit ``rel`` (0 = the MSB of ``byte``) of the 128-bit window that starts
    at byte ``byte``, as an int64 0/1.

    Two ``_span_of`` loads cover every geometry the host admits: a compact
    output byte reads at most 127 bits from its first source bit (the host
    refuses a rate above 8, whose span is 119 + 7), so the second load's
    window is the last one needed.  Both shifts are clamped: Triton evaluates
    both arms of the ``where``, and an LLVM shift of 64 or more is poison
    before the select ever runs.
    """
    w0 = _span_of(words_ptr, byte, mask)
    w1 = _span_of(words_ptr, byte + 8, mask)
    shift0 = tl.maximum(63 - rel, 0)
    shift1 = tl.minimum(tl.maximum(127 - rel, 0), 63)
    return tl.where(rel >= 64, (w1 >> shift1) & 1, (w0 >> shift0) & 1)


@triton.jit
def _span2_select_kernel(words, out, cols, groups_per_col, col0, col_bits,
                         pair0, per, out_bytes_per_col, BLOCK: tl.constexpr):
    """Packed span-2 BODY -> the select plane (one bit per super-symbol).

    Destination byte ``(column j, group k)`` carries the select bits of the
    eight super-symbols ``pair0 + 8k .. +8``, MSB-first, exactly as
    ``lane_planes._pack_columns(padded, 1)`` writes them; the pad byte that
    precedes each column's data is the caller's (the start state, or zero).
    """
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < cols * groups_per_col
    j = idx // groups_per_col
    k = idx % groups_per_col
    first = (col0 + j) * col_bits + (pair0 + k * 8) * per
    base = first // 8
    rel = first - base * 8
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for i in tl.static_range(8):
        bit = _bit_window(words, base, rel + i * per, mask)
        acc = acc | (bit.to(tl.int32) << (7 - i))
    tl.store(out + j * out_bytes_per_col + 1 + k, acc.to(tl.uint8), mask=mask)


@triton.jit
def _span2_label_kernel(words, out, cols, groups_per_col, col0, col_bits,
                        pair0, per, label_off, out_bytes_per_col,
                        BLOCK: tl.constexpr):
    """Packed span-2 BODY -> the label plane (two bits per super-symbol).

    Destination byte ``(column j, group k)`` carries the stored labels of the
    four super-symbols ``pair0 + 4k .. +4``, two bits each, MSB-first, exactly
    as ``lane_planes._pack_columns(label, 2)`` writes them.
    """
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < cols * groups_per_col
    j = idx // groups_per_col
    k = idx % groups_per_col
    first = (col0 + j) * col_bits + (pair0 + k * 4) * per + label_off
    base = first // 8
    rel = first - base * 8
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for i in tl.static_range(8):
        pair = i // 2
        which = i % 2
        bit = _bit_window(words, base, rel + pair * per + which, mask)
        acc = acc | (bit.to(tl.int32) << (7 - i))
    tl.store(out + j * out_bytes_per_col + k, acc.to(tl.uint8), mask=mask)


@triton.jit
def _span2_point_kernel(words, out, cols, groups_per_col, col0, col_bits,
                        step0, per, rate, wid, out_bytes_per_col,
                        BLOCK: tl.constexpr):
    """Packed span-2 BODY -> the point plane (``rate - 1`` bits per position).

    Destination byte ``(column j, group k)`` carries eight consecutive
    positions of the column's point stream (``wid`` bits each, MSB-first),
    exactly as ``lane_planes._pack_columns(point, wid)`` writes them.  Even
    steps read the first position's field (after the select bit, offset 1);
    odd steps read the second's (after the label, offset ``rate + 2``).
    """
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < cols * groups_per_col
    j = idx // groups_per_col
    k = idx % groups_per_col
    col_base = (col0 + j) * col_bits
    stream0 = k * 8
    q0 = stream0 // wid
    r0 = stream0 - q0 * wid
    step_a = step0 + q0
    first = (col_base + (step_a // 2) * per
             + tl.where(step_a % 2 == 0, 1, rate + 2) + r0)
    base = first // 8
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for i in tl.static_range(8):
        stream = k * 8 + i
        q = stream // wid
        r = stream - q * wid
        step = step0 + q
        src = (col_base + (step // 2) * per
               + tl.where(step % 2 == 0, 1, rate + 2) + r)
        bit = _bit_window(words, base, src - base * 8, mask)
        acc = acc | (bit.to(tl.int32) << (7 - i))
    tl.store(out + j * out_bytes_per_col + k, acc.to(tl.uint8), mask=mask)


def pack_span2_select_cuda(plane: torch.Tensor, *, cols: int, groups_per_col: int,
                           col0: int, col_bits: int, pair0: int, per: int,
                           device: torch.device) -> torch.Tensor:
    """The select plane of a span-2 unit (see ``_span2_select_kernel``).

    ``col_bits`` is the source BODY bits per column *of the plane being
    read*, ``pair0`` the first super-symbol of the cut, ``cols`` and
    ``groups_per_col`` the cut's shape.  Returns uint8
    ``[cols * groups_per_col + 8]`` -- the destination's own trailing slack.
    """
    words = _plane_words(plane)
    out = torch.zeros(cols * (groups_per_col + 1) + 8, dtype=torch.uint8,
                      device=device)
    total = cols * groups_per_col
    if total:
        with torch.cuda.device(device):
            _span2_select_kernel[(triton.cdiv(total, 256),)](
                words, out, cols, groups_per_col, col0, col_bits, pair0, per,
                groups_per_col + 1, 256)
    return out


def pack_span2_label_cuda(plane: torch.Tensor, *, cols: int, groups_per_col: int,
                          col0: int, col_bits: int, pair0: int, per: int,
                          label_off: int, device: torch.device) -> torch.Tensor:
    """The label plane of a span-2 unit (see ``_span2_label_kernel``)."""
    words = _plane_words(plane)
    out = torch.zeros(cols * groups_per_col, dtype=torch.uint8, device=device)
    total = cols * groups_per_col
    if total:
        with torch.cuda.device(device):
            _span2_label_kernel[(triton.cdiv(total, 256),)](
                words, out, cols, groups_per_col, col0, col_bits, pair0, per,
                label_off, groups_per_col, 256)
    return out


def pack_span2_point_cuda(plane: torch.Tensor, *, cols: int, groups_per_col: int,
                          col0: int, col_bits: int, step0: int, per: int,
                          rate: int, steps_per_col: int,
                          device: torch.device) -> torch.Tensor:
    """The point plane of a span-2 unit (see ``_span2_point_kernel``)."""
    words = _plane_words(plane)
    wid = rate - 1
    out = torch.zeros(cols * (steps_per_col * wid // 8), dtype=torch.uint8,
                      device=device)
    total = cols * groups_per_col
    if total:
        with torch.cuda.device(device):
            _span2_point_kernel[(triton.cdiv(total, 256),)](
                words, out, cols, groups_per_col, col0, col_bits, step0, per,
                rate, wid, steps_per_col * wid // 8, 256)
    return out


@triton.jit
def _window_repack_kernel(words, out, col_starts, perm, row0, rows_local, rate,
                          tile_rows, tile_bytes, chunk_bytes, group_col0,
                          group_byte0, n_cols, bytes_per_col,
                          BLOCK: tl.constexpr):
    """Packed WINDOW body -> the repacked tile-word stream, byte for byte.

    The documented tile-word layout, for **any** integer rate: a column's
    512-code tile is ``512 * rate`` bits -- always ``16 * rate`` words --
    holding the codes MSB-first with the first stream bit as bit 31 of the
    column's first word (the 4-byte-flip spelling the int32 view reads back
    numerically).  A destination byte holds eight consecutive stream bits, so
    the transform is a bit-string copy; the rate enters only through where a
    column's bits start and the ``rate``-bit alignment of a code.  Codes at or
    past ``rows_local`` (the tail of the last tile) are zero.

    ``group_col0`` is the group's first column in the reference's permuted
    order (what ``rep.runs`` records); ``group_byte0`` is where the group's
    bytes start inside one tile row of the concatenation -- the two differ
    after the first group, because chunks of different rates are different
    widths.
    """
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_cols * bytes_per_col
    j = idx // bytes_per_col
    b = idx % bytes_per_col
    col = tl.load(perm + group_col0 + j, mask=mask, other=0)
    start = tl.load(col_starts + col, mask=mask, other=0)
    g = b // chunk_bytes
    within = b % chunk_bytes
    src0 = start + (row0 + g * tile_rows) * rate + within * 8
    base = src0 // 8
    rel = src0 - base * 8
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for i in tl.static_range(8):
        code = g * tile_rows + (within * 8 + i) // rate
        bit = _bit_window(words, base, rel + i, mask & (code < rows_local))
        acc = acc | (bit.to(tl.int32) << (7 - i))
    word = within // 4
    sub = within % 4
    dest = g * tile_bytes + group_byte0 + j * chunk_bytes + word * 4 + (3 - sub)
    tl.store(out + dest, acc.to(tl.uint8), mask=mask)


def window_repack_stream_cuda(plane: torch.Tensor, *, col_starts: torch.Tensor,
                              perm: torch.Tensor, row0: int, rows_local: int,
                              rate: int, group_col0: int, group_byte0: int,
                              n_cols: int, n_tiles: int, chunk_bytes: int,
                              tile_bytes: int, device: torch.device,
                              tile_rows: int = 512) -> torch.Tensor:
    """One rate group's repacked bytes, in the reference's flat order.

    Returns uint8 ``[n_tiles * tile_bytes]``: the group's ``n_cols`` columns,
    each ``n_tiles * chunk_bytes`` bytes of stream, per tile -- the operand of
    the int32 view ``Repacked.words`` is.
    """
    words = _plane_words(plane)
    out = torch.zeros(n_tiles * tile_bytes, dtype=torch.uint8, device=device)
    bytes_per_col = n_tiles * chunk_bytes
    total = n_cols * bytes_per_col
    if total:
        with torch.cuda.device(device):
            _window_repack_kernel[(triton.cdiv(total, 256),)](
                words, out, col_starts, perm, row0, rows_local, rate,
                tile_rows, tile_bytes, chunk_bytes, group_col0, group_byte0,
                n_cols, bytes_per_col, 256)
    return out
