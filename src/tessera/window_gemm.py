"""The window body's prefill GEMM: the ~4 bits/weight wire read directly,
decoded into registers, and fed to a hardware ``tl.dot``.

WHAT IT SERVES.  ``WindowGemvUnit`` -- the value family, whose table holds
bf16 values and whose per-row fp32 ``scale`` is applied once on the
accumulated output (``kernel_window_gemv.decode_values``).  The M <= 8 GEMV
remains available where its history contract holds; this lane is the direct
packed GEMM for every M, including the tails (0, 1, 8, 9, 15, 16, 17, 32,
128): rows past M are masked, so no multiple-of-16 requirement is imposed on
the caller.  The weight tile is decoded in the mainloop, kept in registers,
and consumed by ``tl.dot`` (bf16 operands, fp32 accumulate) -- no decoded
weight tensor is ever written to global memory.

THE DECODE.  ``reference_states`` defines the state of a column at position
``q`` as the **last ``L`` bits of the MSB-first code stream that end at
``q = (row + 1) * rate``**, zero-padded before the stream starts.  So no
sequential walk is needed: the state of row ``n`` is a windowed bit gather,
and the whole ``[BLOCK_K, BLOCK_N]`` tile is read straight from the repacked
words:

* a column's chunk in tile ``g`` starts at ``g * tile_words + word0 +
  k * 16 * rate`` (the item table's own arithmetic);
* a field of ``length`` bits ending at ``q`` starts at bit ``q - L``, which
  crosses into the previous tile for the first few rows of ``g > 0`` -- there
  the word before the chunk is the *previous tile's* last word of the same run
  (``-tile_words + 16 * rate - 1``), exactly the lookback
  ``csrc/window_gemv.cu`` performs for lane 0.  The following word is the
  current chunk's first word, not the memory-neighbour;
* the padding before position 0 of the first tile is supplied as zero for a
  full unit, or by ``initial_state`` for a tensor-parallel row cut.

TP ROW CUTS.  ``initial_state`` is the window state immediately before local
row 0 -- int32 ``[cols]`` in ORIGINAL column order, re-indexed here by
``rep.perm``.  With it, every window is full ``L`` bits from row 0 and the
first rows' high bits come from the carried history instead of a zero pad.
A unit that carries no history is a zero-start unit by definition; a cut that
arrives without one is the loader's refusal to make, not a silent zero here
(the bundle's cut metadata is the loader's to enforce).

RATE RUNS.  ``rep.runs`` fixes the layout: rate runs are contiguous in
permuted column order, so the K loop walks runs and cuts each into ``BLOCK_K``
segments.  A short segment is masked, never reordered.  ``rep.perm`` maps the
permuted columns back to ``x``'s original ones; ``x`` is permuted once per
call (an activation-sized copy, never a weight materialisation).

WHAT IT REFUSES.  The E4M3 family (FP8 GEMM is a later milestone; the
materialised ``torch._scaled_mm`` path still serves it), a non-bf16 or
wrong-width ``x``, an ``initial_state`` outside the window, and block sizes
that would let one N block straddle a 512-row tile boundary.

WHAT IS NOT HERE.  Grouped expert stacking is a later milestone; this file is
new and edits no shared module.  The loader owns the bundle/dataclass
extension; compute consumes it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .errors import GrammarError
from .kernel_window_gemv import TILE_ROWS, WindowGemvUnit

__all__ = ["window_gemm", "MIN_BLOCK"]

#: ``tl.dot`` needs a 16-row minimum operand; any M is served by masking.
MIN_BLOCK = 16


@triton.jit
def _window_gemm_kernel(
    words_ptr, table_ptr, x_ptr, out_ptr, scale_ptr, runs_ptr, init_ptr,
    n_runs, tile_words, total_words, M, rows, cols,
    L: tl.constexpr, TILE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HAS_INIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n0 = pid_n * BN
    g = n0 // TILE                     # BN divides TILE: one block is one tile
    t = n0 - g * TILE                  # local first row inside the tile

    offs_n = n0 + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    live_n = offs_n < rows
    live_m = offs_m < M

    rows_v = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for r in range(n_runs):
        rate = tl.load(runs_ptr + r * 4 + 0)
        col0 = tl.load(runs_ptr + r * 4 + 1)
        ncols = tl.load(runs_ptr + r * 4 + 2)
        word0 = tl.load(runs_ptr + r * 4 + 3)
        CHUNK = 16 * rate              # words per column chunk (512 rows)
        tile_base = g * tile_words + word0

        for c0 in range(0, ncols, BK):
            offs_k = c0 + tl.arange(0, BK)
            live_k = offs_k < ncols
            live_k2 = live_k[:, None]
            kglob = col0 + offs_k                        # permuted column index [BK]
            base = tile_base + offs_k * CHUNK            # [BK]

            qq = (t + 1 + rows_v) * rate                 # [BN], ends of the windows
            if HAS_INIT:
                # history above row 0 keeps every window full
                length = tl.full((BN,), L, tl.int32)
            else:
                # the first tile's opening is zero-padded; later tiles are not
                length = tl.minimum(qq + g * TILE * rate, L)
            qb = qq - L
            neg = qb < 0
            # stream word holding the first field bit, relative to `base`:
            # -1 means the previous tile's last word of the same run, which is
            # the 32 bits immediately before this tile's first word
            wi = tl.where(neg, -1, qb >> 5)
            d1 = wi + 1                                  # current chunk's first word when neg
            # combined bit of the field's last bit (bit 64-b.L of the pair)
            shift = 64 - qq + 32 * wi

            prev_off = -tile_words + CHUNK - 1
            idx_prev = base[:, None] + prev_off
            idx_norm = base[:, None] + wi[None, :]
            idx1 = base[:, None] + d1[None, :]
            live_prev = live_k2 & (g > 0) & (idx_prev >= 0) & (idx_prev < total_words)
            live_norm = live_k2 & (idx_norm >= 0) & (idx_norm < total_words)
            live1 = live_k2 & (idx1 >= 0) & (idx1 < total_words)
            w0_prev = tl.load(words_ptr + idx_prev, mask=live_prev, other=0).to(tl.int64) & 0xFFFFFFFF
            w0_norm = tl.load(words_ptr + idx_norm, mask=live_norm, other=0).to(tl.int64) & 0xFFFFFFFF
            if HAS_INIT:
                w0_init = tl.load(init_ptr + kglob, mask=live_k, other=0).to(tl.int64) & 0xFFFFFFFF
                w0 = tl.where(neg[None, :] & (g == 0), w0_init[:, None],
                              tl.where(neg[None, :], w0_prev, w0_norm))
            else:
                w0 = tl.where(neg[None, :], w0_prev, w0_norm)
            w1 = tl.load(words_ptr + idx1, mask=live1, other=0).to(tl.int64) & 0xFFFFFFFF

            combined = (w0 << 32) | w1
            state = (combined >> shift[None, :]) & ((1 << length) - 1)[None, :]

            val = tl.load(table_ptr + state, mask=live_k2, other=0.0)

            xk = tl.load(
                x_ptr + offs_m[:, None] * cols + kglob[None, :],
                mask=live_m[:, None] & live_k[None, :], other=0.0
            )
            acc += tl.dot(xk, val, out_dtype=tl.float32)

    scale = tl.load(scale_ptr + offs_n, mask=live_n, other=0.0)
    y = acc * scale[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * rows + offs_n[None, :],
        y.to(tl.bfloat16), mask=live_m[:, None] & live_n[None, :],
    )


def window_gemm(
    unit: WindowGemvUnit,
    x: torch.Tensor,
    *,
    initial_state: "torch.Tensor | None" = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    out: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """``x [M, K] bf16 -> [M, rows] bf16``: the wire read directly, prefill.

    Any M is served (tails masked); ``initial_state`` is the int32 ``[cols]``
    window state before local row 0, original column order, for a TP row cut.
    The per-row fp32 ``unit.scale`` is applied once on the accumulated output,
    the same epilogue ``decode_values`` documents.  ``out`` is a caller-owned
    bf16 buffer, a bench instrument only.
    """
    if unit.family != "value":
        raise GrammarError(
            "window_gemm serves the value (BF16) family; the E4M3 (FP8) family "
            "has no native GEMM in this milestone -- the materialised FP8 path "
            "(torch._scaled_mm) serves it until that lane lands"
        )
    if unit.codes_of_state is not None:
        raise GrammarError("window_gemm serves the value family; the unit carries grid codes")
    if x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise GrammarError("x must be a CUDA bf16 [M, K] tensor")
    if x.shape[1] != unit.cols:
        raise GrammarError(f"x has {x.shape[1]} features, the unit {unit.cols} columns")
    m = int(x.shape[0])
    for name, v in (("block_m", block_m), ("block_n", block_n), ("block_k", block_k)):
        if v < MIN_BLOCK or v & (v - 1):
            raise GrammarError(f"{name}={v} must be a power of two >= {MIN_BLOCK}")
    if TILE_ROWS % block_n:
        raise GrammarError(
            f"block_n={block_n} does not divide the {TILE_ROWS}-row wire tile; "
            "one N block must stay inside one tile (its lookback would be wrong)"
        )

    device = unit.rep.words.device
    if x.device != device:
        raise GrammarError(f"x is on {x.device}, the unit's words on {device}")

    table = unit.table
    if table.dtype == torch.float32:
        rounded = table.to(torch.bfloat16)
        if not torch.equal(rounded.float(), table):
            raise GrammarError(
                "the value family's table must be exactly representable in bf16; "
                "an fp32 table with rounded bits would silently change the values"
            )
        table = rounded
    elif table.dtype != torch.bfloat16:
        raise GrammarError(f"window_gemm reads a bf16 value table; got {table.dtype}")
    table = table.contiguous()
    scale = unit.scale
    if scale.numel() != unit.rows:
        raise GrammarError(f"the unit's scale has {scale.numel()} entries for {unit.rows} rows")

    if initial_state is None:
        initial_state = getattr(unit, "initial_state", None)
    if initial_state is None:
        initial_state = getattr(unit.rep, "initial_state", None)
    limit = 1 << int(unit.window_bits)
    if initial_state is None:
        init_perm = torch.zeros(unit.cols, dtype=torch.int32, device=device)
        has_init = False
    else:
        if initial_state.numel() != unit.cols:
            raise GrammarError(
                f"initial_state has {initial_state.numel()} columns, the unit {unit.cols}"
            )
        bad = (initial_state < 0) | (initial_state >= limit)
        if bool(bad.any()):
            raise GrammarError(
                f"initial_state must hold {unit.window_bits}-bit window states "
                f"in [0, {limit})"
            )
        init_perm = initial_state.to(device=device, dtype=torch.int32)
        init_perm = init_perm.index_select(0, unit.rep.perm.long()).contiguous()
        has_init = True

    x_perm = x.index_select(1, unit.rep.perm.long()).contiguous()
    if m == 0:
        return torch.empty(0, unit.rows, dtype=torch.bfloat16, device=device)
    y = torch.empty(m, unit.rows, dtype=torch.bfloat16, device=device) if out is None else out
    if y.shape != (m, unit.rows) or y.dtype != torch.bfloat16 or y.device != device:
        raise GrammarError(f"out must be a bf16 [{m}, {unit.rows}] tensor on {device}")

    grid = (triton.cdiv(unit.rows, block_n), triton.cdiv(m, block_m))
    _window_gemm_kernel[grid](
        unit.rep.words, table, x_perm, y, scale, unit.rep.runs, init_perm,
        int(unit.rep.runs.shape[0]), int(unit.rep.tile_words), int(unit.rep.words.numel()),
        m, unit.rows, unit.cols,
        L=int(unit.window_bits), TILE=TILE_ROWS,
        BM=block_m, BN=block_n, BK=block_k, HAS_INIT=has_init,
        num_warps=8,
    )
    return y
