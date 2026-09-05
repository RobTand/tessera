"""The kernel lane's plane packers -- **Triton-free**.

Everything a decoder needs from an encoded unit, and nothing that runs on a
GPU: the wire's BODY permuted into column-major select/label/point planes, the
LUT scale plane as nibbles, and the per-unit tables the span-2 and window
decoders index.  ``tessera.kernel`` re-exports every name here for its Triton
GEMVs; a serving runtime that must not import Triton (Gridbook) imports this
module directly and hands the planes to its own native decoder.  The two
decoders read the same bytes, so ``pack_unit_for_kernel`` is the one source of
the plane layout for both.
"""
from __future__ import annotations

import torch

from .alphabet import AnchorForest
from .decode import _replay_tables
from .errors import GrammarError
from .manifest import WINDOW_BITS_MAX, RotationState
from .trellis import ConvCode, SUBSET_COUNT

__all__ = [
    "SELECT_PAD", "pack_kernel_planes", "pack_scale_nibbles", "lut_scale_table",
    "build_history_lut", "build_anchor_values", "build_subset_values",
    "build_span2_luts", "pack_window_planes", "build_window_values",
    "pack_unit_for_kernel", "build_subset_nibbles", "lut_scale_bytes",
    "prepare_span2_planes",
]



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
) -> "tuple[torch.Tensor, ...]":
    """Wire BODY -> kernel planes, column-major, MSB-first.

    ``span == 1``: ``(select plane, point plane)``.  The select plane carries
    ``SELECT_PAD`` zero bits before each column, which is what lets a decoder
    read row 0's history without a boundary test: the pad *is* the initial
    state.

    ``span == 2``: ``(select plane, label plane, point plane)``.  One select
    bit per super-symbol (a pair of codes), padded per column exactly as
    above; the stored two-bit label of every odd position; and the point
    plane, which is byte for byte the span-1 point plane because the point
    field is the same width at both positions of a pair.  The select plane
    ends with eight bytes of slack so the kernel's three-byte window read on
    the last pair of the last column stays inside the tensor.  A column of
    pairs must be a multiple of eight pairs (sixteen codes) so that every
    column's planes start on a byte.
    """
    rows, cols = body_bits.shape
    device = body_bits.device
    if span not in (1, 2):
        raise GrammarError(
            f"the kernel lane decodes span-1 and span-2 bodies; this unit is span {span}"
        )
    body = body_bits.to(torch.int32)
    point = body & ((1 << (rate - 1)) - 1)
    point_plane = _pack_columns(point, rate - 1)
    if span == 1:
        if rows % 8:
            raise GrammarError(f"{rows} rows does not byte-align a column plane")
        select = (body >> (rate - 1)) & 1
        padded = torch.zeros(rows + SELECT_PAD, cols, dtype=torch.int32, device=device)
        padded[SELECT_PAD:] = select
        return _pack_columns(padded, 1), point_plane
    if rows % 16:
        raise GrammarError(
            f"{rows} codes is not a multiple of 16; a span-2 column holds one "
            "select bit and one label per pair and needs a byte-aligned column of pairs"
        )
    select = (body[0::2] >> (rate - 1)) & 1                 # [pairs, cols]
    label = (body[1::2] >> (rate - 1)) & (SUBSET_COUNT - 1)  # [pairs, cols]
    padded = torch.zeros(rows // 2 + SELECT_PAD, cols, dtype=torch.int32, device=device)
    padded[SELECT_PAD:] = select
    select_plane = torch.cat([
        _pack_columns(padded, 1), torch.zeros(8, dtype=torch.uint8, device=device)
    ])
    return select_plane, _pack_columns(label, 2), point_plane


def pack_scale_nibbles(scale_refine: torch.Tensor, rows: int, cols: int, half: int = 16) -> torch.Tensor:
    """The LUT scale plane as the kernel reads it: ``[groups, rows]`` nibbles,
    two per byte, the even row in the high nibble.

    A lane's sixteen output rows of one column group are then eight
    consecutive bytes.  This is the plane at its wire size -- a nibble per
    sixteen weights, 0.25 bpp -- where the span-1 kernel reads a materialised
    E4M3 byte per sixteen (0.5 bpp): the bits the LUT plane saved on disk are
    not spent again in memory.
    """
    if rows % 2:
        raise GrammarError(f"{rows} rows does not pair nibbles into bytes")
    groups = cols // half
    nib = scale_refine.reshape(rows, groups).t().contiguous().to(torch.int32)
    if nib.numel() and int(nib.max()) > 0xF:
        raise GrammarError("a LUT scale index wider than a nibble")
    flat = nib.reshape(-1, 2)
    return ((flat[:, 0] << 4) | flat[:, 1]).to(torch.uint8)


def lut_scale_table(scale_lut: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """``[16]`` fp32 -- the LUT plane's E4M3 entries as numbers, zero past the
    table's end.  The unit's global scale stays a scalar on the wrapper, as
    for the span-1 kernel."""
    table = torch.zeros(16, dtype=torch.float32, device=device)
    n = int(scale_lut.numel())
    if n > 16:
        raise GrammarError(f"a LUT scale plane holds at most 16 entries, got {n}")
    table[:n] = scale_lut.to(device).view(torch.float8_e4m3fn).float()
    return table


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


def build_subset_values(
    forest: AnchorForest, code: ConvCode, device: str = "cuda"
) -> torch.Tensor:
    """``(label, point) -> the anchor's ``arity`` values``, at index
    ``(label * points + point) * arity``.  The PER-UNIT half, in subset order.

    ``build_anchor_values`` is indexed by anchor, so a kernel reaches it
    through a ``(window, point) -> anchor`` table.  The four subsets partition
    the anchors (``_replay_tables``), so permuting the value table into
    subset order makes the anchor index arithmetic -- ``label * points +
    point`` -- and the span-2 kernel, which derives the label per pair, needs
    no ``(label, point)`` table at all.  Same 2 KB per unit; the same
    permutation for every unit at a rate.
    """
    subsets, _table_next, _table_sub = _replay_tables(forest, code, device)
    arity = forest.grid.arity
    values = build_anchor_values(forest, device).reshape(-1, arity)
    return values[subsets.reshape(-1).long()].reshape(-1).contiguous()


def build_span2_luts(
    forest: AnchorForest, code: ConvCode, device: str = "cuda"
) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(label_lut[2^(memory+1)] int32: history window -> super-label,
    subset_lut[4 * points] int16: (label, point) -> anchor)``.  SHARED.

    The span-1 ``build_tuple_index_lut`` fuses these two: ``index[window *
    points + point] == subset_lut[label_lut[window] * points + point]`` (a
    test holds them to it).  At span 2 the fusion cannot be done ahead of
    time, because position 0's label is the super-label minus the pair's
    stored label (``trellis.py``, ``decode._replay_span``): the window gives
    the super-label, the label plane gives position 1's label, and position
    0's is derived per pair in the kernel.  Both halves depend on the replay
    tables and the forest's block layout only, never on what a block
    reconstructs to, so every unit at a rate shares them.
    """
    subsets, _table_next, table_sub = _replay_tables(forest, code, device)
    if len(forest.blocks) > (1 << 15):
        raise GrammarError(
            f"{len(forest.blocks)} anchors does not fit the int16 subset table; "
            "widen it deliberately rather than letting it wrap"
        )
    sub_cpu = table_sub.tolist()
    labels: "list[int]" = []
    for window in range(1 << (code.memory + 1)):
        select = window & 1
        history = window >> 1
        state = 0
        for index in range(code.memory):
            state |= ((history >> (code.memory - 1 - index)) & 1) << index
        labels.append(int(sub_cpu[select][state]))
    label_lut = torch.tensor(labels, dtype=torch.int32, device=device)
    subset_lut = subsets.reshape(-1).to(torch.int16).to(device)
    return label_lut, subset_lut


def pack_window_planes(
    body_bits: torch.Tensor,
    rates: "tuple[int, ...]",
    window_bits: int,
    initial_state: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Wire BODY -> ``(plane uint8, column bit offsets int64, rates int32)``.

    Column-major and MSB-first like every other plane here, with two
    differences the window body forces:

    - **``window_bits`` pad bits lead each column**, exactly as ``SELECT_PAD``
      does for the trellis lanes and for the same reason: the pad *is*
      ``state_{-1}``, so position 0's window needs no boundary test.  The
      pad is ``L`` rather than 8 because the window is ``L`` wide and reaches
      ``L - R`` bits behind position 0.  For a whole unit the pad is zero --
      the pinned start.  For a **shard** (``layout.slice_unit``) it is that
      column's stored start state, written as an ``L``-bit MSB-first integer,
      and the kernel needs no change at all: its window read at ``(t + 1) * R``
      then yields ``(init << R | bits_0) mod 2^L`` at ``t = 0``, which is the
      recursion's own first step.  A shard packed with a zero pad would decode
      to plausible wrong weights, so the state is threaded rather than
      dropped.
    - **each column carries its own rate**, so the columns are not one stride
      apart.  A mixed schedule is the normal case for this body (the TCQ
      span-2 lane refuses one), so the offset of every column is a tensor the
      kernel reads rather than a multiplication it does.  Columns start on a
      byte, which is what makes the offset table small enough to be free.

    With the pad, the L-bit window of code ``t`` in column ``c`` begins at bit
    ``offsets[c] + (t + 1) * rates[c]`` and runs ``L`` bits: the ``L`` pad bits
    plus ``t + 1`` positions of ``R`` bits, minus the ``L`` bits of the window
    itself.  Eight bytes of slack at the end keep the kernel's eight-byte span
    read on the last code of the last column inside the tensor.

    The width bound here is the *wire's* (``WINDOW_BITS_MAX``), not the
    kernel's.  How wide a window a launch can actually read depends on ``vec``
    and the schedule's top rate together -- a half-lane's windows share one
    int64 -- so that check lives in ``tessera_gemv_window``, where both are
    known.
    """
    steps, cols = body_bits.shape
    device = body_bits.device
    if len(rates) != cols:
        raise GrammarError(f"{len(rates)} rates for {cols} columns")
    if not 1 <= window_bits <= WINDOW_BITS_MAX:
        raise GrammarError(
            f"window_bits {window_bits} outside 1..{WINDOW_BITS_MAX}, the widest "
            "window the wire carries"
        )
    if max(rates) > window_bits:
        raise GrammarError(
            f"rate {max(rates)} does not fit a {window_bits}-bit window"
        )
    if initial_state is not None:
        if initial_state.numel() != cols:
            raise GrammarError(
                f"the start state holds {initial_state.numel()} words for "
                f"{cols} columns: one per column"
            )
        if int(initial_state.max()) >= (1 << window_bits):
            raise GrammarError(
                f"a start state of {int(initial_state.max())} does not fit a "
                f"{window_bits}-bit window"
            )
    rate_t = torch.tensor(rates, dtype=torch.int32, device=device)
    col_bytes = (window_bits + steps * rate_t.long() + 7) // 8
    starts = torch.zeros(cols + 1, dtype=torch.int64, device=device)
    starts[1:] = torch.cumsum(col_bytes, 0)
    plane = torch.zeros(int(starts[-1]) + 8, dtype=torch.uint8, device=device)
    body = body_bits.to(torch.int32)
    weights = 1 << torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
    for present in sorted(set(rates)):
        which = torch.tensor(
            [c for c, r in enumerate(rates) if r == present],
            dtype=torch.int64, device=device,
        )
        nbytes = int(col_bytes[which[0]])
        bits = torch.zeros(which.numel(), nbytes * 8, dtype=torch.uint8, device=device)
        values = body[:, which]                                   # [steps, m]
        stop = window_bits + steps * present
        for position in range(present):
            bits[:, window_bits + position : stop : present] = (
                (values >> (present - 1 - position)) & 1
            ).t().to(torch.uint8)
        if initial_state is not None:
            start = initial_state.to(device).long()[which]         # [m]
            for position in range(window_bits):
                bits[:, position] = (
                    (start >> (window_bits - 1 - position)) & 1
                ).to(torch.uint8)
        packed = (bits.reshape(-1, 8) * weights).sum(1, dtype=torch.uint8)
        dest = starts[which][:, None] + torch.arange(nbytes, device=device)[None, :]
        plane[dest.reshape(-1)] = packed
    return plane, starts[:cols] * 8, rate_t


def build_window_values(grid, device: str = "cuda") -> torch.Tensor:
    """``code * arity + a -> the code's value``, fp32.  The SHARED half.

    The per-unit half of a window body is its ``2^L`` table of grid codes,
    which rides the ALPHABET plane and is already a resident byte per state.
    What a code *reconstructs to* is the grid's, shared by every unit over it
    -- the same seam ``build_tuple_index_lut`` / ``build_anchor_values`` cut
    for the trellis lane, with the halves the other way round.

    Fusing them instead would give ``2^L * arity`` floats: 512 KB per unit at
    ``L = 16``, arity 2, against 64 KB for the table and a grid table shared
    across the model.  That is the ratio that decided the trellis lane's split
    and it decides this one.
    """
    from .encode import grid_vector_table

    return grid_vector_table(grid, device).reshape(-1).contiguous()


def _window_code_table(codes: torch.Tensor, grid, device) -> torch.Tensor:
    """The unit's ``2^L`` ALPHABET plane, as wide as the grid declares.

    ``PayloadGrid.code_bytes`` is the width of one stored code, derived from
    the grid's own code space rather than declared anywhere: one byte for the
    three narrow grids, two for BF16, whose code *is* the bf16 bit pattern.
    Cast unconditionally to ``uint8`` that second kind loses its high byte --
    0x3f80 (bf16 1.0) becomes 0x80, which is a legal index into a different
    row of the value table, so nothing downstream can notice.  A grid wider
    than two bytes is refused by ``code_bytes`` itself, by name, here at pack
    time rather than by a wrap in someone's kernel.

    The kernel converts what it loads to int32 in any case, so the wide table
    is stored as int32: ``int16`` cannot hold BF16's top half (0xffff reads
    back negative) and Triton has no settled uint16 pointer type.  That is
    four bytes per state where the byte grids spend one, which is the price
    of a code space that does not fit a byte.
    """
    table = codes.to(device)
    if grid.code_bytes == 1:
        return table.to(torch.uint8).contiguous()
    return table.to(torch.int32).contiguous()


def _require_no_post_decode_transforms(unit) -> None:
    """Refuse the three operations no GEMV on this lane applies.

    A released position is overwritten from the RELEASE plane, diagonals are
    a rank-1 factor outside the dot product and a rotation is a basis change
    -- none of which any kernel here reads, on either body.  Accepting one
    serves the transformed quantisation space as if it were
    ``reconstruct_unit(unit) @ x``: a plausible, wrong answer with no error.

    One rule, one home: the window branch stated these first and the TCQ
    branch did not state them at all, which is how a span-2 unit with a
    rotation packed and served silently.
    """
    if unit.release_index.numel():
        raise GrammarError(
            "this unit has released positions, which overwrite decoded codes "
            "from the RELEASE plane; the kernel lane reads no such plane"
        )
    if unit.diagonals is not None:
        raise GrammarError(
            "this unit carries diagonals; undoing them is a rank-1 factor "
            "outside the GEMV, which the kernel lane does not apply"
        )
    if unit.rotation is not RotationState.NONE:
        raise GrammarError(
            f"this unit is rotated ({unit.rotation.name}); undoing the rotation "
            "is a basis change the kernel lane does not apply"
        )


def _pack_window_unit(unit, grid) -> dict:
    """``pack_unit_for_kernel``'s window branch.  See its docstring."""
    from .manifest import ScalePlaneKind
    from .wire import nvfp4_scale_bytes

    _require_no_post_decode_transforms(unit)
    steps, cols = unit.body_bits.shape
    rows = steps * grid.arity
    device = unit.body_bits.device
    plane, offsets, rates = pack_window_planes(
        unit.body_bits, unit.rates, unit.window_bits,
        getattr(unit, "initial_state", None),
    )
    row_scale = None
    if unit.scale_plane is ScalePlaneKind.LUT:
        scale_plane = pack_scale_nibbles(unit.scale_refine, rows, cols, unit.half)
        scale_table = lut_scale_table(unit.scale_lut, device)
        global_scale = float(unit.scale_global)
    elif unit.scale_plane is ScalePlaneKind.CHANNEL:
        # Schema minor 3: one fp16 word per output row times the global.  A
        # row scale is a factor of the row, not of the dot product, so the
        # kernel runs over an identity block plane (E4M3 0x38 is exactly 1.0,
        # global 1.0 -- both multiplications are exact) and the row scale is
        # applied as an epilogue in ``gemv_from_packed``, computed by the
        # same fp32 expression the reader uses (``channel_scale_field``), so
        # a one-hot column decodes to the reader's bytes bit for bit.
        from .scale_channel import channel_scale_field

        if unit.scale_rows is None:
            raise GrammarError("a CHANNEL scale plane needs the unit's row words")
        scale_plane = torch.full(
            (cols // unit.half, rows), 0x38, dtype=torch.uint8, device=device
        )
        scale_table = None
        global_scale = 1.0
        row_scale = channel_scale_field(
            unit.scale_rows.to(device), unit.scale_global, rows, 1
        )[:, 0].contiguous()
    else:
        e4m3, global_scale = nvfp4_scale_bytes(
            unit.scale_base, unit.scale_refine, unit.group, unit.half
        )
        scale_plane = e4m3.reshape(rows, cols // unit.half).t().contiguous()
        scale_table = None
    return {
        "kind": "window",
        "plane": plane, "offsets": offsets, "rates": rates,
        "table": _window_code_table(unit.window_codes, grid, device),
        "values": build_window_values(grid, device),
        "scale_plane": scale_plane, "scale_table": scale_table,
        "global_scale": global_scale, "row_scale": row_scale,
        "rows": rows, "cols": cols, "window_bits": int(unit.window_bits),
        "arity": grid.arity, "half": unit.half, "max_rate": max(unit.rates),
    }


def pack_unit_for_kernel(unit, forest: AnchorForest, code: ConvCode) -> dict:
    """Everything the lane's GEMV needs, from one encoded unit.

    Dispatches on the unit's **body kind**, and the two branches share nothing
    but this function: a TCQ body packs the span-2 trellis planes for
    ``tessera_gemv_tuple_span2``, a WINDOW body (schema minor 2) packs the
    shift-register plane for ``tessera_gemv_window``.  ``forest`` is the
    unit's ``AnchorForest`` under TCQ and may be a bare ``PayloadGrid`` under
    WINDOW, which has no forest; ``code`` is unused by the window branch for
    the same reason.  ``gemv_from_packed`` reads the ``"kind"`` key back.

    Both branches refuse the three post-decode transforms no GEMV on this
    lane applies -- released positions, diagonals, a rotation -- through the
    one function that states them (``_require_no_post_decode_transforms``).
    Beyond that the TCQ branch refuses what the span-2 kernel does not read:
    a mixed-rate schedule (one forest per unit there) and an S6b plane at
    span 2 (that kernel reads the LUT plane's nibbles; the shipping wire is
    span 2 over a LUT plane).  The window branch reads both scale planes and
    any mixed schedule.
    """
    from .manifest import BodyKind, ScalePlaneKind

    _require_no_post_decode_transforms(unit)
    if getattr(unit, "body", BodyKind.TCQ) is BodyKind.WINDOW:
        grid = forest.grid if isinstance(forest, AnchorForest) else forest
        return _pack_window_unit(unit, grid)
    if getattr(unit, "initial_state", None) is not None:
        # The window branch threads the state through its pad; the span-2
        # trellis planes would need the same treatment on the select plane's
        # SELECT_PAD, in the bit order ``build_span2_luts`` reverses, and that
        # is unwritten and untested.  Refusing is the only honest answer: a
        # shard packed against the pinned zero start decodes to plausible
        # wrong weights, silently.
        raise GrammarError(
            "the span-2 kernel lane does not yet take a start state; this unit "
            f"is a shard beginning at row {getattr(unit, 'row_offset', 0)} of "
            "its parent. Decode it through tessera.decode, or serve it from a "
            "whole unit"
        )
    if unit.span != 2:
        raise GrammarError(f"pack_unit_for_kernel is the span-2 path; this unit is span {unit.span}")
    if unit.scale_plane is not ScalePlaneKind.LUT:
        raise GrammarError(
            "the span-2 kernel reads the LUT scale plane; this unit carries an "
            f"{unit.scale_plane.name} plane"
        )
    rates = set(unit.rates)
    if rates != {forest.rate}:
        raise GrammarError(f"unit rates {sorted(rates)} are not the forest's {forest.rate}")
    # Per-code planes are ``steps`` tall; a code covers ``arity`` rows.
    steps, cols = unit.codes.shape
    rows = steps * forest.grid.arity
    device = unit.body_bits.device
    select, label, point = pack_kernel_planes(unit.body_bits, rate=forest.rate, memory=code.memory, span=2)
    label_lut, _subset_lut = build_span2_luts(forest, code, device)
    return {
        "kind": "span2",
        "select": select, "label": label, "point": point,
        "nibbles": pack_scale_nibbles(unit.scale_refine, rows, cols, unit.half),
        "table": lut_scale_table(unit.scale_lut, device),
        "label_lut": label_lut,
        "values": build_subset_values(forest, code, device),
        "global_scale": float(unit.scale_global),
        "rows": rows, "cols": cols, "rate": forest.rate, "arity": forest.grid.arity,
        "memory": code.memory, "half": unit.half,
    }


# --- what a native (non-Triton) decoder needs beyond the GEMV's tables -------

def build_subset_nibbles(forest: AnchorForest, code: ConvCode, device: str = "cuda") -> torch.Tensor:
    """``(label, point, position) -> E2M1 nibble``, uint8, in the same subset
    order as ``build_subset_values``: a decoder that emits the stock NVFP4 tile
    writes codes, not values, and this is the code of every value in that
    table.

    Built from the anchor's CODE (``blocks[anchor][0]``, split into its E2M1
    digits by ``stock.e2m1_nibbles`` -- the one tuple layout), never from its
    value: E2M1 spells zero twice (+0.0 at nibble 0, -0.0 at nibble 8) and a
    value lookup collapses the pair, so every zero anchor came back as nibble
    8 where ``materialize_stock`` writes 0 -- 32 of 512 entries on the
    shipping E2M1x2 forest, numerically inert and still a byte difference from
    the attested stock tile (found by the CUDA decoder's byte-identity test).
    """
    from .stock import e2m1_nibbles

    subsets, _table_next, _table_sub = _replay_tables(forest, code, device)
    anchors = subsets.reshape(-1).tolist()
    codes = torch.tensor(
        [int(forest.blocks[anchor][0]) for anchor in anchors], dtype=torch.long
    ).reshape(-1, 1)
    # [n * arity, 1]: the tuple's slowest digit on row 0, as ``dequantize`` lays it.
    nibbles = e2m1_nibbles(codes, forest.grid)
    return nibbles.reshape(-1).to(torch.uint8).to(device)


def lut_scale_bytes(scale_lut: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """``[16]`` uint8 -- the LUT scale plane's E4M3 bytes as stored, zero past
    the table's end.  ``lut_scale_table`` is the same table as numbers."""
    table = torch.zeros(16, dtype=torch.uint8, device=device)
    n = int(scale_lut.numel())
    if n > 16:
        raise GrammarError(f"a LUT scale plane holds at most 16 entries, got {n}")
    table[:n] = scale_lut.to(device).view(torch.uint8)
    return table


def prepare_span2_planes(parsed, device: str = "cuda") -> dict:
    """The native decoder's inputs for a parsed span-2 LUT-plane unit
    (``unit_artifact.parse_unit_artifact``): ``pack_unit_for_kernel``'s planes
    plus the two byte tables a code-emitting decoder needs."""
    unit, forests, code = parsed.unit, parsed.forests, parsed.code
    if code is None or not isinstance(forests, dict):
        raise GrammarError("prepare_span2_planes takes a TCQ unit; this one has no forest")
    rates = sorted(set(unit.rates))
    if len(rates) != 1:
        raise GrammarError(f"the span-2 planes take one forest per unit; rates {rates}")
    forest = forests[rates[0]]
    packed = pack_unit_for_kernel(unit, forest, code)
    packed = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in packed.items()}
    packed["subset_nibbles"] = build_subset_nibbles(forest, code, device)
    packed["lut_bytes"] = lut_scale_bytes(unit.scale_lut, device)
    return packed
