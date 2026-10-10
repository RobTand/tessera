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
    "prepare_span2_planes", "require_no_post_decode_transforms",
    "require_no_completion_plane", "require_select_plane_multiple",
    "require_select_plane_rows", "require_window_geometry",
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

#: Bits prepended to each column's select plane so that row 0's history
#: window reads the trellis start state instead of the previous column.
#: Zero for a whole unit -- the encoder's pinned zero start -- and, for a
#: row shard, the shard's INITIAL_STATE plane written into the last ``memory``
#: positions (``_thread_start_state``).  Eight rather than six keeps every
#: column byte-aligned.
SELECT_PAD = 8
#: Super-symbols whose select bits share one byte of a column plane.  The
#: span-L select plane packs one bit per super-symbol, so one byte holds this
#: many of them and a whole column holds ``8 * span`` codes.
SELECT_PLANE_SUPER_SYMBOLS_PER_BYTE = 8


def _thread_start_state(padded: torch.Tensor, memory: int,
                        initial_state: "torch.Tensor | None") -> None:
    """Write a shard's start state into the pad, in STREAM order.

    ``padded`` is ``[SELECT_PAD + steps, cols]`` with the select bits from row
    ``SELECT_PAD`` down; the decoders read step ``t``'s window as the
    ``memory`` positions above it followed by its own bit, oldest first, and
    ``build_history_lut`` / ``build_span2_luts`` map that window to the
    ``ConvCode`` state whose NEWEST bit is the top (``decode._conv_state_stream``:
    ``state_t = sum_{k=1..memory} select_{t-k} << (memory - k)``).  So bit
    ``j`` of the start state is select bit ``-(memory - j)``, which sits at pad
    position ``SELECT_PAD - memory + j``: bit 0 (the oldest) first, bit
    ``memory - 1`` (``select_{-1}``) in the last pad position.  Once the pad
    holds the state the decoders need no other argument -- step 0's window
    reads it exactly as step ``memory`` reads the select bits above it, and
    ``decode._conv_state_stream``'s ``init >> t`` correction is the same
    sliding window read ``t`` positions later.  ``None`` leaves the zero pad
    every whole unit decodes from.
    """
    if initial_state is None:
        return
    cols = padded.shape[1]
    state = initial_state.reshape(-1).to(padded.device, torch.int64)
    if state.numel() != cols:
        raise GrammarError(
            f"a start state names one register per column ({cols}); this one has "
            f"{state.numel()} entries")
    if memory < 1 or memory > SELECT_PAD:
        raise GrammarError(
            f"memory {memory} does not fit the {SELECT_PAD}-bit select pad the start "
            "state is threaded through")
    if state.numel() and (bool((state < 0).any()) or bool((state >> memory).any())):
        raise GrammarError(
            f"a start state is a {memory}-bit register per column; this one holds a "
            f"value outside [0, {1 << memory})")
    for j in range(memory):
        padded[SELECT_PAD - memory + j] = ((state >> j) & 1).to(padded.dtype)


def pack_kernel_planes(
    body_bits: torch.Tensor, rate: int = 3, memory: int = 6, span: int = 1,
    initial_state: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, ...]":
    """Wire BODY -> kernel planes, column-major, MSB-first.

    ``span == 1``: ``(select plane, point plane)``.  The select plane carries
    ``SELECT_PAD`` bits before each column, which is what lets a decoder read
    row 0's history without a boundary test: the pad *is* the initial state
    -- zero for a whole unit, the shard's own register for a row shard
    (``initial_state``, one ``memory``-bit value per column, threaded by
    ``_thread_start_state``).

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
    codes_per_byte = select_plane_codes_per_byte(span)
    if span == 1:
        if rows % codes_per_byte:
            raise GrammarError(f"{rows} rows does not byte-align a column plane")
        select = (body >> (rate - 1)) & 1
        padded = torch.zeros(rows + SELECT_PAD, cols, dtype=torch.int32, device=device)
        padded[SELECT_PAD:] = select
        _thread_start_state(padded, memory, initial_state)
        return _pack_columns(padded, 1), point_plane
    if rows % codes_per_byte:
        raise GrammarError(
            f"{rows} codes is not a multiple of {codes_per_byte}; a span-2 column holds one "
            "select bit and one label per pair and needs a byte-aligned column of pairs"
        )
    select = (body[0::2] >> (rate - 1)) & 1                 # [pairs, cols]
    label = (body[1::2] >> (rate - 1)) & (SUBSET_COUNT - 1)  # [pairs, cols]
    padded = torch.zeros(rows // 2 + SELECT_PAD, cols, dtype=torch.int32, device=device)
    padded[SELECT_PAD:] = select
    _thread_start_state(padded, memory, initial_state)
    select_plane = torch.cat([
        _pack_columns(padded, 1), torch.zeros(8, dtype=torch.uint8, device=device)
    ])
    return select_plane, _pack_columns(label, 2), point_plane


def select_plane_codes_per_byte(span: int) -> int:
    """Codes per column ONE byte of the SELECT plane holds: ``8 * span``.

    The span-L select plane packs one bit per super-symbol, eight
    super-symbols to a byte. This is the single number ``pack_kernel_planes``
    measures its columns against and the native admission below asks for, so
    the two cannot disagree about where a byte ends.
    """
    span = int(span)
    if span < 1:
        raise GrammarError(
            f"a span of {span} is not a positive number of rows per super-symbol"
        )
    return SELECT_PLANE_SUPER_SYMBOLS_PER_BYTE * span


def native_select_plane_admission(parsed) -> "tuple[int, int] | None":
    """``(rows, required_multiple)`` for the NATIVE span-2 decode, or ``None``
    for a body that has no select plane.

    ``pack_kernel_planes`` packs the span-L select plane one bit per
    super-symbol and eight to a byte, column after column in one flat uint8
    array with no per-column offset to resume on, so the native decoder needs
    each rank's rows to be a whole number of columns of
    ``select_plane_codes_per_byte(span)`` super-symbols: ``arity * 8 * span``.
    ``layout.shard_granularity`` reports the finer super-symbol boundary
    (``arity * span``) because that is what ``slice_unit`` measures its offsets
    against, so one super-symbol per column slices cleanly and the PACKER is
    where it lands -- naming bytes rather than the cut that produced them.

    ONLY THE NATIVE DECODER NEEDS THIS.  ``stock.materialize_stock`` (the
    ``when_unavailable`` torch fallback the span-2 route publishes) decodes
    codes, not these planes, so a cut below the byte is a cut it serves; the
    refusal therefore belongs at the native seam and not in
    ``serving.sharding``'s cutter, where it would take that fallback away too.

    A TCQ parse whose geometry cannot be read REFUSES rather than answering
    None: absent geometry means "we cannot tell whether this packs", and a
    coverage gate that reads an unknown as "no requirement" is one that fails
    open -- exactly the shape this function exists to close.  The window body
    is the one true None: ``pack_window_planes`` carries a per-column offset
    table and starts every column on its own byte, so every length packs.
    """
    from .manifest import BodyKind

    unit = getattr(parsed, "unit", parsed)
    body = BodyKind(getattr(unit, "body", BodyKind.TCQ))
    if body is not BodyKind.TCQ:
        return None
    forests = getattr(parsed, "forests", None)
    rates = sorted(set(getattr(unit, "rates", ()) or ()))
    where = (f"forests={type(forests).__name__}, rates={rates}, "
             f"body={body.name}, span={getattr(unit, 'span', None)!r}")
    if not isinstance(forests, dict) or len(rates) != 1:
        raise GrammarError(
            "the native span-2 admission needs exactly one forest per unit to "
            f"read the arity from; this parse reports {where}")
    forest = forests[rates[0]]
    grid = getattr(forest, "grid", forest)
    arity = getattr(grid, "arity", None)
    if not isinstance(arity, int) or arity < 1:
        raise GrammarError(
            "the native span-2 admission needs the grid's arity, which this "
            f"forest does not carry ({where})")
    # ``pack_kernel_planes`` measures the BODY plane's steps (one code per
    # step, ``arity`` rows per code), so the rows named here are
    # ``steps * arity`` -- the same ``rows`` ``pack_unit_for_kernel`` derives
    # -- and the multiple is that step boundary in rows.  Rows divide the
    # multiple exactly when steps divide ``select_plane_codes_per_byte``, so
    # this refuses the packer's own set of cuts and no other.
    body_bits = getattr(unit, "body_bits", None)
    if body_bits is None or getattr(body_bits, "ndim", 0) != 2:
        raise GrammarError(
            "the native span-2 admission needs the unit's [steps, cols] body "
            f"plane to count its rows ({where})")
    rows = int(body_bits.shape[0]) * arity
    return rows, arity * select_plane_codes_per_byte(int(getattr(unit, "span", 1)))


def require_select_plane_multiple(*, rows: int, multiple: int) -> None:
    """Refuse, by name, a native span-2 decode whose rows are not a whole
    number of select columns.

    Names the rank-local rows, the multiple the plane needs, and the fallback
    that does serve the cut, so the operator acts on the cut rather than on a
    byte count in someone else's module.  The facts are named so the compact
    preparer (``tessera.compact_prep``) refuses exactly this cut with exactly
    these words.
    """
    if int(rows) % int(multiple):
        raise GrammarError(
            f"the native span-2 decoder packs "
            f"{SELECT_PLANE_SUPER_SYMBOLS_PER_BYTE} super-symbols to a byte per "
            f"column; this rank's unit is {int(rows)} rows, which is not a whole number of "
            f"that column ({int(multiple)} rows = arity * 8 * span), so the select plane "
            f"cannot be packed. Serve at a tensor_parallel_size that lands on the "
            f"boundary, or take the torch fallback, which decodes codes and serves "
            f"this cut."
        )


def require_select_plane_rows(*, rows: int, arity: int, span: int) -> None:
    """``require_select_plane_multiple`` from the geometry that derives it."""
    require_select_plane_multiple(
        rows=rows, multiple=int(arity) * select_plane_codes_per_byte(int(span)))


def require_native_select_plane_admission(parsed) -> None:
    """Refuse, by name, a native span-2 decode whose rows are not a whole
    number of select columns (``native_select_plane_admission``)."""
    admission = native_select_plane_admission(parsed)
    if admission is None:
        return
    rows, multiple = admission
    require_select_plane_multiple(rows=rows, multiple=multiple)


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


def require_window_geometry(window_bits: int, rates) -> None:
    """The wire's window-width and rate rules, before a plane is packed.

    ONE home: ``pack_window_planes`` states them and the compact window
    repack (``tessera.compact_prep``) refuses the same bytes with the same
    words -- a rate wider than the window would read a window's worth of bits
    from the wrong place and decode to plausible wrong weights.
    """
    if not 1 <= int(window_bits) <= WINDOW_BITS_MAX:
        raise GrammarError(
            f"window_bits {window_bits} outside 1..{WINDOW_BITS_MAX}, the widest "
            "window the wire carries"
        )
    rates = tuple(rates)
    if rates and max(rates) > int(window_bits):
        raise GrammarError(
            f"rate {max(rates)} does not fit a {window_bits}-bit window"
        )


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

    The work per rate group is a fixed number of whole-tensor operations, not
    one strided write per bit position and per pad bit (tessera#501): the
    group's columns expand to ``[m, steps, R]`` bits in one broadcast shift,
    the pad to ``[m, L]`` in another, and the byte layout -- which columns
    share a rate, where each column starts -- is arithmetic on the ``rates``
    tuple the host already holds, so it costs no device read.  A serving
    load packs every role of every routed expert here, and the launch count
    is what bounded it.
    """
    steps, cols = body_bits.shape
    device = body_bits.device
    if len(rates) != cols:
        raise GrammarError(f"{len(rates)} rates for {cols} columns")
    require_window_geometry(window_bits, rates)
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
    rates = tuple(int(r) for r in rates)
    columns_at: "dict[int, list[int]]" = {}
    for column, rate in enumerate(rates):
        columns_at.setdefault(rate, []).append(column)
    col_bytes = [(window_bits + steps * rate + 7) // 8 for rate in rates]
    starts = [0] * (cols + 1)
    for column, nbytes in enumerate(col_bytes):
        starts[column + 1] = starts[column] + nbytes
    rate_t = torch.tensor(rates, dtype=torch.int32, device=device)
    plane = torch.zeros(starts[-1] + 8, dtype=torch.uint8, device=device)
    # uint8 shifts extract the same low bits an int32 copy would for the
    # reader's uint8 body; any wider dtype takes the int32 the loop always used.
    body = body_bits if body_bits.dtype == torch.uint8 else body_bits.to(torch.int32)
    weights = 1 << torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
    start = None if initial_state is None else initial_state.to(device).long()
    pad_shifts = torch.arange(window_bits - 1, -1, -1, device=device, dtype=torch.int64)
    whole = len(columns_at) == 1
    for present in sorted(columns_at):
        columns = columns_at[present]
        m = len(columns)
        nbytes = col_bytes[columns[0]]
        which = None if whole else torch.tensor(columns, dtype=torch.int64, device=device)
        values = body if whole else body.index_select(1, which)          # [steps, m]
        shifts = torch.arange(present - 1, -1, -1, device=device, dtype=torch.int64).to(body.dtype)
        body_part = ((values.t().unsqueeze(-1) >> shifts) & 1).reshape(m, steps * present)
        if start is None:
            pad = torch.zeros(m, window_bits, dtype=torch.uint8, device=device)
        else:
            state = start if whole else start.index_select(0, which)
            pad = ((state.unsqueeze(-1) >> pad_shifts) & 1).to(torch.uint8)
        tail = nbytes * 8 - window_bits - steps * present
        bits = torch.cat([pad, body_part.to(torch.uint8),
                          torch.zeros(m, tail, dtype=torch.uint8, device=device)], 1)
        packed = (bits.reshape(-1, 8) * weights).sum(1, dtype=torch.uint8)
        if whole:
            # One rate: the columns sit back to back at one width, so the
            # plane's head IS the group's bytes in column order.
            plane[: m * nbytes] = packed
        else:
            first = torch.tensor([starts[c] for c in columns], dtype=torch.int64, device=device)
            dest = first[:, None] + torch.arange(nbytes, device=device)[None, :]
            plane[dest.reshape(-1)] = packed
    offsets = torch.tensor([8 * s for s in starts[:cols]], dtype=torch.int64, device=device)
    return plane, offsets, rate_t


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
    """``require_no_post_decode_transforms`` for a unit object."""
    require_no_post_decode_transforms(
        release_positions=int(unit.release_index.numel()),
        diagonals=unit.diagonals is not None,
        rotation=unit.rotation,
    )


def require_no_post_decode_transforms(*, release_positions: int, diagonals: bool,
                                      rotation) -> None:
    """Refuse the three operations no GEMV on this lane applies.

    A released position is overwritten from the RELEASE plane, diagonals are
    a rank-1 factor outside the dot product and a rotation is a basis change
    -- none of which any kernel here reads, on either body.  Accepting one
    serves the transformed quantisation space as if it were
    ``reconstruct_unit(unit) @ x``: a plausible, wrong answer with no error.

    One rule, one home: the window branch stated these first and the TCQ
    branch did not state them at all, which is how a span-2 unit with a
    rotation packed and served silently.  The facts are named rather than
    read off a unit so the compact preparer (``tessera.compact_prep``), which
    holds verified bytes and no expanded unit, refuses exactly these bytes
    with exactly these words.
    """
    if release_positions:
        raise GrammarError(
            "this unit has released positions, which overwrite decoded codes "
            "from the RELEASE plane; the kernel lane reads no such plane"
        )
    if diagonals:
        raise GrammarError(
            "this unit carries diagonals; undoing them is a rank-1 factor "
            "outside the GEMV, which the kernel lane does not apply"
        )
    if RotationState(rotation) is not RotationState.NONE:
        raise GrammarError(
            f"this unit is rotated ({RotationState(rotation).name}); undoing the "
            "rotation is a basis change the kernel lane does not apply"
        )


def _require_no_completion_plane(unit, forest: AnchorForest) -> None:
    """``require_no_completion_plane`` for a unit object."""
    require_no_completion_plane(
        rates=unit.rates, rate=forest.rate, cap=forest.cap,
        limit=getattr(unit, "completion_limit", None))


def require_no_completion_plane(*, rates, rate: int, cap: int, limit, memo: "dict | None" = None) -> None:
    """Refuse a TCQ unit whose wire carries a COMPLETION plane.

    A column at body rate ``R`` under the grid's cap may spend up to
    ``cap - R`` further bits per position choosing among the descendants its
    anchor reaches, and ``reconstruct_unit`` applies them.  The span-2 planes
    this lane packs carry select, label and point bits and nothing else:
    ``build_subset_values`` reads ``forest.blocks[*][0]`` -- the anchor -- for
    every position and ``gemv_from_packed`` forwards no completion field, so
    a deep unit packed to the bytes of the same unit with its plane zeroed and
    served ``reconstruct_unit(zeroed)`` under the deep unit's name (#296).

    The rule is the WRITTEN depth, from the one home that sizes the plane
    (``grammar.completion_widths`` over ``completion_limit``): the plane is on
    the wire whatever its words hold, and a parsed full-depth unit reads its
    limit back as ``None`` (``completion_limit_from_elements``), so a check on
    the limit alone would pass exactly the unit the reader recovers.  A
    full-rate unit (``R == cap``) has no completion axis and passes at any
    limit; that is the shipping span-2 wire.

    The facts are named rather than read off a unit so the compact preparer
    refuses exactly these bytes with exactly these words.
    """
    from .grammar import completion_widths

    written = max(completion_widths(tuple(rates), cap, limit, memo=memo), default=0)
    if written:
        raise GrammarError(
            f"this unit carries a COMPLETION plane {written} level"
            f"{'s' if written != 1 else ''} deep (rate {rate} under a cap "
            f"of {cap}, completion_limit={limit}); the span-2 kernel lane "
            "reads no such plane -- it would serve every position at its anchor "
            "(blocks[anchor][0]) and drop the descendants the plane selects, "
            "which is not reconstruct_unit(unit). Encode at completion=0, or "
            "decode this unit through tessera.decode"
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
    elif unit.scale_plane is ScalePlaneKind.MX:
        # Not a kernel-lane plane: the lane's mainloop reads a per-16 E4M3
        # block scale times a global, and an E8M0-per-32 plane is neither.
        # Refused by name rather than relabelled through nvfp4_scale_bytes
        # (which would die inside the reshape); the block-scaled kernel is
        # tessera#443 bullet 3.
        raise GrammarError(
            "the kernel lane packs LUT and CHANNEL planes; this unit carries the "
            "MX plane (one E8M0 per 32), whose block-scaled kernel is not in "
            "this tree (tessera#443)"
        )
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
    a mixed-rate schedule (one forest per unit there), an S6b plane at span 2
    (that kernel reads the LUT plane's nibbles; the shipping wire is span 2
    over a LUT plane) and a COMPLETION plane of any written depth
    (``_require_no_completion_plane``: the planes packed here stop at the
    anchor, so a deep unit would be served as its zeroed twin).  The window
    branch reads both scale planes and any mixed schedule, and a window body
    has no completion axis by grammar.

    Both branches take a ROW SHARD.  A unit cut below row 0
    (``layout.slice_unit``) carries the register each column is in at the
    cut, and each branch threads it into its own pad -- the window branch's
    ``window_bits`` pad and, here, the span-2 select plane's ``SELECT_PAD``
    (``_thread_start_state``) -- so the decoder that reads the planes starts
    step 0 from the parent's state with no argument of its own.  A whole
    unit carries no state and packs to the zero pad it always did.
    """
    from .manifest import BodyKind, ScalePlaneKind

    _require_no_post_decode_transforms(unit)
    if getattr(unit, "body", BodyKind.TCQ) is BodyKind.WINDOW:
        grid = forest.grid if isinstance(forest, AnchorForest) else forest
        return _pack_window_unit(unit, grid)
    # A row shard (``layout.slice_unit``) carries its columns' trellis
    # register at the cut, ``state_bits`` wide.  The span-2 planes thread it
    # through the select pad exactly as the window branch threads its own
    # (``_thread_start_state``), so the decoders start step 0 from it with no
    # further argument; a whole unit carries None and packs to the pinned
    # zero pad it always did.  The width is checked against the code the
    # planes are packed for: a register of another code's memory would be
    # written into the wrong pad positions and read as a different state.
    initial_state = getattr(unit, "initial_state", None)
    if initial_state is not None:
        state_bits = int(getattr(unit, "state_bits", 0))
        if state_bits != code.memory:
            raise GrammarError(
                f"this shard's start state is {state_bits} bits wide and the code's "
                f"register is {code.memory}; the pad cannot carry a register of another "
                "code (unit begins at row "
                f"{getattr(unit, 'row_offset', 0)} of its parent)")
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
    _require_no_completion_plane(unit, forest)
    # Per-code planes are ``steps`` tall; a code covers ``arity`` rows.
    steps, cols = unit.codes.shape
    rows = steps * forest.grid.arity
    device = unit.body_bits.device
    select, label, point = pack_kernel_planes(
        unit.body_bits, rate=forest.rate, memory=code.memory, span=2,
        initial_state=initial_state)
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
    # The native decoder's own admission, before any packing: a rank cut below
    # the select plane's byte is refused here, by name, naming the cut.  The
    # torch fallback does not come through this function.
    require_native_select_plane_admission(parsed)
    packed = pack_unit_for_kernel(unit, forest, code)
    packed = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in packed.items()}
    packed["subset_nibbles"] = build_subset_nibbles(forest, code, device)
    packed["lut_bytes"] = lut_scale_bytes(unit.scale_lut, device)
    return packed
