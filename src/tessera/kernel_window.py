"""The FP8 route's fused decoder: the 4-bpp window wire, read at wire width.

WHAT THIS IS FOR.  Tessera's E4M3 route (grid ``E4M3``, WINDOW body at
``window_bits`` 14, span 1, over the CHANNEL scale plane -- ``export.
E4M3_RECIPE``) stores ~4.07 bpp on disk and serves as a stock per-channel FP8
tensor.  Serving it *resident* decodes once at load and holds 8 bpp, which is
plain FP8's footprint: the wire's saving lands on disk and nowhere else.
Serving it *streamed* holds the packed wire and decodes per forward, and the
only decoder that existed for that was pure torch (``gridbook.
tessera_window.PreparedWindow.decode``): correct, traceable, and 20-40x off
the memory-bandwidth bound, so the streamed mode paid more in decode than it
saved in bytes.

This module is the fused lane for that wire.  Two entry points:

- ``decode_fp8_tile`` -- the whole tile, bit-identical to ``decode.
  materialize_fp8``, for the prefill/GEMM path.  The stock ``torch.
  _scaled_mm`` mainloop consumes it unchanged.
- ``window_gemv`` -- decode-phase ``x @ W.T`` at small M that never
  materialises the tile, so the per-token weight traffic is the wire's ~4 bpp
  instead of the resident tile's 8.

WHY IT IS KERNELABLE.  ``state_t = ((state_{t-1} << R) | bits_t) mod 2^L``
from ``state_{-1} = 0`` is a shift register, so a state is *literally* the
last ``L`` bits of its column's stream and every position decodes from a
local read: with ``pack_window_planes``' ``L``-bit pad, position ``t`` of
column ``c`` reads the ``L`` bits at ``offsets[c] + (t + 1) * R``.  No walk
down the column, no halo, no replay tables -- the same property
``kernel.py`` had to prove about ``ConvCode.step`` for the trellis lane, here
by construction.  Columns are independent; the trellis axis is the *output*
row axis, so a GEMV's reduction axis (the input columns) is the axis that
does not carry state.

WHAT IS SHARED WITH THE REST OF THE TREE.  The packing is the wire's own
(``lane_planes.pack_window_planes``, the same bytes ``gridbook.
tessera_window.prepare_window`` reads), the code table is the unit's own
ALPHABET plane, the byte map is the grid's ``native`` and the row scale is
``scale_channel.channel_scale_field``'s expression.  Nothing here re-derives
a table the reference decoder builds, so the two cannot drift.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .errors import GrammarError
from .lane_planes import pack_window_planes
from .manifest import BodyKind, RotationState, ScalePlaneKind

__all__ = [
    "GEMV_MAX_M",
    "WINDOW_SPAN_BITS",
    "PreparedWindowUnit",
    "prepare_window_unit",
    "prepare_from_parsed",
    "decode_fp8_tile",
    "window_gemv",
    "window_linear",
    "window_module_decode",
    "window_module_row_scale",
    "window_module_linear",
    "window_code_table",
    "window_value_table",
]

#: The largest M ``window_gemv`` takes.  The accumulator is ``[MBLK, LANES,
#: VEC]`` fp32 in registers, so M grows the block linearly; past this the
#: decode-then-GEMM path is both faster and the contract the lane attests.
GEMV_MAX_M = 8

#: Bits a lane reads in one span.  A lane covers ``VEC`` consecutive codes,
#: whose windows together occupy ``window_bits + (VEC - 1) * R + 7`` bits
#: counting the sub-byte offset of the first, and one int64 holds 64.
WINDOW_SPAN_BITS = 64


# --- tables -----------------------------------------------------------------


def window_code_table(window_codes: torch.Tensor, grid, device=None) -> torch.Tensor:
    """``state -> E4M3 byte``: the unit's ``2^L`` ALPHABET plane through the
    grid's ``native`` byte map, ``uint8 [2^L]``.

    The composition is what ``decode.materialize_fp8`` does in two steps
    (``decode_codes_mixed`` then ``native[codes]``); doing it once at
    preparation means the kernel's inner loop is one gather, and the two
    former-NaN slots and the negative zero land on the legal byte the
    reference decoder writes because it is the same map.
    """
    if grid.native is None or grid.size != 256 or grid.arity != 1:
        raise GrammarError(
            f"the FP8 route decodes a scalar 256-code hardware grid, got {grid.name}"
        )
    codes = window_codes.reshape(-1).to(torch.long)
    native = torch.tensor(grid.native, dtype=torch.long, device=codes.device)
    if codes.numel() and int(codes.max()) >= native.numel():
        raise GrammarError(
            f"the window table names code {int(codes.max())} outside the "
            f"{native.numel()}-code {grid.name} grid"
        )
    table = native[codes].to(torch.uint8).contiguous()
    return table if device is None else table.to(device)


def window_value_table(code_table: torch.Tensor, device=None) -> torch.Tensor:
    """``state -> the E4M3 byte's value``, ``float16 [2^L]``.

    fp16, and exactly: every finite E4M3 value is ``2**(e-7) * (1 + m/8)`` for
    ``e in 1..15`` or ``m * 2**-9``, i.e. three mantissa bits over
    ``2**-9 .. 448``, all inside fp16's normals.  The test asserts the
    round-trip on all 254 legal bytes rather than arguing it.  Half the bytes
    of an fp32 table halves the gather's footprint, and the gather is what
    this kernel spends its time on: 32 KB of table stays hot where 64 KB
    starts to evict.

    Derived from the *decoded byte*, never from ``grid.values`` directly, so
    the GEMV reconstructs exactly what a GEMM over ``decode_fp8_tile``'s tile
    would see.
    """
    byte = code_table.reshape(-1).to(torch.uint8)
    value = byte.view(torch.float8_e4m3fn).to(torch.float16).contiguous()
    return value if device is None else value.to(device)


# --- the prepared unit ------------------------------------------------------


class PreparedWindowUnit:
    """One E4M3 window unit, packed on a device, decoded by the kernels here.

    Holds the wire (``plane_words``, ``offsets``, ``rates``), the two derived
    tables and the per-row scale.  Everything is a contiguous device tensor
    fixed at preparation, which is what lets the ops below be traced at
    static shape inside a compiled forward.
    """

    __slots__ = (
        "plane_words", "wire_bytes", "offsets", "rates", "initial", "code_table",
        "value_table", "row_scale", "rows", "cols", "window_bits", "max_rate", "device",
    )

    def __init__(self, plane_words, wire_bytes, offsets, rates, initial, code_table,
                 value_table, row_scale, rows, cols, window_bits, max_rate):
        self.plane_words = plane_words
        self.wire_bytes = int(wire_bytes)
        self.offsets = offsets
        self.rates = rates
        self.initial = initial
        self.code_table = code_table
        self.value_table = value_table
        self.row_scale = row_scale
        self.rows = int(rows)
        self.cols = int(cols)
        self.window_bits = int(window_bits)
        self.max_rate = int(max_rate)
        self.device = plane_words.device

    @property
    def steps(self) -> int:
        """Codes per column.  At arity 1 that is the unit's output rows -- the
        name the torch reader's prepared object uses, kept so a lane can swap
        one for the other without touching the caller."""
        return self.rows

    def resident_bytes(self) -> int:
        """Alias of ``wire_bytes_resident`` under the torch reader's name."""
        return self.wire_bytes_resident()

    def wire_bytes_resident(self) -> int:
        """Device bytes held: the packed stream plus the tables and the scale.

        The two tables are ``2^L`` entries deterministic in the recipe
        (``encode.window_table``), i.e. one pair for a whole checkpoint --
        every unit of the reach checkpoint carries byte-identical
        ``window_codes`` -- so a lane that shares them across modules pays
        them once, not per unit.  This counts them per unit, which is the
        pessimistic reading.
        """
        return sum(
            t.numel() * t.element_size()
            for t in (self.plane_words, self.offsets, self.rates, self.initial,
                      self.code_table, self.value_table, self.row_scale)
        )

    def decode(self) -> torch.Tensor:
        """``uint8 [rows, cols]`` of E4M3 bytes -- a fresh tensor, every call."""
        return decode_fp8_tile(
            self.plane_words, self.offsets, self.rates, self.initial, self.code_table,
            self.rows, self.cols, self.window_bits, self.max_rate,
        )

    def gemv(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ W.T`` for ``x [M, cols]``, ``[M, rows]`` fp32."""
        return window_gemv(
            x, self.plane_words, self.offsets, self.rates, self.initial,
            self.value_table, self.row_scale, self.rows, self.cols, self.window_bits,
            self.max_rate,
        )


def prepare_window_unit(
    body_bits: torch.Tensor,
    rates,
    window_bits: int,
    window_codes: torch.Tensor,
    grid,
    row_scale: torch.Tensor,
    initial: "torch.Tensor | None" = None,
    device=None,
) -> PreparedWindowUnit:
    """Pack one window unit's wire and build its tables.

    ``body_bits`` is the reader's ``[rows, cols]`` uint8 (the R-bit value per
    position), ``rates`` the per-column schedule, ``window_codes`` the unit's
    ALPHABET plane and ``row_scale`` the CHANNEL plane's fp32 ``[rows]``
    (``scale_rows * scale_global``).  The packing is ``pack_window_planes``:
    the same bytes the torch reader reads, so a disagreement between the two
    decoders is a decoder bug and never a layout difference.
    """
    device = torch.device("cuda" if device is None else device)
    body_bits = body_bits.to(device)
    rows, cols = body_bits.shape
    rates = tuple(int(r) for r in rates)
    if len(rates) != cols:
        raise GrammarError(f"{len(rates)} rates for {cols} columns")
    window_bits = int(window_bits)
    max_rate = max(rates)
    _check_reach(window_bits, max_rate, _VEC)
    plane, offsets, rate_t = pack_window_planes(body_bits, rates, window_bits)
    code_table = window_code_table(window_codes.to(device), grid, device)
    if code_table.numel() != 1 << window_bits:
        raise GrammarError(
            f"the window table holds {code_table.numel()} entries, window_bits "
            f"{window_bits} needs {1 << window_bits}"
        )
    scale = row_scale.to(device, torch.float32).reshape(-1)
    if scale.numel() != rows:
        raise GrammarError(f"{scale.numel()} row scales for {rows} rows")
    if initial is None:
        start = torch.zeros(cols, dtype=torch.int64, device=device)
    else:
        start = initial.to(device, torch.int64).reshape(-1)
        if start.numel() != cols:
            raise GrammarError(f"{start.numel()} initial windows for {cols} columns")
        if start.numel() and (int(start.min()) < 0 or int(start.max()) >= 1 << window_bits):
            raise GrammarError(
                f"an initial window is {window_bits} bits; the plane names "
                f"{int(start.max())}"
            )
    return PreparedWindowUnit(
        _plane_words(plane.contiguous()), plane.numel(), offsets.contiguous(),
        rate_t.contiguous(), start.contiguous(), code_table,
        window_value_table(code_table, device), scale.contiguous(),
        rows, cols, window_bits, max_rate,
    )


def prepare_from_parsed(parsed, device=None) -> PreparedWindowUnit:
    """A ``unit_artifact.ParsedUnit`` of the FP8 route -> a prepared unit.

    Refuses exactly what the reference FP8 materialisation and the trellis
    lane's packer refuse, and for the same reasons: a released position is a
    second plane that overwrites a decoded code, diagonals are a rank-1
    factor outside the product, and a rotation is a basis change.  None of
    the three is applied here, so serving a unit that carries one would serve
    weights the reference decoder does not produce.
    """
    unit, grid = parsed.unit, parsed.grid
    if getattr(unit, "body", BodyKind.TCQ) is not BodyKind.WINDOW:
        raise GrammarError(
            f"this lane decodes the window body; the unit carries {parsed.body.name}"
        )
    if getattr(unit, "scale_plane", None) is not ScalePlaneKind.CHANNEL:
        raise GrammarError(
            "a per-channel FP8 tile takes the CHANNEL plane; the unit carries "
            f"{getattr(getattr(unit, 'scale_plane', None), 'name', None)}"
        )
    if unit.scale_rows is None:
        raise GrammarError("a CHANNEL scale plane needs the unit's row words")
    if unit.release_index.numel():
        raise GrammarError(
            "this unit has released positions, which overwrite decoded codes "
            "from the RELEASE plane; this lane reads no such plane"
        )
    if unit.diagonals is not None:
        raise GrammarError(
            "this unit carries diagonals; undoing them is a rank-1 factor "
            "outside the product, which this lane does not apply"
        )
    if unit.rotation is not RotationState.NONE:
        raise GrammarError(
            f"this unit is rotated ({unit.rotation.name}); undoing the rotation "
            "is a basis change this lane does not apply"
        )
    rows, _cols = unit.body_bits.shape
    row_scale = unit.scale_rows.to(torch.float32) * float(unit.scale_global)
    # ``initial_window`` is absent on a whole unit (the pinned zero start) and
    # present on a tensor-parallel shard, whose columns continue a parent's
    # stream.  Read whichever the artifact carries; never assume the zero.
    return prepare_window_unit(
        unit.body_bits, unit.rates, unit.window_bits, unit.window_codes, grid,
        row_scale.reshape(rows), initial=getattr(unit, "initial_window", None),
        device=device,
    )


# --- the decode kernel ------------------------------------------------------

#: Codes a lane covers out of one 64-bit span.  Eight is what makes the read
#: one int64 for four bytes of wire at R=4 (a 2x L1 overlap with the
#: neighbouring lane, whose span starts four bytes on, and 1x from L2).
_VEC = 8


def _check_reach(window_bits: int, max_rate: int, vec: int) -> None:
    """Refuse a schedule whose ``vec`` windows do not fit one 64-bit span.

    The shifts are silently wrong past the bound and a wrong weight is not an
    error, so this is checked at preparation rather than assumed.
    """
    if not 1 <= max_rate <= window_bits:
        raise GrammarError(f"rate {max_rate} does not fit a {window_bits}-bit window")
    reach = window_bits + (vec - 1) * max_rate + 7
    if reach > WINDOW_SPAN_BITS:
        raise GrammarError(
            f"a {window_bits}-bit window at rate {max_rate} reaches {reach} bits "
            f"across {vec} codes, past the {WINDOW_SPAN_BITS} the kernel reads in "
            "one span"
        )


@triton.jit
def _span_of(words_ptr, byte, mask):
    """Eight plane bytes from ``byte`` as one big-endian int64.

    Two **aligned** int64 loads and a funnel shift, not eight byte loads.
    The plane is stored with each eight-byte word reversed (``plane_words``),
    so a little-endian int64 load already yields the wire's MSB-first field
    and the only thing left is to slide the window to the byte the caller
    asked for.

    This is where the decode's time was.  Eight byte loads is one load
    instruction per *code*; ablating the plane read out of the byte-load
    version took a 1024x3072 decode from 45.6 us to 9.8 us, while ablating
    the ``2^14`` table gather -- the load everyone expects to dominate --
    changed almost nothing (42.3 us), and shrinking that table to 256 entries
    changed nothing at all.  The kernel was LSU-issue bound on the wire read,
    not gather bound.  Two loads per eight codes is a quarter of the
    instructions for the same bytes.

    ``(64 - s) & 63`` rather than ``64 - s``: at ``s = 0`` the second term is
    masked away by ``(1 << s) - 1``, but a shift of 64 is poison in LLVM
    before the mask ever runs.
    """
    s = (byte & 7) * 8
    word = byte >> 3
    w0 = tl.load(words_ptr + word, mask=mask, other=0)
    w1 = tl.load(words_ptr + word + 1, mask=mask, other=0)
    return (w0 << s) | ((w1 >> ((64 - s) & 63)) & ((1 << s) - 1))


def _plane_words(plane: torch.Tensor) -> torch.Tensor:
    """``pack_window_planes``' bytes as int64 words a little-endian load reads
    big-endian: each eight-byte word reversed, padded to a whole number of
    words plus two of slack for the funnel's second load.

    The byte *count* is the wire's; only the order inside a word changes, and
    it changes at preparation, once.  The torch reader's own prepared object
    reorders the same bytes differently (per-rate group rows with four bytes
    of slack), for the same reason: a prepared plane is the wire laid out for
    the machine that reads it.
    """
    n = plane.numel()
    words = (n + 7) // 8 + 2
    buf = torch.zeros(words * 8, dtype=torch.uint8, device=plane.device)
    buf[:n] = plane
    return buf.reshape(-1, 8).flip(1).reshape(-1).view(torch.int64).contiguous()


@triton.jit
def _state_of(span, sub, code, v, rate, init, window: tl.constexpr):
    """The ``window``-bit state of ``VEC`` consecutive codes out of one span.

    ``span`` is the eight wire bytes at the lane's anchor, ``sub`` the anchor's
    sub-byte bit offset, ``code`` the absolute code index, ``v`` its offset
    inside the lane and ``rate`` the column's R.  The arithmetic right shift copies the sign into positions at
    or above the field width, which the mask removes.

    ``init`` is the column's **initial window** -- the ``window`` bits that
    precede its first stored position.  ``pack_window_planes`` pads every
    column with ``window`` zero bits, which *is* ``state_{-1} = 0``, so a
    whole unit reads correctly with ``init = 0``.  A tensor-parallel shard is
    not a whole unit: its first row continues a column whose earlier rows
    live on another rank, and the state it must continue from is that
    parent's last ``window`` bits.  Position ``t`` sits ``d = (t + 1) * R``
    bits into the stream, so the pad still contributes its low ``window - d``
    bits while ``d < window``, in the field's top ``window - d`` places;
    past that the shift register has forgotten it and the term is zero.  The
    ``d & 63`` keeps the dead branch's shift legal.
    """
    MASK: tl.constexpr = (1 << window) - 1
    state = (span >> (64 - sub - v * rate - window)) & MASK
    d = (code + 1) * rate
    return state | tl.where(d < window, (init << (d & 63)) & MASK, 0)


@triton.jit
def _decode_kernel(
    words_ptr, offset_ptr, rate_ptr, init_ptr, table_ptr, out_ptr,
    rows, cols,
    window: tl.constexpr, LANES: tl.constexpr, VEC: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """A ``[LANES * VEC, BLOCK_C]`` tile of E4M3 bytes, straight from the wire.

    The block is ``[BLOCK_C, LANES, VEC]`` while it reads and ``[LANES * VEC,
    BLOCK_C]`` when it writes, and both orders are forced:

    - **Reading, the lane axis is fastest.**  Lane ``l`` of a column anchors
      at bit ``(base + l * VEC + 1) * R``, so at R=4 consecutive lanes' spans
      start four bytes apart and the ``LANES`` span loads cover one
      contiguous ``4 * LANES + 8`` byte run of that column.  With the column
      axis fastest instead, each load is a ``col_bytes``-strided gather.
    - **Writing, the column axis is fastest.**  The tile is row-major
      ``[rows, cols]`` -- the layout ``torch._scaled_mm`` takes -- so a
      contiguous store walks columns.  ``tl.trans`` between the two is a
      register/shared transpose, which is the cheap end of the trade.
    """
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    lane = tl.arange(0, LANES)
    v = tl.arange(0, VEC)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    live_c = c < cols

    offset = tl.load(offset_ptr + c, mask=live_c, other=0)                 # [BLOCK_C]
    rate = tl.load(rate_ptr + c, mask=live_c, other=1).to(tl.int64)        # [BLOCK_C]
    init = tl.load(init_ptr + c, mask=live_c, other=0).to(tl.int64)        # [BLOCK_C]

    base = pid_p * (LANES * VEC) + lane * VEC                              # [LANES]
    # [BLOCK_C, LANES]: the lane axis last, so the span loads coalesce.
    anchor = offset[:, None] + (base[None, :].to(tl.int64) + 1) * rate[:, None]
    live = live_c[:, None] & (base[None, :] < rows)
    span = _span_of(words_ptr, anchor // 8, live)                          # [BLOCK_C, LANES]

    # The initial-window term is provably zero above the first row block, and
    # hoisting it behind `if pid_p == 0` -- a program-uniform branch, so it
    # looked free -- cost 10.6x: 1024x3072 went 23.9 -> 253.1 us, 1024x1024
    # 11.4 -> 25.5, 4096x2560 261.5 -> 740.3 (hoist_ab.py, idle GB10, min of 9
    # rounds).  A branch whose result is a [BLOCK_C, LANES, VEC] tensor makes
    # Triton carry that tensor across an `scf.if`, and it leaves the register
    # file to do it.  The arithmetic the branch saves is three integer ops per
    # code; the spill it buys costs an order of magnitude more.  Compute it
    # unconditionally.
    code = base[None, :, None].to(tl.int64) + v[None, None, :].to(tl.int64)
    state = _state_of(
        span[:, :, None], (anchor % 8)[:, :, None], code,
        v[None, None, :].to(tl.int64), rate[:, None, None], init[:, None, None],
        window,
    )                                                                      # [BLOCK_C, LANES, VEC]
    byte = tl.load(table_ptr + state, mask=live[:, :, None], other=0)

    out = tl.trans(tl.reshape(byte, (BLOCK_C, LANES * VEC)), 1, 0)         # [LANES*VEC, BLOCK_C]
    p = pid_p * (LANES * VEC) + tl.arange(0, LANES * VEC)
    keep = (p[:, None] < rows) & live_c[None, :]
    tl.store(out_ptr + p[:, None].to(tl.int64) * cols + c[None, :], out, mask=keep)


def _decode_impl(plane_words, offsets, rates, initial, code_table, rows, cols,
                 window_bits, max_rate, lanes: int = 32, block_c: int = 64):
    out = torch.empty((rows, cols), dtype=torch.uint8, device=plane_words.device)
    _decode_kernel[(triton.cdiv(rows, lanes * _VEC), triton.cdiv(cols, block_c))](
        plane_words, offsets, rates, initial, code_table, out, rows, cols,
        window=window_bits, LANES=lanes, VEC=_VEC, BLOCK_C=block_c,
        num_warps=4,
    )
    return out


# --- the GEMV kernel --------------------------------------------------------

#: Columns a program reads per loop iteration.  The offsets, rates, initial
#: windows and activations are then vector loads instead of one dependent
#: scalar load apiece per column, which is what the first cut spent its time
#: on: at ``KB = 1`` the same kernel ran 96.7 us where this runs 19.6.
_KB = 8

#: Warps a GEMV program runs on.  Measured, not guessed; see ``_gemv_impl``.
_WARPS = 2


@triton.jit
def _gemv_kernel(
    x_ptr, words_ptr, offset_ptr, rate_ptr, init_ptr, value_ptr, scale_ptr, out_ptr,
    rows, cols, m,
    window: tl.constexpr, LANES: tl.constexpr, VEC: tl.constexpr,
    MBLK: tl.constexpr, KB: tl.constexpr, SPLIT: tl.constexpr,
):
    """``out[m, n] += scale[n] * sum_k value(n, k) * x[m, k]`` off the wire.

    A program owns ``LANES * VEC`` consecutive output rows and a **contiguous**
    slice of the input columns -- contiguous, not strided by ``SPLIT``, so the
    bytes a program touches are one run of each column's stream rather than
    ``cols / SPLIT`` scattered ones.  The row scale is folded into each
    partial before the atomic, so there is no epilogue launch over the output.

    The reduction axis is the *column* axis, the one the trellis does not run
    down, so the split changes only the fp32 summation order and nothing
    about what is decoded.

    Per code the work is one gather out of the ``2^L`` fp16 value table
    (32 KB) and one FMA; per ``VEC`` codes it is one two-word span load.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    lane = tl.arange(0, LANES)
    v = tl.arange(0, VEC)
    mi = tl.arange(0, MBLK)
    j = tl.arange(0, KB)
    live_m = mi < m

    base = pid_n * (LANES * VEC) + lane * VEC                       # [LANES]
    n = base[:, None] + v[None, :]                                  # [LANES, VEC]
    live = n < rows
    scale = tl.load(scale_ptr + n, mask=live, other=0.0)

    acc = tl.zeros((MBLK, LANES, VEC), dtype=tl.float32)
    chunk = tl.cdiv(cols, SPLIT * KB) * KB
    start = pid_k * chunk
    stop = tl.minimum(start + chunk, cols)
    for k0 in range(start, stop, KB):
        kk = k0 + j                                                 # [KB]
        lk = kk < cols
        offset = tl.load(offset_ptr + kk, mask=lk, other=0)
        rate = tl.load(rate_ptr + kk, mask=lk, other=1).to(tl.int64)
        init = tl.load(init_ptr + kk, mask=lk, other=0).to(tl.int64)
        anchor = offset[:, None] + (base[None, :].to(tl.int64) + 1) * rate[:, None]
        ok = lk[:, None] & (base[None, :] < rows)                   # [KB, LANES]
        span = _span_of(words_ptr, anchor // 8, ok)
        code = base[None, :, None].to(tl.int64) + v[None, None, :].to(tl.int64)
        state = _state_of(
            span[:, :, None], (anchor % 8)[:, :, None], code,
            v[None, None, :].to(tl.int64), rate[:, None, None], init[:, None, None],
            window,
        )                                                           # [KB, LANES, VEC]
        value = tl.load(value_ptr + state, mask=ok[:, :, None], other=0.0).to(tl.float32)
        xv = tl.load(x_ptr + mi[:, None] * cols + kk[None, :],
                     mask=live_m[:, None] & lk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(xv[:, :, None, None] * value[None, :, :, :], axis=1)

    tl.atomic_add(
        out_ptr + mi[:, None, None] * rows + n[None, :, :], acc * scale[None, :, :],
        mask=live_m[:, None, None] & live[None, :, :],
    )


def _gemv_split(rows: int, lanes: int) -> int:
    """Column splits: enough programs to fill the box, as few as fill it.

    ``rows / (LANES * VEC)`` programs come from the output axis alone -- four
    for a 1024-row unit -- so the split is what puts work on the other 44 SMs.
    Rounded to a power of two so the launch shape is stable across the five
    shapes of a checkpoint rather than jittering with ``rows``.
    """
    blocks = max(1, -(-rows // (lanes * _VEC)))
    want = max(8, min(128, 256 // blocks))
    return 1 << (want.bit_length() - 1)


def _gemv_impl(x, plane_words, offsets, rates, initial, value_table, row_scale, rows,
               cols, window_bits, max_rate, lanes: int = 32, split: "int | None" = None,
               kb: "int | None" = None, warps: "int | None" = None):
    m = x.shape[0]
    mblk = 1 if m <= 1 else 1 << (m - 1).bit_length()
    # Two warps a program, at every M.  The inner temporary is
    # [MBLK, KB, LANES, VEC] fp32 and the decoded state is [KB, LANES, VEC]
    # int64, so a program is register-hungry however many warps carry it, and
    # widening the launch to cover a larger MBLK -- num_warps = min(8, 2*mblk),
    # which is what this line used to say -- made it worse at every point of a
    # 3-shape x 3-M x 36-config sweep on an idle box: 1024x3072 at M=8 took
    # 96.1 us at eight warps against 29.1 at two.  More warps means more live
    # registers per SM for the same tile, not more parallelism; the parallelism
    # is in the grid (rows/(LANES*VEC) x SPLIT).  Fewer than two lost on the
    # large shapes.  The sweep is `experiments/bench_kernel_window.py --arm
    # gemv` plus /home/rob/tmp/kernel-window/msweep2.py.
    kb = _KB if kb is None else kb
    split = _gemv_split(rows, lanes) if split is None else split
    out = torch.zeros((m, rows), dtype=torch.float32, device=x.device)
    _gemv_kernel[(triton.cdiv(rows, lanes * _VEC), split)](
        x, plane_words, offsets, rates, initial, value_table, row_scale, out,
        rows, cols, m,
        window=window_bits, LANES=lanes, VEC=_VEC, MBLK=mblk, KB=kb, SPLIT=split,
        num_warps=(_WARPS if warps is None else warps),
    )
    return out


# --- the ops ----------------------------------------------------------------
#
# ``torch.library.custom_op`` with ``mutates_args=()``: every op allocates and
# returns its own output, touches no buffer it did not make, reads no data
# pointer and branches on no traced value.  That is the shape a compiled vLLM
# forward can functionalise -- each of those four has broken a serving lane
# here before (``vllm-compiled-forward-breaks-lane-hot-paths``).


@torch.library.custom_op("tessera_window::decode_fp8_tile", mutates_args=())
def decode_fp8_tile(
    plane_words: torch.Tensor,
    offsets: torch.Tensor,
    rates: torch.Tensor,
    initial: torch.Tensor,
    code_table: torch.Tensor,
    rows: int,
    cols: int,
    window_bits: int,
    max_rate: int,
) -> torch.Tensor:
    """The unit's E4M3 tile, ``uint8 [rows, cols]``, decoded from the wire.

    Byte-identical to ``decode.materialize_fp8``'s first return and to a
    stock ``float-quantized`` checkpoint's ``weight`` at ``strategy:
    channel``; the row scale is the prepared unit's ``row_scale``, which is
    the same fp32 expression the reference decoder returns.
    """
    _check_reach(int(window_bits), int(max_rate), _VEC)
    return _decode_impl(plane_words, offsets, rates, initial, code_table,
                        int(rows), int(cols), int(window_bits), int(max_rate))


@decode_fp8_tile.register_fake
def _(plane_words, offsets, rates, initial, code_table, rows, cols, window_bits,
      max_rate):
    return plane_words.new_empty((rows, cols), dtype=torch.uint8)


@torch.library.custom_op("tessera_window::window_gemv", mutates_args=())
def window_gemv(
    x: torch.Tensor,
    plane_words: torch.Tensor,
    offsets: torch.Tensor,
    rates: torch.Tensor,
    initial: torch.Tensor,
    value_table: torch.Tensor,
    row_scale: torch.Tensor,
    rows: int,
    cols: int,
    window_bits: int,
    max_rate: int,
) -> torch.Tensor:
    """``x @ W.T`` for ``x [M, cols]``, decoding the wire in the kernel.

    Returns ``[M, rows]`` fp32.  The tile is never materialised, so the
    weight traffic per token is the wire's, not the resident tile's.  Split-K
    reduces by fp32 atomic add, so the sum order is not fixed run to run --
    a one-hot probe still returns a column exactly, because nothing is
    summed.
    """
    _check_reach(int(window_bits), int(max_rate), _VEC)
    if x.ndim != 2 or x.shape[1] != cols:
        raise GrammarError(f"x is {tuple(x.shape)}, the unit takes [M, {cols}]")
    if not 1 <= x.shape[0] <= GEMV_MAX_M:
        raise GrammarError(
            f"window_gemv is the small-M path: M={x.shape[0]} past {GEMV_MAX_M} grows "
            "the register accumulator past what the launch holds; decode the tile and "
            "run the GEMM (window_linear does this on its own)"
        )
    return _gemv_impl(x.contiguous(), plane_words, offsets, rates, initial, value_table,
                      row_scale, int(rows), int(cols), int(window_bits), int(max_rate))


@window_gemv.register_fake
def _(x, plane_words, offsets, rates, initial, value_table, row_scale, rows, cols,
      window_bits, max_rate):
    return x.new_empty((x.shape[0], rows), dtype=torch.float32)


def _fp8_per_token(x: torch.Tensor):
    """Per-token dynamic E4M3 quantisation: the A side the FP8 route serves."""
    amax = x.abs().amax(dim=1, keepdim=True).to(torch.float32).clamp_min(1e-12)
    scale = amax / 448.0
    return (x.to(torch.float32) / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


@torch.library.custom_op("tessera_window::window_linear", mutates_args=())
def window_linear(
    x: torch.Tensor,
    plane_words: torch.Tensor,
    offsets: torch.Tensor,
    rates: torch.Tensor,
    initial: torch.Tensor,
    code_table: torch.Tensor,
    value_table: torch.Tensor,
    row_scale: torch.Tensor,
    rows: int,
    cols: int,
    window_bits: int,
    max_rate: int,
    gemv_max: int = GEMV_MAX_M,
) -> torch.Tensor:
    """One Linear over the window wire: ``x @ W.T``, ``[..., cols] -> [..., rows]``.

    The M dispatch lives **inside** the op, where ``x.shape[0]`` is a
    concrete integer, and the fake impl reports the output shape without
    reading it.  A dispatch written in the traced forward instead would be a
    guard on the token dimension, which is the exact break the lane hit
    before (``vllm-compiled-forward-breaks-lane-hot-paths``).

    ``gemv_max`` is the largest M the fused GEMV takes; above it the tile is
    decoded and the stock W8A8 ``_scaled_mm`` mainloop runs.  **The two
    branches execute different activation contracts** -- the GEMV multiplies
    the bf16 activation directly (W8A16-class), the GEMM quantises it per
    token to E4M3 (W8A8) -- so a lane that turns the GEMV on must say so in
    its route record (principle 14).  ``gemv_max=0`` disables the GEMV and
    leaves one contract on the path.
    """
    orig = x.shape
    x2 = x.reshape(-1, int(cols))
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    if 0 < x2.shape[0] <= min(int(gemv_max), GEMV_MAX_M):
        y = window_gemv(x2, plane_words, offsets, rates, initial, value_table,
                        row_scale, rows, cols, window_bits, max_rate).to(torch.bfloat16)
    else:
        tile = decode_fp8_tile(plane_words, offsets, rates, initial, code_table,
                               rows, cols, window_bits, max_rate)
        a_q, a_scale = _fp8_per_token(x2.contiguous())
        y = torch._scaled_mm(
            a_q, tile.view(torch.float8_e4m3fn).t(),
            scale_a=a_scale, scale_b=row_scale.view(1, int(rows)),
            out_dtype=torch.bfloat16,
        )
    return y.reshape(*orig[:-1], int(rows))


@window_linear.register_fake
def _(x, plane_words, offsets, rates, initial, code_table, value_table, row_scale,
      rows, cols, window_bits, max_rate, gemv_max=GEMV_MAX_M):
    return x.new_empty((*x.shape[:-1], rows), dtype=torch.bfloat16)


# --- the seam a serving lane calls ------------------------------------------
#
# A vLLM Linear is one *module* of one or more Tessera *units* -- q/k/v are
# three units behind one `qkv_proj`, gate/up two behind one `gate_up_proj` --
# stacked along the output-row axis in role order.  The FP8 route's prepared
# module (`serving/fp8_route.py::prepare_tessera_fp8_module`) holds one
# packed window per role and concatenates their decoded tiles; these three
# take the same list of prepared units and do the same thing through the
# kernels here, so a lane swaps one for the other without changing its own
# shape logic.


def window_module_decode(units) -> torch.Tensor:
    """The module's whole E4M3 tile, ``uint8 [sum(rows), cols]``, role order.

    Byte-identical to ``torch.cat([materialize_fp8(u)[0] for u in units], 0)``,
    which is what the route's ``PreparedTesseraFp8Module.decode`` returns.
    """
    units = list(units)
    if not units:
        raise GrammarError("a module needs at least one unit")
    if len({u.cols for u in units}) != 1:
        raise GrammarError(
            "the units of one module share their input columns; these carry "
            f"{sorted({u.cols for u in units})}"
        )
    if len(units) == 1:
        return units[0].decode()
    return torch.cat([u.decode() for u in units], 0)


def window_module_row_scale(units) -> torch.Tensor:
    """The module's per-row fp32 scale, ``[sum(rows)]``, role order."""
    units = list(units)
    if len(units) == 1:
        return units[0].row_scale
    return torch.cat([u.row_scale for u in units]).contiguous()


def window_module_linear(x: torch.Tensor, units, gemv_max: int = GEMV_MAX_M):
    """``x @ W.T`` for a whole module: ``[..., cols] -> [..., sum(rows)]``.

    Each unit runs its own ``window_linear``, and the results concatenate on
    the output axis -- which is what the stacked tile would have produced,
    because the roles do not interact.  One launch per unit, so a three-role
    ``qkv_proj`` issues three; a single stacked launch is possible (the units
    would have to share one packed plane) and is not built.

    The activation-contract caveat on ``window_linear`` applies here: below
    ``gemv_max`` this multiplies bf16 activations directly and above it
    quantises them per token to E4M3.  A lane that enables the GEMV branch is
    changing what its ``emit_route`` record must say.
    """
    units = list(units)
    if not units:
        raise GrammarError("a module needs at least one unit")
    parts = [
        window_linear(x, u.plane_words, u.offsets, u.rates, u.initial, u.code_table,
                      u.value_table, u.row_scale, u.rows, u.cols, u.window_bits,
                      u.max_rate, gemv_max)
        for u in units
    ]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
