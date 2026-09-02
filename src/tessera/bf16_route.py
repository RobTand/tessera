"""The 16-bit route: a Tessera unit decoded to the plain BF16 tile a stock GEMM takes.

The third family, and the one with no weight-side hardware format to satisfy.
``TESSERA_NVFP4`` decodes to an E2M1 tile plus a per-16 E4M3 scale and runs
W4A4; ``TESSERA_FP8`` decodes to the ``compressed-tensors`` per-channel FP8
pair and runs W8A8; ``TESSERA_BF16`` decodes to an ordinary bfloat16 tensor
and runs the BF16 GEMM the runtime already has.  There is nothing to pack.

**Why the family exists.**  The window body's error over the E4M3 alphabet
saturates at ~0.022 out-space from R=6 upward -- the floor is the *alphabet's*
resolution, not the trellis's -- while the identical body over bf16 keeps
halving at ~1.93x per bit through R=7
(``docs/measurements/tessera16-alphabet-floor-2026-09-02.md``).  Above ~6 bpp
an 8-bit tile has nothing left to buy, so the route that lets an allocator
spend 7 bits usefully is the one whose alphabet is not the constraint.

**Two residency modes, and only one of them is the product.**

* *Resident*: hold ``materialize_bf16``'s tile and row scale.  16 bits a weight, which is
  the source precision -- so as a size claim it is nothing at all.  It exists
  as the correctness path: the tile a stock GEMM consumes, byte-identical to
  the exported stock twin, with no decoder in the serve.
* *Streamed*: hold the wire -- the packed window plane, the ``2^L`` table, the
  fp16 row words -- at the artifact's own 4-8 bpp, and decode when the module
  runs.  That is the product mode; :func:`stream_bf16` is the call,
  for the reason below.

Both are the **same bytes**: the streamed decode is checked against
``decode``'s at load and the two are bit-identical by construction --
they share ``dequantize``'s expression and the single round-to-nearest-even
that ``materialize_bf16_folded`` documents.  Pure torch throughout, no Triton: a
runtime that must not import Triton (Gridbook) imports this module.

**Do not fold the row scale if you do not have to.**  A one-tensor BF16
checkpoint must: it ships a weight and no scale, so ``bf16(code * s)`` is all
it can carry, and that rounding costs ~0.0015 in relative output error on GLM
expert rows *whatever* the coder -- 2% of the error at R=4 and **16% at R=7**
(``docs/measurements/tessera16-alphabet-floor-2026-09-02.md`` §B).  A lane
holding the wire is not stuck with it: a CHANNEL scale is one factor per
**output row**, so it commutes with the matmul, and the lane runs the stock
BF16 GEMM on the *code tile* -- exactly bf16 already, since every table entry
is a bf16 value -- and applies the row scale in fp32 as an epilogue, the same
epilogue ``lane_planes`` builds for this plane on the kernel lane.  Measured
on random activations that is 2.8e-7 relative against the fold's 1.7e-3.
:func:`stream_bf16` is that path and it is what a lane should call;
:func:`stream_bf16_folded` is the twin's rendering, kept because the twin has
to be reproducible bit for bit.
"""
from __future__ import annotations

import torch

from .alphabet import BF16_GRID, PayloadGrid
from .decode import replay_window
from .encode import EncodedUnit, grid_vector_table
from .errors import GrammarError
from .lane_planes import pack_window_planes
from .manifest import BodyKind, RotationState, ScalePlaneKind

__all__ = [
    "BF16_FAMILY",
    "StreamedBF16Unit",
    "window_table_values",
    "prepare_bf16_unit",
    "unpack_window_body",
    "window_pad_state",
    "stream_bf16",
    "stream_bf16_folded",
]

#: The name Gridbook's ``tessera_scheme`` gives this family, and the name the
#: exporter stamps on a module.  One string, spelled here, so the exporter,
#: the plugin and the allocator cannot drift apart on it.
BF16_FAMILY = "TESSERA_BF16"


def window_table_values(
    table: torch.Tensor, grid: PayloadGrid = BF16_GRID, device=None
) -> torch.Tensor:
    """The unit's ``2^L`` table as **bf16 values**: the kernel's gather table.

    On the BF16 grid a code *is* a bf16 bit pattern, so this is a
    reinterpretation and not a lookup -- the ALPHABET plane's little-endian
    uint16 elements viewed as ``torch.bfloat16`` are already this tensor, 32
    KB at L=14.  It is written as a gather through ``grid_vector_table``
    anyway, because that is the definition and the view is the optimisation;
    ``test_bf16_route`` asserts the two agree bit for bit, which is what
    entitles a kernel to take the view.
    """
    if grid.arity != 1 or grid.name != "BF16":
        raise GrammarError(f"the 16-bit route is the BF16 grid, not {grid.name}")
    values = grid_vector_table(grid, table.device)[table.long()].reshape(-1)
    out = values.to(torch.bfloat16)
    return out if device is None else out.to(device)


class StreamedBF16Unit:
    """One Linear's resident wire state for the streamed mode.

    Everything here is the artifact's own bytes, at the artifact's own rate:
    the packed window plane (``lane_planes.pack_window_planes``, the wire BODY
    permuted column-major and led by ``L`` pad bits so position 0 needs no
    boundary test -- the pad IS ``state_{-1}``, zero for a whole unit and the
    stored start state for a TP row shard), the per-column bit offsets and
    rates, the ``2^L`` bf16 table, and one fp16 word per output row times an
    fp32 global.  Nothing is
    expanded until :func:`stream_bf16_folded` is asked for a tile.
    """

    __slots__ = ("plane", "offsets", "rates", "table", "row_scale",
                 "global_scale", "window_bits", "rows", "cols")

    def __init__(self, plane, offsets, rates, table, row_scale, global_scale,
                 window_bits, rows, cols):
        self.plane = plane
        self.offsets = offsets
        self.rates = rates
        self.table = table
        self.row_scale = row_scale
        self.global_scale = float(global_scale)
        self.window_bits = int(window_bits)
        self.rows = int(rows)
        self.cols = int(cols)

    @property
    def resident_bytes(self) -> int:
        """What the streamed mode actually holds, counted and not estimated."""
        return (
            self.plane.numel() * self.plane.element_size()
            + self.offsets.numel() * self.offsets.element_size()
            + self.rates.numel() * self.rates.element_size()
            + self.table.numel() * self.table.element_size()
            + self.row_scale.numel() * self.row_scale.element_size()
        )

    def to(self, device) -> "StreamedBF16Unit":
        return StreamedBF16Unit(
            self.plane.to(device), self.offsets.to(device), self.rates.to(device),
            self.table.to(device), self.row_scale.to(device), self.global_scale,
            self.window_bits, self.rows, self.cols,
        )


def prepare_bf16_unit(
    unit: EncodedUnit, grid: PayloadGrid = BF16_GRID, device=None
) -> StreamedBF16Unit:
    """A parsed unit -> the streamed mode's resident state.

    Refuses everything the streamed decoder does not apply, rather than
    decoding it wrongly: a rotation or segment-2a diagonals are transforms
    outside the tile, and a released position overwrites a decoded code from a
    plane this path does not read.  The BF16 route's own encoder writes none
    of them; this is the check that says so out loud.
    """
    if grid.arity != 1 or grid.name != "BF16":
        raise GrammarError(f"the 16-bit route is the BF16 grid, not {grid.name}")
    if BodyKind(getattr(unit, "body", BodyKind.TCQ)) is not BodyKind.WINDOW:
        raise GrammarError(
            "the 16-bit route is the window body; a TCQ body over 65536 codes "
            "is the anchor count the encoder already refuses"
        )
    if ScalePlaneKind(unit.scale_plane) is not ScalePlaneKind.CHANNEL:
        raise GrammarError(
            "the 16-bit route folds one scale per output row into the value; "
            f"this unit carries a {ScalePlaneKind(unit.scale_plane).name} plane"
        )
    if unit.release_index.numel():
        raise GrammarError("the streamed decoder reads no RELEASE plane")
    if unit.diagonals is not None:
        raise GrammarError("the streamed decoder applies no segment-2a diagonals")
    if unit.rotation is not RotationState.NONE:
        raise GrammarError(
            f"this unit is rotated ({unit.rotation.name}); the streamed decoder "
            "applies no basis change"
        )
    if unit.window_codes is None:
        raise GrammarError("a window body needs the unit's table")
    if unit.scale_rows is None:
        raise GrammarError("a CHANNEL scale plane needs the unit's row words")
    steps, cols = unit.body_bits.shape
    # A ROW shard (``layout.slice_unit``) does not enter its columns from the
    # pinned zero register, and the window plane's ``L``-bit pad IS that start
    # state -- so threading it costs no resident byte, only the pad bits that
    # were being written as zeros anyway.  A whole unit carries ``None`` and
    # takes exactly the path it always did.  Dropped until the 2026-09-02 math
    # audit: a shard's first ``ceil(L/R)`` rows decoded to plausible wrong
    # values, silently.
    plane, offsets, rates = pack_window_planes(
        unit.body_bits, unit.rates, unit.window_bits,
        getattr(unit, "initial_state", None),
    )
    streamed = StreamedBF16Unit(
        plane=plane,
        offsets=offsets,
        rates=rates,
        table=window_table_values(unit.window_codes, grid),
        row_scale=unit.scale_rows.to(torch.float16),
        global_scale=unit.scale_global,
        window_bits=unit.window_bits,
        rows=steps * grid.arity,
        cols=cols,
    )
    return streamed if device is None else streamed.to(device)


def unpack_window_body(
    plane: torch.Tensor,
    offsets: torch.Tensor,
    rates: torch.Tensor,
    window_bits: int,
    steps: int,
) -> torch.Tensor:
    """The packed window plane -> ``[steps, cols]`` body bits.

    The inverse of ``lane_planes.pack_window_planes``, in torch: the plane is
    column-major and MSB-first with ``window_bits`` zero pad bits per column,
    so code ``t`` of column ``c`` occupies bits ``offsets[c] + window_bits +
    t*R`` to ``+R``.  Columns of one rate share a layout and are unpacked in
    one batch, which is why a mixed schedule costs a loop over the *distinct
    rates* and not over columns.
    """
    device = plane.device
    cols = int(rates.numel())
    body = torch.zeros(steps, cols, dtype=torch.int32, device=device)
    bits_all = (
        (plane.to(torch.int32).unsqueeze(1)
         >> torch.arange(7, -1, -1, device=device, dtype=torch.int32)) & 1
    ).reshape(-1)                                              # MSB-first bit array
    rate_list = sorted({int(r) for r in rates.tolist()})
    for present in rate_list:
        which = torch.nonzero(rates == present).reshape(-1)
        start = offsets[which].long() + window_bits             # [m] bit index
        index = (
            start[:, None, None]
            + (torch.arange(steps, device=device) * present)[None, :, None]
            + torch.arange(present, device=device)[None, None, :]
        )                                                       # [m, steps, R]
        got = bits_all[index.reshape(-1)].reshape(-1, steps, present)
        shifts = torch.arange(present - 1, -1, -1, device=device, dtype=torch.int32)
        body[:, which] = (got * (1 << shifts)).sum(dim=2).t().to(torch.int32)
    return body


def window_pad_state(
    plane: torch.Tensor, offsets: torch.Tensor, window_bits: int
) -> torch.Tensor:
    """The ``window_bits`` pad bits opening each column, as ``state_{-1}``.

    ``pack_window_planes`` writes a shard's start state there MSB-first (and
    zeros for a whole unit), so the state need not be carried beside the
    plane: it is *in* the plane, which is what keeps this module's "everything
    here is the artifact's own bytes" true and what makes the resident bytes
    of a shard and a whole unit identical.  Zeros come back for a whole unit,
    and ``replay_window`` with an all-zero start is the same function as
    ``replay_window`` with ``None``.
    """
    device = plane.device
    # ``pack_window_planes`` starts every column on a byte, so the pad is the
    # top ``window_bits`` bits of the column's first ``ceil(L/8)`` bytes.
    # Read those bytes and shift -- not the whole plane expanded to a bit
    # array, which ``unpack_window_body`` already pays for once.
    nbytes = (window_bits + 7) // 8
    index = (
        offsets.long()[:, None] // 8
        + torch.arange(nbytes, device=device, dtype=torch.int64)[None, :]
    )
    head = plane[index.reshape(-1)].reshape(-1, nbytes).to(torch.int64)
    shifts = torch.arange(
        8 * (nbytes - 1), -1, -8, device=device, dtype=torch.int64
    )
    return (head << shifts).sum(1) >> (8 * nbytes - window_bits)


def stream_bf16(
    streamed: StreamedBF16Unit,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """The streamed **no-fold** pair: ``(code tile bf16, row scale fp32)``.

    What a lane should call.  ``stream_bf16_folded`` folds the row scale in and
    rounds, because that is what a one-tensor BF16 checkpoint must do; a lane
    holding the wire can instead run the stock BF16 GEMM on the code tile and
    scale the output rows in fp32, since a CHANNEL scale is an output-row
    factor and commutes with the matmul.  The code tile is exact -- every
    table entry is a bf16 value -- so this path rounds the weight nowhere,
    and it avoids the ~0.0015 output-error floor the fold costs at any rate
    (``decode.materialize_bf16``).
    """
    body = unpack_window_body(
        streamed.plane, streamed.offsets, streamed.rates,
        streamed.window_bits, streamed.rows,
    )
    device = body.device
    # ``unpack_window_body`` steps over the pad to reach position 0, so the
    # start state has to be read from it separately and handed to the replay.
    start = window_pad_state(streamed.plane, streamed.offsets, streamed.window_bits)
    values = torch.zeros(
        streamed.rows, streamed.cols, dtype=torch.float32, device=device
    )
    table = streamed.table.float()
    for present in sorted({int(r) for r in streamed.rates.tolist()}):
        which = torch.nonzero(streamed.rates == present).reshape(-1)
        states = replay_window(
            body[:, which], streamed.window_bits, present, start[which]
        )
        values[:, which] = table[states]
    return (
        values.to(torch.bfloat16),
        streamed.row_scale.float() * streamed.global_scale,
    )


def stream_bf16_folded(streamed: StreamedBF16Unit) -> torch.Tensor:
    """The **twin's** rendering from a resident wire: one folded bf16 tile.

    Packed window stream -> states -> table gather -> row scale -> one
    round-to-nearest-even.  The expression is
    ``decode.materialize_bf16_folded``'s and the result is bit-identical to
    it: the code value comes off the *same* table the reader's
    ``grid_vector_table`` gather produces, the scale off the same
    ``stored.float() * global`` product broadcast down the row, and the
    rounding is that function's single ``.to(torch.bfloat16)``.

    A serving lane wants :func:`stream_bf16` instead -- this exists so that a
    twin can be reproduced from the wire and checked against the checkpoint,
    which is what makes a served comparison between them a comparison of one
    encode.  The fold's cost is the twin's, not the route's.
    """
    values, scale = stream_bf16(streamed)
    return (values.float() * scale[:, None]).to(torch.bfloat16)
