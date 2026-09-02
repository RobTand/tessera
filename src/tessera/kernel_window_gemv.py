"""The window body's GEMV: the ~4 bits/weight wire read directly, in CUDA.

The decode-phase question is whether a GEMV that reads Tessera's E4M3 wire
(window body, CHANNEL plane, ~4.07 bpp) beats the resident per-channel FP8
tensor's ``torch._scaled_mm`` at M=1 -- the wire is half the bytes, so the
bound is ~2x.  ``kernel_window.py`` (the Triton lane on the previous branch)
reads the wire bit-exactly but at 20% of memory bandwidth: 160 registers,
0.6 waves per SM, and a per-code gather through L1.  This module is the same
contract on a layout and a kernel built for bandwidth:

* **A load-time repack** (:func:`repack_window_body`) -- a bijection of the
  BODY plane's bits, done once per unit when the plugin streams it, the
  on-disk wire unchanged: rows padded to a 512-row tile with zero codes
  (appending a zero code never changes an earlier state), columns stably
  sorted by rate so every rate is one contiguous run, and each tile holding
  every column's 512 codes as one contiguous MSB-first chunk in little-endian
  u32 words.  The L-bit zero pad that opens a wire column is not stored --
  it is ``state_{-1} = 0`` by definition and the kernel supplies it.
* **One warp per column chunk**: lane ``l`` owns 16 rows (8 bytes at R=4,
  one ``uint2``), the warp's read is one contiguous 256-byte line pair, and
  the history a lane's first window needs comes from its neighbour by
  shuffle.  A state is a funnel shift by an immediate and a mask; the value
  is one shared-memory lookup (the 2^14 table as bf16, 32 KB, exact for
  E4M3 and for the bf16 value family alike); the product is one FMA per
  batch row.  Warps split an item's columns, reduce through shared memory,
  and retire one coalesced ``atomicAdd`` per output.
* **Blocks loop over items**, so the table is staged once per resident
  block, not once per tile.

The kernel is ``src/tessera/csrc/window_gemv.cu``, JIT-built through
``torch.utils.cpp_extension.load`` on first use (``ninja`` + ``nvcc``; build
directory ``$TORCH_EXTENSIONS_DIR`` or ``~/tmp/torch-ext-gemv`` -- never
``/tmp``).  Bit-exactness against ``materialize_fp8`` is proven by
:func:`decode_codes` (the same state extraction, writing the grid code) in
``tests/test_kernel_window_gemv.py``.

Scope, stated: window body at L=14, CHANNEL plane, scalar 256-code grid with a
``native`` map, no RELEASE plane, no diagonals, no rotation, rates in
{1, 2, 4} (R=3 needs 6-byte lanes and is refused with the fallback named),
M <= 8.  The activation contract is W4A16: ``x`` is bf16, accumulation fp32.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import sys

import torch

from .errors import GrammarError
from .manifest import BodyKind, RotationState, ScalePlaneKind

__all__ = [
    "TILE_ROWS",
    "SUPPORTED_RATES",
    "WINDOW_BITS_SUPPORTED",
    "GEMV_MAX_M",
    "Plan",
    "Repacked",
    "max_item_cols",
    "items_for",
    "default_plan",
    "WindowGemvUnit",
    "repack_window_body",
    "plan_items",
    "prepare_from_parsed",
    "prepare_value_unit",
    "window_gemv",
    "decode_codes",
    "decode_values",
    "decode_fp8",
    "window_linear",
    "reference_states",
]

TILE_ROWS = 512
SUPPORTED_RATES = (1, 2, 4)
WINDOW_BITS_SUPPORTED = (14,)
GEMV_MAX_M = 8


def max_item_cols(mt: int) -> int:
    """Columns a block reduces at once: 1024 at M<=2, 256 above (the x tile in
    shared memory is ``cols * MT`` fp32, double-buffered)."""
    return 1024 if mt <= 2 else 256


# --------------------------------------------------------------------------
# extension
# --------------------------------------------------------------------------

def _ensure_toolchain_on_path() -> None:
    """``cpp_extension.load`` shells out to ninja and nvcc; a venv keeps ninja
    in its bin and CUDA lives in ``/usr/local/cuda`` -- put both on PATH."""
    import shutil
    extra = []
    if shutil.which("ninja") is None:
        try:
            import ninja  # type: ignore
            extra.append(ninja.BIN_DIR)
        except Exception:
            extra.append(os.path.join(sys.prefix, "bin"))
    if shutil.which("nvcc") is None:
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME:
            extra.append(os.path.join(CUDA_HOME, "bin"))
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    _ensure_toolchain_on_path()
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "window_gemv.cu")
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser("~/tmp/torch-ext-gemv")
    build = os.path.join(root, "tessera_window_gemv")
    os.makedirs(build, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    return load(
        name="tessera_window_gemv",
        sources=[src],
        build_directory=build,
        extra_cuda_cflags=[
            "-O3", "-lineinfo", "-std=c++17",
            *(["-Xptxas", "-v"] if os.environ.get("TESSERA_WINDOW_GEMV_VERBOSE") else []),
            "-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}",
        ],
        verbose=bool(os.environ.get("TESSERA_WINDOW_GEMV_VERBOSE")),
    )


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Repacked:
    """The BODY plane in tile order (see the module docstring)."""

    words: torch.Tensor        # int32 [n_tiles * tile_words] on the device
    tile_words: int
    n_tiles: int
    rows: int
    cols: int
    rows_p: int
    perm: torch.Tensor         # int32 [cols]: permuted column -> original column
    runs: torch.Tensor         # int32 [n_runs, 4]: (rate, col0, ncols, word0), permuted columns
    rates: "tuple[int, ...]"

    @property
    def nbytes(self) -> int:
        return self.words.numel() * 4


def repack_window_body(body_bits: torch.Tensor, rates: "tuple[int, ...]") -> Repacked:
    """Wire BODY (``[rows, cols]`` codes of ``rates[c]`` bits) -> tile order.

    A bijection of the plane's bits: the same codes, MSB-first in the same
    per-column stream order, regrouped so that a warp reading rows
    ``[512g, 512g+512)`` of one column reads one contiguous run.  The only
    additions are zero codes past the last row (which change no state that
    exists) and the bookkeeping (``perm``, ``runs``) that says where each
    column went.
    """
    if body_bits.dim() != 2:
        raise GrammarError(f"body_bits must be [rows, cols], got {tuple(body_bits.shape)}")
    rows, cols = body_bits.shape
    if len(rates) != cols:
        raise GrammarError(f"{len(rates)} rates for {cols} columns")
    bad = sorted({int(r) for r in rates} - set(SUPPORTED_RATES))
    if bad:
        raise GrammarError(
            f"rates {bad} have no lane here (supported {SUPPORTED_RATES}); "
            "the materialised FP8 path serves this unit"
        )
    device = body_bits.device
    rows_p = -(-rows // TILE_ROWS) * TILE_ROWS
    n_tiles = rows_p // TILE_ROWS
    body = body_bits.to(torch.int32)
    if rows_p != rows:
        body = torch.cat([body, torch.zeros(rows_p - rows, cols, dtype=torch.int32, device=device)], 0)
    order = sorted(range(cols), key=lambda c: (int(rates[c]), c))   # stable by rate
    perm = torch.tensor(order, dtype=torch.int32, device=device)
    per_tile = []
    runs = []
    word0 = 0
    col0 = 0
    for present in sorted({int(r) for r in rates}):
        which = [c for c in order if int(rates[c]) == present]
        n = len(which)
        idx = torch.tensor(which, dtype=torch.long, device=device)
        codes = body[:, idx]                                          # [rows_p, n]
        if codes.numel() and int(codes.max()) >= (1 << present):
            raise GrammarError(f"a code exceeds {present} bits")
        cpb = 8 // present                                            # codes per byte
        grouped = codes.view(rows_p // cpb, cpb, n)
        byte = torch.zeros(rows_p // cpb, n, dtype=torch.int32, device=device)
        for i in range(cpb):
            byte |= grouped[:, i, :] << (present * (cpb - 1 - i))
        byte = byte.to(torch.uint8)                                   # stream bytes, in order
        chunk_bytes = 64 * present                                    # 512 rows * R bits / 8
        tiles = byte.view(n_tiles, TILE_ROWS // cpb, n).permute(0, 2, 1)   # [G, n, chunk_bytes]
        tiles = tiles.reshape(n_tiles, n, chunk_bytes // 4, 4).flip(-1)     # LE words, MSB-first
        per_tile.append(tiles.reshape(n_tiles, n * chunk_bytes))
        runs.append((present, col0, n, word0))
        word0 += n * 16 * present
        col0 += n
    flat = torch.cat(per_tile, 1).contiguous()                        # [G, tile_bytes]
    words = flat.reshape(-1).view(torch.int32).contiguous()
    return Repacked(
        words=words, tile_words=word0, n_tiles=n_tiles, rows=rows, cols=cols, rows_p=rows_p,
        perm=perm, runs=torch.tensor(runs, dtype=torch.int32, device=device).reshape(-1, 4),
        rates=tuple(int(r) for r in rates),
    )


@dataclasses.dataclass(frozen=True)
class Plan:
    """Launch shape.  ``rpl`` rows per lane (16 for M<=2, 8 above), ``warps``
    per block (8..16), ``blocks`` resident (the grid; blocks loop over items),
    ``cols_per_item`` the most columns a block reduces at once (capped by
    :func:`max_item_cols`), ``item_cost`` the per-item overhead in column
    equivalents the balanced planner charges, ``balanced`` whether items are
    sized so every block gets the same work (else fixed ``cols_per_item``)."""

    rpl: int = 16
    warps: int = 16
    blocks: int = 96
    cols_per_item: int = 1024
    table_dtype: torch.dtype = torch.bfloat16
    item_cost: int = 24
    balanced: bool = True


def default_plan(rows: int, cols: int, M: int = 1, *, sm_count: "int | None" = None,
                 table_dtype: torch.dtype = torch.bfloat16, warps: int = 16) -> Plan:
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 48
    rpl = 16 if M <= 2 else 8
    per_sm = 2 if table_dtype == torch.bfloat16 else 1
    # 256-column items measured 3-5% faster than 1024-column ones on the
    # 9728-row/col shapes and level elsewhere (plan sweep 2026-09-02): two
    # items per block let the second item's setup overlap the first's tail.
    return Plan(rpl=rpl, warps=warps, blocks=sm_count * per_sm, cols_per_item=min(256, max_item_cols(_m_tile(M))),
                table_dtype=table_dtype)


def plan_items(rep: Repacked, cols_per_item: int, *, grid: "int | None" = None,
               item_cost: int = 24, max_cols: "int | None" = None) -> torch.Tensor:
    """int32 ``[n_items, 8]``: (tile, rate, col0, ncols, word0, 0, 0, 0), tile-major.

    With ``grid`` given, the run's columns are cut into ``S`` equal segments
    per tile, ``S`` chosen to minimise ``ceil(items / grid) * (cols_per_segment
    + item_cost)`` -- the time a round-robin persistent grid takes when every
    item costs its columns plus a fixed overhead -- subject to the segment
    fitting the shared-memory x tile.  Without ``grid``, fixed-width items.
    """
    cap = max_cols if max_cols is not None else cols_per_item
    if not 1 <= cols_per_item <= cap:
        raise GrammarError(f"cols_per_item {cols_per_item} outside 1..{cap}")
    runs = rep.runs.tolist()
    items = []
    for rate, col0, n, word0 in runs:
        if grid is None:
            width = cols_per_item
        else:
            best = None
            for S in range(-(-n // cols_per_item), n + 1):
                per_block = -(-(rep.n_tiles * S) // grid)
                cost = per_block * (-(-n // S) + item_cost)
                if best is None or cost < best[0]:
                    best = (cost, S)
                if per_block == 1 and S >= grid:
                    break
            width = -(-n // best[1])
        for g in range(rep.n_tiles):
            for c in range(0, n, width):
                k = min(width, n - c)
                items.append((g, rate, col0 + c, k, word0 + c * 16 * rate, 0, 0, 0))
    # tile-major order so blocks working at the same time read the same region
    items.sort(key=lambda t: (t[0], t[2]))
    return torch.tensor(items, dtype=torch.int32, device=rep.words.device).reshape(-1, 8)


def items_for(rep: Repacked, plan: Plan, mt: int) -> torch.Tensor:
    cap = min(plan.cols_per_item, max_item_cols(mt))
    return plan_items(rep, cap, grid=plan.blocks if plan.balanced else None,
                      item_cost=plan.item_cost, max_cols=max_item_cols(mt))


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

@dataclasses.dataclass
class WindowGemvUnit:
    rep: Repacked
    table: torch.Tensor            # [2^L] bf16 or fp32 values (what a state decodes to)
    scale: torch.Tensor            # [rows] fp32 (ones for the value family)
    window_bits: int
    plan: Plan
    codes_of_state: "torch.Tensor | None" = None   # [2^L] u8 grid codes (E4M3 family)
    native: "torch.Tensor | None" = None           # [256] u8 (E4M3 family)
    family: str = "e4m3"
    items_by_mt: dict = dataclasses.field(default_factory=dict)   # M tile -> (items, max_cols)

    @property
    def rows(self) -> int:
        return self.rep.rows

    @property
    def cols(self) -> int:
        return self.rep.cols

    @property
    def items(self) -> torch.Tensor:
        return self.items_for(1)[0]

    @property
    def uniform(self) -> bool:
        return len(set(self.rep.rates)) == 1

    def items_for(self, mt: int) -> "tuple[torch.Tensor, int]":
        """``(items, max_cols)`` for an M tile, planned once and kept (a
        per-call plan or a device-side ``max`` would cost more than the GEMV)."""
        key = 1 if mt <= 2 else mt
        if key not in self.items_by_mt:
            items = items_for(self.rep, self.plan, key)
            self.items_by_mt[key] = (items, int(items[:, 3].max()) if items.numel() else 0)
        return self.items_by_mt[key]

    def with_plan(self, plan: Plan, *, share_from: "WindowGemvUnit | None" = None) -> "WindowGemvUnit":
        """The same unit under another launch shape.  ``share_from``: a unit of
        the same shape and plan whose item tables this one reuses (replicas)."""
        table = self.table
        if plan.table_dtype != table.dtype:
            table = self.table.float().to(plan.table_dtype)
        shared = share_from.items_by_mt if share_from is not None and share_from.plan == plan else {}
        return dataclasses.replace(self, plan=plan, table=table.contiguous(), items_by_mt=shared)


def _check_window_bits(window_bits: int) -> None:
    if int(window_bits) not in WINDOW_BITS_SUPPORTED:
        raise GrammarError(
            f"window_bits {window_bits}: this build instantiates {WINDOW_BITS_SUPPORTED}"
        )


def prepare_from_parsed(parsed, *, plan: "Plan | None" = None, M: int = 1,
                        table_dtype: torch.dtype = torch.bfloat16) -> WindowGemvUnit:
    """A parsed E4M3 window unit over a CHANNEL plane -> the GEMV's unit.

    Refuses, naming why, what this lane does not read: another body or
    plane, a RELEASE plane, diagonals, rotation, a grid without ``native``
    bytes, a rate outside {1, 2, 4}, a window other than L=14.
    """
    unit, grid = parsed.unit, parsed.grid
    body = BodyKind(getattr(unit, "body", BodyKind.TCQ))
    if body is not BodyKind.WINDOW:
        raise GrammarError(f"the window GEMV reads a WINDOW body, this unit is {body.name}")
    plane = getattr(unit, "scale_plane", ScalePlaneKind.S6B)
    if plane is not ScalePlaneKind.CHANNEL:
        raise GrammarError(f"the window GEMV reads a CHANNEL plane, this unit carries {plane.name}")
    if unit.release_index.numel():
        raise GrammarError(f"the unit carries {unit.release_index.numel()} RELEASE overrides; not read here")
    if getattr(unit, "diagonals", None) is not None:
        raise GrammarError("the unit carries diagonals; not read here")
    if RotationState(getattr(unit, "rotation", RotationState.NONE)) is not RotationState.NONE:
        raise GrammarError("the unit is rotated; not read here")
    if grid.arity != 1 or grid.native is None or grid.size != 256:
        raise GrammarError(f"the window GEMV needs a scalar 256-code hardware grid, got {grid.name}")
    if unit.window_codes is None:
        raise GrammarError("a window body needs the unit's table")
    _check_window_bits(unit.window_bits)
    if unit.scale_rows is None:
        raise GrammarError("a CHANNEL plane needs scale_rows")
    device = unit.body_bits.device
    if device.type != "cuda":
        device = torch.device("cuda")
    codes_of_state = unit.window_codes.to(device=device, dtype=torch.uint8).contiguous()
    if codes_of_state.numel() != 1 << unit.window_bits:
        raise GrammarError(
            f"the window table holds {codes_of_state.numel()} entries, window_bits "
            f"{unit.window_bits} needs {1 << unit.window_bits}"
        )
    native = torch.tensor(grid.native, dtype=torch.uint8, device=device)
    value_of_code = native.view(torch.float8_e4m3fn).float()         # [256]
    table = value_of_code[codes_of_state.long()].to(table_dtype).contiguous()
    scale = (unit.scale_rows.to(device).float() * float(unit.scale_global)).reshape(-1).contiguous()
    rep = repack_window_body(unit.body_bits.to(device), tuple(unit.rates))
    if scale.numel() != rep.rows:
        raise GrammarError(f"{scale.numel()} row scales for {rep.rows} rows")
    if plan is None:
        plan = default_plan(rep.rows, rep.cols, M, table_dtype=table_dtype)
    return WindowGemvUnit(
        rep=rep, table=table, scale=scale, window_bits=int(unit.window_bits), plan=plan,
        codes_of_state=codes_of_state, native=native, family="e4m3",
    )


def prepare_value_unit(body_bits: torch.Tensor, rates: "tuple[int, ...]", window_bits: int,
                       values: torch.Tensor, *, scale: "torch.Tensor | None" = None,
                       plan: "Plan | None" = None, M: int = 1,
                       table_dtype: torch.dtype = torch.bfloat16) -> WindowGemvUnit:
    """The value family: a window body whose table holds values (bf16) rather
    than grid codes.  Same kernel; the optional per-row fp32 ``scale`` is
    applied once on the accumulated output (never folded into the table or a
    bf16 tile -- see :func:`decode_values`); ``None`` means ones."""
    _check_window_bits(window_bits)
    if values.numel() != 1 << window_bits:
        raise GrammarError(f"{values.numel()} table values for window_bits {window_bits}")
    device = body_bits.device if body_bits.is_cuda else torch.device("cuda")
    rep = repack_window_body(body_bits.to(device), tuple(rates))
    table = values.to(device=device, dtype=torch.float32).to(table_dtype).contiguous()
    if scale is None:
        scale = torch.ones(rep.rows, dtype=torch.float32, device=device)
    scale = scale.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
    if plan is None:
        plan = default_plan(rep.rows, rep.cols, M, table_dtype=table_dtype)
    return WindowGemvUnit(
        rep=rep, table=table, scale=scale, window_bits=int(window_bits), plan=plan, family="value",
    )


# --------------------------------------------------------------------------
# ops
# --------------------------------------------------------------------------

def _m_tile(M: int) -> int:
    if M <= 1:
        return 1
    if M <= 2:
        return 2
    if M <= 4:
        return 4
    if M <= 8:
        return 8
    raise GrammarError(f"M={M} exceeds the GEMV's {GEMV_MAX_M}; the materialised path serves prefill")


def window_gemv(unit: WindowGemvUnit, x: torch.Tensor, *, out: "torch.Tensor | None" = None,
                ablation: int = 0) -> torch.Tensor:
    """``x [M, K] bf16 -> [M, rows] fp32``: the wire read directly.

    ``out`` may be a caller-owned zeroed fp32 ``[M_tile, rows]`` buffer (the
    op accumulates into it with atomics); otherwise one is zeroed here.
    ``ablation``: 0 the kernel; 1 no table gather; 2 no wire read; 3 no FMA;
    4 neither read -- the instruments behind the receipt, never a result.
    """
    if x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise GrammarError("x must be a CUDA bf16 [M, K] tensor")
    M, K = x.shape
    if K != unit.cols:
        raise GrammarError(f"x has {K} features, the unit {unit.cols} columns")
    mt = _m_tile(M)
    if mt >= 4 and 1 in unit.rep.rates:
        raise GrammarError(
            f"M={M} runs 8 rows per lane, and a rate-1 column has no 8-row lane "
            "(a byte of history is too short for L=14); the materialised path serves this batch"
        )
    if mt != M:
        x = torch.cat([x, torch.zeros(mt - M, K, dtype=x.dtype, device=x.device)], 0)
    x = x.contiguous()
    plan = unit.plan
    rpl = plan.rpl if mt <= 2 else 8
    items, max_cols = unit.items_for(mt)
    perm = unit.rep.perm if not unit.uniform else unit.rep.perm[:0]     # identity: no gather
    if out is None:
        out = torch.zeros(mt, unit.rows, dtype=torch.float32, device=x.device)
    _ext().window_gemv(
        unit.rep.words, items, int(unit.rep.tile_words), perm,
        unit.table, unit.scale, x, out, int(unit.window_bits), int(rpl), int(plan.warps),
        int(plan.blocks), max_cols, int(ablation),
    )
    return out if mt == M else out[:M]


def window_linear(unit: WindowGemvUnit, x: torch.Tensor) -> torch.Tensor:
    """The module seam: ``x [..., K] bf16 -> [..., rows] bf16``."""
    lead = x.shape[:-1]
    y = window_gemv(unit, x.reshape(-1, x.shape[-1]).to(torch.bfloat16))
    return y.to(torch.bfloat16).reshape(*lead, unit.rows)


def decode_codes(unit: WindowGemvUnit) -> torch.Tensor:
    """Debug: the grid code at every position, ``uint8 [rows, cols]``, from the
    repacked words through the kernel's own state extraction."""
    if unit.codes_of_state is None:
        raise GrammarError("the value family has no grid codes to decode")
    out = torch.empty(unit.rows, unit.cols, dtype=torch.uint8, device=unit.rep.words.device)
    _ext().window_decode(
        unit.rep.words, int(unit.rep.tile_words), int(unit.rep.n_tiles), unit.rep.runs,
        unit.rep.perm, unit.codes_of_state, int(unit.window_bits), out,
    )
    return out


def decode_values(unit: WindowGemvUnit) -> torch.Tensor:
    """The value family's prefill tile: the **raw** table value at every
    position, ``bf16 [rows, cols]``, the row scale NOT folded in.

    Folding ``s_i`` into a bf16 tile adds a rate-independent absolute error
    floor (0.0011-0.0022 measured on GLM experts, 16-28% of the total at
    R=7-8), so the contract is: the tile is the table's values, and the
    per-row fp32 ``unit.scale`` goes to the GEMM epilogue
    (``y_i = s_i * sum_k t_ik x_k``), exactly as the fused GEMV applies it
    once on the accumulated fp32 output.
    """
    if unit.family != "value":
        raise GrammarError("decode_values is the value family's tile; use decode_fp8 for E4M3")
    table = unit.table if unit.table.dtype == torch.bfloat16 else unit.table.to(torch.bfloat16)
    out = torch.empty(unit.rows, unit.cols, dtype=torch.bfloat16, device=unit.rep.words.device)
    _ext().window_decode(
        unit.rep.words, int(unit.rep.tile_words), int(unit.rep.n_tiles), unit.rep.runs,
        unit.rep.perm, table.contiguous(), int(unit.window_bits), out,
    )
    return out


def decode_fp8(unit: WindowGemvUnit) -> "tuple[torch.Tensor, torch.Tensor]":
    """Debug: ``(bytes uint8 [rows, cols], scale fp32 [rows])`` -- what
    ``materialize_fp8`` returns, from the kernel's decode."""
    if unit.native is None:
        raise GrammarError("the value family has no native bytes")
    return unit.native[decode_codes(unit).long()], unit.scale


def reference_states(body_bits: torch.Tensor, rates: "tuple[int, ...]", window_bits: int) -> torch.Tensor:
    """The definition, one step at a time: ``state_t = ((state_{t-1} << R) | bits_t) mod 2^L``
    from ``state_{-1} = 0`` down every column.  int64 ``[rows, cols]``.  Slow; for tests."""
    rows, cols = body_bits.shape
    mask = (1 << window_bits) - 1
    rate = torch.tensor(rates, dtype=torch.int64, device=body_bits.device)
    bits = body_bits.to(torch.int64)
    states = torch.empty(rows, cols, dtype=torch.int64, device=body_bits.device)
    state = torch.zeros(cols, dtype=torch.int64, device=body_bits.device)
    for t in range(rows):
        state = ((state << rate) | bits[t]) & mask
        states[t] = state
    return states
