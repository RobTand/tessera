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

The kernel is ``src/tessera/serving/csrc/window_gemv.cu`` -- the file the
serving contract publishes as the extension's ``source``, resolved through
``tessera.serving.ext.native_source_path`` so the two cannot differ -- JIT-built
through ``torch.utils.cpp_extension.load`` on first use (``ninja`` + ``nvcc``; build
directory ``$TORCH_EXTENSIONS_DIR`` or ``~/tmp/torch-ext-gemv`` -- never
``/tmp``).  Bit-exactness against ``materialize_fp8`` is proven by
:func:`decode_codes` (the same state extraction, writing the grid code) in
``tests/test_kernel_window_gemv.py``.

Scope, stated: window body at L=14, CHANNEL plane, scalar 256-code grid with a
``native`` map, no RELEASE plane, no diagonals, no rotation, **no shard start
state** (the kernel supplies ``state_{-1} = 0``, so a tensor-parallel row
shard is refused with the two lanes that do serve it named), the rates the
kernel declares (:data:`SUPPORTED_RATES`, read off ``window_gemv.cu``: R=3
would need 6-byte lanes and is refused with the fallback named), M <= 8.  The activation contract is W4A16: ``x`` is bf16, accumulation fp32.
The scope is PUBLISHED -- ``runtime_contract.json``
``native_extensions[tessera_window_gemv].lane.requires``, contract v20 --
and :func:`prepare_from_parsed` refuses through the same decision every
gate reads (:func:`lane_refusal_for_parsed` ->
``scheme.decide_lane_requirements``), so preflight and loader agree on the
same bytes by construction (#264); only the 256-code ``native`` table-build
clause is this entry point's own, because :func:`prepare_value_unit` reads
the same wire under the BF16 alphabet without it.

**Under a compiled forward.**  Every route in ``tessera.serving`` is served
eager and compiled, and this lane has been broken by exactly the shape the
first version of :func:`window_gemv` had: a Python branch on the token
dimension (``_m_tile``), a pad and a slice arithmetic on it, an
``lru_cache``d JIT build on the call path and a direct pybind call
(``vllm-compiled-forward-breaks-lane-hot-paths``; RobTand/tessera#52).  So
the GEMV is a functional ``torch.library.custom_op``
(``tessera_window_gemv::gemv``): the M tile, the refusals, the pad and the
slice all happen INSIDE the op, where ``x.shape[0]`` is a concrete integer,
the fake impl reports ``[M, rows]`` without reading it, and the op owns the
output it returns.  The extension is resolved by ``prepare_*`` at load, so
the first call -- which under vLLM's compiled forward is the trace -- never
builds.  The item tables for every M tile a unit can serve are planned at
preparation, not on first use, because planning reads the run table back to
Python and a compiled forward cannot.  ``tests/test_kernel_window_gemv.py``
carries the compiled arm: one graph, the token dim marked dynamic, M = 1..8
without a recompile.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import sys

import torch

from .alphabet import require_hardware_byte_grid
from .errors import GrammarError
from .kernel_roster import SUPPORTED_RATES, WINDOW_BITS_SUPPORTED, WINDOW_GEMV_SOURCE

__all__ = [
    "TILE_ROWS",
    "WINDOW_GEMV_SOURCE",
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
    "lane_refusal_for_parsed",
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
# SUPPORTED_RATES and WINDOW_BITS_SUPPORTED are IMPORTED (above), not declared
# here: tessera.kernel_roster reads them off serving/csrc/window_gemv.cu -- the very
# file _ext() compiles -- so the set this module refuses a unit against is the
# set the kernel instantiates.  Declaring them beside the kernel instead of
# inside it is the drift issue #145 filed: a rate could be added to the .cu and
# stay unreachable here, or dropped from it and stay advertised, with nothing
# failing.
GEMV_MAX_M = 8


def max_item_cols(mt: int) -> int:
    """Columns a block reduces at once: 1024 at M<=2, 256 above (the x tile in
    shared memory is ``cols * MT`` fp32, double-buffered)."""
    return 1024 if mt <= 2 else 256


# --------------------------------------------------------------------------
# extension
# --------------------------------------------------------------------------

def cuda_home_with_nvcc() -> "str | None":
    """The toolkit root that actually holds ``bin/nvcc``, or ``None``.

    Delegates to the serving lane's resolver, which is the one place that
    knows ``/usr/local/cuda`` can be an alternatives symlink to a toolkit
    WITHOUT an ``nvcc`` while a complete one sits beside it under
    ``/usr/local/cuda-<version>``.  That is not hypothetical: on sparky
    ``/usr/local/cuda -> cuda-13.3`` has no compiler and ``cuda-13.0`` does,
    so trusting ``cpp_extension.CUDA_HOME`` reported "no CUDA toolkit on this
    host" for a host that builds this kernel and passes every test on it.
    """
    from .serving.ext import _resolve_cuda_home

    return _resolve_cuda_home(torch)


def _ensure_toolchain_on_path() -> None:
    """``cpp_extension.load`` shells out to ninja and nvcc; a venv keeps ninja
    in its bin and the CUDA toolkit may be under a versioned root -- put both
    on PATH.

    The toolkit root is resolved UNCONDITIONALLY, because the resolver is not
    only a PATH lookup: it ADOPTS what it finds -- ``os.environ["CUDA_HOME"]``
    and ``cpp_extension.CUDA_HOME``, the module global ``load()`` actually
    builds its nvcc path from.  An nvcc already on PATH says nothing about
    that global: torch resolves ``CUDA_HOME`` to ``/usr/local/cuda`` whenever
    the path merely exists, so an alternatives symlink to a toolkit WITHOUT a
    compiler fails the build while a complete one answers ``which nvcc``
    (issue #243).  PATH presence decides only whether PATH itself needs the
    root's ``bin`` prepended.  An explicit ``CUDA_HOME``/``CUDA_PATH`` in the
    environment still wins: the resolver returns it untouched or refuses.
    """
    import shutil
    extra = []
    if shutil.which("ninja") is None:
        try:
            import ninja  # type: ignore
            extra.append(ninja.BIN_DIR)
        except Exception:
            extra.append(os.path.join(sys.prefix, "bin"))
    root = cuda_home_with_nvcc()   # always: repairs torch's cached CUDA_HOME
    if root and shutil.which("nvcc") is None:
        extra.append(os.path.join(root, "bin"))
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])


@functools.lru_cache(maxsize=None)
def _ext():
    from torch.utils.cpp_extension import load

    from tessera.serving import ext as serving_ext

    _ensure_toolchain_on_path()
    # The one source, resolved from the contract's native-extension table: the
    # path the contract publishes IS the file compiled here (#134).
    src = serving_ext.native_source_path(serving_ext.WINDOW_GEMV_MODULE_NAME)
    # The roster and the build read one file: a published source that is not
    # the roster's file would price a lane the build did not instantiate.
    if not os.path.samefile(src, WINDOW_GEMV_SOURCE):
        raise GrammarError(
            f"the contract publishes {src} as the window GEMV source but the "
            f"kernel roster reads {WINDOW_GEMV_SOURCE}; one file, one path")
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser("~/tmp/torch-ext-gemv")
    pf = int(os.environ.get("TESSERA_WINDOW_GEMV_PF", "1"))   # column chunks in flight per warp (1 or 2)
    build = os.path.join(root, "tessera_window_gemv")
    if pf != 1:
        build = build + f"_pf{pf}"
    os.makedirs(build, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    return load(
        name="tessera_window_gemv",  # literal: the contract reader reads it statically
        sources=[WINDOW_GEMV_SOURCE],  # the same file, by the check above; the roster test reads this line
        build_directory=build,
        extra_cuda_cflags=[
            "-O3", "-lineinfo", "-std=c++17",
            *(["-Xptxas", "-v"] if os.environ.get("TESSERA_WINDOW_GEMV_VERBOSE") else []),
            f"-DWINDOW_GEMV_PF={pf}",
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
    items_by_mt: dict = dataclasses.field(default_factory=dict)   # items key -> (items, max_cols)

    def __post_init__(self):
        # Plan every M tile this unit can serve NOW.  ``plan_items`` reads the
        # run table back to Python (``rep.runs.tolist()``) and ``max_cols`` is
        # a device ``max``; neither can run inside a compiled forward, and a
        # first-call plan is exactly that (the first call is the trace).  A
        # dict handed in by ``with_plan(share_from=...)`` is already full and
        # is left alone.
        for key in self.serveable_keys():
            if key not in self.items_by_mt:
                items = items_for(self.rep, self.plan, key)
                self.items_by_mt[key] = (items, int(items[:, 3].max()) if items.numel() else 0)

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

    @staticmethod
    def items_key(mt: int) -> int:
        """The item table an M tile runs on.  ``items_for`` depends on ``mt``
        only through :func:`max_item_cols` (1024 at M <= 2, 256 above), so
        there are two tables, keyed 1 and 4; M = 8 runs the 4-key table."""
        return 1 if mt <= 2 else 4

    def serveable_keys(self) -> "tuple[int, ...]":
        """The item keys this unit can run: the 4-key table needs 8 rows per
        lane, which a rate-1 column does not have (``window_gemv`` refuses it)."""
        return (1,) if 1 in self.rep.rates else (1, 4)

    def items_for(self, mt: int) -> "tuple[torch.Tensor, int]":
        """``(items, max_cols)`` for an M tile, planned at preparation and kept."""
        return self.items_by_mt[self.items_key(mt)]

    def with_plan(self, plan: Plan, *, share_from: "WindowGemvUnit | None" = None) -> "WindowGemvUnit":
        """The same unit under another launch shape.  ``share_from``: a unit of
        the same shape and plan whose item tables this one reuses (replicas)."""
        table = self.table
        if plan.table_dtype != table.dtype:
            table = self.table.float().to(plan.table_dtype)
        shared = share_from.items_by_mt if share_from is not None and share_from.plan == plan else {}
        return dataclasses.replace(self, plan=plan, table=table.contiguous(), items_by_mt=shared)


#: What ``prepare_value_unit`` says when handed a start state.  The RULE --
#: the lane reads no shard start state -- has ONE home, the published
#: predicate (``ext.WINDOW_GEMV_LANE["requires"]["start_state"]``) decided by
#: ``scheme.decide_lane_requirements``, which ``prepare_from_parsed`` runs;
#: this text is the raw-bits entry point's spelling of the same class,
#: because that entry point takes body bits rather than a unit and has no
#: wire facts to hand the core.
_NO_START_STATE = (
    "this unit carries a shard start state (INITIAL_STATE, layout.slice_unit), "
    "and the window GEMV does not take a start state: serving/csrc/window_gemv.cu "
    "supplies state_{-1} = 0 itself -- the L-bit pad that opens a wire column "
    "is not stored, and lane 0 of tile 0 reads its history from that zero pad. "
    "Taking one is a kernel change, not a packing change. Serve this shard "
    "through tessera.kernel_window (the Triton lane, which threads the state) "
    "or through the materialised FP8 path"
)


def _refuse_start_state(initial_state) -> None:
    """Fail closed on a TP shard rather than decode it from zero.

    The same answer ``lane_planes.pack_unit_for_kernel`` already gives a
    span-2 TCQ shard, and for the same reason: a shard packed against the
    pinned start decodes its first ``ceil(L/R)`` rows to plausible wrong
    weights, and a wrong weight raises nothing.
    """
    if initial_state is not None:
        raise GrammarError(_NO_START_STATE)


def lane_refusal_for_parsed(parsed) -> "str | None":
    """WHY this lane's published predicate refuses a unit, or ``None``.

    The loader's spelling of the ONE decision (#264): the facts come off the
    parsed object (``scheme.wire_facts_of_parsed``, the same reader the
    byte-time preflight uses) and the requirements are the build's own copy
    of the published block (``ext.WINDOW_GEMV_LANE`` -- the contract
    validator pins the packaged JSON equal to it), decided by
    ``scheme.decide_lane_requirements`` -- the same core the plan-time and
    byte-time gates run.  So what preflight calls READABLE, this loader
    loads, by construction; before this, the contract published four
    conditions while ``prepare_from_parsed`` refused nine, and a TP row
    shard read READABLE at preflight then fell back module by module.

    Takes a ``ParsedUnit`` or anything duck-shaped like one (the bf16 route
    hands its roles' parses through here).  A bare unit with no grid is
    refused by name rather than passed: absent evidence is not a pass.
    """
    from .serving.ext import WINDOW_GEMV_LANE, WINDOW_GEMV_MODULE_NAME
    from .serving.scheme import decide_lane_requirements, wire_facts_of_parsed

    refusals = decide_lane_requirements(
        WINDOW_GEMV_MODULE_NAME, WINDOW_GEMV_LANE["requires"],
        wire_facts_of_parsed(parsed))
    if not refusals:
        return None
    return "; ".join(refusals) + " -- this lane does not read the unit and the route's materialised path serves it"


def _check_window_bits(window_bits: int) -> None:
    if int(window_bits) not in WINDOW_BITS_SUPPORTED:
        raise GrammarError(
            f"window_bits {window_bits}: this build instantiates {WINDOW_BITS_SUPPORTED}"
        )


def prepare_from_parsed(parsed, *, plan: "Plan | None" = None, M: int = 1,
                        table_dtype: torch.dtype = torch.bfloat16) -> WindowGemvUnit:
    """A parsed E4M3 window unit over a CHANNEL plane -> the GEMV's unit.

    Refuses, naming why, everything the PUBLISHED lane predicate refuses --
    :func:`lane_refusal_for_parsed`, the one decision every gate shares
    (#264), covering the body, the plane, RELEASE overrides, diagonals,
    rotation, a shard start state, the grid's arity, the rates and the
    window width -- plus this entry point's own table-build clause: the
    E4M3 value table is ``native[window_codes]``, so the grid must be a
    256-code hardware grid with ``native`` bytes.  That clause is not part
    of the lane predicate on purpose (the same kernel reads BF16 window
    wire through :func:`prepare_value_unit`, whose scalar grid has 65536
    codes and whose table is the values themselves), and no wire the
    predicate admits on the FP8 route can violate it -- it catches a
    mis-routed parse, not a wire.  Excluded from the contract is not
    homeless: the clause is
    :func:`tessera.alphabet.require_hardware_byte_grid`, which is where all
    three of its legs live (tessera#277).
    """
    unit, grid = parsed.unit, parsed.grid
    refusal = lane_refusal_for_parsed(parsed)
    if refusal is not None:
        raise GrammarError(refusal)
    require_hardware_byte_grid(grid, purpose="the window GEMV")
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
    _ext()   # built (or found) at load, never on the first call -- see the module docstring
    return WindowGemvUnit(
        rep=rep, table=table, scale=scale, window_bits=int(unit.window_bits), plan=plan,
        codes_of_state=codes_of_state, native=native, family="e4m3",
    )


def prepare_value_unit(body_bits: torch.Tensor, rates: "tuple[int, ...]", window_bits: int,
                       values: torch.Tensor, *, scale: "torch.Tensor | None" = None,
                       plan: "Plan | None" = None, M: int = 1,
                       initial_state: "torch.Tensor | None" = None,
                       table_dtype: torch.dtype = torch.bfloat16) -> WindowGemvUnit:
    """The value family: a window body whose table holds values (bf16) rather
    than grid codes.  Same kernel; the optional per-row fp32 ``scale`` is
    applied once on the accumulated output (never folded into the table or a
    bf16 tile -- see :func:`decode_values`); ``None`` means ones.

    ``initial_state`` exists only to be refused.  This entry point takes raw
    bits rather than a unit, so a caller holding a ``layout.SlicedUnit`` would
    otherwise pass ``unit.body_bits`` and drop the shard's start state with
    nowhere for the lane to say so; the keyword is the place it says so.
    """
    _refuse_start_state(initial_state)
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
    _ext()   # at load, as prepare_from_parsed
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


def _gemv_concrete(x: torch.Tensor, words: torch.Tensor, items_1: torch.Tensor, items_4: torch.Tensor,
                   perm: torch.Tensor, table: torch.Tensor, scale: torch.Tensor,
                   tile_words: int, rows: int, window_bits: int, rpl: int, warps: int, blocks: int,
                   max_cols_1: int, max_cols_4: int, rate_one: bool, uniform: bool, ablation: int,
                   out: "torch.Tensor | None" = None) -> torch.Tensor:
    """The GEMV from a CONCRETE ``[M, K]``: the one launch site.

    Everything that reads ``M`` lives here -- the tile, the rate-1 refusal, the
    pad, the ``[:M]`` slice -- and here it is a Python integer: the custom op
    below runs this at call time, outside the trace, and the ``out=``
    instrument path calls it eagerly.
    """
    M, K = x.shape
    mt = _m_tile(M)
    if mt <= 2:
        items, max_cols = items_1, max_cols_1
    else:
        items, max_cols, rpl = items_4, max_cols_4, 8
    if rate_one and rpl != 16:
        # One gate for BOTH ways an 8-row launch reaches a rate-1 column:
        # M > 2 forces rpl=8, and at M <= 2 the plan's own rpl survives --
        # where ``run_item_if_lane<RPL=8, R=1>`` is compiled out and would
        # accumulate NOTHING, silently (issue #240).  The gate reads the
        # resolved rpl, so a supplied Plan(rpl=8) -- or default_plan(M=4)'s --
        # is refused by name at M=1/2 instead of dropping the columns.
        why = (f"M={M} runs 8 rows per lane" if mt >= 4
               else f"the plan's rpl={rpl} runs {rpl} rows per lane at M={M}")
        raise GrammarError(
            f"{why}, and a rate-1 column has no {rpl}-row lane (a byte of "
            "history is too short for L=14); serve rate-1 columns on a 16-row "
            "plan (rpl=16) or through the materialised path"
        )
    if mt != M:
        x = torch.cat([x, torch.zeros(mt - M, K, dtype=x.dtype, device=x.device)], 0)
    x = x.contiguous()
    if out is None:
        out = torch.zeros(mt, rows, dtype=torch.float32, device=x.device)
    _ext().window_gemv(
        words, items, int(tile_words), perm if not uniform else perm[:0],   # identity: no gather
        table, scale, x, out, int(window_bits), int(rpl), int(warps),
        int(blocks), int(max_cols), int(ablation),
    )
    return out if mt == M else out[:M]


# The serving shape (``serving/ops.py`` and ``kernel_window.py`` say why at
# length): a functional custom op, ``mutates_args=()``, that owns the tensor
# it returns.  Dynamo traces it as one opaque node -- no branch on the token
# dim, no pybind symbol, no ``lru_cache`` wrapper, no build lock -- and the
# fake impl below shapes the output from ``x.shape[0]`` symbolically.
@torch.library.custom_op("tessera_window_gemv::gemv", mutates_args=())
def _gemv_op(x: torch.Tensor, words: torch.Tensor, items_1: torch.Tensor, items_4: torch.Tensor,
             perm: torch.Tensor, table: torch.Tensor, scale: torch.Tensor,
             tile_words: int, rows: int, window_bits: int, rpl: int, warps: int, blocks: int,
             max_cols_1: int, max_cols_4: int, rate_one: bool, uniform: bool, ablation: int,
             ) -> torch.Tensor:
    return _gemv_concrete(x, words, items_1, items_4, perm, table, scale, tile_words, rows,
                          window_bits, rpl, warps, blocks, max_cols_1, max_cols_4, rate_one,
                          uniform, ablation)


@_gemv_op.register_fake
def _gemv_op_fake(x, words, items_1, items_4, perm, table, scale, tile_words, rows, window_bits,
                  rpl, warps, blocks, max_cols_1, max_cols_4, rate_one, uniform, ablation):
    return x.new_empty((x.shape[0], rows), dtype=torch.float32)


def _op_args(unit: WindowGemvUnit) -> tuple:
    """The unit as the op's arguments: tensors and Python scalars, nothing
    the trace has to look inside."""
    items_1, max_cols_1 = unit.items_for(1)
    if 4 in unit.items_by_mt:
        items_4, max_cols_4 = unit.items_for(4)
    else:
        items_4, max_cols_4 = items_1[:0], 0     # refused before it is read
    plan = unit.plan
    return (
        unit.rep.words, items_1, items_4, unit.rep.perm, unit.table, unit.scale,
        int(unit.rep.tile_words), int(unit.rows), int(unit.window_bits), int(plan.rpl),
        int(plan.warps), int(plan.blocks), int(max_cols_1), int(max_cols_4),
        1 in unit.rep.rates, unit.uniform,
    )


def window_gemv(unit: WindowGemvUnit, x: torch.Tensor, *, out: "torch.Tensor | None" = None,
                ablation: int = 0) -> torch.Tensor:
    """``x [M, K] bf16 -> [M, rows] fp32``: the wire read directly.

    Without ``out`` this is the functional custom op ``tessera_window_gemv::gemv``
    -- the shape a compiled forward traces (module docstring).  ``out`` is the
    bench's instrument: a caller-owned zeroed fp32 ``[M_tile, rows]`` buffer
    the launch accumulates into with atomics, run eagerly through the same
    concrete launch; never a serving path, and not traceable.
    ``ablation``: 0 the kernel; 1 no table gather; 2 no wire read; 3 no FMA;
    4 neither read -- the instruments behind the receipt, never a result.
    """
    if x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise GrammarError("x must be a CUDA bf16 [M, K] tensor")
    if x.shape[1] != unit.cols:
        raise GrammarError(f"x has {x.shape[1]} features, the unit {unit.cols} columns")
    if out is None:
        return _gemv_op(x, *_op_args(unit), int(ablation))
    return _gemv_concrete(x, *_op_args(unit), int(ablation), out=out)


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

    Folding ``s_i`` into a bf16 tile adds one rate-independent bf16 rounding
    (0.0011-0.0022 absolute measured on GLM experts).  Its share of the error
    grows as the coding error shrinks underneath it, but a share composes in
    quadrature -- the 15.4% share at R = 7 is a 1.2% error gap and a 2.4%
    squared-error gap -- and served at R = 7 the twin's KL is 1.0011x the
    route's on ``all`` and 0.9961x on ``confident``, i.e. below what the
    corpus resolves, so no fold win is claimed
    (``tessera-bf16-route-served-2026-09-02.md`` §3, #45).  The contract is
    unchanged: the tile is the table's values, and the per-row fp32
    ``unit.scale`` goes to the GEMM epilogue
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
