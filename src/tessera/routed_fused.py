"""The fused routed window MoE lane (tessera#640).

One persistent, warp-specialised CUDA kernel (``serving/csrc/routed_fused_
window.cu``) serves a routed expert stack's gate/up projection together with
its SwiGLU epilogue, and a second launch of the same kernel serves the down
projection into a route-sorted buffer that a fixed-order per-token sum
reduces.  It computes the same functions of the wire as the Triton grouped
window GEMM behind :class:`tessera.native_window_moe.NativeWindowMoE` -- the
FOLDED arithmetic for the BF16 (value) family and the epilogue arithmetic for
the E4M3 family, per-token native A quantisation, the bf16 route boundary
before the reduction -- with three differences a reader must know:

* Scheduling is route-count sized: work items are ``(expert, n-block,
  superblock)`` triples counted on the device, claimed through one device
  counter that the caller zeroes in-stream, so no host synchronisation exists
  and a captured forward replays.
* Every weight is decoded once per (item, superblock) into a shared-memory B
  tile and reused across up to 64 routes; the gate and up tiles share one
  read of the activation tile.
* The down reduction is DETERMINISTIC: each route's bf16-rounded, weighted
  row is written by route position and the ``top_k`` rows of a token are
  added in fixed order in fp32.  Two runs are bitwise equal; the legacy
  kernel's fp32 ``atomic_add`` is scheduling-order dependent.

It is a NEW launch identity: ``scheme.ROUTED_FUSED_WINDOW_SYMBOL`` with the
decoders ``native_routed_fused_window`` (E4M3, epilogue) and
``native_routed_fused_window_folded`` (BF16, folded), both lane-bearing
``scheme.ROUTE_LAUNCHES`` rows since contract v42, where the four window routed
cells name them.  The compact adapter's pairs stay attested and stay the dispatch for every stack
this lane refuses (:func:`fused_routed_window_supported`) and for
``TESSERA_ROUTED_FUSED=0``.

THE DENSE CASE (contract v43).  A dense window Linear is the E = 1, top_k = 1,
unweighted case of the down projection, and the same kernel serves it: one
launch per role of a merged Linear (``routed_fused_kernel<FP8, 2, DENSE>``),
row ``m`` of ``x`` as route ``m``, the role's rows written into their column
slice of the module's output, and -- when ``ceil(M / 64) * rows / 128`` items
would leave SMs idle, which is every decode shape -- the K range split ``S``
ways into an fp32 workspace that a fixed-order reduce sums before the one
epilogue (:func:`dense_k_split` states the model that picks ``S``).  It is
its own launch identity, ``tessera::fused_window_dense`` (the functional
custom op in ``serving.native_window``) with the decoders
``native_fused_window_dense`` (E4M3, epilogue) and
``native_fused_window_dense_folded`` (BF16, folded), lane-bearing rows on the
same two extensions.  :func:`fused_dense_window_supported` is the per-role
predicate; the Triton ``tessera::window_gemm_dense`` stays the dispatch for
every module it refuses and for ``TESSERA_DENSE_FUSED=0``.

The kernel is JIT-built through ``torch.utils.cpp_extension`` into three
libraries, the ``native_extensions`` entries the contract publishes for this
source: ``tessera_routed_fused_value`` (BF16 tables, bf16 ``mma.sync``),
``tessera_routed_fused_e4m3`` (the E4M3 family widened to f16 for
``mma.sync.m16n8k16.f16``) and ``tessera_routed_fused_mma_e4m3`` (the E4M3
family on its own instruction, ``mma.sync.m16n8k32.e4m3.e4m3.f32``: the E4M3
byte table goes to the B tile as is, the e4m3 activation is staged
unconverted, 8-bit tables and tiles).  The two E4M3 libraries compute the same
exact products and differ by fp32 accumulation order only, so each is its
own decoder identity (``native_routed_fused_window_e4m3mma``,
``native_fused_window_dense_e4m3mma``); ``TESSERA_FUSED_E4M3_MMA`` picks one
per process (:func:`library_for`).
"""

from __future__ import annotations

import dataclasses
import functools
import glob as _glob
import logging
import os

import torch

from .errors import GrammarError

__all__ = [
    "ENV_TOGGLE",
    "ENV_E4M3_MMA",
    "ENV_TOGGLE_DENSE",
    "ENV_WIDE",
    "DENSE_RATE_MAX",
    "LIBRARIES",
    "FusedDenseWindowRole",
    "FusedRoutedWindowMoE",
    "MODULE_NAME_E4M3",
    "MODULE_NAME_E4M3MMA",
    "MODULE_NAME_VALUE",
    "SOURCE",
    "compose_dense_table",
    "compose_dense_table16",
    "compose_table",
    "compose_table16",
    "compose_table8",
    "dense_forward",
    "dense_k_split",
    "dense_split_max",
    "dense_rates",
    "fused_dense_window_enabled",
    "fused_dense_window_supported",
    "fused_routed_window_supported",
    "fused_routed_window_enabled",
    "library_for",
    "prepare_dense_role",
    "superblock_rows",
    "words_by_expert",
]

log = logging.getLogger(__name__)

#: ``TESSERA_ROUTED_FUSED=0`` keeps the compact Triton adapter for every stack;
#: unset or ``1`` takes this lane wherever :func:`fused_routed_window_supported`
#: admits the stack.
ENV_TOGGLE = "TESSERA_ROUTED_FUSED"
#: ``TESSERA_DENSE_FUSED=0`` keeps the Triton ``tessera::window_gemm_dense``
#: for every dense module; unset or ``1`` takes this kernel's dense identity
#: wherever :func:`fused_dense_window_supported` admits every role.  Its own
#: toggle so the two identities can be measured against their predecessors
#: independently.
ENV_TOGGLE_DENSE = "TESSERA_DENSE_FUSED"
#: The three JIT module names.  Literals: the contract's native-extension
#: scanner reads the ``load(name=...)`` sites statically.
MODULE_NAME_VALUE = "tessera_routed_fused_value"
MODULE_NAME_E4M3 = "tessera_routed_fused_e4m3"
MODULE_NAME_E4M3MMA = "tessera_routed_fused_mma_e4m3"
#: Library key -> ``(module name, family, mma8)``.  ``mma8`` is the E4M3
#: instruction (``TESSERA_ROUTED_FUSED_MMA8=1``): 8-bit tables and tiles.
LIBRARIES = {
    "value": (MODULE_NAME_VALUE, "value", False),
    "e4m3": (MODULE_NAME_E4M3, "e4m3", False),
    "e4m3mma": (MODULE_NAME_E4M3MMA, "e4m3", True),
}
#: Which tensor-core instruction the E4M3 family's fused launches (routed and
#: dense) take in this process: ``e4m3`` (the default) runs
#: ``mma.sync.m16n8k32.e4m3`` on the bytes (``tessera_routed_fused_mma_e4m3``),
#: ``f16`` widens each E4M3 byte to f16 for ``mma.sync.m16n8k16``
#: (``tessera_routed_fused_e4m3``).  The two compute the same exact products
#: and differ only in fp32 summation order, and the E4M3 instruction does twice
#: the work per instruction on half the shared-memory bytes, so it is the
#: default; ``f16`` stays selectable for A/Bs.  Read when an adapter or role is
#: prepared; the prepared object carries its library and stamps that library's
#: decoder.
ENV_E4M3_MMA = "TESSERA_FUSED_E4M3_MMA"
E4M3_MMA_CHOICES = ("f16", "e4m3")
E4M3_MMA_DEFAULT = "e4m3"
#: The one source, as ``ext.NATIVE_EXTENSIONS`` publishes it.
SOURCE = "csrc/routed_fused_window.cu"

# The kernel's geometry, restated for the support predicate (the library's
# attributes of the same names are checked against these at load).
BM = 64
#: The wide superblock (``BM_WIDE`` in the kernel, tessera#741): 128 routes
#: per item, so one decoded B tile feeds twice the rows.  The E4M3 family's
#: one-table launch (routed down, dense) has it in both libraries, and its
#: gate/up launch on the E4M3 instruction (:func:`has_width`).
#: :func:`superblock_rows` picks the width per launch; either width gives the
#: same output bits.
BM_WIDE = 128
HALF = 64
BN = 128
BK = 32
#: The dense identity's row quantum: a role's last ``BN`` block may be partial
#: (the N-tail, tessera#750 WP2), and the epilogue stores at most four columns
#: at a time, so a role's rows are a multiple of 4.  The wire is padded to
#: whole 512-row tiles, so the partial block's pad rows read words the repack
#: wrote (zeros) and are never stored.
DENSE_ROW_QUANTUM = 4
#: The column rates the ROUTED-EXPERT launches (gate/up and down) decode:
#: every rate of the window grammar up to 8, so a stack's one- or two-rate run
#: table (the two rates bracketing its root) is read as the wire lays it out.
#: ``RATE_MAX`` sizes the routed launches' per-column word slot; the kernel
#: publishes it as ``ROUTED_RATE_MAX``.
RATE_MIN = 1
RATE_MAX = 8
RATES = tuple(range(RATE_MIN, RATE_MAX + 1))
SLOT_WORDS_MAX = 2 * RATE_MAX         # the rate-8 word-stage slot, int32 words per (half, column)
#: The largest rate the DENSE launch decodes, per window family (the kernel's
#: ``RATE_MAX``).  A code fits its 14-bit window, so the value family's BF16
#: grid reads rates 1..14 (tessera#750 item 4: rates 15 and 16 widen the
#: window to 15 and 16 bits, whose 64 KB and 128 KB tables leave the one-table
#: block one word stage and none); the E4M3 grids' codes are 8 bits.  The
#: one-table launch fits the rate-14 slot at three word stages (80,144 B).
DENSE_RATE_MAX = {"value": 14, "e4m3": 8}
BDESC_INTS = 12                       # int32 words per 32-column block descriptor
WINDOW_BITS = 14
TABLE_ENTRIES = 1 << WINDOW_BITS
MIN_COLS = 4 * BK
#: The kernel's shared-memory layout, restated for the support predicates (the
#: library's attributes are checked against these at load): the word stages
#: come last and are sized per launch by the stack's rates, so the fixed part
#: is ``SMEM_FIXED[mode]`` (two 32 KB tables for gate/up, one for down/dense,
#: and the two-run block-descriptor ring -- DRING_STAGES chunks of BDESC_INTS
#: int32 per projection) and a launch needs ``SMEM_FIXED[mode] + stages * 2 * BK
#: * slot_words * 4`` bytes at :func:`word_stages` stages.  A block on sm_121
#: may opt in to 101,376 B, so at WORD_STAGES the 16-bit gate/up launch fits
#: slots up to 12 words (rates <= 6) and the down/dense launch every rate; the
#: 16-word slot of rates 7 and 8 takes the gate/up launch at WORD_STAGES_MIN
#: (99,792 B), where each chunk's words are issued one chunk ahead of their
#: decode instead of two.
WORD_STAGES = 3
WORD_STAGES_MIN = 2
DRING_STAGES = 4
SMEM_FIXED = {0: 91_600, 1: 91_600, 2: 58_640}
#: The staged stream history on the E4M3 instruction: one int32 per (half,
#: column) per word stage (``PREV_STAGED`` in ``routed_fused_window.cu``).
PREV_REGION_BYTES_MMA8 = WORD_STAGES * 2 * BK * 4
#: The same fixed part on the E4M3 instruction: 16 KB byte tables, 8-bit A and
#: B stages, and the staged stream history.  The word stages are the same bytes.
SMEM_FIXED_MMA8 = {0: 47_312, 1: 47_312, 2: 30_736}
#: The per-block dynamic shared memory sm_121 (GB10, the contract's target
#: platform) lets a kernel opt in to -- ``cudaDevAttrMaxSharedMemoryPerBlock
#: Optin`` there; the library reads the live value per device, this is the
#: figure the PUBLISHED predicate is derived from.
SM121_MAX_DYNAMIC_SMEM = 101_376


def _round_up_4(words: int) -> int:
    return -(-int(words) // 4) * 4


def chunk_words(rate: int) -> int:
    """int32 words one column at ``rate`` occupies per 512-row tile."""
    return 16 * int(rate)


def dense_rates(family: str) -> "tuple[int, ...]":
    """The column rates the family's dense launch decodes: 1..14 on the value
    family, 1..8 on the E4M3 family."""
    return tuple(range(RATE_MIN, DENSE_RATE_MAX[family] + 1))


def slot_words_for_rate(rate: int) -> int:
    """The word-stage slot one column at ``rate`` needs: its ``2 * rate`` words,
    plus two at an odd rate -- a 64-row half at an odd rate starts on an
    8-byte boundary when its index is odd, and the kernel copies it in 16-byte
    pieces from the aligned word pair before it (``issue_words`` in
    ``routed_fused_window.cu``), which lands in the slot too.  The decode reads
    no word past the half."""
    rate = int(rate)
    return 2 * rate + (2 if rate & 1 else 0)


def slot_words_for_pair(pair: torch.Tensor) -> int:
    """The launch's slot: the larger of the two rates' slots, rounded up to a
    multiple of 4 (16-byte copies stay aligned), at least 4."""
    r_lo, _c0, _n_lo, _w0, r_hi, _c1, n_hi, _w1 = (int(v) for v in pair.reshape(8).tolist())
    need = max(slot_words_for_rate(r_lo), slot_words_for_rate(r_hi) if n_hi > 0 else 0, 4)
    return _round_up_4(need)


def _smem_bytes_at_stages(mode: int, slot_words: int, stages: int, *, mma8: bool = False) -> int:
    fixed = SMEM_FIXED_MMA8 if mma8 else SMEM_FIXED
    return fixed[int(mode)] + int(stages) * 2 * BK * int(slot_words) * 4


def word_stages(mode: int, slot_words: int, *, mma8: bool = False) -> int:
    """The word stages the launch of ``mode`` cycles through at
    ``slot_words``-word slots (``word_stages`` in the kernel): WORD_STAGES
    where they fit sm_121's opt-in block, else WORD_STAGES_MIN.  The library
    publishes both constants and is checked against them at load."""
    fits = _smem_bytes_at_stages(mode, slot_words, WORD_STAGES, mma8=mma8) <= SM121_MAX_DYNAMIC_SMEM
    return WORD_STAGES if fits else WORD_STAGES_MIN


def smem_bytes(mode: int, slot_words: int, *, mma8: bool = False) -> int:
    """Dynamic shared memory the launch of ``mode`` needs at ``slot_words``-word
    slots, at its :func:`word_stages`."""
    return _smem_bytes_at_stages(mode, slot_words, word_stages(mode, slot_words, mma8=mma8), mma8=mma8)


def library_for(family: str) -> str:
    """The library key the family's fused launches take in this process.

    The value family has one library.  The E4M3 family takes
    ``tessera_routed_fused_e4m3`` when ``TESSERA_FUSED_E4M3_MMA=f16`` and
    ``tessera_routed_fused_mma_e4m3`` otherwise (unset or ``e4m3``); any other
    value is refused by name rather than read as the default.
    """
    if family != "e4m3":
        return family
    choice = os.environ.get(ENV_E4M3_MMA, E4M3_MMA_DEFAULT)
    if choice not in E4M3_MMA_CHOICES:
        raise GrammarError(f"{ENV_E4M3_MMA}={choice!r}; one of {E4M3_MMA_CHOICES}")
    return "e4m3mma" if choice == "e4m3" else "e4m3"


def library_mma8(library: str) -> bool:
    return LIBRARIES[library][2]


def a_region_bytes(bm: int, *, mma8: bool = False) -> int:
    """The A region (two tiles of ``bm`` rows) of the kernel's layout."""
    return 2 * int(bm) * BK * (1 if mma8 else 2)


def launch_smem_bytes(mode: int, slot_words: int, *, mma8: bool = False, bm: int = BM) -> int:
    """The dynamic shared memory the launch takes at ``bm``-route
    superblocks: :func:`smem_bytes` with the A region at ``bm`` rows."""
    return smem_bytes(mode, slot_words, mma8=mma8) + a_region_bytes(bm, mma8=mma8) - a_region_bytes(BM, mma8=mma8)


def has_width(library: str, mode: int, bm: int) -> bool:
    """Whether the launch of ``mode`` exists at ``bm``-route superblocks in
    ``library`` (``has_width`` in the kernel): ``BM`` everywhere; ``BM_WIDE``
    in the E4M3 family for the one-table launch, and for gate/up too on the
    E4M3 instruction, whose 8-bit tiles leave the room.  Where the width
    exists it fits every rate the launch decodes on sm_121 (the kernel
    asserts it per pair)."""
    _module, family, mma8 = LIBRARIES[library]
    return int(bm) == BM or (int(bm) == BM_WIDE and family == "e4m3" and (int(mode) == 2 or mma8))


#: ``TESSERA_ROUTED_FUSED_WIDE``: ``auto`` (unset) lets :func:`superblock_rows`
#: pick the width from the launch's rows; ``0`` keeps 64-route superblocks and
#: ``1`` takes 128 wherever the launch has them -- for measuring the two widths
#: against each other, which give the same output bits.
ENV_WIDE = "TESSERA_ROUTED_FUSED_WIDE"
#: The smallest routed step, in tokens, that ``auto`` runs at 128-route
#: superblocks on the E4M3 instruction's library.  Measured over the T8R
#: release's three routed rungs with recorded prefill routing
#: (afetch-ab-20260930T063741Z, tessera#741): 128 routes took 7-17% off the
#: kernel at 2048 tokens on every rung and was within 2.5% of 64 at 512, so
#: 2048 is the smallest measured step where it wins.
WIDE_MIN_ROWS = 2048
#: The threshold where no A/B has measured the width -- a dense role's rows of
#: ``x`` on either library, and the f16 instruction's routed down launch -- so
#: ``auto`` never takes it there.
WIDE_UNMEASURED = 1 << 30


def superblock_rows(library: str, mode: int, rows: int, *, dense: bool = False) -> int:
    """The routes per superblock of one launch: ``BM`` or ``BM_WIDE``.

    ``mode`` is the kernel mode (0/1 gate/up, 2 routed down or dense) and
    ``rows`` the step's tokens (routed) or, with ``dense``, rows of ``x``.  A
    pure function of host-visible integers and the environment -- no device
    read -- so a captured forward records the width with its shapes.
    """
    if not has_width(library, mode, BM_WIDE):
        return BM
    want = os.environ.get(ENV_WIDE, "auto")
    if want == "0":
        return BM
    if want == "1":
        return BM_WIDE
    if want != "auto":
        raise GrammarError(f"{ENV_WIDE}={want!r}: expected auto, 0 or 1")
    floor = WIDE_MIN_ROWS if library_mma8(library) and not dense else WIDE_UNMEASURED
    return BM_WIDE if int(rows) >= floor else BM


#: The rates a ROUTED-EXPERT stack (the two-table gate/up launch, MODE 0/1)
#: reaches on the target platform: those whose one-rate slot fits sm_121's
#: opt-in shared memory at the launch's :func:`word_stages` -- every rate
#: since the 16-word slot of rates 7 and 8 runs at WORD_STAGES_MIN (at
#: WORD_STAGES only slots 8 and 12, rates 1..6, fit; contract v45 published
#: that).  A two-rate pair's slot is the larger rate's, so the set is closed
#: under bracketing.  Published as the fused lanes' ``column_rates_routed_moe``
#: (``serving.ext.ROUTED_FUSED_LANE_REQUIRES``, tessera#694); the down/dense
#: one-table launch reads every rate in ``RATES``.  Derived, not typed: the day
#: the layout changes, this changes with it and the contract's pin fails until
#: the JSON follows.
ROUTED_LANE_RATES = tuple(
    r for r in RATES
    if smem_bytes(0, _round_up_4(slot_words_for_rate(r))) <= SM121_MAX_DYNAMIC_SMEM)


def routed_lane_rates(library: str) -> "tuple[int, ...]":
    """:data:`ROUTED_LANE_RATES` for one library: every rate, on the E4M3
    instruction at three word stages (its 16 KB tables leave room for the
    rate-8 slot), on the 16-bit libraries at two for rates 7 and 8."""
    mma8 = library_mma8(library)
    return tuple(r for r in RATES
                 if smem_bytes(0, _round_up_4(slot_words_for_rate(r)), mma8=mma8) <= SM121_MAX_DYNAMIC_SMEM)


def smem_reason(mode: int, slot_words: int, device: torch.device, library: str) -> "str | None":
    """Why the launch does not fit the device's opt-in shared-memory limit, or ``None``.

    The limit is read from the built library (``cudaDevAttrMaxSharedMemoryPer
    BlockOptin``; torch publishes no such property).  A library that cannot be
    built answers ``None`` here: the build failure is the caller's, reported
    where the adapter is constructed, not a lane refusal.
    """
    try:
        lib = _ext(library)
    except Exception:  # noqa: BLE001 -- the build's failure is reported by from_bundles
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    need = smem_bytes(mode, slot_words, mma8=library_mma8(library))
    have = int(lib.max_dynamic_smem_bytes(index))
    if need <= have:
        return None
    what = "gate/up" if mode != 2 else "down/dense"
    return (f"the {what} launch at {slot_words}-word slots needs {need} bytes of shared memory "
            f"per block and this device allows {have}")


def fused_routed_window_enabled() -> bool:
    return os.environ.get(ENV_TOGGLE, "1") != "0"


def fused_routed_unit_shape_refusal(family: str, part: str, *, rows: int, cols: int,
                                    rates, window_bits: int) -> "str | None":
    """The wire-shape half of :func:`fused_routed_window_supported`, on one
    unit's manifest facts alone -- what an exporter can decide before any
    bundle exists (tessera#624 prices the lane's tables only where the
    stack's shape admits the lane).  ``part`` is ``gate``/``up``/``down``.

    Only what the manifest says is checked here, with the runtime
    predicate's own helpers: family, window bits, the column count the tiles
    need, the run table the packer lays out for these rates (one run per
    distinct rate, sorted by rate) as :func:`run_pair` reads it -- one rate or
    two ADJACENT rates, each in 1..8 (contract v45, tessera#694) -- the
    word-stage slot that pair needs against the target platform's opt-in
    shared memory (``SM121_MAX_DYNAMIC_SMEM``) in the part's own launch (the
    two-table gate/up launch reaches ``ROUTED_LANE_RATES``, the one-table
    down launch every rate), and the row multiples the kernel's tiles need.
    Device, arithmetic, the activation quantizer, the column order and the
    env toggle are runtime facts the runtime predicate keeps.  A stack is
    refused whole when any of its parts is, as at runtime, so a GLM stack
    (one rung for all three parts) is admitted exactly where
    ``column_rates_routed_moe`` admits its rates.  Returns the refusal, or
    ``None`` when the shape serves.
    """
    if family not in ("value", "e4m3"):
        return f"family {family!r} is not a window family"
    if int(window_bits) != WINDOW_BITS:
        return f"{part} window_bits {window_bits} != {WINDOW_BITS}"
    cols = int(cols)
    if cols % BK != 0 or cols < MIN_COLS:
        return f"{part} has {cols} columns; the lane needs a multiple of {BK} and at least {MIN_COLS}"
    rates = tuple(int(r) for r in rates)
    if len(rates) != cols:
        return f"{part} carries {len(rates)} column rates for {cols} columns"
    table, col0, word0 = [], 0, 0
    for rate in sorted(set(rates)):
        count = rates.count(rate)
        table.append((rate, col0, count, word0))
        col0, word0 = col0 + count, word0 + chunk_words(rate) * count
    pair, why = run_pair(torch.tensor(table, dtype=torch.int32), cols)
    if pair is None:
        return f"{part} run table: {why}"
    mode = 2 if part == "down" else 0
    slot = slot_words_for_pair(pair)
    if smem_bytes(mode, slot) > SM121_MAX_DYNAMIC_SMEM:
        what = "down" if mode == 2 else "gate/up"
        return (f"the {what} launch at {slot}-word slots needs {smem_bytes(mode, slot)} bytes of "
                f"shared memory per block; sm_121 allows {SM121_MAX_DYNAMIC_SMEM}")
    rows = int(rows)
    if part == "down":
        if rows % BN != 0:
            return f"the hidden size {rows} is not a multiple of {BN}"
    elif rows % HALF != 0:
        return f"the intermediate size {rows} is not a multiple of {HALF}"
    return None


def fused_dense_window_enabled() -> bool:
    return os.environ.get(ENV_TOGGLE_DENSE, "1") != "0"


def _cflags(token: str, fp8: bool, mma8: bool = False, fp4: bool = False) -> list:
    """A library's compile flags.  ``fp4`` is the E2M1 family's library
    (``tessera.routed_fused_e2m1``): its define, and the architecture-specific
    target its block-scaled FP4 instruction exists on.  The other libraries'
    flags do not move."""
    from .serving.backend import offload_flags

    return ["-O3", "-lineinfo", "-std=c++17",
            f"-DTESSERA_ROUTED_FUSED_FP8={1 if fp8 else 0}",
            f"-DTESSERA_ROUTED_FUSED_MMA8={1 if mma8 else 0}",
            *(["-DTESSERA_ROUTED_FUSED_FP4=1"] if fp4 else []),
            *offload_flags(token, arch_specific=fp4)]


def _probed_or_none(probe):
    try:
        return probe(torch=torch)
    except Exception:  # noqa: BLE001 -- no device answering is the answer
        return None


def _built_library(build: str, module: str) -> "str | None":
    found = sorted(_glob.glob(os.path.join(build, f"{module}*.so")))
    return found[0] if found else None


def build_library(module: str, source_module: str, compile_fn):
    """Build and load one library of :data:`SOURCE` on this process's platform.

    ``module`` names the build directory and the library; ``source_module``
    is the published extension whose source is compiled (#134: the path the
    contract publishes IS the file compiled); ``compile_fn(src, build, token,
    verbose)`` makes the ``cpp_extension.load`` call, whose module name is a
    literal at the call site so the contract scanner reads it.  A library
    built for a platform token this process's device does not probe as is
    refused as a serving path.
    """
    from tessera.serving import ext as serving_ext
    from tessera.serving.backend import (
        PLATFORM_TOKEN_ENV,
        PlatformMismatchError,
        backend as detect_backend,
        ensure_toolchain_on_path,
        pin_build_arch,
        platform_token,
        probed_platform_token,
    )

    from .jit_build_lock import GUARDED_BUILD_SUFFIX, jit_build_lock

    ensure_toolchain_on_path(torch)
    src = serving_ext.native_source_path(source_module)
    if detect_backend(torch) != "cuda":
        raise PlatformMismatchError(
            "the fused routed window kernel is CUDA (mma.sync, ldmatrix, cp.async); this "
            "process's torch is not a CUDA build")
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser("~/tmp/torch-ext-routed-fused")
    token = platform_token(torch=torch)
    build = os.path.join(root, f"{module}_{token}") + GUARDED_BUILD_SUFFIX
    os.makedirs(build, exist_ok=True)
    pin_build_arch(token, torch)
    verbose = bool(os.environ.get("TESSERA_ROUTED_FUSED_VERBOSE"))
    try:
        with jit_build_lock(build):
            lib = compile_fn(src, build, token, verbose)
    except Exception as exc:
        probed = _probed_or_none(probed_platform_token)
        if token == probed or not _built_library(build, module):
            raise
        raise PlatformMismatchError(
            f"the fused routed window extension compiled for {token} ({PLATFORM_TOKEN_ENV}) into "
            f"{build}, and this process's device is {probed}: the library cannot be loaded here "
            f"({type(exc).__name__}: {exc}). A build for an absent device is a compile gate, never "
            "a serving path.") from exc
    probed = _probed_or_none(probed_platform_token)
    if token != probed:
        raise PlatformMismatchError(
            f"the fused routed window extension was built for {token} ({PLATFORM_TOKEN_ENV}) and "
            f"this process's device is {probed}; the library under {build} is a compile-gate "
            "artifact and is refused as a serving path.")
    return lib


@functools.lru_cache(maxsize=None)
def _ext(library: str):
    """The library (a :data:`LIBRARIES` key; a family names its 16-bit
    library), built on first use (the window GEMV's loader shape)."""
    from torch.utils.cpp_extension import load

    if library not in LIBRARIES:
        raise GrammarError(f"the fused routed lane builds the libraries {sorted(LIBRARIES)}, got {library!r}")
    module, family, mma8 = LIBRARIES[library]
    fp8 = family == "e4m3"

    def compile_fn(src, build, token, verbose):
        if mma8:
            return load(
                name="tessera_routed_fused_mma_e4m3",  # literal: the contract scanner reads it
                sources=[src], build_directory=build,
                extra_cuda_cflags=_cflags(token, True, True), verbose=verbose)
        if fp8:
            return load(
                name="tessera_routed_fused_e4m3",   # literal: the contract scanner reads it
                sources=[src], build_directory=build,
                extra_cuda_cflags=_cflags(token, True), verbose=verbose)
        return load(
            name="tessera_routed_fused_value",  # literal: the contract scanner reads it
            sources=[src], build_directory=build,
            extra_cuda_cflags=_cflags(token, False), verbose=verbose)

    lib = build_library(module, module, compile_fn)
    dense_max = DENSE_RATE_MAX["e4m3" if fp8 else "value"]
    for name, want in (("BM", BM), ("BN", BN), ("HALF", HALF), ("BK", BK),
                       ("DENSE_ROW_QUANTUM", DENSE_ROW_QUANTUM),
                       ("RATE_MIN", RATE_MIN), ("ROUTED_RATE_MAX", RATE_MAX),
                       ("RATE_MAX", dense_max), ("SLOT_WORDS_MAX", slot_words_for_rate(dense_max)),
                       ("BDESC_INTS", BDESC_INTS), ("WINDOW_BITS", WINDOW_BITS), ("FAMILY_FP8", fp8),
                       ("FAMILY_MMA8", mma8), ("WORD_STAGES", WORD_STAGES),
                       ("WORD_STAGES_MIN", WORD_STAGES_MIN),
                       ("SMEM_FIXED_GATE_UP", (SMEM_FIXED_MMA8 if mma8 else SMEM_FIXED)[0]),
                       ("SMEM_FIXED_DOWN", (SMEM_FIXED_MMA8 if mma8 else SMEM_FIXED)[2]),
                       ("BM_WIDE", BM_WIDE), ("A_REGION_BYTES_WIDE", a_region_bytes(BM_WIDE, mma8=mma8)),
                       ("HAS_WIDE_GATE_UP", has_width(library, 0, BM_WIDE)),
                       ("HAS_WIDE_DOWN", has_width(library, 2, BM_WIDE)),
                       # the gate/up rates the library instantiates ARE the ones the host admits
                       ("GATE_UP_RATE_MAX", max(routed_lane_rates(library)))):
        if getattr(lib, name) != want:
            raise GrammarError(
                f"{module} was built with {name}={getattr(lib, name)!r}; this module expects {want!r}")
    return lib


def compose_table16(bundle) -> torch.Tensor:
    """The per-expert 16-bit table the kernel looks states up in.

    Value family: the bf16 table as is.  E4M3 family: ``native[codes[state]]``
    composed once, then the e4m3 byte widened to f16 -- an exact conversion,
    so the f16 tensor-core product of two exact values accumulated in fp32 is
    the same function of the wire the fp8 ``tl.dot`` computes.  Returned as an
    int16 view so a torch build without a uint16 dtype can hold it.
    """
    if bundle.family == "value":
        table = bundle.table_all
        if table.dtype != torch.bfloat16:
            raise GrammarError("the value family's table must be bf16")
        return table.contiguous().view(torch.int16)
    codes = bundle.codes_all.to(torch.int64)                       # [E, 2^L]
    bytes_ = torch.gather(bundle.native_all, 1, codes)              # [E, 2^L] uint8
    return bytes_.view(torch.float8_e4m3fn).to(torch.float16).contiguous().view(torch.int16)


def compose_table8(bundle) -> torch.Tensor:
    """The per-expert E4M3 byte table the E4M3 instruction reads: ``native
    [codes[state]]`` composed once, uint8 ``[E, 2^L]`` (16 KB per expert).
    The bytes the f16 table widens, unwidened."""
    if bundle.family != "e4m3":
        raise GrammarError("the E4M3 instruction's byte table belongs to the e4m3 family")
    codes = bundle.codes_all.to(torch.int64)
    return torch.gather(bundle.native_all, 1, codes).contiguous()


def compose_table(bundle, library: str) -> torch.Tensor:
    """The table ``library`` reads: :func:`compose_table8` on the E4M3
    instruction, :func:`compose_table16` otherwise."""
    return compose_table8(bundle) if library_mma8(library) else compose_table16(bundle)


def words_by_expert(bundle) -> torch.Tensor:
    """``words_all`` as the kernel reads it: int32 ``[E, W]``, one row per expert.

    The loader's SoA bundle already has that shape; a bundle built from units
    holds one flat stream with ``word_off``, and both are the same bytes, so
    this is a view.  Refuses (by name) a stack whose experts are not one
    uniform stride apart.
    """
    words = bundle.words_all
    e = int(bundle.experts)
    if words.dim() == 2:
        if int(words.shape[0]) != e:
            raise GrammarError(f"words_all has {int(words.shape[0])} rows for {e} experts")
        return words
    if words.dim() != 1 or words.numel() % e != 0:
        raise GrammarError("words_all must be [E, W] or a flat stream of E equal parts")
    width = words.numel() // e
    want = torch.arange(e, dtype=bundle.word_off.dtype, device=bundle.word_off.device) * width
    if not bool((bundle.word_off == want).all()):
        raise GrammarError("word_off is not one uniform per-expert stride")
    return words.view(e, width)


def run_pair(runs: torch.Tensor, cols: int, *, rate_max: int = RATE_MAX,
             ) -> "tuple[torch.Tensor | None, str | None]":
    """One unit's run table as the kernel's run pair, or why it is refused.

    ``runs`` is the wire's ``[R, 4]`` table of ``(rate, col0, ncols, word0)``
    rows, the packer's stable sort of the columns by (rate, column) into one
    contiguous run per rate.  The kernel reads one or two runs -- the grammar
    mixes only the two rates bracketing a stack's root, which are ADJACENT
    (``grammar.rate_set``) -- as the int32 ``[8]`` pair
    ``(r_lo, 0, n_lo, 0, r_hi, n_lo, n_hi, w_hi)`` with ``r_hi = r_lo + 1``
    and ``w_hi = 16 * n_lo * r_lo`` (a one-run table has ``n_hi = 0``).  Each
    (r_lo, one or two runs) pair is its own kernel instantiation, which the
    host picks from the launch's ``tile_words`` (``pair_of`` in the kernel
    source), so a pair of rates further apart -- which no grammar schedule
    emits -- is refused here by name rather than decoded.  ``rate_max`` is the
    launch's ceiling: :data:`RATE_MAX` on the routed launches, the family's
    :data:`DENSE_RATE_MAX` on the dense one.  Returns ``(pair, None)`` or
    ``(None, reason)``.
    """
    rates = range(RATE_MIN, int(rate_max) + 1)
    runs = runs.reshape(-1, 4)
    n_runs = int(runs.shape[0])
    if n_runs not in (1, 2):
        return None, f"run table has {n_runs} runs; the lane reads one or two (the two rates bracketing the root)"
    rows = [tuple(int(v) for v in row) for row in runs.tolist()]
    r_lo, c_lo, n_lo, w_lo = rows[0]
    if r_lo not in rates or c_lo != 0 or w_lo != 0 or n_lo <= 0:
        return None, f"first run {rows[0]} is not (rate in {RATE_MIN}..{rate_max}, 0, n, 0)"
    if n_runs == 1:
        if n_lo != cols:
            return None, f"the one run covers {n_lo} of {cols} columns"
        pair = (r_lo, 0, n_lo, 0, 0, n_lo, 0, 16 * n_lo * r_lo)
    else:
        r_hi, c_hi, n_hi, w_hi = rows[1]
        if r_hi not in rates or r_hi <= r_lo:
            return None, f"second run rate {r_hi} is not above the first's {r_lo} within {RATE_MIN}..{rate_max}"
        if r_hi != r_lo + 1:
            return None, (f"second run rate {r_hi} is not adjacent to the first's {r_lo}: the lane reads the "
                          "two adjacent rates bracketing a root (grammar.rate_set)")
        if c_hi != n_lo or n_hi <= 0 or n_lo + n_hi != cols or w_hi != 16 * n_lo * r_lo:
            return None, f"runs {rows} do not tile {cols} columns as (rate, 0, n_lo, 0), (rate, n_lo, n_hi, 16 * n_lo * rate)"
        pair = (r_lo, 0, n_lo, 0, r_hi, n_lo, n_hi, w_hi)
    return torch.tensor(pair, dtype=torch.int32, device=runs.device), None


def pair_tile_words(pair: torch.Tensor) -> int:
    """``16 * (n_lo * r_lo + n_hi * r_hi)``: the wire's words per 512-row tile."""
    r_lo, _c0, n_lo, _w0, r_hi, _c1, n_hi, _w1 = (int(v) for v in pair.reshape(8).tolist())
    return 16 * (n_lo * r_lo + n_hi * r_hi)


def perm_reason(perm: torch.Tensor, n_lo: int, cols: int) -> "str | None":
    """Why ``perm`` (``[E, cols]`` or ``[cols]``) is not the packer's order.

    The kernel recovers a column's permuted position from its rank within its
    run, which holds exactly when the permutation lists the low-rate columns
    in ascending column order, then the high-rate columns in ascending order
    -- the stable sort by (rate, column) every packer performs.
    """
    p = perm.reshape(-1, cols).to(torch.int64)
    if int(p.shape[1]) != cols:
        return f"perm has {int(perm.numel())} entries for {cols} columns"
    sorted_p, _ = torch.sort(p, dim=1)
    arange = torch.arange(cols, dtype=torch.int64, device=p.device)
    if not bool((sorted_p == arange).all()):
        return "perm is not a permutation of the columns"
    for lo, hi, name in ((0, n_lo, "low-rate"), (n_lo, cols, "high-rate")):
        seg = p[:, lo:hi]
        if seg.shape[1] > 1 and not bool((seg[:, 1:] > seg[:, :-1]).all()):
            return f"the {name} run's columns are not in ascending column order"
    return None


def block_desc(perm: torch.Tensor, n_lo: int, cols: int) -> torch.Tensor:
    """The per-32-column-block descriptors the kernel maps lane groups with.

    int32 ``[E, cols / 32, BDESC_INTS]``: words 0..7 hold 32 bytes, the
    in-block position (0..31) of the block's low-rate columns in ascending
    order followed by its high-rate columns in ascending order; word 8 is the
    number of low-rate columns in the blocks before this one, word 9 the
    number in this block; words 10..11 pad the descriptor to 48 bytes.  With
    this the kernel decodes the ``m``-th descriptor entry in lane group ``m``,
    so a warp's four columns share a rate except in the one straddling group.
    """
    p = perm.reshape(-1, cols).to(torch.int64)
    e = int(p.shape[0])
    nk = cols // BK
    device = p.device
    is_hi_pos = (torch.arange(cols, device=device) >= n_lo).to(torch.int64)          # per permuted position
    is_hi_col = torch.zeros(e, cols, dtype=torch.int64, device=device)
    is_hi_col.scatter_(1, p, is_hi_pos.expand(e, cols))                              # per original column
    is_hi_blk = is_hi_col.reshape(e, nk, BK)
    cib = torch.arange(BK, device=device).expand(e, nk, BK)
    order = torch.argsort(is_hi_blk * BK + cib, dim=2)                               # lo (asc) then hi (asc)
    packed = order.reshape(e, nk, 8, 4)
    words = (packed[..., 0] | (packed[..., 1] << 8) | (packed[..., 2] << 16) | (packed[..., 3] << 24))
    cnt_lo = BK - is_hi_blk.sum(dim=2)                                               # [E, nk]
    before = torch.cumsum(cnt_lo, dim=1) - cnt_lo
    desc = torch.zeros(e, nk, BDESC_INTS, dtype=torch.int64, device=device)
    desc[..., :8] = words
    desc[..., 8] = before
    desc[..., 9] = cnt_lo
    # 0x80000000 cannot occur (in-block positions are < 32), so the int32 view is exact.
    return desc.to(torch.int32).contiguous()


def has_one_rate_four_run(b, e: int) -> bool:
    """Whether a bundle is the bounded one-run rate-4 body (tessera#739).

    The piece-major resident layout is scoped to exactly this shape: one run,
    rate 4, covering every column.  ``run_pair`` already validates that shape;
    this reads its result rather than the raw table.
    """
    pair, why = run_pair(b.runs_all.reshape(e, -1, 4)[0], int(b.cols))
    if why is not None or pair is None:
        return False
    r_lo, _c0, n_lo, _w0, _r_hi, _c1, n_hi, _w1 = (int(v) for v in pair.reshape(8).tolist())
    return n_hi == 0 and r_lo == 4 and n_lo == int(b.cols)


def _run_stack_reason(name: str, b, e: int) -> "str | None":
    """The wire checks every fused window lane makes on one projection's
    stack: words by expert, one run table for the stack that is the kernel's
    run pair, the packer's column order, the tile stride and the start
    states.  ``None`` when the stack is read as the wire lays it out."""
    try:
        words_by_expert(b)
    except GrammarError as exc:
        return f"{name}: {exc}"
    if b.words_all.dtype != torch.int32:
        return f"{name} words must be int32"
    # One run table for the stack: every expert carries the same one or two
    # runs (run_off == 0, R, 2R, ...), and it is the kernel's run pair.
    runs = b.runs_all.reshape(-1, 4)
    n_runs = int(runs.shape[0])
    if n_runs % e != 0 or (n_runs // e) not in (1, 2):
        return f"{name} carries {n_runs} runs for {e} experts; the lane reads one or two runs per expert"
    per = n_runs // e
    want_off = torch.arange(e + 1, dtype=b.run_off.dtype, device=b.run_off.device) * per
    if not bool((b.run_off == want_off).all()):
        return f"{name} experts do not all carry {per} run(s)"
    runs_e = runs.reshape(e, per, 4)
    if not bool((runs_e == runs_e[:1]).all()):
        return f"{name} experts disagree on their run tables; the lane reads one schedule per stack"
    pair, why = run_pair(runs_e[0], b.cols)
    if pair is None:
        return f"{name} {why}"
    n_lo = int(pair[2])
    if tuple(b.perm_all.shape) != (e, b.cols):
        return f"{name} perm_all is not [E, cols]"
    why = perm_reason(b.perm_all, n_lo, b.cols)
    if why is not None:
        return f"{name} {why}; the lane reads the packer's column order"
    want_tile = pair_tile_words(pair)
    if not bool((b.tile_words == want_tile).all()):
        return f"{name} tile_words is not {want_tile} (from its run table) for every expert"
    if tuple(b.init_all.shape) != (e, b.cols) or b.init_all.dtype != torch.int32:
        return f"{name} init_all must be int32 [E, cols]"
    if b.has_init.dtype != torch.int32 or b.has_init.numel() != e:
        return f"{name} has_init must be int32 [E]"
    return None


def fused_routed_window_supported(gate, up, down) -> "str | None":
    """Why this lane refuses a stack, or ``None`` when it serves it.

    The kernel reads the window wire as the packer lays it out: window bits
    14, one or two column-rate runs per expert (the two rates bracketing the
    stack's root, rates 1..8), the packer's column order (low-rate columns
    ascending, then high-rate ascending), and the family's published
    arithmetic (folded for value, epilogue for e4m3).  One run table and one
    ``tile_words`` per stack: the kernel takes a single tile stride for all
    experts and for both gate and up.  A stack outside that shape keeps the
    compact adapter, and the reason is the string returned here so a log can
    say which.
    """
    if not fused_routed_window_enabled():
        return f"disabled by {ENV_TOGGLE}=0"
    bundles = {"gate": gate, "up": up, "down": down}
    fam = down.family
    if fam not in ("value", "e4m3"):
        return f"family {fam!r} is not a window family"
    want_arith = "folded" if fam == "value" else "epilogue"
    e = int(down.experts)
    for name, b in bundles.items():
        if b.family != fam:
            return f"{name} family {b.family!r} differs from down's {fam!r}"
        if b.arithmetic != want_arith:
            return f"{name} arithmetic {b.arithmetic!r}; the fused lane serves {want_arith!r} for {fam}"
        if b.device.type != "cuda":
            return f"{name} lives on {b.device}; the lane is CUDA"
        if b.window_bits != WINDOW_BITS:
            return f"{name} window_bits {b.window_bits} != {WINDOW_BITS}"
        # The word layout must be an EXACT tag this lane reads: 'legacy' for
        # every stack, or 'piece_major' for the bounded E4M3 one-run rate-4
        # routed body only (tessera#739).  An unknown tag is refused, and the
        # three bundles must agree -- never inferred from one of them.
        lays = {str(getattr(x, "word_layout", "legacy")) for x in bundles.values()}
        if len(lays) != 1:
            return f"the gate/up/down bundles disagree on their word layout: {sorted(lays)}"
        layout = lays.pop()
        if layout not in ("legacy", "piece_major"):
            return f"{name} carries unknown word layout {layout!r}"
        if layout == "piece_major":
            if fam != "e4m3":
                return (f"{name} is piece_major; the lane reads piece-major only for the "
                        f"E4M3 family, not {fam!r}")
            if not has_one_rate_four_run(b, e):
                return (f"{name} is piece_major, which the lane reads only for a single run at "
                        f"rate 4; its run table is not one rate-4 run over {b.cols} columns")
        if int(b.experts) != e:
            return f"{name} has {b.experts} experts, down has {e}"
        if fam == "e4m3" and b.quantizer != "native":
            return f"{name} was prepared without the native activation quantizer"
        if b.cols % BK != 0 or b.cols < MIN_COLS:
            return f"{name} has {b.cols} columns; the lane needs a multiple of {BK} and at least {MIN_COLS}"
        why = _run_stack_reason(name, b, e)
        if why is not None:
            return why
        if b.scale_all.dtype != torch.float32 or tuple(b.scale_all.shape) != (e, b.rows):
            return f"{name} scale_all must be fp32 [E, rows]"
    if gate.rows != up.rows or gate.rows != down.cols:
        return f"gate rows {gate.rows}, up rows {up.rows} and down cols {down.cols} disagree"
    if gate.cols != up.cols:
        return f"gate cols {gate.cols} != up cols {up.cols}"
    if gate.rows % HALF != 0:
        return f"the intermediate size {gate.rows} is not a multiple of {HALF}"
    if down.rows % BN != 0:
        return f"the hidden size {down.rows} is not a multiple of {BN}"
    if int(gate.tile_words[0]) != int(up.tile_words[0]):
        return (f"gate tile_words {int(gate.tile_words[0])} != up tile_words {int(up.tile_words[0])}; "
                "the gate/up launch reads one tile stride for both")
    # The word-stage slot each launch needs, against the device's shared memory.
    try:
        library = library_for(fam)
    except GrammarError as exc:
        return str(exc)
    for mode, bs in ((0, (gate, up)), (2, (down,))):
        slot = max(slot_words_for_pair(run_pair(b.runs_all.reshape(e, -1, 4)[0], b.cols)[0]) for b in bs)
        why = smem_reason(mode, slot, down.device, library)
        if why is not None:
            return why
    return None


def projection_tables(bundle) -> "tuple[torch.Tensor, torch.Tensor, int, int]":
    """The launch tables the lane builds for one projection's bundle.

    ``(runs, bdesc, tile_words, slot_words)``: the stack's run pair broadcast
    to int32 ``[E, 8]``, the block descriptors (int32 ``[E, cols / 32,
    BDESC_INTS]``, :func:`block_desc`), the wire's words per 512-row tile and
    the word-stage slot the pair needs.  :meth:`FusedRoutedWindowMoE.
    from_bundles` builds the lane from these, and ``runs`` and ``bdesc`` are
    two of the tensors its ``resident_bytes`` counts, so the exporter's price
    (``serving_parts.routed_fused_unit_bytes``) is tested against this
    function rather than a restatement.  The caller has admitted the stack
    (:func:`fused_routed_window_supported`).
    """
    e = int(bundle.experts)
    pair, why = run_pair(bundle.runs_all.reshape(e, -1, 4)[0], bundle.cols)
    if pair is None:
        raise GrammarError(f"the fused routed window lane refuses this projection: {why}")
    return (pair.reshape(1, 8).expand(e, 8).contiguous(),
            block_desc(bundle.perm_all, int(pair[2]), int(bundle.cols)),
            pair_tile_words(pair), slot_words_for_pair(pair))


@dataclasses.dataclass(frozen=True)
class _Routing:
    offsets: torch.Tensor      # [E + 1] int32
    flat_sorted: torch.Tensor  # [P] int32
    rw_sorted: torch.Tensor    # [P] fp32
    item_off: torch.Tensor     # [E + 1] int32: prefix sum of ceil(routes_e / BM)
    tokens: int
    top_k: int
    #: The same at ``BM_WIDE``-route superblocks, built only when a launch of
    #: the step takes that width (:func:`superblock_rows`).
    item_off_wide: "torch.Tensor | None" = None

    @property
    def routes(self) -> int:
        return self.tokens * self.top_k

    def superblocks(self, bm: int) -> torch.Tensor:
        """``item_off`` for ``bm``-route superblocks: [E + 1] int32."""
        if int(bm) == BM:
            return self.item_off
        if int(bm) != BM_WIDE or self.item_off_wide is None:
            raise GrammarError(f"no {bm}-route superblock offsets for this step")
        return self.item_off_wide


def _item_off(counts: torch.Tensor, bm: int) -> torch.Tensor:
    """The prefix sum of ``ceil(counts_e / bm)``: [E + 1] int32."""
    item_off = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=counts.device)
    item_off[1:] = torch.cumsum((counts + (bm - 1)) // bm, 0, dtype=torch.int32)
    return item_off


def _routing_tables(expert_ids: torch.Tensor, routing_weights: torch.Tensor, experts: int,
                    device: torch.device, library: "str | None") -> _Routing:
    """The step's routing in the order the fused launches read it: routes
    sorted by expert (stable), their weights, and the superblock offsets at
    each width a launch of ``library`` takes this step (``None``: a library
    with the one width, ``BM``)."""
    if expert_ids.dim() != 2 or routing_weights.shape != expert_ids.shape:
        raise GrammarError("expert_ids and routing_weights must share [T, top_k]")
    if expert_ids.device != device or routing_weights.device != device:
        raise GrammarError("routing tensors must live on the compute device")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise GrammarError("expert_ids must be int32 or int64")
    tokens, top_k = (int(v) for v in expert_ids.shape)
    ids = expert_ids.reshape(-1).to(torch.int64)
    counts = torch.zeros(experts, dtype=torch.int32, device=device)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    order = torch.argsort(ids, stable=True)
    flat_sorted = order.to(torch.int32).contiguous()
    rw_sorted = routing_weights.reshape(-1).to(torch.float32)[order].contiguous()
    wide = library is not None and any(superblock_rows(library, mode, tokens) == BM_WIDE for mode in (0, 2))
    return _Routing(offsets=offsets, flat_sorted=flat_sorted, rw_sorted=rw_sorted,
                    item_off=_item_off(counts, BM), tokens=tokens, top_k=top_k,
                    item_off_wide=_item_off(counts, BM_WIDE) if wide else None)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@dataclasses.dataclass(frozen=True)
class FusedRoutedWindowMoE:
    """The fused lane over a loader-filled grouped stack.

    ``gate``/``up``/``down`` are the compact loader's
    :class:`~tessera.window_gemm_grouped.PreparedGroupedWindowGemm` bundles
    (the same objects the compact adapter would serve); ``library`` is the
    :data:`LIBRARIES` key the launches take (:func:`library_for` at
    construction), the three ``table`` tensors are :func:`compose_table` of
    each for that library and the ``words`` tensors are
    :func:`words_by_expert` views.  ``__call__`` is the routed forward;
    ``gate_up``/``down_routes`` are the teacher-forced stages the routed pair
    oracle calls.
    """
    #: The ``torch.profiler`` range ``moe_route`` runs this adapter's forward
    #: under; the compact adapter's is ``tessera_native_window_moe`` (#640).
    PROFILER_LABEL = "tessera_routed_fused_window"


    gate: object
    up: object
    down: object
    family: str
    arithmetic: str
    library: str
    table_gate: torch.Tensor
    table_up: torch.Tensor
    table_down: torch.Tensor
    words_gate: torch.Tensor
    words_up: torch.Tensor
    words_down: torch.Tensor
    #: The run pairs (int32 ``[E, 8]``, one row per expert, all equal) and the
    #: block descriptors (int32 ``[E, K / 32, BDESC_INTS]``) of each projection.
    runs_gate: torch.Tensor
    runs_up: torch.Tensor
    runs_down: torch.Tensor
    bdesc_gate: torch.Tensor
    bdesc_up: torch.Tensor
    bdesc_down: torch.Tensor
    tile_words_gate_up: int
    tile_words_down: int
    slot_words_gate_up: int
    slot_words_down: int
    counters: torch.Tensor
    activation: str = "silu"

    @property
    def piece_major(self) -> bool:
        """Whether the resident words are the piece-major order (tessera#739).

        Read off the bundles' shared tag (``word_layout``), never recomputed
        from the run table: the reader must agree with how the stack was
        actually written.
        """
        return str(getattr(self.down, "word_layout", "legacy")) != "legacy"

    @classmethod
    def from_bundles(cls, gate, up, down, *, activation: str = "silu") -> "FusedRoutedWindowMoE":
        reason = fused_routed_window_supported(gate, up, down)
        if reason is not None:
            raise GrammarError(f"the fused routed window lane refuses this stack: {reason}")
        if activation != "silu":
            raise GrammarError(f"activation {activation!r} is not served; the fused lane computes silu")
        # Build (or load) the family's extension HERE, at adapter construction:
        # a toolchain that cannot compile it fails in the caller's hands --
        # ``PackedWindowMoeBundles.adapter`` substitutes the compact adapter,
        # the ``when_unavailable`` answer the contract publishes -- and not on
        # the first forward of a serve.
        library = library_for(down.family)
        _ext(library)
        runs_gate, bdesc_gate, tw_gate, sw_gate = projection_tables(gate)
        runs_up, bdesc_up, _tw_up, sw_up = projection_tables(up)
        runs_down, bdesc_down, tw_down, sw_down = projection_tables(down)
        return cls(gate=gate, up=up, down=down, family=down.family, arithmetic=down.arithmetic,
                   library=library,
                   table_gate=compose_table(gate, library), table_up=compose_table(up, library),
                   table_down=compose_table(down, library),
                   words_gate=words_by_expert(gate), words_up=words_by_expert(up),
                   words_down=words_by_expert(down),
                   runs_gate=runs_gate, runs_up=runs_up, runs_down=runs_down,
                   bdesc_gate=bdesc_gate, bdesc_up=bdesc_up, bdesc_down=bdesc_down,
                   tile_words_gate_up=tw_gate, tile_words_down=tw_down,
                   slot_words_gate_up=max(sw_gate, sw_up), slot_words_down=sw_down,
                   counters=torch.zeros(2, dtype=torch.int32, device=down.device),
                   activation=activation)

    # -- identity -----------------------------------------------------------
    @property
    def launch_pair(self) -> "tuple[str, str]":
        from .serving.scheme import ROUTED_FUSED_WINDOW_SYMBOL
        from .serving.telemetry import (DECODER_NATIVE_ROUTED_FUSED_WINDOW,
                                        DECODER_NATIVE_ROUTED_FUSED_WINDOW_E4M3MMA,
                                        DECODER_NATIVE_ROUTED_FUSED_WINDOW_FOLDED)

        decoder = {"value": DECODER_NATIVE_ROUTED_FUSED_WINDOW_FOLDED,
                   "e4m3": DECODER_NATIVE_ROUTED_FUSED_WINDOW,
                   "e4m3mma": DECODER_NATIVE_ROUTED_FUSED_WINDOW_E4M3MMA}[self.library]
        return ROUTED_FUSED_WINDOW_SYMBOL, decoder

    @property
    def experts(self) -> int:
        return int(self.down.experts)

    @property
    def device(self) -> torch.device:
        return self.down.device

    @property
    def fp8(self) -> bool:
        return self.family == "e4m3"

    def named_tables(self):
        """The tensors this lane holds BEYOND the bundles' own planes."""
        yield "routed_fused.table_gate", self.table_gate
        yield "routed_fused.table_up", self.table_up
        yield "routed_fused.table_down", self.table_down
        yield "routed_fused.runs_gate", self.runs_gate
        yield "routed_fused.runs_up", self.runs_up
        yield "routed_fused.runs_down", self.runs_down
        yield "routed_fused.bdesc_gate", self.bdesc_gate
        yield "routed_fused.bdesc_up", self.bdesc_up
        yield "routed_fused.bdesc_down", self.bdesc_down

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for _n, t in self.named_tables())

    # -- routing ------------------------------------------------------------
    def _routing(self, expert_ids: torch.Tensor, routing_weights: torch.Tensor) -> _Routing:
        return _routing_tables(expert_ids, routing_weights, self.experts, self.device, self.library)

    def _launch(self, mode: int, x: torch.Tensor, a_scale: "torch.Tensor | None",
                routing: _Routing, *, a_row_mode: int, mul_weight: bool, limit: float,
                out: torch.Tensor, counter: int) -> None:
        lib = _ext(self.library)
        if mode == 2:
            b0 = b1 = self.down
            w0 = w1 = self.words_down
            t0 = t1 = self.table_down
            r0 = r1 = self.runs_down
            d0 = d1 = self.bdesc_down
            tile_words, slot_words = self.tile_words_down, self.slot_words_down
        else:
            b0, b1 = self.gate, self.up
            w0, w1 = self.words_gate, self.words_up
            t0, t1 = self.table_gate, self.table_up
            r0, r1 = self.runs_gate, self.runs_up
            d0, d1 = self.bdesc_gate, self.bdesc_up
            tile_words, slot_words = self.tile_words_gate_up, self.slot_words_gate_up
        bm = superblock_rows(self.library, mode, routing.tokens)
        empty = self.counters.new_zeros(0, dtype=torch.float32)
        slot = self.counters[counter:counter + 1]
        slot.zero_()   # in-stream: a captured forward replays with a fresh work list
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        lib.routed_fused_forward(
            int(mode), bool(self.fp8),
            x, a_scale if a_scale is not None else empty,
            w0, w1, t0, t1,
            b0.init_all, b1.init_all, b0.has_init, b1.has_init,
            b0.scale_all, b1.scale_all,
            r0, r1, d0, d1,
            int(tile_words), int(slot_words),
            bool(self.piece_major),
            routing.offsets, routing.flat_sorted, routing.rw_sorted, routing.superblocks(bm),
            slot,
            int(routing.top_k), int(a_row_mode), bool(mul_weight), float(limit),
            out, _sm_count(index), int(bm))

    def _quantized(self, x: torch.Tensor, a_scale: "torch.Tensor | None", rows: int):
        """The A operand the family's kernel reads: bf16 as is, or e4m3 + scale."""
        if not self.fp8:
            if x.dtype != torch.bfloat16 or a_scale is not None:
                raise GrammarError("the value family takes a bf16 x and no activation scale")
            return x, None
        if x.dtype == torch.float8_e4m3fn:
            if a_scale is None:
                raise GrammarError("an fp8 x must carry its per-row activation scale")
            a_scale = a_scale.reshape(-1)
            if a_scale.numel() != rows or a_scale.dtype != torch.float32 or a_scale.device != self.device:
                raise GrammarError(f"a_scale must be fp32 [{rows}] on {self.device}")
            return x, a_scale.contiguous()
        if x.dtype != torch.bfloat16:
            raise GrammarError(f"the E4M3 family takes bf16 or fp8 x, got {x.dtype}")
        if a_scale is not None:
            raise GrammarError("a_scale belongs with a prequantized fp8 x")
        from .serving.native_ops import native_fp8_quant

        xq, s = native_fp8_quant(x)
        return xq, s.reshape(-1)

    def _check_x(self, x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        if x.dim() != 2 or tuple(x.shape) != (rows, cols) or x.device != self.device:
            raise GrammarError(f"x must be [{rows}, {cols}] on {self.device}, got {tuple(x.shape)} on {x.device}")
        return x if x.is_contiguous() else x.contiguous()

    # -- the routed forward -------------------------------------------------
    def __call__(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor,
                 *, apply_router_weight_on_input: bool = False,
                 swiglu_limit: "float | None" = None) -> torch.Tensor:
        from .native_window_moe import SUPPORTED_ACTIVATIONS, checked_swiglu_limit

        limit = checked_swiglu_limit(swiglu_limit)
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise GrammarError(f"activation {self.activation!r} has no exact implementation here")
        if expert_ids.dim() != 2:
            raise GrammarError("expert_ids must be [T, top_k]")
        tokens = int(expert_ids.shape[0])
        x = self._check_x(x, tokens, self.gate.cols)
        if apply_router_weight_on_input:
            # The modular prepare's placement (see NativeWindowMoE.__call__):
            # the weight scales x BEFORE the activation quantiser, topk=1 only.
            if int(expert_ids.shape[1]) != 1:
                raise GrammarError(
                    "apply_router_weight_on_input is only implemented for topk=1 (vLLM's own "
                    f"prepare asserts this); got topk={int(expert_ids.shape[1])}")
            x = x * routing_weights.reshape(-1, 1).to(x.dtype)
        hidden, inter = self.down.rows, self.down.cols
        if tokens == 0:
            return torch.empty((0, hidden), dtype=torch.bfloat16, device=self.device)
        routing = self._routing(expert_ids, routing_weights)
        xq, a1 = self._quantized(x, None, tokens)
        act = torch.empty((routing.routes, inter), dtype=torch.bfloat16, device=self.device)
        self._launch(0, xq, a1, routing, a_row_mode=0, mul_weight=False,
                     limit=limit if limit is not None else float("inf"), out=act, counter=0)
        aq, a2 = self._quantized(act, None, routing.routes)
        routed = torch.empty((routing.routes, hidden), dtype=torch.bfloat16, device=self.device)
        self._launch(2, aq, a2, routing, a_row_mode=1, mul_weight=not apply_router_weight_on_input,
                     limit=float("inf"), out=routed, counter=1)
        out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=self.device)
        _ext(self.library).token_sum(routed, out, int(routing.top_k))
        return out

    # -- the teacher-forced stages (the oracle's interface) ------------------
    def gate_up(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor,
                a_scale: "torch.Tensor | None" = None, *, preserve: bool = True,
                apply_router_weight_on_input: bool = False) -> torch.Tensor:
        """Both projections, route-preserved: bf16 ``[T, top_k, 2I]`` (gate | up)."""
        if not preserve or apply_router_weight_on_input:
            raise GrammarError(
                "the fused lane's gate_up stage is the route-preserving, unweighted projection; "
                "the routed forward is __call__")
        routing = self._routing(expert_ids, routing_weights)
        x = self._check_x(x, routing.tokens, self.gate.cols)
        inter = self.down.cols
        out = torch.empty((routing.tokens, routing.top_k, 2 * inter), dtype=torch.bfloat16,
                          device=self.device)
        if routing.tokens == 0:
            return out
        xq, a1 = self._quantized(x, a_scale, routing.tokens)
        self._launch(1, xq, a1, routing, a_row_mode=0, mul_weight=False, limit=float("inf"),
                     out=out, counter=0)
        return out

    def down_routes(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor,
                    a_scale: "torch.Tensor | None" = None, *, route_input: bool = True,
                    apply_router_weight_on_input: bool = False,
                    round_routes: bool = True) -> torch.Tensor:
        """The down projection reduced per token: bf16 ``[T, H]``.

        ``x`` is the route-indexed activation ``[T * top_k, I]`` in ROUTE order
        (row ``t * top_k + j``), as the compact adapter's ``down(...,
        route_input=True)`` takes it.  Only the stock boundary is served:
        ``round_routes=True`` (each route rounded to bf16 before the fixed-order
        fp32 sum, then one rounding).
        """
        if not route_input:
            raise GrammarError("the fused lane's down stage takes route-indexed input")
        if not round_routes:
            raise GrammarError("the fused lane serves the stock bf16 route boundary only (round_routes=True)")
        routing = self._routing(expert_ids, routing_weights)
        hidden, inter = self.down.rows, self.down.cols
        x = self._check_x(x, routing.routes, inter)
        out = torch.empty((routing.tokens, hidden), dtype=torch.bfloat16, device=self.device)
        if routing.tokens == 0:
            return out
        xq, a2 = self._quantized(x, a_scale, routing.routes)
        routed = torch.empty((routing.routes, hidden), dtype=torch.bfloat16, device=self.device)
        self._launch(2, xq, a2, routing, a_row_mode=2, mul_weight=not apply_router_weight_on_input,
                     limit=float("inf"), out=routed, counter=1)
        _ext(self.library).token_sum(routed, out, int(routing.top_k))
        return out


# ---------------------------------------------------------------------------
# the dense identity: one role of a dense Linear as the E = 1 case
# ---------------------------------------------------------------------------

def fused_dense_window_supported(bundle) -> "str | None":
    """Why the dense identity refuses a prepared role, or ``None`` when it serves it.

    ``bundle`` is a ``window_gemm.PreparedWindowGemm`` (the frozen role the
    Triton dense GEMM runs).  The kernel reads the routed lane's wire shape --
    one or two column-rate runs (the two bracketing the root, at rates up to
    the family's :data:`DENSE_RATE_MAX`: 14 on the value family, 8 on E4M3),
    window bits 14, the packer's column order, the family's published
    arithmetic (folded for value, epilogue for e4m3) -- plus the dense tile:
    rows a multiple of ``DENSE_ROW_QUANTUM`` (one 128-column B block per item,
    the last one partial when 128 does not divide the rows) and columns a
    multiple of 32 and at least 128.  A role outside it keeps
    ``tessera::window_gemm_dense``, and the reason is the string returned
    here so a load log can say which.
    """
    if not fused_dense_window_enabled():
        return f"disabled by {ENV_TOGGLE_DENSE}=0"
    fam = bundle.family
    if fam not in ("value", "e4m3"):
        return f"family {fam!r} is not a window family"
    want_arith = "folded" if fam == "value" else "epilogue"
    if bundle.arithmetic != want_arith:
        return f"arithmetic {bundle.arithmetic!r}; the fused identity serves {want_arith!r} for {fam}"
    if bundle.device.type != "cuda":
        return f"the role lives on {bundle.device}; the kernel is CUDA"
    if bundle.window_bits != WINDOW_BITS:
        return f"window_bits {bundle.window_bits} != {WINDOW_BITS}"
    if fam == "e4m3" and bundle.quantizer != "native":
        return "the role was prepared without the native activation quantizer"
    cols, rows = int(bundle.cols), int(bundle.rows)
    if cols % BK != 0 or cols < MIN_COLS:
        return f"{cols} columns; the kernel needs a multiple of {BK} and at least {MIN_COLS}"
    if rows % DENSE_ROW_QUANTUM != 0:
        return f"{rows} rows; the dense identity needs a multiple of {DENSE_ROW_QUANTUM}"
    if bundle.words.dtype != torch.int32 or bundle.words.dim() != 1:
        return "words must be a flat int32 stream"
    pair, why = run_pair(bundle.runs, cols, rate_max=DENSE_RATE_MAX[fam])
    if pair is None:
        return why
    why = perm_reason(bundle.perm, int(pair[2]), cols)
    if why is not None:
        return f"{why}; the kernel reads the packer's column order"
    if int(bundle.tile_words) != pair_tile_words(pair):
        return f"tile_words {bundle.tile_words} is not {pair_tile_words(pair)} (from the run table)"
    try:
        library = library_for(fam)
    except GrammarError as exc:
        return str(exc)
    why = smem_reason(2, slot_words_for_pair(pair), bundle.device, library)
    if why is not None:
        return why
    if bundle.init_perm.dtype != torch.int32 or bundle.init_perm.numel() != cols:
        return "init_perm must be int32 [cols]"
    if bundle.scale.dtype != torch.float32 or bundle.scale.numel() != rows:
        return "scale must be fp32 [rows]"
    if fam == "value":
        if bundle.table.dtype != torch.bfloat16 or bundle.table.numel() != TABLE_ENTRIES:
            return f"the value table must be bf16 [{TABLE_ENTRIES}]"
    else:
        if bundle.codes.dtype != torch.uint8 or bundle.codes.numel() != TABLE_ENTRIES:
            return f"codes must be uint8 [{TABLE_ENTRIES}]"
        if bundle.native.dtype != torch.uint8 or bundle.native.numel() != 256:
            return "native must be uint8 [256]"
    return None


def compose_dense_table16(bundle) -> torch.Tensor:
    """:func:`compose_table16` for one role: int16 ``[1, 2^L]``."""
    if bundle.family == "value":
        return bundle.table.contiguous().view(torch.int16).reshape(1, TABLE_ENTRIES)
    bytes_ = bundle.native[bundle.codes.to(torch.int64)]                 # [2^L] uint8
    return (bytes_.view(torch.float8_e4m3fn).to(torch.float16).contiguous()
            .view(torch.int16).reshape(1, TABLE_ENTRIES))


def compose_dense_table(bundle, library: str) -> torch.Tensor:
    """The table ``library`` reads for one role: the E4M3 bytes, uint8
    ``[1, 2^L]``, on the E4M3 instruction; :func:`compose_dense_table16`
    otherwise."""
    if not library_mma8(library):
        return compose_dense_table16(bundle)
    if bundle.family != "e4m3":
        raise GrammarError("the E4M3 instruction's byte table belongs to the e4m3 family")
    return bundle.native[bundle.codes.to(torch.int64)].contiguous().reshape(1, TABLE_ENTRIES)


@dataclasses.dataclass(frozen=True)
class FusedDenseWindowRole:
    """One role's kernel inputs, frozen at preparation.

    ``words``/``init``/``wscale`` are views of the bundle's own tensors (no new
    storage); ``table16`` is the composed table the role's library reads (the
    16-bit table, 32 KB, or on the E4M3 instruction the uint8 E4M3 bytes,
    16 KB; new storage, counted by the module's residency accounting) --
    its dtype names the library (:attr:`library`); ``has_init`` is the one
    int32 flag the kernel reads per expert; ``runs`` is the run pair (int32
    ``[1, 8]``) and ``bdesc`` the block descriptors (int32 ``[1, K / 32,
    BDESC_INTS]``, 1.5 bytes per column) the kernel maps its lane groups with;
    ``tile_words`` is the wire's words per 512-row tile, from the run pair.
    """
    family: str
    rows: int
    cols: int
    words: torch.Tensor       # int32 [1, W]
    table16: torch.Tensor     # int16 [1, 2^L]
    init: torch.Tensor        # int32 [1, K]
    has_init: torch.Tensor    # int32 [1]
    wscale: torch.Tensor      # fp32 [1, N]
    runs: torch.Tensor        # int32 [1, 8]
    bdesc: torch.Tensor       # int32 [1, K / 32, BDESC_INTS]
    tile_words: int
    slot_words: int

    @property
    def fp8(self) -> bool:
        return self.family == "e4m3"

    @property
    def library(self) -> str:
        """The :data:`LIBRARIES` key this role launches: the table's dtype says
        which (uint8 bytes are the E4M3 instruction's), so a role rebuilt from
        its tensors -- ``tessera::fused_window_dense`` does, per call -- runs
        the library it was prepared for."""
        if self.table16.dtype == torch.uint8:
            return "e4m3mma"
        return self.family

    def named_tables(self):
        yield "fused_table16", self.table16
        yield "fused_has_init", self.has_init
        yield "fused_runs", self.runs
        yield "fused_bdesc", self.bdesc


def prepare_dense_role(bundle) -> FusedDenseWindowRole:
    """The kernel inputs for an admitted role; builds the family's library first."""
    reason = fused_dense_window_supported(bundle)
    if reason is not None:
        raise GrammarError(f"the fused dense identity refuses this role: {reason}")
    library = library_for(bundle.family)
    _ext(library)
    device = bundle.device
    cols = int(bundle.cols)
    pair, why = run_pair(bundle.runs, cols, rate_max=DENSE_RATE_MAX[bundle.family])
    assert pair is not None, why              # the predicate above admitted it
    return FusedDenseWindowRole(
        family=bundle.family, rows=int(bundle.rows), cols=cols,
        words=bundle.words.reshape(1, -1), table16=compose_dense_table(bundle, library),
        init=bundle.init_perm.reshape(1, -1),
        has_init=torch.tensor([1 if bundle.has_init else 0], dtype=torch.int32, device=device),
        wscale=bundle.scale.reshape(1, -1),
        runs=pair.reshape(1, 8), bdesc=block_desc(bundle.perm, int(pair[2]), cols),
        tile_words=pair_tile_words(pair), slot_words=slot_words_for_pair(pair))


def dense_split_max(cols: int) -> int:
    """The largest K split the dense launch takes at ``cols`` columns.

    The producers write item ``i + 2``'s descriptor into item ``i``'s slot
    once item ``i + 1``'s last chunk has waited for the chunk two before it
    to be consumed; the consumers read item ``i``'s slot before they release
    its first chunk.  Items ``i`` and ``i + 1`` of three chunks or more
    between them order the two, so every split item keeps two chunks:
    ``floor(nk / S) >= 2``, ``nk = cols / 32``.  The split launch's epilogue
    writes the raw partial from registers and reads no slot, so this is the
    whole bound.  The library refuses a larger split by name (tessera#805; the
    E2M1 launch's :func:`tessera.routed_fused_e2m1.dense_split_max` is the
    same bound on the same protocol).
    """
    return int(cols) // BK // 2


def dense_k_split(m: int, rows: int, cols: int, sms: int, *, tile_words: "int | None" = None) -> int:
    """How many ways to split K for one role at ``m`` rows: the bandwidth model.

    An item is 64 rows of ``x`` by 128 rows of the role, so ``items0 =
    ceil(m / 64) * ceil(rows / 128)`` (the last block partial on an N-tail).  When ``items0 >= sms`` every SM has work and
    the answer is 1 (prefill is untouched).  Below that, each split adds items
    and costs an fp32 partial written and read back; the time model is the
    wire bytes served by ``min(S * items0, sms)`` SMs at the per-SM share of
    the bandwidth, plus the partial traffic at full bandwidth:

        t(S) = wire * sms / min(S * items0, sms) + 2 * S * m * rows * 4

    with ``wire = rows * tile_words * 4 / 512`` -- the role's wire bytes, from
    its words per 512-row tile (``rows * cols / 2`` at rate 4, the default
    when ``tile_words`` is not given).  The minimiser over the integers
    ``1 .. min(dense_split_max(K), ceil(sms / items0))`` is returned; the
    constants are the SM count and the byte counts, nothing else.  The upper
    end is the launch's legality bound (:func:`dense_split_max`), not a
    tuning choice: the library refuses a larger split.
    """
    items0 = -(-m // BM) * -(-rows // BN)
    if items0 >= sms or m <= 0:
        return 1
    wire = rows * cols // 2 if tile_words is None else rows * int(tile_words) * 4 // 512
    best_s, best_t = 1, None
    for s in range(1, min(dense_split_max(cols), -(-sms // items0)) + 1):
        t = wire * sms / min(s * items0, sms) + 2.0 * s * m * rows * 4
        if best_t is None or t < best_t:
            best_s, best_t = s, t
    return best_s


def dense_forward(role: FusedDenseWindowRole, x: torch.Tensor, a_scale: "torch.Tensor | None",
                  out: torch.Tensor, counter: torch.Tensor, *, zeroed: bool = False) -> None:
    """One role's launch into ``out`` (a ``[M, rows]`` view, unit column stride).

    ``x`` is the family's A operand as the route quantised it (e4m3 + fp32
    ``a_scale`` for E4M3, bf16 for value), contiguous ``[M, cols]``; ``counter``
    is one int32 slot this call zeroes in-stream -- unless ``zeroed`` says the
    caller already zeroed it in-stream before this call (the module op zeroes
    all its roles' slots with one fill).  No host synchronisation, so a
    captured forward replays.
    """
    lib = _ext(role.library)
    m = int(x.shape[0])
    if m == 0:
        return
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    sms = _sm_count(index)
    s = dense_k_split(m, role.rows, role.cols, sms, tile_words=role.tile_words)
    if s > 1 and out.stride(0) % 4 != 0:
        # The split path's reduce stores four bf16 (uint2) per thread at
        # 4-aligned columns; a row stride that is only even would misalign
        # them, so such a view takes the unsplit path.
        s = 1
    # One zero-size fp32 placeholder stands for whichever of the split
    # workspace and ``a_scale`` the launch does not read (an unsplit launch
    # reads no workspace, the value family no ``a_scale``).
    empty = x.new_empty(0, dtype=torch.float32) if (s == 1 or a_scale is None) else None
    partial = torch.empty((s, m, role.rows), dtype=torch.float32, device=x.device) if s > 1 else empty
    slot = counter if counter.numel() == 1 else counter[:1]
    if not zeroed:
        slot.zero_()
    # The wide superblock only where K is not split (``dense_k_split`` is the
    # 64-route model, and a split launch has idle SMs to fill, not rows).
    bm = superblock_rows(role.library, 2, m, dense=True) if s == 1 else BM
    lib.dense_forward(
        bool(role.fp8), x, a_scale if a_scale is not None else empty,
        role.words, role.table16, role.init, role.has_init, role.wscale,
        role.runs, role.bdesc, int(role.tile_words), int(role.slot_words), slot, int(s), partial, out, sms,
        int(bm))
