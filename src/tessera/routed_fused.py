"""Native WINDOW class dispatch and the existing dense kernel entries.

The routed adapter uses one class dispatcher. Every class aliases its exact
word storage and uses the CUDA body for its declared schedule. Two fixed side
streams run the class projections. Events fork and join both streams inside
one opaque native operation. Each replay computes routing prefixes on the
device and copies each class start into its own projection counter.

The adapter retains the existing activation quantizer, SwiGLU arithmetic,
sorted-route activation boundary, flat down writes and fixed-order token sum.
Its execution telemetry names the class dispatcher. It does not qualify a
new serving contract row. Unsupported classes and unavailable native builds
refuse; no feature switch selects a second routed implementation.



THE DENSE CASE (contract v43).  A dense window Linear is the E = 1, top_k = 1,
unweighted case of the down projection, and the same kernel serves it: one
launch per role of a merged Linear (``routed_fused_kernel<FP8, 2, DENSE>``),
row ``m`` of ``x`` as route ``m``, the role's rows written into their column
slice of the module's output, and -- when ``ceil(M / 64) * rows / 128`` items
would leave SMs idle, in the only or the last wave, which is every decode
shape -- the K range split ``S`` ways into an fp32 workspace that a
fixed-order reduce sums before the one epilogue (:func:`dense_k_split` states
the model that picks ``S``).  It is
its own launch identity, ``tessera::fused_window_dense`` (the functional
custom op in ``serving.native_window``) with the decoders
``native_fused_window_dense`` (E4M3) and
``native_fused_window_dense_bf16`` (BF16), lane-bearing rows on the
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
import weakref

import torch

from .errors import GrammarError

__all__ = [
    "ENV_E4M3_MMA",
    "ENV_TOGGLE_DENSE",
    "ENV_DENSE_MODULE",
    "ENV_WIDE",
    "DENSE_RATE_MAX",
    "LIBRARIES",
    "FusedDenseWindowRole",
    "FusedLaunchPair",
    "FusedRoutedWindowMoE",
    "fused_launch_for_rates",
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
    "dense_forward_roles",
    "dense_k_split",
    "dense_fixup_split_max",
    "dense_k_split_bandwidth",
    "dense_k_split_makespan",
    "dense_module_launch_enabled",
    "dense_rates",
    "dense_split_max",
    "fused_dense_window_enabled",
    "fused_dense_window_supported",
    "fused_routed_window_supported",
    "library_for",
    "prepare_dense_role",
    "superblock_rows",
    "words_by_expert",
]

log = logging.getLogger(__name__)




#: ``TESSERA_DENSE_FUSED=0`` keeps the Triton ``tessera::window_gemm_dense``
#: for every dense module; unset or ``1`` takes this kernel's dense identity
#: wherever :func:`fused_dense_window_supported` admits every role.  Its own
#: toggle so the two identities can be measured against their predecessors
#: independently.
ENV_TOGGLE_DENSE = "TESSERA_DENSE_FUSED"
#: Default-off experiment read only by the E4M3 MMA build flags. Separate
#: retained extension banks must be used for matched original/paired binaries.
ENV_PAIRED_K32 = "TESSERA_ROUTED_FUSED_PAIRED_K32"
_paired_k32_choice = os.environ.get(ENV_PAIRED_K32, "0")
if _paired_k32_choice not in ("0", "1"):
    raise GrammarError(f"{ENV_PAIRED_K32}={_paired_k32_choice!r}; compile flag must be 0 or 1")
PAIRED_K32_BUILD = _paired_k32_choice == "1"
del _paired_k32_choice

#: ``TESSERA_DENSE_MODULE_LAUNCH=1`` launches an E4M3 module's roles together
#: (:func:`dense_forward_roles`: one launch, a K split reduced in the kernel)
#: and prices every dense K split with :func:`dense_k_split_makespan`.  Unset
#: or ``0`` -- the default -- keeps one launch per role and
#: :func:`dense_k_split_bandwidth` on both families, the dense identity as it
#: ran before tessera#778, until that PR's decode measurement clears the
#: other.  The two models pick different splits at decode (M <= 192 on the
#: GLM-5.3 MLP shapes) and a different split is a different K order, so the two
#: settings give different bits there; at a split they share, the module
#: launch is bitwise the per-role one.  Read per call by
#: :func:`dense_k_split` and the module op, so a captured forward records it
#: with its shapes.
ENV_DENSE_MODULE = "TESSERA_DENSE_MODULE_LAUNCH"
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
# Build-scoped experiment, frozen before the first extension load. Off/on arms
# require distinct processes and build directories, never a cached-module retarget.
ENV_MMA8_GATE_UP_B_PREFETCH = "TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH"
_mma8_b_choice = os.environ.get(ENV_MMA8_GATE_UP_B_PREFETCH, "0")
if _mma8_b_choice not in ("0", "1"):
    raise GrammarError(f"{ENV_MMA8_GATE_UP_B_PREFETCH}={_mma8_b_choice!r}; one of ('0', '1')")
MMA8_GATE_UP_B_PREFETCH = int(_mma8_b_choice)
del _mma8_b_choice
# Build-scoped experiment (tessera#739), frozen like the one above: the E4M3
# instruction's activation ring (``MMA8_A_RING`` in the kernel).  0 off; 1 the
# routed two-run launches.  It adds WORD_STAGES raw A tiles to the A region
# (``a_region_bytes``), so the layout restated here moves with it.
ENV_MMA8_A_RING = "TESSERA_ROUTED_FUSED_MMA8_A_RING"
_mma8_a_ring_choice = os.environ.get(ENV_MMA8_A_RING, "0")
if _mma8_a_ring_choice not in ("0", "1"):
    raise GrammarError(f"{ENV_MMA8_A_RING}={_mma8_a_ring_choice!r}; one of ('0', '1')")
MMA8_A_RING = int(_mma8_a_ring_choice)
del _mma8_a_ring_choice
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
#: The dense launch's fixed cost per item, as the wire bytes one SM streams in
#: that time at its share of the read rate (:func:`dense_k_split`): the
#: item's claim (one atomic), its first K chunk's latency and its epilogue.
#: Measured on GB10: at 32 items of a 4096 x 4096 role, forced splits of equal
#: share (S = 3, 6, 12: two, four and eight waves) cost c = 1.5 us more per
#: wave, which at 232.2 GB/s / 48 SMs is 7.3 KB (PENDING the sparklina
#: receipt).
DENSE_ITEM_FIXED_BYTES = 7300
#: The kernel's A/B stages (``STAGES`` in the source; the library publishes it).
#: The producers run at most this many K chunks ahead of the consumers, so an
#: item of at least ``STAGES + 1`` chunks guarantees that the descriptor and
#: row-scale slot two items back is free before it is rewritten; a split the
#: kernel reduces itself may not cut an item shorter (:func:`dense_fixup_split_max`).
#: Every launch, the fixup one included, is also bounded by :func:`dense_split_max`
#: (``K / 64``, tessera#805).
STAGES = 2
#: The most roles one dense launch takes (``MAX_ROLES`` in the source): a
#: module with more is launched in groups of at most this many.
MAX_ROLES = 8
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
#: The activation ring (:data:`MMA8_A_RING`) adds its WORD_STAGES 64-route tiles.
A_RING_BYTES_MMA8 = WORD_STAGES * BM * BK if MMA8_A_RING else 0
SMEM_FIXED_MMA8 = {k: v + A_RING_BYTES_MMA8 for k, v in {0: 47_312, 1: 47_312, 2: 30_736}.items()}
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
    """The A region (two tiles of ``bm`` rows, and on the E4M3 instruction the
    activation ring's WORD_STAGES when :data:`MMA8_A_RING` is on) of the
    kernel's layout."""
    tiles = 2 + (WORD_STAGES if mma8 and MMA8_A_RING else 0)
    return tiles * int(bm) * BK * (1 if mma8 else 2)


def launch_smem_bytes(mode: int, slot_words: int, *, mma8: bool = False, bm: int = BM, paired: bool = False) -> int:
    """The dynamic shared memory the launch takes at ``bm``-route
    superblocks: :func:`smem_bytes` with the A region at ``bm`` rows."""
    if paired:
        if not mma8 or mode not in (0, 2) or slot_words != 8 or bm != BM_WIDE:
            raise GrammarError("paired shared memory requires E4M3 MMA, mode0/2, R4 slot8, BMT128")
        # Existing layout owner, with four ordinary microtiles instead of two;
        # four word/history slots instead of the original three. Ring unchanged.
        micros = 2
        tables = 1 if mode == 2 else 2
        decoded = 2 * micros * (BK * BN + bm * BK)
        fixed = tables * TABLE_ENTRIES + decoded + 2 * BN * 4 + 2 * 8 * 4 + 16
        ring = DRING_STAGES * tables * BDESC_INTS * 4
        history = 2 * micros * 2 * BK * 4
        words = 2 * micros * 2 * BK * slot_words * 4
        return fixed + ring + history + words
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


@dataclasses.dataclass(frozen=True)
class FusedLaunchPair:
    """The fused launches one route pair takes on one library.

    ``rates`` is the pair's one or two column rates. ``slot_words`` is the
    launch's word-stage slot, derived the way :func:`slot_words_for_pair`
    derives it: the larger rate's :func:`slot_words_for_rate`, rounded up to
    a multiple of 4. ``stages_gate_up`` / ``smem_gate_up`` describe the
    two-table gate/up launch (mode 0) and ``stages_down`` / ``smem_down``
    the one-table down launch (mode 2) at that slot, each at its own
    :func:`word_stages`. ``fits_sm121`` says both launches fit the target
    platform's per-block opt-in shared memory. Geometry only: whether a
    rung may serve is the contract's, not this object's.
    """
    rates: tuple[int, ...]
    library: str
    slot_words: int
    stages_gate_up: int
    smem_gate_up: int
    stages_down: int
    smem_down: int
    fits_sm121: bool


def fused_launch_for_rates(rates, library: str) -> FusedLaunchPair:
    """The fused launch pair a route's column rates take on ``library``.

    The one home mapping a priced pair to its launch parameters: the census
    proof and any later serve plan read this instead of re-deriving the slot
    beside it. Refuses an unknown library and rates outside ``RATES`` by
    name; a pair the device cannot fit reports ``fits_sm121`` as ``False``
    so the caller, not this geometry, decides admission.
    """
    if library not in LIBRARIES:
        raise GrammarError(f"unknown fused library {library!r}; known: {sorted(LIBRARIES)}")
    seen = tuple(int(r) for r in rates)
    if len(seen) not in (1, 2) or len(set(seen)) != len(seen) or any(r not in RATES for r in seen):
        raise GrammarError(f"a fused launch pair takes one or two distinct rates of {list(RATES)}; got {list(rates)!r}")
    mma8 = library_mma8(library)
    slot = _round_up_4(max(slot_words_for_rate(r) for r in seen))
    stages_gate_up = word_stages(0, slot, mma8=mma8)
    smem_gate_up = _smem_bytes_at_stages(0, slot, stages_gate_up, mma8=mma8)
    stages_down = word_stages(2, slot, mma8=mma8)
    smem_down = _smem_bytes_at_stages(2, slot, stages_down, mma8=mma8)
    return FusedLaunchPair(
        rates=tuple(sorted(seen)),
        library=library,
        slot_words=slot,
        stages_gate_up=stages_gate_up,
        smem_gate_up=smem_gate_up,
        stages_down=stages_down,
        smem_down=smem_down,
        fits_sm121=smem_gate_up <= SM121_MAX_DYNAMIC_SMEM and smem_down <= SM121_MAX_DYNAMIC_SMEM,
    )


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


def fused_routed_unit_shape_refusal(family: str, part: str, *, rows: int, cols: int,
                                    rates, window_bits: int) -> "str | None":
    """The native shape rule on one verified unit manifest.

    The exporter applies this rule before it writes a required class stack.
    Every accepted native unit retains its selected table and launch descriptors.
    An unsupported unit refuses the complete stack; no compact serving fallback remains.
    ``part`` names gate, up or down.

    The rule checks family, window bits, columns, one or two adjacent rates,
    word-stage slots, shared-memory limits and native row multiples.
    The runtime handles device, arithmetic, activation quantization and column order.
    The caller selects the native library.
    Return the refusal, or None for an accepted shape.
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


def _paired_k32_build_enabled(mma8: bool) -> bool:
    return bool(mma8 and PAIRED_K32_BUILD)


def dense_module_launch_enabled() -> bool:
    """:data:`ENV_DENSE_MODULE` as set now: ``1``, or ``0`` / unset (the
    default); any other value is refused by name rather than read as either."""
    want = os.environ.get(ENV_DENSE_MODULE, "0")
    if want not in ("0", "1"):
        raise GrammarError(f"{ENV_DENSE_MODULE}={want!r}: expected 0 or 1")
    return want == "1"


def _cflags(token: str, fp8: bool, mma8: bool = False, fp4: bool = False) -> list:
    """A library's compile flags.  ``fp4`` is the E2M1 family's library
    (``tessera.routed_fused_e2m1``): its define, and the architecture-specific
    target its block-scaled FP4 instruction exists on.  The other libraries'
    flags do not move."""
    from .serving.backend import offload_flags

    return ["-O3", "-lineinfo", "-std=c++17",
            f"-DTESSERA_ROUTED_FUSED_FP8={1 if fp8 else 0}",
            f"-DTESSERA_ROUTED_FUSED_MMA8={1 if mma8 else 0}",
            *([f"-D{ENV_MMA8_GATE_UP_B_PREFETCH}={MMA8_GATE_UP_B_PREFETCH}"] if mma8 else []),
            *([f"-D{ENV_MMA8_A_RING}={MMA8_A_RING}"] if mma8 else []),
            *(["-DTESSERA_ROUTED_FUSED_FP4=1"] if fp4 else []),
            *(["-DTESSERA_ROUTED_FUSED_PAIRED_K32=1"] if _paired_k32_build_enabled(mma8) else []),
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
    # Experimental selection is frozen by the existing per-library owner cache.
    value_prefetch = os.environ.get("TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH", "0") if library == "value" else "0"
    if value_prefetch not in ("0", "4"):
        raise GrammarError("TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH must be 0 or 4")
    source_module = module
    if value_prefetch == "4":
        module = "tessera_routed_fused_value_prefetch4"

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
        if value_prefetch == "4":
            return load(
                name="tessera_routed_fused_value_prefetch4",
                sources=[src], build_directory=build,
                extra_cuda_cflags=_cflags(token, False) + ["-DTESSERA_ROUTED_FUSED_VALUE_A_PREFETCH=4"],
                verbose=verbose)
        return load(
            name="tessera_routed_fused_value",  # literal: the contract scanner reads it
            sources=[src], build_directory=build,
            extra_cuda_cflags=_cflags(token, False), verbose=verbose)

    lib = build_library(module, source_module, compile_fn)
    if library == "value" and lib.VALUE_A_PREFETCH != int(value_prefetch):
        raise GrammarError("the value library prefetch distance differs from its frozen selection")
    dense_max = DENSE_RATE_MAX["e4m3" if fp8 else "value"]
    for name, want in (("BM", BM), ("BN", BN), ("HALF", HALF), ("BK", BK),
                       ("DENSE_ROW_QUANTUM", DENSE_ROW_QUANTUM),
                       ("RATE_MIN", RATE_MIN), ("ROUTED_RATE_MAX", RATE_MAX),
                       ("RATE_MAX", dense_max), ("SLOT_WORDS_MAX", slot_words_for_rate(dense_max)),
                       ("BDESC_INTS", BDESC_INTS), ("WINDOW_BITS", WINDOW_BITS), ("FAMILY_FP8", fp8),
                       ("FAMILY_MMA8", mma8), ("PAIRED_K32_BUILD", _paired_k32_build_enabled(mma8)),
                       ("WORD_STAGES", WORD_STAGES),
                       ("WORD_STAGES_MIN", WORD_STAGES_MIN), ("STAGES", STAGES), ("MAX_ROLES", MAX_ROLES),
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
    if mma8:
        actual = getattr(lib, "MMA8_GATE_UP_B_PREFETCH", None)
        if actual != bool(MMA8_GATE_UP_B_PREFETCH):
            raise GrammarError(
                f"{module} was built with MMA8_GATE_UP_B_PREFETCH={actual!r}; "
                f"this process expects {bool(MMA8_GATE_UP_B_PREFETCH)!r}")
        actual = getattr(lib, "MMA8_A_RING", None)
        if actual != MMA8_A_RING:
            raise GrammarError(
                f"{module} was built with MMA8_A_RING={actual!r}; this process expects {MMA8_A_RING!r}")
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
    ascending, then high-rate ascending), with raw values through the dot
    and FP32 row scales on the accumulator epilogue. One run table and one
    ``tile_words`` per class: the kernel takes a single tile stride for all
    experts in that class and for both gate and up. Unsupported classes
    refuse with this reason; there is no compact runtime substitute.
    """
    bundles = {"gate": gate, "up": up, "down": down}
    fam = down.family
    if fam not in ("value", "e4m3"):
        return f"family {fam!r} is not a window family"
    e = int(down.experts)
    for name, b in bundles.items():
        if getattr(b, "perm_all", ...) is None:
            return f"{name} compact planes were retired; use its already-prepared fused owner"
        if b.family != fam:
            return f"{name} family {b.family!r} differs from down's {fam!r}"
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
            # The piece-major reader is instantiated only in the MMA E4M3
            # library.  An explicit TESSERA_FUSED_E4M3_MMA=f16 selects the
            # f16-byte reader, which reads legacy words only: refuse here, before
            # the device query or any smem/extension build below.
            lib = library_for(fam)
            if not library_mma8(lib):
                return (f"{name} is piece_major, which the lane reads only on the MMA E4M3 "
                        f"reader ({ENV_E4M3_MMA}={os.environ.get(ENV_E4M3_MMA, E4M3_MMA_DEFAULT)!r} "
                        f"selects {lib!r})")
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
    prefixes: dict[int, torch.Tensor]  # Each declared width: [E + 1] int32
    tokens: int
    top_k: int

    @property
    def routes(self) -> int:
        return self.tokens * self.top_k

    def superblocks(self, bm: int) -> torch.Tensor:
        """Return the absolute prefix for a bound-kernel superblock width."""
        try:
            return self.prefixes[bm]
        except KeyError:
            raise GrammarError(f"no {bm}-route superblock offsets for this step") from None


def _item_off(counts: torch.Tensor, bm: int) -> torch.Tensor:
    """The prefix sum of ``ceil(counts_e / bm)``: [E + 1] int32."""
    item_off = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=counts.device)
    item_off[1:] = torch.cumsum((counts + (bm - 1)) // bm, 0, dtype=torch.int32)
    return item_off


def _routing_tables(expert_ids: torch.Tensor, routing_weights: torch.Tensor, experts: int,
                    device: torch.device, widths: tuple[int, ...]) -> _Routing:
    """Sort the routes once and derive every declared width from the same counts."""
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
    prefixes = {bm: _item_off(counts, bm) for bm in widths}
    return _Routing(offsets=offsets, flat_sorted=flat_sorted, rw_sorted=rw_sorted,
                    prefixes=prefixes, tokens=tokens, top_k=top_k)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def grouped_class_view(bundle, start: int, end: int):
    """Bound every grouped plane to one storage class, without copying weights.

    Offsets are read once at load. Dynamic routing offsets are never read here.
    The class has its own constant word/run origins; route origins stay absolute.
    """
    e = int(bundle.experts)
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= e:
        raise GrammarError(f"class extent [{start!r}, {end!r}) is outside [0, {e}) or empty")
    if bundle.word_off is None or bundle.run_off is None:
        raise GrammarError("class views need the load-time word and run bounds, not a retired projection")
    if not bundle.words_all.is_contiguous() or not bundle.runs_all.is_contiguous():
        raise GrammarError("class words and runs must be contiguous zero-copy storage")
    words = bundle.words_all.view(-1)
    runs = bundle.runs_all.view(-1, 4)
    word_off = bundle.word_off.tolist()
    widths = bundle.total_words.tolist()
    run_off = bundle.run_off.tolist()
    if len(word_off) != e or len(widths) != e or len(run_off) != e + 1:
        raise GrammarError("class word/run bounds do not cover the expert axis")
    cursor = 0
    for off, width in zip(word_off, widths):
        if off != cursor or width <= 0:
            raise GrammarError("class word bounds contain a hole, overlap or empty expert")
        cursor += width
    if cursor != words.numel():
        raise GrammarError("class word bounds do not exactly cover the retained words")
    if run_off[0] != 0 or run_off[-1] != runs.shape[0] or any(
            a >= b for a, b in zip(run_off, run_off[1:])):
        raise GrammarError("class run bounds contain a hole, overlap or empty expert")
    width = widths[start]
    if any(w != width for w in widths[start:end]):
        raise GrammarError("class words do not have one per-expert stride")
    w0 = word_off[start]
    r0, r1 = run_off[start], run_off[end]
    count = end - start
    changes = {}
    for name in ("table_all", "codes_all", "native_all", "scale_all", "init_all", "has_init",
                 "tile_words", "total_words", "perm_all"):
        tensor = getattr(bundle, name)
        changes[name] = tensor if tensor is None or tensor.numel() == 0 else tensor[start:end]
    return dataclasses.replace(bundle, **changes, experts=count,
        words_all=words.narrow(0, w0, width * count).view(count, width),
        runs_all=runs.narrow(0, r0, r1 - r0),
        word_off=torch.arange(count, device=bundle.device, dtype=bundle.word_off.dtype) * width,
        run_off=bundle.run_off[start:end + 1] - r0)


def _native_view(bundle, table):
    return dataclasses.replace(bundle, table_all=table, codes_all=None, native_all=None,
        runs_all=None, word_off=None, tile_words=None, total_words=None, run_off=None, perm_all=None)


def _profile_schedule(bundle, q256: int, role: str) -> None:
    from fractions import Fraction
    from .grammar import bresenham_rate_schedule

    schedule = bresenham_rate_schedule(Fraction(q256, 256), int(bundle.cols), cap=RATE_MAX)
    expected = [(rate, schedule.count(rate)) for rate in sorted(set(schedule))]
    for expert_runs in bundle.runs_all.view(bundle.experts, -1, 4):
        actual = [(int(row[0]), int(row[2])) for row in expert_runs.tolist()]
        if actual != expected:
            raise GrammarError(f"{role} class q256 {q256} disagrees with stored schedule {actual}; expected {expected}")


@dataclasses.dataclass(frozen=True)
class _WindowClass:
    start: int
    end: int
    gate: object
    up: object
    down: object

    table_gate: torch.Tensor
    table_up: torch.Tensor
    table_down: torch.Tensor
    words_gate: torch.Tensor
    words_up: torch.Tensor
    words_down: torch.Tensor
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


@dataclasses.dataclass(frozen=True)
class _LutClassKernel:
    """Today's native LUT binding; routing orchestration owns no launch details."""
    library: str
    module: object

    def work_shape(self, mode, tokens, index, parameters):
        down = mode == 2
        projection = 3 * index + (2 if down else 0)
        bm = superblock_rows(self.library, mode, tokens)
        units = parameters["wscales"][projection].shape[1] // (BN if down else HALF)
        return bm, units

    def prepare_input(self, x, a_scale, rows, family, device):
        # This decoder needs no sentinel. A register-direct binding supplies
        # its own M+1 zero-row operand and persistent K-part scratch.
        return quantized_routed_input(x, a_scale, rows, family, device)

    def launch(self, mode, x, a_scale, *, index, start, end, prefix, counter,
               routing, parameters, bm, work_units, empty_scale, a_row_mode,
               mul_weight, limit, out):
        if counter is None:
            raise GrammarError("the LUT class kernel requires a claim counter")
        down = mode == 2
        projection = 1 if down else 0
        p0 = 3 * index + (2 if down else 0)
        p1 = p0 if down else p0 + 1
        p = parameters
        device = x.device.index if x.device.index is not None else torch.cuda.current_device()
        self.module.routed_fused_forward(mode, self.library != "value", x,
            a_scale if a_scale is not None else empty_scale,
            p["words"][p0], p["words"][p1], p["tables"][p0], p["tables"][p1],
            p["inits"][p0], p["inits"][p1], p["has_inits"][p0], p["has_inits"][p1],
            p["wscales"][p0], p["wscales"][p1], p["runs"][p0], p["runs"][p1],
            p["bdescs"][p0], p["bdescs"][p1],
            p["tile_words"][2 * index + projection], p["slot_words"][2 * index + projection],
            p["piece_major"], routing.offsets[start:end + 1], routing.flat_sorted, routing.rw_sorted,
            prefix[start:end + 1], counter, routing.top_k, a_row_mode, mul_weight, limit,
            out, _sm_count(device), bm)


@dataclasses.dataclass(frozen=True)
class _UniformProjection:
    bundle: object
    words: torch.Tensor
    table: torch.Tensor
    runs: torch.Tensor
    bdesc: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _UniformWindowKernel:
    """One load-bound stack; every forward uses the old direct native launch."""
    library: str
    module: object
    gate: _UniformProjection
    up: _UniformProjection
    down: _UniformProjection
    counters: tuple
    empty: torch.Tensor
    tile_words_gate_up: int
    tile_words_down: int
    slot_words_gate_up: int
    slot_words_down: int
    piece_major: bool

    def routing(self, expert_ids, routing_weights, modes=(0, 2)):
        widths = tuple(dict.fromkeys(superblock_rows(self.library, mode, expert_ids.shape[0]) for mode in modes))
        return _routing_tables(expert_ids, routing_weights, self.down.bundle.experts,
                               self.down.bundle.device, widths)

    def prepare_input(self, x, a_scale, rows, family, device):
        return quantized_routed_input(x, a_scale, rows, family, device)

    def launch(self, mode, x, a_scale, routing, *, a_row_mode, mul_weight, limit, out):
        if mode == 2:
            p0 = p1 = self.down
            tile_words, slot_words = self.tile_words_down, self.slot_words_down
            counter = self.counters[1]
        else:
            p0, p1 = self.gate, self.up
            tile_words, slot_words = self.tile_words_gate_up, self.slot_words_gate_up
            counter = self.counters[0]
        counter.zero_()
        bm = superblock_rows(self.library, mode, routing.tokens)
        device = x.device.index if x.device.index is not None else torch.cuda.current_device()
        b0, b1 = p0.bundle, p1.bundle
        self.module.routed_fused_forward(mode, self.library != "value", x,
            a_scale if a_scale is not None else self.empty,
            p0.words, p1.words, p0.table, p1.table,
            b0.init_all, b1.init_all, b0.has_init, b1.has_init,
            b0.scale_all, b1.scale_all, p0.runs, p1.runs, p0.bdesc, p1.bdesc,
            tile_words, slot_words, self.piece_major,
            routing.offsets, routing.flat_sorted, routing.rw_sorted, routing.superblocks(bm),
            counter, routing.top_k, a_row_mode, mul_weight, limit, out, _sm_count(device), bm)

    def forward(self, x, expert_ids, routing_weights, *, input_weight, limit, shared):
        tokens = expert_ids.shape[0]
        if input_weight:
            x = x * routing_weights.reshape(-1, 1).to(x.dtype)
        routing = self.routing(expert_ids, routing_weights)
        family, device = self.down.bundle.family, self.down.bundle.device
        xq, a1 = self.prepare_input(x, None, tokens, family, device)
        act = torch.empty((routing.routes, self.down.bundle.cols), dtype=torch.bfloat16, device=device)
        self.launch(0, xq, a1, routing, a_row_mode=0, mul_weight=False, limit=limit, out=act)
        aq, a2 = self.prepare_input(act, None, routing.routes, family, device)
        routed = torch.empty((routing.routes, self.down.bundle.rows), dtype=torch.bfloat16, device=device)
        self.launch(2, aq, a2, routing, a_row_mode=1, mul_weight=not input_weight,
                    limit=float("inf"), out=routed)
        out = torch.empty((tokens, self.down.bundle.rows), dtype=torch.bfloat16, device=device)
        if shared is None:
            self.module.token_sum(routed, out, routing.top_k)
        else:
            self.module.token_sum_shared(routed, shared, out, routing.top_k)
        return out



@dataclasses.dataclass(frozen=True)
class _DispatchResources:
    streams: tuple
    ready: object
    finished: tuple
    empty: torch.Tensor
    kernel: object


# The registry holds no weights or adapters. The adapter owns these resources.
_dispatch_resources = weakref.WeakValueDictionary()


def _make_dispatch_resources(device: torch.device, kernel) -> _DispatchResources:
    resources = _DispatchResources(
        streams=tuple(torch.cuda.Stream(device=device) for _ in range(2)),
        ready=torch.cuda.Event(), finished=tuple(torch.cuda.Event() for _ in range(2)),
        empty=torch.empty(0, dtype=torch.float32, device=device), kernel=kernel)
    # CUDA creates event handles at the first record. Create them at load,
    # before graph capture or a timed forward uses this adapter.
    caller = torch.cuda.current_stream(device)
    resources.ready.record(caller)
    for stream, finished in zip(resources.streams, resources.finished):
        stream.wait_event(resources.ready)
        finished.record(stream)
        caller.wait_event(finished)
    return resources


def _retain_dispatch_resources(resources: _DispatchResources) -> str:
    key = str(id(resources))
    _dispatch_resources[key] = resources
    return key


def resolve_dispatch_resources(key: str) -> _DispatchResources:
    resources = _dispatch_resources.get(key)
    if resources is None:
        raise GrammarError("the routed dispatcher resource owner no longer exists")
    return resources





@dataclasses.dataclass(frozen=True)
class FusedRoutedWindowMoE:
    """The sole native WINDOW routed path, including the one-class case.

    IDs are STORAGE coordinates. The plugin owns the single global-to-storage
    remap; positions, weights, sorted-route activation boundary and flat down
    writes are unchanged. All class bundles alias the exact loader storage.
    """
    PROFILER_LABEL = "tessera_routed_window_classes"
    supports_shared_fold = True
    gate: object
    up: object
    down: object
    family: str
    library: str
    expert_classes: list
    classes: tuple
    table_gate: torch.Tensor
    table_up: torch.Tensor
    table_down: torch.Tensor
    counters: torch.Tensor
    operands: dict
    class_issue_order: tuple
    dispatch_resources: _DispatchResources | None
    resource_key: str | None
    activation: str = "silu"
    uniform: _UniformWindowKernel | None = None

    @classmethod
    def from_bundles(cls, gate, up, down, *, expert_classes,
                     activation: str = "silu") -> "FusedRoutedWindowMoE":
        from .expert_classes import normalize_expert_classes, validate_gate_up_schedule

        descriptors = normalize_expert_classes(expert_classes, int(down.experts))
        if activation != "silu":
            raise GrammarError(f"activation {activation!r} is not served; the class dispatcher computes silu")
        if any(b.experts != down.experts for b in (gate, up)):
            raise GrammarError("gate/up/down expert counts disagree")
        library = library_for(down.family)
        views = []
        for desc in descriptors:
            start, end = desc["start"], desc["end"]
            g, u, d = (grouped_class_view(b, start, end) for b in (gate, up, down))
            validate_gate_up_schedule(*desc["q256"]["w13"], int(g.cols), target=f"class [{start}, {end})")
            reason = fused_routed_window_supported(g, u, d)
            if reason is not None:
                raise GrammarError(f"native class [{start}, {end}) refuses: {reason}")
            for role, b, q in (("gate", g, desc["q256"]["w13"][0]),
                               ("up", u, desc["q256"]["w13"][1]),
                               ("down", d, desc["q256"]["w2"][0])):
                _profile_schedule(b, q, role)
            views.append((desc, g, u, d))
        # Build at load. There is no compact substitute or late-forward build.
        module = _ext(library)
        tables = tuple(compose_table(b, library) for b in (gate, up, down))
        classes = []
        for desc, g, u, d in views:
            start, end = desc["start"], desc["end"]
            rg, bg, twg, swg = projection_tables(g)
            ru, bu, twu, swu = projection_tables(u)
            rd, bd, twd, swd = projection_tables(d)
            if twg != twu or not torch.equal(rg, ru):
                raise GrammarError(f"class [{start}, {end}) gate/up schedules or tile strides disagree")
            tg, tu, td = (table[start:end] for table in tables)
            classes.append(_WindowClass(start, end, _native_view(g, tg), _native_view(u, tu),
                _native_view(d, td), tg, tu, td, words_by_expert(g), words_by_expert(u),
                words_by_expert(d), rg, ru, rd, bg, bu, bd, twg, twd, max(swg, swu), swd))
        operands = {name: [getattr(c, field) for c in classes for field in fields]
            for name, fields in {
                "words": ("words_gate", "words_up", "words_down"),
                "tables": ("table_gate", "table_up", "table_down"),
                "runs": ("runs_gate", "runs_up", "runs_down"),
                "bdescs": ("bdesc_gate", "bdesc_up", "bdesc_down")}.items()}
        for name, field in (("inits", "init_all"), ("has_inits", "has_init"), ("wscales", "scale_all")):
            operands[name] = [getattr(b, field) for c in classes for b in (c.gate, c.up, c.down)]
        operands.update(starts=[c.start for c in classes], ends=[c.end for c in classes],
            tile_words=[v for c in classes for v in (c.tile_words_gate_up, c.tile_words_down)],
            slot_words=[v for c in classes for v in (c.slot_words_gate_up, c.slot_words_down)],
            piece_major=down.word_layout != "legacy")
        counters = torch.empty((len(classes), 2), dtype=torch.int32, device=down.device)
        uniform = None
        resources = None
        resource_key = None
        if len(classes) == 1:
            c = classes[0]
            projections = tuple(_UniformProjection(getattr(c, role), getattr(c, "words_" + role),
                getattr(c, "table_" + role), getattr(c, "runs_" + role), getattr(c, "bdesc_" + role))
                for role in ("gate", "up", "down"))
            uniform = _UniformWindowKernel(library, module, *projections,
                (counters[0, :1], counters[0, 1:2]), torch.empty(0, dtype=torch.float32, device=down.device),
                c.tile_words_gate_up, c.tile_words_down, c.slot_words_gate_up, c.slot_words_down,
                down.word_layout != "legacy")
        else:
            resources = _make_dispatch_resources(down.device, _LutClassKernel(library, module))
            resource_key = _retain_dispatch_resources(resources)
        return cls(*(_native_view(b, t) for b, t in zip((gate, up, down), tables)),
            family=down.family, library=library,
            expert_classes=descriptors, classes=tuple(classes), table_gate=tables[0],
            table_up=tables[1], table_down=tables[2], counters=counters,
            operands=operands, class_issue_order=tuple(range(len(classes))),
            dispatch_resources=resources, resource_key=resource_key, activation=activation, uniform=uniform)

    @property
    def piece_major(self) -> bool:
        return self.down.word_layout != "legacy"

    @property
    def launch_pair(self):
        from .serving.scheme import routed_class_launch_pair
        return routed_class_launch_pair(self.library, uniform=self.uniform is not None)

    @property
    def experts(self):
        return int(self.down.experts)

    @property
    def device(self):
        return self.down.device

    @property
    def fp8(self):
        return self.family == "e4m3"

    def named_tables(self):
        for role in ("gate", "up", "down"):
            yield f"routed_classes.table_{role}", getattr(self, f"table_{role}")
        for i, c in enumerate(self.classes):
            for field in dataclasses.fields(c):
                value = getattr(c, field.name)
                if isinstance(value, torch.Tensor):
                    yield f"routed_classes.class_{i}.{field.name}", value
        yield "routed_classes.counters", self.counters
        yield "routed_classes.empty_scale", (self.uniform.empty if self.uniform is not None else self.dispatch_resources.empty)

    def resident_bytes(self):
        from .serving.residency import resident_storage_bytes
        return resident_storage_bytes(self.named_tables())

    def _routing(self, expert_ids, routing_weights, *, modes=(0, 2)):
        if self.uniform is not None:
            return self.uniform.routing(expert_ids, routing_weights, modes)
        from .routed_class_dispatch import declared_route_widths
        tokens = expert_ids.shape[0] if expert_ids.ndim else 0
        widths = declared_route_widths(self.dispatch_resources.kernel, tokens,
                                       self.class_issue_order, self.operands, modes)
        return _routing_tables(expert_ids, routing_weights, self.experts, self.device, widths)

    def _launch(self, mode, x, a_scale, routing, *, a_row_mode, mul_weight, limit, out):
        if self.uniform is not None:
            return self.uniform.launch(mode, x, a_scale, routing, a_row_mode=a_row_mode,
                                       mul_weight=mul_weight, limit=limit, out=out)
        from . import routed_class_dispatch
        routed_class_dispatch.dispatch_class_projection(mode, x, a_scale, routing,
            parameters=self.operands,
            starts=self.operands["starts"], ends=self.operands["ends"],
            issue_order=self.class_issue_order, counters=self.counters,
            resources=self.dispatch_resources, mul_weight=mul_weight, limit=limit,
            a_row_mode=a_row_mode, out=out)

    def _quantized(self, x, a_scale, rows):
        kernel = self.uniform if self.uniform is not None else self.dispatch_resources.kernel
        return kernel.prepare_input(x, a_scale, rows, self.family, self.device)

    def _check_x(self, x, rows, cols):
        if x.dim() != 2 or tuple(x.shape) != (rows, cols) or x.device != self.device:
            raise GrammarError(f"x must be [{rows}, {cols}] on {self.device}, got {tuple(x.shape)} on {x.device}")
        return x if x.is_contiguous() else x.contiguous()

    def __call__(self, x, expert_ids, routing_weights, *, apply_router_weight_on_input=False,
                 swiglu_limit=None, shared=None):
        from .native_window_moe import checked_swiglu_limit

        limit = checked_swiglu_limit(swiglu_limit)
        if expert_ids.dim() != 2 or routing_weights.shape != expert_ids.shape:
            raise GrammarError("expert_ids and routing_weights must share [T, top_k]")
        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise GrammarError("expert_ids must be int32 or int64 storage IDs")
        if expert_ids.device != self.device or routing_weights.device != self.device:
            raise GrammarError("routing tensors must live on the compute device")
        tokens, top_k = expert_ids.shape
        x = self._check_x(x, tokens, self.gate.cols)
        if x.dtype != torch.bfloat16:
            raise GrammarError("the routed forward takes bf16 hidden states")
        if apply_router_weight_on_input and top_k != 1:
            raise GrammarError("apply_router_weight_on_input is only implemented for topk=1")
        if shared is not None and (shared.dtype != torch.bfloat16 or not shared.is_contiguous()
                or tuple(shared.shape) != (tokens, self.down.rows) or shared.device != self.device):
            raise GrammarError(f"shared must be contiguous bf16 {(tokens, self.down.rows)} on {self.device}")
        if tokens == 0:
            return torch.empty((0, self.down.rows), dtype=torch.bfloat16, device=self.device)
        if self.uniform is not None:
            return self.uniform.forward(x, expert_ids, routing_weights,
                input_weight=apply_router_weight_on_input, limit=limit if limit is not None else float("inf"), shared=shared)
        from .serving.native_window import _routed_window_classes
        return _routed_window_classes(x, expert_ids, routing_weights, shared,
            **self.operands, issue_order=list(self.class_issue_order), counters=self.counters,
            library=self.library, resource_key=self.resource_key,
            input_weight=apply_router_weight_on_input, swiglu_limit=limit if limit is not None else float("inf"))

    def gate_up(self, x, expert_ids, routing_weights, a_scale=None, *, preserve=True,
                apply_router_weight_on_input=False):
        if not preserve or apply_router_weight_on_input:
            raise GrammarError("gate_up is the unweighted route-preserving projection")
        routing = self._routing(expert_ids, routing_weights, modes=(1,))
        x = self._check_x(x, routing.tokens, self.gate.cols)
        out = torch.empty((routing.tokens, routing.top_k, 2*self.down.cols), dtype=torch.bfloat16,
                          device=self.device)
        if routing.tokens:
            xq, a1 = self._quantized(x, a_scale, routing.tokens)
            self._launch(1, xq, a1, routing, a_row_mode=0, mul_weight=False, limit=float("inf"), out=out)
        return out

    def down_routes(self, x, expert_ids, routing_weights, a_scale=None, *, route_input=True,
                    apply_router_weight_on_input=False, round_routes=True):
        if not route_input or not round_routes:
            raise GrammarError("down_routes needs route-indexed input and the bf16 route boundary")
        routing = self._routing(expert_ids, routing_weights, modes=(2,))
        x = self._check_x(x, routing.routes, self.down.cols)
        out = torch.empty((routing.tokens, self.down.rows), dtype=torch.bfloat16, device=self.device)
        if routing.tokens:
            xq, a2 = self._quantized(x, a_scale, routing.routes)
            routed = torch.empty((routing.routes, self.down.rows), dtype=torch.bfloat16, device=self.device)
            self._launch(2, xq, a2, routing, a_row_mode=2, mul_weight=not apply_router_weight_on_input,
                         limit=float("inf"), out=routed)
            _ext(self.library).token_sum(routed, out, routing.top_k)
        return out


def quantized_routed_input(x, a_scale, rows, family, device):
    if family == "value":
        if x.dtype != torch.bfloat16 or a_scale is not None:
            raise GrammarError("the value family takes bf16 x and no activation scale")
        return x, None
    if x.dtype == torch.float8_e4m3fn:
        if a_scale is None:
            raise GrammarError("an fp8 x must carry its per-row activation scale")
        a_scale = a_scale.reshape(-1)
        if a_scale.numel() != rows or a_scale.dtype != torch.float32 or a_scale.device != device:
            raise GrammarError(f"a_scale must be fp32 [{rows}] on {device}")
        return x, a_scale.contiguous()
    if x.dtype != torch.bfloat16 or a_scale is not None:
        raise GrammarError("the E4M3 family takes bf16 or scaled fp8 x")
    from .serving.native_ops import native_fp8_quant
    xq, scale = native_fp8_quant(x)
    return xq, scale.reshape(-1)





# ---------------------------------------------------------------------------
# the dense identity: one role of a dense Linear as the E = 1 case
# ---------------------------------------------------------------------------

def fused_dense_window_supported(bundle) -> "str | None":
    """Why the dense identity refuses a prepared role, or ``None`` when it serves it.

    ``bundle`` is a ``window_gemm.PreparedWindowGemm`` (the frozen role the
    Triton dense GEMM runs).  The kernel reads the routed lane's wire shape --
    one or two column-rate runs (the two bracketing the root, at rates up to
    the family's :data:`DENSE_RATE_MAX`: 14 on the value family, 8 on E4M3),
    window bits 14, the packer's column order, and the row scale on the fp32
    accumulator after the dot on both families -- plus the dense tile:
    rows a multiple of ``DENSE_ROW_QUANTUM`` (one 128-column B block per item,
    the last one partial when 128 does not divide the rows) and columns a
    multiple of 32 and at least 128.  A role outside it keeps
    ``tessera::window_gemm_dense``, and the reason is the string returned
    here so a load log can say which.
    """
    from .kernel_window_gemv import require_legacy_word_layout
    try:
        require_legacy_word_layout(getattr(bundle, "word_layout", "legacy"),
                                   "the fused dense window reader")
    except GrammarError as exc:
        return str(exc)
    if not fused_dense_window_enabled():
        return f"disabled by {ENV_TOGGLE_DENSE}=0"
    fam = bundle.family
    if fam not in ("value", "e4m3"):
        return f"family {fam!r} is not a window family"
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


def dense_fixup_split_max(cols: int) -> int:
    """The largest K split a launch that reduces the split in the kernel
    (:func:`dense_forward_roles`, ``fixup``) takes at ``cols`` columns: every
    split keeps at least ``STAGES + 1`` of the ``cols / 32`` K chunks
    (:data:`STAGES`), and the library refuses a larger one there.  It is also
    :func:`dense_k_split_makespan`'s range.  Every launch, the fixup one
    included, is also bounded by :func:`dense_split_max` (tessera#805); with
    ``STAGES >= 1`` the fixup bound is the tighter of the two."""
    return max(1, (int(cols) // BK) // (STAGES + 1))


def dense_k_split(m: int, rows: int, cols: int, sms: int, *, tile_words: "int | None" = None,
                  blocks: "int | None" = None) -> int:
    """The K split a dense launch runs at ``m`` rows: :func:`dense_k_split_makespan`
    under ``TESSERA_DENSE_MODULE_LAUNCH=1``, :func:`dense_k_split_bandwidth`
    otherwise (:data:`ENV_DENSE_MODULE`).  Every launch asks this name, so a
    test or bench that forces a split replaces it here."""
    model = dense_k_split_makespan if dense_module_launch_enabled() else dense_k_split_bandwidth
    return model(m, rows, cols, sms, tile_words=tile_words, blocks=blocks)


def dense_k_split_bandwidth(m: int, rows: int, cols: int, sms: int, *, tile_words: "int | None" = None,
                            blocks: "int | None" = None) -> int:
    """How many ways to split K for one launch at ``m`` rows: the bandwidth
    model, the default (:data:`ENV_DENSE_MODULE`).

    An item is 64 rows of ``x`` by 128 rows of a role, so ``items0 =
    ceil(m / 64) * blocks`` with ``blocks = ceil(rows / 128)`` for one role
    (the last block partial on an N-tail; a launch of several roles passes
    their summed rows and blocks).  When ``items0 >= sms`` every SM has work
    and the answer is 1 (prefill is untouched).  Below that, each split adds
    items and costs an fp32 partial written and read back; the time model is
    the wire bytes served by ``min(S * items0, sms)`` SMs at the per-SM share
    of the bandwidth, plus the partial traffic at full bandwidth:

        t(S) = wire * sms / min(S * items0, sms) + 2 * S * m * rows * 4

    with ``wire = rows * tile_words * 4 / 512`` -- the wire bytes, from the
    words per 512-row tile (``rows * cols / 2`` at rate 4, the default when
    ``tile_words`` is not given).  The minimiser over the integers ``1 ..
    min(dense_split_max(K), ceil(sms / items0))`` is returned, the smaller ``S``
    on a tie; the constants are the SM count and the byte counts, nothing else.
    The upper end is the launch's legality bound (:func:`dense_split_max`,
    tessera#805), not a tuning choice: the library refuses a larger split.

    This is the model and the range the default launch takes with
    ``TESSERA_DENSE_MODULE_LAUNCH`` unset, unchanged by tessera#778.  The
    in-kernel fixup keeps the tighter :func:`dense_fixup_split_max` range, so
    a launch that reduces in the kernel is capped there by its caller.
    """
    if m <= 0:
        return 1
    items0 = -(-m // BM) * (-(-rows // BN) if blocks is None else int(blocks))
    nk = cols // BK
    if items0 >= sms:
        return 1
    wire = rows * cols // 2 if tile_words is None else rows * int(tile_words) * 4 // 512
    best_s, best_t = 1, None
    for s in range(1, min(dense_split_max(cols), -(-sms // items0)) + 1):
        t = wire * sms / min(s * items0, sms) + 2.0 * s * m * rows * 4
        if best_t is None or t < best_t:
            best_s, best_t = s, t
    return best_s


def dense_k_split_makespan(m: int, rows: int, cols: int, sms: int, *, tile_words: "int | None" = None,
                           blocks: "int | None" = None) -> int:
    """How many ways to split K for one launch at ``m`` rows: the makespan
    model, under ``TESSERA_DENSE_MODULE_LAUNCH=1`` (:data:`ENV_DENSE_MODULE`).

    An item is 64 rows of ``x`` by 128 rows of a role, so ``items0 =
    ceil(m / 64) * blocks``, with ``blocks = ceil(rows / 128)`` for one role
    (the last block partial on an N-tail).  A launch of several roles
    (:func:`dense_forward_roles`) passes ``rows`` summed over them and
    ``blocks`` summed over their own ``ceil(rows_r / 128)``.
    A split ``S`` makes ``S * items0`` items of ``ceil(K / 32 / S)`` K chunks at
    most (the kernel's ``kc0 = ks * nk / S``), and the persistent grid hands
    them to ``sms`` SMs, so the launch ends when an SM that got
    ``ceil(S * items0 / sms)`` items finishes them.  Each item streams its
    chunks' wire bytes at one SM's share of the read rate and pays
    :data:`DENSE_ITEM_FIXED_BYTES` besides; each split adds an fp32 partial
    written and read back at the full rate:

        t(S) = ceil(S * items0 / sms) * sms * (item * ceil(nk / S) / nk + c)
               + [S > 1] * 2 * S * m * rows * 4

    with ``item = 128 * tile_words * 4 / 512``, the wire bytes of one 128-row
    block over all of K (``tile_words`` the role's words per 512-row tile;
    ``64 * cols``, rate 4, when not given), ``nk = K / 32`` and ``c =``
    :data:`DENSE_ITEM_FIXED_BYTES`.  The integer minimiser over ``1 ..
    min(dense_fixup_split_max(K), sms)`` is returned, the smaller ``S`` on a tie.  The wave count is
    the point: a split that leaves the last wave partly idle costs a whole
    wave (tessera#750: at 32 items, ``S = 2`` is 64 items, two waves of half
    an item, no faster than ``S = 1``; ``S = 3`` is two full waves of a
    third).  The partial term prices the workspace at the read rate even
    where it stays in L2, so it errs toward fewer splits.
    """
    if m <= 0:
        return 1
    items0 = -(-m // BM) * (-(-rows // BN) if blocks is None else int(blocks))
    nk = cols // BK
    words = 64 * cols if tile_words is None else int(tile_words)
    item = BN * words * 4 / 512
    best_s, best_t = 1, None
    for s in range(1, max(1, min(dense_fixup_split_max(cols), sms)) + 1):
        waves = -(-(s * items0) // sms)
        t = waves * sms * (item * -(-nk // s) / nk + DENSE_ITEM_FIXED_BYTES)
        if s > 1:
            t += 2.0 * s * m * rows * 4
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


def dense_forward_roles(roles: "list[FusedDenseWindowRole]", x: torch.Tensor, a_scale: torch.Tensor,
                        out: torch.Tensor, *, fixup: bool = True) -> None:
    """A merged Linear's roles into ``out`` (``[M, sum(rows)]``, unit column
    stride), each role at its column offset in order: one launch per
    :data:`MAX_ROLES` roles (tessera#750 WP2).  The E4M3 libraries only.

    The roles share ``cols``, the run pair, ``tile_words`` and ``slot_words``
    (a module is one rung), so their 128-row blocks form one item list and
    :func:`dense_k_split` prices the launch over all of them.  ``fixup``
    reduces a K split in-kernel: the last split of a tile to arrive sums the
    ``S`` partials in split order and applies the epilogue, bitwise the
    reduce kernel's result.  The fixup stores four bytes at a time, so an
    output view needs only an even row stride; ``fixup=False`` takes the
    reduce kernel (uint2 stores: a row stride of 2 mod 4 runs unsplit) and one
    role per call (the oracle).  The work counter and the per-tile arrival
    counts are one int32 buffer zeroed in-stream here, and the workspace is
    the caching allocator's, so a captured forward replays.
    """
    if not roles:
        return
    lib = _ext(roles[0].library)
    first = roles[0]
    for role in roles:
        if not role.fp8 or role.library != first.library:
            raise ValueError("dense_forward_roles takes roles of one E4M3 library")
        if (role.cols, role.tile_words, role.slot_words) != (first.cols, first.tile_words, first.slot_words):
            raise ValueError("the roles of one launch share cols, tile_words and slot_words (one rung)")
    m = int(x.shape[0])
    if m == 0:
        return
    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    sms = _sm_count(index)
    offset = 0
    for g0 in range(0, len(roles), MAX_ROLES):
        group = roles[g0:g0 + MAX_ROLES]
        rows = sum(r.rows for r in group)
        blocks = sum(-(-r.rows // BN) for r in group)
        s = dense_k_split(m, rows, first.cols, sms, tile_words=first.tile_words, blocks=blocks)
        if fixup:
            # The in-kernel fixup's range (the library refuses past it).  The
            # makespan model never leaves it; the bandwidth model can, for a
            # caller that takes this launch under the default setting.
            s = min(s, dense_fixup_split_max(first.cols))
        if not fixup and len(group) > 1:
            raise ValueError("a split reduced after the launch takes one role per call")
        if s > 1 and not fixup and out.stride(0) % 4 != 0:
            s = 1                     # the reduce kernel stores uint2 at 4-aligned columns
        bm = superblock_rows(first.library, 2, m, dense=True) if s == 1 else BM
        nsb = -(-m // bm)
        counter = torch.zeros(1 + (blocks * nsb if s > 1 and fixup else 0), dtype=torch.int32, device=x.device)
        partial = (torch.empty((s, m, rows), dtype=torch.float32, device=x.device) if s > 1
                   else x.new_empty(0, dtype=torch.float32))
        lib.dense_forward_roles(
            True, x, a_scale,
            [r.words for r in group], [r.table16 for r in group], [r.init for r in group],
            [r.has_init for r in group], [r.wscale for r in group], [r.runs for r in group],
            [r.bdesc for r in group], int(first.tile_words), int(first.slot_words), counter, int(s), partial,
            out.narrow(1, offset, rows), sms, int(bm), bool(fixup))
        offset += rows
