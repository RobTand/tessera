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

The kernel is JIT-built once per family through ``torch.utils.cpp_extension``
into two libraries, ``tessera_routed_fused_value`` and
``tessera_routed_fused_e4m3``, the two ``native_extensions`` entries the
contract publishes for this source.
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
    "ENV_TOGGLE_DENSE",
    "FusedDenseWindowRole",
    "FusedRoutedWindowMoE",
    "MODULE_NAME_E4M3",
    "MODULE_NAME_VALUE",
    "SOURCE",
    "compose_dense_table16",
    "compose_table16",
    "dense_forward",
    "dense_k_split",
    "fused_dense_window_enabled",
    "fused_dense_window_supported",
    "fused_routed_window_supported",
    "fused_routed_window_enabled",
    "prepare_dense_role",
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
#: The two JIT module names, one library per family.  Literals: the contract's
#: native-extension scanner reads the ``load(name=...)`` sites statically.
MODULE_NAME_VALUE = "tessera_routed_fused_value"
MODULE_NAME_E4M3 = "tessera_routed_fused_e4m3"
#: The one source, as ``ext.NATIVE_EXTENSIONS`` publishes it.
SOURCE = "csrc/routed_fused_window.cu"

# The kernel's geometry, restated for the support predicate (the library's
# attributes of the same names are checked against these at load).
BM = 64
HALF = 64
BN = 128
BK = 32
#: The column rates the kernel decodes: every rate of the window grammar, so a
#: stack's one- or two-rate run table (the two rates bracketing its root) is
#: read as the wire lays it out.  ``RATE_MAX`` sizes the per-column word slot.
RATE_MIN = 1
RATE_MAX = 8
RATES = tuple(range(RATE_MIN, RATE_MAX + 1))
SLOT_WORDS_MAX = 2 * RATE_MAX         # the rate-8 word-stage slot, int32 words per (half, column)
BDESC_INTS = 12                       # int32 words per 32-column block descriptor
WINDOW_BITS = 14
TABLE_ENTRIES = 1 << WINDOW_BITS
MIN_COLS = 4 * BK
#: The kernel's shared-memory layout, restated for the support predicates (the
#: library's attributes are checked against these at load): the word stages
#: come last and are sized per launch by the stack's rates, so the fixed part
#: is ``SMEM_FIXED[mode]`` (two 32 KB tables for gate/up, one for down/dense)
#: and a launch needs ``SMEM_FIXED[mode] + WORD_STAGES * 2 * BK * slot_words * 4``
#: bytes.  A block on sm_121 may opt in to 101,376 B, so the gate/up launch
#: fits slots up to 12 words (rates <= 5) and the down/dense launch every rate.
WORD_STAGES = 3
SMEM_FIXED = {0: 91_216, 1: 91_216, 2: 58_448}
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


def slot_words_for_rate(rate: int) -> int:
    """The word-stage slot one column at ``rate`` needs: its ``2 * rate`` words
    plus the one word a lane's re-aligned window reads past them when
    ``8 * rate`` is not a multiple of 32 (those bits enter no field)."""
    rate = int(rate)
    return 2 * rate + (1 if (8 * rate) % 32 else 0)


def slot_words_for_pair(pair: torch.Tensor) -> int:
    """The launch's slot: the larger of the two rates' slots, rounded up to a
    multiple of 4 (16-byte copies stay aligned), at least 4."""
    r_lo, _c0, _n_lo, _w0, r_hi, _c1, n_hi, _w1 = (int(v) for v in pair.reshape(8).tolist())
    need = max(slot_words_for_rate(r_lo), slot_words_for_rate(r_hi) if n_hi > 0 else 0, 4)
    return _round_up_4(need)


def smem_bytes(mode: int, slot_words: int) -> int:
    """Dynamic shared memory the launch of ``mode`` needs at ``slot_words``-word slots."""
    return SMEM_FIXED[int(mode)] + WORD_STAGES * 2 * BK * int(slot_words) * 4


#: The rates a ROUTED-EXPERT stack (the two-table gate/up launch, MODE 0/1)
#: reaches on the target platform: those whose one-rate slot fits sm_121's
#: opt-in shared memory.  A two-rate pair's slot is the larger rate's, so the
#: set is closed under bracketing.  Published as the fused lanes'
#: ``column_rates_routed_moe`` (``serving.ext.ROUTED_FUSED_LANE_REQUIRES``,
#: contract v45, tessera#694); the down/dense one-table launch reads every
#: rate in ``RATES``.  Derived, not typed: the day the layout changes, this
#: changes with it and the contract's pin fails until the JSON follows.
ROUTED_LANE_RATES = tuple(
    r for r in RATES
    if smem_bytes(0, _round_up_4(slot_words_for_rate(r))) <= SM121_MAX_DYNAMIC_SMEM)


def smem_reason(mode: int, slot_words: int, device: torch.device, family: str) -> "str | None":
    """Why the launch does not fit the device's opt-in shared-memory limit, or ``None``.

    The limit is read from the built library (``cudaDevAttrMaxSharedMemoryPer
    BlockOptin``; torch publishes no such property).  A library that cannot be
    built answers ``None`` here: the build failure is the caller's, reported
    where the adapter is constructed, not a lane refusal.
    """
    try:
        lib = _ext(family)
    except Exception:  # noqa: BLE001 -- the build's failure is reported by from_bundles
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    need, have = smem_bytes(mode, slot_words), int(lib.max_dynamic_smem_bytes(index))
    if need <= have:
        return None
    what = "gate/up" if mode != 2 else "down/dense"
    return (f"the {what} launch at {slot_words}-word slots needs {need} bytes of shared memory "
            f"per block and this device allows {have}")


def fused_routed_window_enabled() -> bool:
    return os.environ.get(ENV_TOGGLE, "1") != "0"


def fused_dense_window_enabled() -> bool:
    return os.environ.get(ENV_TOGGLE_DENSE, "1") != "0"


def _cflags(token: str, fp8: bool) -> list:
    from .serving.backend import offload_flags

    return ["-O3", "-lineinfo", "-std=c++17",
            f"-DTESSERA_ROUTED_FUSED_FP8={1 if fp8 else 0}",
            *offload_flags(token)]


def _probed_or_none(probe):
    try:
        return probe(torch=torch)
    except Exception:  # noqa: BLE001 -- no device answering is the answer
        return None


def _built_library(build: str, module: str) -> "str | None":
    found = sorted(_glob.glob(os.path.join(build, f"{module}*.so")))
    return found[0] if found else None


@functools.lru_cache(maxsize=None)
def _ext(family: str):
    """The family's library, built on first use (the window GEMV's loader shape)."""
    from torch.utils.cpp_extension import load

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

    if family not in ("value", "e4m3"):
        raise GrammarError(f"the fused routed lane serves the value and e4m3 families, got {family!r}")
    fp8 = family == "e4m3"
    ensure_toolchain_on_path(torch)
    module = MODULE_NAME_E4M3 if fp8 else MODULE_NAME_VALUE
    # The path the contract publishes IS the file compiled here (#134).
    src = serving_ext.native_source_path(module)
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
            if fp8:
                lib = load(
                    name="tessera_routed_fused_e4m3",   # literal: the contract scanner reads it
                    sources=[src], build_directory=build,
                    extra_cuda_cflags=_cflags(token, True), verbose=verbose)
            else:
                lib = load(
                    name="tessera_routed_fused_value",  # literal: the contract scanner reads it
                    sources=[src], build_directory=build,
                    extra_cuda_cflags=_cflags(token, False), verbose=verbose)
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
    for name, want in (("BM", BM), ("BN", BN), ("HALF", HALF), ("BK", BK),
                       ("RATE_MIN", RATE_MIN), ("RATE_MAX", RATE_MAX), ("SLOT_WORDS_MAX", SLOT_WORDS_MAX),
                       ("BDESC_INTS", BDESC_INTS), ("WINDOW_BITS", WINDOW_BITS), ("FAMILY_FP8", fp8),
                       ("WORD_STAGES", WORD_STAGES), ("SMEM_FIXED_GATE_UP", SMEM_FIXED[0]),
                       ("SMEM_FIXED_DOWN", SMEM_FIXED[2])):
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


def run_pair(runs: torch.Tensor, cols: int) -> "tuple[torch.Tensor | None, str | None]":
    """One unit's run table as the kernel's run pair, or why it is refused.

    ``runs`` is the wire's ``[R, 4]`` table of ``(rate, col0, ncols, word0)``
    rows, the packer's stable sort of the columns by (rate, column) into one
    contiguous run per rate.  The kernel reads one or two runs -- the grammar
    mixes only the two rates bracketing a stack's root -- as the int32
    ``[8]`` pair ``(r_lo, 0, n_lo, 0, r_hi, n_lo, n_hi, w_hi)`` with
    ``w_hi = 16 * n_lo * r_lo`` (a one-run table has ``n_hi = 0``).  Returns
    ``(pair, None)`` or ``(None, reason)``.
    """
    runs = runs.reshape(-1, 4)
    n_runs = int(runs.shape[0])
    if n_runs not in (1, 2):
        return None, f"run table has {n_runs} runs; the lane reads one or two (the two rates bracketing the root)"
    rows = [tuple(int(v) for v in row) for row in runs.tolist()]
    r_lo, c_lo, n_lo, w_lo = rows[0]
    if r_lo not in RATES or c_lo != 0 or w_lo != 0 or n_lo <= 0:
        return None, f"first run {rows[0]} is not (rate in {RATE_MIN}..{RATE_MAX}, 0, n, 0)"
    if n_runs == 1:
        if n_lo != cols:
            return None, f"the one run covers {n_lo} of {cols} columns"
        pair = (r_lo, 0, n_lo, 0, 0, n_lo, 0, 16 * n_lo * r_lo)
    else:
        r_hi, c_hi, n_hi, w_hi = rows[1]
        if r_hi not in RATES or r_hi <= r_lo:
            return None, f"second run rate {r_hi} is not above the first's {r_lo} within {RATE_MIN}..{RATE_MAX}"
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
        if int(b.experts) != e:
            return f"{name} has {b.experts} experts, down has {e}"
        if fam == "e4m3" and b.quantizer != "native":
            return f"{name} was prepared without the native activation quantizer"
        if b.cols % BK != 0 or b.cols < MIN_COLS:
            return f"{name} has {b.cols} columns; the lane needs a multiple of {BK} and at least {MIN_COLS}"
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
    for mode, bs in ((0, (gate, up)), (2, (down,))):
        slot = max(slot_words_for_pair(run_pair(b.runs_all.reshape(e, -1, 4)[0], b.cols)[0]) for b in bs)
        why = smem_reason(mode, slot, down.device, fam)
        if why is not None:
            return why
    return None


@dataclasses.dataclass(frozen=True)
class _Routing:
    offsets: torch.Tensor      # [E + 1] int32
    flat_sorted: torch.Tensor  # [P] int32
    rw_sorted: torch.Tensor    # [P] fp32
    item_off: torch.Tensor     # [E + 1] int32
    tokens: int
    top_k: int

    @property
    def routes(self) -> int:
        return self.tokens * self.top_k


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@dataclasses.dataclass(frozen=True)
class FusedRoutedWindowMoE:
    """The fused lane over a loader-filled grouped stack.

    ``gate``/``up``/``down`` are the compact loader's
    :class:`~tessera.window_gemm_grouped.PreparedGroupedWindowGemm` bundles
    (the same objects the compact adapter would serve); the three ``table``
    tensors are :func:`compose_table16` of each and the ``words`` tensors are
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
        _ext(down.family)
        e = int(down.experts)

        def tables(b):
            pair, why = run_pair(b.runs_all.reshape(e, -1, 4)[0], b.cols)
            assert pair is not None, why          # the predicate above admitted it
            return (pair.reshape(1, 8).expand(e, 8).contiguous(),
                    block_desc(b.perm_all, int(pair[2]), int(b.cols)),
                    pair_tile_words(pair), slot_words_for_pair(pair))

        runs_gate, bdesc_gate, tw_gate, sw_gate = tables(gate)
        runs_up, bdesc_up, _tw_up, sw_up = tables(up)
        runs_down, bdesc_down, tw_down, sw_down = tables(down)
        return cls(gate=gate, up=up, down=down, family=down.family, arithmetic=down.arithmetic,
                   table_gate=compose_table16(gate), table_up=compose_table16(up),
                   table_down=compose_table16(down),
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
                                        DECODER_NATIVE_ROUTED_FUSED_WINDOW_FOLDED)

        decoder = (DECODER_NATIVE_ROUTED_FUSED_WINDOW_FOLDED if self.family == "value"
                   else DECODER_NATIVE_ROUTED_FUSED_WINDOW)
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
        if expert_ids.dim() != 2 or routing_weights.shape != expert_ids.shape:
            raise GrammarError("expert_ids and routing_weights must share [T, top_k]")
        if expert_ids.device != self.device or routing_weights.device != self.device:
            raise GrammarError("routing tensors must live on the compute device")
        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise GrammarError("expert_ids must be int32 or int64")
        tokens, top_k = (int(v) for v in expert_ids.shape)
        ids = expert_ids.reshape(-1).to(torch.int64)
        e = self.experts
        counts = torch.zeros(e, dtype=torch.int32, device=self.device)
        counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
        offsets = torch.zeros(e + 1, dtype=torch.int32, device=self.device)
        offsets[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
        order = torch.argsort(ids, stable=True)
        flat_sorted = order.to(torch.int32).contiguous()
        rw_sorted = routing_weights.reshape(-1).to(torch.float32)[order].contiguous()
        superblocks = (counts + (BM - 1)) // BM
        item_off = torch.zeros(e + 1, dtype=torch.int32, device=self.device)
        item_off[1:] = torch.cumsum(superblocks, 0, dtype=torch.int32)
        return _Routing(offsets=offsets, flat_sorted=flat_sorted, rw_sorted=rw_sorted,
                        item_off=item_off, tokens=tokens, top_k=top_k)

    def _launch(self, mode: int, x: torch.Tensor, a_scale: "torch.Tensor | None",
                routing: _Routing, *, a_row_mode: int, mul_weight: bool, limit: float,
                out: torch.Tensor, counter: int) -> None:
        lib = _ext(self.family)
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
            routing.offsets, routing.flat_sorted, routing.rw_sorted, routing.item_off,
            slot,
            int(routing.top_k), int(a_row_mode), bool(mul_weight), float(limit),
            out, _sm_count(index))

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
        _ext(self.family).token_sum(routed, out, int(routing.top_k))
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
        _ext(self.family).token_sum(routed, out, int(routing.top_k))
        return out


# ---------------------------------------------------------------------------
# the dense identity: one role of a dense Linear as the E = 1 case
# ---------------------------------------------------------------------------

def fused_dense_window_supported(bundle) -> "str | None":
    """Why the dense identity refuses a prepared role, or ``None`` when it serves it.

    ``bundle`` is a ``window_gemm.PreparedWindowGemm`` (the frozen role the
    Triton dense GEMM runs).  The kernel reads the routed lane's wire shape --
    one or two column-rate runs (rates 1..8, the two bracketing the root),
    window bits 14, the packer's column order, the family's published
    arithmetic (folded for value, epilogue for e4m3) -- plus the dense tile:
    rows a multiple of 128 (one 128-column B block per item) and columns a
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
    if rows % BN != 0:
        return f"{rows} rows; the dense identity needs a multiple of {BN}"
    if bundle.words.dtype != torch.int32 or bundle.words.dim() != 1:
        return "words must be a flat int32 stream"
    pair, why = run_pair(bundle.runs, cols)
    if pair is None:
        return why
    why = perm_reason(bundle.perm, int(pair[2]), cols)
    if why is not None:
        return f"{why}; the kernel reads the packer's column order"
    if int(bundle.tile_words) != pair_tile_words(pair):
        return f"tile_words {bundle.tile_words} is not {pair_tile_words(pair)} (from the run table)"
    why = smem_reason(2, slot_words_for_pair(pair), bundle.device, fam)
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


@dataclasses.dataclass(frozen=True)
class FusedDenseWindowRole:
    """One role's kernel inputs, frozen at preparation.

    ``words``/``init``/``wscale`` are views of the bundle's own tensors (no new
    storage); ``table16`` is the composed 16-bit table (32 KB, new storage,
    counted by the module's residency accounting); ``has_init`` is the one
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
    _ext(bundle.family)
    device = bundle.device
    cols = int(bundle.cols)
    pair, why = run_pair(bundle.runs, cols)
    assert pair is not None, why              # the predicate above admitted it
    return FusedDenseWindowRole(
        family=bundle.family, rows=int(bundle.rows), cols=cols,
        words=bundle.words.reshape(1, -1), table16=compose_dense_table16(bundle),
        init=bundle.init_perm.reshape(1, -1),
        has_init=torch.tensor([1 if bundle.has_init else 0], dtype=torch.int32, device=device),
        wscale=bundle.scale.reshape(1, -1),
        runs=pair.reshape(1, 8), bdesc=block_desc(bundle.perm, int(pair[2]), cols),
        tile_words=pair_tile_words(pair), slot_words=slot_words_for_pair(pair))


def dense_k_split(m: int, rows: int, cols: int, sms: int, *, tile_words: "int | None" = None) -> int:
    """How many ways to split K for one role at ``m`` rows: the bandwidth model.

    An item is 64 rows of ``x`` by 128 rows of the role, so ``items0 =
    ceil(m / 64) * rows / 128``.  When ``items0 >= sms`` every SM has work and
    the answer is 1 (prefill is untouched).  Below that, each split adds items
    and costs an fp32 partial written and read back; the time model is the
    wire bytes served by ``min(S * items0, sms)`` SMs at the per-SM share of
    the bandwidth, plus the partial traffic at full bandwidth:

        t(S) = wire * sms / min(S * items0, sms) + 2 * S * m * rows * 4

    with ``wire = rows * tile_words * 4 / 512`` -- the role's wire bytes, from
    its words per 512-row tile (``rows * cols / 2`` at rate 4, the default
    when ``tile_words`` is not given).  The minimiser over the integers
    ``1 .. min(K / 32, ceil(sms / items0))`` is returned; the constants are the
    SM count and the byte counts, nothing else.
    """
    items0 = -(-m // BM) * (rows // BN)
    nk = cols // BK
    if items0 >= sms or m <= 0:
        return 1
    wire = rows * cols // 2 if tile_words is None else rows * int(tile_words) * 4 // 512
    best_s, best_t = 1, None
    for s in range(1, min(nk, -(-sms // items0)) + 1):
        t = wire * sms / min(s * items0, sms) + 2.0 * s * m * rows * 4
        if best_t is None or t < best_t:
            best_s, best_t = s, t
    return best_s


def dense_forward(role: FusedDenseWindowRole, x: torch.Tensor, a_scale: "torch.Tensor | None",
                  out: torch.Tensor, counter: torch.Tensor) -> None:
    """One role's launch into ``out`` (a ``[M, rows]`` view, unit column stride).

    ``x`` is the family's A operand as the route quantised it (e4m3 + fp32
    ``a_scale`` for E4M3, bf16 for value), contiguous ``[M, cols]``; ``counter``
    is one int32 slot this call zeroes in-stream.  No host synchronisation, so
    a captured forward replays.
    """
    lib = _ext(role.family)
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
    if s > 1:
        partial = torch.empty((s, m, role.rows), dtype=torch.float32, device=x.device)
    else:
        partial = x.new_empty(0, dtype=torch.float32)
    slot = counter[:1]
    slot.zero_()
    empty = x.new_empty(0, dtype=torch.float32)
    lib.dense_forward(
        bool(role.fp8), x, a_scale if a_scale is not None else empty,
        role.words, role.table16, role.init, role.has_init, role.wscale,
        role.runs, role.bdesc, int(role.tile_words), int(role.slot_words), slot, int(s), partial, out, sms)
