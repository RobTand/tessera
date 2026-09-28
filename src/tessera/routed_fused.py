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
``native_routed_fused_window_folded`` (BF16, folded), both in
``scheme.EXPERIMENTAL_LAUNCHES`` until a served census earns them cells.  The
compact adapter's pairs stay attested and stay the dispatch for every stack
this lane refuses (:func:`fused_routed_window_supported`) and for
``TESSERA_ROUTED_FUSED=0``.

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
    "FusedRoutedWindowMoE",
    "MODULE_NAME_E4M3",
    "MODULE_NAME_VALUE",
    "SOURCE",
    "compose_table16",
    "fused_routed_window_supported",
    "fused_routed_window_enabled",
    "words_by_expert",
]

log = logging.getLogger(__name__)

#: ``TESSERA_ROUTED_FUSED=0`` keeps the compact Triton adapter for every stack;
#: unset or ``1`` takes this lane wherever :func:`fused_routed_window_supported`
#: admits the stack.
ENV_TOGGLE = "TESSERA_ROUTED_FUSED"
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
RATE = 4
WINDOW_BITS = 14
CHUNK_WORDS = 64          # int32 words per column per 512-row tile at rate 4
MIN_COLS = 4 * BK


def fused_routed_window_enabled() -> bool:
    return os.environ.get(ENV_TOGGLE, "1") != "0"


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
    for name, want in (("BM", BM), ("BN", BN), ("HALF", HALF), ("BK", BK), ("RATE", RATE),
                       ("WINDOW_BITS", WINDOW_BITS), ("FAMILY_FP8", fp8)):
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


def fused_routed_window_supported(gate, up, down) -> "str | None":
    """Why this lane refuses a stack, or ``None`` when it serves it.

    The kernel reads one wire shape: window bits 14, every column at rate 4
    in one run starting at word 0 (the ``[[4, 0, cols, 0]]`` run table every
    q256=1024 GLM stack carries), identity column permutation, and the
    family's published arithmetic (folded for value, epilogue for e4m3).  A
    stack outside that shape -- the mixed-rate q256=896 E4M3 rung, a permuted
    column order -- keeps the compact adapter, and the reason is the string
    returned here so a log can say which.
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
        # One run per expert: run_off == 0, 1, ..., E and the table is [E, 4].
        runs = b.runs_all.reshape(-1, 4)
        want_off = torch.arange(e + 1, dtype=b.run_off.dtype, device=b.run_off.device)
        if int(runs.shape[0]) != e or not bool((b.run_off == want_off).all()):
            return f"{name} carries {int(runs.shape[0])} runs for {e} experts; the lane reads one rate-{RATE} run per expert"
        want_run = torch.tensor([RATE, 0, b.cols, 0], dtype=runs.dtype, device=runs.device)
        if not bool((runs == want_run).all()):
            return f"{name} run table is not [[{RATE}, 0, {b.cols}, 0]] for every expert (mixed rates or a rate other than {RATE})"
        arange = torch.arange(b.cols, dtype=b.perm_all.dtype, device=b.perm_all.device)
        if tuple(b.perm_all.shape) != (e, b.cols) or not bool((b.perm_all == arange).all()):
            return f"{name} permutes its columns; the lane reads the identity order"
        want_tile = b.cols * CHUNK_WORDS
        if not bool((b.tile_words == want_tile).all()):
            return f"{name} tile_words is not {want_tile} for every expert"
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
        return cls(gate=gate, up=up, down=down, family=down.family, arithmetic=down.arithmetic,
                   table_gate=compose_table16(gate), table_up=compose_table16(up),
                   table_down=compose_table16(down),
                   words_gate=words_by_expert(gate), words_up=words_by_expert(up),
                   words_down=words_by_expert(down),
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
        else:
            b0, b1 = self.gate, self.up
            w0, w1 = self.words_gate, self.words_up
            t0, t1 = self.table_gate, self.table_up
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
            int(b0.cols) * CHUNK_WORDS,
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
