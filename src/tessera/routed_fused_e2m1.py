"""The fused window kernel's E2M1 family (tessera#690, tessera#750).

A fourth library of ``serving/csrc/routed_fused_window.cu``,
``tessera_routed_fused_e2m1`` (``-DTESSERA_ROUTED_FUSED_FP4=1``, built for the
architecture-specific target ``sm_121a``), serves the Tessera-4 wire -- the
E2M1x2 window body over the LUT16 scale plane -- on the block-scaled FP4
tensor-core instruction
``mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.e2m1.e2m1.f32.ue4m3``.

What it computes.  The weight is ``e2m1(code) * ue4m3(lut[nibble]) *
global``: the producers decode each 64-column chunk's tuple codes from the
window stream into a packed E2M1 B tile and its LUT16 nibbles into the
instruction's UE4M3 group scales.  The activation is the NVFP4 route's
``scaled_fp4_quant`` at the layer's static global ``gs`` (E2M1 codes and
UE4M3 group-16 scales; ``kernel_a4.a4_quantize_activation``), staged
unconverted.  Accumulation is fp32, and the epilogue is one fp32 multiply by
the per-expert ratio ``global / gs`` before the bf16 boundary -- the
activation contract ``e2m1_group16_ue4m3_static`` the NVFP4 routes already
execute, unchanged.  The launches are the other families': gate/up with the
SwiGLU epilogue (mode 0) or route-preserved (mode 1), the down projection
into route-sorted rows reduced per token in fixed order (mode 2), and the
dense identity (E = 1) with an optional K split.

Geometry.  A chunk is one instruction's K (64 columns); an item is 256
output rows (gate/up: 128 gate rows and the same 128 up rows); a superblock
is 64 routes.  Columns must be a multiple of 64 and at least 256, the
intermediate size a multiple of 128 and the hidden size of 256; a dense
projection's rows a multiple of 32 (the last block is decoded whole from the
wire's padded tile and written only below its rows).  Every
column rate 1..8 and every adjacent two-run table is instantiated at three
word stages; the largest launch (gate/up, rate 8) needs 93,648 B of shared
memory.

Scope: NOT a serving lane yet.  ``ROUTES["TESSERA_NVFP4"]`` admits the TCQ
span-2 body only, and ``nvfp4_moe_route``/``nvfp4_route`` intake span-2 units,
so no serving module can hold a window-body E2M1 stack.  This module is
therefore kept out of the import graph of ``tessera.serving``: the contract's
``native_extensions`` table lists the libraries a serving process can map
(``tests/test_serving_native_extensions.py`` holds reachability to that
table), and publishing this one before a route loads it would claim a load
path that does not exist.  The entry, the route's launch rows and the census
cells come with the route change that admits the window body.
"""

from __future__ import annotations

import dataclasses
import os

import torch

from .errors import GrammarError
from .routed_fused import (
    MODULE_NAME_VALUE,
    RATE_MAX,
    RATE_MIN,
    SLOT_WORDS_MAX,
    TABLE_ENTRIES,
    WINDOW_BITS,
    BM,
    _Routing,
    _cflags,
    _routing_tables,
    _run_stack_reason,
    _sm_count,
    build_library,
    pair_tile_words,
    run_pair,
    slot_words_for_pair,
    words_by_expert,
)

__all__ = [
    "BK",
    "BN",
    "DENSE_ROWS",
    "HALF_ROWS",
    "MIN_COLS",
    "MODULE_NAME",
    "FusedRoutedE2M1MoE",
    "chunk_desc",
    "DenseE2M1Role",
    "dense_forward",
    "dense_role_reason",
    "dense_split_max",
    "fused_routed_e2m1_supported",
    "prepare_dense_role",
    "projection_tables",
    "smem_bytes",
]

MODULE_NAME = "tessera_routed_fused_e2m1"
# Standalone E2M1 window experiment; not the serving WINDOW class selector.
ENV_TOGGLE_E2M1 = "TESSERA_ROUTED_FUSED"
#: The family's geometry (``fp4`` in the kernel source), checked against the
#: built library's attributes by :func:`_ext`.
BN = 256
HALF_ROWS = 128
BK = 64
MIN_COLS = 4 * BK
DESC_WORDS = 4
WORD_STAGES = 3
#: A dense projection's rows are a multiple of this: one 16-byte part of the
#: LUT16 plane's row run (two rows per byte).
DENSE_ROWS = 32
#: Fixed dynamic shared memory per launch mode: the code tables (16 KB each),
#: two 8 KB B stages and their scales, four A slots, the descriptor and
#: scale-plane rings.
SMEM_FIXED = {0: 62_160, 1: 62_160, 2: 45_712}


def smem_bytes(mode: int, slot_words: int) -> int:
    """Dynamic shared memory the launch of ``mode`` needs at
    ``slot_words``-word slots (``fp4::smem_bytes``): three word stages of
    ``2 * 4`` 16-column groups, ``16 * (slot_words + 4) + 8`` words each."""
    group_ints = 16 * (int(slot_words) + 4) + 8
    return SMEM_FIXED[int(mode)] + WORD_STAGES * 2 * (BK // 16) * group_ints * 4


def dense_split_max(cols: int) -> int:
    """The largest K split the dense launch takes at ``cols`` columns.

    The producers write item ``i + 2``'s descriptor into item ``i``'s slot
    once item ``i + 1``'s last chunk has waited for the chunk two before it
    to be consumed; the consumers read item ``i``'s slot before they release
    its first chunk.  Items ``i`` and ``i + 1`` of three chunks or more
    between them order the two, so every split item keeps two chunks:
    ``floor(nk / S) >= 2``.  The library refuses a larger split by name.
    """
    return int(cols) // BK // 2


PREFETCH_ENV = "TESSERA_ROUTED_FUSED_FP4_A_PREFETCH"


def activation_prefetch() -> int:
    """Default-off #875 experiment; invalid arms fail before compilation."""
    value = os.environ.get(PREFETCH_ENV, "0")
    if value not in ("0", "4"):
        raise GrammarError(f"{PREFETCH_ENV} must be 0 (baseline) or 4 (experimental)")
    return int(value)


def _check_library(lib, prefetch: int) -> None:
    """The producer owner's single ABI rule, including retained diagnostic banks."""
    for name, want in (("BM", BM), ("BN", BN), ("HALF_ROWS", HALF_ROWS), ("BK", BK),
                       ("RATE_MIN", RATE_MIN), ("RATE_MAX", RATE_MAX), ("SLOT_WORDS_MAX", SLOT_WORDS_MAX),
                       ("WORD_STAGES", WORD_STAGES), ("DESC_WORDS", DESC_WORDS),
                       ("WINDOW_BITS", WINDOW_BITS), ("SMEM_FIXED_GATE_UP", SMEM_FIXED[0]),
                       ("SMEM_FIXED_DOWN", SMEM_FIXED[2]), ("FAMILY_FP8", False),
                       ("FAMILY_MMA8", False), ("FAMILY_FP4", True), ("A_PREFETCH", prefetch)):
        if getattr(lib, name) != want:
            raise GrammarError(
                f"{MODULE_NAME} was built with {name}={getattr(lib, name)!r}; this module expects {want!r}")


def _ext():
    """The library, built on first use from the source the contract publishes
    for the fused window libraries."""
    global _LIB, _LIB_PREFETCH
    prefetch = activation_prefetch()
    if _LIB is not None:
        if prefetch != _LIB_PREFETCH:
            raise GrammarError(f"{PREFETCH_ENV} changed after the E2M1 library was loaded")
        return _LIB
    from torch.utils.cpp_extension import load

    def compile_fn(src, build, token, verbose):
        flags = _cflags(token, False, False, True)
        if prefetch:
            flags.append(f"-DTESSERA_ROUTED_FUSED_FP4_A_PREFETCH={prefetch}")
            return load(
                name="tessera_routed_fused_e2m1_apf4",
                sources=[src], build_directory=build,
                extra_cuda_cflags=flags, verbose=verbose)
        return load(
            name="tessera_routed_fused_e2m1",   # literal: a scanner reads it
            sources=[src], build_directory=build,
            extra_cuda_cflags=flags, verbose=verbose)

    module = MODULE_NAME + ("_apf4" if prefetch else "")
    lib = build_library(module, MODULE_NAME_VALUE, compile_fn)
    _check_library(lib, prefetch)
    _LIB = lib
    _LIB_PREFETCH = prefetch
    return lib


_LIB = None
_LIB_PREFETCH = None


def smem_reason(mode: int, slot_words: int, device: torch.device) -> "str | None":
    """Why the device cannot hold the launch of ``mode`` at ``slot_words``,
    or ``None``."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    need = smem_bytes(mode, slot_words)
    have = int(_ext().max_dynamic_smem_bytes(index))
    if need <= have:
        return None
    return (f"the E2M1 launch of mode {mode} needs {need} B of shared memory at {slot_words}-word "
            f"slots; this device gives a block {have} B")


def chunk_desc(perm: torch.Tensor, n_lo: int, cols: int) -> torch.Tensor:
    """The E2M1 family's per-64-column chunk descriptors.

    int32 ``[E, cols / 64, 4]``: words 0 and 1 are the chunk's high-rate mask
    (bit ``c`` of word ``c >> 5`` set when original column ``64 kc + c`` is in
    the high-rate run), word 2 the number of low-rate columns in the chunks
    before, word 3 zero.  The kernel keeps the chunk in ORIGINAL column order
    (a producer decodes one scale group's 16 consecutive columns) and recovers
    each column's rank within its run by counting mask bits (``col_map4``).
    A one-run unit's map is computed, not read, but its descriptors are built
    all the same so every launch takes one tensor shape.
    """
    p = perm.reshape(-1, cols).to(torch.int64)
    e = int(p.shape[0])
    nk = cols // BK
    device = p.device
    is_hi_pos = (torch.arange(cols, device=device) >= n_lo).to(torch.int64)
    is_hi_col = torch.zeros(e, cols, dtype=torch.int64, device=device)
    is_hi_col.scatter_(1, p, is_hi_pos.expand(e, cols))
    chunk = is_hi_col.reshape(e, nk, 2, 32)
    masks = (chunk << torch.arange(32, dtype=torch.int64, device=device)).sum(dim=3)    # [E, nk, 2]
    cnt_lo = BK - chunk.sum(dim=(2, 3))
    desc = torch.zeros(e, nk, DESC_WORDS, dtype=torch.int64, device=device)
    desc[..., :2] = masks
    desc[..., 2] = torch.cumsum(cnt_lo, dim=1) - cnt_lo
    # the masks are 32-bit patterns: reinterpret the high bit as int32's sign
    desc = torch.where(desc >= 1 << 31, desc - (1 << 32), desc)
    return desc.to(torch.int32).contiguous()


def fused_routed_e2m1_supported(gate, up, down) -> "str | None":
    """Why the E2M1 library refuses a stack, or ``None`` when it serves it.

    The wire checks of :func:`fused_routed_window_supported` (window bits 14,
    one run table per stack that is the kernel's run pair, the packer's column
    order, one tile stride for gate and up), on the E2M1 family's bundles:
    each carries its code table, LUT16 scale plane, UE4M3 table and global
    (``window_gemm_grouped``'s e2m1 fields).  The geometry is the family's:
    columns a multiple of 64 (one FP4 instruction's K) and at least 256, the
    intermediate size a multiple of 128 rows and the hidden size of 256.
    """
    from .kernel_window_gemv import require_legacy_word_layout
    try:
        for b in (gate, up, down):
            require_legacy_word_layout(getattr(b, "word_layout", "legacy"),
                                       "the E2M1 routed window reader")
    except GrammarError as exc:
        return str(exc)
    if os.environ.get(ENV_TOGGLE_E2M1, "1") == "0":
        return f"disabled by {ENV_TOGGLE_E2M1}=0"
    bundles = {"gate": gate, "up": up, "down": down}
    e = int(down.experts)
    for name, b in bundles.items():
        if b.family != "e2m1":
            return f"{name} family {b.family!r}; this library serves the e2m1 family"
        if b.device.type != "cuda":
            return f"{name} lives on {b.device}; the lane is CUDA"
        if b.window_bits != WINDOW_BITS:
            return f"{name} window_bits {b.window_bits} != {WINDOW_BITS}"
        if int(b.experts) != e:
            return f"{name} has {b.experts} experts, down has {e}"
        if b.cols % BK != 0 or b.cols < MIN_COLS:
            return (f"{name} has {b.cols} columns; the E2M1 lane needs a multiple of {BK} "
                    f"and at least {MIN_COLS}")
        why = _run_stack_reason(name, b, e)
        if why is not None:
            return why
        for field, shape, dtype in (("codes_all", (e, TABLE_ENTRIES), torch.uint8),
                                    ("scale_plane_all", (e, b.rows * (b.cols // 16) // 2), torch.uint8),
                                    ("scale_lut_all", (e, 16), torch.uint8),
                                    ("global_all", (e,), torch.float32)):
            t = getattr(b, field)
            if t is None or tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
                return f"{name} {field} must be a contiguous {dtype} {list(shape)}"
    if gate.rows != up.rows or gate.rows != down.cols:
        return f"gate rows {gate.rows}, up rows {up.rows} and down cols {down.cols} disagree"
    if gate.cols != up.cols:
        return f"gate cols {gate.cols} != up cols {up.cols}"
    if gate.rows % HALF_ROWS != 0:
        return f"the intermediate size {gate.rows} is not a multiple of {HALF_ROWS}"
    if down.rows % BN != 0:
        return f"the hidden size {down.rows} is not a multiple of {BN}"
    if int(gate.tile_words[0]) != int(up.tile_words[0]):
        return (f"gate tile_words {int(gate.tile_words[0])} != up tile_words {int(up.tile_words[0])}; "
                "the gate/up launch reads one tile stride for both")
    for mode, bs in ((0, (gate, up)), (2, (down,))):
        slot = max(slot_words_for_pair(run_pair(b.runs_all.reshape(e, -1, 4)[0], b.cols)[0]) for b in bs)
        why = smem_reason(mode, slot, down.device)
        if why is not None:
            return why
    return None


def projection_tables(bundle) -> "tuple[torch.Tensor, torch.Tensor, int, int]":
    """``(runs, desc, tile_words, slot_words)`` of one E2M1 projection: the
    stack's run pair broadcast to int32 ``[E, 8]``, the chunk descriptors
    (:func:`chunk_desc`), the wire's words per 512-tuple tile and the
    word-stage slot the pair needs (the caller has admitted the stack)."""
    e = int(bundle.experts)
    pair, why = run_pair(bundle.runs_all.reshape(e, -1, 4)[0], bundle.cols)
    if pair is None:
        raise GrammarError(f"the fused E2M1 lane refuses this projection: {why}")
    return (pair.reshape(1, 8).expand(e, 8).contiguous(),
            chunk_desc(bundle.perm_all, int(pair[2]), int(bundle.cols)),
            pair_tile_words(pair), slot_words_for_pair(pair))


def _static_global(value, device: torch.device, what: str) -> torch.Tensor:
    g = torch.as_tensor(value, dtype=torch.float32).to(device).reshape(-1)
    if g.numel() != 1:
        raise GrammarError(f"{what} is one static scalar per GEMM input; got {g.numel()} values")
    if not bool(torch.isfinite(g).all()) or not bool((g > 0).all()):
        raise GrammarError(f"{what} must be a finite positive scalar")
    return g.reshape(())


@dataclasses.dataclass(frozen=True)
class FusedRoutedE2M1MoE:
    """The fused routed lane of the E2M1 family (``tessera_routed_fused_e2m1``).

    ``gate``/``up``/``down`` are the loader's e2m1 bundles; ``gs13`` and
    ``gs2`` the layer's static A-side globals (``scaled_fp4_quant``'s
    capacity-over-amax, one per GEMM input, as the NVFP4 routed route reduces
    them); ``ratio_*`` the per-expert epilogues ``weight global / A-side
    global`` in fp32, frozen at construction so a forward reads no host
    scalar.  The activation contract is ``e2m1_group16_ue4m3_static``: the
    kernel multiplies E2M1 activations with their UE4M3 group scales by the
    E2M1 weights with theirs on the block-scaled FP4 instruction, fp32
    accumulation, and one fp32 multiply by the ratio before the bf16 boundary.
    """
    PROFILER_LABEL = "tessera_routed_fused_window"

    gate: object
    up: object
    down: object
    words_gate: torch.Tensor
    words_up: torch.Tensor
    words_down: torch.Tensor
    runs_gate: torch.Tensor
    runs_up: torch.Tensor
    runs_down: torch.Tensor
    desc_gate: torch.Tensor
    desc_up: torch.Tensor
    desc_down: torch.Tensor
    tile_words_gate_up: int
    tile_words_down: int
    slot_words_gate_up: int
    slot_words_down: int
    gs13: torch.Tensor
    gs2: torch.Tensor
    ratio_gate: torch.Tensor
    ratio_up: torch.Tensor
    ratio_down: torch.Tensor
    counters: torch.Tensor
    activation: str = "silu"
    library: str = "e2m1"
    family: str = "e2m1"

    @classmethod
    def from_bundles(cls, gate, up, down, *, gs13, gs2, activation: str = "silu") -> "FusedRoutedE2M1MoE":
        reason = fused_routed_e2m1_supported(gate, up, down)
        if reason is not None:
            raise GrammarError(f"the fused E2M1 lane refuses this stack: {reason}")
        if activation != "silu":
            raise GrammarError(f"activation {activation!r} is not served; the fused lane computes silu")
        _ext()
        device = down.device
        g13 = _static_global(gs13, device, "gs13")
        g2 = _static_global(gs2, device, "gs2")
        runs_gate, desc_gate, tw_gate, sw_gate = projection_tables(gate)
        runs_up, desc_up, _tw_up, sw_up = projection_tables(up)
        runs_down, desc_down, tw_down, sw_down = projection_tables(down)
        return cls(gate=gate, up=up, down=down,
                   words_gate=words_by_expert(gate), words_up=words_by_expert(up),
                   words_down=words_by_expert(down),
                   runs_gate=runs_gate, runs_up=runs_up, runs_down=runs_down,
                   desc_gate=desc_gate, desc_up=desc_up, desc_down=desc_down,
                   tile_words_gate_up=tw_gate, tile_words_down=tw_down,
                   slot_words_gate_up=max(sw_gate, sw_up), slot_words_down=sw_down,
                   gs13=g13, gs2=g2,
                   ratio_gate=(gate.global_all / g13).contiguous(),
                   ratio_up=(up.global_all / g13).contiguous(),
                   ratio_down=(down.global_all / g2).contiguous(),
                   counters=torch.zeros(2, dtype=torch.int32, device=device),
                   activation=activation)

    @property
    def experts(self) -> int:
        return int(self.down.experts)

    @property
    def device(self) -> torch.device:
        return self.down.device

    def named_tables(self):
        """The tensors this lane holds BEYOND the bundles' own planes."""
        for part in ("gate", "up", "down"):
            yield f"routed_fused.runs_{part}", getattr(self, f"runs_{part}")
            yield f"routed_fused.desc_{part}", getattr(self, f"desc_{part}")
            yield f"routed_fused.ratio_{part}", getattr(self, f"ratio_{part}")

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for _n, t in self.named_tables())

    def _routing(self, expert_ids: torch.Tensor, routing_weights: torch.Tensor) -> _Routing:
        return _routing_tables(expert_ids, routing_weights, self.experts, self.device, None)

    def _quantized(self, x: torch.Tensor, gs: torch.Tensor):
        from .kernel_a4 import a4_quantize_activation

        if x.dtype != torch.bfloat16:
            raise GrammarError(f"the E2M1 lane takes a bf16 x, got {x.dtype}")
        codes, scales = a4_quantize_activation(x, gs)
        return codes.contiguous(), scales.view(torch.uint8).contiguous()

    def _launch(self, mode: int, xq: torch.Tensor, sfa: torch.Tensor, routing: _Routing, *,
                a_row_mode: int, mul_weight: bool, limit: float, out: torch.Tensor, counter: int) -> None:
        lib = _ext()
        if mode == 2:
            b0 = b1 = self.down
            w0 = w1 = self.words_down
            r0 = r1 = self.runs_down
            d0 = d1 = self.desc_down
            q0 = q1 = self.ratio_down
            tile_words, slot_words = self.tile_words_down, self.slot_words_down
        else:
            b0, b1 = self.gate, self.up
            w0, w1 = self.words_gate, self.words_up
            r0, r1 = self.runs_gate, self.runs_up
            d0, d1 = self.desc_gate, self.desc_up
            q0, q1 = self.ratio_gate, self.ratio_up
            tile_words, slot_words = self.tile_words_gate_up, self.slot_words_gate_up
        slot = self.counters[counter:counter + 1]
        slot.zero_()   # in-stream: a captured forward replays with a fresh work list
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        lib.routed_fused_forward_fp4(
            int(mode), xq, sfa, w0, w1, b0.codes_all, b1.codes_all,
            b0.init_all, b1.init_all, b0.has_init, b1.has_init,
            b0.scale_plane_all, b1.scale_plane_all, b0.scale_lut_all, b1.scale_lut_all,
            q0, q1, r0, r1, d0, d1,
            int(b0.rows), int(tile_words), int(slot_words),
            routing.offsets, routing.flat_sorted, routing.rw_sorted, routing.item_off, slot,
            int(routing.top_k), int(a_row_mode), bool(mul_weight), float(limit),
            out, _sm_count(index))

    def _check_x(self, x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        if x.dim() != 2 or tuple(x.shape) != (rows, cols) or x.device != self.device:
            raise GrammarError(f"x must be [{rows}, {cols}] on {self.device}, got {tuple(x.shape)} on {x.device}")
        return x if x.is_contiguous() else x.contiguous()

    def __call__(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor,
                 *, apply_router_weight_on_input: bool = False,
                 swiglu_limit: "float | None" = None) -> torch.Tensor:
        """The routed forward: bf16 ``[T, H]``."""
        from .native_window_moe import SUPPORTED_ACTIVATIONS, checked_swiglu_limit

        limit = checked_swiglu_limit(swiglu_limit)
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise GrammarError(f"activation {self.activation!r} has no exact implementation here")
        if expert_ids.dim() != 2:
            raise GrammarError("expert_ids must be [T, top_k]")
        tokens = int(expert_ids.shape[0])
        x = self._check_x(x, tokens, self.gate.cols)
        if apply_router_weight_on_input:
            if int(expert_ids.shape[1]) != 1:
                raise GrammarError(
                    "apply_router_weight_on_input is only implemented for topk=1 (vLLM's own "
                    f"prepare asserts this); got topk={int(expert_ids.shape[1])}")
            x = x * routing_weights.reshape(-1, 1).to(x.dtype)
        hidden, inter = self.down.rows, self.down.cols
        if tokens == 0:
            return torch.empty((0, hidden), dtype=torch.bfloat16, device=self.device)
        routing = self._routing(expert_ids, routing_weights)
        xq, sfa = self._quantized(x, self.gs13)
        act = torch.empty((routing.routes, inter), dtype=torch.bfloat16, device=self.device)
        self._launch(0, xq, sfa, routing, a_row_mode=0, mul_weight=False,
                     limit=limit if limit is not None else float("inf"), out=act, counter=0)
        aq, sfa2 = self._quantized(act, self.gs2)
        routed = torch.empty((routing.routes, hidden), dtype=torch.bfloat16, device=self.device)
        self._launch(2, aq, sfa2, routing, a_row_mode=1, mul_weight=not apply_router_weight_on_input,
                     limit=float("inf"), out=routed, counter=1)
        out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=self.device)
        _ext().token_sum(routed, out, int(routing.top_k))
        return out

    def gate_up(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        """Both projections, route-preserved: bf16 ``[T, top_k, 2I]`` (gate | up)."""
        routing = self._routing(expert_ids, routing_weights)
        x = self._check_x(x, routing.tokens, self.gate.cols)
        inter = self.down.cols
        out = torch.empty((routing.tokens, routing.top_k, 2 * inter), dtype=torch.bfloat16, device=self.device)
        if routing.tokens == 0:
            return out
        xq, sfa = self._quantized(x, self.gs13)
        self._launch(1, xq, sfa, routing, a_row_mode=0, mul_weight=False, limit=float("inf"),
                     out=out, counter=0)
        return out

    def down_routes(self, x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        """The down projection over the route-indexed activation ``[T * top_k,
        I]`` (row ``t * top_k + j``), weighted and reduced per token: bf16
        ``[T, H]`` (each route rounded to bf16 before the fixed-order fp32 sum)."""
        routing = self._routing(expert_ids, routing_weights)
        hidden, inter = self.down.rows, self.down.cols
        x = self._check_x(x, routing.routes, inter)
        out = torch.empty((routing.tokens, hidden), dtype=torch.bfloat16, device=self.device)
        if routing.tokens == 0:
            return out
        xq, sfa = self._quantized(x, self.gs2)
        routed = torch.empty((routing.routes, hidden), dtype=torch.bfloat16, device=self.device)
        self._launch(2, xq, sfa, routing, a_row_mode=2, mul_weight=True, limit=float("inf"),
                     out=routed, counter=1)
        _ext().token_sum(routed, out, int(routing.top_k))
        return out



@dataclasses.dataclass(frozen=True)
class DenseE2M1Role:
    """One dense E2M1 projection's kernel inputs, frozen at preparation so a
    forward reads no host value (the E = 1 case of the down launch).

    ``words``/``codes``/``plane``/``lut`` are the unit's own tensors viewed
    ``[1, ...]``; ``init``/``has_init`` its start state in the repack's column
    order; ``ratio`` the fp32 epilogue ``global / gs`` at the static A-side
    global ``gs``; ``runs``/``desc`` the run pair and chunk descriptors.
    """
    rows: int
    cols: int
    words: torch.Tensor       # int32 [1, W]
    codes: torch.Tensor       # uint8 [1, 2^L]
    init: torch.Tensor        # int32 [1, K]
    has_init: torch.Tensor    # int32 [1]
    plane: torch.Tensor       # uint8 [1, rows * cols / 32]
    lut: torch.Tensor         # uint8 [1, 16]
    gs: torch.Tensor          # fp32 scalar
    ratio: torch.Tensor       # fp32 [1]
    runs: torch.Tensor        # int32 [1, 8]
    desc: torch.Tensor        # int32 [1, K / 64, 4]
    tile_words: int
    slot_words: int


def dense_role_reason(unit) -> "str | None":
    """Why the dense launch refuses a ``compact_prep.WindowLutUnit``, or ``None``.

    The rows must be a multiple of 32 (``DENSE_ROWS``, one 16-byte part of
    the scale plane); they need not fill the last 256-row block, so GLM-5.3's
    DSA indexer ``wk`` (128 rows) and ``weights_proj`` (32) and a vocab-parallel
    ``lm_head`` (77,440 rows per rank at TP2) are in."""
    from .kernel_window_gemv import require_legacy_word_layout
    try:
        require_legacy_word_layout(getattr(unit.rep, "word_layout", "legacy"),
                                   "the E2M1 dense window reader")
    except GrammarError as exc:
        return str(exc)
    if unit.window_bits != WINDOW_BITS:
        return f"window_bits {unit.window_bits} != {WINDOW_BITS}"
    if unit.arity != 2:
        return f"code arity {unit.arity}; the E2M1 launch reads E2M1x2 tuples"
    if unit.cols % BK != 0 or unit.cols < MIN_COLS:
        return f"{unit.cols} columns; the launch needs a multiple of {BK} and at least {MIN_COLS}"
    if unit.rows % DENSE_ROWS != 0 or unit.rows <= 0:
        return f"{unit.rows} rows; the dense launch needs a positive multiple of {DENSE_ROWS}"
    pair, why = run_pair(unit.rep.runs, unit.cols)
    if pair is None:
        return why
    return None


def prepare_dense_role(unit, gs) -> DenseE2M1Role:
    """The kernel inputs for one admitted dense projection (a
    ``compact_prep.WindowLutUnit``) at the static A-side global ``gs``;
    builds the library first."""
    reason = dense_role_reason(unit)
    if reason is not None:
        raise GrammarError(f"the fused E2M1 dense launch refuses this unit: {reason}")
    _ext()
    device = unit.codes.device
    g = _static_global(gs, device, "gs")
    pair, _ = run_pair(unit.rep.runs, unit.cols)
    init = unit.permuted_start_state()
    cols = int(unit.cols)
    return DenseE2M1Role(
        rows=int(unit.rows), cols=cols,
        words=unit.rep.words.reshape(1, -1).to(torch.int32).contiguous(),
        codes=unit.codes.reshape(1, -1).contiguous(),
        init=(init.to(torch.int32) if init is not None
              else torch.zeros(cols, dtype=torch.int32, device=device)).reshape(1, cols).contiguous(),
        has_init=torch.tensor([0 if init is None else 1], dtype=torch.int32, device=device),
        plane=unit.scale_plane.reshape(1, -1).contiguous(),
        lut=unit.scale_lut.reshape(1, 16).contiguous(),
        gs=g,
        ratio=(torch.tensor([unit.global_scale], dtype=torch.float32, device=device) / g).contiguous(),
        runs=pair.reshape(1, 8).to(device=device, dtype=torch.int32).contiguous(),
        desc=chunk_desc(unit.rep.perm.reshape(1, -1), int(pair[2]), cols).to(device),
        tile_words=pair_tile_words(pair), slot_words=slot_words_for_pair(pair))


def dense_forward(role: DenseE2M1Role, x: torch.Tensor, *, k_split: int = 1,
                  out: "torch.Tensor | None" = None) -> torch.Tensor:
    """bf16 ``[M, rows]``: the role over bf16 ``x [M, cols]``, the activation
    quantised at the role's static global.  ``k_split`` > 1 splits K into
    that many items per 64-row superblock and 256-row block, summed in fixed
    order in fp32 before the one epilogue; at most :func:`dense_split_max`.
    Reads no host value, so a captured forward replays."""
    from .kernel_a4 import a4_quantize_activation

    if x.dtype != torch.bfloat16 or x.dim() != 2 or int(x.shape[1]) != role.cols:
        raise GrammarError(f"x must be bf16 [M, {role.cols}], got {x.dtype} {tuple(x.shape)}")
    if not 1 <= int(k_split) <= dense_split_max(role.cols):
        raise GrammarError(f"k_split {k_split} is outside [1, {dense_split_max(role.cols)}] at "
                           f"{role.cols} columns: every split item keeps two K chunks")
    m = int(x.shape[0])
    device = x.device
    if out is None:
        out = torch.empty((m, role.rows), dtype=torch.bfloat16, device=device)
    if m == 0:
        return out
    codes, scales = a4_quantize_activation(x.contiguous(), role.gs)
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    partial = torch.empty(int(k_split) * m * role.rows if k_split > 1 else 0, dtype=torch.float32, device=device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    _ext().dense_forward_fp4(codes.contiguous(), scales.view(torch.uint8).contiguous(),
                             role.words, role.codes, role.init, role.has_init, role.plane, role.lut,
                             role.ratio, role.runs, role.desc, role.rows, int(role.tile_words),
                             int(role.slot_words), counter, int(k_split), partial, out, _sm_count(index))
    return out
