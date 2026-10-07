"""The register-direct routed window kernel: the Python owner (eng-regdirect-build, stage 1).

The kernel (``serving/csrc/regdirect_routed.cu``) decodes each warp's own MMA
A fragment from the FRAGMENT-ORDER wire (``tessera.fragment_wire``) and never
stages a decoded weight in shared memory.  This module owns its launch
geometry and its scratch; the routing front end (the per-expert class build)
owns the sorted routes, prefixes and streams.

Stage 1 is not a serving path: today's fused routed kernel stays selected until
this one passes D41 rows, a G3 v2 check and an end-to-end serve (CEO, 2026-10-07).
"""
from __future__ import annotations

import dataclasses
import functools
import os

import torch

TILE = 128            # rows per task
KSTEP = 32            # columns per pair group per unit k-step
HIST_LANES = 8        # a history unit keeps lanes (6, t) and (7, t)
KS_MAX = 128          # unit k-steps per expert
SUPERBLOCK_DECODE = 8                 # routes per decode superblock (one route tile)
STAGE1_RATES = (3, 4)                 # T-8 R768 .. R1024 (CEO, 2026-10-07)
MODES = {"gate_up": 0, "down": 2}
GROUPS = {0: 1, 2: 2}                 # activation column groups per unit k-step
TABLES = {0: 2, 2: 1}                 # tables (and scale rows) per expert


def pack_rate(ra: int, rb: int, ksa: int) -> int:
    """One expert's rate profile as the kernel reads it: slots [0, ksa) at ``ra``, then ``rb``."""
    if ra not in STAGE1_RATES or rb not in STAGE1_RATES:
        raise ValueError(f"stage 1 serves rates {STAGE1_RATES}, got ({ra}, {rb})")
    if not 0 <= ksa <= KS_MAX:
        raise ValueError(f"ksa {ksa} outside [0, {KS_MAX}]")
    return ra | (rb << 4) | (ksa << 8)


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The launch geometry the kernel owns (the front-end contract's kernel side)."""
    route_tiles: int         # 1: the decode path; 8 or 16: prefill (gate/up at 16 splits by projection)
    superblock: int          # routes per superblock item (8 x route tiles)
    k_parts: int             # P
    tiles: int               # task tiles per item
    grid: int

    @property
    def prefill(self) -> bool:
        return self.route_tiles > 1

    @property
    def units_per_item(self) -> int:
        """Work units per superblock item: the factor of the front end's interval rule."""
        return self.tiles * self.k_parts


def route_tiles(m: int, top_k: int, experts: int) -> int:
    """The decode path (1) until an expert averages more routes than one decode superblock
    holds (8); then 64-route superblocks (8); above 64 routes per expert, 128 (16), so each
    expert's wire is read once (stage-1 gpu-01: at M = 4096 two 64-route superblocks per
    expert doubled the time)."""
    routes = m * top_k
    if routes <= SUPERBLOCK_DECODE * experts:
        return 1
    return 8 if routes <= 64 * experts else 16


def is_prefill(m: int, top_k: int, experts: int) -> bool:
    return route_tiles(m, top_k, experts) > 1


DECODE_DEPTH = 4       # the decode path's prefetch depth (rd_decode D): the pipeline refill, in k-steps


def k_parts(items: int, tiles: int, grid: int, ks: int) -> int:
    """K parts for the decode path: the P in 1..8 with the least estimated time per CTA,
    rounds x (k-steps per part + the pipeline refill of DECODE_DEPTH k-steps).  A part costs a
    refill and a partial-sum exchange, so P > 1 pays only where it fills idle CTAs (gate/up at
    M = 1: P = 3; M = 16 and down at M = 1: P = 1)."""
    best, best_cost = 1, None
    for p in range(1, min(8, ks) + 1):
        rounds = -(-(items * tiles * p) // grid)
        cost = rounds * (ks / p + DECODE_DEPTH)
        if best_cost is None or cost < best_cost - 1e-9:
            best, best_cost = p, cost
    return best


def geometry(mode: int, m: int, top_k: int, rows: int, experts: int, ks: int, sms: int) -> Geometry:
    rt = route_tiles(m, top_k, experts)
    task_rows = 64 if (mode == 0 and rt == 16) else TILE      # the split gate/up variant
    tiles = rows // task_rows
    grid = sms * _ext().blocks_per_sm(mode, rt, False)
    items = min(experts, m * top_k)              # the balanced estimate; the device sees the real count
    parts = 1 if rt > 1 else k_parts(items, tiles, grid, ks)
    return Geometry(rt, 8 * rt, parts, tiles, grid)


def scratch(geom: Geometry, experts: int, routes: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """K-part partials and self-resetting arrival counters, indexed by ABSOLUTE work unit, so two
    classes on two streams never share a slot.  Allocate once per layer; the counters start at 0."""
    items = experts + routes // geom.superblock + 1
    n_units = items * geom.tiles
    if geom.k_parts > 1:
        part = torch.empty(n_units * geom.k_parts * 2 * TILE * geom.superblock, dtype=torch.float32, device=device)
    else:
        part = torch.empty(1, dtype=torch.float32, device=device)
    return part, torch.zeros(n_units, dtype=torch.int32, device=device)


def with_zero_row(x: torch.Tensor) -> torch.Tensor:
    """The activation buffer the kernel reads: ``x`` with one zero row appended (row M)."""
    out = torch.zeros((x.shape[0] + 1, x.shape[1]), dtype=x.dtype, device=x.device)
    out[: x.shape[0]].copy_(x)
    return out


@functools.lru_cache(maxsize=None)
def _ext():
    """Build and load the kernel on this process's platform.  Not a serving path (stage 1)."""
    from torch.utils.cpp_extension import load

    from tessera.serving.backend import (
        backend as detect_backend,
        ensure_toolchain_on_path,
        pin_build_arch,
        platform_token,
        offload_flags,
    )
    from .jit_build_lock import GUARDED_BUILD_SUFFIX, jit_build_lock

    ensure_toolchain_on_path(torch)
    if detect_backend(torch) != "cuda":
        raise RuntimeError("the register-direct routed kernel is CUDA (mma.sync, cp.async)")
    src = os.path.join(os.path.dirname(__file__), "serving", "csrc", "regdirect_routed.cu")
    token = platform_token(torch=torch)
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser("~/tmp/torch-ext-routed-fused")
    build = os.path.join(root, f"tessera_regdirect_routed_{token}") + GUARDED_BUILD_SUFFIX
    os.makedirs(build, exist_ok=True)
    pin_build_arch(token, torch)
    flags = ["-O3", "-lineinfo", "-std=c++17", *offload_flags(token)]
    with jit_build_lock(build):
        return load(name="tessera_regdirect_routed", sources=[src], build_directory=build,
                    extra_cuda_cflags=flags, verbose=bool(os.environ.get("TESSERA_ROUTED_FUSED_VERBOSE")))


@dataclasses.dataclass
class FragmentStack:
    """One projection mode's fragment-order planes for every expert of a layer (storage order)."""
    mode: int                    # 0 gate/up, 2 down
    wire: torch.Tensor           # int32 [words]
    expert_word0: torch.Tensor   # int64 [E]
    hist: torch.Tensor           # int32 [words]
    expert_hist0: torch.Tensor   # int64 [E]
    rate: torch.Tensor           # int32 [E], pack_rate
    kperm: torch.Tensor          # int16 [E, KS * groups]
    table: torch.Tensor          # uint8 [E, tables, 16384]
    wscale: torch.Tensor         # float32 [E, tables, rows]
    ks: int                      # unit k-steps

    def __post_init__(self):
        if self.mode not in GROUPS:
            raise ValueError(f"mode is 0 (gate/up) or 2 (down), got {self.mode}")
        self.zeros = torch.zeros(4096, dtype=torch.uint8, device=self.wire.device)

    @property
    def experts(self) -> int:
        return self.table.shape[0]

    @property
    def rows(self) -> int:
        return self.wscale.shape[2]

    @property
    def k(self) -> int:
        return self.ks * GROUPS[self.mode] * KSTEP

    def launch(self, geom: Geometry, x_zero_row: torch.Tensor, a_scale: torch.Tensor, offsets: torch.Tensor,
               sorted_routes: torch.Tensor, route_weight: torch.Tensor | None, item_off: torch.Tensor,
               e0: int, e1: int, out: torch.Tensor, part: torch.Tensor, arrive: torch.Tensor, *, top_k: int,
               a_row_mode: int, mul_weight: bool, limit: float, l2_hint: bool | None = None,
               dump: torch.Tensor | None = None) -> None:
        """One launch over experts [e0, e1) on the current stream.  ``item_off`` must count
        superblocks of ``geom.superblock`` routes; ``x_zero_row`` is :func:`with_zero_row`."""
        hint = geom.prefill if l2_hint is None else l2_hint      # activation reuse exists at prefill only
        rw = route_weight if route_weight is not None else torch.empty(0, dtype=torch.float32, device=out.device)
        dbuf = dump if dump is not None else torch.empty(1, dtype=torch.uint8, device=out.device)
        _ext().forward(self.mode, geom.route_tiles, dump is not None, self.wire, self.expert_word0, self.hist,
                       self.expert_hist0, self.rate, self.kperm, self.table, self.wscale, x_zero_row, a_scale,
                       offsets, sorted_routes, rw, item_off, e0, e1, out, part, arrive, dbuf, self.zeros,
                       self.ks, top_k, geom.k_parts, a_row_mode, mul_weight, limit, hint, geom.grid)


class RegDirectClassKernel:
    """The ``routed_class_dispatch.RoutedClassKernel`` binding of this kernel.

    ``parameters["regdirect"]`` maps mode (0 gate/up, 2 down) to one :class:`FragmentStack`
    that holds every expert of the layer in storage order, so a span's absolute
    ``start``/``end`` index it directly.  The binding owns the K-part scratch, by absolute
    work unit, sized at :meth:`reserve` for ``max_tokens``; a launch never allocates it.
    The claim counter is unused: work units are striped over a fixed grid.
    """

    def __init__(self, experts: int, top_k: int, device):
        self.experts, self.top_k = experts, top_k
        self.device = torch.device(device)
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._scratch = {}
        self._max_tokens = 0

    def geometry(self, mode: int, tokens: int, stack: FragmentStack) -> Geometry:
        return geometry(mode, tokens, self.top_k, stack.rows, self.experts, stack.ks, self.sms)

    def reserve(self, parameters: dict, max_tokens: int) -> None:
        """Allocate, per mode, the largest scratch any token count up to ``max_tokens`` needs
        (call at load).  Each launch indexes its own absolute units inside it, and the arrival
        counters return to zero after every use, so one pair serves every geometry."""
        routes = max_tokens * self.top_k
        for mode, stack in parameters["regdirect"].items():
            part = arrive = 0
            for m in range(1, max_tokens + 1):
                p, a = (t.numel() for t in scratch(self.geometry(mode, m, stack), self.experts, routes, "meta"))
                part, arrive = max(part, p), max(arrive, a)
            self._scratch[mode] = (torch.empty(part, dtype=torch.float32, device=self.device),
                                   torch.zeros(arrive, dtype=torch.int32, device=self.device))
        self._max_tokens = max(self._max_tokens, max_tokens)

    def work_shape(self, mode, tokens, index, parameters):
        g = self.geometry(mode, tokens, parameters["regdirect"][mode])
        return g.superblock, g.units_per_item

    def prepare_input(self, x, a_scale, rows, family, device):
        from .errors import GrammarError
        from .routed_fused import quantized_routed_input
        if family == "value":
            raise GrammarError("the register-direct kernel takes E4M3 activations, not the value family")
        xq, scale = quantized_routed_input(x, a_scale, rows, family, device)
        return with_zero_row(xq), scale

    def launch(self, mode, x, a_scale, *, index, start, end, prefix, counter, routing, parameters, bm,
               work_units, empty_scale, a_row_mode, mul_weight, limit, out):
        from .errors import GrammarError
        stack = parameters["regdirect"][mode]
        g = self.geometry(mode, routing.tokens, stack)
        if (g.superblock, g.units_per_item) != (bm, work_units):
            raise GrammarError(f"launch shape ({bm}, {work_units}) differs from work_shape "
                               f"({g.superblock}, {g.units_per_item})")
        if routing.tokens > self._max_tokens:
            raise GrammarError(f"{routing.tokens} tokens exceed the reserved {self._max_tokens}; "
                               "call reserve() at load")
        part, arrive = self._scratch[mode]
        stack.launch(g, x, a_scale, routing.offsets, routing.flat_sorted, routing.rw_sorted if mul_weight else None,
                     prefix, start, end, out, part, arrive, top_k=routing.top_k, a_row_mode=a_row_mode,
                     mul_weight=mul_weight, limit=limit)


# ---------------------------------------------------------------------------
# the layer build: fragment-order stacks from the class build's construction planes
# ---------------------------------------------------------------------------
def fragment_stack(mode: int, experts) -> dict:
    """The :class:`FragmentStack` planes of one projection mode.

    ``experts`` yields, per expert in storage order, ``(codes [P, rows, cols], rates,
    start [P, cols] | None, table uint8 [P, 2^14], scale fp32 [P, rows])``: P = 2 (gate, up) for
    mode 0, 1 (down) for mode 2, codes in original column order.  Each is repacked with
    :func:`tessera.fragment_wire.repack_codes` on the codes' device."""
    from .fragment_wire import repack_codes
    group = "gate_up" if mode == 0 else "down"
    wire, hist, w0, h0, rate, kperm, table, wscale = [], [], [], [], [], [], [], []
    wo = ho = 0
    for codes, rates, start, tab, scale in experts:
        fw = repack_codes(codes, tuple(rates), projection_group=group, start_state=start)
        h = int(fw.unit_offsets[0, 0, 0] - fw.expert_offsets[0])
        slot_rates = fw.rates.tolist()
        ra, rb = slot_rates[0], slot_rates[-1]
        rate.append(pack_rate(ra, rb, slot_rates.count(ra) if ra != rb else len(slot_rates)))
        wire.append(fw.words[h:]); hist.append(fw.words[:h])
        w0.append(wo); h0.append(ho)
        wo += wire[-1].numel(); ho += h
        kperm.append(fw.perm.to(codes.device)); table.append(tab); wscale.append(scale.float())
    device = wire[0].device
    return dict(mode=mode, wire=torch.cat(wire), expert_word0=torch.tensor(w0, dtype=torch.int64, device=device),
                hist=torch.cat(hist), expert_hist0=torch.tensor(h0, dtype=torch.int64, device=device),
                rate=torch.tensor(rate, dtype=torch.int32, device=device), kperm=torch.stack(kperm).contiguous(),
                table=torch.stack(table).contiguous(), wscale=torch.stack(wscale).contiguous(),
                ks=kperm[0].numel() // GROUPS[mode])


def _bundle_expert(bundle, e: int):
    """Expert ``e`` of a routed construction bundle (``window_gemm_grouped.PreparedGroupedWindowGemm``):
    BODY codes ``[rows, cols]``, per-column rates and the start state (or None), original column order."""
    from .kernel_window_gemv import TILE_ROWS, Repacked, unpack_tile_words
    words = bundle.words_all
    total = int(bundle.total_words[e])
    words = words[e, :total] if words.dim() == 2 else words.narrow(0, int(bundle.word_off[e]), total)
    runs = bundle.runs_all[int(bundle.run_off[e]):int(bundle.run_off[e + 1])].reshape(-1, 4)
    perm = bundle.perm_all[e].long()
    n_tiles = -(-int(bundle.rows) // TILE_ROWS)
    rep = Repacked(words=words, tile_words=int(bundle.tile_words[e]), n_tiles=n_tiles, rows=int(bundle.rows),
                   cols=int(bundle.cols), rows_p=n_tiles * TILE_ROWS, perm=perm, runs=runs, rates=(),
                   word_layout=str(bundle.word_layout))
    codes = unpack_tile_words(rep)
    permuted = torch.empty(int(bundle.cols), dtype=torch.int64)
    for r, col0, n, _ in runs.tolist():
        permuted[col0:col0 + n] = r
    rates = torch.empty_like(permuted)
    rates[perm.cpu()] = permuted
    start = None
    if int(bundle.has_init[e]):
        start = torch.empty(int(bundle.cols), dtype=torch.int64, device=codes.device)
        start[perm] = bundle.init_all[e].to(torch.int64)                       # init_perm[j] = init[perm[j]]
    return codes, tuple(rates.tolist()), start


def layer_stacks(gate, up, down, tables) -> dict:
    """``{0: gate/up planes, 2: down planes}`` from the class build's full-layer bundles and their
    composed tables (``routed_fused.compose_table8``: uint8 ``[E, 2^14]``, the E4M3 instruction's)."""
    from .errors import GrammarError
    for b, t in zip((gate, up, down), tables):
        if b.family != "e4m3" or b.arithmetic != "epilogue" or int(b.window_bits) != 14:
            raise GrammarError(f"the register-direct kernel serves the e4m3 family at a 14-bit window with "
                               f"the row-scale epilogue, got {b.family}/{b.arithmetic}/L={b.window_bits}")
        if t.dtype != torch.uint8 or tuple(t.shape) != (int(b.experts), 1 << 14):
            raise GrammarError("the register-direct kernel reads compose_table8's uint8 [E, 2^14] byte table")
    n = int(down.experts)

    def gate_up():
        for e in range(n):
            (cg, rg, sg), (cu, ru, su) = _bundle_expert(gate, e), _bundle_expert(up, e)
            if rg != ru:
                raise GrammarError(f"expert {e}: gate and up differ in rate; one k-step carries both")
            start = None if sg is None and su is None else torch.stack(
                [s if s is not None else torch.zeros_like(cg[0], dtype=torch.int64) for s in (sg, su)])
            yield (torch.stack([cg, cu]), rg, start, torch.stack([tables[0][e], tables[1][e]]),
                   torch.stack([gate.scale_all[e], up.scale_all[e]]))

    def down_():
        for e in range(n):
            c, r, s = _bundle_expert(down, e)
            yield c.unsqueeze(0), r, None if s is None else s.unsqueeze(0), tables[2][e].unsqueeze(0), down.scale_all[e].unsqueeze(0)

    return {0: fragment_stack(0, gate_up()), 2: fragment_stack(2, down_())}


def build_layer(gate, up, down, tables, *, top_k: int, max_tokens: int):
    """``(parameters, kernel)`` for ``routed_class_dispatch``: the fragment stacks of one routed layer
    and its :class:`RegDirectClassKernel`, scratch reserved for ``max_tokens``."""
    stacks = {mode: FragmentStack(**planes) for mode, planes in layer_stacks(gate, up, down, tables).items()}
    parameters = {"regdirect": stacks}
    kernel = RegDirectClassKernel(int(down.experts), top_k, down.words_all.device)
    kernel.reserve(parameters, max_tokens)
    return parameters, kernel
