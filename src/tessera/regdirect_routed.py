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
SERVED_RATES = (3, 4, 5, 6, 7, 8)     # T-8 R768 .. R2048: the code rates the kernel instantiates (by_rate)
MODES = {"gate_up": 0, "down": 2}
GROUPS = {0: 1, 2: 2}                 # activation column groups per unit k-step
TABLES = {0: 2, 2: 1}                 # tables (and scale rows) per expert


def pack_rate(ra: int, rb: int, ksa: int) -> int:
    """One expert's rate profile as the kernel reads it: slots [0, ksa) at ``ra``, then ``rb``."""
    if ra not in SERVED_RATES or rb not in SERVED_RATES:
        raise ValueError(f"the kernel serves rates {SERVED_RATES}, got ({ra}, {rb})")
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
def _load(name: str, source: str):
    """Build and load one of this module's CUDA extensions on this process's platform.
    Not a serving path (stage 1)."""
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
    src = os.path.join(os.path.dirname(__file__), "serving", "csrc", source)
    token = platform_token(torch=torch)
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser("~/tmp/torch-ext-routed-fused")
    build = os.path.join(root, f"{name}_{token}") + GUARDED_BUILD_SUFFIX
    os.makedirs(build, exist_ok=True)
    pin_build_arch(token, torch)
    flags = ["-O3", "-lineinfo", "-std=c++17", *offload_flags(token)]
    with jit_build_lock(build):
        return load(name=name, sources=[src], build_directory=build,
                    extra_cuda_cflags=flags, verbose=bool(os.environ.get("TESSERA_ROUTED_FUSED_VERBOSE")))


def _ext():
    """The register-direct routed kernel."""
    return _load("tessera_regdirect_routed", "regdirect_routed.cu")


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

    def planes(self) -> tuple:
        """The weight planes in :data:`PAYLOAD_FIELDS` order (without the scratch)."""
        return tuple(getattr(self, name) for name in PAYLOAD_FIELDS[:-2])

    def launch(self, geom: Geometry, x_zero_row: torch.Tensor, a_scale: torch.Tensor, offsets: torch.Tensor,
               sorted_routes: torch.Tensor, route_weight: torch.Tensor | None, item_off: torch.Tensor,
               e0: int, e1: int, out: torch.Tensor, part: torch.Tensor, arrive: torch.Tensor, *, top_k: int,
               a_row_mode: int, mul_weight: bool, limit: float, l2_hint: bool | None = None,
               dump: torch.Tensor | None = None) -> None:
        """One launch over experts [e0, e1) on the current stream.  ``item_off`` must count
        superblocks of ``geom.superblock`` routes; ``x_zero_row`` is :func:`with_zero_row`."""
        forward(self.mode, self.planes() + (part, arrive), geom, x_zero_row, a_scale, offsets, sorted_routes,
                route_weight, item_off, e0, e1, out, top_k=top_k, a_row_mode=a_row_mode, mul_weight=mul_weight,
                limit=limit, l2_hint=l2_hint, dump=dump)


#: The tensor payload of one projection mode, the opaque operation's argument list: the
#: fragment planes (``zeros`` is the kernel's zero page), then the K-part scratch (``part``)
#: and the self-resetting arrival counters (``arrive``) sized by :func:`reserve_scratch`.
PAYLOAD_FIELDS = ("wire", "expert_word0", "hist", "expert_hist0", "rate", "kperm", "table", "wscale", "zeros",
                  "part", "arrive")


def payload_rows_ks(mode: int, payload) -> tuple[int, int, int]:
    """(experts, rows, unit k-steps) of a payload, read off its own planes."""
    kperm, wscale = payload[PAYLOAD_FIELDS.index("kperm")], payload[PAYLOAD_FIELDS.index("wscale")]
    return int(wscale.shape[0]), int(wscale.shape[2]), int(kperm.shape[1]) // GROUPS[mode]


def forward(mode: int, payload, geom: Geometry, x_zero_row: torch.Tensor, a_scale: torch.Tensor,
            offsets: torch.Tensor, sorted_routes: torch.Tensor, route_weight: torch.Tensor | None,
            item_off: torch.Tensor, e0: int, e1: int, out: torch.Tensor, *, top_k: int, a_row_mode: int,
            mul_weight: bool, limit: float, l2_hint: bool | None = None, dump: torch.Tensor | None = None) -> None:
    """One launch from a :data:`PAYLOAD_FIELDS` tensor payload, over experts [e0, e1)."""
    wire, w0, hist, h0, rate, kperm, table, wscale, zeros, part, arrive = payload
    _, _, ks = payload_rows_ks(mode, payload)
    hint = geom.prefill if l2_hint is None else l2_hint      # activation reuse exists at prefill only
    rw = route_weight if route_weight is not None else torch.empty(0, dtype=torch.float32, device=out.device)
    dbuf = dump if dump is not None else torch.empty(1, dtype=torch.uint8, device=out.device)
    _ext().forward(mode, geom.route_tiles, dump is not None, wire, w0, hist, h0, rate, kperm, table, wscale,
                   x_zero_row, a_scale, offsets, sorted_routes, rw, item_off, e0, e1, out, part, arrive, dbuf, zeros,
                   ks, top_k, geom.k_parts, a_row_mode, mul_weight, limit, hint, geom.grid)


def _geometry_tokens(top_k: int, experts: int, max_tokens: int) -> list:
    """Token counts that reach every geometry up to ``max_tokens``: :func:`geometry` reads M only
    through ``min(experts, M * top_k)`` (the K parts) and the route-tile thresholds."""
    cut = {1, max_tokens}
    cut.update(range(1, min(max_tokens, -(-experts // top_k)) + 1))
    for limit in (SUPERBLOCK_DECODE * experts, 64 * experts):     # route_tiles changes past these routes
        m = limit // top_k
        cut.update(v for v in (m, m + 1) if 1 <= v <= max_tokens)
    return sorted(cut)


def reserve_scratch(mode: int, rows: int, ks: int, experts: int, top_k: int, max_tokens: int, sms: int,
                    device) -> tuple[torch.Tensor, torch.Tensor]:
    """The largest K-part scratch and arrival counters any token count up to ``max_tokens`` needs.
    Each launch indexes its own absolute units inside them, and the counters return to zero after
    every use, so one pair serves every geometry of the mode."""
    routes = max_tokens * top_k
    part = arrive = 0
    for m in _geometry_tokens(top_k, experts, max_tokens):
        g = geometry(mode, m, top_k, rows, experts, ks, sms)
        p, a = (t.numel() for t in scratch(g, experts, routes, "meta"))
        part, arrive = max(part, p), max(arrive, a)
    return (torch.empty(part, dtype=torch.float32, device=device),
            torch.zeros(arrive, dtype=torch.int32, device=device))


class RegDirectClassKernel:
    """The ``routed_class_dispatch.RoutedClassKernel`` binding of this kernel.

    ``parameters["regdirect"]`` maps mode (0 gate/up, 2 down) to one :data:`PAYLOAD_FIELDS`
    tensor payload holding every expert of the layer in storage order, so a span's absolute
    ``start``/``end`` index it directly.  The binding holds no tensor: the planes and the K-part
    scratch (by absolute work unit) are the caller's payload.  The claim counter is unused:
    work units are striped over a fixed grid.
    """

    def __init__(self, experts: int, top_k: int, device):
        self.experts, self.top_k = experts, top_k
        self.sms = torch.cuda.get_device_properties(torch.device(device)).multi_processor_count

    def geometry(self, mode: int, tokens: int, payload) -> Geometry:
        _, rows, ks = payload_rows_ks(mode, payload)
        return geometry(mode, tokens, self.top_k, rows, self.experts, ks, self.sms)

    def payload(self, mode: int, stack: FragmentStack, max_tokens: int) -> tuple:
        """``stack``'s planes with the scratch :func:`reserve_scratch` sizes for ``max_tokens``."""
        return stack.planes() + reserve_scratch(mode, stack.rows, stack.ks, self.experts, self.top_k,
                                                max_tokens, self.sms, stack.wire.device)

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
        payload = parameters["regdirect"][mode]
        g = self.geometry(mode, routing.tokens, payload)
        if (g.superblock, g.units_per_item) != (bm, work_units):
            raise GrammarError(f"launch shape ({bm}, {work_units}) differs from work_shape "
                               f"({g.superblock}, {g.units_per_item})")
        need = scratch(g, self.experts, routing.routes, "meta")
        if need[0].numel() > payload[-2].numel() or need[1].numel() > payload[-1].numel():
            raise GrammarError(f"{routing.tokens} tokens exceed the payload's reserved scratch; "
                               "reserve it for the serving max_tokens at load")
        forward(mode, payload, g, x, a_scale, routing.offsets, routing.flat_sorted,
                routing.rw_sorted if mul_weight else None, prefix, start, end, out, top_k=routing.top_k,
                a_row_mode=a_row_mode, mul_weight=mul_weight, limit=limit)


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


def layer_stacks(gate, up, down, tables, rungs: "dict | None" = None) -> dict:
    """``{0: gate/up planes, 2: down planes}`` from the class build's full-layer bundles and their
    composed tables (``routed_fused.compose_table8``: uint8 ``[E, 2^14]``, the E4M3 instruction's).
    ``rungs[e]`` = (gate, up, down) q256: each expert's rates must spend exactly that rung."""
    from .errors import GrammarError
    for b, t in zip((gate, up, down), tables):
        if b.family != "e4m3" or b.arithmetic != "epilogue" or int(b.window_bits) != 14:
            raise GrammarError(f"the register-direct kernel serves the e4m3 family at a 14-bit window with "
                               f"the row-scale epilogue, got {b.family}/{b.arithmetic}/L={b.window_bits}")
        if t.dtype != torch.uint8 or tuple(t.shape) != (int(b.experts), 1 << 14):
            raise GrammarError("the register-direct kernel reads compose_table8's uint8 [E, 2^14] byte table")
    n = int(down.experts)

    def spend(e, which, rates):
        if rungs is not None and sum(rates) * 256 != rungs[e][which] * len(rates):
            raise GrammarError(f"expert {e}: rates spend {sum(rates) * 256 / len(rates):g} q256, "
                               f"its class declares {rungs[e][which]}")

    def gate_up():
        for e in range(n):
            (cg, rg, sg), (cu, ru, su) = _bundle_expert(gate, e), _bundle_expert(up, e)
            if rg != ru:
                raise GrammarError(f"expert {e}: gate and up differ in rate; one k-step carries both")
            spend(e, 0, rg)
            spend(e, 1, ru)
            start = None if sg is None and su is None else torch.stack(
                [s if s is not None else torch.zeros_like(cg[0], dtype=torch.int64) for s in (sg, su)])
            yield (torch.stack([cg, cu]), rg, start, torch.stack([tables[0][e], tables[1][e]]),
                   torch.stack([gate.scale_all[e], up.scale_all[e]]))

    def down_():
        for e in range(n):
            c, r, s = _bundle_expert(down, e)
            spend(e, 2, r)
            yield c.unsqueeze(0), r, None if s is None else s.unsqueeze(0), tables[2][e].unsqueeze(0), down.scale_all[e].unsqueeze(0)

    return {0: fragment_stack(0, gate_up()), 2: fragment_stack(2, down_())}


def _field(c, name):
    return c[name] if isinstance(c, dict) else getattr(c, name)


def build_layer(gate, up, down, classes, device, *, top_k: int, max_tokens: int):
    """``(parameters, kernel)`` for ``routed_class_dispatch``: ``parameters["regdirect"]`` maps
    modes 0 and 2 to their :data:`PAYLOAD_FIELDS` tensor payloads, scratch reserved for
    ``max_tokens``; ``kernel`` is the tensor-free :class:`RegDirectClassKernel`.

    ``gate``/``up``/``down`` are the full storage-ordered construction bundles before retirement
    (TP cut applied); ``classes`` give ``start``, ``end`` and ``q256`` (``w13``: gate, up;
    ``w2``: down), and every expert's actual rates must spend exactly its class's rung."""
    from .errors import GrammarError
    from .routed_fused import compose_table8
    n = int(down.experts)
    spans = sorted((int(_field(c, "start")), int(_field(c, "end")), _field(c, "q256")) for c in classes)
    if [s[0] for s in spans] != [0] + [s[1] for s in spans[:-1]] or spans[-1][1] != n:
        raise GrammarError(f"classes {[(a, b) for a, b, _ in spans]} do not tile the {n} experts in order")
    rungs = {}
    for a, b, q in spans:
        for e in range(a, b):
            rungs[e] = (int(q["w13"][0]), int(q["w13"][1]), int(q["w2"][0]))
    tables = tuple(compose_table8(b).to(device) for b in (gate, up, down))
    # The device transcode: bit-identical to layer_stacks (tests/test_regdirect_transcode_cuda.py), ~260x faster.
    stacks = {mode: FragmentStack(**planes) for mode, planes in transcode_stacks(gate, up, down, tables, rungs).items()}
    kernel = RegDirectClassKernel(n, top_k, device)
    return {"regdirect": {mode: kernel.payload(mode, st, max_tokens) for mode, st in stacks.items()}}, kernel


def _transcode_ext():
    """Tile-order words -> fragment words on the device (``serving/csrc/regdirect_transcode.cu``)."""
    return _load("tessera_regdirect_transcode", "regdirect_transcode.cu")


def _bundle_columns(bundle):
    """Per expert and ORIGINAL column: the chunk word offset in a 512-row tile, the rate and the
    start state (0 when none), all ``[E, cols]`` on the bundle's device, from runs/perm/init."""
    e_n, cols = int(bundle.experts), int(bundle.cols)
    dev = bundle.words_all.device
    runs = bundle.runs_all.reshape(-1, 4).to(dev).long()
    run_off = bundle.run_off.to(dev).long()
    expert = torch.searchsorted(run_off[1:], torch.arange(runs.shape[0], device=dev), right=True)
    rate, col0, n, word0 = runs.unbind(1)
    within = torch.arange(int(n.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
    pcol = torch.repeat_interleave(col0, n) + within                     # permuted column
    e_of = torch.repeat_interleave(expert, n)
    base = torch.repeat_interleave(word0, n) + within * 16 * torch.repeat_interleave(rate, n)
    perm = bundle.perm_all.to(dev).long()
    orig = perm[e_of, pcol]
    colbase = torch.empty(e_n, cols, dtype=torch.int64, device=dev)
    colrate = torch.empty(e_n, cols, dtype=torch.int64, device=dev)
    colbase[e_of, orig] = base
    colrate[e_of, orig] = torch.repeat_interleave(rate, n)
    start = torch.zeros(e_n, cols, dtype=torch.int64, device=dev)
    start.scatter_(1, perm, bundle.init_all.to(dev).long())                 # init_perm[j] = init[perm[j]]
    start *= (bundle.has_init.to(dev).long() != 0).unsqueeze(1)
    return colbase.to(torch.int32), colrate, start.to(torch.int32)


def transcode_stacks(gate, up, down, tables, rungs: "dict | None" = None) -> dict:
    """:func:`layer_stacks` on the device: one transcode launch per projection group instead of a
    per-expert repack.  Same planes, bit for bit; the same refusals."""
    from .errors import GrammarError
    for b, t in zip((gate, up, down), tables):
        if b.family != "e4m3" or b.arithmetic != "epilogue" or int(b.window_bits) != 14:
            raise GrammarError(f"the register-direct kernel serves the e4m3 family at a 14-bit window with "
                               f"the row-scale epilogue, got {b.family}/{b.arithmetic}/L={b.window_bits}")
        if t.dtype != torch.uint8 or tuple(t.shape) != (int(b.experts), 1 << 14):
            raise GrammarError("the register-direct kernel reads compose_table8's uint8 [E, 2^14] byte table")
    ext = _transcode_ext()
    out = {}
    for mode, bundles in ((0, (gate, up)), (2, (down, down))):
        ng = GROUPS[mode]
        info = [_bundle_columns(b) for b in bundles[: 2 if mode == 0 else 1]]
        colrate = info[0][1]
        e_n, cols = colrate.shape
        rows = int(bundles[0].rows)
        groups = colrate.reshape(e_n, cols // KSTEP, KSTEP)
        if not bool((groups == groups[:, :, :1]).all()):
            raise GrammarError("a 32-column k-step holds two rates; the fragment wire needs one rate per k-step")
        if mode == 0 and not torch.equal(colrate, info[1][1]):
            raise GrammarError("gate and up differ in rate; one k-step carries both")
        if rungs is not None:
            for which in ((0, 1) if mode == 0 else (2,)):
                want = torch.tensor([rungs[e][which] for e in range(e_n)], device=colrate.device)
                bad = (colrate.sum(1) * 256 != want * cols).nonzero()
                if bad.numel():
                    raise GrammarError(f"expert {int(bad[0])}: rates spend another rung than its class declares")
        grate = groups[:, :, 0]
        sorted_rate, order = torch.sort(grate, dim=1, stable=True)
        if ng == 2 and not bool((sorted_rate.reshape(e_n, -1, 2)[:, :, 0] == sorted_rate.reshape(e_n, -1, 2)[:, :, 1]).all()):
            raise GrammarError("down needs an even 32-column group count at each rate")
        slot_rate = sorted_rate[:, ::ng]
        ks = slot_rate.shape[1]
        ra, rb = slot_rate[:, 0], slot_rate[:, -1]
        if not bool(((rb - ra) <= 1).all()) or not bool(torch.isin(slot_rate, torch.tensor(SERVED_RATES, device=ra.device)).all()):
            raise GrammarError(f"the kernel serves rates {SERVED_RATES}, one or two adjacent per expert")
        ksa = torch.where(ra == rb, torch.full_like(ra, ks), (slot_rate == ra.unsqueeze(1)).sum(1))
        prof = (ra | (rb << 4) | (ksa << 8)).to(torch.int32)
        nt = -(-rows // TILE)
        wire_n = nt * (ksa * 256 * ra + (ks - ksa) * 256 * rb)
        hist_n = ksa * 8 * ra + (ks - ksa) * 8 * rb
        wire0 = torch.cumsum(wire_n, 0) - wire_n
        hist0 = torch.cumsum(hist_n, 0) - hist_n
        wire = torch.empty(int(wire_n.sum()), dtype=torch.int32, device=ra.device)
        hist = torch.empty(int(hist_n.sum()), dtype=torch.int32, device=ra.device)
        both = info if mode == 0 else info * 2
        ext.transcode([b.words_all.reshape(-1) for b in bundles],
                      [b.word_off.to(ra.device).long() for b in bundles],
                      [b.tile_words.to(ra.device).int() for b in bundles],
                      [i[0] for i in both], [i[2] for i in both], order.to(torch.int16).contiguous(), prof,
                      wire0.long(), hist0.long(), wire, hist, rows, ks, ng)
        tabs = (tables[0], tables[1]) if mode == 0 else (tables[2],)
        scales = (gate.scale_all, up.scale_all) if mode == 0 else (down.scale_all,)
        out[mode] = dict(mode=mode, wire=wire, expert_word0=wire0.long(), hist=hist, expert_hist0=hist0.long(),
                         rate=prof, kperm=order.to(torch.int16).contiguous(),
                         table=torch.stack(tabs, 1).contiguous(), wscale=torch.stack(scales, 1).float().contiguous(),
                         ks=ks)
    return out
