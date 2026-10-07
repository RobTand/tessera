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
SUPERBLOCK = {False: 8, True: 64}     # routes per superblock: decode, prefill
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
    prefill: bool
    superblock: int          # routes per superblock item
    k_parts: int             # P
    tiles: int               # NT: 128-row tiles of the projection
    grid: int

    @property
    def units_per_item(self) -> int:
        """Work units per superblock item: the factor of the front end's interval rule."""
        return self.tiles * self.k_parts


def is_prefill(m: int, top_k: int, experts: int) -> bool:
    """The prefill path (64-route superblocks, staged activations) once an expert averages more
    routes than one decode superblock holds; below that, the decode path's 8-route items."""
    return m * top_k > SUPERBLOCK[False] * experts


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
    prefill = is_prefill(m, top_k, experts)
    tiles = rows // TILE
    grid = sms * _ext().blocks_per_sm(mode, prefill, False)
    items = min(experts, m * top_k)              # the balanced estimate; the device sees the real count
    parts = 1 if prefill else k_parts(items, tiles, grid, ks)
    return Geometry(prefill, SUPERBLOCK[prefill], parts, tiles, grid)


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
        _ext().forward(self.mode, geom.prefill, dump is not None, self.wire, self.expert_word0, self.hist,
                       self.expert_hist0, self.rate, self.kperm, self.table, self.wscale, x_zero_row, a_scale,
                       offsets, sorted_routes, rw, item_off, e0, e1, out, part, arrive, dbuf, self.zeros,
                       self.ks, top_k, geom.k_parts, a_row_mode, mul_weight, limit, hint, geom.grid)
