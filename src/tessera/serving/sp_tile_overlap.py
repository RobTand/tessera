"""Uninstalled SP mHC pipeline primitive (#858, children of #783/#803).

This module does not select a communicator, rebind a model, or enable a serve.
An inspected caller must supply the same TP2 communicator and explicit stock
split-k mHC kernels as its serialized path. Real NCCL exactness and usefulness
remain GPU/served gates. CPU fixtures cover ownership and the stream protocol.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import threading
from typing import Any, Callable

from .stock_interface import InspectedInterface, import_modules, match_modules, stock_attribute


# Reuse the communicator read set inspected for #804; the route below reads
# the same CUDA/PyNccl owners for SP's dim-0 gather/reduce-scatter branches.
COMM_MODULES = (
    "vllm.distributed.device_communicators.cuda_communicator",
    "vllm.distributed.device_communicators.all_reduce_utils",
    "vllm.distributed.device_communicators.pynccl",
    "vllm.distributed.parallel_state",
    "vllm.utils.torch_utils",
)
COMM_INTERFACES = (InspectedInterface("nightly-20260929", (
    "2a0695d8b46757be83b38fe657df605ba0bfb2ef2dfa690c78c57419bbad0762",
    "b41e3ab17e21d81c7516cbef29850b0096ad3efd73d23538300af652ffce42d6",
    "4c51becc2ebfd41910526b93d7bd8e9e36312df444fdfb55fa2fbc92b0f8f88e",
    "a33fc846e0f4e682a644ce712d862806ecfa12dca6e5f641806e2e7991cd0087",
    "eab9ea0a3d3b9792fd7e3076cca1a3484031b22ce662baa7709cb5a68dc2115f",
)),)


def inspected_communication() -> tuple[Any, str]:
    modules, why = import_modules(COMM_MODULES)
    if modules is None:
        return None, why
    interface, why = match_modules(modules, COMM_MODULES, COMM_INTERFACES)
    return (modules, "") if interface is not None else (None, why)


def pynccl_sp_decline(dc: Any, *, symmetric_ag_rs: bool, rocm: bool = False) -> str | None:
    """Conservatively require stock dim-0 SP to use this active PyNccl owner.

    Stock first tries ca_comm's custom SP operations. Its CUDA communicator's
    ordinary gather/scatter branches also select symmetric memory, and gather
    selects the base class on ROCm. Unknown facts decline before any enqueue.
    """
    if dc is None or stock_attribute(dc, "world_size") != 2:
        return "no TP2 CUDA communicator"
    if stock_attribute(dc, "ca_comm", object()) is not None:
        return "custom SP communicator present or unknown"
    if type(symmetric_ag_rs) is not bool or symmetric_ag_rs:
        return "symmetric-memory SP enabled or unknown"
    if type(rocm) is not bool or rocm:
        return "ROCm/base-class gather selected or unknown"
    nccl = stock_attribute(dc, "pynccl_comm")
    if (nccl is None or stock_attribute(nccl, "disabled") is not False
            or stock_attribute(nccl, "world_size") != 2):
        return "no active TP2 pynccl communicator"
    if any(not callable(stock_attribute(nccl, name)) for name in ("reduce_scatter", "all_gather")):
        return "pynccl SP methods unavailable"
    return None


@dataclass(frozen=True)
class SpTile:
    global_begin: int
    global_end: int
    local_begin: int
    local_end: int
    padded_end: int


@dataclass(frozen=True)
class SpTilePlan:
    """Tile-major ownership: rank r owns half r of every global tile.

    Only the final global tile may be padded. The gather writes each tile to
    its original global interval; attention and the MLP see full token order.
    """

    tokens: int
    tile: int
    tp_size: int = 2

    def __post_init__(self):
        for name in ("tokens", "tile", "tp_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.tp_size != 2:
            raise ValueError("SP tile overlap requires TP2")
        if self.tile % self.tp_size:
            raise ValueError("global tile must be divisible by TP size")

    @property
    def padded_tokens(self) -> int:
        return self.tokens + (-self.tokens) % self.tp_size

    @property
    def local_tokens(self) -> int:
        return self.padded_tokens // self.tp_size

    @property
    def tiles(self) -> tuple[SpTile, ...]:
        return tuple(SpTile(a, min(a + self.tile, self.tokens), a // self.tp_size,
                            min(a + self.tile, self.padded_tokens) // self.tp_size,
                            min(a + self.tile, self.padded_tokens))
                     for a in range(0, self.tokens, self.tile))

    def pad_global(self, x: Any, torch: Any) -> Any:
        if x.shape[0] != self.tokens:
            raise ValueError("global input token count does not match plan")
        padding = self.padded_tokens - self.tokens
        if not padding:
            return x
        return torch.cat((x, x.new_zeros((padding, *x.shape[1:]))), dim=0)

    def shard(self, x: Any, rank: int, torch: Any) -> Any:
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < self.tp_size:
            raise ValueError("rank must be 0 or 1")
        padded = self.pad_global(x, torch)
        pieces = []
        for tile in self.tiles:
            half = tile.local_end - tile.local_begin
            a = tile.global_begin + rank * half
            pieces.append(padded[a:a + half])
        return torch.cat(pieces, dim=0).contiguous()


@dataclass(frozen=True)
class SiteOutputs:
    residual: Any
    post: Any
    comb: Any
    layer_input: Any

    def slice(self, a: int, b: int) -> SiteOutputs:
        return SiteOutputs(*(getattr(self, name)[a:b]
                             for name in ("residual", "post", "comb", "layer_input")))


def stock_post_pre_into(k: Any, *, fn: Any, scale: Any, base: Any,
                        rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float,
                        hc_post_mult_value: float, sinkhorn_repeat: int,
                        norm_weight: Any = None, norm_eps: float = 1e-6) -> Callable:
    """Adapt the three explicit stock split-k kernels to output slices.

    ``k.post/gemm/pre`` are the pinned post, prenorm-GEMM and pre kernels.
    The runner supplies SplitForcer.full_batch around every tile. Calling the
    high-level stock op here would let a short tail select its fused-small
    kernel, whose split cannot be forced. No projection arithmetic is changed.
    """
    if norm_weight is not None:
        if norm_weight.dtype != k.torch.bfloat16:
            norm_weight = norm_weight.to(k.torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    def into(x: Any, residual: Any, post: Any, comb: Any, out: SiteOutputs) -> None:
        tokens, hc_mult, hidden = residual.shape
        k.post(comb.view(tokens, hc_mult, hc_mult), residual,
               post.view(tokens, hc_mult), x, out.residual, hc_mult, hidden)
        mul, sqrsum = k.gemm(out.residual.view(tokens, hc_mult * hidden), fn,
                             hidden_size=hidden, hc_mult=hc_mult)
        k.pre(mul, sqrsum, scale, base, out.residual, out.post.view(tokens, hc_mult),
              out.comb.view(tokens, hc_mult * hc_mult), out.layer_input,
              rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
              norm_weight=norm_weight, norm_eps=norm_eps)

    return into


class SpTileOverlap:
    """One communication stream: all RS tiles, then ordered AG tiles.

    Compute waits for RS_i, executes mHC_i and records M_i. The side stream
    waits for M_i before AG_i; RS_(i+1) can overlap mHC_i. A final compute wait
    retires every side-stream collective before later communicator work.
    Allocation registration precedes any enqueue. Errors join the side stream
    even when a partially enqueued collective has no completion event (#847).
    The stream/event/keep interface follows TileOverlap in the #804/#847 stack.
    """

    def __init__(self, *, torch: Any, reduce_scatter_into: Callable,
                 all_gather_into: Callable, side: Any, compute: Callable,
                 new_event: Callable, keep: Callable, full_split: Callable,
                 capturing: Callable):
        self.torch = torch
        self.reduce_scatter_into, self.all_gather_into = reduce_scatter_into, all_gather_into
        self.side, self.compute = side, compute
        self.new_event, self.keep = new_event, keep
        self.full_split, self.capturing = full_split, capturing
        self._events: list[Any] = []
        self._lock = threading.Lock()

    def _event(self, i: int) -> Any:
        while len(self._events) <= i:
            self._events.append(self.new_event())
        return self._events[i]

    @contextlib.contextmanager
    def _retained_side(self, owners: tuple[Any, ...], compute: Any):
        # No side work is issued before every owner is registered. This also
        # covers a registration failure without leaving partial side work.
        for owner in owners:
            self.keep(owner, self.side)
        try:
            ready = self._event(0)
            ready.record(compute)
            self.side.wait_event(ready)
            yield
        except BaseException:
            self.side.synchronize()
            raise

    @contextlib.contextmanager
    def _call(self):
        if self.capturing():
            raise RuntimeError("SP tile overlap does not support graph capture")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("SP tile overlap call is already in flight")
        try:
            yield self.compute()
        finally:
            self._lock.release()

    def gather(self, plan: SpTilePlan, local: Any) -> Any:
        """Reconstruct initial-pre or terminal-post output in global token order."""
        if local.shape[0] != plan.local_tokens or not local.is_contiguous():
            raise ValueError("local gather input does not match tile plan")
        with self._call() as compute:
            out = local.new_empty((plan.padded_tokens, *local.shape[1:]))
            with self._retained_side((local, out), compute):
                for tile in plan.tiles:
                    self.all_gather_into(out[tile.global_begin:tile.padded_end],
                                         local[tile.local_begin:tile.local_end], self.side)
                done = self._event(1)
                done.record(self.side)
                compute.wait_event(done)
            return out[:plan.tokens]

    def run(self, plan: SpTilePlan, x: Any, residual: Any, post: Any, comb: Any,
            mhc_into: Callable) -> SiteOutputs:
        """Consume unreduced global x plus tile-owned state; return tile state/full input.

        The explicit stock adapter writes every output slice in place, avoiding
        a separate residual cache or concatenation of mHC outputs. The caller
        must use plan.shard at model entry and this runner's gather at the terminal site
        for the entire pass; a contiguous stock rank shard is incompatible.
        """
        torch = self.torch
        if x.ndim != 2 or residual.ndim != 3:
            raise ValueError("expected global [T,H] and local [T/2,n,H] tensors")
        hc_mult, hidden = residual.shape[1:]
        if (x.shape != (plan.tokens, hidden) or hc_mult <= 0
                or residual.shape != (plan.local_tokens, hc_mult, hidden)
                or post.shape != (plan.local_tokens, hc_mult, 1)
                or comb.shape != (plan.local_tokens, hc_mult, hc_mult)):
            raise ValueError("mHC tensor shapes do not match tile plan")
        if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16 \
                or post.dtype != torch.float32 or comb.dtype != torch.float32:
            raise ValueError("mHC requires BF16 inputs and FP32 mixes")
        if any(t.device != x.device or not t.is_contiguous() for t in (x, residual, post, comb)):
            raise ValueError("mHC inputs must be contiguous on one device")
        with self._call() as compute:
            packed_x = plan.pad_global(x, torch)
            reduced = x.new_empty((plan.local_tokens, hidden))
            local = SiteOutputs(torch.empty_like(residual), torch.empty_like(post),
                                torch.empty_like(comb), torch.empty_like(reduced))
            gathered = torch.empty_like(packed_x)
            tiles = plan.tiles
            owners = (x, packed_x, reduced, local.layer_input, gathered)
            with self._retained_side(owners, compute):
                for i, tile in enumerate(tiles):
                    self.reduce_scatter_into(reduced[tile.local_begin:tile.local_end],
                                             packed_x[tile.global_begin:tile.padded_end], self.side)
                    self._event(1 + i).record(self.side)
                with self.full_split(plan.tokens):
                    for i, tile in enumerate(tiles):
                        a, b = tile.local_begin, tile.local_end
                        compute.wait_event(self._event(1 + i))
                        mhc_into(reduced[a:b], residual[a:b], post[a:b], comb[a:b], local.slice(a, b))
                        done = self._event(1 + len(tiles) + i)
                        done.record(compute)
                        self.side.wait_event(done)
                        self.all_gather_into(gathered[tile.global_begin:tile.padded_end],
                                             local.layer_input[a:b], self.side)
                done = self._event(1 + 2 * len(tiles))
                done.record(self.side)
                compute.wait_event(done)
            return SiteOutputs(local.residual, local.post, local.comb, gathered[:plan.tokens])


def make_pynccl_runner(torch: Any, nccl: Any, full_split: Callable) -> SpTileOverlap:
    """Bind an already-inspected active stock PyNccl owner; do not select a route.

    PyNccl.reduce_scatter's third positional argument is the reduction op,
    whereas all_gather's is the stream. Bind stream by name for both methods.
    """
    return SpTileOverlap(torch=torch,
        reduce_scatter_into=lambda out, x, stream: nccl.reduce_scatter(out, x, stream=stream),
        all_gather_into=lambda out, x, stream: nccl.all_gather(out, x, stream=stream),
        side=torch.cuda.Stream(), compute=torch.cuda.current_stream,
        new_event=torch.cuda.Event, keep=lambda t, s: t.record_stream(s),
        full_split=full_split, capturing=torch.cuda.is_current_stream_capturing)
