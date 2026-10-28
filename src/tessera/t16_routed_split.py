"""Experimental one-table T16 routed execution; not a serving admission.

Each projection reads its original raw-value wire and applies FP32 row scales
after the BF16 dot. Gate/up round to BF16 before the native SwiGLU stage, which
also restores the sorted-route activation boundary. Down writes flat routes;
the existing token sum reduces them in fixed top-k order. No serving dispatcher
imports this owner. Its ceiling comes from the existing one-table value reader.
"""
from __future__ import annotations

import dataclasses

import torch

from . import routed_fused as rf
from .errors import GrammarError


def supported(gate, up, down) -> str | None:
    """Why the bounded experimental path refuses these stacks, or None."""
    if any(bundle.family != "value" for bundle in (gate, up, down)):
        return "split T16 execution requires the value family for gate, up and down"
    if any(bundle.device != down.device for bundle in (gate, up)):
        return "split T16 gate, up and down must share a CUDA device"
    return rf._routed_window_reason(gate, up, down,
                                   rate_max=rf.DENSE_RATE_MAX["value"], split=True)


@dataclasses.dataclass(frozen=True)
class _Projection:
    words: torch.Tensor
    table: torch.Tensor
    init: torch.Tensor
    has_init: torch.Tensor
    scale: torch.Tensor
    runs: torch.Tensor
    bdesc: torch.Tensor
    tile_words: int
    slot_words: int
    rows: int
    cols: int

    @classmethod
    def prepare(cls, bundle, module):
        ceiling = rf.DENSE_RATE_MAX["value"]
        runs, bdesc, tile_words, slot_words = rf.projection_tables(bundle, rate_max=ceiling)
        pair, reason = rf.run_pair(bundle.runs_all.reshape(bundle.experts, -1, 4)[0],
                                   bundle.cols, rate_max=ceiling)
        if pair is None:
            raise GrammarError(reason)
        if not module.single_projection_decodes(int(pair[0]), bool(pair[6])):
            raise GrammarError("the native one-table reader refuses this run pair")
        return cls(rf.words_by_expert(bundle), rf.compose_table16(bundle),
                   bundle.init_all, bundle.has_init, bundle.scale_all,
                   runs, bdesc, tile_words, slot_words, bundle.rows, bundle.cols)


@dataclasses.dataclass(frozen=True)
class SplitT16RoutedMoE:
    """Explicit experimental owner, never a replacement for the production lane.

    Per-forward BF16 intermediates include two flat gate/up projections and
    one sorted activation (6 * routes * intermediate bytes), plus flat down
    output and the existing routing tensors. Retained storage is enumerated
    by named_tensors; callers must measure peak allocator and native resources
    on their admitted tuple before requesting any serving promotion.
    """
    module: object
    gate: _Projection
    up: _Projection
    down: _Projection
    counter: torch.Tensor

    @classmethod
    def from_bundles(cls, gate, up, down):
        reason = supported(gate, up, down)
        if reason is not None:
            raise GrammarError(f"split T16 routed execution refuses: {reason}")
        module = rf._ext("value")
        projections = [_Projection.prepare(bundle, module) for bundle in (gate, up, down)]
        counter = torch.zeros(1, dtype=torch.int32, device=down.device)
        return cls(module, *projections, counter)

    @property
    def device(self):
        return self.down.words.device

    def named_tensors(self):
        for role in ("gate", "up", "down"):
            for field in dataclasses.fields(_Projection):
                value = getattr(getattr(self, role), field.name)
                if isinstance(value, torch.Tensor):
                    yield f"split_t16.{role}.{field.name}", value
        yield "split_t16.counter", self.counter

    @property
    def resident_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for _, tensor in self.named_tensors())

    def _project(self, projection, x, routing, *, a_row_mode, mul_weight):
        out = torch.empty((routing.routes, projection.rows), dtype=torch.bfloat16, device=self.device)
        self.counter.zero_()
        self.module.routed_single_forward(
            x, projection.words, projection.table, projection.init, projection.has_init,
            projection.scale, projection.runs, projection.bdesc,
            projection.tile_words, projection.slot_words, routing.offsets,
            routing.flat_sorted, routing.rw_sorted, routing.prefixes[rf.BM], self.counter,
            routing.top_k, a_row_mode, mul_weight, out, rf._sm_count(self.device.index), rf.BM)
        return out

    def __call__(self, x, expert_ids, routing_weights, *, apply_router_weight_on_input=False,
                 swiglu_limit=None):
        from .native_window_moe import checked_swiglu_limit

        limit = checked_swiglu_limit(swiglu_limit)
        if expert_ids.dim() != 2 or expert_ids.shape != routing_weights.shape:
            raise GrammarError("expert_ids and routing_weights must share [T, top_k]")
        tokens, top_k = expert_ids.shape
        if top_k < 1 or (apply_router_weight_on_input and top_k != 1):
            raise GrammarError("input routing weight requires top_k=1; top_k must be positive")
        if x.dtype != torch.bfloat16 or x.device != self.device or tuple(x.shape) != (tokens, self.gate.cols):
            raise GrammarError(f"x must be bf16 [{tokens}, {self.gate.cols}] on {self.device}")
        routing = rf._routing_tables(expert_ids, routing_weights, self.down.words.shape[0],
                                     self.device, (rf.BM,))
        if tokens == 0:
            return torch.empty((0, self.down.rows), dtype=torch.bfloat16, device=self.device)
        if apply_router_weight_on_input:
            x = (x.float() * routing_weights[:, :1].float()).to(torch.bfloat16)
        x = x.contiguous()
        gate = self._project(self.gate, x, routing, a_row_mode=0, mul_weight=False)
        up = self._project(self.up, x, routing, a_row_mode=0, mul_weight=False)
        activation = torch.empty_like(gate)
        self.module.split_swiglu(gate, up, routing.flat_sorted,
                                  float("inf") if limit is None else limit, activation)
        routed = self._project(self.down, activation, routing, a_row_mode=1,
                               mul_weight=not apply_router_weight_on_input)
        out = torch.empty((tokens, self.down.rows), dtype=torch.bfloat16, device=self.device)
        self.module.token_sum(routed, out, top_k)
        return out
