"""The native window MoE adapter: two grouped window GEMMs around the
activation, for the FP8 and BF16 routed-expert families.

WHAT IT IS.  ``PreparedGroupedWindowGemm`` (``window_gemm_grouped``) computes
one projection under a token->expert routing.  A routed MoE layer is two of
them -- gate/up, then down -- with the activation between, and that is what
this module owns:

* ``gate/up`` runs in ``preserve=True`` mode: per-route outputs
  ``[T, top_k, 2I]`` (or two ``[T, top_k, I]`` stacks if gate and up are
  separate).  The routing weight multiplies this projection's fp32
  accumulator iff ``apply_router_weight_on_input``, exactly as vLLM dispatches
  gemm1 (``fused_moe.py``).
* the activation is applied per route.  Only ``silu`` is served;
  ``swiglu_alpha``/``swiglu_beta``/``swiglu_limit``/another activation **fail
  closed** until an implementation reproduces vLLM's exact arithmetic -- no
  silent substitution.
* ``down`` runs with ``route_input=True``: it consumes the per-route
  activations, and weights each route's fp32 accumulator iff the input did
  not already carry the weights (vLLM's gemm2 / ``topk_weight_and_reduce``
  placement), summing over ``top_k``.

ARITHMETIC.  The weight-side contract is the grouped bundle's and is explicit
at preparation: ``"epilogue"`` (dense BF16 row scale on the fp32 accumulator,
or the FP8 ``acc * a_scale * w_scale`` contract) or ``"folded"`` (the
research BF16 contract, one bf16 rounding of ``value * row_scale`` in
registers before ``tl.dot``).  The adapter inherits it from the bundles; the
FP8 family has no folded form.

WHAT REMAINS THE OWNER'S.  Routing (top-k ids and weights), shared experts,
the TP all-reduce and the final scale/dtype presentation are the serving
layer's; this adapter returns the routed-expert result ``[T, rows]`` bf16.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import torch

from .errors import GrammarError
from .kernel_window_gemv import WindowGemvUnit
from .window_gemm_grouped import PreparedGroupedWindowGemm, prepare_grouped_window_gemm

__all__ = ["NativeWindowMoE", "prepare_native_window_moe", "PackedWindowUnits",
           "SUPPORTED_ACTIVATIONS"]

#: The activations this adapter reproduces exactly.  Everything else refuses.
SUPPORTED_ACTIVATIONS = ("silu",)


@dataclasses.dataclass(frozen=True)
class PackedWindowUnits:
    """Rank-local per-expert units in GLOBAL expert order, one entry per
    expert: what a loader hands the adapter without a second weight walk."""

    gate: tuple
    up: tuple
    down: tuple
    family: str

    @property
    def experts(self) -> int:
        return len(self.down)

    def resident_bytes(self) -> int:
        total = 0
        for unit in (*self.gate, *self.up, *self.down):
            rep = unit.rep
            total += rep.words.numel() * rep.words.element_size()
            for t in (unit.table, unit.scale, unit.initial_state, unit.codes_of_state, unit.native):
                if t is not None:
                    total += t.numel() * t.element_size()
        return total

    def prepare(self, *, block_m: int = 64, block_n: int = 64, block_k: int = 64,
                arithmetic: "str | None" = None, activation: str = "silu",
                quantizer: "str | None" = "native") -> "NativeWindowMoE":
        """Build the adapter.  The default arithmetic is each family's
        published contract: folded for the research BF16 wire, epilogue for
        the FP8 wire.  An explicit value must match the family's served
        contract; nothing here silently swaps them."""
        if arithmetic is None:
            arithmetic = "folded" if self.family == "value" else "epilogue"
        up = list(self.up) if self.up else None
        return prepare_native_window_moe(
            list(self.gate), list(self.down), up=up,
            block_m=block_m, block_n=block_n, block_k=block_k,
            quantizer=quantizer, arithmetic=arithmetic, activation=activation)


@dataclasses.dataclass(frozen=True)
class NativeWindowMoE:
    """Two grouped projections around the activation.  ``gate_up`` is the
    fused ``[2I]`` stack; ``gate``/``up`` are the separate-stack spelling."""

    gate_up: "PreparedGroupedWindowGemm | None"
    gate: "PreparedGroupedWindowGemm | None"
    up: "PreparedGroupedWindowGemm | None"
    down: PreparedGroupedWindowGemm
    activation: str = "silu"

    def __call__(self, x: torch.Tensor,
                 expert_ids: torch.Tensor,
                 routing_weights: torch.Tensor,
                 *,
                 apply_router_weight_on_input: bool = False) -> torch.Tensor:
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise GrammarError(
                f"the native window MoE serves {SUPPORTED_ACTIVATIONS}; "
                f"activation {self.activation!r} has no exact implementation here "
                "and refusing beats approximating vLLM's arithmetic"
            )
        inter = self.down.cols
        if self.gate_up is not None:
            route = self.gate_up(x, expert_ids, routing_weights, preserve=True,
                                 apply_router_weight_on_input=apply_router_weight_on_input)
            if route.shape[-1] != 2 * inter:
                raise GrammarError(
                    f"the fused gate/up stack produces {route.shape[-1]} rows, "
                    f"the down stack consumes {inter} per half"
                )
            gate, up = route[..., :inter], route[..., inter:]
        else:
            gate = self.gate(x, expert_ids, routing_weights, preserve=True,
                             apply_router_weight_on_input=apply_router_weight_on_input)
            up = self.up(x, expert_ids, routing_weights, preserve=True,
                         apply_router_weight_on_input=apply_router_weight_on_input)
        act = _silu_and_mul(gate, up)
        t_tokens, top_k, _ = act.shape
        return self.down(act.reshape(t_tokens * top_k, inter), expert_ids, routing_weights,
                         route_input=True,
                         apply_router_weight_on_input=apply_router_weight_on_input,
                         round_routes=True)


def _silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` with one bf16 rounding, the same arithmetic vLLM's
    ``SiluAndMul`` performs on the gemm output (fp32 activation, cast once)."""
    if gate.shape != up.shape:
        raise GrammarError(f"gate {tuple(gate.shape)} and up {tuple(up.shape)} must match")
    act = torch.nn.functional.silu(gate.float()) * up.float()
    return act.to(torch.bfloat16)


def prepare_native_window_moe(
    gate: Sequence[WindowGemvUnit],
    down: Sequence[WindowGemvUnit],
    *,
    up: "Sequence[WindowGemvUnit] | None" = None,
    initial_state: "torch.Tensor | None" = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    quantizer: "str | None" = "native",
    arithmetic: str = "epilogue",
    activation: str = "silu",
) -> NativeWindowMoE:
    """Prepare both stages once.

    ``gate`` is the fused gate/up stack (``rows == 2 * down.cols``) unless
    ``up`` is given, in which case ``gate`` and ``up`` are separate stacks of
    ``down.cols`` rows each.  ``arithmetic`` is the grouped bundles' explicit
    weight-side contract (``"epilogue"`` default, ``"folded"`` for the
    research BF16 contract); the down stack must agree with the gate/up
    stacks.  ``activation`` must be in :data:`SUPPORTED_ACTIVATIONS`.
    """
    if activation not in SUPPORTED_ACTIVATIONS:
        raise GrammarError(
            f"activation {activation!r} is not served; supported: {SUPPORTED_ACTIVATIONS}"
        )
    if up is None:
        gate_up = prepare_grouped_window_gemm(
            gate, initial_state=initial_state, block_m=block_m, block_n=block_n,
            block_k=block_k, quantizer=quantizer, arithmetic=arithmetic)
        gate_bundle = up_bundle = None
    else:
        gate_up = None
        gate_bundle = prepare_grouped_window_gemm(
            gate, initial_state=initial_state, block_m=block_m, block_n=block_n,
            block_k=block_k, quantizer=quantizer, arithmetic=arithmetic)
        up_bundle = prepare_grouped_window_gemm(
            up, initial_state=initial_state, block_m=block_m, block_n=block_n,
            block_k=block_k, quantizer=quantizer, arithmetic=arithmetic)
    down_bundle = prepare_grouped_window_gemm(
        down, initial_state=initial_state, block_m=block_m, block_n=block_n,
        block_k=block_k, quantizer=quantizer, arithmetic=arithmetic)
    first = gate_up or gate_bundle
    if first.family != down_bundle.family:
        raise GrammarError(
            f"gate/up family {first.family!r} and down family {down_bundle.family!r} "
            "must agree; a mixed-family MoE has no single arithmetic contract"
        )
    if first.arithmetic != down_bundle.arithmetic:
        raise GrammarError(
            "the gate/up and down stacks must share one weight arithmetic "
            f"({first.arithmetic!r} vs {down_bundle.arithmetic!r})"
        )
    if first.experts != down_bundle.experts:
        raise GrammarError(
            f"gate/up has {first.experts} experts, down has {down_bundle.experts}"
        )
    for bundle in (gate_bundle, up_bundle):
        if bundle is not None and bundle.experts != down_bundle.experts:
            raise GrammarError(
                f"a gate/up stack has {bundle.experts} experts, down has "
                f"{down_bundle.experts}"
            )
    return NativeWindowMoE(gate_up=gate_up, gate=gate_bundle, up=up_bundle,
                           down=down_bundle, activation=activation)
