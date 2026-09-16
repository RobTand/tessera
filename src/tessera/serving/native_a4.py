"""Native A4 serving adapter: compact span-2 planes -> fused W4A4 compute.

The loader's ``compact_prep`` reads a verified wire container into rank-local
kernel planes without expanding the parent's weight planes; this module is the
serving seam between that bundle and ``kernel_a4``'s fused kernels.  Nothing
here decodes a stock tile: the dense route holds the compact planes and runs
the fused GEMM, and the routed route holds one stacked bundle per expert axis
and runs the grouped operator over the runtime's dispatch.

One activation contract for both: ``e2m1_group16_ue4m3_static`` through the
runtime's registered quantizer, with the static global the selected backend
uses (the FlashInfer MoE backends intentionally aggregate the routed
activation scale per layer/projection -- ``flashinfer_fp4_moe.py``'s
``is_global_sf_supported_for_nvfp4_backend`` + ``amax_for_moe_activation_quant``
-- so one scalar per GEMM is the served contract, not a simplification this
adapter invents).
"""
from __future__ import annotations

import torch

from ..compact_prep import prepare_span2_compact
from ..errors import GrammarError
from ..kernel_a4 import (A4Unit, A4UnitStack, a4_quantize_activation,
                         a4_span2_gemm, a4_span2_grouped_gemm)

__all__ = [
    "prepare_a4_unit",
    "stack_epilogues",
    "a4_dense_apply",
    "A4ExpertAxis",
    "a4_grouped_apply",
]


def prepare_a4_unit(wire, *, rows=None, cols=None, global_scale=None) -> A4Unit:
    """One rank-local role's ``A4Unit`` from a verified compact wire.

    ``wire`` is ``compact_prep.CompactWire`` (metadata verified, no parent
    expansion); ``rows``/``cols`` name the rank-local cut exactly as the
    loader's shard plan does.  ``global_scale`` overrides the wire's own
    shared global only for a fused module whose roles were moved onto one
    global by ``fused.shared_lut_global`` -- the caller passes the moved
    value, never a re-derived one.
    """
    prepared = prepare_span2_compact(wire, rows=rows, cols=cols)
    return A4Unit.from_prepared(prepared, global_scale=global_scale)


def a4_dense_apply(
    x: torch.Tensor,
    unit: A4Unit,
    input_global_scale: torch.Tensor,
    *,
    epilogue: "torch.Tensor | None" = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    """``[M, rows]`` for a dense Linear over one compact role bundle.

    ``input_global_scale`` is the checkpoint's static A-side scale (a device
    scalar); ``epilogue`` may be the prepared ``unit.epilogue_for(...)`` tensor
    so a captured forward never reads a host scalar.  The returned dtype
    matches the stock route's ``_scaled_mm`` output (bf16 by default).
    """
    if x.dim() not in (2, 3):
        raise GrammarError(
            f"a4_dense_apply takes 2-D or 3-D activations, got {tuple(x.shape)}")
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16)
    if x2.shape[1] != unit.cols:
        raise GrammarError(
            f"a4_dense_apply: activation reduces over {x2.shape[1]} columns and "
            f"the unit expects {unit.cols}")
    if epilogue is None:
        epilogue = unit.epilogue_for(input_global_scale)
    packed, scales = a4_quantize_activation(x2.contiguous(), input_global_scale)
    y = a4_span2_gemm(packed, scales, unit, epilogue, block_m=block_m,
                      block_n=block_n, block_k=block_k, out_dtype=torch.float32)
    return y.to(out_dtype).reshape(*shape[:-1], unit.rows)


class A4ExpertAxis:
    """One stacked compact bundle per plane kind over the expert axis.

    Mirrors the routed loader's intake: units are placed as their callbacks
    fire and the axis refuses heterogeneous geometry.  ``finish`` stacks once
    per plane kind -- one allocation per plane, the same shape
    ``A4UnitStack`` consumes -- and drops the per-expert unit objects.
    """

    def __init__(self, experts: int):
        if experts < 1:
            raise GrammarError("an expert axis needs at least one expert")
        self.experts = int(experts)
        self._units: "list[A4Unit | None]" = [None] * self.experts

    def put(self, expert: int, unit: A4Unit) -> None:
        if not 0 <= expert < self.experts:
            raise GrammarError(
                f"expert {expert} is outside the axis's {self.experts}")
        if self._units[expert] is not None:
            raise GrammarError(f"expert {expert} was placed twice")
        self._units[expert] = unit

    def finish(self) -> A4UnitStack:
        missing = [index for index, unit in enumerate(self._units) if unit is None]
        if missing:
            raise GrammarError(
                f"the expert axis is missing experts {missing[:8]}"
                f"{'...' if len(missing) > 8 else ''}; a partial axis cannot serve")
        stack = A4UnitStack.stack(self._units)
        self._units = []
        return stack


def a4_grouped_apply(
    x: torch.Tensor,
    stack: A4UnitStack,
    input_global_scale: torch.Tensor,
    *,
    expert_offsets: torch.Tensor,
    route_ids: torch.Tensor,
    num_routes: "int | None" = None,
    epilogues: "torch.Tensor | None" = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """One expert-scoped W4A4 GEMM per route, in dispatch order.

    ``x`` is the layer's activation for the rows the dispatch names: token
    rows for the gate/up stage, per-route rows for the down stage.
    ``route_ids`` is the row each dispatch entry reads (token id or route id);
    ``expert_offsets`` is the runtime's fixed ``[E+1]`` run table.  The result
    keeps one row per route -- no token reduction and no routing weights, which
    a routed MoE applies only in its final combine.
    """
    x2 = x.reshape(x.shape[0], -1).to(torch.bfloat16).contiguous()
    if epilogues is None:
        # One vectorized device expression, never a per-expert host loop: the
        # per-expert weight globals over the one A-side global.
        gscale = torch.as_tensor(input_global_scale, dtype=torch.float32,
                                 device=stack.globals.device).reshape(1)
        epilogues = (stack.globals / gscale).to(torch.float32).contiguous()
    packed, scales = a4_quantize_activation(x2, input_global_scale)
    return a4_span2_grouped_gemm(
        packed, scales, stack, epilogues,
        expert_offsets=expert_offsets, token_ids=route_ids,
        num_routes=num_routes, out_dtype=out_dtype)


def stack_epilogues(stack: A4UnitStack, input_global_scale: torch.Tensor) -> torch.Tensor:
    """``[E]`` fp32: every expert's weight global over the A-side global.

    Device-side and vectorized; the caller freezes it once at preparation.
    """
    gscale = torch.as_tensor(input_global_scale, dtype=torch.float32,
                             device=stack.globals.device).reshape(1)
    return (stack.globals / gscale).to(torch.float32).contiguous()
