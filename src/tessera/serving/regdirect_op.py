"""The register-direct routed MoE as one opaque Torch operation (stage 1; not selected).

``tessera::routed_regdirect_classes`` is the explicit argument list of the register-direct
kernel behind ``routed_class_dispatch``.  Every tensor the kernel reads or writes is an
argument: each mode's nine weight planes (``regdirect_routed.PAYLOAD_FIELDS`` without the
scratch) and each mode's two mutable scratch tensors, which ``mutates_args`` declares.  The
resource registry holds only streams, events and the tensor-free binding
(``routed_fused._make_dispatch_resources``); it owns no weight and no scratch.

The flow is the LUT class operation's (``serving.native_window._routed_window_classes``):
gate/up with the SwiGLU epilogue at sorted route positions, the activation quantizer, down with
the route weight at flat route ids, then the LUT library's ``token_sum``.  Today's LUT class
kernel stays the default; this operation is selected nowhere until G3 v2 and an end-to-end serve.
"""
from __future__ import annotations

from typing import List, Optional

import torch

#: The ``routed_fused`` library whose ``token_sum`` adds each token's routes (no weights).
TOKEN_SUM_LIBRARY = "e4m3mma"


@torch.library.custom_op("tessera::routed_regdirect_classes", mutates_args=("gate_up_scratch", "down_scratch"))
def routed_regdirect_classes(
    x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor, shared: Optional[torch.Tensor],
    gate_up_planes: List[torch.Tensor], down_planes: List[torch.Tensor],
    gate_up_scratch: List[torch.Tensor], down_scratch: List[torch.Tensor],
    starts: List[int], ends: List[int], issue_order: List[int],
    resource_key: str, input_weight: bool, swiglu_limit: float,
) -> torch.Tensor:
    """One routed MoE forward on the register-direct kernel.

    ``*_planes``: wire, expert_word0, hist, expert_hist0, rate, kperm, table, wscale, zeros.
    ``*_scratch``: part (fp32 K-part partials), arrive (int32 self-resetting counters).
    """
    from .. import regdirect_routed as rr
    from .. import routed_class_dispatch
    from .. import routed_fused as rf

    tokens = expert_ids.shape[0]
    resources = rf.resolve_dispatch_resources(resource_key)
    kernel = resources.kernel
    parameters = {"regdirect": {0: tuple(gate_up_planes) + tuple(gate_up_scratch),
                                2: tuple(down_planes) + tuple(down_scratch)}}
    hidden = int(down_planes[rr.PAYLOAD_FIELDS.index("wscale")].shape[2])
    inter = int(gate_up_planes[rr.PAYLOAD_FIELDS.index("wscale")].shape[2])
    if input_weight:
        x = x * routing_weights.reshape(-1, 1).to(x.dtype)
    widths = routed_class_dispatch.declared_route_widths(kernel, tokens, issue_order, parameters)
    routing = rf._routing_tables(expert_ids, routing_weights, ends[-1], x.device, widths)
    args = dict(parameters=parameters, starts=starts, ends=ends, issue_order=issue_order, counters=None,
                resources=resources)
    xq, a1 = kernel.prepare_input(x, None, tokens, "e4m3", x.device)
    act = torch.empty((routing.routes, inter), dtype=torch.bfloat16, device=x.device)
    routed_class_dispatch.dispatch_class_projection(0, xq, a1, routing, **args, a_row_mode=0, mul_weight=False,
                                                    limit=swiglu_limit, out=act)
    aq, a2 = kernel.prepare_input(act, None, routing.routes, "e4m3", x.device)
    routed = torch.empty((routing.routes, hidden), dtype=torch.bfloat16, device=x.device)
    routed_class_dispatch.dispatch_class_projection(2, aq, a2, routing, **args, a_row_mode=1,
                                                    mul_weight=not input_weight, limit=float("inf"), out=routed)
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=x.device)
    if shared is None:
        rf._ext(TOKEN_SUM_LIBRARY).token_sum(routed, out, expert_ids.shape[1])
    else:
        rf._ext(TOKEN_SUM_LIBRARY).token_sum_shared(routed, shared, out, expert_ids.shape[1])
    return out


@routed_regdirect_classes.register_fake
def _routed_regdirect_classes_fake(x, expert_ids, routing_weights, shared, gate_up_planes, down_planes,
                                   gate_up_scratch, down_scratch, starts, ends, issue_order, resource_key,
                                   input_weight, swiglu_limit):
    from .. import regdirect_routed as rr
    hidden = down_planes[rr.PAYLOAD_FIELDS.index("wscale")].shape[2]
    return torch.empty((x.shape[0], hidden), dtype=torch.bfloat16, device=x.device)


def bind(parameters: dict, kernel, device) -> tuple:
    """The operation's argument groups for one layer from :func:`regdirect_routed.build_layer`:
    ``(gate_up_planes, down_planes, gate_up_scratch, down_scratch, resources, resource_key)``.
    The caller keeps ``resources`` alive (the registry holds it weakly) for the layer's life."""
    from .. import routed_fused as rf
    payload = parameters["regdirect"]
    resources = rf._make_dispatch_resources(torch.device(device), kernel)
    return (list(payload[0][:-2]), list(payload[2][:-2]), list(payload[0][-2:]), list(payload[2][-2:]),
            resources, rf._retain_dispatch_resources(resources))
