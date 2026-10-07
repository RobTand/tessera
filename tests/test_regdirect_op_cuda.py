"""``tessera::routed_regdirect_classes``: the register-direct MoE as one explicit opaque operation.

``opcheck`` verifies the schema, the fake kernel and that the operation mutates only the
scratch it declares (never a weight plane).  Its output does not depend on how the experts are
split into classes, and equals the same flow composed from direct launches.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for the register-direct kernel", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "regdirect_stage1"))
from fragment_synth import make_stack  # noqa: E402

from tessera import regdirect_routed as rr  # noqa: E402
from tessera import routed_fused as rf  # noqa: E402

E, TOP_K, HIDDEN, INTER, LIMIT = 6, 2, 512, 256, 10.0


def _layer(device, max_tokens):
    from tessera.serving import regdirect_op
    stacks = {0: rr.FragmentStack(**make_stack(0, E, INTER, HIDDEN // 32, [(4, 3, 8), (3, 3, 16), (4, 4, 16)] * 2, 11, device)),
              2: rr.FragmentStack(**make_stack(2, E, HIDDEN, INTER // 64, [(4, 3, 2), (3, 3, 4), (4, 4, 4)] * 2, 12, device))}
    kernel = rr.RegDirectClassKernel(E, TOP_K, device)
    parameters = {"regdirect": {m: kernel.payload(m, st, max_tokens) for m, st in stacks.items()}}
    return stacks, kernel, regdirect_op.bind(parameters, kernel, device)


def _inputs(m, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.stack([torch.randperm(E, device=device, generator=g)[:TOP_K] for _ in range(m)]).to(torch.int32)
    return (torch.randn(m, HIDDEN, device=device, generator=g) * 0.5).to(torch.bfloat16), ids, \
        torch.rand(m, TOP_K, device=device, generator=g)


def _op(bound, x, ids, w, spans):
    gu, dn, gus, dns, _resources, key = bound
    return torch.ops.tessera.routed_regdirect_classes(x, ids, w, None, gu, dn, gus, dns, [s[0] for s in spans],
                                                      [s[1] for s in spans], list(range(len(spans))), key, False, LIMIT)


@pytest.mark.parametrize("m", [1, 150])
def test_opcheck_declares_every_mutation(m):
    device = torch.device("cuda")
    _, _, bound = _layer(device, 256)
    gu, dn, gus, dns, _resources, key = bound
    x, ids, w = _inputs(m, 3, device)
    torch.library.opcheck(torch.ops.tessera.routed_regdirect_classes.default,
                          (x, ids, w, None, gu, dn, gus, dns, [0, 3], [3, 6], [0, 1], key, False, LIMIT),
                          test_utils=("test_schema", "test_faketensor"))


@pytest.mark.parametrize("m", [1, 16, 400])
def test_output_is_independent_of_the_class_split_and_equals_direct_launches(m):
    device = torch.device("cuda")
    stacks, kernel, bound = _layer(device, 400)
    x, ids, w = _inputs(m, 7 + m, device)
    one = _op(bound, x, ids, w, [(0, E)])
    three = _op(bound, x, ids, w, [(0, 2), (2, 4), (4, 6)])
    torch.cuda.synchronize()
    assert torch.equal(one, three)

    # The same flow from direct launches, without the dispatcher.
    from tessera.routed_class_dispatch import declared_route_widths
    parameters = {"regdirect": {0: tuple(bound[0]) + tuple(bound[2]), 2: tuple(bound[1]) + tuple(bound[3])}}
    routing = rf._routing_tables(ids, w, E, device, declared_route_widths(kernel, m, [0], parameters))
    xq, a1 = kernel.prepare_input(x, None, m, "e4m3", device)
    act = torch.empty((routing.routes, INTER), dtype=torch.bfloat16, device=device)
    g0 = kernel.geometry(0, m, parameters["regdirect"][0])
    stacks[0].launch(g0, xq, a1, routing.offsets, routing.flat_sorted, None, routing.superblocks(g0.superblock), 0, E,
                     act, bound[2][0], bound[2][1], top_k=TOP_K, a_row_mode=0, mul_weight=False, limit=LIMIT)
    aq, a2 = kernel.prepare_input(act, None, routing.routes, "e4m3", device)
    routed = torch.empty((routing.routes, HIDDEN), dtype=torch.bfloat16, device=device)
    g2 = kernel.geometry(2, m, parameters["regdirect"][2])
    stacks[2].launch(g2, aq, a2, routing.offsets, routing.flat_sorted, routing.rw_sorted,
                     routing.superblocks(g2.superblock), 0, E, routed, bound[3][0], bound[3][1], top_k=TOP_K,
                     a_row_mode=1, mul_weight=True, limit=float("inf"))
    want = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
    rf._ext("e4m3mma").token_sum(routed, want, TOP_K)
    torch.cuda.synchronize()
    assert one.abs().sum() > 0
    assert torch.equal(one, want)
