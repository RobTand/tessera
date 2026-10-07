"""The register-direct binding behind ``routed_class_dispatch`` equals one direct launch.

Three storage classes on two streams must write the same bytes as one launch over every
expert: the binding passes absolute expert bounds, the full superblock prefix and scratch
indexed by absolute work unit.  A captured graph replays with new routing in place.
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
from tessera import routed_class_dispatch as rcd  # noqa: E402
from tessera.routed_fused import _item_off, _Routing  # noqa: E402

E, TOP_K, HIDDEN, INTER = 6, 2, 512, 256
CLASSES = ((0, 2), (2, 4), (4, 6))
LIMIT = 10.0


class _AnyWidthRouting(_Routing):
    """Superblock prefixes at any width the bound kernel declares (8, 64, 128 routes)."""

    def superblocks(self, bm):
        return _item_off(self.offsets[1:] - self.offsets[:-1], int(bm))


def _routing(ids, weights):
    flat = ids.reshape(-1).to(torch.int64)
    counts = torch.zeros(E, dtype=torch.int32, device=ids.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=ids.device)
    offsets[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    order = torch.argsort(flat, stable=True)
    return _AnyWidthRouting(offsets=offsets, flat_sorted=order.to(torch.int32).contiguous(),
                            rw_sorted=weights.reshape(-1).float()[order].contiguous(),
                            item_off=_item_off(counts, 64), tokens=ids.shape[0], top_k=TOP_K)


def _stacks(device):
    gate_up = make_stack(0, E, INTER, HIDDEN // 32, [(4, 3, 8), (3, 3, 16), (4, 4, 16)] * 2, 11, device)
    down = make_stack(2, E, HIDDEN, INTER // 64, [(4, 3, 2), (3, 3, 4), (4, 4, 4)] * 2, 12, device)
    return {0: rr.FragmentStack(**gate_up), 2: rr.FragmentStack(**down)}


def _inputs(m, mode, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    rows, k = (m, HIDDEN) if mode == 0 else (m * TOP_K, INTER)
    x = (torch.randn(rows, k, device=device, generator=g) * 0.5).to(torch.float8_e4m3fn)
    return x, torch.rand(rows, device=device, generator=g) * 0.1 + 0.01


def _ids(m, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.stack([torch.randperm(E, device=device, generator=g)[:TOP_K] for _ in range(m)]).to(torch.int32)
    return ids, torch.rand(m, TOP_K, device=device, generator=g)


class _Resources:
    def __init__(self, kernel, device):
        self.kernel = kernel
        self.streams = tuple(torch.cuda.Stream(device=device) for _ in range(2))
        self.ready = torch.cuda.Event()
        self.finished = tuple(torch.cuda.Event() for _ in range(2))
        self.empty = torch.empty(0, dtype=torch.float32, device=device)


def _run(kernel, resources, parameters, mode, x, a_scale, routing, out, *, classes):
    rows_out = out.shape[0]
    xz, scale = kernel.prepare_input(x, a_scale, x.shape[0], "e4m3", x.device)
    starts, ends = [c[0] for c in classes], [c[1] for c in classes]
    rcd.dispatch_class_projection(mode, xz, scale, routing, parameters=parameters, starts=starts, ends=ends,
                                  issue_order=list(range(len(classes))), counters=None, resources=resources,
                                  mul_weight=mode == 2, limit=LIMIT, a_row_mode=0 if mode == 0 else 1, out=out)
    assert out.shape[0] == rows_out


@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("m", [1, 150, 400])        # route tiles 1 (decode), 8 and 16
def test_class_dispatch_equals_one_launch(mode, m):
    device = torch.device("cuda")
    parameters = {"regdirect": _stacks(device)}
    kernel = rr.RegDirectClassKernel(E, TOP_K, device)
    kernel.reserve(parameters, 400)
    resources = _Resources(kernel, device)
    stack = parameters["regdirect"][mode]
    ids, w = _ids(m, 5 + m, device)
    routing = _routing(ids, w)
    x, a_scale = _inputs(m, mode, 7 + m, device)
    outs = []
    for classes in (((0, E),), CLASSES):
        out = torch.zeros(m * TOP_K, stack.rows, dtype=torch.bfloat16, device=device)
        _run(kernel, resources, parameters, mode, x, a_scale, routing, out, classes=classes)
        torch.cuda.synchronize()
        outs.append(out)
    assert outs[0].abs().sum() > 0
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("m", [1, 16])
def test_graph_replay_follows_new_routing(mode, m):
    device = torch.device("cuda")
    parameters = {"regdirect": _stacks(device)}
    kernel = rr.RegDirectClassKernel(E, TOP_K, device)
    kernel.reserve(parameters, 64)
    resources = _Resources(kernel, device)
    stack = parameters["regdirect"][mode]
    ids, w = _ids(m, 21, device)
    routing = _routing(ids, w)
    x, a_scale = _inputs(m, mode, 22, device)
    out = torch.zeros(m * TOP_K, stack.rows, dtype=torch.bfloat16, device=device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _run(kernel, resources, parameters, mode, x, a_scale, routing, out, classes=CLASSES)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(kernel, resources, parameters, mode, x, a_scale, routing, out, classes=CLASSES)
    for seed in (31, 32, 33):
        ids2, w2 = _ids(m, seed, device)
        fresh = _routing(ids2, w2)
        for name in ("offsets", "flat_sorted", "rw_sorted"):
            getattr(routing, name).copy_(getattr(fresh, name))
        x2, s2 = _inputs(m, mode, seed + 100, device)
        x.copy_(x2)
        a_scale.copy_(s2)
        graph.replay()
        want = torch.zeros_like(out)
        _run(kernel, resources, parameters, mode, x, a_scale, fresh, want, classes=CLASSES)
        torch.cuda.synchronize()
        assert torch.equal(out, want), f"replay with routing seed {seed} differs from eager"
