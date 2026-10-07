"""The register-direct kernel decodes T-8 rates 3 to 8 bitwise (stage 2: R1280 to R2048).

The kernel's decode dump (the E4M3 bytes each warp forms for its MMA A fragment) must equal
``fragment_synth.reference_decode`` of the same fragment planes, for every pure rate and every
adjacent mixed pair, in both modes, on the decode path (one route tile) and on prefill.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for the register-direct kernel", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "regdirect_stage1"))
from fragment_synth import TABLES, make_stack, reference_decode  # noqa: E402

from tessera import regdirect_routed as rr  # noqa: E402
from tessera.routed_fused import _item_off  # noqa: E402

E, TOP_K = 4, 2
SHAPES = {0: (256, 512), 2: (512, 256)}            # (rows N, input K) per expert
PROFILES = [(r, r, None) for r in range(3, 9)] + [(r + 1, r, "half") for r in range(3, 8)]


def _routing(m, device):
    ids = (torch.arange(m * TOP_K, device=device) % E).reshape(m, TOP_K).to(torch.int32)
    flat = ids.reshape(-1).long()
    counts = torch.bincount(flat, minlength=E).to(torch.int32)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, 0)
    return ids, counts, offsets, torch.argsort(flat, stable=True).to(torch.int32).contiguous()


@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: f"R{p[0]}-{p[1]}")
@pytest.mark.parametrize("m", [1, 300])                  # the decode path and a prefill route tile
def test_the_kernel_decodes_each_rate_bitwise(mode, profile, m):
    device = torch.device("cuda", torch.cuda.current_device())
    rows, k = SHAPES[mode]
    ks = k // (rr.GROUPS[mode] * rr.KSTEP)
    ra, rb, half = profile
    planes = make_stack(mode, E, rows, ks, [(ra, rb, ks if half is None else ks // 2)] * E, 17 + ra + rb, device)
    stack = rr.FragmentStack(**planes)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    g = rr.geometry(mode, m, TOP_K, rows, E, ks, sms)
    g = rr.Geometry(g.route_tiles, g.superblock, g.k_parts, g.tiles, sms * rr._ext().blocks_per_sm(mode, g.route_tiles, True))
    ids, counts, offsets, order = _routing(m, device)
    item_off = _item_off(counts, g.superblock)
    routes = m * TOP_K
    xrows = m if mode == 0 else routes
    x = rr.with_zero_row((torch.randn(xrows, k, device=device) * 0.5).to(torch.float8_e4m3fn))
    a_scale = torch.rand(xrows, device=device) * 0.1 + 0.01
    part, arrive = rr.scratch(g, E, routes, device)
    out = torch.zeros(routes, rows, dtype=torch.bfloat16, device=device)
    dump = torch.zeros(E, TABLES[mode], rows, k, dtype=torch.uint8, device=device)
    rw = torch.full((routes,), 0.5, device=device) if mode == 2 else None
    stack.launch(g, x, a_scale, offsets, order, rw, item_off, 0, E, out, part, arrive, top_k=TOP_K,
                 a_row_mode=0 if mode == 0 else 1, mul_weight=mode == 2, limit=10.0, dump=dump)
    torch.cuda.synchronize()
    routed = [e for e in range(E) if int(counts[e]) > 0]     # the kernel dumps only the experts it runs
    assert routed, "the routing reaches no expert"
    for e in routed:
        want = reference_decode(planes, e)
        assert torch.equal(dump[e], want), f"expert {e}: {(dump[e] != want).sum().item()} bytes differ"
