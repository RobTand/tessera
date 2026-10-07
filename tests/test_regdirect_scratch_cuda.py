"""``reserve_scratch`` visits only the token counts that change the geometry, and reserves exactly
what visiting every token count reserves (the serve-load cost of the layer build)."""
from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("the launch geometry reads the kernel's occupancy (CUDA)", allow_module_level=True)

from tessera import regdirect_routed as rr  # noqa: E402


@pytest.mark.parametrize("mode, rows, ks, experts, top_k, max_tokens", [
    (0, 1024, 128, 288, 8, 2048), (2, 4096, 16, 288, 8, 2048), (0, 256, 16, 6, 2, 400), (2, 512, 4, 6, 2, 64)])
def test_the_reduced_token_set_reserves_what_every_token_count_reserves(mode, rows, ks, experts, top_k, max_tokens):
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    routes = max_tokens * top_k
    full = [0, 0]
    for m in range(1, max_tokens + 1):
        g = rr.geometry(mode, m, top_k, rows, experts, ks, sms)
        sizes = [t.numel() for t in rr.scratch(g, experts, routes, "meta")]
        full = [max(a, b) for a, b in zip(full, sizes)]
    tokens = rr._geometry_tokens(top_k, experts, max_tokens)
    assert len(tokens) < max_tokens
    part, arrive = rr.reserve_scratch(mode, rows, ks, experts, top_k, max_tokens, sms, "cuda")
    assert [part.numel(), arrive.numel()] == full
