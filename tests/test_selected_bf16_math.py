"""Selected BF16 dots preserve route contributions and BF16 stage boundaries."""
from types import SimpleNamespace

import pytest
import torch

from tessera.serving.moe_route import (
    PreparedTesseraSelectedBf16MoeExperts,
    _selected_bf16_apply,
)


@pytest.mark.parametrize("top_k", [1, 2])
def test_selected_bf16_keeps_midpoint_boundaries_and_repeated_routes(top_k):
    selected = PreparedTesseraSelectedBf16MoeExperts(
        torch.tensor([[[1.0], [1.3671875]]], dtype=torch.bfloat16),
        torch.ones((1, 1, 1), dtype=torch.bfloat16),
        torch.tensor([[[1.0 + 2.0 ** -8], [1.0]]], dtype=torch.float32),
        torch.full((1, 1, 1), 1.0 + 2.0 ** -8, dtype=torch.float32),
    )
    layer = SimpleNamespace(apply_router_weight_on_input=False, swiglu_limit=None)
    x = torch.ones((1, 1), dtype=torch.bfloat16)
    ids = torch.zeros((1, top_k), dtype=torch.int32)
    weights = torch.full((1, top_k), 0.75 / top_k, dtype=torch.float32)
    got = _selected_bf16_apply(
        x, selected, weights, ids, torch.tensor([0], dtype=torch.int32),
        layer=layer, prefix="selected BF16 boundary",
    )
    # The gate midpoint rounds to 1.0 before SwiGLU. The activation rounds
    # to 1.0. Each route then receives its row scale and router weight.
    expected = torch.full_like(got, 0.75390625)
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), (
        f"top_k={top_k}: got {float(got[0, 0])}, expected 0.75390625"
    )
