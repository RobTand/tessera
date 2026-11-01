"""The routed numerical oracle reads canonical factors, not stock BF16 tiles."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tessera.alphabet import BF16_GRID
from tessera.export import encode_linear_planes
from tessera.fused import pack_fused
from tessera.unit_artifact import build_unit_artifact, read_unit_artifact


def oracle_module():
    path = Path(__file__).resolve().parents[1] / "experiments/routed_pair_oracle.py"
    spec = importlib.util.spec_from_file_location("routed_pair_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bf16_wires(module, sign=1.0, scale=None, hidden=128, inter=32):
    if scale is None:
        scale = 1.0 + torch.finfo(torch.bfloat16).eps / 2
    shapes = {"gate_proj": (inter, hidden), "up_proj": (inter, hidden),
              "down_proj": (hidden, inter)}
    blobs, decoded = {}, {}
    for projection, shape in shapes.items():
        _, unit, forests = encode_linear_planes(
            torch.full(shape, sign), grid=BF16_GRID, q256=1024,
            name=projection, verify=False)
        unit.scale_rows.fill_(1.0)
        unit.scale_global = float(scale)
        _, _, blob = build_unit_artifact(
            unit, projection, forests, q256=1024, fixture_id=None)
        blobs[projection] = [pack_fused([(projection, shape[0], blob)])]
        decoded[projection] = read_unit_artifact(blob, device="cpu")
    return blobs, decoded


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_bf16_reference_preserves_fp32_effective_weights(sign):
    module = oracle_module()
    family = module.FAMILIES["bf16"]
    blobs, decoded = bf16_wires(module, sign=sign)
    scheme, _ = module.scheme_for(family, blobs, 1)
    reference, _ = module.reference_weights(
        family, scheme, blobs, 1, "oracle.midpoint", device="cpu")
    route_expert = torch.zeros(1, dtype=torch.int64)
    router_weight = 0.75
    for projection, weights in decoded.items():
        activations = torch.zeros(1, weights.shape[1], dtype=torch.float64)
        activations[0, 0] = 1.0
        actual, _, _, _ = module.expert_gemm(
            activations, route_expert, reference, projection)
        expected = activations @ weights.double().t()
        expected *= router_weight
        actual *= router_weight
        assert torch.equal(actual, expected), (
            f"{projection}: the oracle rounded an effective BF16 weight before its dot; "
            f"got {float(actual[0, 0])}, expected {float(expected[0, 0])}")
        assert torch.equal(actual.bfloat16().view(torch.int16),
                           expected.bfloat16().view(torch.int16))
