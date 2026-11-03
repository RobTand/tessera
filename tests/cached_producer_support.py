"""Shared cached-unit wires and identities."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from tessera.alphabet import E4M3_GRID
from tessera.export import encode_linear


ROOT = Path(__file__).resolve().parents[1]


STACK = "model.layers.2.feed_forward.experts"


TENSOR = STACK + ".0.w1.weight"


UNIT = TENSOR.removesuffix(".weight")


DIM = 128


def _api():
    return importlib.import_module("tessera.cached_unit")


def _exporter():
    return importlib.import_module("tessera.export_serving")


@pytest.fixture(scope="module")
def encoded():
    weight = torch.randn(DIM, DIM, generator=torch.Generator().manual_seed(183)).bfloat16()
    unit = encode_linear(weight.float(), grid=E4M3_GRID, q256=1024,
                         name="TESSERA_E4M3_K1_R1024", verify=False)
    return weight, unit.blob


def _projection():
    return {"tensor": TENSOR, "source_tensor": TENSOR,
            "source_layout": "unpacked_per_expert", "expert": 0,
            "source_slice": {"expert": 0, "selector": "whole", "transpose": False},
            "projection": "gate_proj", "group": "w13", "rows": DIM, "cols": DIM}


def _record(encoded, activation=None):
    api = _api()
    weight, blob = encoded
    identity = api.unit_input_identity(weight, _projection(), E4M3_GRID, 1024,
                                       activation=activation)
    return api.make_unit_record(blob, identity, filename="unit.tessera"), identity
