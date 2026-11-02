"""Shared expert wire containers and stock-reference tensors."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tessera.serving.scheme import TESSERA_FP8


HIDDEN, INTER, EXPERTS, Q256 = 64, 32, 3, 1024


def _tessera():
    return (pytest.importorskip("tessera.export"), pytest.importorskip("tessera.stock"),
            pytest.importorskip("tessera.alphabet"), pytest.importorskip("tessera.fused"))


def _encode(rows, cols, name, seed):
    export, stock, alphabet, fused = _tessera()
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=generator) * 0.02
    exported, unit, forests = export.encode_linear_planes(
        w.contiguous(), grid=alphabet.E4M3_GRID, q256=Q256, name=name, verify=False)
    blob = fused.pack_fused([(name, rows, exported.blob)])
    return blob, stock.materialize_stock(unit, forests, export.DEFAULT_CODE)


def _stack(experts=EXPERTS, hidden=HIDDEN, inter=INTER):
    """E experts of (gate, up, down) wires, plus the stock reference tensors."""
    w13_blobs, w2_blobs, reference = [], [], []
    for e in range(experts):
        gate, gate_ref = _encode(inter, hidden, "gate_proj", 100 + e)
        up, up_ref = _encode(inter, hidden, "up_proj", 200 + e)
        down, down_ref = _encode(hidden, inter, "down_proj", 300 + e)
        w13_blobs.append([gate, up])
        w2_blobs.append(down)
        reference.append({"gate": gate_ref, "up": up_ref, "down": down_ref})
    scheme = {
        "family": TESSERA_FP8, "structure": "routed_moe", "grid": "E4M3", "body": "WINDOW",
        "plane": "CHANNEL", "experts": experts,
        "expert_ids": list(range(experts)),
        "expert_classes": [{"start": 0, "end": experts,
                            "q256": {"w13": [Q256, Q256], "w2": [Q256]}}],
        "groups": {
            "w13": {"rows": 2 * inter, "columns": hidden, "q256": Q256,
                    "wire_stride": max(len(b) for pair in w13_blobs for b in pair),
                    "roles": [["gate_proj", inter], ["up_proj", inter]]},
            "w2": {"rows": hidden, "columns": inter, "q256": Q256,
                   "wire_stride": max(len(b) for b in w2_blobs),
                   "roles": [["down_proj", hidden]]}},
    }
    return w13_blobs, w2_blobs, scheme, reference
