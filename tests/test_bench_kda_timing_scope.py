"""Full KDA geometry reaches benchmark consumers (tessera#1020 follow-up).

The served-module contract is already corrected. Neither the large-M FP8
comparison nor the default geometry sweep may silently retain the old scope.
These are CPU-side integration checks, not new GPU timing or D41 evidence.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tessera import routed_fused as rf

T8R = Path(__file__).resolve().parents[1] / "experiments" / "t8r_speed"


def load_bench(name):
    sys.path.insert(0, str(T8R))
    try:
        spec = importlib.util.spec_from_file_location(f"{name}_timing_scope", T8R / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(T8R))


def kda_shape():
    producer = load_bench("bench_dense_module")
    producer.require_pinned_kda_contract()
    roles, cols = producer.MODULES["kda_in"]
    return sum(rows for _, rows in roles), cols


def test_large_m_kda_uses_complete_producer_shape():
    bench = load_bench("bench_fp8_large_m")
    expected = kda_shape()
    assert bench.SHAPES["kda_in"] == expected, (
        "kda_in must measure the complete producer shape, not the historical "
        f"12448-row proxy: expected {expected}, got {bench.SHAPES['kda_in']}"
    )


@pytest.mark.parametrize("family", ["e4m3", "value"])
def test_default_geometry_kda_reaches_dense_builder(monkeypatch, family):
    bench = load_bench("bench_geometry")
    rows, cols = kda_shape()
    assert (rows, cols) == bench.PROTOCOL_DENSE["kda_in"]
    assert rows % rf.DENSE_ROW_QUANTUM == 0
    assert rows % rf.BN != 0  # the supported tail the stale guard drops
    built = []
    projection = {"tile_words": 1}

    def build_projection(owner, experts, n, k, *args, **kwargs):
        built.append((owner, experts, n, k))
        return projection

    # Only replace wire generation: run the real build_dense admission and
    # geometry code, without allocating the 12576x4096 wire or loading CUDA.
    monkeypatch.setattr(bench, "build_projection", build_projection)
    sweep = SimpleNamespace(rf=rf, fp8=family == "e4m3", family=family,
                            mma8=family == "e4m3", dev=torch.device("cpu"))
    head, make, keep = bench.Sweep.build_dense(sweep, 1024, "kda_in")

    assert "refused" not in head, (
        "the full default KDA shape must reach timing; the dense reader "
        f"already supports a partial final BN block: {head.get('refused')}"
    )
    assert (head["rows"], head["cols"]) == (rows, cols)
    assert built == [(rf, 1, rows, cols)]
    assert callable(make), "a default KDA cell needs a timing callable, not a refusal"
    assert keep is projection
