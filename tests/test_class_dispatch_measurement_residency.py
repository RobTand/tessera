"""D41 reports the tensor storage that the production owner retains."""
import gc
from pathlib import Path
from types import SimpleNamespace
import sys
import weakref

import pytest

torch = pytest.importorskip("torch")
from test_routed_window_classes import allow_cpu, descriptors, projection
from tessera.native_window_moe import PackedWindowMoeBundles
from tessera.serving.residency import resident_storage_bytes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/t8r_speed"))
import bench_class_dispatch as bench


def test_measure_charges_only_production_owner(tmp_path, monkeypatch):
    allow_cpu(monkeypatch)
    monkeypatch.setattr(bench, "EXPERTS", 4)
    monkeypatch.setattr(bench, "HIDDEN", 128)
    monkeypatch.setattr(bench, "INTER", 128)
    expected = {}
    load_refs = []

    def build(*args, **kwargs):
        roles = [projection(family="e4m3") for _ in range(3)]
        prepared = PackedWindowMoeBundles(*roles, family="e4m3", expert_classes=descriptors())
        inverse = torch.arange(4, dtype=torch.int32)
        owner = prepared.native_owner()
        expected["production"] = resident_storage_bytes(owner.named_tensors()) + inverse.untyped_storage().nbytes()
        expected["preparation"] = prepared.resident_bytes() + inverse.untyped_storage().nbytes()
        for role in roles:
            load_refs.extend(weakref.ref(getattr(role, field)) for field in ("codes_all", "native_all", "perm_all"))
        return prepared, {}, inverse

    monkeypatch.setattr(bench, "packed_constants", build)
    observed = []

    def timing(*args):
        gc.collect()
        observed.append(all(ref() is None for ref in load_refs))
        return {}

    monkeypatch.setattr(bench, "timed_samples", timing)
    monkeypatch.setattr(bench, "class_geometry", lambda *args: [])
    monkeypatch.setattr(bench, "kernel_profile", lambda *args, **kwargs: {})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    report = SimpleNamespace(data={"groups": {"q3_q4:M1": {"measurements": {}}}}, save=lambda: None)
    args = SimpleNamespace(seed=1, projections="full", prof_reps=1, out=str(tmp_path),
                           power_s=0, ncu=False, admission_label="GPU-exclusive, not quiet-host certified")
    bench.measure(args, report, "q3_q4", (768, 1024), 1, "eager", "F", "production", {},
                  SimpleNamespace(sample_during=lambda *args, **kwargs: {}), SimpleNamespace(read=lambda: {}))
    group = report.data["groups"]["q3_q4:M1"]
    assert group["resident_bytes"] == expected["production"], "D41 reports preparation storage as production residency."
    assert group["preparation_storage_bytes"] == expected["preparation"]
    assert observed == [True], "Load-only planes must retire before timing."
