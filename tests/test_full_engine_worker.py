"""CPU control-flow tests; stock GPU startup remains separately qualified."""
import importlib
import os
import sys
from types import SimpleNamespace

import pytest

from experiments.capture_full_engine_resources import canonical_roster
from experiments import full_engine_bootstrap as bootstrap


def test_roster_preserves_fused_groups_without_duplicate_members():
    census = {"anchor_groups": {"g:model.qkv": ["model.q", "model.k", "model.v"],
                                 "s:model.experts": ["model.experts.0.w1"]},
              "dense_targets": ["model.q", "model.k", "model.v", "model.out"]}
    rows = canonical_roster(census)
    assert [(row["unit_id"], row["module"]) for row in rows] == [
        ("g:model.qkv", "model.qkv"), ("l:model.out", "model.out"),
        ("s:model.experts", "model.experts")]


def test_roster_refuses_ambiguous_modules():
    with pytest.raises(ValueError, match="distinct"):
        canonical_roster({"anchor_groups": {"g:model.x": ["model.a"],
                                            "s:model.x": ["model.b"]}, "dense_targets": []})


@pytest.mark.parametrize("condition", ["absent", "claimed", "forked", "late", "errors"])
def test_worker_cannot_claim_incomplete_or_inherited_bootstrap(monkeypatch, condition):
    recorder = SimpleNamespace(process_id=os.getpid(), _early=True, _errors=[])
    if condition == "forked":
        recorder.process_id += 1
    if condition == "late":
        recorder._early = False
    if condition == "errors":
        recorder._errors.append("fixture history failed")
    monkeypatch.setattr(bootstrap, "_recorder", None if condition == "absent" else recorder)
    monkeypatch.setattr(bootstrap, "_claimed", condition == "claimed")
    with pytest.raises(RuntimeError):
        bootstrap.claim()


def test_worker_claim_is_unique(monkeypatch):
    recorder = SimpleNamespace(process_id=os.getpid(), _early=True, _errors=[])
    plan = {"identity": "fixture"}
    monkeypatch.setattr(bootstrap, "_recorder", recorder)
    monkeypatch.setattr(bootstrap, "_plan", plan)
    monkeypatch.setattr(bootstrap, "_claimed", False)
    assert bootstrap.claim() == (recorder, plan)
    with pytest.raises(RuntimeError):
        bootstrap.claim()


def test_late_bootstrap_fails_before_loading_collector(monkeypatch):
    monkeypatch.setattr(bootstrap, "_recorder", None)
    monkeypatch.setenv("TESSERA_ENGINE_RESOURCE_PLAN", "/unused.json")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    with pytest.raises(RuntimeError, match="precede"):
        bootstrap.start()


@pytest.fixture
def worker_module(monkeypatch):
    class StockWorker:
        def __init__(self, *args, **kwargs):
            self.model_runner = None
        def init_device(self):
            return "device"
        def execute_model(self, scheduler_output):
            return scheduler_output
        def sample_tokens(self, grammar_output):
            return grammar_output
        def initialize_from_config(self, kv_cache_config):
            return "kv initialized"
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", SimpleNamespace(Worker=StockWorker))
    sys.modules.pop("experiments.full_engine_worker", None)
    module = importlib.import_module("experiments.full_engine_worker")
    yield module
    sys.modules.pop("experiments.full_engine_worker", None)


def test_parameter_classification_keeps_router_and_tied_head_fixed(worker_module):
    classify = worker_module.parameter_category
    units = ["model.layers.2.feed_forward.experts", "model.layers.0.feed_forward.w13"]
    assert classify("model.layers.2.feed_forward.experts.w13_weight", units) == "candidate"
    assert classify("model.layers.2.feed_forward.gate.weight", units) == "fixed"
    assert classify("model.layers.2.feed_forward.experts_extra.weight", units) == "fixed"
    assert classify("lm_head.weight", units) == "fixed"
    assert classify("model.embed_tokens.weight", units) == "fixed"


def test_worker_bounds_execute_capture_but_preserves_stock_outputs(monkeypatch, worker_module):
    seen = []
    recorder = SimpleNamespace(snapshot=lambda label, **kwargs: seen.append(label))
    plan = {"max_execute_calls": 2}
    monkeypatch.setattr(worker_module, "claim", lambda: (recorder, plan))
    worker = worker_module.ResourceCaptureWorker()
    assert worker.init_device() == "device"
    for index in range(1, 4):
        assert worker.execute_model(index) == index
        assert worker.sample_tokens(index) == index
    assert seen == ["device_initialized", "execute:1:begin", "execute:1:end", "sample:1:end",
                    "execute:2:begin", "execute:2:end", "sample:2:end"]
    assert worker._resource_active is False


def test_failed_stock_execute_still_records_boundary_and_disarms(monkeypatch, worker_module):
    seen = []
    recorder = SimpleNamespace(snapshot=lambda label, **kwargs: seen.append(label))
    monkeypatch.setattr(worker_module, "claim", lambda: (recorder, {"max_execute_calls": 1}))
    def fail(*args):
        raise RuntimeError("stock fixture failed")
    monkeypatch.setattr(worker_module.Worker, "execute_model", fail)
    worker = worker_module.ResourceCaptureWorker()
    with pytest.raises(RuntimeError, match="stock fixture"):
        worker.execute_model(None)
    assert seen == ["execute:1:begin", "execute:1:end"]
    assert worker._resource_active is False


def test_worker_preserves_stock_1970_kv_placement_descriptors(monkeypatch, worker_module):
    seen = []
    recorder = SimpleNamespace(snapshot=lambda label, **kwargs: seen.append(label))
    monkeypatch.setattr(worker_module, "claim", lambda: (recorder, {}))
    worker = worker_module.ResourceCaptureWorker()
    tensor = SimpleNamespace(size=8192, layers=["attention", "recurrent"],
                             layer_stride=512, block_stride=1024, offset=128)
    config = SimpleNamespace(num_blocks=8, kv_cache_tensors=[tensor])
    assert worker.initialize_from_config(config) == "kv initialized"
    assert worker._resource_kv_description["tensors"] == [{
        "size": 8192, "layers": ["attention", "recurrent"],
        "layer_stride": 512, "block_stride": 1024, "offset": 128}]
    assert seen == ["before_kv_allocation", "kv_allocated"]
