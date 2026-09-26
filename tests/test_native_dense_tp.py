"""CPU-only TP2 driver tests: fake collectives, never native GPU evidence."""
import copy
from types import SimpleNamespace

import pytest
from test_native_operator_receipt import (
    _fake_lifecycle,
    _fake_preparation,
    _panel_fixture,
    _tensor_identity,
)

from experiments import bench_native_operator as dense

torch = pytest.importorskip("torch")


def execution(axis):
    return {**dense.EXECUTION, "tensor_parallel": 2, "tensor_parallel_cut_axis": axis}


def distributed(rank=0):
    return {"world_size": 2, "rank": rank, "init_method": "tcp://192.0.2.1:29517",
            "timeout_seconds": 60}


@pytest.mark.parametrize("axis,local", [("output", [64, 256]), ("input", [128, 128])])
def test_tp2_geometry_uses_local_tile_but_keeps_whole_wire(axis, local):
    assert dense.local_shape([128, 256], execution(axis)) == local
    assert dense.whole_shape(local, execution(axis)) == [128, 256]


@pytest.mark.parametrize("axis", ["output", "input"])
@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_source_identity_uses_the_served_rank_slice(axis, rank):
    weight = torch.arange(128 * 256).reshape(128, 256)
    actual = dense.rank_weight(weight, execution(axis), rank)
    expected = weight[rank*64:(rank+1)*64] if axis == "output" else weight[:, rank*128:(rank+1)*128]
    assert torch.equal(actual, expected)
    assert actual.is_contiguous()


@pytest.mark.parametrize("axis", ["output", "input"])
@pytest.mark.parametrize("rank", [0, 1])
def test_complete_apply_reduces_row_once_and_column_never(monkeypatch, axis, rank):
    calls = []
    layer = SimpleNamespace(tp_size=2, tp_rank=rank)
    prepared = {"layer": layer, "runtime": {"execution": execution(axis)},
                "method": SimpleNamespace(apply=lambda layer, x: calls.append("apply") or x + rank)}
    monkeypatch.setattr(dense, "_all_reduce", lambda x: calls.append("reduce") or x + 10)
    actual = dense.apply_complete(prepared, torch.ones(1, 8))
    assert calls == (["apply", "reduce"] if axis == "input" else ["apply"])
    assert torch.equal(actual, torch.ones(1, 8) + rank + (10 if axis == "input" else 0))


@pytest.mark.parametrize("bad", [True, 0, 3, 2.0])
def test_execution_refuses_non_supported_worlds(bad):
    with pytest.raises(ValueError):
        dense.validate_execution({**execution("output"), "tensor_parallel": bad})


@pytest.mark.parametrize("block", [None, {**distributed(), "world_size": 1},
                                      {**distributed(), "rank": True},
                                      {**distributed(), "init_method": "file:///not-a-world"}])
def test_tp2_requires_matching_explicit_distributed_block(block):
    with pytest.raises(ValueError):
        dense.execution_distributed(execution("output"), block)


@pytest.mark.parametrize("axis", ["output", "input"])
@pytest.mark.parametrize("family,rates", [("BF16", (1024, 1792)), ("E4M3", (1024, 1536)), ("E2M1", (896, 1024))])
def test_tp2_panel_preserves_local_shapes_and_world_for_every_rate(axis, family, rates):
    for rate in rates:
        panel, _, _, _ = _panel_fixture()
        fmt = f"TESSERA_{family}_K{2 if family == 'E2M1' else 1}_R{rate}"
        panel["format"] = panel["joint_operator_identity"]["format"] = fmt
        panel["joint_operator_identity_sha256"] = dense.identity_sha256(panel["joint_operator_identity"])
        panel["execution"] = execution(axis)
        panel["runtime"]["execution"] = execution(axis)
        panel["runtime"]["distributed"] = distributed()
        # The fixture describes this rank's measured tile. The retained wire is global.
        panel["wire"]["record"]["identity"]["source"]["shape"] = dense.whole_shape(panel["shape"], execution(axis))
        assert dense.validate_panel(panel) == panel
        bad = copy.deepcopy(panel)
        bad["runtime"]["execution"]["tensor_parallel"] = 1
        with pytest.raises(ValueError):
            dense.validate_panel(bad)


@pytest.mark.parametrize("axis", ["output", "input"])
@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_preparation_loads_whole_wire_with_local_create_weights(monkeypatch, axis, rank):
    from experiments import bench_native_moe_operator as routed
    from tessera.serving import lane
    module, blob, record, source, rendered, kwargs, trace = _fake_preparation(monkeypatch)
    original_build = lane.build_tessera_method
    original_observe = module.observe_runtime
    def observe(image):
        return {**original_observe(image), "gpu": {"uuid": f"CPU-FIXTURE-RANK-{rank}"}}
    monkeypatch.setattr(module, "observe_runtime", observe)
    local = [64, 256] if axis == "output" else [128, 128]
    def build(scheme, unit, mode):
        method = original_build(scheme, unit, mode)
        def create(layer, **sizes):
            assert sizes == {"input_size_per_partition": local[1], "output_partition_sizes": [local[0]],
                             "input_size": 256, "output_size": 128, "params_dtype": torch.bfloat16}
            assert (layer.tp_rank, layer.tp_size) == (rank, 2)
            layer.register_buffer("wire_bytes", torch.empty(0, dtype=torch.uint8))
        method.create_weights = create
        method.apply = lambda layer, value: value
        return method
    monkeypatch.setattr(lane, "build_tessera_method", build)
    monkeypatch.setattr(routed, "bind_owner_rank", lambda d: (d["rank"], d["world_size"]))
    calls = []
    monkeypatch.setattr(module, "_all_reduce", lambda x: calls.append(True) or x)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    prepared = module.prepare_native_operator(blob, record, source, rendered, **kwargs,
        execution=execution(axis), distributed=distributed(rank),
        phase_inputs={phase: torch.zeros(1, local[1], dtype=torch.bfloat16) for phase in module.PHASES})
    expected = source[rank*64:(rank+1)*64] if axis == "output" else source[:, rank*128:(rank+1)*128]
    assert prepared["operator"]["source_weight"] == _tensor_identity(expected)
    assert prepared["operator"]["rendered_weight"] == _tensor_identity(expected)
    assert prepared["operator"]["scheme"]["rows"] == 128
    assert prepared["operator"]["scheme"]["columns"] == 256
    assert prepared["runtime"]["execution"] == execution(axis)
    assert prepared["runtime"]["distributed"] == distributed(rank)
    assert prepared["runtime"]["gpu"]["uuid"] == f"CPU-FIXTURE-RANK-{rank}"
    assert len(calls) == (2 if axis == "input" else 0)
    assert trace[-1] == "observe runtime after native load"


@pytest.mark.parametrize("axis", ["output", "input"])
def test_all_measurement_paths_use_complete_apply(monkeypatch, axis):
    module, panel, prepared, tensors, trace = _fake_lifecycle(monkeypatch)
    panel["execution"] = execution(axis)
    panel["runtime"]["execution"] = execution(axis)
    panel["runtime"]["distributed"] = distributed()
    panel["wire"]["record"]["identity"]["source"]["shape"] = dense.whole_shape(panel["shape"], execution(axis))
    prepared["runtime"] = copy.deepcopy(panel["runtime"])
    prepared["layer"].tp_size = 2
    global_shape = dense.whole_shape(panel["shape"], execution(axis))
    prepared["operator"]["wire_record_sha256"] = dense.identity_sha256(panel["wire"]["record"])
    prepared["operator"]["scheme"].update(rows=global_shape[0], columns=global_shape[1],
                                           roles=[["weight", global_shape[0]]])
    prepared["operator"]["scheme_sha256"] = dense.identity_sha256(prepared["operator"]["scheme"])
    panel["scheme_sha256"] = prepared["operator"]["scheme_sha256"]
    prepared["layer"].tessera_scheme = copy.deepcopy(prepared["operator"]["scheme"])
    from experiments import bench_native_moe_operator as routed
    monkeypatch.setattr(routed, "bind_owner_rank", lambda d: (d["rank"], d["world_size"]))
    monkeypatch.setattr(module, "_agree_numerical_status", lambda prepared, passed: passed)
    monkeypatch.setattr(module, "_agree_output", lambda *a: None)
    reductions = []
    monkeypatch.setattr(module, "_all_reduce", lambda x: reductions.append(True) or x)
    receipt = module.measure_prepared_operator(prepared, panel, tensors, warmup_iterations=2, iterations=3)
    assert receipt["runtime"]["execution"]["tensor_parallel"] == 2
    assert receipt["panel"]["shape"] == [128, 256]
    # CPU-side producer schema check against PQ admit_native_rows/consume_native_receipt:
    # dense roster is rank 0; runtime+panel world, local member shape, joint identity,
    # wire record and route shape must travel together. No PQ import in Tessera.
    assert receipt["panel"]["execution"] == receipt["runtime"]["execution"]
    assert receipt["runtime"]["distributed"]["rank"] == 0
    assert receipt["panel"]["runtime"] == receipt["runtime"]
    assert receipt["operator"]["source_weight"] == panel["joint_operator_identity"]["source_weight"]
    assert receipt["operator"]["rendered_weight"] == panel["joint_operator_identity"]["rendered_weight"]
    assert receipt["operator"]["wire_record_sha256"] == dense.identity_sha256(panel["wire"]["record"])
    for phase in module.PHASES:
        assert receipt["phases"][phase]["route"]["shape"] == f"M{panel['phases'][phase]['m']}:N128:K256"
    assert len(reductions) == (4 if axis == "input" else 0)
    reductions.clear()
    class Collector:
        _finished = True
        def observe_apply(self, apply, phase, *, device):
            return apply(), {"fixture": "CPU fake"}
    receipt = module.measure_prepared_operator(prepared, panel, tensors,
        warmup_iterations=2, iterations=3, resource_collector=Collector())
    assert len(reductions) == (8 if axis == "input" else 0)
    receipt["resources"]["status"] = "complete_operator_bound"
    module.time_after_resource_collection(prepared, panel, tensors, receipt,
        collector=Collector(), warmup_iterations=2, iterations=3)
    assert len(reductions) == (10 if axis == "input" else 0)
