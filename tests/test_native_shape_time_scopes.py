"""Scope admission for the shape-time panel: dense TP1/TP2, routed TP2."""
from __future__ import annotations

import copy
import gzip
import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import box_artifacts
import pytest
import torch

from test_native_timing_panel import canonical_wire, panel
from tools import tessera_shape_time_panel as app
from tools import tessera_shape_time_worker as worker
from tessera.serving import census_plan, contract, scheme
from tessera.serving import timing_panel as tp


def _save(root, name, value, raw=False):
    path = Path(root) / name
    path.write_bytes(value if raw else tp.canonical(value))
    return path


def _request_value(panel, monkeypatch, *, scope, runtime, scheme_decl, wire_binding,
                   prefix="test.scope", contract_binding=None):
    monkeypatch.setattr(contract, "contract_path", lambda: Path(
        (contract_binding or panel["evidence"]["contract"])["path"]))
    value = {"schema": app.REQUEST_SCHEMA, "expected_runtime": runtime, "scope": scope,
             "prefix": prefix, "scheme": scheme_decl, "wire": wire_binding,
             "contract": contract_binding or panel["evidence"]["contract"],
             "record_verifier": app.tp.file_binding(box_artifacts.skip_now("prismabuild_tools", "pbtest_pins.py")),
             "runtime_python": app.tp.file_binding(sys.executable), "worker_timeout_s": 120,
             "sampling": {"samples": 4, "warmup_iterations": 1, "steady_s": 20.0, "seed": 688},
             "netdata_hosts": {"sparky": "sparky", "sparklina": "sparklina"}}
    source = {"schema": app.PRODUCER_SCHEMA, "commit": "a" * 40, "commit_source": "sealed_checkout",
              **app.producer_source_identity()}
    value["producer_identity"] = app.publish_json(
        source, Path(panel["evidence"]["wire"]["path"]).parent / "producer-input.json")
    path = Path(panel["evidence"]["wire"]["path"]).parent / "request.json"
    path.write_bytes(app.tp.canonical(value))
    return path, value


def _dense_request(panel, monkeypatch, *, tp_degree, tp_rank=0):
    scope = dict(panel["plan"]["rows"][0]["scope"], tp_degree=tp_degree)
    runtime = dict(panel["runtime"], tp_degree=tp_degree, tp_rank=tp_rank)
    return _request_value(panel, monkeypatch, scope=scope, runtime=runtime,
                          scheme_decl=copy.deepcopy(panel["rows"][0]["scheme"]),
                          wire_binding=panel["evidence"]["wire"])


@pytest.mark.parametrize("tp_degree", [1, 2])
def test_request_accepts_dense_tp1_and_tp2(panel, monkeypatch, tp_degree):
    path, value = _dense_request(panel, monkeypatch, tp_degree=tp_degree)
    got, scope, wire = app.read_request(path)
    assert got == value and scope == value["scope"]
    assert wire == app.tp.read_bound(value["wire"])
    assert app.main(["check-request", str(path)]) == 0


def _routed_stack(tmp_path):
    from _routed_classes_plugin_fixture import wire_fixture
    scheme_decl, wires = wire_fixture("e4m3", "uniform")
    groups = {}
    for group in scheme.MOE_GROUPS:
        entries = []
        for storage in range(scheme_decl["experts"]):
            shards = []
            for shard in scheme.MOE_GROUP_SHARDS[group]:
                blob = wires[(storage, shard)]
                bound = tp.file_binding(_save(tmp_path, f"{group}.{storage}.{shard}.wire", blob, raw=True))
                shards.append(bound)
            entries.append(shards)
        groups[group] = entries
    index = {"schema": tp.MOE_WIRE_INDEX_SCHEMA, "groups": groups}
    index_path = tmp_path / "moe-wire-index.json"
    index_path.write_bytes(tp.canonical(index))
    return scheme_decl, tp.file_binding(index_path)


def _routed_cell_contract():
    doc = copy.deepcopy(contract.load_serving_contract())
    family = contract.PAYLOAD_FAMILY_BY_ROUTE[scheme.TESSERA_FP8]
    cell = next(c for c in doc["lane_eligibility"]["cells"] if c["platform"] == "sm_121"
                and c["family"] == family and c["structure"] == "routed_moe" and c["regime"] == "batch"
                and "resident" in contract.cell_residency_modes(c)
                and "eager" in contract.cell_runtime_scope(c)[1]
                and contract.cell_is_device_backed(c)
                and contract.cell_covers_rung(c, 1024, next(f for f in doc["formats"] if f["family"] == family)))
    cell["runtime"].update(tessera_commit="1" * 40, serving_source_sha256="2" * 64)
    packed = tp.canonical(doc)
    return doc, cell, packed


def _routed_scope_and_runtime(cell, packed):
    scope = {"route": scheme.TESSERA_FP8, "grid": "E4M3", "q256": 1024, "structure": "routed_moe",
             "mode": "resident", "execution_mode": "eager", "regime": "batch", "tp_degree": 2,
             "requested_platform": "sm_121",
             "shape": {"M": 64, "N": 512, "K": 256, "experts": 8, "topk": 8}}
    runtime = {"image": cell["runtime"]["image"], "tessera_commit": "1" * 40,
               "serving_source_sha256": "2" * 64, "contract_sha256": hashlib.sha256(packed).hexdigest(),
               "platform": "sm_121", "torch": cell["runtime"]["torch"], "vllm": cell["runtime"]["vllm"],
               "serve_flags": {"TESSERA_SERVE_MODE": "resident"}, "residency": "resident",
               "execution_mode": "eager", "tp_rank": 0, "tp_degree": 2, "package_root": "/CPU-fixture/tessera"}
    return scope, runtime


def _routed_request(panel, monkeypatch, tmp_path, *, scope=None, runtime=None):
    scheme_decl, wire_binding = _routed_stack(tmp_path)
    doc, cell, packed = _routed_cell_contract()
    default_scope, default_runtime = _routed_scope_and_runtime(cell, packed)
    contract_binding = tp.file_binding(_save(tmp_path, "contract.json", packed, raw=True))
    return _request_value(panel, monkeypatch, scope=scope or default_scope,
                          runtime=runtime or default_runtime, scheme_decl=scheme_decl,
                          wire_binding=wire_binding, prefix="test.routed",
                          contract_binding=contract_binding)


def test_request_accepts_routed_tp2(panel, monkeypatch, tmp_path):
    path, value = _routed_request(panel, monkeypatch, tmp_path)
    got, scope, wire = app.read_request(path)
    assert got == value and scope == value["scope"]
    assert app.main(["check-request", str(path)]) == 0


@pytest.mark.parametrize("fault,match", [
    ("routed_tp1", "admits dense TP1/TP2 and routed_moe TP2 only"),
    ("dense_tp3", "admits dense TP1/TP2 and routed_moe TP2 only"),
    ("streamed", "supports eager/resident only"),
    ("compiled", "supports eager/resident only"),
    ("tp_mismatch", "scope TP degree differs"),
    ("unknown_route", "dispatch registry"),
])
def test_request_refuses_other_scopes_by_name(panel, monkeypatch, fault, match):
    scope = copy.deepcopy(panel["plan"]["rows"][0]["scope"])
    if fault == "routed_tp1":
        scope.update(structure="routed_moe", tp_degree=1,
                     shape={"M": 512, "N": 16, "K": 128, "experts": 2, "topk": 2})
    elif fault == "dense_tp3":
        scope["tp_degree"] = 3
    elif fault == "streamed":
        scope["mode"] = "streamed"
    elif fault == "compiled":
        scope["execution_mode"] = "compiled"
    elif fault == "tp_mismatch":
        scope["tp_degree"] = 2
    else:
        scope["route"] = "INVENTED"
    path, _ = _request_value(panel, monkeypatch, scope=scope, runtime=copy.deepcopy(panel["runtime"]),
                             scheme_decl=copy.deepcopy(panel["rows"][0]["scheme"]),
                             wire_binding=panel["evidence"]["wire"])
    with pytest.raises(ValueError, match=match):
        app.read_request(path)


def test_request_refuses_routed_geometry_mismatch(panel, monkeypatch, tmp_path):
    scheme_decl, wire_binding = _routed_stack(tmp_path)
    doc, cell, packed = _routed_cell_contract()
    scope, runtime = _routed_scope_and_runtime(cell, packed)
    scope = dict(scope, shape=dict(scope["shape"], N=999))
    contract_binding = tp.file_binding(_save(tmp_path, "contract.json", packed, raw=True))
    path, _ = _request_value(panel, monkeypatch, scope=scope, runtime=runtime,
                             scheme_decl=scheme_decl, wire_binding=wire_binding, prefix="test.routed",
                             contract_binding=contract_binding)
    with pytest.raises(ValueError, match="stack geometry differs"):
        app.read_request(path)


def _pick_routed_launch(doc, cell, scope, runtime, roles):
    for launch in scheme.route_launches(scope["route"], structure="routed_moe",
                                        regime=scope["regime"], mode="resident"):
        if not launch["lane"]:
            continue
        if (launch["symbol"], launch["decoder"]) not in {
                tuple(entry[:2]) for entry in cell["executes"]}:
            continue
        try:
            tp.admitted_cell(doc, scope, runtime, (launch["symbol"], launch["decoder"]), roles)
        except ValueError:
            continue
        return launch
    raise AssertionError("no positively backed routed lane for the fixture stack")


def _routed_panel(tmp_path, monkeypatch):
    scheme_decl, wire_binding = _routed_stack(tmp_path)
    doc, cell, packed = _routed_cell_contract()
    scope, runtime = _routed_scope_and_runtime(cell, packed)
    monkeypatch.setattr(contract, "contract_path",
                        lambda: _save(tmp_path, "contract.json", packed, raw=True))
    plan = census_plan.build_census_plan([scope], raw_contract=packed)
    declared, roles = tp.wire_facts(tp.read_bound(wire_binding), scheme_decl)
    launch = _pick_routed_launch(doc, cell, scope, runtime, roles)
    record = {"kind": "moe", "policy": scheme.TESSERA_FP8 + ":resident", "state": "served",
              "symbol": launch["symbol"], "decoder": launch["decoder"], "shape": "M64:N512:K256",
              "contract": scheme.ROUTES[scheme.TESSERA_FP8]["activation_contract"], "platform": "sm_121"}
    samples = [1.0, 2.0, 3.0, 4.0]
    trace = {"traceEvents": [{"cat": "kernel", "name": "CPU trace fixture", "dur": 1.0}]}
    native = next(x for x in doc["native_extensions"] if x["module_name_prefix"] == launch["lane"])
    filename = native["filename_glob"].replace("*", "CPU-fixture")
    telemetry = {"interval_unix": [10.0, 11.0], "fast_power_samples": [[10.5, 40.0]],
                 "netdata": {box: {context: {"query": "CPU fixture", "raw_response": {"view": {}, "result": {"labels": ["time", "fixture"], "data": [[10, [0, 0, 0]]]}},
                                            "returned_view": {}} for context in tp.NETDATA_CONTEXTS}
                             for box in ("sparky", "sparklina")}}
    evidence = {
        "runtime": tp.file_binding(_save(tmp_path, "runtime.json", runtime)),
        "runtime_origins": tp.file_binding(_save(tmp_path, "origins.json", {
            "package_root": runtime["package_root"],
            "installation": {"module": "tessera", "distribution": "tessera-quant",
                             "expected_commit": runtime["tessera_commit"],
                             "installed_commit": runtime["tessera_commit"],
                             "origin": runtime["package_root"] + "/__init__.py", "verified_files": 43},
            "record_verifier": {"path": str(box_artifacts.path("prismabuild_tools", "pbtest_pins.py")),
                                "bytes": 1, "sha256": "a" * 64},
            "modules": {name: {"path": str(Path(runtime["package_root"]) / ("__init__.py" if name == "tessera" else name.removeprefix("tessera.").replace(".", "/") + ".py")),
                               "bytes": 1, "sha256": "a" * 64} for name in tp.RUNTIME_MODULES}})),
        "producer": tp.file_binding(_save(tmp_path, "producer.json", {
            "schema": "tessera.native_panel_producer_identity.v1", "commit": "3" * 40,
            "commit_source": "sealed_checkout", "source_tree_sha256": "6" * 64,
            "source_tree_members": 1, "tool_source_sha256": "4" * 64})),
        "contract": tp.file_binding(_save(tmp_path, "contract.json", packed, raw=True)),
        "wire": wire_binding,
        "preparation": tp.file_binding(_save(tmp_path, "preparation.json", {
            "builder": "tessera.serving.moe_route.build_tessera_moe_method",
            "wire_sha256": wire_binding["sha256"], "roles": roles, "shape": scope["shape"],
            "topk": 8, "tp_rank": 0, "tp_degree": 2, "grid": "E4M3",
            "local_shape": {"N": 512, "K": 256}, "native_packed_bytes": 1})),
        "samples": tp.file_binding(_save(tmp_path, "samples.json", {"samples_ms": samples,
                                                                    "warmup_iterations": 1,
                                                                    "interval_unix": [10.0, 11.0]})),
        "routes": tp.file_binding(_save(tmp_path, "routes.json", {"records": [record] * len(samples)})),
        "trace": tp.file_binding(_save(tmp_path, "trace.json.gz", gzip.compress(tp.canonical(trace)), raw=True)),
        "telemetry": tp.file_binding(_save(tmp_path, "telemetry.json", telemetry)),
        "native_binary": tp.file_binding(_save(tmp_path, filename, b"\x7fELF CPU binary fixture", raw=True)),
    }
    return {"schema": tp.SCHEMA, "status": "measured", "claims": dict(tp.CLAIMS), "runtime": runtime,
            "plan": plan, "rows": [{"scope_id": plan["rows"][0]["id"], "prefix": "test.routed",
                                    "scheme": scheme_decl, "timing": tp.timing_summary(samples),
                                    "cell_id": cell["id"]}],
            "evidence": evidence,
            "energy": {"status": "hold", "reason": "cross_host_clock_alignment_unqualified", "reference_w": 140}}


def test_routed_tp2_panel_validates(tmp_path, monkeypatch):
    candidate = _routed_panel(tmp_path, monkeypatch)
    result = tp.validate_panel(candidate, expected_runtime=copy.deepcopy(candidate["runtime"]))
    assert result["cell_id"] == candidate["rows"][0]["cell_id"]
    assert result["timing"] == candidate["rows"][0]["timing"]


def _dense_tp2_panel(panel, tmp_path):
    scope = dict(panel["plan"]["rows"][0]["scope"], tp_degree=2)
    runtime = dict(panel["runtime"], tp_degree=2, tp_rank=1)
    packed = tp.read_bound(panel["evidence"]["contract"])
    plan = census_plan.build_census_plan([scope], raw_contract=packed)
    shape = scope["shape"]
    pair = _launch_pair(panel)
    record = {"kind": "dense", "policy": scheme.TESSERA_FP8 + ":resident", "state": "served",
              "symbol": pair["symbol"], "decoder": pair["decoder"],
              "shape": f"M{shape['M']}:N{shape['N'] // 2}:K{shape['K']}",
              "contract": scheme.ROUTES[scheme.TESSERA_FP8]["activation_contract"], "platform": "sm_121"}
    samples = [1.0, 2.0, 3.0, 4.0]
    old = panel["evidence"]
    prep = dict(tp.json_bytes(tp.read_bound(old["preparation"])))
    prep.update(tp_rank=1, tp_degree=2, axis="row")
    evidence = dict(old)
    evidence["preparation"] = tp.file_binding(_save(tmp_path, "preparation-tp2.json", prep))
    evidence["runtime"] = tp.file_binding(_save(tmp_path, "runtime-tp2.json", runtime))
    evidence["routes"] = tp.file_binding(_save(tmp_path, "routes-tp2.json", {"records": [record] * len(samples)}))
    return {"schema": tp.SCHEMA, "status": "measured", "claims": dict(tp.CLAIMS), "runtime": runtime,
            "plan": plan, "rows": [{"scope_id": plan["rows"][0]["id"], "prefix": "test.dense",
                                    "scheme": panel["rows"][0]["scheme"],
                                    "timing": tp.timing_summary(samples),
                                    "cell_id": panel["rows"][0]["cell_id"]}],
            "evidence": evidence,
            "energy": {"status": "hold", "reason": "cross_host_clock_alignment_unqualified", "reference_w": 140}}


def _launch_pair(panel):
    record = tp.json_bytes(tp.read_bound(panel["evidence"]["routes"]))["records"][0]
    return {"symbol": record["symbol"], "decoder": record["decoder"]}


def test_dense_tp2_panel_validates(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "contract_path",
                        lambda: Path(panel["evidence"]["contract"]["path"]))
    candidate = _dense_tp2_panel(panel, tmp_path)
    result = tp.validate_panel(candidate, expected_runtime=copy.deepcopy(candidate["runtime"]))
    assert result["cell_id"] == candidate["rows"][0]["cell_id"]
    assert result["timing"] == candidate["rows"][0]["timing"]


def test_tp1_preparation_carries_no_partition_key(panel):
    prep = tp.json_bytes(tp.read_bound(panel["evidence"]["preparation"]))
    assert set(prep) == {"builder", "wire_sha256", "roles", "shape", "tp_rank",
                         "tp_degree", "grid", "native_packed_bytes"}
    assert (prep["tp_rank"], prep["tp_degree"]) == (0, 1)


def test_worker_dense_tp2_uses_public_builder(panel, monkeypatch):
    from tessera.serving import lane, native_window
    events = []

    path, request = _dense_request(panel, monkeypatch, tp_degree=2, tp_rank=1)
    wire = app.tp.read_bound(request["wire"])

    class Method:
        def create_weights(self, layer, **kw):
            events.append(("create", kw))
            layer.wire_bytes = torch.nn.Parameter(torch.empty(len(wire), dtype=torch.uint8),
                                                  requires_grad=False)

        def process_weights_after_loading(self, layer):
            layer.tessera_shard_plan = SimpleNamespace(tp_rank=1, tp_size=2)
            published = native_window.DENSE_LANES[native_window.LANE_FUSED]
            layer.tessera_native = SimpleNamespace(
                rows=8, columns=128, packed_bytes=lambda: 123, lane=native_window.LANE_FUSED,
                launch_pair=(published[0], published[1][native_window.NATIVE_WINDOW_FAMILY["TESSERA_FP8"]]))

    method = Method()
    monkeypatch.setattr(lane, "build_tessera_method", lambda *a: (events.append(("build", a)) or method))
    layer, got, prep = worker.prepare_dense(
        dict(request, _wire_roles=app.tp.wire_facts(wire, request["scheme"])[1]), wire)
    assert got is method
    assert (layer.tp_rank, layer.tp_size) == (1, 2)
    assert events[1][1]["output_partition_sizes"] == [8]
    assert events[1][1]["input_size_per_partition"] == 128
    assert events[1][1]["output_size"] == 16
    assert prep["axis"] == "row" and (prep["tp_rank"], prep["tp_degree"]) == (1, 2)
    assert prep["shape"] == request["scope"]["shape"]


def test_worker_routed_tp2_uses_production_builder(panel, monkeypatch, tmp_path):
    from tessera.serving import moe_route
    calls = []

    class FakeMethod:
        def create_weights(self, layer, **kw):
            calls.append(("create", kw))
            layer.register_parameter("w13_wire", torch.nn.Parameter(torch.empty(0, dtype=torch.uint8),
                                                                    requires_grad=False))
            layer.register_parameter("w2_wire", torch.nn.Parameter(torch.empty(0, dtype=torch.uint8),
                                                                   requires_grad=False))

        def _load_wire(self, param, blob, weight_name, shard_id, expert_id):
            calls.append(("load", shard_id, expert_id, int(blob.numel())))

        def process_weights_after_loading(self, layer):
            layer.tessera_rows, layer.tessera_columns = 512, 256
            self._native = SimpleNamespace(launch_pair=("fake.symbol", "fake.decoder"))
            self._packed = SimpleNamespace(resident_bytes=lambda: 77)

    def build_fake_moe_method(scheme_decl, prefix, mode, layer):
        calls.append(("build", prefix, mode))
        return FakeMethod()

    path, request = _routed_request(panel, monkeypatch, tmp_path)
    wire = app.tp.read_bound(request["wire"])
    module = types.ModuleType("shape_time_fake_moe")
    module.build_fake_moe_method = build_fake_moe_method
    monkeypatch.setitem(sys.modules, "shape_time_fake_moe", module)
    monkeypatch.setattr(scheme, "MOE_BUILDERS",
                        {scheme.TESSERA_FP8: ("shape_time_fake_moe", "build_fake_moe_method")})
    monkeypatch.setattr(moe_route, "prepare_tessera_packed_moe_experts",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("research packed path is not the TP2 intake")))
    monkeypatch.setattr(moe_route, "prepare_tessera_packed_bf16_moe_experts",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("research packed path is not the TP2 intake")))
    layer, got, prep = worker.prepare_routed(
        dict(request, _wire_roles=app.tp.wire_facts(wire, request["scheme"])[1]), wire)
    assert isinstance(got, FakeMethod)
    assert calls[0] == ("build", "test.routed", "resident")
    assert calls[1][1]["intermediate_size_per_partition"] == 256
    assert calls[1][1]["num_experts"] == 8 and calls[1][1]["hidden_size"] == 256
    loads = [c for c in calls if c[0] == "load"]
    assert len(loads) == 8 * 3
    assert {(shard, expert) for _, shard, expert, _ in loads} == {
        (shard, expert) for expert in range(8) for shard in ("w1", "w3", "w2")}
    assert prep["builder"] == "shape_time_fake_moe.build_fake_moe_method"
    assert prep["local_shape"] == {"N": 512, "K": 256} and prep["topk"] == 8
    assert (prep["tp_rank"], prep["tp_degree"]) == (0, 2)
    assert prep["native_packed_bytes"] == 77


def test_worker_refuses_routed_tp1(panel, monkeypatch, tmp_path):
    scheme_decl, wire_binding = _routed_stack(tmp_path)
    doc, cell, packed = _routed_cell_contract()
    scope, runtime = _routed_scope_and_runtime(cell, packed)
    scope = dict(scope, tp_degree=1)
    runtime = dict(runtime, tp_degree=1)
    contract_binding = tp.file_binding(_save(tmp_path, "contract.json", packed, raw=True))
    path, request = _request_value(panel, monkeypatch, scope=scope, runtime=runtime,
                                   scheme_decl=scheme_decl, wire_binding=wire_binding,
                                   prefix="test.routed", contract_binding=contract_binding)
    wire = app.tp.read_bound(request["wire"])
    with pytest.raises(ValueError, match="TP2 only"):
        worker.prepare_routed(dict(request, _wire_roles=[]), wire)
