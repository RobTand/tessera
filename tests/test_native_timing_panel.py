"""CPU evidence controls; synthetic records do not claim GPU qualification."""
from __future__ import annotations
import copy
import gzip
import json
import hashlib
import importlib.abc
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tessera.serving import timing_panel as tp, census_plan, contract, scheme


def save(root, name, value, raw=False):
    path = root / name
    path.write_bytes(value if raw else tp.canonical(value))
    return tp.file_binding(path)


@pytest.fixture(scope="module")
def canonical_wire():
    import torch
    from tessera.alphabet import E4M3_GRID
    from tessera.export import encode_linear_planes
    from tessera.fused_frame import pack_fused
    generator = torch.Generator().manual_seed(688)
    weight = torch.randn(16, 128, generator=generator) * 0.02
    exported, _, _ = encode_linear_planes(weight, grid=E4M3_GRID, q256=1024,
                                          name="weight", verify=False)
    blob = pack_fused([("weight", 16, exported.blob)])
    declaration = {"family": scheme.TESSERA_FP8, "structure": "dense", "grid": "E4M3",
                   "body": "WINDOW", "plane": "CHANNEL", "q256": 1024,
                   "rows": 16, "columns": 128, "roles": [["weight", 16]], "wire_bytes": len(blob)}
    return blob, declaration


@pytest.fixture
def panel(tmp_path, canonical_wire):
    blob, declaration = canonical_wire
    doc = copy.deepcopy(contract.load_serving_contract())
    family = contract.PAYLOAD_FAMILY_BY_ROUTE[scheme.TESSERA_FP8]
    cell = next(c for c in doc["lane_eligibility"]["cells"] if c["platform"] == "sm_121"
                and c["family"] == family and c["structure"] == "dense" and c["regime"] == "batch"
                and "resident" in contract.cell_residency_modes(c)
                and "eager" in contract.cell_runtime_scope(c)[1]
                and contract.cell_is_device_backed(c))
    cell["runtime"].update(tessera_commit="1" * 40, serving_source_sha256="2" * 64)
    packed_contract = tp.canonical(doc)
    runtime = {"image": cell["runtime"]["image"], "tessera_commit": "1" * 40,
               "serving_source_sha256": "2" * 64, "contract_sha256": hashlib.sha256(packed_contract).hexdigest(),
               "platform": "sm_121", "torch": cell["runtime"]["torch"], "vllm": cell["runtime"]["vllm"],
               "serve_flags": {"TESSERA_SERVE_MODE": "resident"}, "residency": "resident",
               "execution_mode": "eager", "tp_rank": 0, "tp_degree": 1, "package_root": "/CPU-fixture/tessera"}
    scope = {"route": scheme.TESSERA_FP8, "grid": "E4M3", "q256": 1024, "structure": "dense",
             "mode": "resident", "execution_mode": "eager", "regime": "batch", "tp_degree": 1,
             "requested_platform": "sm_121", "shape": {"M": 512, "N": 16, "K": 128}}
    plan = census_plan.build_census_plan([scope], raw_contract=packed_contract)
    launch = next(x for x in scheme.route_launches(scheme.TESSERA_FP8, structure="dense", regime="batch", mode="resident")
                  if x["lane"] and {"symbol": x["symbol"], "decoder": x["decoder"]} in cell["executes"])
    _, roles = tp.wire_facts(blob, declaration)
    record = {"kind": "dense", "policy": scheme.TESSERA_FP8 + ":resident", "state": "served",
              "symbol": launch["symbol"], "decoder": launch["decoder"], "shape": "M512:N16:K128",
              "contract": scheme.ROUTES[scheme.TESSERA_FP8]["activation_contract"], "platform": "sm_121"}
    samples = [1.0, 2.0, 3.0, 4.0]
    trace = {"traceEvents": [{"cat": "kernel", "name": "CPU trace fixture", "dur": 1.0}]}
    native = next(x for x in doc["native_extensions"] if x["module_name_prefix"] == launch["lane"])
    filename = native["filename_glob"].replace("*", "CPU-fixture")
    telemetry = {"interval_unix": [10.0, 11.0], "fast_power_samples": [[10.5, 40.0]],
                 "netdata": {box: {context: {"query": "CPU fixture", "raw_response": {"view": {}, "result": {"labels": ["time", "fixture"], "data": [[10, [0, 0, 0]]] }},
                                            "returned_view": {}} for context in tp.NETDATA_CONTEXTS}
                             for box in ("sparky", "sparklina")}}
    evidence = {
        "runtime": save(tmp_path, "runtime.json", runtime),
        "runtime_origins": save(tmp_path, "origins.json", {"package_root": runtime["package_root"], "modules": {name: {"path": str(Path(runtime["package_root"]) / ("__init__.py" if name == "tessera" else name.removeprefix("tessera.").replace(".", "/") + ".py")), "bytes": 1, "sha256": "a"*64} for name in tp.RUNTIME_MODULES}}),
        "producer": save(tmp_path, "producer.json", {"commit": "3" * 40, "tool_source_sha256": "4" * 64}),
        "contract": save(tmp_path, "contract.json", packed_contract, True),
        "wire": save(tmp_path, "wire.bin", blob, True),
        "preparation": save(tmp_path, "preparation.json", {"builder": "tessera.serving.lane.build_tessera_method",
            "wire_sha256": hashlib.sha256(blob).hexdigest(), "roles": roles, "shape": scope["shape"],
            "tp_rank": 0, "tp_degree": 1, "grid": "E4M3", "native_packed_bytes": 1}),
        "samples": save(tmp_path, "samples.json", {"samples_ms": samples, "warmup_iterations": 1,
                                                  "interval_unix": [10.0, 11.0]}),
        "routes": save(tmp_path, "routes.json", {"records": [record] * len(samples)}),
        "trace": save(tmp_path, "trace.json.gz", gzip.compress(tp.canonical(trace)), True),
        "telemetry": save(tmp_path, "telemetry.json", telemetry),
        "native_binary": save(tmp_path, filename, b"\x7fELF CPU binary fixture", True),
    }
    return {"schema": tp.SCHEMA, "status": "measured", "claims": dict(tp.CLAIMS), "runtime": runtime,
            "plan": plan, "rows": [{"scope_id": plan["rows"][0]["id"], "prefix": "test.dense",
                                    "scheme": declaration, "timing": tp.timing_summary(samples), "cell_id": cell["id"]}],
            "evidence": evidence,
            "energy": {"status": "hold", "reason": "cross_host_clock_alignment_unqualified", "reference_w": 140}}


def rewrite(panel, key, fn):
    bound = panel["evidence"][key]
    value = tp.json_bytes(tp.read_bound(bound))
    fn(value)
    panel["evidence"][key] = save(Path(bound["path"]).parent, Path(bound["path"]).name, value)


def test_true_even_median_and_positive_binding(panel):
    result = tp.validate_panel(panel, expected_runtime=copy.deepcopy(panel["runtime"]))
    assert result["timing"]["median_ms"] == 2.5
    assert result["energy_status"] == "hold"
    assert panel["plan"]["gpu_executed"] is False


@pytest.mark.parametrize("fault", ["producer_as_runtime", "wrong_cell_source", "missing_code", "pair", "state",
    "shape", "activation", "platform", "wire", "native", "raw_samples", "fake_median",
    "nan", "sample_bool", "sample_count", "prep_wire", "prep_geometry", "prep_roles", "no_trace",
    "single_box", "missing_context", "wrong_interval", "no_fast_power", "energy", "duplicate_row", "extra_field", "noncanonical_wire", "cpu_only_trace", "float_count", "numeric_claim", "empty_netdata", "error_netdata", "missing_scope", "wrong_trace_type"])
def test_positive_receipt_refuses_false_evidence(panel, fault):
    expected = copy.deepcopy(panel["runtime"])
    if fault == "producer_as_runtime": panel["runtime"]["tessera_commit"] = "3" * 40
    elif fault in ("wrong_cell_source", "missing_code"):
        def change(doc):
            cell = next(c for c in doc["lane_eligibility"]["cells"] if c["id"] == panel["rows"][0]["cell_id"])
            if fault == "missing_code":
                cell["runtime"].pop("tessera_commit");cell["runtime"].pop("serving_source_sha256")
            else: cell["runtime"]["serving_source_sha256"] = "5" * 64
        rewrite(panel, "contract", change)
        panel["runtime"]["contract_sha256"] = panel["evidence"]["contract"]["sha256"]
        expected["contract_sha256"] = panel["runtime"]["contract_sha256"]
        rewrite(panel, "runtime", lambda v: v.update(panel["runtime"]))
        panel["plan"] = census_plan.build_census_plan([panel["plan"]["rows"][0]["scope"]], raw_contract=tp.read_bound(panel["evidence"]["contract"]))
        panel["rows"][0]["scope_id"] = panel["plan"]["rows"][0]["id"]
    elif fault in ("pair", "state", "shape", "activation", "platform"):
        field = {"pair": "symbol", "state": "state", "shape": "shape", "activation": "contract", "platform": "platform"}[fault]
        rewrite(panel, "routes", lambda v: v["records"][0].update({field: "wrong"}))
    elif fault in ("wire", "native", "raw_samples", "no_trace"):
        key = {"native": "native_binary", "raw_samples": "samples", "no_trace": "trace"}.get(fault, fault)
        Path(panel["evidence"][key]["path"]).write_bytes(b"tampered")
    elif fault == "float_count": panel["rows"][0]["timing"]["n"] = 4.0
    elif fault == "numeric_claim": panel["claims"]["certifies_placement"] = 0
    elif fault == "empty_netdata": rewrite(panel, "telemetry", lambda v: v["netdata"]["sparky"][next(iter(tp.NETDATA_CONTEXTS))]["raw_response"].update(result={"labels": [], "data": []}))
    elif fault == "error_netdata": rewrite(panel, "telemetry", lambda v: v["netdata"]["sparky"][next(iter(tp.NETDATA_CONTEXTS))]["raw_response"].update(result={"error": "unknown context"}))
    elif fault == "missing_scope": panel["plan"]["rows"][0].pop("scope")
    elif fault == "wrong_trace_type":
        bound = panel["evidence"]["trace"];panel["evidence"]["trace"] = save(Path(bound["path"]).parent, "trace.json.gz", gzip.compress(tp.canonical([])), True)
    elif fault == "fake_median": panel["rows"][0]["timing"]["median_ms"] = 3
    elif fault in ("nan", "sample_bool", "sample_count"):
        rewrite(panel, "samples", lambda v: v.update(samples_ms=[True, 1, 2] if fault == "sample_bool" else [1, 2]))
        if fault == "nan": panel["rows"][0]["timing"]["median_ms"] = float("nan")
    elif fault.startswith("prep_"):
        field, value = {"prep_wire": ("wire_sha256", "5" * 64), "prep_geometry": ("shape", {}),
                        "prep_roles": ("roles", [])}[fault]
        rewrite(panel, "preparation", lambda v: v.update({field: value}))
    elif fault in ("single_box", "missing_context", "wrong_interval", "no_fast_power"):
        def change(v):
            if fault == "single_box": v["netdata"].pop("sparky")
            elif fault == "missing_context": v["netdata"]["sparky"].pop(next(iter(tp.NETDATA_CONTEXTS)))
            elif fault == "wrong_interval": v["interval_unix"] = [9, 10]
            else: v["fast_power_samples"] = []
        rewrite(panel, "telemetry", change)
    elif fault == "energy": panel["energy"]["status"] = "qualified"
    elif fault == "duplicate_row": panel["rows"].append(copy.deepcopy(panel["rows"][0]))
    elif fault == "extra_field": panel["rows"][0]["default_topology"] = True
    elif fault == "noncanonical_wire":
        bound = panel["evidence"]["wire"];raw = bytearray(tp.read_bound(bound));raw[-1] ^= 1
        panel["evidence"]["wire"] = save(Path(bound["path"]).parent, "wire.bin", bytes(raw), True)
        rewrite(panel, "preparation", lambda v: v.update(wire_sha256=panel["evidence"]["wire"]["sha256"]))
    elif fault == "cpu_only_trace":
        bound = panel["evidence"]["trace"]
        panel["evidence"]["trace"] = save(Path(bound["path"]).parent, "trace.json.gz", gzip.compress(tp.canonical({"traceEvents": [{"cat": "cpu_op"}]})), True)
    with pytest.raises((ValueError, OSError)):
        tp.validate_panel(panel, expected_runtime=expected)

