"""Shared CPU timing-panel fixtures; synthetic records claim no GPU qualification."""
from __future__ import annotations
import copy
import gzip
import hashlib
from pathlib import Path

import pytest

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


def build_dense_fixture_panel(tmp_path, blob, declaration):
    """The CPU fixture panel body; the ``panel`` fixture delegates here so
    sibling tests can build the same receipt without calling a fixture."""
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
    verifier = tmp_path / "_install_identity.py"
    verifier.write_text("# CPU fixture verifier identity; never executed.\n")
    evidence = {
        "runtime": save(tmp_path, "runtime.json", runtime),
        "runtime_origins": save(tmp_path, "origins.json", {"install_verifier": tp.file_binding(verifier), "package_root": runtime["package_root"], "installation": {"module": "tessera", "distribution": "tessera-quant", "expected_commit": runtime["tessera_commit"], "installed_commit": runtime["tessera_commit"], "origin": runtime["package_root"]+"/__init__.py", "verified_files": 43}, "modules": {name: {"path": str(Path(runtime["package_root"]) / ("__init__.py" if name == "tessera" else name.removeprefix("tessera.").replace(".", "/") + ".py")), "bytes": 1, "sha256": "a"*64} for name in tp.RUNTIME_MODULES}}),
        "producer": save(tmp_path, "producer.json", {"schema": "tessera.native_panel_producer_identity.v1", "commit": "3" * 40, "commit_source": "sealed_checkout", "source_tree_sha256": "6"*64, "source_tree_members": 1, "tool_source_sha256": "4" * 64}),
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


@pytest.fixture
def panel(tmp_path, canonical_wire):
    return build_dense_fixture_panel(tmp_path, blob=canonical_wire[0],
                                     declaration=canonical_wire[1])
