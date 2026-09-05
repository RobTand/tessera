"""The MoE campaign closes only over the complete planned owner population."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tessera.serving.contract import load_serving_contract
from tessera.serving.scheme import (
    ROUTES, MOE_GEMM_SYMBOL, TESSERA_FP8, expert_role_declarations, launch_pairs,
    validate_tessera_moe_scheme)

ACTIVATION_CONTRACT = ROUTES[TESSERA_FP8]["activation_contract"]
GEMM_SYMBOL = MOE_GEMM_SYMBOL
DECODER_TORCH_STOCK = next(iter(launch_pairs(
    TESSERA_FP8, structure="routed_moe", regime="decode", mode="resident")))[1]

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "example/runtime@sha256:" + "1" * 64
TARGETS = ["model.layers.2.feed_forward.experts", "model.layers.5.feed_forward.experts"]
SIDECARS = {"config.json": "a" * 64, "tessera_serving_manifest.json": "b" * 64}


def _checker():
    spec = importlib.util.spec_from_file_location(
        "ts5_census_check", ROOT / "experiments" / "ts5_census_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_census_cli_resolves_its_own_tools_beside_another_repository(tmp_path):
    """A regular tools package elsewhere must not hide this checkout's gate."""
    other = tmp_path / "tools"
    other.mkdir()
    (other / "__init__.py").write_text("")
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "ts5_census_check.py"), "--help"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--census" in result.stdout


def _fixture():
    plan = {name: {"grid": "E4M3", "q256": 1024} for name in TARGETS}
    plan["model.layers.0.conv.in_proj.weight"] = "PASSTHROUGH"
    groups, modules = {}, {}
    for target in TARGETS:
        scheme = {"family": TESSERA_FP8, "structure": "routed_moe", "grid": "E4M3",
            "body": "WINDOW", "plane": "CHANNEL", "experts": 2, "groups": {
                "w13": {"rows": 128, "columns": 128, "q256": 1024,
                    "roles": [["gate_proj", 64], ["up_proj", 64]], "wire_stride": 4096},
                "w2": {"rows": 128, "columns": 64, "q256": 1024,
                    "roles": [["down_proj", 128]], "wire_stride": 4096}}}
        groups[target] = {"targets": [target], "format": "TESSERA", "scheme": scheme}
        normalized = validate_tessera_moe_scheme(scheme, target)
        roles = []
        for expert in range(normalized["experts"]):
            for group, declaration in normalized["groups"].items():
                for role in expert_role_declarations(declaration):
                    roles.append({"expert": expert, "group": group, "role": role["roles"][0][0],
                        "q256": role["q256"], "grid": scheme["grid"], "family": scheme["family"],
                        "rows": role["rows"], "cols": role["columns"]})
        modules[target] = {"structure": "routed_moe", "family": scheme["family"],
            "grid": scheme["grid"], "q256": 1024, "experts": scheme["experts"], "roles": roles}
    config = {"architectures": ["Lfm2MoeForCausalLM"], "quantization_config": {
        "quant_method": "tessera", "config_groups": groups, "ignore": []}}
    total_units = sum(len(row["roles"]) for row in modules.values())
    manifest = {"modules": modules, "export_identity": {
        "runtime_image": IMAGE, "options": {"plan": copy.deepcopy(plan)}},
        "routed_moe": {"quantized_stacks": list(TARGETS), "quantized_logical_units": total_units},
        "totals": {"modules": len(TARGETS), "units": total_units}}
    records, owners = {}, {}
    for phase in ("decode", "prefill"):
        records[phase], owners[phase] = {}, {}
        for target in TARGETS:
            child = f"{target}.routed_experts"
            records[phase][child] = {"kind": "moe", "policy": f"{TESSERA_FP8}:resident",
                "symbol": f"{GEMM_SYMBOL}:TRITON", "decoder": DECODER_TORCH_STOCK,
                "contract": ACTIVATION_CONTRACT, "state": "served",
                "shape": "M1:N128:K128" if phase == "decode" else "M64:N128:K128"}
            owners[phase][child] = target
    census = {"schema": "tessera.serving.route_census/2", "checkpoint": "/merged",
        "checkpoint_sidecars": dict(SIDECARS),
        "runtime": {"image": IMAGE, "execution_mode": "eager"}, "compiled": False,
        "env": {"TESSERA_SERVE_MODE": "resident"}, "verdict": "served", "problems": [],
        "records": records, "record_owner": owners,
        "declared_names_mapped_to_module_space": True,
        "declared_name_mapping": {target: target for target in TARGETS},
        "device": {"capability": [12, 1]},
        "cell_launch_agreement": {"agrees": True}}
    contract = load_serving_contract()
    return plan, config, manifest, census, contract


def _check(case, **kw):
    plan, config, manifest, census, contract = case
    return _checker().check_census(plan, config, manifest, census,
        runtime_image=IMAGE, checkpoint=Path("/merged"), checkpoint_sidecars=SIDECARS,
        contract=contract, **kw)


def test_complete_population_passes_without_pretending_moe_is_attested():
    result = _check(_fixture())
    assert result["verdict"] == "passed"
    assert result["expected_owners"] == TARGETS
    assert result["cell_launch_agreement"]["agrees"] is None
    assert result["symbols"] == {"decode": [f"{GEMM_SYMBOL}:TRITON"],
                                  "prefill": [f"{GEMM_SYMBOL}:TRITON"]}


@pytest.mark.parametrize("phase", ["decode", "prefill"])
@pytest.mark.parametrize("defect", ["empty", "partial", "duplicate_owner", "extra", "owner_lie"])
def test_each_phase_requires_an_exact_owner_bijection(phase, defect):
    case = _fixture()
    census = case[3]
    records = census["records"][phase]
    first = next(iter(records))
    if defect == "empty":
        records.clear()
    elif defect == "partial":
        records.pop(first)
    elif defect == "duplicate_owner":
        records[first + ".duplicate"] = copy.deepcopy(records[first])
        census["record_owner"][phase][first + ".duplicate"] = TARGETS[0]
    elif defect == "extra":
        records["model.unplanned.experts.routed_experts"] = copy.deepcopy(records[first])
    else:
        census["record_owner"][phase][first] = TARGETS[1]
    with pytest.raises(ValueError):
        _check(case)


@pytest.mark.parametrize("defect", ["image", "compiled", "residency", "problems", "verdict",
    "phase", "mapping", "policy", "launch", "state", "shape_missing", "shape_same",
    "activation"])
def test_wrong_raw_receipt_context_is_refused(defect):
    case = _fixture()
    census = case[3]
    first = next(iter(census["records"]["decode"].values()))
    if defect == "image":
        census["runtime"]["image"] = "example/other@sha256:" + "2" * 64
    elif defect == "compiled":
        census["compiled"] = True
        census["runtime"]["execution_mode"] = "compiled"
    elif defect == "residency":
        census["env"]["TESSERA_SERVE_MODE"] = "streamed"
    elif defect == "problems":
        census["problems"] = ["failed"]
    elif defect == "verdict":
        census["verdict"] = "REFUSED"
    elif defect == "phase":
        del census["records"]["prefill"]
    elif defect == "mapping":
        census["declared_name_mapping"][TARGETS[0]] = "wrong.target"
    elif defect == "policy":
        first["policy"] = f"{TESSERA_FP8}:streamed"
    elif defect == "launch":
        first["symbol"] = "wrong.kernel"
    elif defect == "shape_missing":
        first.pop("shape")
    elif defect == "shape_same":
        first["shape"] = next(iter(census["records"]["prefill"].values()))["shape"]
    elif defect == "activation":
        first["contract"] = "wrong.activation"
    else:
        first["state"] = "failed"
    with pytest.raises(ValueError):
        _check(case)


def test_host_container_alias_and_encoder_image_do_not_change_serving_context():
    case = _fixture()
    case[3]["checkpoint"] = "/model"
    case[2]["export_identity"]["runtime_image"] = "encoder/image@sha256:" + "2" * 64
    assert _check(case)["verdict"] == "passed"


@pytest.mark.parametrize("phase,shape", [
    ("decode", "arbitrary"), ("decode", "M2:N128:K128"),
    ("decode", "M*:N128:K128"), ("decode", "M01:N128:K128"),
    ("decode", "M1:N0:K128"), ("decode", "M1:N256:K128"),
    ("prefill", "M64:N128:K256")])
def test_eager_shapes_prove_actual_regime_and_declared_geometry(phase, shape):
    case = _fixture()
    next(iter(case[3]["records"][phase].values()))["shape"] = shape
    with pytest.raises(ValueError, match="shape"):
        _check(case)


@pytest.mark.parametrize("defect", ["empty_plan", "different_plan", "extra_target", "duplicate_target",
    "missing_target", "manifest_target", "role_missing", "role_duplicate", "role_extra",
    "role_rung", "role_shape", "rung", "grid", "experts", "units", "stack_summary"])
def test_plan_config_and_projection_manifest_must_describe_one_population(defect):
    case = _fixture()
    plan, config, manifest, _, _ = case
    groups = config["quantization_config"]["config_groups"]
    module = manifest["modules"][TARGETS[0]]
    if defect == "empty_plan":
        plan.clear()
    elif defect == "different_plan":
        manifest["export_identity"]["options"]["plan"] = {}
    elif defect == "extra_target":
        groups["extra"] = copy.deepcopy(groups[TARGETS[0]])
        groups["extra"]["targets"] = ["model.unplanned.experts"]
    elif defect == "duplicate_target":
        groups["extra"] = copy.deepcopy(groups[TARGETS[0]])
    elif defect == "missing_target":
        groups.pop(TARGETS[0])
    elif defect == "manifest_target":
        manifest["modules"].pop(TARGETS[0])
    elif defect == "role_missing":
        module["roles"].pop()
    elif defect == "role_duplicate":
        module["roles"].append(copy.deepcopy(module["roles"][0]))
    elif defect == "role_extra":
        module["roles"][0]["role"] = "wrong_projection"
    elif defect == "role_rung":
        module["roles"][0]["q256"] = 768
    elif defect == "role_shape":
        module["roles"][0]["cols"] += 1
    elif defect == "rung":
        groups[TARGETS[0]]["scheme"]["groups"]["w2"]["q256"] = 768
    elif defect == "grid":
        module["grid"] = "BF16"
    elif defect == "experts":
        module["experts"] += 1
    elif defect == "units":
        manifest["totals"]["units"] += 1
    else:
        manifest["routed_moe"]["quantized_stacks"] = []
    with pytest.raises(ValueError):
        _check(case)


def _promote(case):
    cells = case[4]["lane_eligibility"]["cells"]
    for regime in ("decode", "batch"):
        cells.append({"id": f"synthetic_moe_{regime}", "platform": "sm_121",
            "structure": "routed_moe", "family": "TESSERA_E4M3_K1", "regime": regime,
            "rungs_q256": [1024], "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
            "runtime": {"image": IMAGE, "execution_modes": ["eager"]},
            "executes": [{"symbol": GEMM_SYMBOL, "decoder": DECODER_TORCH_STOCK}]})


def test_promotion_replays_current_cells_instead_of_embedded_agreement():
    case = _fixture()
    with pytest.raises(ValueError, match="attest"):
        _check(case, require_attested=True)
    _promote(case)
    case[3]["cell_launch_agreement"] = {"agrees": False}
    assert _check(case, require_attested=True)["verdict"] == "passed"
    case[4]["lane_eligibility"]["cells"][-1]["runtime"]["image"] = (
        "example/other@sha256:" + "2" * 64)
    with pytest.raises(ValueError, match="attest"):
        _check(case, require_attested=True)


def test_duplicate_json_keys_are_refused_before_population_is_collapsed(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"records": {}, "records": {}}')
    with pytest.raises(ValueError, match="duplicate"):
        _checker().read_json(path)


@pytest.mark.parametrize("defect", ["missing", "null", "wrong", "extra"])
def test_raw_sidecar_identity_must_match_supplied_files(defect):
    case = _fixture()
    if defect == "missing":
        case[3].pop("checkpoint_sidecars")
    elif defect == "null":
        case[3]["checkpoint_sidecars"]["tessera_serving_manifest.json"] = None
    elif defect == "wrong":
        case[3]["checkpoint_sidecars"]["config.json"] = "c" * 64
    else:
        case[3]["checkpoint_sidecars"]["other.json"] = "c" * 64
    with pytest.raises(ValueError, match="sidecar"):
        _check(case)


def test_census_stamps_exact_sidecar_bytes_and_explicit_absent_manifest(tmp_path):
    from tools import tessera_route_census
    config = tmp_path / "config.json"
    config.write_text('{ "x": 1 }\n')
    assert tessera_route_census.checkpoint_sidecar_hashes(tmp_path) == {
        "config.json": hashlib.sha256(config.read_bytes()).hexdigest(),
        "tessera_serving_manifest.json": None}
    manifest = tmp_path / "tessera_serving_manifest.json"
    manifest.write_text('{}\n')
    assert tessera_route_census.checkpoint_sidecar_hashes(tmp_path) == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (config, manifest)}


@pytest.mark.parametrize("changed", ["config.json", "tessera_serving_manifest.json"])
def test_sidecar_seal_refuses_changes_during_serve(tmp_path, changed):
    from tools import tessera_route_census
    for name in SIDECARS:
        (tmp_path / name).write_text('{}\n')
    before = tessera_route_census.checkpoint_sidecar_hashes(tmp_path)
    (tmp_path / changed).write_text('{}\n\n')
    with pytest.raises(ValueError, match="sidecar"):
        tessera_route_census.checkpoint_sidecar_hashes(tmp_path, expected=before)


@pytest.mark.parametrize("refused", [False, "problems", "sidecar_whitespace"])
def test_cli_fingerprints_inputs_and_preserves_existing_receipts(tmp_path, refused):
    plan, config, manifest, census, _ = _fixture()
    if refused == "problems":
        census["problems"] = ["incomplete run"]
    files = {"plan": tmp_path / "plan.json", "config": tmp_path / "config.json",
             "manifest": tmp_path / "tessera_serving_manifest.json", "census": tmp_path / "census.json"}
    for name, value in (("plan", plan), ("config", config), ("manifest", manifest)):
        files[name].write_text(json.dumps(value))
    census["checkpoint_sidecars"] = {files[name].name: hashlib.sha256(files[name].read_bytes()).hexdigest()
                                     for name in ("config", "manifest")}
    files["census"].write_text(json.dumps(census))
    if refused == "sidecar_whitespace":
        files["config"].write_text(files["config"].read_text() + "\n")
    out = tmp_path / "check.json"
    command = [sys.executable, str(ROOT / "experiments" / "ts5_census_check.py"),
               "--plan", str(files["plan"]), "--checkpoint", str(tmp_path),
               "--census", str(files["census"]), "--runtime-image", IMAGE, "--out", str(out)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == int(bool(refused)), result.stderr
    receipt = json.loads(out.read_text())
    assert receipt["verdict"] == ("REFUSED" if refused else "passed")
    if not refused:
        assert receipt["inputs"] == {name: {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in files.items()}
        assert receipt["current_contract_sha256"]
    before = out.read_bytes()
    rerun = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert rerun.returncode != 0
    assert out.read_bytes() == before
