"""The MoE campaign closes only over the complete planned owner population."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tessera.serving.contract import load_serving_contract
from tessera.serving.moe_route import ACTIVATION_CONTRACT, GEMM_SYMBOL
from tessera.serving.telemetry import DECODER_TORCH_STOCK
from tessera.serving.scheme import (
    TESSERA_FP8, expert_role_declarations, validate_tessera_moe_scheme)

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "example/runtime@sha256:" + "1" * 64
TARGETS = ["model.layers.2.feed_forward.experts", "model.layers.5.feed_forward.experts"]


def _checker():
    spec = importlib.util.spec_from_file_location(
        "ts5_census_check", ROOT / "experiments" / "ts5_census_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        runtime_image=IMAGE, checkpoint=Path("/merged"), contract=contract, **kw)


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
