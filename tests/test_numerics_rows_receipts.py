"""Receipts keep exact NumPy values and completed H1 sections."""
import json
import sys

import numpy as np
import pytest

from experiments.numerics_rows.common import write_json
from experiments.numerics_rows import h1
from experiments.numerics_rows import row
from experiments.numerics_rows.common import IMAGES, Invalid, artifact_limit, check_runtime


def test_numpy_scalars_and_arrays_are_exact_json_values(tmp_path):
    path = tmp_path / "receipt.json"
    write_json(path, {"count": np.int64(2**63 - 1), "value": np.float32(0.1),
                      "passed": np.bool_(True), "array": np.array([[0, -2**63]], dtype=np.int64),
                      "float_array": np.array([2**-149, -0.0], dtype=np.float32)})
    result = json.loads(path.read_text())
    assert result == {"count": 2**63 - 1, "value": float(np.float32(0.1)), "passed": True,
                      "array": [[0, -2**63]], "float_array": [2**-149, -0.0]}
    assert type(result["count"]) is int and type(result["passed"]) is bool
    assert np.signbit(result["float_array"][1])


@pytest.mark.parametrize("value", [np.float32("nan"), np.float64("inf"), np.array([float("-inf")])])
def test_nonfinite_numpy_values_do_not_replace_a_finished_receipt(tmp_path, value):
    path = tmp_path / "receipt.json"
    write_json(path, {"finished": True})
    with pytest.raises(ValueError, match="Out of range float values"):
        write_json(path, {"value": value})
    assert json.loads(path.read_text()) == {"finished": True}


def test_later_h1_error_keeps_finished_sections(tmp_path, monkeypatch):
    def checks(args, report):
        report["P1"] = {"samples": np.int64(1_000_000), "mismatches": np.int64(0)}
        report["preconditions_confirmed"] = True
        raise RuntimeError("later section failed")

    monkeypatch.setattr(h1, "run_checks", checks)
    monkeypatch.setattr(sys, "argv", ["h1", "--out", str(tmp_path), "--image", "G"])
    with pytest.raises(SystemExit) as stopped:
        h1.main()
    assert stopped.value.code == 3
    result = json.loads((tmp_path / "h1.json").read_text())
    assert result["P1"] == {"samples": 1_000_000, "mismatches": 0}
    assert result["preconditions_confirmed"] is False
    assert result["error"]["type"] == "RuntimeError"
    assert result["error"]["message"] == "later section failed"


def population_cases():
    pair = ["tessera::fused_window_dense", "native_fused_window_dense"]
    population = {"modules": [{"root": "/wires", "name": name, "library": "e4m3",
                               "contract_family": "TESSERA_E4M3_K1", "pair": pair}
                              for name in ("module-a", "module-b")]}
    manifest = {"modules": [{"root": "/wires", "name": name, "structure": "dense", "q256": 1024,
                             "cuts": [{"tp": {"size": size, "rank": rank}, "m_configs": [{"M": 1}, {"M": 2}]}
                                      for size, rank in ((1, 0), (2, 0), (2, 1))]}
                            for name in ("module-a", "module-b")]}
    contract = {"lane_eligibility": {"cells": [
        {"id": regime, "family": "TESSERA_E4M3_K1", "structure": "dense", "regime": regime,
         "runtime": {"image": "sha256:image"}, "rungs_q256": [1024],
         "executes": [{"symbol": pair[0], "decoder": pair[1]}]}
        for regime in ("decode", "batch")]}}
    return contract, population, manifest


def test_expected_units_exist_before_a_loader_runs():
    contract, population, manifest = population_cases()
    groups = row.planned_units(contract, "image", population, manifest)
    assert len(groups) == 6
    units = [unit for _, _, _, group in groups for unit in group]
    assert len(units) == 12 and all(unit["outcome"] == "STOPPED" for unit in units)
    assert {unit["cell"] for unit in units} == {"decode", "batch"}


@pytest.mark.parametrize("omitted", ("module", "cut", "configs", "pair"))
def test_manifest_cannot_omit_an_expected_unit(omitted):
    contract, population, manifest = population_cases()
    if omitted == "module":
        manifest["modules"][1]["name"] = "wrong-module"
    elif omitted == "cut":
        manifest["modules"][0]["cuts"].pop()
    elif omitted == "configs":
        manifest["modules"][0]["cuts"][0]["m_configs"] = []
    else:
        population["modules"][0]["pair"][1] = "wrong-decoder"
    with pytest.raises(Invalid):
        row.planned_units(contract, "image", population, manifest)


def test_adapter_failure_keeps_unrun_units_stopped(tmp_path, monkeypatch):
    contract, population, manifest = population_cases()
    def run(args, receipt):
        groups = row.planned_units(contract, "image", population, manifest)
        receipt["units"] = [unit for _, _, _, group in groups for unit in group]
        receipt["units"][0]["completed"] = True
        raise RuntimeError("second adapter failed")

    argv = ["row", "--image", "G", "--name", "test", "--out", str(tmp_path)]
    for name in ("source", "manifest", "population", "build", "q1", "h1", "rule", "freeze"):
        argv.extend(("--" + name, str(tmp_path / name)))
    monkeypatch.setattr(row, "run", run)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stopped:
        row.main()
    assert stopped.value.code == 3
    receipt = json.loads((tmp_path / "numerics_row.json").read_text())
    assert len(receipt["units"]) == 12
    assert all(unit["outcome"] == "STOPPED" for unit in receipt["units"][1:])
    assert all(cell["verdict"] == "UNATTESTED" for cell in receipt["cells"])
    assert receipt["row"]["stop_cause"] == "second adapter failed"


def test_observation_cannot_claim_decoder_bit_exact():
    unit = row.new_unit({"role": "observation"})
    unit["completed"] = True
    row.finish_unit(unit)
    assert not unit["decoder_bit_exact"]


@pytest.mark.parametrize("mutation", (None, "id", "digest", "version", "missing"))
def test_runtime_checks_the_observed_image(tmp_path, monkeypatch, mutation):
    expected = IMAGES["G"]
    record = {"Id": "sha256:" + expected["local_id"] + "0" * 52, "Digest": expected["image_digest"]}
    if mutation == "id":
        record["Id"] = "sha256:wrong-image"
    if mutation == "digest":
        record["Digest"] = "sha256:wrong-digest"
    path = tmp_path / "inspect.json"
    path.write_text(json.dumps([record]))
    monkeypatch.setenv("TESSERA_NUMERICS_IMAGE_INSPECT", str(path))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "wrong-version" if mutation == "version" else expected[name])
    if mutation == "missing":
        monkeypatch.delenv("TESSERA_NUMERICS_IMAGE_INSPECT")
    if mutation is None:
        result = check_runtime("G")
        assert result["local_id"] == record["Id"][7:] and result["image_inspect_sha256"]
    else:
        with pytest.raises(ValueError):
            check_runtime("G")


@pytest.mark.parametrize("config,expected", (({"swiglu_limit": 4.0}, 4.0),
    ({"swiglu_limit": 2.0, "text_config": {"swiglu_limit": 8.0}}, 8.0), ({}, float("inf"))))
def test_limit_comes_from_the_artifact_config(tmp_path, config, expected):
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert artifact_limit(tmp_path) == expected


def test_profiler_names_resolve_to_exact_sass_symbols():
    symbols, unresolved = row.resolve_symbols(["_Zkernel0", "_Zkernel2"],
        ["void routed_fused_kernel<0>( Params )", "quantizer", "_Zkernel2"],
        ["void routed_fused_kernel<0>(Params)", "void routed_fused_kernel<2>(Params)"])
    assert symbols == ["_Zkernel0", "_Zkernel2"] and unresolved == []
    assert row.resolve_symbols(["_Zkernel0"], ["void routed_fused_kernel<1>(Params)"],
        ["void routed_fused_kernel<0>(Params)"]) == ([], ["void routed_fused_kernel<1>(Params)"])


@pytest.mark.parametrize("different", (None, "symbol", "binary", "rung", "tp", "pair"))
def test_separation_attaches_only_to_the_same_compiled_function(different):
    import copy
    scope = {"symbol": "pair", "decoder": "decoder", "rung": 1024, "tp": {"rank": 0, "size": 1}}
    source, target = row.new_unit(scope), row.new_unit(scope)
    identity = [{"symbol": "_Zrouted_fused_kernel0", "library_so_sha256": "binary-a"}]
    source.update(outcome="PASS", separation_census_complete=True, m_config={"M": 64, "compiled_kernel_symbols": identity})
    source["exact"]["separation"]["records"] = [{"elements": 10, "both": 10}]
    target.update(needs_separation_transfer=True, m_config={"M": 2, "compiled_kernel_symbols": copy.deepcopy(identity)})
    if different == "symbol":
        target["m_config"]["compiled_kernel_symbols"][0]["symbol"] += "different"
    elif different == "binary":
        target["m_config"]["compiled_kernel_symbols"][0]["library_so_sha256"] = "binary-b"
    elif different == "rung":
        target["rung"] = 1088
    elif different == "tp":
        target["tp"] = {"rank": 1, "size": 2}
    elif different == "pair":
        target["decoder"] = "other-decoder"
    row.attach_separation([source, target])
    if different:
        assert target["unattested"] and "separation_from" not in target
    else:
        assert target["separation_from"]["M"] == 64 and not target["unattested"]
        assert target["exact"]["separation"] == source["exact"]["separation"]


def test_receipt_aggregates_keep_each_array_and_separation_record():
    unit = row.new_unit({"role": "unit"})
    unit["exact"]["separation"]["records"] = [{"stage": stage, "elements": 8, "lower_excluded": 8, "upper_excluded": 8, "both": 8,
        "by_sign": {"positive": [4, 4], "negative": [4, 4]}} for stage in ("gate", "up")]
    unit["exact"]["scale_index"]["arrays"] = [{"scale": scale, "base": 8, "families": ["plain"], "passes_per_family": 2,
        "passes_total": 2, "reads": 8, "reads_covered": 8, "uncovered_reads": [], "mismatches": 0, "launches": 16}
        for scale in ("row", "a_s")]
    row.finish_unit(unit)
    separation, scales = unit["exact"]["separation"], unit["exact"]["scale_index"]
    assert separation["elements"] == separation["both"] == 16
    assert separation["by_sign"] == {"positive": [8, 8], "negative": [8, 8]}
    assert scales["scale"] == ["a_s", "row"] and scales["base"] == [8]
    assert scales["reads"] == scales["reads_covered"] == 16 and scales["launches"] == 32
    assert scales["passes_total"] == scales["passes_per_family"]["plain"] == 4


def test_rule_receipt_requires_decisions_and_records_the_h1_job(tmp_path):
    path = tmp_path / "h1.json"
    h1 = {"flux_job_id": "test-job", "preconditions_confirmed": True, "P2": {"result": "PASS"}}
    path.write_text(json.dumps(h1))
    freeze = {"decisions": {f"D{i}": "reviewed" for i in range(1, 9)}}
    receipt = row.rule_receipt(tmp_path / "rule.md", freeze, h1, path)
    assert set(receipt["preconditions"]) == {"P1", "P2", "P3", "P4", "P5"}
    assert all(result == {"result": "PASS", "h1_job": "test-job"} for result in receipt["preconditions"].values())
    with pytest.raises(Invalid, match="D1 through D8"):
        row.rule_receipt(tmp_path / "rule.md", {}, h1, path)
