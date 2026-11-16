"""Receipts keep exact NumPy values and completed H1 sections."""
import json
import sys
import copy
from types import SimpleNamespace

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


@pytest.mark.parametrize("size,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("fails", [False, True])
def test_routed_build_passes_rank_local_width_and_restores_full_width(monkeypatch, size, rank, fails):
    from experiments import routed_pair_oracle as oracle
    from experiments.numerics_rows.routed import build_served

    config = SimpleNamespace(intermediate_size=512,
                             moe_parallel_config=SimpleNamespace(tp_size=size, tp_rank=rank))
    monkeypatch.setattr(oracle, "INTER", 512)
    calls = []
    def loader(family, scheme, blobs, scales, experts, cfg, limit, prefix):
        calls.append((oracle.INTER, cfg.intermediate_size, cfg.moe_parallel_config.tp_rank))
        if fails:
            raise ValueError("loader failure")
        return "layer", "method", "info"
    monkeypatch.setattr(oracle, "build_after", loader)
    if fails:
        with pytest.raises(ValueError, match="loader failure"):
            build_served({}, {}, {}, {}, 2, config, 10.0, "synthetic.experts")
    else:
        assert build_served({}, {}, {}, {}, 2, config, 10.0, "synthetic.experts") == ("layer", "method", "info")
    assert calls == [(512 // size, 512, rank)]
    assert oracle.INTER == 512


@pytest.mark.parametrize("where,owner", [("inside", "injected-launch"), ("outside", None)])
def test_cuda_child_attribution_is_present_in_parent_record(where, owner):
    child = {"where": where, "attributed_to": owner, "boundaries": [{"call": "injected-launch"}]}
    run = SimpleNamespace(returncode=0, stdout="runtime log\n" + json.dumps(child) + "\n", stderr="assert log")
    record = h1.cuda_control_record(where, run)
    assert record["attributed_to"] == owner and record["attribution_confirmed"]
    assert record["boundaries"] == child["boundaries"]
    run.stdout = json.dumps({"where": where})
    assert not h1.cuda_control_record(where, run)["attribution_confirmed"]


def completed_gpu_receipt():
    return {"flux_job_id": "gpu-job", "runtime": {"libraries": ["e4m3"]},
            "P1": {"samples": 1_000_000, "mismatches": 0}, "P2": {"result": "PASS"},
            "P3": {"e4m3": {"library_so_sha256": "a" * 64, "probe_sha256": "b" * 64,
                            "matched": False, "status": "DIFFERENT"}},
            "libraries": {"e4m3": {"P4": True, "P5": True, "routed": {"P4": True, "P5": True},
                                    "subnormal_operand": {"result": "UNATTESTED"}}},
            "negative_controls": [{"flagged_exactly": True}],
            "cuda_controls": [{"where": where, "exit_status": 0, "attribution_confirmed": True,
                               "attributed_to": "injected-launch" if where == "inside" else None}
                              for where in ("inside", "outside")], "preconditions_confirmed": False}


def completed_p3_receipt():
    return {"e4m3": {"library_so_sha256": "a" * 64, "probe_sha256": "b" * 64,
                     "matched": True, "status": "MATCHED", "unresolved": [], "differences": [],
                     "mode0_instantiations": 1, "quotients": 1, "quotients_examined": 1}}


@pytest.mark.parametrize("mutation", ["library_so_sha256", "probe_sha256", "missing", "gpu"])
def test_cpu_p3_confirmation_requires_the_h1_binary_and_completed_gpu_checks(mutation):
    report, proof = completed_gpu_receipt(), completed_p3_receipt()
    h1.confirm_p3(report, proof)
    assert report["preconditions_confirmed"] and report["gpu_preconditions_confirmed"]
    assert report["P3_at_gpu_run"]["e4m3"]["status"] == "DIFFERENT"
    changed, proof = completed_gpu_receipt(), copy.deepcopy(proof)
    if mutation in ("library_so_sha256", "probe_sha256"):
        proof["e4m3"][mutation] = "c" * 64
    elif mutation == "missing":
        proof.clear()
    else:
        changed["libraries"]["e4m3"]["P4"] = False
    with pytest.raises(ValueError):
        h1.confirm_p3(changed, proof)
    assert changed["preconditions_confirmed"] is False


@pytest.mark.parametrize("field,value", [("matched", False), ("status", "UNSUPPORTED"),
    ("unresolved", ["unknown"]), ("differences", [{"changed": True}]), ("mode0_instantiations", 0),
    ("quotients", 0), ("quotients_examined", 2)])
def test_cpu_p3_confirmation_refuses_an_incomplete_proof(field, value):
    report, proof = completed_gpu_receipt(), completed_p3_receipt()
    h1.confirm_p3(report, proof)
    assert report["preconditions_confirmed"]
    proof["e4m3"][field] = value
    h1.confirm_p3(report, proof)
    assert report["preconditions_confirmed"] is False


def test_h1_can_confirm_cpu_p3_without_loading_cuda(tmp_path, monkeypatch):
    gpu, proof = tmp_path / "gpu.json", tmp_path / "p3.json"
    write_json(gpu, completed_gpu_receipt())
    write_json(proof, completed_p3_receipt())
    def no_gpu(*args):
        raise AssertionError("confirmation must not run GPU checks")
    monkeypatch.setattr(h1, "run_checks", no_gpu)
    monkeypatch.setattr(sys, "argv", ["h1", "--gpu-receipt", str(gpu), "--p3-receipt", str(proof), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as stopped:
        h1.main()
    result = json.loads((tmp_path / "out/h1.json").read_text())
    assert stopped.value.code == 0 and result["preconditions_confirmed"]
    assert result["gpu_receipt"]["sha256"] and result["p3_receipt"]["sha256"]
    assert result["flux_job_id"] == "gpu-job"
    assert not json.loads(gpu.read_text())["preconditions_confirmed"]


def population_cases():
    pair = ["tessera::fused_window_dense", "native_fused_window_dense"]
    population = {"sms": 1, "modules": [{"root": "/wires", "name": name, "library": "e4m3",
                               "contract_family": "TESSERA_E4M3_K1", "pair": pair}
                              for name in ("module-a", "module-b")]}
    manifest = {"modules": [{"root": "/wires", "name": name, "structure": "dense", "q256": 1024,
                             "cuts": [{"tp": {"size": size, "rank": rank},
                                       "launch_geometry": {"rows": 128, "cols": 64, "tile_words": 4096, "blocks": 1, "merged": False},
                                       "m_configs": [{"M": m, "k_split": 1, "last_tile_partial": m % 64 != 0,
                                                      "superblock_width": 64, "chunk_size": 32, "route_counts": []}
                                                     for m in (1, 2, 8, 64, 65, 128, 512, 2048)]}
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
    assert len(units) == 48 and all(unit["outcome"] == "STOPPED" for unit in units)
    assert {unit["cell"] for unit in units} == {"decode", "batch"}


@pytest.mark.parametrize("omitted", ("module", "cut", "configs", "one_config", "pair"))
def test_manifest_cannot_omit_an_expected_unit(omitted):
    contract, population, manifest = population_cases()
    if omitted == "module":
        manifest["modules"][1]["name"] = "wrong-module"
    elif omitted == "cut":
        manifest["modules"][0]["cuts"].pop()
    elif omitted == "configs":
        manifest["modules"][0]["cuts"][0]["m_configs"] = []
    elif omitted == "one_config":
        manifest["modules"][0]["cuts"][0]["m_configs"].pop()
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
    assert len(receipt["units"]) == 48
    assert all(unit["outcome"] == "STOPPED" for unit in receipt["units"][1:])
    assert all(cell["verdict"] == "UNATTESTED" for cell in receipt["cells"])
    assert receipt["row"]["stop_cause"] == "second adapter failed"


def test_observation_cannot_claim_decoder_bit_exact():
    unit = row.new_unit({"role": "unit"})
    unit["completed"] = True
    unit["exact"]["sweep"]["elements"] = 1
    unit["exact"]["separation"]["records"] = [{"both": 1}]
    unit["exact"]["scale_index"]["arrays"] = [{"launches": 1}]
    unit["exact"]["chunk_pairs"]["x_elements"] = 1
    unit["exact"]["composition"]["cases"] = 1
    row.finish_unit(unit)
    assert unit["outcome"] == "PASS" and unit["decoder_bit_exact"]
    unit["role"] = "observation"
    row.finish_unit(unit)
    assert unit["outcome"] == "PASS" and not unit["decoder_bit_exact"]


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


def test_canonical_population_units_keep_distinct_wire_identities():
    from experiments.numerics_rows.population import module_key
    base = {"root": "/checkpoint", "name": "module", "wire_root": "/cache", "layer": 3}
    first = {**base, "wire_family": {"rung": "R832", "payload": "TESSERA_E4M3_K1"}}
    second = {**base, "wire_family": {"rung": "R960", "payload": "TESSERA_E4M3_K1"}}
    assert module_key(first) != module_key(second)
    assert module_key({"root": "/checkpoint", "name": "module", "wire_path": "/cache/a"}) != module_key(
        {"root": "/checkpoint", "name": "module", "wire_path": "/cache/b"})


def test_population_metadata_requires_every_named_shard(tmp_path):
    from experiments.numerics_rows.population import entries
    scheme = {"family": "TESSERA_FP8", "q256": 1024, "rows": 128, "columns": 64, "roles": [["down_proj", 128]]}
    name = "model.language_model.layers.0.mlp.down_proj"
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"config_groups": {"one": {"scheme": scheme, "targets": [name]}}}}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name + ".wire_bytes": "weights.safetensors"}}))
    with pytest.raises(ValueError, match="missing wire shard"):
        entries(tmp_path)
    (tmp_path / "weights.safetensors").write_bytes(b"metadata-only-path-check")
    entry, = entries(tmp_path)
    assert entry["paths_exist"] and entry["q256"] == 1024 and entry["layer"] == 0
    assert entry["wire_stamp"] is None and entry["runs_table"] is None


@pytest.fixture(params=["e4m3-1024-window-channel-256c.tessera",
                        "bf16-1024-window-channel-256c.tessera",
                        "e2m1x2-640-window-lut-256c.tessera"])
def population_wire(request):
    from pathlib import Path
    from experiments.numerics_rows.cases import rate_layout
    from tessera.unit_artifact import parse_unit_artifact

    blob = (Path(__file__).parent / "data/legacy" / request.param).read_bytes()
    parsed = parse_unit_artifact(blob, device="cpu")
    unit = parsed.unit
    stamp = {"q256": int(parsed.manifest.branch.root_q256) // int(parsed.grid.arity),
             "body": parsed.body.name.lower(), "span": unit.span,
             "plane": "lut16" if "lut" in request.param else "channel",
             "window_bits": unit.window_bits, "seed": unit.window_seed,
             "sigma": unit.window_sigma, "channel_sigma": unit.channel_sigma}
    return blob, parsed.manifest.geometry.rows, stamp, rate_layout(unit.rates)


@pytest.mark.parametrize("storage", ["raw", "fused", "safetensors"])
def test_population_inspection_reads_only_metadata(tmp_path, monkeypatch, population_wire, storage, capsys):
    import torch
    from safetensors.torch import save_file
    from experiments.dense_fused_oracle import Store
    from experiments.numerics_rows.population import inspect
    from tessera.fused import pack_fused

    blob, rows, stamp, layout = population_wire
    role, name = "down_proj", "model.layers.0.mlp.down_proj"
    fused = pack_fused([(role, rows, blob)])
    scheme = {"roles": [[role, rows]]}
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"config_groups": {
        "one": {"scheme": scheme, "targets": [name]}}}}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name + ".wire_bytes": "w.safetensors"}}))
    entry = {"root": str(tmp_path), "name": name, "structure": "dense", "q256": stamp["q256"]}
    if storage == "safetensors":
        save_file({name + ".wire_bytes": torch.tensor(list(fused), dtype=torch.uint8)}, tmp_path / "w.safetensors")
    else:
        path = tmp_path / "wire.tessera"
        path.write_bytes(blob if storage == "raw" else fused)
        entry["wire_path"] = str(path)
    def full_tensor_forbidden(*args):
        raise AssertionError("population must not read a complete wire tensor")
    monkeypatch.setattr(Store, "get", full_tensor_forbidden)
    population = inspect({"arm": "test", "modules": [entry]})
    record, = entry["wire_stamp"]
    assert record["wire_stamp"] == stamp
    assert {key: record[key] for key in layout} == layout
    assert "wire_sha256" not in record and len(record["manifest_sha256"]) == 64
    assert 0 < entry["inspection"]["metadata_bytes"] < entry["inspection"]["wire_bytes"]
    assert population["wire_stamps_read"] and "payload verification remains" in population["inspection_scope"]
    assert "module-start" in capsys.readouterr().out


def test_population_range_reads_never_enter_the_payload(population_wire):
    from experiments.numerics_rows.population import read_stamps
    from tessera.container import _HEADER, HEADER_BYTES

    blob, _, stamp, layout = population_wire
    side = HEADER_BYTES + _HEADER.unpack_from(blob)[4]
    def read(offset, size):
        assert offset + size <= side, "population read touched a payload plane"
        return blob[offset:offset + size]
    record, = read_stamps(read, len(blob), "down_proj")
    assert record["wire_stamp"] == stamp
    assert record["runs_table"] == layout["runs_table"]
    with pytest.raises(ValueError, match="lengths|terminal|truncated"):
        read_stamps(read, side - 1, "down_proj")
    with pytest.raises(ValueError, match="short"):
        read_stamps(lambda offset, size: b"", len(blob), "down_proj")


def test_rate_layout_uses_the_served_word_spans():
    from experiments.numerics_rows.cases import rate_layout
    assert rate_layout([4, 3, 4, 3]) == {"runs_table": [[3, 0, 2, 0], [4, 2, 2, 96]], "tile_words": 224}


def test_dense_and_routed_widths_are_distinct_at_2048(monkeypatch):
    from experiments.numerics_rows.cases import configurations
    from tessera import routed_fused as rf
    monkeypatch.setenv(rf.ENV_WIDE, "auto")
    dense = configurations(128, 128, "e4m3mma", 1, "dense", tile_words=8192, blocks=1)
    routed = configurations(128, 128, "e4m3mma", 1, "routed_moe", tile_words=8192, blocks=1)
    assert next(c for c in dense if c["M"] == 2048)["superblock_width"] == 64
    assert next(c for c in routed if c["M"] == 2048)["superblock_width"] == 128
