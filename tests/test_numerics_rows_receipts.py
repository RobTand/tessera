"""Receipts keep exact NumPy values and completed H1 sections."""
import json
import hashlib
import struct
import sys
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.numerics_rows.common import write_json
from experiments.numerics_rows.build import code_identity
from experiments.numerics_rows import h1
from experiments.numerics_rows import row
from experiments.numerics_rows import dense_phases  # noqa: F401  (load before a fixture patches row.Dense)
from experiments.numerics_rows.common import IMAGES, Invalid, artifact_limit, check_runtime, load_build


@pytest.mark.parametrize("keys", (("e4m3mma", "e4m3"), ("value",), ("e2m1",), ("e4m3", "e4m3"), ()))
def test_img_n_build_selects_its_quantizers_and_h1_library_scope(tmp_path, keys):
    runtime = copy.deepcopy(IMAGES["N"])
    path = tmp_path / "build.json"
    write_json(path, {"runtime": runtime, "libraries": [{"key": key} for key in keys]})
    if keys in (("e4m3mma", "e4m3"), ("value",)):
        built = load_build(path, runtime)
        assert runtime["libraries"] == [r["key"] for r in built["libraries"]] == list(keys)
    else:
        with pytest.raises(Invalid, match="build libraries"):
            load_build(path, runtime)


def test_build_from_another_image_is_refused(tmp_path):
    path = tmp_path / "build.json"
    write_json(path, {"runtime": IMAGES["G"], "libraries": [{"key": "e4m3mma"}]})
    with pytest.raises(Invalid, match="build runtime"):
        load_build(path, copy.deepcopy(IMAGES["N"]))


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
    from experiments.numerics_rows import launch_state
    captured = launch_state.LaunchState()
    monkeypatch.setattr(launch_state, "LaunchState", lambda *args, **kwargs: captured)
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
                                    "subnormal_operand": {"result": "NOT_APPLICABLE", "reason": "not representable; native code range"}}},
            "negative_controls": [{"flagged_exactly": True}],
            "cuda_controls": [{"where": where, "exit_status": 0, "attribution_confirmed": True,
                               "attributed_to": "injected-launch" if where == "inside" else None}
                              for where in ("inside", "outside")], "preconditions_confirmed": False}


def completed_p3_receipt():
    return {"e4m3": {"library_so_sha256": "a" * 64, "probe_sha256": "b" * 64,
                     "matched": True, "status": "MATCHED", "unresolved": [], "differences": [],
                     "mode0_instantiations": 1, "quotients": 1, "quotients_examined": 1}}


def test_source_check_has_no_implicit_commit(tmp_path, monkeypatch):
    from experiments.numerics_rows import common

    monkeypatch.setattr(common, "sha256", lambda path: common.KERNEL_SHA256)
    monkeypatch.setattr(common.subprocess, "check_output", lambda command, **kw:
                        b"" if "status" in command else "a" * 40)
    with pytest.raises(TypeError):
        common.check_source(tmp_path)


def test_p_accepts_the_explicit_campaign_commit(tmp_path, monkeypatch):
    from experiments.numerics_rows import build

    class Checked(Exception):
        pass

    def check(source, revision):
        assert revision == "b" * 40
        raise Checked

    monkeypatch.setattr(build, "check_source", check)
    monkeypatch.setattr("sys.argv", ["build", "--source", str(tmp_path), "--out", str(tmp_path),
                                     "--image", "N", "--revision", "b" * 40])
    with pytest.raises(Checked):
        build.main()


@pytest.mark.parametrize("entrypoint", ("quantizers", "h1"))
def test_gpu_checks_use_the_p_commit(tmp_path, monkeypatch, entrypoint):
    from experiments.numerics_rows import h1, quantizers

    module = {"h1": h1, "quantizers": quantizers}[entrypoint]

    class Checked(Exception):
        pass

    def check(source, revision=None):
        assert revision == "b" * 40
        raise Checked

    monkeypatch.setattr(module, "check_source", check)
    monkeypatch.setattr(module, "check_runtime", lambda image: {})
    monkeypatch.setattr(module, "load_build", lambda *args: {"commit": "b" * 40})
    with pytest.raises(Checked):
        if entrypoint == "h1":
            h1.run_checks(SimpleNamespace(source=tmp_path, image="N", build=tmp_path), {})
        else:
            monkeypatch.setattr("sys.argv", ["quantizers", "--source", str(tmp_path), "--out", str(tmp_path),
                                             "--image", "N", "--build", str(tmp_path), "--artifact", str(tmp_path)])
            quantizers.main()


def test_p_uses_the_original_serving_flags(tmp_path, monkeypatch):
    from pathlib import Path
    from experiments.numerics_rows import build
    from tessera import routed_fused
    from torch.utils import cpp_extension

    observed = []
    binary = tmp_path / "library.so"
    binary.write_bytes(b"fixture")
    monkeypatch.setattr(build, "check_source", lambda *args: "b" * 40)
    monkeypatch.setattr(build, "check_runtime", lambda image: {"libraries": ["e4m3mma"]})
    monkeypatch.setattr(build, "library_identity", lambda *args: {})
    monkeypatch.setattr(build.subprocess, "run", lambda command, **kw:
                        Path(command[-1]).write_bytes(b"probe") if command[0] == "nvcc" else None)

    def load(**kwargs):
        observed.append(kwargs["extra_cuda_cflags"])
        return SimpleNamespace(__file__=str(binary))

    monkeypatch.setattr(cpp_extension, "load", load)
    monkeypatch.setattr("sys.argv", ["build", "--source", str(tmp_path), "--out", str(tmp_path),
                                     "--image", "N", "--revision", "e89757b9"])
    build.main()
    assert observed == [routed_fused._cflags("sm_121", True, True, False)]


@pytest.mark.parametrize("step", ("Q1-N", "H1-N"))
def test_generated_jobs_refuse_empty_container_success(tmp_path, monkeypatch, step):
    import os
    from pathlib import Path
    import subprocess
    from experiments.numerics_rows import prepare_job

    with monkeypatch.context() as patch:
        patch.setattr(prepare_job, "Path", lambda value: tmp_path / Path(value).name
                      if str(value).startswith("/mnt/shared/flux-jobs/") else Path(value))
        patch.setattr(prepare_job.subprocess, "check_output", lambda *a, **kw: "a" * 40)
        patch.setattr(prepare_job.subprocess, "run", lambda *a, **kw: None)
        out = prepare_job.prepare(step, "N", ("--build", str(tmp_path / "build.json")), source_revision="b" * 40)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("git", "sha256sum", "podman"):
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + (
            f'if [ "$3" = rev-parse ]; then if [ "$2" = code ]; then echo {"a" * 40}; else echo {"b" * 40}; fi; fi\n'
            if name == "git" else "") + "exit 0\n")
        path.chmod(0o755)
    run = subprocess.run(["/bin/bash", str(out / "job.sh")], capture_output=True, text=True,
                         env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FLUX_JOB_ID": "test"})
    assert run.returncode != 0
    assert int((out / "tool-exit.txt").read_text()) != 0
    assert not (out / "results" / f"{step.split('-')[0].lower()}.json").exists()


def test_generated_job_refuses_another_source_commit_before_podman(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    import subprocess
    from experiments.numerics_rows import prepare_job

    with monkeypatch.context() as patch:
        patch.setattr(prepare_job, "Path", lambda value: tmp_path / Path(value).name
                      if str(value).startswith("/mnt/shared/flux-jobs/") else Path(value))
        patch.setattr(prepare_job.subprocess, "check_output", lambda *a, **kw: "a" * 40)
        patch.setattr(prepare_job.subprocess, "run", lambda *a, **kw: None)
        out = prepare_job.prepare("P-N", "N", source_revision="b" * 40)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text(f'#!/bin/sh\nif [ "$3" = rev-parse ]; then if [ "$2" = code ]; then echo {"a" * 40}; else echo {"c" * 40}; fi; fi\n')
    marker = tmp_path / "container-started"
    (bin_dir / "podman").write_text(f'#!/bin/sh\nif [ "$1" = run ]; then touch "{marker}"; fi\n')
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    run = subprocess.run(["/bin/bash", str(out / "job.sh")], capture_output=True, text=True,
                         env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FLUX_JOB_ID": "test"})
    assert run.returncode != 0 and not marker.exists()
    assert "checkout --detach " + "b" * 40 in (out / "job.sh").read_text()
    assert (out / "tool-exit.txt").read_text().strip() != "0"


@pytest.mark.parametrize("step", ("C1-N", "ROW-N"))
def test_generated_rows_use_only_the_required_resident_selector(tmp_path, monkeypatch, step):
    from pathlib import Path
    from experiments.numerics_rows import prepare_job

    monkeypatch.setattr(prepare_job, "Path", lambda value: tmp_path / Path(value).name
                        if str(value).startswith("/mnt/shared/flux-jobs/") else Path(value))
    monkeypatch.setattr(prepare_job.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    monkeypatch.setattr(prepare_job.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(prepare_job, "stage_population", lambda path, out: (path, ""))
    monkeypatch.setattr(prepare_job, "source_layout", lambda *a: (tmp_path / "retained", None))
    out = prepare_job.prepare(step, "N", ("--population", str(tmp_path / "population.json"),
        "--build", str(tmp_path / "build.json"), "--build-provenance", str(tmp_path / "provenance.json")), source_revision="b" * 40)
    script = (out / "job.sh").read_text()
    assert "--env TESSERA_SERVE_MODE=resident" in script
    assert "--env TESSERA_PLATFORM_TOKEN=" not in script and "--env TESSERA_FUSED_E4M3_MMA=" not in script
    assert "/run/numerics/image-inspect.json:ro" in script and "--env NUMERICS_IMAGE_INSPECT=" not in script


@pytest.mark.parametrize("dev_mode", ("0", "1"))
def test_population_selector_refuses_in_both_dev_modes(monkeypatch, dev_mode):
    from experiments.numerics_rows.scope import check_selectors

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    population = {"modules": [{"cell_ids": [], "selector_env": {"TESSERA_DENSE_FUSED": "0"}}]}
    contract = {"lane_eligibility": {"cells": []}}
    check_selectors(contract, population, {"TESSERA_DENSE_FUSED": "0"})
    with pytest.raises(Invalid, match="population selector"):
        check_selectors(contract, population, {"TESSERA_DENSE_FUSED": "1"})


@pytest.fixture
def finished_job(tmp_path):
    import struct
    from experiments.numerics_rows.common import sha256

    registration = {"result": "PASS", "registration": "import vllm._custom_ops", "requirements": []}
    write_json(tmp_path / "native_ops.json", registration)
    binary = tmp_path / "binary"
    binary.write_bytes(b"measured binary")
    identity = {"path": str(binary), "sha256": sha256(binary)}
    runtime = {**IMAGES["N"], "libraries": ["e4m3"]}
    build = tmp_path / "build.json"
    write_json(build, {"runtime": runtime, "libraries": [{"key": "e4m3", "library": identity, "probe": identity}]})
    table = tmp_path / "table.f32"
    table.write_bytes(struct.pack("<fff", -1, 0, 1))
    certificate = {"elements": 3, "finite": True, "magnitude_bounded": True, "held": True}
    proof = {"key": "e4m3", "library_so_sha256": identity["sha256"], "probe_sha256": identity["sha256"],
             "dom_size": 3, "d_nondecreasing_0_L": True, "finite_certificate": certificate,
             **{f"{prefix}_{field}": value for prefix in ("dom", "d_table")
                for field, value in (("path", str(table)), ("sha256", sha256(table)))}}
    record = {**identity, "family": "e4m3", "gs": 1.0, "repeat_equal": True, "finite_codes": True,
              "finite_scales": True, "labels": [{"zero": True}]}
    q1 = {"native_ops": registration, "runtime": runtime, "tables": [proof], "quantizers": [record],
          "artifact_scales": {}, "boundaries": [{"kind": "quantizer", "sync_before": True, "sync_after": True}] * 2}
    gpu = completed_gpu_receipt()
    gpu.update(runtime=runtime, native_ops=registration, build_sha256=sha256(build))
    proof = completed_p3_receipt()
    proof["e4m3"].update(library_so_sha256=identity["sha256"], probe_sha256=identity["sha256"])
    gpu["P3"] = proof
    h1.confirm_p3(gpu, proof)
    write_json(tmp_path / "q1.json", q1)
    write_json(tmp_path / "h1.json", gpu)
    return tmp_path, build


@pytest.mark.parametrize("step", ("Q1", "H1"))
@pytest.mark.parametrize("mutation", ("missing_receipt", "registration", "scope", "failed_check", "missing_check", "changed_binary"))
def test_job_receipt_requires_completed_checks(finished_job, step, mutation):
    from experiments.numerics_rows import job_receipt

    out, build = finished_job
    assert job_receipt.check(step, out, build)["verdict"] == "PASS"
    path = out / f"{step.lower()}.json"
    record = json.loads(path.read_text())
    if mutation == "missing_receipt":
        path.unlink()
    elif mutation == "registration":
        (out / "native_ops.json").unlink()
    elif mutation == "changed_binary":
        (out / "binary").write_bytes(b"other binary")
    else:
        if mutation == "scope":
            record["runtime"]["libraries"] = ["value"]
        elif step == "Q1":
            if mutation == "missing_check":
                record["tables"].clear()
            else:
                record["quantizers"][0]["repeat_equal"] = False
        elif mutation == "missing_check":
            record.pop("cuda_controls")
        else:
            record["libraries"]["e4m3"]["P4"] = False
        write_json(path, record)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        job_receipt.check(step, out, build)


def gpu_receipt_with_inapplicable_checks():
    report = completed_gpu_receipt()
    template = report["libraries"]["e4m3"]
    report["runtime"]["libraries"] = ["e2m1", "e4m3", "e4m3mma"]
    report["libraries"] = {key: copy.deepcopy(template) for key in report["runtime"]["libraries"]}
    for key, value in report["libraries"].items():
        value["P5"] = None if key == "e2m1" else True
        value["wrong_scale_detected"] = True if key == "e2m1" else None
        value["routed"]["negative_control"] = {"flagged_exactly": True if key == "e2m1" else None}
    return report


def test_gpu_preconditions_record_only_inapplicable_checks():
    report = gpu_receipt_with_inapplicable_checks()
    assert h1.gpu_preconditions(report)
    assert report["libraries"]["e2m1"]["P5"]["result"] == "NOT_APPLICABLE"
    for key in ("e4m3", "e4m3mma"):
        entry = report["libraries"][key]
        assert entry["wrong_scale_detected"]["result"] == "NOT_APPLICABLE"
        assert entry["routed"]["negative_control"]["result"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("key,path", [("e4m3", ("P5",)), ("e4m3mma", ("routed", "P5")),
    ("e2m1", ("wrong_scale_detected",)), ("e2m1", ("routed", "negative_control", "flagged_exactly")),
    ("e2m1", ("P4",)), ("e4m3", ("routed", "P4"))])
@pytest.mark.parametrize("value", [None, False])
def test_gpu_preconditions_do_not_exempt_required_checks(key, path, value):
    report = gpu_receipt_with_inapplicable_checks()
    assert h1.gpu_preconditions(report)
    entry = report["libraries"][key]
    for field in path[:-1]:
        entry = entry[field]
    entry[path[-1]] = value
    assert not h1.gpu_preconditions(report)


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
    population = {"sms": 1, "modules": [{"root": "/wires", "name": name, "library": "e4m3", "roles": [["down_proj", 128]], "experts": 1,
                               "contract_family": "TESSERA_E4M3_K1", "pair": pair, "structure": "dense", "q256": 1024, "cell_ids": ["batch", "decode"]}
                              for name in ("module-a", "module-b")]}
    manifest = {"generation_finished": True, "modules": [{"root": "/wires", "name": name, "cell_ids": ["batch", "decode"], "structure": "dense", "q256": 1024,
                             "cuts": [{"tp": {"size": size, "rank": rank}, "roles": [{"name": "down_proj", "expert": 0}],
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


def test_explicit_module_selection_keeps_the_signed_population_and_all_cut_configs():
    from experiments.numerics_rows.population import select_modules

    contract, population, manifest = population_cases()
    original = copy.deepcopy(population)
    selected = select_modules(population, ["module-b"])
    manifest["modules"] = manifest["modules"][1:]
    groups = row.planned_units(contract, "image", selected, manifest)
    assert len(groups) == 3 and sum(len(units) for *_, units in groups) == 24
    assert population == original
    with pytest.raises(Invalid, match="modules differ"):
        row.planned_units(contract, "image", population, manifest)


@pytest.mark.parametrize("names", [["missing"], ["module-a", "module-a"]])
def test_module_selection_refuses_unsigned_or_repeated_names(names):
    from experiments.numerics_rows.population import select_modules

    with pytest.raises(Invalid, match="module selection"):
        select_modules(population_cases()[1], names)


def test_a_complete_sample_is_still_not_a_row_manifest():
    contract, population, manifest = population_cases()
    assert row.planned_units(contract, "image", population, manifest)
    manifest.update(complete=True, sample={"expert": 0, "tp": {"size": 2, "rank": 0}})
    with pytest.raises(Invalid, match="sample"):
        row.planned_units(contract, "image", population, manifest)


def test_routed_sample_reads_only_its_expert_and_keeps_the_total_expert_count():
    from types import SimpleNamespace
    import torch
    from experiments.numerics_rows.cases import checkpoint_routed

    requested = []
    def get(name):
        requested.append(name)
        return torch.tensor([17], dtype=torch.uint8)
    store = SimpleNamespace(schemes={"module": {"experts": 288, "family": "TESSERA_FP8"}}, get=get)
    scheme, blobs, scales = checkpoint_routed(store, {"name": "module"}, [17])
    assert scheme["experts"] == 288 and scales == {}
    assert requested == [f"module.17.{p}.wire" for p in ("gate_proj", "up_proj", "down_proj")]
    assert all(wires == [b"\x11"] for wires in blobs.values())


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
    from experiments.numerics_rows import launch_state

    captured = launch_state.LaunchState()
    monkeypatch.setattr(launch_state, "LaunchState", lambda *args, **kwargs: captured)
    contract, population, manifest = population_cases()
    def run(args, receipt):
        groups = row.planned_units(contract, "image", population, manifest)
        receipt["units"] = [unit for _, _, _, group in groups for unit in group]
        receipt["units"][0]["completed"] = True
        raise RuntimeError("second adapter failed")

    argv = ["row", "--image", "G", "--name", "test", "--out", str(tmp_path)]
    for name in ("source", "manifest", "population", "build", "build-provenance", "q1", "h1", "rule", "freeze"):
        argv.extend(("--" + name, str(tmp_path / name)))
    monkeypatch.setattr(row, "run", run)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stopped:
        row.main()
    assert stopped.value.code == 3
    receipt = json.loads((tmp_path / "numerics_row.json").read_text())
    assert len(receipt["units"]) == 48
    assert all(unit["outcome"] == "STOPPED" for unit in receipt["units"][1:])
    assert all(cell["verdict"] == "UNATTESTED" for cell in receipt.get("diagnostic_cells", receipt.get("cells")))
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
    monkeypatch.setenv("NUMERICS_IMAGE_INSPECT", str(path))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "wrong-version" if mutation == "version" else expected[name])
    if mutation == "missing":
        monkeypatch.delenv("NUMERICS_IMAGE_INSPECT")
    if mutation is None:
        result = check_runtime("G")
        assert result["local_id"] == record["Id"][7:]
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
    assert symbols == [{"profiler_name": "void routed_fused_kernel<0>( Params )", "symbol": "_Zkernel0"},
                       {"profiler_name": "_Zkernel2", "symbol": "_Zkernel2"}] and unresolved == ["quantizer"]
    assert row.resolve_symbols(["_Zkernel0"], ["void routed_fused_kernel<1>(Params)"],
        ["void routed_fused_kernel<0>(Params)"]) == ([], ["void routed_fused_kernel<1>(Params)"])


@pytest.mark.parametrize("different", (None, "symbol", "binary", "rung", "tp", "pair", "unresolved", "source-unresolved", "missing"))
def test_separation_attaches_only_to_the_same_compiled_function(different):
    import copy
    scope = {"cell": "cell", "symbol": "pair", "decoder": "decoder", "rung": 1024, "tp": {"rank": 0, "size": 1}}
    source, target = row.new_unit(scope), row.new_unit(scope)
    identity = [{"profiler_name": "void routed_fused_kernel<0>()", "symbol": "_Zrouted_fused_kernel0", "library_so_sha256": "binary-a"}]
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
    elif different in ("unresolved", "source-unresolved"):
        (source if different == "source-unresolved" else target)["arm"] = {"unresolved_kernel_names": ["routed_unknown"]}
    elif different == "missing":
        target["m_config"]["compiled_kernel_symbols"] = []
    target["m_config"]["compiled_kernel_symbols"] = [{**r, "profiler_name": "a different profiler spelling"} for r in target["m_config"]["compiled_kernel_symbols"]]
    row.attach_separation([source, target])
    if different:
        assert target["unattested"] and target["m_config"]["separation_source"] is None
    else:
        assert target["m_config"]["separation_source"] == {"unit_id": row.unit_identity(source), "symbol": "_Zrouted_fused_kernel0", "library_so_sha256": "binary-a"}
        assert row.unit_identity(source) == ["cell", "pair", "decoder", 1024, {"M": 64}, {"rank": 0, "size": 1}, None]
        assert not target["unattested"]
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
    assert scales["scales"] == ["a_s", "row"] and scales["bases"] == [8]
    assert scales["reads"] == scales["reads_covered"] == 16 and scales["launches"] == 32
    assert scales["passes_total"] == 4 and scales["families"] == ["plain"]
    assert not {"scale", "base", "passes_per_family"} & scales.keys()
    assert all(record["passes_per_family"] == 2 for record in scales["arrays"])


def test_rule_receipt_requires_decisions_and_records_the_h1_job(tmp_path):
    path = tmp_path / "h1.json"
    h1 = {"flux_job_id": "test-job", "preconditions_confirmed": True, "P2": {"result": "PASS"}}
    path.write_text(json.dumps(h1))
    freeze = {"decisions": {f"D{i}": "reviewed" for i in range(1, 10)}}
    receipt = row.rule_receipt(tmp_path / "rule.md", freeze, h1, path)
    assert set(receipt["preconditions"]) == {"P1", "P2", "P3", "P4", "P5"}
    assert all(result == {"result": "PASS", "h1_job": "test-job"} for result in receipt["preconditions"].values())
    with pytest.raises(Invalid, match="D1 through D9"):
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


@pytest.mark.parametrize("mutation", ["review", "reviewed", "wire_stamps_read", "sms", "module", "source"])
def test_signed_population_is_bound_to_the_reviewed_source(tmp_path, mutation):
    import hashlib
    from experiments.numerics_rows.population import load_signed_population

    source = tmp_path / "source.json"
    population = {"reviewed": False, "wire_stamps_read": True, "sms": 48, "modules": [{"name": "module"}]}
    source.write_text(json.dumps(population))
    review = {"signed_by": "lead", "signed_utc": "2026-11-16T13:18:32Z", "source_file": str(source),
              "source_flux_job": "population-job", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "spec": "rows draft 8"}
    signed = {**population, "reviewed": True, "review": review}
    path = tmp_path / "signed.json"
    path.write_text(json.dumps(signed))
    result, digest = load_signed_population(path)
    assert result == signed and digest == hashlib.sha256(path.read_bytes()).hexdigest()
    if mutation == "module":
        signed["modules"][0]["name"] = "omitted-or-replaced"
    elif mutation == "source":
        source.write_text(source.read_text() + "\n")
    elif mutation == "review":
        signed.pop("review")
    else:
        signed[mutation] = None
    path.write_text(json.dumps(signed))
    with pytest.raises(Invalid, match="signed population"):
        load_signed_population(path)


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


def test_uncovered_reads_retain_scale_names_at_unit_and_array_levels():
    unit = row.new_unit({"role": "unit"})
    unit["exact"]["scale_index"]["arrays"] = [{"scale": scale, "uncovered_reads": [{"read": [0, 1], "reason": "joint collision"}]} for scale in ("row", "sfa")]
    row.aggregate_exact(unit)
    assert unit["exact"]["scale_index"]["uncovered_reads"] == [{"scale": "row", "read": [0, 1]}, {"scale": "sfa", "read": [0, 1]}]
    assert all(isinstance(r["uncovered_reads"], list) for r in unit["exact"]["scale_index"]["arrays"])


def test_own_separation_needs_no_borrowed_symbol_resolution():
    unit = row.new_unit({"m_config": {"M": 64}, "arm": {"unresolved_kernel_names": ["unknown"]}})
    row.attach_separation([unit])
    assert unit["m_config"]["separation_source"] is None and not unit["unattested"]


@pytest.mark.parametrize("key", ["e2m1", "e4m3", "e4m3mma"])
def test_non_bf16_subnormal_case_records_its_native_range(key):
    result = h1.subnormal_case(key, None)
    assert result["result"] == "NOT_APPLICABLE" and "native" in result["reason"]


def test_legacy_subnormal_receipt_is_normalized_only_with_its_recorded_reason():
    report = completed_gpu_receipt()
    entry = report["libraries"]["e4m3"]
    entry["subnormal_operand"] = {"result": "UNATTESTED", "reason": "not representable; this library's FP32 products cannot be subnormal"}
    assert h1.gpu_preconditions(report)
    assert entry["subnormal_operand"]["result"] == "NOT_APPLICABLE"
    assert entry["subnormal_operand"]["original_result"] == "UNATTESTED"
    entry["subnormal_operand"].pop("reason")
    assert not h1.gpu_preconditions(report)


def elf_bytes(*, wide=True, order="<", text=b"\x01\x02\x03\x04", strtab=b"\0left\0", bss_size=8,
              text_type=1, text_flags=6, reverse=False, extra_sections=()):
    names = b"\0.shstrtab\0.text\0.strtab\0.bss\0"
    sections = [("", 0, 0, b"", 0), (".shstrtab", 3, 0, names, len(names)),
                (".text", text_type, text_flags, text, len(text)), (".strtab", 3, 0, strtab, len(strtab)),
                (".bss", 8, 3, b"", bss_size)]
    if extra_sections:
        sections.extend(extra_sections)
        names = b"\0" + b"".join(s[0].encode() + b"\0" for s in sections[1:])
        sections[1] = (".shstrtab", 3, 0, names, len(names))
    if reverse:
        sections[2:] = reversed(sections[2:])
    size, stride = (64, 64) if wide else (52, 40)
    data = bytearray(size)
    headers = []
    for name, kind, flags, content, length in sections:
        headers.append((names.index(name.encode() + b"\0"), kind, flags, 0, len(data), length,
                        3 if kind == 2 else 0, 0, 1, (24 if wide else 16) if kind == 2 else 0))
        data.extend(content)
    table_offset = len(data)
    for header in headers:
        data.extend(struct.pack(order + ("IIQQQQIIQQ" if wide else "IIIIIIIIII"), *header))
    ident = b"\x7fELF" + bytes((2 if wide else 1, 1 if order == "<" else 2, 1)) + b"\0" * 9
    data[:size] = ident + struct.pack(order + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH"),
                                     3, 62, 1, 0, 0, table_offset, 0, size, 0, 0, stride, len(headers), 1)
    return bytes(data)


@pytest.mark.parametrize("wide,order", [(False, "<"), (False, ">"), (True, "<"), (True, ">")])
def test_code_digest_has_exact_header_and_payload_encoding(tmp_path, wide, order):
    path = tmp_path / "library.so"
    path.write_bytes(elf_bytes(wide=wide, order=order))
    names = b"\0.shstrtab\0.text\0.strtab\0.bss\0"
    expected = (b'["",0,0,0]\n[".shstrtab",3,0,30]\n' + names
                + b'[".text",1,6,4]\n\x01\x02\x03\x04[".bss",8,3,8]\n')
    result = code_identity(path)
    assert result["library_code_sha256"] == hashlib.sha256(expected).hexdigest()
    sections = result["library_code_sections"]
    assert [s["name"] for s in sections] == ["", ".shstrtab", ".text", ".bss"]
    assert [s["index"] for s in sections] == [0, 1, 2, 4]
    assert sections[-1] == {"index": 4, "name": ".bss", "sh_type": 8, "sh_flags": 3,
                            "sh_size": 8, "sha256": hashlib.sha256(b"").hexdigest()}


@pytest.mark.parametrize("change,equal", [({"strtab": b"\0right\0"}, True),
                                        ({"text": b"\x01\x02\x03\x05"}, False),
                                        ({"bss_size": 9}, False), ({"text_type": 7}, False),
                                        ({"text_flags": 2}, False), ({"reverse": True}, False)])
def test_code_digest_ignores_only_strtab(tmp_path, change, equal):
    original, changed = tmp_path / "original.so", tmp_path / "changed.so"
    original.write_bytes(elf_bytes())
    changed.write_bytes(elf_bytes(**change))
    assert original.read_bytes() != changed.read_bytes()
    assert (code_identity(original)["library_code_sha256"] == code_identity(changed)["library_code_sha256"]) is equal


@pytest.mark.parametrize("data", [b"not ELF", elf_bytes()[:-1]])
def test_code_digest_refuses_invalid_elf(tmp_path, data):
    path = tmp_path / "broken.so"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        code_identity(path)


@pytest.fixture
def retained_binary_receipts(tmp_path, monkeypatch):
    from pathlib import Path
    from experiments.numerics_rows import build
    from experiments.numerics_rows.common import sha256

    def extract(command, *, cwd, **kwargs):
        assert command[:3] == ["cuobjdump", "-xelf", "all"]
        assert Path(command[3]).is_absolute()
        for name in ("sm_121.cubin", "sm_121a.cubin"):
            (cwd / name).write_bytes(elf_bytes(text=name.encode()))
    monkeypatch.setattr(build.subprocess, "run", extract)
    library, probe = tmp_path / "library.so", tmp_path / "probe"
    library.write_bytes(elf_bytes())
    probe.write_bytes(b"retained probe bytes")
    record = {"key": "e4m3", "library": {"path": str(library), "sha256": sha256(library)},
              "probe": {"path": str(probe), "sha256": sha256(probe)}}
    build_path, h1_path = tmp_path / "build.json", tmp_path / "h1.json"
    write_json(build_path, {"libraries": [record]})
    report = completed_gpu_receipt()
    proof = completed_p3_receipt()
    proof["e4m3"].update(library_so_sha256=sha256(library), probe_sha256=sha256(probe))
    report.update(P3=copy.deepcopy(proof), build_sha256=sha256(build_path))
    h1.confirm_p3(report, proof)
    assert report["preconditions_confirmed"]
    write_json(h1_path, report)
    return build_path, h1_path


def test_code_identity_enriches_copies_without_changing_gpu_inputs(retained_binary_receipts, tmp_path):
    from experiments.numerics_rows import build

    build_path, h1_path = retained_binary_receipts
    before = [p.read_bytes() for p in (build_path, h1_path)]
    out = tmp_path / "enriched"
    record, = build.add_identities(build_path, out)["libraries"]
    report = json.loads(h1_path.read_text())
    proof = {**report["P3"]["e4m3"], **{k: record[k] for k in build.IDENTITY_FIELDS}}
    h1.confirm_p3(report, {"e4m3": proof})
    assert report["preconditions_confirmed"]
    assert [p.read_bytes() for p in (build_path, h1_path)] == before
    assert record["cubin_sha256s"] == [hashlib.sha256(elf_bytes(text=name.encode())).hexdigest()
                                         for name in ("sm_121.cubin", "sm_121a.cubin")]
    for field in build.IDENTITY_FIELDS:
        assert report["P3"]["e4m3"][field] == record[field]
    assert report["build_sha256"] == hashlib.sha256(before[0]).hexdigest()


@pytest.mark.parametrize("mutation", ["library", "probe"])
def test_code_identity_refuses_changed_binary(retained_binary_receipts, tmp_path, mutation):
    from pathlib import Path
    from experiments.numerics_rows import build

    build_path, h1_path = retained_binary_receipts
    build.add_identities(build_path, tmp_path / "positive")
    record, = json.loads(build_path.read_text())["libraries"]
    Path(record[mutation]["path"]).write_bytes(b"changed binary")
    with pytest.raises(ValueError):
        build.add_identities(build_path, tmp_path / "negative")


@pytest.mark.parametrize("field", ["library_so_sha256", "library_code_sha256", "library_code_sections", "cubin_sha256s"])
def test_row_requires_h1_code_identity(retained_binary_receipts, tmp_path, field):
    from experiments.numerics_rows import build

    build_path, h1_path = retained_binary_receipts
    record, = build.add_identities(build_path, tmp_path / "identity")["libraries"]
    proof = {k: record[k] for k in build.IDENTITY_FIELDS}
    row.bind_library_identity(record, record, proof)
    proof.pop(field)
    with pytest.raises(Invalid, match="H1 code identity"):
        row.bind_library_identity(record, record, proof)


def test_population_job_stages_shared_draft_without_changing_signed_bytes(tmp_path):
    from pathlib import Path
    from experiments.numerics_rows.prepare_job import stage_population

    shared = tmp_path / "shared"
    (shared / "drafts").mkdir(parents=True)
    draft = shared / "drafts/population.json"
    draft.write_text('{"reviewed":false,"wire_stamps_read":true,"sms":48,"modules":[]}\n')
    original = Path("/home/rob/unavailable-on-sparks/population.json")
    signed = shared / "signed.json"
    signed.write_text(json.dumps({"review": {"source_file": str(original),
                                             "source_sha256": hashlib.sha256(draft.read_bytes()).hexdigest()}}))
    before = signed.read_bytes()
    job = tmp_path / "job"
    job.mkdir()
    staged, mount = stage_population(signed, job)
    assert staged.read_bytes() == before == signed.read_bytes()
    assert (job / "inputs/draft.json").read_bytes() == draft.read_bytes()
    assert mount == f"-v {job}/inputs/draft.json:{original}:ro "
    draft.write_text(draft.read_text() + " ")
    other = tmp_path / "wrong-job"
    other.mkdir()
    with pytest.raises(ValueError, match="draft identity"):
        stage_population(signed, other)


@pytest.mark.parametrize("field", ["library_canonical_code_sha256", "cubin_canonical_sha256s"])
def test_row_refuses_served_code_identity_before_output_checks(field):
    measured = {"library_so_sha256": "a" * 64, "library_code_sha256": "b" * 64, "cubin_sha256s": ["c" * 64],
                "library_canonical_code_sha256": "b" * 64, "cubin_canonical_sha256s": ["c" * 64]}
    served = {**measured, "library_so_sha256": "d" * 64, "path": "/shared/worker/loaded.so",
              "pid": 1234, "executable_mappings": ["recorded executable mapping"]}
    served.update(library_code_sha256="f" * 64, cubin_sha256s=["f" * 64])
    unit = row.new_unit({"decoder": "native_fused_window_dense_e4m3mma", "arm": {}})
    row.check_served_identity(unit, [served], measured)
    assert unit["arm"]["library_identity_verified"]
    assert unit["arm"]["loaded_modules"][0]["library_so_sha256"] == "d" * 64
    matching = copy.deepcopy(served)
    served[field] = "e" * 64 if field == "library_canonical_code_sha256" else ["e" * 64]
    with pytest.raises(Invalid, match="served library code or cubin identity") as error:
        row.check_served_identity(unit, [matching, served], measured)
    assert not unit["arm"]["library_identity_verified"]
    assert not unit["findings"] and not unit["completed"]
    unit["invalid"].append(str(error.value))
    unit["completed"] = True
    assert row.outcome(unit) == "INVALID"


def test_served_native_empty_and_triton_exemption():
    unit = row.new_unit({"decoder": "native_fused_window_dense_e4m3mma", "arm": {}})
    with pytest.raises(Invalid, match="served library"):
        row.check_served_identity(unit, [], {})
    unit["decoder"] = "native_window_gemm"
    row.check_served_identity(unit, [], {})
    assert unit["arm"]["library_identity_verified"] is None
    with pytest.raises(Invalid, match="served library"):
        row.check_served_identity(unit, [{"name": "unexpected native library"}], {})


@pytest.mark.parametrize("key", ["e4m3mma", "e4m3", "value"])
def test_dense_library_selection_comes_from_the_served_role(monkeypatch, key):
    from tessera import routed_fused as rf

    calls = []
    monkeypatch.setattr(rf, "_ext", lambda key: calls.append(key) or key)
    native = SimpleNamespace(lane="fused", _PreparedDenseNativeModule__fused=[SimpleNamespace(library=key)])
    adapter = SimpleNamespace(family="value" if key == "value" else "e4m3", library="wrong-population-key",
                              layer=SimpleNamespace(tessera_native=native))
    assert row.served_library(adapter, False) == key and calls == [key]
    native.lane = "triton"
    assert row.served_library(adapter, False) is None and calls == [key]


def test_unit_libraries_uses_bound_snapshot_and_rechecks_on_cache_hits(tmp_path, monkeypatch):
    from pathlib import Path
    from experiments.numerics_rows import build
    from tessera._dev import native_identity

    live, source = tmp_path / "live.so", tmp_path / "kernel.cu"
    data = elf_bytes()
    live.write_bytes(data)
    source.write_bytes(b"kernel source")
    calls = []
    def mapped(*args):
        calls.append("mapped")
        return {"path": str(live), "sha256": hashlib.sha256(live.read_bytes()).hexdigest(),
                "executable_mappings": ["same inode and device"], "pid": 1}
    def dump(command, *, stdout, cwd=None, **kwargs):
        snapshot = Path(command[-1])
        assert snapshot != live and snapshot.read_bytes() == data
        calls.append(command[1])
        if command[1] == "-xelf":
            live.write_bytes(elf_bytes(text=b"new!"))
            (cwd / "sm_121.cubin").write_bytes(elf_bytes())
        else:
            stdout.write("served SASS\n")
    monkeypatch.setattr(native_identity, "loaded_native_identity", mapped)
    monkeypatch.setattr(build.subprocess, "run", dump)
    cache = {}
    lib = SimpleNamespace(__name__="native", __file__=str(live))
    before = row.unit_libraries(lib, source, tmp_path / "evidence", cache)
    assert before[0]["library_so_sha256"] == hashlib.sha256(data).hexdigest()
    assert before[0]["library_code_sha256"] == build.code_identity(data)["library_code_sha256"]
    assert Path(before[0]["sass"]["path"]).read_text() == "served SASS\n"
    live.write_bytes(data)
    assert row.unit_libraries(lib, source, tmp_path / "evidence", cache) == before
    assert calls == ["mapped", "-xelf", "-sass", "mapped"]
    monkeypatch.setattr(native_identity, "loaded_native_identity", lambda *args: {**mapped(), "sha256": "0" * 64})
    with pytest.raises(Invalid, match="changed during hashing"):
        row.unit_libraries(lib, source, tmp_path / "evidence", cache)


def test_compile_path_refusal_is_bound_to_the_retained_p_json(tmp_path, monkeypatch):
    from pathlib import Path
    from experiments.numerics_rows import serving_build
    from tessera.serving import ext
    from tessera import routed_fused as rf

    source = tmp_path / "P-cell" / serving_build.KERNEL
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pinned test kernel")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(serving_build, "KERNEL_SHA256", digest)
    record = {"key": "value", "name": "tessera_routed_fused_value", "cflags": ["retained flags"],
              "library": {"path": str(tmp_path / "library.so"), "sha256": "a" * 64}}
    build, proof = tmp_path / "build.json", tmp_path / "p.json"
    write_json(build, {"libraries": [record]})
    original = build.read_bytes()
    write_json(proof, {"build_sha256": hashlib.sha256(original).hexdigest(),
                       "libraries": [{**record, "native_source_path": str(source), "native_source_sha256": digest}]})
    assert build.read_bytes() == original
    assert serving_build.source_layout(build, proof)[0] == source.parents[4]
    monkeypatch.setattr(ext, "native_source_path", lambda name: str(source))
    monkeypatch.setattr(rf, "_cflags", lambda *args: record["cflags"])
    assert serving_build.check_compile_paths(build, proof)[0]["native_source_path"] == str(source)
    monkeypatch.setattr(ext, "native_source_path", lambda name: str(tmp_path / "row-cell" / serving_build.KERNEL))
    with pytest.raises(Invalid, match="compile path"):
        serving_build.check_compile_paths(build, proof)


@pytest.fixture
def row_execution(tmp_path, monkeypatch):
    """Drive run() on CPU; replace device and filesystem boundaries, keep its policy."""
    from contextlib import nullcontext
    import torch
    from experiments import dense_fused_oracle as dense
    from experiments.numerics_rows import cases, common, population, sass_slice, scale_launch
    try:
        from experiments.numerics_rows import dependency_code
    except ImportError:
        pass
    else:
        monkeypatch.setattr(dependency_code.CodeAudit, "install", lambda self: self)
    from experiments import routed_pair_oracle
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: nullcontext())
    monkeypatch.setattr(routed_pair_oracle, "kernel_table", lambda *args, **kwargs: [])

    args = SimpleNamespace(**{name: tmp_path / name for name in
        ("source", "out", "population", "manifest", "build", "build_provenance", "q1", "h1", "rule", "freeze")}, image="N", module=[])
    # The simulated worker has one start environment, separate from pytest phases.
    import _pytest.runner
    monkeypatch.setattr(_pytest.runner, "_update_current_test_var", lambda *args: None)
    from experiments.numerics_rows.launch_state import LaunchState
    monkeypatch.setenv("TESSERA_SERVE_MODE", "resident")
    args.launch_state = LaunchState()
    args.launch_state.bind_math(torch)
    args.out.mkdir()
    contract, pop, manifest = population_cases()
    pop["modules"] = pop["modules"][:1]
    args._population = pop
    entry = pop["modules"][0]
    module = {**manifest["modules"][0], **manifest["modules"][0]["cuts"][0], "roles": [{"matches_attested_wire": True}]}
    scope = row.new_unit({"cell": "cell", "symbol": entry["pair"][0], "decoder": entry["pair"][1], "rung": 1024,
                         "structure": "dense", "role": "unit", "arm": {}, "tp": {}, "m_config": {"M": 1, "k_split": 1}})
    identity = {"library_so_sha256": "a" * 64, "library_code_sha256": "b" * 64, "library_code_sections": [], "cubin_sha256s": ["c" * 64],
                "library_canonical_code_sha256": "b" * 64, "cubin_canonical_sha256s": ["c" * 64]}
    served = {**identity, "name": "native", "path": "/mapped/native.so", "sha256": "a" * 64, "executable_mappings": ["device inode mapping"]}
    runtime = {**IMAGES["N"], "local_id": "image-local-id"}
    library = {"key": "e4m3", "library": {"sha256": "a" * 64, "sass": {"path": "P.library.sass"}}, "probe": {"sha256": "p" * 64}, "probe_source_sha256": "s", "cflags": []}
    write_json(args.build, {"commit": "cell-commit", "runtime": runtime, "libraries": [library]})
    args.rule.write_text("test rule")
    args.build_provenance.write_text("bound source provenance")
    write_json(args.freeze, {"sha256": common.sha256(args.rule), "launch_levers_sha256": args.launch_state.levers_sha256, "frozen_utc": "now", "reviewer": "reviewer", "decisions": {f"D{i}": "yes" for i in range(1, 10)}})
    write_json(args.q1, {"tables": [{"key": "e4m3", "library_so_sha256": "a" * 64, "probe_sha256": "p" * 64,
        "L_bits": 0, "dom_size": 1, "d_table_sha256": "d", "d_nondecreasing_0_L": True}]})
    write_json(args.h1, {**args.launch_state.fields(), "preconditions_confirmed": True, "build_sha256": common.sha256(args.build), "P1": {"mismatches": 0},
        "P2": {"result": "PASS"}, "P3": {"e4m3": identity}, "libraries": {"e4m3": {"P4": True, "P5": True}}, "flux_job_id": "h1-job"})
    write_json(args.manifest, {"population_sha256": "population-hash", "q1_table_sha256": common.sha256(args.q1), **args.launch_state.fields(), "modules": [module]})
    contract_path = args.source / "src/tessera/serving/runtime_contract.json"
    contract_path.parent.mkdir(parents=True)
    write_json(contract_path, contract)
    doc = json.loads(args.manifest.read_text())
    doc.update(source_commit="cell-commit", contract_sha256=common.sha256(contract_path))
    write_json(args.manifest, doc)
    args._load_population = population.load_signed_population
    monkeypatch.setattr(population, "load_signed_population", lambda path: (pop, "population-hash"))
    monkeypatch.setattr(row, "check_source", lambda path: "cell-commit")
    monkeypatch.setattr(row, "check_runtime", lambda image: runtime)
    try:
        from experiments.numerics_rows import serving_build
    except ImportError:
        pass  # The pre-fix checkout has no compile-path verifier.
    else:
        args._compile_paths = serving_build.check_compile_paths
        args._retained_source = serving_build.check_retained_source
        monkeypatch.setattr(serving_build, "check_compile_paths", lambda *args: [{"key": "e4m3", "native_source_path": "/P/kernel.cu"}])
        if hasattr(serving_build, "check_retained_source"):
            monkeypatch.setattr(serving_build, "check_retained_source", lambda *args: {"commit": "cell-commit"})
        def comparison(measured, served, sass, directory):
            directory.mkdir(parents=True, exist_ok=True)
            return {"identity_equal": True, "raw_file_audit": {"unexplained_bytes": 0}, "cubin_audits": [],
                    "sass_token_masked_equal": True, "ptx": {}}
        if hasattr(serving_build, "compare_binaries"):
            monkeypatch.setattr(serving_build, "compare_binaries", comparison)
    monkeypatch.setattr(common, "register_native_ops", lambda image: {"result": "PASS"})
    monkeypatch.setattr(common, "source_digest", lambda *args: "source-hash")
    monkeypatch.setattr(row.subprocess, "check_output", lambda *args, **kwargs: "test-value\n")
    for name, value in (("get_device_capability", (12, 1)), ("get_device_name", "CPU test device"), ("synchronize", None)):
        monkeypatch.setattr(torch.cuda, name, lambda *args, _value=value: _value)
    monkeypatch.setattr(row, "selftest", lambda **kwargs: {"mismatches": 0})
    args._negative_controls = h1.negative_controls
    monkeypatch.setattr(h1, "negative_controls", lambda: [{"flagged_exactly": True}])
    args._match_library, args._dtable = sass_slice.match_library, cases.DTable
    monkeypatch.setattr(sass_slice, "match_library", lambda *args: {**identity, "matched": True})
    monkeypatch.setattr(cases, "DTable", lambda value: SimpleNamespace(finite_certificate=True))
    monkeypatch.setattr(row, "planned_units", lambda *args: [(entry, module, {"rank": 0, "size": 1}, [scope])])
    monkeypatch.setattr(dense, "init_vllm_world1", nullcontext)
    adapter = SimpleNamespace(info={}, pairs=lambda: [entry["pair"]])
    monkeypatch.setattr(row, "Dense", lambda *args: adapter)
    monkeypatch.setattr(row, "module_receipt", lambda *args: {})
    monkeypatch.setattr(row, "tp_facts", lambda *args: {})
    monkeypatch.setattr(row, "actual_geometry", lambda *args: module["launch_geometry"])
    monkeypatch.setattr(row, "scale_receipt", lambda *args: {"equal": True})
    monkeypatch.setattr(row, "native_boundaries", lambda *args: nullcontext())
    monkeypatch.setattr(row, "historical_units", lambda *args: [])
    monkeypatch.setattr(row, "served_library", lambda *args: SimpleNamespace(__name__="native"))
    events = []
    def libraries(*args):
        events.append("identity")
        return [copy.deepcopy(served)]
    monkeypatch.setattr(row, "unit_libraries", libraries)
    def case(*args, **kwargs):
        assert scope["arm"]["library_identity_verified"] is True
        events.append("case")
        exact = scope["exact"]
        exact["sweep"]["elements"] = exact["chunk_pairs"]["x_elements"] = exact["composition"]["cases"] = 1
        exact["separation"]["records"] = [{"both": 1}]
        exact["scale_index"]["arrays"] = [{"launches": 1}]
    for name in ("verify_cases", "dense_cases", "dense_diagnostics", "profile_unit"):
        monkeypatch.setattr(row, name, case)
    monkeypatch.setattr(scale_launch, "check", case)
    receipt = {"row": {"stop_cause": None}, "tessera": {}, "inputs": {}, "units": [], "modules": [], "findings": []}
    return args, receipt, scope, events, served


def test_generated_compare_worker_setup(tmp_path, monkeypatch, capsys, row_execution, retained_binary_receipts):
    from itertools import count
    import shlex
    from pathlib import Path
    from experiments.numerics_rows import build, cases, common, dense_phases, launch_state, population, prepare_split, sass_slice, serving_build, start_env
    from experiments.numerics_rows.prepare_job import stage_population

    fixture, _, _, _, _ = row_execution
    monkeypatch.setattr(population, "load_signed_population", fixture._load_population)
    entry, = fixture._population["modules"]
    entry["selector_env"] = {"TESSERA_SERVE_MODE": "resident"}
    draft = tmp_path / "draft.json"
    saved_draft = tmp_path / "drafts" / draft.name
    saved_draft.parent.mkdir()
    document = dict(fixture._population, reviewed=False, wire_stamps_read=True)
    write_json(saved_draft, document)
    write_json(fixture.population, dict(document, reviewed=True, review={
        "signed_by": "lead", "signed_utc": "then", "source_file": str(draft), "source_flux_job": "inventory",
        "source_sha256": common.sha256(saved_draft), "spec": "test"}))
    fixture.population, draft_mount = stage_population(fixture.population, fixture.out)
    staged, target, mode = shlex.split(draft_mount)[1].split(":")
    assert Path(target) == draft and mode == "ro" and not draft.exists()
    draft.write_bytes(Path(staged).read_bytes())
    h1_job = tmp_path / "h1-job"
    h1_job.mkdir()
    (h1_job / "artifacts").mkdir()
    fixture.h1 = fixture.h1.rename(h1_job / "artifacts/h1.json")
    (h1_job / "jobid").write_text("\n")
    contract_path = fixture.source / "src/tessera/serving/runtime_contract.json"
    contract = json.loads(contract_path.read_text())
    for cell in contract["lane_eligibility"]["cells"]:
        cell["requires_serve_flags"] = ["TESSERA_SERVE_MODE=resident"]
    write_json(contract_path, contract)

    kernel = fixture.source / common.KERNEL
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"retained kernel")
    kernel_sha = common.sha256(kernel)
    monkeypatch.setattr(common, "KERNEL_SHA256", kernel_sha)
    monkeypatch.setattr(serving_build, "KERNEL_SHA256", kernel_sha)
    monkeypatch.setattr(serving_build, "check_retained_source", fixture._retained_source)
    dirty = []

    def git(command, **kwargs):
        if command[1] == "status":
            return " M src\n" if dirty else ""
        return "cell-commit\n"

    monkeypatch.setattr(row.subprocess, "check_output", git)
    image = IMAGES["N"]
    options = start_env.container_options(tmp_path)
    mount = next(word for word in options if ":/run/numerics/image-inspect.json:" in word)
    inspect_path = Path(mount.split(":", 1)[0])
    write_json(inspect_path, [{"Id": "sha256:" + image["local_id"] + "0" * 52, "Digest": image["image_digest"]}])
    monkeypatch.setenv("NUMERICS_IMAGE_INSPECT", str(inspect_path))
    import importlib.metadata
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: image[name] if name in image else version(name))
    monkeypatch.setattr(row, "check_runtime", common.check_runtime)
    monkeypatch.setattr(cases, "DTable", fixture._dtable)
    monkeypatch.setattr(sass_slice, "match_library", fixture._match_library)
    monkeypatch.setattr(sys, "path", sys.path[:])

    retained, _ = retained_binary_receipts
    library, = json.loads(retained.read_text())["libraries"]
    identity = build.library_identity(library["library"]["path"], tmp_path / "p-cubins")
    for label in ("library", "probe"):
        for kind in ("sass", "elf"):
            path = tmp_path / f"{label}.{kind}"
            path.write_text("")
            library[label][kind] = {"path": str(path), "sha256": common.sha256(path)}
    library.update(probe_source_sha256="probe-source", cflags=[])
    write_json(fixture.build, {"commit": "cell-commit", "runtime": image, "libraries": [library]})
    write_json(fixture.build_provenance, {"commit": "cell-commit", "build_sha256": common.sha256(fixture.build),
        "libraries": [{**library, "native_source_path": str(kernel), "native_source_sha256": kernel_sha}]})
    dom, table = tmp_path / "domain", tmp_path / "table"
    np.array([-1, 0, 1], dtype="<f4").tofile(dom)
    np.array([-0.25, 0, 0.75], dtype="<f4").tofile(table)
    q1 = {"tables": [{"key": "e4m3", "library_so_sha256": library["library"]["sha256"],
                     "probe_sha256": library["probe"]["sha256"], "L_bits": 0, "dom_size": 3,
                     "d_nondecreasing_0_L": True, "dom_path": str(dom), "dom_sha256": common.sha256(dom),
                     "d_table_path": str(table), "d_table_sha256": common.sha256(table)}]}
    write_json(fixture.q1, q1)
    state = launch_state.LaunchState()
    state.bind_math(sys.modules["torch"])
    for path, fields in ((fixture.manifest, {"q1_table_sha256": common.sha256(fixture.q1),
                            "population_sha256": common.sha256(fixture.population), "contract_sha256": common.sha256(contract_path)}),
                         (fixture.h1, {"build_sha256": common.sha256(fixture.build), "P3": {"e4m3": identity}})):
        write_json(path, {**json.loads(path.read_text()), **state.fields(), **fields})
    monkeypatch.setattr(launch_state, "LaunchState", lambda path=launch_state.LEVERS, **kwargs: type(state)(path))
    addendum = tmp_path / "addendum.json"
    write_json(addendum, {"freeze_record_sha256": common.sha256(fixture.freeze), "rule_sha256": common.sha256(fixture.rule),
                          "reviewer": "lead", "frozen_utc": "then", "launch_levers_sha256": state.levers_sha256})
    monkeypatch.setattr(row, "FREEZE_ADDENDUM", addendum)
    monkeypatch.setattr(prepare_split, "RULE", fixture.rule)
    monkeypatch.setattr(prepare_split, "FREEZE", fixture.freeze)
    paths = {key: getattr(fixture, key) for key in prepare_split.INPUTS if key != "freeze_addendum"}
    paths["freeze_addendum"] = addendum

    def launch(attempt, phase, part, **unit):
        output = attempt / phase
        output.mkdir(parents=True)
        write_json(output / "instance.json", {"name": "fv1"})
        command = shlex.split(prepare_split.row_arguments({"module": "module-a", "canary": False}, tmp_path,
                              fixture.source, fixture.population, paths, phase,
                              unit={"case_set": part, **unit}, attempt=attempt))
        return output, command[command.index("experiments.numerics_rows.row") + 1:]

    from experiments.numerics_rows.reference import selftest
    monkeypatch.setattr(row, "selftest", lambda *, device: selftest(samples=32, device=device))
    monkeypatch.setattr(h1, "negative_controls", fixture._negative_controls)
    entries = []

    def compare(args, receipt, phases, groups):
        entries.append(args.out)
        receipt["row"]["exit_status"] = 0

    monkeypatch.setattr(dense_phases, "host_compare", compare)
    planned = row.planned_units
    def groups(*args):
        result = planned(*args)
        result[0][0]["family"] = "TESSERA_FP8"
        result[0][1].update(structure="dense", roles=[{"cut_axis": "whole", "cut_range": [0, 1],
            "cols": 32, "separation": {"recipes": []}, "scale_index": []}])
        result[0][3][0]["m_config"]["chunk_size"] = 32
        return result
    monkeypatch.setattr(row, "planned_units", groups)

    attempts = count(1)

    def run(status, flag=None, value=None):
        output, argv = launch(tmp_path / "compare-attempts" / str(next(attempts)), "compare", "all")
        if flag is not None:
            index = argv.index(flag)
            argv[index:index + 2] = [] if value is None else [flag, value]
        monkeypatch.setattr(sys, "argv", ["row", *argv])
        with pytest.raises(SystemExit) as exited:
            row.main()
        assert exited.value.code == status
        if status != 2:
            return json.loads((output / "numerics_row.json").read_text())

    receipt = run(0)
    out = tmp_path / "compare-attempts/1/compare"
    assert entries == [out] and (out / "native_ops.json").is_file()
    assert receipt["row"]["phase_jobs"]["compare"]["name"] == "fv1"
    assert receipt["tessera"]["retained_source"]["kernel_source_sha256"] == kernel_sha
    assert receipt["probe"]["finite_certificate"]["held"]
    assert receipt["rule"]["preconditions"]["P5"]["h1_job"] == "h1-job"
    from experiments.numerics_rows import gpu_time

    def two_cuts(*inputs):
        group, = groups(*inputs)
        entry, module, _, units = copy.deepcopy(group)
        cut = {"size": 2, "rank": 0}
        units[0]["tp"] = cut
        return [group, (entry, module, cut, units)]

    with monkeypatch.context() as patch:
        patch.setattr(row, "planned_units", two_cuts)
        patch.setattr(gpu_time, "RowMeter", lambda: SimpleNamespace(finish=lambda: {}))
        patch.setattr(gpu_time, "preflight", lambda **kwargs: {"verdict": "FREE"})
        patch.setattr(gpu_time, "self_visibility", lambda torch: {"verdict": "VISIBLE"})
        patch.setattr(dense_phases, "host_prepare", compare)
        patch.setattr(dense_phases, "device_capture", lambda *a, **kw: compare(*a))
        for phase in ("prepare", "capture"):
            for part in ("gate", "diagnostics"):
                for size_selected in (True, False):
                    attempt = tmp_path / f"{phase}-{part}-{size_selected}"
                    output, argv = launch(attempt, phase, part, size=1)
                    if not size_selected:
                        start = argv.index("--tp-size")
                        assert argv[start + 2] == "--tp-rank"
                        argv = argv[:start] + argv[start + 4:]
                    patch.setattr(sys, "argv", ["row", *argv])
                    with pytest.raises(SystemExit) as exited:
                        row.main()
                    result = json.loads((output / "numerics_row.json").read_text())
                    assert exited.value.code == (0 if size_selected else 3)
                    if size_selected:
                        assert entries.pop() == output
                        assert result["inputs"]["selected_cuts"] == [{"size": 1, "rank": 0}]
                    else:
                        assert result["row"]["stop_cause"] == "a case-set job requires one module and one TP unit"
    draft.write_bytes(draft.read_bytes() + b"\n")
    assert "signed population source sha256 differs" in run(3)["row"]["stop_cause"]
    draft.write_bytes(saved_draft.read_bytes())
    run(2, "--phase")
    assert "--phase-local-output requires --phase" in capsys.readouterr().err
    run(2, "--phase", "invalid")
    assert "argument --phase: invalid choice" in capsys.readouterr().err

    for path, field, value, reason in (
        (addendum, "launch_levers_sha256", "bad", "lever freeze addendum has no lever-file digest"),
        (addendum, "launch_levers_sha256", "b" * 64, "lever freeze addendum differs from frozen lever binding"),
        (fixture.manifest, "population_sha256", "other", "case manifest input identity differs"),
        (fixture.manifest, "modules", [], "case manifest omitted a population module"),
        (fixture.manifest, "launch_state_names", ["duplicate", "duplicate"], "launch state names are missing or malformed"),
        (fixture.manifest, "launch_state", {}, "launch state values must not be recorded"),
        (fixture.manifest, "launch_state_sha256", "bad", "launch state binding is missing or malformed"),
        (fixture.manifest, "launch_selectors", [], "launch state selectors are missing or malformed"),
        (fixture.manifest, "launch_selectors", {"TESSERA_SERVE_MODE": "other"}, "outside the non-secret list"),
        (fixture.manifest, "launch_state_sha256", "b" * 64, "C1 launch state or lever file differs"),
        (fixture.h1, "launch_state_sha256", "b" * 64, "H1 launch state or lever file differs"),
        (fixture.h1, "preconditions_confirmed", False, "H1 did not confirm this binary set"),
        (fixture.h1, "P3", {"e4m3": {}}, "H1 code identity differs"),
        (fixture.h1, "flux_job_id", "", "rule receipt requires a confirmed H1 job"),
    ):
        original = path.read_bytes()
        write_json(path, {**json.loads(original), field: value})
        assert reason in run(3)["row"]["stop_cause"]
        path.write_bytes(original)
    assert "split phase requires --phase-instance" in run(3, "--phase-instance")["row"]["stop_cause"]
    assert "lever freeze addendum is not at its fixed rule path" in run(
        3, "--freeze-addendum", str(tmp_path / "other-addendum.json"))["row"]["stop_cause"]
    dirty.append(True)
    assert "kernel source is not clean commit" in run(3)["row"]["stop_cause"]
    dirty.clear()
    kernel.write_bytes(b"changed kernel")
    assert "kernel source sha256 differs" in run(3)["row"]["stop_cause"]
    kernel.write_bytes(b"retained kernel")
    inspect_path.unlink()
    assert "missing Podman image inspection" in run(3)["row"]["stop_cause"]
    write_json(inspect_path, [{"Id": "sha256:" + image["local_id"] + "0" * 52, "Digest": image["image_digest"]}])
    evidence = library["library"]["sass"]["path"]
    Path(evidence).write_text("changed")
    assert "changed library sass evidence" in run(3)["row"]["stop_cause"]
    Path(evidence).write_text("")
    original_contract, original_manifest = contract_path.read_bytes(), fixture.manifest.read_bytes()
    for fields, reason in (({"runtime": {"execution_modes": ["graph"]}}, "source cell does not admit the eager row"),
                           ({"requires_serve_flags": ["TESSERA_SERVE_MODE=other"]}, "required selector differs from the source cell")):
        changed = json.loads(original_contract)
        changed["lane_eligibility"]["cells"][0].update(fields)
        write_json(contract_path, changed)
        write_json(fixture.manifest, {**json.loads(original_manifest), "contract_sha256": common.sha256(contract_path)})
        assert reason in run(3)["row"]["stop_cause"]
    contract_path.write_bytes(original_contract)
    fixture.manifest.write_bytes(original_manifest)
    table.write_bytes(b"changed")
    assert "changed D table" in run(3)["row"]["stop_cause"]
    np.array([0.25, 0, 0.75], dtype="<f4").tofile(table)
    q1["tables"][0]["d_table_sha256"] = common.sha256(table)
    write_json(fixture.q1, q1)
    write_json(fixture.manifest, {**json.loads(fixture.manifest.read_text()), "q1_table_sha256": common.sha256(fixture.q1)})
    assert "D table does not establish the gate sign facts" in run(3)["row"]["stop_cause"]
    assert entries == [out]


def test_generated_launch_policy(tmp_path, monkeypatch):
    import csv
    from experiments.numerics_rows import launch_state, start_env

    options = start_env.container_options(tmp_path)
    mapping = dict(options[i + 1].split("=", 1) for i, word in enumerate(options) if word == "--env")
    for name, value in mapping.items():
        monkeypatch.setenv(name, value)
    state = launch_state.LaunchState()
    state.check()
    assert set(mapping) <= state.mapping.keys()
    assert len(launch_state.digest(mapping)) == 64
    with pytest.raises(Invalid, match="launch state must map names to exact strings"):
        launch_state.digest({**mapping, "TESSERA_SERVE_MODE": 1})
    monkeypatch.delenv("TESSERA_SERVE_MODE")
    with pytest.raises(Invalid, match="launch state changed after worker start: TESSERA_SERVE_MODE"):
        state.check()

    with launch_state.LEVERS.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
    path = tmp_path / "launch-levers.tsv"
    for name, field, value, reason in (
        ("TESSERA_SERVE_MODE", "name", "", "no environment name"),
        ("TESSERA_SERVE_MODE", "launch_state", "malformed", "policy is malformed or inconsistent"),
        ("PATH", "evidence", "", "exclusion has no cited reason"),
        ("TESSERA_SERVE_MODE", "public_values", "1", "malformed non-secret enum policy"),
        ("TESSERA_SERVE_MODE", "public_values", '["different"]', "inconsistent non-secret enum policy"),
        ("TESSERA_SERVE_MODE", None, None, "must retain the residency selector"),
        ("PATH", None, None, "must exclude the seven toolchain diagnostics"),
        ("PRISMAQUANT_DEV_MODE", None, None, "must include PRISMAQUANT_DEV_MODE"),
    ):
        updated = rows[:]
        index = next(i for i, record in enumerate(rows) if record["name"] == name)
        if field is None:
            updated = [record for record in rows if record["name"] != name]
        elif "inconsistent non-secret enum" in reason:
            updated.append({**updated[index], field: value})
        else:
            updated[index] = {**updated[index], field: value}
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, reader.fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(updated)
        with pytest.raises(Invalid, match=reason):
            launch_state.LaunchState(path)


def test_run_checks_identity_before_every_case_and_rechecks_after(row_execution):
    args, receipt, scope, events, _ = row_execution
    with pytest.raises(SystemExit):
        row.run(args, receipt)
    assert events == ["identity", *(["case"] * 4), "identity", "identity", "case", "identity"]
    assert scope["outcome"] == "PASS"
    assert scope["arm"]["library_identity_verified"] and scope["arm"]["library_identity_unchanged"]


def test_run_refuses_a_changed_compile_path_before_any_case(row_execution, monkeypatch):
    from experiments.numerics_rows.common import KERNEL, sha256
    from tessera.serving import ext
    from tessera import routed_fused as rf

    args, receipt, _, events, _ = row_execution
    source = args.source / KERNEL
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pinned test kernel")
    built = json.loads(args.build.read_text())
    built["libraries"][0]["name"] = "tessera_routed_fused_e4m3"
    write_json(args.build, built)
    h1_report = json.loads(args.h1.read_text())
    h1_report["build_sha256"] = sha256(args.build)
    write_json(args.h1, h1_report)
    write_json(args.build_provenance, {**built, "build_sha256": sha256(args.build), "libraries": [
        {**built["libraries"][0], "native_source_path": str(source), "native_source_sha256": sha256(source)}]})
    if hasattr(args, "_compile_paths"):
        from experiments.numerics_rows import serving_build

        monkeypatch.setattr(serving_build, "check_compile_paths", args._compile_paths)
        monkeypatch.setattr(serving_build, "KERNEL_SHA256", sha256(source))
    monkeypatch.setattr(ext, "native_source_path", lambda name: str(source))
    monkeypatch.setattr(rf, "_cflags", lambda *args: [])
    with pytest.raises(SystemExit) as passed:
        row.run(args, receipt)
    assert passed.value.code == 0 and "case" in events
    events.clear()
    monkeypatch.setattr(ext, "native_source_path", lambda name: str(args.source / "different" / KERNEL))
    with pytest.raises(Invalid, match="compile path"):
        row.run(args, receipt)
    assert events == []


def test_run_identity_runtime_error_is_an_invalid_unit(row_execution, monkeypatch):
    args, receipt, scope, events, _ = row_execution
    def error(*args):
        raise RuntimeError("mapped inode changed")
    monkeypatch.setattr(row, "unit_libraries", error)
    with pytest.raises(SystemExit):
        row.run(args, receipt)
    assert events == [] and scope["outcome"] == "INVALID"
    assert receipt["row"]["stop_cause"] is None and not receipt["findings"]
    assert "mapped inode changed" in scope["invalid"][0]


def test_run_library_change_after_cases_is_invalid(row_execution, monkeypatch):
    args, receipt, scope, events, served = row_execution
    def libraries(*args):
        value = copy.deepcopy(served)
        if "case" in events:
            value["library_so_sha256"] = "changed after cases"
        return [value]
    monkeypatch.setattr(row, "unit_libraries", libraries)
    with pytest.raises(SystemExit):
        row.run(args, receipt)
    assert len(events) == 4 and scope["outcome"] == "INVALID"
    assert not scope["arm"]["library_identity_unchanged"] and not receipt["findings"]


@pytest.mark.parametrize("field", ["source_commit", "contract_sha256"])
def test_run_refuses_a_c1_source_mismatch(row_execution, field):
    args, receipt, _, events, _ = row_execution
    with pytest.raises(SystemExit) as passed:
        row.run(args, receipt)
    assert passed.value.code == 0
    events.clear()
    manifest = json.loads(args.manifest.read_text())
    manifest[field] = "different source"
    write_json(args.manifest, manifest)
    with pytest.raises(Invalid, match="source commit or contract"):
        row.run(args, receipt)
    assert events == []


@pytest.mark.parametrize("finished", [False, None, "missing"])
def test_unfinished_c1_is_refused(finished):
    contract, population, manifest = population_cases()
    assert row.planned_units(contract, "image", population, manifest)
    if finished == "missing":
        manifest.pop("generation_finished")
    else:
        manifest["generation_finished"] = finished
    with pytest.raises(Invalid, match="generation is not finished"):
        row.planned_units(contract, "image", population, manifest)


@pytest.mark.parametrize("kind", ["dense", "routed_expert", "routed_projection", "repeated"])
def test_manifest_requires_every_declared_expert_role(kind):
    from experiments.numerics_rows.cases import configurations

    contract, population, manifest = population_cases()
    if kind.startswith("routed"):
        for entry, module in zip(population["modules"], manifest["modules"]):
            entry["experts"] = 2
            entry["structure"] = "routed_moe"
            module["structure"] = "routed_moe"
            for cut in module["cuts"]:
                cut["roles"] = [{"expert": e, "name": p} for e in range(2) for p in ("gate_proj", "up_proj", "down_proj")]
                cut["m_configs"] = configurations(
                    **cut["launch_geometry"], library="e4m3", sms=1, structure="routed_moe")
        for cell in contract["lane_eligibility"]["cells"]:
            cell["structure"] = "routed_moe"
    assert row.planned_units(contract, "image", population, manifest)
    roles = manifest["modules"][0]["cuts"][0]["roles"]
    if kind == "repeated":
        roles.append(copy.deepcopy(roles[0]))
    elif kind == "routed_expert":
        roles[:] = [r for r in roles if r["expert"] != 1]
    else:
        roles.pop()
    with pytest.raises(Invalid, match="declared role"):
        row.planned_units(contract, "image", population, manifest)


def joined_inputs(monkeypatch):
    from experiments.numerics_rows import join as join_module
    monkeypatch.setattr(join_module, "RULE_SHA", "rule", raising=False)
    monkeypatch.setattr(join_module, "H1_SHA", "h1", raising=False)
    from experiments.numerics_rows import scope
    from experiments.numerics_rows.common import IMAGES
    from experiments.numerics_rows.launch_state import LaunchState
    import torch
    worker = LaunchState()
    worker.bind_math(torch)
    state = worker.fields()

    contract, population, manifest = population_cases()
    population["image"] = "N"
    for entry in population["modules"]:
        entry.update(structure="dense", q256=1024)
    for cell in contract["lane_eligibility"]["cells"]:
        cell["runtime"]["image"] = IMAGES["N"]["image_digest"]
    geometry = {"rows": 128, "cols": 64, "tile_words": 4096, "blocks": 1, "merged": False}
    monkeypatch.setattr(scope, "wire_geometries", lambda entry: [({"size": size, "rank": rank}, dict(geometry)) for size, rank in ((1, 0), (2, 0), (2, 1))])
    monkeypatch.setattr(scope, "contract_document", lambda path=None: (contract, "contract"))
    manifests, receipts = [], []
    for entry, module in zip(population["modules"], manifest["modules"]):
        name = entry["name"]
        case = {"population_sha256": "signed", "selected_modules": [name], "generation_finished": True,
                "source_commit": "source", "contract_sha256": "contract", "q1_table_sha256": "q1", **state, "modules": [module]}
        manifests.append(("c1-" + name, case))
        units = row.planned_units(contract, IMAGES["N"]["image_digest"], {**population, "modules": [entry]}, case)
        owned = [u for _, _, _, group in units for u in group]
        for unit in owned:
            unit.update(outcome="PASS", launch_state_sha256=state["launch_state_sha256"])
            unit.update(triton_specializations=[], kernel_origins=[], unresolved_kernel_origins=[],
                        dependency_code={"observation_complete": True})
            unit.update({k: v for k, v in state.items() if k.startswith("dependency_math")})
            unit["unit_id"] = row.unit_identity(unit)
        receipts.append(("row-" + name, {"inputs": {"population_sha256": "signed", "selected_modules": [name],
            "case_manifest_sha256": "c1-" + name, "build_sha256": "build", "q1_table_sha256": "q1", "launch_levers_sha256": state["launch_levers_sha256"]},
            "tool": {"commit": "tool", "tree_sha256": "tree"}, "runtime": {"image_digest": IMAGES["N"]["image_digest"], **state},
            "tessera": {"commit": "source", "contract_sha256": "contract"}, "rule": {"h1_sha256": "h1", "sha256": "rule", "launch_levers_sha256": state["launch_levers_sha256"], "frozen_utc": "2026-11-16T20:00:00Z"},
            "row": {"utc_end": "finished", "exit_status": 0, "finalization_finished": True}, "units": owned}))
    return population, "signed", manifests, receipts


@pytest.mark.parametrize("mutation", ["missing", "repeated", "unfinished"])
def test_join_refuses_missing_repeated_or_unfinished_signed_modules(mutation, monkeypatch):
    from experiments.numerics_rows.join import join

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    result = join(population, digest, manifests, receipts)
    assert result["verdict"] == "PASS" and all(len(u["modules"]) == 2 for u in result["units"])
    if mutation == "missing":
        manifests.pop()
        receipts.pop()
    elif mutation == "repeated":
        manifests.append(manifests[0])
        receipts.append(receipts[0])
    else:
        manifests[0][1]["generation_finished"] = False
    if mutation == "missing":
        assert join(population, digest, manifests, receipts)["verdict"] == "UNATTESTED"
    else:
        with pytest.raises(Invalid):
            join(population, digest, manifests, receipts)


@pytest.mark.parametrize("left,right,expected", [("PASS", "FAIL", "FAIL"), ("STOPPED", "INVALID", "STOPPED"),
    ("PASS", "UNATTESTED", "UNATTESTED"), ("UNATTESTED", "STOPPED", "STOPPED")])
def test_join_uses_the_worst_module_outcome(left, right, expected, monkeypatch):
    from experiments.numerics_rows.join import join

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    receipts[0][1]["units"][0]["outcome"], receipts[1][1]["units"][0]["outcome"] = left, right
    result = join(population, digest, manifests, receipts)
    assert result["verdict"] == expected
    assert any(u["outcome"] == expected for u in result["units"])


@pytest.mark.parametrize("mutation", [None, "dense", "fp4", "many", "negative", "large", "tp"])
def test_routed_sample_guard_accepts_tp1_and_refuses_invalid_scopes(mutation):
    from experiments.numerics_rows.cases import sample_cut

    entry = {"structure": "routed_moe", "family": "TESSERA_FP8", "experts": 288}
    population, expert, size, rank = {"modules": [entry]}, 144, 1, 0
    assert sample_cut(population, expert, size, rank) == (1, 0)
    assert sample_cut(population, expert, 2, 1) == (2, 1)
    if mutation is None:
        return
    if mutation == "dense": entry["structure"] = "dense"
    elif mutation == "fp4": entry["family"] = "TESSERA_NVFP4"
    elif mutation == "many": population["modules"].append(copy.deepcopy(entry))
    elif mutation == "negative": expert = -1
    elif mutation == "large": expert = 288
    else: rank = 1
    with pytest.raises(Invalid, match="routed sample"):
        sample_cut(population, expert, size, rank)


def test_two_sample_rate_includes_tp1_and_all_2592_items():
    from experiments.numerics_rows.join import sample_rate

    base = {"population_sha256": "signed", "q1_table_sha256": "q1", "source_commit": "source", "contract_sha256": "contract",
            "generation_finished": True, "items": [{}, {}, {}], "modules": [{"name": "module", "experts": 288}]}
    first = {**base, "sample": {"expert": 0, "tp": {"size": 2, "rank": 0}}, "reference_seconds": 30}
    second = {**base, "sample": {"expert": 144, "tp": {"size": 1, "rank": 0}}, "reference_seconds": 60}
    report = sample_rate(first, second)
    assert report["mean_seconds_per_item"] == 15 and report["module_items"] == 2592
    assert report["projected_module_seconds"] == 38880 and report["sample_scope_only"]


def test_unattributed_cuda_error_rechecks_identity_without_dispatch_failure(row_execution, monkeypatch):
    from experiments.numerics_rows.common import BoundaryError

    args, receipt, scope, events, _ = row_execution
    row.Dense().pairs = lambda: []
    def failed(*args):
        raise BoundaryError({"call": "apply", "kind": "apply", "sync_before": False, "sync_after": None}, RuntimeError("outside launch"))
    monkeypatch.setattr(row, "dense_cases", failed)
    with pytest.raises(SystemExit) as stopped:
        row.run(args, receipt)
    assert stopped.value.code == 3 and scope["outcome"] == "STOPPED"
    assert scope["arm"]["library_identity_unchanged"] and events[-1] == "identity"
    assert not scope["completed"] and not scope["findings"]
    assert "emitted_pairs" not in scope["arm"]


@pytest.mark.parametrize("path", ["tool.commit", "tool.tree_sha256", "runtime.image_digest", "tessera.commit", "tessera.contract_sha256",
                                  "inputs.build_sha256", "inputs.q1_table_sha256", "rule.h1_sha256", "rule.sha256"])
def test_join_refuses_mixed_execution_identity(path, monkeypatch):
    from experiments.numerics_rows.join import join

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    owner, field = path.split(".")
    receipts[1][1][owner][field] = "different"
    with pytest.raises(Invalid, match="execution identities"):
        join(population, digest, manifests, receipts)


@pytest.mark.parametrize("field", ["h1_sha256", "sha256"])
def test_join_refuses_rows_that_agree_on_another_rule_or_h1(field, monkeypatch):
    from experiments.numerics_rows.join import join

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    for _, receipt in receipts:
        receipt["rule"][field] = "0" * 64
    try:
        join(population, digest, manifests, receipts)
    except Invalid as error:
        assert "frozen rule and its H1" in str(error)
    else:
        raise AssertionError("JOIN_ACCEPTS_ANOTHER_RULE_OR_H1")


@pytest.mark.parametrize("field", ["source_commit", "contract_sha256", "q1_table_sha256"])
def test_join_refuses_mixed_manifest_identity(field, monkeypatch):
    from experiments.numerics_rows.join import join

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    manifests[1][1][field] = "different"
    with pytest.raises(Invalid, match="execution identities"):
        join(population, digest, manifests, receipts)


@pytest.mark.parametrize("field", ["utc_end", "exit_status"])
def test_join_refuses_unfinished_rows(field, monkeypatch):
    from experiments.numerics_rows.join import join

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    receipts[1][1]["row"][field] = None
    with pytest.raises(Invalid, match="finished row"):
        join(population, digest, manifests, receipts)


def test_join_projects_launch_configuration_and_tp_rank_instead_of_tensor_geometry(monkeypatch):
    from experiments.numerics_rows.join import join

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    original = join(population, digest, manifests, receipts)
    unit = receipts[1][1]["units"][0]
    unit["tp"].update(cut_axis=1, cut_range=[0, 1024])
    unit["unit_id"], unit["outcome"] = row.unit_identity(unit), "FAIL"
    result = join(population, digest, manifests, receipts)
    assert len(result["units"]) == len(original["units"]) and result["verdict"] == "FAIL"
    unit["m_config"]["k_split"] = 2
    unit["unit_id"] = row.unit_identity(unit)
    with pytest.raises(Invalid, match="required signed scope"):
        join(population, digest, manifests, receipts)


def test_attributed_cuda_error_survives_a_failed_post_case_identity_read(row_execution, monkeypatch):
    from experiments.numerics_rows.common import BoundaryError

    args, receipt, scope, _, _ = row_execution
    original, calls = row.unit_libraries, []
    def identity(*args):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("post-case mapping disappeared")
        return original(*args)
    def failed(*args):
        raise BoundaryError({"call": "native-launch", "kind": "tessera", "sync_before": True, "sync_after": False}, RuntimeError("device assert"))
    monkeypatch.setattr(row, "unit_libraries", identity)
    monkeypatch.setattr(row, "dense_cases", failed)
    with pytest.raises(SystemExit):
        row.run(args, receipt)
    assert scope["outcome"] == "FAIL" and scope["cuda_errors"][0]["attributed_to"] == "native-launch"
    assert "post-case mapping disappeared" in scope["invalid"][0]


@pytest.mark.parametrize("marker", [False, None, "missing"])
def test_join_refuses_failed_finalization(marker, monkeypatch):
    from experiments.numerics_rows.join import join

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    final = receipts[1][1]["row"]
    final.update(exit_status=3, finalization_finished=marker)
    if marker == "missing":
        final.pop("finalization_finished")
    with pytest.raises(Invalid, match="finished row"):
        join(population, digest, manifests, receipts)


@pytest.mark.parametrize("step", ["attach_separation", "historical_units"])
def test_row_marks_finalization_only_after_all_steps(row_execution, monkeypatch, step):
    args, receipt, _, _, _ = row_execution
    with pytest.raises(SystemExit):
        row.run(args, receipt)
    assert receipt["row"].get("finalization_finished") is True
    def failed(*args):
        raise RuntimeError("finalization failed")
    monkeypatch.setattr(row, step, failed)
    with pytest.raises(RuntimeError, match="finalization failed"):
        row.run(args, receipt)
    assert receipt["row"]["finalization_finished"] is False


def test_row_audits_served_bytes_before_first_case(row_execution, monkeypatch):
    from experiments.numerics_rows import serving_build

    args, receipt, scope, events, _ = row_execution
    with pytest.raises(SystemExit):
        row.run(args, receipt)
    assert scope["outcome"] == "PASS" and "case" in events
    events.clear()
    original = serving_build.compare_binaries
    def failed(*args):
        result = original(*args)
        result.update(identity_equal=False, raw_file_audit={"unexplained_bytes": 1})
        return result
    monkeypatch.setattr(serving_build, "compare_binaries", failed)
    with pytest.raises(SystemExit):
        row.run(args, receipt)
    assert events == ["identity"] and scope["outcome"] == "INVALID"
    assert scope["arm"]["library_identity_comparisons"][0]["unexplained_bytes"] == 1
    assert "byte audit" in scope["invalid"][0]


def test_retained_source_checks_the_p_path_tree_and_records_mtime(tmp_path, monkeypatch):
    from experiments.numerics_rows import serving_build, common

    source = tmp_path / common.KERNEL
    source.parent.mkdir(parents=True)
    source.write_bytes(b"kernel")
    monkeypatch.setattr(serving_build, "source_layout", lambda *args: (tmp_path, {"commit": "P"}))
    monkeypatch.setattr(serving_build, "check_source", lambda *args: "P")
    monkeypatch.setattr(common, "source_digest", lambda *args: "tree-sha256")
    monkeypatch.setattr(serving_build.subprocess, "check_output", lambda *args, **kwargs: "tree\n")
    observed = serving_build.check_retained_source(tmp_path, None, None)
    assert observed["source_mtime_ns"] == source.stat().st_mtime_ns and observed["git_tree_sha"] == "tree"
    assert observed["kernel_source_sha256"] == common.sha256(source)
    with pytest.raises(Invalid, match="retained source tree"):
        serving_build.check_retained_source(tmp_path / "clone", None, None)
    monkeypatch.setattr(serving_build.subprocess, "check_output", lambda cmd, **kwargs: "changed\n" if cmd[-1] == "HEAD:src" else "tree\n")
    with pytest.raises(Invalid, match="tree differs"):
        serving_build.check_retained_source(tmp_path, None, None)


def test_dense_graph_capture_prepares_metadata_before_capture(monkeypatch):
    import torch
    from experiments.numerics_rows.common import BoundaryRecorder
    from experiments.numerics_rows.reference import FiniteBound
    from experiments.t4_code import fused_e2m1_check
    from tessera import routed_fused as rf

    capturing, calls = False, []
    def host_copy():
        assert not capturing, "Cannot copy between CPU and CUDA tensors during CUDA graph capture"
        return torch.tensor([1]).tolist()
    def prepare(bundle):
        host_copy()
        calls.append("prepare")
        return bundle
    def launch(role, q, sf, out, counter):
        calls.append("launch")
        out.zero_()
    def capture(function, output, **kwargs):
        nonlocal capturing
        function()
        capturing = True
        try:
            function()
        finally:
            capturing = False
        return True
    monkeypatch.setattr(rf, "prepare_dense_role", prepare)
    monkeypatch.setattr(rf, "dense_forward", launch)
    monkeypatch.setattr(fused_e2m1_check, "graph_equal", capture)
    for name in ("randn", "linspace", "Generator"):
        function = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *args, _function=function, **kwargs: _function(*args, **{**kwargs, "device": "cpu"}))
    adapter = row.Dense.__new__(row.Dense)
    adapter.device = "cpu"
    adapter.family, adapter.library, adapter.rows, adapter.cols = "value", "value", 2, 2
    adapter.layer = SimpleNamespace(tessera_native=SimpleNamespace(role_bundles=[SimpleNamespace(rows=2)]))
    adapter.roles = [("down", torch.zeros(2, 2, dtype=torch.float64), torch.tensor(1.0), None)]
    adapter.quantized = lambda x, recorder: (x, None, x.double(), None)
    adapter.apply = lambda x: torch.zeros_like(x)
    adapter.reference = lambda *args, **kwargs: (torch.zeros(2, 2, dtype=torch.bfloat16), None, FiniteBound(torch.zeros(2, 2)))
    monkeypatch.setattr(row, "diagnostic", lambda *args: {})
    unit = row.new_unit({"m_config": {"M": 2, "k_split": 1}})
    row.dense_diagnostics(adapter, unit, BoundaryRecorder(lambda: None, []))
    assert unit["graph_replay_equal"] == [{"call": "dense-staged", "equal": True}]
    assert calls.count("prepare") == 10 and calls.count("launch") == 11
    assert unit["exact"]["composition"]["cases"] == 9 and not unit["findings"]


def test_all_gate_cases_precede_diagnostics_and_diagnostic_errors_do_not_stop_them(row_execution, monkeypatch):
    args, receipt, first, events, _ = row_execution
    from experiments.numerics_rows import scale_launch
    planned = row.planned_units(None)
    second = copy.deepcopy(first)
    second["m_config"]["M"] = 2
    entry, module, cut, _ = planned[0]
    monkeypatch.setattr(row, "planned_units", lambda *args: [(entry, module, cut, [first]), (entry, module, cut, [second])])
    def gate(adapter, unit, *args, **kwargs):
        assert unit["arm"]["library_identity_verified"]
        events.append(f"gate-{unit['m_config']['M']}")
        exact = unit["exact"]
        exact["sweep"]["elements"] = exact["chunk_pairs"]["x_elements"] = exact["composition"]["cases"] = 1
        exact["separation"]["records"] = [{"both": 1}]
        exact["scale_index"]["arrays"] = [{"launches": 1}]
    def diagnostic(adapter, unit, recorder):
        events.append("diagnostic")
        raise RuntimeError("capture diagnostic failed")
    for name in ("verify_cases", "dense_cases", "profile_unit"):
        monkeypatch.setattr(row, name, gate)
    monkeypatch.setattr(scale_launch, "check", gate)
    monkeypatch.setattr(row, "dense_diagnostics", diagnostic)
    with pytest.raises(SystemExit) as finished:
        row.run(args, receipt)
    assert finished.value.code == 0
    assert [e for e in events if e != "identity"] == [*(["gate-1"] * 4), *(["gate-2"] * 4), "diagnostic", "diagnostic"]
    assert all(u["outcome"] == "PASS" and u["completed"] and u["diagnostic_errors"] for u in (first, second))
    assert len(receipt["findings"]) == 2 and receipt["row"]["stop_cause"] is None
    assert set(receipt["row"]["phase_wall_seconds"]) == {"setup", "identity_audit", "gate_cases", "diagnostics"}


def test_phase_wall_times_are_exclusive_and_survive_errors(monkeypatch):
    from experiments.numerics_rows import gpu_time

    now, receipt = 0, {}
    monkeypatch.setattr(gpu_time.time, "monotonic", lambda: now)
    with pytest.raises(RuntimeError), gpu_time.PhaseTimes(receipt) as phases:
        now = 3
        with phases.measure("identity_audit"):
            now = 8
        now = 10
        phases.change("gate_cases")
        now = 17
        phases.change("diagnostics")
        now = 19
        raise RuntimeError("diagnostic interrupted")
    assert receipt["phase_wall_seconds"] == {"setup": 5, "identity_audit": 5, "gate_cases": 7, "diagnostics": 2}


def test_gpu_sampler_keeps_power_and_compute_app_memory_when_memory_used_is_na(monkeypatch):
    from experiments.numerics_rows import gpu_time
    import torch

    times, stop = iter((10, 20, 30)), iter((False, True))
    monkeypatch.setattr(gpu_time.time, "monotonic", lambda: next(times))
    meter = gpu_time.RowMeter.__new__(gpu_time.RowMeter)
    meter.start, meter.samples, meter.errors = 0, [], []
    meter.done = SimpleNamespace(wait=lambda seconds: next(stop), set=lambda: None)
    meter.thread = SimpleNamespace(join=lambda **kwargs: None, is_alive=lambda: False)
    lines = iter(("GPU-GB10, 0, 10.0, [N/A]\n", "GPU-GB10, [N/A], 10.7, [N/A]\n"))
    def query(command, **kwargs):
        assert command[0] == "nvidia-smi" and command[-1] == "--format=csv,noheader,nounits"
        return "701, 128\n702, 220\n" if "--query-compute-apps=pid,used_memory" in command else next(lines)
    monkeypatch.setattr(gpu_time.subprocess, "check_output", query)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2048)
    meter.sample()
    result = meter.finish()
    assert len(result["samples"]) == 2 and not result["errors"]
    assert all(s["memory_mib"] is None and s["compute_apps_memory_mib"] == 348 for s in result["samples"])
    assert result["peak_device_memory_mib"] == 348 and result["peak_power_watts"] == 10.7
    assert result["torch_peak_allocated_bytes"] == 1024 and result["torch_peak_reserved_bytes"] == 2048
    assert result["gpu_busy_minutes"] is None and result["busy_sampled_seconds"] == 0


def test_gpu_memory_query_failure_preserves_power_and_recovers(monkeypatch):
    from experiments.numerics_rows import gpu_time

    meter = gpu_time.RowMeter.__new__(gpu_time.RowMeter)
    meter.samples, meter.errors = [], []
    stop, values = iter((False, True)), iter((TimeoutError("compute query timed out"), "701, 16\n702, 32\n"))
    meter.done = SimpleNamespace(wait=lambda seconds: next(stop))
    def query(command):
        if "--query-gpu=" in command:
            return "GPU-GB10, 0, 10.0, [N/A]"
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(gpu_time.subprocess, "check_output", lambda command, **kwargs: query(command[1]))
    meter.sample()
    assert len(meter.samples) == 2 and meter.errors == ["compute query timed out"]
    assert [s["power_watts"] for s in meter.samples] == [10.0, 10.0]
    assert [s["compute_apps_memory_mib"] for s in meter.samples] == [None, 48]


def test_gpu_sampler_zeroes_stuck_counter_without_compute_process(monkeypatch):
    from experiments.numerics_rows import gpu_time

    meter = gpu_time.RowMeter.__new__(gpu_time.RowMeter)
    meter.samples, meter.errors = [], []
    stop, times = iter((False, True)), iter((10, 20))
    meter.done = SimpleNamespace(wait=lambda seconds: next(stop))
    monkeypatch.setattr(gpu_time.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(gpu_time.subprocess, "check_output", lambda command, **kw: "GPU-GB10, 96, 10, [N/A]" if command[1].startswith("--query-gpu=") else "")
    meter.sample()
    result = gpu_time.summarize(meter.samples, 0, 30)
    assert result["gpu_busy_minutes"] == 0
    assert all(s.get("compute_process_count") == 0 and s.get("compute_apps_memory_mib") == 0
               and s.get("nvml_busy_percent") == 96 and s["busy_percent"] == 0 for s in result["samples"])


def test_gpu_sampler_uses_any_compute_process_and_records_pids_once(monkeypatch):
    import os
    from experiments.numerics_rows import gpu_time

    meter = gpu_time.RowMeter.__new__(gpu_time.RowMeter)
    meter.start, meter.samples, meter.errors = 0, [], []
    stop, times, apps = iter((False, True)), iter((10, 20, 30)), iter(("123456789, 128", "123456790, 256"))
    meter.done = SimpleNamespace(wait=lambda seconds: next(stop), set=lambda: None)
    meter.thread = SimpleNamespace(join=lambda **kwargs: None, is_alive=lambda: False)
    monkeypatch.setattr(gpu_time.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(gpu_time.subprocess, "check_output", lambda command, **kw: "GPU-GB10, 96, 10, [N/A]" if command[1].startswith("--query-gpu=") else next(apps))
    meter.sample()
    result = meter.finish()
    assert result["gpu_busy_minutes"] == pytest.approx(0.16)
    assert result.get("pid_diagnostic", {}).get("nvml_pids") == [123456789]
    assert result["pid_diagnostic"]["in_container_pid"] == os.getpid()
    assert [s.get("compute_process_memory_mib") for s in result["samples"]] == [[128], [256]]
    assert all(s.get("compute_process_count") == 1 and "pid" not in s and "nvml_pids" not in s for s in result["samples"])


@pytest.mark.parametrize("apps,memory,servers,verdict", [
    ("", "[N/A]", [], "FREE"), ("", "307.19", [], "FREE"),
    ("", "307.2", [], "STOPPED"), ("701, 128", "[N/A]", [], "STOPPED"),
    ("", "[N/A]", [701], "STOPPED"), ("", "-1", [], "STOPPED")],
    ids=("empty-gb10", "below-limit", "at-limit", "compute-process", "vllm", "negative-memory"))
def test_capture_preflight_requires_free_memory_no_compute_and_no_vllm(apps, memory, servers, verdict):
    from experiments.numerics_rows import gpu_time

    assert hasattr(gpu_time, "preflight"), "feature absent: GPU-free capture preflight"
    def query(request):
        return f"GPU-GB10, 96, 10, {memory}" if request.startswith("--query-gpu=") else apps
    result = gpu_time.preflight(query, lambda: servers)
    assert result["verdict"] == verdict
    assert result["compute_process_count"] == (1 if apps else 0)
    if memory == "[N/A]":
        assert result["device_memory_mib"] is None
        assert result.get("memory_check") == "waived: GB10 reports memory.used as [N/A]; free judged by zero compute processes (lead decision 2026-11-17)"


def test_capture_preflight_detects_vllm_without_a_cuda_context(tmp_path):
    from experiments.numerics_rows import gpu_time

    assert hasattr(gpu_time, "vllm_processes"), "feature absent: host vLLM preflight"
    for pid, name, command in ((701, "python", b"python\0-m\0vllm.entrypoints.openai.api_server\0"),
                               (702, "VLLM::EngineCore", b"python\0"),
                               (703, "python", b"python\0capture.py\0vllm.log\0/path/vllm\0")):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "comm").write_text(name)
        (directory / "cmdline").write_bytes(command)
    assert sorted(gpu_time.vllm_processes(tmp_path)) == [701, 702]


@pytest.fixture
def capture_worker(tmp_path, monkeypatch):
    from experiments.numerics_rows import gpu_time, launch_state

    instance = tmp_path / "instance.json"
    write_json(instance, {"name": "fv1"})
    argv = ["row", "--image", "N", "--name", "capture", "--out", str(tmp_path), "--phase", "capture"]
    for name in ("source", "manifest", "population", "build", "build-provenance", "q1", "h1", "rule", "freeze"):
        argv.extend(("--" + name, str(tmp_path / name)))
    argv.extend(("--phase-instance", str(instance)))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(gpu_time, "RowMeter", lambda: SimpleNamespace(finish=lambda: {}))
    monkeypatch.setattr(launch_state, "LaunchState", lambda *a, **kw: SimpleNamespace(fields=lambda: {"launch_levers_sha256": "a" * 64}))
    return argv


@pytest.mark.parametrize("phase", ("prepare", "capture", "compare"))
def test_split_phase_requires_instance_record(capture_worker, tmp_path, monkeypatch, phase):
    del capture_worker[-2:]
    capture_worker[capture_worker.index("--phase") + 1] = phase
    monkeypatch.setattr(row, "run", lambda *args: pytest.fail("missing instance reached row setup"))
    with pytest.raises(SystemExit) as stopped:
        row.main()
    saved = json.loads((tmp_path / "numerics_row.json").read_text())
    assert stopped.value.code == 3 and saved["row"]["outcome"] == "INVALID"
    assert saved["row"]["stop_cause"] == "split phase requires --phase-instance"


def test_capture_preflight_stops_before_first_device_call(capture_worker, tmp_path, monkeypatch):
    from experiments.numerics_rows import gpu_time

    calls, free = [], True
    monkeypatch.setattr(gpu_time, "preflight", lambda **kw: {"verdict": "FREE" if free else "STOPPED"}, raising=False)
    monkeypatch.setattr(gpu_time, "self_visibility", lambda torch: {"verdict": "VISIBLE"}, raising=False)
    monkeypatch.setattr(row, "run", lambda *args: calls.append("first-device-call"))
    row.main()
    assert calls == ["first-device-call"]
    calls.clear()
    free = False
    with pytest.raises(SystemExit) as stopped:
        row.main()
    assert stopped.value.code == 3 and not calls
    saved = json.loads((tmp_path / "numerics_row.json").read_text())
    assert saved["row"]["outcome"] == "STOPPED" and saved["row"]["finalization_finished"] is False


@pytest.mark.parametrize("failure", ("empty", "query-error", "unknown-memory", "zero-memory"))
def test_capture_self_visibility_stops_before_row_setup(capture_worker, tmp_path, monkeypatch, failure):
    import torch
    from experiments.numerics_rows import gpu_time

    monkeypatch.setattr(gpu_time, "preflight", lambda **kw: {"verdict": "FREE"})
    monkeypatch.setattr(gpu_time.time, "sleep", lambda seconds: None)
    calls, visible = [], True
    monkeypatch.setattr(torch, "empty", lambda count, *, device: calls.append(("allocation", count, device)) or object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    def query(request):
        calls.append(request)
        if request.startswith("--query-gpu="):
            return "GPU-GB10, 96, 10, [N/A]"
        if visible:
            return "987654321, 32"
        if failure == "query-error":
            raise OSError("NVML query failed")
        return {"unknown-memory": "701, [N/A]", "zero-memory": "701, 0"}.get(failure, "")
    monkeypatch.setattr(gpu_time, "query", query)
    monkeypatch.setattr(row, "run", lambda *args: calls.append("row-setup"))
    row.main()
    assert calls == [("allocation", 1, "cuda"), "synchronize", "--query-compute-apps=pid,used_memory",
                     "--query-gpu=uuid,utilization.gpu,power.draw,memory.used", "row-setup"]
    visible = False
    calls.clear()
    with pytest.raises(SystemExit) as stopped:
        row.main()
    assert stopped.value.code == 3 and "row-setup" not in calls
    saved = json.loads((tmp_path / "numerics_row.json").read_text())
    assert saved["row"]["outcome"] == "STOPPED" and saved["row"]["finalization_finished"] is False
    assert "NVML self-visibility" in saved["row"]["stop_cause"]
    assert saved["row"]["gpu_self_visibility"]["verdict"] == "STOPPED"
    if failure in ("unknown-memory", "zero-memory"):
        assert saved["row"]["gpu_self_visibility"]["reason"] == "NVML self-visibility found no per-process memory"


def test_missing_gpu_fields_do_not_invent_zero_memory_or_activity():
    from experiments.numerics_rows.gpu_time import gpu_reading, summarize

    reading = gpu_reading("GPU-GB10, [N/A], 10.7, [N/A]")
    result = summarize([{**reading, "t": 1, "compute_apps_memory_mib": None},
                        {**reading, "t": 2, "compute_apps_memory_mib": None}], 0, 3)
    assert result["gpu_busy_minutes"] is None and result["peak_device_memory_mib"] is None
    assert result["peak_power_watts"] == 10.7


def test_unknown_compute_count_under_stuck_counter_stays_unknown(monkeypatch):
    from experiments.numerics_rows import gpu_time
    meter = gpu_time.RowMeter.__new__(gpu_time.RowMeter)
    meter.samples, meter.errors = [], []
    stop, times = iter((False, True)), iter((10, 20))
    meter.done = SimpleNamespace(wait=lambda seconds: next(stop))
    monkeypatch.setattr(gpu_time.time, "monotonic", lambda: next(times))
    def query(command, **kw):
        if command[1].startswith("--query-gpu="):
            return "GPU-GB10, 96, 10, [N/A]"
        raise TimeoutError("NVML compute query failed")
    monkeypatch.setattr(gpu_time.subprocess, "check_output", query)
    meter.sample()
    result = gpu_time.summarize(meter.samples, 0, 30)
    assert result["gpu_busy_minutes"] is None and result["busy_sampled_seconds"] == 0
    assert all(s.get("compute_process_count") is None and s["busy_percent"] is None for s in result["samples"])


def test_gpu_phase_idle_excludes_unknown_sample_intervals(monkeypatch):
    from experiments.numerics_rows import gpu_time
    meter = gpu_time.RowMeter.__new__(gpu_time.RowMeter)
    meter.start, meter.errors = 0, []
    meter.samples = [{"t": t, "phase": "gate_cases", "busy_percent": 96, "nvml_busy_percent": 96,
                      "compute_process_count": count} for t, count in ((0, None), (10, 0), (20, 0))]
    monkeypatch.setattr(gpu_time.time, "monotonic", lambda: 30)
    result = meter.snapshot()["phases"]["gate_cases"]
    assert result["wall_without_resident_kernels_seconds"] == 10
    assert result["known_sampled_wall_seconds"] == 10 and result["unknown_sampled_wall_seconds"] == 10
    assert result["max_compute_process_count"] == 0


@pytest.mark.parametrize("failure", ("empty", "query-error", "unknown-memory"))
def test_capture_self_visibility_retries_before_stopping(monkeypatch, failure):
    from experiments.numerics_rows import gpu_time
    calls, delays = [], []
    fake_torch = SimpleNamespace(empty=lambda *a, **kw: object(), cuda=SimpleNamespace(synchronize=lambda: None))
    def query(request):
        if request.startswith("--query-gpu="):
            return "GPU-GB10, 96, 10, [N/A]"
        calls.append(request)
        if len(calls) == 3:
            return "701, 128"
        if failure == "query-error":
            raise TimeoutError("NVML compute query failed")
        return "701, [N/A]" if failure == "unknown-memory" else ""
    monkeypatch.setattr(gpu_time, "query", query)
    monkeypatch.setattr(gpu_time.time, "sleep", delays.append)
    report = gpu_time.self_visibility(fake_torch)
    assert report["verdict"] == "VISIBLE" and report["compute_process_count"] == 1
    assert report.get("attempts") == 3 and report.get("nvml_busy_percent") == 96
    assert len(calls) == 3 and delays == [1, 1]


def test_host_preflight_main_writes_stopped_on_query_error(tmp_path, monkeypatch):
    import runpy
    from experiments.numerics_rows import gpu_time
    path = tmp_path / "gpu-preflight-host.json"
    monkeypatch.setattr(sys, "argv", ["gpu_time", "--preflight", str(path)])
    def query(*a, **kw):
        raise TimeoutError("NVML query failed")
    monkeypatch.setattr(gpu_time.subprocess, "check_output", query)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(gpu_time.__file__, run_name="__main__")
    assert stopped.value.code == 3
    assert json.loads(path.read_text()) == {"verdict": "STOPPED", "exception_type": "TimeoutError"}


def test_worker_preflight_names_its_limited_vllm_scope():
    from experiments.numerics_rows import gpu_time
    def query(request):
        return "GPU-GB10, 96, 10, [N/A]" if request.startswith("--query-gpu=") else ""
    report = gpu_time.preflight(query, processes=None)
    assert report["verdict"] == "FREE" and report["vllm_process_count"] is None
    assert report["nvml_busy_percent"] == 96 and "container PID namespace only" in report["vllm_check_scope"]


def test_worker_launch_state_encoding_and_named_exclusions(monkeypatch):
    from experiments.numerics_rows.launch_state import LaunchState, digest

    monkeypatch.setenv("TESSERA_SERVE_MODE", "resident")
    monkeypatch.setenv("TESSERA_UNRECOGNIZED_LEVER", "\u00e9")
    monkeypatch.setenv("TESSERA_ROUTE_TRACE", "/different/trace")
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    state = LaunchState()
    assert state.mapping["TESSERA_SERVE_MODE"] == "resident"
    assert state.mapping["TESSERA_UNRECOGNIZED_LEVER"] == "\u00e9"
    assert state.mapping["PRISMAQUANT_DEV_MODE"] == "0" and "TESSERA_ROUTE_TRACE" not in state.mapping
    encoded = json.dumps(state.mapping, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    assert state.sha256 == hashlib.sha256(encoded).hexdigest()
    monkeypatch.setenv("TESSERA_ROUTE_TRACE", "/another/trace")
    state.check()
    assert digest({}) != digest({"TESSERA_EMPTY": ""})
    with pytest.raises(Invalid, match="exact strings"):
        digest({"TESSERA_BAD": 1})


@pytest.mark.parametrize("prefix", ("TRITON_", "TORCH_", "PYTORCH_", "CUBLAS_", "CUBLASLT_", "CUDNN_", "NVIDIA_"))
@pytest.mark.parametrize("dev_mode", ("0", "1"))
def test_dependency_environment_refusal_in_both_dev_modes(monkeypatch, prefix, dev_mode):
    from experiments.numerics_rows.launch_state import LaunchState
    from experiments.numerics_rows.common import BoundaryRecorder

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    name = prefix + "UNLISTED_CONTROL"
    monkeypatch.setenv(name, "")
    state, calls = LaunchState(), []
    assert state.mapping[name] == "" and state.mapping["PRISMAQUANT_DEV_MODE"] == dev_mode
    recorder = BoundaryRecorder(lambda: None, [])
    recorder.before_call = state.check
    recorder.call("positive", "tessera", lambda: calls.append("positive"))
    monkeypatch.setenv(name, "changed")
    with pytest.raises(Invalid, match="launch state changed"):
        recorder.call("changed", "tessera", lambda: calls.append("changed"))
    assert calls == ["positive"]


@pytest.mark.parametrize("name", ("PATH", "CUDA_HOME", "CUDA_PATH", "PYTORCH_NVCC", "PYTORCH_ROCM_ARCH", "ROCM_HOME", "ROCM_PATH"))
def test_toolchain_changes_are_recorded_as_diagnostics(monkeypatch, name):
    from experiments.numerics_rows.launch_state import LaunchState

    monkeypatch.setenv(name, "/original")
    state = LaunchState()
    assert name not in state.mapping and name in state.diagnostics()["excluded_names"]
    monkeypatch.setenv(name, "/changed")
    state.check()
    assert name in state.diagnostics()["excluded_names"]


@pytest.mark.parametrize("package", ("tessera", "torch", "triton", "vllm"))
def test_worker_capture_refuses_each_prior_dependency_import(monkeypatch, package):
    from experiments.numerics_rows import launch_state

    monkeypatch.setattr(launch_state, "sys", SimpleNamespace(modules={}))
    state = launch_state.LaunchState(fresh=True)
    launch_state.sys.modules[package + ".submodule"] = object()
    state.check()
    with pytest.raises(Invalid, match="fresh worker"):
        launch_state.LaunchState(fresh=True)


@pytest.mark.parametrize("dev_mode", ("0", "1"))
@pytest.mark.parametrize("setting", ("get_float32_matmul_precision", "are_deterministic_algorithms_enabled", "is_deterministic_algorithms_warn_only_enabled"))
def test_math_mutation_refuses_before_native_call(row_execution, monkeypatch, dev_mode, setting):
    import torch
    from experiments.numerics_rows.common import BoundaryRecorder

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    from experiments.numerics_rows.launch_state import LaunchState
    worker, calls = LaunchState(), []
    worker.bind_math(torch)
    recorder = BoundaryRecorder(lambda: None, [])
    recorder.before_call = worker.check
    recorder.call("positive", "tessera", lambda: calls.append("positive"))
    previous = getattr(torch, setting)()
    monkeypatch.setattr(torch, setting, lambda: "high" if previous != "high" and isinstance(previous, str) else "highest" if isinstance(previous, str) else not previous)
    with pytest.raises(Invalid, match="math settings changed"):
        recorder.call("changed", "tessera", lambda: calls.append("changed"))
    assert calls == ["positive"]


@pytest.mark.parametrize("source", ("manifest", "h1"))
@pytest.mark.parametrize("dev_mode", ("0", "1"))
def test_row_refuses_foreign_math_settings_before_cases(row_execution, monkeypatch, source, dev_mode):
    from experiments.numerics_rows.launch_state import LaunchState, json_digest
    import torch

    args, receipt, _, events, _ = row_execution
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    args.launch_state = LaunchState()
    args.launch_state.bind_math(torch)
    for path in (args.manifest, args.h1):
        record = json.loads(path.read_text())
        record.update(args.launch_state.fields())
        write_json(path, record)
    with pytest.raises(SystemExit) as positive:
        row.run(args, receipt)
    assert positive.value.code == 0
    events.clear()
    path = getattr(args, source)
    record = json.loads(path.read_text())
    record["dependency_math"]["deterministic_algorithms"] = not record["dependency_math"]["deterministic_algorithms"]
    record["dependency_math_sha256"] = json_digest(record["dependency_math"])
    write_json(path, record)
    with pytest.raises(Invalid, match="math settings differ"):
        row.run(args, receipt)
    assert not events


def test_worker_refuses_a_tessera_import_before_capture(tmp_path, monkeypatch):
    from experiments.numerics_rows import gpu_time

    argv = ["row", "--image", "N", "--name", "fresh", "--out", str(tmp_path)]
    for name in ("source", "manifest", "population", "build", "build-provenance", "q1", "h1", "rule", "freeze"):
        argv.extend(("--" + name, str(tmp_path / name)))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setitem(sys.modules, "tessera.already_imported", SimpleNamespace())
    monkeypatch.setattr(gpu_time, "RowMeter", lambda: SimpleNamespace(finish=lambda: {}))
    monkeypatch.setattr(row, "run", lambda args, receipt: None)
    with pytest.raises(SystemExit) as stopped:
        row.main()
    assert stopped.value.code == 3
    assert "before its first Tessera import" in json.loads((tmp_path / "numerics_row.json").read_text())["row"]["stop_cause"]


def test_changed_worker_input_refuses_before_the_next_native_call(monkeypatch):
    from experiments.numerics_rows.launch_state import LaunchState
    from experiments.numerics_rows.common import BoundaryRecorder

    monkeypatch.setenv("TESSERA_NEW_UNLISTED_INPUT", "original")
    state, calls = LaunchState(), []
    recorder = BoundaryRecorder(lambda: None, [])
    recorder.before_call = state.check
    recorder.call("native", "tessera", lambda: calls.append("positive"))
    monkeypatch.setenv("TESSERA_NEW_UNLISTED_INPUT", "1")
    with pytest.raises(Invalid, match="launch state changed"):
        recorder.call("native", "tessera", lambda: calls.append("changed"))
    assert calls == ["positive"]


def test_row_refuses_input_changed_during_prepare(row_execution, monkeypatch):
    args, receipt, unit, events, _ = row_execution
    with pytest.raises(SystemExit) as positive:
        row.run(args, receipt)
    assert positive.value.code == 0 and unit["outcome"] == "PASS"
    events.clear()
    build = row.Dense
    def prepare(*args):
        adapter = build(*args)
        monkeypatch.setenv("TESSERA_SERVE_MODE", "changed during prepare")
        return adapter
    monkeypatch.setattr(row, "Dense", prepare)
    with pytest.raises(SystemExit):
        row.run(args, receipt)
    assert unit["outcome"] == "INVALID" and not events


@pytest.mark.parametrize("field", ("launch_state_names", "launch_state_sha256", "launch_levers_sha256"))
def test_row_refuses_c1_launch_state_or_policy_before_cases(row_execution, field):
    args, receipt, _, events, _ = row_execution
    with pytest.raises(SystemExit) as positive:
        row.run(args, receipt)
    assert positive.value.code == 0
    events.clear()
    manifest = json.loads(args.manifest.read_text())
    manifest[field] = ["TESSERA_CHANGED"] if field == "launch_state_names" else "b" * 64
    write_json(args.manifest, manifest)
    with pytest.raises(Invalid, match="C1 launch state"):
        row.run(args, receipt)
    assert not events


@pytest.mark.parametrize("field", ("cell_ids", "manifest_cell_ids"))
def test_population_and_c1_require_full_source_cell_ids(field):
    contract, population, manifest = population_cases()
    assert row.planned_units(contract, "image", population, manifest)
    target = population["modules"][0] if field == "cell_ids" else manifest["modules"][0]
    target["cell_ids"] = ["historical-v68-alias"]
    with pytest.raises(Invalid, match="cell IDs"):
        row.planned_units(contract, "image", population, manifest)


@pytest.mark.parametrize("field", ("cell", "launch_state_sha256"))
def test_separation_cannot_borrow_across_cell_or_launch_state(field):
    unit = row.new_unit({"cell": "cell", "symbol": "pair", "decoder": "decoder", "rung": 1024,
                         "tp": {"size": 2, "rank": 0}, "launch_state_sha256": "a" * 64})
    unit.update(outcome="PASS", separation_census_complete=True,
                m_config={"M": 64, "compiled_kernel_symbols": [{"symbol": "_Zsame", "library_so_sha256": "same"}]})
    target = copy.deepcopy(unit)
    target.update(needs_separation_transfer=True, separation_census_complete=False)
    row.attach_separation([unit, target])
    assert target["m_config"]["separation_source"]
    target["m_config"]["separation_source"] = None
    target[field] = "different"
    row.attach_separation([unit, target])
    assert target["m_config"]["separation_source"] is None and target["unattested"]


@pytest.mark.parametrize("mutation", (None, "gate", "up", "down", "missing", "ambiguous", "unsupported", "missing_count", "pm_false", "pm_false_on_piece_major", "duplicate_demangled"))
def test_profile_checks_prepared_layout_against_exact_compiled_symbols(tmp_path, monkeypatch, mutation):
    import torch
    from contextlib import nullcontext
    from experiments import routed_pair_oracle
    from experiments.t4_code import sass_dump
    from experiments.numerics_rows.common import BoundaryRecorder

    path = tmp_path / "sass.txt"
    path.write_text("fixture SASS function roster")
    modes = (0, 2)
    names = [f"void routed_fused_kernel<true, {mode}, false, false, 4, false, 64, true, false>(Params)" for mode in modes]
    functions = ["_Zgate", "_Zdown"]
    if mutation in ("pm_false", "pm_false_on_piece_major"):
        names = [name.replace("64, true, false", "64, false, false") for name in names]
    if mutation == "duplicate_demangled":
        functions.append("_Zgate_alias")
        names.append(names[0])
    if mutation == "unsupported":
        names[0] = names[0].replace("64, true, false", "64, unknown, false")
    profiler_names = names if mutation != "ambiguous" else ["routed_fused_kernel<...>"]
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: nullcontext())
    calls = []
    def kernels(*args, **kwargs):
        calls.append(True)
        selected = profiler_names[:1] if mutation == "missing_count" and len(calls) == 2 else profiler_names
        return [{"name": n} for n in selected]
    monkeypatch.setattr(routed_pair_oracle, "kernel_table", kernels)
    monkeypatch.setattr(sass_dump, "parse_sass", lambda text: dict.fromkeys(functions))
    monkeypatch.setattr(row.subprocess, "check_output", lambda *args, **kwargs: "\n".join(names))
    for name in ("zeros", "tensor"):
        function = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *args, _function=function, **kwargs: _function(*args, **{**kwargs, "device": "cpu"}))
    library = {"sass": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
               "library_so_sha256": "a" * 64, "library_code_sha256": "b" * 64, "cubin_sha256s": []}
    native = SimpleNamespace(**{role: SimpleNamespace(word_layout="piece_major") for role in ("gate", "up", "down")})
    if mutation == "pm_false":
        for role in ("gate", "up", "down"):
            getattr(native, role).word_layout = "legacy"
    if mutation in ("gate", "up", "down"):
        getattr(native, mutation).word_layout = "legacy"
    elif mutation == "missing":
        del native.up.word_layout
    adapter = SimpleNamespace(native=native, cols=2, experts=16, served_identity=library, apply=lambda *args: None)
    unit = row.new_unit({"m_config": {"M": 2, "route_counts": [1, 2]}, "arm": {}})
    row.profile_unit(adapter, unit, BoundaryRecorder(lambda: None, []), routed=True)
    if mutation in ("gate", "up", "down", "pm_false_on_piece_major"):
        assert unit["findings"] and not unit["unattested"]
    elif mutation not in (None, "pm_false"):
        assert unit["unattested"] and not unit["findings"]
    else:
        assert unit.get("prepared_layouts") == {r: "legacy" if mutation == "pm_false" else "piece_major" for r in ("gate", "up", "down")}
        assert len(unit["layout_symbol_proof"]) == 2 and not unit["findings"] and not unit["unattested"]
        assert [r["route_count"] for r in unit["arm"]["profiler_launches"]] == [1, 2]


def test_unit_key_contains_launch_state():
    unit = row.new_unit({"cell": "cell", "symbol": "pair", "decoder": "decoder", "rung": 1024,
                         "m_config": {"M": 1}, "tp": {"size": 2, "rank": 0}, "launch_state_sha256": "a" * 64})
    before = row.unit_identity(unit)
    unit["launch_state_sha256"] = "b" * 64
    assert before != row.unit_identity(unit)


def test_row_refuses_a_different_frozen_lever_file(row_execution):
    args, receipt, _, events, _ = row_execution
    with pytest.raises(SystemExit) as positive:
        row.run(args, receipt)
    assert positive.value.code == 0
    events.clear()
    frozen = json.loads(args.freeze.read_text())
    frozen["launch_levers_sha256"] = "b" * 64
    write_json(args.freeze, frozen)
    with pytest.raises(Invalid, match="frozen file"):
        row.run(args, receipt)
    assert not events


def test_row_refuses_missing_required_residency_selector(row_execution, monkeypatch):
    args, receipt, _, events, _ = row_execution
    from experiments.numerics_rows.launch_state import LaunchState
    monkeypatch.delenv("TESSERA_SERVE_MODE", raising=False)
    args.launch_state = LaunchState()
    # The positive control binds both prerequisite records to this worker.
    for path in (args.manifest, args.h1):
        record = json.loads(path.read_text())
        record.update(args.launch_state.fields())
        write_json(path, record)
    with pytest.raises(SystemExit) as positive:
        row.run(args, receipt)
    assert positive.value.code == 0
    events.clear()
    path = args.source / "src/tessera/serving/runtime_contract.json"
    contract = json.loads(path.read_text())
    for cell in contract["lane_eligibility"]["cells"]:
        cell["requires_serve_flags"] = ["TESSERA_SERVE_MODE=resident"]
    # This is a selector test, so keep the source-contract binding valid.
    write_json(path, contract)
    manifest = json.loads(args.manifest.read_text())
    manifest["contract_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.manifest, manifest)
    with pytest.raises(Invalid, match="required selector"):
        row.run(args, receipt)
    assert not events


def test_changed_launch_state_inside_a_nested_boundary_stays_invalid(monkeypatch):
    from experiments.numerics_rows.launch_state import LaunchState
    from experiments.numerics_rows.common import BoundaryRecorder

    monkeypatch.setenv("TESSERA_UNKNOWN_NEW_INPUT", "original")
    state, calls = LaunchState(), []
    recorder = BoundaryRecorder(lambda: None, [])
    recorder.before_call = state.check
    recorder.call("outer", "apply", lambda: recorder.call("inner", "tessera", lambda: calls.append("positive")))
    def changed():
        monkeypatch.setenv("TESSERA_UNKNOWN_NEW_INPUT", "1")
        recorder.call("inner", "tessera", lambda: calls.append("changed"))
    with pytest.raises(Invalid, match="launch state changed"):
        recorder.call("outer", "apply", changed)
    assert calls == ["positive"]


def test_changed_launch_state_refuses_before_graph_replay(monkeypatch):
    from experiments.numerics_rows.launch_state import LaunchState
    from experiments.numerics_rows.common import BoundaryRecorder
    from experiments.numerics_rows.run_routed import graph_record
    from experiments.t4_code import fused_e2m1_check

    monkeypatch.setenv("TESSERA_UNKNOWN_NEW_INPUT", "original")
    state, calls = LaunchState(), []
    recorder = BoundaryRecorder(lambda: None, [])
    recorder.before_call = state.check
    change = False
    def graph(function, output, *, before_call=lambda: None):
        function()
        if change:
            monkeypatch.setenv("TESSERA_UNKNOWN_NEW_INPUT", "1")
        before_call()
        calls.append("replay")
        return True
    monkeypatch.setattr(fused_e2m1_check, "graph_equal", graph)
    unit = row.new_unit({})
    graph_record(unit, "graph", lambda: None, None, recorder)
    change = True
    with pytest.raises(Invalid, match="launch state changed"):
        graph_record(unit, "graph", lambda: None, None, recorder)
    assert calls == ["replay"] and recorder.enabled


@pytest.mark.parametrize("dev_mode", ("0", "1"))
@pytest.mark.parametrize("mutation", ("change", "remove"))
def test_whole_start_environment_refuses_uncatalogued_input(monkeypatch, dev_mode, mutation):
    from experiments.numerics_rows.launch_state import LaunchState
    from experiments.numerics_rows.common import BoundaryRecorder

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    monkeypatch.setenv("UNCLASSIFIED_VENDOR_CONTROL", "initial")
    worker, calls = LaunchState(), []
    recorder = BoundaryRecorder(lambda: None, [])
    recorder.before_call = worker.check
    recorder.call("positive", "tessera", lambda: calls.append("positive"))
    if mutation == "change":
        monkeypatch.setenv("UNCLASSIFIED_VENDOR_CONTROL", "changed")
    else:
        monkeypatch.delenv("UNCLASSIFIED_VENDOR_CONTROL")
    with pytest.raises(Invalid, match="launch state changed"):
        recorder.call("changed", "tessera", lambda: calls.append("changed"))
    assert calls == ["positive"]


def test_code_added_names_do_not_change_the_start_binding(monkeypatch):
    from experiments.numerics_rows.launch_state import LaunchState

    monkeypatch.delenv("TESSERA_IMPORT_ADDED_INPUT", raising=False)
    state = LaunchState()
    before = state.fields()
    monkeypatch.setenv("TESSERA_IMPORT_ADDED_INPUT", "added by pinned import")
    state.check()
    assert state.fields() == before
    assert "TESSERA_IMPORT_ADDED_INPUT" in state.diagnostics()["added_names"]


def test_environment_records_never_write_private_values_or_per_name_hashes(monkeypatch):
    from experiments.numerics_rows.launch_state import LaunchState, digest

    private = "do-not-record-this-private-value"
    monkeypatch.setenv("UNCLASSIFIED_VENDOR_CONTROL", private)
    monkeypatch.setenv("PATH", private)
    monkeypatch.setenv("VLLM_API_KEY", private)
    state = LaunchState()
    fields = state.fields()
    if private in json.dumps([fields, state.diagnostics()]):
        pytest.fail("environment record contains a private value; contents suppressed", pytrace=False)
    assert "UNCLASSIFIED_VENDOR_CONTROL" in fields["launch_state_names"]
    assert "VLLM_API_KEY" not in state.mapping
    assert fields["launch_state_sha256"] == digest(state.mapping)
    assert "launch_state" not in fields
    latched = state.latched_diagnostic({"UNLISTED_SECRET": private, "TESSERA_SERVE_MODE": "resident"})
    assert latched["names"] == ["TESSERA_SERVE_MODE", "UNLISTED_SECRET"]
    assert latched["non_secret_values"] == {"TESSERA_SERVE_MODE": "resident"}
    recorded = json.dumps([fields, state.diagnostics(), latched])
    assert private not in recorded and hashlib.sha256(private.encode()).hexdigest() not in recorded


def test_start_environment_recipe_has_no_job_values():
    from experiments.numerics_rows.start_env import container_options, worker_command

    first, second = container_options("/var/tmp/job-a"), container_options("/var/tmp/job-b")
    assert [v for v in first if "=" in v] == [v for v in second if "=" in v]
    assert "--device" not in first
    assert "--env PYTHONPATH=" not in worker_command("/retained/cell", "module", [])


def test_environment_proof_checks_container_exit_before_socket_wait():
    from experiments.numerics_rows import env_proof
    assert hasattr(env_proof, "receive"), "feature absent: environment proof process and stage monitor"
    receive = env_proof.receive
    from types import SimpleNamespace

    events = []
    listener = SimpleNamespace(settimeout=lambda _: None,
                               accept=lambda: pytest.fail("waited for a dead container"))
    with pytest.raises(RuntimeError, match="before worker connection"):
        receive(listener, SimpleNamespace(poll=lambda: 125), lambda stage, **data: events.append((stage, data)))
    assert events == [("container-exit", {"container_exit": 125})]


def test_environment_proof_records_import_stage_before_read_timeout():
    from experiments.numerics_rows import env_proof
    assert hasattr(env_proof, "receive"), "feature absent: environment proof process and stage monitor"
    receive = env_proof.receive
    from types import SimpleNamespace

    class Channel:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def settimeout(self, _): pass
        def makefile(self, _): return self
        def readline(self):
            if not hasattr(self, "sent"):
                self.sent = True
                return b'{"stage":"import-vllm"}\n'
            raise TimeoutError()
    events = []
    listener = SimpleNamespace(settimeout=lambda _: None, accept=lambda: (Channel(), None))
    with pytest.raises(TimeoutError):
        receive(listener, SimpleNamespace(poll=lambda: None), lambda stage: events.append(stage))
    assert events == ["import-vllm"]


def test_environment_proof_exercises_h1_lazy_imports_and_records_writers(monkeypatch):
    import os
    from experiments.numerics_rows import env_proof

    assert hasattr(env_proof, "import_dependencies"), "feature absent: lazy H1 import and environment-writer proof"
    private = "private-library-search-value"
    monkeypatch.setenv("LD_LIBRARY_PATH", private)
    original = type(os.environ).__setitem__
    loaded, stages = [], []
    def importer(name):
        loaded.append(name)
        if name == "cv2":
            os.environ["LD_LIBRARY_PATH"] = "/image/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
        return SimpleNamespace()
    monkeypatch.setattr(env_proof.importlib, "import_module", importer)
    imports, writes = env_proof.import_dependencies(SimpleNamespace(bind_math=lambda _: None), stages.append)
    assert loaded == ["torch", "triton", "vllm", "tessera", "vllm.config", "vllm.distributed", "cv2"]
    assert stages == ["import-" + name for name in loaded]
    assert all(r["imported"] for r in imports)
    assert len(writes) == 1 and writes[0]["name"] == "LD_LIBRARY_PATH"
    assert writes[0]["file"] == __file__ and writes[0]["line"] > 0
    assert type(os.environ).__setitem__ is original
    if private in json.dumps([imports, writes]):
        pytest.fail("environment writer proof disclosed a value", pytrace=False)


def test_library_search_path_exclusion_does_not_exclude_other_inputs(monkeypatch):
    from experiments.numerics_rows.launch_state import LaunchState

    monkeypatch.setenv("LD_LIBRARY_PATH", "start")
    monkeypatch.setenv("UNLISTED_LIBRARY_CONTROL", "start")
    state = LaunchState()
    state.check()
    assert "LD_LIBRARY_PATH" not in state.fields()["launch_state_names"]
    monkeypatch.setenv("LD_LIBRARY_PATH", "changed-by-opencv")
    state.check()
    assert "LD_LIBRARY_PATH" in state.diagnostics()["excluded_names"]
    monkeypatch.setenv("UNLISTED_LIBRARY_CONTROL", "changed")
    with pytest.raises(Invalid, match="UNLISTED_LIBRARY_CONTROL"):
        state.check()


@pytest.mark.parametrize("field", ("launch_state_sha256", "start_environment_spec_sha256", "launch_levers_sha256"))
def test_row_refuses_a_different_h1_launch_binding_before_cases(row_execution, field):
    args, receipt, _, events, _ = row_execution
    with pytest.raises(SystemExit) as positive:
        row.run(args, receipt)
    assert positive.value.code == 0
    events.clear()
    doc = json.loads(args.h1.read_text())
    doc[field] = "b" * 64
    write_json(args.h1, doc)
    with pytest.raises(Invalid, match="H1 launch state or lever file"):
        row.run(args, receipt)
    assert not events


def test_public_environment_values_exclude_paths_commands_and_invalid_enums(monkeypatch):
    from experiments.numerics_rows.launch_state import LaunchState, check_fields
    private = "private-path-or-command"
    for name in ("TESSERA_EXT_DIR", "TESSERA_SOURCE_VERIFIER", "TESSERA_PRODUCER_SOURCE", "TESSERA_GIT", "TESSERA_SERVE_MODE"):
        monkeypatch.setenv(name, private)
    state = LaunchState()
    fields = state.fields()
    if private in json.dumps([fields, state.latched_diagnostic(state.mapping)]):
        pytest.fail("non-secret enum policy wrote a private value", pytrace=False)
    check_fields(fields)
    fields["launch_selectors"]["TESSERA_EXT_DIR"] = private
    with pytest.raises(Invalid, match="non-secret"):
        check_fields(fields)


def test_latched_flag_error_records_only_the_type():
    from tessera.serving import flags
    from experiments.numerics_rows.common import BoundaryRecorder, error_text, error_trace
    flags.reset_for_tests("TESSERA_ROUTE_TRACE")
    flags._latch("TESSERA_ROUTE_TRACE", "private-old-path")
    records = []
    try:
        with pytest.raises(Exception) as caught:
            BoundaryRecorder(lambda: None, records).call("flag", "tessera", lambda: flags._latch("TESSERA_ROUTE_TRACE", "private-new-path"))
        rendered = json.dumps([records, error_text(caught.value), error_trace(caught.value)])
        assert "private-old-path" not in rendered and "private-new-path" not in rendered
    finally:
        flags.reset_for_tests("TESSERA_ROUTE_TRACE")


def test_profile_copy_and_memset_events_are_diagnostics_only():
    import torch
    from experiments.routed_pair_oracle import kernel_table
    names = ["Memcpy HtoD (Pageable -> Device)", "Memcpy DtoD (Device -> Device)", "Memset (Device)", "native_kernel"]
    events = [SimpleNamespace(key=n, count=1, device_type=torch.autograd.DeviceType.CUDA, self_device_time_total=7) for n in names]
    profile = SimpleNamespace(key_averages=lambda: events, events=lambda: events)
    counts = {}
    records = kernel_table(profile, device_only=True, diagnostics=counts)
    assert [r["name"] for r in records] == ["native_kernel"]
    assert counts == {"memcpy": 2, "memset": 1}


def test_fixed_worker_launcher_reads_a_private_script_from_an_existing_mount(tmp_path):
    import subprocess
    from pathlib import Path
    from experiments.numerics_rows.start_env import container_options, worker_command
    options = container_options(str(tmp_path))
    workdir, = [v.split("=", 1)[1] for v in options if v.startswith("--workdir=")]
    mounts = [options[i + 1].rsplit(":", 2) for i, v in enumerate(options) if v == "-v"]
    locations = [Path(source) for source, target, mode in mounts if target == workdir]
    assert len(locations) == 1 and locations[0].is_dir(), "fixed work directory must be an existing mount"
    source = tmp_path / "cell with spaces"
    module = source / "src" / "launcher_fixture.py"
    module.parent.mkdir(parents=True)
    module.write_text("import json,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n")
    output = tmp_path / "arguments.json"
    expected = ["one value", "literal '$()' and ;"]
    command = worker_command(source, "launcher_fixture", [str(output), *expected])
    script = tmp_path / "row.sh"
    script.write_text("#!/bin/bash\nset -euo pipefail\nexec " + command + "\n")
    tmp_path.chmod(0o700)
    script.chmod(0o600)
    result = subprocess.run(["/bin/bash", "-euo", "pipefail", str(script)], cwd=locations[0],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    assert result.returncode == 0
    assert json.loads(output.read_text()) == expected
    assert script.stat().st_mode & 0o111 == 0


def test_compare_ignores_its_running_attempt(tmp_path, monkeypatch, row_execution):
    from experiments.numerics_rows import h1, launch_state, prepare_split, row
    from tests.test_numerics_unit_templates import payload_arguments, templates

    fixture, _, _, _, _ = row_execution
    addendum = tmp_path / "addendum.json"
    addendum.write_text(json.dumps({"freeze_record_sha256": row.sha256(fixture.freeze),
        "rule_sha256": row.sha256(fixture.rule), "reviewer": "lead", "frozen_utc": "then",
        "launch_levers_sha256": fixture.launch_state.levers_sha256}))
    for name in ("rule", "freeze"):
        monkeypatch.setattr(prepare_split, name.upper(), getattr(fixture, name))
    monkeypatch.setattr(row, "FREEZE_ADDENDUM", addendum)
    monkeypatch.setattr(launch_state, "LaunchState", lambda path=launch_state.LEVERS, **kwargs: type(fixture.launch_state)(path))
    monkeypatch.setattr(h1, "negative_controls", lambda **kwargs: [{"flagged_exactly": True}])
    monkeypatch.setattr(sys, "path", sys.path[:])
    paths = {key: str(getattr(fixture, key)) for key in prepare_split.INPUTS if key != "freeze_addendum"}
    paths["freeze_addendum"] = str(addendum)
    root = tmp_path / "module"
    config, directories = templates(root, module="module-a", paths=paths)
    directory = next(p for p in directories if p.name == "compare")
    attempt = root / "units" / directory.parent.name / "attempts/1"
    out = attempt / "compare"
    out.mkdir(parents=True)
    (out / "instance.json").write_text('{"name":"fv1"}')
    selected = {key: root / "units" / key / "attempts/1" for key in config["unit_keys"]}
    for path in selected.values():
        path.mkdir(parents=True)
    (attempt / "compare-inputs.json").write_text(json.dumps({key: str(path) for key, path in selected.items()}))
    groups = []
    for size, rank in ((1, 0), (2, 0), (2, 1)):
        cut = {"size": size, "rank": rank}
        role = {"cols": 32, "separation": {"recipes": []}, "scale_index": [], "cut_axis": "whole", "cut_range": [0, 1]}
        unit = row.new_unit({"module": "module-a", "cell": "dense", "role": "unit", "tp": cut,
                             "m_config": {"M": 1, "k_split": 1, "chunk_size": 32}})
        groups.append(({"name": "module-a", "family": "TESSERA_FP8"},
                       {"structure": "dense", "wire_sha256": "wire", "roles": [role]}, cut, [unit]))
    monkeypatch.setattr(row, "planned_units", lambda *args: groups)
    monkeypatch.setattr(sys, "argv", ["row", *payload_arguments(directory, attempt, tmp_path / "tools")])
    with pytest.raises(SystemExit) as exited:
        row.main()
    saved = json.loads((out / "numerics_row.json").read_text())
    assert saved["row"]["finalization_finished"], "COMPARE_REFUSES_ITS_OWN_ATTEMPT"
    assert exited.value.code == 3 and saved["row"]["outcome"] == "STOPPED"
    assert {entry["unit"] for entry in saved["attempts"]["listing"]} == set(selected)
    assert saved["phase_artifacts"]["compare"]["sha256"] == row.sha256(out / "phase.json")
