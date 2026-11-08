"""Native container authority, starting with independently known commands."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import _suite_container as container
import merge_suite as merge


def composed(tmp_path, targets=None, cpus=2):
    arm = {**merge.ARMS["gpu"], "python": "/usr/bin/python3", "container": True,
           "fans_out": True, "targets": targets or ["tests/test_grammar.py"]}
    surface = tmp_path / "output/surface.gpu.json"
    inner = merge._command(arm, surface, [], cpus)
    inner += ["--junitxml", str(surface.parent / "junit.gpu.xml"), "--durations=10"]
    return ["/usr/bin/python3", container.RUNNER,
            "--image", "sha256:" + "a" * 64,
            "--deps-site", str(tmp_path / "dependencies"), "--deps-sha256", "b" * 64,
            "--surface-dir", str(surface.parent), "--cache-dir", str(tmp_path / "cache"),
            "--", *inner]


def test_actual_generated_command_passes_finite_parser(tmp_path):
    command = composed(tmp_path)
    spec = container.command_spec(command)
    assert spec["workers"] == 2
    assert spec["surface"] == str(tmp_path / "output/surface.gpu.json")
    assert spec["junit"] == str(tmp_path / "output/junit.gpu.xml")
    assert merge._effective_surface_json(merge._timed_command(command, 60)) == (spec["surface"], None)


@pytest.mark.parametrize("shadow", [
    ["-o", "pythonpath=/foreign"], ["-o", "addopts="], ["-c", "/foreign"],
    ["--junitxml", "/foreign.xml"], ["--surface-json", "/foreign.json"],
    ["-n", "1"], ["-p", "xdist.plugin"], ["-p", "foreign.plugin"],
    ["tests/test_other.py"], ["tests/test_grammar.py::test_x"],
])
def test_composed_controls_cannot_be_shadowed(tmp_path, shadow):
    command = composed(tmp_path)
    assert container.command_spec(command)["workers"] == 2
    with pytest.raises(ValueError):
        container.command_spec(command + shadow)


@pytest.mark.parametrize("targets", [
    ["tests/a.py", "tests/a.py"], ["tests/./a.py"],
    ["tests/a", "tests/a-else.py", "tests/a/b.py"],
    ["tests/a.py::test_a", "tests/a.py::test_a-other", "tests/a.py::test_a[param]"],
])
def test_parser_refuses_target_aliases_and_nonadjacent_overlap(tmp_path, targets):
    command = composed(tmp_path)
    assert container.command_spec(command)["workers"] == 2
    start = command.index("pytest") + 1
    command[start:start + 1] = targets
    with pytest.raises(ValueError, match="target"):
        container.command_spec(command)


@pytest.fixture
def native_record(tmp_path):
    """Known hand-authored execution, independent of producer functions."""
    import _suite_flux as flux
    directory = tmp_path / "native"
    directory.mkdir()
    source = {"verification": "verified", "snapshot_commit": "c" * 40, "sha256": "d" * 64}
    intent = {"container": {"--image": "sha256:" + "a" * 64, "--deps-sha256": "b" * 64},
              "source": source, "resources": {"architecture": "x86_64", "gpus": 0}}
    entry = {"schema": "tessera.suite_container_runtime.v1", "job_id": "ƒ123",
             "launch_sha256": "e" * 64, "source": source, "dependencies": "b" * 64,
             "affinity": [2, 42], "driver": None,
             "runtime": {"schema": "tessera.suite_runtime.v1", "sha256": "f" * 64}}
    record = {"schema": "tessera.suite_container_execution.v1", "job_id": "f123",
              "launch_sha256": "e" * 64, "source": source,
              "image": {"Id": "sha256:" + "a" * 64, "Architecture": "amd64", "Os": "linux"},
              "gpus": 0, "affinity": [2, 42], "dependencies": "b" * 64,
              "removed": True, "cleanup_errors": [], "entry": entry, "exit": copy.deepcopy(entry),
              "driver": None, "cid": "1" * 64,
              "inspection": {"Id": "1" * 64, "Image": "sha256:" + "a" * 64, "CpusetCpus": "2,42"}}
    def publish():
        flux.save(directory / "container.json", record)
        digest = flux.hashlib.sha256((directory / "container.json").read_bytes()).hexdigest()
        (directory / "stdout").write_text(f"Tessera container evidence {directory}/container.json sha256={digest}\n")
    publish()
    assert container.verify_record(directory, intent, "e" * 64, "ƒ123", [2, 42]) == record
    return directory, intent, record, publish


@pytest.mark.parametrize("fact", ["source", "image", "image-arch", "deps", "job", "launch", "cpus", "gpu", "cid",
                                   "runtime", "inside-source", "inside-deps", "inside-cpus", "driver", "cleanup", "removed",
                                   "missing", "replaced", "symlink"])
def test_native_verifier_accepts_positive_before_mutation(native_record, fact):
    directory, intent, record, publish = native_record
    if fact in ("source", "image", "deps", "job", "launch", "cpus", "gpu", "cid", "image-arch"):
        if fact == "source": record["source"] = {"sha256": "0" * 64}
        elif fact == "image": record["image"]["Id"] = "sha256:" + "0" * 64
        elif fact == "image-arch": record["image"]["Architecture"] = "arm64"
        elif fact == "deps": record["dependencies"] = "0" * 64
        elif fact == "job": record["job_id"] = "f124"
        elif fact == "launch": record["launch_sha256"] = "0" * 64
        elif fact == "cpus": record["affinity"] = [3, 43]
        elif fact == "gpu": record["gpus"] = 1
        elif fact == "cid": record["cid"] = "2" * 64
    elif fact == "runtime": record["exit"]["runtime"]["sha256"] = "0" * 64
    elif fact.startswith("inside-") or fact == "driver":
        key = {"inside-source": "source", "inside-deps": "dependencies", "inside-cpus": "affinity", "driver": "driver"}[fact]
        record["entry"][key] = ["unexpected driver"] if fact == "driver" else None
        record["exit"][key] = ["unexpected driver"] if fact == "driver" else None
    elif fact == "cleanup": record["cleanup_errors"] = ["failed to delete owned container"]
    elif fact == "removed": record["removed"] = False
    publish()
    path = directory / "container.json"
    if fact == "missing": path.unlink()
    elif fact == "replaced": path.write_text(path.read_text() + " ")
    elif fact == "symlink":
        target = directory / "replacement.json"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        container.verify_record(directory, intent, "e" * 64, "f123", [2, 42])


def test_native_source_authentication_and_mutation_replace_bundle_fixture(tmp_path, monkeypatch):
    import subprocess
    import _suite_flux as flux
    from tessera._dev.suite_source import measured_source
    checkout = tmp_path / "source"
    checkout.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], stderr=subprocess.DEVNULL)
    git("init", "--quiet")
    (checkout / "runner.py").write_text("print('original authenticated source')\n")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@local", "commit", "--quiet", "-m", "original")
    git("checkout", "--detach", "HEAD")
    identity = measured_source(checkout, verifier=None)
    assert identity["verification"] == "verified"
    directory = tmp_path / "record"
    directory.mkdir()
    intent = {"directory": str(directory), "checkout": str(checkout), "source": identity}
    flux.save(directory / "launch.json", intent)
    digest = flux.hashlib.sha256((directory / "launch.json").read_bytes()).hexdigest()
    flux.save(directory / "submission.json", {"state": "submitted", "job_id": "f123", "launch_sha256": digest})
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(flux, "source", lambda root: measured_source(root, verifier=None))
    assert container.admitted_context(directory, digest, {"FLUX_JOB_ID": "ƒ123"}) == intent
    (checkout / "runner.py").write_text("print('forged source')\n")
    with pytest.raises(ValueError, match="source authentication"):
        container.admitted_context(directory, digest, {"FLUX_JOB_ID": "f123"})
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@local", "commit", "--quiet", "-m", "replacement")
    with pytest.raises(ValueError, match="source authentication"):
        container.admitted_context(directory, digest, {"FLUX_JOB_ID": "f123"})
    with pytest.raises(ValueError, match="job ID"):
        container.admitted_context(directory, digest, {"FLUX_JOB_ID": "f124"})


@pytest.mark.parametrize("gpus,arch", [(2, "aarch64"), (0.5, "aarch64"), (True, "aarch64"), (1, "x86_64"), (0, "arm")])
def test_finite_resource_architecture_and_whole_gpu_intent(gpus, arch):
    import _suite_flux as flux
    assert flux.resources(2, 16, 600, gpus=1, architecture="aarch64") == {
        "schema": "tessera.suite_resources.v1", "cores": 2, "memory_gib": 16,
        "timeout_s": 600, "gpus": 1, "architecture": "aarch64"}
    with pytest.raises(ValueError):
        flux.resources(2, 16, 600, gpus=gpus, architecture=arch)


def test_cleanup_failure_is_retained_even_when_remove_fails(tmp_path, monkeypatch):
    cid = "1" * 64
    path = tmp_path / "container.cid"
    path.write_text(cid)
    record = {"cidfile": str(path), "cid": cid, "cleanup_errors": [], "removed": False}
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=125, stderr="owned cleanup denied")
    monkeypatch.setattr(container.subprocess, "run", run)
    container.cleanup_cid(record)
    assert len(record["cleanup_errors"]) == 3
    assert all(argv[-1] == cid for argv in calls)
    assert record["removed"] is False


def test_original_native_wrapper_authenticates_launch_bytes():
    import _suite_flux as flux
    intent = {"directory": "/owned/native", "command": ["/named/python", "payload"],
              "resources": {"cores": 2}}
    original = {"tasks": [{"command": ["/named/python", "tools/_suite_flux.py", "payload", "/owned/native", "a" * 64]}]}
    flux.require_original_launch(intent, "a" * 64, original)
    with pytest.raises(ValueError, match="launch digest"):
        flux.require_original_launch(intent, "b" * 64, original)
    with pytest.raises(ValueError, match="launch digest"):
        flux.require_original_launch({**intent, "command": ["/other/python"]}, "a" * 64, original)


@pytest.mark.parametrize("fact", ["cores", "memory_gib", "gpus", "architecture", "affinity", "performance"])
def test_actual_allocation_refuses_different_resource_ownership(fact):
    import _suite_flux as flux
    demand = {"architecture": "x86_64", "cores": 1, "memory_gib": 1, "gpus": 0}
    core = {"type": "core", "properties": {"os-pus": "2,42", "core-type": "perf"}}
    allocation = {"scheduling": {"graph": {"nodes": [{"metadata": item} for item in [
        core, {"type": "perf"}, {"type": "memory"}, {"type": "node", "properties": {"arch": "x86_64"}}]]}}}
    flux.require_allocation(allocation, demand, [2, 42], [2, 42])
    if fact in ("cores", "memory_gib", "gpus"):
        demand[fact] += 1
    elif fact == "architecture": demand[fact] = "aarch64"
    elif fact == "performance": core["properties"]["core-type"] = "efficiency"
    with pytest.raises(ValueError, match="allocation"):
        flux.require_allocation(allocation, demand, [3, 43] if fact == "affinity" else [2, 42], [2, 42])
