"""Regression oracles for the independent native suite review."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def suite():
    spec = importlib.util.spec_from_file_location("review_merge_suite", ROOT / "tools/merge_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARMS["x86"]["python"] = "/opt/test-cpu/bin/python"
    return module


@pytest.mark.parametrize("extra", [["-n", "64"], ["-n64"], ["--numprocesses=auto"],
                                  ["-o", "addopts=-n64"], ["--override-ini=addopts=-n64"],
                                  ["-c", "other.ini"], ["tests/extra.py"],
                                  ["-k", "wanted", "tests/extra.py"]])
def test_resource_and_target_controls_are_owned(suite, extra):
    assert suite._command(suite.ARMS["x86"], Path("/out/population.json"), ["-k", "wanted"], 2)
    with pytest.raises(ValueError):
        suite._command(suite.ARMS["x86"], Path("/out/population.json"), extra, 2)


@pytest.mark.parametrize("targets", [
    ["tests/test_a.py", "tests/./test_a.py"],
    ["tests/test_a.py", "tests//test_a.py"],
    ["tests", "tests/test_a.py"],
    ["tests/test_a.py", "tests/test_a.py::test_one"],
    ["tests/test_a.py::test_one", "tests/test_a.py::test_one[value]"],
])
def test_repeated_effective_targets_refuse(suite, targets):
    arm = {**suite.ARMS["x86"], "targets": targets}
    with pytest.raises(ValueError, match="targets"):
        suite._command(arm, Path("/out/population.json"), [], 1)


def test_distinct_nodes_with_shared_name_prefix_remain_selectable(suite):
    targets = ["tests/test_a.py::test_one", "tests/test_a.py::test_one_more"]
    arm = {**suite.ARMS["x86"], "targets": targets}
    assert suite._targets_of(suite._command(arm, Path("/out/population.json"), [], 1)) == targets


@pytest.mark.parametrize("failed", [{"surface": None, "returncode": 9},
                                    {"surface": {"counts": {"failed": 1}}, "returncode": None}])
def test_conclusive_failure_precedes_incomplete_sibling(suite, failed):
    record = {"arm": "x86", **failed}
    assert suite._verdict([record, {"arm": "gpu", "surface": None}]).startswith("red")
    assert suite._arm_results([record])["x86"].startswith("red")


@pytest.fixture
def native_world(suite, tmp_path, monkeypatch):
    flux = suite.suite_flux
    launch = {"schema": "tessera.flux.launch.v1", "directory": str(tmp_path),
              "uri": "local:///historical-instance", "broker_starttime": "generation-A",
              "command": ["python", "-m", "pytest"], "source": {"sha256": "a" * 64},
              "checkout": str(tmp_path / "source"), "runtime": {"sha256": "c" * 64},
              "application": None, "qualification": {"profile": "fixture-A", "broker_starttime": "generation-A"}}
    flux.save(tmp_path / "launch.json", launch)
    digest = flux.hashlib.sha256(flux.encoded(launch)).hexdigest()
    flux.save(tmp_path / "submission.json", {"state": "submitted", "job_id": "ƒ123",
                                            "launch_sha256": digest})
    spec = {"tasks": [{"command": ["/usr/bin/python3", "-s", "tools/_suite_flux.py", "payload", str(tmp_path), digest]}]}
    flux.save(tmp_path / "jobspec.json", spec)
    flux.save(tmp_path / "payload.json", {"job_id": "ƒ123", "launch_sha256": digest,
                                         "source": launch["source"], "scratch_removed": True,
                                         "runtime": launch["runtime"], "runtime_verified": True})
    state = {"exit": 0, "exception": False, "result": 1}
    calls = []

    def native(argv, timeout=30):
        calls.append(argv)
        code = 0
        if argv == ["getattr", "broker.starttime"]:
            raw = "generation-A"
        elif argv[:3] == ["job", "info", "--original"]:
            raw = json.dumps(spec)
        elif argv[:2] == ["job", "eventlog"]:
            raw = "\n".join(json.dumps(event) for event in [
                {"name": "alloc"}, {"name": "start"},
                {"name": "finish", "context": {"status": state["exit"] * 256}},
                *([{ "name": "exception", "context": {"severity": 0, "type": "exec"}}]
                  if state["exception"] else []), {"name": "clean"}])
        elif argv[:2] == ["job", "status"]:
            code = 125 if state["exception"] else state["exit"]
            raw = json.dumps({"id": 1, "result": state["result"],
                              "exception_occurred": state["exception"],
                              "waitstatus": state["exit"] * 256})
        elif argv[:2] == ["job", "info"] and argv[-1] == "R":
            raw = '{"version":1,"execution":{"R_lite":[{"rank":"0"}]}}'
        else:
            raise AssertionError(f"unexpected native operation {argv}")
        return SimpleNamespace(returncode=code, stdout=raw, stderr="")

    monkeypatch.setattr(flux, "broker_starttime", lambda **kw: "generation-A")
    monkeypatch.setattr(flux, "native", native)
    assert flux.collect(tmp_path)["returncode"] == 0
    return SimpleNamespace(flux=flux, directory=tmp_path, state=state, calls=calls)


@pytest.mark.parametrize("cleanup", ["missing", "failed"])
def test_cleanup_gap_cannot_erase_confirmed_failure(native_world, cleanup):
    world = native_world
    world.state.update(exit=7, result=2)
    path = world.directory / "payload.json"
    if cleanup == "missing":
        path.unlink()
    else:
        payload = json.loads(path.read_bytes())
        payload["scratch_removed"] = False
        world.flux.save(path, payload)
    record = world.flux.collect(world.directory)
    assert record["returncode"] == 7
    assert record["exit_status_observed"] is True
    assert record.get("evidence_errors")


def test_late_fatal_exception_cannot_read_as_success(native_world):
    native_world.state["exception"] = True
    record = native_world.flux.collect(native_world.directory)
    assert record["returncode"] == 125
    assert record["exit_status_observed"] is True


def test_acknowledged_resume_never_submits(native_world):
    native_world.calls.clear()
    assert native_world.flux.collect(native_world.directory)["returncode"] == 0
    assert not any("submit" in call for call in native_world.calls)


def test_ascii_native_ack_and_unicode_payload_name_one_saved_job(native_world):
    world = native_world
    sent_path = world.directory / "submission.json"
    sent = json.loads(sent_path.read_bytes())
    sent["job_id"] = "f123"
    world.flux.save(sent_path, sent)
    record = world.flux.collect(world.directory)
    assert record["returncode"] == 0 and record["exit_status_observed"], record
    assert record["job_id"] == "f123"
    assert not any("submit" in call for call in world.calls)
