"""Native suite authority refuses changed facts after a positive control."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def suite():
    return load("native_merge_suite", ROOT / "tools/merge_suite.py")


def publication(suite, tmp_path):
    from tessera._dev.surface_publication import publication_line, POPULATION
    path = tmp_path / "surface.x86.json"
    identity = {"verification": "verified", "snapshot_commit": "a" * 40,
                "sha256": "b" * 64}
    payload = {"schema": "tessera.test_surface.v3", "role": "population",
               "commit": "a" * 40, "source_identity": identity,
               "counts": {"passed": 3, "failed": 0, "skipped": 0}}
    path.write_text(json.dumps(payload))
    published = suite._read_publication(path)
    native = {"command": ["/usr/bin/python3", "-m", "pytest", "tests/test_a.py",
                          "--surface-json", str(path)],
              "source": identity, "returncode": 0, "exit_status_observed": True}
    stdout = publication_line(POPULATION, path, published.digest) + "\n3 passed in 0.10s\n"
    return published, native, stdout


def test_native_transport_has_no_pb_discovery(suite):
    assert callable(suite.suite_flux.collect)
    assert not hasattr(suite, "_pool_actions_that_wrote")
    assert not hasattr(suite, "use_pool_root")


@pytest.mark.parametrize("fact", ["path", "digest", "counts", "source", "commit", "command", "query"])
def test_native_publication_binding_refuses_one_changed_fact(suite, tmp_path, fact):
    published, native, stdout = publication(suite, tmp_path)
    assert suite._native_binding_refusal(native, published, stdout) is None
    native = copy.deepcopy(native)
    if fact == "path":
        native["command"][-1] += ".other"
    elif fact == "digest":
        stdout = stdout.replace(published.digest, "c" * 64)
    elif fact == "counts":
        stdout = stdout.replace("3 passed", "2 passed")
    elif fact in ("source", "commit"):
        native["source"]["sha256" if fact == "source" else "snapshot_commit"] = "c" * 64
    elif fact == "command":
        native["command"] = ["echo", *native["command"]]
    elif fact == "query":
        native.update(query_errors=["broker unreachable"], returncode=None, exit_status_observed=False)
    assert suite._native_binding_refusal(native, published, stdout)


def test_binding_uses_the_single_publication_read(suite, tmp_path):
    published, native, stdout = publication(suite, tmp_path)
    published.path.write_text('{"counts":{"passed":100}}')
    assert suite._native_binding_refusal(native, published, stdout) is None
    replacement = suite._read_publication(published.path)
    assert suite._native_binding_refusal(native, replacement, stdout)


@pytest.mark.parametrize("cores,memory,timeout", [(0, 1, 1), (5, 1, 1), (1, 17, 1),
                                                   (1, 1, 1791), (1, 1, float("nan"))])
def test_finite_resource_profile(suite, cores, memory, timeout):
    suite.suite_flux.validate(4, 16, 1790, {})
    with pytest.raises(ValueError, match="profile"):
        suite.suite_flux.validate(cores, memory, timeout, {})


def test_gpu_preflight_cannot_start_a_partial_launch(suite, tmp_path):
    args = SimpleNamespace(cpus=2, mem_gb=4, timeout_s=60, wait_s=90, artifact_root=[], gpu_image="")
    assert suite._preflight(suite.ARMS["x86"], args) == {}
    with pytest.raises(ValueError, match="GPU/container"):
        suite._preflight(suite.ARMS["gpu"], args)
    assert list(tmp_path.iterdir()) == []


def test_artifact_environment_cannot_replace_source_verifier(suite, tmp_path):
    with pytest.raises(ValueError, match="artifact root"):
        suite.suite_flux.validate(1, 1, 60, {"TESSERA_SOURCE_VERIFIER": str(tmp_path)})


def test_unknown_saved_submission_never_submits(suite, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite.suite_flux, "native", lambda argv, **kwargs: calls.append(argv))
    result = suite.suite_flux.collect(tmp_path)
    assert result["state"] == "unknown"
    assert result["returncode"] is None
    assert calls == []


def test_late_nonzero_native_exit_stays_red(suite, tmp_path, monkeypatch):
    published, native, stdout = publication(suite, tmp_path)
    native.update(returncode=7, job_id="ƒ123")
    directory = tmp_path / "flux.x86"
    directory.mkdir()
    (directory / "stdout").write_text(stdout)
    monkeypatch.setattr(suite.suite_flux, "collect", lambda directory: native)
    record = suite._resume("x86", suite.ARMS["x86"], tmp_path)
    assert record["returncode"] == 7
    assert suite._verdict([record]).startswith("red")
