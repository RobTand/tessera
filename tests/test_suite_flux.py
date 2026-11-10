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
    identity = {"schema": "tessera.suite_source.v1", "verification": "verified", "snapshot_commit": "a" * 40,
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


def test_submit_environment_and_runtime_budget_are_explicit(suite, tmp_path, monkeypatch):
    flux = suite.suite_flux
    source = {"snapshot_commit": "a" * 40, "sha256": "b" * 64}
    monkeypatch.setattr(flux, "source", lambda path: source)
    monkeypatch.setattr(flux, "instance_identity", lambda **kw: {"broker_starttime": "A"})
    monkeypatch.setattr(flux, "require_capacity", lambda *a, **kw: None)
    monkeypatch.setenv("FLUX_URI", "local:///test-instance")
    monkeypatch.setenv("FLUX_CLIENT", "/usr/bin/python3")
    monkeypatch.setattr(flux, "instance_settings", lambda: tmp_path)
    budgets, calls = [], []
    def runtime(python, checkout, *, timeout=None):
        budgets.append(timeout)
        return {"schema": "tessera.suite_runtime.v1", "sha256": "c" * 64}
    monkeypatch.setattr(flux, "runtime_identity", runtime)
    def native(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["submit", "--env=-*", "--dry-run"]:
            return SimpleNamespace(returncode=0, stdout='{"attributes":{"system":{"environment":{}}}}', stderr="")
        assert argv[:2] == ["job", "submit"]
        return SimpleNamespace(returncode=0, stdout="ƒ123\n", stderr="")
    monkeypatch.setattr(flux, "native", native)
    monkeypatch.setattr(flux, "collect", lambda directory: {"state": "running"})
    flux.submit(tmp_path / "new", tmp_path, ["/named/python", "-s", "-m", "pytest"], 2, 8, 900,
                {"KL_TOOL_DIR": str(tmp_path)})
    assert budgets == [900], "full native seal must share the declared finite launch budget"
    render = calls[0]
    assert render[:3] == ["submit", "--env=-*", "--dry-run"]
    assert "--resources-shape=slot/[core=2;memory=8]" in render
    assert "--requires=x86_64" in render
    environment = dict(token.removeprefix("--env=").split("=", 1) for token in render[3:] if token.startswith("--env="))
    assert set(environment) == {"PATH", "HOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "PRISMAQUANT_DEV_MODE",
                                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL",
                                "MAKEFLAGS", "NINJAFLAGS", "KL_TOOL_DIR"}
    assert environment["PYTHONPATH"] == "src"
    assert all(environment[name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL"))
    assert environment["MAKEFLAGS"] == environment["NINJAFLAGS"] == "-j1"


def test_resource_shape_is_the_stock_slot_with_no_node_wrapper_or_core_class(suite):
    flux = suite.suite_flux
    assert flux.resource_shape(4, 16) == "slot/[core=4;memory=16]"
    assert flux.resource_shape(2, 8, 1) == "slot/[core=2;memory=8;gpu=1]"
    for shape in (flux.resource_shape(1, 1), flux.resource_shape(1, 1, 1)):
        assert not shape.startswith("node/") and "perf" not in shape



def test_image_host_is_a_finite_architecture_bound_dependency(suite):
    flux = suite.suite_flux
    demand = flux.resources(2, 16, 600, gpus=1, architecture="aarch64", host="sparky")
    assert demand == {"schema": "tessera.suite_resources.v1", "cores": 2, "memory_gib": 16,
                      "timeout_s": 600, "gpus": 1, "architecture": "aarch64", "host": "sparky"}
    for host in ("bad host", "host:injection", ""):
        with pytest.raises(ValueError, match="host"):
            flux.resources(2, 16, 600, host=host)


def test_portable_noncontainer_cannot_pin_a_host(suite, tmp_path):
    with pytest.raises(ValueError, match="local container image dependency"):
        suite.suite_flux.submit(tmp_path / "unused", tmp_path, ["python", "-m", "pytest"],
                                1, 1, 60, {}, host="dl380g10")
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize('targets', [
    ['tests/a', 'tests/a-b.py', 'tests/a/test_x.py'],
    ['tests/test_a.py', 'tests/test_a.py.bak', 'tests/test_a.py::test_x'],
])
def test_nonadjacent_target_overlap_refuses(suite, targets):
    """Pre-fix: unit3-parent-overlap-pre retained two DID NOT RAISE failures."""
    arm = {**suite.ARMS['x86'], 'targets': ['tests/a-b.py', 'tests/a/test_x.py']}
    assert suite._command(arm, Path('/out/surface.json'), [], 1)
    with pytest.raises(ValueError, match='overlap'):
        suite._command({**arm, 'targets': targets}, Path('/out/surface.json'), [], 1)


def test_gpu_jobspec_is_rendered_in_the_stock_shape_and_never_run():
    spec = load("flux_gpu_jobspec", ROOT / "tools/_flux_gpu_jobspec.py")
    argv = spec.render(["bash", "x.sh", "a b"], cores=8, memory_gib=16, walltime="45m", env=["PROFILE=0"], directory="/mnt/shared/render-fixture")
    assert argv[:3] == ["flux", "submit", "--env=-*"]
    assert "--env=PATH=/usr/bin:/bin" in argv and "--env=PROFILE=0" in argv
    assert "--resources-shape=slot/[core=8;memory=16;gpu=1]" in argv
    assert not any(token.startswith("--requires=") for token in argv)
    assert "--input=/dev/null" in argv
    assert "--output=/mnt/shared/render-fixture/stdout" in argv
    assert "--error=/mnt/shared/render-fixture/stderr" in argv
    assert "--cwd" in argv
    assert argv[argv.index("-t") + 1] == "45m"
    assert argv[argv.index("--") + 1:] == ["bash", "x.sh", "a b"]
    assert not any("perf" in part or "core-type" in part for part in argv)
    for bad in ({"command": []}, {"walltime": ""}, {"cores": 0}, {"memory_gib": 0}, {"env": ["NOVALUE"]}):
        kwargs = dict(command=["x"], cores=1, memory_gib=1, walltime="1m", env=[], directory="/mnt/shared/render-fixture", **{})
        kwargs.update(bad)
        with pytest.raises(ValueError):
            spec.render(kwargs.pop("command"), **kwargs)
    source = (ROOT / "tools/_flux_gpu_jobspec.py").read_text()
    assert "subprocess" not in source and "os.exec" not in source and "os.system" not in source
