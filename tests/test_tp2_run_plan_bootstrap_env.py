"""Both TP2 ranks hand the resource bootstrap to their spawned vLLM workers.

WHY THIS TEST EXISTS.  The PACT U4 M3 TP2 resource capture on 2026-09-27
failed on both ranks with ``TypeError: 'NoneType' object is not
subscriptable`` at ``full_engine_worker.py``: ``full_engine_bootstrap._plan``
was ``None`` because no worker ran the bootstrap. vLLM spawns every worker as
a fresh interpreter, so a worker runs ``resource_bootstrap/sitecustomize.py``
only when its environment carries ``TESSERA_ENGINE_RESOURCE_PLAN`` and puts
``<tree>/experiments/resource_bootstrap`` on ``PYTHONPATH``.

* Head: ``step4_capture_driver`` launched the TP2 ``--run-plan`` process with
  the container environment unchanged, so neither variable was set.
* Peer: ``step4_tp2_peer`` derived the tree root from its own script path.
  The launcher runs it from a separate read-only mount of ``experiments/`` at
  ``/control``, so the root resolved to ``/`` and ``PYTHONPATH`` named
  directories that do not exist.

Each test drives the real entry point, then starts a fresh interpreter with
the environment it produced and checks that THIS tree's sitecustomize ran:
only that bootstrap exits 78 with "Fatal full-engine resource bootstrap" on a
plan that names no usable collector. No CUDA, vLLM or collector library is
needed.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "experiments" / "resource_bootstrap"
# The head test replaces subprocess.run with a fake; the child check needs the real one.
_REAL_RUN = subprocess.run


def _driver():
    spec = importlib.util.spec_from_file_location(
        "step4_capture_driver_bootstrap_env_under_test", ROOT / "experiments" / "step4_capture_driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(mode, output):
    return {
        "world_size": 2,
        "observation_mode": mode,
        "output_directory": str(output),
        "model": "/stub/model",
        "rank_runtime_evidence": {},
        "selected_configuration": {
            "engine_args": {"nnodes": 2, "node_rank": 0, "tensor_parallel_size": 2,
                            "distributed_executor_backend": "mp", "master_addr": "10.0.0.1"},
            "environment": {"PYTORCH_CUDA_ALLOC_CONF": "unset", "VLLM_HOST_IP": "10.0.0.1",
                            "TESSERA_SERVE_MODE": "resident"}},
        "observer_engine_args": {"worker_cls": "experiments.full_engine_worker.ResourceCaptureWorker"},
        "observer_environment": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
    }


@pytest.fixture
def restored_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _bootstrap_ran(env):
    """Start a fresh interpreter the way a spawned worker starts; did our bootstrap run?"""
    child = _REAL_RUN([sys.executable, "-c", "pass"], env=env, capture_output=True,
                      text=True, timeout=60)
    return child.returncode == 78 and "Fatal full-engine resource bootstrap" in child.stderr


def _assert_mode_environment(env, mode, plan):
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    if mode == "resources":
        assert Path(env["TESSERA_ENGINE_RESOURCE_PLAN"]) == plan.resolve()
        assert parts == [str(BOOTSTRAP), str(ROOT)]
        assert "TESSERA_ENGINE_TIMING_PLAN" not in env
        assert _bootstrap_ran(env)
    elif mode == "timings":
        assert Path(env["TESSERA_ENGINE_TIMING_PLAN"]) == plan.resolve()
        assert parts == [str(ROOT)]
        assert "TESSERA_ENGINE_RESOURCE_PLAN" not in env
    else:
        assert parts == [str(ROOT)]
        assert "TESSERA_ENGINE_RESOURCE_PLAN" not in env
        assert "TESSERA_ENGINE_TIMING_PLAN" not in env


@pytest.mark.parametrize("mode", ["resources", "timings", "kv"])
def test_tp2_head_run_plan_carries_the_bootstrap_environment(tmp_path, monkeypatch, mode):
    driver = _driver()
    import experiments.step4_cache_preflight as cache_preflight
    monkeypatch.setattr(cache_preflight, "check_worker_caches", lambda path: None)
    monkeypatch.setattr(driver, "native_preflight",
                        lambda out, serve_mode, expected: {"families": {}, "dense_launches": {}})
    # Stale plan variables from the container must not survive into the worker.
    monkeypatch.setenv("TESSERA_ENGINE_RESOURCE_PLAN", "/stale/resource-plan.json")
    monkeypatch.setenv("TESSERA_ENGINE_TIMING_PLAN", "/stale/timing-plan.json")
    monkeypatch.setenv("PYTHONPATH", "/tessera")
    out = tmp_path / "out"
    capture = out / "capture"
    (out / "rank-1").mkdir(parents=True)
    evidence = out / "rank-1" / "per-job-runtime.json"
    evidence.write_text('{"rank": 1}\n')
    (out / "head-session.json").write_text(json.dumps({"session_id": "fixture-session"}))
    plan = capture / "observer-plan.json"
    seen = {}

    def fake_run(command, **kwargs):
        if "--prepare-only" in command:
            capture.mkdir(parents=True, exist_ok=True)
            plan.write_text(json.dumps(_plan(mode, capture)))
            (out / "rank-1" / "peer-ready.json").write_text(json.dumps({
                "session_id": "fixture-session", "plan_sha256": driver.digest(plan),
                "runtime_evidence_sha256": driver.digest(evidence)}))
            return subprocess.CompletedProcess(command, 0)
        seen["command"] = command
        seen["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "step4_capture_driver.py", "--out", str(out), "--capture-output", str(capture),
        "--route-trace", str(out / "route-trace.json"),
        "--expected-modules", json.dumps({"TESSERA_FP8": {"count": 1}}),
        "--serve-mode", "resident", "--observation-mode", mode, "--tp2-head", "--peer-wait-s", "5",
        "--", "--world-size", "2"])
    assert driver.main() != 0  # the fake capture exits 1; the driver refuses at "capture"
    assert seen["command"][-2:] == ["--run-plan", str(plan.resolve())]
    env = seen["env"]
    assert env is not None, "TP2 --run-plan launched with the container environment unchanged"
    _assert_mode_environment(env, mode, plan)


def test_tp2_peer_root_is_the_imported_tree_not_the_control_mount(tmp_path, monkeypatch, restored_environ):
    # The launcher mounts experiments/ read-only at /control and runs the peer
    # from there. A COPY (never a symlink, which resolve() would follow back)
    # reproduces that: the script's own parents[1] is not the tree root.
    control = tmp_path / "mount" / "control"
    control.mkdir(parents=True)
    shutil.copy2(ROOT / "experiments" / "step4_tp2_peer.py", control / "step4_tp2_peer.py")
    out = tmp_path / "head-out"
    capture = out / "capture"
    capture.mkdir(parents=True)
    peer_out = tmp_path / "peer-out"
    peer_out.mkdir()
    (out / "head-session.json").write_text(json.dumps({"session_id": "fixture-session"}))
    evidence = peer_out / "per-job-runtime.json"
    evidence.write_text('{"rank": 1}\n')
    plan = capture / "observer-plan.json"
    document = _plan("resources", capture)
    document["rank_runtime_evidence"] = {"1": {"sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}}
    plan.write_text(json.dumps(document))

    class Reached(Exception):
        pass

    identity = types.ModuleType("experiments.full_engine_worker_identity")
    identity.actual_host_ip = lambda: {"ip": "10.0.0.2"}
    caches = types.ModuleType("experiments.step4_cache_preflight")
    caches.check_worker_caches = lambda path: None
    control_driver = types.ModuleType("step4_capture_driver")
    control_driver.native_preflight = lambda *args: {}
    control_driver.observer_preflight = lambda *args: {}
    vllm = types.ModuleType("vllm")

    def reached(name):
        raise Reached(name)

    vllm.__getattr__ = reached
    for name, module in (("experiments.full_engine_worker_identity", identity),
                         ("experiments.step4_cache_preflight", caches),
                         ("step4_capture_driver", control_driver), ("vllm", vllm)):
        monkeypatch.setitem(sys.modules, name, module)
    os.environ["PYTHONPATH"] = str(ROOT)
    monkeypatch.setattr(sys, "argv", [
        str(control / "step4_tp2_peer.py"), "--plan", str(plan), "--evidence", str(evidence),
        "--ready", str(peer_out / "peer-ready.json"), "--host-ip", "10.0.0.2",
        "--out", str(peer_out), "--serve-mode", "resident",
        "--expected-modules", json.dumps({"TESSERA_FP8": {"count": 1}}),
        "--collector", "/nonexistent/collector.so", "--wait-s", "5"])
    with pytest.raises(Reached):
        runpy.run_path(str(control / "step4_tp2_peer.py"), run_name="__main__")
    env = dict(os.environ)
    assert (peer_out / "peer-ready.json").is_file()
    assert env["VLLM_HOST_IP"] == "10.0.0.2"
    _assert_mode_environment(env, "resources", plan)
