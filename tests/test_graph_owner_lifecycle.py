"""Additional CPU regressions through the real graph observer's import hooks.

Stand-ins supply runtime calls, not replacement accounting. These tests make no
CUDA execution or device-completion claim.
"""
import json
import subprocess
import sys

from child_env import child_env
from test_graph_owner_hooks import HOOK, MANAGER, RUNNER


PROBE = r'''
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

hook, scratch, scenario = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(scratch))
spec = importlib.util.spec_from_file_location("owner_lifecycle_probe", hook)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
# Keep the actual daemon for the fork regression: publication is the contract.
if scenario != "fork":
    observer._ga_flusher = lambda: None
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager, Descriptor
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
manager, runner = ModelCudaGraphManager(), GPUModelRunner()
owner = f"ModelCudaGraphManager@{id(manager):x}"
desc = Descriptor(1)

def step():
    return runner.execute_model(manager, desc)

def read(pid=None):
    return json.loads(Path(f"{os.environ['GA_DISPATCH_LOG']}.{pid or os.getpid()}.json").read_text())

def replay_count(record):
    return sum(d['count'] for d in record.get('owner_evidence', {}).get(owner, {}).get('replays', []))

def await_replays(count):
    deadline = time.monotonic() + 4.0
    record = None
    while time.monotonic() < deadline:
        try:
            record = read()
        except FileNotFoundError:
            pass
        if record is not None and replay_count(record) == count:
            break
        time.sleep(0.025)
    return record

if scenario == "missing-step":
    from tessera.serving import glm53_graphs as gg
    import vllm.v1.worker.gpu.cudagraph_utils as utils
    gg._STATE = NS(installed=True, bounds=(192, 640, 8448), capture_bound=None,
                   bound_applied=False, step_max_seq_len=None)
    gg._patch_managers(utils, (ModelCudaGraphManager,))
    manager.capture(NS(modules=lambda: []))
    gg._STATE.step_max_seq_len = 500
    assert step() is manager.result
    gg._STATE.step_max_seq_len = None
    try:
        step()
    except RuntimeError as exc:
        assert str(exc) == ("tessera.glm53_graphs: a FULL replay with no step max_seq_len "
                            "recorded; the class it needs is unknown")
        print("ERROR=" + json.dumps(str(exc)))
    else:
        raise AssertionError("missing step bound did not raise the original RuntimeError")
else:
    assert manager.capture() is manager.result
    if scenario == "nested-dummy":
        assert step() is manager.result
    elif scenario == "fork":
        assert step() is manager.result
        assert step() is manager.result
        parent = await_replays(2)
        assert parent is not None and replay_count(parent) == 2, parent
        print("PARENT=" + json.dumps(parent), flush=True)
        child = os.fork()
        if child == 0:
            # No capture, module reimport or explicit flush: the inherited hooks
            # must reset history AND arrange publication after this return.
            assert step() is manager.result
            record = await_replays(1)
            print("CHILD=" + json.dumps(record), flush=True)
            os._exit(0)
        assert os.waitpid(child, 0)[1] == 0
        print("OWNER=" + owner)
        sys.exit(0)
    elif scenario != "pruned-capture":
        raise AssertionError(scenario)
observer._ga_flush(force=True)
print("OWNER=" + owner)
print("RECORD=" + json.dumps(read()))
'''


def _probe(tmp_path, scenario):
    package = tmp_path / "vllm/v1/worker/gpu"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent,
                      package.parent.parent.parent):
        (directory / "__init__.py").write_text("")
    manager_source, runner_source = MANAGER, RUNNER
    if scenario == "pruned-capture":
        manager_source = MANAGER.replace(
            "        return super().capture(*args)",
            "        out = super().capture(*args)\n"
            "        del self.graphs[Descriptor(2)]\n"
            "        return out")
    if scenario == "nested-dummy":
        runner_source = RUNNER.replace(
            "        selected = manager.dispatch(desc)",
            "        if not dummy_run:\n"
            "            assert self.execute_model(manager, type(desc)(2), dummy_run=True) is manager.result\n"
            "        selected = manager.dispatch(desc)")
    (package / "cudagraph_utils.py").write_text(manager_source)
    (package / "model_runner.py").write_text(runner_source)
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(HOOK), str(tmp_path), scenario],
        env=child_env(GA_DISPATCH_LOG=str(tmp_path / "dispatch"), T508_DIGEST="",
                      T508_CAPTURE_LOG="", T508_PROF_DIR="", T508_MOE_DETERMINISTIC="",
                      T695_GC_BEFORE_DRAFTER="", T695_DRAFT_LOG=""),
        capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr
    result = {}
    for line in done.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in ("OWNER", "RECORD", "PARENT", "CHILD", "ERROR"):
            result[key] = value if key == "OWNER" else json.loads(value)
    return result


def _owner(result, key="RECORD"):
    return result[key]["owner_evidence"][result["OWNER"]]


def _assert_inventory(owner, tokens=(1, 2), bounds=(None,)):
    assert owner["capture_complete"] is True
    assert owner["capture_error"] is None
    assert {(d["mode"], d["tokens"], d["num_reqs"], d["max_seq_len"])
            for d in owner["captured"]} == {
                ("FULL", token, 1, bound) for token in tokens for bound in bounds}


def test_nested_dummy_does_not_inherit_real_serving_credit(tmp_path):
    result = _probe(tmp_path, "nested-dummy")
    owner = _owner(result)
    _assert_inventory(owner)
    # Check both diagnostics together so neither can hide the other regression.
    assert (sum(d["count"] for d in owner["replays"]),
            sum(result["RECORD"]["counts"].values())) == (1, 1)
    assert {d["tokens"] for d in owner["replays"]} == {1}
    assert not owner["replay_failures"]


def test_model_override_publishes_only_final_capture_inventory(tmp_path):
    result = _probe(tmp_path, "pruned-capture")
    _assert_inventory(_owner(result), tokens=(1,))
    assert result["RECORD"]["captured"][result["OWNER"]] == [1]


def test_fork_child_automatically_publishes_fresh_successful_replay(tmp_path):
    result = _probe(tmp_path, "fork")
    parent = _owner(result, "PARENT")
    _assert_inventory(parent)
    assert sum(d["count"] for d in parent["replays"]) == 2
    assert sum(result["PARENT"]["counts"].values()) == 2
    child = result["CHILD"]
    assert child is not None, "child never automatically published within four seconds"
    assert child["pid"] != result["PARENT"]["pid"]
    evidence = _owner(result, "CHILD")
    assert sum(d["count"] for d in evidence["replays"]) == 1, child
    assert sum(child["counts"].values()) == 1
    assert not evidence["capture_complete"] and not evidence["captured"]
    assert not evidence["replay_failures"]


def test_missing_step_bound_preserves_error_and_records_unselected_failure(tmp_path):
    result = _probe(tmp_path, "missing-step")
    owner = _owner(result)
    _assert_inventory(owner, bounds=(192, 640, 8448))
    assert [(d["tokens"], d["max_seq_len"], d["count"])
            for d in owner["replays"]] == [(1, 640, 1)]
    failures = owner["replay_failures"]
    assert len(failures) == 1, failures
    failure = failures[0]
    assert (failure["mode"], failure["tokens"], failure["num_reqs"],
            failure["max_seq_len"], failure["count"]) == ("FULL", 1, 1, None, 1)
    assert result["ERROR"] in failure["error"]
