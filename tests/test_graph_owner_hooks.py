"""CPU hook-to-wire acceptance for #1259; no CUDA/device-completion claim.

Import the real observer and let its real import finder patch stand-in vLLM
modules. The proposed additive owner_evidence wire is shared with
 test_graph_replay_coverage.py. Existing counts remain dispatch diagnostics.
"""
from pathlib import Path
import json
import subprocess
import sys

import pytest

from child_env import child_env

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "experiments/glm53_508_graph_qual/digest/usercustomize.py"

MANAGER = '''
from dataclasses import dataclass
from types import SimpleNamespace as NS
@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int = 1
    @property
    def cg_mode(self):
        return NS(name="FULL")
class CudaGraphManager:
    def __init__(self):
        self.graphs = {}
        self._graphs_captured = False
        self.fail = False
        self.incomplete = False
        self.replay_fail = False
        self.result = object()
    def capture(self, *args):
        import sys
        gg = sys.modules.get("tessera.serving.glm53_graphs")
        if gg is not None and gg._STATE.capture_bound is not None:
            gg._record_build(lambda **kw: None)(max_seq_len=8448, for_cudagraph_capture=True)
        self.graphs = {Descriptor(1): object(), Descriptor(2): object()}
        if self.fail:
            raise ValueError("capture-sentinel")
        self._graphs_captured = not self.incomplete
        return self.result
    def captured_token_counts(self):
        return [d.num_tokens for d in self.graphs]
    def dispatch(self, desc):
        return desc
    def run_fullgraph(self, desc):
        if self.replay_fail:
            raise LookupError("replay-sentinel")
        return self.result
    def release_graphs(self):
        self.graphs = {}
        self._graphs_captured = False
class ModelCudaGraphManager(CudaGraphManager):
    def capture(self, *args):
        return super().capture(*args)
def _extrapolate_full_graph_memory(samples, total):
    return total
'''
RUNNER = '''
class GPUModelRunner:
    def execute_model(self, manager, desc, *, replay=True, dummy_run=False):
        selected = manager.dispatch(desc)
        return manager.run_fullgraph(selected) if replay else selected
    def sample_tokens(self, manager, desc):
        return manager.run_fullgraph(manager.dispatch(desc))
'''
PROBE = r'''
import importlib.util
import json
import os
from pathlib import Path
import sys

hook, scratch, scenario = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(scratch))
spec = importlib.util.spec_from_file_location("owner_hook_probe", hook)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
# Deterministic flushing, not replacement accounting. No background I/O/fork race.
observer._ga_flusher = lambda: None
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager, Descriptor
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
manager, runner, desc = ModelCudaGraphManager(), GPUModelRunner(), Descriptor(1)
owner = f"ModelCudaGraphManager@{id(manager):x}"
if scenario == "branch":
    from types import SimpleNamespace as NS
    from tessera.serving import glm53_graphs as gg
    import vllm.v1.worker.gpu.cudagraph_utils as utils
    gg._STATE = NS(installed=True, bounds=(192, 640, 8448), capture_bound=None,
                   bound_applied=False, step_max_seq_len=None)
    gg._patch_managers(utils, (ModelCudaGraphManager,))
    # Invoke the real serving capture wrappers around the already instrumented
    # manager. The stand-in builder marks each branch's actual bound applied.
    manager.capture(NS(modules=lambda: []))
else:
    assert manager.capture() is manager.result

def step(m=manager, replay=True):
    return runner.execute_model(m, desc, replay=replay)

def read():
    observer._ga_flush(force=True)
    return json.loads(Path(f"{os.environ['GA_DISPATCH_LOG']}.{os.getpid()}.json").read_text())

if scenario in ("success", "complete", "failed-replay", "dispatch-only", "same-class", "fork", "eager-dummy"):
    if scenario != "dispatch-only":
        assert step() is manager.result
    if scenario == "complete":
        assert runner.execute_model(manager, Descriptor(2)) is manager.result
    elif scenario == "success":
        assert runner.sample_tokens(manager, desc) is manager.result
    elif scenario == "failed-replay":
        manager.replay_fail = True
        try:
            step()
        except LookupError as exc:
            assert str(exc) == "replay-sentinel"
        else:
            raise AssertionError("replay exception swallowed")
    elif scenario == "dispatch-only":
        assert step(replay=False) is desc
    elif scenario == "same-class":
        other = ModelCudaGraphManager()
        assert other.capture() is other.result
        other_owner = f"ModelCudaGraphManager@{id(other):x}"
        assert step(other, replay=False) is desc
    elif scenario == "eager-dummy":
        assert runner.execute_model(manager, desc, dummy_run=True) is manager.result
        assert manager.run_fullgraph(desc) is manager.result  # outside serving scope
    elif scenario == "fork":
        read()
        child = os.fork()
        if child == 0:
            assert step(replay=False) is desc
            read()
            os._exit(0)
        assert os.waitpid(child, 0)[1] == 0
        inherited = json.loads(Path(f"{os.environ['GA_DISPATCH_LOG']}.{child}.json").read_text())
        print("CHILD=" + json.dumps(inherited))
elif scenario == "branch":
    for seq_len in (64, 500, 1000):
        gg._STATE.step_max_seq_len = seq_len
        assert step() is manager.result
elif scenario in ("capture-failed", "capture-incomplete"):
    assert step() is manager.result  # earlier success cannot redeem a later broken capture
    manager.fail = scenario == "capture-failed"
    manager.incomplete = scenario == "capture-incomplete"
    try:
        manager.capture()
    except ValueError as exc:
        assert manager.fail and str(exc) == "capture-sentinel"
    else:
        assert not manager.fail
else:
    raise AssertionError(scenario)
print("OWNER=" + owner)
if scenario == "same-class":
    print("OTHER=" + other_owner)
print("RECORD=" + json.dumps(read()))
'''


def _probe(tmp_path, scenario):
    package = tmp_path / "vllm/v1/worker/gpu"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent,
                      package.parent.parent.parent):
        (directory / "__init__.py").write_text("")
    (package / "cudagraph_utils.py").write_text(MANAGER)
    (package / "model_runner.py").write_text(RUNNER)
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
        if sep and key in ("OWNER", "OTHER", "RECORD", "CHILD"):
            result[key] = json.loads(value) if key in ("RECORD", "CHILD") else value
    return result


def _evidence(result):
    record = result["RECORD"]
    assert "owner_evidence" in record, "hook emitted dispatch totals, not owner-local replay outcomes"
    return record["owner_evidence"][result["OWNER"]]


def _count(owner):
    return sum(item["count"] for item in owner["replays"])


@pytest.mark.parametrize("scenario, count", [
    ("success", 2), ("failed-replay", 1), ("dispatch-only", 0), ("eager-dummy", 1),
])
def test_real_hooks_credit_only_successful_serving_returns(tmp_path, scenario, count):
    result = _probe(tmp_path, scenario)
    owner = _evidence(result)
    assert owner["capture_complete"] is True and owner["capture_error"] is None
    assert {tuple(d[k] for k in ("mode", "tokens", "num_reqs", "max_seq_len"))
            for d in owner["captured"]} == {("FULL", 1, 1, None), ("FULL", 2, 1, None)}
    assert _count(owner) == count
    if scenario == "failed-replay":
        assert any(d["tokens"] == 1 and "replay-sentinel" in d["error"]
                   and d["count"] == 1 for d in owner["replay_failures"])
    assert all(d["tokens"] == 1 and d["num_reqs"] == 1 and d["max_seq_len"] is None
               for d in owner["replays"])
    # The legacy diagnostic is additive, not renamed to successful replay coverage.
    assert result["RECORD"]["counts"]


@pytest.mark.parametrize("scenario", ["capture-failed", "capture-incomplete"])
def test_broken_capture_cannot_reuse_earlier_success(tmp_path, scenario):
    owner = _evidence(_probe(tmp_path, scenario))
    assert owner["capture_complete"] is False
    if scenario == "capture-failed":
        assert "capture-sentinel" in owner["capture_error"]
    # Retaining earlier replay history is fine only when the current capture
    # remains explicitly nonqualifying; the verifier tests enforce that gate.


def test_distinct_same_class_managers_do_not_share_replay_credit(tmp_path):
    result = _probe(tmp_path, "same-class")
    owner = _evidence(result)
    other = result["RECORD"]["owner_evidence"][result["OTHER"]]
    assert result["OWNER"] != result["OTHER"]
    assert _count(owner) == 1 and _count(other) == 0
    assert other["capture_complete"] is True


def test_branch_inventory_serializes_actual_descriptor_and_bound(tmp_path):
    owner = _evidence(_probe(tmp_path, "branch"))
    assert owner["capture_complete"] is True and owner["capture_error"] is None
    assert {(d["tokens"], d["num_reqs"], d["max_seq_len"]) for d in owner["captured"]} == {
        (tokens, 1, bound) for tokens in (1, 2) for bound in (192, 640, 8448)}
    assert {(d["tokens"], d["max_seq_len"], d["count"]) for d in owner["replays"]} == {
        (1, bound, 1) for bound in (192, 640, 8448)}


@pytest.mark.parametrize("role", ["target", "drafter"])
def test_serving_class_counter_increments_only_after_return(monkeypatch, role):
    from test_glm53_graphs import _install, _Model, TOPK
    from tessera.serving import glm53_graphs as gg
    utils, Drafter, State = _install(monkeypatch)
    state = State()
    for manager_type in (utils.ModelCudaGraphManager if role == "target" else Drafter,):
        manager = manager_type(["decode-b1"])
        manager.capture(_Model(TOPK), state)
        state.prepare_attn(for_capture=False, step_max_seq_len=64)
        key = "decode-b1"
        bound = min(gg._STATE.bounds)
        counter = (manager_type.__name__, bound)
        assert manager.run_fullgraph(key) == "fill"
        before = gg.REPLAYS[counter]
        # The original call raises naturally for a missing descriptor. Target
        # and drafter wrappers must preserve that error and earn no credit.
        with pytest.raises(KeyError, match="uncaptured"):
            manager.run_fullgraph("uncaptured")
        assert gg.REPLAYS[counter] == before, (
            "failed replay was counted as a successful class return")


def test_actual_hook_inventory_survives_complete_population_producer_verify(tmp_path):
    from test_graph_replay_coverage import Measurement
    measurement_root = tmp_path / "measurement"
    measurement_root.mkdir()
    m = Measurement(measurement_root, tp=1)
    result = _probe(tmp_path / "hooks", "complete")
    owner = _evidence(result)
    assert _count(owner) == len(owner["captured"]) == 2
    m.path().unlink()
    path = m.graph_dir / m.basename(pid=result["RECORD"]["pid"])
    path.write_text(json.dumps(result["RECORD"]))
    m.valid()  # complete W1 equality population plus actual W2 evidence through W3
    m.require_retention()
    owner["replays"] = [d for d in owner["replays"] if d["tokens"] != 2]
    path.write_text(json.dumps(result["RECORD"]))
    # Leave the historical positive dispatch totals untouched. They cannot
    # redeem a missing successful replay in the actual owner inventory.
    m.refused()


def test_forked_process_cannot_inherit_successful_history(tmp_path):
    result = _probe(tmp_path, "fork")
    assert _count(_evidence(result)) == 1
    child = result["CHILD"]
    assert child["pid"] != result["RECORD"]["pid"]
    assert sum(_count(o) for o in child.get("owner_evidence", {}).values()) == 0
    assert sum(child["counts"].values()) == 1, (
        "child dispatch bookkeeping must reset as well as successful replay counts")
