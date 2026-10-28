"""CPU hook-to-wire acceptance for #1259; no CUDA/device-completion claim.

Import the real observer and let its real import finder patch stand-in vLLM
modules. The additive owner_evidence wire is the W2 handoff to W3 (#1260).
Adapted from the earlier reviewed, unmerged CPU probes; no runtime git access.
Existing counts remain dispatch diagnostics.

Research evidence (base 17b3fdb396410698c9a142615345f046d2f94a5d):
The final plain x86 PrismaBuild selection of these two new acceptance files and
 test_glm53_graphs.py failed all 17 new cases; the existing 51 cases passed.
Failures demonstrate missing owner_evidence, premature target/drafter counters
(2 != 1), and captured=[2] instead of the observed token union [1, 2]. The
competing-writer probe currently stops at the missing evidence prerequisite; it
is not a newly reproduced publication race on this base. Layered probes preserve
the real original capture errors before asserting the missing evidence. An
intermediate run had an incorrect exception-message assertion; that harness
assertion was corrected before this final failing-before run.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b09fa51a215454b8953be358a08eb5b1b1b81eaad698f6771e332df1d241b972","size":33444}},"operation":"631f9a70e7f4b4d02bfc4aecdb79630294957b4fc5693e556aefd3934ff8423e","receipt":"3405a4c5771d8ba08a09670d3c0f9fd04fb096c275facdf19b57445a42f12972"}]
A separate baseline action passed tests/test_glm53_graphs.py: 51 passed.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e9eeecc93d817f4b1f03364bb61b3bea68d4da077b26191bad04faa18758ce49","size":578}},"operation":"c91f64e76d3e51ec429d054ec4504bd21112d707ae89f9c6cca466e4318f99aa","receipt":"7e59ed32c55cc2c9965e5907b6e2c79f0fa16e5bba97e9b41f7c8557fa6f3ec2"}]
Both actions reported NO CUDA, zero skips/uncollected modules/device allocations.
A fresh continuation replay of the same combined command again observed 17 failed,
51 passed, zero skips/uncollected modules/device allocations on the x86 arm.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c0a42a83de9d4c9c8353decb4ecf6f7000d4e68402f005c6b2467b05d440e17f","size":33418}},"operation":"ccf28e1ea30dc15a79d6225715a9dcc05214feb7c4c0816b099a23637243f135","receipt":"12c27da7fa47d5f9a93157067217b8cf983fe63816394949ec9680570ca01b3f"}]
No GPU run or device/compiled qualification is claimed. Implementer must retain
post-fix passing action receipts and obtain separate-session implementation review.
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
        self.fail_bound = self.incomplete_bound = self.bypass_bound = None
    def capture(self, *args):
        import sys
        gg = sys.modules.get("tessera.serving.glm53_graphs")
        bound = gg._STATE.capture_bound if gg is not None else None
        if bound is not None and bound != self.bypass_bound:
            gg._record_build(lambda **kw: None)(max_seq_len=8448, for_cudagraph_capture=True)
        self.graphs = {Descriptor(1): object(), Descriptor(2): object()}
        if self.fail or (bound is not None and bound == self.fail_bound):
            raise ValueError("capture-sentinel")
        self._graphs_captured = not (self.incomplete or
            (bound is not None and bound == self.incomplete_bound))
        return self.result
    def captured_token_counts(self):
        return [d.num_tokens for d in self.graphs]
    def dispatch(self, desc):
        return desc
    def run_fullgraph(self, desc):
        if self.replay_fail:
            raise LookupError("replay-sentinel")
        self.graphs[desc]  # a real unknown descriptor cannot be replayed
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
if scenario.startswith("branch"):
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

if scenario in ("success", "none-return", "complete", "failed-replay", "dispatch-only", "same-class", "fork", "eager-dummy"):
    if scenario == "none-return":
        manager.result = None
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
elif scenario.startswith("branch-"):
    gg._STATE.step_max_seq_len = 64
    assert step() is manager.result
    if scenario == "branch-failed":
        manager.fail_bound = 640
    elif scenario == "branch-incomplete":
        manager.incomplete_bound = 640
    elif scenario == "branch-unobserved":
        manager.bypass_bound = 640
    else:
        raise AssertionError(scenario)
    try:
        manager.capture(NS(modules=lambda: []))
    except (ValueError, RuntimeError) as exc:
        assert scenario != "branch-incomplete"
        if scenario == "branch-failed":
            assert str(exc) == "capture-sentinel"
        else:
            assert "without building its metadata" in str(exc)
    else:
        assert scenario == "branch-incomplete"
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
    # Inspect immediate invalidation, then ensure later successful host returns
    # and another same-class owner's progress cannot restore completeness.
    print("BROKEN=" + json.dumps(read()))
    other = ModelCudaGraphManager()
    assert other.capture() is other.result
    other_owner = f"ModelCudaGraphManager@{id(other):x}"
    assert step(other) is other.result
    assert step() is manager.result
else:
    raise AssertionError(scenario)
print("OWNER=" + owner)
if scenario in ("same-class", "capture-failed", "capture-incomplete"):
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
        if sep and key in ("OWNER", "OTHER", "RECORD", "CHILD", "BROKEN"):
            result[key] = json.loads(value) if key in ("RECORD", "CHILD", "BROKEN") else value
    return result


def _evidence(result):
    record = result["RECORD"]
    assert "owner_evidence" in record, "hook emitted dispatch totals, not owner-local replay outcomes"
    return record["owner_evidence"][result["OWNER"]]


def _count(owner):
    return sum(item["count"] for item in owner["replays"])


@pytest.mark.parametrize("scenario, count", [
    ("success", 2), ("none-return", 1), ("failed-replay", 1), ("dispatch-only", 0), ("eager-dummy", 1),
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
    result = _probe(tmp_path, scenario)
    owner = _evidence(result)
    assert owner["capture_complete"] is False
    if scenario == "capture-failed":
        assert "capture-sentinel" in owner["capture_error"]
    # Old capture generations may be retained only as separately labelled
    # history, never as current qualifying inventory or successful returns.
    broken = result["BROKEN"]["owner_evidence"][result["OWNER"]]
    assert broken["capture_complete"] is False and not broken["replays"]
    other = result["RECORD"]["owner_evidence"][result["OTHER"]]
    assert other["capture_complete"] is True and _count(other) == 1


@pytest.mark.parametrize("scenario", ["branch-failed", "branch-incomplete", "branch-unobserved"])
def test_layered_capture_requires_every_branch_and_outer_success(tmp_path, scenario):
    owner = _evidence(_probe(tmp_path, scenario))
    assert owner["capture_complete"] is False
    assert not owner["replays"], "partial recapture reused the old generation's credit"
    if scenario == "branch-failed":
        assert "capture-sentinel" in owner["capture_error"]
    elif scenario == "branch-unobserved":
        assert "without building its metadata" in owner["capture_error"]


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


def test_forked_process_cannot_inherit_successful_history(tmp_path):
    result = _probe(tmp_path, "fork")
    assert _count(_evidence(result)) == 1
    child = result["CHILD"]
    assert child["pid"] != result["RECORD"]["pid"]
    assert sum(_count(o) for o in child.get("owner_evidence", {}).values()) == 0
    assert sum(child["counts"].values()) == 1, (
        "child dispatch bookkeeping must reset as well as successful replay counts")
