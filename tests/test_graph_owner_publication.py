"""CPU regressions for the completed owner-evidence review findings.

Use the real layered observer/serving path, without editing protected acceptance
probes. Competing flushes are event-controlled rather than relying on the daemon
flusher's one-second polling interval. No device work or completion is observed.
"""
import json
from pathlib import Path
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
import threading
from types import SimpleNamespace as NS

hook, scratch, scenario = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(scratch))
spec = importlib.util.spec_from_file_location("publication_hook_probe", hook)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
# We drive two real flush calls below instead of the daemon's polling loop.
observer._ga_flusher = lambda: None
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager, Descriptor
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
manager, runner = ModelCudaGraphManager(), GPUModelRunner()
owner = f"ModelCudaGraphManager@{id(manager):x}"
path = Path(f"{os.environ['GA_DISPATCH_LOG']}.{os.getpid()}.json")

def read():
    return json.loads(path.read_text())

if scenario == "non-cartesian":
    from tessera.serving import glm53_graphs as gg
    import vllm.v1.worker.gpu.cudagraph_utils as utils
    gg._STATE = NS(installed=True, bounds=(192, 640, 8448), capture_bound=None,
                   bound_applied=False, step_max_seq_len=None)
    gg._patch_managers(utils, (ModelCudaGraphManager,))
    manager.capture(NS(modules=lambda: []))
    for seq_len, token in ((64, 1), (500, 2), (1000, 2)):
        gg._STATE.step_max_seq_len = seq_len
        assert runner.execute_model(manager, Descriptor(token, 1 if seq_len == 64 else 2)) is manager.result
    observer._ga_flush(force=True)
    record = read()
    assert record["captured"][owner] == [1, 2], record
    evidence = record["owner_evidence"][owner]
    assert {(d["tokens"], d["num_reqs"], d["max_seq_len"]) for d in evidence["captured"]} == {
        (1, 1, 192), (2, 2, 640), (2, 2, 8448)}
    assert {(d["tokens"], d["num_reqs"], d["max_seq_len"], d["count"])
            for d in evidence["replays"]} == {
        (1, 1, 192, 1), (2, 2, 640, 1), (2, 2, 8448, 1)}
elif scenario == "stale-publication":
    manager.capture()
    for token in (1, 2):
        assert runner.execute_model(manager, Descriptor(token)) is manager.result
    observer._ga_flush(force=True)
    assert read()["owner_evidence"][owner]["capture_complete"] is True

    snapshot_ready, release_snapshot = threading.Event(), threading.Event()
    manager.capture_entered, manager.capture_release = threading.Event(), threading.Event()
    original_lock = observer._GA["lock"]
    original_write_lock = observer._GA["write_lock"]
    publication_attempted = threading.Event()
    publication = {}
    errors = []
    class PublicationLock:
        def __enter__(self):
            if threading.current_thread().name == "recapture":
                # Distinguish the repaired order (old writer owns the lock)
                # from the buggy order (new invalidation can publish first).
                acquired = original_write_lock.acquire(blocking=False)
                publication["acquired"] = acquired
                publication_attempted.set()
                if acquired:
                    return
            original_write_lock.acquire()
        def __exit__(self, *exc):
            original_write_lock.release()
    class SnapshotLock:
        def __enter__(self):
            original_lock.acquire()
        def __exit__(self, *exc):
            original_lock.release()
            # Pause AFTER releasing the state lock, not during JSON copying:
            # recapture must be free to invalidate while the old writer waits.
            if threading.current_thread().name == "older-flush":
                snapshot_ready.set()
                assert release_snapshot.wait(10), "test did not release old snapshot"
    def checked(call):
        try:
            call()
        except BaseException as exc:
            errors.append(exc)
    observer._GA["lock"] = SnapshotLock()
    observer._GA["write_lock"] = PublicationLock()
    old = threading.Thread(name="older-flush", target=lambda: checked(
        lambda: observer._ga_flush(force=True)))
    recapture = threading.Thread(name="recapture", target=lambda: checked(manager.capture))
    old.start()
    try:
        assert snapshot_ready.wait(10), "older writer did not construct a snapshot"
        recapture.start()
        # Before the fix the new invalidation publishes and raw capture enters
        # while the old snapshot is delayed. After serialization it must wait
        # behind the old writer, then publish invalidation before raw capture.
        assert publication_attempted.wait(10), "recapture did not attempt invalidation"
        if publication["acquired"]:
            # Buggy order: insist invalidation is published and capture enters
            # before releasing the old writer. Scheduling cannot hide the race.
            assert manager.capture_entered.wait(10), "invalidation did not publish"
        release_snapshot.set()
        old.join(10)
        assert not old.is_alive()
        assert manager.capture_entered.wait(10), "recapture never entered"
        record = read()  # Do not flush: inspect what competing writers published.
        assert record["owner_evidence"][owner]["capture_complete"] is False, record
        assert not record["owner_evidence"][owner]["replays"], record
    finally:
        release_snapshot.set()
        manager.capture_release.set()
        old.join(10)
        if recapture.ident is not None:
            recapture.join(10)
        observer._GA["lock"] = original_lock
        observer._GA["write_lock"] = original_write_lock
    assert not old.is_alive() and not recapture.is_alive()
    assert not errors, errors
else:
    raise AssertionError(scenario)
print("RECORD=" + json.dumps(record))
'''


def _probe(tmp_path, scenario):
    package = tmp_path / "vllm/v1/worker/gpu"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent,
                      package.parent.parent.parent):
        (directory / "__init__.py").write_text("")
    # The CPU stand-in varies token sets across actual capture bounds, not only
    # request dimensions. The wrappers under test are unmodified production code.
    source = MANAGER.replace(
        "        self.graphs = {Descriptor(1): object(), Descriptor(2): object()}",
        "        self.graphs = {Descriptor(1, 1) if gg._STATE.capture_bound == 192 else Descriptor(2, 2): object()}"
        if scenario == "non-cartesian" else
        "        self.graphs = {Descriptor(1): object(), Descriptor(2): object()}")
    source = source.replace(
        "    def capture(self, *args):\n        import sys",
        "    def capture(self, *args):\n"
        "        if hasattr(self, 'capture_entered'):\n"
        "            self.capture_entered.set()\n"
        "            assert self.capture_release.wait(10), 'test did not release recapture'\n"
        "        import sys")
    (package / "cudagraph_utils.py").write_text(source)
    (package / "model_runner.py").write_text(RUNNER)
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(HOOK), str(tmp_path), scenario],
        env=child_env(GA_DISPATCH_LOG=str(tmp_path / "dispatch"), T508_DIGEST="",
                      T508_CAPTURE_LOG="", T508_PROF_DIR="", T508_MOE_DETERMINISTIC="",
                      T695_GC_BEFORE_DRAFTER="", T695_DRAFT_LOG=""),
        capture_output=True, text=True, timeout=45)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(next(line.removeprefix("RECORD=") for line in done.stdout.splitlines()
                           if line.startswith("RECORD=")))


def test_competing_flush_cannot_overwrite_recapture_invalidation(tmp_path):
    _probe(tmp_path, "stale-publication")


def test_non_cartesian_token_inventory_and_legacy_diagnostic_agree(tmp_path):
    _probe(tmp_path, "non-cartesian")
