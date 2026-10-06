"""Real CPU subprocesses run the production protocol with simulated local device adapters.

No Docker, CUDA, NCCL, live peer claim, physical Spark floor or broker cleanup is
qualified here. Claims/scopes/CIDs are fixtures; process-group death is observed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import managed_window as window
import rank_window
import window_driver


class CpuArm:
    def __init__(self, root, owned, envelope, scenario):
        self.root, self.owned, self.envelope, self.scenario = root, owned, envelope, scenario
        self.guard = None
        self.server = None
        self.active = None

    def preflight(self):
        if self.scenario == "preflight" and self.owned["rank"] == 1:
            raise window.Refused("simulated MemAvailable 113.5 GiB < unchanged 114")
        metadata = dict(image="fixture-image", src_sha256="a" * 64, config_sha256="b" * 64)
        for key in metadata:
            if self.scenario == f"peer-mismatch-{key}" and self.owned["rank"] == 1:
                metadata[key] = f"different-rank1-{key}"
        return metadata

    def start(self, arm):
        self.active = arm["arm"]
        with (self.root / f"{self.active}.rank{self.owned['rank']}.server.log").open("w") as stream:
            self.server = subprocess.Popen([sys.executable, "-c", "import time;print('ready',flush=True);time.sleep(30)"],
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (self.root / f"{self.active}.rank{self.owned['rank']}.pid").write_text(str(self.server.pid))
        if self.scenario == "launch_failure" and self.owned["rank"] == 1:
            raise window.Refused("simulated CLI failure after local child created")
        return dict(cid=f"simulated-pid-{self.server.pid}", scope_id=self.owned["scope_id"], pid=self.server.pid)

    def tick(self):
        if self.guard: self.guard()
        if self.server and self.server.poll() is not None:
            raise window.Refused("simulated local server unexpectedly exited")
        if self.scenario == "floor" and self.owned["rank"] == 1:
            raise window.Refused("simulated physical floor 15.9 GiB < 16")

    def ready(self, arm):
        self.tick()
        banner = "Using network IB" if self.scenario == "fabric" and self.owned["rank"] == 1 else "Using network Socket"
        return dict(fabric=rank_window.fabric_from_log(banner))

    def probes(self, arm, peer):
        mode = window.read_json(self.root.parent / "scenario-mode.json")["mode"]
        failed_arm = {"window4-eager-2048-4096": "eager2048", "ship-eager-4096-8192": "eager4096",
                      "ship-graph-2048": "graph2048_off"}.get(mode, "aGR")
        failure = self.scenario in ("probe_failure", "timeout") and arm["arm"] == failed_arm
        program = "import time;print('partial',flush=True);time.sleep(30)" if failure and self.scenario == "timeout" else (
                  "print('partial',flush=True);raise SystemExit(7)" if failure else "print('two full passes and screens simulated',flush=True)")
        with (self.root / f"{arm['arm']}.probes.log").open("w") as stream:
            self.envelope.run([sys.executable, "-c", program], stdout=stream, tick=self.tick,
                              limit=self.envelope.remaining())

    def cleanup(self, arm):
        if self.server:
            if self.scenario == "removal_failure" and self.owned["rank"] == 1:
                self.envelope.run([sys.executable, "-c", "raise SystemExit(7)"], cleanup=True)
            os.killpg(self.server.pid, signal.SIGTERM)
            self.server.wait(timeout=min(1, self.envelope.remaining(cleanup=True)))
            self.server = None
        if self.scenario == "copy_failure" and self.owned["rank"] == 1:
            self.envelope.run(["cp", str(self.root / "absent-output"), str(self.root / "copy.log")], cleanup=True)
        return dict(containers_empty=True, gpu_descendants_empty=True, scope_id=self.owned["scope_id"])


def write_claim(root, owned):
    (root / "claimed").mkdir(parents=True, exist_ok=True)
    window.atomic_json(root / "claimed" / (owned["action_key"] + ".json"),
        dict(action_key=owned["action_key"], claimed_unix=owned["claimed_unix"], claimed_host=owned["host"],
             container_owner=owned["container_owner"],
             resource_scope={key: owned[key] for key in ("action_key", "nonce", "scope_id")}))


def make_identity(rank, start, end):
    key, nonce = str(rank + 1) * 64, str(rank + 3) * 32
    return dict(rank=rank, action_key=key, nonce=nonce, scope_id=window.scope_name(key, nonce),
                host=window.HOSTS[rank], container_owner=f"simulated-owner-{rank}",
                claimed_unix=start, run_id="simulated-fresh-run", input_sha256="a" * 64,
                window_end_unix=end)


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split()[2] != "Z"


def scenario(root, name, *, mode="graph-control"):
    queue, rdv = root / "queue", root / "rdv"
    rdv.mkdir()
    window.atomic_json(root / "scenario-mode.json", dict(mode=mode))
    start = time.time()
    # Shortened only in this simulated CPU fixture; production always has the fixed 5400s envelope.
    end = start + (3 if name == "timeout" else 8)
    identities = [make_identity(rank, start, end) for rank in (0, 1)]
    children = []
    started = time.monotonic()
    try:
        for rank, owned in enumerate(identities):
            write_claim(queue, owned)
            window.atomic_json(root / f"identity{rank}.json", owned)
            log = (root / f"controller{rank}.log").open("w")
            child = subprocess.Popen([sys.executable, str(Path(__file__)), "--scenario-rank", str(rank),
                                      str(root), name], stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log))
        for child, log in children:
            child.wait(timeout=12)
            log.close()
        outcomes = [window.read_json(rdv / f"outcome-rank{rank}.json") for rank in (0, 1)]
        pids = [int(path.read_text()) for path in rdv.glob("*.pid")]
        living = [pid for pid in pids if alive(pid)]
        return outcomes, living, time.monotonic() - started
    finally:
        for child, log in children:
            if child.poll() is None: child.kill(); child.wait()
            log.close()
        # Fault scenarios intentionally retain an exact owned child; never leave it on the PB test host.
        for path in rdv.glob("*.pid"):
            pid = int(path.read_text())
            if alive(pid): os.killpg(pid, signal.SIGKILL)


@pytest.mark.parametrize("name", ["success", "probe_failure", "timeout", "launch_failure", "copy_failure",
                                   "removal_failure", "preflight", "floor", "fabric"])
def test_real_bounded_protocol_and_failures_keep_partial_evidence(tmp_path, name):
    outcomes, living, seconds = scenario(tmp_path, name)
    assert all(outcome["simulation"] is True for outcome in outcomes)
    assert all(outcome["ownership_released"] is False for outcome in outcomes)
    assert seconds < 10
    rdv = tmp_path / "rdv"
    if name == "success":
        assert [outcome["returncode"] for outcome in outcomes] == [0, 0]
        assert all(outcome["completed_arms"] == ["aE1", "aGR", "aE2"] for outcome in outcomes)
        assert not living
    else:
        assert any(outcome["returncode"] for outcome in outcomes)
        assert not (rdv / "aE2.probes.log").exists()
        assert not (rdv / "aE2.rank0.pid").exists()
        assert not (rdv / "aE2.rank1.pid").exists()
        if name in ("probe_failure", "timeout"):
            assert "partial" in (rdv / "aGR.probes.log").read_text()
            assert all(outcome["completed_arms"] == ["aE1"] for outcome in outcomes)
        if name == "timeout":
            assert any(outcome["returncode"] == 124 for outcome in outcomes)
            assert all(outcome["ended_unix"] <= outcome["window_end_unix"] for outcome in outcomes)
        if name in ("copy_failure", "removal_failure"):
            assert any("cleanup_error" in outcome or "command failed" in outcome.get("error", "") for outcome in outcomes)
        assert bool(living) == (name == "removal_failure")


@pytest.mark.parametrize("field", ["image", "src_sha256", "config_sha256"])
@pytest.mark.parametrize("mode", ["graph-control", "window4-eager-2048-4096", "ship-eager-4096-8192"])
@pytest.mark.parametrize("dev_mode", [None, "1", "0"], ids=["default-dev", "explicit-dev", "certified"])
def test_mismatched_rank_preflight_refuses_before_either_launch(tmp_path, monkeypatch, field, mode, dev_mode):
    # Both controller subprocesses use the real rendezvous/protocol; only the
    # local device adapter is simulated. D32 never permits different gang halves.
    if dev_mode is None:
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    else:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    outcomes, living, seconds = scenario(tmp_path, f"peer-mismatch-{field}", mode=mode)
    assert not living and seconds < 10
    assert all(outcome["returncode"] == 1 and outcome["invocation_failed"]
               and outcome["completed_arms"] == [] for outcome in outcomes)
    assert any(f"rank preflight differs in {field}" in outcome["error"] for outcome in outcomes)
    rdv = tmp_path / "rdv"
    assert not list(rdv.glob("*.pid"))  # neither local server started
    assert not list(rdv.glob("*-started-rank*.json"))
    assert not list(rdv.glob("*-ready-rank*.json"))
    assert not list(rdv.glob("*-probes-rank*.json"))
    assert not list(rdv.glob("*.probes.log"))


def test_supported_campaign_rows_own_each_rank_and_aggregate_peaks(tmp_path):
    (tmp_path / "inputs.json").write_text("{}")
    env = dict(TS=str(tmp_path / "source"), ARTIFACT=str(tmp_path / "control"), RECEIPTS=str(tmp_path / "receipts"),
               FABRIC="socket", SOURCE_COMMIT="c" * 40, SOURCE_SHA256="a" * 64,
               PRODUCER_COMMIT="d" * 40, PRODUCER_SHA256="e" * 64)
    config = dict(ts=env["TS"], image=rank_window.recipe.IMAGE)
    rows = window_driver.rows(tmp_path, config, env)
    assert [row["tags"] for row in rows] == [["sparklina"], ["sparky"]]
    assert sum(row["demand"]["cpu"] for row in rows) == 14
    assert sum(row["demand"]["mem_gb"] for row in rows) == 208
    assert sum(row["gpu_memory_gb"] for row in rows) == 204
    assert all(row["exclusive"] and row["demand"]["gpu"] == 1 and row["timeout_s"] == 5400 for row in rows)
    assert all(row["measurement"] is True and row["host_class"] == "gb10" and row["priority"] == 10 for row in rows)
    assert all(row["env"]["GRAPH_PEER_WAIT_SECONDS"] == "3600" and row["priority_reason"].startswith("Goal:") for row in rows)
    assert all(row["max_attempts"] == 1 and row["env"]["OMP_NUM_THREADS"] == "1" for row in rows)
    assert all("ssh" not in word for row in rows for word in row["argv"])


def test_no_admission_means_no_rank_child(tmp_path):
    env = {**os.environ}
    for key in ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE", "PRISMABUILD_ACTION_SCOPE"):
        env.pop(key, None)
    done = subprocess.run([sys.executable, str(HERE / "rank_window.py"), "--rank", "0", "--run", str(tmp_path / "absent")],
                          env=env, capture_output=True, text=True, timeout=10)
    assert done.returncode == 3
    assert "requires an admitted PB action/attempt/scope" in done.stderr
    assert not (tmp_path / "rank0.json").exists()


@pytest.mark.parametrize("log", ["", "Using network Socket\nUsing network IB", "Using network Mystery"])
def test_fabric_cannot_be_inferred_from_a_request(log):
    with pytest.raises(window.Refused, match="actual NCCL"):
        rank_window.fabric_from_log(log)


if __name__ == "__main__":
    rank, root, name = int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    owned = window.read_json(root / f"identity{rank}.json")
    envelope = window.Envelope(owned["window_end_unix"], cleanup_seconds=.6)
    mode = window.read_json(root / "scenario-mode.json")["mode"]
    names = {"window4-eager-2048-4096": ("eager2048", "eager4096"),
             "ship-eager-4096-8192": ("eager4096", "eager8192"),
             "ship-graph-2048": ("graph2048_off", "graph2048_on")}
    arms = [dict(arm=arm) for arm in names.get(mode, ("aE1", "aGR", "aE2"))]
    adapter = CpuArm(root / "rdv", owned, envelope, name)
    raise SystemExit(rank_window.run_rank(dict(fabric="socket", window_mode=mode), owned, root / "queue", root / "rdv", arms,
                                         adapter, envelope, poll_seconds=.02))
