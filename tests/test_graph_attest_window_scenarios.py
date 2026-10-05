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
        return dict(image="fixture-image", src_sha256="a" * 64, config_sha256="b" * 64)

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
        failure = self.scenario in ("probe_failure", "timeout") and arm["arm"] == "aGR"
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


def scenario(root, name):
    queue, rdv = root / "queue", root / "rdv"
    rdv.mkdir()
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


def test_supported_campaign_rows_own_each_rank_and_aggregate_peaks(tmp_path):
    (tmp_path / "inputs.json").write_text("{}")
    env = dict(TS="/mnt/shared/source", ARTIFACT="/mnt/shared/control", RECEIPTS="/mnt/shared/receipts",
               FABRIC="socket", SOURCE_COMMIT="c" * 40, SOURCE_SHA256="a" * 64)
    config = dict(ts=env["TS"], image=rank_window.recipe.IMAGE)
    rows = window_driver.rows(tmp_path, config, env)
    assert [row["tags"] for row in rows] == [["sparklina"], ["sparky"]]
    assert sum(row["demand"]["cpu"] for row in rows) == 14
    assert sum(row["demand"]["mem_gb"] for row in rows) == 208
    assert sum(row["gpu_memory_gb"] for row in rows) == 204
    assert all(row["exclusive"] and row["demand"]["gpu"] == 1 and row["timeout_s"] == 5400 for row in rows)
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
    arms = [dict(arm=arm) for arm in ("aE1", "aGR", "aE2")]
    adapter = CpuArm(root / "rdv", owned, envelope, name)
    raise SystemExit(rank_window.run_rank(dict(fabric="socket"), owned, root / "queue", root / "rdv", arms,
                                         adapter, envelope, poll_seconds=.02))
