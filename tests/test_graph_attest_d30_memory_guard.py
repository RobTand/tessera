"""D30 Window4 memory-guard regressions (#959): threshold boundary, the 1 Hz
dual-rank abort guard, and owned TERM→SIGKILL grace — all through the real
production APIs.

CPU-only and deterministic: the boundary, cadence and ownership tests drive the
real ``LocalArm.headroom``/``tick``/``_term_server``/``_stop_server`` and the
real ``Rendezvous``/``run_rank`` failure path on a fake clock, substituting
only the physical Docker, /proc/meminfo and subprocess boundaries. The
real-process tests spawn actual children: a TERM-ignoring process group must be
SIGKILLed only after the full ten-second grace, a cooperative child must exit
within the grace without any SIGKILL, and a finite cleanup deadline must
shorten the grace. Exactly one test needs real Docker inside an admitted PB
action; it is marked for the published PB scope and skips with its reason
anywhere else — never silently.

The selected before/after population covers observable threshold, cadence and
process-grace changes. Container ownership cases cover the new helper separately;
they are not advertised as before-source behavioral failures.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import managed_window as window
import tp2_recipe as recipe

REFUSAL = "local MemAvailable below 107 GiB preflight"
FLOOR = "local MemAvailable below 2 GiB physical memory floor"
POLL = 0.2
CID = "a" * 64


class FakeClock:
    """One clock shared by rank_window and managed_window during a test."""

    def __init__(self, monotonic=100.0):
        self.monotonic_value, self.unix_value = monotonic, 1_800_000_000.0
        self.sleeps = []

    def time(self):
        return self.unix_value

    def monotonic(self):
        return self.monotonic_value

    def sleep(self, seconds):
        assert seconds >= 0, "no wait may compute a negative sleep"
        self.sleeps.append(seconds)
        self.monotonic_value += seconds
        self.unix_value += seconds

    def advance(self, seconds):
        self.monotonic_value += seconds
        self.unix_value += seconds


def memory_trace(clock, schedule):
    """available_gib keyed by fake elapsed time, as one real meminfo poll."""
    def read():
        value = schedule[0][1]
        for moment, gib in schedule:
            if clock.monotonic_value >= moment:
                value = gib
        return value
    return read


def identity(rank=0, **fields):
    key, nonce = str(rank + 1) * 64, str(rank + 3) * 32
    return dict(rank=rank, host=window.HOSTS[rank], scope_id=window.scope_name(key, nonce),
                container_owner=f"owned-parent-{rank}", run_id="run-" + "a" * 8,
                nonce=nonce, action_key=key, input_sha256="a" * 64,
                claimed_unix=time.time() - 5, window_end_unix=time.time() + 600, **fields)


def write_claim(root, owned):
    (root / "claimed").mkdir(parents=True, exist_ok=True)
    window.atomic_json(root / "claimed" / (owned["action_key"] + ".json"),
                       dict(action_key=owned["action_key"], claimed_unix=owned["claimed_unix"],
                            claimed_host=owned["host"], container_owner=owned["container_owner"],
                            resource_scope={key: owned[key] for key in ("action_key", "nonce", "scope_id")}))


def new_adapter(tmp_path, owned, rank):
    """A LocalArm shaped exactly as __init__ leaves it, on tmp directories."""
    import rank_window
    adapter = rank_window.LocalArm.__new__(rank_window.LocalArm)
    adapter.identity, adapter.rank = owned, rank
    adapter.config = {}
    adapter.envelope = window.Envelope(time.time() + 600, cleanup_seconds=60)
    adapter.work, adapter.rdv = tmp_path / "work", tmp_path / "rdv"
    adapter.work.mkdir(parents=True, exist_ok=True)
    (adapter.work / "ext").mkdir(parents=True, exist_ok=True)
    adapter.rdv.mkdir(parents=True, exist_ok=True)
    adapter.ext = adapter.work / "ext"
    adapter.active, adapter.containers, adapter.last_sample = None, [], 0
    adapter.last_maintenance_sample = 0
    adapter.memory_summary = dict(rank=rank, host=owned["host"], samples=0,
                                  baseline_gib=None, minimum_gib=None)
    adapter.server_terminations = {}
    adapter.abort = None
    adapter.image_env = {}
    adapter.guard_calls = []
    adapter.guard = lambda: adapter.guard_calls.append("check")
    adapter.poll_seconds = POLL
    return adapter


def d30_adapter(tmp_path, monkeypatch, *, rank=0, read, clock, eager=False):
    import rank_window
    monkeypatch.setattr(rank_window, "available_gib", read)
    monkeypatch.setattr(rank_window, "time", clock)
    monkeypatch.setattr(window, "time", clock)
    owned = identity(rank)
    if eager:
        monkeypatch.setattr(rank_window, "scope_memory",
                            lambda owned: dict(rank=owned["rank"], scope_id=owned["scope_id"]))
    adapter = new_adapter(tmp_path, owned, rank)
    adapter.config = dict(window_mode=recipe.EAGER_MODE if eager else "graph-control")
    return adapter


def stub_boundaries(adapter, monkeypatch):
    """Substitutes only the subprocess and frozen-source boundaries."""
    import rank_window
    adapter.config.update(ts=str(ROOT / "src"),
                          image="frozen-full-image@" + "d" * 32,
                          src_sha256="e" * 64, config_sha256="f" * 64)
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "ps"] or argv[0] == "nvidia-smi":
            payload = ""
        elif "resolve" in argv:
            payload = json.dumps(dict(resolved_reference=adapter.config["image"],
                                      local_id="sha256:" + "c" * 64))
        elif "container-env" in argv:
            payload = "TESSERA_IMAGE_ENV=1\n"
        else:
            raise AssertionError(argv)
        return SimpleNamespace(stdout=payload)

    adapter.command = command
    monkeypatch.setattr(rank_window.recipe, "inputs",
                        lambda environ, live, runner: adapter.config)
    return calls


def reports(adapter):
    path = adapter.rdv / f"headroom-preflight-rank{adapter.rank}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def await_dead(pids, deadline):
    """SIGKILL teardown is asynchronous; death is awaited, never sampled once."""
    while time.monotonic() < deadline and any(alive(pid) for pid in pids):
        time.sleep(.05)


# --- the deterministic 107 GiB start boundary -------------------------------

def test_headroom_accepts_exactly_the_107_gib_start_boundary(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = d30_adapter(tmp_path, monkeypatch, read=lambda: 107.0, clock=clock)
    calls = stub_boundaries(adapter, monkeypatch)
    metadata = adapter.preflight()
    assert metadata["image"] == adapter.config["image"]
    assert calls, "a ready preflight continues into the ordinary image/source checks"
    (report,) = reports(adapter)
    assert report["threshold_gib"] == 107 and report["reason"] == "ready"
    assert [s["available_gib"] for s in report["samples"]] == [107.0]
    assert report["elapsed_seconds"] == pytest.approx(0.0) and clock.sleeps == []


def test_headroom_refuses_106_99_with_the_current_text_and_no_model(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = d30_adapter(tmp_path, monkeypatch,
                          read=memory_trace(clock, [(100.0, 106.99)]), clock=clock)

    def forbidden(argv, **kwargs):
        raise AssertionError(f"a refused wait must start no image or model command: {argv}")

    adapter.command = forbidden
    with pytest.raises(window.Refused) as excinfo:
        adapter.preflight()
    assert str(excinfo.value) == REFUSAL
    (report,) = reports(adapter)
    assert report["threshold_gib"] == 107 and report["reason"] == "headroom_timeout"
    assert report["last_available_gib"] == 106.99
    assert report["elapsed_seconds"] == pytest.approx(900.0)


def test_launch_recheck_refuses_106_99_after_a_ready_preflight(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = d30_adapter(tmp_path, monkeypatch, read=lambda: 107.25, clock=clock)
    stub_boundaries(adapter, monkeypatch)
    assert adapter.preflight()["image"] == adapter.config["image"]
    monkeypatch.setattr(rank_window, "available_gib", lambda: 106.99)

    def forbidden(*args, **kwargs):
        raise AssertionError("a launch below 107 GiB is refused before any argv is rendered")

    monkeypatch.setattr(rank_window.recipe, "container", forbidden)
    with pytest.raises(window.Refused) as excinfo:
        adapter.start(dict(arm="aGR"))
    assert str(excinfo.value) == REFUSAL
    assert not (adapter.work / "aGR").exists()
    sample = json.loads((adapter.rdv / "aGR-launch-headroom-rank0.json").read_text())
    assert sample["mem_available_gib"] == 106.99 and sample["threshold_gib"] == 107
    assert sample["rank"] == 0


# --- the 1 Hz guard, its exact 2 GiB boundary and its dual-rank publication --

def test_guard_accepts_exactly_2_gib_then_refuses_1_99_one_second_later(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    rdv = tmp_path / "rdv"
    rdv.mkdir()
    owned = identity(0)
    write_claim(tmp_path, owned)
    meeting = window.Rendezvous(rdv, owned, tmp_path,
                                window.Envelope(time.time() + 600, cleanup_seconds=60))
    adapter = d30_adapter(tmp_path, monkeypatch,
                          read=memory_trace(clock, [(100.0, 2.0), (101.0, 1.99)]), clock=clock)
    adapter.abort = lambda **fields: meeting.publish("failed", **fields)  # run_rank's wiring
    adapter.tick()  # exactly 2.0 GiB: strictly-below is false, the rank runs on
    assert not (rdv / "failed-rank0.json").exists()
    clock.advance(1.0)
    with pytest.raises(window.Refused) as excinfo:
        adapter.tick()
    assert str(excinfo.value) == FLOOR
    failed = window.read_json(rdv / "failed-rank0.json")
    assert failed["rank"] == 0 and failed["action_key"] == owned["action_key"]
    assert FLOOR in failed["error"], "the abort publishes the peer-facing failure"
    samples = [json.loads(line) for line in (rdv / "memory-samples-rank0.jsonl").read_text().splitlines()]
    assert [sample["mem_available_gib"] for sample in samples] == [2.0, 1.99]
    summary = window.read_json(rdv / "memory-summary-rank0.json")
    assert summary["baseline_gib"] == 2.0 and summary["minimum_gib"] == 1.99
    assert summary["samples"] == 2, "per-host start/minimum/raw samples are retained"


def test_guard_samples_at_one_hz_while_docker_health_keeps_five_seconds(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = d30_adapter(tmp_path, monkeypatch, read=lambda: 200.0, clock=clock)
    owned = adapter.identity
    inspects = []

    def command(argv, **kwargs):
        if argv[:2] == ["docker", "inspect"]:
            inspects.append(clock.monotonic())
            labels = {"prismabuild.scope": owned["scope_id"], "prismabuild.action": owned["container_owner"],
                      "org.prismaquant.graph-window": owned["run_id"], "org.prismaquant.attempt": owned["nonce"]}
            payload = [dict(Id=CID, Config=dict(Labels=labels),
                            HostConfig=dict(CgroupParent=owned["scope_id"]),
                            State=dict(Running=True, Pid=99))]
            return SimpleNamespace(stdout=json.dumps(payload))
        raise AssertionError(argv)

    adapter.command = command
    adapter.active = dict(arm="aGR", cid=CID)
    for _ in range(6):
        adapter.tick()
        clock.advance(1.0)
    memwatch = (adapter.work / "memwatch.txt").read_text().splitlines()
    assert len(memwatch) == 6, "the guard samples once per second"
    samples = (adapter.rdv / "memory-samples-rank0.jsonl").read_text().splitlines()
    assert len(samples) == 6
    assert inspects == [100.0, 105.0], "docker health keeps its five-second cadence"
    assert len(adapter.guard_calls) == 6, "the peer guard still runs on every tick"
    summary = window.read_json(adapter.rdv / "memory-summary-rank0.json")
    assert summary["samples"] == 6 and summary["minimum_gib"] == 200.0


def test_eager_scope_sampling_follows_the_one_hz_guard(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = d30_adapter(tmp_path, monkeypatch, read=lambda: 200.0, clock=clock, eager=True)
    for _ in range(3):
        adapter.tick()
        clock.advance(1.0)
    scope = (adapter.work / "memory-scope.jsonl").read_text().splitlines()
    assert len(scope) == 3, "the eager scope sampler rides the 1 Hz guard, not the 5 s cadence"
    assert all(json.loads(line)["rank"] == 0 for line in scope)




# --- real Rendezvous propagation: one rank's abort fails both ranks ----------

def test_peer_abort_failure_propagates_and_rejects_the_run(tmp_path, monkeypatch):
    import rank_window
    queue, rdv = tmp_path / "queue", tmp_path / "rdv"
    rdv.mkdir()
    owned = [identity(rank) for rank in (0, 1)]
    for row in owned:
        write_claim(queue, row)
    # Rank 1 aborts below the floor and publishes through the real Rendezvous,
    # exactly as run_rank's assigned abort callback does.
    aborting = window.Rendezvous(rdv, owned[1], queue,
                                 window.Envelope(time.time() + 600, cleanup_seconds=60))
    aborting.publish("failed", error=f"Refused: {FLOOR}")
    aborting.publish("failed-cleaned", local_cleanup=[], cleanup_error=None)
    adapter = new_adapter(tmp_path, owned[0], 0)

    def command(argv, **kwargs):
        if argv[:2] == ["docker", "ps"] or argv[0] == "nvidia-smi":
            return SimpleNamespace(stdout="")
        raise AssertionError(argv)

    adapter.command = command

    def forbidden(*args, **kwargs):
        raise AssertionError("no container may be rendered after a peer abort")

    monkeypatch.setattr(rank_window.recipe, "container", forbidden)
    envelope = window.Envelope(time.time() + 600, cleanup_seconds=60)
    config = dict(window_mode="graph-control", source_commit="s" * 40, producer_commit="p" * 40)
    code = rank_window.run_rank(config, owned[0], queue, rdv,
                                [dict(arm="aGR", eager=False, compilation="{}", spec="{}")],
                                adapter, envelope, poll_seconds=.02)
    assert code == 1
    outcome = window.read_json(rdv / "outcome-rank0.json")
    assert outcome["completed_arms"] == [], "a peer abort rejects the whole run: no scores"
    assert "rank 1 failed" in outcome["error"] and FLOOR in outcome["error"]
    assert "stale or mismatched" not in outcome["error"], "the failed record is exact-attempt"
    assert (rdv / "failed-rank0.json").exists(), "this rank recorded its own failure too"
    assert not (rdv / "aGR-preflight-rank0.json").exists()
    assert not (adapter.work / "aGR").exists()


# --- real Envelope children: TERM, the ten-second grace, and the SIGKILL -----

STUBBORN_CHILD = (
    "import json,os,signal,subprocess,sys,time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "grand = subprocess.Popen([sys.executable, '-c',\n"
    "    \"import signal,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);time.sleep(120)\"])\n"
    "open(sys.argv[1], 'w').write(json.dumps(dict(child=os.getpid(), grandchild=grand.pid)))\n"
    "time.sleep(120)\n"
)

COOPERATIVE_CHILD = (
    "import os,signal,sys,time\n"
    "log = open(sys.argv[1], 'a')\n"
    "log.write(f'pid {os.getpid()}\\n'); log.flush()\n"
    "def on_term(signum, frame):\n"
    "    log.write(f'TERM {time.monotonic():.3f}\\n'); log.flush(); time.sleep(3)\n"
    "    log.write('graceful-exit\\n'); log.flush(); raise SystemExit(0)\n"
    "signal.signal(signal.SIGTERM, on_term)\n"
    "time.sleep(120)\n"
)

STUBBORN_ALONE_CHILD = (
    "import os,signal,sys,time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    "time.sleep(120)\n"
)


def signal_names(record):
    return [event["signal"] for event in record["signals"]]


def test_term_ignoring_child_group_is_killed_only_after_the_full_grace(tmp_path):
    sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"],
                                start_new_session=True)
    pids = tmp_path / "group.json"
    envelope = window.Envelope(time.time() + 60, cleanup_seconds=15)
    errors = []
    def run():
        try:
            envelope.run([sys.executable, "-c", STUBBORN_CHILD, str(pids)], limit=1.0)
        except TimeoutError:
            return
        except BaseException as exc:
            errors.append(exc)
        else:
            errors.append(AssertionError("TERM-resistant child unexpectedly completed"))
    thread = threading.Thread(target=run)
    try:
        thread.start()
        until = time.monotonic() + 3
        while not pids.exists() and time.monotonic() < until:
            time.sleep(.02)
        ids = json.loads(pids.read_text())
        time.sleep(5)  # observe DURING Envelope.run's grace, not after it returns
        assert alive(ids["child"]) and alive(ids["grandchild"])
        assert alive(sentinel.pid), "an unrelated session sentinel is never touched"
        thread.join(timeout=12)
        assert not thread.is_alive() and not errors, errors
        await_dead(list(ids.values()), time.monotonic() + 2)
        assert not any(alive(pid) for pid in ids.values())
        assert alive(sentinel.pid)
        (record,) = envelope.terminations
        assert record["pid"] == ids["child"]
        assert signal_names(record) == ["SIGTERM", "SIGKILL"]
        assert record["signals"][1]["monotonic"] - record["signals"][0]["monotonic"] >= 10.0
        assert record["returncode"] == -9
    finally:
        if pids.exists():
            try:
                os.killpg(json.loads(pids.read_text())["child"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        thread.join(timeout=3)
        sentinel.terminate()
        sentinel.wait(timeout=3)


def test_cooperative_child_exits_within_grace_without_a_kill(tmp_path):
    marker = tmp_path / "cooperative.log"
    envelope = window.Envelope(time.time() + 300, cleanup_seconds=30)
    with pytest.raises(TimeoutError):
        envelope.run([sys.executable, "-c", COOPERATIVE_CHILD, str(marker)], limit=2.0)
    child = int(marker.read_text().splitlines()[0].split()[1])
    assert not alive(child)
    assert "graceful-exit" in marker.read_text(), "the cooperative handler ran to completion"
    (record,) = envelope.terminations
    grace = record["ended_unix"] - record["signals"][0]["unix"]
    assert 2.5 <= grace <= 9.5, "the cooperative handler gets its actual TERM grace"
    assert record["pid"] == child
    assert signal_names(record) == ["SIGTERM"], "a cooperative exit needs no SIGKILL"
    assert record["returncode"] == 0


def test_finite_cleanup_deadline_shortens_the_kill_grace(tmp_path):
    pids = tmp_path / "stubborn.pid"
    envelope = window.Envelope(time.time() + 8, cleanup_seconds=2)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        envelope.run([sys.executable, "-c", STUBBORN_ALONE_CHILD, str(pids)], limit=1.0, cleanup=True)
    elapsed = time.monotonic() - started
    assert not alive(int(pids.read_text()))
    assert 7 <= elapsed <= 9.5, "termination respects the original eight-second envelope"
    (record,) = envelope.terminations
    assert signal_names(record) == ["SIGTERM", "SIGKILL"]
    assert record["deadline_shortened_grace"] is True


# --- exact-attempt container termination -------------------------------------

def owned_labels(owned):
    return {"prismabuild.scope": owned["scope_id"], "prismabuild.action": owned["container_owner"],
            "org.prismaquant.graph-window": owned["run_id"], "org.prismaquant.attempt": owned["nonce"]}


class DockerStub:
    """The docker inspect/kill boundary: state flips only when a signal lands."""

    def __init__(self, labels, dies_on_term=True, cgroup_parent=None):
        self.labels, self.dies_on_term = labels, dies_on_term
        self.cgroup_parent = cgroup_parent or labels["prismabuild.scope"]
        self.running = True
        self.commands = []

    def __call__(self, argv, **kwargs):
        self.commands.append(list(argv))
        if argv[:2] == ["docker", "inspect"]:
            payload = [dict(Id=CID, Config=dict(Labels=dict(self.labels)),
                            HostConfig=dict(CgroupParent=self.cgroup_parent),
                            State=dict(Running=self.running, Pid=4242))]
            return SimpleNamespace(stdout=json.dumps(payload))
        if argv[:2] == ["docker", "kill"]:
            if argv[argv.index("--signal") + 1] == "KILL" or self.dies_on_term:
                self.running = False
            return SimpleNamespace(stdout="", returncode=0)
        raise AssertionError(argv)


def server_adapter(tmp_path, monkeypatch, stub, clock=None):
    import rank_window
    adapter = new_adapter(tmp_path, identity(0), 0)
    adapter.command = stub
    if clock is not None:
        monkeypatch.setattr(rank_window, "time", clock)
    return adapter


@pytest.mark.parametrize("field,value", [("scope", "foreign.slice"), ("action", "foreign-owner"),
                                         ("run", "run-" + "f" * 5), ("attempt", "f" * 32),
                                         ("parent", "foreign.slice")])
def test_server_termination_refuses_anything_outside_the_exact_attempt(tmp_path, monkeypatch, field, value):
    import rank_window
    owned = identity(0)
    labels = owned_labels(owned)
    parent = owned["scope_id"]
    if field == "scope": labels["prismabuild.scope"] = value
    if field == "action": labels["prismabuild.action"] = value
    if field == "run": labels["org.prismaquant.graph-window"] = value
    if field == "attempt": labels["org.prismaquant.attempt"] = value
    if field == "parent": parent = value
    stub = DockerStub(labels, cgroup_parent=parent)
    adapter = server_adapter(tmp_path, monkeypatch, stub)
    with pytest.raises(window.Refused, match="exact PB attempt"):
        adapter._term_server(CID)
    with pytest.raises(window.Refused, match="exact PB attempt"):
        adapter._stop_server(CID)
    assert not any(argv[:2] == ["docker", "kill"] for argv in stub.commands), (
        "a foreign container is never signalled")
    assert not (adapter.rdv / "container-termination-rank0.json").exists()


def test_stop_server_lets_a_cooperative_container_exit_on_term_without_kill(tmp_path, monkeypatch):
    import rank_window
    stub = DockerStub(owned_labels(identity(0)), dies_on_term=True)
    adapter = server_adapter(tmp_path, monkeypatch, stub)
    record = adapter._stop_server(CID)
    assert signal_names(record) == ["SIGTERM"], "a container that exits on TERM is never killed"
    assert record["deadline_shortened_grace"] is False and record["ended_unix"] > 0
    kills = [argv for argv in stub.commands if argv[:2] == ["docker", "kill"]]
    assert kills == [["docker", "kill", "--signal", "TERM", CID]]
    published = window.read_json(adapter.rdv / "container-termination-rank0.json")
    assert published[CID]["cid"] == CID and published[CID]["scope_id"] == adapter.identity["scope_id"]
    assert adapter._term_server(CID) is record, "the stop is recorded exactly once per container"
    assert len([argv for argv in stub.commands if argv[:2] == ["docker", "kill"]]) == 1


def test_stop_server_kills_a_stubborn_container_only_after_the_full_grace(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    stub = DockerStub(owned_labels(identity(0)), dies_on_term=False)
    adapter = server_adapter(tmp_path, monkeypatch, stub, clock=clock)
    record = adapter._stop_server(CID)
    events = record["signals"]
    assert [event["signal"] for event in events] == ["SIGTERM", "SIGKILL"]
    assert events[1]["monotonic"] - events[0]["monotonic"] == pytest.approx(10.0), (
        "the container SIGKILL waits the full ten-second grace")
    assert record["deadline_shortened_grace"] is False
    kills = [argv for argv in stub.commands if argv[:2] == ["docker", "kill"]]
    assert [argv[argv.index("--signal") + 1] for argv in kills] == ["TERM", "KILL"]


# --- the physical rehearsal: real Docker, admitted PB scope only -------------

def _published_pb_scope() -> bool:
    return all(os.environ.get(name) for name in ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
                                                 "PRISMABUILD_ACTION_SCOPE", "PRISMABUILD_CONTAINER_OWNER"))


PHYSICAL_IMAGES = ("debian:12", "ubuntu:24.04", "busybox:1.36", "alpine:3.20")


@pytest.mark.skipif(not _published_pb_scope(),
                    reason="physical container TERM→SIGKILL rehearsal requires the published PB scope")
def test_owned_container_grace_under_published_pb_scope(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("physical rehearsal needs the docker CLI on the PB host")
    image = next((name for name in PHYSICAL_IMAGES
                  if subprocess.run(["docker", "image", "inspect", name],
                                    capture_output=True).returncode == 0), None)
    if image is None:
        pytest.skip("physical rehearsal needs a pre-pulled local container image")
    import rank_window
    nonce = os.environ["PRISMABUILD_ACTION_NONCE"]
    owned = dict(rank=0, action_key=os.environ["PRISMABUILD_ACTION_KEY"], nonce=nonce,
                 scope_id=os.environ["PRISMABUILD_ACTION_SCOPE"], host=socket.gethostname(),
                 container_owner=os.environ["PRISMABUILD_CONTAINER_OWNER"],
                 run_id="d30-guard-" + nonce, claimed_unix=time.time(),
                 window_end_unix=time.time() + 900)
    adapter = new_adapter(tmp_path, owned, 0)
    adapter.command = adapter.envelope.run
    labels = []
    for key, value in owned_labels(owned).items():
        labels += ["--label", f"{key}={value}"]

    cooperative = stubborn = None
    try:
        def launch(name, entrypoint):
            argv = ["docker", "run", "-d", "--network", "none", "--name", f"d30-guard-{nonce[:8]}-{name}",
                    "--cgroup-parent", owned["scope_id"], *labels, image, "sh", "-c", entrypoint]
            cid = adapter.command(argv, limit=60).stdout.strip()
            assert cid, "the rehearsal container must start with its exact cid"
            return cid

        cooperative = launch("coop", 'trap "exit 0" TERM; echo ready; while :; do sleep 1; done')
        stubborn = launch("stub", 'trap "" TERM; echo ready; sleep 600')
        for cid in (cooperative, stubborn):
            until = time.monotonic() + 5
            while "ready" not in adapter.command(["docker", "logs", cid]).stdout:
                assert time.monotonic() < until, "container signal handler did not become ready"
                time.sleep(.05)
        started = time.monotonic()
        record = adapter._stop_server(cooperative)
        assert time.monotonic() - started < 10, "the cooperative container exits inside the grace"
        assert signal_names(record) == ["SIGTERM"]
        started = time.monotonic()
        record = adapter._stop_server(stubborn)
        assert time.monotonic() - started >= 10, "the stubborn container gets the full grace"
        assert signal_names(record) == ["SIGTERM", "SIGKILL"]
        for cid in (cooperative, stubborn):
            assert adapter.inspect_owned(cid)["State"]["Running"] is False
            adapter.command(["docker", "rm", cid], cleanup=True, limit=10)
            assert adapter.command(["docker", "inspect", cid], check=False, cleanup=True,
                                   limit=10).returncode != 0, "both containers are physically gone"
        published = window.read_json(adapter.rdv / "container-termination-rank0.json")
        assert cooperative in published and stubborn in published
    finally:
        for cid in (cooperative, stubborn):
            if cid:
                adapter.command(["docker", "rm", "-f", cid], check=False, cleanup=True, limit=20)
