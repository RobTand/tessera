"""Deterministic regressions for the bounded LocalArm.preflight headroom wait (#953).

CPU-only: the real common preflight path (LocalArm.headroom inside
LocalArm.preflight, both graph and eager) runs against one shared FakeClock
patched into both ``rank_window.time`` and ``managed_window.time``, so the
real Envelope, Refused, tick guard/floor machinery and the rendezvous report
file all move on the same deterministic clock. The subprocess, frozen-source
comparison and eager cgroup-counter boundaries are substituted. Real admitted
cgroup sampling is covered separately by test_graph_attest_eager_window.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import managed_window as window
import tp2_recipe as recipe

REFUSAL = "local MemAvailable below unchanged 114 GiB preflight"
FLOOR = "local 16 GiB physical memory floor breached"
REPORT = "headroom-preflight-rank{rank}.jsonl"
POLL = 0.2


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
        assert seconds >= 0, "the wait bound must never compute a negative sleep"
        self.sleeps.append(seconds)
        self.monotonic_value += seconds
        self.unix_value += seconds

    def advance(self, seconds):
        self.monotonic_value += seconds
        self.unix_value += seconds


def memory_trace(clock, schedule):
    """available_gib keyed by fake elapsed time: tick and wait read the same
    value inside one poll, exactly as one real /proc/meminfo poll would."""
    def read():
        value = schedule[0][1]
        for moment, gib in schedule:
            if clock.monotonic_value >= moment:
                value = gib
        return value
    return read




def identity(rank=0, **fields):
    key = str(rank + 1) * 64
    return dict(rank=rank, host="this-box", scope_id="owned.slice",
                container_owner="owned-parent", run_id="run-" + "a" * 8,
                nonce="b" * 32, action_key=key,
                window_end_unix=1_800_000_000.0 + 10, **fields)


def rank_adapter(tmp_path, monkeypatch, *, rank=0, read, clock, eager=False,
                 window_end_seconds=10_000, cleanup_seconds=1.0):
    import rank_window
    monkeypatch.setattr(rank_window, "available_gib", read)
    monkeypatch.setattr(rank_window, "time", clock)
    monkeypatch.setattr(window, "time", clock)
    owned = identity(rank)
    if eager:
        monkeypatch.setattr(rank_window, "scope_memory",
                            lambda identity: dict(rank=identity["rank"],
                                                  scope_id=identity["scope_id"]))
    adapter = rank_window.LocalArm.__new__(rank_window.LocalArm)
    adapter.identity, adapter.rank = owned, rank
    adapter.config = dict(window_mode=recipe.EAGER_MODE if eager else "graph-control")
    adapter.envelope = window.Envelope(clock.time() + window_end_seconds,
                                       cleanup_seconds=cleanup_seconds)
    adapter.work, adapter.rdv = tmp_path / "work", tmp_path / "rdv"
    (adapter.work / "ext").mkdir(parents=True)
    adapter.rdv.mkdir()
    adapter.ext = adapter.work / "ext"
    adapter.active, adapter.containers, adapter.last_sample = None, [], 0
    adapter.image_env = {}
    adapter.guard_calls = []
    adapter.guard = lambda: adapter.guard_calls.append("check")
    adapter.poll_seconds = POLL
    return adapter


def stub_boundaries(adapter, monkeypatch):
    """Substitutes only the subprocess and frozen-source boundaries; the wait,
    guard, floor and envelope machinery stay real."""
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


def forbid_commands(adapter):
    def command(argv, **kwargs):
        raise AssertionError("a failed wait must start no image or model command")
    adapter.command = command


def reports(adapter):
    path = adapter.rdv / REPORT.format(rank=adapter.rank)
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("eager", [False, True], ids=["graph", "eager"])
def test_below_threshold_then_ready_passes_after_waiting(tmp_path, monkeypatch, rank, eager):
    import rank_window
    clock = FakeClock()
    read = memory_trace(clock, [(100.0, 113.9), (100.2, 113.95), (100.4, 114.25)])
    adapter = rank_adapter(tmp_path, monkeypatch, rank=rank, read=read, clock=clock,
                           eager=eager)
    calls = stub_boundaries(adapter, monkeypatch)
    metadata = adapter.preflight()
    assert metadata["image"] == adapter.config["image"]
    assert calls, "a ready preflight continues into the ordinary image/source checks"
    (report,) = reports(adapter)
    assert report["threshold_gib"] == 114 and report["wait_bound_seconds"] == 900.0
    assert report["reason"] == "ready"
    assert report["initial_available_gib"] == 113.9
    assert report["last_available_gib"] == 114.25
    assert report["elapsed_seconds"] == pytest.approx(sum(clock.sleeps))
    assert [s["available_gib"] for s in report["samples"]] == [113.9, 113.95, 114.25]
    assert all(set(s) == {"unix", "monotonic", "available_gib"} for s in report["samples"])
    assert clock.sleeps == [POLL, POLL]
    assert len(adapter.guard_calls) >= 3
    if eager:
        scope_lines = (adapter.work / "memory-scope.jsonl").read_text().splitlines()
        assert scope_lines, "the eager tick sampler really executes and records"
        assert json.loads(scope_lines[0])["rank"] == rank
    # The report file is shared and appended, one terminal record per call.
    adapter.preflight()
    assert len(reports(adapter)) == 2
    assert reports(adapter)[1]["reason"] == "ready"


def test_never_reaching_threshold_times_out_at_900_and_refuses_without_a_model(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0,
                           read=memory_trace(clock, [(100.0, 100.0)]), clock=clock)
    forbid_commands(adapter)
    with pytest.raises(window.Refused) as excinfo:
        adapter.preflight()
    assert str(excinfo.value) == REFUSAL
    (report,) = reports(adapter)
    assert report["reason"] == "headroom_timeout"
    assert report["threshold_gib"] == 114 and report["wait_bound_seconds"] == 900.0
    assert report["initial_available_gib"] == 100.0 and report["last_available_gib"] == 100.0
    assert report["elapsed_seconds"] == pytest.approx(900.0)
    assert clock.monotonic_value == pytest.approx(100.0 + 900.0)
    assert report["samples"][-1]["monotonic"] == pytest.approx(clock.monotonic())
    assert len(adapter.guard_calls) >= 1, "the live guard is checked during the whole wait"


def test_cleanup_reserve_shortens_the_wait_deadline(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0,
                           read=memory_trace(clock, [(100.0, 100.0)]), clock=clock,
                           window_end_seconds=600, cleanup_seconds=300.0)
    forbid_commands(adapter)
    with pytest.raises(TimeoutError) as excinfo:
        adapter.preflight()
    assert "cleanup reserve begins" in str(excinfo.value)
    (report,) = reports(adapter)
    assert report["reason"] == "deadline"
    assert report["elapsed_seconds"] == pytest.approx(300.0)
    assert report["samples"] and report["last_available_gib"] == 100.0


def test_tightened_peer_window_shortens_the_wait_deadline(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0,
                           read=memory_trace(clock, [(100.0, 100.0)]), clock=clock)
    adapter.envelope.tighten(clock.time() + 9)
    forbid_commands(adapter)
    with pytest.raises(TimeoutError):
        adapter.preflight()
    (report,) = reports(adapter)
    assert report["reason"] == "deadline"
    assert report["elapsed_seconds"] == pytest.approx(8.0)


def test_already_expired_envelope_never_reports_ready_even_with_headroom(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0,
                           read=memory_trace(clock, [(100.0, 200.0)]), clock=clock,
                           window_end_seconds=-5)
    forbid_commands(adapter)
    with pytest.raises(TimeoutError):
        adapter.preflight()
    (report,) = reports(adapter)
    assert report["reason"] == "deadline"
    assert report["initial_available_gib"] == 200.0
    assert report["elapsed_seconds"] == pytest.approx(0.0)


def test_guard_peer_cancellation_propagates_and_persists_the_report(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=1,
                           read=memory_trace(clock, [(100.0, 100.0)]), clock=clock)
    cancelled = window.Refused("peer rank published lifecycle-cancelled")
    def guard():
        adapter.guard_calls.append("check")
        if len(adapter.guard_calls) == 2:
            raise cancelled
    adapter.guard = guard
    with pytest.raises(window.Refused) as excinfo:
        adapter.preflight()
    assert excinfo.value is cancelled
    (report,) = reports(adapter)
    assert report["reason"] == "lifecycle_cancelled"
    assert len(report["samples"]) == 2
    assert report["last_available_gib"] == 100.0


def test_sixteen_gib_floor_breach_during_the_wait_refuses_and_persists(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    # Real tick gates its memory sample on a five-second cadence; the drop to
    # 15 GiB lands exactly on the second gated sample at fake 10 s.
    read = memory_trace(clock, [(100.0, 113.0), (110.0, 15.0)])
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0, read=read, clock=clock)
    with pytest.raises(window.Refused) as excinfo:
        adapter.preflight()
    assert str(excinfo.value) == FLOOR
    (report,) = reports(adapter)
    assert report["reason"] == "lifecycle_cancelled"
    assert report["initial_available_gib"] == 113.0
    assert report["last_available_gib"] == 15.0


def test_guard_latency_past_900_reports_headroom_timeout_without_negative_sleep(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0,
                           read=memory_trace(clock, [(100.0, 100.0)]), clock=clock)
    def slow_guard():
        adapter.guard_calls.append("slow")
        clock.advance(1000.0)  # the live guard/tick may consume far more than one poll
    adapter.guard = slow_guard
    with pytest.raises(window.Refused) as excinfo:
        adapter.preflight()
    assert str(excinfo.value) == REFUSAL
    (report,) = reports(adapter)
    assert report["reason"] == "headroom_timeout"
    assert report["elapsed_seconds"] == pytest.approx(1000.0)
    assert clock.sleeps == [], "no sleep is attempted once the bound has passed"


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("eager", [False, True], ids=["graph", "eager"])
def test_launch_rechecks_headroom_after_successful_preflight(tmp_path, monkeypatch, rank, eager):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=rank, read=lambda: 114.25,
                           clock=clock, eager=eager)
    stub_boundaries(adapter, monkeypatch)
    assert adapter.preflight()["image"] == adapter.config["image"]
    monkeypatch.setattr(rank_window, "available_gib", lambda: 113.9)
    launches = []
    def forbidden_container(*args, **kwargs):
        launches.append(True)
        raise AssertionError("model launch reached below 114 GiB after a ready preflight")
    monkeypatch.setattr(rank_window.recipe, "container", forbidden_container)
    with pytest.raises(window.Refused, match=REFUSAL):
        adapter.start(dict(arm="launch-check"))
    assert not launches
    assert not (adapter.work / "launch-check").exists()
    sample = json.loads((adapter.rdv / f"launch-check-launch-headroom-rank{rank}.json").read_text())
    assert sample["mem_available_gib"] == 113.9 and sample["threshold_gib"] == 114
    assert sample["rank"] == rank


def test_headroom_read_error_retains_unknown_values_and_propagates(tmp_path, monkeypatch):
    clock = FakeClock()
    failure = OSError("meminfo reading unavailable")
    def unreadable():
        raise failure
    adapter = rank_adapter(tmp_path, monkeypatch, read=unreadable, clock=clock)
    forbid_commands(adapter)
    with pytest.raises(OSError) as excinfo:
        adapter.preflight()
    assert excinfo.value is failure
    (report,) = reports(adapter)
    assert report["reason"] == "error"
    assert report["samples"] == []
    assert report["initial_available_gib"] is None and report["last_available_gib"] is None
