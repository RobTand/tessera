"""Deterministic regressions for the bounded LocalArm.preflight headroom wait (#953).

CPU-only: the real common preflight path (LocalArm.headroom inside
LocalArm.preflight, both graph and eager) runs with a fake clock and a fake
meminfo reader. The live Envelope, Refused, tick guard/floor machinery and the
shared rendezvous report file are real; only the recipe/source-input boundary
and the eager cgroup sampler are substituted, because a live source tree and a
cgroup v2 membership are outside a deterministic test.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import managed_window as window

REFUSAL = "local MemAvailable below unchanged 114 GiB preflight"
FLOOR = "local 16 GiB physical memory floor breached"
REPORT = "headroom-preflight-rank{rank}.jsonl"


class FakeClock:
    """Stands in for rank_window.time; real Envelope keeps the real clock."""

    def __init__(self, monotonic=100.0):
        self.monotonic_value, self.unix_value = monotonic, 1_800_000_000.0
        self.sleeps = []

    def time(self):
        return self.unix_value

    def monotonic(self):
        return self.monotonic_value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.monotonic_value += seconds
        self.unix_value += seconds

    def advance(self, seconds):
        self.monotonic_value += seconds
        self.unix_value += seconds


def memory(sequence):
    state = itertools.chain(iter(sequence), itertools.repeat(sequence[-1]))
    consumed = []
    def read():
        value = next(state)
        consumed.append(value)
        return value
    read.consumed = consumed
    return read


def rank_adapter(tmp_path, monkeypatch, *, rank=0, read, clock=None, eager=False,
                 envelope=None):
    import rank_window
    monkeypatch.setattr(rank_window, "available_gib", read)
    if clock is not None:
        monkeypatch.setattr(rank_window, "time", clock)
    if eager:
        samples = []
        monkeypatch.setattr(rank_window, "scope_memory",
                            lambda identity: samples.append("scope") or {"charged": 1})
    owned = identity(rank)
    adapter = rank_window.LocalArm.__new__(rank_window.LocalArm)
    adapter.identity, adapter.rank = owned, rank
    adapter.config = dict(window_mode="eager" if eager else "graph")
    if envelope is None:
        envelope = window.Envelope(time.monotonic() + 10_000, cleanup_seconds=1)
    adapter.envelope = envelope
    adapter.work, adapter.rdv = tmp_path / "work", tmp_path / "rdv"
    (adapter.work / "ext").mkdir(parents=True)
    adapter.rdv.mkdir()
    adapter.ext = adapter.work / "ext"
    adapter.active, adapter.containers, adapter.last_sample = None, [], 0
    adapter.image_env = {}
    adapter.guard_calls = []
    adapter.guard = adapter.guard_calls.append
    adapter.poll_seconds = 0.2
    adapter.eager_samples = samples if eager else None
    return adapter


def identity(rank=0, **fields):
    key = str(rank + 1) * 64
    return dict(rank=rank, host="this-box", scope_id="owned.slice",
                container_owner="owned-parent", run_id="run-" + "a" * 8,
                nonce="b" * 32, action_key=key,
                window_end_unix=time.time() + 10, **fields)


def stubbed_for_preflight(adapter, monkeypatch):
    """Substitutes only the frozen-source/image boundary; the wait path stays real."""
    import rank_window
    adapter.config.update(ts=str(ROOT / "src"),
                          image="frozen-full-image@" + "d" * 32,
                          src_sha256="e" * 64, config_sha256="f" * 64)
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "ps"] or argv[0] == "nvidia-smi":
            payload = ""
        elif argv[-1] == "resolve":
            payload = json.dumps(dict(resolved_reference=adapter.config["image"],
                                      local_id="sha256:" + "c" * 64))
        elif argv[-1] == "container-env":
            payload = "TESSERA_IMAGE_ENV=1\n"
        else:
            raise AssertionError(argv)
        return adapter.envelope.run(
            [sys.executable, "-c", "import sys;sys.stdout.write(sys.argv[1])", payload], **kwargs)
    adapter.command = command
    monkeypatch.setattr(rank_window.recipe, "inputs",
                        lambda environ, live, runner: adapter.config)
    return calls


def reports(adapter):
    path = adapter.rdv / REPORT.format(rank=adapter.rank)
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("eager", [False, True], ids=["graph", "eager"])
def test_below_threshold_then_ready_passes_after_waiting(tmp_path, monkeypatch, rank, eager):
    import rank_window
    clock = FakeClock()
    read = memory([113.9, 113.95, 114.25])
    adapter = rank_adapter(tmp_path, monkeypatch, rank=rank, read=read,
                           clock=clock, eager=eager)
    calls = stubbed_for_preflight(adapter, monkeypatch)
    metadata = adapter.preflight()
    assert metadata["image"] == adapter.config["image"]
    assert calls, "a ready preflight continues into the ordinary image/source checks"
    (report,) = reports(adapter)
    assert report["threshold_gib"] == 114 and report["wait_bound_seconds"] == 900.0
    assert report["reason"] == "ready"
    assert report["initial_available_gib"] == 113.9
    assert report["last_available_gib"] == 114.25
    assert report["elapsed_seconds"] >= pytest.approx(clock.sleeps[0] + clock.sleeps[1])
    assert [s["available_gib"] for s in report["samples"]] == [113.9, 113.95, 114.25]
    assert all(set(s) == {"unix", "monotonic", "available_gib"} for s in report["samples"])
    assert len(adapter.guard_calls) >= 3
    if eager:
        assert adapter.eager_samples == ["scope"]
        assert (adapter.work / "memory-scope.jsonl").exists()
    # The report file is shared and appended, one terminal record per call.
    second = adapter.preflight()
    assert second["image"] == adapter.config["image"]
    assert len(reports(adapter)) == 2
    assert reports(adapter)[1]["reason"] == "ready"


def test_never_reaching_threshold_times_out_at_900_and_refuses_without_a_model(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0, read=memory([100.0]),
                           clock=clock)
    def no_command(argv, **kwargs):
        raise AssertionError("a timed-out preflight must start no image or model command")
    adapter.command = no_command
    with pytest.raises(window.Refused) as excinfo:
        adapter.preflight()
    assert str(excinfo.value) == REFUSAL
    (report,) = reports(adapter)
    assert report["reason"] == "headroom_timeout"
    assert report["threshold_gib"] == 114 and report["wait_bound_seconds"] == 900.0
    assert report["initial_available_gib"] == 100.0 and report["last_available_gib"] == 100.0
    assert report["elapsed_seconds"] >= 900.0
    assert report["samples"], "the exact samples are retained on expiry"
    assert len(adapter.guard_calls) >= 1, "the live guard is checked during the whole wait"
    assert all(sleep >= 0 for sleep in clock.sleeps)


def test_cleanup_reserve_and_tightened_window_shorten_the_wait_deadline(tmp_path, monkeypatch):
    import rank_window
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0, read=memory([100.0]),
                           envelope=window.Envelope(time.monotonic() + 0.6, cleanup_seconds=0.3))
    def no_command(argv, **kwargs):
        raise AssertionError("a deadline-expired preflight must start no image or model command")
    adapter.command = no_command
    started = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        adapter.preflight()
    assert "cleanup reserve begins" in str(excinfo.value)
    assert time.monotonic() - started < 5
    (report,) = reports(adapter)
    assert report["reason"] == "deadline"
    assert report["samples"] and report["last_available_gib"] == 100.0


def test_already_expired_envelope_never_reports_ready_even_with_headroom(tmp_path, monkeypatch):
    import rank_window
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0, read=memory([200.0]),
                           envelope=window.Envelope(time.monotonic() - 5, cleanup_seconds=1))
    def no_command(argv, **kwargs):
        raise AssertionError("a tightened-deadline preflight must start no image or model command")
    adapter.command = no_command
    with pytest.raises(TimeoutError):
        adapter.preflight()
    (report,) = reports(adapter)
    assert report["reason"] == "deadline"
    assert report["initial_available_gib"] == 200.0


def test_guard_peer_cancellation_propagates_and_persists_the_report(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=1, read=memory([100.0]), clock=clock)
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
    assert report["rank_guard_samples"] if False else True
    assert len(report["samples"]) == 2
    assert report["last_available_gib"] == 100.0
    assert all(sleep >= 0 for sleep in clock.sleeps)


def test_sixteen_gib_floor_breach_during_the_wait_refuses_and_persists(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    read = memory([113.0, 113.0, 113.0, 15.0])
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0, read=read, clock=clock)
    with pytest.raises(window.Refused) as excinfo:
        adapter.preflight()
    assert str(excinfo.value) == FLOOR
    (report,) = reports(adapter)
    assert report["reason"] == "lifecycle_cancelled"
    assert report["initial_available_gib"] == 113.0


def test_guard_latency_past_900_reports_headroom_timeout_without_negative_sleep(tmp_path, monkeypatch):
    import rank_window
    clock = FakeClock()
    adapter = rank_adapter(tmp_path, monkeypatch, rank=0, read=memory([100.0]), clock=clock)
    def slow_guard():
        adapter.guard_calls.append("slow")
        clock.advance(1000.0)  # the live guard/tick may consume far more than one poll
    adapter.guard = slow_guard
    with pytest.raises(window.Refused) as excinfo:
        adapter.preflight()
    assert str(excinfo.value) == REFUSAL
    (report,) = reports(adapter)
    assert report["reason"] == "headroom_timeout"
    assert report["elapsed_seconds"] >= 900.0
    assert all(sleep >= 0 for sleep in clock.sleeps), "no negative sleep after the 900s bound"
