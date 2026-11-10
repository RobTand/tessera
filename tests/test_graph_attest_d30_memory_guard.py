"""D30 owned TERM->SIGKILL grace regressions (#959) for the real ``Envelope``.

CPU-only and deterministic: the real-process tests spawn actual children (a
TERM-ignoring process group must be SIGKILLed only after the full ten-second
grace, a cooperative child must exit within the grace without any SIGKILL, and a
finite cleanup deadline must shorten the grace); a fake clock drives the delayed
reap through the real ``Envelope.run`` / ``_terminate``.

The rank-window adapter, rendezvous and container-ownership cases of this file
belonged to the PrismaBuild-admitted window driver, now archived under
``archive/flux-migration/R-CONTESTED/``.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import managed_window as window


from managed_window_support import FakeClock, alive


def await_dead(pids, deadline):
    """SIGKILL teardown is asynchronous; death is awaited, never sampled once."""
    while time.monotonic() < deadline and any(alive(pid) for pid in pids):
        time.sleep(.05)




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
    child = int(pids.read_text())
    # SIGKILL is asynchronous. Keep the original elapsed bound for physical death.
    await_dead([child], started + 9.5)
    elapsed = time.monotonic() - started
    assert not alive(child)
    assert 7 <= elapsed <= 9.5, "termination respects the original eight-second envelope"
    (record,) = envelope.terminations
    assert signal_names(record) == ["SIGTERM", "SIGKILL"]
    assert record["deadline_shortened_grace"] is True


def test_delayed_kill_reap_preserves_deadline_and_records_cleanup(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(window, "time", clock)
    envelope = window.Envelope(clock.time() + 8, cleanup_seconds=2)
    signals, waits = [], []

    class DelayedProcess:
        pid = 4242
        returncode = None

        def communicate(self, *, input, timeout):
            clock.advance(timeout)
            raise subprocess.TimeoutExpired("delayed-child", timeout)

        def poll(self):
            return self.returncode

        def wait(self, *, timeout):
            waits.append(timeout)
            assert signals == [(self.pid, signal.SIGTERM), (self.pid, signal.SIGKILL)]
            clock.advance(timeout)
            assert .02 > timeout
            raise subprocess.TimeoutExpired("delayed-child", timeout)

    process = DelayedProcess()
    monkeypatch.setattr(window.subprocess, "Popen", lambda *args, **kwargs: process)

    def killpg(pid, sig):
        assert pid == process.pid, "cleanup must signal only the owned process group"
        if sig:
            signals.append((pid, sig))

    monkeypatch.setattr(window.os, "killpg", killpg)
    started = clock.monotonic()
    with pytest.raises(TimeoutError, match="subprocess deadline: delayed-child") as excinfo:
        envelope.run(["delayed-child"], limit=1, cleanup=True)
    assert type(excinfo.value) is TimeoutError
    assert waits == [.01], "the final reap has only 10 ms after the absolute deadline"
    assert clock.monotonic() - started == pytest.approx(8.01)
    (record,) = envelope.terminations
    assert record["pid"] == process.pid
    assert signal_names(record) == ["SIGTERM", "SIGKILL"]
    assert record["deadline_shortened_grace"] is True
    assert record["reap_timed_out"] is True
    assert record["reap_timeout_seconds"] == .01
    assert record["returncode"] is None
    assert record["ended_unix"] == clock.time()
