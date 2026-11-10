"""Live Envelope coverage ported from the archived managed-window tests.

Clocks drive deadlines. Real children prove that cleanup reaches the owned group.
"""
import importlib.util
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from managed_window_support import FakeClock, alive

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('live_envelope', ROOT / 'experiments/graph_attest_702/managed_window.py')
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)


def test_one_envelope_cannot_reset_for_later_arms(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(window, 'time', clock)
    envelope = window.Envelope(clock.time() + 20, cleanup_seconds=3)
    end = envelope.end
    envelope.tighten(clock.time() + 5400)
    assert envelope.end == end
    clock.advance(18)
    with pytest.raises(TimeoutError, match='cleanup reserve'):
        envelope.remaining()
    assert envelope.remaining(cleanup=True) == 2
    clock.advance(2)
    with pytest.raises(TimeoutError, match='whole graph window expired'):
        envelope.remaining(cleanup=True)


def test_tighten_can_only_shorten_the_absolute_deadline(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(window, 'time', clock)
    envelope = window.Envelope(clock.time() + 30, cleanup_seconds=3)
    clock.advance(4)
    envelope.tighten(clock.time() + 10)
    assert envelope.remaining() == 7
    assert envelope.remaining(cleanup=True) == 10
    clock.advance(2)
    envelope.tighten(clock.time() + 100)
    assert envelope.remaining() == 5


def test_wall_clock_correction_cannot_extend_the_monotonic_deadline(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(window, 'time', clock)
    envelope = window.Envelope(clock.time() + 30, cleanup_seconds=3)
    earlier_peer_end = envelope.end_unix - 1
    clock.advance(4)
    clock.unix_value -= 60
    envelope.tighten(earlier_peer_end)
    assert envelope.remaining() == 23
    assert envelope.remaining(cleanup=True) == 26


@pytest.mark.parametrize('value', [0, -1, True, float('inf'), float('nan')])
def test_deadlines_require_finite_positive_times(value):
    with pytest.raises(window.Refused):
        window.Envelope(value)


def test_deadline_kills_real_cpu_subprocess_group_and_retains_output(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(window, 'time', clock)
    output = tmp_path / 'partial.log'
    ready = tmp_path / 'ready'
    envelope = window.Envelope(clock.time() + 30, cleanup_seconds=3)
    failure_bound = time.monotonic() + 20
    def tick():
        if ready.exists():
            clock.advance(28)
        elif time.monotonic() > failure_bound:
            raise AssertionError('owned group never became ready')
    child = "import time; print('ready', flush=True); time.sleep(120)"
    parent = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[2]],stdout=subprocess.PIPE,text=True); "
        "assert p.stdout.readline().strip()=='ready'; "
        "print(p.pid,flush=True); pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(120)"
    )
    try:
        with output.open('w') as stream, pytest.raises(TimeoutError):
            envelope.run([sys.executable, '-c', parent, str(ready), child],
                         stdout=stream, tick=tick, limit=30)
        pid = int(output.read_text().strip())
        until = time.monotonic() + 10
        while alive(pid) and time.monotonic() < until:
            time.sleep(.01)
        assert not alive(pid)
        assert envelope.terminations[-1]['signals'][0]['signal'] == 'SIGTERM'
    finally:
        for record in envelope.terminations:
            try:
                os.killpg(record['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
