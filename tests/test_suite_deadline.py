"""``tessera._dev.suite_deadline``: the per-attempt deadline never waits without a bound.

After SIGKILL the runner waits for the group leader.  A leader that cannot be
reaped (a D-state process on a wedged GPU is the case this tree has met) made
that wait unbounded, so the deadline that existed to bound a run had no bound
itself (tessera#154).  The grace-period no-poll between SIGTERM and SIGKILL is
deliberate and untouched.
"""

import subprocess

from tessera._dev import suite_deadline


class _Unreapable:
    pid = 4242
    returncode = None

    def __init__(self):
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        raise subprocess.TimeoutExpired("leader", timeout)


def test_the_final_wait_after_sigkill_is_bounded_by_the_grace(capsys):
    process = _Unreapable()
    assert suite_deadline._wait_bounded(process, 2.5, "deadline") is None
    assert process.waits == [2.5]
    assert "not reaped" in capsys.readouterr().err


def test_an_interrupt_kill_waits_no_longer_than_the_grace(capsys, monkeypatch):
    process = _Unreapable()
    monkeypatch.setattr(suite_deadline, "_signal_group", lambda pid, signum: None)
    suite_deadline._kill_owned(process, 0.5)
    assert process.waits == [0.5]
    assert "not reaped" in capsys.readouterr().err
