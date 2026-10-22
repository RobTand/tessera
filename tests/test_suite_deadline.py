"""``tessera._dev.suite_deadline``: the per-attempt deadline never waits without a bound.

After SIGKILL the runner waits for the group leader.  A leader that cannot be
reaped (a D-state process on a wedged GPU is the case this tree has met) made
that wait unbounded, so the deadline that existed to bound a run had no bound
itself (tessera#154).  The grace-period no-poll between SIGTERM and SIGKILL is
deliberate and untouched.
"""

import signal
import subprocess

import pytest

from tessera._dev import suite_deadline


@pytest.mark.parametrize("failed_signal", [signal.SIGINT, signal.SIGCHLD])
def test_partial_handler_installation_restores_previous_dispositions(monkeypatch, failed_signal):
    """A later installer failure must roll back an already-installed SIGTERM.

    Exercise real dispositions with a one-shot failure, but always repair the
    worker's signal state even when the regression assertion fails. No child
    process or GPU work is started.
    """
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGCHLD)
    original = {sig: signal.getsignal(sig) for sig in signals}
    real_signal = signal.signal
    sentinels = {sig: (lambda signum, frame: None) for sig in signals}
    error = RuntimeError("injected handler installation failure")
    failed = False
    launches = []

    def install(sig, handler):
        nonlocal failed
        if sig == failed_signal and not failed:
            failed = True
            raise error
        return real_signal(sig, handler)

    def unexpected_launch(*args, **kwargs):
        launches.append((args, kwargs))
        pytest.fail("deadline launched a child after incomplete signal setup")

    try:
        for sig, handler in sentinels.items():
            real_signal(sig, handler)
        monkeypatch.setattr(signal, "signal", install)
        monkeypatch.setattr(suite_deadline.subprocess, "Popen", unexpected_launch)
        with pytest.raises(RuntimeError) as raised:
            suite_deadline.run(["unused-inert-child"], 1, 0.1)
        assert raised.value is error
        assert not launches
        for sig, previous in sentinels.items():
            assert signal.getsignal(sig) is previous, (
                f"{sig.name} handler leaked after {failed_signal.name} installation failed")
    finally:
        for sig, previous in original.items():
            real_signal(sig, previous)



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
