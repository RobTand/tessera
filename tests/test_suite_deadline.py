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


_HANDLER_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGCHLD)


def _handlers():
    return {sig: signal.getsignal(sig) for sig in _HANDLER_SIGNALS}


@pytest.fixture(params=["custom", "ignored"])
def caller_handlers(request):
    # Use the real setter even if a fault-injection patch is still active when
    # an assertion fails. Never leak the regression's signal state into pytest.
    set_handler = signal.signal
    original = _handlers()

    def custom_handler():
        def handler(signum, frame):
            pass
        return handler

    previous = {sig: custom_handler() if request.param == "custom" else signal.SIG_IGN
                for sig in _HANDLER_SIGNALS}
    try:
        for sig, handler in previous.items():
            set_handler(sig, handler)
        yield previous
    finally:
        for sig, handler in original.items():
            set_handler(sig, handler)


def _check_completed_lifecycles(monkeypatch, previous):
    # These paths already restore correctly. Exercise them in-process before
    # the failing partial-setup regression, so the fix cannot regress them.
    for outcome in ("success", "launch_failure", "timeout_term", "timeout_kill",
                    signal.SIGTERM, signal.SIGINT):
        groups, sleeps = [], []

        class Process:
            pid = 4242
            returncode = None

            def __init__(self):
                self.waits = []

            def wait(self, timeout=None):
                self.waits.append(timeout)
                if len(self.waits) == 1:
                    if outcome in ("timeout_term", "timeout_kill"):
                        raise subprocess.TimeoutExpired("leader", timeout)
                    if outcome in (signal.SIGTERM, signal.SIGINT):
                        signal.getsignal(outcome)(outcome, None)
                self.returncode = (-signal.SIGKILL if outcome == "timeout_kill"
                                   or isinstance(outcome, signal.Signals) else 7)
                return self.returncode

        process = Process()
        launch_error = OSError("injected launch failure")

        def popen(command, *, start_new_session):
            assert start_new_session is True
            assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
            assert all(callable(signal.getsignal(sig)) and
                       signal.getsignal(sig) is not previous[sig]
                       for sig in (signal.SIGTERM, signal.SIGINT))
            if outcome == "launch_failure":
                raise launch_error
            return process

        with monkeypatch.context() as patch:
            patch.setattr(suite_deadline.subprocess, "Popen", popen)
            patch.setattr(suite_deadline.os, "killpg", lambda *args: groups.append(args))
            patch.setattr(suite_deadline.time, "sleep", sleeps.append)
            if outcome == "launch_failure":
                with pytest.raises(OSError) as exc:
                    suite_deadline.run(["leader"], 2.0, 0.25)
                assert exc.value is launch_error
            else:
                expected = {"success": 7, "timeout_term": 124, "timeout_kill": 137,
                            signal.SIGTERM: 143, signal.SIGINT: 130}[outcome]
                assert suite_deadline.run(["leader"], 2.0, 0.25) == expected
        assert _handlers() == previous, outcome
        if outcome in ("timeout_term", "timeout_kill"):
            assert groups == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
            assert sleeps == [0.25]
            assert process.waits == [2.0, 0.25]
        elif outcome in (signal.SIGTERM, signal.SIGINT):
            assert groups == [(process.pid, signal.SIGKILL)]
            assert process.waits == [2.0, 0.25]
        else:
            assert groups == []


@pytest.mark.parametrize("setup_signal,after_mutation", [
    pytest.param(signal.SIGTERM, True, id="term-after"),
    pytest.param(signal.SIGINT, False, id="int-before"),
    pytest.param(signal.SIGINT, True, id="int-after"),
    pytest.param(signal.SIGCHLD, False, id="chld-before"),
    pytest.param(signal.SIGCHLD, True, id="chld-after"),
])
@pytest.mark.parametrize("interrupt", [None, signal.SIGTERM, signal.SIGINT],
                         ids=["exception", "sigterm", "sigint"])
def test_run_restores_exact_handlers_through_lifecycle_and_failed_setup(
        monkeypatch, caller_handlers, setup_signal, after_mutation, interrupt):
    _check_completed_lifecycles(monkeypatch, caller_handlers)
    set_handler = signal.signal
    failure = RuntimeError("injected handler installation failure")
    injected, launches = [], []

    def install(sig, handler):
        if sig != setup_signal or injected:
            return set_handler(sig, handler)
        injected.append(sig)  # One-shot: restoration must be able to succeed.
        if after_mutation:
            set_handler(sig, handler)
        if interrupt is not None:
            # Deterministically invoke the installed handler, without a signal
            # delivery race. TERM is installed by every injection point here.
            signal.getsignal(signal.SIGTERM)(interrupt, None)
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(suite_deadline.signal, "signal", install)
        patch.setattr(suite_deadline.subprocess, "Popen", lambda *a, **kw: launches.append(a))
        result, error = None, None
        try:
            result = suite_deadline.run(["leader"], 2.0, 0.25)
        except BaseException as exc:
            error = exc
        assert _handlers() == caller_handlers, "partial setup leaked a caller's signal handler"
        assert injected == [setup_signal]
        assert launches == [], "interrupted/failed setup must not launch a child"
        if interrupt is None:
            assert error is failure
        else:
            assert error is None
            assert result == 128 + interrupt
