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


_HANDLER_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGCHLD)


def _handlers():
    return {sig: signal.getsignal(sig) for sig in _HANDLER_SIGNALS}


@pytest.fixture(params=["custom", "ignored"])
def caller_handlers(request):
    # Keep the real setter: teardown must repair pytest's state even if a
    # fault-injection patch or an assertion is still active.
    set_handler = signal.signal
    original = _handlers()
    previous = {sig: (lambda signum, frame: None) if request.param == "custom"
                else signal.SIG_IGN for sig in _HANDLER_SIGNALS}
    try:
        for sig, handler in previous.items():
            set_handler(sig, handler)
        yield previous
    finally:
        for sig, handler in original.items():
            set_handler(sig, handler)


def _check_completed_lifecycles(monkeypatch, previous):
    # These paths already restore correctly. Check them before the failing
    # setup regression so fixing setup cannot silently break body cleanup.
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
                        signal.raise_signal(outcome)
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


@pytest.mark.parametrize("setup_signal", _HANDLER_SIGNALS, ids=lambda sig: sig.name)
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt],
                         ids=["exception", "keyboard-interrupt"])
def test_setup_mutation_before_setter_returns_restores_exact_handlers(
        monkeypatch, caller_handlers, setup_signal, failure_type):
    _check_completed_lifecycles(monkeypatch, caller_handlers)
    set_handler = signal.signal
    failure = failure_type("injected after signal disposition changed")
    injected, launches = [], []

    def install(sig, handler):
        previous = set_handler(sig, handler)
        if sig == setup_signal and not injected:
            injected.append(sig)  # One shot: restoration can succeed.
            raise failure
        return previous

    with monkeypatch.context() as patch:
        patch.setattr(suite_deadline.signal, "signal", install)
        patch.setattr(suite_deadline.subprocess, "Popen", lambda *a, **kw: launches.append(a))
        error = None
        try:
            suite_deadline.run(["leader"], 2.0, 0.25)
        except BaseException as exc:
            error = exc
        assert injected == [setup_signal]
        assert launches == [], "failed setup must not launch a child"
        assert _handlers() == caller_handlers, "partial setup leaked a caller's signal handler"
        assert error is failure, "the original setup failure must propagate unchanged"


@pytest.mark.parametrize("setup_signal,after_mutation,interrupt", [
    pytest.param(signal.SIGTERM, True, signal.SIGTERM, id="term-after-term"),
    pytest.param(signal.SIGINT, False, signal.SIGTERM, id="int-before-term"),
    pytest.param(signal.SIGINT, True, signal.SIGTERM, id="int-after-term"),
    pytest.param(signal.SIGCHLD, False, signal.SIGTERM, id="chld-before-term"),
    pytest.param(signal.SIGCHLD, True, signal.SIGTERM, id="chld-after-term"),
    pytest.param(signal.SIGINT, True, signal.SIGINT, id="int-after-int"),
    pytest.param(signal.SIGCHLD, True, signal.SIGINT, id="chld-after-int"),
])
def test_setup_signal_interruption_restores_handlers_and_returns_signal_status(
        monkeypatch, caller_handlers, setup_signal, after_mutation, interrupt):
    _check_completed_lifecycles(monkeypatch, caller_handlers)
    set_handler = signal.signal
    injected, launches = [], []

    def install(sig, handler):
        if sig != setup_signal or injected:
            return set_handler(sig, handler)
        injected.append(sig)
        if after_mutation:
            set_handler(sig, handler)
        # Deliver only a signal whose deadline handler is already installed.
        # This runs synchronously in pytest's main thread, with no timer race.
        signal.raise_signal(interrupt)
        pytest.fail("the installed deadline interruption handler did not run")

    with monkeypatch.context() as patch:
        patch.setattr(suite_deadline.signal, "signal", install)
        patch.setattr(suite_deadline.subprocess, "Popen", lambda *a, **kw: launches.append(a))
        result, error = None, None
        try:
            result = suite_deadline.run(["leader"], 2.0, 0.25)
        except BaseException as exc:
            error = exc
        assert injected == [setup_signal]
        assert launches == [], "interrupted setup must not launch a child"
        assert _handlers() == caller_handlers, "interrupted setup leaked a caller's signal handler"
        assert error is None, f"setup interruption escaped instead of returning status: {error!r}"
        assert result == 128 + interrupt


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
