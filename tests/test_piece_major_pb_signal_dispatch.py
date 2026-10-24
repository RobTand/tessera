"""Adversarial dispatch ordering for the real subprocess regression.

This model queues the self-SIGTERM until the subject waits for its receipt pipe.
The named test must still use real OS delivery when run without this model.
A pipe/select receipt is the deliberately chosen synchronization contract: no
thread scheduling or CPython signal-dispatch latency decides this regression.
"""
from pathlib import Path
import runpy
import subprocess


TARGET = Path(__file__).with_name('test_piece_major_pb_action.py')

# Runs only in the subject's isolated child. The controller and its signal
# registration/restoration are real; only asynchronous signal dispatch is modeled.
DEFER_DISPATCH = r'''
import os as _dispatch_os
import select as _dispatch_select
import signal as _dispatch_signal

_dispatch_real_signal = _dispatch_signal.signal
_dispatch_real_select = _dispatch_select.select
_dispatch_registered = []
_dispatch_requests = []
_dispatch_deliveries = []
_dispatch_receipts = []

def _dispatch_register(signum, handler):
    previous = _dispatch_real_signal(signum, handler)
    if signum == _dispatch_signal.SIGTERM:
        _dispatch_registered.append(handler)
    return previous

def _dispatch_queue(pid, signum):
    assert pid == _dispatch_os.getpid(), 'subject must signal its isolated child'
    assert signum == _dispatch_signal.SIGTERM
    assert not _dispatch_requests, 'duplicate signal is not synchronization'
    _dispatch_requests.append(signum)
    # Sending is deliberately not an acknowledgement of Python-level dispatch.

def _dispatch_wait(readers, writers, exceptional, timeout=None):
    if _dispatch_requests and not _dispatch_deliveries:
        assert readers, 'wait must observe a receipt descriptor'
        assert timeout is not None and timeout > 0, 'wait needs a failure bound'
        assert not _dispatch_real_select(readers, [], [], 0)[0], (
            'receipt was readable before the previous handler ran')
        callback = _dispatch_signal.getsignal(_dispatch_signal.SIGTERM)
        assert callback is _dispatch_registered[0], (
            'controller did not restore the exact previous SIGTERM handler')
        _dispatch_deliveries.append(_dispatch_requests[0])
        callback(_dispatch_requests[0], None)
        # The actual callback, not the harness, must publish the receipt.
        assert _dispatch_real_select(readers, [], [], 0)[0], (
            'previous handler did not acknowledge receipt on the pipe')
        _dispatch_receipts.append(True)
    return _dispatch_real_select(readers, writers, exceptional, timeout)

_dispatch_signal.signal = _dispatch_register
_dispatch_os.kill = _dispatch_queue
_dispatch_select.select = _dispatch_wait
'''

VERIFY_DISPATCH = r'''
assert _dispatch_requests == [_dispatch_signal.SIGTERM], _dispatch_requests
assert _dispatch_deliveries == [_dispatch_signal.SIGTERM], (
    'test returned without waiting for deferred dispatch', _dispatch_deliveries)
assert _dispatch_receipts == [True], 'previous handler receipt was not observed'
print('deferred SIGTERM dispatch acknowledged', flush=True)
'''


def test_subprocess_regression_observes_deferred_sigterm_receipt(tmp_path, monkeypatch):
    """Execute the existing test, not a duplicated example of a repaired test."""
    subject = runpy.run_path(str(TARGET))
    real_run = subprocess.run
    children = []

    def run_with_deferred_dispatch(argv, *args, **kwargs):
        assert argv[1] == '-c', 'expected the signal-isolating child program'
        wrapped = [*argv]
        wrapped[2] = DEFER_DISPATCH + '\n' + argv[2] + '\n' + VERIFY_DISPATCH
        result = real_run(wrapped, *args, **kwargs)
        children.append(result)
        return result

    monkeypatch.setattr(subprocess, 'run', run_with_deferred_dispatch)
    subject['test_subsequent_sigterm_reaches_previous_handler_in_subprocess'](tmp_path)
    assert len(children) == 1, 'the real-signal regression must remain isolated'
    assert 'deferred SIGTERM dispatch acknowledged' in children[0].stdout
