"""A per-attempt deadline for the suite's one owned subprocess group."""
from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time

_INTERRUPT_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number of seconds")
    return seconds


class _Interrupted(BaseException):
    pass


def _interrupt(signum, _frame):
    raise _Interrupted(signum)


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def _kill_owned(process):
    if process is not None and process.returncode is None:
        _signal_group(process.pid, signal.SIGKILL)
        process.wait()


def run(command, timeout_s, kill_after_s):
    timeout_s, kill_after_s = positive_seconds(timeout_s), positive_seconds(kill_after_s)
    process = None
    previous = {sig: signal.signal(sig, _interrupt) for sig in _INTERRUPT_SIGNALS}
    # SIG_IGN auto-reaps children: wait() can then report0 for a failed child,
    # and the leader's PID no longer anchors group ownership during grace.
    previous[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    try:
        process = subprocess.Popen(command, start_new_session=True)
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _signal_group(process.pid, signal.SIGTERM)
            # Do not reap/poll the leader during this grace. Its unreaped PID
            # anchors ownership even if it exits before a resistant child.
            time.sleep(kill_after_s)
            _signal_group(process.pid, signal.SIGKILL)
            code = process.wait()
            status = 137 if code == -signal.SIGKILL else 124
            print(f"tessera suite deadline expired: inner_status={status}", file=sys.stderr, flush=True)
            return status
        return 128 - code if code < 0 else code
    except _Interrupted as exc:
        for sig in _INTERRUPT_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        _kill_owned(process)
        return 128 + exc.args[0]
    except BaseException:
        _kill_owned(process)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=positive_seconds, required=True)
    parser.add_argument("--kill-after-s", type=positive_seconds, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    return run(command, args.timeout_s, args.kill_after_s)
