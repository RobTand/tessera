"""Finite deadlines and owned process-group cleanup for graph qualifications.

This is payload coordination, not admission.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

# Retained D30 process-group TERM grace; unrelated window admission policy is archived.
CLEANUP_SECONDS = 180
TERM_GRACE_SECONDS = 10.0


class Refused(RuntimeError):
    pass


def positive_time(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise Refused(f"{label} is not a finite positive time")
    return float(value)


class Envelope:
    def __init__(self, end_unix: float, cleanup_seconds: float = CLEANUP_SECONDS):
        self.end_unix = positive_time(end_unix, "window end")
        self.end = time.monotonic() + (self.end_unix - time.time())
        self.terminations = []
        self.cleanup_seconds = cleanup_seconds

    def tighten(self, end_unix: float) -> None:
        end_unix = positive_time(end_unix, "peer window end")
        if end_unix < self.end_unix:
            self.end = min(self.end, time.monotonic() + (end_unix - time.time()))
            self.end_unix = end_unix

    def remaining(self, *, cleanup: bool = False) -> float:
        left = self.end - time.monotonic() - (0 if cleanup else self.cleanup_seconds)
        if left <= 0:
            raise TimeoutError("whole graph window expired" if cleanup else "graph work deadline; cleanup reserve begins")
        return left

    def run(self, argv: list[str], *, check=True, cleanup=False, tick=None,
            stdout=None, env=None, input_text=None, limit=30, text=True) -> subprocess.CompletedProcess:
        """A subprocess and its process group share the same finite envelope."""
        cap = min(self.remaining(cleanup=cleanup), limit)
        process = subprocess.Popen(argv, start_new_session=True, text=text, env=env,
                                   stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                   stdout=stdout if stdout is not None else subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        until = time.monotonic() + cap
        try:
            first = True
            while True:
                if tick:
                    tick()
                left = min(until - time.monotonic(), self.remaining(cleanup=cleanup))
                if left <= 0:
                    raise TimeoutError(f"subprocess deadline: {argv[0]}")
                try:
                    out, _ = process.communicate(input=input_text if first else None,
                                                 timeout=min(.2, left))
                    break
                except subprocess.TimeoutExpired:
                    first = False
            result = subprocess.CompletedProcess(argv, process.returncode, out)
            if check and result.returncode:
                raise Refused(f"command failed ({result.returncode}): {argv!r}: {out}")
            return result
        finally:
            self._terminate(process, argv)

    def _terminate(self, process, argv):
        """Stop only the session this Envelope created, including owned descendants."""
        record = dict(pid=process.pid, argv=argv, signals=[], started_unix=time.time())
        self.terminations.append(record)
        def alive():
            process.poll()  # reap leader; a remaining descendant may still own the group
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False
        def send(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            record["signals"].append(dict(signal=sig.name, unix=time.time(), monotonic=time.monotonic()))
        if alive():
            send(signal.SIGTERM)
            stop_at = min(time.monotonic() + TERM_GRACE_SECONDS, self.end)
            while alive() and time.monotonic() < stop_at:
                time.sleep(min(.05, max(0, stop_at - time.monotonic())))
            if alive():
                record["deadline_shortened_grace"] = stop_at == self.end
                send(signal.SIGKILL)
        reap_timeout = max(.01, min(1, self.end - time.monotonic()))
        record.update(reap_timeout_seconds=reap_timeout, reap_timed_out=False)
        try:
            process.wait(timeout=reap_timeout)
        except subprocess.TimeoutExpired:
            # A delayed reap is cleanup evidence, not a replacement work failure.
            record["reap_timed_out"] = True
        finally:
            record.update(returncode=process.returncode, ended_unix=time.time())
