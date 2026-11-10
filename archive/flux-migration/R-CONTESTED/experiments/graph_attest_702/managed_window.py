"""Deadlines, process-group envelope and memory policy for the TP2 graph window recipe.

This is payload coordination, not admission.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

# The script checkout supplies the same standalone helper as the package; no
# serving or encoder module is imported into this lifecycle controller.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tessera.dev_mode import NOT_COMPUTED, dev_mode_enabled, seal_check

WINDOW_SECONDS = 5400
CLEANUP_SECONDS = 180
PEER_WAIT_SECONDS = 3600
HOSTS = ("sparklina", "sparky")

# One owner for the rank lifecycle and its memory admission/abort policy.
MEMORY_POLICY = dict(schema="tessera.window_memory_policy.d30.v1",
                     model_kv_estimate_gib=98, unmeasured_graph_host_allowance_gib=6,
                     reserve_gib=3, headroom_wait_seconds=900.0,
                     host_cap_gib=104, gpu_subset_cap_gib=102,
                     sample_seconds=1.0, abort_below_gib=2.0, term_grace_seconds=10.0,
                     authority="CEO final D30 amendment, 2026-10-05; new reviewed invocation only",
                     provenance="98 is ceil(max(94.4,96.1)+1.625), a conservative hybrid estimate, not a measured peak. "
                                "6 is UNMEASURED retained graph/host allowance; 3 is reserve. "
                                "104 is host cap; 102 caps its GPU subset, not additive memory.",
                     historical_source="archive/flux-migration/R-CONTESTED/experiments/graph_attest_702/RUNPLAN-artifact.md; "
                                       "kernels-window4-headroom-equation-packet-20261005",
                     historical_packet_sha256="1a28bff770790a09b0df4573d46acf31e44aa12c6118cf31b9a32a279ffdb4f8",
                     counter_definitions={
                         "mem_available_gib": "Whole-host /proc/meminfo MemAvailable kB / 1048576; includes reclaimable cache, not process RSS.",
                         "scope_memory": "Owned cgroup v2 host charge; current sampled, peak claim-lifetime; CUDA coverage unproven.",
                         "host_rss_and_driver_context": "UNKNOWN: not recorded or decomposed by MemAvailable minima; never zero.",
                         "overlap": "Shared DRAM counters overlap; do not sum GPU used, UVM residual, RSS, cgroup charge or whole-box used.",
                         "sampling_limit": "One-second sampled guard, not continuous protection against transient OOM."})
MEMORY_POLICY["start_gib"] = sum(MEMORY_POLICY[key] for key in
    ("model_kv_estimate_gib", "unmeasured_graph_host_allowance_gib", "reserve_gib"))
MEMORY_POLICY["hosts"] = {host: {"start_gib": MEMORY_POLICY["start_gib"]} for host in HOSTS}


class Refused(RuntimeError):
    pass


def check_memory_policy(recorded, *, where):
    """The recorded policy versus the running one is a run-identity seal (D32):
    dev mode stamps one [DEV-MODE] line and the live MEMORY_POLICY keeps
    governing; certified mode keeps the verbatim refusal. Byte damage against
    the policy's own recorded digest stays integrity at the call sites."""
    seal_check("memory policy identity", recorded, MEMORY_POLICY, where=where,
               refusal=Refused("prepared memory policy changed; never restamp a reviewed window"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    if path.stat().st_size > 1 << 20:
        raise Refused(f"oversized protocol record: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Refused(f"protocol record is not an object: {path}")
    return value


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

    def run_container_control(self, cid, operation, *, check=True):
        """Five-second cleanup-only control, even after the work envelope expires.

        LocalArm verifies labels/parent before sending a signal. This bounded
        rescue cannot launch a model or extend its work lifetime.
        """
        if not re.fullmatch("[a-f0-9]{64}", cid) or operation not in ("inspect", "TERM", "KILL"):
            raise Refused("invalid cleanup-only container control")
        argv = (["docker", "inspect", cid] if operation == "inspect" else
                ["docker", "kill", "--signal", operation, cid])
        process = subprocess.Popen(argv, start_new_session=True, text=True,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        try:
            out, _ = process.communicate(timeout=5)
            result = subprocess.CompletedProcess(argv, process.returncode, out)
            if check and result.returncode:
                raise Refused(f"owned cleanup control failed ({result.returncode}): {argv!r}: {out}")
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
            stop_at = min(time.monotonic() + MEMORY_POLICY["term_grace_seconds"], self.end)
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
