"""Attempt-bound rendezvous and deadlines for the existing TP2 graph recipe.

This is payload coordination, not admission: only pbcampaign/pbrun admit ranks.
The worker, not a payload, releases the PB scope after its controller exits.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

WINDOW_SECONDS = 5400
CLEANUP_SECONDS = 180
HOSTS = ("sparklina", "sparky")


class Refused(RuntimeError):
    pass


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


def scope_name(key: str, nonce: str) -> str:
    return "prismabuild-job" + hashlib.sha256((key + nonce).encode()).hexdigest()[:32] + ".slice"


def positive_time(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise Refused(f"{label} is not a finite positive time")
    return float(value)


def require_claim(identity: dict, queue: Path) -> dict:
    """Read only CLAIMED; DONE/FAILED markers can never authorize a launch."""
    key, nonce, scope = (identity.get(k) for k in ("action_key", "nonce", "scope_id"))
    if not isinstance(key, str) or not re.fullmatch("[a-f0-9]{64}", key):
        raise Refused("rank has no full PB action key")
    if not isinstance(nonce, str) or not re.fullmatch("[a-f0-9]{32}", nonce):
        raise Refused("rank has no PB attempt nonce")
    if scope != scope_name(key, nonce):
        raise Refused("rank PB scope does not bind its action and nonce")
    claim = read_json(queue / "claimed" / (key + ".json"))
    control = claim.get("resource_scope", {})
    if (claim.get("action_key") != key or claim.get("claimed_host") != identity["host"]
            or control.get("action_key") != key or control.get("nonce") != nonce
            or control.get("scope_id") != scope
            or claim.get("container_owner") != identity["container_owner"]):
        raise Refused("rank no longer owns this live PB claim/attempt/host/container owner")
    if positive_time(claim.get("claimed_unix"), "claim start") != identity["claimed_unix"]:
        raise Refused("rank claim start changed")
    return claim


class Envelope:
    def __init__(self, end_unix: float, cleanup_seconds: float = CLEANUP_SECONDS):
        self.end_unix = positive_time(end_unix, "window end")
        self.end = time.monotonic() + (self.end_unix - time.time())
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
            stdout=None, env=None, input_text=None, limit=30) -> subprocess.CompletedProcess:
        """A subprocess and its process group share the same finite envelope."""
        cap = min(self.remaining(cleanup=cleanup), limit)
        process = subprocess.Popen(argv, start_new_session=True, text=True, env=env,
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
            # A command may exit while a grandchild still owns its session.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=min(1, max(.01, self.end - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=max(.01, min(1, self.end - time.monotonic())))


class Rendezvous:
    def __init__(self, root: Path, identity: dict, queue: Path, envelope: Envelope,
                 *, poll_seconds=.2):
        self.root, self.identity, self.queue, self.envelope = root, identity, queue, envelope
        self.rank = identity["rank"]
        self.peer = None
        self.poll_seconds = poll_seconds
        self.path = root / f"rank{self.rank}.json"
        if self.path.exists():
            raise Refused("rank rendezvous path already used; never adopt an old attempt")
        require_claim(identity, queue)
        atomic_json(self.path, identity)

    def publish(self, stage: str, **fields) -> None:
        atomic_json(self.root / f"{stage}-rank{self.rank}.json",
                    {**self.identity, **fields, "stage": stage, "written_unix": time.time()})

    def check(self) -> None:
        require_claim(self.identity, self.queue)
        if self.peer is not None:
            require_claim(self.peer, self.queue)
        for rank in (0, 1):
            path = self.root / f"failed-rank{rank}.json"
            if path.exists():
                value = self.checked(path, rank)
                raise Refused(f"rank {rank} failed: {value.get('error')}")

    def checked(self, path: Path, rank: int) -> dict:
        value = read_json(path)
        expected = self.identity if rank == self.rank else self.peer
        if expected is None:
            raise Refused("unbound peer stage")
        for key in ("rank", "action_key", "nonce", "scope_id", "host", "container_owner",
                    "claimed_unix", "run_id", "input_sha256"):
            if value.get(key) != expected[key]:
                raise Refused(f"stale or mismatched {key} in {path.name}")
        written = positive_time(value.get("written_unix"), "stage timestamp")
        if not expected["claimed_unix"] <= written <= time.time():
            raise Refused("stage timestamp outside this admitted attempt")
        return value

    def bind_peer(self) -> None:
        path = self.root / f"rank{1 - self.rank}.json"
        while not path.exists():
            self.envelope.remaining()
            require_claim(self.identity, self.queue)
            time.sleep(min(self.poll_seconds, self.envelope.remaining()))
        value = read_json(path)
        if (value.get("rank") != 1 - self.rank or value.get("host") != HOSTS[1 - self.rank]
                or value.get("run_id") != self.identity["run_id"]
                or value.get("input_sha256") != self.identity["input_sha256"]):
            raise Refused("peer belongs to a different host/run/input tuple")
        require_claim(value, self.queue)
        self.peer = value
        self.envelope.tighten(value["claimed_unix"] + WINDOW_SECONDS)
        self.check()

    def wait(self, stage: str) -> dict:
        path = self.root / f"{stage}-rank{1 - self.rank}.json"
        while True:
            self.envelope.remaining()
            self.check()
            if path.exists():
                return self.checked(path, 1 - self.rank)
            time.sleep(min(self.poll_seconds, self.envelope.remaining()))


def terminal_cleanup(identity: dict, terminal: dict) -> dict:
    """A successful payload is not a physical owned-scope handoff."""
    control = terminal.get("resource_scope", {})
    cleanup = terminal.get("resource_scope_cleanup", {})
    export = cleanup.get("export", {})
    if (terminal.get("action_key") != identity["action_key"]
            or terminal.get("claimed_host") != identity["host"]
            or terminal.get("claimed_unix") != identity["claimed_unix"]
            or control.get("nonce") != identity["nonce"]
            or control.get("scope_id") != identity["scope_id"]
            or cleanup.get("complete") is not True
            or cleanup.get("nonce") != identity["nonce"]
            or cleanup.get("settle_error")
            or export.get("scope_id") != identity["scope_id"]
            or export.get("empty") is not True or export.get("tickets_pending") is not False
            or cleanup.get("remaining")):
        raise Refused("terminal has no exact-attempt physical scope/container cleanup proof")
    stopped = positive_time(export.get("stopped_unix"), "broker stopped time")
    if stopped > identity["window_end_unix"]:
        raise Refused("broker cleanup completed outside the whole-window envelope")
    return cleanup
