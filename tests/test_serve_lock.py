"""The serve lock must not outlive its holder.

On 2026-09-04 a withdrawn pool action's TERM landed inside
``serve_lock_release``, between ``rm -f owner`` and ``rmdir``.  What was left
was the lock directory with no owner file -- and the only staleness rule the
loop had (older than an hour, on a box with no container running) could not see
it: the directory was seconds old and a container was still up.  The next serve
queued behind a lock nobody held, and would have queued for an hour.

These tests drive ``experiments/serve_lock.sh`` as a shell library, with
``SERVE_LOCK`` pointed at a temp directory and ``SERVE_LOCK_POLL`` at one
second, so each reaping rule is exercised for what it is.
"""

import os
import subprocess
from pathlib import Path

import pytest

LOCK_SH = Path(__file__).resolve().parents[1] / "experiments" / "serve_lock.sh"


def run_sh(script: str, lock: Path, timeout: float = 30.0):
    """Source the lock library and run `script`, returning the CompletedProcess."""
    body = f'set -euo pipefail\nsource "{LOCK_SH}"\n{script}\n'
    env = dict(os.environ, SERVE_LOCK=str(lock), SERVE_LOCK_POLL="1")
    return subprocess.run(
        ["bash", "-c", body], capture_output=True, text=True, timeout=timeout, env=env
    )


def test_acquire_on_a_free_lock_records_its_own_pid(tmp_path):
    lock = tmp_path / "serve.lock"
    proc = run_sh('serve_lock_acquire; cat "$SERVE_LOCK/owner"', lock)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[0].isdigit()
    assert (lock / "owner").exists()


def test_release_only_removes_a_lock_this_process_owns(tmp_path):
    """The 2026-09-02 rule, kept: an EXIT trap in a waiter must not free
    somebody else's lock."""
    lock = tmp_path / "serve.lock"
    lock.mkdir()
    (lock / "owner").write_text("999999 2026-09-04T00:00:00Z somebody-else\n")
    proc = run_sh("serve_lock_release", lock)
    assert proc.returncode == 0, proc.stderr
    assert (lock / "owner").exists(), "release freed a lock it did not own"


def test_a_lock_with_no_owner_file_is_reaped(tmp_path):
    """Rule 2, and the exact shape the withdrawn action left behind."""
    lock = tmp_path / "serve.lock"
    lock.mkdir()  # no owner file: release was killed between rm and rmdir
    proc = run_sh('serve_lock_acquire; echo TOOK "$(cat "$SERVE_LOCK/owner")"', lock)
    assert proc.returncode == 0, proc.stderr
    assert "has had no owner for a full poll" in proc.stderr
    assert proc.stdout.startswith("TOOK ")


def test_a_lock_whose_owner_is_gone_is_reaped(tmp_path):
    """Rule 1: the owner file names a pid that is not a live process."""
    lock = tmp_path / "serve.lock"
    lock.mkdir()
    dead = subprocess.run(["bash", "-c", "echo $$"], capture_output=True, text=True)
    dead_pid = dead.stdout.strip()
    assert not Path(f"/proc/{dead_pid}").exists()
    (lock / "owner").write_text(f"{dead_pid} 2026-09-04T00:00:00Z dead-serve\n")
    proc = run_sh("serve_lock_acquire", lock)
    assert proc.returncode == 0, proc.stderr
    assert f"owner pid {dead_pid} is gone" in proc.stderr
    assert (lock / "owner").read_text().split()[0] != dead_pid


def test_a_lock_with_a_live_owner_is_not_reaped(tmp_path):
    """The property that matters more than any of the reaping rules: two serves
    on one box is an OOM, so a held lock must block."""
    lock = tmp_path / "serve.lock"
    lock.mkdir()
    holder = subprocess.Popen(["sleep", "20"])
    try:
        (lock / "owner").write_text(f"{holder.pid} 2026-09-04T00:00:00Z live-serve\n")
        with pytest.raises(subprocess.TimeoutExpired):
            run_sh("serve_lock_acquire", lock, timeout=6.0)
        assert (lock / "owner").read_text().split()[0] == str(holder.pid)
    finally:
        holder.kill()
        holder.wait()


def test_the_ownerless_rule_needs_two_observations(tmp_path):
    """A healthy acquire writes the owner file microseconds after ``mkdir``.
    The ownerless rule must not race that window, which is why it fires on the
    second consecutive ownerless poll and not the first."""
    lock = tmp_path / "serve.lock"
    lock.mkdir()  # seen ownerless once
    holder = subprocess.Popen(["sleep", "20"])
    writer = subprocess.Popen(
        ["bash", "-c", f'sleep 0.4; echo "{holder.pid} 2026-09-04T00:00:00Z slow" > "{lock}/owner"']
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_sh("serve_lock_acquire", lock, timeout=6.0)
        assert (lock / "owner").read_text().split()[0] == str(holder.pid), (
            "the ownerless reap fired on a lock whose owner was one poll behind"
        )
    finally:
        writer.wait()
        holder.kill()
        holder.wait()
