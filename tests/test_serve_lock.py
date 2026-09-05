"""The host-local serve lock is atomic and released by owner death."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "experiments" / "serve_lock.sh"
GPU_PROBES = (
    "moe_decode_target_probe.sh",
    "moe_route_load_probe.sh",
    "nvfp4_moe_oracle_probe.sh",
)


def _environment(tmp_path: Path) -> dict[str, str]:
    """Use a Docker probe that truthfully reports no test containers."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    docker = bindir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = ps ]; then exit 0; fi\n"
        "echo unexpected-docker-command >&2\n"
        "exit 64\n"
    )
    docker.chmod(0o755)
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")


def _race_environment(tmp_path: Path, contender: str) -> dict[str, str]:
    """Pause two dead-owner reapers at the vulnerable compare/unlink edge.

    Every pause here waits for a marker another contender publishes, never for
    a number of seconds.  A ``sleep``-counted window is a bet on how fast the
    box is, and on a saturated one the bet loses in the direction that makes
    this test USELESS rather than red: if A stops waiting before B has reached
    the guard, B inspects A's live token instead of the dead one, and the test
    passes without the race it exists to exercise (#182).
    """
    env = _environment(tmp_path)
    bindir = tmp_path / "bin"
    sync = tmp_path / "sync"
    sync.mkdir(exist_ok=True)
    # B announces that it is BLOCKED at the transition guard, which it learns
    # by trying the lock non-blockingly first.  "Blocked" is the load-bearing
    # word: it is proof that B cannot be about to capture anything, which is
    # what lets A stop waiting for it.  A marker written merely on the way IN
    # to the guard proves nothing -- A would release B while B still had the
    # guard free ahead of it, and the two would then run in an order that never
    # reaches the race.  A non-blocking acquisition that SUCCEEDS is kept: an
    # flock lock lives on the open file description, so the fd the parent still
    # holds keeps it after this shim exits, exactly as a direct ``flock -x fd``
    # would.
    flock = bindir / "flock"
    flock.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "if [ \"$SERVE_LOCK_RACE_ID\" = B ] && [ \"${1:-}\" = -x ]; then\n"
        "  if /usr/bin/flock -nx \"${@:2}\"; then\n"
        "    exit 0\n"
        "  fi\n"
        "  : > \"$SERVE_LOCK_RACE_SYNC/b_blocked_at_guard\"\n"
        "fi\n"
        "exec /usr/bin/flock \"$@\"\n"
    )
    flock.chmod(0o755)
    readlink = bindir / "readlink"
    readlink.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "counter=$SERVE_LOCK_RACE_SYNC/readlink_$SERVE_LOCK_RACE_ID\n"
        "count=0; [ ! -f \"$counter\" ] || read -r count < \"$counter\"\n"
        "count=$((count + 1)); printf '%s\\n' \"$count\" > \"$counter\"\n"
        "value=$(/usr/bin/readlink \"$@\")\n"
        # B's second readlink is the observation the whole test turns on.  With
        # the guard broken it is the compare-recheck of the DEAD token inside
        # one acquire pass -- B has captured the old token and is one unlink
        # away from A's replacement -- so it holds here until A has published,
        # to make that unlink actually happen.  With the guard intact B cannot
        # reach the recheck at all: this is instead its second polling pass,
        # against A's live token, and A published long ago, so the wait returns
        # at once.  Either way it is B's own progress, never a clock.
        "if [ \"$SERVE_LOCK_RACE_ID\" = B ] && [ \"$count\" = 2 ]; then\n"
        "  : > \"$SERVE_LOCK_RACE_SYNC/b_second_readlink\"\n"
        "  while [ ! -e \"$SERVE_LOCK_RACE_SYNC/a_acquired\" ]; do\n"
        "    sleep 0.01\n"
        "  done\n"
        "fi\n"
        "printf '%s\\n' \"$value\"\n"
    )
    readlink.chmod(0o755)
    unlink = bindir / "unlink"
    unlink.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "target=${!#}\n"
        "if [ \"$SERVE_LOCK_RACE_ID\" = A ] && "
        "[ \"$target\" = \"$SERVE_LOCK_RACE_PATH\" ]; then\n"
        "  : > \"$SERVE_LOCK_RACE_SYNC/a_at_unlink\"\n"
        # Hold the dead token in place until B has either captured it (which a
        # broken guard permits, and which is the race) or proved it is blocked
        # out of the guard (which a working one guarantees).  Exactly one of
        # the two happens, on every box, at every speed -- so this wait needs
        # no count and cannot expire early into a vacuous pass.
        "  while [ ! -e \"$SERVE_LOCK_RACE_SYNC/b_blocked_at_guard\" ] && "
        "[ ! -e \"$SERVE_LOCK_RACE_SYNC/b_second_readlink\" ]; do\n"
        "    sleep 0.01\n"
        "  done\n"
        "fi\n"
        "exec /usr/bin/unlink \"$@\"\n"
    )
    unlink.chmod(0o755)
    env.update(
        SERVE_LOCK_RACE_ID=contender,
        SERVE_LOCK_RACE_PATH=str(tmp_path / "serve.lock"),
        SERVE_LOCK_RACE_SYNC=str(sync),
    )
    return env


_ROUND_TRIP_S: float | None = None


def _acquire_round_trip_s() -> float:
    """Seconds one whole ``serve_lock_acquire``/``release`` costs HERE and NOW.

    Every wait in this file is waiting for some part of that round trip, so a
    fixed wall-clock deadline is really a claim about how fast the box is.
    Under ``pytest -n 24`` on an 80-core box the claim is false and the test
    goes red for the load rather than for the lock (#182).  Measuring the round
    trip at the moment of the wait makes the bound a multiple of observed work
    instead: whatever slows the awaited subprocess slows this measurement by
    the same factor, so the bound tracks the box.  It is measured once per
    session, lazily, which is inside the loaded window whenever the load is
    what is being survived.
    """
    global _ROUND_TRIP_S
    if _ROUND_TRIP_S is None:
        with tempfile.TemporaryDirectory() as raw:
            box = Path(raw)
            env = _environment(box)
            # Against a DEAD owner, so the reference walks the same path the
            # waits wait on: readlink, the liveness probe, the container probe,
            # unlink, republication.  An acquire of an absent lock is one
            # ``ln`` and measures nothing this file cares about.
            script = (
                f'lock="{box / "unit.lock"}"; '
                f'ln -sfT "999999999:1:{"a" * 32}" "$lock"; '
                f'SERVE_LOCK="$lock"; source "{HELPER}"; '
                'SERVE_LOCK_OWNER=round-trip; serve_lock_acquire; serve_lock_release'
            )
            samples = []
            for _ in range(3):
                started = time.monotonic()
                subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env=env, text=True, capture_output=True, check=True,
                )
                samples.append(time.monotonic() - started)
        _ROUND_TRIP_S = sorted(samples)[1]
    return _ROUND_TRIP_S


#: How many acquire round trips a wait may take before it is called a hang.
#: Set from two measurements rather than from taste.  Above: the worst wait in
#: this file costs 3.1 round trips idle and 5.5 starved, so 200 is ~36x the
#: work.  Below: 200 x the idle round trip is ~3.4 s, so replacing the old
#: three-second literal cannot make any wait TIGHTER than it was on an idle
#: box -- it can only make it looser, and only in proportion to how much slower
#: the box has made the work.
_HANG_GUARD_ROUND_TRIPS = 200


def _wait_until(
    reached, process: subprocess.Popen[str], what: str,
    *, round_trips: int = _HANG_GUARD_ROUND_TRIPS,
) -> None:
    """Wait for a subprocess to reach an observable point of its own.

    The bound is a hang guard, not a deadline: every point this file waits for
    is reached within a handful of acquire round trips of the wait starting,
    and the guard is two hundred of them -- headroom in units of the work, not
    in seconds, so a box that runs the work ten times slower gets ten times the
    wall clock instead of the same three seconds.
    """
    unit = _acquire_round_trip_s()
    deadline = time.monotonic() + round_trips * unit
    while time.monotonic() < deadline:
        if reached():
            return
        if process.poll() is not None:
            pytest.fail(f"lock holder exited {process.returncode}: {process.stderr.read()}")
        time.sleep(0.01)
    pytest.fail(
        f"{what} did not happen within {round_trips} acquire round trips "
        f"({unit:.3f}s each as measured on this box)"
    )


def _wait_for(path: Path, process: subprocess.Popen[str], **kwargs) -> None:
    """Wait for a marker a subprocess publishes."""
    _wait_until(path.exists, process, f"the marker {path.name}", **kwargs)


def _outer_bound_s(
    busy_deadline_s: int = 0, *, round_trips: int = _HANG_GUARD_ROUND_TRIPS
) -> str:
    """A wall bound for a contender that is meant to reach its OWN verdict.

    ``timeout`` here is a hang guard around a subprocess whose result the test
    reads.  Sized as a literal it competes with the helper's own busy deadline,
    and on a loaded box it wins -- turning "the helper refused" (3) into "the
    harness killed it" (124), which is #182 again in another spelling.  So it
    is the helper's deadline, which is whole seconds compared against
    ``date +%s`` and so fires between ``busy`` and ``busy + 1``, plus headroom
    measured in acquire round trips on this box.
    """
    return f"{busy_deadline_s + 1 + round_trips * _acquire_round_trip_s():.2f}"


def test_serve_lock_acquisition_publishes_one_atomic_owner(tmp_path: Path) -> None:
    """There is no mkdir-to-owner gap for a withdrawal to strand."""
    lock = tmp_path / "serve.lock"
    ready = tmp_path / "ready"
    script = (
        f'SERVE_LOCK="{lock}"; source "{HELPER}"; '
        'SERVE_LOCK_OWNER=test-holder; serve_lock_acquire; '
        f'printf ready > "{ready}"; sleep 300'
    )
    holder = subprocess.Popen(
        ["bash", "-euo", "pipefail", "-c", script],
        env=_environment(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _wait_for(ready, holder)
        assert lock.is_symlink(), "the owner must be the atomic lock object"
        fields = os.readlink(lock).split(":")
        assert len(fields) == 3
        assert fields[0] == str(holder.pid)
        assert fields[1].isdigit() and int(fields[1]) > 0
        assert len(fields[2]) == 32 and set(fields[2]) <= set("0123456789abcdef")
        contender_script = (
            f'SERVE_LOCK="{lock}"; SERVE_LOCK_TIMEOUT=1; SERVE_LOCK_POLL_S=0.01; '
            f'source "{HELPER}"; serve_lock_acquire'
        )
        contender = subprocess.run(
            ["timeout", _outer_bound_s(1), "bash", "-euo", "pipefail", "-c",
             contender_script],
            env=_environment(tmp_path),
            text=True,
            capture_output=True,
        )
        assert contender.returncode == 3, contender.stdout + contender.stderr
        assert os.readlink(lock) == ":".join(fields)
    finally:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait()


def test_serve_lock_reclaims_a_dead_atomic_owner(tmp_path: Path) -> None:
    """SIGKILL releases ownership without waiting for an arbitrary age."""
    lock = tmp_path / "serve.lock"
    lock.symlink_to(f"999999999:1:{'a' * 32}")
    script = (
        f'SERVE_LOCK="{lock}"; SERVE_LOCK_TIMEOUT=1; SERVE_LOCK_POLL_S=0.01; '
        f'source "{HELPER}"; SERVE_LOCK_OWNER=test-contender; '
        'serve_lock_acquire; printf acquired; serve_lock_release'
    )
    result = subprocess.run(
        ["timeout", _outer_bound_s(), "bash", "-euo", "pipefail", "-c", script],
        env=_environment(tmp_path),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "acquired"
    assert not os.path.lexists(lock)


@pytest.mark.parametrize("legacy", [False, True], ids=["atomic", "legacy"])
def test_release_only_removes_a_lock_this_process_owns(tmp_path: Path, legacy: bool) -> None:
    """A waiter or an unrelated EXIT handler may not release the holder."""
    lock = tmp_path / "serve.lock"
    if legacy:
        lock.mkdir()
        (lock / "owner").write_text(f"{os.getpid()} live-holder\n")
    else:
        lock.symlink_to(f"{os.getpid()}:1:{'c' * 32}")
    script = f'SERVE_LOCK="{lock}"; source "{HELPER}"; serve_lock_release'
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env=_environment(tmp_path), text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    if legacy:
        assert (lock / "owner").read_text() == f"{os.getpid()} live-holder\n"
    else:
        assert os.readlink(lock) == f"{os.getpid()}:1:{'c' * 32}"


def _live_legacy_owner_contender(tmp_path: Path, *, probe_denied: bool = False):
    lock = tmp_path / "serve.lock"
    lock.mkdir()
    owner = lock / "owner"
    owner.write_text(f"{os.getpid()} live-holder\n")
    old = time.time() - 7200
    os.utime(lock, (old, old))
    script = (
        f'SERVE_LOCK="{lock}"; SERVE_LOCK_TIMEOUT=1; SERVE_LOCK_POLL_S=0.01; '
        f'source "{HELPER}"; '
        + ('kill() { return 1; }; ' if probe_denied else '')
        + 'serve_lock_acquire; serve_lock_release'
    )
    result = subprocess.run(
        ["timeout", _outer_bound_s(1), "bash", "-euo", "pipefail", "-c", script],
        env=_environment(tmp_path), text=True, capture_output=True,
    )
    return result, owner


def test_live_legacy_owner_blocks_atomic_acquisition(tmp_path: Path) -> None:
    """An atomic publication may not succeed inside an existing directory lock."""
    result, owner = _live_legacy_owner_contender(tmp_path)
    assert result.returncode == 3, result.stdout + result.stderr
    assert owner.read_text() == f"{os.getpid()} live-holder\n"


def test_live_legacy_owner_survives_a_denied_signal_probe(tmp_path: Path) -> None:
    """EPERM from kill -0 is not evidence that an existing process is dead."""
    result, owner = _live_legacy_owner_contender(tmp_path, probe_denied=True)
    assert result.returncode == 3, result.stdout + result.stderr
    assert owner.read_text() == f"{os.getpid()} live-holder\n"


def test_two_dead_owner_reapers_cannot_unlink_the_new_holder(tmp_path: Path) -> None:
    """Compare+unlink and the replacement acquire are one serialized transition."""
    lock = tmp_path / "serve.lock"
    sync = tmp_path / "sync"
    lock.symlink_to(f"999999999:1:{'b' * 32}")

    def start(contender: str) -> subprocess.Popen[str]:
        ready = sync / f"{contender.lower()}_acquired"
        script = (
            # No SERVE_LOCK_TIMEOUT: A acquires on its first pass and B is
            # meant to block for as long as the test looks at it.  A busy
            # deadline here is the helper's own ``date +%s`` wall clock, and a
            # loaded box would make B exit 3 -- ending the contention this test
            # needs to keep alive.  The finally block reaps both groups.
            f'SERVE_LOCK="{lock}"; SERVE_LOCK_POLL_S=0.01; '
            f'source "{HELPER}"; SERVE_LOCK_OWNER=test-{contender}; '
            f'serve_lock_acquire; printf ready > "{ready}"; sleep 300'
        )
        return subprocess.Popen(
            ["bash", "-euo", "pipefail", "-c", script],
            env=_race_environment(tmp_path, contender),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    first = start("A")
    second: subprocess.Popen[str] | None = None
    try:
        _wait_for(sync / "a_at_unlink", first)
        second = start("B")
        # B has reached the point where it either races A or is held out of the
        # race, so A's transition is made against a contender that is really
        # contending.
        _wait_until(
            lambda: ((sync / "b_blocked_at_guard").exists()
                     or (sync / "b_second_readlink").exists()),
            second, "B reaching the transition guard",
        )
        _wait_for(sync / "a_acquired", first)
        # B has taken a second readlink.  A broken guard would have spent it on
        # the compare-recheck of the dead token, leaving B one unlink away from
        # A's new owner; the serialized helper spends it on a second polling
        # pass against A's live token, which is a decline.  Waiting for B's own
        # progress replaces a fixed negative window that a loaded box could
        # expire before B had entered the guard at all.
        _wait_for(sync / "b_second_readlink", second)

        def b_finished_that_pass() -> bool:
            """B has left the pass a broken guard would have let it win.

            Its readlink counter is B's own step count.  Reaching three means
            the pass that took the second readlink ended and another began --
            and it ended in a decline, because the only other way out of that
            pass is unlink-and-acquire, which publishes ``b_acquired`` and
            takes no third readlink.  So either side of the disjunction is B
            telling us it is done deciding; nothing here is a guess about how
            long deciding takes.
            """
            if (sync / "b_acquired").exists():
                return True
            try:
                return int((sync / "readlink_B").read_text().strip() or 0) >= 3
            except (OSError, ValueError):
                return False  # bash is mid-write; look again next poll

        _wait_until(b_finished_that_pass, second, "B's pass against A's published owner")
        assert not (sync / "b_acquired").exists(), (
            "both contenders entered the protected region: the second reaper "
            "unlinked the first contender's newly published owner"
        )
        assert second.poll() is None, "the second contender should remain blocked"
        assert os.readlink(lock).split(":", 1)[0] == str(first.pid)
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


@pytest.mark.parametrize("name", GPU_PROBES)
def test_gpu_probes_use_the_one_serve_lock_protocol(name: str) -> None:
    """A copied mkdir loop would reintroduce the owner-publication gap."""
    body = (ROOT / "experiments" / name).read_text()
    assert "serve_lock_acquire" in body
    assert 'until mkdir "$SERVE_LOCK"' not in body
