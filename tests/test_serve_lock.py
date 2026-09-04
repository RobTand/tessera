"""The host-local serve lock is atomic and released by owner death."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
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
    """Pause two dead-owner reapers at the vulnerable compare/unlink edge."""
    env = _environment(tmp_path)
    bindir = tmp_path / "bin"
    sync = tmp_path / "sync"
    sync.mkdir(exist_ok=True)
    readlink = bindir / "readlink"
    readlink.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "counter=$SERVE_LOCK_RACE_SYNC/readlink_$SERVE_LOCK_RACE_ID\n"
        "count=0; [ ! -f \"$counter\" ] || read -r count < \"$counter\"\n"
        "count=$((count + 1)); printf '%s\\n' \"$count\" > \"$counter\"\n"
        "value=$(/usr/bin/readlink \"$@\")\n"
        "if [ \"$SERVE_LOCK_RACE_ID\" = B ] && [ \"$count\" = 2 ]; then\n"
        "  : > \"$SERVE_LOCK_RACE_SYNC/b_captured_old_token\"\n"
        "  for _ in $(seq 1 500); do\n"
        "    [ ! -e \"$SERVE_LOCK_RACE_SYNC/a_acquired\" ] || break\n"
        "    sleep 0.01\n"
        "  done\n"
        "  [ -e \"$SERVE_LOCK_RACE_SYNC/a_acquired\" ] || exit 70\n"
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
        "  for _ in $(seq 1 100); do\n"
        "    [ ! -e \"$SERVE_LOCK_RACE_SYNC/b_captured_old_token\" ] || break\n"
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


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            pytest.fail(f"lock holder exited {process.returncode}: {process.stderr.read()}")
        time.sleep(0.01)
    pytest.fail("lock holder did not publish its ready marker")


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
            ["timeout", "3", "bash", "-euo", "pipefail", "-c", contender_script],
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
        ["timeout", "3", "bash", "-euo", "pipefail", "-c", script],
        env=_environment(tmp_path),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "acquired"
    assert not os.path.lexists(lock)


def test_two_dead_owner_reapers_cannot_unlink_the_new_holder(tmp_path: Path) -> None:
    """Compare+unlink and the replacement acquire are one serialized transition."""
    lock = tmp_path / "serve.lock"
    sync = tmp_path / "sync"
    lock.symlink_to(f"999999999:1:{'b' * 32}")

    def start(contender: str) -> subprocess.Popen[str]:
        ready = sync / f"{contender.lower()}_acquired"
        script = (
            f'SERVE_LOCK="{lock}"; SERVE_LOCK_TIMEOUT=10; SERVE_LOCK_POLL_S=0.01; '
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
        _wait_for(sync / "a_acquired", first)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not (sync / "b_acquired").exists():
            if second.poll() is not None:
                pytest.fail(
                    f"second contender exited {second.returncode}: {second.stderr.read()}"
                )
            time.sleep(0.01)
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
