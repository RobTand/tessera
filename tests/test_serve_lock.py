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


@pytest.mark.parametrize("name", GPU_PROBES)
def test_gpu_probes_use_the_one_serve_lock_protocol(name: str) -> None:
    """A copied mkdir loop would reintroduce the owner-publication gap."""
    body = (ROOT / "experiments" / name).read_text()
    assert "serve_lock_acquire" in body
    assert 'until mkdir "$SERVE_LOCK"' not in body
