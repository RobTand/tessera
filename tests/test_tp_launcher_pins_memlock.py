"""The two-box RoCE launcher must lift the container's pinned-memory ceiling.

A container inherits the docker daemon's ``RLIMIT_MEMLOCK``, which is 8 MB on
both Sparks while the hosts are unlimited.  NCCL's IB transport registers its
per-channel proxy buffers with ``ibv_reg_mr``, the kernel charges those pinned
pages against that limit, and above it the registration returns ENOMEM --
measured, six interleaved trials per arm on a two-rank NCCL all-reduce over
these RoCE devices: 0/6 without the flag, 6/6 with it (tessera#550).

The assertion runs the launcher's own ``tp_docker_args`` rather than reading
the file for a string, so it fails the way the serve would: on the argument
vector the container is actually given.  ``--device /dev/infiniband`` is
asserted beside it because the ceiling only matters once a rank can reach the
RDMA devices at all, and the two flags are one requirement.
"""
from pathlib import Path
import subprocess

LAUNCHER = Path(__file__).resolve().parents[1] / "experiments" / "tessera_plugin_served_tp.sh"


def _docker_args() -> list[str]:
    """The argument vector the launcher hands ``docker run``, really evaluated.

    Only the two argument-building functions are lifted out; sourcing the whole
    file would start a serve.
    """
    script = f"""
    set -uo pipefail
    TS=/nonexistent-checkout
    EXT=/nonexistent-ext
    MODE=resident
    eval "$(sed -n '/^tp_fabric_env()/,/^}}/p' {LAUNCHER})"
    eval "$(sed -n '/^tp_docker_args()/,/^}}/p' {LAUNCHER})"
    tp_docker_args
    """
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def test_tp_launcher_lifts_the_memlock_ceiling() -> None:
    args = _docker_args()
    assert "--ulimit" in args, args
    ulimits = [args[i + 1] for i, a in enumerate(args) if a == "--ulimit" and i + 1 < len(args)]
    assert "memlock=-1:-1" in ulimits, ulimits


def test_tp_launcher_passes_the_rdma_devices() -> None:
    args = _docker_args()
    assert "/dev/infiniband" in args, args
