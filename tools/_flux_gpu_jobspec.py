"""Render, and never run, a stock-shape Flux GPU submission line.

GPU work on Flux needs the owner's per-run approval, so this module only prints
the ``flux submit`` argv a caller would run: the stock slot shape with one GPU,
GPU-eligible hosts, a closed environment (``--env=-*`` plus an explicit allowlist)
and explicit walltime and I/O paths. Create the named output directory before using
the printed line. It imports no process runner.

    python3 tools/_flux_gpu_jobspec.py --cores 8 --memory-gib 16 --walltime 45m --out /mnt/shared/my-owned-job \\
        --env PROFILE=0 -- bash experiments/example.sh arg
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _suite_flux import resource_shape  # noqa: E402

BASE_ENV = ("PATH=/usr/bin:/bin",)


def render(command, *, cores, memory_gib, walltime, env=(), hosts=None, directory=None, cwd=None):
    """Return the flux submit argv for ``command``. Nothing is executed."""
    if not command or not walltime or cores < 1 or memory_gib < 1:
        raise ValueError("a GPU jobspec needs a command, a walltime, cores >= 1 and memory >= 1")
    if directory is None or not Path(directory).is_absolute() or not Path(directory).is_relative_to("/mnt/shared"):
        raise ValueError("GPU output directory must be an owned unique path under /mnt/shared")
    directory = Path(directory)
    cwd = str(Path.cwd() if cwd is None else Path(cwd))
    allowlist = [*BASE_ENV, *env]
    for entry in allowlist:
        if "=" not in entry or entry.startswith("="):
            raise ValueError(f"environment allowlist entry is not NAME=VALUE: {entry!r}")
    return ["flux", "submit", "--env=-*", *[f"--env={entry}" for entry in allowlist],
            f"--resources-shape={resource_shape(cores, memory_gib, 1)}",
            *([f"--requires=host:{hosts}"] if hosts else []), "-t", walltime,
            "--cwd", cwd, "--input=/dev/null", f"--output={directory}/stdout", f"--error={directory}/stderr",
            "--", *command]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--memory-gib", type=int, required=True)
    parser.add_argument("--walltime", required=True, help="explicit walltime, e.g. 45m")
    parser.add_argument("--out", required=True, help="unique owned shared output directory; create it before using the line")
    parser.add_argument("--cwd", default=str(Path.cwd()), help="payload working directory")
    parser.add_argument("--env", action="append", default=[], help="NAME=VALUE allowed into the job")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    print(shlex.join(render(command, cores=args.cores, memory_gib=args.memory_gib,
                            walltime=args.walltime, env=args.env, directory=args.out, cwd=args.cwd)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
