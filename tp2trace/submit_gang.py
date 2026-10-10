#!/usr/bin/env python3
"""tessera#1185: submit the TP2 trace gang (runs on celestia, OFF REPO).

Usage: submit_gang.py --run NAME [--small] [--timeout-s S]
       [--evidence PACKET]

--small submits the D30 probe gang (full artifact, tiny context, two
profiled steps); without it, the full measurement gang. --evidence names
a pbevidence packet from a gb10 worker; with it the members seal
measurement + host_class, without it they run exclusive on pinned hosts
and the doc states the admission. Prints the pbgang JSON line (group +
member keys) for wait.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[0]
PB = Path("/mnt/shared/prismabuild-fleet/repo/tools")
IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5")
FULL_ARTIFACT = ("/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928"
                 "/release-t8/exported")
HOSTS = ("sparklina", "sparky")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=None)
    ap.add_argument("--evidence", default=None)
    args = ap.parse_args(argv)

    sealed = bool(args.evidence and Path(args.evidence).is_file())
    print(f"[tp2trace] sealed measurement: {sealed}", flush=True)

    env = {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MAX_JOBS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TP2TRACE_RUN": args.run,
        "TP2TRACE_ARTIFACT": FULL_ARTIFACT,
        "TP2TRACE_PLUGIN": "tp2trace/plugin-83460680",
        "TP2TRACE_PLUGIN_COMMIT":
            "83460680ed84e33c82eb62b31345381cc151aa58",
    }
    if args.small:
        env.update(TP2TRACE_MLM="512", TP2TRACE_MBT="512",
                   TP2TRACE_UTIL="0.3", TP2TRACE_KVBYTES="268435456",
                   TP2TRACE_STEPS="1", TP2TRACE_WANT="64",
                   TP2TRACE_WARM_TOKENS="32", TP2TRACE_MEASURE_TOKENS="64")
        timeout = args.timeout_s or 3600
        reason = ("issuegraph tessera#1185 D30 small TP2 gang: full artifact "
                  "at tiny context, one profiled step")
    else:
        timeout = args.timeout_s or 8100
        reason = ("issuegraph tessera#1185 TP2 prefill trace: full T8R "
                  "release at 2048-token chunks, torch profiler both ranks")
    members = []
    for rank, host in enumerate(HOSTS):
        member = {
            "tag": host,
            "cwd": str(WORKTREE),
            "argv": ["python3", "tp2trace/rank_trace.py",
                     "--gang-rank", str(rank)],
            "demand": {"cpu": 8, "mem_gb": 100, "gpu": 1},
            "gpu_memory_gb": 96,
            "exclusive": True,
            "max_attempts": 1,
            "container_images": [IMAGE],
            "env": env,
        }
        if sealed:
            member.update(measurement=True, host_class="gb10")
        members.append(member)
    manifest = {"priority": 0, "priority_reason": reason,
                "timeout_s": timeout, "members": members}
    path = HERE / f"gang-{args.run}.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    cmd = [sys.executable, str(PB / "pbgang.py"), "--manifest", str(path),
           "--cwd", str(WORKTREE)]
    if sealed:
        cmd += ["--target-evidence", args.evidence]
    done = subprocess.run(cmd, capture_output=True, text=True)
    print(done.stdout, end="")
    print(done.stderr, end="", file=sys.stderr)
    if done.returncode:
        raise SystemExit(f"pbgang refused: {done.returncode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
