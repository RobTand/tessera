"""tessera#702: publish each arm of a plan as one PrismaBuild GPU action (arm.sh), detached.

  submit.py PLAN MANIFEST [--cwd CHECKOUT] [--only ARM,ARM] [--pbrun-python PY]

PLAN lines are ``ARM KEY=VALUE ...`` (``#`` comments); every KEY=VALUE becomes
an ``--env`` of the action. Each arm reserves one box's whole GPU
(``--exclusive``) and declares the serving image, so PrismaBuild places it on
any box holding the image. MANIFEST (JSON, kept outside the checkout so it does
not change the sealed snapshot) records each arm's action key and env.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

PBRUN = "/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py"
IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a")


def parse_plan(path: pathlib.Path) -> list[tuple[str, dict[str, str]]]:
    arms = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        arm, *pairs = line.split()
        env = {}
        for pair in pairs:
            key, sep, value = pair.partition("=")
            if not sep:
                raise SystemExit(f"{arm}: {pair!r} is not KEY=VALUE")
            env[key] = value
        arms.append((arm, env))
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=pathlib.Path)
    ap.add_argument("manifest", type=pathlib.Path)
    ap.add_argument("--cwd", default=str(pathlib.Path(__file__).resolve().parents[2]))
    ap.add_argument("--only", default="")
    ap.add_argument("--pbrun-python", default="/home/rob/tmp/pb-submit-celestia-20261003/bin/python")
    ap.add_argument("--timeout-s", type=int, default=5400)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="add to every arm's env (e.g. RECEIPTS=<a fresh root>)")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    extra = dict(pair.split("=", 1) for pair in args.set)
    for arm, env in parse_plan(args.plan):
        if only and arm not in only:
            continue
        env = {**env, **extra}
        command = [args.pbrun_python, PBRUN, "--cwd", args.cwd, "--anywhere", "--gpu",
                   "--exclusive", "--demand", "mem_gb=48", "--cpus", "4",
                   "--container-image", env.get("IMG", IMAGE),
                   "--timeout-s", str(args.timeout_s), "--detach"]
        for key, value in env.items():
            command += ["--env", f"{key}={value}"]
        command += ["--", "bash", "experiments/graph_attest_702/arm.sh", arm]
        done = subprocess.run(command, capture_output=True, text=True)
        tail = done.stdout.strip().splitlines()[-1:] or [""]
        try:
            key = json.loads(tail[0])["action_key"]
        except (ValueError, KeyError):
            print(f"{arm}: submission failed\n{done.stdout}\n{done.stderr}", file=sys.stderr)
            return 1
        manifest[arm] = {"action_key": key, "env": env, "cwd": args.cwd}
        args.manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        print(arm, key[:12], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
