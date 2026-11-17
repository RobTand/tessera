"""Prepare the F dense split canary. No command in this module submits a job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from experiments.numerics_rows.common import Invalid, sha256, write_json
from experiments.numerics_rows.prepare_job import stage_population
from experiments.numerics_rows.start_env import container_options, worker_command

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "272507f18de39fc90f03c1588373c031a9e43717"
RULE = Path("/mnt/shared/flux-jobs/tessera-1717-rule/pass-rule-v2.md")
RULE_SHA = "ec8d4b3bbbb8df5f6c1be860bd20204f0f0a14ff3faa26eadc8c5e764fff5d53"
FREEZE = RULE.with_name("freeze-record-v2.json")
MODULE = "model.language_model.layers.32.mlp.shared_experts.down_proj"
INPUTS = ("source", "population", "manifest", "build", "build_provenance", "q1", "h1", "freeze_addendum")


def bind(root, paths):
    root = Path(root)
    config = json.loads((root / "campaign.json").read_text())
    if set(paths) != set(INPUTS) or any(not str(p).startswith("/mnt/shared/") or not Path(p).exists() for p in paths.values()):
        raise Invalid("bind each F chain input from an existing shared path")
    source = Path(paths["source"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != SOURCE:
        raise Invalid("split canary source is not F")
    if sha256(RULE) != RULE_SHA:
        raise Invalid("split canary rule is not frozen v2")
    from experiments.numerics_rows.serving_build import check_retained_source
    check_retained_source(source, Path(paths["build"]), Path(paths["build_provenance"]))
    from experiments.numerics_rows.row import bind_freeze_addendum, check_rule_freeze
    from experiments.numerics_rows.launch_state import LEVERS
    freeze = bind_freeze_addendum(json.loads(FREEZE.read_text()), FREEZE, RULE, paths["freeze_addendum"])
    check_rule_freeze(RULE, freeze, canary=True, launch_levers_sha256=sha256(LEVERS))
    population, mount = stage_population(paths["population"], root)
    config.update(paths={k: str(v) for k, v in paths.items()}, digests={k: sha256(v) for k, v in paths.items() if k != "source"},
                  staged_population=str(population), population_mount=mount, bound=True)
    write_json(root / "campaign.json", config)
    for phase in ("prepare", "capture", "compare"):
        directory = root / phase
        extra = []
        if phase != "prepare":
            extra += ["--prepare-artifact", str(root / "prepare/results/prepare")]
        if phase == "compare":
            extra += ["--capture-artifact", str(root / "capture/results/capture")]
        arguments = ["--image", "N", "--name", "N-A8S-L32-r0-split-canary", "--canary", "--phase", phase,
                     "--out", str(directory / "results"), "--rule", str(RULE), "--freeze", str(FREEZE),
                     "--module", MODULE, "--tp-size", "2", "--tp-rank", "0"]
        for key, value in paths.items():
            arguments += ["--" + key.replace("_", "-"), str(population) if key == "population" else str(value)]
        arguments += extra
        payload = worker_command(source, "experiments.numerics_rows.row", arguments)
        (directory / "payload.sh").write_text(f"#!/bin/bash\nset -euo pipefail\nexec {payload}\n")
    return root


def verify(root):
    config = json.loads((root / "campaign.json").read_text())
    if not config.get("bound") or config["source_revision"] != SOURCE or config["rule_sha256"] != RULE_SHA:
        raise Invalid("split canary needs its F chain and lever freeze addendum")
    if sha256(RULE) != RULE_SHA or sha256(FREEZE) != config["freeze_sha256"]:
        raise Invalid("frozen v2 rule or record changed")
    checks = json.loads(Path(config["cpu_check"]).read_text())
    if checks.get("verdict") != "PASS" or checks.get("source") != config["tool_commit"]:
        raise Invalid("split canary requires a passing CPU check at its tool commit")
    for name, digest in config["digests"].items():
        if sha256(config["paths"][name]) != digest:
            raise Invalid("split canary bound input changed: " + name)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config["paths"]["source"], text=True).strip() != SOURCE:
        raise Invalid("split canary source moved away from F")
    return config


def prepare(cpu_check):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"/mnt/shared/flux-jobs/tessera-1717-SPLIT-CANARY-N-{revision[:8]}-{stamp}")
    out.mkdir()
    subprocess.run(["git", "clone", "--bare", "--no-hardlinks", str(ROOT), str(out / "repository.git")], check=True)
    write_json(out / "campaign.json", {"tool_commit": revision, "source_revision": SOURCE, "rule": str(RULE),
        "rule_sha256": RULE_SHA, "freeze": str(FREEZE), "freeze_sha256": sha256(FREEZE),
        "module": MODULE, "cuts": [{"size": 2, "rank": 0}], "canary": True, "bound": False,
        "paths": dict.fromkeys(INPUTS), "cpu_check": str(cpu_check)})
    shutil.copyfile(__file__, out / "prepare.py")
    for phase, minutes in (("prepare", 1440), ("capture", 15), ("compare", 120)):
        job = out / phase
        job.mkdir()
        gpu = phase == "capture"
        options = shlex.join(container_options("$task_scratch")).replace("'$task_scratch", '"$task_scratch').replace(":rw'", ':rw"').replace(":ro'", ':ro"')
        bootstrap = f"PYTHONPATH={job}/code python3 -m experiments.numerics_rows.prepare_split --verify {out}"
        prerequisites = ""
        for prior in (("prepare",) if gpu else ("prepare", "capture") if phase == "compare" else ()):
            prerequisites += f"test -s {out}/{prior}/results/{prior}/phase.json\n"
        (job / "job.sh").write_text(f'''#!/bin/bash
set -euo pipefail
umask 077
cd {job}
task_scratch=$(mktemp -d /var/tmp/tessera-dense-phase.XXXXXX)
name=tessera-dense-{phase}-"$FLUX_JOB_ID"
trap 'rc=$?; podman rm -f "$name" >/dev/null 2>&1 || true; rm -rf -- "$task_scratch"; printf "%s\\n" "$rc" > tool-exit.txt; exit "$rc"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
mkdir -p results "$task_scratch/home"
grep Cpus_allowed_list /proc/self/status > results/affinity.txt
git clone --shared {out}/repository.git code
git -C code checkout --detach {revision}
test "$(git -C code rev-parse HEAD)" = {revision}
{bootstrap}
{prerequisites}podman image inspect acc684c06ec7 > "$task_scratch/image-inspect.json"
# The population draft mount is supplied from the bound, reviewed campaign file.
mapfile -t population_mount < <(PYTHONPATH=code python3 -c 'import json,shlex; print("\\n".join(shlex.split(json.load(open("{out}/campaign.json"))["population_mount"])))')
rc=0
timeout --signal=TERM --kill-after=20s {minutes * 60 - 60}s podman run --rm --name "$name" --pull=never --cgroups=disabled --network=none {'--device nvidia.com/gpu=all' if gpu else ''} -v /mnt/shared:/mnt/shared:ro -v {job}:{job}:rw -v {job}/code:/numerics:ro "${{population_mount[@]}}" {options} --entrypoint /bin/bash acc684c06ec7 -euo pipefail {job}/payload.sh > results/payload.stdout 2> results/payload.stderr || rc=$?
printf '%s\\n' "$rc" > results/podman-exit.txt
printf '%s\\n' "$rc" > results/payload-exit.txt
test "$rc" -eq 0
test -s results/{phase}/phase.json
''')
        base = Path("/mnt/shared/flux-jobs/tessera-1717-RECEIPT-CPU-7d69093d-20261116T235639Z")
        submit = (base / "submit-after-review.sh").read_text().replace(str(base), str(job))
        submit = submit.replace("tessera-1717-RECEIPT-CPU", "tessera-1717-DENSE-" + phase.upper())
        submit = submit.replace("-t30m", f"-t{minutes}m").replace("core=4;memory=16", "core=4;memory=32" + (";gpu=1" if gpu else "")).replace("host:dl380g10", "host:sparklina")
        (job / "submit-after-review.sh").write_text(submit)
    (out / "bind-inputs.py").write_text(f'''import argparse, json, sys
import subprocess
subprocess.run(["git", "clone", "--shared", {str(out / 'repository.git')!r}, {str(out / 'binding-code')!r}], check=True)
subprocess.run(["git", "-C", {str(out / 'binding-code')!r}, "checkout", "--detach", {revision!r}], check=True)
sys.path.insert(0, {str(out / 'binding-code')!r})
from experiments.numerics_rows.prepare_split import bind
p=argparse.ArgumentParser()
p.add_argument("paths", help="JSON object with the eight F chain paths")
a=p.parse_args()
bind({str(out)!r}, json.load(open(a.paths)))
''')
    (out / "README.md").write_text(f'''# Split dense canary

Prepared only. No job was submitted. Tool {revision}; Tessera F {SOURCE}.
Rule {RULE}, SHA-256 {RULE_SHA}. Freeze record {FREEZE}.
The lever freeze addendum and the clean F P/Q1/H1/C1 chain are required inputs.
No old source tree or receipt can fill them. Run bind-inputs.py with their eight
shared paths, then obtain Fleet OK. Each job rechecks every bound file and F.
The CPU prerequisite is {cpu_check}. It must report PASS at this tool commit.
The canary flag is true in each receipt. The join must refuse this canary.

| Job | Cores | Memory GiB | GPU | Limit | Time basis |
|---|---:|---:|---:|---|---|
| prepare | 4 | 32 | 0 | 24 h | Unknown. Old host work exceeded 88 min and was unfinished. |
| capture | 4 | 32 | 1 | 15 min | Provisional: old setup + identity + resident time about 4.6 min. |
| compare | 4 | 32 | 0 | 120 min | Unknown. Streamed comparisons, no reference construction. |

All jobs use aarch64 IMG-N on sparklina. These are prepared limits, not new measurements.
No case, M, distribution or seed is removed. Prepare records an in-process
cProfile of all diagnostic reference work, with per-case progress and phase times.
Capture writes the gate-complete checkpoint before diagnostics. It runs only
device work plus preflight, code audits and byte I/O. Each case is profiled.
The receipt records NVML resident time and power, torch memory, and cgroup
whole-job peak memory. Inspect the coverage and errors beside each number.
A handled TERM or INT writes STOPPED and preserves known FAIL. Only compare
can set finalization_finished=true. Every phase binds the full unit scope,
common identities and content hashes. Reused module artifacts are refused.
Run prepare, capture, compare in that order. Read each Flux status, exit files,
phase.json, payload.stdout and payload.stderr. Do not fan out counted rows
until the canary and its compare pass, reviews finish and Fleet approves.
''')
    print(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--cpu-check", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    elif args.cpu_check:
        prepare(args.cpu_check)
    else:
        parser.error("prepare requires --cpu-check")
