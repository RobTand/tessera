"""Prepare a Flux job. This command never submits work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from experiments.numerics_rows.common import IMAGES, sha256
from experiments.numerics_rows.serving_build import source_layout
from experiments.numerics_rows.start_env import container_options, worker_command

ROOT = Path(__file__).resolve().parents[2]


def stage_population(path, out):
    """Copy signed bytes and their bound draft. Mount the draft at its signed path."""
    path = Path(path)
    review = json.loads(path.read_text())["review"]
    original = Path(review["source_file"])
    draft = path.parent / "drafts" / original.name
    if not draft.is_file():
        draft = original
    if not original.is_absolute() or sha256(draft) != review["source_sha256"]:
        raise ValueError("population draft identity differs")
    inputs = out / "inputs"
    inputs.mkdir()
    shutil.copyfile(path, inputs / "population.json")
    shutil.copyfile(draft, inputs / "draft.json")
    return inputs / "population.json", f"-v {shlex.quote(str(inputs / 'draft.json') + ':' + str(original) + ':ro')} "


def prepare(step, image, extra=(), *, source_revision):
    if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
        raise ValueError("source revision must be one full lowercase commit SHA")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"/mnt/shared/flux-jobs/tessera-1717-{step}-{revision[:8]}-{stamp}")
    out.mkdir()
    subprocess.run(["git", "clone", "--bare", "--no-hardlinks", str(ROOT), str(out / "repository.git")], check=True)
    command = {"P": "build", "Q1": "quantizers", "H1": "h1", "C1": "cases", "ROW": "row", "CPU": "cpu_checks"}[step.split("-")[0]]
    extra = list(extra)
    if command == "build":
        extra += ["--revision", source_revision]
    elif command in ("cases", "cpu_checks"):
        extra += ["--source-revision", source_revision]
    source = out / "cell"
    population_mount = ""
    if command in ("cases", "row"):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--population", type=Path, required=True)
        inputs, extra = parser.parse_known_args(extra)
        population, population_mount = stage_population(inputs.population, out)
        extra += ["--population", str(population)]
    if command == "row":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--build", type=Path, required=True)
        parser.add_argument("--build-provenance", type=Path, required=True)
        inputs = parser.parse_known_args(extra)[0]
        source, _ = source_layout(inputs.build, inputs.build_provenance)
    gpu = step.startswith(("Q1", "H1", "ROW"))
    shape, minutes = ("core=2;memory=16;gpu=1", 9) if gpu else ("core=4;memory=8", 20)
    host = "sparklina"
    if command == "cases":
        shape, minutes = "core=4;memory=32", 60
    if command == "row":
        shape, minutes = "core=4;memory=32;gpu=1", 90
    if command == "cpu_checks":
        shape, minutes, host = "core=4;memory=16", 30, "dl380g10"
    arguments = ["--source", str(source), "--image", image, "--out", str(out / "results"), *extra]
    if command == "row":
        arguments += ["--phase-instance", str(out / "results/instance.json")]
    payload = worker_command(source, f"experiments.numerics_rows.{command}", arguments)
    if gpu:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--build", type=Path, required=True)
        build = parser.parse_known_args(extra)[0].build
    name = f'tessera-1717-{step.lower()}-"$FLUX_JOB_ID"'
    # All worker starts use the same shell depth, working directory and constants.
    (out / "payload.sh").write_text(f'''#!/bin/bash
set -euo pipefail
exec > {out}/results/payload.stdout 2> {out}/results/payload.stderr
exec {payload}
''')
    options = shlex.join(container_options("$task_scratch")).replace("'$task_scratch", '"$task_scratch').replace(":rw'", ':rw"').replace(":ro'", ':ro"')
    container = (f"timeout --signal=TERM --kill-after=20s {480 if gpu and command != 'row' else minutes * 60 - 60}s "
                 f"podman run --rm --name {name} --pull=never --cgroups=disabled --network=none "
                 f"{'--device nvidia.com/gpu=all ' if gpu else ''}"
                 f"-v /mnt/shared:/mnt/shared:ro -v {out}:{out}:rw -v {out}/code:/numerics:ro "
                 f"{population_mount}{options} "
                 f"--entrypoint /bin/bash {IMAGES[image]['local_id']} -euo pipefail {out}/payload.sh")
    container = f'''rc=0
{container} > results/podman.stdout 2> results/podman.stderr || rc=$?
printf '%s\\n' "$rc" > results/podman-exit.txt
printf '%s\\n' "$rc" > results/payload-exit.txt
test "$rc" -eq 0
'''
    if gpu:
        container += (f"test -s {out}/results/numerics_row.json\n" if command == "row" else
                      f"python3 -m experiments.numerics_rows.job_receipt {step.split('-')[0]} --out {out}/results --build {shlex.quote(str(build))}\n")
    if command == "cpu_checks":
        payload = shlex.join(["python3", "-m", f"experiments.numerics_rows.{command}", *arguments])
        container = f"env HOME={out}/home /home/rob/.local/bin/uv run --python 3.12 --no-project --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --with-requirements code/tools/ci/gate_torch_requirements.txt --with-requirements code/tools/ci/gate_requirements.txt {payload}"
    cell_checkout = "" if command == "row" else f"git clone --shared repository.git cell\ngit -C cell checkout --detach {source_revision}"
    script = f'''#!/bin/bash
set -euo pipefail
umask 077
cd {out}
task_scratch=$(mktemp -d /var/tmp/tessera-1717-worker.XXXXXX)
trap 'rc=$?; podman rm -f {name} >/dev/null 2>&1 || true; rm -rf -- "$task_scratch"; test ! -e "$task_scratch" || rc=3; printf "%s\\n" "$rc" > tool-exit.txt; exit "$rc"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export TMPDIR="$task_scratch"
export UV_CACHE_DIR={out}/cache/uv UV_PYTHON_INSTALL_DIR={out}/python
export XDG_CACHE_HOME={out}/cache
export PATH=/home/rob/.local/bin:$PATH
export PYTHONPYCACHEPREFIX={out}/cache/pycache
export PYTHONPATH={source}/src:{out}/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS={2 if gpu else 4}
mkdir -p cache results home "$task_scratch/home"
python3 instance.py check {out} > results/instance.json
# The cell source stays separate from the implementation source.
git clone --shared repository.git code
git -C code checkout --detach {revision}
test "$(git -C code rev-parse HEAD)" = {revision}
{cell_checkout}
test "$(git -C {source} rev-parse HEAD)" = {source_revision}
printf '%s\\n' {source_revision} > results/tessera-commit.txt
printf '%s  %s\\n' 8a1536dfbc3df70edbdc37001a0db6947e97d9263926bc85a4d4d3ca231eae4f {source}/src/tessera/serving/csrc/routed_fused_window.cu | sha256sum -c -
{f"podman image inspect {IMAGES[image]['local_id']} > $task_scratch/image-inspect.json" if command != "cpu_checks" else ""}
{container}
'''
    (out / "job.sh").write_text(script)
    submit = (f"flux submit --job-name=tessera-1717-{step} -n1 -t{minutes}m --resources-shape='slot/[{shape}]' --requires=host:{host} "
              f"--cwd={out} --input=/dev/null '--env=-*' --env=PATH=/usr/bin:/bin "
              "--env=HOME=/home/rob --env=XDG_RUNTIME_DIR=/run/user/1000 "
              "--env=FLUX_F58_FORCE_ASCII=1 --env=FLUX_SCRATCH=/var/tmp "
              "--env=OMP_NUM_THREADS=1 --env=MKL_NUM_THREADS=1 --env=OPENBLAS_NUM_THREADS=1 "
              f"--output={out}/stdout --error={out}/stderr /bin/bash {out}/job.sh")
    (out / "README.md").write_text(
        f"# {step}\n\nTool: {revision}. Tessera: {source_revision}. No work has been submitted.\n\n"
        "The lead must obtain Fleet OK before submission.\n"
        "GPU work cannot run from 12:15 through 13:45 UTC on 2026-11-16.\n\n"
        f"```bash\n{submit}\n```\n\nRead tool-exit.txt, the Flux status, stdout, and stderr after the job.\n")
    (out / "submit-after-review.sh").write_text(f'''#!/bin/bash
set -euo pipefail
root={out}
flux() {{ python3 "$root/instance.py" submit "$root" "$@"; }}
test ! -e "$root/jobid"
{submit} 2> "$root/submit.stderr"
cat "$root/jobid"
''')
    shutil.copyfile(Path(__file__).with_name('phase_instance.py'), out / 'instance.py')
    print(out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("P-G", "P-N", "Q1-G", "Q1-N", "H1-G", "H1-N", "C1-G", "C1-N", "ROW-G", "ROW-N", "CPU"))
    p.add_argument("--image", choices=IMAGES, required=True)
    p.add_argument("--source-revision", required=True)
    args, extra = p.parse_known_args()
    prepare(args.step, args.image, extra, source_revision=args.source_revision)
