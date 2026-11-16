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


def prepare(step, image, extra=()):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"/mnt/shared/flux-jobs/tessera-1717-{step}-{revision[:8]}-{stamp}")
    out.mkdir()
    subprocess.run(["git", "clone", "--bare", "--no-hardlinks", str(ROOT), str(out / "repository.git")], check=True)
    command = {"P": "build", "Q1": "quantizers", "H1": "h1", "C1": "cases", "CPU": "cpu_checks"}[step.split("-")[0]]
    population_mount = ""
    if command == "cases":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--population", type=Path, required=True)
        inputs, extra = parser.parse_known_args(extra)
        population, population_mount = stage_population(inputs.population, out)
        extra += ["--population", str(population)]
    gpu = step.startswith(("Q1", "H1"))
    shape, minutes = ("core=2;memory=16;gpu=1", 9) if gpu else ("core=4;memory=8", 20)
    host = "sparklina"
    if command == "cases":
        shape, minutes = "core=4;memory=32", 60
    if command == "cpu_checks":
        shape, minutes, host = "core=4;memory=16", 30, "dl380g10"
    payload = shlex.join(["python3", "-m", f"experiments.numerics_rows.{command}", "--source", str(out / "cell"),
                          "--image", image, "--out", str(out / "results"), *extra])
    if gpu:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--build", type=Path, required=True)
        build = parser.parse_known_args(extra)[0].build
        (out / "payload.sh").write_text(f'''#!/bin/bash
set -euo pipefail
exec > {out}/results/payload.stdout 2> {out}/results/payload.stderr
rc=0
{payload} || rc=$?
printf '%s\\n' "$rc" > {out}/results/payload-exit.txt
exit "$rc"
''')
        payload = f"/bin/bash {out}/payload.sh"
    name = f'tessera-1717-{step.lower()}-"$FLUX_JOB_ID"'
    container = (f"timeout --signal=TERM --kill-after=20s {480 if gpu else minutes * 60 - 60}s "
                 f"podman run --rm --name {name} --pull=never --cgroups=disabled --network=none "
                 f"{'--device nvidia.com/gpu=all ' if gpu else ''}"
                 f"-v /mnt/shared:/mnt/shared:ro -v {out}:{out}:rw -w {out}/code "
                 f"{population_mount}"
                 f"--env HOME={out}/home --env TMPDIR={out}/scratch --env XDG_CACHE_HOME={out}/cache "
                 f"--env CUDA_CACHE_PATH={out}/cuda-cache --env VLLM_CACHE_ROOT={out}/vllm-cache "
                 f"--env TORCH_EXTENSIONS_DIR={out}/extensions --env TRITON_CACHE_DIR={out}/triton "
                 f"--env TESSERA_NUMERICS_IMAGE_INSPECT={out}/image-inspect.json "
                 '--env FLUX_JOB_ID="$FLUX_JOB_ID" '
                 "--env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 "
                 f"--env MAX_JOBS={2 if gpu else 4} --env TESSERA_PLATFORM_TOKEN=sm_121 "
                 f"--entrypoint /bin/bash {IMAGES[image]['local_id']} -euo pipefail -c {shlex.quote(payload)}")
    if gpu:
        container = f'''rc=0
{container} > results/podman.stdout 2> results/podman.stderr || rc=$?
printf '%s\\n' "$rc" > results/podman-exit.txt
test "$rc" -eq 0
test "$(cat results/payload-exit.txt)" = 0
python3 -m experiments.numerics_rows.job_receipt {step.split('-')[0]} --out {out}/results --build {shlex.quote(str(build))}
'''
    if command == "cpu_checks":
        container = f"env HOME={out}/home /home/rob/.local/bin/uv run --python 3.12 --no-project --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --with-requirements code/tools/ci/gate_torch_requirements.txt --with-requirements code/tools/ci/gate_requirements.txt {payload}"
    script = f'''#!/bin/bash
set -euo pipefail
cd {out}
trap 'rc=$?; podman rm -f {name} >/dev/null 2>&1 || true; rm -rf -- scratch; test ! -e scratch || rc=3; printf "%s\\n" "$rc" > tool-exit.txt; exit "$rc"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export TMPDIR={out}/scratch
export UV_CACHE_DIR={out}/cache/uv UV_PYTHON_INSTALL_DIR={out}/python
export XDG_CACHE_HOME={out}/cache
export PATH=/home/rob/.local/bin:$PATH
export PYTHONPYCACHEPREFIX={out}/cache/pycache
export PYTHONPATH={out}/cell/src:{out}/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS={2 if gpu else 4}
mkdir -p scratch cache results extensions triton home cuda-cache vllm-cache
# The cell source stays separate from the implementation source.
git clone --shared repository.git code
git -C code checkout --detach {revision}
git clone --shared repository.git cell
git -C cell checkout --detach 8bf496ab
printf '%s  %s\\n' 8a1536dfbc3df70edbdc37001a0db6947e97d9263926bc85a4d4d3ca231eae4f cell/src/tessera/serving/csrc/routed_fused_window.cu | sha256sum -c -
{f"podman image inspect {IMAGES[image]['local_id']} > image-inspect.json" if command != "cpu_checks" else ""}
{container}
'''
    (out / "job.sh").write_text(script)
    submit = (f"flux submit --job-name=tessera-1717-{step} -n1 -t{minutes}m --resources-shape='slot/[{shape}]' --requires=host:{host} "
              f"--cwd={out} --input=/dev/null '--env=-*' --env=PATH=/usr/bin:/bin "
              "--env=HOME=/home/rob --env=XDG_RUNTIME_DIR=/run/user/1000 "
              "--env=FLUX_F58_FORCE_ASCII=1 --env=FLUX_SCRATCH=/var/tmp "
              "--env=OMP_NUM_THREADS=1 --env=MKL_NUM_THREADS=1 --env=OPENBLAS_NUM_THREADS=1 "
              f"--output={out}/stdout --error={out}/stderr /bin/bash {out}/job.sh > {out}/jobid")
    (out / "README.md").write_text(
        f"# {step}\n\nSource: {revision}. No work has been submitted.\n\n"
        "The lead must obtain Fleet OK before submission.\n"
        "GPU work cannot run from 12:15 through 13:45 UTC on 2026-11-16.\n\n"
        f"```bash\n{submit}\n```\n\nRead tool-exit.txt, the Flux status, stdout, and stderr after the job.\n")
    (out / "submit-after-review.sh").write_text(f'''#!/bin/bash
set -euo pipefail
root={out}
read -r FLUX_PREFIX FLUX_RUN < /mnt/shared/flux-migration/INSTANCE
export PATH=$FLUX_PREFIX/install/bin:$FLUX_PREFIX/deps/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$FLUX_PREFIX/install/lib:$FLUX_PREFIX/deps/lib
export FLUX_URI=$(/mnt/shared/flux-migration/flux-uri.sh)
test ! -e "$root/jobid"
cp /mnt/shared/flux-migration/INSTANCE "$root/instance.txt"
{submit} 2> "$root/submit.stderr"
cat "$root/jobid"
''')
    print(out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("P-G", "P-N", "Q1-G", "Q1-N", "H1-G", "H1-N", "C1-G", "C1-N", "CPU"))
    p.add_argument("--image", choices=IMAGES, required=True)
    args, extra = p.parse_known_args()
    prepare(args.step, args.image, extra)
