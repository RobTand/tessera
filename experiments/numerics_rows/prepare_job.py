"""Prepare a Flux job. This command never submits work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess

from experiments.numerics_rows.common import IMAGES

ROOT = Path(__file__).resolve().parents[2]


def prepare(step, image, extra=()):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"/mnt/shared/flux-jobs/tessera-1717-{step}-{revision[:8]}-{stamp}")
    out.mkdir()
    subprocess.run(["git", "clone", "--bare", "--no-hardlinks", str(ROOT), str(out / "repository.git")], check=True)
    command = {"P": "build", "Q1": "quantizers", "H1": "h1", "CPU": "CPU"}[step.split("-")[0]]
    gpu = step.startswith(("Q1", "H1"))
    shape, minutes = ("core=2;memory=8;gpu=1", 9) if gpu else ("core=4;memory=8", 20)
    host = "sparklina"
    if command == "CPU":
        shape, minutes, host = "core=4;memory=16", 30, "dl380g10"
    payload = shlex.join(["python3", "-m", f"experiments.numerics_rows.{command}", "--source", str(out / "cell"),
                          "--image", image, "--out", str(out / "results"), *extra])
    container = (f"podman run --rm --pull=never --cgroups=disabled --network=none "
                 f"{'--device nvidia.com/gpu=all ' if gpu else ''}"
                 f"-v /mnt/shared:/mnt/shared:ro -v {out}:{out}:rw -w {out}/code "
                 f"--env TMPDIR={out}/scratch --env XDG_CACHE_HOME={out}/cache "
                 f"--env CUDA_CACHE_PATH={out}/cuda-cache --env VLLM_CACHE_ROOT={out}/vllm-cache "
                 f"--env TORCH_EXTENSIONS_DIR={out}/extensions --env TRITON_CACHE_DIR={out}/triton "
                 "--env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 "
                 f"--env MAX_JOBS={2 if gpu else 4} --env TESSERA_PLATFORM_TOKEN=sm_121 "
                 f"--entrypoint /bin/bash {IMAGES[image]['local_id']} -euo pipefail -c {shlex.quote(payload)}")
    script = f'''#!/bin/bash
set -euo pipefail
cd {out}
trap 'rc=$?; printf "%s\\n" "$rc" > tool-exit.txt; exit "$rc"' EXIT
export TMPDIR={out}/scratch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS={2 if gpu else 4}
mkdir -p scratch cache results extensions triton
# The cell source stays separate from the implementation source.
git clone --shared repository.git code
git -C code checkout --detach {revision}
git clone --shared repository.git cell
git -C cell checkout --detach 8bf496ab
printf '%s  %s\\n' 8a1536dfbc3df70edbdc37001a0db6947e97d9263926bc85a4d4d3ca231eae4f cell/src/tessera/serving/csrc/routed_fused_window.cu | sha256sum -c -
{container}
'''
    (out / "job.sh").write_text(script)
    submit = (f"flux submit -n1 -t{minutes}m --resources-shape='slot/[{shape}]' --requires=host:{host} "
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
    print(out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("P-G", "P-N", "Q1-G", "Q1-N", "H1-G", "H1-N", "CPU"))
    p.add_argument("--image", choices=IMAGES, required=True)
    args, extra = p.parse_known_args()
    prepare(args.step, args.image, extra)
