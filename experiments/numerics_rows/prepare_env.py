"""Prepare the IMG-N environment proof. Never submit it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def prepare(source_revision):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_revision = subprocess.check_output(["git", "rev-parse", source_revision], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"/mnt/shared/flux-jobs/tessera-1717-ENV-PROOF-N-{revision[:8]}-{stamp}")
    out.mkdir()
    subprocess.run(["git", "clone", "--bare", "--no-hardlinks", str(ROOT), str(out / "repository.git")], check=True)
    (out / "job.sh").write_text(f'''#!/bin/bash
set -euo pipefail
umask 077
cd {out}
task_scratch=$(mktemp -d /var/tmp/tessera-1717-env-proof.XXXXXX)
cleanup() {{
 rc=$?
 for kind in row c1 serve-spec; do podman rm -f tessera-env-proof-{out.name}-"$kind" >/dev/null 2>&1 || true; done
 rm -rf -- "$task_scratch"
 test ! -e "$task_scratch" || rc=3
 printf '%s\\n' "$rc" > tool-exit.txt
 exit "$rc"
}}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
mkdir -p results "$task_scratch/home"
grep Cpus_allowed_list /proc/self/status > results/affinity.txt
git clone --shared repository.git code
git -C code checkout --detach {revision}
test "$(git -C code rev-parse HEAD)" = {revision}
git clone --shared repository.git cell
git -C cell checkout --detach {source_revision}
test "$(git -C cell rev-parse HEAD)" = {source_revision}
podman image inspect acc684c06ec7 > "$task_scratch/image-inspect.json"
PYTHONPATH={out}/code timeout --signal=TERM --kill-after=20s 1100s python3 -m experiments.numerics_rows.env_proof --root {out} --scratch "$task_scratch"
''')
    # Preserve the standing Flux connection recipe. This file only submits when the lead runs it.
    base = Path("/mnt/shared/flux-jobs/tessera-1717-RECEIPT-CPU-7d69093d-20261116T235639Z")
    submit = (base / "submit-after-review.sh").read_text().replace(str(base), str(out))
    submit = submit.replace("tessera-1717-RECEIPT-CPU", "tessera-1717-ENV-PROOF-N")
    submit = submit.replace("-t30m", "-t20m").replace("core=4;memory=16", "core=1;memory=4").replace("host:dl380g10", "host:sparklina")
    (out / "submit-after-review.sh").write_text(submit)
    shutil.copyfile(__file__, out / "prepare.py")
    (out / "README.md").write_text(f'''# IMG-N environment proof

Prepared only. Do not submit before Fleet approval.
Tool {revision}; source {source_revision}. Each clone refuses another commit.
Shape: sparklina, core=1, memory=4 GiB, limit 20 min. No GPU device and no network.
These are bookings. No wall-time or memory measurement exists for this proof.

The job compares row, C1 and a serve-spec start. It imports torch,
Triton, vLLM and Tessera in the row worker, then checks captured names again.
The result contains names and one launch-state digest. It contains no values
or per-name value hashes. Temporary comparison digests use a Unix socket only.
Import stdout and stderr are discarded. Import status names the module and error type.
Raw image inspection stays in /var/tmp scratch. Cleanup removes that scratch.
The image environment result contains names only.

The fixed work directory /ext is already mounted. Bash reads its 0600 script.
No execute permission is added to the script.
The serve-spec arm uses the shared recipe and checks tessera_plugin_served.sh's -e set.
Both sides add the fixed OCI constants from start_env.py. The serving launcher
must receive these through TESSERA_LANE_DOCKER_EXTRA. The policy excludes none
of these names. The build-identity deterministic switch stays unset.
This is not an observed A8S release serve. No such environment was saved.
This CPU proof cannot establish Docker --gpus versus Podman CDI equivalence.
The GPU canary must compare its captured names and digest with this proof.
A difference refuses attestation. Do not add an exclusion to force equality.

Read results/environment-proof.json, results/image-environment-names.json,
stdout, stderr, tool-exit.txt and the Flux status. A non-SAME result exits 3.
''')
    print(out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-revision", required=True)
    prepare(p.parse_args().source_revision)
