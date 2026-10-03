#!/usr/bin/env bash
# PB CPU-only producer; Triton import comes from the pinned CUDA image, no GPU visible.
set -euo pipefail
ROOT=${1:?native root}; BANK=${2:?new synthetic bank}
IMAGE=${ORACLE_IMAGE:?immutable image required}
RUNNER=${TEST_RUNNER_SP:?Python3.12 pure Python runner required}
PB_CLIENT_ROOT=${PB_CLIENT_ROOT:?published PB SDK required for manifest validation}
source experiments/runtime_image.sh
runtime_image_require "$IMAGE"
CPUS=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
WORK="${BANK}.work"
mkdir -p "$WORK/home" "$WORK/tmp"
docker run --rm --network=none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" \
  -v "$PWD":/work:ro -v "$ROOT":"$ROOT" -v "$WORK":"$WORK" -v "$RUNNER":"$RUNNER":ro \
  -v /usr/local/cuda-13.0/compute-sanitizer:/sealed-sanitizer:ro \
  -v /mnt/shared/prismabuild-fleet:/mnt/shared/prismabuild-fleet:ro \
  -e PYTHONPATH="/work/src:/work/tests:/work:$RUNNER:$PB_CLIENT_ROOT/src" -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME="$WORK/home" -e TMPDIR="$WORK/tmp" -e XDG_CACHE_HOME="$WORK/home/.cache" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PRISMABUILD_ACTION_KEY="${PRISMABUILD_ACTION_KEY:?PB admission required}" \
  --entrypoint python3 -w /work "$IMAGE" experiments/t8r_speed/value_prefetch_numeric.py prepare \
  --bank "$BANK" --native-root "$ROOT" --sanitizer /sealed-sanitizer/compute-sanitizer --runner "$RUNNER"
