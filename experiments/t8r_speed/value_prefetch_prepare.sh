#!/usr/bin/env bash
# PB CPU-only producer; Triton import comes from the pinned CUDA image, no GPU visible.
set -euo pipefail
ROOT=${1:?native root}; BANK=${2:?new synthetic bank}
IMAGE=${ORACLE_IMAGE:?immutable image required}
RUNNER=${TEST_RUNNER_SP:?Python3.12 pure Python runner required}
source experiments/runtime_image.sh
runtime_image_require "$IMAGE"
CPUS=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
docker run --rm --network=none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" \
  -v "$PWD":/work:ro -v "$ROOT":"$ROOT" -v "$RUNNER":"$RUNNER":ro \
  -v /usr/local/cuda-13.0/compute-sanitizer:/sealed-sanitizer:ro \
  -e PYTHONPATH="/work/src:/work/tests:/work:$RUNNER" -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PRISMABUILD_ACTION_KEY="${PRISMABUILD_ACTION_KEY:?PB admission required}" \
  --entrypoint python3 -w /work "$IMAGE" experiments/t8r_speed/value_prefetch_numeric.py prepare \
  --bank "$BANK" --native-root "$ROOT" --sanitizer /sealed-sanitizer/compute-sanitizer --runner "$RUNNER"
