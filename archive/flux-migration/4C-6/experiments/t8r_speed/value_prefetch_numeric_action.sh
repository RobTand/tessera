#!/usr/bin/env bash
# PB qualification action. Consume/sanitize require root GPU GO; state-proof is CPU-only.
set -euo pipefail
BANK=${1:?synthetic bank}; OUT=${2:?output}; MODE=${3:?consume or sanitize}; MANIFEST=${4:?frozen readset}
[[ "$MODE" == consume || "$MODE" == sanitize || "$MODE" == state-proof || "$MODE" == loader-proof || "$MODE" == native-preflight ]] || exit 2
GPU_ARGS=()
[[ "$MODE" == state-proof || "$MODE" == loader-proof || "$MODE" == native-preflight ]] || GPU_ARGS=(--gpus all)
IMAGE=${ORACLE_IMAGE:?immutable image required}
PB_CLIENT_ROOT=${PB_CLIENT_ROOT:?published PB SDK required}
source experiments/runtime_image.sh
runtime_image_require "$IMAGE"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
CTX=()
for key in PRISMABUILD_ACTION_KEY PRISMABUILD_ACTION_NONCE PRISMABUILD_ACTION_SCOPE PRISMABUILD_QUEUE_ROOT PRISMABUILD_RESIDENCY_MAP PRISMABUILD_READER_HELPER_ROOT; do
  [[ -n "${!key:-}" ]] || { echo "missing admitted context $key" >&2; exit 2; }
  CTX+=(-e "$key=${!key}")
done
STAGE_ROOT=$(PYTHONPATH="$PB_CLIENT_ROOT/src" python3 -c 'import os; from prismabuild.client import read_residency_map; print(read_residency_map(os.environ["PRISMABUILD_RESIDENCY_MAP"])["stage_root"])')
CPUS=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
mkdir -p "$OUT"
docker run --rm "${GPU_ARGS[@]}" --ipc=host --network=none --pid=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" -v "$PWD":/work:ro -v "$BANK":"$BANK":ro \
  -v "$OUT":"$OUT" -v /mnt/shared/prismabuild-fleet:/mnt/shared/prismabuild-fleet \
  -v "$STAGE_ROOT":"$STAGE_ROOT" -e HOME="$OUT" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH="/work/src:/work/tests:/work:$PB_CLIENT_ROOT/src" \
  -e TORCH_EXTENSIONS_DIR="$OUT/owner-build" -e MAX_JOBS=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  "${CTX[@]}" "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$IMAGE" \
  experiments/t8r_speed/value_prefetch_numeric.py "$MODE" --bank "$BANK" \
  --manifest "$MANIFEST" --out "$OUT/syntheticgeometry-nonshipping"
