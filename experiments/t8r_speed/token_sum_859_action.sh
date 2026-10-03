#!/usr/bin/env bash
# PB-owned finite diagnostics; GPU mode requires root GO.
set -euo pipefail
MODE=${1:?cpu or gpu}; MANIFEST=${2:?sealed readset}; OUT=${3:?bounded output}
[[ "$MODE" == cpu || "$MODE" == gpu ]] || exit 2
IMAGE=${ORACLE_IMAGE:?immutable image}; PB_CLIENT_ROOT=${PB_CLIENT_ROOT:?published SDK}
source experiments/runtime_image.sh
runtime_image_require "$IMAGE"
IMAGE_ENV=()
while IFS= read -r line; do [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line"); done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
CTX=()
for key in PRISMABUILD_ACTION_KEY PRISMABUILD_ACTION_NONCE PRISMABUILD_ACTION_SCOPE PRISMABUILD_QUEUE_ROOT PRISMABUILD_RESIDENCY_MAP PRISMABUILD_READER_HELPER_ROOT; do
  [[ -n "${!key:-}" ]] || { echo "missing admitted context $key" >&2; exit 2; }
  CTX+=(-e "$key=${!key}")
done
STAGE_ROOT=$(PYTHONPATH="$PB_CLIENT_ROOT/src" python3 -c 'import os; from prismabuild.client import read_residency_map; print(read_residency_map(os.environ["PRISMABUILD_RESIDENCY_MAP"])["stage_root"])')
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
DEVICE=(--runtime=runc -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void)
if [[ "$MODE" == gpu ]]; then DEVICE=(--gpus all); fi
mkdir -p "$OUT"
docker run --rm "${DEVICE[@]}" --network=none --pid=host --cpuset-cpus "$CPUS" --memory=4g \
  --user "$(id -u):$(id -g)" -v "$PWD":/work:ro -v "$OUT":"$OUT" \
  -v /mnt/shared/prismabuild-fleet:/mnt/shared/prismabuild-fleet \
  -v "$STAGE_ROOT":"$STAGE_ROOT" -v "$MANIFEST":"$MANIFEST":ro \
  -e HOME="$OUT" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH="/work:$PB_CLIENT_ROOT/src" -e TORCH_EXTENSIONS_DIR="$OUT/owner-build" \
  -e MAX_JOBS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  "${CTX[@]}" "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$IMAGE" \
  experiments/t8r_speed/token_sum_859.py "$MODE" --manifest "$MANIFEST" --out "$OUT/diagnostic" "${@:4}"
