#!/usr/bin/env bash
# An admitted D41 native-span2 measurement; PB owns placement and exclusivity.
set -euo pipefail
OUT=$(realpath -m "$1"); shift
IMAGE=${ORACLE_IMAGE:?actual immutable measurement image}
PB_CLIENT_ROOT=${PB_CLIENT_ROOT:?published PrismaBuild SDK}
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
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton"
docker run --rm --gpus all --ipc=host --network=none --pid=host --cpuset-cpus "$CPUS" \
    --user "$(id -u):$(id -g)" -v "$PWD":/work:ro -v "$OUT":"$OUT" \
    -v /mnt/shared:/mnt/shared -v "$STAGE_ROOT":"$STAGE_ROOT" \
    -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
    -e TORCH_EXTENSIONS_DIR="$OUT/home/torch_extensions" -e MAX_JOBS=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="/work/src:/work/experiments/t8r_speed:$PB_CLIENT_ROOT/src" \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e HOST_NAME="$(hostname)" -e TESSERA_HEAD="${TESSERA_HEAD:-unknown}" \
    -e PB_ACTION_KEY="$PRISMABUILD_ACTION_KEY" -e ORACLE_IMAGE="$IMAGE" \
    "${CTX[@]}" "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$IMAGE" \
    experiments/t4_code/bench_geometry_e2m1.py --out "$OUT" "$@"
