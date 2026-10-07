#!/usr/bin/env bash
# PrismaBuild admits this entry point on both processor and device actions.
set -euo pipefail
OUT=$(realpath -m "$1"); shift
mkdir -p "$OUT/home" "$OUT/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
if [[ " $* " == *" --cpu-preflight "* ]]; then
    export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1
    exec /home/rob/venvs/pb-cpu/bin/python experiments/t4_code/fp4_arithmetic_attest.py --out "$OUT" "$@"
fi
IMAGE=${ORACLE_IMAGE:?the admitted native runtime image is required}
source experiments/runtime_image.sh
RUNTIME_IMAGE_JSON=$(PYTHONPATH="$PWD/src" "$RUNTIME_IMAGE_PY" experiments/t4_code/geometry_runtime_image.py --image "$IMAGE")
printf '%s\n' "$RUNTIME_IMAGE_JSON" > "$OUT/image-context.json"
IMAGE_ENV=()
while IFS= read -r line; do
    [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done < <(printf '%s' "$RUNTIME_IMAGE_JSON" | _runtime_image_cli container-env)
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
exec docker run --rm --gpus all --ipc=host --network=none --cpuset-cpus "$CPUS" \
    --user "$(id -u):$(id -g)" -v "$PWD":/work:ro -v "$OUT":"$OUT" \
    -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/work/src -e MAX_JOBS=1 \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e PRISMABUILD_ACTION_KEY="${PRISMABUILD_ACTION_KEY:?}" -e ORACLE_IMAGE="$IMAGE" \
    "${IMAGE_ENV[@]}" --entrypoint bash -w /work "$IMAGE" \
    -c 'source experiments/cuda_home_shadow.sh "$TMPDIR"; exec python3 experiments/t4_code/fp4_arithmetic_attest.py "$@"' -- --out "$OUT" "$@"
