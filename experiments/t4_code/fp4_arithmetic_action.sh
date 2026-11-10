#!/usr/bin/env bash
# Flux runs this entry point in CPU or GPU jobs.
set -euo pipefail
export TESSERA_CONTAINER_ENGINE=podman
OUT=$(realpath -m "$1"); shift
mkdir -p "$OUT/home" "$OUT/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
if [[ " $* " == *" --cpu-preflight "* ]]; then
    export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1
    exec "${TESSERA_CPU_PYTHON:?set TESSERA_CPU_PYTHON to the scoped CPU interpreter}" experiments/t4_code/fp4_arithmetic_attest.py --out "$OUT" "$@"
fi
IMAGE=${ORACLE_IMAGE:?the admitted native runtime image is required}
source experiments/runtime_image.sh
RUNTIME_IMAGE_JSON=$(PYTHONPATH="$PWD/src" "$RUNTIME_IMAGE_PY" experiments/t4_code/geometry_runtime_image.py --engine podman --image "$IMAGE")
printf '%s\n' "$RUNTIME_IMAGE_JSON" > "$OUT/image-context.json"
IMAGE_ENV=()
while IFS= read -r line; do
    [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done < <(printf '%s' "$RUNTIME_IMAGE_JSON" | _runtime_image_cli container-env)
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
INPUT_MOUNTS=()
previous=
for argument in "$@"; do
    if [[ "$previous" == "--native-attestation" || "$previous" == "--boundary-attestation" ]]; then
        [[ -f "$argument" ]] || { echo "missing native attestation input: $argument" >&2; exit 2; }
        INPUT_MOUNTS+=(-v "$argument:$argument:ro")
    fi
    previous=$argument
done
exec python3 "$PWD/tools/_flux_gpu_jobspec.py" run --out "$OUT" --scratch-root "${FLUX_SCRATCH:-/tmp/flux-vanilla}" -- \
    --user "$(id -u):$(id -g)" -v "$PWD":/work:ro -v "$OUT":"$OUT" \
    -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/work/src -e MAX_JOBS=1 \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e FLUX_JOB_ID="${FLUX_JOB_ID:?}" -e ORACLE_IMAGE="$IMAGE" -e HOST_NAME="$(hostname)" \
    "${INPUT_MOUNTS[@]}" "${IMAGE_ENV[@]}" --entrypoint bash -w /work "$IMAGE" \
    -c 'source experiments/cuda_home_shadow.sh "$TMPDIR"; exec python3 experiments/t4_code/fp4_arithmetic_attest.py "$@"' -- --out "$OUT" "$@"
