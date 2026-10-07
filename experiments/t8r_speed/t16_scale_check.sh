#!/usr/bin/env bash
# One source entry serves the D38 CPU preflight and actual GPU checks.
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton" "$OUT/extensions"
if [[ " $* " == *" --cpu-preflight "* ]]; then
  PYTHONPATH="$CHECKOUT/src:$CHECKOUT/tests:/tmp/eng-t16-cpu-deps:${PYTHONPATH:-}" \
    exec /home/rob/venvs/pq-cpu312/bin/python \
    "$CHECKOUT/experiments/t8r_speed/t16_scale_check.py" --out "$OUT" "$@"
fi
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "${ORACLE_IMAGE:?immutable PB-declared image required}"
IMAGE_ENV=()
while IFS= read -r line; do [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line"); done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
DEPS=${T16_TEST_DEPS:?scoped pytest dependency directory required}
[[ -d "$DEPS" ]] || { echo "The scoped test dependency directory is absent." >&2; exit 2; }
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
docker run --rm --gpus all --ipc=host --network=none --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT" -v "$DEPS":"$DEPS":ro \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$OUT/extensions" -e MAX_JOBS=1 \
  -e PYTHONPATH="/work/src:/work/tests:$DEPS" -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e TESSERA_PLATFORM_TOKEN=sm_121 -e TESSERA_SERVE_MODE=resident \
  -e PRISMABUILD_ACTION_KEY="${PRISMABUILD_ACTION_KEY:?PB admission required}" \
  "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$ORACLE_IMAGE" \
  /work/experiments/t8r_speed/t16_scale_check.py --out "$OUT" "$@"
