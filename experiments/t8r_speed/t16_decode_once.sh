#!/usr/bin/env bash
# Run the T16 screen or its GPU tests in the existing serving image.
set -euo pipefail
CHECKOUT=$(realpath "$1")
OUT=$(realpath -m "$2")
shift 2
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the PB-declared serving image}
OWNER=${T16_OWNER_TOKEN:?the supervisor supplies the container owner token}
if [[ "${1:-}" == --tests ]]; then
  shift
  export T16_TEST_OUT="$OUT"
  # Preserve the shared test runner and PB Docker shim. Add only owned cleanup fields.
  docker() {
    if [[ "${1:-}" == run ]]; then
      shift
      command docker run --cidfile "$T16_TEST_OUT/owned.cid" \
        --label "tessera.t16_owner=$T16_OWNER_TOKEN" \
        --memory 16g --memory-swap 16g --pids-limit 512 "$@"
    else
      command docker "$@"
    fi
  }
  export -f docker
  exec bash "$CHECKOUT/experiments/routed_fused_tests.sh" "$CHECKOUT" "$OUT" "$@"
fi
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
for name in TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH TESSERA_DENSE_MODULE_LAUNCH; do
  [[ ! -v "$name" ]] || IMAGE_ENV+=(-e "$name=${!name}")
done
mkdir -p "$OUT/home/torch_extensions" "$OUT/tmp" "$OUT/triton"
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD)}
KERNEL_SHA=$(sha256sum "$CHECKOUT/src/tessera/serving/csrc/routed_fused_window.cu" | cut -d' ' -f1)
printf 'host=%s cpus=%s head=%s image=%s kernel_sha=%s\n' "$(hostname)" "$CPUS" "$HEAD" "$IMAGE_REF" "$KERNEL_SHA"
exec docker run --rm --gpus all --ipc=host --network=host \
  --cidfile "$OUT/owned.cid" --label "tessera.t16_owner=$OWNER" \
  --cpuset-cpus "$CPUS" --memory 16g --memory-swap 16g --pids-limit 512 \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$OUT/home/torch_extensions" \
  -e PYTHONPATH=/work/src:/work/tests -e HOST_NAME="$(hostname)" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 -e MAX_JOBS=1 -e PYTHONUNBUFFERED=1 \
  -e TESSERA_SERVE_MODE=resident -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" \
  -e KERNEL_SHA="$KERNEL_SHA" \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$IMAGE_REF" \
  /work/experiments/t8r_speed/bench_t16_decode_once.py --out "$OUT" "$@"
