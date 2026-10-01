#!/usr/bin/env bash
# Run one experiments/t4_code script inside the serving image on one GB10.
# Submitted through PrismaBuild (pbrun --container-image ...); this script is
# the admitted action's command.  Usage:
#   run_in_image.sh <checkout> <out_dir> <script.py under experiments/t4_code> [args...]
# The container mounts the checkout, the GLM-5.3-Flash BF16 source and the
# activation capture read-only, and writes only under <out_dir> (HOME, TMPDIR
# and the Triton / torch-extension caches included).  ORACLE_IMAGE is the
# image's digest reference; experiments/runtime_image.sh refuses it before the
# container starts unless docker resolves it to that digest.
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); SCRIPT=$3; shift 3
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
SRC=/mnt/shared/models/GLM-5.3-Flash-BF16
ACT=/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act
for d in "$SRC" "$ACT"; do [[ -d "$d" ]] || { echo "missing input: $d" >&2; exit 2; }; done
[[ -f "$CHECKOUT/experiments/t4_code/$SCRIPT" ]] || { echo "no script $SCRIPT" >&2; exit 2; }
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
NCPU=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD image=$IMAGE_REF id=${RUNTIME_IMAGE_LOCAL_ID:-} start=$(date -u +%FT%TZ)"
EXTRA=()
if [[ -n "${T4_NCU:-}" ]]; then
  NCU_ROOT=/opt/nvidia/nsight-compute/2025.3.1
  EXTRA+=(--mount "type=bind,src=$NCU_ROOT,dst=$NCU_ROOT,readonly")
fi
rc=0
docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$SRC":"$SRC":ro -v "$ACT":"$ACT":ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$OUT/home/torch_extensions" \
  -e PYTHONPATH=/work/src:/work/experiments/t4_code -e HOST_NAME="$(hostname)" \
  -e OMP_NUM_THREADS="$NCPU" -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PYTHONUNBUFFERED=1 -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  "${IMAGE_ENV[@]}" "${EXTRA[@]}" --entrypoint python3 -w /work "$IMAGE_REF" \
  "/work/experiments/t4_code/$SCRIPT" --out "$OUT" "$@" || rc=$?
echo "end=$(date -u +%FT%TZ) rc=$rc"
exit $rc
