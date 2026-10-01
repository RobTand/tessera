#!/usr/bin/env bash
# KDA fusion build gate: build stock FlashKDA through tessera.flashkda inside the
# serving image and compare its cubin with the image's own _flashkda_C, kernel for
# kernel (experiments/kda/build_gate.py).  CPU only: no GPU is requested or used.
# Submitted through PrismaBuild (pbrun --container-image ...); this script is the
# admitted action's command.  Usage:
#   build_gate.sh <checkout> <out_dir>
# The container mounts the checkout and the CUTLASS directory read-only and writes
# only under <out_dir>.  TESSERA_FLASHKDA_CUTLASS overrides the CUTLASS directory.
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared serving image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
CUTLASS=${TESSERA_FLASHKDA_CUTLASS:-/mnt/shared/tessera-measurements/kda-fusion-20261001/deps/cutlass-5c149f52a436782210263fb2f19b354443a61c6a}
[[ -f "$CUTLASS/COMMIT" ]] || { echo "missing CUTLASS directory: $CUTLASS" >&2; exit 2; }
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/torch-ext"
exec > >(tee -a "$OUT/run.log") 2>&1
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
STATE=${TESSERA_STATE:-$(git -C "$CHECKOUT" status --short 2>/dev/null | tr '\n' ';' || echo unknown)}
echo "start $(date -u +%FT%TZ) host=$(hostname) cpus=$CPUS head=$HEAD state=[$STATE] image=$IMAGE_REF cutlass=$CUTLASS"
rc=0
docker run --rm --network=none --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$CUTLASS":"$CUTLASS":ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TORCH_EXTENSIONS_DIR="$OUT/torch-ext" \
  -e TESSERA_FLASHKDA_CUTLASS="$CUTLASS" -e TESSERA_PLATFORM_TOKEN=sm_121 -e MAX_JOBS=2 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work/src -e HOST_NAME="$(hostname)" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
  -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" -e TESSERA_STATE="$STATE" \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$IMAGE_REF" \
  /work/experiments/kda/build_gate.py --out "$OUT" "$@" || rc=$?
echo "end $(date -u +%FT%TZ) rc=$rc"
exit $rc
