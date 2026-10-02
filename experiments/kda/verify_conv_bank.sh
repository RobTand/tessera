#!/usr/bin/env bash
# CPU-only admitted artifact check. No CUDA workload or recompilation.
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); OLD=$(realpath "$3"); NEW=$(realpath "$4")
BANK=$(realpath "$5"); CONTROL=$(realpath "$6")
BANK_SHA=$7; CONTROL_SHA=$8
IMAGE_REF=${ORACLE_IMAGE:?immutable declared image required}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
CPUS=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
mkdir -p "$OUT/old" "$OUT/new"
printf '%s\n' "$OLD" > "$OUT/old-module-path.txt"
printf '%s\n' "$NEW" > "$OUT/new-module-path.txt"
exec docker run --rm --network none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT" \
  -v "$(dirname "$OLD")":"$(dirname "$OLD")":ro \
  -v "$(dirname "$NEW")":"$(dirname "$NEW")":ro \
  -v "$(dirname "$BANK")":"$(dirname "$BANK")":ro \
  -v "$(dirname "$CONTROL")":"$(dirname "$CONTROL")":ro \
  -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  --entrypoint bash "$IMAGE_REF" -c '
    set -euo pipefail
    cd "$1/old"
    /usr/local/cuda/bin/cuobjdump --extract-elf all "$2"
    cd "$1/new"
    /usr/local/cuda/bin/cuobjdump --extract-elf all "$3"
    python3 /work/experiments/kda/verify_conv_bank.py "$1" "$4" "$5" "$6" "$7"
  ' bash "$OUT" "$OLD" "$NEW" "$BANK" "$CONTROL" "$BANK_SHA" "$CONTROL_SHA"
