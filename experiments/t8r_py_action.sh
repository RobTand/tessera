#!/usr/bin/env bash
# PB-admitted entrypoint: run one experiments/ script inside the pinned
# serving image, writing only under OUT_DIR.
#   t8r_py_action.sh IMAGE OUT_DIR LABEL SCRIPT [script args...]
set -uo pipefail
IMG=$1; OUT=$2; LABEL=$3; SCRIPT=$4; shift 4
TREE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT" "$OUT/home" "$OUT/triton" "$OUT/torch-ext"
source "$TREE/experiments/runtime_image.sh"
runtime_image_require "$IMG" > "$OUT/image.txt" || { cat "$OUT/image.txt"; exit 2; }
imgenv=()
while IFS= read -r _kv; do
  [ -n "$_kv" ] && imgenv+=(-e "$_kv")
done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
echo "host=$(hostname) cpus=$CPUS head=$(git -C "$TREE" rev-parse HEAD 2>/dev/null) image=$IMG"
docker run --rm --gpus all --ipc host --network none --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" \
  -v "$TREE":/work:ro -v /mnt/shared:/mnt/shared:ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/home" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$OUT/torch-ext" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/work/src -e OMP_NUM_THREADS=2 -e PYTHONUNBUFFERED=1 \
  "${imgenv[@]}" --entrypoint python3 -w /work "$IMG" "$SCRIPT" "$@" \
  > "$OUT/$LABEL.log" 2>&1
rc=$?
tail -n 20 "$OUT/$LABEL.log"
echo "rc=$rc"
exit $rc
