#!/usr/bin/env bash
# PB-admitted entrypoint for experiments/routed_load_memory_probe.py: one
# process per spec, all inside the pinned serving image, under the runtime's
# load-time allocator context (max_split_size_mb:20, gpu_worker's scope).
#
#   routed_load_memory_action.sh IMAGE OUT_DIR SPEC [SPEC ...]
#   SPEC = label|checkpoint|layers|rank[|extra probe args]
#
# Submit through PrismaBuild, e.g.
#   pbrun.py --cwd <tree> --gpu --tag gb10 --priority -10 --timeout-s 3600 \
#     --container-image <IMG> -- bash experiments/routed_load_memory_action.sh <IMG> <OUT> ...
set -uo pipefail
IMG=$1; OUT=$2; shift 2
TREE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
source "$TREE/experiments/runtime_image.sh"
runtime_image_require "$IMG" > "$OUT/image.txt" || { cat "$OUT/image.txt"; exit 2; }
imgenv=()
while IFS= read -r _kv; do
  [ -n "$_kv" ] && imgenv+=(-e "$_kv")
done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
HEAD=$(git -C "$TREE" rev-parse HEAD 2>/dev/null || echo unknown)
echo "host=$(hostname) cpus=$CPUS head=$HEAD image=$IMG out=$OUT"
printf '%s\n' "$@" > "$OUT/specs.txt"
mkdir -p "$OUT/home" "$OUT/triton"
status=0
for spec in "$@"; do
  IFS='|' read -r label data layers rank extra <<<"$spec"
  echo "== $label layers=$layers rank=$rank $(date -u +%FT%TZ)"
  docker run --rm --gpus all --ipc host --network none --cpuset-cpus "$CPUS" \
    --user "$(id -u):$(id -g)" \
    -v "$TREE":/work:ro -v /mnt/shared:/mnt/shared:ro -v "$OUT":"$OUT" \
    -e HOME="$OUT/home" -e TMPDIR="$OUT/home" -e TRITON_CACHE_DIR="$OUT/triton" \
    -e TORCH_EXTENSIONS_DIR="$OUT/torch-ext" -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/work/src -e OMP_NUM_THREADS=2 -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF="${PROBE_ALLOC_CONF-max_split_size_mb:20}" \
    "${imgenv[@]}" --entrypoint python3 -w /work "$IMG" \
    experiments/routed_load_memory_probe.py --data "$data" --layers "$layers" \
    --rank "$rank" --out "$OUT/$label.json" $extra \
    > "$OUT/$label.log" 2>&1
  rc=$?
  tail -n 3 "$OUT/$label.log"
  echo "rc=$rc"
  [ "$rc" = 0 ] || status=$rc
done
exit "$status"
