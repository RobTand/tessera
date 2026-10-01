#!/usr/bin/env bash
# D1 row: time and profile FlashInfer's SM120 sparse MLA prefill kernel
# (sparse_mla_prefill_mg_kernel<GLM53_NOPE, FP8, 32, 64, 2>) inside the serving
# image, at the GLM-5.3 served shape. One PrismaBuild GPU action, exclusive GPU.
#
#   MLA_IMAGE=<digest ref> d1_row.sh <out_dir>
#
# Legs, each with its UTC window printed for the Netdata series:
#   timing_pools   CUDA events + kernel-only profiler times + power, kpool indices
#   timing_random  the same with uniform-random causal indices
#   ncu_pools      Nsight Compute --set full, one launch per shape
# Exit status: 1 if any leg failed (all legs always run), 2 on a launch error.
set -uo pipefail
OUT=$(realpath -m "${1:?out_dir}")
mkdir -p "$OUT/home" "$OUT/tmp"
export TMPDIR="$OUT/tmp"
IMAGE=${MLA_IMAGE:?set MLA_IMAGE to the PB-declared serving image}
HERE=$(dirname "$(realpath "$0")")
CHECKOUT=$(realpath "$HERE/../..")
NCU_ROOT=/opt/nvidia/nsight-compute/2025.3.1
[[ -x "$NCU_ROOT/ncu" ]] || { echo "missing profiler: $NCU_ROOT/ncu" >&2; exit 2; }
# Refuse a floating image and pass its declared identity into the container
# (experiments/runtime_image.sh; issues #100, #132).
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE" || exit 2
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
echo "host=$(hostname) cpus=$CPUS head=$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)" \
     "image=$IMAGE start=$(date -u +%FT%TZ)"
sha256sum "$CHECKOUT/experiments/mla_prefill/d1_bench.py"

DOCKER=(docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS"
  --user "$(id -u):$(id -g)" -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT"
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e HOST_NAME="$(hostname)"
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONUNBUFFERED=1
  "${IMAGE_ENV[@]}" -w /work)
BENCH=/work/experiments/mla_prefill/d1_bench.py
FAILED=()

leg() {  # leg NAME docker-args... -- command...
  local name=$1; shift
  local dargs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do dargs+=("$1"); shift; done
  shift
  local p0; p0=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)
  echo "== $name start=$(date -u +%FT%TZ) gpu_w=$p0"
  "${DOCKER[@]}" "${dargs[@]}" --entrypoint "$1" "$IMAGE" "${@:2}" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "== $name rc=$rc end=$(date -u +%FT%TZ)"
  tail -6 "$OUT/$name.log"
  ((rc == 0)) || FAILED+=("$name:$rc")
}

leg timing_pools -- python3 "$BENCH" --out "$OUT" --index-mode pools
leg timing_random -- python3 "$BENCH" --out "$OUT" --index-mode random
leg ncu_pools --mount "type=bind,src=$NCU_ROOT,dst=$NCU_ROOT,readonly" -- \
  "$NCU_ROOT/ncu" --profile-from-start off --target-processes all \
  --kernel-name "regex:sparse_mla_prefill_mg_kernel" --clock-control none \
  --set full --import-source no --export "$OUT/d1_mg" --force-overwrite \
  python3 "$BENCH" --out "$OUT" --ncu --index-mode pools
if [[ -f "$OUT/d1_mg.ncu-rep" ]]; then
  "$NCU_ROOT/ncu" --import "$OUT/d1_mg.ncu-rep" --csv --page details > "$OUT/ncu_details.csv" 2> "$OUT/ncu_import.err" \
    || FAILED+=("ncu_import:$?")
  "$NCU_ROOT/ncu" --import "$OUT/d1_mg.ncu-rep" --csv --page raw > "$OUT/ncu_raw.csv" 2>> "$OUT/ncu_import.err" \
    || FAILED+=("ncu_raw:$?")
fi
echo "end=$(date -u +%FT%TZ)"
((${#FAILED[@]} == 0)) || { echo "FAILED_LEGS ${FAILED[*]}"; exit 1; }
