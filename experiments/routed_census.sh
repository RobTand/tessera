#!/usr/bin/env bash
# Run experiments/routed_census.py (and optionally the PACT routed timing
# harness) inside the pinned serving image (tessera#640).  Submitted through
# PrismaBuild as the admitted action's command; same container discipline as
# routed_pair_oracle.sh (checkout read-only at /work, measurement trees
# read-only, every write under <out_dir>).
#   routed_census.sh <checkout> <out_dir> [census.py args...]
# Environment:
#   ORACLE_IMAGE   the immutable image reference (required)
#   PACT_BENCH_DIR host directory holding bench_linears.py; when set, the
#                  harness runs after the census with PACT_BENCH_ARGS
#   PACT_BENCH_ARGS  arguments for bench_linears.py (default: the routed groups)
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
MEAS=/mnt/shared/tessera-measurements/glm-canonical-census-20260908
RUNS=/mnt/shared/tessera-runs
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
STATE=${TESSERA_STATE:-$(git -C "$CHECKOUT" status --short 2>/dev/null | tr '\n' ';' || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD state=[$STATE] image=$IMAGE_REF"
EXTRA_MOUNTS=()
if [[ -n "${PACT_BENCH_DIR:-}" ]]; then
  EXTRA_MOUNTS+=(-v "$(realpath "$PACT_BENCH_DIR")":/pact:ro)
fi
run_in_image() {
  docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
    --user "$(id -u):$(id -g)" \
    -v "$CHECKOUT":/work:ro -v "$MEAS":"$MEAS":ro -v "$RUNS":"$RUNS":ro -v "$OUT":"$OUT" \
    -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
    -e TORCH_EXTENSIONS_DIR="$OUT/torch-ext" \
    -e PYTHONPATH=/work/src:/work/tests:/work/experiments -e HOST_NAME="$(hostname)" \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
    -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" -e TESSERA_STATE="$STATE" \
    -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
    "${IMAGE_ENV[@]}" "${EXTRA_MOUNTS[@]}" ${TESSERA_ROUTED_ENV:+-e "$TESSERA_ROUTED_ENV"} \
    --entrypoint python3 -w /work "$IMAGE_REF" "$@"
}
status=0
run_in_image /work/experiments/routed_census.py --out "$OUT" "$@" || status=$?
if [[ -n "${PACT_BENCH_DIR:-}" ]]; then
  ARGS=${PACT_BENCH_ARGS:---groups experts.T16,experts.T8,experts.T4,rate.experts.E4M3_R896}
  # shellcheck disable=SC2086
  run_in_image /pact/bench_linears.py --out "$OUT/bench" $ARGS || status=$?
fi
exit "$status"
