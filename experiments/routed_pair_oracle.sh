#!/usr/bin/env bash
# Run experiments/routed_pair_oracle.py inside the serving image (tessera#604).
# Submitted through PrismaBuild (pbrun --gpu --container-image ...); this script
# is the admitted action's command.  Usage:
#   routed_pair_oracle.sh <checkout> <out_dir> <oracle.py args...>
# PB's docker shim preserves the action's CPU mask; the container mounts the
# checkout read-only at /work and the measurement tree read-only, and writes
# only under <out_dir> (HOME/TMPDIR/Triton cache included).
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
# tessera#640: the fused routed lane's opt-out rides into the container when set,
# so one action can profile the compact adapter as the dispatch (TESSERA_ROUTED_FUSED=0).
[[ -z "${TESSERA_ROUTED_FUSED:-}" ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED=$TESSERA_ROUTED_FUSED")
[[ -z "${TESSERA_ROUTED_FUSED_VERBOSE:-}" ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED_VERBOSE=$TESSERA_ROUTED_FUSED_VERBOSE")
MEAS=/mnt/shared/tessera-measurements/glm-canonical-census-20260908
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
STATE=${TESSERA_STATE:-$(git -C "$CHECKOUT" status --short 2>/dev/null | tr '\n' ';' || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD state=[$STATE] image=$IMAGE_REF"
EXTRA_MOUNTS=()
COMMAND=(python3 /work/experiments/routed_pair_oracle.py)
if [[ "${ORACLE_NCU:-0}" == 1 ]]; then
  NCU_ROOT=/opt/nvidia/nsight-compute/2025.3.1
  [[ -x "$NCU_ROOT/ncu" ]] || { echo "missing profiler: $NCU_ROOT/ncu" >&2; exit 2; }
  EXTRA_MOUNTS=(--mount "type=bind,src=$NCU_ROOT,dst=$NCU_ROOT,readonly")
  COMMAND=("$NCU_ROOT/ncu" --profile-from-start off --target-processes all
    --kernel-name 'regex:_grouped_window_gemm_kernel|_a4_span2_grouped_kernel|routed_fused_kernel|token_sum_kernel'
    --section LaunchStats --section Occupancy --section SpeedOfLight
    --section MemoryWorkloadAnalysis --csv --log-file "$OUT/ncu.csv"
    --export "$OUT/grouped" --force-overwrite
    python3 /work/experiments/routed_pair_ncu.py)
fi
exec docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$MEAS":"$MEAS":ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e PYTHONPATH=/work/src:/work/tests -e HOST_NAME="$(hostname)" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
  -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" -e TESSERA_STATE="$STATE" \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  "${IMAGE_ENV[@]}" "${EXTRA_MOUNTS[@]}" --entrypoint "${COMMAND[0]}" -w /work "$IMAGE_REF" \
  "${COMMAND[@]:1}" --out "$OUT" "$@"
