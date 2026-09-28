#!/usr/bin/env bash
# Run experiments/dense_fused_oracle.py inside the serving image (contract v43,
# the fused window kernel's dense identity).  Submitted through PrismaBuild
# (pbrun --gpu --container-image ...); this script is the admitted action's
# command.  Usage:
#   dense_fused_oracle.sh <checkout> <out_dir> <oracle.py args...>
# PB's docker shim preserves the action's CPU mask; the container mounts the
# checkout read-only at /work and the stub checkpoint read-only, and writes
# only under <out_dir> (HOME/TMPDIR/Triton cache/JIT build included).
# ORACLE_NCU=1 runs dense_fused_ncu.py under Nsight Compute instead, gated to
# the dense window launches of both lanes.
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
# The dense identity's opt-out and the fused library's build chatter ride into
# the container when set; the oracle itself builds both lanes per module.
[[ -z "${TESSERA_DENSE_FUSED:-}" ]] || IMAGE_ENV+=(-e "TESSERA_DENSE_FUSED=$TESSERA_DENSE_FUSED")
[[ -z "${TESSERA_ROUTED_FUSED_VERBOSE:-}" ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED_VERBOSE=$TESSERA_ROUTED_FUSED_VERBOSE")
STUB=${DENSE_ORACLE_STUB:-/mnt/shared/tessera-runs/moe/u1-stubs-20260926/stub-B}
[[ -d "$STUB" ]] || { echo "missing stub checkpoint: $STUB" >&2; exit 2; }
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton" "$OUT/torch-ext"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
STATE=${TESSERA_STATE:-$(git -C "$CHECKOUT" status --short 2>/dev/null | tr '\n' ';' || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD state=[$STATE] image=$IMAGE_REF stub=$STUB"
EXTRA_MOUNTS=()
# cuda_home_shadow.sh fills the CUDA header gap of the pinned vllm-openai image
# for the JIT build (a no-op where /usr/local/cuda/include is complete).
PREFIX=(bash -c 'source /work/experiments/cuda_home_shadow.sh "$TMPDIR/.." && exec "$@"' bash)
COMMAND=(python3 /work/experiments/dense_fused_oracle.py)
if [[ "${ORACLE_NCU:-0}" == 1 ]]; then
  NCU_ROOT=/opt/nvidia/nsight-compute/2025.3.1
  [[ -x "$NCU_ROOT/ncu" ]] || { echo "missing profiler: $NCU_ROOT/ncu" >&2; exit 2; }
  EXTRA_MOUNTS=(--mount "type=bind,src=$NCU_ROOT,dst=$NCU_ROOT,readonly")
  COMMAND=("$NCU_ROOT/ncu" --profile-from-start off --target-processes all
    --kernel-name 'regex:routed_fused_kernel|dense_reduce_kernel|_window_gemm_kernel'
    --section LaunchStats --section Occupancy --section SpeedOfLight
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
    --section WarpStateStats --section SchedulerStats --section InstructionStats
    --csv --log-file "$OUT/ncu.csv"
    --export "$OUT/dense" --force-overwrite
    python3 /work/experiments/dense_fused_ncu.py)
fi
exec docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$STUB":"$STUB":ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$OUT/torch-ext" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/work/src:/work/tests:/work/experiments -e HOST_NAME="$(hostname)" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
  -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" -e TESSERA_STATE="$STATE" \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  "${IMAGE_ENV[@]}" "${EXTRA_MOUNTS[@]}" --entrypoint "${PREFIX[0]}" -w /work "$IMAGE_REF" \
  "${PREFIX[@]:1}" "${COMMAND[@]}" --out "$OUT" "$@"
