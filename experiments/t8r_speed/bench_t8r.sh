#!/usr/bin/env bash
# Run experiments/t8r_speed/bench_t8r.py inside the serving image on one GB10.
# Submitted through PrismaBuild (pbrun --measurement --gpu --container-image ...);
# this script is the admitted action's command.  Usage:
#   bench_t8r.sh <checkout> <out_dir> <bench_t8r.py args...>
# BENCH_NCU=1 wraps the process in Nsight Compute (routed_fused_kernel and the
# dense window kernels; one profiled call per (group, M) via --profile-from-start off).
# The container mounts the checkout and the artifact read-only and writes only
# under <out_dir> (HOME/TMPDIR/Triton and torch-extension caches included).
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
[[ -z "${TESSERA_ROUTED_FUSED:-}" ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED=$TESSERA_ROUTED_FUSED")
ART=/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported
[[ -f "$ART/config.json" ]] || { echo "missing artifact: $ART" >&2; exit 2; }
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
STATE=${TESSERA_STATE:-$(git -C "$CHECKOUT" status --short 2>/dev/null | tr '\n' ';' || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD state=[$STATE] image=$IMAGE_REF start=$(date -u +%FT%TZ)"
free -g | sed -n 2p
# BENCH_SRC: an A/B arm's Tessera source tree, mounted over the checkout's src
# (the harness stays the checkout's); unset runs the checkout's own src.
SRC_MOUNT=()
if [[ -n "${BENCH_SRC:-}" ]]; then
  [[ -f "$BENCH_SRC/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "BENCH_SRC is not a tessera src tree: $BENCH_SRC" >&2; exit 2; }
  SRC_MOUNT=(-v "$BENCH_SRC":/work/src:ro)
  KSRC="$BENCH_SRC"
else
  KSRC="$CHECKOUT/src"
fi
KERNEL_SHA=$(sha256sum "$KSRC/tessera/serving/csrc/routed_fused_window.cu" | cut -d' ' -f1)
echo "arm src=$KSRC kernel_sha=$KERNEL_SHA"
EXTRA_MOUNTS=()
COMMAND=(python3 /work/experiments/t8r_speed/bench_t8r.py)
if [[ "${BENCH_NCU:-0}" == 1 ]]; then
  NCU_ROOT=/opt/nvidia/nsight-compute/2025.3.1
  [[ -x "$NCU_ROOT/ncu" ]] || { echo "missing profiler: $NCU_ROOT/ncu" >&2; exit 2; }
  EXTRA_MOUNTS=(--mount "type=bind,src=$NCU_ROOT,dst=$NCU_ROOT,readonly")
  COMMAND=("$NCU_ROOT/ncu" --profile-from-start off --target-processes all
    --kernel-name "regex:${BENCH_NCU_KERNELS:-routed_fused_kernel|token_sum_kernel|window}"
    --section LaunchStats --section Occupancy --section SpeedOfLight
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
    --section WarpStateStats --section SchedulerStats --section InstructionStats
    --section SourceCounters --import-source yes
    --csv --log-file "$OUT/ncu.csv"
    --export "$OUT/t8r" --force-overwrite
    python3 /work/experiments/t8r_speed/bench_t8r.py --ncu)
fi
rc=0
docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro "${SRC_MOUNT[@]}" -v "$ART":"$ART":ro -v "$OUT":"$OUT" \
  -e KERNEL_SHA="$KERNEL_SHA" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$OUT/home/torch_extensions" \
  -e PYTHONPATH=/work/src:/work/tests -e HOST_NAME="$(hostname)" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 -e TESSERA_SERVE_MODE=resident \
  -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" -e TESSERA_STATE="$STATE" \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  "${IMAGE_ENV[@]}" "${EXTRA_MOUNTS[@]}" --entrypoint "${COMMAND[0]}" -w /work "$IMAGE_REF" \
  "${COMMAND[@]:1}" --out "$OUT" "$@" || rc=$?
echo "end=$(date -u +%FT%TZ) rc=$rc"
exit $rc
