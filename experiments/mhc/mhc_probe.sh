#!/usr/bin/env bash
# TMPDIR and thread bounds are launcher-owned; temporary writes use removable job scratch.
# Run experiments/mhc/mhc_probe.py inside the serving image (the mHC and
# elementwise before-measurements).  Run inside an admitted Flux GPU job; this
# script is the job's command.  Usage:
#   mhc_probe.sh <checkout> <out_dir> <mhc_probe.py args...>
# The container mounts the checkout and the served checkpoint read-only and
# keeps persistent writes under <out_dir>.  ORACLE_NCU=1 runs the NCU-gated mHC calls under
# Nsight Compute instead, filtered to the three stock mHC kernels (ORACLE_NCU_KERNELS
# overrides the filter, e.g. 'regex:_flash_kda_fwd' with --ncu-part kda).
set -euo pipefail
export TESSERA_CONTAINER_ENGINE=podman
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable digest-pinned measurement image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
MODEL=${MHC_PROBE_MODEL:-/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported}
[[ -f "$MODEL/model.safetensors.index.json" ]] || { echo "missing checkpoint: $MODEL" >&2; exit 2; }
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton" "$OUT/torch-ext"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
STATE=${TESSERA_STATE:-$(git -C "$CHECKOUT" status --short 2>/dev/null | tr '\n' ';' || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD state=[$STATE] image=$IMAGE_REF model=$MODEL"
EXTRA_MOUNTS=()
OBS_ENV=()
declare -A OBS_MOUNTED=()
for OBS_VAR in TESSERA_PROFILE_TORCH_OUT; do
  OBS_VALUE=${!OBS_VAR:-}
  if [[ -n "$OBS_VALUE" ]]; then
    OBS_PARENT=$(dirname "$OBS_VALUE")
    mkdir -p "$OBS_PARENT"
    if [[ -z "${OBS_MOUNTED[$OBS_PARENT]:-}" ]]; then
      EXTRA_MOUNTS+=(--mount "type=bind,src=$OBS_PARENT,dst=$OBS_PARENT")
      OBS_MOUNTED[$OBS_PARENT]=1
    fi
    OBS_ENV+=(-e "$OBS_VAR=$OBS_VALUE")
  fi
done
PREFIX=(bash -c 'source /work/experiments/cuda_home_shadow.sh "$TMPDIR/.." && exec "$@"' bash)
# MHC_PROBE_SCRIPT selects a sibling probe with the same contract (mhc_fused_probe.py, #783).
PROBE=${MHC_PROBE_SCRIPT:-mhc_probe.py}
[[ "$PROBE" =~ ^[a-z_]+\.py$ && -f "$CHECKOUT/experiments/mhc/$PROBE" ]] || { echo "bad MHC_PROBE_SCRIPT: $PROBE" >&2; exit 2; }
COMMAND=(python3 "/work/experiments/mhc/$PROBE")
if [[ "${ORACLE_NCU:-0}" == 1 ]]; then
  NCU_ROOT=/opt/nvidia/nsight-compute/2025.3.1
  [[ -x "$NCU_ROOT/ncu" ]] || { echo "missing profiler: $NCU_ROOT/ncu" >&2; exit 2; }
  EXTRA_MOUNTS+=(--mount "type=bind,src=$NCU_ROOT,dst=$NCU_ROOT,readonly")
  COMMAND=("$NCU_ROOT/ncu" --profile-from-start off --target-processes all
    --kernel-name "${ORACLE_NCU_KERNELS:-regex:mhc_post|hc_prenorm_gemm|mhc_pre_big_fuse}"
    --section LaunchStats --section Occupancy --section SpeedOfLight
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
    --section WarpStateStats --section SchedulerStats
    --csv --log-file "$OUT/ncu.csv"
    --export "$OUT/mhc" --force-overwrite
    python3 /work/experiments/mhc/mhc_probe.py --ncu)
fi
exec python3 "$CHECKOUT/tools/_flux_gpu_jobspec.py" run --out "$OUT" --scratch-root "${FLUX_SCRATCH:-/tmp/flux-vanilla}" -- \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$MODEL":"$MODEL":ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$OUT/torch-ext" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/work/src:/work/tests:/work/experiments -e HOST_NAME="$(hostname)" \
  -e PYTHONUNBUFFERED=1 \
  -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" -e TESSERA_STATE="$STATE" \
  -e FLUX_JOB_ID="${FLUX_JOB_ID:-}" \
  "${IMAGE_ENV[@]}" "${OBS_ENV[@]}" "${EXTRA_MOUNTS[@]}" --entrypoint "${PREFIX[0]}" -w /work "$IMAGE_REF" \
  "${PREFIX[@]:1}" "${COMMAND[@]}" --out "$OUT" --model "$MODEL" "$@"
