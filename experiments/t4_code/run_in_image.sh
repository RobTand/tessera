#!/usr/bin/env bash
# TMPDIR and thread bounds are launcher-owned; temporary writes use removable job scratch.
# Run one experiments/t4_code script inside the serving image on one GB10.
# Run inside an admitted Flux GPU job; this script is the job's command.  Usage:
#   run_in_image.sh <checkout> <out_dir> <script.py under experiments/t4_code> [args...]
# The container mounts the checkout, the GLM-5.3-Flash BF16 source and the
# activation capture read-only, and keeps persistent writes under <out_dir> (HOME
# and the Triton / torch-extension caches included).  ORACLE_IMAGE is the
# image's digest reference; experiments/runtime_image.sh refuses it before the
# container starts unless Podman resolves it to that digest.
set -euo pipefail
export TESSERA_CONTAINER_ENGINE=podman
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); SCRIPT=$3; shift 3
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable digest-pinned measurement image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
SRC=${T4_SOURCE_ROOT:-/mnt/shared/models/GLM-5.3-Flash-BF16}
ACT=${T4_ACTIVATION_ROOT:-/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act}
for d in "$SRC" "$ACT"; do [[ -d "$d" ]] || { echo "missing input: $d" >&2; exit 2; }; done
[[ -f "$CHECKOUT/experiments/t4_code/$SCRIPT" ]] || { echo "no script $SCRIPT" >&2; exit 2; }
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD image=$IMAGE_REF id=${RUNTIME_IMAGE_LOCAL_ID:-} start=$(date -u +%FT%TZ)"
EXTRA=()
if [[ -n "${T4_NCU:-}" ]]; then
  NCU_ROOT=${NCU_ROOT:-/opt/nvidia/nsight-compute/2025.3.1}
  EXTRA+=(--mount "type=bind,src=$NCU_ROOT,dst=$NCU_ROOT,readonly")
fi
rc=0
python3 "$CHECKOUT/tools/_flux_gpu_jobspec.py" run --out "$OUT" --scratch-root "${FLUX_SCRATCH:-/tmp/flux-vanilla}" -- \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$SRC":"$SRC":ro -v "$ACT":"$ACT":ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$OUT/home/torch_extensions" \
  -e PYTHONPATH=/work/src:/work/experiments/t4_code -e HOST_NAME="$(hostname)" \
  -e PYTHONUNBUFFERED=1 -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" \
  -e FLUX_JOB_ID="${FLUX_JOB_ID:-}" \
  "${IMAGE_ENV[@]}" "${EXTRA[@]}" --entrypoint bash -w /work "$IMAGE_REF" \
  -c 'source /work/experiments/cuda_home_shadow.sh "$TMPDIR"; exec python3 "$@"' -- \
  "/work/experiments/t4_code/$SCRIPT" --out "$OUT" "$@" || rc=$?
echo "end=$(date -u +%FT%TZ) rc=$rc"
exit $rc
