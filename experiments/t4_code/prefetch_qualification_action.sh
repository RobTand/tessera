#!/usr/bin/env bash
# Admitted PB payload only. GPU modes require separate root authorization.
set -euo pipefail
OUT=${1:?output}; MODE=${2:?staged-read/cpu-map/numeric/native/sanitize}; MANIFEST=${3:?validated staged readset}; shift 3
IMAGE=${ORACLE_IMAGE:?immutable image required}
PB_CLIENT_ROOT=${PB_CLIENT_ROOT:?published PB SDK required}
GPU_ARGS=(); DRIVER_ARGS=(); VISIBILITY=(); OUT_ARGS=(--out "$OUT/diagnostic")
case "$MODE" in
  staged-read) DRIVER_ARGS=(staged-read); OUT_ARGS=() ;;
  cpu-map) DRIVER_ARGS=(consume --cpu-map) ;;
  tool-preflight) DRIVER_ARGS=(tool-preflight) ;;
  numeric|native) GPU_ARGS=(--gpus all); DRIVER_ARGS=(consume) ;;
  sanitize) GPU_ARGS=(--gpus all); DRIVER_ARGS=(sanitize) ;;
  *) echo "unsupported qualification mode $MODE" >&2; exit 2 ;;
esac
[[ ${#GPU_ARGS[@]} -ne 0 ]] || VISIBILITY=(-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void)
source experiments/runtime_image.sh
runtime_image_require "$IMAGE"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
CTX=()
for key in PRISMABUILD_ACTION_KEY PRISMABUILD_ACTION_NONCE PRISMABUILD_ACTION_SCOPE PRISMABUILD_QUEUE_ROOT PRISMABUILD_RESIDENCY_MAP PRISMABUILD_READER_HELPER_ROOT; do
  [[ -n "${!key:-}" ]] || { echo "missing admitted PB context $key" >&2; exit 2; }
  CTX+=(-e "$key=${!key}")
done
STAGE_ROOT=$(PYTHONPATH="$PB_CLIENT_ROOT/src" python3 -c 'import os; from prismabuild.client import read_residency_map; print(read_residency_map(os.environ["PRISMABUILD_RESIDENCY_MAP"])["stage_root"])')
CPUS=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
mkdir -p "$OUT"
# The SDK reader helper and executable map proof require the admitted PID view;
# every child stays in PB's container scope and its assigned CPU affinity.
docker run --rm "${GPU_ARGS[@]}" "${VISIBILITY[@]}" --network=none --ipc=host --pid=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" -v "$PWD":/work:ro -v "$OUT":"$OUT" \
  -v /mnt/shared/tessera-measurements/t4-875-cpu-20261003:/mnt/shared/tessera-measurements/t4-875-cpu-20261003:ro \
  -v /mnt/shared/prismabuild-fleet:/mnt/shared/prismabuild-fleet \
  -v "$STAGE_ROOT":"$STAGE_ROOT" -e HOME="$OUT" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH="/work/src:/work/tests:/work/experiments/t4_code:/work/experiments/t8r_speed:$PB_CLIENT_ROOT/src" \
  -e ORACLE_IMAGE="$IMAGE" -e TORCH_EXTENSIONS_DIR="$OUT/owner-build" -e MAX_JOBS=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  "${CTX[@]}" "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$IMAGE" \
  experiments/t4_code/prefetch_qualification.py "${DRIVER_ARGS[@]}" --manifest "$MANIFEST" "${OUT_ARGS[@]}" "$@"
