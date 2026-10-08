#!/usr/bin/env bash
# Run this payload only through PrismaBuild. PB owns placement and containment.
set -euo pipefail
ARGS=("$@")
OUT=""; MODE=""; CPU=0; PREVIOUS=""
for arg in "$@"; do
    case "$PREVIOUS" in
        --out) OUT="$arg" ;;
        --mode) MODE="$arg" ;;
    esac
    [[ "$arg" != --dry-run ]] || CPU=1
    PREVIOUS="$arg"
done
[[ -n "$OUT" && -n "$MODE" ]] || { echo "Specify --mode and --out." >&2; exit 2; }
case "$MODE" in
    dry-run|research-fixture|prepare-inputs) CPU=1 ;;
esac
CHECKOUT="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ "$CPU" == 1 ]]; then
    export TMPDIR=/tmp PYTHONPATH="$CHECKOUT/src:$CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
    exec "${D38_PYTHON:-/home/rob/venvs/pb-cpu/bin/python}" \
        experiments/t4_code/t4_fused_qualify.py "${ARGS[@]}"
fi
IMAGE=${ORACLE_IMAGE:?Specify the PB-declared serving image.}
PB_CLIENT_ROOT=${PB_CLIENT_ROOT:-${PRISMABUILD_READER_HELPER_ROOT:-/mnt/shared/prismabuild-fleet/repo}}
# Use the existing image resolver. D32 identity changes produce stamps.
source experiments/runtime_image.sh
IMAGE_JSON=$(PYTHONPATH="$CHECKOUT/src" "$RUNTIME_IMAGE_PY" \
    experiments/t4_code/geometry_runtime_image.py --image "$IMAGE")
IMAGE_ENV=()
while IFS= read -r value; do
    [[ -z "$value" ]] || IMAGE_ENV+=(-e "$value")
done < <(printf '%s' "$IMAGE_JSON" | _runtime_image_cli container-env)
printf '%s\n' "$IMAGE_JSON"
CTX=()
for key in PRISMABUILD_ACTION_KEY PRISMABUILD_ACTION_NONCE PRISMABUILD_ACTION_SCOPE \
           PRISMABUILD_QUEUE_ROOT PRISMABUILD_READER_HELPER_ROOT; do
    [[ -n "${!key:-}" ]] || { echo "Missing admitted PB context: $key" >&2; exit 2; }
    CTX+=(-e "$key=${!key}")
done
STAGE=()
if [[ -n "${PRISMABUILD_RESIDENCY_MAP:-}" ]]; then
    CTX+=(-e "PRISMABUILD_RESIDENCY_MAP=$PRISMABUILD_RESIDENCY_MAP")
    STAGE_ROOT=$(PYTHONPATH="$PB_CLIENT_ROOT/src" python3 -c \
        'import os; from prismabuild.client import read_residency_map; print(read_residency_map(os.environ["PRISMABUILD_RESIDENCY_MAP"])["stage_root"])')
    STAGE=(-v "$STAGE_ROOT:$STAGE_ROOT")
fi
OUT=$(realpath -m "$OUT")
if [[ "$MODE" == prepare-inputs ]]; then WORK="$OUT"; else WORK="$(dirname "$OUT")"; fi
mkdir -p "$WORK/home/torch_extensions" "$WORK/tmp" "$WORK/triton"
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
HEAD=$(git rev-parse HEAD)
# Input mounts are read-only. Only this action's output and the PB lease store
# remain writable. The PB Docker shim retains the admitted CPU mask and scope.
docker run --rm --gpus all --ipc=host --network=none --pid=host --cpuset-cpus "$CPUS" \
    --user "$(id -u):$(id -g)" -v "$CHECKOUT:/work:ro" -v "/mnt/shared:/mnt/shared:ro" \
    -v "$WORK:$WORK" -v "/mnt/shared/prismabuild-fleet:/mnt/shared/prismabuild-fleet" \
    "${STAGE[@]}" -e HOME="$WORK/home" -e TMPDIR="$WORK/tmp" \
    -e TORCH_EXTENSIONS_DIR="$WORK/home/torch_extensions" -e TRITON_CACHE_DIR="$WORK/triton" \
    -e PYTHONPATH="/work:/work/src:/work/experiments/t8r_speed:$PB_CLIENT_ROOT/src" \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MAX_JOBS=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e HOST_NAME="$(hostname)" \
    -e TESSERA_HEAD="$HEAD" -e ORACLE_IMAGE="$IMAGE" "${CTX[@]}" "${IMAGE_ENV[@]}" \
    --entrypoint bash -w /work "$IMAGE" -c \
    'source /work/experiments/cuda_home_shadow.sh "$TMPDIR"; exec python3 /work/experiments/t4_code/t4_fused_qualify.py "$@"' \
    -- "${ARGS[@]}"
