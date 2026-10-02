#!/usr/bin/env bash
# One admitted kernel quantum; independent invocations share no mutable build/output state.
set -euo pipefail
SCRIPT=${MLA_SCRIPT:-mask_gate.py}
[[ "$SCRIPT" = mask_gate.py || "$SCRIPT" = mask_abba.py ]] || exit 2
OUT=$(realpath -m "${1:?output_dir}");shift
HERE=$(dirname "$(realpath "$0")");CHECKOUT=$(realpath "$HERE/../..")
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "${MLA_IMAGE:?immutable PB-declared image required}"
mkdir -p "$OUT/home" "$OUT/tmp"
CPUS=$(python3 -c 'import os;print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
IMAGE_ENV=();while IFS= read -r line;do [[ -z "$line" ]]||IMAGE_ENV+=(-e "$line");done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" \
 -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT" -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" \
 -e TORCH_EXTENSIONS_DIR="$OUT/home/torch_extensions" -e PYTHONPATH=/work/src:/work/experiments/mla_prefill \
 -e MAX_JOBS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
 -e MLA_SCRIPT="$SCRIPT" -e HOST_NAME="$(hostname)" "${IMAGE_ENV[@]}" -w /work --entrypoint bash "$MLA_IMAGE" -c \
 'source experiments/cuda_home_shadow.sh "$HOME"; python3 "experiments/mla_prefill/$MLA_SCRIPT" "$@"' _ --out "$OUT" "$@"
