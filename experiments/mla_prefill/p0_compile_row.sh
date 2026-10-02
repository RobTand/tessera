#!/usr/bin/env bash
# CPU-only compile gate for the MLA pass-buffer schedule, run INSIDE the pinned
# serving image with no GPU. It compiles the serving source twice -- once with
# TESSERA_MLA_P0_BUFFERS=0 (the shipped L0) and once with =1 (the new schedule)
# -- and dumps ptxas resource usage and SASS for both, so a register/spill
# regression is visible before any GPU request.
#
# --container-image on pbrun DECLARES placement only; it does NOT run the
# command in the image. This wrapper actually starts the
# pinned CUDA container (experiments/runtime_image.sh resolves the digest and
# passes its declared identity in), the same way experiments/t8r_speed/
# build_ext.sh does for its libraries.
#
#   MLA_IMAGE=<digest ref> p0_compile_row.sh <out_dir>
#
# Refuses a floating image reference. Never passes --gpus.
set -euo pipefail
OUT=$(realpath -m "${1:?out_dir}")
mkdir -p "$OUT/home" "$OUT/tmp"
HERE=$(dirname "$(realpath "$0")")
CHECKOUT=$(realpath "$HERE/../..")
IMAGE=${MLA_IMAGE:?set MLA_IMAGE to the immutable PB-declared serving image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE" || exit 2
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
CPUS=$(python3 -c 'import os;print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
SRC="$CHECKOUT/src/tessera/serving/csrc/mla_prefill_mg.cu"
echo "host=$(hostname) cpus=$CPUS image=$IMAGE kernel_sha256=$(sha256sum "$SRC" | cut -d' ' -f1)" \
     "start=$(date -u +%FT%TZ)"
# --network=none, no --gpus: a compile gate, never a GPU probe.
docker run --rm --network=none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" \
  --read-only \
  -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e MAX_JOBS=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work/src \
  -e PYTHONUNBUFFERED=1 "${IMAGE_ENV[@]}" -w "$OUT" --entrypoint bash "$IMAGE" -c '
    set -euo pipefail
    source /work/experiments/cuda_home_shadow.sh "$HOME"
    # One admitted process owns these three dependent build selections.
    # Separate directories retain each final DSO and its actual build.ninja.
    python3 /work/experiments/mla_prefill/p0_build_only.py --out "$PWD/p0_0"
    python3 /work/experiments/mla_prefill/p0_build_only.py --out "$PWD/p0_1" --p0-buffers
    python3 /work/experiments/mla_prefill/p0_build_only.py --out "$PWD/p0_1m" --p0-buffers --p0-wrong-pass
  ' | tee "$OUT/compile.log"
echo "end=$(date -u +%FT%TZ)"
