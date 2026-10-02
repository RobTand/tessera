#!/usr/bin/env bash
# CPU-only compile gate for the MLA pass-buffer schedule, run INSIDE the pinned
# serving image with no GPU. It compiles the serving source twice -- once with
# TESSERA_MLA_P0_BUFFERS=0 (the shipped L0) and once with =1 (the new schedule)
# -- and dumps ptxas resource usage and SASS for both, so a register/spill
# regression is visible before any GPU request.
#
# --container-image on pbrun DECLARES placement only; it does NOT run the
# command in the image. This wrapper is the thing that actually starts the
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
  -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e MAX_JOBS=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PYTHONUNBUFFERED=1 "${IMAGE_ENV[@]}" -w "$OUT" --entrypoint bash "$IMAGE" -c '
    set -euo pipefail
    D=/usr/local/lib/python3.12/dist-packages/flashinfer/data
    INC=$D/include
    CCCL=(-I$D/cccl/cub -I$D/cccl/libcudacxx/include -I$D/cccl/thrust)
    FLAGS=(-std=c++17 --threads 1 -use_fast_math -DTESSERA_MLA_DECLARED_FAST_MATH=1
      -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3
      -DFLASHINFER_ENABLE_FP8_E5M2 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1
      -DNDEBUG -O3 -gencode=arch=compute_121a,code=sm_121a
      -D_GLIBCXX_USE_CXX11_ABI=1 -DPy_LIMITED_API=0x03090000
      --expt-relaxed-constexpr -static-global-template-stub=false
      -Xfatbin=-compress-all --compress-mode=size)
    SRC=/work/src/tessera/serving/csrc/mla_prefill_mg.cu
    echo "nvcc: $(nvcc --version | tail -2 | head -1)"
    # name:flags -- 0 L0, 1 pass-buffers, 1m pass-buffers + wrong-pass mutant
    build() { # name extra-defines...
      local name=$1; shift
      echo "== $name : $* =="
      nvcc "${CCCL[@]}" -isystem "$INC" "${FLAGS[@]}" "$@" \
        -Xptxas -v -c "$SRC" -o "$name.o" 2> "compile_$name.log"
      grep -E "error|registers|spill|lmem|smem|Function properties|Compiling entry" "compile_$name.log" || true
      cuobjdump -sass "$name.o" > "$name.sass" 2>/dev/null
      cuobjdump -res-usage "$name.o" > "$name.res" 2>&1
      echo "sass_lines=$(wc -l < $name.sass) obj_sha256=$(sha256sum $name.o | cut -d" " -f1)"
    }
    for n in 0 1; do
      build "p0_$n" -DTESSERA_MLA_P0_BUFFERS=$n
    done
    build p0_1m -DTESSERA_MLA_P0_BUFFERS=1 -DTESSERA_MLA_P0_WRONG_PASS=1
    echo "== L0 (P0_BUFFERS=0) resource usage =="; cat p0_0.res
    echo "== P0-buffers (P0_BUFFERS=1) resource usage =="; cat p0_1.res
    echo "== wrong-pass mutant (P0_BUFFERS=1,WRONG_PASS=1) resource usage =="; cat p0_1m.res
  ' | tee "$OUT/compile.log"
echo "end=$(date -u +%FT%TZ)"
