#!/usr/bin/env bash
# Rebuild FlashKDA's device code with the image's nvcc and compare it to the
# image's cubin byte for byte (KDA fusion step 1, CPU only, x86).
#
# Usage: rebuild_sass_diff.sh OUT_DIR
#
# The question: does nvcc 13.0.88 with vLLM af5b4857's FlashKDA flags
# (cmake/external_projects/flashkda.cmake) reproduce the image's
# _flashkda_C sm_120f cubin exactly? If it does, a Tessera-built FlashKDA
# variant starts from the same SASS as stock, and any later SASS difference
# is the fusion's own.
set -euo pipefail
OUT=${1:?OUT_DIR}
ROOT=/mnt/shared/tessera-measurements/kda-fusion-20261001
TC=$ROOT/toolchain
CU=$TC/cuda-13.0.88/nvidia/cu13
OBJ=$TC/cuobjdump-13.2.86/nvidia/cu13/bin
IMG_CUBIN=$ROOT/image/_flashkda_C.abi3.1.sm_120.cubin
HERE=$(cd "$(dirname "$0")" && pwd)
FKDA_SHA=17a037d98da546deb4591e967cf961a43c034d8b
CUTLASS_SHA=5c149f52a436782210263fb2f19b354443a61c6a
mkdir -p "$OUT"/tmp "$OUT"/src
export TMPDIR=$OUT/tmp
exec > >(tee -a "$OUT/run.log") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) arch=$(uname -m)"
sha256sum "$0" "$HERE/cubin_cmp.py" "$IMG_CUBIN"

fetch() {  # fetch REPO_URL SHA DIR
  if [ "$(git -C "$3" rev-parse HEAD 2>/dev/null || true)" != "$2" ]; then
    rm -rf "$3"; git init -q "$3"
    git -C "$3" fetch -q --depth 1 "$1" "$2"
    git -C "$3" checkout -q FETCH_HEAD
  fi
  echo "$3 HEAD=$(git -C "$3" rev-parse HEAD) tree=$(git -C "$3" rev-parse 'HEAD^{tree}')"
}
fetch https://github.com/vllm-project/FlashKDA.git $FKDA_SHA "$OUT/src/FlashKDA"
fetch https://github.com/NVIDIA/cutlass.git $CUTLASS_SHA "$OUT/src/FlashKDA/cutlass"
S=$OUT/src/FlashKDA

"$CU/bin/nvcc" --version | tail -2
g++ --version | head -1

# vLLM af5b4857 flashkda.cmake: VLLM_GPU_FLAGS (torch common flags with the
# half/bf16 no-conversion defines stripped, plus -DENABLE_FP8), gencode for
# 12.0f, TORCH_TARGET_VERSION, USE_CUDA, and the FlashKDA CUDA options
# -UPy_LIMITED_API --expt-relaxed-constexpr --expt-extended-lambda
# --use_fast_math -O3. Host-side -g/-O2 from the build type does not reach
# ptxas (the image's tkinfo note records ptxas options "-arch sm_120f -m 64").
FLAGS=(-cubin -gencode arch=compute_120f,code=sm_120f -std=c++17
  -DNDEBUG -DENABLE_FP8 -DUSE_CUDA -UPy_LIMITED_API
  -DTORCH_TARGET_VERSION=0x020B000000000000ULL
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -O3
  -I"$S/csrc" -I"$S/cutlass/include" -I"$S/cutlass/examples/common"
  -I"$S/cutlass/tools/util/include")
build() {  # build TAG EXTRA...
  local tag=$1; shift
  local t0=$(date +%s)
  if "$CU/bin/nvcc" "${FLAGS[@]}" "$@" -o "$OUT/rebuilt-$tag.cubin" \
      "$S/csrc/smxx/fwd_launch.cu" > "$OUT/nvcc-$tag.log" 2>&1; then
    echo "build $tag ok $(( $(date +%s) - t0 ))s"
  else
    echo "build $tag FAILED rc=$?"; tail -20 "$OUT/nvcc-$tag.log"; return 1
  fi
}
# g++ 15 is the only host C++ toolchain on dl380g10; the image used GCC 13.3.
# Host compiler identity reaches device code only through preprocessor
# macros, so it is recorded, and the comparison decides whether it matters.
build gxx15 || build gxx15-unsupported -allow-unsupported-compiler
CUBIN=$(ls -1 "$OUT"/rebuilt-*.cubin | head -1)
sha256sum "$CUBIN"

python3 "$HERE/cubin_cmp.py" "$IMG_CUBIN" "$CUBIN" --out "$OUT/cubin_cmp.json" \
  | tee "$OUT/cubin_cmp.summary.json" | grep -v '"path"'

# Same disassembler on both sides: cuobjdump 13.2.86 (no 13.0 x86 wheel).
for side in image rebuilt; do
  f=$IMG_CUBIN; [ $side = rebuilt ] && f=$CUBIN
  PATH=$OBJ:$PATH "$OBJ/cuobjdump" -sass "$f" > "$OUT/$side.sass"
  PATH=$OBJ:$PATH "$OBJ/cuobjdump" -res-usage "$f" > "$OUT/$side.res-usage.txt"
done
diff -q "$OUT/image.res-usage.txt" "$OUT/rebuilt.res-usage.txt" \
  && echo "res-usage identical" || echo "res-usage DIFFERS"
if diff "$OUT/image.sass" "$OUT/rebuilt.sass" > "$OUT/sass.diff"; then
  echo "sass identical ($(grep -c '^        /\*[0-9a-f]*\*/' "$OUT/image.sass") instructions)"
else
  echo "sass DIFFERS: $(grep -c '^[<>]' "$OUT/sass.diff") lines"
fi
echo "end $(date -u +%FT%TZ)"
