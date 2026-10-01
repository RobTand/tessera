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

# The host side needs glibc < 2.41: CUDA 13.0's crt/math_functions.h
# conflicts with the C23 rsqrt/rsqrtf declarations in newer glibc
# (dl380g10 runs Ubuntu 26.04). The image built FlashKDA with
# "GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0" (its .comment section), so the
# compile runs in ubuntu:24.04 with that g++-13. Only the compile runs in the
# container; fetch and comparison stay on the host.
FLAGS=(-cubin -gencode arch=compute_120f,code=sm_120f -std=c++17
  -DNDEBUG -DENABLE_FP8 -DUSE_CUDA -UPy_LIMITED_API
  -DTORCH_TARGET_VERSION=0x020B000000000000ULL
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -O3
  -I"$S/csrc" -I"$S/cutlass/include" -I"$S/cutlass/examples/common"
  -I"$S/cutlass/tools/util/include")
GXX_VERSION=${REBUILD_GXX_VERSION:-13.3.0-6ubuntu2~24.04.1}
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
printf '%q ' "$CU/bin/nvcc" -ccbin g++-13 "${FLAGS[@]}" \
  -o "$OUT/rebuilt-gxx13.cubin" "$S/csrc/smxx/fwd_launch.cu" > "$OUT/nvcc.cmd"
t0=$(date +%s)
if docker run --rm --network=host --cpuset-cpus "$CPUS" \
    -v "$TC":"$TC":ro -v "$OUT":"$OUT" -e TMPDIR="$OUT/tmp" \
    "${REBUILD_IMAGE:?set REBUILD_IMAGE to the PB-declared ubuntu:24.04 image}" \
    bash -c "set -e; apt-get update -qq >/dev/null; \
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        g++-13=$GXX_VERSION gcc-13=$GXX_VERSION cpp-13=$GXX_VERSION >/dev/null; \
      g++-13 --version | head -1; dpkg-query -W g++-13 gcc-13 libstdc++-13-dev libc6-dev || true; \
      rc=0; bash $OUT/nvcc.cmd > $OUT/nvcc-gxx13.log 2>&1 || rc=\$?; \
      chown -R $(id -u):$(id -g) $OUT; exit \$rc"; then
  echo "build gxx13 ok $(( $(date +%s) - t0 ))s"
else
  echo "build gxx13 FAILED rc=$?"; tail -20 "$OUT/nvcc-gxx13.log"; exit 1
fi
CUBIN=$OUT/rebuilt-gxx13.cubin
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
