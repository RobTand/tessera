#!/usr/bin/env bash
# Build routed_fused_window.cu from two sources for each of its three
# libraries (value, E4M3, E4M3-MMA) and compare their SASS per kernel.
# Usage: TORCH_INCLUDE=<torch include dir> sass_identity.sh OLD.cu NEW.cu OUT_DIR
#   TORCH_INCLUDE: the include/ directory of the torch the extension builds
#   against (``torch.utils.cpp_extension.include_paths()[0]``).
#   ARCH (default 121): the -gencode target, as routed_fused's build passes it.
#   NVCC (default /usr/local/cuda/bin/nvcc).
# The flags are routed_fused._cflags' (-O3 -lineinfo -std=c++17 and the two
# library defines) plus torch's extension defines. Host-only: compiles, no GPU.
set -uo pipefail
OLD=${1:?old .cu}; NEW=${2:?new .cu}; OUT=${3:?out dir}
TI=${TORCH_INCLUDE:?set TORCH_INCLUDE}
ARCH=${ARCH:-121}; NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
PYI=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
mkdir -p "$OUT"; export TMPDIR="$OUT/tmp"; mkdir -p "$TMPDIR"
HERE=$(cd "$(dirname "$0")" && pwd)
build() {  # tag src fp8 mma8
  local tag=$1 src=$2
  cp "$src" "$OUT/$tag.cu"
  "$NVCC" -DTORCH_EXTENSION_NAME=x -DTORCH_API_INCLUDE_EXTENSION_H -isystem "$TI" \
    -isystem "$TI/torch/csrc/api/include" -isystem "$PYI" \
    -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options -fPIC -O3 -lineinfo -std=c++17 \
    -DTESSERA_ROUTED_FUSED_FP8="$3" -DTESSERA_ROUTED_FUSED_MMA8="$4" \
    -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xptxas -v -c "$OUT/$tag.cu" -o "$OUT/$tag.o" > "$OUT/$tag.log" 2>&1 \
    && "${NVCC%/nvcc}/cuobjdump" -sass "$OUT/$tag.o" > "$OUT/$tag.sass"
}
rc=0
for spec in "value 0 0" "e4m3 1 0" "mma8 1 1"; do
  set -- $spec
  build "old_$1" "$OLD" "$2" "$3" & build "new_$1" "$NEW" "$2" "$3" & wait
  [[ -s "$OUT/old_$1.sass" && -s "$OUT/new_$1.sass" ]] || { echo "$1: build failed (see $OUT/*_$1.log)"; rc=1; continue; }
  echo "== $1"
  python3 "$HERE/sass_compare.py" "$OUT/old_$1.sass" "$OUT/new_$1.sass" | grep -v '^NEW'
  echo "spills: $(grep -c 'bytes spill' "$OUT/new_$1.log") kernels, $(grep 'bytes spill' "$OUT/new_$1.log" | grep -vc ' 0 bytes spill stores, 0 bytes spill loads') with a spill"
done
exit $rc
