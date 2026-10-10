#!/usr/bin/env bash
# Probe whether the f16 activation prefetch reaches PTX and SASS (tessera#739).
#
# Builds the fused routed kernel TU twice with nvcc -- once from master's
# source, once from this checkout's -- under the f16 library's own flags, and
# counts the prefetch instruction in the PTX and in the disassembled object.
# A volatile-asm hint must appear in both; SASS counts that differ only by the
# new prefetch instructions prove the gate's codegen claim directly.
#
# Usage: f16_prefetch_ptx.sh <out_root>
# Requires ORACLE_IMAGE (a CUDA image with nvcc, cuobjdump and torch).
set -uo pipefail
OUT=$(realpath -m "${1:?out_root}")
MASTER_CU=$(realpath -m "${MASTER_CU:-$PWD/pb-arms/master-src/tessera/serving/csrc/routed_fused_window.cu}")
NEW_CU=$(realpath -m "${NEW_CU:-$PWD/src/tessera/serving/csrc/routed_fused_window.cu}")
IMG=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../runtime_image.sh"
runtime_image_require "$IMG" || exit 2
cp "$MASTER_CU" "$OUT/base.cu"
cp "$NEW_CU" "$OUT/new.cu"
sha256sum "$OUT/base.cu" "$OUT/new.cu"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
docker run --rm --network=none --ipc=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" -v "$OUT":"$OUT" \
  -e HOME="$OUT" -e TMPDIR="$OUT" \
  --entrypoint bash "$IMG" -c "
set -uo pipefail
cd '$OUT'
command -v nvcc cuobjdump || exit 3
INC=\$(python3 -c 'import sysconfig, torch.utils.cpp_extension as e; print(\" \".join([\"-I\"+p for p in e.include_paths()] + [\"-I\"+sysconfig.get_paths()[\"include\"]]))')
FLAGS=\"-O3 -lineinfo -std=c++17 -DTESSERA_ROUTED_FUSED_FP8=1 -DTESSERA_ROUTED_FUSED_MMA8=0 -gencode arch=compute_121,code=sm_121\"
for a in base new; do
  nvcc \$FLAGS \$INC --ptx \$a.cu -o \$a.ptx || exit 4
  nvcc \$FLAGS \$INC -c \$a.cu -o \$a.o || exit 5
  cuobjdump -sass \$a.o > \$a.sass || exit 6
done
" > "$OUT/build.log" 2>&1
rc=$?
tail -5 "$OUT/build.log"
((rc == 0)) || { echo "BUILD FAILED rc=$rc"; exit 1; }
for a in base new; do
  echo "== $a"
  echo "ptx prefetch lines: $(grep -ci 'prefetch' "$OUT/$a.ptx")"
  grep -ai -m4 'prefetch' "$OUT/$a.ptx" | cut -c1-160
  echo "sass PREFETCH lines: $(grep -ci 'prefetch' "$OUT/$a.sass")"
  grep -ai -m4 'prefetch' "$OUT/$a.sass" | cut -c1-160
done
echo "ptx bytes: base=$(stat -c %s "$OUT/base.ptx") new=$(stat -c %s "$OUT/new.ptx")"
echo "sass bytes: base=$(stat -c %s "$OUT/base.sass") new=$(stat -c %s "$OUT/new.sass")"
echo ALL_DONE
