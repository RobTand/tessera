#!/usr/bin/env bash
# Observe PTX hints and SASS opcodes under the f16 flags for issue 739.
# Cache-control opcodes are candidates. Text counts cannot prove hint removal.
# This probe supplies no timing, wait, or complete acceptance result.
#
# Usage: f16_prefetch_ptx.sh <out_root>
# Requires ORACLE_IMAGE (a CUDA image with nvcc, cuobjdump and torch).
set -euo pipefail
: "${PRISMABUILD_ACTION_KEY:?codegen probe requires PrismaBuild admission}"
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
HERE=$(cd "$(dirname "$0")" && pwd)
for a in base new; do
  python3 "$HERE/f16_prefetch_accept.py" --ptx "$OUT/$a.ptx" --sass "$OUT/$a.sass" \
    > "$OUT/$a-codegen.json" || exit 1
  cat "$OUT/$a-codegen.json"
done
echo "SELECTED_STAGES_DONE stages=[codegen] hint_retention=unknown issue_acceptance=not_established"
