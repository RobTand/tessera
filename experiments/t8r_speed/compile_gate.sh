#!/usr/bin/env bash
# One PB CPU action (x86, no GPU): compile routed_fused_window.cu's device code for sm_121
# from source snapshots under the production library flags, then compare the SASS and
# resource usage per kernel (sass_cmp.py).
#
# The toolchain is the Spark image's, rebuilt off the Sparks:
# - nvcc and ptxas 13.0.88, the ptxas of the image's library builds (their .note.nv.tkinfo);
# - g++-13 from apt, the image's host compiler release;
# - the image's torch (2.13.0+cu130), cuBLAS, cuSPARSE, cuSOLVER and Python 3.12 headers.
#   They are host-side declarations only.
# The parity pair in pairs.txt (a Spark-built library object against the same source built
# here) is what licenses the rest.
#
# Usage: compile_gate.sh <gate_root>
#   Runs itself in GATE_IMAGE (the PB-declared ubuntu:24.04 image) under the action's CPU
#   affinity: apt installs g++-13 and the Python 3.12 headers as the image's root, then the
#   gate runs as the calling user, so everything under <gate_root> stays that user's.
#   <gate_root>/src/<variant>/routed_fused_window.cu  source snapshots, one per variant
#   <gate_root>/builds.txt  "<build> <variant> <library> [-D...]" per line; <library> is a
#                           routed_fused.LIBRARIES key (value, e4m3, e4m3mma) or e2m1
#   <gate_root>/bins.txt    "<name> <path>" per line: binaries built elsewhere (optional)
#   <gate_root>/pairs.txt   "<label> <ref> <cand> <expect>" per line (see sass_cmp.py)
# Env: GATE_IMAGE, GATE_NVCC, GATE_CUOBJDUMP, GATE_HDR (holds torch/include and
#      nvidia/cu13/include), GATE_RO (read-only mounts: the toolchain, headers and bins.txt
#      paths; default /mnt/shared/tessera-measurements), GATE_JOBS (parallel compiles;
#      default: the CPUs the action holds).
# Output under <gate_root>/out: <build>.cubin, <build>.nvcc.log, <build>.rc, spec.json,
# gate.json and gate.txt.  Exit status: sass_cmp.py's (0 = every verdict passed), 2 when a
# compile failed, 3 when apt failed.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
case ${1:-} in
--in-image)   # the image's root: the toolchain packages, then the gate as the caller
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      g++-13 python3.12-dev; } > /var/log/gate-apt.log 2>&1 || { echo "apt failed"; tail -20 /var/log/gate-apt.log; exit 3; }
  echo "apt ok"
  exec setpriv --reuid="$GATE_UID" --regid="$GATE_GID" --clear-groups -- bash "$0" --as-user "$2" ;;
--as-user) ROOT=$2 ;;
*)
  ROOT=$(realpath "${1:?gate_root}")
  IMAGE_REF=${GATE_IMAGE:?set GATE_IMAGE to the PB-declared ubuntu:24.04 image}
  CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
  NCPU=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
  RO=${GATE_RO:-/mnt/shared/tessera-measurements}
  mkdir -p "$ROOT/out/home" "$ROOT/out/tmp"
  echo "host=$(hostname) cpus=$CPUS image=$IMAGE_REF start=$(date -u +%FT%TZ)"
  docker run --rm --network=host --cpuset-cpus "$CPUS" \
    -v "$RO":"$RO":ro -v "$ROOT":"$ROOT" -v "$HERE":"$HERE":ro \
    -e GATE_UID="$(id -u)" -e GATE_GID="$(id -g)" -e GATE_NVCC="${GATE_NVCC:?}" \
    -e GATE_CUOBJDUMP="${GATE_CUOBJDUMP:?}" -e GATE_HDR="${GATE_HDR:?}" -e GATE_JOBS="${GATE_JOBS:-$NCPU}" \
    -e HOME="$ROOT/out/home" -e TMPDIR="$ROOT/out/tmp" -e PYTHONUNBUFFERED=1 \
    --entrypoint bash "$IMAGE_REF" "$HERE/compile_gate.sh" --in-image "$ROOT"
  rc=$?
  echo "end=$(date -u +%FT%TZ) rc=$rc"
  exit $rc ;;
esac
NVCC=${GATE_NVCC:?GATE_NVCC}
CUOBJDUMP=${GATE_CUOBJDUMP:?GATE_CUOBJDUMP}
HDR=${GATE_HDR:?GATE_HDR}
JOBS=${GATE_JOBS:-$(nproc)}
OUT=$ROOT/out
mkdir -p "$OUT"
echo "start $(date -u +%FT%TZ) arch=$(uname -m) uid=$(id -u) nproc=$(nproc) jobs=$JOBS"
sha256sum "$0" "$HERE/sass_cmp.py" "$ROOT"/builds.txt "$ROOT"/pairs.txt "$ROOT"/src/*/routed_fused_window.cu
[[ -f $ROOT/bins.txt ]] && sha256sum "$ROOT/bins.txt"
dpkg-query -W g++-13 libpython3.12-dev 2>/dev/null
"$NVCC" --version | tail -2
"$CUOBJDUMP" --version | tail -2
g++-13 --version | head -1
PY=$(command -v python3.12 || command -v python3)

libflags() {   # the library's own flags, in routed_fused._cflags's order
  case $1 in
    value)   echo "-DTORCH_EXTENSION_NAME=tessera_routed_fused_value -DTESSERA_ROUTED_FUSED_FP8=0 -DTESSERA_ROUTED_FUSED_MMA8=0 -gencode arch=compute_121,code=sm_121" ;;
    e4m3)    echo "-DTORCH_EXTENSION_NAME=tessera_routed_fused_e4m3 -DTESSERA_ROUTED_FUSED_FP8=1 -DTESSERA_ROUTED_FUSED_MMA8=0 -gencode arch=compute_121,code=sm_121" ;;
    e4m3mma) echo "-DTORCH_EXTENSION_NAME=tessera_routed_fused_mma_e4m3 -DTESSERA_ROUTED_FUSED_FP8=1 -DTESSERA_ROUTED_FUSED_MMA8=1 -gencode arch=compute_121,code=sm_121" ;;
    e2m1)    echo "-DTORCH_EXTENSION_NAME=tessera_routed_fused_e2m1 -DTESSERA_ROUTED_FUSED_FP8=0 -DTESSERA_ROUTED_FUSED_MMA8=0 -DTESSERA_ROUTED_FUSED_FP4=1 -gencode arch=compute_121a,code=sm_121a" ;;
    *) return 1 ;;
  esac
}
# torch.utils.cpp_extension's cuda_cflags (the Spark build.ninja), minus -c/-MD and with the
# headers' paths here; -cubin stops after ptxas.
COMMON=(-DTORCH_API_INCLUDE_EXTENSION_H -isystem "$HDR/torch/include"
        -isystem "$HDR/torch/include/torch/csrc/api/include" -isystem "$HDR/nvidia/cu13/include"
        -isystem /usr/include/python3.12 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
        -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr
        --compiler-options -fPIC -O3 -lineinfo -std=c++17)
build() {   # build variant library [extra...]
  local b=$1 v=$2 lib=$3; shift 3
  local lf t0=$SECONDS
  lf=$(libflags "$lib") || { echo "unknown library $lib" > "$OUT/$b.nvcc.log"; echo 2 > "$OUT/$b.rc"; return; }
  # shellcheck disable=SC2086
  "$NVCC" -ccbin g++-13 -cubin "${COMMON[@]}" $lf "$@" -o "$OUT/$b.cubin" \
    "$ROOT/src/$v/routed_fused_window.cu" > "$OUT/$b.nvcc.log" 2>&1
  local rc=$?
  echo "$rc" > "$OUT/$b.rc"
  echo "build $b ($v $lib $*) rc=$rc $((SECONDS - t0))s"
}
while read -r b v lib extra; do
  [[ -z $b || $b == \#* ]] && continue
  # shellcheck disable=SC2086
  build "$b" "$v" "$lib" $extra &
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
done < "$ROOT/builds.txt"
wait
fail=0
while read -r b _; do
  [[ -z $b || $b == \#* ]] && continue
  if [[ $(cat "$OUT/$b.rc" 2>/dev/null) != 0 ]]; then
    fail=1; echo "COMPILE FAILED: $b"; tail -15 "$OUT/$b.nvcc.log"
  fi
done < "$ROOT/builds.txt"
(( fail == 0 )) || exit 2
"$PY" - "$ROOT" "$OUT/spec.json" <<'PY'
import json, sys
root, out = sys.argv[1], sys.argv[2]
def rows(name):
    try:
        lines = open(f"{root}/{name}").read().splitlines()
    except FileNotFoundError:
        return []
    return [l.split() for l in lines if l.strip() and not l.startswith("#")]
bins = {r[0]: f"{root}/out/{r[0]}.cubin" for r in rows("builds.txt")}
bins.update({r[0]: r[1] for r in rows("bins.txt")})
json.dump({"bins": bins, "pairs": [r[:4] for r in rows("pairs.txt")]}, open(out, "w"), indent=1)
PY
"$PY" "$HERE/sass_cmp.py" "$CUOBJDUMP" "$OUT/spec.json" "$OUT/gate.json" | tee "$OUT/gate.txt"
rc=${PIPESTATUS[0]}
echo "end $(date -u +%FT%TZ) rc=$rc"
exit "$rc"
