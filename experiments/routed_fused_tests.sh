#!/usr/bin/env bash
# Run the fused routed window lane's GPU tests (tests/test_routed_fused_window.py,
# tessera#640) inside the pinned serving image.  Submitted through PrismaBuild
# as the admitted action's command; same container discipline as
# routed_census.sh (checkout read-only at /work, every write under <out_dir>).
#
# The image ships no pytest.  The runner is borrowed, pure Python, from a host
# venv of the SAME interpreter minor (3.12): only pytest, _pytest, pluggy,
# iniconfig, packaging and the py.py shim are copied under <out_dir>/runner-sp and appended to
# PYTHONPATH, so the image's torch, triton and vLLM are the ones imported.
#   routed_fused_tests.sh <checkout> <out_dir> [pytest args...]
# Environment:
#   ORACLE_IMAGE      the immutable image reference (required)
#   TEST_RUNNER_SP    site-packages holding the pure-Python runner (required)
#   TEST_RO_MOUNTS    host directories mounted read-only at the same path
#   TEST_LOCAL_TMP    1: TMPDIR and pytest's basetemp on a container tmpfs
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
SP=${TEST_RUNNER_SP:?set TEST_RUNNER_SP to a python3.12 site-packages holding the test runner}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
EXT=${BENCH_EXT_DIR:-$OUT/torch-ext}
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton" "$EXT" "$OUT/runner-sp"
# py.py is the one-file shim _pytest/compat.py imports (the retired ``py``
# library's name); a runner copy without it fails at import.
for pkg in pytest _pytest pluggy iniconfig packaging py.py; do
  [[ -e "$SP/$pkg" ]] || { echo "missing $SP/$pkg" >&2; exit 2; }
  rm -rf "$OUT/runner-sp/$pkg"
  cp -r "$SP/$pkg" "$OUT/runner-sp/$pkg"
done
XDIST=()
if [[ "${TEST_XDIST:-0}" == 1 ]]; then
  for pkg in xdist execnet; do
    [[ -d "$SP/$pkg" ]] || { echo "missing $SP/$pkg" >&2; exit 2; }
    rm -rf "$OUT/runner-sp/$pkg"
    cp -r "$SP/$pkg" "$OUT/runner-sp/$pkg"
  done
  XDIST=(-p xdist.plugin)
fi
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
STATE=${TESSERA_STATE:-$(git -C "$CHECKOUT" status --short 2>/dev/null | tr '\n' ';' || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD state=[$STATE] image=$IMAGE_REF"
# TEST_RO_MOUNTS: host directories tests read (fixtures such as the A4 wires),
# mounted read-only at the same path.  TEST_LOCAL_TMP=1 puts TMPDIR and pytest's
# basetemp on a container tmpfs: flock on an NFS out directory fails with EBADF.
EXTRA=()
if [[ -n "${BENCH_SRC:-}" ]]; then
  [[ -f "$BENCH_SRC/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "BENCH_SRC is not a tessera src tree" >&2; exit 2; }
  EXTRA+=(-v "$BENCH_SRC":/work/src:ro)
fi
if [[ -n "${BENCH_EXT_DIR:-}" ]]; then EXTRA+=(-v "$EXT":"$EXT"); fi
for d in ${TEST_RO_MOUNTS:-}; do [[ -d "$d" ]] || { echo "missing $d" >&2; exit 2; }; EXTRA+=(-v "$d":"$d":ro); done
BT="$OUT/tmp/pytest-tmp"; TD="$OUT/tmp"
if [[ "${TEST_LOCAL_TMP:-0}" == 1 ]]; then EXTRA+=(--tmpfs /pbtmp:rw,exec,size=8g); BT=/pbtmp/pytest-tmp; TD=/pbtmp; fi
docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT" \
  -e HOME="$OUT/home" -e TMPDIR="$TD" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$EXT" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/work/src:/work/tests:/work/experiments:"$OUT/runner-sp" \
  -e HOST_NAME="$(hostname)" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
  -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" -e TESSERA_STATE="$STATE" \
  -e TESSERA_ROUTED_FUSED_VERBOSE="${TESSERA_ROUTED_FUSED_VERBOSE:-}" \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  "${IMAGE_ENV[@]}" "${EXTRA[@]}" ${TESSERA_ROUTED_ENV:+-e "$TESSERA_ROUTED_ENV"} \
  --entrypoint python3 -w /work "$IMAGE_REF" \
  -m pytest -p no:cacheprovider "${XDIST[@]}" -q -rA --junitxml="$OUT/junit.xml" \
  -o "cache_dir=$OUT/tmp/pytest-cache" --basetemp="$BT" "$@"
