#!/usr/bin/env bash
# Selected f16 synthetic probe stages, not complete issue 739 acceptance.
# All benchmark, NCU, and disassembly execution requires PrismaBuild admission.
# AB_STEPS=analyze validates retained artifacts without Docker or a GPU.
set -euo pipefail
OUT=$(realpath -m "${1:?out_root}")
HERE=$(cd "$(dirname "$0")" && pwd)
CHECKOUT=$(realpath "$HERE/../..")
MASTER_SRC=$(realpath "${MASTER_SRC:-$CHECKOUT/pb-arms/master-src}")
NEW_SRC=$(realpath "${NEW_SRC:-$CHECKOUT/src}")
STEPS=${AB_STEPS:-timed ncu sass}
PYTHON=${AB_PYTHON:-python3}
CASES=${AB_CASES:-r4,r4+q,r3+q}
MODES=${AB_MODES:-0,2}
MS=${AB_MS:-1,512,2048}
VALIDATOR="$HERE/f16_prefetch_accept.py"
VALIDATE=("$PYTHON" "$VALIDATOR" --out "$OUT" --master-src "$MASTER_SRC" --new-src "$NEW_SRC"
          --cases "$CASES" --modes "$MODES" --ms "$MS")
mkdir -p "$OUT"
rm -f "$OUT/ab_acceptance.json"
for stage in $STEPS; do
  case "$stage" in timed|ncu|sass|analyze) ;; *) echo "unknown probe stage: $stage" >&2; exit 2 ;; esac
done
if [[ " $STEPS " != " analyze " ]]; then
  : "${PRISMABUILD_ACTION_KEY:?probe execution requires PrismaBuild admission}"
  IMG=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared image}
fi
FAILED=()
step() {
  local name=$1; shift
  local rc=0
  "$@" > "$OUT/$name.log" 2>&1 || rc=$?
  echo "step=$name rc=$rc"
  if ((rc)); then cat "$OUT/$name.log"; FAILED+=("$name:$rc"); fi
}
run_arm() {
  local source=$1 arm=$2
  shift 2
  env BENCH_PY=bench_pairs.py BENCH_SRC="$source" NATIVE_CONTAINER_SRC="$source" \
    BENCH_PROJECT_FILE="$CHECKOUT/pyproject.toml" \
    NATIVE_CONTAINER_EXT="$OUT/$arm/home/torch_extensions" \
    BENCH_EXT_DIR="" \
    bash "$HERE/bench_t8r.sh" "$CHECKOUT" "$OUT/$arm" "$@"
}
if [[ " $STEPS " == *" timed "* ]]; then
  for arm in base new newb baseb; do mkdir -p "$OUT/$arm/home/torch_extensions"; done
  ARGS=(--cases "$CASES" --modes "$MODES" --ms "$MS" --library e4m3 --warmup 10 --iters 50 --power-s 0)
  step r-base run_arm "$MASTER_SRC" base "${ARGS[@]}"
  step r-new run_arm "$NEW_SRC" new "${ARGS[@]}"
  step rb-new run_arm "$NEW_SRC" newb "${ARGS[@]}"
  step rb-base run_arm "$MASTER_SRC" baseb "${ARGS[@]}"
fi
if [[ " $STEPS " == *" ncu "* ]]; then
  for arm in ncu-base ncub-new; do mkdir -p "$OUT/$arm/home/torch_extensions"; done
  step ncu-base env BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel \
    BENCH_PY=bench_pairs.py BENCH_SRC="$MASTER_SRC" NATIVE_CONTAINER_SRC="$MASTER_SRC" \
    BENCH_PROJECT_FILE="$CHECKOUT/pyproject.toml" NATIVE_CONTAINER_EXT="$OUT/ncu-base/home/torch_extensions" \
    BENCH_EXT_DIR="" \
    bash "$HERE/bench_t8r.sh" "$CHECKOUT" "$OUT/ncu-base" --cases r4 --modes 0 --ms 512 --library e4m3 --ncu
  step ncu-new env BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel \
    BENCH_PY=bench_pairs.py BENCH_SRC="$NEW_SRC" NATIVE_CONTAINER_SRC="$NEW_SRC" \
    BENCH_PROJECT_FILE="$CHECKOUT/pyproject.toml" NATIVE_CONTAINER_EXT="$OUT/ncub-new/home/torch_extensions" \
    BENCH_EXT_DIR="" \
    bash "$HERE/bench_t8r.sh" "$CHECKOUT" "$OUT/ncub-new" --cases r4 --modes 0 --ms 512 --library e4m3 --ncu
fi
if [[ " $STEPS " == *" sass "* ]]; then
  source "$HERE/../runtime_image.sh"
  runtime_image_require "$IMG"
  for arm in base new; do
    if ! lib=$("${VALIDATE[@]}" --native-path "$arm"); then FAILED+=("sass-$arm:identity"); continue; fi
    step "sass-$arm" docker run --rm --network=none --user "$(id -u):$(id -g)" \
      -v "$OUT":"$OUT" --entrypoint cuobjdump "$IMG" -sass "$lib"
    if [[ -f "$OUT/sass-$arm.log" ]]; then cp "$OUT/sass-$arm.log" "$OUT/sass-$arm.txt"; fi
  done
fi
step analyze "${VALIDATE[@]}" --steps "$STEPS"
if ((${#FAILED[@]})); then
  echo "FAILED=[${FAILED[*]}]"
  rm -f "$OUT/ab_acceptance.json"
  exit 1
fi
cat "$OUT/analyze.log"
echo "SELECTED_STAGES_DONE stages=[$STEPS] issue_acceptance=not_established"
