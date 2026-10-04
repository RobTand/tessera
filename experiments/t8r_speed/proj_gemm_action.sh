#!/usr/bin/env bash
# One PB measurement action (one GB10, exclusive): BF16 projection GEMM levers
# at GLM-5.3-Flash's served prefill shapes (M = 2048, TP2 per rank).
#   proj_gemm_action.sh <out_root>
# Step 1 (proj): bench_proj_gemm.py -- cuBLAS default vs every cuBLASLt
#   heuristic algo, a Triton sweep, E4M3 _scaled_mm + its quantiser, and
#   same-input concatenations.
# Step 2 (t8): bench_dense_module.py -- the Tessera T-8 dense lane at the KDA
#   input and o_proj shapes, M = 2048, with the bf16 and _scaled_mm references.
# A later step still runs after an earlier failure; the action exits 1 if any
# step failed and 2 before running anything when ORACLE_IMAGE is unset.
set -uo pipefail
OUT=${1:?out_root}
[[ -n "${ORACLE_IMAGE:-}" ]] || { echo "REFUSED: ORACLE_IMAGE is unset" >&2; exit 2; }
mkdir -p "$OUT"
H=experiments/t8r_speed/bench_t8r.sh
FAILED=()
step() {
  local name=$1; shift
  echo "== step $name start=$(date -u +%FT%TZ) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "== step $name rc=$rc end=$(date -u +%FT%TZ)"
  tail -n 40 "$OUT/$name.log"
  ((rc == 0)) || FAILED+=("$name:$rc")
}
STEPS=" ${PROJ_STEPS:-proj t8} "
[[ $STEPS == *" proj "* ]] && step proj env BENCH_PY=bench_proj_gemm.py bash $H . "$OUT/proj" ${PROJ_ARGS:-}
[[ $STEPS == *" t8 "* ]] && step t8 env BENCH_PY=bench_dense_module.py bash $H . "$OUT/t8" \
  --modules kda_in,o_proj --ms 2048 --refs --power-ms 2048 --numerics-ms 2048
((${#FAILED[@]} == 0)) || { echo "FAILED: ${FAILED[*]}"; exit 1; }
echo "all steps rc=0"
