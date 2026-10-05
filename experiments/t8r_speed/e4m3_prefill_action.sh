#!/usr/bin/env bash
# One PB action (one GB10, exclusive): the decode-once E4M3 prefill lane
# (tessera#931) -- its GPU tests, then its operator timings.
#   e4m3_prefill_action.sh <out_root>
# Step test: tests/test_e4m3_prefill.py through routed_fused_tests.sh (pinned image).
# Step bench: bench_e4m3_prefill.py through bench_t8r.sh.
# Step probe: bench_fp8_large_m.py (the M*K >= 2^25 slowdown), run before the
#   tests so it times an idle device.
# E4M3_PREFILL_TESTS overrides the test files (default tests/test_e4m3_prefill.py);
# TEST_XDIST=1 with E4M3_PREFILL_XDIST=N runs them under -n N.
# A later step still runs after an earlier failure; exit 1 if any failed, 2
# before running anything when ORACLE_IMAGE or TEST_RUNNER_SP is unset.
set -uo pipefail
OUT=${1:?out_root}
[[ -n "${ORACLE_IMAGE:-}" && -n "${TEST_RUNNER_SP:-}" ]] || { echo "REFUSED: ORACLE_IMAGE and TEST_RUNNER_SP are required" >&2; exit 2; }
mkdir -p "$OUT"
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
STEPS=" ${E4M3_PREFILL_STEPS:-test bench} "
[[ $STEPS == *" probe "* ]] && step probe env BENCH_PY=bench_fp8_large_m.py bash experiments/t8r_speed/bench_t8r.sh . "$OUT/probe"
XD=()
[[ "${TEST_XDIST:-0}" == 1 ]] && XD=(-n "${E4M3_PREFILL_XDIST:-4}" --dist worksteal)
# shellcheck disable=SC2086
[[ $STEPS == *" test "* ]] && step test bash experiments/routed_fused_tests.sh . "$OUT/test" \
  ${E4M3_PREFILL_TESTS:-tests/test_e4m3_prefill.py} "${XD[@]}" -q -rfEs --durations=15 \
  --surface-json "$OUT/test/surface.json"
[[ $STEPS == *" bench "* ]] && step bench env BENCH_PY=bench_e4m3_prefill.py bash experiments/t8r_speed/bench_t8r.sh . "$OUT/bench" ${E4M3_PREFILL_ARGS:-}
((${#FAILED[@]} == 0)) || { echo "FAILED: ${FAILED[*]}"; exit 1; }
echo "all steps rc=0"
