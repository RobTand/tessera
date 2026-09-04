#!/usr/bin/env bash
# Issue #104, one GPU slot: the rate preflight over every checkpoint we hold,
# then the two route censuses.
#
#   0. a re-export of the rate-constrained arm THROUGH --require-lane, so the
#      artifact declares the lane it was built for (requires_lanes in its
#      manifest) and the census requires it off the bytes rather than off a
#      shell history.  The encoder is deterministic, so this also reports
#      whether the re-export's wire is byte-identical to the first one's.
#   1. tools/tessera_lane_preflight.py over the new rate-constrained arm and
#      all six allocated checkpoints (read-only), reading EVERY unit's rates
#      off the wire through parse_fused + parse_unit_artifact.
#   2. experiments/ts104_gemv_census.sh -- the readable arm and the untouched
#      R1006 baseline, same command, streamed + eager.
#
# One job because re-queueing between the legs on a two-slot box spreads one
# experiment over hours (the lesson ts91_chain.sh records).
set -x
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
RUNS=${RUNS:-/home/rob/tessera-runs/ts104}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
NEW=${NEW:-/mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024}
ALLOC=/mnt/shared/tessera-runs/allocated
mkdir -p "$RUNS"

sha_before=$(sha256sum "$NEW/model.safetensors" 2>/dev/null | cut -d" " -f1)
"$PY" "$WT/experiments/export_tessera_serving.py" /home/rob/models/Qwen3-0.6B "$NEW" \
  --grid E4M3 --q256 1024 --require-lane tessera_window_gemv \
  > "$RUNS/export-requirelane.log" 2>&1
tail -3 "$RUNS/export-requirelane.log"
sha_after=$(sha256sum "$NEW/model.safetensors" | cut -d" " -f1)
echo "wire sha before=$sha_before after=$sha_after"
[ "$sha_before" = "$sha_after" ] && echo "BYTE-IDENTICAL re-export" || echo "WIRE MOVED (investigate)"
grep -o '"requires_lanes": \[[^]]*\]' "$NEW/tessera_serving_manifest.json"

"$PY" "$WT/tools/tessera_lane_preflight.py" "$NEW" \
  --lane tessera_window_gemv --json "$RUNS/preflight-new.json" 2>&1 | tail -20

# The plan-time refusal, on the rung uniform-R1006 was built at.  Nothing is
# encoded: the log must carry no "[n/196]" progress line and the output
# directory must not exist afterwards.
REFUSED_OUT=/home/rob/tmp/ts104-refused-must-not-exist
rm -rf "$REFUSED_OUT"
"$PY" "$WT/experiments/export_tessera_serving.py" /home/rob/models/Qwen3-0.6B "$REFUSED_OUT" \
  --grid E4M3 --q256 1006 --require-lane tessera_window_gemv \
  > "$RUNS/refusal-R1006.log" 2>&1
echo "plan refusal exit: $?"
cat "$RUNS/refusal-R1006.log"
echo "encode progress lines in that log: $(grep -c '\[[0-9]*/[0-9]*\]' "$RUNS/refusal-R1006.log")"
echo "output dir after the refusal: $(ls -d "$REFUSED_OUT" 2>&1)"

"$PY" "$WT/tools/tessera_lane_preflight.py" \
  "$ALLOC/qwen3-0.6b-uniform-R750" "$ALLOC/qwen3-0.6b-uniform-R1006" \
  "$ALLOC/qwen3-0.6b-uniform-R1262" "$ALLOC/qwen3-0.6b-alloc-3.0" \
  "$ALLOC/qwen3-0.6b-alloc-4.0" "$ALLOC/qwen3-0.6b-alloc-5.0" \
  --lane tessera_window_gemv --json "$RUNS/preflight-allocated.json" 2>&1 | tail -40

WT="$WT" RUNS="$RUNS" bash "$WT/experiments/ts104_gemv_census.sh"

# --- (g): the served pair, same session, same box, same corpus -------------
# NOT byte-matched: R1024 is +1.8% wire over R1006 (root 4.0 vs 3.93), which is
# the price of the constraint at this rung and is stated as such.  Served
# through Tessera's own plugin, streamed (the residency in which the GEMV lane
# exists at all), eager.
export TS=$WT RUNS
for arm in "R1024:$NEW" "R1006:$ALLOC/qwen3-0.6b-uniform-R1006"; do
  name=${arm%%:*}; model=${arm#*:}
  echo "=== served KL $name  $model  $(date -Is)"
  bash "$WT/experiments/tessera_plugin_served.sh" "$model" "ts104-$name-streamed" streamed \
    2>&1 | tail -30
  echo "served KL $name exit: ${PIPESTATUS[0]}"
done

# --- (e): the tests, and their FAIL-BEFORE on a pristine master -------------
NEWTESTS="tests/test_lane_reachability.py tests/test_census_engagement.py"
echo "=== new tests on this branch"
(cd "$WT" && "$PY" -m pytest -q $NEWTESTS 2>&1 | tail -25)
echo "=== impacted tests (tools/impacted_tests.py --ref master...HEAD)"
(cd "$WT" && "$PY" tools/impacted_tests.py --ref master...HEAD 2>&1 | tee "$RUNS/impacted.txt" | tail -20)
IMPACTED=$(grep -oE 'tests/[A-Za-z0-9_./-]+\.py' "$RUNS/impacted.txt" | sort -u | tr '\n' ' ')
echo "impacted: $IMPACTED"
if [ -n "$IMPACTED" ]; then (cd "$WT" && "$PY" -m pytest -q $IMPACTED 2>&1 | tail -25); fi
echo "=== FAIL-BEFORE: the same two files against master 42615e4 (AGENTS.md rule 8)"
MASTER=/home/rob/tmp/ts104-master
cp $NEWTESTS "$MASTER/tests/"
(cd "$MASTER" && "$PY" -m pytest -q tests/test_lane_reachability.py tests/test_census_engagement.py 2>&1 | tail -30)
rm -f "$MASTER/tests/test_lane_reachability.py" "$MASTER/tests/test_census_engagement.py"
echo "CHAIN DONE $(date -Is)"
