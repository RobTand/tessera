#!/usr/bin/env bash
# One PB action: a kernel change's narrow bitwise gate on the served artifact,
# then its timing A/B, over the same arms (ab_arms.sh twice).
# Usage: d1_gate.sh <gate_root> <timing_root> <ref_arm> <arm> [<arm> ...]
#   1. <gate_root>: GATE_ARTIFACT's routed and dense groups (GATE_ROUTED,
#      GATE_DENSE) at GATE_MS, balanced routing plus the recorded routing under
#      <gate_root>/routing, --hash-only (each cell's output bytes twice per
#      process), forward then reverse.
#   2. <timing_root>: ab_arms.sh's default cells (the release-t8 routed groups
#      R1024 L10, R1088 L11, R832 L42 and the four dense families at M 1..2048,
#      each timed cell's output hashed; NCU at M 1 and 512).
# Both roots hold src-<arm> and ext-<arm>; each root's ab_summary.json carries
# its own verdict.  The gate runs first, so a timeout leaves the gate whole.
set -uo pipefail
GATE=${1:?gate_root}; TIMING=${2:?timing_root}; shift 2
H=experiments/t8r_speed/ab_arms.sh
echo "== part 1 (bitwise gate) start=$(date -u +%FT%TZ)"
BENCH_ARTIFACT=${GATE_ARTIFACT:?} AB_ROUTED=${GATE_ROUTED:?} AB_DENSE=${GATE_DENSE:?} AB_MS=${GATE_MS:?} \
  AB_ROUTING="$GATE/routing" AB_BENCH_ARGS=--hash-only AB_STEPS="routed dense" bash $H "$GATE" "$@"
rc1=$?
echo "== part 2 (timing A/B) start=$(date -u +%FT%TZ)"
bash $H "$TIMING" "$@"
rc2=$?
echo "== d1_gate rc1=$rc1 rc2=$rc2 end=$(date -u +%FT%TZ)"
exit $(( rc1 || rc2 ))
