#!/usr/bin/env bash
# One PB action (sparklina, exclusive): routed-kernel timing + NCU at the A8S
# release artifact's R1024 layer 10, for the L512 TTFT work (2026-09-30).
#   kern_action.sh <out_root> <arm> [<arm> ...]
# Each arm's source is <out_root>/src-<arm>/src; the harness is the cwd
# (this tree).  Steps per arm:
#   t-<arm>   timing: M 1,512,2048 balanced + recorded routing (T8R L512/L8192 ids)
#   n-<arm>   NCU (full sections + source counters) at M 512, balanced routing
#             (KERN_NCU_MS other Ms; KERN_NCU_ROUTING DIR adds its recorded cases)
# Timing arms run forward then reverse (t-<arm>, then tb-<arm>) for drift symmetry.
# Build every arm's libraries first, in a separate non-measurement row on any GB10
# (build_ext.sh <out_root>/src-<arm> <out_root>/ext-<arm>); this action only loads
# them (BENCH_EXT_DIR) and refuses an arm without a prebuilt ext-<arm>, so no
# compile runs inside the measurement host's window.  KERN_ALLOW_BUILD=1 overrides.
set -uo pipefail
OUT=${1:?out_root}; shift
ARMS=("$@")
H=experiments/t8r_speed/bench_t8r.sh
ROUTING=/mnt/shared/tessera-measurements/t8r-speed-20260929/prefill-routing-20260930
export BENCH_ARTIFACT=${BENCH_ARTIFACT:-/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported}
GROUPS_=${KERN_GROUPS:-experts.R1024.L10}
STEPS=" ${KERN_STEPS:-time ncu} "
step() {
  local name=$1; shift
  echo "== step $name start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "== step $name rc=$rc end=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  tail -4 "$OUT/$name.log"
}
armenv() {
  local f="$OUT/src-$1/env"; [[ -f "$f" ]] && grep -E '^[A-Z_][A-Z0-9_]*=' "$f" | tr '\n' ' '
  [[ -d "$OUT/ext-$1" ]] && echo "BENCH_EXT_DIR=$OUT/ext-$1"
  return 0
}
N=${#ARMS[@]}
for a in "${ARMS[@]}"; do
  if [[ ! -d "$OUT/ext-$a" && ${KERN_ALLOW_BUILD:-0} != 1 ]]; then
    echo "REFUSED: $OUT/ext-$a is missing; build it off the measurement host first (build_ext.sh)" >&2
    exit 2
  fi
done
if [[ $STEPS == *" time "* ]]; then
  for ((i = 0; i < N; i++)); do
    # shellcheck disable=SC2046
    step "t-${ARMS[i]}" env $(armenv "${ARMS[i]}") BENCH_SRC="$OUT/src-${ARMS[i]}/src" \
      bash $H . "$OUT/${ARMS[i]}-time" --groups "$GROUPS_" --ms ${KERN_MS:-1,512,2048} --routing "$ROUTING"
  done
  for ((i = N - 1; i >= 0; i--)); do
    # shellcheck disable=SC2046
    step "tb-${ARMS[i]}" env $(armenv "${ARMS[i]}") BENCH_SRC="$OUT/src-${ARMS[i]}/src" \
      bash $H . "$OUT/${ARMS[i]}-timeb" --groups "$GROUPS_" --ms ${KERN_MS:-1,512,2048} --routing "$ROUTING"
  done
fi
if [[ $STEPS == *" ncu "* ]]; then
  for ((i = 0; i < N; i++)); do
    # shellcheck disable=SC2046
    step "n-${ARMS[i]}" env $(armenv "${ARMS[i]}") BENCH_SRC="$OUT/src-${ARMS[i]}/src" BENCH_NCU=1 \
      BENCH_NCU_KERNELS=routed_fused_kernel bash $H . "$OUT/${ARMS[i]}-ncu" --groups "$GROUPS_" --ms ${KERN_NCU_MS:-512} \
      ${KERN_NCU_ROUTING:+--routing "$KERN_NCU_ROUTING"}
  done
fi
echo "ALL_DONE $(date -u +%FT%TZ)"
