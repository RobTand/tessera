#!/usr/bin/env bash
# The served leg of tessera#60: ldlq_block=4 against a RE-RUN of the default 32,
# on dense Qwen3-0.6B, at byte-identical bytes, scored against the same teacher
# the incumbent 0.531028 was scored against.
#
# ONE SESSION IS NOT AVAILABLE AND THIS SCRIPT DOES NOT PRETEND IT IS.  vLLM
# serves one model per engine, and every wrapper in this repo is
# start-serve-dump-reap, so two checkpoints are two containers.  What replaces
# the missing guarantee is a MATCHED-PAIR DRIFT CONTROL: the default arm is
# served FIRST and LAST, either side of the candidate, on one box, one image,
# one corpus, one teacher.  If the two default dumps agree, nothing between
# them drifted and the candidate's delta is the candidate's; if they disagree,
# their disagreement is the error bar on the delta and is reported as such.
# That is the same control the weight-space sweep this serve is testing used,
# and it is evidence rather than an assertion.
#
# Eager on purpose (TESSERA_KL_EAGER defaults to 1): the one known source of
# cross-container disagreement on this stack is inductor build nondeterminism
# in the compiled forward, and an eager serve does not have one.
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
RUNS=${RUNS:-/mnt/shared/tessera-runs/ldlq-block-serve}
KLDIR=/mnt/shared/tessera-kl
# The teacher the incumbent was read against, on the box it was dumped on.
# Overridable, because the bracket is not tied to one box: the two BF16
# teacher dumps on disk -- one from gx10-6b77, one from sparky, twelve hours
# apart -- agree at KL 0.000000 with 100% top-1 over all 4088 positions
# (experiments/results/kl_teacher_cross_box.json), so this teacher may be read
# from either box without a cross-box term entering the comparison.
TEACHER=${TESSERA_KL_TEACHER:-$KLDIR/qwen_rot_teacher_lina.json.npz}
# Optional second teacher.  Every arm is scored against it too, into a
# separate kl_<label>.x.json, so the box-invariance above is re-checked per
# ARM rather than only once at the teacher level.  Comparing is CPU work; a
# second reading of an already-dumped student costs no serve.
TEACHER_X=${TESSERA_KL_TEACHER_X:-}

export PYTHONPATH=$REPO/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
export TESSERA_KL_PORT=${TESSERA_KL_PORT:-8003}
export TESSERA_GPU_MEM_UTIL=${TESSERA_GPU_MEM_UTIL:-0.30}
export TESSERA_KL_NAME=${TESSERA_KL_NAME:-tessera-kl-ts60serve}
export TESSERA_KL_IMAGE=${TESSERA_KL_IMAGE:-vllm/vllm-openai:latest}
export TESSERA_KL_CORPUS=$KLDIR/corpus_qwen_n8_s512.json
export TESSERA_KL_LOGDIR=$RUNS
cd "$REPO" || exit 1

# arm-label -> twin directory.  b32 appears twice, deliberately, and both
# dumps are kept: the second is not a retry of the first, it is the control.
run () {   # dump-label, twin-name
  local label="$1" twin="$RUNS/$2-stock-twin"
  local out=$KLDIR/qwen_lb_$label.json
  [ -f "$twin/model.safetensors" ] || { echo "no export for $2"; return 1; }
  if [ ! -f "$out.npz" ]; then
    echo "== serving $label ($twin) $(date -u +%FT%TZ)"; uptime
    experiments/serve_and_dump_kl.sh "$twin" "$out" student || return 1
  fi
  $PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER" "$out.npz" \
      --out "$RUNS/kl_$label.json" 2>&1 | tee "$RUNS/kl_$label.log" | tail -8
  [ -n "$TEACHER_X" ] || return 0
  echo "-- cross-check teacher $(basename "$TEACHER_X")"
  $PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER_X" "$out.npz" \
      --out "$RUNS/kl_$label.x.json" 2>&1 | tee "$RUNS/kl_$label.x.log" | tail -4
}

# usage: ldlq_block_serve_ab.sh <candidate-arm> [more candidates...]
# Every candidate named goes inside ONE pair of default readings, so two
# candidates are compared against the same bracket rather than against two.
CANDS=("$@")
[ ${#CANDS[@]} -gt 0 ] || { echo "usage: $0 <candidate-arm> [more...]"; exit 64; }
SUF=$(IFS=_; echo "${CANDS[*]}")
run "b32a_$SUF" b32 || exit 1
for c in "${CANDS[@]}"; do run "$c" "$c" || exit 1; done
run "b32b_$SUF" b32 || exit 1
echo "SERVE_AB_DONE ${CANDS[*]} $(date -u +%FT%TZ)"
