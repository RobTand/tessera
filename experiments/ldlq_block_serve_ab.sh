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
TEACHER=$KLDIR/qwen_rot_teacher_lina.json.npz

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
}

run b32a b32 || exit 1
run b4   b4   || exit 1
run b32b b32 || exit 1
echo "SERVE_AB_DONE $(date -u +%FT%TZ)"
