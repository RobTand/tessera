#!/usr/bin/env bash
# Issue #5, item 5: serve a routed-MoE Tessera checkpoint, census its routes,
# and measure its KL against a BF16 teacher of the SAME model.
#
# THE MODEL IS A CUT, AND THAT IS STATED EVERYWHERE THE NUMBER IS.  The only
# routed-MoE model this box can serve is GLM-5.3-Flash-4layer, whose three
# stacks are 21.74 G routed parameters and ~3.75 GPU-hours of encode.  So the
# arms below run on the 16-expert cut written by ``moe_expert_cut.py``: same
# model class, same tokenizer, same real weights, expert dimension narrowed.
# The KL is student-against-teacher on that cut, so it measures the error the
# Tessera expert route introduces on these experts -- it is NOT a quality
# claim about GLM-5.3-Flash, whose routing the cut changes.
#
# THREE SERVES, SEQUENTIALLY, because the box has one serve lock and one GPU:
#   1. the BF16 cut  -> teacher logprobs
#   2. the Tessera cut -> student logprobs
#   3. the Tessera cut -> route census (a second load; the census drives two
#      forward shapes of its own and reads the route record from inside the
#      worker, which the OpenAI-server dump cannot)
# then the KL compare, which needs no GPU.
#
# usage: ts5_moe_served.sh <bf16-cut> <tessera-cut> [outdir]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TS="$(cd "$HERE/.." && pwd)"
BF16=${1:?bf16 cut}
WIRE=${2:?tessera cut}
OUT=${3:-/mnt/shared/tessera-runs/ts5/served}
export TMPDIR=${TMPDIR:-/home/rob/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/home/rob/.triton-cache}
export TS RUNS=${RUNS:-$OUT/run} EXT=${EXT:-$OUT/run/ext}
mkdir -p "$OUT" "$RUNS" "$EXT"
COMMIT=$(cd "$TS" && git rev-parse HEAD 2>/dev/null || echo unknown)
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python

# Mia's GLM build for every leg: it is the only runtime that registers
# Glm5Next, so "the pin" is not an option here and the image is stamped
# instead of assumed (serve_and_dump_kl.sh defaults to it already).
export IMG=${IMG:-prismaquant/glm53-mia-sm121:487ecf187}
export TESSERA_KL_IMAGE=$IMG
export TESSERA_KL_PORT=${TESSERA_KL_PORT:-8137}
export TESSERA_KL_LOGDIR=$OUT
export TESSERA_SERVE_MODE=${TESSERA_SERVE_MODE:-resident}
export TESSERA_GPU_MEM_UTIL=${TESSERA_GPU_MEM_UTIL:-0.35}

rc_teacher=skipped rc_student=skipped rc_census=skipped rc_compare=skipped

if [ ! -s "$OUT/teacher_bf16.json" ]; then
  echo "=== 1/4 teacher (BF16 cut)  $(date -Is)"
  TESSERA_KL_NAME=ts5-kl-teacher "$HERE/serve_and_dump_kl.sh" \
    "$BF16" "$OUT/teacher_bf16.json" teacher
  rc_teacher=$?
else
  echo "=== 1/4 teacher already dumped"; rc_teacher=0
fi

if [ "$rc_teacher" = 0 ] && [ ! -s "$OUT/student_tessera.json" ]; then
  echo "=== 2/4 student (Tessera cut)  $(date -Is)"
  TESSERA_KL_NAME=ts5-kl-student "$HERE/serve_and_dump_kl.sh" \
    "$WIRE" "$OUT/student_tessera.json" student "bf16-cut"
  rc_student=$?
elif [ -s "$OUT/student_tessera.json" ]; then
  echo "=== 2/4 student already dumped"; rc_student=0
fi

# The census is its own load: the route record lives on the layer objects
# inside the worker, and an OpenAI serve cannot hand them out.
echo "=== 3/4 route census  $(date -Is)"
"$HERE/tessera_plugin_run.sh" \
  -e TESSERA_SERVE_MODE="$TESSERA_SERVE_MODE" \
  -v /mnt/shared:/mnt/shared -- \
  "python3 tools/tessera_route_census.py '$WIRE' '$OUT/census.json' \
     --tessera-commit $COMMIT --gpu-memory-utilization 0.35 --max-model-len 1024" \
  2>&1 | tee "$OUT/census.log"
rc_census=${PIPESTATUS[0]}

if [ -s "$OUT/teacher_bf16.json" ] && [ -s "$OUT/student_tessera.json" ]; then
  echo "=== 4/4 KL  $(date -Is)"
  $PY /home/rob/dq-runs/kl_tool.py compare \
    --teacher "$OUT/teacher_bf16.json" --student "$OUT/student_tessera.json" \
    --out "$OUT/kl.json" 2>&1 | tee "$OUT/kl.log"
  rc_compare=${PIPESTATUS[0]}
fi

echo "=== ts5_moe_served: teacher=$rc_teacher student=$rc_student census=$rc_census compare=$rc_compare"
[ "$rc_teacher" = 0 ] && [ "$rc_student" = 0 ] && [ "$rc_compare" = 0 ] || exit 1
exit 0
