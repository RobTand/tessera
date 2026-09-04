#!/usr/bin/env bash
# Issue #5, item 5: serve a routed-MoE Tessera checkpoint, census its routes,
# and measure its KL against a BF16 teacher of the SAME model.
#
# THE MODEL IS A CUT, AND THAT IS STATED WHEREVER THE NUMBER IS.  The only
# routed-MoE model this box can serve is GLM-5.3-Flash-4layer, whose three
# stacks are 21.74 G routed parameters and ~3.75 GPU-hours of encode.  So the
# arms below run on the 16-expert cut written by ``moe_expert_cut.py``: same
# model class, same tokenizer, same real weights, expert dimension narrowed.
# The KL is student-against-teacher on that cut, so it measures the error the
# Tessera expert route introduces on THESE experts.  It is NOT a quality claim
# about GLM-5.3-Flash, whose routing the cut changes.
#
# THREE SERVES, SEQUENTIALLY, because the box has one serve lock and the two
# arms of a KL must not be resident at once:
#   1. the BF16 cut through the stock serve  -> teacher logprobs
#   2. the Tessera cut through the PLUGIN serve -> student logprobs + KL
#   3. the Tessera cut again -> route census.  A second load on purpose: the
#      route record lives on the layer objects inside the worker and an
#      OpenAI-protocol serve cannot hand them out.
#
# The corpus contract is the GLM-tokenized default (``corpus_n8_s512.json``,
# built against GLM-5.3-Flash-4layer), and the cut copies that tokenizer byte
# for byte, so ``kl_tool``'s tokenizer-identity gate passes without
# --allow-tokenizer-mismatch.
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
export TS RUNS=${RUNS:-$OUT} EXT=${EXT:-$OUT/ext}
mkdir -p "$OUT" "$EXT"
COMMIT=$(cd "$TS" && git rev-parse HEAD 2>/dev/null || echo unknown)

# Mia's GLM build for every leg: it is the only runtime that registers
# Glm5Next, so "the pin" is not an option here and the image is stamped
# instead of assumed.
export IMAGE=${IMAGE:-prismaquant/glm53-mia-sm121:487ecf187}
export IMG=$IMAGE TESSERA_KL_IMAGE=$IMAGE
export TESSERA_KL_PORT=${TESSERA_KL_PORT:-8137}
export TESSERA_KL_LOGDIR=$OUT
export TESSERA_KL_CORPUS=${TESSERA_KL_CORPUS:-/mnt/shared/tessera-kl/corpus_n8_s512.json}
# 0.15 of 121 GB is ~18 GB, against a 7.07 GiB teacher and four decoder
# layers of KV at 4096 x 8.  The number has to be small AND has to match
# what the pool granted: vLLM compares its fraction of the box's TOTAL
# memory against what is FREE, and a serve that asks for more than the
# box has left does not run slowly, it refuses to start ("Free memory ...
# is less than desired GPU memory utilization").  Declare mem_gb=20 on
# the pool action that runs this.
export TESSERA_GPU_MEM_UTIL=${TESSERA_GPU_MEM_UTIL:-0.15}
MODE=${TESSERA_SERVE_MODE:-resident}
# THE CENSUS NEEDS MORE THAN THE SERVES DO, and for a reason that is not
# about the model: it drives ``LLM(...)`` with vLLM's default chunked-prefill
# budget of 8192 batched tokens, so its profiling peak is several times the
# serve's at --max-num-seqs 8.  On the first run 0.15 (18.24 GiB) left
# "Available KV cache memory: -2.02 GiB" AFTER the model had loaded, which
# reads like a Tessera failure and is not one.  One knob per consumer.
CENSUS_MEM_UTIL=${TESSERA_CENSUS_MEM_UTIL:-0.35}

rc_teacher=skipped rc_student=skipped rc_census=skipped

if [ -s "$OUT/teacher_bf16.json.npz" ]; then
  echo "=== 1/3 teacher already dumped"; rc_teacher=0
else
  echo "=== 1/3 teacher (BF16 cut)  $(date -Is)"
  TESSERA_KL_NAME=ts5-kl-teacher "$HERE/serve_and_dump_kl.sh" \
    "$BF16" "$OUT/teacher_bf16.json" teacher
  rc_teacher=$?
fi

if [ "$rc_teacher" = 0 ]; then
  echo "=== 2/3 student (Tessera cut) + KL  $(date -Is)"
  TESSERA_KL_NAME=ts5-plugin-student \
  TESSERA_KL_TEACHER="$OUT/teacher_bf16.json" \
  TESSERA_KL_DUMP="$OUT/student_tessera.json" \
  TESSERA_KL_LOG="$OUT/serve_student_tessera.log" \
    "$HERE/tessera_plugin_served.sh" "$WIRE" ts5moe "$MODE"
  rc_student=$?
fi

# The census runs whatever the KL did: a route receipt is worth having even if
# the comparison could not be made, and a census that refuses says why.
echo "=== 3/3 route census  $(date -Is)"
"$HERE/tessera_plugin_run.sh" \
  -e TESSERA_SERVE_MODE="$MODE" \
  -v /mnt/shared:/mnt/shared -- \
  "python3 tools/tessera_route_census.py '$WIRE' '$OUT/census.json' \
     --tessera-commit $COMMIT --gpu-memory-utilization ${CENSUS_MEM_UTIL} \
     --max-model-len 1024" 2>&1 | tee "$OUT/census.log"
rc_census=${PIPESTATUS[0]}

echo "=== ts5_moe_served: teacher=$rc_teacher student=$rc_student census=$rc_census"
echo "    teacher  $OUT/teacher_bf16.json"
echo "    student  $OUT/student_tessera.json"
echo "    KL       $OUT/kl_tessera_ts5moe.json"
echo "    census   $OUT/census.json"
[ "$rc_teacher" = 0 ] && [ "$rc_student" = 0 ] || exit 1
exit 0
