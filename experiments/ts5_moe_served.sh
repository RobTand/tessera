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
# CONTAINER SCRATCH IS LOCAL, ALWAYS.  $EXT is bind-mounted as /ext and carries
# TORCH_EXTENSIONS_DIR, TMPDIR and (in tessera_plugin_run.sh) TRITON_CACHE_DIR,
# so it is written by the container's ROOT -- which /mnt/shared squashes.  A
# default of $OUT/ext puts it on NFS and the census dies at model inspection
# with "PermissionError: '/ext/triton'", which surfaces as vLLM failing to
# inspect Glm5NextForConditionalGeneration and looks like a model problem.
# (The first census run survived because its pbrun command line exported
# EXT=/home/rob/tmp/ts5-ext by hand and the re-run's wrapper script did not --
# the difference was between two submissions of mine, not inside this file.
# The default below is what removes that difference: a caller who forgets is
# now correct, and a caller who wants a different scratch dir must name a
# LOCAL one.)
export TS RUNS=${RUNS:-$OUT} EXT=${EXT:-/home/rob/tmp/ts5-ext}
mkdir -p "$OUT" "$EXT"
# EVERYTHING THIS SCRIPT SAYS GOES TO A FILE AS IT SAYS IT.  Under the
# PrismaBuild pool the client buffers an action's stdout and publishes it
# only when the action ends, so an arm that fails inside a long action is
# invisible until the action ends -- and an action the pool RETRIES never
# ends, which on 2026-09-04 hid one dump failure through three attempts.
# A serve's own container log is not a substitute: the failing message is
# the CLIENT's.
exec > >(tee -a "$OUT/driver.log") 2>&1
echo "=== ts5_moe_served $(date -Is)  bf16=$BF16  wire=$WIRE"
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
  # THE FOURTH ARGUMENT IS NOT OPTIONAL FOR A TEACHER.  kl_tool refuses a
  # teacher dump with no --teacher-label in about two seconds, before it makes
  # a single request -- the label travels into every compare line as
  # "KL-vs-<label>", and it would rather refuse than let the reference identity
  # be supplied by convention later.  Omitting it cost three pool attempts and
  # about ten GPU-minutes on 2026-09-04, each of which stood a serve up for
  # three minutes to be refused by a client-side check.
  TESSERA_KL_NAME=ts5-kl-teacher "$HERE/serve_and_dump_kl.sh" \
    "$BF16" "$OUT/teacher_bf16.json" teacher BF16
  rc_teacher=$?
fi

if [ -s "$OUT/student_tessera.json.npz" ]; then
  echo "=== 2/3 student already dumped"; rc_student=0
elif [ "$rc_teacher" = 0 ]; then
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
# THE CENSUS WRITES TO LOCAL DISK, NOT TO /mnt/shared.  /mnt/shared is NFSv4
# with sec=sys and the export squashes root; the census runs as root inside the
# container, so its receipt lands as `nobody` against a `drwxrwxr-x rob:rob`
# directory and the open() is refused -- AFTER the model has loaded, generated
# twice and the whole census has been gathered, which is the most expensive
# place to discover a permission.  (Verified: container root gets
# "Permission denied" on $OUT and "OK" on a bind mount of /home/rob/tmp.)  The
# wrapper's own default RUNS is local for exactly this reason; this driver is
# the caller that pointed it at NFS.  Write local, copy on the host.
CENSUS_LOCAL=${TESSERA_CENSUS_LOCAL:-/home/rob/tmp/ts5-census}
mkdir -p "$CENSUS_LOCAL"
"$HERE/tessera_plugin_run.sh" \
  -e TESSERA_SERVE_MODE="$MODE" \
  -v /mnt/shared:/mnt/shared -v "$CENSUS_LOCAL":/census -- \
  "python3 tools/tessera_route_census.py '$WIRE' /census/census.json \
     --tessera-commit $COMMIT --gpu-memory-utilization ${CENSUS_MEM_UTIL} \
     --max-model-len 1024" 2>&1 | tee -a "$OUT/census.log"
rc_census=${PIPESTATUS[0]}
[ -s "$CENSUS_LOCAL/census.json" ] && cp "$CENSUS_LOCAL/census.json" "$OUT/census.json"

echo "=== ts5_moe_served: teacher=$rc_teacher student=$rc_student census=$rc_census"
echo "    teacher  $OUT/teacher_bf16.json"
echo "    student  $OUT/student_tessera.json"
echo "    KL       $OUT/kl_tessera_ts5moe.json"
echo "    census   $OUT/census.json"
[ "$rc_teacher" = 0 ] && [ "$rc_student" = 0 ] || exit 1
exit 0
