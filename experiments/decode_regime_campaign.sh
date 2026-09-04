#!/usr/bin/env bash
# The #102 campaign: one decode-regime teacher and two arms on one checkpoint.
#
# Three serves, sequential, each taking the box's serve lock in turn:
#
#   1. the BF16 teacher, RE-DUMPED in the decode regime.  A decode-regime
#      student against the prefill teacher would be a comparison against a
#      differently-produced reference -- the same weights, but its positions
#      came out of 512-row forwards.  There is no way to fix that in the
#      compare, which is why the tool refuses it.
#   2. arm A: the streamed FP8 route with its window-GEMV lane available.
#   3. arm B: the same bytes (a hardlink, one inode) with a READ-ONLY
#      extensions root, so the lane cannot build and the route takes its
#      published torch fallback.
#
# Each arm dumps BOTH regimes off its own serve, so the receipt carries the
# matched pair: the prefill regime is where #83 found its bit-identical null,
# and the decode regime is where the kernel actually runs.
#
# Skip-if-exists per stage, the repo's convention: a lost arm costs its own
# serve and not the ones that landed.  Delete the payload to force a re-run.
#
# usage: decode_regime_campaign.sh [teacher|arms|all]
set -euo pipefail
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
RUNS=${RUNS:-/home/rob/tessera-runs/ts102}
ARMS=${ARMS:-/home/rob/tessera-runs/ts83}
KLDIR=/mnt/shared/tessera-kl
STAGE=${1:-all}
# The campaign tag that names this run's arms and dumps.  Default = #102's own,
# so nothing here changes.  The compiled re-take #113 asks for, over the same
# two hardlinked arms, is
#   ARMTAG=compiled TESSERA_LANE_EAGER=0 TESSERA_KL_DUMP_PREFIX=qwen_ts113 \
#   RUNS=/home/rob/tessera-runs/ts113 decode_regime_campaign.sh arms
ARMTAG=${ARMTAG:-ts102}
export TESSERA_KL_DUMP_PREFIX=${TESSERA_KL_DUMP_PREFIX:-qwen_ts102}
DP=$TESSERA_KL_DUMP_PREFIX
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
# The campaign's own name.  #102 is the default so its receipt reproduces; a
# re-run over the same arms after a lane change sets both and writes beside
# the evidence rather than over it (TAG=ts110 PREFIX=qwen_ts110).
# Two names for one value: #83/#113 drive this with ARMTAG, #102/#110 with
# TESSERA_KL_ARM_TAG.  Either works and they cannot disagree -- picking one
# side of the merge would have silently dropped the other issue's invocation.
TAG=${TESSERA_KL_ARM_TAG:-$ARMTAG}
ARMTAG=$TAG
PREFIX=${TESSERA_KL_DUMP_PREFIX:-qwen_ts102}
DP=$PREFIX
export TESSERA_KL_DUMP_PREFIX=$PREFIX
export TS=$WT RUNS
export TESSERA_KL_CORPUS=${TESSERA_KL_CORPUS:-$KLDIR/corpus_qwen_n8_s512.json}
export TESSERA_KL_IMAGE=${TESSERA_KL_IMAGE:-vllm/vllm-openai:latest}
export IMAGE=${IMAGE:-$TESSERA_KL_IMAGE}
export TESSERA_KL_LOGDIR=$RUNS
export TESSERA_GPU_MEM_UTIL=${TESSERA_GPU_MEM_UTIL:-0.45}
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
TEACHER=$KLDIR/qwen_teacher_bf16_v028_decode.json
mkdir -p "$RUNS"
cd "$WT"

if [ "$STAGE" = teacher ] || [ "$STAGE" = all ]; then
  if [ -f "$TEACHER.npz" ]; then
    echo "=== decode teacher already at $TEACHER.npz"
  else
    echo "=== decode-regime teacher  $(date -Is)"
    TESSERA_KL_REGIME=decode TESSERA_KL_NAME=tessera-ts102-teacher \
      experiments/serve_and_dump_kl.sh /home/rob/models/Qwen3-0.6B \
      "$TEACHER" teacher BF16 2>&1 | tee "$RUNS/teacher-decode.log"
  fi
fi

if [ "$STAGE" = arms ] || [ "$STAGE" = all ]; then
  # The arms must be one inode when the run STARTS, not just when they were
  # made: a re-export between the two serves is the confound this guards.
  INO_A=$(stat -c %i "$ARMS/armA/model.safetensors")
  INO_B=$(stat -c %i "$ARMS/armB/model.safetensors")
  [ "$INO_A" = "$INO_B" ] || { echo "armA and armB are not one inode"; exit 2; }
  echo "arms share inode $INO_A ($(stat -c %s "$ARMS/armA/model.safetensors") bytes)"
  EXT_B_RO=$ARMS/ext-B-readonly
  mkdir -p "$EXT_B_RO"; chmod a-w "$EXT_B_RO"
  for ARM in armA armB; do
    if [ -f "$KLDIR/${PREFIX}_${TAG}-${ARM}_decode.json.npz" ]; then
      echo "=== $ARM already dumped"; continue
    fi
    EXTRA=""
    [ "$ARM" = armB ] && EXTRA="-v $EXT_B_RO:/ext-ro:ro -e TORCH_EXTENSIONS_DIR=/ext-ro"
    echo "=== $ARM  $(date -Is)"
    EXT=$ARMS/ext-A TESSERA_LANE_DOCKER_EXTRA="$EXTRA" \
      experiments/decode_regime_kl.sh "$ARMS/$ARM" "$TAG-$ARM" streamed \
      2>&1 | tee "$RUNS/$ARM.log"
  done

  echo "=== mutual KL between the two arms, per regime  $(date -Is)"
  for REG in decode prefill; do
    A=$KLDIR/${PREFIX}_${TAG}-armA_$REG.json.npz
    B=$KLDIR/${PREFIX}_${TAG}-armB_$REG.json.npz
    [ -f "$A" ] && [ -f "$B" ] || { echo "  $REG: missing an arm"; continue; }
    echo "--- $REG: armB against armA, byte-identical bytes ---"
    $PY /home/rob/dq-runs/kl_tool.py compare "$A" "$B" \
      --teacher-label-override "$TAG-armA-GEMV-$REG" \
      --out "$RUNS/mutual_$REG.json" | tail -6
  done
fi
echo "=== campaign done $(date -Is)"
