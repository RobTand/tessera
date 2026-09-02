#!/usr/bin/env bash
# Third wave.
#
# Two mechanism arms that take the separator apart.  The separator says the
# allocation loses 1.93x on the seven units it priced; the per-role Delta-loss
# table says it bought that trade on gate/up/v and paid for it by cutting
# down_proj from R1006 to R749.  These two arms move only down_proj:
#
#   l0-down749      the uniform arm with down_proj alone cut to R749
#   l0-allocnodown  the allocation with down_proj alone restored to R1006
#
# Bytes are deliberately not matched -- these are mechanism arms, not controls.
#
# And the layer-0 allocation's stock twin, which is the only checkpoint here
# whose BF16 passthrough rides vanilla vLLM's own `ignore` list.  Whether a
# FUSED name is accepted there is a claim about another runtime; it gets served,
# not asserted.
set -uo pipefail
TS=/home/rob/tmp/wt-allocated
RUNS=/home/rob/tessera-runs/allocated
OUT=/mnt/shared/tessera-runs/allocated
KLDIR=/mnt/shared/tessera-kl
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export TS RUNS
export EXT=$RUNS/ext VLLM_CACHE=$RUNS/vllm-cache
export TESSERA_GPU_MEM_UTIL=0.30
export PORT=8004 TESSERA_KL_PORT=8004
export TESSERA_KL_CORPUS=$KLDIR/corpus_qwen_n8_s512.json
mkdir -p "$RUNS" "$EXT"

kl () {       # <checkpoint> <arm>
  local ckpt=$1 arm=$2
  [ -f "$ckpt/tessera_serving_manifest.json" ] || { echo "== kl $arm: no export"; return 0; }
  [ -f "$KLDIR/qwen_tessera_$arm.json.npz" ] && { echo "== kl $arm: already dumped"; return 0; }
  echo "== kl $arm"
  TESSERA_KL_NAME=alloc-serve-$arm TESSERA_LANE_EAGER=1 \
    "$TS/experiments/tessera_plugin_served.sh" "$ckpt" "$arm" resident \
    > "$RUNS/arm_$arm.log" 2>&1
  echo "   rc=$?"
  sed -n '/--- KL vs teacher ---/,$p' "$RUNS/arm_$arm.log" | head -8
}

kl "$OUT/qwen3-0.6b-l0-down749"      l0-down749
kl "$OUT/qwen3-0.6b-l0-allocnodown"  l0-allocnodown

TWIN=$OUT/qwen3-0.6b-l0-alloc-twin
if [ -d "$TWIN" ] && [ ! -f "$KLDIR/qwen_tessera_l0-alloc-twin.json.npz" ]; then
  echo "== kl l0-alloc-twin (stock twin, vanilla vLLM, no plugin)"
  TESSERA_KL_IMAGE=vllm/vllm-openai:latest TESSERA_KL_NAME=alloc-serve-l0twin \
    "$TS/experiments/serve_and_dump_kl.sh" "$TWIN" \
    "$KLDIR/qwen_tessera_l0-alloc-twin.json" student > "$RUNS/arm_l0-alloc-twin.log" 2>&1
  echo "   rc=$?"; tail -8 "$RUNS/arm_l0-alloc-twin.log"
  if [ -f "$KLDIR/qwen_tessera_l0-alloc-twin.json.npz" ]; then
    $PY /home/rob/dq-runs/kl_tool.py compare "$KLDIR/qwen_teacher_bf16_v028.json.npz" \
        "$KLDIR/qwen_tessera_l0-alloc-twin.json.npz" \
        --out "$RUNS/kl_tessera_l0-alloc-twin.json" | tail -8
  fi
fi
echo "PROBE CHAIN DONE"
