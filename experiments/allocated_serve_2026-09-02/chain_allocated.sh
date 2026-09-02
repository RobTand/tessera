#!/usr/bin/env bash
# The allocated-Tessera serving chain, on sparklina.
#
# Six arms: the PrismaQuant-allocated 4.0-bpp checkpoint and its matched-BYTES
# uniform control, each through the Tessera plugin resident/eager and
# resident/compiled; the allocated one also streamed (the residency control) and
# as its stock compressed-tensors twin on vanilla vLLM (the no-plugin control).
# Every arm takes the box's serve lock for itself.
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
mkdir -p "$RUNS" "$EXT" "$VLLM_CACHE"
COMMIT=${COMMIT:-unknown}
source "$TS/experiments/serve_lock.sh"

ALLOC=$OUT/qwen3-0.6b-alloc-4.0
UNIF=$OUT/qwen3-0.6b-uniform-R1006
TWIN=$OUT/qwen3-0.6b-alloc-4.0-stocktwin

census () {   # <checkpoint> <name> <mode> [--compiled]
  local ckpt=$1 name=$2 mode=$3 extra=${4:-}
  [ -f "$RUNS/census_$name.json" ] && { echo "== census $name: already done"; return 0; }
  echo "== census $name ($mode $extra)"
  SERVE_LOCK_OWNER="census $name" serve_lock_acquire
  "$TS/experiments/tessera_plugin_run.sh" \
    -e TESSERA_SERVE_MODE="$mode" -v /mnt/shared:/mnt/shared -v "$RUNS":"$RUNS" -- \
    "python3 /work/tools/tessera_route_census.py $ckpt $RUNS/census_$name.json \
       --expect-modules 112 --gpu-memory-utilization 0.30 --tessera-commit $COMMIT $extra" \
    > "$RUNS/census_$name.log" 2>&1
  echo "   rc=$? -> $RUNS/census_$name.json"
  serve_lock_release
}

kl () {       # <checkpoint> <arm> <mode> <eager 1|0>
  local ckpt=$1 arm=$2 mode=$3 eager=$4
  [ -f "$KLDIR/qwen_tessera_$arm.json.npz" ] && { echo "== kl $arm: already dumped"; return 0; }
  echo "== kl $arm (mode=$mode eager=$eager)"
  TESSERA_KL_NAME=alloc-serve-$arm TESSERA_LANE_EAGER=$eager \
    "$TS/experiments/tessera_plugin_served.sh" "$ckpt" "$arm" "$mode" \
    > "$RUNS/arm_$arm.log" 2>&1
  echo "   rc=$?"
  tail -14 "$RUNS/arm_$arm.log"
}

# The acceptance triple first, so a contended box still yields the result:
# the census is the evidence that 112 mixed-rung modules all dispatched, and
# the two eager/resident KLs are the allocated-vs-uniform comparison itself.
census "$ALLOC" alloc4-resident resident
kl "$ALLOC" alloc4-resident        resident 1
kl "$UNIF"  unif1006-resident      resident 1

# The layer-0-only separator: the same seven Linears the allocator priced,
# allocated vs byte-matched uniform, everything else BF16.  The whole-body arms
# change the allocation AND broadcast it to 28 depths; this pair changes only
# the allocation, so it says which of the two the served gap belongs to.
L0A=$OUT/qwen3-0.6b-l0-alloc
L0U=$OUT/qwen3-0.6b-l0-unif1006
[ -f "$L0A/tessera_serving_manifest.json" ] && kl "$L0A" l0-alloc     resident 1
[ -f "$L0U/tessera_serving_manifest.json" ] && kl "$L0U" l0-unif1006  resident 1

# everything after this point is corroboration
census "$UNIF"  unif1006-resident resident
census "$ALLOC" alloc4-streamed streamed
census "$ALLOC" alloc4-resident-graph resident --compiled
kl "$ALLOC" alloc4-resident-graph  resident 0
kl "$UNIF"  unif1006-resident-graph resident 0
kl "$ALLOC" alloc4-streamed        streamed 1

# the stock twin: the SAME wires materialised to compressed-tensors, on vanilla
# vLLM with no plugin at all.
if [ -d "$TWIN" ] && [ ! -f "$KLDIR/qwen_tessera_alloc4-twin.json.npz" ]; then
  echo "== kl alloc4-twin (stock twin, vanilla vLLM, no plugin)"
  TESSERA_KL_IMAGE=vllm/vllm-openai:latest TESSERA_KL_NAME=alloc-serve-twin \
    "$TS/experiments/serve_and_dump_kl.sh" "$TWIN" \
    "$KLDIR/qwen_tessera_alloc4-twin.json" student > "$RUNS/arm_alloc4-twin.log" 2>&1
  echo "   rc=$?"; tail -6 "$RUNS/arm_alloc4-twin.log"
  $PY /home/rob/dq-runs/kl_tool.py compare "$KLDIR/qwen_teacher_bf16_v028.json.npz" \
      "$KLDIR/qwen_tessera_alloc4-twin.json.npz" --out "$RUNS/kl_tessera_alloc4-twin.json" | tail -8
fi

echo "=== mutual comparisons"
for pair in "alloc4-resident alloc4-streamed" "alloc4-resident alloc4-twin" \
            "alloc4-resident alloc4-resident-graph" "alloc4-resident unif1006-resident"; do
  set -- $pair
  [ -f "$KLDIR/qwen_tessera_$1.json.npz" ] && [ -f "$KLDIR/qwen_tessera_$2.json.npz" ] || continue
  echo "--- $1 vs $2"
  $PY /home/rob/dq-runs/kl_tool.py compare "$KLDIR/qwen_tessera_$1.json.npz" \
      "$KLDIR/qwen_tessera_$2.json.npz" --out "$RUNS/mutual_$1__$2.json" | tail -6
done
echo "CHAIN DONE"
