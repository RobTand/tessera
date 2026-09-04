#!/usr/bin/env bash
# Is the eager-vs-compiled KL gap the dispatch switch, or something else?  (#16)
#
# THE CLAIM UNDER TEST.  vLLM 0.28 does not run one program two ways.  When it
# compiles, two config defaults flip together:
#
#   vllm/config/vllm.py:1392-1399   custom_ops base mode  "all" -> "none"
#   vllm/platforms/cuda.py:690-700  ir_op_priority        ["vllm_c","native"] -> ["native"]
#
# so the compiled arm runs ``forward_native``/``native`` -- torch decompositions
# that inductor then fuses -- where the eager arm ran the CUDA kernels.  Both
# arms of the 2026-09-02 divergence recorded exactly that in their own startup
# logs.  On a W4A4 route the resulting bf16-ulp difference is re-drawn by an FP4
# quantizer whose codes are ~40% apart, which is the amplification the gap's
# size needs.
#
# THE LADDER.  Seven arms of one checkpoint -- vanilla vLLM, no plugin, so
# nothing Tessera does is in the loop -- on one box, one image, one corpus, one
# compile-cache root:
#
#   eager              --enforce-eager                       (the control)
#   compiled           vLLM's default compiled forward       (reproduces the gap)
#   compiled-ir        compiled, rms_norm/fused_add_rms_norm pinned to vllm_c
#   compiled-ops       compiled, custom_ops pinned to "all"
#   compiled-both      compiled, both pinned
#   compiled-both-noauto  both pinned, and FlashInfer autotune held off
#   compiled-eagerbackend compiled machinery, backend="eager", autotune off
#
# The last two exist because --enforce-eager flips a THIRD switch that the first
# five leave moving.  O0 sets ``enable_flashinfer_autotune: False``
# (config/vllm.py:241) and O1/O2/O3 set it True (:264,:287,:310), so every
# compiled arm autotunes FlashInfer at warmup where the eager arm does not, and
# a different mainloop is a different accumulation.  ``compiled-both-noauto``
# holds it still.  ``compiled-eagerbackend`` is the cleanest control of the
# ladder: ``using_inductor`` is ``backend == "inductor" and mode != NONE``
# (config/vllm.py:1392-1399, platforms/cuda.py:690-700), so backend="eager"
# resolves BOTH dispatch defaults to their eager values on its own, with dynamo
# and the cudagraphs still in the loop.  It also holds autotune off, so that it
# differs from ``eager`` in the machinery ALONE; without that it would differ in
# two things and answer neither.  The ladder then decomposes:
#   eager -> eagerbackend         dynamo and the cudagraphs
#   eagerbackend -> both-noauto   inductor's codegen on the glue
#   both-noauto -> both           the FlashInfer autotune
#   both -> compiled              the dispatch switch itself
#
# Every compiled arm holds the fusion passes off, because enabling either
# dispatch pin flips ``enable_norm_fusion``/``enable_act_fusion`` on
# (config/vllm.py:123-146) and that would be a second change in one arm.  The
# 2026-09-02 compiled arm ran with all three off, so off is also what matches it.
#
# PREDICTIONS, WRITTEN BEFORE THE RUN (a prediction recorded after is a
# postdiction):
#   eager_2026-09-02 vs eager   =  0.000000  (this box reproduces that one)
#   eager vs compiled          ~= 0.2473 +- the day's rebuild floor.  This arm
#                                 builds into a fresh cache root, so it does not
#                                 REPRODUCE 0.244481/0.2473, it lands near it;
#                                 the floor is measured below, not assumed.
#   eager vs compiled-eagerbackend ~ 0.00x   dynamo and cudagraphs alone change
#                                            nothing; if not, there is a
#                                            mechanism I have not named
#   eager vs compiled-both     ~  0.00x   if dispatch is the mechanism
#                              ~  0.02    if dispatch plus the known build-to-build
#                                         autotune floor (0.017117, measured)
#                              >> 0.05    a second mechanism exists -- look at
#                                         compiled-both-noauto next, and do not
#                                         claim closure
#   compiled-ir, compiled-ops between the two, splitting the gap by op family
#
# Each compiled arm has its own config, so each keys its own cache slot and each
# carries its own autotune draw; the floor above is why an exact zero is not
# expected even if the mechanism is fully named.  The root is shared and never
# emptied mid-run: a replayed build is bit-identical, a rebuilt one is not.
set -euo pipefail

MODEL=${MODEL:-/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4}
R=${R:-/home/rob/tessera-runs/compile-dispatch}
KLDIR=/mnt/shared/tessera-kl
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$R" "$R/vllm-cache"
# The pinned image, not a tag.  `latest` breaks the #100 gate silently on
# the day upstream repoints it, and `arm()` swallows a failure and still
# prints DISPATCH_LADDER_DONE -- so a repointed tag would read as a clean
# run of seven arms that never happened.
source "$HERE/runtime_image.sh"
export TESSERA_KL_IMAGE=${TESSERA_KL_IMAGE:-$(runtime_image_pin)}
export TESSERA_KL_CORPUS=$KLDIR/corpus_qwen_n8_s512.json
export TESSERA_KL_LOGDIR=$R
export TESSERA_KL_VLLM_CACHE=$R/vllm-cache
# Every arm died at engine init asking for 0.85 of a 121 GiB device on a
# box running three concurrent GPU jobs.  Qwen3-0.6B at max-model-len
# 4096 does not need more than this, and every arm uses the same value,
# so nothing about the comparison moves.  It lived in a shell export
# until now, which is to say it was not in the experiment at all.
export TESSERA_GPU_MEM_UTIL=${TESSERA_GPU_MEM_UTIL:-0.15}
export TESSERA_KL_NAME=${TESSERA_KL_NAME:-tessera-dispatch-serve}
export TMPDIR=${TMPDIR:-/home/rob/tmp}

docker image inspect "$TESSERA_KL_IMAGE" --format 'image {{.Id}} created {{.Created}}' | tee "$R/image.txt"

NOFUSE='"pass_config":{"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false}'
IRPIN='--kernel-config {"ir_op_priority":{"rms_norm":["vllm_c"],"fused_add_rms_norm":["vllm_c"]}}'
IRPIN_NOAUTO='--kernel-config {"ir_op_priority":{"rms_norm":["vllm_c"],"fused_add_rms_norm":["vllm_c"]},"enable_flashinfer_autotune":false}'

arm() {  # arm <name> <eager 0|1> <require-ere> [extra vllm args...]
  local name=$1 eager=$2 require=$3; shift 3
  local dump=$KLDIR/qwen_dispatch_$name.json
  if [ -f "$dump.npz" ]; then echo "=== $name: dump exists, keeping it ==="; return 0; fi
  echo "=================== $name ==================="
  # A failed arm does not take the ladder down with it: the arms are
  # independent measurements, and losing the four that would have run because
  # the third mistyped a flag is how a GPU hour becomes nothing.
  if ! TESSERA_KL_EAGER=$eager \
       TESSERA_KL_REQUIRE_IN_LOG="$require" \
       TESSERA_KL_VLLM_EXTRA="$*" \
       "$HERE/serve_and_dump_kl.sh" "$MODEL" "$dump" student; then
    echo "ARM FAILED (continuing): $name"
    return 0
  fi
}

arm eager 1 "'custom_ops': \['all'\]"
arm compiled 0 "enforce_eager=False.*'custom_ops': \['none'\]"
arm compiled-ir 0 "enforce_eager=False.*rms_norm=\['vllm_c'" \
  "$IRPIN" "--compilation-config" "{$NOFUSE}"
arm compiled-ops 0 "enforce_eager=False.*'custom_ops': \['all'\]" \
  "--compilation-config" "{\"custom_ops\":[\"all\"],$NOFUSE}"
arm compiled-both 0 "enforce_eager=False.*'custom_ops': \['all'\].*rms_norm=\['vllm_c'" \
  "$IRPIN" "--compilation-config" "{\"custom_ops\":[\"all\"],$NOFUSE}"
arm compiled-both-noauto 0 "enforce_eager=False.*'custom_ops': \['all'\].*rms_norm=\['vllm_c'" \
  "$IRPIN_NOAUTO" "--compilation-config" "{\"custom_ops\":[\"all\"],$NOFUSE}"
arm compiled-eagerbackend 0 "enforce_eager=False.*'backend': 'eager'" \
  "--kernel-config" "{\"enable_flashinfer_autotune\":false}" \
  "--compilation-config" "{\"backend\":\"eager\",$NOFUSE}"

echo "=================== compare ==================="
compare() {  # compare <a> <b>
  local a=$1 b=$2
  if [ ! -f "$KLDIR/qwen_dispatch_$a.json.npz" ] || [ ! -f "$KLDIR/qwen_dispatch_$b.json.npz" ]; then
    echo "--- $a vs $b: SKIPPED, an arm did not produce a dump ---"; return 0
  fi
  $PY /home/rob/dq-runs/kl_tool.py compare \
    "$KLDIR/qwen_dispatch_$a.json.npz" "$KLDIR/qwen_dispatch_$b.json.npz" \
    --teacher-label-override "dispatch_$a" \
    --out "$R/kl_${a}__vs__${b}.json" | tail -20
}
compare eager compiled
compare eager compiled-ir
compare eager compiled-ops
compare eager compiled-both
compare eager compiled-both-noauto
compare eager compiled-eagerbackend
compare compiled compiled-both
compare compiled-both compiled-eagerbackend
# The historical arms of the receipt, same checkpoint, on the same corpus: this
# is the check that today's box reproduces 2026-09-02 rather than a new number.
# Guarded like its sibling below.  Under `set -euo pipefail` a missing dump
# here killed the ladder before all three diagnostic loops -- which is exactly
# how the 21:09 run ended, with five arms served and nothing compared.
if [ -f "$KLDIR/qwen_stock_tessera-k2.json.npz" ] && \
   [ -f "$KLDIR/qwen_dispatch_eager.json.npz" ]; then
  $PY /home/rob/dq-runs/kl_tool.py compare \
    "$KLDIR/qwen_stock_tessera-k2.json.npz" "$KLDIR/qwen_dispatch_eager.json.npz" \
    --teacher-label-override "stock_eager_2026-09-02" \
    --out "$R/kl_historical-eager__vs__eager.json" | tail -20
else
  echo "--- historical-eager vs eager: SKIPPED, a dump is absent ---"
fi
# Same configuration, different build, six days apart: this is TODAY's
# rebuild-to-rebuild floor, and it is the yardstick the "~0.02" predictions
# above are read against.  Without it "eager vs compiled-both = 0.02" has no
# scale.
if [ -f "$KLDIR/qwen_dispatch_compiled.json.npz" ]; then
  $PY /home/rob/dq-runs/kl_tool.py compare \
    "$KLDIR/qwen_stock_tessera-k2-graph.json.npz" "$KLDIR/qwen_dispatch_compiled.json.npz" \
    --teacher-label-override "stock_compiled_2026-09-02" \
    --out "$R/kl_historical-compiled__vs__compiled.json" | tail -20
fi

echo "--- resolved dispatch, read off each arm's own log ---"
for a in eager compiled compiled-ir compiled-ops compiled-both compiled-both-noauto compiled-eagerbackend; do
  [ -f "$R/serve_qwen_dispatch_$a.log" ] || continue
  printf '%-14s ' "$a"
  grep -o "'custom_ops': \[[^]]*\]" "$R/serve_qwen_dispatch_$a.log" | head -1 | tr -d '\n'
  printf '  '
  grep -o "ir_op_priority=IrOpPriorityConfig([^)]*)" "$R/serve_qwen_dispatch_$a.log" | head -1
done

# Which GEMM each arm actually rode.  The checkpoint is compressed-tensors
# NVFP4, and vLLM picks that kernel at load; if the arms disagree here, the KL
# gap has a second author and the dispatch table above is not the whole story.
# Pinning the dispatch turns the fusion defaults back on -- enable_norm_fusion
# is true as soon as ir_op_priority.rms_norm[0] != "native", and
# enable_act_fusion as soon as silu_and_mul is an enabled custom op
# (config/vllm.py:123-146) -- so every pinned arm sets pass_config explicitly to
# hold them off.  Explicit values survive: only an unset (None) field is
# defaulted (config/compilation.py:228-247).  Read back what actually resolved.
echo "--- resolved fusion passes, read off each arm's own log ---"
for a in eager compiled compiled-ir compiled-ops compiled-both compiled-both-noauto compiled-eagerbackend; do
  [ -f "$R/serve_qwen_dispatch_$a.log" ] || continue
  printf '%-22s ' "$a"
  grep -o "'fuse_norm_quant': [A-Za-z]*\|'fuse_act_quant': [A-Za-z]*\|'fuse_attn_quant': [A-Za-z]*" \
    "$R/serve_qwen_dispatch_$a.log" | head -3 | tr '\n' ' '
  echo
done

echo "--- selected kernels, read off each arm's own log ---"
for a in eager compiled compiled-ir compiled-ops compiled-both compiled-both-noauto compiled-eagerbackend; do
  [ -f "$R/serve_qwen_dispatch_$a.log" ] || continue
  echo "  [$a]"
  grep -Eio "using [a-z0-9_ .-]* (for|as) [a-z0-9_ .-]*|selected [a-z0-9_ .-]*kernel[a-z0-9_ .-]*|flashinfer autotune[a-z .]*" \
    "$R/serve_qwen_dispatch_$a.log" | sort -u | sed 's/^/    /' | head -12
done
echo DISPATCH_LADDER_DONE
