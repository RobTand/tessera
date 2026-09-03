#!/usr/bin/env bash
# The window GEMV, SERVED: the census, the two-arm KL and the latency that #10
# wired but never measured (issue #83).
#
# #10 put ``fp8_gemv.streamed_apply`` in the streamed FP8 route's forward and
# proved it bit-exact against the torch decoder at load.  Everything it
# measured was in-process on this box -- kernel against ``_scaled_mm`` at M=1,
# a read-bandwidth number, an agreement bound.  None of that is a serving
# result.  This script is the serving result.
#
# THE TWO ARMS ARE ONE CHECKPOINT.  ``armA`` and ``armB`` under $RUNS are
# ``cp -al`` hardlinks of the same export: every file is the same inode, so the
# bytes are identical by construction rather than by a hash someone remembered
# to take (the hashes are taken anyway, into arm-hashes.txt).  The arms differ
# in exactly one process fact: whether the window-GEMV extension can be built.
#
# HOW ARM B IS MADE.  ``TORCH_EXTENSIONS_DIR`` points at a READ-ONLY mount, so
# ``kernel_window_gemv._ext()``'s ``os.makedirs(build, exist_ok=True)`` raises
# before nvcc is ever consulted.  That is the smallest possible perturbation:
# CUDA_HOME is untouched, so nothing else in the container behaves differently,
# and the route takes exactly the published fallback
# (``ext.NATIVE_EXTENSIONS``'s ``when_unavailable`` -> ``torch_window`` in both
# residencies).  ``prepare_fp8_gemv`` raising is the SAME state a rate-3 wire
# or a shard start state reaches, so arm B is the route's real fallback and not
# a test double.
#
# WHY THE ARMS GET SEPARATE COMPILE CACHES (issue #91).  The two streamed arms
# trace structurally different graphs -- arm A a single
# ``tessera::fp8_streamed_apply`` node, arm B a window decode plus
# ``torch._scaled_mm`` -- and ``config.py`` declares only ``serve_mode`` into
# vLLM's compile-cache key, so the two would share one slot.  A separate
# VLLM_CACHE per arm is a MEASUREMENT WORKAROUND, not a fix; #91 owns the fix.
#
# "SAME vLLM SESSION" IS IMPOSSIBLE HERE and the receipt says so: extension
# residency is a process fact, so the two arms are necessarily two processes.
# What is held equal is everything else -- one image, one teacher npz, one
# corpus contract, one box, back to back.
#
# usage: window_gemv_served.sh [census|kl|all]
set -euo pipefail

WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
RUNS=${RUNS:-/home/rob/tessera-runs/ts83}
KLDIR=/mnt/shared/tessera-kl
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
TEACHER=$KLDIR/qwen_teacher_bf16_v028.json
STAGE=${1:-all}

# THE TREE UNDER TEST IS THIS WORKTREE.  Both plugin wrappers default TS to
# /home/rob/tessera, the shared checkout another agent is editing for #47; a
# serve that installed THAT tree would measure someone else's working copy.
export TS=$WT
export RUNS
export TESSERA_KL_CORPUS=${TESSERA_KL_CORPUS:-$KLDIR/corpus_qwen_n8_s512.json}
# Image-matched teacher: the npz above was dumped on vllm/vllm-openai:latest.
export TESSERA_KL_IMAGE=${TESSERA_KL_IMAGE:-vllm/vllm-openai:latest}
export TESSERA_KL_LOGDIR=$RUNS
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
# A 0.6B model on a shared 121.6 GiB unified pool: 0.85 refuses to start the
# moment another job holds ~21 GiB (the lesson in bf16_route_served.sh).
export TESSERA_GPU_MEM_UTIL=${TESSERA_GPU_MEM_UTIL:-0.45}

HEAD_COMMIT=${TESSERA_COMMIT:-$(cd "$WT" && git rev-parse HEAD)}
EXT_A=$RUNS/ext-A
EXT_B_RO=$RUNS/ext-B-readonly
mkdir -p "$RUNS" "$EXT_A" "$EXT_B_RO"
chmod a-w "$EXT_B_RO"
cd "$WT"

[ -f "$TEACHER.npz" ] || { echo "no image-matched teacher at $TEACHER.npz"; exit 2; }
for A in armA armB; do
  [ -f "$RUNS/$A/config.json" ] || { echo "no hardlinked arm at $RUNS/$A"; exit 2; }
done
# The arms must still be one inode when the run starts, not just when they were
# made: a re-export between the two serves is exactly the confound this guards.
INO_A=$(stat -c %i "$RUNS/armA/model.safetensors")
INO_B=$(stat -c %i "$RUNS/armB/model.safetensors")
[ "$INO_A" = "$INO_B" ] || { echo "armA and armB are no longer the same inode ($INO_A vs $INO_B)"; exit 2; }
echo "arms share inode $INO_A ($(stat -c %s "$RUNS/armA/model.safetensors") bytes)"

# Arm B's docker arguments: a read-only extensions root, mounted over the one
# the wrappers pass.  TMPDIR stays on the writable /ext, so only the JIT build
# is refused.
ARMB_EXTRA="-v $EXT_B_RO:/ext-ro:ro -e TORCH_EXTENSIONS_DIR=/ext-ro"

# EVERY GPU RUN GOES THROUGH THE BOX LOCK.  Not politeness: vLLM's own startup
# memory profile ASSERTS that free memory does not RISE while it profiles, and
# another agent's job releasing a few GiB mid-startup kills the load outright
# ("Initial free memory 72.8 GiB, current free memory 74.46 GiB", 2026-09-03
# 22:00).  A census that dies at startup measures nothing, and a latency taken
# beside another job measures the other job.
GPULOCK=${GPULOCK:-/home/rob/tmp/arb/gpulock.sh}

census_one() {  # arm mode regime
  local arm=$1 mode=$2 regime=$3
  local flag="" extra=""
  [ "$regime" = compiled ] && flag="--compiled"
  [ "$arm" = armB ] && extra="$ARMB_EXTRA"
  local out=$RUNS/census-$arm-$mode-$regime.json
  # Skip-if-exists, the repo's stage convention: a census is deterministic in
  # what it asserts, and re-running a passing one costs a model load on a box
  # several agents are queueing for.  Delete the receipt to force a re-run.
  if [ -f "$out" ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('$out'))['verdict']=='served' else 1)"; then
    echo "=== census $arm/$mode/$regime already served at $out"; return 0
  fi
  echo "=== census $arm/$mode/$regime  $(date -Is)"
  EXT=$EXT_A TESSERA_SERVE_MODE=$mode "$GPULOCK" experiments/tessera_plugin_run.sh \
    -e TESSERA_SERVE_MODE="$mode" \
    -v "$RUNS":"$RUNS" $extra -- \
    "python3 tools/tessera_route_census.py '$RUNS/$arm' '$out' --tessera-commit $HEAD_COMMIT $flag" \
    2>&1 | tee "$RUNS/census-$arm-$mode-$regime.log"
  # tee eats the exit status; read the verdict off the receipt instead.
  $PY -c "
import json,sys
d=json.load(open('$out'))
print('verdict', d['verdict'])
sys.exit(0 if d['verdict']=='served' else 1)"
}

kl_one() {  # arm mode regime
  local arm=$1 mode=$2 regime=$3
  local name=ts83-$arm-$mode-$regime
  local eager=1; [ "$regime" = compiled ] && eager=0
  echo "=== KL $arm/$mode/$regime  $(date -Is)"
  local extra=""
  [ "$arm" = armB ] && extra="$ARMB_EXTRA"
  EXT=$EXT_A VLLM_CACHE=$RUNS/vllm-cache-$arm \
    TESSERA_LANE_EAGER=$eager TESSERA_KL_NAME=tessera-ts83-$name \
    TESSERA_LANE_DOCKER_EXTRA="$extra" \
    "$GPULOCK" experiments/tessera_plugin_served.sh "$RUNS/$arm" "$name" "$mode" \
    2>&1 | tee "$RUNS/kl-$name.log"
}

if [ "$STAGE" = census ] || [ "$STAGE" = all ]; then
  # Arm A in all four (mode x regime) combinations: the deliverable.  Arm B
  # streamed only -- the resident route never reaches the GEMV lane, so a
  # resident arm B is the same serve as a resident arm A by construction.
  for MODE in resident streamed; do
    for REGIME in eager compiled; do census_one armA "$MODE" "$REGIME"; done
  done
  for REGIME in eager compiled; do census_one armB streamed "$REGIME"; done
fi

if [ "$STAGE" = kl ] || [ "$STAGE" = all ]; then
  for ARM in armA armB; do
    for REGIME in eager compiled; do kl_one "$ARM" streamed "$REGIME"; done
  done
  echo "=== KL summary  $(date -Is)"
  $PY - "$RUNS" <<'PYEOF'
import json, pathlib, sys
runs = pathlib.Path(sys.argv[1])
rows = []
for arm in ("armA", "armB"):
    for regime in ("eager", "compiled"):
        p = runs / f"kl_tessera_ts83-{arm}-streamed-{regime}.json"
        if not p.exists():
            print(f"MISSING {p}"); continue
        d = json.loads(p.read_text())
        rows.append((arm, regime, d["all"]["kl_lower_mean"],
                     d["confident"]["kl_lower_mean"], d["all"]["top1_agree_pct"]))
for arm, regime, a, c, t in rows:
    print(f"{arm:6s} {regime:9s} all={a:.6f} confident={c:.6f} top1={t:.3f}%")
by = {(a, r): (k, c) for a, r, k, c, _ in rows}
for regime in ("eager", "compiled"):
    if ("armA", regime) in by and ("armB", regime) in by:
        ka, _ = by[("armA", regime)]
        kb, _ = by[("armB", regime)]
        print(f"{regime}: GEMV {ka:.6f} vs fallback {kb:.6f}  delta {ka-kb:+.6f} "
              f"({ka/kb:.4f}x)")
PYEOF
fi
echo "=== done $(date -Is)"
