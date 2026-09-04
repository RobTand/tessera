#!/usr/bin/env bash
# The 16-bit route, served: the gate issue #9 asks for, end to end.
#
# The BF16 wire has had an encoder, a recipe, an exporter and a stock twin
# since 2026-09-02 and the best 8-bpp weight-space quality Tessera has measured
# (dense screen 0.00783 against E4M3's 0.02280 and FP8 RTN's 0.02341, at 1%
# more bytes).  What it has never had is a SERVED number, because ``ROUTES``
# had two entries and a checkpoint declaring ``TESSERA_BF16`` was refused at
# config parse.  The route now exists; this is what turns that into evidence.
#
# WHAT IT PROVES, AND WHAT IT DOES NOT.  It proves the plugin loads a BF16
# checkpoint, that every declared module executes the BF16 route in both the
# prefill and the decode shape and under both the eager and the compiled
# forward (the census, which compares what the serve RECORDED against what the
# checkpoint DECLARES), and what the artifact's KL-vs-BF16 is on the gold
# corpus in both residency modes.  It does not license a ``lane_eligibility``
# cell on its own: a cell says a route status was observed on a platform at a
# rung, so read the census JSON and add the cell deliberately, with the receipt
# path in the changelog.
#
# THE TWIN IS THE CONTROL, AND IT IS NOT AN EQUAL ARM.  ``--stock-twin`` writes
# a plain BF16 safetensors of ``materialize_bf16_folded`` -- the SAME encode,
# with the row scale folded into the tile because a one-tensor checkpoint has
# no way to carry it separately.  Vanilla vLLM serves it with no plugin.  So
# twin-vs-route isolates exactly one thing: the fold.  The route should be the
# BETTER arm (the fold costs 0.0011-0.0022 absolute relative output error at
# any rate, ~16% of the total at R=7), and a route that merely TIES the twin is
# a finding -- it would mean the epilogue is rounding where it should not.
#
# usage: bf16_route_served.sh [q256 ...]        # default 1792 (R=7)
#
# One serve at a time: every serving step below goes through
# ``serve_lock.sh``'s lock, taken by the script that starts the container.
# Do NOT run this concurrently with another serving arm on the same box.
set -euo pipefail

WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
SRC=${SRC:-/home/rob/models/Qwen3-0.6B}
OUT=${OUT:-/mnt/shared/tessera-runs/bf16}
RUNS=${RUNS:-/home/rob/tessera-runs/bf16route}
KLDIR=/mnt/shared/tessera-kl
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
TEACHER=$KLDIR/qwen_teacher_bf16_v028.json
export TESSERA_KL_CORPUS=${TESSERA_KL_CORPUS:-$KLDIR/corpus_qwen_n8_s512.json}
# IMAGE-MATCHED, and it has to be said out loud: serve_and_dump_kl.sh defaults
# to the GLM image, and the teacher this compares against was dumped on
# the pinned vLLM 0.28 image.  A KL taken against a teacher from another image is
# a number about two runtimes.
source "$(dirname "$0")/runtime_image.sh"
export TESSERA_KL_IMAGE=${TESSERA_KL_IMAGE:-$(runtime_image_pin)}
export TESSERA_KL_LOGDIR=$RUNS
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
# THE BOX IS SHARED AND THE MODEL IS 0.6B.  Both serve scripts default to
# --gpu-memory-utilization 0.85, which on a 121.6 GiB unified pool asks for
# 103 GiB and REFUSES TO START ("Free memory on device cuda:0 (100.7/121.63
# GiB) is less than desired") the moment another job on the box holds ~21 GiB
# -- which is what killed the first resident serve of this arm at 22:22 on
# 2026-09-02.  A 0.6B model with a 4096 context needs a small fraction of
# that, so ask for a fraction: the KL is a number about the weights, not
# about how much of the pool the serve reserved.
export TESSERA_GPU_MEM_UTIL=${TESSERA_GPU_MEM_UTIL:-0.45}
# THE TREE UNDER TEST IS THIS ONE.  ``tessera_plugin_{run,served}.sh`` default
# TS to /home/rob/tessera, so a branch that is not the main checkout would
# install MASTER into the container -- and the serve would then refuse this
# checkpoint with the very error this branch removes, an arm that measures
# nothing while looking like it ran.  RUNS is exported for the same reason:
# the KL JSON this script's summary reads is written by the serve script under
# ITS ``RUNS``, which defaults elsewhere.
export TS=$WT
export RUNS
# The JIT build cache is shared with the other plugin arms on purpose: it is
# keyed by source, and a private one would rebuild the span-2 extension for a
# family that never calls it.
export EXT=${EXT:-/home/rob/tessera-runs/tsplugin/ext}
# The commit the container is running.  A worktree's .git is a POINTER file, so
# the census's own `git rev-parse` inside the container resolves nowhere and
# would stamp the receipt with None; it is read here, on the host, instead.
HEAD_COMMIT=${TESSERA_COMMIT:-$(cd "$WT" && git rev-parse HEAD)}
mkdir -p "$RUNS" "$EXT"
cd "$WT"

# The KL corpus contract is per TOKENIZER, and this is a Qwen artifact: the
# default contract is GLM-tokenized and ``kl_tool`` refuses the mismatch, which
# is the refusal you want rather than a number computed on the wrong text.
[ -f "$TEACHER.npz" ] || {
  echo "no image-matched teacher at $TEACHER.npz -- run stock_lane_served.sh's teacher arm first"
  exit 2
}

for Q in "${@:-1792}"; do
  R=$((Q / 256))
  ARM=bf16-r$R
  WIRE=$OUT/qwen0.6b-bf16-r$R
  TWIN=$WIRE-twin
  PLUG=$WIRE-plugin

  # 1. Export the wire and its twin, and verify every twin tensor IS
  #    materialize_bf16_folded of the wire.  Skipped if it is already there:
  #    the encode is deterministic, and re-exporting under a serve is how a
  #    box runs out of memory.
  if [ ! -f "$WIRE/config.json" ]; then
    echo "=== export R=$R  $(date -Is)"
    WT="$WT" SRC="$SRC" OUT="$OUT" PY="$PY" experiments/bf16_export_qwen.sh "$Q"
  else
    echo "=== export R=$R already at $WIRE"
  fi

  # 1b. Point it at the plugin.  A wire written before the plugin move carries
  #     ``quant_method: "gridbook"``; the retarget HARDLINKS the weights and
  #     rewrites config.json only, so the served bytes are the exported bytes
  #     by construction rather than by a copy someone has to trust.
  QM=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['quantization_config']['quant_method'])" "$WIRE/config.json")
  if [ "$QM" = "tessera" ]; then
    PLUG=$WIRE
    echo "=== retarget R=$R not needed (already quant_method=tessera)"
  elif [ ! -f "$PLUG/config.json" ]; then
    echo "=== retarget R=$R -> $PLUG  $(date -Is)"
    $PY experiments/retarget_checkpoint_to_plugin.py "$WIRE" "$PLUG"
  else
    echo "=== retarget R=$R already at $PLUG"
  fi

  # 2. The census, in both residency modes and both forward regimes.  This is
  #    the principle-9 leg: a route nothing reads is a confession log, so the
  #    receipt is the record the serve wrote, read from inside the worker.
  #    The compiled arm is not a formality -- every route bug this lane has
  #    had (int() on the token dim, data_ptr fingerprints, a mutating op on an
  #    aliased pool) was invisible to an eager census.
  for MODE in resident streamed; do
    for REGIME in eager compiled; do
      FLAG=""
      [ "$REGIME" = compiled ] && FLAG="--compiled"
      echo "=== census R=$R $MODE/$REGIME  $(date -Is)"
      TESSERA_SERVE_MODE=$MODE experiments/tessera_plugin_run.sh \
        -e TESSERA_SERVE_MODE="$MODE" -v /mnt/shared:/mnt/shared -v "$RUNS":"$RUNS" -- \
        "python3 tools/tessera_route_census.py '$PLUG' '$RUNS/census-$ARM-$MODE-$REGIME.json' --runtime-image \"\$TESSERA_CENSUS_RUNTIME_IMAGE\" --tessera-commit $HEAD_COMMIT $FLAG" \
        || { echo "census FAILED for $ARM/$MODE/$REGIME"; exit 3; }
    done
  done

  # 3. The served KL, both modes.  Byte-identical inputs, one flag apart: the
  #    two modes decode the same wire and must agree to the last digit, which
  #    is the same acceptance the other two families' modes carry.
  for MODE in resident streamed; do
    echo "=== serve R=$R $MODE  $(date -Is)"
    experiments/tessera_plugin_served.sh "$PLUG" "$ARM-$MODE" "$MODE"
  done

  # 4. The control: the twin, on vanilla vLLM with no plugin at all.
  if [ ! -f "$KLDIR/qwen_bf16twin_r$R.npz" ]; then
    echo "=== serve twin R=$R  $(date -Is)"
    experiments/serve_and_dump_kl.sh "$TWIN" "$KLDIR/qwen_bf16twin_r$R" student
  else
    echo "=== twin R=$R already dumped"
  fi

  echo "=== R=$R summary  $(date -Is)"
  $PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER.npz" \
      "$KLDIR/qwen_bf16twin_r$R.npz" --out "$RUNS/kl_$ARM-twin.json" | tail -6
  for MODE in resident streamed; do
    echo "--- route $MODE ---"
    tail -6 "$RUNS/kl_tessera_$ARM-$MODE.json" 2>/dev/null || true
  done
  # The two modes must be one number; the twin should be WORSE than both.
  # THE KEY IS ``all.kl_lower_mean``, AND A MISSING ONE IS A FAILURE.  This
  # block first read a top-level ``mean_kl``, which ``kl_tool.py compare`` has
  # never written: every arm read None, both acceptance checks were skipped,
  # and the run printed "resident=None streamed=None twin=None" as though that
  # were a pass.  A check that cannot fail is worse than no check, so the
  # reader below refuses a file it cannot find the number in.
  #
  # Only ONE of the two checks exits non-zero.  The modes decode the same wire
  # and must agree exactly -- a difference there is a decode defect.  The fold
  # is a quality claim whose size depends on the rung: at q256=1792 it moves
  # ``all`` by +0.11% and moves ``confident`` the OTHER way by -0.39%, i.e. it
  # sits below what this corpus resolves, so a sign flip there is reported and
  # not treated as a failure.
  $PY - "$RUNS" "$ARM" <<'PYEOF'
import json, sys, pathlib
runs, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
def kl(p):
    if not p.exists():
        raise SystemExit(f"no KL json at {p} -- the arm did not run")
    d = json.loads(p.read_text())
    try:
        return d["all"]["kl_lower_mean"], d["confident"]["kl_lower_mean"], d["all"]["top1_agree_pct"]
    except KeyError:
        raise SystemExit(f"{p} has no all.kl_lower_mean; keys={sorted(d)}")
res, res_c, res_t = kl(runs / f"kl_tessera_{arm}-resident.json")
strm, strm_c, strm_t = kl(runs / f"kl_tessera_{arm}-streamed.json")
twin, twin_c, twin_t = kl(runs / f"kl_{arm}-twin.json")
print(f"{arm}: resident={res!r} streamed={strm!r} twin(folded)={twin!r}")
print(f"  confident: resident={res_c!r} twin={twin_c!r}")
print(f"  top1 agree: resident={res_t:.3f}% twin={twin_t:.3f}%")
if (res, res_c, res_t) != (strm, strm_c, strm_t):
    print("  MODES DISAGREE -- the two residencies decode the same wire and must not")
    raise SystemExit(4)
print(f"  fold cost: all {twin - res:+.6f} nats ({twin / res:.4f}x), "
      f"confident {twin_c - res_c:+.6f} nats ({twin_c / res_c:.4f}x)")
if res >= twin or res_c >= twin_c:
    print("  THE FOLD IS BELOW WHAT THIS CORPUS RESOLVES on at least one subset --"
          " that is a statement about the rung, not a pass; do not cite a fold win from it")
PYEOF
done
echo "=== done $(date -Is)"
