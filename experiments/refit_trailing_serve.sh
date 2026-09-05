#!/usr/bin/env bash
# tessera#75's fair pair, exported and served -- BOTH arms built by the same
# encoder, on the same day.
#
# The measurement half of #75 is on master (``experiments/refit_trailing_pair.py``,
# merged as 9add21d): at the wire, matched pass count, the trailing full-H swap
# is 0.9191x on six dense Qwen units (6/6) and 0.9999x on the GLM six experts,
# so it clears the GLM gate.  ``assert_plane_promotion`` then refuses it on one
# leg only -- "the served KL measures arm None".  This is that leg.
#
#   A  (the control)  a4h1   LDLQ 1.0/32 + refit h^1.0 on every pass.
#   B  (the arm)      bjac   the same with the TRAILING refit under the full H,
#                            stepped in parallel.  Same pass count, same wire.
#
# **A is re-exported, not quoted.**  The obvious A was the 2026-09-02 ldlqH1
# twin, whose served KL is 0.5310275686796917 and whose bytes are still on
# disk.  It is the wrong A: its manifest records ``git: "unknown"`` and a write
# time of 2026-09-02T19:05, and about forty encoder commits have landed since
# (the exact LUT table fit f2f319d, the epsilon-derived swap test 2f6a15a, the
# swap budget 56b4a26, the block-landed refit 3c35ee0, Lloyd's fixed point
# c175c7a/072155f, ...).  Serving it against a 2026-09-04 B measures the
# trailing objective AND that drift, which is two treatments and no control.
# It was caught by the pair check, not by reading the log: 81 of 196 units'
# codes differ between ldlqH1 and bjac, and #75's own screen says the codes of
# this pair are identical -- so the difference could not be the objective.
# The old bytes stay in the run as a DRIFT reading (``compare-drift``), which
# is what they are now evidence of.
#
# Everything that is not the trailing refit's objective is held fixed: the same
# source model, the same Hessian capture, the same static A4 input scales, the
# same teacher npz, the same corpus, the same image, eager, one box, one day.
#
#   refit_trailing_serve.sh export a4h1|bjac   # ~45 min of GPU each
#   refit_trailing_serve.sh compare            # the matched pair, 196 units
#   refit_trailing_serve.sh compare-drift      # what the encoder moved since 09-02
#   refit_trailing_serve.sh serve a4h1|bjac    # dump + KL against the same teacher
#
set -uo pipefail
REPO="${TESSERA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${TESSERA_PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}"
SRC="${TESSERA_SRC:-/home/rob/models/Qwen3-0.6B}"
RUNS="${TESSERA_RUNS:-/mnt/shared/tessera-runs/refit-trailing}"
H="${TESSERA_H:-/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt}"
SCALES="${TESSERA_SCALES:-/mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors}"
# The incumbent's own bytes, exported 2026-09-02 by experiments/ldlq_lut_export_arms.sh.
INCUMBENT="${TESSERA_INCUMBENT:-/mnt/shared/tessera-runs/ldlq-lut/ldlqH1-stock-twin}"
INCUMBENT_WIRE="${TESSERA_INCUMBENT_WIRE:-/mnt/shared/tessera-runs/ldlq-lut/ldlqH1-tessera}"
TEACHER="${TESSERA_TEACHER:-/mnt/shared/tessera-kl/qwen_rot_teacher_lina.json.npz}"
# Where this pair's dumps live.  A variable and not a literal so the stage's
# exit paths can be exercised without a serve (tests/test_refit_trailing_serve.py).
KLDIR="${TESSERA_KL_DIR:-/mnt/shared/tessera-kl}"
export PYTHONPATH=$REPO/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
mkdir -p "$RUNS"
cd "$REPO" || exit 1

case "${1:-export}" in
export)
  # Both objectives named explicitly on the arm.  The LUT plane's default IS
  # h^1.0, but a default is a thing that can move, and this pair has to keep
  # meaning the same pair after it does.
  ARM="${2:-bjac}"
  case "$ARM" in
    a4h1) objective=() ;;                              # the control: uniform
    bjac) objective=(--refit-metric-trailing hessian) ;;
    *) echo "unknown arm $ARM"; exit 2 ;;
  esac
  if [ -f "$RUNS/$ARM-stock-twin/model.safetensors" ]; then echo "$ARM already exported"; exit 0; fi
  echo "== exporting $ARM $(date +%H:%M:%S)"
  nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader > "$RUNS/idle_power_before_$ARM.txt" 2>&1
  $PY -u experiments/export_tessera_serving.py "$SRC" "$RUNS/$ARM-tessera" \
      --grid E2M1x2 --q256 896 --input-scales "$SCALES" \
      --stock-twin "$RUNS/$ARM-stock-twin" \
      --hessian "$H" --refit-metric h^1.0 "${objective[@]}" \
      > "$RUNS/export_$ARM.log" 2>&1 \
    || { echo "export $ARM FAILED"; tail -20 "$RUNS/export_$ARM.log"; exit 1; }
  tail -3 "$RUNS/export_$ARM.log"
  ;;
compare)
  # The matched pair, on all 196 units of the real checkpoint rather than the
  # six the screen measured: codes identical, scale plane moved, wire the same
  # length.  ``compare_stock_checkpoints.py`` is the wrong instrument here --
  # it exits 1 on the expected outcome and names only its first few diffs,
  # where this pair needs a per-unit count and a receipt file.
  $PY experiments/refit_trailing_bytes.py \
      "$RUNS/a4h1-stock-twin" "$RUNS/bjac-stock-twin" \
      --wire-a "$RUNS/a4h1-tessera" --wire-b "$RUNS/bjac-tessera" \
      --out experiments/results/refit_trailing_bytes.json \
      2>&1 | tee "$RUNS/compare_pair.log" | tail -40
  exit ${PIPESTATUS[0]}
  ;;
compare-drift)
  # The SAME recipe, two encoders: 2026-09-02's ldlqH1 against today's a4h1.
  # Nothing here is a promotion input; it is the reading that says why the
  # 09-02 bytes cannot be this A/B's control.
  $PY experiments/refit_trailing_bytes.py \
      "$INCUMBENT" "$RUNS/a4h1-stock-twin" \
      --wire-a "$INCUMBENT_WIRE" --wire-b "$RUNS/a4h1-tessera" \
      --out experiments/results/refit_trailing_encoder_drift.json \
      2>&1 | tee "$RUNS/compare_drift.log" | tail -40
  ;;
serve)
  # serve ARM -- a4h1 | bjac | incumbent.  `incumbent` re-serves the 2026-09-02
  # ldlqH1 twin, which is a drift reading and NOT this pair's bar.
  ARM="${2:-bjac}"
  case "$ARM" in
    incumbent) ARM=ldlqH1; twin=$INCUMBENT ;;
    *)         twin=$RUNS/$ARM-stock-twin ;;
  esac
  [ -f "$twin/model.safetensors" ] || { echo "no export for $ARM"; exit 1; }
  export TESSERA_KL_PORT="${TESSERA_KL_PORT:-8005}"
  # 0.15 of 121 GB is ~18 GB, and the twin is a 0.6B model whose whole
  # checkpoint is 0.84 GB: the rest is KV cache for eight sequences of 512
  # tokens, which is orders of magnitude more than they need.  It was 0.30,
  # and the number is not free -- it is what the pool action has to declare.
  # At mem_gb=40 this action starved: sparky offers 48, so it only fits when
  # the box is nearly empty, and small GPU items refilled the box faster than
  # 40 GB ever came free (1507 denied passes in three hours, past the
  # 900 s withhold ceiling, so it had lost its veto and kept only its place).
  # At 24 -- the demand the bjac export actually ran under -- it fits beside
  # one other item.  Both arms serve at the same utilization, so the A/B is
  # matched whatever the value; what changes is only how long the pair waits.
  export TESSERA_GPU_MEM_UTIL="${TESSERA_GPU_MEM_UTIL:-0.15}"
  export TESSERA_KL_NAME="${TESSERA_KL_NAME:-tessera-kl-ts75}"
  source "$(dirname "$0")/runtime_image.sh"
  export TESSERA_KL_IMAGE=$(runtime_image_pin)
  export TESSERA_KL_CORPUS="${TESSERA_KL_CORPUS:-$KLDIR/corpus_qwen_n8_s512.json}"
  export TESSERA_KL_LOGDIR=$RUNS
  dump=$KLDIR/qwen_ts75_$ARM.json
  npz=$dump.npz
  if [ ! -f "$npz" ]; then
    experiments/serve_and_dump_kl.sh "$twin" "$dump" student || exit 1
  fi
  # The comparison is a STAGE of this step, not a postscript to it.  This
  # script runs under `set -uo pipefail` and not errexit, so a failed
  # `kl_tool compare` was recorded by pipefail for the pipeline -- and then
  # discarded, because the pipeline is the branch's last command and the
  # unconditional `echo STEP_DONE; date` below became the exit status.  A
  # missing or refused teacher therefore reported a completed stage
  # (tessera#251).  Capture PIPESTATUS[0], the way the `compare` branch above
  # already does, and refuse.
  receipt=$RUNS/kl_$ARM.json
  attempt=$receipt.attempt.$$
  rm -f "$attempt"
  $PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER" "$npz" \
      --out "$attempt" 2>&1 | tee "$RUNS/kl_$ARM.log" | tail -12
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    rm -f "$attempt"
    # An earlier attempt's receipt sits at the path a reader looks in, and
    # nothing in it says which attempt wrote it.  Move it aside by name
    # rather than leave it to certify this failure.
    if [ -f "$receipt" ]; then
      mv "$receipt" "$receipt.stale"
      echo "REFUSED: $receipt was written by an earlier attempt and does not" \
           "describe this one; moved to $receipt.stale" >&2
    fi
    echo "REFUSED: KL compare for $ARM exited $rc; log at $RUNS/kl_$ARM.log" >&2
    exit "$rc"
  fi
  # Published only by a compare that succeeded, and in one move, so the
  # canonical path never holds a partial or an older attempt's receipt.
  if [ ! -f "$attempt" ]; then
    echo "REFUSED: KL compare for $ARM exited 0 and wrote no receipt" >&2
    exit 1
  fi
  mv "$attempt" "$receipt"
  ;;
*) echo "usage: $0 export ARM|compare|compare-drift|serve ARM"; exit 2;;
esac
echo "STEP_DONE ${1:-export}"; date
