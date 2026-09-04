#!/usr/bin/env bash
# tessera#75's served leg, end to end, in ONE pool action.
#
# One action because the two arms must share a box and a day, and because the
# thing that goes wrong here is a chain half-run: an A/B whose control was
# built by a different encoder is what this run exists to stop making.  So the
# script enforces its own preconditions rather than reporting them:
#
#   export A -> the PAIR CHECK -> (only if it passes) serve A, serve B -> gate
#
# A failing pair check exits before either serve.  Same-day arms whose codes
# differ would mean the encoder is not deterministic, and a served number on
# top of that measures two treatments again -- which is the failure the whole
# re-export exists to avoid, arriving by a different door.
#
# Everything it produces lands in $R as well as in the checkout, and $R/DONE
# carries the verdict line, so the results survive the client that submitted
# the action.
set -uo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
R=/mnt/shared/tessera-runs/refit-trailing
PY="${TESSERA_PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}"
rm -f "$R/DONE"
nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader > "$R/idle_power_before_serve.txt" 2>&1
echo "idle before: $(cat "$R/idle_power_before_serve.txt")"

echo "== A: the control, exported by the same encoder as B"
bash experiments/refit_trailing_serve.sh export a4h1 || { echo "EXPORT_A_FAILED"; exit 1; }

echo "== the matched pair, 196 units"
bash experiments/refit_trailing_serve.sh compare; pair=$?
echo "PAIR_EXIT=$pair"
echo "== the encoder drift the 2026-09-02 bytes carry (a reading, not a gate input)"
bash experiments/refit_trailing_serve.sh compare-drift; echo "DRIFT_EXIT=$?"
if [ $pair -ne 0 ]; then
  echo "PAIR_FAILED: the arms are not the matched pair, so no served number here"
  echo "would be about the trailing objective.  Stopping before the serves."
  { echo "PAIR_FAILED"; } > "$R/DONE"
  exit 1
fi

echo "== serve A"
bash experiments/refit_trailing_serve.sh serve a4h1 || { echo "SERVE_A_FAILED"; exit 1; }
echo "== serve B"
bash experiments/refit_trailing_serve.sh serve bjac || { echo "SERVE_B_FAILED"; exit 1; }

echo "== the gate, verbatim"
# A distinct --out: the merged refit_trailing_pair_gate.json is the screen's
# own verdict and stays the record of what a screen earns.
$PY experiments/refit_trailing_pair_gate.py \
    --served-arm "B-Jac" \
    --served-kl-json "$R/kl_bjac.json" \
    --served-bar-json "$R/kl_a4h1.json" \
    --out experiments/results/refit_trailing_pair_gate_served.json \
    2>&1 | tee "$R/gate_served.log"
echo "GATE_EXIT=${PIPESTATUS[0]}"

cp -f experiments/results/refit_trailing_bytes.json \
      experiments/results/refit_trailing_encoder_drift.json \
      experiments/results/refit_trailing_pair_gate_served.json "$R/" 2>/dev/null
{
  echo "pair: $(python3 -c "import json;print(json.load(open('experiments/results/refit_trailing_bytes.json'))['verdict'])")"
  echo "A  served: $($PY -c "import json;print(json.load(open('$R/kl_a4h1.json'))['all']['kl_lower_mean'])")"
  echo "B  served: $($PY -c "import json;print(json.load(open('$R/kl_bjac.json'))['all']['kl_lower_mean'])")"
  grep -E "assert_plane_promotion|geomean" "$R/gate_served.log"
} > "$R/DONE"
cat "$R/DONE"
echo RUN_ALL_DONE
