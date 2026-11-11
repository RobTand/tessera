#!/usr/bin/env bash
# The #91 reproduction, four censuses under ONE GPU slot.
#
#   A -> cacheX   the window-GEMV lane, into a fresh cache
#   B -> cacheY   the torch-window lane, into a fresh cache  (control: B alone works)
#   B -> cacheX   the torch-window lane into the GEMV lane's cache
#   A -> cacheY   the GEMV lane into the torch-window lane's cache
#
# The first two answer "do the two lane states compute the same AOT key?" by
# reading the directory name vLLM 0.28 derives it into.  The last two answer
# "what does that cost?".  Pass a suffix to run the same four into fresh cache
# roots (e.g. after the fix).
set -uo pipefail
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
SUF=${1:-before}
# GPU work is admitted by the scheduler, not by a box-local lock: a
# local lock cannot balance two boxes, and it starved this very experiment for
# two hours.  Not --exclusive: this is a CORRECTNESS measurement (which key
# does each arm compute), so it needs the arms honestly separated -- separate
# cache roots and separate extension dirs -- and not a quiet box.
#
# The job runs the recorded argv under a CLOSED environment, so everything
# the chain needs travels as an argument.  Nothing here may be passed by
# export: see the header of ts91_chain_body.sh for what that cost.
RENDER="$WT/tools/_flux_gpu_jobspec.py"
RUNS=${RUNS:-/home/rob/tessera-runs/ts91}
# The serve pin, resolved HERE and handed to the body, so a before-arm that
# predates issue #100 runs the same bytes as the after-arm.
source "$WT/experiments/runtime_image.sh"
IMG=${IMG:-$(runtime_image_pin)}
# Submit and collect one Flux job. FLUX_SUBMIT=0 renders a preview.
# The retained 60-minute workload is refused until its duration is approved.
SUBMIT_ARGS=(--submit)
[[ "${FLUX_SUBMIT:-1}" != 0 ]] || SUBMIT_ARGS=()
exec /usr/bin/python3 "$RENDER" "${SUBMIT_ARGS[@]}" --cores 4 --memory-gib 32 --walltime 60m \
  --out "${FLUX_OUT:?set FLUX_OUT to a unique owned output directory under /mnt/shared}" --cwd "$WT" -- \
  bash "$WT/experiments/ts91_chain_body.sh" "$WT" "$SUF" \
       "$RUNS/ext-A-$SUF" "$RUNS/ext-B-readonly-$SUF" "$IMG"
