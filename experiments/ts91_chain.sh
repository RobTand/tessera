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
# GPU work is admitted by the PrismaBuild pool, not by a box-local lock: a
# local lock cannot balance two boxes, and it starved this very experiment for
# two hours.  Not --exclusive: this is a CORRECTNESS measurement (which key
# does each arm compute), so it needs the arms honestly separated -- separate
# cache roots and separate extension dirs -- and not a quiet box.
#
# The pool runs the recorded argv under a CLOSED environment, so everything
# the chain needs travels as an argument.  Nothing here may be passed by
# export: see the header of ts91_chain_body.sh for what that cost.
PBRUN=${PBRUN:-/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py}
RUNS=${RUNS:-/home/rob/tessera-runs/ts91}
# The serve pin, resolved HERE and handed to the body, so a before-arm that
# predates issue #100 runs the same bytes as the after-arm.
source "$WT/experiments/runtime_image.sh"
IMG=${IMG:-$(runtime_image_pin)}
exec /usr/bin/python3 "$PBRUN" --gpu -- \
  bash "$WT/experiments/ts91_chain_body.sh" "$WT" "$SUF" \
       "$RUNS/ext-A-$SUF" "$RUNS/ext-B-readonly-$SUF" "$IMG"
