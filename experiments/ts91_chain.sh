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
# local flock cannot balance two boxes, and it starved this very experiment
# for two hours.  Not --exclusive: this is a CORRECTNESS measurement (which
# key does each arm compute), so it needs the arms honestly separated --
# separate cache roots and separate extension dirs -- and not a quiet box.
PBRUN=${PBRUN:-/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py}
GPUWRAP=(/usr/bin/python3 "$PBRUN" --gpu --)
export TS91_NO_LOCK=1 WT SUF
exec "${GPUWRAP[@]}" bash "$WT/experiments/ts91_chain_body.sh"
