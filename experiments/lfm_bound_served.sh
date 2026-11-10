#!/usr/bin/env bash
# One source-bound LFM score in the reviewed serving supervisor.
set -euo pipefail
source "$(dirname "$0")/flux_serve.sh"
stage=${1:?usage: lfm_bound_served.sh teacher|student [student options]}
shift
case "$stage" in
  teacher) exec python3 "$_FLUX_SERVE_ROOT/experiments/ts5_lfm_teacher_bound.py" "$@" ;;
  student) exec python3 "$_FLUX_SERVE_ROOT/experiments/ts5_lfm_served_bound.py" student "$@" ;;
  *) echo 'REFUSED: LFM_SCORED_STAGE_REQUIRED' >&2; exit 2 ;;
esac
