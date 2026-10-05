#!/usr/bin/env bash
# One-arm command inspection only. Real execution is rank_window.py in each local PB action.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ ${1:-} != --dry-run ]]; then
  echo 'arm_tp2.sh no longer launches ranks directly; use drive_tp2.sh to prepare a published PB campaign' >&2
  exit 3
fi
shift
exec "${RUNTIME_IMAGE_PY:-python3}" -c \
  'import os,sys; sys.path.insert(0,sys.argv[1]); from window_driver import dry_arm; from managed_window import Refused
try: dry_arm(sys.argv[2],os.environ)
except (Refused,ValueError,OSError) as e: print(e); sys.exit(3)' \
  "$HERE" "${1:?usage: arm_tp2.sh --dry-run ARM}"
