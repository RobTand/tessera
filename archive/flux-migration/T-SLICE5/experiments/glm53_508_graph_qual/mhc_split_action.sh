#!/usr/bin/env bash
# PB-admitted entrypoint for mhc_split_repro.py (tessera#508): runs the repro in
# the pinned serving image through mhc_split_run.sh and prints the JSON report
# on stdout so it returns in PB's receipt.
#   mhc_split_action.sh IMAGE_REF OUT_DIR
set -uo pipefail
if [[ -z "${PRISMABUILD_ACTION_KEY:-${PB_ACTION_KEY:-}}" ]]; then
  echo 'mhc_split_action.sh requires PrismaBuild admission' >&2
  exit 2
fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$here/mhc_split_run.sh" "$1" "$2"
status=$?
echo "MHC_SPLIT_REPORT_BEGIN"
cat "$2/mhc_split.json" 2>/dev/null
echo "MHC_SPLIT_REPORT_END"
exit "$status"
