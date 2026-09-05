#!/usr/bin/env bash
# Retired historical campaign: fixed output names reuse the 2026-09-04 arms.
# The original implementation and duplicate-job finding remain in git history.
set -eu
printf '%s\n' \
  'This completed campaign is retired; it cannot run against cached historical arms.' \
  'Receipts and disposition: docs/measurements/branch-recovery-2026-09-05.md' >&2
exit 2
