#!/usr/bin/env bash
# Thin entry/RPC adapter. Lifecycle and refusals belong to tools/_flux_serve.py.
_FLUX_SERVE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${FLUX_JOB_ID:-}" ]; then
  echo 'REFUSED: OUTSIDE_FLUX_JOB' >&2
  exit 2
fi
if [ -z "${TESSERA_SERVE_SOCKET:-}" ]; then
  if [ -z "${TESSERA_SERVE_SPEC:-}" ] || [ -z "${TESSERA_SERVE_OUT:-}" ]; then
    echo "REFUSED: SERVE_SPEC_REQUIRED: set TESSERA_SERVE_SPEC and TESSERA_SERVE_OUT" >&2
    exit 2
  fi
  exec python3 "$_FLUX_SERVE_ROOT/tools/_flux_serve.py" wrapper \
    --spec "${TESSERA_SERVE_SPEC:?set TESSERA_SERVE_SPEC to the reviewed normalized spec}" \
    --out "${TESSERA_SERVE_OUT:?set TESSERA_SERVE_OUT to the unique shared job directory}" \
    -- bash "$0" "$@"
fi
flux_serve_start() { python3 "$_FLUX_SERVE_ROOT/tools/_flux_serve.py" start "$@"; }
flux_serve_stop() { python3 "$_FLUX_SERVE_ROOT/tools/_flux_serve.py" stop; }
