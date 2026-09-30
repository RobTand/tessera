#!/bin/bash
# Export one T-16 dense census stub: the 16 dense MLP modules encoded fresh at
# the plan's BF16 rungs (GPU), the five routed stacks from u1 stub B's cached
# expert units (no encode).  Run with cwd = the Tessera checkout.
# usage: PRODUCER_AUTHORITY=/abs/authority.py export_census_stub.sh NAME
#   (plan: $R/plan-NAME.json, out: $R/stub-NAME).  The Hessian references bind
#   the producer's calibration cache, which the exporter reads only through the
#   producer's own authority file (--producer-authority, tessera#599).
#   The routed units come from stub B's cache through a routed-only manifest
#   ($R/cache-B-routed: the bundle's 4320 expert units, hard-linked (the bundle refuses a symlink that resolves outside it); the exporter
#   requires the manifest to cover exactly the planned cached units).
set -uo pipefail
S="$1"
AUTH="${PRODUCER_AUTHORITY:?set PRODUCER_AUTHORITY to the producer authority file}"
R=/mnt/shared/tessera-measurements/t16-coverage-20260930/stubs
SRCR=/mnt/shared/tessera-runs/moe/u1-stubs-20260926
U=/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/union-a4a8a16-01/cache
PRODUCER=/mnt/shared/tessera-measurements/glm-canonical-census-20260908/identity-reseal-20260915/producer-source-c92826fa4/src/tessera
PY=/home/rob/venvs/pq-pb461728e4-tessera-07bfcc0e/bin/python
OUT="$R/stub-$S"
LOG="$R/logs/export-$S-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[export_census_stub] $S host=$(hostname) start=$(date -u +%FT%TZ) head=${TESSERA_HEAD:-$(git rev-parse HEAD 2>/dev/null)} gpu=$(nvidia-smi --query-gpu=name,power.draw --format=csv,noheader 2>/dev/null)" | tee "$LOG"
PYTHONPATH=src:experiments TMPDIR=$R/tmp "$PY" experiments/export_tessera_serving.py "$SRCR/source-l8" "$OUT" \
  --plan-json "$R/plan-$S.json" --device cuda --producer-authority "$AUTH" \
  --hessian "$U/hessian_capture.references.json" \
  --cached-expert-units "$R/cache-B-routed/cached_units.u1-stub-B.routed.v1.json" --cached-hessian-identity committed \
  --cached-producer-package "$PRODUCER" \
  --cached-producer-source-sha256 a4c9209437c7601d4f8cd3ab8ac1e7a2d2db33461a4245e0fbbbdf74a9d8de83 \
  --cached-intake-threads 4 --source-digest-cache "$R/source-digests" --allow-unserveable >> "$LOG" 2>&1
rc=$?
echo "[export_census_stub] $S rc=$rc end=$(date -u +%FT%TZ)" | tee -a "$LOG"
exit $rc
