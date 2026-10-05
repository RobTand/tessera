#!/bin/bash
# Export one dense census stub: the 16 dense MLP modules encoded fresh at the
# plan's rungs and grid (GPU; BF16 for the T-16 stubs, E4M3 for the T-8 ones),
# the five routed stacks from u1 stub B's cached expert units (no encode).  Run
# with cwd = the Tessera checkout.
# usage: TESSERA_PRODUCER_PYTHON=/abs/producer/bin/python \
#        TESSERA_PRODUCER_SOURCE=/abs/genuine-checkout/src/tessera \
#        PRODUCER_AUTHORITY=/abs/authority.py [CENSUS_ROOT=/abs/root] \
#        [PART_BOUND_S<=2700] export_census_stub.sh NAME
#   (plan: $R/plan-NAME.json, out: $R/stub-NAME, R = $CENSUS_ROOT/stubs; the
#   default root is the T-16 coverage directory).  ROUTED_CACHE names the
#   routed-only manifest's directory (default $R/cache-B-routed).  The Hessian
#   references bind the producer's calibration cache, which the exporter reads
#   only through the producer's own authority file (--producer-authority,
#   tessera#599).  The routed units come from stub B's cache through a
#   routed-only manifest ($R/cache-B-routed: the bundle's 4320 expert units,
#   hard-linked (the bundle refuses a symlink that resolves outside it); the
#   exporter requires the manifest to cover exactly the planned cached units).
#
# The producer (#944): TESSERA_PRODUCER_PYTHON names the interpreter that runs
# the exporter and TESSERA_PRODUCER_SOURCE names, explicitly and immutably,
# the qualified genuine source checkout's src/tessera it must authenticate
# against.  Both are required by name and passed through UNCHANGED -- never
# derived from $PWD (a PrismaBuild snapshot is an intentionally parentless
# tree and cannot qualify), never overwritten, and a relative reference
# refuses.  The interpreter is authenticated BEFORE anything runs through
# tessera.export_serving.authenticate_producer_python: the installed
# distribution it imports must BE the genuine source package, at a clean HEAD
# descending from the genuine producer commit, under the selected
# sys.executable.  The same two variables are exported, unchanged, to the
# actual exporter process, which re-proves them in-process; the exporter then
# runs as "$TESSERA_PRODUCER_PYTHON" -m tessera.export_serving on this host:
# no hard-coded interpreter, no PYTHONPATH source shadow.  The cached routed
# units keep their ORIGINAL producer's immutable package and sealed source
# sha256 (--cached-producer-*, tessera#599): that historical identity is
# intake provenance, never the interpreter executing this export.  The action
# is bounded by PART_BOUND_S (default and cap 2700 s); a larger or
# non-numeric request refuses by name.
set -uo pipefail
S=${1:?NAME}
AUTH=${PRODUCER_AUTHORITY:?set PRODUCER_AUTHORITY to the producer authority file}
PRODUCER_SOURCE=${TESSERA_PRODUCER_SOURCE:-}
R=${CENSUS_ROOT:-/mnt/shared/tessera-measurements/t16-coverage-20260930}/stubs
ROUTED_CACHE=${ROUTED_CACHE:-$R/cache-B-routed}
SRCR=/mnt/shared/tessera-runs/moe/u1-stubs-20260926
U=/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/union-a4a8a16-01/cache
# The CACHED routed units' original producer (intake provenance, tessera#599).
PRODUCER=/mnt/shared/tessera-measurements/glm-canonical-census-20260908/identity-reseal-20260915/producer-source-c92826fa4/src/tessera
OUT="$R/stub-$S"
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R/logs" "$R/source-digests" "$R/tmp" || exit 2
LOG="$R/logs/export-$S-$TS.log"

refuse() { echo "[export_census_stub] refuse: $*" | tee -a "$LOG" >&2; exit 2; }

BOUND=${PART_BOUND_S:-2700}
case "$BOUND" in
  ''|*[!0-9]*) refuse "PART_BOUND_S=${PART_BOUND_S:-} is not a whole number of seconds";;
esac
[ "$BOUND" -ge 1 ] && [ "$BOUND" -le 2700 ] || refuse "PART_BOUND_S=$BOUND must be in 1..2700 seconds"

PRODUCER_PY=${TESSERA_PRODUCER_PYTHON:-}
[ -n "$PRODUCER_PY" ] || refuse "TESSERA_PRODUCER_PYTHON is not set: name the producer interpreter that will run the exporter"
[ -x "$PRODUCER_PY" ] || refuse "TESSERA_PRODUCER_PYTHON=$PRODUCER_PY is not an executable file"
[ -n "$PRODUCER_SOURCE" ] || refuse "TESSERA_PRODUCER_SOURCE must name the qualified immutable source"
case "$PRODUCER_SOURCE" in
  /*) ;;
  *) refuse "TESSERA_PRODUCER_SOURCE=$PRODUCER_SOURCE is not an absolute path; a qualified source reference is never derived from the working directory";;
esac
AUTH_ERR="$R/logs/auth-$S-$TS.err"
AUTH_RECEIPT=$(TESSERA_PRODUCER_PYTHON="$PRODUCER_PY" TESSERA_PRODUCER_SOURCE="$PRODUCER_SOURCE" \
  "$PRODUCER_PY" -c 'import json
from tessera.export_serving import authenticate_producer_python
print(json.dumps(authenticate_producer_python()))' 2>"$AUTH_ERR") || {
  refuse "TESSERA_PRODUCER_PYTHON=$PRODUCER_PY failed producer authentication against $PRODUCER_SOURCE (stderr kept at $AUTH_ERR): $(tail -n 3 "$AUTH_ERR" | tr '\n' ' ')"
}
PRODUCER_SHA=$(python3 -c 'import json,sys
r = json.loads(sys.argv[1])
print(r.get("package_sha256") or "")' "$AUTH_RECEIPT") || refuse "producer authentication receipt is not JSON: $AUTH_RECEIPT"
[ -n "$PRODUCER_SHA" ] || refuse "producer authentication receipt names no package_sha256: $AUTH_RECEIPT"
# The SAME selected interpreter and the SAME explicit source reference travel,
# unchanged, to the actual exporter process; it re-authenticates in-process.
export TESSERA_PRODUCER_PYTHON="$PRODUCER_PY"
export TESSERA_PRODUCER_SOURCE="$PRODUCER_SOURCE"

CODE=$(git rev-parse HEAD 2>/dev/null || echo unknown)
echo "[export_census_stub] $S host=$(hostname) start=$(date -u +%FT%TZ) code=$CODE producer=$PRODUCER_PY producer_package_sha256=$PRODUCER_SHA source=$PRODUCER_SOURCE bound=${BOUND}s gpu=$(nvidia-smi --query-gpu=name,power.draw --format=csv,noheader 2>/dev/null)" | tee "$LOG"
timeout "$BOUND" env TMPDIR=$R/tmp "$PRODUCER_PY" -m tessera.export_serving "$SRCR/source-l8" "$OUT" \
  --plan-json "$R/plan-$S.json" --device cuda --producer-authority "$AUTH" \
  --hessian "$U/hessian_capture.references.json" \
  ${INPUT_SCALES:+--input-scales "$INPUT_SCALES"} \
  --cached-expert-units "$ROUTED_CACHE/cached_units.u1-stub-B.routed.v1.json" --cached-hessian-identity committed \
  --cached-producer-package "$PRODUCER" \
  --cached-producer-source-sha256 a4c9209437c7601d4f8cd3ab8ac1e7a2d2db33461a4245e0fbbbdf74a9d8de83 \
  --cached-intake-threads 4 --source-digest-cache "$R/source-digests" --allow-unserveable >> "$LOG" 2>&1
rc=$?
echo "[export_census_stub] $S rc=$rc end=$(date -u +%FT%TZ)" | tee -a "$LOG"
exit $rc
