#!/usr/bin/env bash
# Run experiments/moe_wire_loader_probe.py inside the pinned serving build.
#
# NO GPU and NO serve lock, unlike nvfp4_moe_oracle_probe.sh: this probe imports
# ``routed_experts`` and calls two functions, so it creates no CUDA context and
# takes nothing a serve would want.  Asking for the lock here would only make a
# real serve wait for a question that does not touch the device.
#
# usage: experiments/moe_wire_loader_probe.sh [image] [out.json]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE=${1:-prismaquant/glm53-mia-sm121:487ecf187}
OUT=${2:-$HERE/results/moe_wire_loader_probe.json}
export TMPDIR=${TMPDIR:-/home/rob/tmp}

mkdir -p "$(dirname "$OUT")"
LOG=$(mktemp "$TMPDIR/moe_wire_loader_probe.XXXXXX.log")
docker run --rm --ipc=host \
  -v "$HERE":/probe:ro -e TMPDIR=/tmp \
  --entrypoint python3 "$IMAGE" /probe/moe_wire_loader_probe.py 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
sed -n 's/^TESSERA_MOE_LOADER_JSON //p' "$LOG" | tail -1 | python3 -m json.tool > "$OUT" || true
[ -s "$OUT" ] || { echo "probe produced no JSON (rc=$rc); see $LOG" >&2; exit 1; }
echo "-> $OUT"
exit "$rc"
