#!/usr/bin/env bash
# Run experiments/nvfp4_moe_oracle_probe.py inside the pinned serving build.
#
# This is NOT a serve: no model is loaded and no port is bound.  It still takes
# the serve lock, because it creates a CUDA context and a serve starting beside
# it would see less free memory than it asked for.  The wait is bounded, so a
# probe never sits on a worker slot for an hour the way a serve legitimately can.
#
# usage: experiments/nvfp4_moe_oracle_probe.sh [image] [out.json]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# Mia's GLM image is a different runtime and carries no pin, so this resolves
# and REPORTS its digest rather than refusing it; an image on the pinned
# repository is refused unless it is the pin.  See runtime_image.sh.
IMAGE=${1:-prismaquant/glm53-mia-sm121:487ecf187}
OUT=${2:-$HERE/results/nvfp4_moe_oracle_probe.json}
export TMPDIR=${TMPDIR:-/home/rob/tmp}

source "$HERE/runtime_image.sh"
# Before the lock, for the same reason gridbook_lane_served.sh gates there.
runtime_image_require "$IMAGE" || exit 2
source "$HERE/serve_lock.sh"
SERVE_LOCK_OWNER="nvfp4_moe_oracle_probe"
waited=0
until mkdir "$SERVE_LOCK" 2>/dev/null; do
  waited=$((waited + 10)); sleep 10
  if [ "$waited" -ge 300 ]; then
    echo "serve lock busy after ${waited}s ($(cat "$SERVE_LOCK/owner" 2>/dev/null)); not probing" >&2
    exit 3
  fi
done
echo "$$ $(date -u +%FT%TZ) $SERVE_LOCK_OWNER" > "$SERVE_LOCK/owner"
# Ownership-checked release, same rule as serve_lock_release: never remove a
# lock this process did not write.
trap 'if [ "$(awk "NR==1{print \$1}" "$SERVE_LOCK/owner" 2>/dev/null)" = "$$" ]; then
        rm -f "$SERVE_LOCK/owner"; rmdir "$SERVE_LOCK" 2>/dev/null; fi' EXIT

mkdir -p "$(dirname "$OUT")"
LOG=$(mktemp "$TMPDIR/nvfp4_moe_oracle_probe.XXXXXX.log")
# experiments/ is mounted read-only and the probe opens only its own file; the
# result is written out here from the container's stdout, never from inside it.
docker run --rm --gpus all --ipc=host \
  -v "$HERE":/probe:ro -e TMPDIR=/tmp \
  --entrypoint python3 "$IMAGE" /probe/nvfp4_moe_oracle_probe.py 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
# The probe prints the record twice: indented for a reader, then one compact
# tagged line.  Take the tagged line and re-indent it here, so a result file is
# either the whole record or empty -- never the first line of one.
sed -n 's/^TESSERA_ORACLE_JSON //p' "$LOG" | tail -1 | python3 -m json.tool > "$OUT" || true
[ -s "$OUT" ] || { echo "probe produced no JSON (rc=$rc); see $LOG" >&2; exit 1; }
echo "-> $OUT   (image $IMAGE -> ${RUNTIME_IMAGE_DIGEST:-unresolved})"
exit "$rc"
