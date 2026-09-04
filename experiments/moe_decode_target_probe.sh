#!/usr/bin/env bash
# Run experiments/moe_decode_target_probe.py inside the pinned serving build.
#
# WITH A GPU, and therefore with the serve lock.  The probe's decisive leg calls
# the runtime's own ``is_supported_config`` for each fp8 MoE backend, and every
# one of those kernels answers "kernel does not support current device None"
# when there is no device -- controls included, which is how the CPU run showed
# that the missing GPU was the reason rather than the per-channel weight scale.
# So the table is only a table about sm_121 when it is taken on sm_121.  No
# model is loaded and no port is bound, but a CUDA context is created, so a
# serve starting beside it would see less free memory than it asked for.
#
# usage: experiments/moe_decode_target_probe.sh [image] [out.json]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE=${1:-prismaquant/glm53-mia-sm121:487ecf187}
OUT=${2:-$HERE/results/moe_decode_target_probe.json}
export TMPDIR=${TMPDIR:-/home/rob/tmp}

source "$HERE/runtime_image.sh"
runtime_image_require "$IMAGE" || exit 2
source "$HERE/serve_lock.sh"
SERVE_LOCK_OWNER="moe_decode_target_probe"
SERVE_LOCK_TIMEOUT=300
SERVE_LOCK_POLL_S=10
serve_lock_acquire || exit $?
trap serve_lock_release EXIT

mkdir -p "$(dirname "$OUT")"
LOG=$(mktemp "$TMPDIR/moe_decode_target_probe.XXXXXX.log")
docker run --rm --gpus all --ipc=host \
  -v "$HERE":/probe:ro -e TMPDIR=/tmp \
  --entrypoint python3 "$IMAGE" /probe/moe_decode_target_probe.py 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
sed -n 's/^TESSERA_MOE_DECODE_JSON //p' "$LOG" | tail -1 | python3 -m json.tool > "$OUT" || true
[ -s "$OUT" ] || { echo "probe produced no JSON (rc=$rc); see $LOG" >&2; exit 1; }
echo "-> $OUT   (image $IMAGE -> ${RUNTIME_IMAGE_DIGEST:-unresolved})"
exit "$rc"
