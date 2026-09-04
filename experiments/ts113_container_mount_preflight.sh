#!/usr/bin/env bash
# Prove the #113 container can atomically publish into its trace/profile binds.
set -euo pipefail

WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
ROOT=${TS113_MOUNT_PREFLIGHT_ROOT:?set TS113_MOUNT_PREFLIGHT_ROOT to a fresh disposable path}

[ "$(hostname)" = gx10-6b77 ] || {
  echo "REFUSED: container-mount preflight requires gx10-6b77" >&2
  exit 2
}
[ "${CUDA_VISIBLE_DEVICES+x}" = x ] && [ -z "$CUDA_VISIBLE_DEVICES" ] || {
  echo "REFUSED: container-mount preflight must run in a GPU-hidden CPU slot" >&2
  exit 2
}

verify_complete() {
  [ -f "$ROOT/PREFLIGHT_COMPLETE" ] && [ -f "$ROOT/SHA256SUMS" ] || return 1
  (
    cd "$ROOT"
    sha256sum --check --strict SHA256SUMS
  )
  [ "$(cat "$ROOT/trace/route-trace.json")" = route-atomic-rename-ok ]
  [ "$(cat "$ROOT/prof/trace.json")" = profile-atomic-rename-ok ]
  [ ! -e "$ROOT/trace/.route-trace.json.tmp" ]
  [ ! -e "$ROOT/prof/.trace.json.tmp" ]
}

if [ -e "$ROOT" ]; then
  if verify_complete; then
    echo "TS113_CONTAINER_MOUNT_PREFLIGHT_ALREADY_OK $ROOT"
    exit 0
  fi
  echo "REFUSED: preserved partial container-mount preflight exists: $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT/trace" "$ROOT/prof"
# The exact condition under test: /mnt/shared root-squashes the container's
# root identity, so host ownership is insufficient for a bind-mounted writer.
chmod a+rwx "$ROOT/trace" "$ROOT/prof"

source "$WT/experiments/runtime_image.sh"
IMAGE=$(runtime_image_pin)
runtime_image_require "$IMAGE"
printf '%s\n' "$RUNTIME_IMAGE_JSON" > "$ROOT/runtime-image.json"
printf 'host=%s\nimage=%s\ncuda_visible_devices=%q\n' \
  "$(hostname)" "$IMAGE" "$CUDA_VISIBLE_DEVICES" > "$ROOT/identity.txt"

# Deliberately no --gpus: this is a filesystem/identity gate in a CPU slot.
docker run --rm \
  -v "$ROOT/trace:/trace" -v "$ROOT/prof:/prof" \
  --entrypoint bash "$IMAGE" -ceu '
    printf %s route-atomic-rename-ok > /trace/.route-trace.json.tmp
    mv /trace/.route-trace.json.tmp /trace/route-trace.json
    printf %s profile-atomic-rename-ok > /prof/.trace.json.tmp
    mv /prof/.trace.json.tmp /prof/trace.json
  '

verify_complete_payload() {
  [ "$(cat "$ROOT/trace/route-trace.json")" = route-atomic-rename-ok ]
  [ "$(cat "$ROOT/prof/trace.json")" = profile-atomic-rename-ok ]
  [ ! -e "$ROOT/trace/.route-trace.json.tmp" ]
  [ ! -e "$ROOT/prof/.trace.json.tmp" ]
}
verify_complete_payload
(
  cd "$ROOT"
  sha256sum identity.txt runtime-image.json \
    trace/route-trace.json prof/trace.json > .SHA256SUMS.tmp
  mv .SHA256SUMS.tmp SHA256SUMS
)
printf 'schema=tessera.ts113.container-mount-preflight.v1\nhost=%s\n' \
  "$(hostname)" > "$ROOT/.PREFLIGHT_COMPLETE.tmp"
mv "$ROOT/.PREFLIGHT_COMPLETE.tmp" "$ROOT/PREFLIGHT_COMPLETE"
verify_complete
echo "TS113_CONTAINER_MOUNT_PREFLIGHT_OK $ROOT"
