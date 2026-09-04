#!/usr/bin/env bash
# Publish #113's box-local source inputs into one immutable shared namespace.
set -euo pipefail

FINAL=${TS113_STAGE_ROOT:-/mnt/shared/tessera-runs/ts113-fresh-sparklina-aa6-r1}
PARTIAL=${FINAL%/*}/.${FINAL##*/}.staging
BF16_SOURCE=${TS113_BF16_SOURCE:-/home/rob/models/Qwen3-0.6B}
ARMS_SOURCE=${TS113_ARMS_SOURCE:-/home/rob/tessera-runs/ts83}

EXPECTED_HOST=sparky
EXPECTED_UNIQUE_BYTES=2398074295
EXPECTED_LOGICAL_BYTES=3244800413
EXPECTED_BF16_BYTES=1519210493
EXPECTED_ARM_BYTES=862794960
EXPECTED_ARM_NONWEIGHT_BYTES=16068842
EXPECTED_WEIGHT_BYTES=846726118
EXPECTED_BF16_WEIGHT_SHA=f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
EXPECTED_BF16_CONFIG_SHA=660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd
EXPECTED_ARM_WEIGHT_SHA=ff17a8c64a2d95d23f44b8cc14585b8e942d1b19531a9a419233f52aa904c6ad
EXPECTED_ARM_CONFIG_SHA=e529fa91cd929d0c77f5e93d3ca2840ffcd1ba649f4af52e71f3bf47c573356b

sum_files() {
  find "$1" -type f -printf '%s\n' | awk '{total += $1} END {printf "%.0f\n", total}'
}

sum_unique_inodes() {
  find "$1" -type f -printf '%D:%i %s\n' |
    awk '!seen[$1]++ {total += $2} END {printf "%.0f\n", total}'
}

require_sha() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  [ "$actual" = "$expected" ] || {
    echo "REFUSED: $path sha256=$actual, expected $expected" >&2
    return 2
  }
}

verify_final() {
  local root=$1 logical unique ino_a ino_b
  [ -f "$root/STAGE_COMPLETE" ] || {
    echo "REFUSED: completed namespace lacks STAGE_COMPLETE: $root" >&2
    return 2
  }
  (
    cd "$root"
    sha256sum --check --strict SHA256SUMS
  )
  require_sha "$EXPECTED_BF16_WEIGHT_SHA" "$root/inputs/bf16/model.safetensors"
  require_sha "$EXPECTED_BF16_CONFIG_SHA" "$root/inputs/bf16/config.json"
  require_sha "$EXPECTED_ARM_WEIGHT_SHA" "$root/inputs/armA/model.safetensors"
  require_sha "$EXPECTED_ARM_WEIGHT_SHA" "$root/inputs/armB/model.safetensors"
  require_sha "$EXPECTED_ARM_CONFIG_SHA" "$root/inputs/armA/config.json"
  require_sha "$EXPECTED_ARM_CONFIG_SHA" "$root/inputs/armB/config.json"
  [ "$(sum_files "$root/inputs/bf16")" = "$EXPECTED_BF16_BYTES" ]
  [ "$(sum_files "$root/inputs/armA")" = "$EXPECTED_ARM_BYTES" ]
  [ "$(sum_files "$root/inputs/armB")" = "$EXPECTED_ARM_BYTES" ]
  logical=$(sum_files "$root/inputs")
  unique=$(sum_unique_inodes "$root/inputs")
  [ "$logical" = "$EXPECTED_LOGICAL_BYTES" ] || {
    echo "REFUSED: logical bytes=$logical, expected $EXPECTED_LOGICAL_BYTES" >&2
    return 2
  }
  [ "$unique" = "$EXPECTED_UNIQUE_BYTES" ] || {
    echo "REFUSED: unique bytes=$unique, expected $EXPECTED_UNIQUE_BYTES" >&2
    return 2
  }
  ino_a=$(stat -c '%D:%i' "$root/inputs/armA/model.safetensors")
  ino_b=$(stat -c '%D:%i' "$root/inputs/armB/model.safetensors")
  [ "$ino_a" = "$ino_b" ] || {
    echo "REFUSED: staged A/B weights are not one inode ($ino_a != $ino_b)" >&2
    return 2
  }
  printf 'TS113_STAGE_OK root=%s logical_bytes=%s unique_bytes=%s shared_weight_inode=%s\n' \
    "$root" "$logical" "$unique" "$ino_a"
}

[ "$(hostname)" = "$EXPECTED_HOST" ] || {
  echo "REFUSED: staging requires $EXPECTED_HOST, got $(hostname)" >&2
  exit 2
}

if [ -e "$FINAL" ]; then
  [ ! -e "$PARTIAL" ] || {
    echo "REFUSED: both final and partial namespaces exist" >&2
    exit 2
  }
  verify_final "$FINAL"
  exit 0
fi
[ ! -e "$PARTIAL" ] || {
  echo "REFUSED: preserved partial staging namespace exists: $PARTIAL" >&2
  exit 2
}

[ "$(sum_files "$BF16_SOURCE")" = "$EXPECTED_BF16_BYTES" ]
[ "$(sum_files "$ARMS_SOURCE/armA")" = "$EXPECTED_ARM_BYTES" ]
[ "$(sum_files "$ARMS_SOURCE/armB")" = "$EXPECTED_ARM_BYTES" ]
require_sha "$EXPECTED_BF16_WEIGHT_SHA" "$BF16_SOURCE/model.safetensors"
require_sha "$EXPECTED_BF16_CONFIG_SHA" "$BF16_SOURCE/config.json"
require_sha "$EXPECTED_ARM_WEIGHT_SHA" "$ARMS_SOURCE/armA/model.safetensors"
require_sha "$EXPECTED_ARM_WEIGHT_SHA" "$ARMS_SOURCE/armB/model.safetensors"
require_sha "$EXPECTED_ARM_CONFIG_SHA" "$ARMS_SOURCE/armA/config.json"
require_sha "$EXPECTED_ARM_CONFIG_SHA" "$ARMS_SOURCE/armB/config.json"
[ "$(stat -c '%D:%i' "$ARMS_SOURCE/armA/model.safetensors")" = \
  "$(stat -c '%D:%i' "$ARMS_SOURCE/armB/model.safetensors")" ] || {
  echo "REFUSED: source A/B weights are not one inode" >&2
  exit 2
}

mkdir -p "$PARTIAL/inputs/bf16" "$PARTIAL/inputs/armA" "$PARTIAL/inputs/armB"
cp -a "$BF16_SOURCE/." "$PARTIAL/inputs/bf16/"
cp -a "$ARMS_SOURCE/armA/." "$PARTIAL/inputs/armA/"
find "$ARMS_SOURCE/armB" -mindepth 1 -maxdepth 1 ! -name model.safetensors \
  -exec cp -a -t "$PARTIAL/inputs/armB" -- {} +
ln "$PARTIAL/inputs/armA/model.safetensors" \
  "$PARTIAL/inputs/armB/model.safetensors"

(
  cd "$PARTIAL"
  find inputs -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
{
  printf 'contract=ts113-input-stage-v1\n'
  printf 'source_host=%s\n' "$(hostname)"
  printf 'bf16_hf_revision=%s\n' c1899de289a04d12100db370d81485cdf75e47ca
  printf 'bf16_weight_sha256=%s\n' "$EXPECTED_BF16_WEIGHT_SHA"
  printf 'arm_weight_sha256=%s\n' "$EXPECTED_ARM_WEIGHT_SHA"
  printf 'logical_bytes=%s\n' "$EXPECTED_LOGICAL_BYTES"
  printf 'unique_bytes=%s\n' "$EXPECTED_UNIQUE_BYTES"
} > "$PARTIAL/STAGE_COMPLETE"

verify_final "$PARTIAL"
mv "$PARTIAL" "$FINAL"
verify_final "$FINAL"
