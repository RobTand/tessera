#!/usr/bin/env bash
# Refuse #113 unless the staged population and exact serving image are present.
set -euo pipefail

WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
STAGE_ROOT=${TS113_STAGE_ROOT:-/mnt/shared/tessera-runs/ts113-fresh-sparklina-aa6-r1}
POP_ROOT=${TS113_POP_ROOT:-/mnt/shared/tessera-runs/ts113-sparklina-population-aa6-r4}
LOCAL_ROOT=${TS113_LOCAL_ROOT:-/home/rob/tessera-runs/ts113-sparklina-aa6-r4}
source "$WT/experiments/runtime_image.sh"
IMAGE=$(runtime_image_pin)
CORPUS=/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json
EXPECTED_CORPUS_SHA=cf96c4744a58e925f62673b6fc09c3bd584b5d7a49c00b901d7f0bce0ab57002
EXPECTED_UNIQUE_BYTES=2398074295
EXPECTED_LOGICAL_BYTES=3244800413
EXPECTED_BF16_WEIGHT_SHA=f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
EXPECTED_ARM_WEIGHT_SHA=ff17a8c64a2d95d23f44b8cc14585b8e942d1b19531a9a419233f52aa904c6ad

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

[ "$(hostname)" = gx10-6b77 ] || {
  echo "REFUSED: Sparklina preflight placed on $(hostname)" >&2
  exit 2
}
[ -f "$STAGE_ROOT/STAGE_COMPLETE" ] || {
  echo "REFUSED: staged-input completion marker absent" >&2
  exit 2
}
[ ! -e "${STAGE_ROOT%/*}/.${STAGE_ROOT##*/}.staging" ] || {
  echo "REFUSED: staged-input partial namespace still exists" >&2
  exit 2
}
(
  cd "$STAGE_ROOT"
  sha256sum --check --strict SHA256SUMS
)
require_sha "$EXPECTED_BF16_WEIGHT_SHA" "$STAGE_ROOT/inputs/bf16/model.safetensors"
require_sha "$EXPECTED_ARM_WEIGHT_SHA" "$STAGE_ROOT/inputs/armA/model.safetensors"
require_sha "$EXPECTED_ARM_WEIGHT_SHA" "$STAGE_ROOT/inputs/armB/model.safetensors"
[ "$(sum_files "$STAGE_ROOT/inputs")" = "$EXPECTED_LOGICAL_BYTES" ] || {
  echo "REFUSED: staged logical byte count changed" >&2
  exit 2
}
[ "$(sum_unique_inodes "$STAGE_ROOT/inputs")" = "$EXPECTED_UNIQUE_BYTES" ] || {
  echo "REFUSED: staged unique-inode byte count changed" >&2
  exit 2
}
ino_a=$(stat -c '%D:%i' "$STAGE_ROOT/inputs/armA/model.safetensors")
ino_b=$(stat -c '%D:%i' "$STAGE_ROOT/inputs/armB/model.safetensors")
[ "$ino_a" = "$ino_b" ] || {
  echo "REFUSED: staged A/B weights are not one inode" >&2
  exit 2
}
require_sha "$EXPECTED_CORPUS_SHA" "$CORPUS"
python3 - "$CORPUS" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "schema": "prismaquant.kl_corpus_contract/1",
    "source_sha256": "076d33efc4476dcc417a2bb249c0bc950bb54bbb471d73f69c15cef0010b53d0",
    "n_chunks": 8,
    "seqlen": 512,
    "tokens": 4096,
    "scored_positions": 4088,
    "contract_sha256": "cfbddc2c49078256564dffd32dc5033515ce11f30057c33f0fe457ed5aded59d",
}
for field, value in expected.items():
    if record.get(field) != value:
        raise SystemExit(
            f"REFUSED: corpus {field}={record.get(field)!r}, expected {value!r}"
        )
tokenizer = record.get("tokenizer", {})
if tokenizer.get("identity_sha256") != "76f13c8e6e553e35b09733ed5543274fdcd97285d3fcd7e1cccd4e0ad8089891":
    raise SystemExit("REFUSED: corpus tokenizer identity changed")
print("TS113_CORPUS_OK positions=4088 tokens=4096")
PY

inspect=$(docker image inspect --format '{{.Id}}|{{.Size}}|{{json .RepoDigests}}' "$IMAGE")
case "$inspect" in
  *"$IMAGE"*) ;;
  *) echo "REFUSED: exact RepoDigest absent from local image: $inspect" >&2; exit 2 ;;
esac
version=$(docker run --rm --entrypoint python3 "$IMAGE" -c \
  'import vllm; print(vllm.__version__)' | tail -n 1)
[ "$version" = 0.28.0 ] || {
  echo "REFUSED: vLLM version=$version, expected 0.28.0" >&2
  exit 2
}

[ ! -e "$POP_ROOT" ] || {
  echo "REFUSED: population output namespace already exists: $POP_ROOT" >&2
  exit 2
}
[ ! -e "$LOCAL_ROOT" ] || {
  echo "REFUSED: host-local population namespace already exists: $LOCAL_ROOT" >&2
  exit 2
}

printf 'TS113_SPARKLINA_PREFLIGHT_OK\n'
printf 'host=%s\nimage=%s\nimage_inspect=%s\nvllm=%s\n' \
  "$(hostname)" "$IMAGE" "$inspect" "$version"
printf 'stage_root=%s\nlogical_bytes=%s\nunique_bytes=%s\nshared_weight_inode=%s\n' \
  "$STAGE_ROOT" "$EXPECTED_LOGICAL_BYTES" "$EXPECTED_UNIQUE_BYTES" "$ino_a"
df -B1 --output=target,avail "${LOCAL_ROOT%/*}" "${POP_ROOT%/*}" | sort -u
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
