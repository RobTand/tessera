#!/usr/bin/env bash
# Fresh same-host compiled teacher/lane/fallback population for Tessera #113.
set -euo pipefail

WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
STAGE_ROOT=${TS113_STAGE_ROOT:-/mnt/shared/tessera-runs/ts113-fresh-sparklina-aa6-r1}
PROMOTIONAL_POP_ROOT=/mnt/shared/tessera-runs/ts113-sparklina-population-aa6-r6
PROMOTIONAL_LOCAL_ROOT=/home/rob/tessera-runs/ts113-sparklina-aa6-r6
POP_ROOT=${TS113_POP_ROOT:-$PROMOTIONAL_POP_ROOT}
LOCAL_ROOT=${TS113_LOCAL_ROOT:-$PROMOTIONAL_LOCAL_ROOT}
source "$WT/experiments/runtime_image.sh"
IMAGE=$(runtime_image_pin)
CORPUS=/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json
BF16=$STAGE_ROOT/inputs/bf16
ARMA=$STAGE_ROOT/inputs/armA
ARMB=$STAGE_ROOT/inputs/armB
PREFIX=qwen_ts113_sparklina_aa6_r6
MODE=${1:-campaign}
DECODE_STRIDE=16
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
KL=${KL:-/home/rob/dq-runs/kl_tool.py}
EXT_LANE=$LOCAL_ROOT/ext-lane
EXT_FALLBACK=$LOCAL_ROOT/ext-fallback-readonly

[ "$MODE" = campaign ] || [ "$MODE" = preflight-stage ] || {
  echo "usage: $0 [campaign|preflight-stage]" >&2
  exit 64
}

[ "$(hostname)" = gx10-6b77 ] || {
  echo "REFUSED: #113 campaign requires gx10-6b77, got $(hostname)" >&2
  exit 2
}
[ -f "$STAGE_ROOT/STAGE_COMPLETE" ] || {
  echo "REFUSED: immutable input stage is incomplete" >&2
  exit 2
}
[ ! -e "${STAGE_ROOT%/*}/.${STAGE_ROOT##*/}.staging" ] || {
  echo "REFUSED: immutable input stage still has a partial namespace" >&2
  exit 2
}
(
  cd "$STAGE_ROOT"
  sha256sum --check --strict --quiet SHA256SUMS
)
[ "$(sha256sum "$BF16/model.safetensors" | cut -d' ' -f1)" = \
  f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b ]
[ "$(sha256sum "$ARMA/model.safetensors" | cut -d' ' -f1)" = \
  ff17a8c64a2d95d23f44b8cc14585b8e942d1b19531a9a419233f52aa904c6ad ]
[ "$(stat -c '%D:%i' "$ARMA/model.safetensors")" = \
  "$(stat -c '%D:%i' "$ARMB/model.safetensors")" ] || {
  echo "REFUSED: staged A/B weights are no longer one inode" >&2
  exit 2
}

sample_power() {
  printf 'utc=%s host=%s load=%s power=%s mem_available_kib=%s\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$(cut -d' ' -f1-3 /proc/loadavg)" \
    "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null | tr '\n' ' ')" \
    "$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
}

verify_stage() {
  local dir=$1
  [ -f "$dir/STAGE_COMPLETE" ] && [ -f "$dir/STAGE_SHA256" ] || return 1
  (cd "$dir" && sha256sum --check --strict STAGE_SHA256)
}

reap_partial_container() {
  local name=$1 dir=$2
  if docker ps -aq -f "name=^${name}$" | grep -q .; then
    docker inspect "$name" > "$dir/retry-container-inspect.json" 2>&1 || true
    docker logs "$name" > "$dir/retry-container.log" 2>&1 || true
    docker rm -f "$name" >/dev/null 2>&1 || true
  fi
}

prepare_stage() {
  local stage=$1 name=$2
  local dir=$POP_ROOT/stages/$stage
  if [ -e "$dir" ]; then
    if verify_stage "$dir"; then
      echo "=== $stage already complete and verified"
      return 1
    fi
    reap_partial_container "$name" "$dir"
    echo "REFUSED: preserved partial stage exists: $dir" >&2
    exit 2
  fi
  mkdir -p "$dir"
  sample_power > "$dir/power-before.txt"
  return 0
}

seal_stage() {
  local stage=$1
  local dir=$POP_ROOT/stages/$stage
  local tmp=$POP_ROOT/stages/.$stage.sha256
  sample_power > "$dir/power-after.txt"
  (
    cd "$dir"
    find . -type f ! -name STAGE_SHA256 ! -name STAGE_COMPLETE -print0 |
      sort -z | xargs -0 sha256sum > "$tmp"
  )
  mv "$tmp" "$dir/STAGE_SHA256"
  printf 'stage=%s\nhost=%s\ncompleted_at_utc=%s\n' \
    "$stage" "$(hostname)" "$(date -u +%FT%TZ)" > "$dir/STAGE_COMPLETE"
  verify_stage "$dir"
}

validate_build() {
  local path=$1 artifact=$2
  "$PY" - "$path" "$artifact" "$IMAGE" <<'PY'
import json
import sys

path, artifact, image = sys.argv[1:]
record = json.load(open(path, encoding="utf-8"))
identity = record.get("identity", {})
provenance = record.get("provenance", {})
assert record.get("complete") is True, record
assert identity.get("compiled_forward") is True, record
assert identity.get("eager") is False, record
assert identity.get("vllm_version") == "0.28.0", record
assert identity.get("image") == image, record
assert identity.get("image_digest") == image.split("@", 1)[1], record
assert identity.get("dispatch") is not None, record
assert provenance.get("artifact_path") == artifact, record
assert provenance.get("fresh_compiles", 0) > 0, record
print("TS113_BUILD_OK", path, record["build_fingerprint"])
PY
}

freeze_cache_choice() {
  local cache=$1 out=$2
  local list=$out.list
  (
    cd "$cache"
    find . -type f \( -name '*.best_config' -o -name cache_key_factors.json \
      -o -name computation_graph.py \) -print0 | sort -z > "$list"
    [ -s "$list" ] || { echo "REFUSED: no compile-cache evidence in $cache" >&2; exit 2; }
    tar --null --files-from "$list" -cf "$out"
  )
  rm "$list"
}

shopt -s nullglob
closure_stamps=("$WT"/.pbrun-closure.*.json)
[ "${#closure_stamps[@]}" = 1 ] || {
  echo "REFUSED: expected one verified pbrun closure stamp, found ${#closure_stamps[@]}" >&2
  exit 2
}
closure_stamp=${closure_stamps[0]}
snapshot_commit=$(git -C "$WT" rev-parse HEAD)
[ -z "$(git -C "$WT" status --porcelain)" ] || {
  echo "REFUSED: materialized PrismaBuild snapshot is dirty" >&2
  exit 2
}
if [ "$MODE" = preflight-stage ]; then
  [ "${TS113_POP_ROOT+x}" = x ] && [ "${TS113_LOCAL_ROOT+x}" = x ] || {
    echo "REFUSED: preflight-stage requires explicit disposable roots" >&2
    exit 2
  }
  [ "$POP_ROOT" != "$PROMOTIONAL_POP_ROOT" ] && \
    [ "$LOCAL_ROOT" != "$PROMOTIONAL_LOCAL_ROOT" ] || {
    echo "REFUSED: preflight-stage may not seed the promotional namespaces" >&2
    exit 2
  }
fi

pop_existed=0
[ -e "$POP_ROOT" ] && pop_existed=1
if [ "$pop_existed" = 0 ] && [ -e "$LOCAL_ROOT" ]; then
  echo "REFUSED: fresh shared population has a pre-existing host-local namespace" >&2
  exit 2
fi
mkdir -p "$POP_ROOT/stages" "$LOCAL_ROOT" "$EXT_LANE" "$EXT_FALLBACK"
chmod a-w "$EXT_FALLBACK"
identity_path=$POP_ROOT/CAMPAIGN_IDENTITY.json
if [ "$pop_existed" = 1 ] && [ ! -f "$identity_path" ]; then
  echo "REFUSED: existing population lacks CAMPAIGN_IDENTITY.json" >&2
  exit 2
fi
read -r DECODE_POSITIONS ELIGIBLE_MODULES ELIGIBLE_UNITS EXPECTED_LAUNCHES < <(
  PYTHONPATH="$WT/src" "$PY" - \
    "$closure_stamp" "$snapshot_commit" "$IMAGE" \
    "$ARMA/tessera_gridbook_manifest.json" \
    "$WT/src/tessera/serving/runtime_contract.json" "$CORPUS" \
    "$identity_path" "$DECODE_STRIDE" \
    "$(stat -c '%D:%i' "$ARMA/model.safetensors")" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import sys

(closure_path, snapshot_commit, image, manifest_path, contract_path,
 corpus_path, identity_path, stride_raw, weight_inode) = sys.argv[1:]
closure_bytes = Path(closure_path).read_bytes()
closure = json.loads(closure_bytes)
if set(closure) != {"cwd", "dirty_sha256", "head"}:
    raise SystemExit(f"REFUSED: malformed pbrun closure identity: {closure}")
if closure["cwd"] != ".":
    raise SystemExit(
        f"REFUSED: closure logical cwd is {closure['cwd']!r}, expected '.'"
    )
if not re.fullmatch(r"[0-9a-f]{40}", closure["head"]):
    raise SystemExit("REFUSED: closure source head is not a commit id")
empty_dirty = hashlib.sha256(b"").hexdigest()
if closure["dirty_sha256"] != empty_dirty:
    raise SystemExit(
        f"REFUSED: campaign source was dirty: {closure['dirty_sha256']}"
    )
if not re.fullmatch(r"[0-9a-f]{40}", snapshot_commit):
    raise SystemExit("REFUSED: snapshot commit is not a commit id")

manifest = json.load(open(manifest_path, encoding="utf-8"))
contract = json.load(open(contract_path, encoding="utf-8"))
corpus = json.load(open(corpus_path, encoding="utf-8"))
formats = {row["grid"]: row for row in contract["formats"]}
eligible_pairs = {
    (cell["family"], int(q256))
    for cell in contract["lane_eligibility"]["cells"]
    if cell["platform"] == "sm_121"
    and cell["structure"] == "dense"
    and cell["regime"] == "decode"
    and "TESSERA_SERVE_MODE=streamed" in cell["requires_serve_flags"]
    and any(
        launch == {
            "symbol": "tessera_window_gemv::gemv",
            "decoder": "window_gemv",
        }
        for launch in cell["executes"]
    )
    for q256 in cell["rungs_q256"]
}
eligible = []
for name, module in manifest["modules"].items():
    fmt = formats.get(module["grid"])
    if fmt is not None and (fmt["family"], int(module["q256"])) in eligible_pairs:
        eligible.append(name)
module_total = int(manifest["totals"]["modules"])
if module_total != len(manifest["modules"]):
    raise SystemExit("REFUSED: manifest module total disagrees with its roster")
if len(eligible) != module_total:
    raise SystemExit(
        f"REFUSED: only {len(eligible)}/{module_total} staged modules are lane-eligible"
    )
unit_total = int(manifest["totals"]["units"])
roster_units = sum(len(module["roles"]) for module in manifest["modules"].values())
if unit_total != roster_units:
    raise SystemExit("REFUSED: manifest unit total disagrees with its role roster")
eligible_units = sum(len(manifest["modules"][name]["roles"]) for name in eligible)
if eligible_units != unit_total:
    raise SystemExit(
        f"REFUSED: only {eligible_units}/{unit_total} staged units are lane-eligible"
    )

stride = int(stride_raw)
if len(corpus["chunks"]) != int(corpus["n_chunks"]):
    raise SystemExit("REFUSED: corpus chunk roster disagrees with n_chunks")
if any(len(chunk) != int(corpus["seqlen"]) for chunk in corpus["chunks"]):
    raise SystemExit("REFUSED: corpus chunk length disagrees with seqlen")
decode_per_chunk = len(range(1, int(corpus["seqlen"]), stride))
decode_positions = int(corpus["n_chunks"]) * decode_per_chunk
eligible_modules = len(eligible)
expected_launches = decode_positions * eligible_units
expected = {
    "schema": "tessera.ts113.sparklina-population-identity.v3",
    "host": os.uname().nodename,
    "image": image,
    "checkout": {
        "snapshot_commit": snapshot_commit,
        "closure_stamp": Path(closure_path).name,
        "closure_stamp_sha256": hashlib.sha256(closure_bytes).hexdigest(),
        "source_cwd": closure["cwd"],
        "original_head": closure["head"],
        "dirty_sha256": closure["dirty_sha256"],
    },
    "inputs": {
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest(),
        "corpus_contract_sha256": corpus["contract_sha256"],
        "weight_inode": weight_inode,
        "weight_sha256": "ff17a8c64a2d95d23f44b8cc14585b8e942d1b19531a9a419233f52aa904c6ad",
    },
    "derived_gate": {
        "decode_stride": stride,
        "chunks": int(corpus["n_chunks"]),
        "positions_per_chunk": decode_per_chunk,
        "decode_positions": decode_positions,
        "eligible_modules": eligible_modules,
        "eligible_units": eligible_units,
        "expected_window_gemv_launches": expected_launches,
        "fallback_refusals": eligible_modules,
    },
}
path = Path(identity_path)
if path.exists():
    observed = json.loads(path.read_text())
    if observed != expected:
        raise SystemExit("REFUSED: existing campaign identity does not match this retry")
    print("validated existing campaign identity", file=sys.stderr)
else:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print("published campaign identity", file=sys.stderr)
print(decode_positions, eligible_modules, eligible_units, expected_launches)
PY
)
[ -n "$DECODE_POSITIONS" ] && [ -n "$ELIGIBLE_MODULES" ] && \
  [ -n "$ELIGIBLE_UNITS" ] && [ -n "$EXPECTED_LAUNCHES" ]
echo "TS113_DERIVED_GATE positions=$DECODE_POSITIONS modules=$ELIGIBLE_MODULES units=$ELIGIBLE_UNITS launches=$EXPECTED_LAUNCHES"

if [ "$MODE" = preflight-stage ]; then
  if prepare_stage path-expansion ts113-aa6-r6-preflight; then
    printf 'stage_path=%s\n' "$POP_ROOT/stages/path-expansion" \
      > "$POP_ROOT/stages/path-expansion/no-docker-proof.txt"
    seal_stage path-expansion
  fi
  verify_stage "$POP_ROOT/stages/path-expansion"
  echo "TS113_STAGE_EXPANSION_PREFLIGHT_OK $POP_ROOT"
  exit 0
fi

if [ -f "$POP_ROOT/CAMPAIGN_COMPLETE" ]; then
  for stage in teacher A1 A2 B1 B2; do verify_stage "$POP_ROOT/stages/$stage"; done
  echo "TS113_CAMPAIGN_ALREADY_COMPLETE $POP_ROOT"
  exit 0
fi

TEACHER_DIR=$POP_ROOT/stages/teacher
TEACHER=$TEACHER_DIR/teacher_decode.json
if prepare_stage teacher ts113-aa6-r6-teacher; then
  if ! env \
      TESSERA_KL_IMAGE="$IMAGE" TESSERA_KL_EAGER=0 TESSERA_KL_REGIME=decode \
      TESSERA_KL_NAME=ts113-aa6-r6-teacher TESSERA_KL_LOGDIR="$TEACHER_DIR" \
      TESSERA_KL_VLLM_CACHE="$LOCAL_ROOT/cache-teacher" \
      TESSERA_KL_CORPUS="$CORPUS" TESSERA_GPU_MEM_UTIL=0.45 \
      TESSERA_KL_REQUIRE_IN_LOG='enforce_eager=False' \
      "$WT/experiments/serve_and_dump_kl.sh" "$BF16" "$TEACHER" teacher BF16 \
      2>&1 | tee "$TEACHER_DIR/driver.log"; then
    echo "teacher stage failed; evidence preserved at $TEACHER_DIR" >&2
    exit 3
  fi
  validate_build "$TEACHER_DIR/teacher_decode.build.json" "$BF16"
  freeze_cache_choice "$LOCAL_ROOT/cache-teacher" "$TEACHER_DIR/compile-cache-evidence.tar"
  seal_stage teacher
fi

run_arm() {
  local stage=$1 model=$2 lane=$3 profile=$4
  local dir=$POP_ROOT/stages/$stage name=ts113-aa6-r6-$stage
  local cache=$LOCAL_ROOT/cache-$stage extra= profile_dir=
  local dump_decode=$dir/${PREFIX}_${stage}_decode.json
  if ! prepare_stage "$stage" "$name"; then return 0; fi
  if [ "$lane" = fallback ]; then
    extra="-v $EXT_FALLBACK:/ext-ro:ro -e TORCH_EXTENSIONS_DIR=/ext-ro"
  fi
  [ "$profile" = yes ] && profile_dir=$dir/profile
  if ! env \
      WT="$WT" TS="$WT" RUNS="$dir" EXT="$EXT_LANE" TRACEDIR="$dir/route" \
      VLLM_CACHE="$cache" IMAGE="$IMAGE" TESSERA_KL_DIR="$dir" \
      TESSERA_KL_DUMP_PREFIX="$PREFIX" TESSERA_KL_CORPUS="$CORPUS" \
      TESSERA_KL_DECODE_STRIDE="$DECODE_STRIDE" \
      TESSERA_LANE_EAGER=0 TESSERA_KL_NAME="$name" TESSERA_GPU_MEM_UTIL=0.45 \
      TEACHER_DECODE="$TEACHER" TEACHER_PREFILL="$POP_ROOT/no-prefill-teacher" \
      TESSERA_LANE_DOCKER_EXTRA="$extra" TESSERA_KL_PROFILE_DIR="$profile_dir" \
      "$WT/experiments/decode_regime_kl.sh" "$model" "$stage" streamed \
      2>&1 | tee "$dir/driver.log"; then
    echo "$stage failed; evidence preserved at $dir" >&2
    exit 3
  fi
  validate_build "${dump_decode%.json}.build.json" "$model"
  refusals=$(grep -c 'the window GEMV lane did not prepare' "$dir/serve_$stage.log" || true)
  if [ "$lane" = fallback ]; then expected=$ELIGIBLE_MODULES; else expected=0; fi
  [ "$refusals" = "$expected" ] || {
    echo "REFUSED: $stage lane refusals=$refusals, expected $expected" >&2
    exit 2
  }
  if [ "$profile" = yes ]; then
    if [ "$lane" = lane ]; then
      expected_profile_launches=$EXPECTED_LAUNCHES
    else
      expected_profile_launches=0
    fi
    "$PY" - "$dir/profile-$stage-summary.json" \
      "$expected_profile_launches" "$DECODE_POSITIONS" "$ELIGIBLE_UNITS" \
      "$ELIGIBLE_MODULES" "$lane" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
launches = sum(
    row.get("by_bucket", {}).get("window_gemv", {}).get("launches", 0)
    for row in rows
)
expected, positions, units, modules = map(int, sys.argv[2:6])
lane = sys.argv[6]
assert launches == expected, (launches, expected)
if lane == "lane":
    assert expected == positions * units
else:
    assert lane == "fallback" and expected == 0
print(
    f"TS113_LAUNCH_OK window_gemv={launches} "
    f"lane={lane} positions={positions} eligible_modules={modules} "
    f"eligible_units={units}"
)
PY
  fi
  freeze_cache_choice "$cache" "$dir/compile-cache-evidence.tar"
  (
    cd "$EXT_LANE"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
  ) > "$dir/extension-files.sha256"
  [ -s "$dir/extension-files.sha256" ] || {
    echo "REFUSED: lane extension evidence is empty" >&2
    exit 2
  }
  seal_stage "$stage"
}

run_arm A1 "$ARMA" lane yes
run_arm A2 "$ARMA" lane yes
run_arm B1 "$ARMB" fallback yes
run_arm B2 "$ARMB" fallback yes

for stage in teacher A1 A2 B1 B2; do
  verify_stage "$POP_ROOT/stages/$stage"
done
{
  printf 'schema=tessera.ts113.sparklina-population-complete.v1\n'
  printf 'host=%s\ncompleted_at_utc=%s\n' "$(hostname)" "$(date -u +%FT%TZ)"
  for stage in teacher A1 A2 B1 B2; do
    sha256sum "$POP_ROOT/stages/$stage/STAGE_SHA256"
  done
} > "$POP_ROOT/CAMPAIGN_COMPLETE"
echo "TS113_CAMPAIGN_OK $POP_ROOT"
