#!/usr/bin/env bash
# Fresh same-host compiled teacher/lane/fallback population for Tessera #113.
set -euo pipefail

WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
STAGE_ROOT=${TS113_STAGE_ROOT:-/mnt/shared/tessera-runs/ts113-fresh-sparklina-aa6-r1}
POP_ROOT=${TS113_POP_ROOT:-/mnt/shared/tessera-runs/ts113-sparklina-population-aa6-r1}
LOCAL_ROOT=${TS113_LOCAL_ROOT:-/home/rob/tessera-runs/ts113-sparklina-aa6-r1}
IMAGE=vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
CORPUS=/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json
BF16=$STAGE_ROOT/inputs/bf16
ARMA=$STAGE_ROOT/inputs/armA
ARMB=$STAGE_ROOT/inputs/armB
PREFIX=qwen_ts113_sparklina_aa6_r1
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
KL=${KL:-/home/rob/dq-runs/kl_tool.py}
EXT_LANE=$LOCAL_ROOT/ext-lane
EXT_FALLBACK=$LOCAL_ROOT/ext-fallback-readonly

[ "$(hostname)" = gx10-6b77 ] || {
  echo "REFUSED: #113 campaign requires gx10-6b77, got $(hostname)" >&2
  exit 2
}
[ -f "$STAGE_ROOT/STAGE_COMPLETE" ] || {
  echo "REFUSED: immutable input stage is incomplete" >&2
  exit 2
}
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
  printf 'utc=%s host=%s load=%s power=%s\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$(cut -d' ' -f1-3 /proc/loadavg)" \
    "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
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
  local stage=$1 name=$2 dir=$POP_ROOT/stages/$stage
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
  local stage=$1 dir=$POP_ROOT/stages/$stage tmp=$POP_ROOT/stages/.$stage.sha256
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
  local cache=$1 out=$2 list=$out.list
  (
    cd "$cache"
    find . -type f \( -name '*.best_config' -o -name cache_key_factors.json \
      -o -name computation_graph.py \) -print0 | sort -z > "$list"
    [ -s "$list" ] || { echo "REFUSED: no compile-cache evidence in $cache" >&2; exit 2; }
    tar --null --files-from "$list" -cf "$out"
  )
  rm "$list"
}

mkdir -p "$POP_ROOT/stages" "$LOCAL_ROOT" "$EXT_LANE" "$EXT_FALLBACK"
chmod a-w "$EXT_FALLBACK"
if [ ! -f "$POP_ROOT/CAMPAIGN_IDENTITY" ]; then
  {
    printf 'schema=tessera.ts113.sparklina-population.v1\n'
    printf 'host=%s\nimage=%s\n' "$(hostname)" "$IMAGE"
    printf 'tessera_commit=%s\n' "$(git -C "$WT" rev-parse HEAD)"
    printf 'staged_weight_sha256=%s\n' \
      ff17a8c64a2d95d23f44b8cc14585b8e942d1b19531a9a419233f52aa904c6ad
    printf 'corpus_contract_sha256=%s\n' \
      cfbddc2c49078256564dffd32dc5033515ce11f30057c33f0fe457ed5aded59d
  } > "$POP_ROOT/CAMPAIGN_IDENTITY"
fi

TEACHER_DIR=$POP_ROOT/stages/teacher
TEACHER=$TEACHER_DIR/teacher_decode.json
if prepare_stage teacher ts113-aa6-teacher; then
  if ! env \
      TESSERA_KL_IMAGE="$IMAGE" TESSERA_KL_EAGER=0 TESSERA_KL_REGIME=decode \
      TESSERA_KL_NAME=ts113-aa6-teacher TESSERA_KL_LOGDIR="$TEACHER_DIR" \
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
  local dir=$POP_ROOT/stages/$stage name=ts113-aa6-$stage
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
  if [ "$lane" = fallback ]; then expected=112; else expected=0; fi
  [ "$refusals" = "$expected" ] || {
    echo "REFUSED: $stage lane refusals=$refusals, expected $expected" >&2
    exit 2
  }
  if [ "$profile" = yes ]; then
    "$PY" - "$dir/profile-$stage-summary.json" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
launches = sum(
    row.get("by_bucket", {}).get("window_gemv", {}).get("launches", 0)
    for row in rows
)
expected = 256 * 112
assert launches == expected, (launches, expected)
print(f"TS113_LAUNCH_OK window_gemv={launches} derived=256x112")
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
run_arm A2 "$ARMA" lane no
run_arm B1 "$ARMB" fallback no
run_arm B2 "$ARMB" fallback no

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
