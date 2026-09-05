#!/usr/bin/env bash
# Greedy smoke on BOTH arms of a routed-MoE cell, one serve at a time (#198).
#
# Arm "bf16" is the source checkpoint on the pinned image's own vLLM; arm
# "tessera" is the E4M3/q1024 artifact through Tessera's plugin, resident,
# eager -- the two serves `experiments/ts5_lfm_teacher_bound.py` and
# `experiments/ts5_lfm_served_bound.py` ran for the campaign, with the same
# image, flags, utilisation and memory cap, minus the KL dump.  Each arm is
# served, asked every prompt in $PROMPTS by `moe_greedy_smoke.py run`, logged,
# and reaped before the other starts; the two receipts are then joined by
# `moe_greedy_smoke.py compare` into the pair the measurement doc quotes.
#
# Host memory is gated, not assumed: the GPU and host share one pool, and this
# script refuses to start a serve while MemAvailable is under $MIN_AVAIL_GIB.
# A telemetry log per arm samples MemAvailable every 10 s so the receipt can
# state the high-water mark rather than guess it.
#
# usage: IMAGE=<pinned digest> EXT=<dir> VLLM_CACHE=<dir> \
#          moe_greedy_smoke_pair.sh <bf16-source-dir> <tessera-artifact-dir> <seal.json> <out-dir>
set -euo pipefail
SOURCE="$1"; ARTIFACT="$2"; SEAL="$3"; OUT="$4"
TS=${TS:-$(cd "$(dirname "$0")/.." && pwd)}
IMAGE=${IMAGE:?set IMAGE to the pinned image digest of the cells under test}
EXT=${EXT:?set EXT to a torch-extensions dir outside the tree}
VLLM_CACHE=${VLLM_CACHE:?set VLLM_CACHE to a vLLM cache dir outside the tree}
PY=${PY:-python3}
PORT=${PORT:-8198}
NAME_PREFIX=${NAME_PREFIX:-ts198-smoke}
PROMPTS=${PROMPTS:-$TS/experiments/moe_greedy_smoke_prompts.json}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-40}
UTIL=${TESSERA_GPU_MEM_UTIL:-0.35}
export RUNTIME_IMAGE_PY="$PY" BUILD_IDENTITY_PY="$PY"
source "$TS/experiments/runtime_image.sh"
source "$TS/experiments/build_identity.sh"
source "$TS/experiments/serve_metrics.sh"
source "$TS/experiments/serve_lock.sh"
mkdir -p "$OUT" "$EXT" "$VLLM_CACHE"
[ -e "$OUT/pair.json" ] && { echo "REFUSED: $OUT/pair.json exists; a rerun gets its own directory"; exit 2; }

mem_avail_gib() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
require_memory() {
  local avail; avail=$(mem_avail_gib)
  echo "MemAvailable ${avail} GiB before $1 (gate ${MIN_AVAIL_GIB} GiB)" | tee -a "$OUT/driver.log"
  if [ "$avail" -lt "$MIN_AVAIL_GIB" ]; then
    echo "REFUSED: ${avail} GiB available < ${MIN_AVAIL_GIB} GiB; not starting $1" | tee -a "$OUT/driver.log"
    exit 3
  fi
}
telemetry() {  # every 10 s: wall clock, GPU power, MemAvailable -- until the arm is reaped
  while :; do
    date --iso-8601=seconds
    nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader 2>&1 || true
    grep -E '^(MemAvailable|SwapFree):' /proc/meminfo
    sleep 10
  done
}
identity() {  # sha256 of every file the serve reads, in the campaign's own shape
  PYTHONPATH="$TS/src" "$PY" -c '
import json, sys
from tessera.serving_parts import source_identity
print(json.dumps(source_identity(sys.argv[1]), indent=2, sort_keys=True))' "$1"
}

serve_arm() {
  local arm="$1" model="$2" name="${NAME_PREFIX}-$1" log="$OUT/serve_$1.log"
  local model_mount; model_mount="$(cd "$model" && pwd)"
  require_memory "$arm"
  identity "$model" > "$OUT/identity_${arm}_before.json"
  serve_lock_acquire
  trap 'serve_lock_release' EXIT
  docker rm -f "$name" >/dev/null 2>&1 || true
  telemetry > "$OUT/telemetry_$arm.log" 2>&1 &
  local tele=$!
  if [ "$arm" = "bf16" ]; then
    # experiments/ts5_lfm_teacher_bound.py's serve, verbatim in effect: the
    # image's own vLLM, no plugin, eager, the model mounted read-only.
    docker run -d --name "$name" --gpus all --ipc=host -p "${PORT}:8000" \
      -v /mnt/shared:/mnt/shared:ro -v "${model_mount}:${model_mount}:ro" \
      --memory=64g --memory-swap=64g $(build_identity_docker_env) \
      --entrypoint=vllm "$IMAGE" serve "$model" --served-model-name kl-target \
      --host 0.0.0.0 --port 8000 --max-model-len 4096 --max-num-seqs 8 \
      --gpu-memory-utilization "$UTIL" --max-logprobs 1024 --enforce-eager \
      --trust-remote-code >/dev/null
  else
    # experiments/tessera_plugin_served.sh's container body, verbatim: link the
    # CUDA headers, install this tree's plugin editable, serve resident + eager.
    docker run -d --name "$name" --gpus all --ipc=host -p "${PORT}:8000" \
      -v /mnt/shared:/mnt/shared:ro -v "${model_mount}:${model_mount}:ro" \
      -v "$TS/src":/work/src:ro -v "$TS/pyproject.toml":/work/pyproject.toml:ro \
      -v "$EXT":/ext -v "$VLLM_CACHE":/root/.cache/vllm \
      -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext \
      -e TESSERA_SERVE_MODE=resident -e TESSERA_GPU_MEM_UTIL="$UTIL" \
      --memory=64g --memory-swap=64g $(build_identity_docker_env) \
      --entrypoint bash "$IMAGE" -c '
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include; for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /work 2>&1 | tail -2
python3 -c "import importlib.metadata as m; print(\"[plugin] vllm.general_plugins:\", [e.name for e in m.entry_points(group=\"vllm.general_plugins\")])"
exec vllm serve '"$model"' --served-model-name kl-target --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL}" \
  --max-logprobs 1024 --enforce-eager --trust-remote-code' >/dev/null
  fi
  local up=0
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "$arm up after ${i}0s" | tee -a "$OUT/driver.log"; up=1; break
    fi
    if ! docker ps -q -f "name=^/${name}$" | grep -q .; then break; fi
    sleep 10
  done
  local rc=0
  if [ "$up" = 1 ] && serve_require_no_spec_decode "$PORT" "$OUT/metrics_$arm.txt"; then
    "$PY" "$TS/experiments/moe_greedy_smoke.py" run --url "http://127.0.0.1:${PORT}/v1/completions" \
      --tokenizer "$ARTIFACT" --prompts "$PROMPTS" --arm "$arm" --out "$OUT/smoke_$arm.json" \
      --pure-greedy 2>&1 | tee -a "$OUT/driver.log" || rc=$?
  else
    echo "$arm: serve did not come up or is speculative; see $log" | tee -a "$OUT/driver.log"; rc=1
  fi
  docker logs "$name" > "$log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  kill "$tele" 2>/dev/null || true; wait "$tele" 2>/dev/null || true
  serve_lock_release; trap - EXIT
  # After the reap, never before: a stamp that failed first would leave a
  # headless serve holding the GPU.
  local mode=""; [ "$arm" = "tessera" ] && mode=resident
  build_identity_stamp "$log" "$OUT/smoke_$arm.build.json" "$VLLM_CACHE" "$IMAGE" "$mode" 1 "$model"
  identity "$model" > "$OUT/identity_${arm}_after.json"
  cmp -s "$OUT/identity_${arm}_before.json" "$OUT/identity_${arm}_after.json" \
    || { echo "REFUSED: $arm checkpoint changed across the serve" | tee -a "$OUT/driver.log"; exit 4; }
  echo "MemAvailable $(mem_avail_gib) GiB after $arm reaped; high-water $(awk '/^MemAvailable/ {print $2}' "$OUT/telemetry_$arm.log" | sort -n | head -1) kB MemAvailable during serve" | tee -a "$OUT/driver.log"
  return "$rc"
}

echo "== $(date --iso-8601=seconds) image $IMAGE tree $TS commit ${TESSERA_COMMIT:-$(git -C "$TS" rev-parse HEAD 2>/dev/null || echo unknown)}" | tee -a "$OUT/driver.log"
# Not through a pipe: the gate exports the resolved digest for the build
# stamps, and a pipeline would leave it in a subshell.
runtime_image_require "$IMAGE" > "$OUT/runtime_image.txt"
tee -a "$OUT/driver.log" < "$OUT/runtime_image.txt"
printf '%s\n' "$RUNTIME_IMAGE_JSON" > "$OUT/runtime_image.json"
# The student's bytes are the campaign's sealed bytes or the serve is of
# something else: bind the seal before the first model load.
PYTHONPATH="$TS/src" "$PY" -c '
import json, sys
from pathlib import Path
from tessera.serving_parts import source_identity, sha256_file
artifact, seal_path = Path(sys.argv[1]), Path(sys.argv[2])
seal = json.loads(seal_path.read_text())
assert seal["checkpoint"] == str(artifact), ("seal names a different checkpoint", seal["checkpoint"])
assert source_identity(artifact) == seal["checkpoint_identity"], "artifact differs from its seal"
print(json.dumps({"seal": str(seal_path), "seal_sha256": sha256_file(seal_path),
                  "checkpoint": str(artifact), "bound": True}, indent=2))' "$ARTIFACT" "$SEAL" \
  | tee "$OUT/seal_binding.json"
serve_arm bf16 "$SOURCE"
serve_arm tessera "$ARTIFACT"
# --subject/--reference make the pair carry the evidence.smoke.record block a
# contract cell transcribes, and the status word contract.derive_smoke_status
# reads off it, so the receipt and the contract cannot disagree about the word
# (#327).  The cell is about the Tessera student; the BF16 arm is the reference.
"$PY" "$TS/experiments/moe_greedy_smoke.py" compare "$OUT/smoke_bf16.json" "$OUT/smoke_tessera.json" \
  --out "$OUT/pair.json" --markdown "$OUT/pair.md" \
  --subject tessera --reference bf16_source | tee -a "$OUT/driver.log"
echo "== done $(date --iso-8601=seconds); receipts under $OUT" | tee -a "$OUT/driver.log"
