#!/usr/bin/env bash
# Serve a Tessera-wire checkpoint through Tessera's own vLLM plugin on the
# selected digest-pinned image (the dense vLLM 0.28 pin is the default). Dump
# its logprobs on the selected model-matched corpus, compare to the supplied
# image-matched teacher, and record the route. Campaigns must supply matching
# model, corpus, teacher and runtime settings together.
#
# For the historical dense Gridbook retargeting experiment, the checkpoints
# were hardlinks retargeted by config only (retarget_checkpoint_to_plugin.py),
# so mutual KL against that lane tested preservation of its decoder arithmetic.
# That experiment's acceptance rule is not a general MoE promotion criterion;
# each campaign must retain its own matched-byte served evidence and gate.
#
# There is NO enable flag: ``quant_method: "tessera"`` in the checkpoint selects
# the plugin.  TESSERA_SERVE_MODE declares the residency.
#
# usage: tessera_plugin_served.sh <model-dir> <arm-name> [resident|streamed]
set -euo pipefail
MODEL="$1"; ARM="$2"; MODE="${3:-resident}"
TS=${TS:-/home/rob/tessera}
RUNS=${RUNS:-/home/rob/tessera-runs/tsplugin}
EXT=${EXT:-$RUNS/ext}
VLLM_CACHE=${VLLM_CACHE:-$RUNS/vllm-cache}
source "$(dirname "$0")/runtime_image.sh"
IMAGE=${IMAGE:-$(runtime_image_pin)}
KLDIR=/mnt/shared/tessera-kl
PORT=${PORT:-${TESSERA_KL_PORT:-8000}}
# Per-arm container name (overridable, like serve_and_dump_kl.sh's TESSERA_KL_NAME):
# two workers on one box must never contend for a container name, and a stale
# container of MINE must never be a name another worker's `docker rm -f` matches.
NAME=${TESSERA_KL_NAME:-tessera-plugin-serve-$ARM}
CORPUS=${TESSERA_KL_CORPUS:-$KLDIR/corpus_qwen_n8_s512.json}
# The three paths a NON-Qwen arm has to move, defaulted to what every arm
# before them used so an existing command line records the same files.  The
# corpus was already overridable; these were not, and a model with another
# tokenizer needs all four to move together or it compares its logprobs to
# another model's.
TEACHER=${TESSERA_KL_TEACHER:-$KLDIR/qwen_teacher_bf16_v028.json}
DUMP=${TESSERA_KL_DUMP:-$KLDIR/qwen_tessera_$ARM.json}
LOG=${TESSERA_KL_LOG:-$RUNS/serve_qwen_tessera_$ARM.log}
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
# GPU-MEMORY UTILISATION IS PASSED IN, NOT EXPANDED IN THE CONTAINER.  The
# `vllm serve` line lives inside a single-quoted `bash -c` string, so
# "${TESSERA_GPU_MEM_UTIL:-0.85}" there is expanded by the CONTAINER's shell,
# where only the variables named with -e exist.  Without the -e below the
# serve silently asked for 0.85 of a shared box and refused to start
# ("Free memory ... is less than desired GPU memory utilization") while the
# host had exported 0.30.  Host-expanded values (the model, --max-logprobs,
# the eager flag) close the quote instead; this one cannot, because the
# default belongs with the reader.
# TESSERA_LANE_EAGER=0 serves under vLLM's default compiled forward + CUDA
# graphs; the eager serve is the numerics arm, the compiled serve is principle
# 9's second leg.  Compiled mode contributes no optional flag;
# --trust-remote-code is appended once below.
EAGER_FLAG=--enforce-eager; [ "${TESSERA_LANE_EAGER:-1}" = "0" ] && EAGER_FLAG=
# Which compiled build served this dump, recorded beside it (issue #30).  This
# wrapper already pins ONE $VLLM_CACHE across every arm, so the stamp can read
# the cache slot's contents and the sidecar is complete: the AOT key alone
# would not be, since both of the two builds 0.017117 apart sit under one key.
source "$(dirname "$0")/build_identity.sh"
mkdir -p "$EXT" "$VLLM_CACHE" "$RUNS"
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"
echo "serving $MODEL via the tessera plugin ($IMAGE, mode=$MODE, port=$PORT)"
# Refuse a floating image BEFORE the serve lock (issue #100): a wrapper that
# is going to refuse must not first take the box's one serve lock.  The
# resolved digest reaches this arm's build sidecar through build_identity.sh.
runtime_image_require "$IMAGE" || exit 2
# One serve at a time on this box: the GPU and host share one 128 GB pool.
source "$(dirname "$0")/serve_lock.sh"; SERVE_LOCK_OWNER="$0 $ARM"; serve_lock_acquire
trap serve_lock_release EXIT
# Reaping a stale container of my own happens INSIDE the lock: outside it, a
# `docker rm -f` races a serve another worker is starting under the same GPU.
docker rm -f "$NAME" >/dev/null 2>&1 || true
# ONE compile cache across every arm, deliberately: resident-after-streamed in
# one cache is the regression test for the compile-cache identity key (the mode
# is folded into additional_config by tessera.serving.compile_identity).
# Separate caches would hide a broken key.
docker run -d --name "$NAME" --gpus all --ipc=host -p "${PORT}:8000" \
  -v /mnt/shared:/mnt/shared -v "${MODEL_MOUNT}:${MODEL_MOUNT}" \
  -v "$TS/src":/work/src:ro -v "$TS/pyproject.toml":/work/pyproject.toml:ro \
  -v "$EXT":/ext -v "$VLLM_CACHE":/root/.cache/vllm \
  -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext \
  -e TESSERA_SERVE_MODE="$MODE" \
  -e TESSERA_GPU_MEM_UTIL="${TESSERA_GPU_MEM_UTIL:-0.85}" \
  $(build_identity_docker_env) \
  ${TESSERA_LANE_DOCKER_EXTRA:-} \
  --entrypoint bash "$IMAGE" -c '
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include; for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /work 2>&1 | tail -2
python3 -c "import importlib.metadata as m; print(\"[plugin] vllm.general_plugins:\", [e.name for e in m.entry_points(group=\"vllm.general_plugins\")])"
exec vllm serve '"$MODEL"' --served-model-name kl-target --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL:-0.85}" \
  --max-logprobs '"${TESSERA_KL_TOPK:-1024}"' '"${EAGER_FLAG}"' --trust-remote-code' >/dev/null
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "  up after ${i}0s"; break; fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    docker logs "$NAME" > "$LOG" 2>&1 || true
    echo "serve died; log at $LOG"; tail -40 "$LOG"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1
  fi
  sleep 10
done
if curl -s "http://127.0.0.1:${PORT}/metrics" | grep -q 'vllm:spec_decode'; then
  echo "REFUSED: spec-decode active"; docker rm -f "$NAME" >/dev/null; exit 2
fi
# greedy smoke first: the answer to the other arms' prompt
curl -s "http://127.0.0.1:${PORT}/v1/completions" -H 'content-type: application/json' \
  -d '{"model":"kl-target","prompt":"The capital of France is","max_tokens":16,"temperature":0}' | $PY -c "import json,sys; print('completion:', repr(json.load(sys.stdin)['choices'][0]['text']))"
if ! $PY /home/rob/dq-runs/kl_tool.py dump --model kl-target --out "$DUMP" --url "http://127.0.0.1:${PORT}/v1/completions" \
  --corpus-contract "$CORPUS" --role student --artifact-path "$MODEL"; then
  docker logs "$NAME" > "$LOG" 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1
  echo "dump FAILED; serve log at $LOG"; exit 3
fi
docker logs "$NAME" > "$LOG" 2>&1 || true
docker rm -f "$NAME" >/dev/null
# After the reap, never before: a stamp that failed first would leave a
# headless serve holding the GPU.
build_identity_stamp "$LOG" "${DUMP%.json}.build.json" "$VLLM_CACHE" "$IMAGE" \
  "$MODE" "${TESSERA_LANE_EAGER:-1}" "$MODEL"
echo "--- route ---"
grep -i "TESSERA_NVFP4\|TESSERA_FP8\|tessera\|Using .* for .*GEMM\|Selected .*Kernel for" "$LOG" | grep -iv "warning.*deprecat" | sed 's/.*INFO[^ ]* //' | sort | uniq -c | sort -rn | head -12 || true
echo "--- KL vs teacher ---"
$PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER.npz" "$DUMP.npz" --out "$RUNS/kl_tessera_$ARM.json" | tail -12
