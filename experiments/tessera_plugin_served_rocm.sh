#!/usr/bin/env bash
# Serve a Tessera-wire checkpoint through Tessera's own vLLM plugin on an AMD
# device, dump its logprobs on the model-matched corpus, and record the route.
#
# The HIP sibling of ``tessera_plugin_served.sh``. It is a separate file rather
# than four more knobs on that one, and the reason is risk, not taste: the ten
# sm_121 cells' receipts were taken through that script, and a wrapper that
# changes under them cannot be re-read. What differs here is exactly what the
# platform forces, and each difference is listed in
# ``tessera_plugin_run_rocm.sh``, which this script calls to start the
# container -- so the image gate, the digest declaration and the plugin install
# have ONE home for the AMD lane and this file holds none of them.
#
# The KL instrument is not duplicated either: the dump and the compare are
# ``kl_tool.py``'s, the same binary the CUDA arms call, reached through
# ``TESSERA_KL_PYTHON``/``TESSERA_KL_TOOL`` because the AMD box is not the box
# that file lives on.
#
#   usage: tessera_plugin_served_rocm.sh <model-dir> <arm-name> [resident|streamed]
#
# Environment:
#   IMG                  the ROCm serve image digest (required; no default --
#                        the packaged pin is the sm_121 image and an AMD arm
#                        that fell back to it would name a runtime it never ran)
#   TESSERA_LANE_EAGER   1 (default) serves --enforce-eager; 0 serves compiled
#   TESSERA_KL_TEACHER   the image-matched teacher dump stem (no .npz)
#   TESSERA_KL_PYTHON    an interpreter with numpy, for the dump and compare
#   TESSERA_KL_TOOL      the path to kl_tool.py
#   TESSERA_KL_REGIME    prefill (default) or decode. The decode regime scores
#                        every position off an M=1 forward, which is the only
#                        regime a decode cell's launch is made in; the two are
#                        different metrics and a compare across them is refused
#                        by the tool, not by this wrapper.
set -euo pipefail
MODEL="$1"; ARM="$2"; MODE="${3:-resident}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TS=${TS:-$(cd "$HERE/.." && pwd)}
RUNS=${RUNS:-$HOME/agents/t460-cells/runs}
KLDIR=${KLDIR:-/mnt/shared/tessera-kl}
PORT=${PORT:-${TESSERA_KL_PORT:-8000}}
NAME=${TESSERA_KL_NAME:-tessera-plugin-serve-rocm-$ARM}
CORPUS=${TESSERA_KL_CORPUS:-$KLDIR/corpus_qwen_n8_s512.json}
ROLE=${TESSERA_KL_ROLE:-student}
# A teacher arm dumps the unquantized model on the SAME image and compares
# nothing: it is the denominator every student arm is read against, and a
# teacher from another runtime is a number about two runtimes.
TEACHER=${TESSERA_KL_TEACHER:-}
[ "$ROLE" = teacher ] || [ -n "$TEACHER" ] || {
  echo "set TESSERA_KL_TEACHER to the image-matched teacher dump stem" >&2; exit 2; }
LABEL=${TESSERA_KL_TEACHER_LABEL:-}
[ "$ROLE" != teacher ] || [ -n "$LABEL" ] || {
  echo "a teacher dump needs TESSERA_KL_TEACHER_LABEL (it travels into every compare line)" >&2; exit 2; }
DUMP=${TESSERA_KL_DUMP:-$RUNS/kl/tessera_$ARM.json}
LOG=${TESSERA_KL_LOG:-$RUNS/serve_tessera_$ARM.log}
PY=${TESSERA_KL_PYTHON:?set TESSERA_KL_PYTHON to an interpreter that can run kl_tool.py}
KL_TOOL=${TESSERA_KL_TOOL:?set TESSERA_KL_TOOL to the path of kl_tool.py}
TOPK=${TESSERA_KL_TOPK:-1024}
REGIME=${TESSERA_KL_REGIME:-prefill}
# The stride must be the serve's KV block size; kl_tool checks it against the
# serve's own `usage.prompt_tokens` and refuses rather than mislabelling.
DECODE_STRIDE=${TESSERA_KL_DECODE_STRIDE:-16}
EAGER_FLAG=--enforce-eager; [ "${TESSERA_LANE_EAGER:-1}" = "0" ] && EAGER_FLAG=
mkdir -p "$RUNS/kl" "$(dirname "$LOG")"
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"

echo "serving $MODEL via the tessera plugin on HIP (mode=$MODE, port=$PORT, eager=${TESSERA_LANE_EAGER:-1})"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# The launcher owns the image gate, the WSL device flags and the non-editable
# install; -d and --name are ordinary docker arguments it passes through.
TS="$TS" bash "$HERE/tessera_plugin_run_rocm.sh" \
  -d --name "$NAME" --ipc=host --shm-size=8g -p "${PORT}:8000" \
  -v "${MODEL_MOUNT}:${MODEL_MOUNT}" \
  -e TESSERA_SERVE_MODE="$MODE" \
  -e TESSERA_GPU_MEM_UTIL="${TESSERA_GPU_MEM_UTIL:-0.55}" \
  ${TESSERA_LANE_DOCKER_EXTRA:-} \
  -- "exec vllm serve '$MODEL' --served-model-name kl-target --host 0.0.0.0 --port 8000 \
      --max-model-len 4096 --max-num-seqs 8 \
      --gpu-memory-utilization \"\${TESSERA_GPU_MEM_UTIL}\" \
      --max-logprobs $TOPK $EAGER_FLAG --trust-remote-code \
      ${TESSERA_LANE_SERVE_EXTRA:-}" >/dev/null

reap() { docker logs "$NAME" > "$LOG" 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap 'reap' EXIT
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "  up after $((i*10))s"; break; fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    reap; echo "serve died; log at $LOG"; tail -40 "$LOG"; exit 1
  fi
  sleep 10
done

# Spec-decode poisons a logprob readout: /v1/completions returns the DRAFT
# model's numbers. The refusal is the shared one.
source "$HERE/serve_metrics.sh"
if ! serve_require_no_spec_decode "$PORT" "$RUNS/metrics_$ARM.txt"; then reap; exit 2; fi

# The greedy smoke, before anything is measured: principle 5's "generates
# correctly" leg, recorded verbatim rather than eyeballed.
curl -s "http://127.0.0.1:${PORT}/v1/completions" -H 'content-type: application/json' \
  -d '{"model":"kl-target","prompt":"The capital of France is","max_tokens":16,"temperature":0}' \
  | "$PY" -c "import json,sys; print('completion:', repr(json.load(sys.stdin)['choices'][0]['text']))"

DUMP_ARGS=(dump --model kl-target --out "$DUMP"
  --url "http://127.0.0.1:${PORT}/v1/completions" --top-k "$TOPK"
  --corpus-contract "$CORPUS" --role "$ROLE" --artifact-path "$MODEL"
  --regime "$REGIME")
[ "$REGIME" != decode ] || DUMP_ARGS+=(--decode-stride "$DECODE_STRIDE")
[ -z "$LABEL" ] || DUMP_ARGS+=(--teacher-label "$LABEL")
if ! "$PY" "$KL_TOOL" "${DUMP_ARGS[@]}"; then
  reap; echo "dump FAILED; serve log at $LOG"; exit 3
fi
reap
trap - EXIT
if [ "$ROLE" = teacher ]; then
  echo "teacher dumped -> $DUMP"
  exit 0
fi
echo "--- KL vs teacher ---"
"$PY" "$KL_TOOL" compare "$TEACHER.npz" "$DUMP.npz" --out "$RUNS/kl_tessera_$ARM.json" | tail -12
