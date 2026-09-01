#!/usr/bin/env bash
# Serve one checkpoint, dump its per-position logprobs, stop the serve.
#
# The two arms of a KL must not be resident at once: each is a 45 GB BF16
# model and the box has ~121 GB of unified memory shared with the GPU, so a
# concurrent pair is an OOM, not a measurement.  Sequential dumps also mean
# each arm gets an identical, uncontended box -- which matters because the
# reference arm's numbers are the denominator of everything downstream.
#
# usage: serve_and_dump_kl.sh <model-dir> <out.json> <role> [teacher-label]
set -euo pipefail

MODEL="$1"; OUT="$2"; ROLE="$3"; LABEL="${4:-}"
IMAGE="${TESSERA_KL_IMAGE:-prismaquant/glm53-mia-sm121:487ecf187}"
PORT="${TESSERA_KL_PORT:-8000}"
CORPUS="${TESSERA_KL_CORPUS:-/mnt/shared/tessera-kl/corpus_n8_s512.json}"
NAME="tessera-kl-serve"
LOG="${TESSERA_KL_LOGDIR:-/mnt/shared/tessera-kl}/serve_$(basename "$OUT" .json).log"

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "serving $MODEL  ($IMAGE)"
docker run -d --rm --name "$NAME" --gpus all --ipc=host \
  -p "${PORT}:8000" \
  -v /mnt/shared:/mnt/shared \
  "$IMAGE" \
  --model "$MODEL" --served-model-name kl-target \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager --trust-remote-code \
  >/dev/null

# The serve is the long pole; give it room but fail rather than hang forever.
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  up after ${i}0s"; break
  fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    docker logs "$NAME" > "$LOG" 2>&1 || true
    echo "serve died; log at $LOG"; tail -30 "$LOG"; exit 1
  fi
  sleep 10
done

# Spec-decode poisons a logprob readout: /v1/completions returns the DRAFT
# model's numbers when vLLM serves with a speculative config.  Refuse rather
# than record a number that silently belongs to another model.
if curl -s "http://127.0.0.1:${PORT}/metrics" | grep -q 'vllm:spec_decode'; then
  echo "REFUSED: serve has spec-decode active; the logprobs would be the draft model's"
  docker rm -f "$NAME" >/dev/null; exit 2
fi

ARGS=(dump --model kl-target --out "$OUT" --url "http://127.0.0.1:${PORT}"
      --corpus-contract "$CORPUS" --role "$ROLE" --artifact-path "$MODEL")
[ -n "$LABEL" ] && ARGS+=(--teacher-label "$LABEL")
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python /home/rob/dq-runs/kl_tool.py "${ARGS[@]}"

docker logs "$NAME" > "$LOG" 2>&1 || true
docker rm -f "$NAME" >/dev/null
echo "dumped -> $OUT   (serve log $LOG)"
