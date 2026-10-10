#!/usr/bin/env bash
# Served top-K KL, greedy smoke and long-context smoke for tessera#1172.
#
# The exact full-vocabulary number comes from the in-runtime dump
# (full_vocab_kl_dump.py); this script supplies the run-parity half: served
# top-K bounds in both regimes off the same frozen corpus, a greedy smoke on
# both arms under the instrument's own degeneration rule, one long-context
# smoke prompt per arm, and the serve logs, metrics and exit statuses the KL
# report reads. Stock vLLM serves: no tessera route executes, and the report
# says so rather than printing an empty census.
#
# usage: full_vocab_kl_campaign_serve.sh <teacher-dir> <student-dir> <outdir>
set -euo pipefail

TEACHER="$1"; STUDENT="$2"; OUT="$3"
WT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT/logs"
CORPUS="${TESSERA_KL_CORPUS:-/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json}"
IMAGE="${TESSERA_KL_IMAGE:-prismaquant/glm53-mia-sm121:487ecf187}"
STATUS="$OUT/exit_status.txt"
: > "$STATUS"
note() { echo "$1=$2" >> "$STATUS"; }
commit() {  # <units>; durable progress so a quiet serve is not a stall
  [ -n "${PRISMABUILD_ACTION_PROGRESS_HELPER:-}" ] && python3 \
    "$PRISMABUILD_ACTION_PROGRESS_HELPER" --phase serve --units "$1" || true
}
KL="${TESSERA_KL_KL:-/home/rob/dq-runs/kl_tool.py}"

# ---- served top-K dumps, both arms, both regimes ---------------------------
export TESSERA_KL_CORPUS="$CORPUS" TESSERA_KL_IMAGE="$IMAGE"
export TESSERA_KL_PORT="$PORT" TESSERA_KL_NAME="$NAME"
export TESSERA_KL_LOGDIR="$OUT/logs" TESSERA_KL_TOPK=1024
export TESSERA_KL_PY="$PY" TESSERA_KL_KL="$KL"
# NOTE: serve_and_dump_kl.sh hardcodes its own PY/KL paths; the exports above
# document the intended pair and the calls below verify the files exist.
[ -x "$PY" ] || { echo "REFUSED: no $PY" >&2; exit 2; }
[ -f "$KL" ] || { echo "REFUSED: no $KL" >&2; exit 2; }

"$WT/experiments/serve_and_dump_kl.sh" "$TEACHER" \
  "$OUT/topk_teacher_prefill.json" teacher BF16
note topk_teacher_prefill $?
commit 1
TESSERA_KL_REGIME=decode "$WT/experiments/serve_and_dump_kl.sh" "$TEACHER" \
  "$OUT/topk_teacher_decode.json" teacher BF16
note topk_teacher_decode $?
commit 2
"$WT/experiments/serve_and_dump_kl.sh" "$STUDENT" \
  "$OUT/topk_student_prefill.json" student
note topk_student_prefill $?
commit 3
TESSERA_KL_REGIME=decode "$WT/experiments/serve_and_dump_kl.sh" "$STUDENT" \
  "$OUT/topk_student_decode.json" student
note topk_student_decode $?
commit 4

"$PY" "$KL" compare "$OUT/topk_teacher_prefill.json.npz" \
  "$OUT/topk_student_prefill.json.npz" \
  --out "$OUT/topk_compare_prefill.json"
note topk_compare_prefill $?
"$PY" "$KL" compare "$OUT/topk_teacher_decode.json.npz" \
  "$OUT/topk_student_decode.json.npz" \
  --out "$OUT/topk_compare_decode.json"
note topk_compare_decode $?
commit 5

# ---- prompts: seven short raw prompts plus one long-context smoke ---------
# The long prompt decodes held-out corpus chunks 4..7 (2048 tokens) and
# truncates to 2000, so it is WikiText-2 test text the KL never scores for
# quality; at --max-model-len 4096 the full 32k native window stays a bare
# area and the report labels it.
"$PY" - "$CORPUS" "$TEACHER" "$OUT/smoke_prompts.json" <<'EOF'
import json, sys
corpus = json.loads(open(sys.argv[1]).read())
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[2] + "/tokenizer.json")
long_ids = [t for c in corpus["chunks"][4:8] for t in c][:2000]
text = tok.decode(long_ids)
back = tok.encode(text).ids
print(f"long prompt tokens: {len(back)}", flush=True)
prompts = [
    {"id": f"P{i}", "prompt": p, "max_tokens": 64}
    for i, p in enumerate([
        "The capital of France is",
        "In 1969, humans first",
        "The quick brown fox",
        "To bake bread, first",
        "Python is a programming",
        "The largest ocean on Earth is",
        "Once upon a time,",
    ])]
prompts.append({"id": "PLONG", "prompt": text, "max_tokens": 64})
json.dump({"prompts": prompts}, open(sys.argv[3], "w"), indent=1)
EOF
note smoke_prompts $?
commit 6

# ---- smoke serves, one arm at a time under the shared serve lock -----------
# shellcheck source=serve_lock.sh
source "$WT/experiments/serve_lock.sh"
serve_arm() {  # <model-dir> <arm>
  local model="$1" arm="$2"
  local mount log
  mount="$(cd "$(dirname "$model")" && pwd)"
  log="$OUT/logs/serve_smoke_$arm.log"
  SERVE_LOCK_OWNER="$0 smoke-$arm"; serve_lock_acquire
  # shellcheck disable=SC2064
  trap "docker rm -f $NAME >/dev/null 2>&1 || true; serve_lock_release" RETURN
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --gpus all --ipc=host \
    -p "${PORT}:8000" -v /mnt/shared:/mnt/shared \
    -v "${mount}:${mount}" \
    "$IMAGE" "$model" --served-model-name kl-target \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 --max-num-seqs 8 \
    --gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL:-0.85}" \
    --max-logprobs 1024 --enforce-eager --trust-remote-code \
    >"$log.docker" 2>&1
  : >"$log"
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
    sleep 5
    [ "$i" = 240 ] && { echo "REFUSED: serve never healthy" >&2; return 1; }
  done
  docker logs "$NAME" >"$log" 2>&1 || true
  "$PY" "$WT/experiments/moe_greedy_smoke.py" run \
    --url "http://127.0.0.1:${PORT}/v1/completions" \
    --tokenizer "$TEACHER" --prompts "$OUT/smoke_prompts.json" \
    --arm "$arm" --out "$OUT/smoke_$arm.json"
  echo "smoke_$arm=$?" >>"$STATUS"
  curl -sf "http://127.0.0.1:${PORT}/metrics" -o "$OUT/metrics_$arm.txt" || true
  echo "serve_smoke_${arm}_log=$log" >>"$STATUS"
}
serve_arm "$TEACHER" teacher_bf16
serve_arm "$STUDENT" student_fp8rtn
commit 7

"$PY" "$WT/experiments/moe_greedy_smoke.py" compare \
  "$OUT/smoke_student_fp8rtn.json" "$OUT/smoke_teacher_bf16.json" \
  --out "$OUT/smoke_pair.json" \
  --subject student_fp8rtn --reference bf16_source
note smoke_compare $?
commit 8

echo "=== smoke pair ==="
"$PY" - "$OUT/smoke_pair.json" <<'EOF'
import json, sys
pair = json.load(open(sys.argv[1]))
print("status:", pair.get("status"), "attribution:", pair.get("attribution"))
for row in pair.get("record", {}).get("rows", pair.get("rows", [])):
    print(" ", row)
EOF
echo "=== done ==="
