#!/usr/bin/env bash
# One served arm of the issue-#545 step-3 A/B (host side).
#
# Serves MODEL_DIR through the tessera runtime tree TS_TREE (the arm's
# independent variable) inside the pinned runtime image, and runs
# served_fused_ab_profile.py in the same container: fixed prompts, greedy
# sampling, CUDA-event + wall call timing, vLLM's OWN in-engine profiler
# (workers write traces into the arm's profiler dir), a 10 Hz board-power
# sampler, and route records as the arm's dispatch proof.
#
# usage: served_fused_ab_arm.sh TS_TREE MODEL_DIR ARM MODE OUT SNAP \
#            [--glm] [extra driver args...]
set -euo pipefail
TS="$1"; MODEL="$2"; ARM="$3"; MODE="$4"; OUT="$5"; SNAP="$6"; shift 6
source "$SNAP/experiments/runtime_image.sh"
source "$SNAP/experiments/served_fused_ab_env.sh"
IMAGE=${IMAGE:-$(runtime_image_pin)}
# Refuse a floating image before any container work (issue #100); the resolved
# digest reaches the driver through the census declaration env (#132).
runtime_image_require "$IMAGE" || exit 2
mkdir -p "$OUT/ext" "$OUT/profiles"
TAG=$(basename "$MODEL")
NAME="tessera-545ab-${ARM}-${TAG}"
LOG="$OUT/log-${ARM}-${TAG}.txt"
GLM=0; EXTRA=()
for a in "$@"; do
  case "$a" in
    --glm) GLM=1 ;;
    *) EXTRA+=("$a") ;;
  esac
done
# The launcher's image declaration, exported for the in-container cross-check.
imgenv=()
while IFS= read -r kv; do
  [ -n "$kv" ] && imgenv+=(-e "$kv")
done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
# Explicit off/on: the GLM env args exist ONLY when GLM=1 (finding 4: ${VAR:+}
# tests nonempty, so GLM=0 used to leak the flag into the h1 arms).
glm_env=()
while IFS= read -r kv; do
  [ -n "$kv" ] && glm_env+=("$kv")
done <<<"$(served_fused_ab_glm_env "$GLM")"
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"
commit=$(git -C "$TS" rev-parse HEAD)
echo "[arm $ARM] tree=$TS commit=$commit image=$IMAGE model=$MODEL mode=$MODE glm=$GLM"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run --rm --gpus all --ipc=host --name "$NAME" \
  -v "$TS/src":/work/src:ro -v "$TS/pyproject.toml":/work/pyproject.toml:ro \
  -v "$SNAP":/drv:ro -v "$OUT":/out -v "$OUT/ext":/ext \
  -v /mnt/shared:/mnt/shared \
  -v "${MODEL_MOUNT}:${MODEL_MOUNT}":ro \
  -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton \
  -e TESSERA_SERVE_MODE="$MODE" \
  -e TESSERA_ARM_COMMIT="$commit" -e TESSERA_ARM_NAME="$ARM" \
  -e VLLM_TORCH_PROFILER_DIR=/out/profiles \
  ${glm_env[@]+"${glm_env[@]}"} \
  ${imgenv[@]+"${imgenv[@]}"} \
  ${TESSERA_545AB_DOCKER_EXTRA:-} \
  --entrypoint bash "$IMAGE" -c '
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include; for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /work
python3 -c "import importlib.metadata as m; print(\"[plugin] vllm.general_plugins:\", [e.name for e in m.entry_points(group=\"vllm.general_plugins\")])"
exec python3 /drv/experiments/served_fused_ab_profile.py \
  --model "'"$MODEL"'" --arm "'"$ARM"'" --tag "'"$TAG"'" --mode "'"$MODE"'" \
  --out /out --profiler-dir /out/profiles '"${EXTRA[*]}"'
' 2>&1 | tee "$LOG"
st=${PIPESTATUS[0]}
echo "[arm $ARM] exit=$st log=$LOG"
exit "$st"
