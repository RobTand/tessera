#!/usr/bin/env bash
# Bounded small A4 TP2 serve: the 4-layer stub on both Sparks, native routes.
#
# Settings are SMALL-SERVE-SETTINGS.md's (context 2048, KV 1 GiB/rank,
# max_num_seqs 8, max_num_batched_tokens 256, eager) -- NOT the full GLM
# target.  The serve worktree's src is mounted so vLLM imports the native
# routes; the image is the pinned a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb.
#
# Run from /home/rob/tmp/tessera-native-a4-serve on each host with the role:
#   bash experiments/small_a4_serve.sh rank0   # on sparky
#   bash experiments/small_a4_serve.sh rank1   # on sparklina
set -euo pipefail

TS=${TS:-$(cd "$(dirname "$0")/.." && pwd)}
IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb
MODEL=${MODEL:-/mnt/shared/tessera-runs/moe/glm53-4layer-a4-e2m1x2-q896-l2}
ROLE=${1:?rank0|rank1}
NAME="a4-small-${ROLE}"

# Gated like every other wrapper here that starts a container (issue #100):
# this one names an explicit digest, so the gate verifies that digest against
# the daemon's RepoDigests and stamps what actually ran rather than trusting the
# literal above.
source "$TS/experiments/runtime_image.sh"
runtime_image_require "$IMAGE" || exit 2

if [ "$ROLE" = "rank0" ]; then
  RANK=0; HOST_IP=192.168.100.1; PEER_IP=192.168.100.2; PORT=8000
else
  RANK=1; HOST_IP=192.168.100.2; PEER_IP=192.168.100.1; PORT=8000
fi

exec docker run --rm --gpus all --name "$NAME" \
  --network host --ipc host --shm-size 8g \
  --user 1000:1000 -e HOME=/tmp \
  -e VLLM_HOST_IP="$HOST_IP" -e NCCL_IB_DISABLE=1 \
  -e OMP_NUM_THREADS=8 -e PYTHONPATH=/work/src \
  -e TESSERA_SERVE_MODE=resident \
  -v "$PWD":/work:ro -v /mnt/shared:/mnt/shared:ro \
  --entrypoint bash "$IMAGE" -lc "
    python3 -m vllm.entrypoints.openai.api_server \
      --model '$MODEL' \
      --served-model-name tessera-a4-small \
      --tensor-parallel-size 2 --distributed-executor-backend mp \
      --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 256 \
      --gpu-memory-utilization 0.55 \
      --enforce-eager \
      --port $PORT --host 127.0.0.1
  "
