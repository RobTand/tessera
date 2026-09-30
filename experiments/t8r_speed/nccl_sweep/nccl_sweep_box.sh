#!/usr/bin/env bash
# One box's half of the two-box NCCL sweep: one container, hard-bounded.
# usage: nccl_sweep_box.sh RANK OUT_DIR [extra nccl_sweep.py box args...]
#   RANK 0 runs on sparklina (head, 10.100.96.2, hosts the rendezvous), RANK 1 on sparky.
# Container flags and NCCL env mirror the A8SE-SH serve (run/cmd/latency-rank{0,1}.sh).
# Bound: the container is killed at NCCL_SWEEP_BOUND_S (default 290 s); exit 124 then.
set -uo pipefail
RANK=$1; OUT=$2; shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMG=${NCCL_SWEEP_IMAGE:-localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a}
BOUND=${NCCL_SWEEP_BOUND_S:-290}
MASTER=${NCCL_SWEEP_MASTER:-10.100.96.2}
PORT=${NCCL_SWEEP_PORT:-29611}
NAME=nccl-sweep-$(basename "$OUT")-rank$RANK
mkdir -p "$OUT/rank$RANK" "$OUT/tmp"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host --ipc host --device /dev/infiniband --gpus all \
  --label org.prismaquant.campaign=pact-u4 --label org.prismaquant.phase=nccl-sweep \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK --shm-size 4g \
  -v /mnt/shared:/mnt/shared:ro -v "$OUT":"$OUT" \
  -e TMPDIR="$OUT/tmp" -e PYTHONDONTWRITEBYTECODE=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_DISABLE=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_DEBUG=WARN \
  --entrypoint python3 "$IMG" "$HERE/nccl_sweep.py" box --rank "$RANK" \
  --master "$MASTER" --port "$PORT" --out "$OUT" "$@" >/dev/null || { echo "[nccl-sweep] rank$RANK: docker run failed"; exit 2; }
rc=$(timeout "$BOUND" docker wait "$NAME"); trc=$?
if [ $trc -ne 0 ]; then
  echo "[nccl-sweep] rank$RANK: bound ${BOUND}s hit, killing $NAME"
  docker kill "$NAME" >/dev/null 2>&1; rc=124
fi
docker logs "$NAME" > "$OUT/rank$RANK/container.log" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
echo "[nccl-sweep] rank$RANK: exit $rc (log $OUT/rank$RANK/container.log)"
exit "$rc"
