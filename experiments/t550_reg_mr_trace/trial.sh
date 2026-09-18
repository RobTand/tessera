#!/usr/bin/env bash
# tessera#550 traced reproduction: one two-Spark TP2 vLLM serve of the 4-layer
# GLM stub over NCCL/RoCE (f0 ports), with bpftrace on both boxes recording the
# kernel side of every ibv_reg_mr. vLLM serve -> exempt from PrismaBuild.
# usage: trial.sh TAG [IB_DISABLE=0] ; env: TRACE=1 (default) EAGER=1 NCCL_COMPAT=1
set -uo pipefail
TAG=${1:?tag}
HERE=/home/rob/tmp/tessera-550-tp2-rdma-20260918/scratch/repro
OUT=$HERE/out/$TAG; mkdir -p "$OUT"
TS=/home/rob/tmp/tessera-pin-4c384e60
MODEL=${MODEL:-/mnt/shared/models/GLM-5.3-Flash-4layer}
IMG=localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb
WORKER=sparklina; HEAD_ADDR=10.100.96.1; PORT=29611; HTTP=8011; NAME=t550
EXT=/home/rob/tmp/glm-tp2-smoke/ext
TRACE=${TRACE:-1}
IB_DISABLE=${IB_DISABLE:-0}
KV_BYTES=4294967296
GPU_UTIL=${GPU_UTIL:-0.5}
LOAD=${LOAD:-}
LOAD_SECS=${LOAD_SECS:-240}

FABRIC=(-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0
        -e NCCL_IB_HCA=${IB_HCA:-rocep1s0f0,roceP2p1s0f0} -e NCCL_IB_DISABLE=$IB_DISABLE
        -e TESSERA_RESEARCH_GLM53_NOPE=1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1)
[ "${NCCL_COMPAT:-1}" = 1 ] && FABRIC+=(-e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 -e NCCL_DMABUF_ENABLE=0)
[ -n "${NCCL_EXTRA:-}" ] && for kv in $NCCL_EXTRA; do FABRIC+=(-e "$kv"); done

PREP='inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /tessera >/dev/null 2>&1
echo "[t550] ulimit -l: $(ulimit -l); caps: $(grep CapEff /proc/self/status)"'

SERVE_ARGS="--tensor-parallel-size 2 --nnodes 2 --master-addr $HEAD_ADDR --master-port $PORT \
 --distributed-executor-backend mp --attention-backend CUSTOM --kv-cache-dtype fp8_ds_mla \
 $([ "${EAGER:-1}" = 1 ] && echo --enforce-eager) $([ "${MOE_TRITON:-1}" = 1 ] && echo --moe-backend triton) --kernel-config '{\"enable_flashinfer_autotune\":false}' \
 --max-model-len 4096 --gpu-memory-utilization $GPU_UTIL --kv-cache-memory-bytes $KV_BYTES --trust-remote-code ${SERVE_EXTRA:-}"

docker_args() {  # $1 = VLLM_HOST_IP
  printf '%s\n' --network host --ipc host --device /dev/infiniband --gpus all \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK --shm-size 16g \
    -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET,ALLOC \
    -v "$TS/src":/tessera/src:ro -v "$TS/pyproject.toml":/tessera/pyproject.toml:ro \
    -v "$EXT":/ext -v /mnt/shared:/mnt/shared:ro \
    -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton \
    -e VLLM_HOST_IP="$1" -w /tessera
}
on() { local h=$1; shift; if [ "$h" = "$(hostname)" ]; then bash -c "$*"; else ssh "$h" "$@"; fi; }
guard() {  # $1 host: MemAvailable >= 16 GiB and PSI full avg10 < 20
  local a p; a=$(on "$1" "awk '/MemAvailable/{print \$2}' /proc/meminfo"); p=$(on "$1" "awk '/^full/{split(\$2,x,\"=\"); print int(x[2])}' /proc/pressure/memory")
  echo "$1 MemAvailable=$((a/1024/1024))GiB psi_full=$p"
  [ "$a" -ge $((16*1024*1024)) ] && [ "$p" -lt 20 ] || { echo "guard refused on $1"; exit 3; }
}
cleanup() { docker rm -f $NAME-rank0 >/dev/null 2>&1; ssh $WORKER "docker rm -f $NAME-rank1" >/dev/null 2>&1; }
stop_trace() {
  [ "$TRACE" = 1 ] || return 0
  sudo -n systemctl stop t550-bt-$TAG.service 2>/dev/null; ssh $WORKER "sudo -n systemctl stop t550-bt-$TAG.service" 2>/dev/null
  systemctl --user stop t550-jk-$TAG.service 2>/dev/null; ssh $WORKER "systemctl --user stop t550-jk-$TAG.service" 2>/dev/null
  sleep 2
  scp -q $WORKER:"$OUT/trace-sparklina.txt" $WORKER:"$OUT/journal-sparklina.txt" "$OUT/" 2>/dev/null
  scp -q -r $WORKER:"$HERE/out/snap" "$OUT/snap-sparklina" 2>/dev/null; ssh $WORKER "rm -rf $HERE/out/snap"
  [ -d "$HERE/out/snap" ] && mv "$HERE/out/snap" "$OUT/snap-sparky"
  true
}

guard sparky; guard $WORKER
cleanup
[ -z "$(docker ps -q --filter name=$NAME)" ] && [ -z "$(ssh $WORKER docker ps -q --filter name=$NAME)" ] || { echo "leftover $NAME containers"; exit 3; }
ssh $WORKER "mkdir -p $OUT $EXT"
printf '%s\n' "$SERVE_ARGS" > "$OUT/engine-args.txt"; echo "IB_DISABLE=$IB_DISABLE NCCL_COMPAT=${NCCL_COMPAT:-1} NCCL_EXTRA=${NCCL_EXTRA:-} EAGER=${EAGER:-1} TRACE=$TRACE GPU_UTIL=$GPU_UTIL LOAD=$LOAD LOAD_SECS=$LOAD_SECS" > "$OUT/config.txt"
mkc() { on "$1" "sudo -n cat /sys/kernel/debug/mlx5/0000:01:00.0/commands/CREATE_MKEY/failed /sys/kernel/debug/mlx5/0000:01:00.0/commands/CREATE_MKEY/n 2>/dev/null | paste - -"; }
echo "create_mkey_failed_n before: sparky=$(mkc sparky) lina=$(mkc $WORKER)" >> "$OUT/config.txt"
systemd-run --user --unit t550-mm-$TAG --collect $HERE/memmon.sh $OUT/mem-sparky.txt 600 >/dev/null 2>&1
ssh $WORKER "systemd-run --user --unit t550-mm-$TAG --collect $HERE/memmon.sh $OUT/mem-sparklina.txt 600" >/dev/null 2>&1
if [ "${COLD:-1}" = 1 ]; then echo "evict: sparky $($HERE/evict.sh) / lina $(ssh $WORKER $HERE/evict.sh)"; fi
if [ -n "$LOAD" ]; then
  systemd-run --user --unit t550-load-$TAG --collect $HERE/load.sh "$LOAD" $LOAD_SECS >/dev/null 2>&1
  ssh $WORKER "systemd-run --user --unit t550-load-$TAG --collect $HERE/load.sh $LOAD $LOAD_SECS" >/dev/null 2>&1
  sleep ${LOAD_LEAD:-8}
fi

if [ "$TRACE" = 1 ]; then
  sudo -n systemd-run --unit t550-bt-$TAG --collect -p WorkingDirectory=$HERE bpftrace --unsafe -B none -o "$OUT/trace-sparky.txt" "$HERE/trace.bt" >/dev/null
  ssh $WORKER "sudo -n systemd-run --unit t550-bt-$TAG --collect -p WorkingDirectory=$HERE bpftrace --unsafe -B none -o $OUT/trace-sparklina.txt $HERE/trace.bt" >/dev/null
  systemd-run --user --unit t550-jk-$TAG --collect bash -c "journalctl -kf -o short-precise > $OUT/journal-sparky.txt" >/dev/null 2>&1
  ssh $WORKER "systemd-run --user --unit t550-jk-$TAG --collect bash -c 'journalctl -kf -o short-precise > $OUT/journal-sparklina.txt'" >/dev/null 2>&1
  for _ in $(seq 1 30); do grep -q "trace start" "$OUT/trace-sparky.txt" 2>/dev/null && ssh $WORKER "grep -q 'trace start' $OUT/trace-sparklina.txt" 2>/dev/null && break; sleep 1; done
  echo "trace: $(head -1 "$OUT/trace-sparky.txt" 2>/dev/null) / lina: $(ssh $WORKER head -1 $OUT/trace-sparklina.txt 2>/dev/null)"
fi

t0=$(date +%s)
mapfile -t wargs < <(docker_args 10.100.96.2)
ssh $WORKER docker run -d --name $NAME-rank1 "$(printf '%q ' "${wargs[@]}" "${FABRIC[@]}")" --entrypoint bash $IMG -c "$(printf '%q' "$PREP
exec vllm serve $MODEL --node-rank 1 --headless $SERVE_ARGS")" >/dev/null
mapfile -t hargs < <(docker_args $HEAD_ADDR)
docker run -d --name $NAME-rank0 "${hargs[@]}" "${FABRIC[@]}" --entrypoint bash $IMG -c "$PREP
exec vllm serve $MODEL --node-rank 0 --host 0.0.0.0 --port $HTTP $SERVE_ARGS" >/dev/null
systemd-run --user --unit t550-memwd-rank0-$TAG --collect /home/rob/tmp/glm-tp2-smoke/mem-watchdog.sh $NAME-rank0 $WORKER $NAME-rank1 16 >/dev/null 2>&1
ssh $WORKER systemd-run --user --unit t550-memwd-rank1-$TAG --collect /home/rob/tmp/glm-tp2-smoke/mem-watchdog.sh $NAME-rank1 sparky $NAME-rank0 16 >/dev/null 2>&1

state=timeout
while [ $(( $(date +%s) - t0 )) -lt 480 ]; do
  if curl -sf -m 3 localhost:$HTTP/health >/dev/null; then state=ready; break; fi
  r0=$(docker ps -q --filter name=$NAME-rank0); r1=$(ssh $WORKER docker ps -q --filter name=$NAME-rank1)
  if [ -z "$r0" ] || [ -z "$r1" ]; then state=exited; sleep 3; break; fi
  sleep 3
done
secs=$(( $(date +%s) - t0 ))
gen=none
snap_maps() { for p in $(docker top $NAME-rank0 -eo pid 2>/dev/null | tail -n +2); do sudo -n cat /proc/$p/maps > "$OUT/maps-sparky-$p.txt" 2>/dev/null; done; ssh $WORKER "for p in \$(docker top $NAME-rank1 -eo pid 2>/dev/null | tail -n +2); do sudo -n cat /proc/\$p/maps > $OUT/maps-sparklina-\$p.txt 2>/dev/null; done"; }
if [ $state = ready ]; then
  snap_maps
  gen=$(curl -s -m 120 localhost:$HTTP/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print("ok", d["usage"]["completion_tokens"])' 2>/dev/null || echo fail)
fi
docker logs $NAME-rank0 > "$OUT/rank0.log" 2>&1
ssh $WORKER docker logs $NAME-rank1 > "$OUT/rank1.log" 2>&1
cleanup
stop_trace
systemctl --user stop t550-load-$TAG.service t550-mm-$TAG.service 2>/dev/null; ssh $WORKER "systemctl --user stop t550-load-$TAG.service t550-mm-$TAG.service" 2>/dev/null
echo "create_mkey_failed_n after: sparky=$(mkc sparky) lina=$(mkc $WORKER)" >> "$OUT/config.txt"
scp -q $WORKER:"$OUT/mem-sparklina.txt" $WORKER:"$OUT/maps-sparklina-*.txt" "$OUT/" 2>/dev/null
ib=$(cat "$OUT/rank0.log" "$OUT/rank1.log" | grep -c "Using network IB")
enomem=$(cat "$OUT/rank0.log" "$OUT/rank1.log" | grep -c "ibv_reg_mr.*Cannot allocate memory")
echo "$TAG state=$state secs=$secs gen=$gen ib_lines=$ib enomem_lines=$enomem" | tee -a "$HERE/out/summary.txt"
