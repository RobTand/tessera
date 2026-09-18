#!/usr/bin/env bash
# tessera#550 reproduction stimulus. usage: load.sh MODES SECONDS
#   MODES: comma list of nfsread (2 O_DIRECT + 1 buffered stream over the NFS/RDMA
#          mount, the workload the census/prepare jobs impose) and compact (force
#          full memory compaction every 2 s: migrate_pages storms, pageblock churn).
# Pure I/O and sysctl load, bounded by SECONDS; no GPU work.
MODES=$1; SECS=${2:-180}
MODEL=/mnt/shared/models/GLM-5.3-Flash-4layer
pids=()
if [[ ",$MODES," == *,nfsread,* ]]; then
  for s in 1 2; do
    ( while :; do for f in "$MODEL"/model-*.safetensors; do dd if="$f" of=/dev/null bs=16M iflag=direct status=none; done; done ) & pids+=($!)
  done
  ( while :; do cat "$MODEL"/model-0000[1-6]*.safetensors > /dev/null; done ) & pids+=($!)
fi
if [[ ",$MODES," == *,compact,* ]]; then
  ( while :; do echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null; sleep 2; done ) & pids+=($!)
fi
sleep "$SECS"
kill "${pids[@]}" 2>/dev/null; pkill -P $$ 2>/dev/null; sleep 1; pkill -9 -P $$ 2>/dev/null
