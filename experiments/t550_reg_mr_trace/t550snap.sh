#!/bin/bash
# tessera#550 tracing helper: snapshot a failing process's VMAs at the instant
# ib_umem_get fails. Called from bpftrace system(); args: pid tid
OUT=/home/rob/tmp/tessera-550-tp2-rdma-20260918/scratch/repro/out/snap
mkdir -p "$OUT"
cat /proc/$1/maps > "$OUT/maps-$1-$2.txt" 2>&1
cat /proc/$1/smaps > "$OUT/smaps-$1-$2.txt" 2>&1
cat /proc/$1/status > "$OUT/status-$1-$2.txt" 2>&1
cat /proc/$1/limits > "$OUT/limits-$1-$2.txt" 2>&1
