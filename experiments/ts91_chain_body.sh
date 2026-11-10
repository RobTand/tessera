#!/usr/bin/env bash
# The four censuses of the #91 reproduction, in one process.
#
# Every parameter arrives positionally. The Flux launcher starts from an empty
# environment with an explicit allow-list, so these censuses set the values
# they need rather than relying on submitter environment inheritance.
#
# usage: ts91_chain_body.sh <worktree> <suffix> [ext-a] [ext-b-readonly] [image]
#
# The image argument exists because the two arms of this experiment are two
# CHECKOUTS, and only one of them may know how to resolve the serve pin: the
# before-arm predates issue #100's runtime_image.sh.  Passing the resolved
# digest to both is what makes them a matched pair -- the arms must differ in
# src/tessera/serving and in nothing else.  It is a caller's argument, never a
# constant here: the pin lives in runtime_contract.json and is read from there.
set -uo pipefail
WT=$1
SUF=$2
RUNS=${RUNS:-/home/rob/tessera-runs/ts91}
# Per chain, so two chains running concurrently never share a JIT build dir:
# the point of the experiment is that the two lane states stay separated.
EXT_A=${3:-$RUNS/ext-A-$SUF}
EXT_B_RO=${4:-$RUNS/ext-B-readonly-$SUF}
IMAGE=${5:-}

# A closed environment reaches here, so name what the censuses need rather
# than inheriting it.  HOME is docker's config lookup; TMPDIR and the Triton
# cache are the house rules (never /tmp, never /mnt/shared).
export HOME=${HOME:-/home/rob}
export PATH=${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
# The caller already holds a GPU slot: this whole chain is one pool job, and
# a box-local lock inside it would be a second admission point.
export TS91_NO_LOCK=1
export WT RUNS
export TS91_EXT_A=$EXT_A TS91_EXT_B_RO=$EXT_B_RO
[ -n "$IMAGE" ] && export IMG=$IMAGE

REPRO=$WT/experiments/ts91_cache_key_repro.sh
[ -x "$REPRO" ] || { echo "ts91: no repro script at $REPRO" >&2; exit 2; }

run() {  # arm cache-name tag
  if [ "${TS91_DRY:-0}" = 1 ]; then
    echo "would run: $REPRO $1 $2 compiled $3   (EXT_A=$EXT_A EXT_B_RO=$EXT_B_RO IMG=${IMG:-tree default})"
  else
    "$REPRO" "$1" "$2" compiled "$3"
  fi
}

run A "X-$SUF" "$SUF-A-into-X"
run B "Y-$SUF" "$SUF-B-into-Y"
run B "X-$SUF" "$SUF-B-into-X"
run A "Y-$SUF" "$SUF-A-into-Y"
