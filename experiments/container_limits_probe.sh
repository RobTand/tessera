#!/usr/bin/env bash
# Do host thread exports cross into a container, and which limit flag do
# affinity-blind counts see?  ubuntu:24.04, CPU only, no ports, no GPU.
set -uo pipefail
TS=${TS:-$(cd "$(dirname "$0")/.." && pwd)}
IMAGE=${IMAGE:-ubuntu:24.04}
source "$TS/experiments/runtime_image.sh"
# Gated like every other wrapper here that starts a container (#100): this one
# runs ubuntu, not the serving image, and the gate stamps the digest that
# actually produced the numbers below instead of leaving the reader with a
# floating tag.  The rule caught this script when it was written ungated --
# which is the same rule this issue is about, applied to its author.
runtime_image_require "$IMAGE" || exit 2
mask=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
echo "host Cpus_allowed_list: $mask   host nproc=$(nproc) onln=$(getconf _NPROCESSORS_ONLN)"
first=$(echo "$mask" | cut -d, -f1)
lo=${first%%-*}; hi=${first##*-}; [ "$hi" = "$lo" ] && hi=$lo
sub="$lo-$hi"
probe='echo "  in-container nproc=$(nproc) onln=$(getconf _NPROCESSORS_ONLN) OMP_NUM_THREADS=[${OMP_NUM_THREADS:-unset}]"'
echo "A) no limit flags, host exports OMP_NUM_THREADS=3:"
OMP_NUM_THREADS=3 docker run --rm --name tess375-probe-a "$IMAGE" bash -c "$probe"
echo "B) --cpus=2 (CFS quota), host exports OMP_NUM_THREADS=3:"
OMP_NUM_THREADS=3 docker run --rm --cpus=2 --name tess375-probe-b "$IMAGE" bash -c "$probe"
echo "C) --cpuset-cpus=$sub:"
docker run --rm --cpuset-cpus="$sub" --name tess375-probe-c "$IMAGE" bash -c "$probe"
echo "D) --cpuset-cpus=$sub with an explicit -e OMP_NUM_THREADS:"
docker run --rm --cpuset-cpus="$sub" -e OMP_NUM_THREADS=4 --name tess375-probe-d "$IMAGE" bash -c "$probe"
