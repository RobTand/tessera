#!/usr/bin/env bash
# tessera#508: the kpool-tail slot-mapping read under the CUDA memcheck tool.
# A GPU unit reproduction, not a serve: submit through PrismaBuild, e.g.
#   pbrun.py --cwd <tree> --gpu --priority -10 --timeout-s 1800 \
#     --container-image <IMG> -- bash experiments/glm53_508_graph_qual/kpool_tail_slotmap_repro.sh <IMG> <OUT>
# Four arms, each its own process in one container:
#   stock-cached    generic tail mapping enabled (the pinned runtime's rule),
#                   caching allocator, no sanitizer: the read goes unnoticed
#   stock-memcheck  same rule, PYTORCH_NO_CUDA_MEMORY_CACHING=1, memcheck
#   fixed-memcheck  tail opted out of the generic mapping (vLLM #57317)
#   fixed-cached    same, caching allocator, no sanitizer
set -uo pipefail
IMG=$1
OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
mkdir -p "$OUT"
# Gated like every wrapper that starts a container (issue #100): the digest in
# IMG is checked against the daemon's RepoDigests and what ran is stamped.
source "$TREE/experiments/runtime_image.sh"
runtime_image_require "$IMG" > "$OUT/image.txt" || { cat "$OUT/image.txt"; exit 2; }
cat "$OUT/image.txt"
# What a process inside may check its own image against (issue #132): the
# reference the daemon resolved, injected after this wrapper's own -e flags.
imgenv=()
while IFS= read -r _kv; do
  [ -n "$_kv" ] && imgenv+=(-e "$_kv")
done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
# The container runs as the invoking user: OUT may be on the root-squashed
# shared mount, where the image's root cannot write.
docker run --rm --gpus all --ipc host --network none --user "$(id -u):$(id -g)" \
  -v "$TREE/experiments/glm53_508_graph_qual":/w:ro -v "$OUT":/out \
  -e HOME=/out -e TRITON_CACHE_DIR=/out/triton -e TMPDIR=/out "${imgenv[@]}" \
  --entrypoint bash "$IMG" -c '
set -u
run() {  # name, env, sanitize(0|1), rule
  echo "== $1 (rule=$4) $(date -u +%FT%TZ)"
  san=""
  [ "$3" = 1 ] && san="compute-sanitizer --tool memcheck --print-limit 50 --kernel-name kns=_compute_slot_mappings_kernel --log-file /out/$1.memcheck.log"
  env $2 $san python3 /w/kpool_tail_slotmap_repro.py --tail-slot-mapping $4 > /out/$1.json 2> /out/$1.err
  echo "rc=$?" | tee /out/$1.rc
  [ "$3" = 1 ] && { echo "invalid_global_reads=$(grep -c "Invalid __global__" /out/$1.memcheck.log)"; tail -3 /out/$1.memcheck.log; }
  true
}
run stock-cached   "X=1" 0 enabled
run stock-memcheck "PYTORCH_NO_CUDA_MEMORY_CACHING=1" 1 enabled
run fixed-memcheck "PYTORCH_NO_CUDA_MEMORY_CACHING=1" 1 disabled
run fixed-cached   "X=1" 0 disabled
' 2>&1 | tee "$OUT/driver.log"
