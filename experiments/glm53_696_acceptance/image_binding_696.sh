#!/usr/bin/env bash
# tessera#696: bind the GLM serving images to their vLLM source and the causal
# backport, by extraction from the images themselves (not from prose).
#
# For each of the stock, kpool-tail-backport and nightly GLM serving images
# this records the image's own answer (image_binding_probe.py) for:
# python/torch/vLLM/FlashInfer versions, the sha256 of the V2 runner
# (v1/worker/gpu/model_runner.py) and of the generic slot-mapping kernel
# (v1/worker/gpu/block_table.py), and a sha256 manifest of every .py file
# under the installed vllm package. It then diffs the stock and backport
# manifests (the gate's binding claim: that runner file is the only vLLM file
# that differs), diffs the runner file itself, and inspects the nightly
# runner's slot-mapping region so the deployed-image question is answered
# from bytes, not from a version number.
#
# usage: image_binding_696.sh [OUT_DIR]
# Env: STOCK_IMG / BACKPORT_IMG / NIGHTLY_IMG override the pinned digests.
# CPU-only: no --gpus; the inner containers never see a device.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/mnt/shared/tessera-runs/receipts/696-serving-acceptance-20261004/image-binding}
STOCK_IMG=${STOCK_IMG:-localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5}
BACKPORT_IMG=${BACKPORT_IMG:-localhost/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b}
NIGHTLY_IMG=${NIGHTLY_IMG:-localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a}
RUNNER=v1/worker/gpu/model_runner.py
BLOCK_TABLE=v1/worker/gpu/block_table.py

mkdir -p "$OUT"
chmod a+rwx "$OUT" 2>/dev/null || true

# Inside one image: identity, the two hashes, the manifest, the two files.
inspect() {
  local name=$1 img=$2
  local dir="$OUT/$name"
  mkdir -p "$dir"
  docker run --rm --entrypoint /usr/bin/python3 --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PROBE_OUT=/out \
    -v "$dir":/out -v "$HERE":/probe:ro \
    "$img" /probe/image_binding_probe.py > "$dir/identity.txt" 2>&1
  # The package root the image itself reported; the file extracts need the
  # absolute path (a relative cat reads nothing).
  local root
  root=$(sed -n 's/.*"vllm_path": "\([^"]*\)".*/\1/p' "$dir/identity.txt" | head -1)
  local rel base
  for rel in "$RUNNER" "$BLOCK_TABLE"; do
    base=$(basename "$rel")
    if [ -n "$root" ]; then
      docker run --rm --entrypoint /bin/cat --user "$(id -u):$(id -g)" \
        -e HOME=/tmp "$img" "$root/$rel" > "$dir/$base" 2>/dev/null
    else
      echo "(no vllm_path in identity.txt; cannot extract $rel)" > "$dir/$base"
    fi
  done
}

inspect stock "$STOCK_IMG"
inspect backport "$BACKPORT_IMG"
inspect nightly "$NIGHTLY_IMG"

# The gate's binding claim, checked: stock and backport differ only in the runner.
{
  echo "# vllm .py files that differ between stock f8dbe1a0 and backport c2e75e03"
  diff "$OUT/stock/manifest.txt" "$OUT/backport/manifest.txt" | grep '^[<>]' || true
} > "$OUT/stock-vs-backport-manifest.diff" 2>&1

{
  echo "# diff vllm/v1/worker/gpu/model_runner.py: stock f8dbe1a0 -> backport c2e75e03"
  diff -u "$OUT/stock/model_runner.py" "$OUT/backport/model_runner.py" || true
} > "$OUT/runner.diff" 2>&1

# The deployed-image question from bytes: which slot-mapping rule does the
# nightly V2 runner apply to the kpool tail?
{
  echo "# nightly $NIGHTLY_IMG"
  echo "# KpoolTailSpec / uses_slot_mapping mentions in the nightly runner:"
  grep -n "uses_slot_mapping\|KpoolTail" "$OUT/nightly/model_runner.py" || echo "(none)"
  echo "# nightly generic slot-mapping kernel (block_table.py) load and mask:"
  grep -n "position // \|// block_size\|slot_mappings\|is_local" "$OUT/nightly/block_table.py" | head -20
} > "$OUT/nightly-slot-mapping.txt" 2>&1

echo "binding evidence written under $OUT"
