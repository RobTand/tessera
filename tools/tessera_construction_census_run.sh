#!/usr/bin/env bash
# Take a construction census inside the pinned serving image, as a PrismaBuild
# action run from a snapshot of this tree (the snapshot is $PWD).
#
# Usage: tools/tessera_construction_census_run.sh <model-or-config-dir> <out-dir> <receipt-name.json>
#
# The model directory needs only config.json and the tokenizer files; nothing
# is read on the meta device.  The receipt lands root-owned under <out-dir>
# (the container runs as root; the tree is mounted read-only) and is copied
# into docs/measurements/construction/ by the caller, then
# tools/tessera_update_construction_block.py regenerates the contract block.
set -euo pipefail
model=$1; out=$2; name=$3
export TS="$PWD"
export RUNS="${RUNS:-$out/runs}" EXT="${EXT:-$out/ext}"
mkdir -p "$out" "$EXT"
exec experiments/tessera_plugin_run.sh --network none \
  -v "$model":/model:ro -v "$out":/census \
  -- "python3 tools/tessera_construction_census.py /model /census/$name --device meta \
&& python3 - <<PY
import json
d = json.load(open('/census/$name'))
print('runtime', d['runtime'])
for r in d['linears']:
    print(r['prefix_pattern'], r['class'], r.get('output_sizes'))
PY"
