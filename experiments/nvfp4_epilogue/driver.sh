#!/usr/bin/env bash
# Inside the bench container, after full_engine_plugin_install.py exec'd us.
# The six prepared cells are copied out of the read-only prep mount so the bench
# resolves the same bytes PrismaQuant prepared for PQ #563/#573.
set -eu
ROWS=${1:?comma-separated M values}
ARMS=${2:?comma-separated bench-local arms}
BASELINE=${3:-}
POWER_S=${4:-5.0}
export PYTHONDONTWRITEBYTECODE=1

python3 -c "import tessera, sys; print('tessera', tessera.__file__, file=sys.stderr)"
mkdir -p /out/cells /out/profile
for unit in gate_proj up_proj down_proj; do
  for fmt in TESSERA_E2M1_K2_R896 TESSERA_E4M3_K1_R1006; do
    cell=model.layers.0.mlp.${unit}__${fmt}
    [ -d "/prep/cells/$cell" ] || { echo "MISSING CELL $cell" >&2; exit 3; }
    mkdir -p "/out/cells/$cell"
    cp "/prep/cells/$cell/request.json" "/prep/cells/$cell/inputs.json" "/prep/cells/$cell/wire-record.json" \
       "/prep/cells/$cell/weight.tessera" "/prep/cells/$cell/tensors.safetensors" "/out/cells/$cell/"
  done
done

BASELINE_FLAG=()
if [ -n "$BASELINE" ]; then BASELINE_FLAG=(--baseline-route "/tessera/$BASELINE"); fi

echo "=== BENCH start $(date -u +%FT%TZ) rows=$ROWS arms=$ARMS baseline=${BASELINE:-installed}"
PYTHONPATH=/tessera python3 -u /tessera/experiments/nvfp4_epilogue/bench_epilogue_fold.py \
  --cells-root /out/cells --rows "$ROWS" --arms "$ARMS" --out /out/bench.json \
  --power-seconds "$POWER_S" --profile-dir /out/profile "${BASELINE_FLAG[@]}"
python3 - <<'EOF'
import hashlib, importlib.util, json
from pathlib import Path
stock = json.loads(Path("/mnt/shared/tessera-runs/receipts/399-qwen3-0.6b-20260913/observer-build/runtime-inventory.json").read_text())
core = Path(importlib.util.find_spec("vllm").origin).parent
def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
files = {str(p.relative_to(core)): {"sha256": digest(p), "bytes": p.stat().st_size}
         for p in sorted(core.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
same = files == stock["files"]
Path("/out/post-bench-core.json").write_text(json.dumps({"core_files_unchanged": same, "files": len(files)}) + "\n")
print("post-bench vLLM core unchanged:", same, flush=True)
raise SystemExit(0 if same else 4)
EOF
echo "=== BENCH COMPLETE $(date -u +%FT%TZ)"
