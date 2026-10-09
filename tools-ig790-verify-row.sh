#!/bin/bash
# ig790 caller byte-verification row: bridge cache-hit + caller re-read of projected units.
# One-time measurement script. Shared package and cache paths are historical inputs.
# Args: <caller-python> <out-dir> [source] [stack] [cache]
set -u
PY="$1"; OUT="$2"
SRC="${3:-/mnt/shared/models/GLM-5.3-Flash-BF16}"
STACK="${4:-model.language_model.layers.10.mlp.experts}"
CACHE="${5:-/mnt/shared/tessera-measurements/ig790-source-digest-cache}"
echo "HOST $(hostname)"
export PYTHONPATH="$PWD/src"
echo "ROW-START $(date -u +%FT%TZ)"
START=$SECONDS
"$PY" - "$OUT" "$SRC" "$STACK" "$CACHE" <<'PYEOF' &
import json, sys, time
from pathlib import Path
out = Path(sys.argv[1])
sys.path.insert(0, "/mnt/shared/tessera-measurements/ig790-pqbridge-pkg/..")
sys.path.insert(0, "/mnt/shared/tessera-measurements/ig790-pqbridge-pkg")
import os
linkroot = "/tmp/ig790-pqverify"
os.makedirs(linkroot, exist_ok=True)
link = os.path.join(linkroot, "prismaquant")
if not os.path.exists(link):
    os.symlink("/mnt/shared/tessera-measurements/ig790-pqbridge-pkg", link)
sys.path.insert(0, linkroot)
import prismaquant.tessera_expert_projection as tep
import inspect
print("bridge:", inspect.getsourcefile(tep))
print("cache opt:", getattr(tep, "SOURCE_DIGEST_CACHE_OPTION", None))
SRC, STACK, cache = sys.argv[2:5]
t0 = time.time()
proj = tep.request_expert_projection(
    SRC, {STACK: ("E4M3", 1024)},
    out_path=str(out / "pq-projection.json"), source_digest_cache=cache)
print("PQ-ELAPSED %.1f" % (time.time() - t0))
print("files:", len(proj.get("source", {}).get("files", {})))
r = proj.get("source_digest_cache")
print("producer-receipt:", json.dumps(r))
print("caller-use:", json.dumps(proj.get("source_digest_cache_use")) if proj.get("source_digest_cache_use") else "ABSENT")
# Keep the descriptor verification logic separate from row telemetry.
sys.path.insert(0, str(Path.cwd() / "tools"))
from ig790_verify import verify_projected_units
receipt = verify_projected_units(SRC, proj, STACK)
(out / "caller-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("caller-verification:", json.dumps(receipt))
PYEOF
CHILD=$!
echo "CHILD-PID $CHILD"
SAMPLE=0
while kill -0 $CHILD 2>/dev/null; do
  sleep 20
  SAMPLE=$((SAMPLE+1))
  RB=$(awk '/read_bytes/ {print $2}' /proc/$CHILD/io 2>/dev/null || echo NA)
  ELAPSED=$((SECONDS-START))
  echo "SAMPLE $SAMPLE elapsed_s=$ELAPSED child_read_bytes=$RB"
  if [ $ELAPSED -gt 1500 ]; then echo "ROW-TIMEOUT 1500s"; kill $CHILD; wait $CHILD; exit 99; fi
done
wait $CHILD; RC=$?
echo "ROW-END rc=$RC elapsed_s=$((SECONDS-START))"
nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv 2>/dev/null | head -2 || echo NO-NVIDIA-SMI
echo ROW-DONE
exit "$RC"
