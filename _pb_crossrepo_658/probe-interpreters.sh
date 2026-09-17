#!/usr/bin/env bash
# Read-only probe: which interpreters on THIS worker can import the sealed PQ
# consumer module? Nothing is admitted as a test here; this only lists facts.
set -uo pipefail
echo "host: $(hostname)"
for py in /home/rob/venvs/*/bin/python /usr/bin/python3; do
  [ -x "$py" ] || continue
  "$py" - <<PY 2>/dev/null
import sys
mods = {}
for m in ("torch", "pytest", "compressed_tensors", "safetensors", "numpy"):
    try:
        __import__(m); mods[m] = "ok"
    except Exception:
        mods[m] = "-"
print(f"{sys.executable} py={sys.version.split()[0]} " + " ".join(f"{k}={v}" for k, v in mods.items()))
PY
done
echo "--- venvs present ---"
ls -d /home/rob/venvs/*/ 2>/dev/null
