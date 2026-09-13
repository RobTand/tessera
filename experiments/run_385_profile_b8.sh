#!/bin/bash
# Real B=8 profile of E2M1_K2@896, sized from the measured capture-cost law.
#
# The law (from rungs r1 and r2, which both completed): the torch.profiler host-side event
# table for this body costs ~25.7 MiB per unit-column for the unbatched+batched pair
#   r1  128 unit-columns -> 3.68 GiB table (29.4 MiB/unit-col)
#   r2  512 unit-columns -> 12.85 GiB table (25.7 MiB/unit-col)
# It is linear, and it predicts the full span (2 units x 2048 cols = 4096 unit-columns) at
# ~103 GiB. That prediction was then confirmed the hard way: action 04fc0c8ca1a2 reserved
# 64 GiB, ran the full span, and was OOM-killed at exactly 64.00 GiB (oom_local 1, box
# psi_mem_full_avg10_max 0.0) at 376.6 s of an 1100 s deadline. Full-span capture of this
# body does not fit on a 128 GiB box; that is a principle-15 instrument gap, recorded.
#
# The same law says a REAL B=8 capture fits with room to spare, and B=8 is far closer to
# the B=32 operating point than r1/r2's effective B=2 (profile_experts caps at 8, so B=8 is
# the widest batched profile this harness can take). Ladder, safe first:
#   r4  8 units x  64 cols =  512 unit-cols -> ~13 GiB   directly comparable to r1 (64 cols)
#   r5  8 units x 256 cols = 2048 unit-cols -> ~51 GiB   directly comparable to r2 (256 cols)
# Matching the column counts matters: it makes B=1 / B=2 / B=8 a three-point curve at one
# span rather than three measurements of different things.
set -uo pipefail

BASE="${1:?base out-dir required}"
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
BENCH=experiments/tessera385_bench.py

run_rung () {
  local name="$1" cols="$2"
  echo "=== RUNG $name: experts=8 profile-batch=8 cols=$cols ===" >&2
  "$PY" - "$PY" "$BENCH" "$BASE/$name" "$cols" "$name" <<'PYEOF'
import resource, subprocess, sys
py, bench, out, cols, name = sys.argv[1:6]
rc = subprocess.call([py, "-u", bench, "--out-dir", out, "--families", "E2M1_K2@896",
                      "--expert-count", "32", "--arms", "both", "--profile", "--only-profile",
                      "--profile-experts", "8", "--profile-batch", "8",
                      "--profile-input-columns", cols, "--warmup-repeats", "1",
                      "--run-label", f"profile-2026-09-13-e2m1-{name}"])
kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(f"[mem] {name}: rc={rc} child ru_maxrss={kb / 2**20:.3f} GiB of the 64 GiB reserved",
      flush=True)
sys.exit(rc)
PYEOF
  local rc=$?; echo "[runner] rung $name rc=$rc" >&2; return $rc
}

run_rung r4-experts8-b8-cols64  64;  RC4=$?
run_rung r5-experts8-b8-cols256 256; RC5=$?
echo "[runner] rc r4=$RC4 r5=$RC5" >&2
exit "$RC4"
