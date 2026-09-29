#!/usr/bin/env bash
# Follow-up to ab_two_path.sh, one PB action on one GB10.  Usage:
#   ab_followup.sh <out_root> <base_src>
# Steps, each recorded with its rc, the host load and the GPU power:
#   1..6 R1024 at M 1..8, base and new arms interleaved three times (the
#        one-run stack's small-M cost under the column-map change)
#   7    base, every bf16 group (attention, indexer, router, lm_head)
#   8    base, R1024 and R1088 routed at M 4096 and 8192 (the chunk-size lever)
set -uo pipefail
OUT=${1:?out_root}; BASE_SRC=${2:?base_src}
mkdir -p "$OUT"
step() {
  local name=$1; shift
  echo "== step $name start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.sm --format=csv,noheader 2>/dev/null)"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "== step $name rc=$rc end=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  tail -3 "$OUT/$name.log"
}
H=experiments/t8r_speed/bench_t8r.sh
for rep in 1 2 3; do
  step "r$rep-base" env BENCH_SRC="$BASE_SRC" bash $H . "$OUT/r1024-base-$rep" --groups experts.R1024.L10 --ms 1,2,4,8 --iters 200
  step "r$rep-new"  bash $H . "$OUT/r1024-new-$rep" --groups experts.R1024.L10 --ms 1,2,4,8 --iters 200
done
step bf16 env BENCH_SRC="$BASE_SRC" bash $H . "$OUT/base-bf16" --groups bf16 --ms 1,2,4,8,512,2048
step bigm env BENCH_SRC="$BASE_SRC" bash $H . "$OUT/base-bigm" --groups experts.R1024.L10,experts.R1088.L11 --ms 4096,8192
python3 - "$OUT" <<'PY'
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
def kus(arm, m):
    p = root / arm / "bench_t8r.json"
    if not p.exists():
        return None
    for r in json.load(open(p))["results"]:
        c = r.get("cells", {}).get(str(m))
        if c:
            return c.get("profile", {}).get("kernel_us_per_call"), c.get("out_sha256")
    return None
rows = []
for m in (1, 2, 4, 8):
    b = [kus(f"r1024-base-{i}", m) for i in (1, 2, 3)]
    n = [kus(f"r1024-new-{i}", m) for i in (1, 2, 3)]
    bt = [x[0] for x in b if x and x[0]]; nt = [x[0] for x in n if x and x[0]]
    shas = {x[1] for x in b + n if x}
    rows.append({"M": m, "base_us": bt, "new_us": nt, "bitwise": len(shas) == 1,
                 "ratio_median": statistics.median(nt) / statistics.median(bt) if bt and nt else None})
json.dump(rows, open(root / "followup_summary.json", "w"), indent=1)
for r in rows:
    print(json.dumps(r))
PY
echo "ALL_DONE $(date -u +%FT%TZ)"
