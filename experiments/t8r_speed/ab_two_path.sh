#!/usr/bin/env bash
# One PB measurement action: the T8R per-module baseline and the two-run
# column-map A/B, back to back on one GB10.  Usage:
#   ab_two_path.sh <out_root> <base_src>
# <base_src> is the base arm's Tessera src tree (the master the change is on);
# the new arm is this checkout's own src.  Steps, each recorded with its rc:
#   1 base  routed bench (R1024 L10, R1088 L11, R832 L42; M 1..2048)
#   2 new   routed bench (same cells; outputs hashed for the bitwise A/B)
#   3 base  NCU, routed, M 1 and 512 (SourceCounters)
#   4 new   NCU, same cells
#   5 base  every other group (dense, shared, bf16) for the attribution table
set -uo pipefail
OUT=${1:?out_root}; BASE_SRC=${2:?base_src}
mkdir -p "$OUT"
ROUTED=experts.R1024.L10,experts.R1088.L11,experts.R832.L42
MS=1,2,4,8,512,2048
step() {
  local name=$1; shift
  echo "== step $name start=$(date -u +%FT%TZ)"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "== step $name rc=$rc end=$(date -u +%FT%TZ)"
  tail -5 "$OUT/$name.log"
}
H=experiments/t8r_speed/bench_t8r.sh
step 1-base-routed env BENCH_SRC="$BASE_SRC" bash $H . "$OUT/base-routed" --groups "$ROUTED" --ms $MS
step 2-new-routed  bash $H . "$OUT/new-routed" --groups "$ROUTED" --ms $MS
step 3-base-ncu    env BENCH_SRC="$BASE_SRC" BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel bash $H . "$OUT/base-ncu" --groups "$ROUTED" --ms 1,512
step 4-new-ncu     env BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel bash $H . "$OUT/new-ncu" --groups "$ROUTED" --ms 1,512
REST=dense_gate_up,dense_down,shared_gate_up,shared_down,kda_qkv,kda_q,kda_o,kda_fa_ga,kda_fb,kda_b,mla_qa_kva,mla_qb,mla_kvb,mla_o,idx_wqb,idx_wk,idx_weights,router,lm_head
step 5-base-rest   env BENCH_SRC="$BASE_SRC" bash $H . "$OUT/base-rest" --groups "$REST" --ms $MS
python3 - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def cells(arm):
    p = root / arm / "bench_t8r.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    return {(r["group"], m): c for r in d["results"] for m, c in r.get("cells", {}).items()}
b, n = cells("base-routed"), cells("new-routed")
rows = []
for k in sorted(set(b) | set(n)):
    cb, cn = b.get(k, {}), n.get(k, {})
    tb = cb.get("profile", {}).get("kernel_us_per_call"); tn = cn.get("profile", {}).get("kernel_us_per_call")
    rows.append({"group": k[0], "M": int(k[1]),
                 "bitwise": cb.get("out_sha256") is not None and cb.get("out_sha256") == cn.get("out_sha256"),
                 "base_kernel_us": tb, "new_kernel_us": tn,
                 "ratio": (tn / tb) if tb and tn else None,
                 "base_wall_ms": cb.get("wall", {}).get("median_ms"), "new_wall_ms": cn.get("wall", {}).get("median_ms"),
                 "base_W": cb.get("power", {}).get("mean_w"), "new_W": cn.get("power", {}).get("mean_w")})
json.dump(rows, open(root / "ab_summary.json", "w"), indent=1)
for r in rows:
    print(json.dumps(r))
PY
echo "ALL_DONE $(date -u +%FT%TZ)"
