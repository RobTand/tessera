#!/usr/bin/env bash
# One PB action: the per-pair kernel instantiation A/B on one GB10, three arms
# timed interleaved (base, new, rev2; twice) with the host load recorded at
# each step's start and end.  Usage:
#   ab_pair_kernel.sh <out_root> <base_src> <rev2_src>
# <base_src> is master's Tessera src tree; <rev2_src> the two-run column map's
# revision 2 (one kernel per mode, per-item pair switch); the new arm is this
# checkout's own src.  Steps, each recorded with its rc:
#   1/2/3   base, new, rev2 routed bench (R1024 L10, R1088 L11, R832 L42;
#           M 1..2048; outputs hashed for the bitwise A/B), then 1b/2b/3b again
#   4/5     base, new: NCU of the routed launches at M 1 and 512 (rev2's is
#           the ab2 receipt's; instruction counts do not depend on the session)
#   6/7     base, new: the Tessera dense and shared-expert groups (the DENSE
#           instantiations changed too), then 6b/7b again
set -uo pipefail
OUT=${1:?out_root}; BASE_SRC=${2:?base_src}; REV2_SRC=${3:?rev2_src}
mkdir -p "$OUT"
ROUTED=experts.R1024.L10,experts.R1088.L11,experts.R832.L42
DENSE=shared_gate_up,shared_down,dense_gate_up,dense_down
MS=1,2,4,8,512,2048
step() {
  local name=$1; shift
  echo "== step $name start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "== step $name rc=$rc end=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  tail -3 "$OUT/$name.log"
}
H=experiments/t8r_speed/bench_t8r.sh
for pass in "" b; do
  step 1$pass-base-routed env BENCH_SRC="$BASE_SRC" bash $H . "$OUT/base-routed$pass" --groups "$ROUTED" --ms $MS
  step 2$pass-new-routed  bash $H . "$OUT/new-routed$pass" --groups "$ROUTED" --ms $MS
  step 3$pass-rev2-routed env BENCH_SRC="$REV2_SRC" bash $H . "$OUT/rev2-routed$pass" --groups "$ROUTED" --ms $MS
done
step 4-base-ncu env BENCH_SRC="$BASE_SRC" BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel bash $H . "$OUT/base-ncu" --groups "$ROUTED" --ms 1,512
step 5-new-ncu  env BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel bash $H . "$OUT/new-ncu" --groups "$ROUTED" --ms 1,512
for pass in "" b; do
  step 6$pass-base-dense env BENCH_SRC="$BASE_SRC" bash $H . "$OUT/base-dense$pass" --groups "$DENSE" --ms $MS
  step 7$pass-new-dense  bash $H . "$OUT/new-dense$pass" --groups "$DENSE" --ms $MS
done
python3 - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def cells(arm):
    p = root / arm / "bench_t8r.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    return {(r["group"], int(m)): c for r in d["results"] for m, c in r.get("cells", {}).items()}
def kus(c):
    return (c or {}).get("profile", {}).get("kernel_us_per_call")
arms = {a: cells(a) for a in ("base-routed", "new-routed", "rev2-routed", "base-routedb", "new-routedb",
                              "rev2-routedb", "base-dense", "new-dense", "base-denseb", "new-denseb")}
rows = []
for fam, names in (("routed", ("base", "new", "rev2")), ("dense", ("base", "new"))):
    keys = sorted(set(arms[f"base-{fam}"]) | set(arms[f"new-{fam}"]))
    for k in keys:
        r = {"family": fam, "group": k[0], "M": k[1]}
        sha = {}
        for n in names:
            for p in ("", "b"):
                c = arms[f"{n}-{fam}{p}"].get(k)
                r[f"{n}{p}_us"] = kus(c)
                r[f"{n}{p}_W"] = (c or {}).get("power", {}).get("mean_w")
                sha[f"{n}{p}"] = (c or {}).get("out_sha256")
        r["bitwise"] = sha["base"] is not None and len(set(sha.values())) == 1
        for n in names[1:]:
            for p in ("", "b"):
                b, x = r[f"base{p}_us"], r[f"{n}{p}_us"]
                r[f"{n}{p}_ratio"] = round(x / b, 4) if b and x else None
        rows.append(r)
json.dump(rows, open(root / "ab_summary.json", "w"), indent=1)
for r in rows:
    print(json.dumps({k: v for k, v in r.items() if not k.endswith("_W")}))
PY
echo "ALL_DONE $(date -u +%FT%TZ)"
