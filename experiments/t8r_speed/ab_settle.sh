#!/usr/bin/env bash
# One PB action: the chunk loop's load-settle A/B on one GB10.  Three arms,
# each from an immutable source snapshot under <out_root>/src-<arm>/src (the
# checkout is read only for these scripts), timed interleaved in forward then
# reverse order so neither arm always runs first on a cold GPU:
#   base    master's kernel (one kernel per mode, per-item pair switch)
#   pair    the per-pair kernel (the run pair a template parameter)
#   settle  the per-pair kernel with the first chunk's loads settled before
#           the chunk loop
# Usage: ab_settle.sh <out_root>   (the snapshots must already be in place)
# Steps, each recorded with its rc and the host load at start and end:
#   1-3     base, pair, settle routed bench (R1024 L10, R1088 L11, R832 L42;
#           M 1..2048; outputs hashed for the bitwise A/B), then 3b-1b reversed
#   4-6     base, pair, settle: NCU of the routed launches at M 1 and 512
#   7/8     base, settle: the Tessera dense and shared-expert groups, then 8b/7b
set -uo pipefail
OUT=${1:?out_root}
for arm in base pair settle; do
  [[ -f "$OUT/src-$arm/src/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "missing snapshot: $OUT/src-$arm" >&2; exit 2; }
done
sha256sum "$OUT"/src-*/src/tessera/serving/csrc/routed_fused_window.cu
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
routed() { step "$1-$2-routed$3" env BENCH_SRC="$OUT/src-$2/src" bash $H . "$OUT/$2-routed$3" --groups "$ROUTED" --ms $MS; }
dense() { step "$1-$2-dense$3" env BENCH_SRC="$OUT/src-$2/src" bash $H . "$OUT/$2-dense$3" --groups "$DENSE" --ms $MS; }
routed 1 base ""; routed 2 pair ""; routed 3 settle ""
routed 3b settle b; routed 2b pair b; routed 1b base b
for i in "4 base" "5 pair" "6 settle"; do
  set -- $i
  step "$1-$2-ncu" env BENCH_SRC="$OUT/src-$2/src" BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel bash $H . "$OUT/$2-ncu" --groups "$ROUTED" --ms 1,512
done
dense 7 base ""; dense 8 settle ""
dense 8b settle b; dense 7b base b
python3 - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def cells(arm):
    p = root / arm / "bench_t8r.json"
    if not p.exists():
        return {}, None
    d = json.load(open(p))
    return {(r["group"], int(m)): c for r in d["results"] for m, c in r.get("cells", {}).items()}, d["meta"].get("kernel_sha")
arms, shas = {}, {}
for fam in ("routed", "dense"):
    for n in ("base", "pair", "settle"):
        for p in ("", "b"):
            arms[f"{n}-{fam}{p}"], shas[f"{n}-{fam}{p}"] = cells(f"{n}-{fam}{p}")
rows = []
for fam, names in (("routed", ("base", "pair", "settle")), ("dense", ("base", "settle"))):
    keys = sorted(set(arms[f"base-{fam}"]) | set(arms[f"settle-{fam}"]))
    for k in keys:
        r = {"family": fam, "group": k[0], "M": k[1]}
        sha = {}
        for n in names:
            for p in ("", "b"):
                c = arms[f"{n}-{fam}{p}"].get(k)
                r[f"{n}{p}_us"] = (c or {}).get("profile", {}).get("kernel_us_per_call")
                r[f"{n}{p}_W"] = (c or {}).get("power", {}).get("mean_w")
                sha[f"{n}{p}"] = (c or {}).get("out_sha256")
        r["bitwise"] = sha["base"] is not None and len(set(sha.values())) == 1
        for n in names[1:]:
            for p in ("", "b"):
                b, x = r[f"base{p}_us"], r[f"{n}{p}_us"]
                r[f"{n}{p}_ratio"] = round(x / b, 4) if b and x else None
        rows.append(r)
json.dump({"kernel_sha": shas, "rows": rows}, open(root / "ab_summary.json", "w"), indent=1)
print(json.dumps(shas))
for r in rows:
    print(json.dumps({k: v for k, v in r.items() if not k.endswith("_W")}))
PY
echo "ALL_DONE $(date -u +%FT%TZ)"
