#!/usr/bin/env bash
# One PB action: a kernel A/B on one GB10 over any number of arms, each from
# an immutable source snapshot under <out_root>/src-<arm>/src (the checkout
# supplies only the harness), timed interleaved in forward then reverse order
# so no arm always runs first on a cold GPU.  The first arm is the reference.
# Usage: ab_arms.sh <out_root> <arm> <arm> [<arm> ...]
# Steps, each recorded with its rc and the host load and GPU power at start:
#   r<i>/r<i>b  routed bench per arm (R1024 L10, R1088 L11, R832 L42; M 1..2048;
#               outputs hashed for the bitwise A/B), forward then reverse
#   n<i>        NCU of the routed launches at M 1 and 512, per arm
#   d<i>/d<i>b  the Tessera dense and shared-expert groups, forward then reverse
# ab_summary.json: per (family, group, M) each arm's kernel time and power per
# pass, the bitwise verdict over every arm and pass, and each arm's time over
# the reference arm's in the same pass.
set -uo pipefail
OUT=${1:?out_root}; shift
ARMS=("$@")
(( ${#ARMS[@]} >= 2 )) || { echo "need at least two arms" >&2; exit 2; }
for arm in "${ARMS[@]}"; do
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
# An arm whose <out_root>/ext-<arm> exists (build_ext.sh, run as its own row off
# the measurement host) loads its libraries from there instead of compiling
# them inside this action.
extenv() { [[ -d "$OUT/ext-$1" ]] && echo "BENCH_EXT_DIR=$OUT/ext-$1"; return 0; }
bench() {   # step-name arm family suffix groups [extra env...]
  local name=$1 arm=$2 fam=$3 sfx=$4 groups=$5; shift 5
  # shellcheck disable=SC2046
  step "$name-$arm-$fam$sfx" env $(extenv "$arm") BENCH_SRC="$OUT/src-$arm/src" "$@" bash $H . "$OUT/$arm-$fam$sfx" --groups "$groups" --ms $MS
}
N=${#ARMS[@]}
for ((i = 0; i < N; i++)); do bench "r$i" "${ARMS[i]}" routed "" "$ROUTED"; done
for ((i = N - 1; i >= 0; i--)); do bench "r${i}b" "${ARMS[i]}" routed b "$ROUTED"; done
for ((i = 0; i < N; i++)); do
  # shellcheck disable=SC2046
  step "n$i-${ARMS[i]}-ncu" env $(extenv "${ARMS[i]}") BENCH_SRC="$OUT/src-${ARMS[i]}/src" BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel \
    bash $H . "$OUT/${ARMS[i]}-ncu" --groups "$ROUTED" --ms 1,512
done
for ((i = 0; i < N; i++)); do bench "d$i" "${ARMS[i]}" dense "" "$DENSE"; done
for ((i = N - 1; i >= 0; i--)); do bench "d${i}b" "${ARMS[i]}" dense b "$DENSE"; done
python3 - "$OUT" "${ARMS[@]}" <<'PY'
import json, pathlib, sys
root, arms = pathlib.Path(sys.argv[1]), sys.argv[2:]
ref = arms[0]
def cells(name):
    p = root / name / "bench_t8r.json"
    if not p.exists():
        return {}, None
    d = json.load(open(p))
    return {(r["group"], int(m)): c for r in d["results"] for m, c in r.get("cells", {}).items()}, d["meta"].get("kernel_sha")
data, shas = {}, {}
for fam in ("routed", "dense"):
    for a in arms:
        for p in ("", "b"):
            data[(fam, a, p)], shas[f"{a}-{fam}{p}"] = cells(f"{a}-{fam}{p}")
rows = []
for fam in ("routed", "dense"):
    keys = sorted(set().union(*(set(data[(fam, a, p)]) for a in arms for p in ("", "b"))))
    for k in keys:
        r = {"family": fam, "group": k[0], "M": k[1]}
        sha = {}
        for a in arms:
            for p in ("", "b"):
                c = data[(fam, a, p)].get(k) or {}
                r[f"{a}{p}_us"] = c.get("profile", {}).get("kernel_us_per_call")
                r[f"{a}{p}_W"] = c.get("power", {}).get("mean_w")
                sha[f"{a}{p}"] = c.get("out_sha256")
        r["bitwise"] = None not in sha.values() and len(set(sha.values())) == 1
        r["missing"] = sorted(n for n, v in sha.items() if v is None)
        for a in arms[1:]:
            for p in ("", "b"):
                b, x = r[f"{ref}{p}_us"], r[f"{a}{p}_us"]
                r[f"{a}{p}_ratio"] = round(x / b, 4) if b and x else None
        rows.append(r)
json.dump({"ref": ref, "arms": arms, "kernel_sha": shas, "rows": rows}, open(root / "ab_summary.json", "w"), indent=1)
print(json.dumps(shas))
for r in rows:
    print(json.dumps({k: v for k, v in r.items() if not k.endswith("_W")}))
PY
echo "ALL_DONE $(date -u +%FT%TZ)"
