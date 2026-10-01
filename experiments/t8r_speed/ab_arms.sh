#!/usr/bin/env bash
# One PB action: a kernel A/B on one GB10 over any number of arms, each from
# an immutable source snapshot under <out_root>/src-<arm>/src (the checkout
# supplies only the harness), timed interleaved in forward then reverse order
# so no arm always runs first on a cold GPU.  The first arm is the reference.
# Usage: ab_arms.sh <out_root> <arm> <arm> [<arm> ...]
# Steps, each recorded with its rc and the host load and GPU power at start:
#   r<i>/r<i>b  routed bench per arm (AB_ROUTED; default R1024 L10, R1088 L11, R832 L42; M 1..2048;
#               outputs hashed for the bitwise A/B), forward then reverse
#   n<i>        NCU of the routed launches at M 1 and 512, per arm
#   d<i>/d<i>b  the Tessera dense and shared-expert groups, forward then reverse
# ab_summary.json: per (family, group, M) each arm's kernel time and power per
# pass, the bitwise verdict over every arm and pass, and each arm's time over
# the reference arm's in the same pass.
# AB_STEPS (default "routed ncu dense") picks the steps; AB_MS (default
# 1,2,4,8,512,2048) the routed and dense benches' M; AB_NCU_MS (default 1,512)
# the NCU step's.  AB_ROUTING=<dir> adds the recorded-routing cells
# (bench_t8r.py --routing; keyed "<M>@<file>") to the routed benches.  An arm
# whose snapshot holds an ``env`` file (VAR=value lines) runs every step with
# those variables, so two arms can be one source under two settings.  A
# diagnostic ceiling arm (a load replaced by a register value; wrong output by
# design) needs only "routed".
set -uo pipefail
OUT=${1:?out_root}; shift
ARMS=("$@")
(( ${#ARMS[@]} >= 2 )) || { echo "need at least two arms" >&2; exit 2; }
for arm in "${ARMS[@]}"; do
  [[ -f "$OUT/src-$arm/src/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "missing snapshot: $OUT/src-$arm" >&2; exit 2; }
done
sha256sum "$OUT"/src-*/src/tessera/serving/csrc/routed_fused_window.cu
ROUTED=${AB_ROUTED:-experts.R1024.L10,experts.R1088.L11,experts.R832.L42}
DENSE=${AB_DENSE:-shared_gate_up,shared_down,dense_gate_up,dense_down}
# AB_BENCH_ARGS: extra bench_t8r.py words for the routed and dense benches
# (e.g. "--hash-only" for a correctness row).
read -r -a BENCH_ARGS <<< "${AB_BENCH_ARGS:-}"
MS=${AB_MS:-1,2,4,8,512,2048}
NCU_MS=${AB_NCU_MS:-1,512}
ROUTING=()
[[ -z "${AB_ROUTING:-}" ]] || ROUTING=(--routing "$AB_ROUTING")
armenv() {   # the arm's env file as VAR=value words (none: nothing)
  local f="$OUT/src-$1/env"
  [[ -f "$f" ]] && grep -E '^[A-Z_][A-Z0-9_]*=' "$f" | tr '\n' ' '
  return 0
}
STEPS=" ${AB_STEPS:-routed ncu dense} "
step() {
  local name=$1; shift
  echo "== step $name start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "== step $name rc=$rc end=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  tail -3 "$OUT/$name.log"
}
H=experiments/t8r_speed/bench_t8r.sh
bench() {   # step-name arm family suffix groups [bench args...]
  local name=$1 arm=$2 fam=$3 sfx=$4 groups=$5; shift 5
  # shellcheck disable=SC2046
  step "$name-$arm-$fam$sfx" env $(armenv "$arm") BENCH_SRC="$OUT/src-$arm/src" BENCH_EXT_DIR="$OUT/ext-$arm" \
    bash $H . "$OUT/$arm-$fam$sfx" --groups "$groups" --ms $MS "${BENCH_ARGS[@]}" "$@"
}
N=${#ARMS[@]}
if [[ $STEPS == *" routed "* ]]; then
  for ((i = 0; i < N; i++)); do bench "r$i" "${ARMS[i]}" routed "" "$ROUTED" "${ROUTING[@]}"; done
  for ((i = N - 1; i >= 0; i--)); do bench "r${i}b" "${ARMS[i]}" routed b "$ROUTED" "${ROUTING[@]}"; done
fi
if [[ $STEPS == *" ncu "* ]]; then
  for ((i = 0; i < N; i++)); do
    # shellcheck disable=SC2046
    step "n$i-${ARMS[i]}-ncu" env $(armenv "${ARMS[i]}") BENCH_SRC="$OUT/src-${ARMS[i]}/src" BENCH_NCU=1 \
      BENCH_NCU_KERNELS=routed_fused_kernel bash $H . "$OUT/${ARMS[i]}-ncu" --groups "$ROUTED" --ms $NCU_MS
  done
fi
if [[ $STEPS == *" dense "* ]]; then
  for ((i = 0; i < N; i++)); do bench "d$i" "${ARMS[i]}" dense "" "$DENSE"; done
  for ((i = N - 1; i >= 0; i--)); do bench "d${i}b" "${ARMS[i]}" dense b "$DENSE"; done
fi
python3 - "$OUT" "${ARMS[@]}" <<'PY'
import json, pathlib, sys
root, arms = pathlib.Path(sys.argv[1]), sys.argv[2:]
ref = arms[0]
def cells(name):
    p = root / name / "bench_t8r.json"
    if not p.exists():
        return {}, None
    d = json.load(open(p))
    return {(r["group"], m): c for r in d["results"] for m, c in r.get("cells", {}).items()}, d["meta"].get("kernel_sha")
data, shas = {}, {}
for fam in ("routed", "dense"):
    for a in arms:
        for p in ("", "b"):
            data[(fam, a, p)], shas[f"{a}-{fam}{p}"] = cells(f"{a}-{fam}{p}")
rows = []
for fam in ("routed", "dense"):
    keys = sorted(set().union(*(set(data[(fam, a, p)]) for a in arms for p in ("", "b"))),
                  key=lambda k: (k[0], int(k[1].split("@")[0]), k[1]))
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
        # each arm against the reference arm's forward pass, and each arm
        # against itself (forward, reverse, and a --hash-only repeat call)
        r["sha"] = {n: (v[:16] if v else None) for n, v in sha.items()}
        for a in arms:
            mine = [sha[f"{a}{p}"] for p in ("", "b")]
            mine += [(data[(fam, a, p)].get(k) or {}).get("out_sha256_repeat") for p in ("", "b")]
            mine = [v for v in mine if v]
            r[f"{a}_self_equal"] = bool(mine) and len(set(mine)) == 1
            r[f"{a}_eq_ref"] = (sha[f"{a}"] is not None and sha[f"{a}"] == sha[f"{ref}"]
                                and sha[f"{a}b"] == sha[f"{ref}b"])
        for a in arms[1:]:
            for p in ("", "b"):
                b, x = r[f"{ref}{p}_us"], r[f"{a}{p}_us"]
                r[f"{a}{p}_ratio"] = round(x / b, 4) if b and x else None
        rows.append(r)
verdict = {"rows": len(rows), "bitwise_all_arms": sum(r["bitwise"] for r in rows),
           "missing_rows": sum(bool(r["missing"]) for r in rows)}
for a in arms:
    verdict[f"{a}_self_equal"] = sum(r[f"{a}_self_equal"] for r in rows)
    verdict[f"{a}_eq_ref"] = sum(r[f"{a}_eq_ref"] for r in rows)
json.dump({"ref": ref, "arms": arms, "kernel_sha": shas, "verdict": verdict, "rows": rows},
          open(root / "ab_summary.json", "w"), indent=1)
print("VERDICT", json.dumps(verdict))
print(json.dumps(shas))
for r in rows:
    print(json.dumps({k: v for k, v in r.items() if not k.endswith("_W")}))
PY
echo "ALL_DONE $(date -u +%FT%TZ)"
