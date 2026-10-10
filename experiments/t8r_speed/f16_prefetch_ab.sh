#!/usr/bin/env bash
# A/B of the 16-bit E4M3 activation prefetch (tessera#739) on one GB10.
#
# Base arm: master's source (MASTER_SRC, default ./pb-arms/master-src), whose
# f16 library reads prefetch distance 0. New arm: this checkout's source.
# Both arms run experiments/t8r_speed/bench_pairs.py on the f16 library
# (--library e4m3) with identical deterministic inputs, forward then reverse,
# so drift cannot favor one arm. Outputs are hashed per cell for the bitwise
# verdict; wall medians give the timing ratio. An NCU capture of the one-run
# pair at M = 512 and a SASS prefetch count per arm follow.
#
# Usage: f16_prefetch_ab.sh <out_root>
# Requires ORACLE_IMAGE (the pinned serving image) and PrismaBuild admission
# only for the NCU steps' held execution; the bench steps run anywhere docker
# holds the image. AB_STEPS selects from "timed ncu sass" (default all).
# OUT and every derived bench path are absolute: bench_t8r.sh and docker
# refuse relative native paths.
set -uo pipefail
OUT=$(realpath -m "${1:?out_root}"); shift || true
MASTER_SRC=${MASTER_SRC:-$PWD/pb-arms/master-src}
MASTER_SRC=$(realpath -m "$MASTER_SRC")
STEPS=" ${AB_STEPS:-timed ncu sass} "
IMG=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
mkdir -p "$OUT"
echo "master_src=$MASTER_SRC"
echo "master_kernel_sha=$(sha256sum "$MASTER_SRC/tessera/serving/csrc/routed_fused_window.cu" | cut -d' ' -f1)"
echo "new_kernel_sha=$(sha256sum "$PWD/src/tessera/serving/csrc/routed_fused_window.cu" | cut -d' ' -f1)"
H=experiments/t8r_speed/bench_t8r.sh
ARGS=(--cases r4,r4+q,r3+q --modes 0,2 --ms 1,512,2048 --library e4m3 --warmup 10 --iters 50 --power-s 0)
FAILED=()
step() {
  local name=$1; shift
  echo "== step $name start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "== step $name rc=$rc end=$(date -u +%FT%TZ)"
  if ((rc == 0)); then tail -3 "$OUT/$name.log"; else tail -20 "$OUT/$name.log"; FAILED+=("$name:$rc"); fi
}
echo "native_container_src=${NATIVE_CONTAINER_SRC:-unset} native_container_ext=${NATIVE_CONTAINER_EXT:-unset}"
if [[ "$STEPS" == *" timed "* ]]; then
step r-base env BENCH_PY=bench_pairs.py BENCH_SRC="$MASTER_SRC" bash $H . "$OUT/base" "${ARGS[@]}"
step r-new env BENCH_PY=bench_pairs.py bash $H . "$OUT/new" "${ARGS[@]}"
step rb-new env BENCH_PY=bench_pairs.py bash $H . "$OUT/newb" "${ARGS[@]}"
step rb-base env BENCH_PY=bench_pairs.py BENCH_SRC="$MASTER_SRC" bash $H . "$OUT/baseb" "${ARGS[@]}"
fi
# NCU of the one-run pair (r4), gate/up (mode 0), M = 512, one profiled call
# per arm. Libraries are reused from the timed arms (same source and flags).
if [[ "$STEPS" == *" ncu "* ]]; then
step ncu-base env BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel BENCH_PY=bench_pairs.py \
  BENCH_SRC="$MASTER_SRC" BENCH_EXT_DIR="$OUT/base/home/torch_extensions" \
  bash $H . "$OUT/ncu-base" --cases r4 --modes 0 --ms 512 --library e4m3 --ncu
step ncu-new env BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel BENCH_PY=bench_pairs.py \
  BENCH_EXT_DIR="$OUT/new/home/torch_extensions" \
  bash $H . "$OUT/ncub-new" --cases r4 --modes 0 --ms 512 --library e4m3 --ncu
fi
# SASS: the prefetch hint count per arm's f16 library (built above). The build
# root is the retained container path when the worker sets it, else the arm's
# own torch-extensions directory; the newest library wins.
if [[ "$STEPS" == *" sass "* ]]; then
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../runtime_image.sh"
runtime_image_require "$IMG" || exit 2
for a in base new; do
  root=${NATIVE_CONTAINER_EXT:-$OUT/$a/home/torch_extensions}
  lib=$(find "$root" -name tessera_routed_fused_e4m3.so -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  if [[ -z "${lib:-}" ]]; then echo "$a: no f16 library under $root"; FAILED+=("sass-$a:missing"); continue; fi
  echo "$a: lib=$lib"
  docker run --rm --network=none --user "$(id -u):$(id -g)" -v "$OUT":"$OUT" ${NATIVE_CONTAINER_EXT:+-v "$NATIVE_CONTAINER_EXT":"$NATIVE_CONTAINER_EXT"} --entrypoint bash "$IMG" -c \
    "cuobjdump -sass '$lib' > '$OUT/sass-$a.txt'" > "$OUT/sass-$a.log" 2>&1 || FAILED+=("sass-$a:$?")
  echo "$a: prefetch.global.L1 = $(grep -c 'prefetch.global.L1' "$OUT/sass-$a.txt" 2>/dev/null || echo 0)"
done
fi
python3 - "$OUT" <<'PY'
import glob, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def cells(d):
    p = root / d / "bench_pairs.json"
    if not p.exists():
        return {}
    j = json.load(open(p))
    return {(r["case"], r["mode"], int(m)): c for r in j["results"] for m, c in r.get("cells", {}).items()}
arms = {a: cells(a) for a in ("base", "new", "newb", "baseb")}
keys = sorted(set(arms["base"]) | set(arms["new"]))
rows, mismatches = [], 0
for k in keys:
    shas = {a: (arms[a].get(k) or {}).get("out_sha256") for a in arms}
    ok = len(set(shas.values())) == 1 and None not in set(shas.values())
    mismatches += (not ok)
    r = {"case": k[0], "mode": k[1], "M": k[2], "bitwise": ok}
    for a in arms:
        w = (arms[a].get(k) or {}).get("wall", {})
        r[f"{a}_ms"] = w.get("median_ms")
    for new_a, base_a in (("new", "base"), ("newb", "baseb")):
        b, x = r[f"{base_a}_ms"], r[f"{new_a}_ms"]
        r[f"{new_a}_over_{base_a}"] = round(x / b, 4) if b and x else None
    rows.append(r)
json.dump(rows, open(root / "ab_summary.json", "w"), indent=1)
for r in rows:
    print(json.dumps(r))
print(f"cells={len(rows)} mismatches={mismatches}")
def scoreboard(d):
    hits = []
    for f in glob.glob(str(root / d / "ncu.csv")):
        with open(f, newline="") as fh:
            for line in fh:
                if "scoreboard" in line.lower():
                    hits.append(line.strip())
    return hits
for d in ("ncu-base", "ncub-new"):
    print(f"== {d} scoreboard lines:")
    for h in scoreboard(d)[:20]:
        print(f"  {h}")
PY
echo "FAILED=[${FAILED[*]:-none}]"
echo "ALL_DONE $(date -u +%FT%TZ)"
(( ${#FAILED[@]} == 0 ))
