"""Markdown tables for the window-GEMV receipt, straight from the bench JSON.

    python experiments/gemv_receipt_tables.py --dir /mnt/shared/tessera-runs/gemv \
        --gemv 'bench_gemv_v2_*.json' --plans 'bench_plans_v2plans_*.json' \
        --ablate 'bench_ablate_v2_*.json' --power 'bench_power_v2_*.json' \
        --ncu 'ncu_v2_{shape}_M1.txt'

Every row carries the box state it was measured under (other CUDA
processes, mean power, SM clock) because the box was shared.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

SPEC = 273.0

# Qwen3-4B Linear list: shape -> occurrences per layer (x 36 layers); the
# bench's MODELS table is the source of truth, this mirrors it for weighting.
QWEN3_4B_WEIGHTS = {"4096x2560": 36, "2560x4096": 36, "9728x2560": 36 * 2, "2560x9728": 36, "1024x2560": 36 * 2}


def _load(d, pat):
    if not pat:
        return None
    hits = sorted(glob.glob(os.path.join(d, pat)))
    if not hits:
        raise SystemExit(f"no match for {pat} in {d}")
    return json.load(open(hits[-1]))


def _state(r):
    return f"{r.get('cuda_procs_min', '?')}-{r.get('cuda_procs_max', '?')} / {r.get('mean_w', '?')} W / {r.get('sm_mhz_mean', '?')} MHz"


def gemv_tables(g, read_key="wire_read"):
    out = []
    shapes = g["gemv"]
    for M in sorted({m for s in shapes.values() for m in s["M"]}, key=int):
        out.append(f"\n### M={M}: per shape (us, min over interleaved rounds)\n")
        out.append("| shape (out x in) | wire MB | fused kernel | fused op | fp8 lane (quant+mm) | fp8 mm only | bf16 | wire read | kernel GB/s | /273 | /read | box: procs / W / MHz |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, s in shapes.items():
            r = s["M"].get(M)
            if not r:
                continue
            k = r["fused_kernel"]

            def us(a, r=r):
                return f"{r[a]['us']:.1f}" if a in r else "-"

            out.append(f"| {name} | {s['wire_bytes'] / 1e6:.2f} | {us('fused_kernel')} | {us('fused_op')} | {us('fp8_lane_quant_plus_mm')} | {us('fp8_mm_only')} | {us('bf16_linear')} | {us(read_key)} | {k.get('GB_per_s', '-')} | {k.get('frac_of_273', '-')} | {k.get('frac_of_wire_read', '-')} | {_state(k)} |")
    out.append("\n### Per token (us summed over each model's Linear list x layers)\n")
    out.append("| model | M | fused kernel | fused op | fp8 lane | fp8 mm only | bf16 | wire read | op / fp8 lane | kernel / fp8 lane | kernel / mm only | op / bf16 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model, byM in g["totals"].items():
        for M, t in byM.items():
            out.append(f"| {model} | {M} | {t['fused_kernel']:.0f} | {t['fused_op']:.0f} | {t['fp8_lane_quant_plus_mm']:.0f} | {t['fp8_mm_only']:.0f} | {t['bf16_linear']:.0f} | {t['wire_read']:.0f} | **{t['speedup_op_vs_fp8_lane']:.3f}x** | {t['speedup_kernel_vs_fp8_lane']:.3f}x | {t['speedup_kernel_vs_mm_only']:.3f}x | {t['speedup_op_vs_bf16']:.2f}x |")
    return "\n".join(out)


def plans_tables(p, top=6, weights=None):
    out = []
    for shape, s in p["plans"].items():
        rows = sorted(s["rows"].items(), key=lambda kv: kv[1]["us"])
        default = s.get("default")
        dflt = f" = {s['rows'][default]['us']:.1f} us" if default in s["rows"] else ""
        out.append(f"\n**{shape}** ({len(rows)} plans; default `{default}`{dflt}; box {_state(rows[0][1])})\n")
        out.append("| plan | us | GB/s | items |")
        out.append("|---|---|---|---|")
        for name, r in rows[:top]:
            out.append(f"| `{name}` | {r['us']:.1f} | {r.get('GB_per_s', '-')} | {r.get('items', '-')} |")
    if weights:
        shapes = [sh for sh in weights if sh in p["plans"]]
        names = set.intersection(*[set(p["plans"][sh]["rows"]) for sh in shapes])
        tot = {n: sum(p["plans"][sh]["rows"][n]["us"] * weights[sh] for sh in shapes) for n in names}
        best = sorted(tot.items(), key=lambda kv: kv[1])
        out.append(f"\n**Weighted over the Qwen3-4B list ({', '.join(shapes)}; us per token)**\n")
        out.append("| plan | per token us |")
        out.append("|---|---|")
        for n, t in best[:10]:
            out.append(f"| `{n}` | {t:.0f} |")
        for sh, s in p["plans"].items():
            d = s.get("default")
            if d in tot:
                out.append(f"| default `{d}` | {tot[d]:.0f} |")
                break
    return "\n".join(out)


def ablate_tables(a):
    out = ["\n| shape | kernel | no gather | no wire read | no FMA | neither read | box |", "|---|---|---|---|---|---|---|"]
    for shape, s in a["ablate"].items():
        def c(k, s=s):
            r = s.get(k)
            return f"{r['us']:.1f} ({r['delta_vs_kernel']:+.0%})" if r else "-"

        out.append(f"| {shape} | {s['kernel']['us']:.1f} | {c('no_gather')} | {c('no_wire_read')} | {c('no_fma')} | {c('no_reads')} | {_state(s['kernel'])} |")
    return "\n".join(out)


def power_table(pw):
    out = ["\n| shape | us (2000 back-to-back) | GB/s | mean W | max W | MHz | procs |", "|---|---|---|---|---|---|---|"]
    for shape, r in pw["power"].items():
        out.append(f"| {shape} | {r['us']:.1f} | {r['GB_per_s']} | {r.get('mean_w', '?')} | {r.get('max_w', '?')} | {r.get('sm_mhz_mean', '?')} | {r.get('cuda_procs_min', '?')}-{r.get('cuda_procs_max', '?')} |")
    return "\n".join(out)


NCU_KEYS = ["Duration", "Registers Per Thread", "Theoretical Occupancy", "Achieved Occupancy", "Waves Per SM",
            "Memory Throughput", "Compute (SM) Throughput", "L1/TEX Cache Throughput", "L2 Cache Throughput",
            "No Eligible", "Block Limit Registers", "Block Limit Shared Mem", "Dynamic Shared Memory Per Block"]


def ncu_table(d, pat, shapes):
    out = ["\n| metric | " + " | ".join(shapes) + " |", "|---|" + "---|" * len(shapes)]
    vals = {}
    for sh in shapes:
        path = os.path.join(d, pat.format(shape=sh))
        vals[sh] = {}
        if not os.path.exists(path):
            continue
        for line in open(path, errors="replace"):
            for k in NCU_KEYS:
                m = re.match(rf"\s+{re.escape(k)}\s+(\S+)\s+([\d.,]+)\s*$", line)
                if m and k not in vals[sh]:
                    vals[sh][k] = f"{m.group(2)} {m.group(1)}"
    for k in NCU_KEYS:
        out.append(f"| {k} | " + " | ".join(vals[sh].get(k, "-") for sh in shapes) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gemv")
    ap.add_argument("--plans")
    ap.add_argument("--ablate")
    ap.add_argument("--power")
    ap.add_argument("--ncu", help="pattern with {shape}")
    ap.add_argument("--ncu-shapes", default="1024x2560,2560x4096,2560x9728,4096x2560,9728x2560")
    a = ap.parse_args()
    if a.gemv:
        print(gemv_tables(_load(a.dir, a.gemv)))
    if a.plans:
        print(plans_tables(_load(a.dir, a.plans), weights=QWEN3_4B_WEIGHTS))
    if a.ablate:
        print(ablate_tables(_load(a.dir, a.ablate)))
    if a.power:
        print(power_table(_load(a.dir, a.power)))
    if a.ncu:
        print(ncu_table(a.dir, a.ncu, a.ncu_shapes.split(",")))


if __name__ == "__main__":
    main()
