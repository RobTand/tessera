"""Issue #12: the exact three-way decomposition of the dense 4-bit C/A residual.

Reads the two completed censuses and factors the census ratio identically:

    C/A = (P_lut16 / P_e4m3)          plane  -- Tessera's 4-bit scale index vs NVFP4's byte
        x (P_e4m3 / A_nvfp4)          gptq   -- what NVFP4's GPTQ+JSO buys over NVFP4-RTN
        x (B / P_lut16)^-1            body   -- what the trellis buys over E2M1 RTN, same plane
        x (C / B)                     comp   -- what LDLQ + the refit buys over the trellis

Every factor is a ratio of two measured `hq` numbers on the same unit and the
same Hessian, so the product is an identity, not a model: it is checked per unit
to 1e-9 and printed.  Nothing here is a render and nothing promotes anything.
"""
from __future__ import annotations
import json, math, statistics, sys

PLANE = "/mnt/shared/tessera-runs/ldlq-lut/dense4_plane_census.json"

def geo(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v)
        i = 0
        while i < len(o):
            j = i
            while j+1 < len(o) and v[o[j+1]] == v[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2.0 + 1
            i = j+1
        return r
    ra, rb = rank(a), rank(b); n = len(a)
    ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x-ma)**2 for x in ra) * sum((y-mb)**2 for y in rb))
    return num/den if den else float("nan")

d = json.load(open(PLANE))
units = [u for u in d["units"] if "census" in u]
print(f"plane census units {len(d['units'])}, joined to residual census {len(units)}")
print(f"A_check (production NVFP4 hq reproduces the residual census): "
      f"{sum(u.get('A_check') is True for u in units)}/{len(units)}")
sh = d["h_shared"]
print(f"H bit-identical q==k==v on {sum(s['H_q_eq_k'] and s['H_q_eq_v'] for s in sh)}/{len(sh)} layers, "
      f"gate==up on {sum(s['H_gate_eq_up'] for s in sh)}/{len(sh)}")
print("drift control:", all(c["same"] for c in d["control"]),
      f"({len(d['control'])} units re-run last, bit-identical)")

worst = 0.0
for u in units:
    c = u["census"]
    plane = u["plane_lut_over_e4m3"]
    gptq  = u["gptq_jso_gain"]
    body  = u["P_lut16"]["hq"] / c["B_hq"]
    comp  = c["B_hq"] / c["C_hq"]
    u["f"] = {"plane": plane, "gptq": gptq, "body": body, "comp": comp,
              "comp_over_gptq": comp / gptq}
    lhs = c["C_over_A"]; rhs = plane * gptq / (body * comp)
    worst = max(worst, abs(math.log(lhs/rhs)))
print(f"identity max |log(C/A) - log(plane*gptq/(body*comp))| = {worst:.2e}\n")

roles = ["o_proj","down_proj","up_proj","v_proj","gate_proj","q_proj","k_proj"]
hdr = (f"{'role':<10}{'n':>3}{'C/A':>8}{'plane':>8}{'gptq':>8}{'body':>7}"
       f"{'comp':>7}{'comp/gptq':>11}{'A_hq':>9}{'btwRow':>8}{'inRow':>7}{'oct':>6}")
print(hdr); print("-"*len(hdr))
tab = {}
for r in roles:
    s = [u for u in units if u["role"] == r]
    if not s: continue
    tab[r] = dict(n=len(s), CA=geo([u["census"]["C_over_A"] for u in s]),
        plane=geo([u["f"]["plane"] for u in s]), gptq=geo([u["f"]["gptq"] for u in s]),
        body=geo([u["f"]["body"] for u in s]), comp=geo([u["f"]["comp"] for u in s]),
        cg=geo([u["f"]["comp_over_gptq"] for u in s]),
        Ahq=geo([u["census"]["A_hq"] for u in s]),
        btw=statistics.median([u["field"]["between_rows_sd_log2"] for u in s]),
        inr=statistics.median([u["field"]["within_row_sd_log2"] for u in s]),
        oct=statistics.median([u["field"]["octaves"] for u in s]))
    t = tab[r]
    print(f"{r:<10}{t['n']:>3}{t['CA']:>8.4f}{t['plane']:>8.4f}{t['gptq']:>8.4f}"
          f"{t['body']:>7.4f}{t['comp']:>7.4f}{t['cg']:>11.4f}{t['Ahq']:>9.5f}"
          f"{t['btw']:>8.3f}{t['inr']:>7.3f}{t['oct']:>6.2f}")
allu = dict(CA=geo([u["census"]["C_over_A"] for u in units]),
            plane=geo([u["f"]["plane"] for u in units]), gptq=geo([u["f"]["gptq"] for u in units]),
            body=geo([u["f"]["body"] for u in units]), comp=geo([u["f"]["comp"] for u in units]),
            cg=geo([u["f"]["comp_over_gptq"] for u in units]))
print(f"{'ALL':<10}{len(units):>3}{allu['CA']:>8.4f}{allu['plane']:>8.4f}{allu['gptq']:>8.4f}"
      f"{allu['body']:>7.4f}{allu['comp']:>7.4f}{allu['cg']:>11.4f}")

print("\n== how much of log(C/A)'s spread across the 196 units each factor explains")
y = [math.log(u["census"]["C_over_A"]) for u in units]
print(f"   sd log(C/A) = {statistics.stdev(y):.4f}")
for k in ("plane","gptq","body","comp","comp_over_gptq"):
    x = [math.log(u["f"][k]) for u in units]
    print(f"   {k:<15} sd {statistics.stdev(x):.4f}  spearman vs log(C/A) {spearman(x,y):+.3f}")
for k, f in (("between_rows_sd_log2", lambda u: u["field"]["between_rows_sd_log2"]),
             ("within_row_sd_log2", lambda u: u["field"]["within_row_sd_log2"]),
             ("octaves", lambda u: u["field"]["octaves"]),
             ("crest_mean", lambda u: u["field"]["crest_mean"])):
    print(f"   {k:<15} spearman vs log(C/A) {spearman([f(u) for u in units], y):+.3f}")

print("\n== matched pairs: same layer, H bit-identical; k/v also identical shape")
byname = {u["name"]: u for u in units}
for a, b in (("k_proj","v_proj"), ("q_proj","v_proj"), ("gate_proj","up_proj")):
    rows = []
    for L in sorted({u["layer"] for u in units}):
        pa = byname.get(f"model.layers.{L}.self_attn.{a}") or byname.get(f"model.layers.{L}.mlp.{a}")
        pb = byname.get(f"model.layers.{L}.self_attn.{b}") or byname.get(f"model.layers.{L}.mlp.{b}")
        if pa and pb: rows.append((pa, pb))
    if not rows: continue
    print(f"\n  {a} vs {b}  (n={len(rows)} layers)")
    print(f"    {'factor':<14}{'geomean ratio a/b':>20}{'a>b count':>11}")
    for k in ("plane","gptq","body","comp","comp_over_gptq"):
        g = geo([pa["f"][k]/pb["f"][k] for pa, pb in rows])
        print(f"    {k:<14}{g:>20.4f}{sum(pa['f'][k]>pb['f'][k] for pa,pb in rows):>11}")
    g = geo([pa["census"]["C_over_A"]/pb["census"]["C_over_A"] for pa, pb in rows])
    print(f"    {'C/A':<14}{g:>20.4f}{sum(pa['census']['C_over_A']>pb['census']['C_over_A'] for pa,pb in rows):>11}")
