"""Issue #12: why `q_proj` and `k_proj`, and not the other five roles.

The 196-Linear census (``dense4_residual_census.py``,
``docs/measurements/tessera-bf16-gauge-and-dense4-residual-2026-09-02.md``)
put the dense 4-bit weight-leg residual entirely in ``q_proj`` (1.0648) and
``k_proj`` (1.0976) on ``hq``, with the other five roles winning.  A role
label is not a mechanism.  This is the isolation.

**The natural experiment the model hands us.**  Within one layer ``q_proj``,
``k_proj`` and ``v_proj`` read the *same* hidden state, so their Hessians are
**bit-identical** (checked here, printed, not assumed), and ``k_proj`` and
``v_proj`` have the *identical shape*.  Same H, same shape, same rate, same
alphabet -- and the census says one loses and the other wins.  Whatever the
mechanism is, it is a property of ``W`` and of nothing else.  ``gate_proj`` /
``up_proj`` is the second such pair.

**The one thing the two arms' wires do not share.**  Tessera's E2M1x2 q896
wire carries ``ScalePlaneKind.LUT``: *sixteen* distinct E4M3 scales for the
whole unit, one 4-bit index per 16-column half, times an fp32 global.
Production NVFP4 carries one *full* E4M3 byte per 16-column half.  Both plane
targets are the same field -- ``amax_16 / 6``, which is ``_pack_scales_lut``'s
``target`` and NVFP4's block scale, the same quantity under two names -- so
the planes are directly comparable and the Tessera one is 4 bits per half
cheaper (0.25 bpp, most of the 4.00-vs-4.50 gap).

So the arms below hold **the alphabet, the rounding and the weights fixed**
and change **only the plane**:

* ``exact``  -- fp32 ``amax_16/6`` per half.  The plane that costs nothing.
* ``e4m3``   -- nearest E4M3 per half, one byte: **NVFP4's plane**.
* ``lut16``  -- ``encode._pack_scales_lut``, the encoder's own fitter:
  **Tessera's plane**.

E2M1 nearest-value rounding on every arm (the alphabet NVFP4 deploys and the
one E2M1x2 tuples), no GPTQ, no LDLQ, no refit, no trellis.  This is not a
render and cannot promote anything; it is a *predictor*, and its job is to say
whether the plane is where ``q``/``k`` differ from ``v``.

``e4m3`` doubles as the **NVFP4-RTN** arm the census lacked, which separates
"NVFP4's *format* is better on q/k" from "NVFP4's *GPTQ+JSO* is better on
q/k".  ``A_rtn/A >= 1`` on every unit is the arm's own sanity gate: a
violation means the RTN arm is wrong, not that compensation hurt.

Metric: ``hq = sqrt(E H E^T / W H W^T)`` -- the census's metric, on the census's
Hessian, so the numbers join.  Plain relative Frobenius carried alongside.

Run with ``PYTHONPATH=src TMPDIR=/home/rob/tmp
TRITON_CACHE_DIR=/home/rob/.triton-cache`` under the prismaquant-cu130 python.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import time

import torch
from safetensors import safe_open

from tessera.encode import _pack_scales_lut, e4m3_positive_values

SRC = "/home/rob/models/Qwen3-0.6B"
NVFP4 = "/home/rob/dq-runs/fc45-0p6b-nvfp4/exported"
HFULL = "/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt"
CENSUS = "/mnt/shared/tessera-runs/ldlq-lut/dense4_residual_census.json"
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
HALF = 16
PEAK = 6.0


def open_all(d):
    idx = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        h = safe_open(f, framework="pt")
        for k in h.keys():
            idx[k] = h
    return idx


def get(idx, key, device):
    return idx[key].get_tensor(key).to(device)


def hq(W, Wd, H):
    E = (Wd - W).float()
    return math.sqrt(max(float((E @ H * E).sum()), 0.0) / float((W @ H * W).sum()))


def rel(W, Wd):
    E = (Wd - W).float()
    return math.sqrt(float((E * E).sum()) / float((W * W).sum()))


def e2m1_rtn(W, s):
    """Nearest E2M1 value per weight given a per-half scale ``s`` [halves]."""
    vals = torch.tensor(E2M1_VALUES, device=W.device, dtype=torch.float32)
    halves = W.reshape(-1, HALF)
    x = halves / s[:, None]
    q = vals[(x.abs()[..., None] - vals).abs().argmin(dim=-1)] * torch.sign(x)
    return (q * s[:, None]).reshape(W.shape)


def nvfp4_export_arm(idx, name, rows, cols, device):
    """Production NVFP4, dequantised from its own bytes and priced from them."""
    pk = get(idx, name + ".weight_packed", device)
    s = get(idx, name + ".weight_scale", device).float()
    g = get(idx, name + ".weight_global_scale", device).float()
    vals = torch.tensor(E2M1_VALUES, device=device)
    lo, hi = (pk & 0xF).long(), (pk >> 4).long()

    def dq(n):
        return vals[n & 7] * torch.where(n >= 8, -1.0, 1.0)

    w = torch.stack([dq(lo), dq(hi)], -1).reshape(rows, cols)
    w = w * (s / g).repeat_interleave(HALF, dim=1)
    bits = pk.numel() * 8 + s.numel() * 8 + g.numel() * 32
    return w, bits / (rows * cols)


def geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    blob = torch.load(HFULL, map_location="cpu")
    Hs = blob["H"]
    census = {u["name"]: u for u in json.load(open(CENSUS))["units"]}
    src, nv = open_all(SRC), open_all(NVFP4)
    names = sorted(Hs)
    if args.units:
        names = names[: args.units]
    print(f"{len(names)} units; census join {len(census)}", flush=True)

    # The natural experiment, checked rather than assumed.
    shared = []
    for L in sorted({int(n.split(".")[2]) for n in names}):
        q = Hs.get(f"model.layers.{L}.self_attn.q_proj")
        k = Hs.get(f"model.layers.{L}.self_attn.k_proj")
        v = Hs.get(f"model.layers.{L}.self_attn.v_proj")
        g = Hs.get(f"model.layers.{L}.mlp.gate_proj")
        u = Hs.get(f"model.layers.{L}.mlp.up_proj")
        if q is None:
            continue
        shared.append({"layer": L, "H_q_eq_k": bool(torch.equal(q, k)),
                       "H_q_eq_v": bool(torch.equal(q, v)),
                       "H_gate_eq_up": bool(torch.equal(g, u))})
    nqk = sum(s["H_q_eq_k"] and s["H_q_eq_v"] for s in shared)
    ngu = sum(s["H_gate_eq_up"] for s in shared)
    print(f"H bit-identical: q==k==v on {nqk}/{len(shared)} layers, "
          f"gate==up on {ngu}/{len(shared)}", flush=True)

    rows_out, t0, first = [], time.time(), {}
    for i, name in enumerate(names):
        W = get(src, name + ".weight", dev).to(torch.float32).contiguous()
        r, c = W.shape
        H = Hs[name].to(dev, torch.float32)
        rec = {"name": name, "rows": r, "cols": c,
               "role": name.split(".")[-1], "layer": int(name.split(".")[2])}

        # --- the plane target field, the encoder's own definition
        t = W.reshape(-1, HALF).abs().amax(dim=1).clamp_min(1e-30) / PEAK

        # exact plane
        w_ex = e2m1_rtn(W, t)
        # NVFP4's plane: one E4M3 byte per half, two-level with an fp32 global
        gnv = float(t.max()) / 448.0
        grid = e4m3_positive_values(dev)
        s_nv = grid[(t[:, None] / gnv - grid[None, :]).abs().argmin(dim=1)] * gnv
        w_nv = e2m1_rtn(W, s_nv)
        # Tessera's plane: the encoder's own sixteen-entry fitter
        _tb, _ix, s_lut, _gs = _pack_scales_lut(W, HALF)
        w_lut = e2m1_rtn(W, s_lut)

        for tag, wq in (("exact", w_ex), ("e4m3", w_nv), ("lut16", w_lut)):
            rec[f"P_{tag}"] = {"hq": hq(W, wq, H), "rel": rel(W, wq)}
        rec["plane_lut_over_e4m3"] = rec["P_lut16"]["hq"] / rec["P_e4m3"]["hq"]
        rec["plane_e4m3_over_exact"] = rec["P_e4m3"]["hq"] / rec["P_exact"]["hq"]
        rec["plane_lut_over_exact"] = rec["P_lut16"]["hq"] / rec["P_exact"]["hq"]

        # --- how wide a field the sixteen entries must cover
        lg = torch.log2(t)
        tb = t.reshape(r, c // HALF)
        rowmed = tb.median(dim=1).values
        rec["field"] = {
            "octaves": float(lg.max() - lg.min()),
            "sd_log2": float(lg.std()),
            "between_rows_sd_log2": float(torch.log2(rowmed).std()),
            "within_row_sd_log2": float(torch.log2(tb / rowmed[:, None]).std()),
            "crest_mean": float((W.reshape(-1, HALF).abs().amax(dim=1)
                                 / W.reshape(-1, HALF).pow(2).mean(dim=1).sqrt()
                                 .clamp_min(1e-30)).mean()),
        }

        # --- production NVFP4, for the compensation split
        wa, bpp_a = nvfp4_export_arm(nv, name, r, c, dev)
        rec["A_nvfp4"] = {"bpp": bpp_a, "hq": hq(W, wa, H)}
        rec["gptq_jso_gain"] = rec["P_e4m3"]["hq"] / rec["A_nvfp4"]["hq"]

        cu = census.get(name)
        if cu:
            rec["census"] = {"A_hq": cu["A_nvfp4"]["hq"],
                             "B_hq": cu["B_weights_only"]["hq"],
                             "C_hq": cu["C_h_aware"]["hq"],
                             "C_over_A": cu["ratio_C_over_A"],
                             "B_over_A": cu["ratio_B_over_A"]}
            rec["A_check"] = abs(rec["A_nvfp4"]["hq"] - cu["A_nvfp4"]["hq"]) \
                <= 1e-6 * max(1.0, cu["A_nvfp4"]["hq"])
        first.setdefault(name, rec["P_lut16"]["hq"])
        rows_out.append(rec)
        print(f"[{i+1:3d}/{len(names)}] {name:<44} {r}x{c} "
              f"exact {rec['P_exact']['hq']:.5f} e4m3 {rec['P_e4m3']['hq']:.5f} "
              f"lut16 {rec['P_lut16']['hq']:.5f} | lut/e4m3 "
              f"{rec['plane_lut_over_e4m3']:.4f} | oct {rec['field']['octaves']:.2f} "
              f"| gptq {rec['gptq_jso_gain']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        del W, H, wa, w_ex, w_nv, w_lut
        if dev == "cuda":
            torch.cuda.empty_cache()
        with open(args.out, "w") as fh:
            json.dump({"units": rows_out, "h_shared": shared, "partial": True}, fh, indent=1)

    # drift control: the same arm, last, same process
    print("\n== repeat control (lut16 plane, re-run last)", flush=True)
    control = []
    for name in names[: args.repeat]:
        W = get(src, name + ".weight", dev).to(torch.float32).contiguous()
        H = Hs[name].to(dev, torch.float32)
        _tb, _ix, s_lut, _gs = _pack_scales_lut(W, HALF)
        again = hq(W, e2m1_rtn(W, s_lut), H)
        ok = abs(again - first[name]) <= 1e-9 * max(1.0, first[name])
        control.append({"name": name, "first": first[name], "again": again, "same": ok})
        print(f"   {name:<44} {first[name]:.8f} -> {again:.8f} "
              f"{'SAME' if ok else '!! DIFFER'}", flush=True)
        del W, H

    roles = sorted({r["role"] for r in rows_out})
    summary = {"by_role": {}}
    for role in roles:
        sub = [r for r in rows_out if r["role"] == role]
        summary["by_role"][role] = {
            "n": len(sub),
            "plane_lut_over_e4m3": geomean([r["plane_lut_over_e4m3"] for r in sub]),
            "plane_e4m3_over_exact": geomean([r["plane_e4m3_over_exact"] for r in sub]),
            "gptq_jso_gain": geomean([r["gptq_jso_gain"] for r in sub]),
            "octaves": statistics.median([r["field"]["octaves"] for r in sub]),
            "between_rows_sd": statistics.median(
                [r["field"]["between_rows_sd_log2"] for r in sub]),
            "within_row_sd": statistics.median(
                [r["field"]["within_row_sd_log2"] for r in sub]),
            "crest": statistics.median([r["field"]["crest_mean"] for r in sub]),
            "census_C_over_A": geomean(
                [r["census"]["C_over_A"] for r in sub if "census" in r]),
        }
    print(f"\n{'role':<11} {'n':>3} {'lut/e4m3':>9} {'e4m3/exact':>11} {'gptq':>7} "
          f"{'oct':>6} {'btwRow':>7} {'inRow':>7} {'crest':>6} {'C/A':>7}")
    for role, s in summary["by_role"].items():
        print(f"{role:<11} {s['n']:>3} {s['plane_lut_over_e4m3']:>9.4f} "
              f"{s['plane_e4m3_over_exact']:>11.4f} {s['gptq_jso_gain']:>7.3f} "
              f"{s['octaves']:>6.2f} {s['between_rows_sd']:>7.3f} "
              f"{s['within_row_sd']:>7.3f} {s['crest']:>6.3f} "
              f"{s['census_C_over_A']:>7.4f}", flush=True)

    # matched pairs: identical H, identical shape for (k,v); identical H for (gate,up)
    pairs = []
    byname = {r["name"]: r for r in rows_out}
    for L in sorted({r["layer"] for r in rows_out}):
        for a, b in (("self_attn.k_proj", "self_attn.v_proj"),
                     ("self_attn.q_proj", "self_attn.v_proj"),
                     ("mlp.gate_proj", "mlp.up_proj")):
            ra = byname.get(f"model.layers.{L}.{a}")
            rb = byname.get(f"model.layers.{L}.{b}")
            if not ra or not rb or "census" not in ra or "census" not in rb:
                continue
            pairs.append({
                "layer": L, "pair": f"{a.split('.')[-1]}-{b.split('.')[-1]}",
                "d_log_CA": math.log(ra["census"]["C_over_A"] / rb["census"]["C_over_A"]),
                "d_log_plane": math.log(ra["plane_lut_over_e4m3"] / rb["plane_lut_over_e4m3"]),
                "d_octaves": ra["field"]["octaves"] - rb["field"]["octaves"],
                "d_between": ra["field"]["between_rows_sd_log2"] - rb["field"]["between_rows_sd_log2"],
                "d_within": ra["field"]["within_row_sd_log2"] - rb["field"]["within_row_sd_log2"],
                "d_crest": ra["field"]["crest_mean"] - rb["field"]["crest_mean"],
                "d_log_gptq": math.log(ra["gptq_jso_gain"] / rb["gptq_jso_gain"]),
            })
    print("\n== matched pairs (same layer; H bit-identical; k/v same shape)")
    print(f"{'pair':<16} {'n':>3} {'dlogC/A>0':>10} {'dlogPlane>0':>12} "
          f"{'agree':>7} {'med dOct':>9} {'med dBtw':>9} {'med dlogGPTQ':>13}")
    for tag in sorted({p["pair"] for p in pairs}):
        sub = [p for p in pairs if p["pair"] == tag]
        agree = sum((p["d_log_CA"] > 0) == (p["d_log_plane"] > 0) for p in sub)
        print(f"{tag:<16} {len(sub):>3} "
              f"{sum(p['d_log_CA'] > 0 for p in sub):>10} "
              f"{sum(p['d_log_plane'] > 0 for p in sub):>12} "
              f"{agree:>7} {statistics.median([p['d_octaves'] for p in sub]):>9.3f} "
              f"{statistics.median([p['d_between'] for p in sub]):>9.3f} "
              f"{statistics.median([p['d_log_gptq'] for p in sub]):>13.4f}", flush=True)
    summary["pairs"] = pairs

    with open(args.out, "w") as fh:
        json.dump({"units": rows_out, "control": control, "summary": summary,
                   "h_shared": shared, "args": vars(args), "partial": False},
                  fh, indent=1)
    print(f"\nwrote {args.out} [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
