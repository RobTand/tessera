"""Issue #12: is the q/k residual structural, or is it hq's row weighting?

``dense4_plane_census.py`` + ``dense4_residual_census.py`` factor the census
ratio exactly (checked to 4e-16 on all 196 units):

    C/A = plane x gptq / (body x comp)

and on the H-matched pairs the ordering is carried by ``gptq`` -- NVFP4's
GPTQ+JSO gains 11.1% more on ``k_proj`` than on ``v_proj``, 28 layers of 28 --
not by Tessera's own compensation, which is flat (1.004x).  ``q_proj`` and
``k_proj`` are also the two roles with by far the widest *between-row* block
scale spread (median sd log2 0.567 / 0.479 against ``v_proj``'s 0.195), and
they are the only two projections Qwen3 follows with a per-head RMSNorm
(``q_norm`` / ``k_norm``), which leaves a head's output scale free.

Those two facts have one benign reading and one alarming one, and a per-tensor
scalar cannot tell them apart:

* **structural** -- GPTQ's compensation really is more effective on these
  weights, row by row; or
* **aggregation** -- ``hq = sqrt(sum_r E_r H E_r^T / sum_r W_r H W_r^T)`` is a
  *ratio of sums over rows*.  Wide between-row spread concentrates both sums
  on a few large-norm rows, so on ``q``/``k`` the census scalar is close to a
  few-row statistic, and whichever arm happens to do better on those rows wins
  the aggregate by more than it wins the tensor.

This measures both.  Every arm is scored **per output row** -- rows are
independent under this quadratic, `E_r H E_r^T` involves no other row -- and
each leg is reported three ways on the same numbers:

* ``agg``  -- the census aggregate, ``sqrt(sum num / sum den)``.  Joined back to
  the two censuses per unit as a check, not assumed.
* ``med``  -- the median over rows of the per-row leg ratio.  Every leg ratio
  is ``sqrt(num_x,r / num_y,r)``: scale-free per row, so a row's norm cannot
  buy it influence.
* ``relmed`` -- the median over rows of the per-row *relative* error
  ``sqrt(num_r / den_r)``, an aggregate with no row weighting at all.

Plus the row concentration that decides whether the two can differ: the
participation ratio ``(sum den)^2 / sum den^2`` as a fraction of the row count.

Restricted to ``q_proj``/``k_proj``/``v_proj``, all 28 layers, because that is
the triple whose Hessian is bit-identical (asserted here per layer) -- so W is
the only thing that varies, and ``k``/``v`` are the identical shape as well.

Arms, all five, one process, one device:
  ``P_e4m3``  NVFP4's plane, E2M1 nearest rounding, no compensation  (NVFP4-RTN)
  ``A``       production NVFP4 GPTQ+JSO, dequantised from its own bytes
  ``P_lut16`` Tessera's plane, same alphabet and rounding             (Tessera-RTN)
  ``B``       Tessera weights-only at the E2M1x2 q896 wire
  ``C``       Tessera + LDLQ 1.0/32 + refit h^1.0 -- the served default

A screen, weight leg only, in-sample on H for arm C.  Promotes nothing.
"""
from __future__ import annotations

import argparse, glob, json, math, statistics, time
import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.encode import _pack_scales_lut, e4m3_positive_values
from tessera.export import DEFAULT_LDLQ_BLOCK, encode_linear_planes, wire_recipe
from tessera.compensate import block_ldl, regularize_hessian
from tessera.unit_artifact import read_unit_artifact

SRC = "/home/rob/models/Qwen3-0.6B"
NVFP4 = "/home/rob/dq-runs/fc45-0p6b-nvfp4/exported"
HFULL = "/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt"
PLANE_CENSUS = "/mnt/shared/tessera-runs/ldlq-lut/dense4_plane_census.json"
RES_CENSUS = "/mnt/shared/tessera-runs/ldlq-lut/dense4_residual_census.json"
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
HALF, PEAK = 16, 6.0
ROLES = ("q_proj", "k_proj", "v_proj")


def open_all(d):
    idx = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        h = safe_open(f, framework="pt")
        for k in h.keys():
            idx[k] = h
    return idx


def get(idx, key, device):
    return idx[key].get_tensor(key).to(device)


def e2m1_rtn(W, s):
    vals = torch.tensor(E2M1_VALUES, device=W.device, dtype=torch.float32)
    halves = W.reshape(-1, HALF)
    x = halves / s[:, None]
    q = vals[(x.abs()[..., None] - vals).abs().argmin(dim=-1)] * torch.sign(x)
    return (q * s[:, None]).reshape(W.shape)


def nvfp4_arm(idx, name, rows, cols, device):
    pk = get(idx, name + ".weight_packed", device)
    s = get(idx, name + ".weight_scale", device).float()
    g = get(idx, name + ".weight_global_scale", device).float()
    vals = torch.tensor(E2M1_VALUES, device=device)
    lo, hi = (pk & 0xF).long(), (pk >> 4).long()

    def dq(n):
        return vals[n & 7] * torch.where(n >= 8, -1.0, 1.0)

    w = torch.stack([dq(lo), dq(hi)], -1).reshape(rows, cols)
    return w * (s / g).repeat_interleave(HALF, dim=1)


def rowquad(M, H):
    """Per-row ``M_r H M_r^T`` -- rows are independent under this quadratic."""
    return ((M @ H) * M).sum(dim=1).clamp_min(0.0)


def geo(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=DEFAULT_LDLQ_BLOCK)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    grid = tuple_grid(E2M1_GRID, 2)
    print(f"wire_recipe(E2M1x2, {args.q256}) = {wire_recipe(grid, args.q256)}", flush=True)

    blob = torch.load(HFULL, map_location="cpu")
    Hs = blob["H"]
    pc = {u["name"]: u for u in json.load(open(PLANE_CENSUS))["units"]}
    rc = {u["name"]: u for u in json.load(open(RES_CENSUS))["units"]}
    src, nv = open_all(SRC), open_all(NVFP4)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else sorted({int(n.split(".")[2]) for n in Hs}))

    def encode(W, H, *, ldlq):
        kw = {}
        if ldlq:
            h = torch.diagonal(H)
            kw = {"ldl": block_ldl(regularize_hessian(H, sigma_reg=args.sigma), args.block),
                  "ldl_block": args.block, "refit_metric": (h / h.mean()).clone()}
        exported, _u, _f = encode_linear_planes(W, grid=grid, q256=args.q256,
                                                name="u", verify=False, **kw)
        return read_unit_artifact(exported.blob, device=W.device).float()

    out, t0, first = [], time.time(), {}
    for L in layers:
        Hq = Hs.get(f"model.layers.{L}.self_attn.q_proj")
        assert all(torch.equal(Hq, Hs[f"model.layers.{L}.self_attn.{r}"]) for r in ROLES), \
            f"layer {L}: q/k/v Hessians are not bit-identical"
        for role in ROLES:
            name = f"model.layers.{L}.self_attn.{role}"
            W = get(src, name + ".weight", dev).to(torch.float32).contiguous()
            r, c = W.shape
            H = Hs[name].to(dev, torch.float32)
            t = W.reshape(-1, HALF).abs().amax(dim=1).clamp_min(1e-30) / PEAK
            gnv = float(t.max()) / 448.0
            gridv = e4m3_positive_values(dev)
            s_nv = gridv[(t[:, None] / gnv - gridv[None, :]).abs().argmin(dim=1)] * gnv
            _tb, _ix, s_lut, _gs = _pack_scales_lut(W, HALF)
            arms = {
                "P_e4m3": e2m1_rtn(W, s_nv),
                "A": nvfp4_arm(nv, name, r, c, dev),
                "P_lut16": e2m1_rtn(W, s_lut),
                "B": encode(W, H, ldlq=False),
                "C": encode(W, H, ldlq=True),
            }
            den = rowquad(W, H)
            num = {k: rowquad(v - W, H) for k, v in arms.items()}
            rec = {"name": name, "role": role, "layer": L, "rows": r, "cols": c}
            # census join, per unit, checked not assumed
            agg = {k: math.sqrt(float(num[k].sum()) / float(den.sum())) for k in arms}
            rec["agg"] = agg
            j = {}
            for k, ref in (("A", rc[name]["A_nvfp4"]["hq"]), ("B", rc[name]["B_weights_only"]["hq"]),
                           ("C", rc[name]["C_h_aware"]["hq"]),
                           ("P_e4m3", pc[name]["P_e4m3"]["hq"]), ("P_lut16", pc[name]["P_lut16"]["hq"])):
                j[k] = abs(agg[k] - ref) <= 1e-6 * max(1.0, ref)
            rec["census_join"] = j
            # row concentration: how many rows the aggregate effectively counts
            rec["participation"] = float(den.sum() ** 2 / (den * den).sum()) / r
            rec["den_top1pct_share"] = float(
                den.sort(descending=True).values[: max(1, r // 100)].sum() / den.sum())
            # the three readings of each leg
            legs = {"plane": ("P_lut16", "P_e4m3"), "gptq": ("P_e4m3", "A"),
                    "body": ("P_lut16", "B"), "comp": ("B", "C")}
            rec["leg_agg"] = {k: agg[a] / agg[b] for k, (a, b) in legs.items()}
            rec["leg_rowmed"] = {k: float(torch.sqrt(num[a] / num[b].clamp_min(1e-300)).median())
                                 for k, (a, b) in legs.items()}
            rec["relmed"] = {k: float(torch.sqrt(num[k] / den.clamp_min(1e-300)).median())
                             for k in arms}
            first.setdefault(name, agg["B"])
            out.append(rec)
            print(f"[{len(out):3d}] {name:<40} {r}x{c} part {rec['participation']:.3f} "
                  f"top1% {rec['den_top1pct_share']:.3f} | gptq agg "
                  f"{rec['leg_agg']['gptq']:.4f} rowmed {rec['leg_rowmed']['gptq']:.4f} | "
                  f"comp agg {rec['leg_agg']['comp']:.4f} rowmed {rec['leg_rowmed']['comp']:.4f} "
                  f"| join {sum(j.values())}/5 [{time.time()-t0:.0f}s]", flush=True)
            del W, H, arms, num, den
            if dev == "cuda":
                torch.cuda.empty_cache()
            with open(args.out, "w") as fh:
                json.dump({"units": out, "args": vars(args), "partial": True}, fh, indent=1)

    print("\n== repeat control (arm B, re-run last, same process)", flush=True)
    control = []
    for rec in out[: args.repeat]:
        name = rec["name"]
        W = get(src, name + ".weight", dev).to(torch.float32).contiguous()
        H = Hs[name].to(dev, torch.float32)
        again = math.sqrt(float(rowquad(encode(W, H, ldlq=False) - W, H).sum())
                          / float(rowquad(W, H).sum()))
        ok = abs(again - first[name]) <= 1e-9 * max(1.0, first[name])
        control.append({"name": name, "first": first[name], "again": again, "same": ok})
        print(f"   {name:<40} {first[name]:.8f} -> {again:.8f} {'SAME' if ok else '!! DIFFER'}",
              flush=True)

    bad = [u["name"] for u in out if sum(u["census_join"].values()) != 5]
    print(f"\ncensus join 5/5 on {len(out)-len(bad)}/{len(out)} units"
          + (f"; MISMATCH {bad}" if bad else ""))

    print(f"\n{'role':<8}{'n':>3}{'part':>7}{'top1%':>7}"
          f"{'  |  plane agg':>14}{'rowmed':>8}"
          f"{'  |  gptq agg':>13}{'rowmed':>8}"
          f"{'  |  comp agg':>13}{'rowmed':>8}"
          f"{'  |  relmed C/A':>15}{'aggC/A':>8}")
    summ = {}
    for role in ROLES:
        s = [u for u in out if u["role"] == role]
        summ[role] = {
            "n": len(s),
            "participation": statistics.median([u["participation"] for u in s]),
            "top1": statistics.median([u["den_top1pct_share"] for u in s]),
            **{f"{leg}_{how}": geo([u[f"leg_{how}"][leg] for u in s])
               for leg in ("plane", "gptq", "body", "comp") for how in ("agg", "rowmed")},
            "relmed_C_over_A": geo([u["relmed"]["C"] / u["relmed"]["A"] for u in s]),
            "agg_C_over_A": geo([u["agg"]["C"] / u["agg"]["A"] for u in s]),
        }
        t = summ[role]
        print(f"{role:<8}{t['n']:>3}{t['participation']:>7.3f}{t['top1']:>7.3f}"
              f"{t['plane_agg']:>14.4f}{t['plane_rowmed']:>8.4f}"
              f"{t['gptq_agg']:>13.4f}{t['gptq_rowmed']:>8.4f}"
              f"{t['comp_agg']:>13.4f}{t['comp_rowmed']:>8.4f}"
              f"{t['relmed_C_over_A']:>15.4f}{t['agg_C_over_A']:>8.4f}")

    print("\n== matched triple, ratio to v_proj in the same layer (H bit-identical)")
    byname = {u["name"]: u for u in out}
    pairs = {}
    for a in ("q_proj", "k_proj"):
        rows = [(byname[f"model.layers.{L}.self_attn.{a}"],
                 byname[f"model.layers.{L}.self_attn.v_proj"]) for L in layers]
        pairs[a] = {}
        print(f"\n  {a} / v_proj  (n={len(rows)})")
        print(f"    {'leg':<8}{'agg':>10}{'a>b':>5}{'rowmed':>10}{'a>b':>5}")
        for leg in ("plane", "gptq", "body", "comp"):
            ga = geo([x["leg_agg"][leg] / y["leg_agg"][leg] for x, y in rows])
            gm = geo([x["leg_rowmed"][leg] / y["leg_rowmed"][leg] for x, y in rows])
            na = sum(x["leg_agg"][leg] > y["leg_agg"][leg] for x, y in rows)
            nm = sum(x["leg_rowmed"][leg] > y["leg_rowmed"][leg] for x, y in rows)
            pairs[a][leg] = {"agg": ga, "rowmed": gm, "n_agg": na, "n_rowmed": nm}
            print(f"    {leg:<8}{ga:>10.4f}{na:>5}{gm:>10.4f}{nm:>5}")
        ca_a = geo([(x["agg"]["C"]/x["agg"]["A"]) / (y["agg"]["C"]/y["agg"]["A"]) for x, y in rows])
        ca_m = geo([(x["relmed"]["C"]/x["relmed"]["A"]) / (y["relmed"]["C"]/y["relmed"]["A"])
                    for x, y in rows])
        pairs[a]["C_over_A"] = {"agg": ca_a, "relmed": ca_m}
        print(f"    {'C/A':<8}{ca_a:>10.4f}{'':>5}{ca_m:>10.4f}")

    with open(args.out, "w") as fh:
        json.dump({"units": out, "control": control, "by_role": summ, "pairs": pairs,
                   "args": vars(args), "partial": False}, fh, indent=1)
    print(f"\nwrote {args.out} [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
