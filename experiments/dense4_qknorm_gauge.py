"""Issue #12: q_norm/k_norm make a head's scale a gauge, and hq does not know.

The 196-Linear census attributes the dense 4-bit weight-leg residual to
``q_proj`` (1.0648) and ``k_proj`` (1.0976) with the other five roles winning,
on ``hq = sqrt(sum_r E_r H E_r^T / sum_r W_r H W_r^T)``.  ``k_proj`` and
``v_proj`` of one layer have **bit-identical Hessians and identical shape**, so
the only thing that can separate them is ``W`` -- and Qwen3 hands us exactly
one structural difference:

    model.layers.L.self_attn.q_norm.weight   exists
    model.layers.L.self_attn.k_norm.weight   exists
    model.layers.L.self_attn.v_norm.weight   does not

``q_proj``'s and ``k_proj``'s outputs pass through a per-head RMSNorm over
``head_dim`` before RoPE; ``v_proj``'s do not.  Two consequences, both measured
here rather than argued.

**1. A head's output scale is a gauge freedom on q/k and not on v.**  Scale
``W_head`` by alpha and the post-norm output is *exactly* unchanged
(``y/rms(y)`` is scale invariant), so training is free to leave a head's
overall magnitude anywhere.  Prediction: the between-head component of the
row-scale spread is large on ``q``/``k`` and small on ``v``.  Measured as the
sd of ``log2`` head medians against the sd within a head.

**2. So ``hq`` weights those heads wrongly, and it is the two roles the
residual is attributed to.**  ``hq`` is a *ratio of sums over rows*: a head
carrying alpha^2 times the energy contributes alpha^2 times the numerator.  But
downstream of the norm that alpha cancels -- to first order the post-norm error
is ``(I - yhat yhat^T) delta / ||y||``, i.e. the head's **relative** error, and
the radial component is annihilated outright.  So the weighting the next op
actually applies is *equal per head*, not proportional to head energy.  The
derived aggregate is therefore

    hq_head = sqrt( mean_j  E_j H E_j^T / W_j H W_j^T )

the RMS over heads of the per-head relative error -- not a heuristic reweighting
but the Jacobian of the op that follows.  For ``v_proj`` and every MLP role
there is no such norm and the plain aggregate stays correct; this arm changes
the reading of ``q``/``k`` only, which is why ``v_proj`` is carried as the
control that must *not* move much.

Arms are the census's own five, one process, one device, and every unit's plain
aggregate is joined back to the two censuses as a check.  Weight leg only,
in-sample on H for arm C, no serve: this cannot promote anything.  What it can
do is say whether the localisation the issue rests on survives its own metric.
"""
from __future__ import annotations

import argparse, glob, json, math, statistics, time
import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.compensate import block_ldl, regularize_hessian
from tessera.encode import _pack_scales_lut, e4m3_positive_values
from tessera.export import DEFAULT_LDLQ_BLOCK, encode_linear_planes, wire_recipe
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
    return ((M @ H) * M).sum(dim=1).clamp_min(0.0)


def geo(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=DEFAULT_LDLQ_BLOCK)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, hd = args.device, args.head_dim
    grid = tuple_grid(E2M1_GRID, 2)
    print(f"wire_recipe(E2M1x2, {args.q256}) = {wire_recipe(grid, args.q256)}", flush=True)

    blob = torch.load(HFULL, map_location="cpu")
    Hs = blob["H"]
    pc = {u["name"]: u for u in json.load(open(PLANE_CENSUS))["units"]}
    rc = {u["name"]: u for u in json.load(open(RES_CENSUS))["units"]}
    src, nv = open_all(SRC), open_all(NVFP4)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else sorted({int(n.split(".")[2]) for n in Hs}))

    # The structural asymmetry, checked on the checkpoint rather than assumed.
    norms = {}
    for L in layers:
        norms[L] = {r: f"model.layers.{L}.self_attn.{r.split('_')[0]}_norm.weight" in src
                    for r in ROLES}
    print("per-head RMSNorm on the output, from the checkpoint's own keys: "
          + ", ".join(f"{r}={all(norms[L][r] for L in layers)}" for r in ROLES), flush=True)

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
        Hq = Hs[f"model.layers.{L}.self_attn.q_proj"]
        assert all(torch.equal(Hq, Hs[f"model.layers.{L}.self_attn.{r}"]) for r in ROLES)
        for role in ROLES:
            name = f"model.layers.{L}.self_attn.{role}"
            W = get(src, name + ".weight", dev).to(torch.float32).contiguous()
            r, c = W.shape
            assert r % hd == 0, f"{name}: {r} rows is not a multiple of head_dim {hd}"
            nh = r // hd
            H = Hs[name].to(dev, torch.float32)
            t = W.reshape(-1, HALF).abs().amax(dim=1).clamp_min(1e-30) / PEAK
            gnv = float(t.max()) / 448.0
            gridv = e4m3_positive_values(dev)
            s_nv = gridv[(t[:, None] / gnv - gridv[None, :]).abs().argmin(dim=1)] * gnv
            _tb, _ix, s_lut, _gs = _pack_scales_lut(W, HALF)
            arms = {"P_e4m3": e2m1_rtn(W, s_nv), "A": nvfp4_arm(nv, name, r, c, dev),
                    "P_lut16": e2m1_rtn(W, s_lut), "B": encode(W, H, ldlq=False),
                    "C": encode(W, H, ldlq=True)}
            den = rowquad(W, H)
            num = {k: rowquad(v - W, H) for k, v in arms.items()}
            den_h = den.reshape(nh, hd).sum(dim=1)
            num_h = {k: v.reshape(nh, hd).sum(dim=1) for k, v in num.items()}
            plain = {k: math.sqrt(float(num[k].sum()) / float(den.sum())) for k in arms}
            # The derived aggregate: RMS over heads of the per-head relative error.
            head = {k: math.sqrt(float((num_h[k] / den_h).mean())) for k in arms}
            rec = {"name": name, "role": role, "layer": L, "rows": r, "cols": c,
                   "n_heads": nh, "has_out_norm": norms[L][role],
                   "plain": plain, "head": head,
                   "C_over_A_plain": plain["C"] / plain["A"],
                   "C_over_A_head": head["C"] / head["A"],
                   "B_over_A_plain": plain["B"] / plain["A"],
                   "B_over_A_head": head["B"] / head["A"]}
            # the gauge itself: between-head vs within-head row-scale spread
            rowamax = W.abs().amax(dim=1).clamp_min(1e-30)
            hmed = rowamax.reshape(nh, hd).median(dim=1).values
            rec["gauge"] = {
                "between_heads_sd_log2": float(torch.log2(hmed).std()) if nh > 1 else 0.0,
                "within_head_sd_log2": float(
                    torch.log2(rowamax.reshape(nh, hd) / hmed[:, None]).std()),
                "head_energy_top_share": float(den_h.max() / den_h.sum()),
                "head_energy_participation": float(den_h.sum() ** 2 / (den_h * den_h).sum()) / nh,
            }
            j = {}
            for k, ref in (("A", rc[name]["A_nvfp4"]["hq"]), ("B", rc[name]["B_weights_only"]["hq"]),
                           ("C", rc[name]["C_h_aware"]["hq"]),
                           ("P_e4m3", pc[name]["P_e4m3"]["hq"]),
                           ("P_lut16", pc[name]["P_lut16"]["hq"])):
                j[k] = abs(plain[k] - ref) <= 1e-6 * max(1.0, ref)
            rec["census_join"] = j
            first.setdefault(name, plain["B"])
            out.append(rec)
            print(f"[{len(out):3d}] {name:<40} {nh:>2}h norm={int(rec['has_out_norm'])} "
                  f"btwHead {rec['gauge']['between_heads_sd_log2']:.3f} "
                  f"inHead {rec['gauge']['within_head_sd_log2']:.3f} "
                  f"topHead {rec['gauge']['head_energy_top_share']:.3f} | "
                  f"C/A plain {rec['C_over_A_plain']:.4f} head {rec['C_over_A_head']:.4f} "
                  f"| join {sum(j.values())}/5 [{time.time()-t0:.0f}s]", flush=True)
            del W, H, arms, num, den
            if dev == "cuda":
                torch.cuda.empty_cache()
            with open(args.out, "w") as fh:
                json.dump({"units": out, "args": vars(args), "partial": True}, fh, indent=1)

    print("\n== repeat control (arm B plain aggregate, re-run last, same process)", flush=True)
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
    print(f"\ncensus join 5/5 on {len(out)-len(bad)}/{len(out)}"
          + (f"; MISMATCH {bad}" if bad else ""))

    print(f"\n{'role':<8}{'n':>3}{'norm':>5}{'btwHead':>9}{'inHead':>8}{'topHead':>9}{'part':>7}"
          f"{'  C/A plain':>12}{'C/A head':>10}{'  B/A plain':>12}{'B/A head':>10}")
    summ = {}
    for role in ROLES:
        s = [u for u in out if u["role"] == role]
        summ[role] = {
            "n": len(s), "has_out_norm": all(u["has_out_norm"] for u in s),
            "between_heads_sd_log2": statistics.median(
                [u["gauge"]["between_heads_sd_log2"] for u in s]),
            "within_head_sd_log2": statistics.median(
                [u["gauge"]["within_head_sd_log2"] for u in s]),
            "head_energy_top_share": statistics.median(
                [u["gauge"]["head_energy_top_share"] for u in s]),
            "head_energy_participation": statistics.median(
                [u["gauge"]["head_energy_participation"] for u in s]),
            "C_over_A_plain": geo([u["C_over_A_plain"] for u in s]),
            "C_over_A_head": geo([u["C_over_A_head"] for u in s]),
            "B_over_A_plain": geo([u["B_over_A_plain"] for u in s]),
            "B_over_A_head": geo([u["B_over_A_head"] for u in s]),
        }
        t = summ[role]
        print(f"{role:<8}{t['n']:>3}{int(t['has_out_norm']):>5}"
              f"{t['between_heads_sd_log2']:>9.3f}{t['within_head_sd_log2']:>8.3f}"
              f"{t['head_energy_top_share']:>9.3f}{t['head_energy_participation']:>7.3f}"
              f"{t['C_over_A_plain']:>12.4f}{t['C_over_A_head']:>10.4f}"
              f"{t['B_over_A_plain']:>12.4f}{t['B_over_A_head']:>10.4f}")

    print("\n== the reading the norm licenses, against the one the census used")
    for role in ROLES:
        t = summ[role]
        verdict = ("q/k: the norm cancels the head scale, so C/A head is the reading"
                   if t["has_out_norm"] else "v: no norm, plain stays the reading (control)")
        print(f"  {role:<8} plain {t['C_over_A_plain']:.4f} -> head {t['C_over_A_head']:.4f}  "
              f"({t['C_over_A_head']/t['C_over_A_plain']:.4f}x)   {verdict}")

    with open(args.out, "w") as fh:
        json.dump({"units": out, "control": control, "by_role": summ,
                   "args": vars(args), "partial": False}, fh, indent=1)
    print(f"\nwrote {args.out} [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
