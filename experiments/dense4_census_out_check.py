"""Is the 4-bit residual census's metric fair to the arm that was not fit to it?

`dense4_residual_census.py` scores every arm on `sqrt(E H E^T / W H W^T)` with
the **fit-row** Hessian.  Arm C's LDLQ factor and block-scale refit were built
from that same `H`, so C is scored in-sample.  Arm A -- production NVFP4
GPTQ+JSO -- was calibrated on PrismaQuant's own activations somewhere else, so
it is scored out-of-sample.  The receipt bounds C's in-sample advantage at 1-3%
on one unit; nothing bounds A's out-of-sample penalty, and a census that
concludes "Tessera wins the weight leg" on a quadratic one arm was fit to has
concluded something about the quadratic.

`x_eval_qwen06b.pt` carries **held-out** activation rows -- wikitext-2 train,
the eval slice, disjoint from the fit slice `H` was accumulated on -- for six
Linears.  Neither arm saw them.  This scores both arms both ways on those six
and compares `C/A` under each metric.  If the two agree, the census metric is
not doing the work; if `C/A(out)` is materially the larger, it is, and the
census's reading has to say so.

Arms run back to back in one process, weights-only re-run last as the control,
same construction as the census so the numbers are comparable to it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.compensate import block_ldl, regularize_hessian
from tessera.export import DEFAULT_LDLQ_BLOCK, encode_linear_planes
from tessera.unit_artifact import read_unit_artifact

SRC = "/home/rob/models/Qwen3-0.6B"
NVFP4 = "/home/rob/dq-runs/fc45-0p6b-nvfp4/exported"
HFULL = "/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt"
XEVAL = "/mnt/shared/tessera-runs/ldlq/x_eval_qwen06b.pt"
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def open_all(d):
    idx = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        h = safe_open(f, framework="pt")
        for k in h.keys():
            idx[k] = h
    return idx


def get(idx, key, device):
    return idx[key].get_tensor(key).to(device)


def nvfp4_arm(idx, name, rows, cols, device):
    pk = get(idx, name + ".weight_packed", device)
    s = get(idx, name + ".weight_scale", device).float()
    g = get(idx, name + ".weight_global_scale", device).float()
    vals = torch.tensor(E2M1_VALUES, device=device)
    lo, hi = (pk & 0xF).long(), (pk >> 4).long()

    def dq(n):
        return vals[n & 7] * torch.where(n >= 8, -1.0, 1.0)

    w = torch.stack([dq(lo), dq(hi)], -1).reshape(rows, cols)
    return w * (s / g).repeat_interleave(16, dim=1)


def hq(W, Wd, H):
    E = (Wd - W).float()
    return math.sqrt(max(float((E @ H * E).sum()), 0.0) / float((W @ H * W).sum()))


def out_err(W, Wd, X):
    """||X (W - Wd)^T||_F / ||X W^T||_F on held-out rows."""
    E = (Wd - W).float()
    return float((X @ E.T).norm() / (X @ W.float().T).norm())


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--ldlq-sigma", type=float, default=1.0)
    ap.add_argument("--ldlq-block", type=int, default=DEFAULT_LDLQ_BLOCK)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    grid = tuple_grid(E2M1_GRID, 2)
    Hs = torch.load(HFULL, map_location="cpu")["H"]
    xblob = torch.load(XEVAL, map_location="cpu")
    Xs, xprov = xblob["x"], xblob.get("provenance", {})
    src, nv = open_all(SRC), open_all(NVFP4)

    def encode(W, H, *, ldlq):
        kw = {}
        if ldlq:
            h = torch.diagonal(H)
            kw = {"ldl": block_ldl(regularize_hessian(H, sigma_reg=args.ldlq_sigma),
                                   args.ldlq_block),
                  "ldl_block": args.ldlq_block,
                  "refit_metric": (h / h.mean()).clone()}
        exported, _u, _f = encode_linear_planes(
            W, grid=grid, q256=args.q256, name="u", verify=False, **kw)
        return read_unit_artifact(exported.blob, device=W.device), float(exported.bpp)

    rows_out, first = [], {}
    print(f"{'unit':<38} {'A hq':>8} {'C hq':>8} {'C/A hq':>7} | "
          f"{'A out':>8} {'C out':>8} {'C/A out':>8}", flush=True)
    for name in sorted(Xs):
        W = get(src, name + ".weight", dev).to(torch.float32).contiguous()
        r, c = W.shape
        H = Hs[name].to(dev, torch.float32)
        X = Xs[name].to(dev, torch.float32)
        wa = nvfp4_arm(nv, name, r, c, dev)
        wb, _ = encode(W, H, ldlq=False)
        wc, bpp_c = encode(W, H, ldlq=True)
        rec = {"name": name, "rows": r, "cols": c, "bpp_c": bpp_c,
               "A": {"hq": hq(W, wa, H), "out": out_err(W, wa, X)},
               "B": {"hq": hq(W, wb.float(), H), "out": out_err(W, wb.float(), X)},
               "C": {"hq": hq(W, wc.float(), H), "out": out_err(W, wc.float(), X)}}
        rec["C_over_A_hq"] = rec["C"]["hq"] / rec["A"]["hq"]
        rec["C_over_A_out"] = rec["C"]["out"] / rec["A"]["out"]
        first[name] = rec["B"]["hq"]
        rows_out.append(rec)
        print(f"{name.replace('model.layers.',''):<38} {rec['A']['hq']:>8.5f} "
              f"{rec['C']['hq']:>8.5f} {rec['C_over_A_hq']:>7.4f} | "
              f"{rec['A']['out']:>8.5f} {rec['C']['out']:>8.5f} "
              f"{rec['C_over_A_out']:>8.4f}", flush=True)
        del W, H, X, wa, wb, wc
        torch.cuda.empty_cache()

    print("\n== repeat control (weights-only, re-run last)", flush=True)
    control = []
    for name in sorted(Xs):
        W = get(src, name + ".weight", dev).to(torch.float32).contiguous()
        H = Hs[name].to(dev, torch.float32)
        w2, _ = encode(W, H, ldlq=False)
        again = hq(W, w2.float(), H)
        ok = abs(again - first[name]) <= 1e-9 * max(1.0, abs(first[name]))
        control.append({"name": name, "first": first[name], "again": again, "same": ok})
        print(f"   {name.replace('model.layers.',''):<38} {first[name]:.8f} -> "
              f"{again:.8f}  {'SAME' if ok else '!! DIFFER'}", flush=True)
        del W, H, w2
        torch.cuda.empty_cache()

    g_hq = geomean([r["C_over_A_hq"] for r in rows_out])
    g_out = geomean([r["C_over_A_out"] for r in rows_out])
    print(f"\ngeomean C/A: hq {g_hq:.4f}   out {g_out:.4f}   "
          f"out/hq {g_out / g_hq:.4f}", flush=True)
    print("out/hq > 1 means the census metric flattered arm C, the arm fit to it.",
          flush=True)
    with open(args.out, "w") as fh:
        json.dump({"units": rows_out, "control": control,
                   "geomean": {"C_over_A_hq": g_hq, "C_over_A_out": g_out,
                               "out_over_hq": g_out / g_hq},
                   "x_provenance": xprov, "args": vars(args)}, fh, indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
