"""Issue #12: the compensation-reach test the mechanism predicts.

``dense4_plane_census.py`` decomposes the census ratio exactly:

    C/A  =  g_C  x  g_body  x  h_plane  x  g_A

where ``g_C = C/B`` is what Tessera's LDLQ + refit removes, ``g_A =
RTN_e4m3/A`` is what NVFP4's GPTQ+JSO removes, ``h_plane`` is the LUT16
plane's cost against NVFP4's per-half E4M3 byte at fixed alphabet, and
``g_body`` is the trellis against plain E2M1 rounding.  Over 196 Linears the
plane and the body are flat (Spearman +0.19 and ~1% respectively) and the
compensation terms are not: ``g_C`` is **insensitive to how hard the unit is**
(Spearman -0.06 against the unit's own error) while ``g_A`` **rises steeply as
the unit gets easier** (-0.75).  Rank the seven roles by which side's
compensation removes more and the census's ``C/A`` ordering comes back exactly
(Spearman -1.000).

That names a lever.  ``block_ldl``'s own docstring says the ``block`` columns
quantised together "see no correction from each other" and that "smaller
blocks therefore compensate more"; the served default is ``block=32`` and
production GPTQ compensates column by column.  If the deficit on ``q_proj`` and
``k_proj`` is Tessera's compensation being coarser than the comparator's, then
**narrowing the block must buy more on those two roles than on ``v_proj``** --
and if it buys the same everywhere, reach is not the mechanism and the
hypothesis is dead.  Either answer is worth the run.

**The control the model hands us.**  ``q_proj``, ``k_proj`` and ``v_proj`` of one
layer read the same hidden state, so their fit Hessians are bit-identical
(checked and printed) and their held-out rows are the same rows.  ``k`` and ``v``
have the identical shape.  One arm changes, and it is ``ldl_block``.

Every arm scores ``out`` on the held-out eval slice (disjoint from the rows the
Hessian was accumulated on -- neither the encoder nor the refit saw them) and
``hfit`` = ``sqrt(E H E^T / W H W^T)`` on the fit rows, the quadratic the refit
is monotone in.  The served default runs **first and again last** on every
unit; a disagreement is printed, not absorbed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.compensate import block_ldl, regularize_hessian
from tessera.export import encode_linear_planes, wire_recipe
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


def hq(W, Wd, H):
    E = (Wd - W).float()
    return math.sqrt(max(float((E @ H * E).sum()), 0.0) / float((W @ H * W).sum()))


def outspace(W, Wd, X):
    E = (Wd - W).float()
    num = float((X @ E.T).pow(2).sum())
    den = float((X @ W.T).pow(2).sum())
    return math.sqrt(num / den)


def nvfp4_arm(idx, name, rows, cols, device):
    pk = idx[name + ".weight_packed"].get_tensor(name + ".weight_packed").to(device)
    s = idx[name + ".weight_scale"].get_tensor(name + ".weight_scale").to(device).float()
    g = idx[name + ".weight_global_scale"].get_tensor(
        name + ".weight_global_scale").to(device).float()
    vals = torch.tensor(E2M1_VALUES, device=device)
    lo, hi = (pk & 0xF).long(), (pk >> 4).long()

    def dq(n):
        return vals[n & 7] * torch.where(n >= 8, -1.0, 1.0)

    w = torch.stack([dq(lo), dq(hi)], -1).reshape(rows, cols)
    return w * (s / g).repeat_interleave(16, dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,1")
    ap.add_argument("--blocks", default="16,32,64,128")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    grid = tuple_grid(E2M1_GRID, 2)
    print(f"wire_recipe(E2M1x2, {args.q256}) = {wire_recipe(grid, args.q256)}", flush=True)

    Hs = torch.load(HFULL, map_location="cpu")["H"]
    xb = torch.load(XEVAL, map_location="cpu")
    src, nv = open_all(SRC), open_all(NVFP4)
    blocks = [int(b) for b in args.blocks.split(",")]
    layers = [int(x) for x in args.layers.split(",")]

    units = []
    for L in layers:
        # the held-out rows for this layer's attention block: q/k/v read one
        # hidden state, so one capture serves all three.  Asserted below.
        xk = next((k for k in xb["x"]
                   if k.startswith(f"model.layers.{L}.self_attn.")), None)
        if xk is None:
            print(f"layer {L}: no held-out rows cached, skipped", flush=True)
            continue
        for role in ("q_proj", "k_proj", "v_proj"):
            units.append((f"model.layers.{L}.self_attn.{role}", xk))

    # the natural experiment, asserted rather than assumed
    for L in layers:
        hq_ = Hs.get(f"model.layers.{L}.self_attn.q_proj")
        hk_ = Hs.get(f"model.layers.{L}.self_attn.k_proj")
        hv_ = Hs.get(f"model.layers.{L}.self_attn.v_proj")
        if hq_ is None:
            continue
        ok = bool(torch.equal(hq_, hk_)) and bool(torch.equal(hq_, hv_))
        print(f"layer {L}: H(q) == H(k) == H(v) bit-identical: {ok}", flush=True)
        if not ok:
            raise SystemExit("the shared-H premise failed; the pairing is not a control")

    rows, t0 = [], time.time()
    for name, xk in units:
        W = src[name + ".weight"].get_tensor(name + ".weight").to(dev).float().contiguous()
        r, c = W.shape
        H = Hs[name].to(dev, torch.float32)
        X = xb["x"][xk].to(dev, torch.float32)
        h = torch.diagonal(H)
        metric = (h / h.mean()).clone()
        Hreg = regularize_hessian(H, sigma_reg=args.sigma)
        rec = {"name": name, "role": name.split(".")[-1],
               "layer": int(name.split(".")[2]), "rows": r, "cols": c,
               "x_from": xk, "arms": {}}
        wa = nvfp4_arm(nv, name, r, c, dev)
        rec["A_nvfp4"] = {"hq": hq(W, wa, H), "out": outspace(W, wa, X)}
        del wa

        def run(block, tag):
            exported, _u, _f = encode_linear_planes(
                W, grid=grid, q256=args.q256, name="u", verify=False,
                ldl=block_ldl(Hreg, block), ldl_block=block,
                refit_metric=metric)
            Wd = read_unit_artifact(exported.blob, device=dev).float()
            rec["arms"][tag] = {"block": block, "bpp": float(exported.bpp),
                                "hq": hq(W, Wd, H), "out": outspace(W, Wd, X)}
            del Wd
            a = rec["arms"][tag]
            print(f"   {tag:<22} block {block:>4}  bpp {a['bpp']:.4f}  "
                  f"out {a['out']:.5f}  hfit {a['hq']:.5f}  "
                  f"out/A {a['out']/rec['A_nvfp4']['out']:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

        print(f"\n== {name} {r}x{c}  A(nvfp4 4.5bpp) out "
              f"{rec['A_nvfp4']['out']:.5f} hfit {rec['A_nvfp4']['hq']:.5f}", flush=True)
        run(32, "control FIRST b32")
        for b in blocks:
            run(b, f"b{b}")
        run(32, "control LAST b32")
        f, l = rec["arms"]["control FIRST b32"], rec["arms"]["control LAST b32"]
        rec["drift_same"] = abs(f["out"] - l["out"]) <= 1e-12 and abs(f["hq"] - l["hq"]) <= 1e-12
        print(f"   drift control: {'SAME' if rec['drift_same'] else '!! DIFFER'}", flush=True)
        rows.append(rec)
        del W, H, X, Hreg
        if dev == "cuda":
            torch.cuda.empty_cache()
        with open(args.out, "w") as fh:
            json.dump({"units": rows, "args": vars(args), "partial": True}, fh, indent=1)

    print(f"\n{'unit':<34} {'A out':>8} " +
          "".join(f"{'b'+str(b)+' out/A':>12}" for b in blocks))
    for rec in rows:
        cells = "".join(f"{rec['arms']['b'+str(b)]['out']/rec['A_nvfp4']['out']:>12.4f}"
                        for b in blocks)
        print(f"{rec['name'].replace('model.layers.','L'):<34} "
              f"{rec['A_nvfp4']['out']:>8.5f} {cells}", flush=True)
    print(f"\ngain of the narrowest block against the default, per unit "
          f"(out-space, <1 = narrower is better)")
    for rec in rows:
        d = rec["arms"][f"b{blocks[0]}"]["out"] / rec["arms"]["b32"]["out"]
        print(f"   {rec['name'].replace('model.layers.','L'):<34} "
              f"b{blocks[0]}/b32 {d:.4f}", flush=True)
    with open(args.out, "w") as fh:
        json.dump({"units": rows, "args": vars(args), "partial": False}, fh, indent=1)
    print(f"\nwrote {args.out} [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
