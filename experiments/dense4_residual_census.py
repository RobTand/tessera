"""Issue #12, the matched comparison: where does the residual 1.039x live?

#12 opened at **1.254x** -- served Qwen3-0.6B, Tessera W4A4 0.640 against
PrismaQuant's NVFP4 GPTQ+JSO 0.511 at equal residency.  Two things have
happened to that number since, and only one of them is the one the issue
predicted:

* The **reach-aware per-row start** cannot have moved it.  It lives in
  ``initial_channel_scale``, which ``encode_unit`` calls only under
  ``ScalePlaneKind.CHANNEL``; ``wire_recipe(E2M1x2, 896)`` is ``TCQ`` over
  ``LUT``.  Monkeypatching the function to raise and encoding both wires gives
  zero calls on the 4-bit wire and an immediate raise on the 8-bit one.
* **LDLQ + an H-solved block-scale refit on the LUT plane** did move it, at
  byte-identical bytes: 0.640404 -> 0.531028 served, 220,301,312 wire bytes
  either way, so the gap is now **1.039x on 11% fewer bits**
  (``docs/measurements/tessera-ldlq-lut-plane-served-2026-09-02.md``).

So the issue's premise is stale and its three open items are addressed.  What
is left is a residual, and a residual is a scalar: nobody has asked *where in
the model it sits*.  This is that census -- the 4-bit analogue of
``dense_spread_census.py``, which is what found the outlier mechanism on the
8-bit route.

**What this prices, and what it does not.** The **weight leg only**.  Both
arms deploy the same A4 activation leg, which is why #12 calls the gap a
weight-leg problem, and the served numbers already exist and are the gate
(principle 3).  A census cannot promote anything; it can say whether the
residual is *concentrated* -- in a role, a depth, a shape -- and therefore
whether a targeted lever exists, or *flat*, in which case #12 closes as
"the 4-bit route is for residency, not quality".

**The metric.** ``sqrt(E H E^T / W H W^T)`` on the full per-Linear Hessian, the
quadratic the refit is provably monotone in.  Held-out activation rows exist
for six units only, and on the one unit that carries both columns this
quadratic and the held-out out-space error agree to 1-3% and never disagree on
ordering (the receipt's objective table).  Plain relative Frobenius is carried
alongside, unweighted, so a reader can see what the weighting is doing.

**Conditions, pinned.**  Every arm of every unit runs back to back in one
process on one box, and the weights-only arm is **re-run at the end** on the
first ``--repeat`` units.  A disagreement between the two baselines is state
leakage or drift and is printed as ``!! DIFFER``; it is not absorbed into a
ratio.  Comparing two treatments and calling one a control is the error that
cost this project 19.2x.

**Pricing.** Each Tessera arm is priced at the bytes it actually wrote
(``ExportedUnit.bpp``).  NVFP4 is priced from its own export -- packed nibbles
+ per-16 E4M3 block scale + fp32 global -- not from a plane borrowed from
Tessera.  Prices are printed per unit, not assumed from the label.

Run with ``PYTHONPATH=src TMPDIR=/home/rob/tmp
TRITON_CACHE_DIR=/home/rob/.triton-cache`` under the prismaquant-cu130 python.
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
from tessera.export import DEFAULT_LDLQ_BLOCK, encode_linear_planes, wire_recipe
from tessera.unit_artifact import read_unit_artifact

SRC = "/home/rob/models/Qwen3-0.6B"
NVFP4 = "/home/rob/dq-runs/fc45-0p6b-nvfp4/exported"
HFULL = "/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt"
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
    """Dequantise production NVFP4, and price it at what the export stores."""
    pk = get(idx, name + ".weight_packed", device)
    s = get(idx, name + ".weight_scale", device).float()
    g = get(idx, name + ".weight_global_scale", device).float()
    vals = torch.tensor(E2M1_VALUES, device=device)
    lo, hi = (pk & 0xF).long(), (pk >> 4).long()

    def dq(n):
        return vals[n & 7] * torch.where(n >= 8, -1.0, 1.0)

    w = torch.stack([dq(lo), dq(hi)], -1).reshape(rows, cols)
    w = w * (s / g).repeat_interleave(16, dim=1)
    # As it deploys: 4 bits/weight + one E4M3 per 16 + one fp32 per tensor.
    bits = pk.numel() * 8 + s.numel() * 8 + g.numel() * 32
    return w, bits / (rows * cols)


def hq(W, Wd, H):
    """sqrt(E H E^T / W H W^T) -- the refit's own quadratic, per tensor."""
    E = (Wd - W).float()
    num = float((E @ H * E).sum())
    den = float((W @ H * W).sum())
    return math.sqrt(max(num, 0.0) / den)


def rel(W, Wd):
    E = (Wd - W).float()
    return math.sqrt(float((E * E).sum()) / float((W * W).sum()))


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", type=int, default=0, help="0 = every Linear")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth unit")
    ap.add_argument("--repeat", type=int, default=3, help="units to re-run at the end")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--ldlq-sigma", type=float, default=1.0)
    ap.add_argument("--ldlq-block", type=int, default=DEFAULT_LDLQ_BLOCK)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    grid = tuple_grid(E2M1_GRID, 2)
    print(f"wire_recipe(E2M1x2, {args.q256}) = {wire_recipe(grid, args.q256)}", flush=True)

    blob = torch.load(HFULL, map_location="cpu")
    Hs, prov = blob["H"], blob.get("provenance", {})
    print(f"H: {len(Hs)} Linears, provenance {json.dumps(prov, default=str)[:200]}", flush=True)

    src, nv = open_all(SRC), open_all(NVFP4)
    names = sorted(Hs)[:: args.stride]
    if args.units:
        names = names[: args.units]
    print(f"{len(names)} units, repeat control on the first {args.repeat}\n", flush=True)

    rows_out, t0 = [], time.time()
    baseline_first: dict[str, float] = {}

    def encode(W, H, *, ldlq):
        # Arm C is the served default, built the way `ldlq_window_sweep`
        # builds it: LDLQ 1.0/32 over the regularised H, refit on h^1.0 (the
        # mean-normalised diagonal).  Anything else would be a different arm
        # wearing the default's name.
        kw = {}
        if ldlq:
            h = torch.diagonal(H)
            kw = {
                "ldl": block_ldl(regularize_hessian(H, sigma_reg=args.ldlq_sigma),
                                 args.ldlq_block),
                "ldl_block": args.ldlq_block,
                "refit_metric": (h / h.mean()).clone(),
            }
        exported, _unit, _f = encode_linear_planes(
            W, grid=grid, q256=args.q256, name="u", verify=False, **kw
        )
        return read_unit_artifact(exported.blob, device=W.device), float(exported.bpp)

    for i, name in enumerate(names):
        W = get(src, name + ".weight", dev).to(torch.float32).contiguous()
        r, c = W.shape
        H = Hs[name].to(dev, torch.float32)
        rec = {"name": name, "rows": r, "cols": c,
               "role": name.split(".")[-1], "layer": int(name.split(".")[2])}

        wa, bpp_a = nvfp4_arm(nv, name, r, c, dev)
        rec["A_nvfp4"] = {"bpp": bpp_a, "hq": hq(W, wa, H), "rel": rel(W, wa)}

        wb, bpp_b = encode(W, H, ldlq=False)
        rec["B_weights_only"] = {"bpp": bpp_b, "hq": hq(W, wb.float(), H), "rel": rel(W, wb.float())}
        baseline_first[name] = rec["B_weights_only"]["hq"]

        wc, bpp_c = encode(W, H, ldlq=True)
        rec["C_h_aware"] = {"bpp": bpp_c, "hq": hq(W, wc.float(), H), "rel": rel(W, wc.float())}

        rec["ratio_C_over_A"] = rec["C_h_aware"]["hq"] / rec["A_nvfp4"]["hq"]
        rec["ratio_B_over_A"] = rec["B_weights_only"]["hq"] / rec["A_nvfp4"]["hq"]
        rows_out.append(rec)
        print(f"[{i+1:3d}/{len(names)}] {name:<44} {r}x{c}  "
              f"A {bpp_a:.3f}bpp {rec['A_nvfp4']['hq']:.5f} | "
              f"B {bpp_b:.3f} {rec['B_weights_only']['hq']:.5f} | "
              f"C {bpp_c:.3f} {rec['C_h_aware']['hq']:.5f} | "
              f"C/A {rec['ratio_C_over_A']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        del H, W, wa, wb, wc
        torch.cuda.empty_cache()
        with open(args.out, "w") as fh:
            json.dump({"units": rows_out, "provenance": prov, "partial": True}, fh, indent=1)

    # The control: the same weights-only arm, last, same process.
    print("\n== repeat control (weights-only, re-run last)", flush=True)
    control = []
    for name in names[: args.repeat]:
        W = get(src, name + ".weight", dev).to(torch.float32).contiguous()
        H = Hs[name].to(dev, torch.float32)
        w2, _ = encode(W, H, ldlq=False)
        again = hq(W, w2.float(), H)
        first = baseline_first[name]
        ok = abs(again - first) <= 1e-9 * max(1.0, abs(first))
        control.append({"name": name, "first": first, "again": again, "same": ok})
        print(f"   {name:<44} {first:.8f} -> {again:.8f}  "
              f"{'SAME' if ok else '!! DIFFER'}", flush=True)
        del H, W, w2
        torch.cuda.empty_cache()

    summary = {"n": len(rows_out),
               "geomean_C_over_A": geomean([r["ratio_C_over_A"] for r in rows_out]),
               "geomean_B_over_A": geomean([r["ratio_B_over_A"] for r in rows_out]),
               "by_role": {}}
    for role in sorted({r["role"] for r in rows_out}):
        sub = [r for r in rows_out if r["role"] == role]
        summary["by_role"][role] = {
            "n": len(sub),
            "C_over_A": geomean([r["ratio_C_over_A"] for r in sub]),
            "B_over_A": geomean([r["ratio_B_over_A"] for r in sub]),
            "A_hq": geomean([r["A_nvfp4"]["hq"] for r in sub]),
            "C_hq": geomean([r["C_h_aware"]["hq"] for r in sub]),
        }

    print(f"\n{'role':<12} {'n':>3} {'A hq':>9} {'C hq':>9} {'B/A':>8} {'C/A':>8}")
    for role, s in summary["by_role"].items():
        print(f"{role:<12} {s['n']:>3} {s['A_hq']:>9.5f} {s['C_hq']:>9.5f} "
              f"{s['B_over_A']:>8.4f} {s['C_over_A']:>8.4f}", flush=True)
    print(f"\ngeomean over {summary['n']} Linears: B/A {summary['geomean_B_over_A']:.4f}  "
          f"C/A {summary['geomean_C_over_A']:.4f}", flush=True)

    with open(args.out, "w") as fh:
        json.dump({"units": rows_out, "control": control, "summary": summary,
                   "provenance": prov, "args": vars(args), "partial": False}, fh, indent=1)
    print(f"wrote {args.out}  [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
