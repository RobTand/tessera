"""Issue #12: decompose the LUT plane's octave span, and test it causally.

``dense4_plane_census.py`` measures the handicap: holding alphabet, rounding
and weights fixed and changing **only the scale plane**, Tessera's sixteen
per-unit E4M3 entries cost more than NVFP4's one E4M3 byte per half -- and the
cost concentrates in ``q_proj`` and ``k_proj``.  This script asks *why their
field is wide* and then removes the width to see the handicap go.

Three synthetic controls, each removing one term of the span and leaving the
other two.  Every arm is the identical plane comparison (``lut16`` against
``e4m3``, E2M1 nearest, per-16 ``amax/6`` targets) run on a **rescaled** copy
of the same weight matrix, so the handicap is a within-arm ratio and the
rescaling cannot flatter either side:

* ``rownorm``  -- every output row rescaled to the same amax.  Removes the
  between-row term entirely; leaves the within-row term.
* ``headnorm`` -- every attention head (128 rows) rescaled to the same amax,
  rows inside a head untouched.  Removes only the *between-head* term.
  Qwen3 puts an RMSNorm on ``q_proj``'s and ``k_proj``'s outputs
  (``self_attn.q_norm`` / ``k_norm``, per head, over ``head_dim``) and none on
  ``v_proj``'s, so a head's overall output scale is exactly the quantity the
  norm divides out -- free for training to place anywhere.  If that is the
  mechanism, ``headnorm`` alone collapses the handicap on ``q``/``k`` and does
  nothing on ``v``.
* ``colnorm``  -- every *input column* rescaled to the same amax.  The null
  control: the plane's halves run along a row, so a column rescale changes the
  weights without removing any term of the field's span.

Rescaling is exact and invertible; it is a diagnostic and proposes no wire.

The causal arm (``--stage encode``) is the real encoder: the shipping E2M1x2
q896 wire (``ScalePlaneKind.LUT``) against the same encode with
``scale_plane=ScalePlaneKind.S6B`` -- a per-group E8M0 base plus a per-half
``(d, m)`` nibble, which spends more bits and can follow a wider field.
Weights-only on both arms, because S6b refuses ``refit_metric`` by design, and
one changed thing per comparison.  Every arm is priced at the bytes it wrote.
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

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.encode import _pack_scales_lut, e4m3_positive_values
from tessera.export import encode_linear_planes, wire_recipe
from tessera.manifest import ScalePlaneKind
from tessera.unit_artifact import read_unit_artifact

SRC = "/home/rob/models/Qwen3-0.6B"
HFULL = "/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt"
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
HALF, PEAK, HEAD_DIM = 16, 6.0, 128


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


def e2m1_rtn(W, s):
    vals = torch.tensor(E2M1_VALUES, device=W.device, dtype=torch.float32)
    halves = W.reshape(-1, HALF)
    x = halves / s[:, None]
    q = vals[(x.abs()[..., None] - vals).abs().argmin(dim=-1)] * torch.sign(x)
    return (q * s[:, None]).reshape(W.shape)


def plane_handicap(W):
    """lut16/e4m3 relative-Frobenius handicap on this exact matrix, plus the field."""
    t = W.reshape(-1, HALF).abs().amax(dim=1).clamp_min(1e-30) / PEAK
    g = float(t.max()) / 448.0
    grid = e4m3_positive_values(W.device)
    s_nv = grid[(t[:, None] / g - grid[None, :]).abs().argmin(dim=1)] * g
    _tb, _ix, s_lut, _gs = _pack_scales_lut(W, HALF)
    den = float((W * W).sum())

    def err(s):
        E = e2m1_rtn(W, s) - W
        return math.sqrt(float((E * E).sum()) / den)

    e_nv, e_lut = err(s_nv), err(s_lut)
    lg = torch.log2(t)
    tb = t.reshape(W.shape[0], W.shape[1] // HALF)
    rowmed = tb.median(dim=1).values
    return {"e4m3": e_nv, "lut16": e_lut, "handicap": e_lut / e_nv,
            "octaves": float(lg.max() - lg.min()),
            "between_rows_sd": float(torch.log2(rowmed).std()),
            "within_row_sd": float(torch.log2(tb / rowmed[:, None]).std())}


def rescale(W, mode):
    """An exact, invertible rescale that removes one term of the field's span."""
    if mode == "orig":
        return W
    if mode == "rownorm":
        a = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        return W / a
    if mode == "colnorm":
        a = W.abs().amax(dim=0, keepdim=True).clamp_min(1e-30)
        return W / a
    if mode == "headnorm":
        r = W.shape[0]
        if r % HEAD_DIM:
            return None
        h = W.reshape(r // HEAD_DIM, HEAD_DIM, W.shape[1])
        a = h.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30)
        return (h / a).reshape(W.shape)
    raise ValueError(mode)


def head_spread(W):
    """Octaves of amax spread BETWEEN heads and, for contrast, WITHIN a head."""
    r = W.shape[0]
    if r % HEAD_DIM:
        return None
    h = W.reshape(r // HEAD_DIM, HEAD_DIM, W.shape[1])
    per_head = h.abs().amax(dim=(1, 2))
    per_row = W.abs().amax(dim=1).reshape(r // HEAD_DIM, HEAD_DIM)
    return {
        "heads": int(r // HEAD_DIM),
        "between_head_octaves": float(torch.log2(per_head.max() / per_head.min())),
        "between_head_sd_log2": float(torch.log2(per_head).std()),
        "within_head_sd_log2": float(
            torch.log2(per_row / per_head[:, None]).std()),
    }


def geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="controls", choices=["controls", "encode"])
    ap.add_argument("--layers", default="")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    src = open_all(SRC)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(28)))
    roles = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    names = [f"model.layers.{L}.{r}" for L in layers for r in roles]

    if args.stage == "controls":
        out, t0, first = [], time.time(), {}
        for i, name in enumerate(names):
            W = src[name + ".weight"].get_tensor(name + ".weight").to(dev).float()
            rec = {"name": name, "role": name.split(".")[-1],
                   "layer": int(name.split(".")[2]), "shape": list(W.shape)}
            for mode in ("orig", "rownorm", "headnorm", "colnorm"):
                Wm = rescale(W, mode)
                if Wm is None:
                    continue
                rec[mode] = plane_handicap(Wm)
            rec["head"] = head_spread(W)
            first.setdefault(name, rec["orig"]["handicap"])
            out.append(rec)
            hn = rec.get("headnorm", {}).get("handicap")
            print(f"[{i+1:3d}/{len(names)}] {name:<44} orig {rec['orig']['handicap']:.4f} "
                  f"rownorm {rec['rownorm']['handicap']:.4f} "
                  f"headnorm {hn if hn is None else round(hn,4)} "
                  f"colnorm {rec['colnorm']['handicap']:.4f} "
                  f"| oct {rec['orig']['octaves']:.2f} [{time.time()-t0:.0f}s]", flush=True)
            del W
            if dev == "cuda":
                torch.cuda.empty_cache()
        # drift control, same arm, last
        print("\n== repeat control (orig handicap, re-run last)", flush=True)
        control = []
        for name in names[:4]:
            W = src[name + ".weight"].get_tensor(name + ".weight").to(dev).float()
            again = plane_handicap(W)["handicap"]
            ok = abs(again - first[name]) <= 1e-9 * max(1.0, first[name])
            control.append({"name": name, "first": first[name], "again": again, "same": ok})
            print(f"   {name:<44} {first[name]:.8f} -> {again:.8f} "
                  f"{'SAME' if ok else '!! DIFFER'}", flush=True)
        print(f"\n{'role':<11} {'n':>3} {'orig':>7} {'rownorm':>8} {'headnorm':>9} "
              f"{'colnorm':>8} {'oct':>6} {'btwHeadOct':>11} {'inHeadSd':>9}")
        summary = {}
        for role in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                     "up_proj", "down_proj"]:
            sub = [r for r in out if r["role"] == role]
            if not sub:
                continue
            hn = [r["headnorm"]["handicap"] for r in sub if "headnorm" in r]
            bh = [r["head"]["between_head_octaves"] for r in sub if r.get("head")]
            ih = [r["head"]["within_head_sd_log2"] for r in sub if r.get("head")]
            summary[role] = {
                "n": len(sub),
                "orig": geomean([r["orig"]["handicap"] for r in sub]),
                "rownorm": geomean([r["rownorm"]["handicap"] for r in sub]),
                "headnorm": geomean(hn) if hn else None,
                "colnorm": geomean([r["colnorm"]["handicap"] for r in sub]),
                "octaves": statistics.median([r["orig"]["octaves"] for r in sub]),
                "between_head_octaves": statistics.median(bh) if bh else None,
                "within_head_sd": statistics.median(ih) if ih else None,
            }
            s = summary[role]
            print(f"{role:<11} {s['n']:>3} {s['orig']:>7.4f} {s['rownorm']:>8.4f} "
                  f"{'   n/a  ' if s['headnorm'] is None else format(s['headnorm'], '>9.4f')} "
                  f"{s['colnorm']:>8.4f} {s['octaves']:>6.2f} "
                  f"{'  n/a  ' if s['between_head_octaves'] is None else format(s['between_head_octaves'], '>11.2f')} "
                  f"{'  n/a  ' if s['within_head_sd'] is None else format(s['within_head_sd'], '>9.3f')}",
                  flush=True)
        with open(args.out, "w") as fh:
            json.dump({"units": out, "control": control, "summary": summary,
                       "args": vars(args)}, fh, indent=1)
        print(f"\nwrote {args.out} [{time.time()-t0:.0f}s]", flush=True)
        return 0

    # --- causal: the real encoder, one changed thing (the plane)
    blob = torch.load(HFULL, map_location="cpu")
    Hs = blob["H"]
    grid = tuple_grid(E2M1_GRID, 2)
    print(f"wire_recipe(E2M1x2, {args.q256}) = {wire_recipe(grid, args.q256)}", flush=True)
    out, t0, first = [], time.time(), {}
    for i, name in enumerate(names):
        W = src[name + ".weight"].get_tensor(name + ".weight").to(dev).float().contiguous()
        H = Hs[name].to(dev, torch.float32)
        rec = {"name": name, "role": name.split(".")[-1],
               "layer": int(name.split(".")[2]), "shape": list(W.shape)}
        for tag, plane in (("lut", None), ("s6b", ScalePlaneKind.S6B)):
            exported, _u, _f = encode_linear_planes(
                W, grid=grid, q256=args.q256, name="u", verify=False,
                scale_plane=plane)
            Wd = read_unit_artifact(exported.blob, device=dev).float()
            rec[tag] = {"bpp": float(exported.bpp), "hq": hq(W, Wd, H)}
            del Wd
        rec["s6b_over_lut"] = rec["s6b"]["hq"] / rec["lut"]["hq"]
        rec["d_bpp"] = rec["s6b"]["bpp"] - rec["lut"]["bpp"]
        first.setdefault(name, rec["lut"]["hq"])
        out.append(rec)
        print(f"[{i+1:3d}/{len(names)}] {name:<44} lut {rec['lut']['hq']:.5f} "
              f"@{rec['lut']['bpp']:.4f} | s6b {rec['s6b']['hq']:.5f} "
              f"@{rec['s6b']['bpp']:.4f} | s6b/lut {rec['s6b_over_lut']:.4f} "
              f"(+{rec['d_bpp']:.3f} bpp) [{time.time()-t0:.0f}s]", flush=True)
        del W, H
        if dev == "cuda":
            torch.cuda.empty_cache()
        with open(args.out, "w") as fh:
            json.dump({"units": out, "partial": True, "args": vars(args)}, fh, indent=1)
    print("\n== repeat control (lut arm, re-run last)", flush=True)
    control = []
    for name in names[:2]:
        W = src[name + ".weight"].get_tensor(name + ".weight").to(dev).float().contiguous()
        H = Hs[name].to(dev, torch.float32)
        exported, _u, _f = encode_linear_planes(
            W, grid=grid, q256=args.q256, name="u", verify=False)
        again = hq(W, read_unit_artifact(exported.blob, device=dev).float(), H)
        ok = abs(again - first[name]) <= 1e-9 * max(1.0, first[name])
        control.append({"name": name, "first": first[name], "again": again, "same": ok})
        print(f"   {name:<44} {first[name]:.8f} -> {again:.8f} "
              f"{'SAME' if ok else '!! DIFFER'}", flush=True)
    print(f"\n{'role':<11} {'n':>3} {'s6b/lut':>8} {'d bpp':>7}")
    summary = {}
    for role in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                 "up_proj", "down_proj"]:
        sub = [r for r in out if r["role"] == role]
        if not sub:
            continue
        summary[role] = {"n": len(sub),
                         "s6b_over_lut": geomean([r["s6b_over_lut"] for r in sub]),
                         "d_bpp": statistics.median([r["d_bpp"] for r in sub])}
        print(f"{role:<11} {summary[role]['n']:>3} "
              f"{summary[role]['s6b_over_lut']:>8.4f} {summary[role]['d_bpp']:>7.3f}",
              flush=True)
    with open(args.out, "w") as fh:
        json.dump({"units": out, "control": control, "summary": summary,
                   "partial": False, "args": vars(args)}, fh, indent=1)
    print(f"\nwrote {args.out} [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
