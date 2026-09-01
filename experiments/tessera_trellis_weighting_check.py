"""Does the trellis minimise the right thing?  Scale-weighted branch metric.

``viterbi_columns`` runs on ``work / scale`` -- targets normalised per half.
Unweighted, the path minimises ``sum (w/c - q)^2``: each half's error divided
by its own scale squared, so a column's quiet halves are served at the loud
halves' expense.  The per-16 scale varies by up to ~3.5 octaves within a
column (128x in weight).  ``encode_unit(trellis_weighting="scale")`` weights
each code's branch metric by ``c^2`` so the path minimises the true
``sum (w - c q)^2`` -- the objective the plane refit already descends.  No
wire change.  Same harness as ``tessera_wire_default_check.py``: six GLM
experts, held-out 1024 rows, served activation quantiser.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera_wire_default_check import ACT, CODE  # noqa: E402
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid  # noqa: E402
from tessera.decode import reconstruct_unit  # noqa: E402
from tessera.encode import encode_unit  # noqa: E402
from tessera.manifest import ScalePlaneKind  # noqa: E402
from prismaquant.nvfp4_activation_contract import (  # noqa: E402
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

ARMS = [
    # name, span, plane, refit, weighting, bpp
    ("span2 LUT refit4 unweighted (default)", 2, ScalePlaneKind.LUT, 4, "none", 4.00),
    ("span2 LUT refit4 scale-weighted",       2, ScalePlaneKind.LUT, 4, "scale", 4.00),
    ("span2 LUT refit0 unweighted",           2, ScalePlaneKind.LUT, 0, "none", 4.00),
    ("span2 LUT refit0 scale-weighted",       2, ScalePlaneKind.LUT, 0, "scale", 4.00),
    ("span1 S6b refit0 unweighted (export)",  1, ScalePlaneKind.S6B, 0, "none", 4.00),
    ("span1 S6b refit0 scale-weighted",       1, ScalePlaneKind.S6B, 0, "scale", 4.00),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--out", default="experiments/results/tessera_trellis_weighting_check.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    grid = tuple_grid(E2M1_GRID, 2)
    forests = {grid.rate_cap: build_forest(grid.rate_cap, grid=grid)}
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
    prior = json.load(open("experiments/results/tessera_wire_default_check.json"))
    out = {"tensors": [], "act": [], "arms": {}, "exl3_k4": F.EXL3_K4, "args": vars(a)}
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        n_fit = xa.shape[0] - a.eval_rows
        x_fit, x_ev = xa[:n_fit].contiguous().cuda(), xa[n_fit:].contiguous().cuda()
        g = select_mse_grid_input_global_scale([x_fit])
        xq_s = nvfp4_activation_qdq_served(x_ev, g).float()
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{F.SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            R, C = w.shape
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            act = float((xq_s @ w.T - y).norm() / ny)
            tname = f"L{layer}.{proj}"
            out["tensors"].append(tname); out["act"].append(act)
            log(f"\n== {tname} {tuple(w.shape)}  eval {x_ev.shape[0]} rows  act leg served {act:.5f}")
            log(f"    {'arm':<40} {'bpp':>5} {'wt':>8} {'out':>8} {'W4A4':>8} {'enc s':>6}")
            for nm, span, plane, refit, weighting, bpp in ARMS:
                torch.cuda.synchronize(); t0 = time.time()
                unit = encode_unit(w, forests, (grid.rate_cap,) * C, CODE, span=span,
                                   scale_plane=plane, scale_refit=refit, completion=0,
                                   trellis_weighting=weighting)
                torch.cuda.synchronize(); enc = time.time() - t0
                hat = reconstruct_unit(unit, forests, CODE)
                r_ = {"wt": float((hat - w).norm() / nw), "out": float((x_ev @ hat.T - y).norm() / ny),
                      "both_served": float((xq_s @ hat.T - y).norm() / ny), "act": act, "bpp": bpp,
                      "encode_s": enc}
                out["arms"].setdefault(nm, []).append(r_)
                log(f"    {nm:<40} {bpp:>5.2f} {r_['wt']:.5f} {r_['out']:.5f} {r_['both_served']:.5f} {enc:>6.1f}")
                del unit, hat
            # The unweighted arms must reproduce the wire-default check to the digit.
            i = prior["tensors"].index(tname)
            for nm, pnm in (("span2 LUT refit4 unweighted (default)", "span2 LUT refit4 (new default)"),
                            ("span1 S6b refit0 unweighted (export)", "span1 S6b refit0 (the 151 GiB export)")):
                mine, theirs = out["arms"][nm][-1]["out"], prior["arms"][pnm][i]["out"]
                log(f"    identity check {nm}: {mine:.6f} vs prior {theirs:.6f} "
                    f"{'OK' if abs(mine - theirs) < 1e-6 else 'DRIFT'}")
            json.dump(out, open(a.out, "w"), indent=1)
            del w; torch.cuda.empty_cache()
        del xa, x_fit, x_ev, xq_s; torch.cuda.empty_cache()

    mean = lambda v: sum(v) / len(v)
    base = out["arms"][ARMS[0][0]]
    log("\n| arm | bpp | out-space weight leg | vs default | W4A4 | vs EXL3@A4 | encode s |")
    log("|---|---:|---:|---:|---:|---:|---:|")
    for nm, *_ in ARMS:
        v = out["arms"][nm]
        ratios = [b["out"] / r["out"] for b, r in zip(base, v)]
        exl = mean([r["both_served"] / (F.EXL3_K4 ** 2 + r["act"] ** 2) ** 0.5 for r in v])
        log(f"| {nm} | {v[0]['bpp']:.2f} | {mean([r['out'] for r in v]):.5f} | {mean(ratios):.3f}x "
            f"({min(ratios):.3f}-{max(ratios):.3f}) | {mean([r['both_served'] for r in v]):.5f} | {exl:.3f}x "
            f"| {mean([r['encode_s'] for r in v]):.1f} |")
    for pair in ((0, 1), (2, 3), (4, 5)):
        u, wgt = out["arms"][ARMS[pair[0]][0]], out["arms"][ARMS[pair[1]][0]]
        ratios = [x["out"] / y["out"] for x, y in zip(u, wgt)]
        log(f"weighted / unweighted at {ARMS[pair[0]][0]!r}: {mean(ratios):.4f}x "
            f"({min(ratios):.4f}-{max(ratios):.4f})")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
