"""The new default wire, built by the PRODUCTION encoder, on the real tensors.

``experiments/tessera_index_plane.py`` measured the index plane + Wei L=2 at
1.111x over the per-16 E4M3 reference with a harness that warm-started the
LUT from a full per-16 refit encode.  The production path cold-starts the LUT
from the amax targets and alternates from there (``encode._pack_scales_lut``,
``_refit_scales_lut``).  This script is the number that matters for the
default flip: ``encode_unit`` at (span 2, LUT, refit 4) against ``encode_unit``
at (span 1, S6b, refit 4) -- the artifact today -- on the six GLM experts, the
held-out eval rows and the served activation quantiser of every earlier
measurement.  Attribution arms isolate the plane and the trellis.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid  # noqa: E402
from tessera.decode import reconstruct_unit  # noqa: E402
from tessera.encode import encode_unit  # noqa: E402
from tessera.manifest import ScalePlaneKind  # noqa: E402
from tessera.trellis import ConvCode  # noqa: E402
from prismaquant.nvfp4_activation_contract import (  # noqa: E402
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"
CODE = ConvCode(memory=6)
ARMS = [
    # name, span, plane, refit, bpp (E2M1x2 at the cap; forest overhead ~0)
    ("span1 S6b refit4 (today's default)", 1, ScalePlaneKind.S6B, 4, 4.00),
    ("span2 LUT refit4 (new default)",     2, ScalePlaneKind.LUT, 4, 4.00),
    ("span2 LUT refit6",                    2, ScalePlaneKind.LUT, 6, 4.00),
    ("span1 LUT refit4 (plane alone)",      1, ScalePlaneKind.LUT, 4, 3.75),
    ("span2 S6b refit4 (trellis alone)",    2, ScalePlaneKind.S6B, 4, 4.25),
    ("span1 S6b refit0 (the 151 GiB export)", 1, ScalePlaneKind.S6B, 0, 4.00),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--out", default="experiments/results/tessera_wire_default_check.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    grid = tuple_grid(E2M1_GRID, 2)
    forests = {grid.rate_cap: build_forest(grid.rate_cap, grid=grid)}
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
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
            out["tensors"].append(f"L{layer}.{proj}"); out["act"].append(act)
            log(f"\n== L{layer} {proj} {tuple(w.shape)}  eval {x_ev.shape[0]} rows  act leg served {act:.5f}")
            log(f"    {'arm':<40} {'bpp':>5} {'wt':>8} {'out':>8} {'W4A4':>8} {'enc s':>6}")
            for nm, span, plane, refit, bpp in ARMS:
                torch.cuda.synchronize(); t0 = time.time()
                unit = encode_unit(w, forests, (grid.rate_cap,) * C, CODE, span=span,
                                   scale_plane=plane, scale_refit=refit, completion=0)
                torch.cuda.synchronize(); enc = time.time() - t0
                hat = reconstruct_unit(unit, forests, CODE)
                r_ = {"wt": float((hat - w).norm() / nw), "out": float((x_ev @ hat.T - y).norm() / ny),
                      "both_served": float((xq_s @ hat.T - y).norm() / ny), "act": act, "bpp": bpp,
                      "encode_s": enc, "sse": unit.sse}
                if plane is ScalePlaneKind.LUT:
                    r_["lut"] = unit.scale_lut.tolist(); r_["global"] = unit.scale_global
                out["arms"].setdefault(nm, []).append(r_)
                log(f"    {nm:<40} {bpp:>5.2f} {r_['wt']:.5f} {r_['out']:.5f} {r_['both_served']:.5f} {enc:>6.1f}")
                del unit, hat
            json.dump(out, open(a.out, "w"), indent=1)
            del w; torch.cuda.empty_cache()
        del xa, x_fit, x_ev, xq_s; torch.cuda.empty_cache()

    mean = lambda v: sum(v) / len(v)
    base = out["arms"][ARMS[0][0]]
    log("\n| arm | bpp | out-space weight leg | vs today's default | W4A4 | vs EXL3@A4 | encode s |")
    log("|---|---:|---:|---:|---:|---:|---:|")
    for nm, *_ in ARMS:
        v = out["arms"][nm]
        ratio = mean([b["out"] / r["out"] for b, r in zip(base, v)])
        exl = mean([r["both_served"] / (F.EXL3_K4 ** 2 + r["act"] ** 2) ** 0.5 for r in v])
        log(f"| {nm} | {v[0]['bpp']:.2f} | {mean([r['out'] for r in v]):.5f} | {ratio:.3f}x "
            f"| {mean([r['both_served'] for r in v]):.5f} | {exl:.3f}x | {mean([r['encode_s'] for r in v]):.1f} |")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
