"""Where the codebook upgrades land against EXL3 under a 4-bit activation contract.

Under W4A16 the weight leg is the whole story.  Under W4A4 it is not: both
formats pay the same activation error, and a term common to both compresses
every ratio between them.  That is not a rhetorical device -- it is the contract
Tessera is actually aiming at, so it is the axis the comparison should be read
on.

The legs are MEASURED, not modelled.  Each arm is scored three ways on the
held-out half of real cached GLM activations: the weight leg alone (exact
activations), the activation leg alone (exact weights), and the composite that
actually ships.  Measuring the composite is what lets the quadrature rule
``both^2 = w^2 + act^2`` be *checked* rather than assumed -- and the check
matters, because EXL3 cannot be re-run here (it lives in the vLLM container) so
its composite has to be projected from its weight leg by exactly that rule.  A
rule verified on four arms and applied to a fifth is a projection; a rule
asserted is a guess.

**Read the EXL3 column with today's caveat.**  Its 0.05653 was fitted with a
real Hessian and LDL error compensation through a rank-128 Hessian over 4096
features, and scored on 128 held-out tokens adjacent to its 128 fit tokens.
Every Tessera arm here is uncompensated.  The gap is a mechanism and
calibration gap as much as a format gap -- see docs/tessera_rd_gap_anatomy.md --
so these ratios are an upper bound on Tessera's true disadvantage, not a
measurement of it.
"""
import argparse, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.codebook import learn_tree_codebook
from tessera.trellis import TCQ, ConvCode
from tessera_free_codebook_trellis import colour4, lloyd
from tessera_memory_and_codebook import labels, run

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
CC = ConvCode(memory=6)
EXL3_K4 = 0.05653


def quant_a4(x, group=16):
    """NVFP4-shaped activations: per-group-16 amax onto the E2M1 eight."""
    v = torch.tensor(E2M1_GRID.values, dtype=torch.float32, device=x.device)
    peak = float(v.abs().max())
    g = x.reshape(x.shape[0], -1, group)
    s = g.abs().amax(-1, keepdim=True).clamp(min=1e-12) / peak
    q = v[torch.cdist((g / s).reshape(-1, 1), v.reshape(-1, 1)).argmin(1)]
    return (q.reshape(g.shape) * s).reshape(x.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--out", default="experiments/results/tessera_w4a4_projection.json")
    a = ap.parse_args()

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2)
    prod = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(prod.abs().max())
    forest = build_forest(grid.rate_cap, grid=grid)
    coset = labels(TCQ(forest, CC).subsets, forest.anchors, grid.size, "cuda")

    acc = {}
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        x = xa[xa.shape[0] // 2:].contiguous()
        xq = quant_a4(x)
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            R, C = w.shape
            y = x @ w.T
            ny = y.norm()
            rel = lambda yh: float((yh - y).norm() / ny)
            acc.setdefault(("activation leg only", 0.0), []).append(rel(xq @ w.T))

            s = (w.reshape(R // 32, 32, C).abs().amax(1, keepdim=True)
                 .clamp(min=1e-12) / peak)
            xn = (w.reshape(R // 32, 32, C) / s).reshape(R, C)
            seq = xn.reshape(R // 2, 2, C).permute(0, 2, 1).contiguous()
            samp = seq.reshape(-1, 2)[::7][:300000].contiguous()
            tree = learn_tree_codebook(samp, depth=8)
            tvals = torch.tensor(tree.values, device="cuda").reshape(-1, 2)
            tforest = build_forest(grid.rate_cap, grid=tree)
            tlab = labels(TCQ(tforest, CC).subsets, tforest.anchors, tree.size, "cuda")
            flat = lloyd(samp, 256)

            for tag, cb, lb in (("product grid (today)", prod, coset),
                                ("tree codebook", tvals, tlab),
                                ("flat Lloyd (no ladder)", flat, colour4(flat))):
                q = run(seq, cb, lb, CC).permute(0, 2, 1).reshape(R, C)
                hat = (q.reshape(R // 32, 32, C) * s).reshape(R, C)
                acc.setdefault((tag, 4.0), []).append((rel(x @ hat.T), rel(xq @ hat.T)))
            print(f"  {layer:>3} {proj:<10} done", flush=True)
            del w, hat, seq
            torch.cuda.empty_cache()
        del xa, x, xq
        torch.cuda.empty_cache()

    m = lambda v: sum(v) / len(v)
    act = m(acc[("activation leg only", 0.0)])
    print(f"\nactivation leg (measured, exact weights): {act:.5f}")
    print(f"{'arm':<24}{'W4A16':>9}{'W4A4':>9}{'quadrature':>12}{'err':>7}"
          f"{'vs EXL3 A16':>13}{'vs EXL3 A4':>12}")
    rows = {}
    proj_exl3 = (EXL3_K4 ** 2 + act ** 2) ** 0.5
    for tag in ("product grid (today)", "tree codebook", "flat Lloyd (no ladder)"):
        w_leg = m([r[0] for r in acc[(tag, 4.0)]])
        both = m([r[1] for r in acc[(tag, 4.0)]])
        quad = (w_leg ** 2 + act ** 2) ** 0.5
        rows[tag] = {"w": w_leg, "both": both, "quad": quad}
        print(f"{tag:<24}{w_leg:>9.5f}{both:>9.5f}{quad:>12.5f}"
              f"{(quad/both-1)*100:>6.1f}%{w_leg/EXL3_K4:>12.2f}x"
              f"{both/proj_exl3:>11.2f}x")
    print(f"{'EXL3 K=4':<24}{EXL3_K4:>9.5f}{proj_exl3:>9.5f}{'(projected)':>12}"
          f"{'':>7}{1.0:>12.2f}x{1.0:>11.2f}x")
    json.dump({"activation_leg": act, "exl3_projected": proj_exl3, "arms": rows},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
