"""Does LDLQ's 1.134x survive a Hessian fit on OTHER documents?

Every LDLQ number so far (`tessera_fp4_native_levers.py` F2,
`tessera_plane_alternatives.py`) fit the Hessian on the first 128 of 256
random rows of ONE calibration document and scored on the other 128 -- shared
low-rank structure flatters generalisation, and the Hessian had rank <=128
over 4096 features.  The 2026-09-01 capture keeps every token of every
document in order (16 x 512), so fit and eval can be split by whole
documents.  Two folds (fit docs 0-7 / eval 8-15 and the reverse), the
regulariser swept again because a full-rank Hessian may want a different
sigma, and the old adjacent-halves-of-one-document split kept as the
control that reproduces the screen.

    PYTHONPATH=src:experiments:/home/rob/prismaquant python experiments/tessera_ldlq_generalisation.py \
        --act /mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera_plane_alternatives import full_ldlq_refit  # noqa: E402
from tessera.compensate import regularize_hessian  # noqa: E402
from tessera.trellis import TCQ  # noqa: E402
from tessera_memory_and_codebook import labels  # noqa: E402
from prismaquant.nvfp4_activation_contract import (  # noqa: E402
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)


def full_ldlq_refit_per32_L2(w, Hreg, G, iters):
    """The frontier's 4.0 bpp point (one E4M3 per 32, Wei L=2) inside the same
    LDLQ schedule -- LDLQ x L=2 has no number otherwise."""
    from tessera.compensate import block_ldl, compensated_targets
    from tessera_rate_plane_frontier import ls_shared
    Ld = block_ldl(Hreg, F.GROUP)

    def encode(target, start, stop):
        eff = F.e4m3(F.amax_half(target).reshape(-1, 2).amax(1).repeat_interleave(2), G)
        q = F.multidim(target, F.expand(eff, *target.shape), 2)
        for _ in range(iters):
            eff = F.e4m3(ls_shared(target, q, eff, 2), G)
            q = F.multidim(target, F.expand(eff, *target.shape), 2)
        return q * F.expand(eff, *target.shape)

    return compensated_targets(w, Ld, encode, block=F.GROUP)[1]


def refit_only(w, plane, requant, iters):
    eff = plane(w)
    q = F.trellis(w, F.expand(eff, *w.shape))
    for _ in range(iters):
        eff = requant(F.ls_scale(w, q, eff))
        q = F.trellis(w, F.expand(eff, *w.shape))
    return q * F.expand(eff, *w.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.025, 0.1, 1.0, 3.0],
                    help="0.025 is EXL3's own and stays in: a head-to-head must damp both sides alike")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--eval-rows", type=int, default=1024,
                    help="rows scored per fold (a random subset of the eval documents' tokens)")
    ap.add_argument("--out", default="experiments/results/tessera_ldlq_generalisation.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    F.PROD = torch.tensor([F.GRID.vector(c) for c in range(F.GRID.size)], dtype=torch.float32, device="cuda")
    F.COSET = labels(TCQ(F.FOREST, F.CC).subsets, F.FOREST.anchors, F.GRID.size, "cuda")
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"tensors": [], "folds": {}, "arms": {}, "exl3_k4": F.EXL3_K4, "args": vars(a)}
    half_docs = a.docs // 2
    folds = {"fit0-7/eval8-15": (range(0, half_docs), range(half_docs, a.docs)),
             "fit8-15/eval0-7": (range(half_docs, a.docs), range(0, half_docs)),
             "control:doc0 halves": None}
    gen = torch.Generator().manual_seed(0)

    for layer in a.layers:
        blob = torch.load(f"{a.act}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        assert xa.shape[0] == a.docs * a.seqlen, (xa.shape, a.docs, a.seqlen)
        docs = xa.reshape(a.docs, a.seqlen, -1)
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{F.SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            out["tensors"].append(f"L{layer}.{proj}")
            G = F.E4M3_MAX / F.amax_half(w).max()
            e4m3_plane = lambda t: F.e4m3(F.amax_half(t), G)
            e4m3_requant = lambda s_: F.e4m3(s_, G)
            for fold, split in folds.items():
                if split is None:
                    d0 = docs[0]
                    perm = torch.randperm(a.seqlen, generator=gen)
                    x_fit = d0[perm[: a.seqlen // 2]].cuda().contiguous()
                    x_ev = d0[perm[a.seqlen // 2:]].cuda().contiguous()
                else:
                    fit_docs, ev_docs = split
                    x_fit = docs[list(fit_docs)].reshape(-1, docs.shape[-1]).cuda().contiguous()
                    x_all = docs[list(ev_docs)].reshape(-1, docs.shape[-1])
                    perm = torch.randperm(x_all.shape[0], generator=gen)[: a.eval_rows]
                    x_ev = x_all[perm].cuda().contiguous()
                n = x_fit.shape[0]
                g = select_mse_grid_input_global_scale([x_fit])
                xq_s = nvfp4_activation_qdq_served(x_ev, g).float()
                y = x_ev @ w.T
                ny, nw = y.norm(), w.norm()
                act = float((xq_s @ w.T - y).norm() / ny)
                out["folds"].setdefault(fold, []).append({"fit_rows": n, "eval_rows": int(x_ev.shape[0]), "act": act})
                log(f"\n== L{layer} {proj} [{fold}] fit {n} rows, eval {x_ev.shape[0]} rows, act leg {act:.5f}")
                t0 = time.time()

                def rec(nm, hat):
                    r_ = {"wt": float((hat - w).norm() / nw), "out": float((x_ev @ hat.T - y).norm() / ny),
                          "both_served": float((xq_s @ hat.T - y).norm() / ny), "act": act,
                          "fold": fold, "tensor": f"L{layer}.{proj}"}
                    out["arms"].setdefault(nm, []).append(r_)
                    log(f"    {nm:<44} wt={r_['wt']:.5f} out={r_['out']:.5f} W4A4={r_['both_served']:.5f}"
                        f"  {time.time() - t0:6.1f}s")

                rec("refit only (s6b)", refit_only(w, F.s6b, F.two_tier, a.iters))
                rec("refit only (e4m3)", refit_only(w, e4m3_plane, e4m3_requant, a.iters))
                H = (x_fit.T @ x_fit).contiguous()
                for sig in a.sigmas:
                    Hreg = regularize_hessian(H, count=n, sigma_reg=sig)
                    rec(f"LDLQ s={sig} + in-block LS (e4m3)",
                        full_ldlq_refit(w, Hreg, e4m3_plane, e4m3_requant, a.iters))
                    if sig == 1.0:
                        rec(f"LDLQ s={sig} + in-block LS (s6b)",
                            full_ldlq_refit(w, Hreg, F.s6b, F.two_tier, a.iters))
                        rec(f"LDLQ s={sig} + in-block LS (per-32 e4m3, L=2)",
                            full_ldlq_refit_per32_L2(w, Hreg, G, a.iters))
                    del Hreg
                del H, x_fit, x_ev, xq_s, y
                torch.cuda.empty_cache()
                json.dump(out, open(a.out, "w"), indent=1)
            del w; torch.cuda.empty_cache()

    mean = lambda v: sum(v) / len(v)
    log("\n| fold | arm | out-space weight leg | vs refit-only (same plane) | W4A4 |")
    log("|---|---|---:|---:|---:|")
    for fold in folds:
        rows = {k: [r for r in v if r["fold"] == fold] for k, v in out["arms"].items()}
        for k, v in rows.items():
            plane = "e4m3" if "e4m3" in k else "s6b"   # the L=2 per-32 arm reads against the e4m3 refit-only
            base = rows[f"refit only ({plane})"]
            ratio = mean([b["out"] / r["out"] for b, r in zip(base, v)])
            log(f"| {fold} | {k} | {mean([r['out'] for r in v]):.5f} | {ratio:.3f}x | {mean([r['both_served'] for r in v]):.5f} |")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
