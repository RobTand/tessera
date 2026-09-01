"""How does LDLQ's out-of-document gain scale with calibration tokens?

The theoretical-limits run says ideal feedback is worth 1.16x with the
sigma_reg=1.0 Hessian a 4096-token capture forces and 1.91x with the true
one.  Before asking for a 200k-token capture, measure the slope: fit H on
the first k documents (k in --fit-docs), evaluate on a FIXED tail of
documents, at several sigma.  Same arms as `tessera_ldlq_generalisation`.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera_plane_alternatives import full_ldlq_refit  # noqa: E402
from tessera_ldlq_generalisation import refit_only  # noqa: E402
from tessera.compensate import regularize_hessian  # noqa: E402
from tessera.trellis import TCQ  # noqa: E402
from tessera_memory_and_codebook import labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj"])
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--fit-docs", type=int, nargs="+", default=[2, 4, 8, 14])
    ap.add_argument("--eval-docs", type=int, default=2, help="the LAST n documents, fixed across fits")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--out", default="experiments/results/tessera_ldlq_token_scaling.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    F.PROD = torch.tensor([F.GRID.vector(c) for c in range(F.GRID.size)], dtype=torch.float32, device="cuda")
    F.COSET = labels(TCQ(F.FOREST, F.CC).subsets, F.FOREST.anchors, F.GRID.size, "cuda")
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"args": vars(a), "rows": []}
    gen = torch.Generator().manual_seed(0)
    for layer in a.layers:
        blob = torch.load(f"{a.act}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        assert xa.shape[0] == a.docs * a.seqlen, (xa.shape, a.docs, a.seqlen)
        docs = xa.reshape(a.docs, a.seqlen, -1)
        ev_all = docs[a.docs - a.eval_docs:].reshape(-1, docs.shape[-1])
        perm = torch.randperm(ev_all.shape[0], generator=gen)[: a.eval_rows]
        x_ev = ev_all[perm].cuda().contiguous()
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{F.SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            G = F.E4M3_MAX / F.amax_half(w).max()
            e4m3_plane = lambda t: F.e4m3(F.amax_half(t), G)
            e4m3_requant = lambda s_: F.e4m3(s_, G)
            y = x_ev @ w.T
            base = refit_only(w, e4m3_plane, e4m3_requant, a.iters)
            out_base = float(((x_ev @ base.T - y).norm() / y.norm()))
            log(f"== L{layer} {proj}: refit-only (flat E4M3 plane) out={out_base:.5f} on the last {a.eval_docs} docs")
            for k in a.fit_docs:
                assert k <= a.docs - a.eval_docs
                x_fit = docs[:k].reshape(-1, docs.shape[-1]).cuda()
                n = x_fit.shape[0]
                H = (x_fit.T @ x_fit).contiguous()
                del x_fit
                for sig in a.sigmas:
                    t0 = time.time()
                    Hreg = regularize_hessian(H, count=n, sigma_reg=sig)
                    hat = full_ldlq_refit(w, Hreg, e4m3_plane, e4m3_requant, a.iters)
                    o = float(((x_ev @ hat.T - y).norm() / y.norm()))
                    row = {"layer": layer, "proj": proj, "fit_docs": k, "fit_rows": n, "sigma": sig,
                           "out": o, "out_base": out_base, "gain": out_base / o}
                    out["rows"].append(row)
                    log(f"   fit {k:2d} docs ({n:5d} rows) sigma={sig:<5} out={o:.5f} gain {out_base / o:.3f}x  {time.time()-t0:.0f}s")
                    del Hreg, hat
                del H
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
