"""The FP4-native rate/plane frontier: where should a bit go, the plane or L?

Every arm decodes to E2M1 codes x a per-16 E4M3 scale the MMA reads.  The
plane axis is how many per-16 slots share one stored scale (1 = NVFP4's
flat per-16 E4M3 at 0.5 bpp, 2 = per-32 at 0.25, 4 = per-64 at 0.125, 8 =
per-128 at 0.0625; the S6b two-tier plane at 0.5 is today's wire), the rate
axis is Wei's L (payload 4 - 1/(2L)).  Each cell gets the same least-squares
refit landed on its own plane (shared scales use the exact shared LS, not the
mean of per-half fits), so cells differ only in where their bits are.  Scored
as the lever battery scores: held-out 128 tokens, served activation
quantiser, decoder start state fixed.

    PYTHONPATH=src:experiments:/home/rob/prismaquant python experiments/tessera_rate_plane_frontier.py
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera.trellis import TCQ  # noqa: E402
from tessera_memory_and_codebook import labels  # noqa: E402
from prismaquant.nvfp4_activation_contract import (  # noqa: E402
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

HALF = F.HALF


def ls_shared(w, q, prev, k):
    """Exact least-squares scale shared by k consecutive halves, keep-previous
    where the codes are all zero."""
    R, C = w.shape
    num = (w * q).reshape(R, C // HALF, HALF).sum(-1).reshape(-1, k).sum(1)
    den = (q * q).reshape(R, C // HALF, HALF).sum(-1).reshape(-1, k).sum(1)
    prev_k = prev.reshape(-1, k)[:, 0]
    s = torch.where(den > 0, num / den.clamp_min(1e-30), prev_k)
    s = torch.where(s > 0, s, prev_k)
    return s.repeat_interleave(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--shares", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="experiments/results/tessera_rate_plane_frontier.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    F.PROD = torch.tensor([F.GRID.vector(c) for c in range(F.GRID.size)], dtype=torch.float32, device="cuda")
    F.COSET = labels(TCQ(F.FOREST, F.CC).subsets, F.FOREST.anchors, F.GRID.size, "cuda")
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"tensors": [], "act": [], "arms": {}, "exl3_k4": F.EXL3_K4, "args": vars(a)}

    def enc(L):
        return (lambda w_, s_: F.trellis(w_, s_)) if L == 1 else (lambda w_, s_: F.multidim(w_, s_, L))

    for layer in a.layers:
        blob = torch.load(f"{F.ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        n = xa.shape[0] // 2
        x_fit, x_ev = xa[:n].contiguous(), xa[n:].contiguous()
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
            log(f"\n== L{layer} {proj} {tuple(w.shape)}  act leg served {act:.5f}")
            log(f"    {'arm':<34} {'bpp':>6} {'wt':>7} {'out':>7} {'W4A4':>7}")
            t0 = time.time()

            def rec(nm, hat, bpp):
                r_ = {"wt": float((hat - w).norm() / nw), "out": float((x_ev @ hat.T - y).norm() / ny),
                      "both_served": float((xq_s @ hat.T - y).norm() / ny), "act": act, "bpp": bpp}
                out["arms"].setdefault(nm, []).append(r_)
                log(f"    {nm:<34} {bpp:>6.4f} {r_['wt']:.5f} {r_['out']:.5f} {r_['both_served']:.5f}"
                    f"  {time.time() - t0:5.1f}s")

            G = F.E4M3_MAX / F.amax_half(w).max()
            amax = F.amax_half(w)
            for L in a.levels:
                payload = 4 - 1 / (2 * L)
                # today's wire: the S6b two-tier plane
                eff = F.s6b(w)
                q = enc(L)(w, F.expand(eff, R, C))
                for _ in range(a.iters):
                    eff = F.two_tier(F.ls_scale(w, q, eff))
                    q = enc(L)(w, F.expand(eff, R, C))
                rec(f"S6b per-16 + LS  L={L}", q * F.expand(eff, R, C), payload + 0.5)
                # flat E4M3 shared by k halves
                for k in a.shares:
                    eff = F.e4m3(amax.reshape(-1, k).amax(1).repeat_interleave(k), G)
                    q = enc(L)(w, F.expand(eff, R, C))
                    for _ in range(a.iters):
                        eff = F.e4m3(ls_shared(w, q, eff, k), G)
                        q = enc(L)(w, F.expand(eff, R, C))
                    rec(f"E4M3 per-{16 * k} + LS  L={L}", q * F.expand(eff, R, C), payload + 0.5 / k)
            json.dump(out, open(a.out, "w"), indent=1)
            del w; torch.cuda.empty_cache()
        del xa, x_fit, x_ev, xq_s; torch.cuda.empty_cache()

    mean = lambda v: sum(v) / len(v)
    base = out["arms"]["S6b per-16 + LS  L=1"]
    log("\n| arm | bpp | out-space weight leg | vs S6b L=1 | W4A4 | vs EXL3@A4 |")
    log("|---|---:|---:|---:|---:|---:|")
    rows = sorted(out["arms"].items(), key=lambda kv: (kv[1][0]["bpp"], kv[0]))
    for k, v in rows:
        ratio = mean([b["out"] / r["out"] for b, r in zip(base, v)])
        exl = mean([r["both_served"] / (F.EXL3_K4 ** 2 + r["act"] ** 2) ** 0.5 for r in v])
        log(f"| {k} | {v[0]['bpp']:.4f} | {mean([r['out'] for r in v]):.5f} | {ratio:.3f}x "
            f"| {mean([r['both_served'] for r in v]):.5f} | {exl:.3f}x |")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
