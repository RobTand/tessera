"""The bitshift trellis on the E2M1x2 tile: does a state-dependent window map
beat the coset-partition trellis at Tessera-4's own rung?

Tessera-4 at the cap (K=7 bits per pair, 3.5 b/wt body) has every tuple as an
anchor and two fixed coset families; memory and code are measured closed
there (0.0-0.3%). A bitshift trellis with L bits of state reaches a
different 2^K-tuple subset per state, chosen from all 256 tuples by a
Gaussian-density-matched random map. Sub-cap rungs (K=6, 3.0 b/wt; K=5,
2.5 b/wt) are where the builder's k-d anchors lose to Gridbook FP4-CB.

Gaussian oracle: 2048 rows x 512 cols of N(0,1) as 1024 pairs down each
column, each arm at its own best global scale (the scale is a free per-group
parameter, so a comparator gets the one that suits it -- fp8-band lesson).
Experts: per-16 amax/6 fp32 block scale + LS refit x2 for both arms, priced
0.5 bpp (E4M3 scale), so the K=7 arms land at 4.0 bpp against the default
wire's 4.0 (3.75 body + 0.25 LUT plane).

Screen, principle 3.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, E2M1_VALUES, PayloadGrid, build_forest, tuple_grid
from tessera.encode import viterbi_columns
from tessera.trellis import ConvCode
from tessera8_bounds import gaussian_sample
from tessera8_targets import ACT, EXL3, SRC, UNGERBOECK6, LARSEN6, UNG6
from tessera_bitshift_tile import bitshift_viterbi

E2M1 = torch.tensor(sorted(set(E2M1_VALUES)))                  # 15 distinct values (one zero)
GRID2 = tuple_grid(E2M1_GRID, 2)
TUPLES = torch.tensor([GRID2.vector(c) for c in range(GRID2.size)])   # [256, 2]


def snap_tuple(v: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 value per coordinate: [.., 2] -> [.., 2] on the tuple grid."""
    grid = E2M1.to(v.device)
    idx = torch.bucketize(v, grid).clamp(1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    return torch.where((v - lo).abs() <= (hi - v).abs(), lo, hi)


def window_map2(L: int, kind: str, sigma: float, seed: int, device) -> torch.Tensor:
    """[2^L, 2] reconstruction pairs in grid units (sigma = source std in grid units)."""
    S = 1 << L
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(S, 2, generator=g).to(device) * sigma
    if kind == "free":
        return v
    if kind == "e2m1x2":
        return snap_tuple(v)
    raise ValueError(kind)


def tessera_pairs(x2: torch.Tensor, K: int, code: ConvCode) -> torch.Tensor:
    """Tessera's own trellis over the tuple grid: x2 [steps, cols, 2] -> hat."""
    steps, cols, _ = x2.shape
    flat = x2.permute(0, 2, 1).reshape(steps * 2, cols).contiguous()   # pairs = consecutive rows
    forest = build_forest(K, grid=GRID2)
    anchors, _, _ = viterbi_columns(flat, forest, code, completion=0)
    # viterbi_columns returns the ANCHOR index; the grid code is forest.anchors[index]
    codes = torch.tensor(forest.anchors, device=x2.device)[anchors.long()]
    hat = TUPLES.to(x2.device)[codes]                                     # [steps, cols, 2]
    return hat


def best_scale(fn, x2: torch.Tensor, sigmas) -> tuple[float, float]:
    """Run ``fn(x2 / s)`` at each grid-unit scale, return (best rms, sigma)."""
    best = (float("inf"), None)
    for sg in sigmas:
        # source std 1.0 -> grid units: multiply by sg (sigma in grid units)
        hat = fn(x2 * sg) / sg
        r = float((hat - x2).pow(2).mean().sqrt())
        if r < best[0]:
            best = (r, sg)
    return best


def gaussian_oracle(a, log, out):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(a.rows, a.gauss_cols, device=dev)
    x2 = x.view(a.rows // 2, 2, a.gauss_cols).permute(0, 2, 1).contiguous()   # [steps, cols, 2]
    rows = {}
    for K in a.rates:
        ref = 2.0 ** -(K / 2)                                           # Shannon at K/2 b/wt
        for name, code in (("larsen6", LARSEN6), ("ung6", UNG6)):
            t0 = time.time()
            r, sg = best_scale(lambda t: tessera_pairs(t, K, code), x2, a.sigmas)
            rows[f"K={K} tessera-coset-{name}"] = {"ratio": r / ref, "sigma": sg, "s": time.time() - t0}
            log(f"  K={K} {f'Tessera coset trellis {name}':50s} {r/ref:.4f}x  sigma={sg:.2f}  {time.time()-t0:5.1f}s")
        for kind in a.kinds:
            for L in a.memories:
                if L < K + 1:
                    continue
                t0 = time.time()

                def run(t, L=L, kind=kind):
                    book = window_map2(L, kind, 1.0, a.seed, dev)     # unit-sigma map ...
                    # ... scaled with the data: the map's sigma tracks the source's grid sigma
                    sg = float(t.std())
                    book = book * sg if kind == "free" else snap_tuple(book * sg)
                    st, _ = bitshift_viterbi(t, book, L, K, chunk=a.chunk)
                    return book[st.long()]

                r, sg = best_scale(run, x2, a.sigmas)
                rows[f"K={K} bitshift-{kind}-L{L}"] = {"ratio": r / ref, "sigma": sg, "L": L, "s": time.time() - t0}
                log(f"  K={K} {f'bitshift {kind:8s} L={L:2d}':50s} {r/ref:.4f}x  sigma={sg:.2f}  {time.time()-t0:5.1f}s")
    out["gaussian"] = rows


def expert_arms(a, log, out):
    from safetensors import safe_open
    from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm
    from prismaquant.nvfp4_activation_contract import (
        nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    out["experts"] = {}
    dev = "cuda"
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        n_fit = xa.shape[0] - a.eval_rows
        x_fit, x_ev = xa[:n_fit].contiguous().cuda(), xa[n_fit:].contiguous().cuda()
        g = select_mse_grid_input_global_scale([x_fit])
        xq4 = nvfp4_activation_qdq_served(x_ev, g).float()
        xq8 = fp8_dynamic_activation_qdq_vllm(x_ev).dequant.float()
        del x_fit, xa
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            tname = f"L{layer}.{proj}"
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            res = {}
            log(f"\n== {tname} {tuple(w.shape)}")

            def rec(arm, hat, bpp):
                r = {"bpp": bpp, "wt": float((hat - w).norm() / nw),
                     "out": float((x_ev @ hat.T - y).norm() / ny),
                     "a4": float((xq4 @ hat.T - y).norm() / ny),
                     "a8": float((xq8 @ hat.T - y).norm() / ny)}
                res[arm] = r
                log(f"    {arm:<56} {bpp:6.3f} {r['wt']:8.5f} {r['out']:8.5f} {r['a4']:8.5f} {r['a8']:8.5f}")

            for Kx in a.exl3:
                p = Path(EXL3) / f"L{layer}_{proj}_K{Kx}.pt"
                if p.exists():
                    rec(f"EXL3 K={Kx}", torch.load(p, map_location=dev).float(), Kx + 0.011723)
            R_, C = w.shape
            # per-16 block scale along the input dim (the NVFP4 block), amax / 6, fp32
            wb = w.view(R_, C // 16, 16)
            s0 = wb.abs().amax(dim=2, keepdim=True) / 6.0
            for K in a.rates:
                bpp = K / 2 + 0.5
                for name, code in (("larsen6", LARSEN6), ("ung6", UNG6)):
                    s = s0.clone()
                    for it in range(a.refits + 1):
                        t = (wb / s).view(R_, C)                                    # grid units
                        # pairs are consecutive ROWS down a column for the tuple grid
                        forest = build_forest(K, grid=GRID2)
                        anchors, _, _ = viterbi_columns(t.contiguous(), forest, code, completion=0)
                        codes = torch.tensor(forest.anchors, device=dev)[anchors.long()]
                        q = TUPLES.to(dev)[codes]                                   # [R_/2, C, 2]
                        q = q.permute(0, 2, 1).reshape(R_, C).view(R_, C // 16, 16)
                        if it < a.refits:
                            num, den = (wb * q).sum(2, keepdim=True), (q * q).sum(2, keepdim=True)
                            s = torch.where(den > 0, num / den, s)
                    rec(f"Tessera-4 K={K} coset {name}, per-16 fp32 scale refit{a.refits}", (q * s).view(R_, C), bpp)
                for L in a.expert_memories:
                    if L < K + 1:
                        continue
                    s = s0.clone()
                    t0 = time.time()
                    for it in range(a.refits + 1):
                        t = (wb / s).view(R_, C)
                        t2 = t.view(R_ // 2, 2, C).permute(0, 2, 1).contiguous()    # [steps, cols, 2]
                        book = snap_tuple(window_map2(L, "free", 1.0, a.seed, dev) * float(t.std()))
                        st, _ = bitshift_viterbi(t2, book, L, K, chunk=a.chunk)
                        q = book[st.long()]                                         # [steps, C, 2]
                        q = q.permute(0, 2, 1).reshape(R_, C).view(R_, C // 16, 16)
                        if it < a.refits:
                            num, den = (wb * q).sum(2, keepdim=True), (q * q).sum(2, keepdim=True)
                            s = torch.where(den > 0, num / den, s)
                    rec(f"Tessera-4 K={K} bitshift E2M1x2 L={L}, per-16 fp32 scale refit{a.refits}",
                        (q * s).view(R_, C), bpp + L / R_)
                    log(f"      ({time.time()-t0:.1f}s)")
            out["experts"][tname] = res
        del x_ev, xq4, xq8
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=int, nargs="+", default=[7, 6, 5])
    ap.add_argument("--memories", type=int, nargs="+", default=[10, 12, 14, 16])
    ap.add_argument("--expert-memories", type=int, nargs="+", default=[12, 14, 16])
    ap.add_argument("--kinds", nargs="+", default=["free", "e2m1x2"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.4, 1.7, 2.0, 2.3, 2.6, 3.0])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--gauss-cols", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--refits", type=int, default=2)
    ap.add_argument("--exl3", type=int, nargs="+", default=[4])
    ap.add_argument("--no-gaussian", action="store_true")
    ap.add_argument("--no-experts", action="store_true")
    ap.add_argument("--out", default="experiments/results/tessera_bitshift_tuple.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    log(f"\n##### tessera_bitshift_tuple {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(a)}")
    out = {"args": vars(a)}
    if not a.no_gaussian:
        log("\n### Gaussian oracle on E2M1x2: RMS / 2^-(K/2), each arm at its best global scale")
        gaussian_oracle(a, log, out)
        json.dump(out, open(a.out, "w"), indent=1)
    if not a.no_experts:
        log("\n### Six experts, per-16 fp32 block scale, held-out last 1024 rows")
        expert_arms(a, log, out)
        json.dump(out, open(a.out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
