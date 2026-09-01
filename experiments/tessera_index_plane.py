"""Index plane: can the scale plane's redundancy be cashed WITHOUT losing per-16 loading?

The limits doc (docs/measurements/tessera-theoretical-limits-2026-09-01.md) found the
per-16 E4M3 plane carries 0.215 bpp of information in 0.5 bpp of bytes, and that the
one arm that tried to cash the difference (per-32 E4M3 + Wei L=2 at 4.0 bpp, 1.047x)
paid for its bits by halving the loading granularity (per-16 vs per-32 at L=2 differ
by 1.078x).  This arm keeps per-16 loading and cashes the redundancy instead: a
per-tensor codebook of k E4M3 values, a log2(k)-bit index per 16 weights.

  bpp = payload(L) + log2(k)/16      k=8: 0.1875   k=16: 0.25   k=32: 0.3125

Decode is still E2M1 codes x a per-16 E4M3 scale: the kernel lane reads the scale
through a k-entry LUT, the stock lane materialises the same E4M3 tile.  FP4-native
either way.  The codebook is weighted k-means on the LS-optimal per-half scale s*
with weight A = <q,q> (exact: the per-half SSE is A (s - s*)^2 + const), centroids
snapped to E4M3, quantisation = nearest centroid in the linear domain (the SSE
minimiser).  Alternating with the trellis like the refit schedule.

Reference arms recomputed in-script: flat E4M3 per-16 (0.5 bpp) and per-32 (0.25)
at L=1 and L=2, so every matched-bpp comparison is on the same eval rows.
"""
import argparse, json, math, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera.trellis import TCQ  # noqa: E402
from tessera_memory_and_codebook import labels  # noqa: E402
from tessera_rate_plane_frontier import ls_shared  # noqa: E402
from prismaquant.nvfp4_activation_contract import (  # noqa: E402
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

HALF = F.HALF
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"


def ls_parts(w, q):
    """Per-half B = <w,q> and A = <q,q>; s* = B/A minimises A s^2 - 2 B s."""
    R, C = w.shape
    B = (w * q).reshape(R, C // HALF, HALF).sum(-1).reshape(-1)
    A = (q * q).reshape(R, C // HALF, HALF).sum(-1).reshape(-1)
    return B, A


def weighted_kmeans(s, A, k, G, iters=40):
    """Lloyd on 1-D targets s with weights A, init at weighted log-quantiles, centroids
    snapped to E4M3 behind the global G at the end of every iteration (so the
    assignment always sees representable values)."""
    ok = A > 0
    s, A = s[ok], A[ok]
    order = torch.argsort(s)
    cw = torch.cumsum(A[order], 0) / A.sum()
    qs = (torch.arange(k, device=s.device, dtype=s.dtype) + 0.5) / k
    idx = torch.searchsorted(cw, qs).clamp(max=s.numel() - 1)
    c = F.e4m3(s[order][idx], G)
    for _ in range(iters):
        assign = torch.argmin((s[:, None] - c[None, :]).abs(), dim=1)
        num = torch.zeros(k, device=s.device, dtype=s.dtype).index_add_(0, assign, A * s)
        den = torch.zeros(k, device=s.device, dtype=s.dtype).index_add_(0, assign, A)
        new = torch.where(den > 0, num / den.clamp_min(1e-30), c)
        new = F.e4m3(new, G)
        if torch.equal(new, c):
            break
        c = new
    return torch.sort(c).values


def e4m3_grid(G, lo, hi, device):
    """Every positive finite E4M3FN value (behind G) in [lo, hi], one step wider each side."""
    v = torch.arange(1, 127, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn).float()
    v = torch.sort(v[torch.isfinite(v) & (v > 0)]).values / G
    i0 = int((v < lo).sum().clamp(min=1)) - 1
    i1 = int((v <= hi).sum().clamp(max=v.numel() - 1)) + 1
    return v[i0:i1]


def _cost(s, A, c):
    d = (s[:, None] - c[None, :]).abs().amin(1)
    return float((A * d * d).sum())


def e4m3_subset(s, A, k, G, swaps=2):
    """Choose k DISTINCT E4M3 values minimising sum_i A_i (s_i - nearest)^2: greedy
    backward elimination from the full in-range E4M3 grid, then pairwise swap passes.
    Exact per-assignment cost, no continuous centroids to collapse under snapping."""
    ok = A > 0
    s, A = s[ok], A[ok]
    cand = e4m3_grid(G, float(s.min()), float(s.max()), s.device)
    cur = cand.clone()
    while cur.numel() > k:
        best, best_i = None, -1
        for i in range(cur.numel()):
            trial = torch.cat([cur[:i], cur[i + 1:]])
            cst = _cost(s, A, trial)
            if best is None or cst < best:
                best, best_i = cst, i
        cur = torch.cat([cur[:best_i], cur[best_i + 1:]])
    for _ in range(swaps):
        improved = False
        base = _cost(s, A, cur)
        unused = cand[~torch.isin(cand, cur)]
        for i in range(cur.numel()):
            for u in unused:
                trial = cur.clone(); trial[i] = u
                cst = _cost(s, A, trial)
                if cst < base - 1e-12:
                    cur, base, improved = trial, cst, True
        if not improved:
            break
    return torch.sort(cur).values


def quantise_to(s, c, prev, A):
    assign = torch.argmin((s[:, None] - c[None, :]).abs(), dim=1)
    out = c[assign]
    return torch.where((A > 0) & (s > 0), out, prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--codebook", choices=["kmeans", "subset"], default="subset")
    ap.add_argument("--out", default="experiments/results/tessera_index_plane.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    F.PROD = torch.tensor([F.GRID.vector(c) for c in range(F.GRID.size)], dtype=torch.float32, device="cuda")
    F.COSET = labels(TCQ(F.FOREST, F.CC).subsets, F.FOREST.anchors, F.GRID.size, "cuda")
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"tensors": [], "act": [], "arms": {}, "codebooks": {}, "exl3_k4": F.EXL3_K4, "args": vars(a)}

    def enc(L):
        return (lambda w_, s_: F.trellis(w_, s_)) if L == 1 else (lambda w_, s_: F.multidim(w_, s_, L))

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
            log(f"    {'arm':<36} {'bpp':>6} {'wt':>7} {'out':>7} {'W4A4':>7}")
            t0 = time.time()

            def rec(nm, hat, bpp):
                r_ = {"wt": float((hat - w).norm() / nw), "out": float((x_ev @ hat.T - y).norm() / ny),
                      "both_served": float((xq_s @ hat.T - y).norm() / ny), "act": act, "bpp": bpp}
                out["arms"].setdefault(nm, []).append(r_)
                log(f"    {nm:<36} {bpp:>6.4f} {r_['wt']:.5f} {r_['out']:.5f} {r_['both_served']:.5f}"
                    f"  {time.time() - t0:5.1f}s")

            G = F.E4M3_MAX / F.amax_half(w).max()
            amax = F.amax_half(w)
            for L in a.levels:
                payload = 4 - 1 / (2 * L)
                # references: flat E4M3 shared by 1 and 2 halves, LS refit
                planes = {}
                for k in (1, 2):
                    eff = F.e4m3(amax.reshape(-1, k).amax(1).repeat_interleave(k), G)
                    q = enc(L)(w, F.expand(eff, R, C))
                    for _ in range(a.iters):
                        eff = F.e4m3(ls_shared(w, q, eff, k), G)
                        q = enc(L)(w, F.expand(eff, R, C))
                    rec(f"E4M3 per-{16 * k} + LS  L={L}", q * F.expand(eff, R, C), payload + 0.5 / k)
                    planes[k] = (eff, q)
                # index plane: start from the per-16 refit plane's codes
                eff16, q16 = planes[1]
                for k in a.ks:
                    B, A = ls_parts(w, q16)
                    s_star = B / A.clamp_min(1e-30)
                    fit = weighted_kmeans if a.codebook == "kmeans" else e4m3_subset
                    c = fit(s_star, A, k, G)
                    eff = quantise_to(s_star, c, eff16, A)
                    q = enc(L)(w, F.expand(eff, R, C))
                    for _ in range(a.iters):
                        B, A = ls_parts(w, q)
                        s_star = B / A.clamp_min(1e-30)
                        c = fit(s_star, A, k, G)
                        eff = quantise_to(s_star, c, eff, A)
                        q = enc(L)(w, F.expand(eff, R, C))
                    nm = f"idx{k} per-16 + LS  L={L}"
                    rec(nm, q * F.expand(eff, R, C), payload + math.log2(k) / 16)
                    out["codebooks"].setdefault(nm, []).append((c * G).tolist())
                    log(f"      codebook x{G:.3g}: {[round(float(v), 1) for v in c * G]}")
            json.dump(out, open(a.out, "w"), indent=1)
            del w; torch.cuda.empty_cache()
        del xa, x_fit, x_ev, xq_s; torch.cuda.empty_cache()

    mean = lambda v: sum(v) / len(v)
    base = out["arms"]["E4M3 per-16 + LS  L=1"]
    log("\n| arm | bpp | out-space weight leg | vs E4M3/16 L=1 (4.0) | W4A4 | vs EXL3@A4 |")
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
