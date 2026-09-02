"""A large-state bitshift trellis on the E4M3 tile: states and cosets, not dimension.

Question (2026-09-02): the remaining gap between Tessera's 64-state, 4-subset
trellis and EXL3 is mostly boundary/shaping loss, which QTIP-style trellises
recover with a huge state count and a state-dependent Gaussian codebook. On a
hardware tile the codebook can only be tile values. How far does a bitshift
trellis whose window map lands on E4M3 close toward EXL3?

The bitshift trellis (QTIP): the state is the last ``L`` bits of the
bitstream, ``K`` new bits shift in per position, and the reconstruction is
``book[state]``. Predecessors of ``s'`` are every ``s`` whose low ``L-K``
bits equal ``s' >> K``, so the Viterbi's minimum over predecessors is a
reshape-and-min, O(2^L) per position with no gather. Decode is a window
lookup, which is what the Tessera kernel lane already does.

Arms, Gaussian oracle (2048x512 i.i.d. N(0,1), RMS over 2^-R):
  * Tessera today: 4 subsets, LM+midpoint anchors, Ungerboeck memory 6.
  * bitshift, free Gaussian window map (QTIP/EXL3-like, not a tile): L in
    {8..16}.  EXL3 K4/K5 measured 1.068/1.085 on this protocol.
  * bitshift, the same map snapped to E4M3 at sigma = 64 units: THE arm.
  * bitshift over the 2^(R+1)-value LM+midpoint alphabet assigned by hash:
    is it the alphabet size or the state count that pays?
  * bitshift with a sorted (non-random) map: the control that says the
    randomness matters.

Arms, six GLM experts (per-channel protocol of tessera8_targets.py: row
scale rms/64, E4M3 window map, LS row-scale refit x2, s^2 position weights),
scored on the last 1024 capture rows with the served A4/A8 legs, against the
Ungerboeck per-channel arm and EXL3 K4/K5 on the same rows.

Screen, principle 3. No wire is written; the window map is a per-unit table
(2^L bytes) that a schema minor would carry.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import PayloadGrid, build_forest
from tessera.encode import viterbi_columns
from tessera.trellis import ConvCode
from tessera8_bounds import gaussian_sample, lloyd_max
from tessera8_targets import (ACT, E4M3_ALL, E4M3_MAX, EXL3, SRC, UNG6, midpoint_codebook,
                              snap_unique)

E4M3_SORTED = E4M3_ALL.clone()                              # 255 finite values, sorted


def snap_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Nearest E4M3 value (collisions allowed; a window map need not be injective)."""
    grid = E4M3_SORTED.to(x.device)
    idx = torch.bucketize(x, grid).clamp(1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    return torch.where((x - lo).abs() <= (hi - x).abs(), lo, hi)


def window_map(L: int, kind: str, R: int, sigma_units: float, seed: int, device) -> torch.Tensor:
    """The per-state reconstruction table, in E4M3 units (sigma = ``sigma_units``)."""
    S = 1 << L
    g = torch.Generator(device="cpu").manual_seed(seed)
    quant = gaussian_sample(S, device) * sigma_units          # sorted Gaussian quantiles
    if kind == "sorted":
        return snap_e4m3(quant)
    perm = torch.randperm(S, generator=g).to(device)
    if kind == "free":                                           # not a tile: reference only
        return quant[perm]
    if kind == "e4m3":
        return snap_e4m3(quant[perm])
    if kind == "lm-mid":                                         # 2^(R+1)-value alphabet by hash
        book = snap_unique(midpoint_codebook(gaussian_sample(1 << 16, device), R) * sigma_units)
        idx = torch.randint(0, len(book), (S,), generator=g).to(device)
        return book[idx]
    if kind == "computed":
        # QTIP's 1MAD idea, table-free: one 32-bit multiply-add of the state, the
        # four bytes of the product summed (CLT Gaussian, sigma ~147.2), scaled to
        # sigma_units and converted to E4M3.  A kernel computes this per window
        # with a MAD, three byte extracts, an FMA and a cvt.e4m3 -- no table.
        s = torch.arange(S, dtype=torch.int64)
        v = ((s + 0x1234567) * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF
        b = sum(((v >> (8 * i)) & 0xFF) for i in range(4)).double() - 510.0
        sigma_bytes = math.sqrt(4 * (255.0 ** 2 - 1) / 12.0)
        return snap_e4m3((b / sigma_bytes * sigma_units).float().to(device))
    raise ValueError(kind)


@torch.no_grad()
def bitshift_viterbi(targets: torch.Tensor, book: torch.Tensor, L: int, K: int,
                     weights=None, chunk: int = 512, pinned: bool = False):
    """Exact Viterbi down every column, batched over columns. Free initial state
    (the first L bits are transmitted; L/rows bits per column, ignored here).

    ``targets`` [rows, cols] in book units; ``weights`` [rows] or [rows, cols]
    or None. Returns (states [rows, cols] int32, sse float)."""
    arity = 1 if targets.dim() == 2 else targets.shape[2]        # [steps, cols] or [steps, cols, k]
    rows, cols = targets.shape[0], targets.shape[1]
    S, P = 1 << L, 1 << K
    low = S >> K                                                   # 2^(L-K) predecessor classes
    dev = targets.device
    states = torch.empty(rows, cols, dtype=torch.int32, device=dev)
    sse = 0.0
    bookf = book.float().view(S, arity)
    for c0 in range(0, cols, chunk):
        x = targets[:, c0:c0 + chunk].float().view(rows, -1, arity)
        n = x.shape[1]
        if weights is None:
            wt = None
        elif weights.dim() == 1:
            wt = weights.view(rows, 1).float()
        else:
            wt = weights[:, c0:c0 + chunk].float()
        if pinned:                                             # the wire's start: state 0, free-start bias gone
            cost = torch.full((S, n), float("inf"), device=dev)
            cost[0] = 0.0
        else:
            cost = torch.zeros(S, n, device=dev)
        back = torch.empty(rows, low, n, dtype=torch.uint8, device=dev)
        for t in range(rows):
            m, p = cost.view(P, low, n).min(dim=0)               # best predecessor per low class
            back[t] = p.to(torch.uint8)
            d = (x[t].view(1, n, arity) - bookf.view(S, 1, arity)).pow(2).sum(-1)
            if wt is not None:
                d = d * wt[t].view(-1, n) if wt.shape[1] == n else d * wt[t]
            cost = m.repeat_interleave(P, dim=0) + d
        best, s = cost.min(dim=0)                                  # [n]
        sse += float(best.sum())
        st = torch.empty(rows, n, dtype=torch.int32, device=dev)
        for t in range(rows - 1, -1, -1):
            st[t] = s
            p = back[t].gather(0, (s >> K).view(1, n).long()).view(n).long()
            s = (s >> K) | (p << (L - K))
        states[:, c0:c0 + chunk] = st
    return states, sse


def gaussian_oracle(a, log, out):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(a.rows, a.gauss_cols, device=dev)
    z = gaussian_sample(1 << 16, dev)
    rows = {}
    for R in a.rates:
        ref = 2.0 ** -R
        # Tessera today: 4 subsets, LM+mid anchors, Ungerboeck m6, one global scale
        book = midpoint_codebook(z, R)
        grid = PayloadGrid(f"lm{R}", tuple(float(v) for v in book.tolist()))
        forest = build_forest(R, grid=grid)
        t0 = time.time()
        anchors, _, _ = viterbi_columns(x, forest, UNG6, completion=0)
        hat = book[anchors]
        r = float((hat - x).pow(2).mean().sqrt() / ref)
        rows[f"R={R} tessera-4subset-ung6-lmmid"] = {"ratio": r, "s": time.time() - t0}
        log(f"  R={R} {'Tessera 4-subset Ungerboeck m6, LM+mid (free values)':58s} {r:.4f}x  {time.time()-t0:5.1f}s")
        # E4M3-snapped version of the same, at sigma units, for the tile baseline
        booke = snap_unique(book * a.sigma_units)
        gride = PayloadGrid(f"lme{R}", tuple(float(v) for v in booke.tolist()))
        foreste = build_forest(R, grid=gride)
        t0 = time.time()
        anchors, _, _ = viterbi_columns(x * a.sigma_units, foreste, UNG6, completion=0)
        hat = booke[anchors] / a.sigma_units
        r = float((hat - x).pow(2).mean().sqrt() / ref)
        rows[f"R={R} tessera-4subset-ung6-lmmid-e4m3"] = {"ratio": r, "s": time.time() - t0}
        log(f"  R={R} {'Tessera 4-subset Ungerboeck m6, LM+mid snapped E4M3':58s} {r:.4f}x  {time.time()-t0:5.1f}s")
        for kind in a.kinds:
            for L in a.memories:
                if L < R + 1:
                    continue
                book = window_map(L, kind, R, a.sigma_units, a.seed, dev)
                t0 = time.time()
                st, _ = bitshift_viterbi(x * a.sigma_units, book, L, R, chunk=a.chunk)
                hat = book[st.long()] / a.sigma_units
                r = float((hat - x).pow(2).mean().sqrt() / ref)
                key = f"R={R} bitshift-{kind}-L{L}"
                rows[key] = {"ratio": r, "s": time.time() - t0, "L": L, "kind": kind}
                log(f"  R={R} {f'bitshift {kind:7s} L={L:2d} K={R}':58s} {r:.4f}x  {time.time()-t0:5.1f}s")
    out["gaussian"] = rows


def expert_arms(a, log, out):
    from safetensors import safe_open
    from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm
    from prismaquant.nvfp4_activation_contract import (
        nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    out["experts"] = {}
    dev = "cuda"
    z = gaussian_sample(1 << 16, dev)
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

            for K in a.exl3:
                p = Path(EXL3) / f"L{layer}_{proj}_K{K}.pt"
                if p.exists():
                    rec(f"EXL3 K={K}", torch.load(p, map_location=dev).float(), K + 0.011723)
            rows_, cols_ = w.shape
            rms = w.pow(2).mean(dim=1, keepdim=True).sqrt()
            for R in a.rates:
                # reference: the Ungerboeck per-channel arm (as tessera8_targets)
                book = snap_unique(midpoint_codebook(z, R) * a.sigma_units)
                grid = PayloadGrid(f"pc{R}", tuple(float(v) for v in book.tolist()))
                forest = build_forest(R, grid=grid)
                s = rms / a.sigma_units
                for it in range(a.refits + 1):
                    wts = (s / s.max()).pow(2).expand(rows_, cols_).contiguous()
                    anchors, _, _ = viterbi_columns((w / s).contiguous(), forest, UNG6, completion=0, weights=wts)
                    q = book[anchors]
                    if it < a.refits:
                        num, den = (w * q).sum(1, keepdim=True), (q * q).sum(1, keepdim=True)
                        s = torch.where(den > 0, num / den, s)
                rec(f"Tessera-8 R={R} per-channel 4-subset Ung6 (today)", q * s, R + 32 / cols_)
                for L in a.expert_memories:
                    if L < R + 1:
                        continue
                    book = window_map(L, "e4m3", R, a.sigma_units, a.seed, dev)
                    s = rms / a.sigma_units
                    t0 = time.time()
                    for it in range(a.refits + 1):
                        wts = (s / s.max()).pow(2).view(-1)
                        st, _ = bitshift_viterbi((w / s).contiguous(), book, L, R, weights=wts, chunk=a.chunk,
                                                 pinned=a.pinned_start)
                        q = book[st.long()]
                        if it < a.refits:
                            num, den = (w * q).sum(1, keepdim=True), (q * q).sum(1, keepdim=True)
                            s = torch.where(den > 0, num / den, s)
                    start_bits = 0.0 if a.pinned_start else L / rows_
                    tag = "pinned" if a.pinned_start else "free-start"
                    rec(f"Tessera-8 R={R} per-channel bitshift E4M3 L={L} ({tag})", q * s,
                        R + 32 / cols_ + start_bits + (1 << L) * 8 / (rows_ * cols_))
                    log(f"      ({time.time()-t0:.1f}s)")
            out["experts"][tname] = res
        del x_ev, xq4, xq8
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--memories", type=int, nargs="+", default=[8, 10, 12, 14, 16])
    ap.add_argument("--expert-memories", type=int, nargs="+", default=[10, 12, 14])
    ap.add_argument("--kinds", nargs="+", default=["free", "e4m3", "lm-mid", "sorted"])
    ap.add_argument("--sigma-units", type=float, default=64.0)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--gauss-cols", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--refits", type=int, default=2)
    ap.add_argument("--exl3", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--no-gaussian", action="store_true")
    ap.add_argument("--pinned-start", action="store_true",
                    help="start the trellis at state 0 as the wire does (default: free start, ~0.3%% optimistic)")
    ap.add_argument("--no-experts", action="store_true")
    ap.add_argument("--out", default="experiments/results/tessera_bitshift_tile.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    log(f"\n##### tessera_bitshift_tile {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(a)}")
    out = {"args": vars(a)}
    if not a.no_gaussian:
        log("\n### Gaussian oracle: RMS / 2^-R (EXL3 K4 1.068, K5 1.085 on this protocol)")
        gaussian_oracle(a, log, out)
        json.dump(out, open(a.out, "w"), indent=1)
    if not a.no_experts:
        log("\n### Six experts, per-channel protocol, held-out last 1024 rows")
        expert_arms(a, log, out)
        json.dump(out, open(a.out, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
