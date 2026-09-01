"""Tessera-8's two hard constraints, computed before any encoder arm runs.

Rob's 8-bit mandate names two targets: the theoretical bound, and EXL3 with
8-bit activations.  Before the first real-weight arm there are two numbers
that bound what ANY E4M3-tile trellis can do, and both are cheap:

1. **The alphabet floor.**  A trellis constrains which E4M3 value a position
   may take; it never adds a representable value.  So no E4M3-tile format at
   any rate can put a weight closer to its target than the nearest E4M3 value
   under the tile's scale.  Per-channel FP8 RTN at 8 bpp IS that floor (with
   an LS-refit row scale it is the floor to the digit), and it is
   rate-independent.  Measured on the six GLM experts in weight space, output
   space (held-out tokens), and under the served per-token FP8 activation
   quantiser -- the W8A8 contract every stock FP8 GEMM takes.

2. **The mechanism bound.**  Memory-6 TCQ over four Ungerboeck subsets of an
   ideal 2^(R+1)-point codebook on i.i.d. Gaussian, at R = 4 and 5, over the
   Shannon bound 2^-R.  That ratio is what THIS encoder costs independent of
   alphabet and plane.  Memory 8 and 10 are run too: the earlier "memory is
   closed" verdict was measured on E2M1x2 at the cap, where every code is an
   anchor and a bigger trellis has no expanded codebook to exploit; on a fine
   alphabet at a sub-cap rate that reasoning does not transfer, and a larger
   conv code changes no bit layout.  The same codebook snapped to E4M3 values
   at the spread a per-channel scale gives it says what the alphabet costs on
   Gaussian; the real-weight arms (tessera8_targets.py) say the rest.

Everything here starts the Viterbi from state 0 (the decodable path; the
standalone free-start bias is on record) by going through the production
``viterbi_columns``.
"""
import argparse, json, math, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import PayloadGrid, build_forest
from tessera.encode import viterbi_columns
from tessera.trellis import ConvCode
from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
E4M3_MAX = 448.0
# Larsen's maximum-free-distance rate-1/2 codes for K = 11; memory 6 and 8
# come from the encoder's own table.  Wire-relevant only inside this
# experiment: no artifact is written with them.
GENERATORS = {10: (0o2335, 0o3661)}


def e4m3_rtn(x):
    """Round-to-nearest-even onto E4M3FN, saturating at +-448 (the cast does not)."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def ls_refit(w, s, rounds, dim):
    """Alternate RTN and the least-squares scale along ``dim`` (rows or blocks)."""
    for _ in range(rounds):
        q = e4m3_rtn(w / s)
        num = (w * q).sum(dim=dim, keepdim=True)
        den = (q * q).sum(dim=dim, keepdim=True)
        s = torch.where(den > 0, num / den, s)
        s = s.clamp_min(w.abs().amax(dim=dim, keepdim=True) / E4M3_MAX * 0.5)
    return s


def lloyd_max(z, k, iters=200):
    """Lloyd-Max levels for the sample ``z`` (1-D, sorted not required)."""
    lo, hi = z.min(), z.max()
    levels = torch.linspace(lo, hi, k, device=z.device)
    for _ in range(iters):
        cuts = (levels[:-1] + levels[1:]) / 2
        idx = torch.bucketize(z, cuts)
        sums = torch.zeros(k, device=z.device).index_add_(0, idx, z)
        cnt = torch.zeros(k, device=z.device).index_add_(0, idx, torch.ones_like(z))
        levels = torch.where(cnt > 0, sums / cnt.clamp_min(1), levels)
        levels, _ = levels.sort()
    return levels


def gaussian_sample(n, device):
    u = (torch.arange(n, device=device, dtype=torch.float64) + 0.5) / n
    return (math.sqrt(2.0) * torch.erfinv(2 * u - 1)).float()


def tcq(targets, values, R, memory, weights=None):
    """One trellis pass over ``targets`` [rows, cols] with a 2^(R+1)-value codebook."""
    grid = PayloadGrid(f"ideal{R}", tuple(float(v) for v in values.tolist()))
    assert grid.rate_cap == R, (grid.payload_bits, R)
    forest = build_forest(R, grid=grid)
    code = ConvCode(memory=memory, generators=GENERATORS.get(memory))
    anchors, _, sse = viterbi_columns(targets, forest, code, completion=0, weights=weights)
    hat = values.to(targets.device)[anchors]
    return hat, float(sse)


# ------------------------------------------------------------ part 1: floor
def alphabet_floor(a, log):
    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"tensors": [], "arms": {}}
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        n = xa.shape[0] // 2
        x_ev = xa[n:].contiguous()
        xq8 = fp8_dynamic_activation_qdq_vllm(x_ev).dequant.float()
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            act8 = float((xq8 @ w.T - y).norm() / ny)
            log(f"\n== L{layer} {proj} {tuple(w.shape)}  A8 leg (served per-token E4M3) {act8:.5f}")
            log(f"    {'arm':<40} {'bpp':>6} {'wt':>8} {'out':>8} {'W?A8':>8}")

            def rec(arm, hat, bpp):
                r = {"wt": float((hat - w).norm() / nw),
                     "out": float((x_ev @ hat.T - y).norm() / ny),
                     "both8": float((xq8 @ hat.T - y).norm() / ny),
                     "act8": act8, "bpp": bpp}
                out["arms"].setdefault(arm, []).append(r)
                log(f"    {arm:<40} {bpp:6.3f} {r['wt']:8.5f} {r['out']:8.5f} {r['both8']:8.5f}")

            R, C = w.shape
            s = w.abs().amax() / E4M3_MAX
            rec("FP8 RTN per-tensor amax", e4m3_rtn(w / s) * s, 8.0)
            s = w.abs().amax(dim=1, keepdim=True) / E4M3_MAX
            rec("FP8 RTN per-channel amax", e4m3_rtn(w / s) * s, 8.0 + 16 / C)
            s = ls_refit(w, s, 6, dim=1)
            rec("FP8 RTN per-channel LS-refit", e4m3_rtn(w / s) * s, 8.0 + 16 / C)
            for g in (16, 32, 128):
                blocks = w.reshape(R, C // g, g)
                s = blocks.abs().amax(dim=2, keepdim=True) / E4M3_MAX
                hat = (e4m3_rtn(blocks / s) * s).reshape(R, C)
                rec(f"FP8 RTN per-{g} amax (fp32 scale)", hat, 8.0 + 32 / g)
                s = ls_refit(blocks, s, 6, dim=2)
                hat = (e4m3_rtn(blocks / s) * s).reshape(R, C)
                rec(f"FP8 RTN per-{g} LS-refit (fp32 scale)", hat, 8.0 + 32 / g)
            out["tensors"].append(f"L{layer}.{proj}")
            del w, y
        del xa, x_ev, xq8
        torch.cuda.empty_cache()
    return out


# -------------------------------------------------------- part 2: mechanism
def mechanism_bound(a, log):
    torch.manual_seed(0)
    dev = "cuda"
    targets = torch.randn(a.rows, a.cols, device=dev)
    N = targets.numel()
    z = gaussian_sample(1 << 16, dev)
    out = {"rows": a.rows, "cols": a.cols, "arms": {}}

    def rec(R, arm, hat, extra=None):
        rms = float((hat - targets).norm() / math.sqrt(N))
        r = {"rms": rms, "bound": 2.0 ** -R, "ratio": rms * 2.0 ** R}
        if extra:
            r.update(extra)
        out["arms"].setdefault(str(R), {})[arm] = r
        log(f"    R={R} {arm:<48} rms {rms:.5f}  x{r['ratio']:.3f} of 2^-{R}")
        return r

    log("\n== mechanism bound: i.i.d. N(0,1), one global scale, no plane")
    for R in a.rates:
        # scalar references
        lv = lloyd_max(z, 1 << R)
        cuts = (lv[:-1] + lv[1:]) / 2
        rec(R, f"Lloyd-Max scalar 2^{R} levels (RTN)", lv[torch.bucketize(targets, cuts)])
        values = lloyd_max(z, 1 << (R + 1))
        for memory in a.memories:
            t0 = time.time()
            hat, sse = tcq(targets, values, R, memory)
            r = rec(R, f"TCQ m={memory}, Lloyd 2^{R+1} codebook", hat,
                    {"seconds": time.time() - t0})
            # Codebook refinement: the TCQ analogue of Lloyd's step, each
            # anchor moved to the conditional mean of what it reconstructed.
            vals = values.clone()
            best = r["rms"]
            for it in range(a.refine):
                grid = PayloadGrid("tmp", tuple(float(v) for v in vals.tolist()))
                forest = build_forest(R, grid=grid)
                code = ConvCode(memory=memory, generators=GENERATORS.get(memory))
                anchors, _, _ = viterbi_columns(targets, forest, code, completion=0)
                flat = anchors.reshape(-1)
                sums = torch.zeros(vals.numel(), device=dev).index_add_(0, flat, targets.reshape(-1))
                cnt = torch.zeros(vals.numel(), device=dev).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
                vals = torch.where(cnt > 0, sums / cnt.clamp_min(1), vals).sort().values
                hat, _ = tcq(targets, vals, R, memory)
                rr = float((hat - targets).norm() / math.sqrt(N))
                best = min(best, rr)
            rec(R, f"TCQ m={memory}, codebook refined x{a.refine}", hat,
                {"best_rms": best, "best_ratio": best * 2.0 ** R})
        # The same codebook snapped to E4M3 values at the spread a per-channel
        # amax scale gives a 4096-wide Gaussian row (amax ~ 3.9 sigma -> 448).
        sigma_units = E4M3_MAX / a.row_amax_sigmas
        snapped = e4m3_rtn(values * sigma_units) / sigma_units
        dup = int(values.numel() - snapped.unique().numel())
        hat, _ = tcq(targets, snapped, R, a.memories[0])
        rec(R, f"TCQ m={a.memories[0]}, Lloyd codebook snapped to E4M3", hat,
            {"duplicate_values": dup, "sigma_in_e4m3_units": sigma_units})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--rates", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--memories", type=int, nargs="+", default=[6, 8, 10])
    ap.add_argument("--refine", type=int, default=3)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--row-amax-sigmas", type=float, default=3.9)
    ap.add_argument("--parts", default="floor,mechanism")
    ap.add_argument("--out", default="experiments/results/tessera8_bounds.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"\n##### tessera8_bounds {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(a)}")
    out = {"args": vars(a)}
    if "mechanism" in a.parts:
        out["mechanism"] = mechanism_bound(a, log)
        json.dump(out, open(a.out, "w"), indent=1)
    if "floor" in a.parts:
        out["floor"] = alphabet_floor(a, log)
        json.dump(out, open(a.out, "w"), indent=1)
        log("\n== alphabet floor, mean over tensors")
        log(f"    {'arm':<40} {'bpp':>6} {'wt':>8} {'out':>8} {'W?A8':>8}")
        for arm, rs in out["floor"]["arms"].items():
            m = lambda k: sum(r[k] for r in rs) / len(rs)
            log(f"    {arm:<40} {m('bpp'):6.3f} {m('wt'):8.5f} {m('out'):8.5f} {m('both8'):8.5f}")
        log(f"    {'A8 leg alone (served per-token E4M3)':<40} {'':>6} {'':>8} {'':>8} "
            f"{sum(r['act8'] for r in rs) / len(rs):8.5f}")
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
