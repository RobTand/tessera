"""FP4-native levers: what moves Tessera-4's 4.0 bpp point without leaving the NVFP4 tile.

The constraint this battery is built around.  Blackwell's FP4 tensor-core MMA
consumes E2M1 bit patterns times one E4M3 scale per 16 consecutive K elements.
That is what "natively use NVIDIA's 4-bit hardware" means at the instruction
level, and it decides the lever set.  A learned codebook (tree or flat Lloyd)
puts arbitrary floats in the tile and is out; the trellis, the E2M1 x E2M1
product grid, and every choice of *scale value* are in.  So this measures the
levers that stay inside the tile:

  P  the scale plane's rule and format, at the same 0.5 bpp or less
       - today's S6b two-tier (E8M0 per 32 + 4-bit refine per 16), headroom swept
       - a flat per-16 E4M3 plane (NVFP4's own scale tensor: same bits, no octave lock)
       - E8M0-only per 32 (0.25 bpp; rides the NVFP4 route with each scale duplicated)
  R  re-fitting the scale VALUES after the trellis has spoken: least squares per
     half, plain and weighted by the 16x16 diagonal block of the input Hessian,
     re-quantised into the plane's own format and iterated with the trellis
  M  Wei's multidimensional partition at L=2: payload 3.75 for half the redundancy
  F  error feedback the calibration can actually support -- LDLQ inside each
     32-column scale group (a 32x32 Hessian block is full rank at 128 tokens) --
     and the whole-Hessian LDLQ that produced EXL3's number, for the record

Every arm is scored on the plane the artifact ships: scale groups of 32 along K
with the S6b bytes.  No prior experiment here did that; they all grouped 32 rows
down a column with an fp32 amax scale (self-consistent, not the format).  The
baseline is pinned to `encode_unit` + `reconstruct_unit` by an equality check
on the first tensor, so "the format" is measured rather than asserted.

Scored on the held-out 128 tokens of the cached GLM activations: weight-space
relative error (deterministic; the tie-breaker), the output-space weight leg,
and the W4A4 composite under the SERVED activation quantiser
(`nvfp4_activation_qdq_served` + MSE global scale, the deployed contract) and
under plain per-16 amax (what the earlier projection used).  EXL3's composite is
projected from its weight leg by the quadrature rule the projection experiment
verified.  256 tokens per layer cap what any feedback arm can show; that cap is
the calibration's, not the format's.
"""
import argparse, json, re, sys, time
from fractions import Fraction
from pathlib import Path

import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.compensate import block_ldl, compensated_targets, regularize_hessian
from tessera.decode import reconstruct_unit
from tessera.encode import _pack_scales, encode_unit
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import RotationState
from tessera.trellis import TCQ, ConvCode
from tessera_free_codebook_trellis import viterbi
from tessera_memory_and_codebook import labels
from tessera_multidim_partition import encode_multidim
from tessera_w4a4_projection import quant_a4
from prismaquant.nvfp4_activation_contract import (
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
CC = ConvCode(memory=6)
EXL3_K4 = 0.05653
PEAK, GROUP, HALF = 6.0, 32, 16
E4M3_MAX = 448.0
PLANE_BPP = {"s6b": 0.5, "e4m3": 0.5, "e8m0": 0.25}

GRID = tuple_grid(E2M1_GRID, 2)
FOREST = build_forest(GRID.rate_cap, grid=GRID)
PROD = COSET = None                      # device tables, built in main


# ---------------------------------------------------------------- planes
# Every plane function returns the per-half effective scale, flat, in the order
# `_pack_scales` uses: row-major, so a half is 16 consecutive K of one row.
def expand(eff, R, C):
    return torch.repeat_interleave(eff, HALF).reshape(R, C)


def s6b(w, headroom=1.0):
    """Today's plane, from the encoder's own packer."""
    return _pack_scales(w, GROUP, HALF, peak=PEAK, headroom=headroom)[2].float()


def amax_half(w, headroom=1.0):
    return w.reshape(-1, HALF).abs().amax(1).clamp_min(1e-30) / (PEAK * headroom)


def e4m3(s, G=None):
    """Nearest E4M3FN per half behind one fp32 global that puts the largest scale at 448."""
    G = E4M3_MAX / s.max() if G is None else G
    return (s * G).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).float() / G


def e8m0_amax(w, t):
    """One power of two per 32: floor(log2(amax/6)), bumped an octave when the
    leftover fraction exceeds t.  t=1 is ceil (never clips), t=2 is floor,
    sqrt(2) rounds in the log domain."""
    r = w.reshape(-1, GROUP).abs().amax(1).clamp_min(1e-30) / PEAK
    E = torch.floor(torch.log2(r))
    E = E + (r / 2.0 ** E > t).float()
    return (2.0 ** E).repeat_interleave(GROUP // HALF)


def e8m0_nearest(s):
    """Re-quantise a desired per-half scale to one power of two per 32."""
    E = torch.round(torch.log2(s.reshape(-1, GROUP // HALF)).mean(1))
    return (2.0 ** E).repeat_interleave(GROUP // HALF)


def two_tier(s):
    """Re-quantise a desired per-half scale into S6b: a shared E8M0 base per 32
    and a (d, m) refinement per half.  Three base candidates, picked by relative
    squared error; a half whose ratio to the base leaves [1, 4) is clamped --
    that clamp is the octave lock S6b imposes and the flat E4M3 plane does not."""
    g = s.reshape(-1, GROUP // HALF)
    lo = torch.floor(torch.log2(g.min(1).values))
    hi = torch.floor(torch.log2(g.max(1).values)) - 1
    best_eff = best_cost = None
    for E in (lo, lo + 1, hi):
        r = (g / 2.0 ** E[:, None]).clamp(1.0, 4.0 - 1e-6)
        d = (r >= 2).float()
        m = torch.round((r / 2.0 ** d - 1) * 8)
        carry = (m >= 8) & (d == 0)
        m = torch.where(carry, torch.zeros_like(m), m)
        d = torch.where(carry, torch.ones_like(d), d)
        m = m.clamp(max=7)
        eff = 2.0 ** (E[:, None] + d) * (1 + m / 8)
        cost = (((eff - g) / g) ** 2).sum(1)
        if best_eff is None:
            best_eff, best_cost = eff, cost
        else:
            take = cost < best_cost
            best_eff = torch.where(take[:, None], eff, best_eff)
            best_cost = torch.where(take, cost, best_cost)
    return best_eff.reshape(-1)


def octaves(eff):
    return float(torch.log2(eff.max() / eff.min()))


# --------------------------------------------------------------- encoders
def trellis(w, scale):
    """Rung 7 of E2M1x2 over K-grouped targets: the shipping encoder's body."""
    R, C = w.shape
    seq = (w / scale).reshape(R // 2, 2, C).permute(0, 2, 1).contiguous()
    return viterbi(seq, PROD, COSET, CC).permute(0, 2, 1).reshape(R, C)


def multidim(w, scale, L):
    R, C = w.shape
    seq = (w / scale).reshape(R // 2, 2, C).permute(0, 2, 1).contiguous()
    return encode_multidim(seq, PROD, COSET, CC, L).permute(0, 2, 1).reshape(R, C)


def clip_fraction(w, scale):
    return float(((w / scale).abs() > PEAK).float().mean())


def ls_scale(w, q, prev, Hb=None):
    """Per-half least-squares scale given the trellis's units q: <w,q>_H / <q,q>_H.
    Hb is the (C/16, 16, 16) diagonal-block Hessian or None for plain LS.
    A half with no usable fit keeps its previous scale."""
    R, C = w.shape
    W, Q = w.reshape(R, C // HALF, HALF), q.reshape(R, C // HALF, HALF)
    if Hb is None:
        num, den = (W * Q).sum(-1), (Q * Q).sum(-1)
    else:
        num = torch.einsum("rbi,bij,rbj->rb", Q, Hb, W)
        den = torch.einsum("rbi,bij,rbj->rb", Q, Hb, Q)
    s = (num / den.clamp_min(1e-30)).reshape(-1)
    return torch.where((den.reshape(-1) > 0) & (s > 0), s, prev)


def diag_blocks(H, size):
    idx = torch.arange(H.shape[0], device=H.device).reshape(-1, size)
    return H[idx[:, :, None], idx[:, None, :]].contiguous()


def group_feedback(w, scale, Hb32):
    """GPTQ/LDLQ with the Hessian restricted to each 32-column scale group, every
    group in parallel: quantise column j of all groups in one Viterbi batch, push
    its residual onto columns j+1.. of its own group.  The scale is the plane set
    on the uncompensated group, so feedback can push a target past 6*s; that
    fraction is returned alongside the units."""
    R, C = w.shape
    nb = C // GROUP
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(Hb32)), upper=True)
    Wb = w.reshape(R, nb, GROUP).clone()
    Sb = scale.reshape(R, nb, GROUP)
    Qb = torch.empty_like(Wb)
    clipped = 0
    for j in range(GROUP):
        col, sc = Wb[:, :, j], Sb[:, :, j]
        t = col / sc
        clipped += int((t.abs() > PEAK).sum())
        seq = t.reshape(R // 2, 2, nb).permute(0, 2, 1).contiguous()
        q = viterbi(seq, PROD, COSET, CC).permute(0, 2, 1).reshape(R, nb)
        Qb[:, :, j] = q
        err = (col - q * sc) / Hinv[:, j, j][None, :]
        if j + 1 < GROUP:
            Wb[:, :, j + 1:] -= err[:, :, None] * Hinv[:, j, j + 1:][None, :, :]
    return Qb.reshape(R, C), clipped / w.numel()


def full_ldlq(w, Hreg, plane):
    """The EXL3 procedure on Tessera: block-LDL of the whole regularised Hessian,
    32-column blocks, via the repo's own scheduler.  `plane(slice) -> eff`."""
    L = block_ldl(Hreg, GROUP)
    clips = []

    def encode(target, start, stop):
        sc = expand(plane(target), *target.shape)
        clips.append(clip_fraction(target, sc))
        return trellis(target, sc) * sc

    _, recon = compensated_targets(w, L, encode, block=GROUP)
    return recon, sum(clips) / len(clips)


# ---------------------------------------------------------------- scoring
class Scorer:
    def __init__(self, w, x_ev, xq_served, xq_amax):
        self.w, self.x = w, x_ev
        self.y = x_ev @ w.T
        self.ny, self.nw = self.y.norm(), w.norm()
        self.xq_s, self.xq_a = xq_served, xq_amax

    def __call__(self, hat, **extra):
        r = {"wt": float((hat - self.w).norm() / self.nw),
             "out": float((self.x @ hat.T - self.y).norm() / self.ny),
             "both_served": float((self.xq_s @ hat.T - self.y).norm() / self.ny),
             "both_amax": float((self.xq_a @ hat.T - self.y).norm() / self.ny)}
        r.update(extra)
        return r


def verify_against_encoder(w):
    """The harness's baseline must be the artifact's own encode -> decode."""
    rates = bresenham_rate_schedule(Fraction(GRID.rate_cap), w.shape[1], cap=GRID.rate_cap)
    forests = {r: build_forest(r, grid=GRID) for r in sorted(set(rates))}
    unit = encode_unit(w, forests, rates, CC, rotation=RotationState.NONE,
                       with_diagonals=False, completion=0, group=GROUP, half=HALF)
    ref = reconstruct_unit(unit, forests, CC)
    sc = expand(s6b(w), *w.shape)
    mine = trellis(w, sc) * sc
    diff = (ref - mine).abs()
    nw = w.norm()
    return {"max_abs_diff": float(diff.max()), "differing": int((diff > 0).sum()),
            "numel": w.numel(), "rel_encoder": float((ref - w).norm() / nw),
            "rel_harness": float((mine - w).norm() / nw)}


# ------------------------------------------------------------------- arms
def run_tensor(w, sc, H_raw, n_fit, a, log):
    R, C = w.shape
    res = {}
    t0 = time.time()

    def rec(name, hat, bpp, **extra):
        res[name] = sc(hat, bpp=bpp, **extra)
        log(f"    {name:<44} {res[name]['wt']:.5f} {res[name]['out']:.5f} "
            f"{res[name]['both_served']:.5f}  {time.time() - t0:6.1f}s")

    want = lambda name: a.arms is None or re.search(a.arms, name)

    # ---- P: the plane
    planes = {}
    eff0 = s6b(w)
    planes["s6b"] = eff0
    name = "P0 s6b h=1.00 (artifact)"
    if want(name):
        sc0 = expand(eff0, R, C)
        rec(name, trellis(w, sc0) * sc0, 4.0, clip=clip_fraction(w, sc0), octaves=octaves(eff0))
    for h in a.headrooms:
        name = f"P1 s6b h={h:.2f}"
        if want(name):
            eff = s6b(w, h)
            scx = expand(eff, R, C)
            rec(name, trellis(w, scx) * scx, 4.0, clip=clip_fraction(w, scx))
    name = "P1 s6b nearest-mantissa"
    if want(name):
        eff = two_tier(amax_half(w))
        scx = expand(eff, R, C)
        rec(name, trellis(w, scx) * scx, 4.0, clip=clip_fraction(w, scx))
    G = E4M3_MAX / amax_half(w).max()
    for h in [1.0] + a.headrooms:
        name = f"P2 e4m3 flat h={h:.2f}"
        eff = e4m3(amax_half(w, h), G)
        if h == 1.0:
            planes["e4m3"] = eff
        if want(name):
            scx = expand(eff, R, C)
            rec(name, trellis(w, scx) * scx, 4.0, clip=clip_fraction(w, scx), octaves=octaves(eff))
    best_t, best_e = None, None
    for t in a.thresholds:
        name = f"P3 e8m0-only t={t:.2f}"
        if want(name):
            eff = e8m0_amax(w, t)
            scx = expand(eff, R, C)
            rec(name, trellis(w, scx) * scx, 3.75, clip=clip_fraction(w, scx))
            if best_e is None or res[name]["wt"] < best_e:
                best_t, best_e = t, res[name]["wt"]
    if best_t is not None:
        planes["e8m0"] = e8m0_amax(w, best_t)

    # ---- R: refit the scale values, in the plane's own format
    requant = {"s6b": two_tier, "e4m3": lambda s: e4m3(s, G), "e8m0": e8m0_nearest,
               "fp32": lambda s: s}
    Hb16 = {}
    for sig in a.sigmas:
        Hb16[sig] = diag_blocks(regularize_hessian(H_raw, count=n_fit, sigma_reg=sig), HALF)
    for pname in ("s6b", "e4m3", "e8m0"):
        if pname not in planes:
            continue
        bpp = 3.5 + PLANE_BPP[pname]
        for weighting in ["plain"] + [f"H16 s={s}" for s in a.sigmas]:
            Hb = None if weighting == "plain" else Hb16[float(weighting.split("=")[1])]
            for fmt in ([pname, "fp32"] if pname == "s6b" and weighting == "plain" else [pname]):
                tag = f"R {pname} LS {weighting} -> {fmt}"
                if not want(tag):
                    continue
                eff = planes[pname]
                q = trellis(w, expand(eff, R, C))
                for it in range(1, a.iters + 1):
                    eff = requant[fmt](ls_scale(w, q, eff, Hb))
                    scx = expand(eff, R, C)
                    q = trellis(w, scx)
                    rec(f"{tag} x{it}", q * scx, bpp if fmt != "fp32" else float("nan"),
                        clip=clip_fraction(w, scx))

    # ---- M: half the redundancy
    for pname, bpp in (("s6b", 4.25), ("e8m0", 4.0)):
        if pname in planes and want("M"):
            scx = expand(planes[pname], R, C)
            rec(f"M L=2 on {pname}", multidim(w, scx, 2) * scx, bpp, clip=clip_fraction(w, scx))
            if a.iters and want("M"):
                q = multidim(w, scx, 2)
                eff = requant[pname](ls_scale(w, q, planes[pname], Hb16[a.sigmas[0]]))
                scx = expand(eff, R, C)
                rec(f"M L=2 on {pname} + H16-LS x1", multidim(w, scx, 2) * scx, bpp,
                    clip=clip_fraction(w, scx))

    # ---- F: feedback the calibration supports, and the one EXL3 used
    for pname in ("s6b", "e4m3"):
        if pname not in planes:
            continue
        bpp = 3.5 + PLANE_BPP[pname]
        for sig in a.sigmas:
            tag = f"F1 {pname} group-LDLQ s={sig}"
            if not want(tag):
                continue
            Hb32 = diag_blocks(regularize_hessian(H_raw, count=n_fit, sigma_reg=sig), GROUP)
            eff = planes[pname]
            for it in range(1, 3):
                scx = expand(eff, R, C)
                q, clip = group_feedback(w, scx, Hb32)
                rec(f"{tag} x{it}", q * scx, bpp, clip=clip)
                eff = requant[pname](ls_scale(w, q, eff, Hb16[sig]))
                scx = expand(eff, R, C)
                rec(f"{tag} x{it} + H16-LS", q * scx, bpp, clip=clip)
        for sig in a.sigmas:
            tag = f"F2 {pname} full-LDLQ s={sig}"
            if not want(tag):
                continue
            Hreg = regularize_hessian(H_raw, count=n_fit, sigma_reg=sig)
            plane = s6b if pname == "s6b" else (lambda t: e4m3(amax_half(t), G))
            hat, clip = full_ldlq(w, Hreg, plane)
            rec(tag, hat, bpp, clip=clip)
    return res


def main():
    global PROD, COSET
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--headrooms", type=float, nargs="+", default=[0.89, 0.94, 0.97, 1.05, 1.10, 1.20])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 1.19, 1.41, 1.68, 2.0])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.025, 0.1, 1.0])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--arms", default=None, help="regex over arm names; default all")
    ap.add_argument("--out", default="experiments/results/tessera_fp4_native_levers.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    PROD = torch.tensor([GRID.vector(c) for c in range(GRID.size)], dtype=torch.float32, device="cuda")
    COSET = labels(TCQ(FOREST, CC).subsets, FOREST.anchors, GRID.size, "cuda")
    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]

    out = {"tensors": [], "act": [], "verify": None, "arms": {}, "exl3_k4": EXL3_K4,
           "args": vars(a)}
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        n = xa.shape[0] // 2
        x_fit, x_ev = xa[:n].contiguous(), xa[n:].contiguous()
        g = select_mse_grid_input_global_scale([x_fit])
        xq_s = nvfp4_activation_qdq_served(x_ev, g).float()
        xq_a = quant_a4(x_ev)
        H_raw = (x_fit.T @ x_fit).contiguous()
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            sc = Scorer(w, x_ev, xq_s, xq_a)
            act = {"served": float((xq_s @ w.T - sc.y).norm() / sc.ny),
                   "amax": float((xq_a @ w.T - sc.y).norm() / sc.ny)}
            log(f"\n== L{layer} {proj} {tuple(w.shape)}  act leg served {act['served']:.5f} "
                f"amax {act['amax']:.5f}  (fit {n} / held-out {x_ev.shape[0]} tokens)")
            if out["verify"] is None:
                out["verify"] = verify_against_encoder(w)
                log(f"   verify vs encode_unit/reconstruct_unit: {out['verify']}")
            log(f"    {'arm':<44} {'wt':>7} {'out':>7} {'W4A4':>7}")
            res = run_tensor(w, sc, H_raw, n, a, log)
            out["tensors"].append(f"L{layer}.{proj}")
            out["act"].append(act)
            for k, v in res.items():
                out["arms"].setdefault(k, []).append(v)
            json.dump(out, open(a.out, "w"), indent=1)
            del w, sc
            torch.cuda.empty_cache()
        del xa, x_fit, x_ev, xq_s, xq_a, H_raw
        torch.cuda.empty_cache()

    summarise(out, log)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


def summarise(out, log):
    mean = lambda v: sum(v) / len(v)
    arms = out["arms"]
    base = arms.get("P0 s6b h=1.00 (artifact)")
    act_s = [x["served"] for x in out["act"]]
    act_a = [x["amax"] for x in out["act"]]
    log(f"\nact leg: served {mean(act_s):.5f}  amax {mean(act_a):.5f}   over {len(act_s)} tensors")
    log(f"{'arm':<44}{'bpp':>6}{'wt':>9}{'out':>9}{'W4A4':>9}{'out vs P0':>12}"
        f"{'min':>7}{'max':>7}{'vs EXL3@A4':>12}{'clip':>8}")
    for k, v in arms.items():
        bpp = v[0]["bpp"]
        wt, o, b = mean([r["wt"] for r in v]), mean([r["out"] for r in v]), mean([r["both_served"] for r in v])
        if base and len(base) == len(v):
            ratio = [bb["out"] / rr["out"] for bb, rr in zip(base, v)]
            rs = f"{mean(ratio):>11.3f}x{min(ratio):>7.3f}{max(ratio):>7.3f}"
        else:
            rs = f"{'-':>12}{'':>14}"
        exl = [rr["both_served"] / (EXL3_K4 ** 2 + aa ** 2) ** 0.5 for rr, aa in zip(v, act_s)]
        clip = mean([r.get("clip", 0.0) for r in v])
        log(f"{k:<44}{bpp:>6.2f}{wt:>9.5f}{o:>9.5f}{b:>9.5f}{rs}{mean(exl):>11.3f}x{clip:>8.4f}")
    exl_a4 = mean([(EXL3_K4 ** 2 + aa ** 2) ** 0.5 for aa in act_s])
    log(f"{'EXL3 K=4 (weight leg measured; W4A4 projected)':<44}{4.01:>6.2f}{'-':>9}{EXL3_K4:>9.5f}"
        f"{exl_a4:>9.5f}")


if __name__ == "__main__":
    main()
