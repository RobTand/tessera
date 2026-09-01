"""Which 4-way subset partition of the E2M1x2 grid does the trellis want?

`TCQ.subsets` uses (sum of ranks) mod 4 at the rate cap.  On Z^2 that
sublattice has intra-subset distance^2 = 2 ((1,-1) is in it); the Ungerboeck
4-way partition of Z^2 is the cosets of 2Z^2 ((i mod 2, j mod 2)), distance^2
= 4, and (i + 2j) mod 4 is another index-4 sublattice with distance^2 = 4.
Measured on i.i.d. Gaussian at one optimal global scale, then on real experts
via the harness.  A partition is a per-grid rule, so a better one is a new
grid variant (profile id), not a container change.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera.trellis import TCQ  # noqa: E402
from tessera_memory_and_codebook import labels  # noqa: E402
from tessera_free_codebook_trellis import viterbi  # noqa: E402

HALF = 16


def partitions(grid, device):
    keys = grid.keys
    ri = torch.tensor([keys[c][0] for c in range(grid.size)], device=device)
    rj = torch.tensor([keys[c][1] for c in range(grid.size)], device=device)
    base = labels(TCQ(F.FOREST, F.CC).subsets, F.FOREST.anchors, grid.size, device)
    cands = {
        "sum mod 4 (today)": base,
        "(i mod 2, j mod 2) = 2Z^2 cosets": (ri & 1) | ((rj & 1) << 1),
        "(i + 2j) mod 4": (ri + 2 * rj) & 3,
        "(2i + j) mod 4": (2 * ri + rj) & 3,
    }
    for k, v in cands.items():
        counts = torch.bincount(v, minlength=4).tolist()
        assert counts == [grid.size // 4] * 4, (k, counts)
    return cands


def encode(w, scale, lab):
    R, C = w.shape
    seq = (w / scale).reshape(R // 2, 2, C).permute(0, 2, 1).contiguous()
    return viterbi(seq, F.PROD, lab, F.CC).permute(0, 2, 1).reshape(R, C)


def gaussian_arm(lab, rows=2048, cols=4096, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(rows, cols, generator=g, device="cuda")
    best = None
    for k in [2.2 + 0.1 * i for i in range(16)]:
        sc = F.expand(torch.full((rows * cols // HALF,), k / F.PEAK, device="cuda"), rows, cols)
        u = encode(z, sc, lab) * sc
        a = float((u * z).sum() / (u * u).sum().clamp_min(1e-30))
        err = float((z - a * u).norm() / z.norm())
        if best is None or err < best[0]:
            best = (err, k)
    return best


def refit_arm(w, lab, iters=3):
    """refit-only, flat E4M3 plane, weight-space and (if given) nothing else."""
    G = F.E4M3_MAX / F.amax_half(w).max()
    eff = F.e4m3(F.amax_half(w), G)
    q = encode(w, F.expand(eff, *w.shape), lab)
    for _ in range(iters):
        eff = F.e4m3(F.ls_scale(w, q, eff), G)
        q = encode(w, F.expand(eff, *w.shape), lab)
    return float((w - q * F.expand(eff, *w.shape)).norm() / w.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--out", default="experiments/results/tessera_partition_gaussian.json")
    a = ap.parse_args()
    F.PROD = torch.tensor([F.GRID.vector(c) for c in range(F.GRID.size)], dtype=torch.float32, device="cuda")
    cands = partitions(F.GRID, "cuda")
    out = {"gaussian": {}, "experts": {}}
    print("== i.i.d. Gaussian 2048x4096, one optimal global scale, rate 3.5 (Shannon 0.08839)")
    for k, lab in cands.items():
        err, clip = gaussian_arm(lab)
        out["gaussian"][k] = {"rms_rel": err, "clip_k": clip, "loss_x": err / 2 ** -3.5}
        print(f"   {k:36s} rms_rel {err:.5f}  loss {err / 2 ** -3.5:.3f}x  clip {clip:.1f} sigma", flush=True)
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
    for layer in a.layers:
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{F.SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            row = {}
            for k, lab in cands.items():
                row[k] = refit_arm(w, lab)
            base = row["sum mod 4 (today)"]
            out["experts"][f"L{layer} {proj}"] = row
            print(f"== L{layer} {proj}: " + "  ".join(f"{k.split(' ')[0]}={v:.5f} ({base / v:.3f}x)" for k, v in row.items()), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
