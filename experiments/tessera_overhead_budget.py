"""Tessera spends 1.0 of its 4.0 bpp on overhead.  EXL3 spends 0.0117.

The redundancy bit gets all the attention because it is structural -- ``cap =
payload_bits - 1`` is written into the grammar.  But it is only half the story:

    Tessera rung 7   3.500 payload + 0.500 scale planes  = 4.000 bpp
    EXL3 K=4         4.000 payload + 0.0117 rank-1 diag  = 4.0117 bpp

The scale term is the same size as the redundancy term and nobody has touched
it.  EXL3 gets away with rank-1 because after a Hadamard every group of the
tensor is statistically the same, so a row scale times a column scale is a
sufficient description and per-group amax is redundant precision.

Three questions, in the order that matters:

  1. What do the group-32 scale planes actually BUY?  Drop them for a rank-1
     fit and measure the error we pay for the 0.49 bpp we get back.
  2. Does a rotation change that answer?  (It should be worth more here than
     in the error-reduction test that measured 1.003x -- there the rotation was
     asked to lower error at fixed rate; here it is asked to make one scale
     describe the whole tensor.)
  3. What is the CEILING for anything built on a fixed 256-value alphabet?
     Lloyd-optimal free 2D centroids at 8 bits/pair is that ceiling: no
     trellis, no partition and no alphabet design can beat it, so if EXL3 is
     already below it the remaining gap is dimensional and no amount of
     puncturing or k-tupling reaches it.
"""
import argparse, json
import torch
from safetensors import safe_open
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
CC = ConvCode(memory=6)
EXL3_K4 = 0.05653          # experiments/results/exl3_rate_sweep_K4.json, 6 tensors


def rank1(w, iters=8):
    """diag(r) @ Wn @ diag(c) -- EXL3's suh/svh, fitted by norm equalisation."""
    r = torch.ones(w.shape[0], 1, device=w.device)
    c = torch.ones(1, w.shape[1], device=w.device)
    for _ in range(iters):
        n = w / c
        r = n.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
        n = w / r
        c = n.pow(2).mean(0, keepdim=True).sqrt().clamp(min=1e-12)
    return r, c


def group_scale(w, grid_peak, group=32):
    g = w.reshape(w.shape[0] // group, group, w.shape[1])
    return (g.abs().amax(1, keepdim=True).clamp(min=1e-12) / grid_peak)


def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / (h.shape[0] ** 0.5)


def lloyd(x, k, iters=40, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for d in range(x.shape[1]):
            s = torch.zeros(k, device=x.device).index_add_(0, a, x[:, d])
            n = torch.zeros(k, device=x.device).index_add_(0, a, torch.ones_like(a, dtype=x.dtype))
            c[:, d] = torch.where(n > 0, s / n.clamp(min=1), c[:, d])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--out", default="experiments/results/tessera_overhead_budget.json")
    a = ap.parse_args()

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(vals.abs().max())
    forests = {grid.rate_cap: build_forest(grid.rate_cap, grid=grid)}

    arms, rows_out = {}, []
    for layer in a.layers:
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w0 = f.get_tensor(name)[:a.rows, :a.cols].contiguous().cuda().float()
            R, C = w0.shape
            nrm = torch.linalg.norm(w0)
            rel = lambda r: (torch.linalg.norm(w0 - r.float()) / nrm).item()
            # bpp of the overhead each arm actually writes
            bpp_group = (8 * (R // 32) * C + 4 * (R // 16) * C) / (R * C)
            bpp_rank1 = 16 * (R + C) / (R * C)

            H = hadamard(C, w0.device)
            got = {}
            for rot in (False, True):
                w = w0 @ H if rot else w0
                nrm_r = torch.linalg.norm(w)
                # error is always measured in the ORIGINAL basis: H is orthogonal
                # so ||W - R||  is preserved, but undo it anyway rather than
                # asserting the identity we are relying on.
                back = (lambda r: r @ H.T) if rot else (lambda r: r)

                for scaling in ("group32", "rank1"):
                    if scaling == "group32":
                        s = group_scale(w, peak)
                        xn = (w.reshape(R // 32, 32, C) / s).reshape(R, C)
                        unscale = lambda q: (q.reshape(R // 32, 32, C) * s).reshape(R, C)
                        ovh = bpp_group
                    else:
                        r1, c1 = rank1(w)
                        base = w / (r1 * c1)
                        # one global multiplier, line-searched: amax would be set
                        # by the single worst outlier in the whole tensor.
                        best, alpha = None, None
                        for t in torch.linspace(0.5, 1.0, 26):
                            cand = float(base.abs().amax()) * float(t) / peak
                            q = vals[torch.cdist((base / cand).reshape(R // 2, 2, C)
                                     .permute(0, 2, 1).reshape(-1, 2), vals).argmin(1)]
                            e = ((q.reshape(R // 2, C, 2).permute(0, 2, 1).reshape(R, C)
                                  * cand - base) ** 2).sum()
                            if best is None or e < best:
                                best, alpha = e, cand
                        xn = base / alpha
                        unscale = lambda q: q * alpha * (r1 * c1)
                        ovh = bpp_rank1

                    pairs = xn.reshape(R // 2, 2, C).permute(0, 2, 1).reshape(-1, 2)
                    unpair = lambda q: q.reshape(R // 2, C, 2).permute(0, 2, 1).reshape(R, C)

                    # -- arm: plain NN on the grid, 8 bits/pair = 4.0 payload
                    e_nn = rel(back(unscale(unpair(vals[torch.cdist(pairs, vals).argmin(1)]))))
                    got[(rot, scaling, "NN-grid")] = (4.0 + ovh, e_nn)

                    # -- arm: Lloyd-optimal FREE 2D centroids, same 8 bits/pair.
                    #    The ceiling for every fixed-alphabet scheme.
                    cb = lloyd(pairs[torch.randperm(len(pairs))[:200000]], 256)
                    e_ll = rel(back(unscale(unpair(cb[torch.cdist(pairs, cb).argmin(1)]))))
                    got[(rot, scaling, "Lloyd-256")] = (4.0 + ovh, e_ll)

                    # -- arm: Tessera's trellis at rung 7, 7 bits/pair = 3.5
                    u = encode_unit(w if scaling == "group32" else (xn * alpha) * 1.0,
                                    forests, (grid.rate_cap,) * C, code=CC,
                                    rotation=RotationState.NONE, completion=0,
                                    group=32, half=16)
                    rec = reconstruct_unit(u, forests, CC)
                    if scaling == "rank1":
                        rec = rec * (r1 * c1)
                        ovh_t = bpp_rank1 + bpp_group   # trellis still wrote group scales
                    else:
                        ovh_t = bpp_group
                    got[(rot, scaling, "trellis-r7")] = (3.5 + ovh_t, rel(back(rec)))

            rows_out.append({"layer": layer, "proj": proj,
                             "arms": {f"{'rot' if k[0] else 'raw'}/{k[1]}/{k[2]}": v
                                      for k, v in got.items()}})
            for k, v in got.items():
                arms.setdefault(k, []).append(v[1])
            print(f"  {layer:>3} {proj:<10} done")

    print(f"\n{'arm':<34}{'bpp':>8}{'rel err':>10}{'vs EXL3':>10}")
    order = sorted(arms, key=lambda k: sum(arms[k]) / len(arms[k]))
    for k in order:
        m = sum(arms[k]) / len(arms[k])
        bpp = rows_out[0]["arms"][f"{'rot' if k[0] else 'raw'}/{k[1]}/{k[2]}"][0]
        print(f"{('rot' if k[0] else 'raw')+'/'+k[1]+'/'+k[2]:<34}{bpp:>8.3f}"
              f"{m:>10.5f}{m/EXL3_K4:>9.2f}x")
    print(f"{'EXL3 K=4 (reference)':<34}{4.0117:>8.3f}{EXL3_K4:>10.5f}{1.0:>9.2f}x")
    json.dump({"rows": rows_out, "exl3_k4": EXL3_K4}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
