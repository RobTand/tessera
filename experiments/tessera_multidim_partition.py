"""Fractional redundancy IS reachable with a fixed alphabet.  Wei (1987).

An earlier commit here claimed the redundancy bit cannot be split without
abandoning the fixed alphabet.  The proof it rested on is sound but narrower
than the claim: every rate-k/(k+1) convolutional code spends exactly one bit per
POSITION, so no member of the per-position Ungerboeck family does better.  The
escape is not a different code, it is a different partition -- the same
multidimensional set-partitioning that lets V.34 modems pay half a bit of
constellation expansion instead of a whole one.

Group L positions into a super-symbol and label it by the SUM of its positions'
subset labels mod 4.  One conv-code bit picks the super-label; 2(L-1) bits pick
which member of that class the L positions take; 6L bits pick points inside the
chosen per-position subsets:

    bits per super-symbol = 1 + 2(L-1) + 6L        over 2L weights
    payload/weight        = (8L - 1) / 2L = 4 - 1/(2L)

    L = 1   3.5000   (today)
    L = 2   3.7500
    L = 4   3.8750
    L = 8   3.9375

Same asymptote as k-tupling -- and that is the whole point of running it.
k-tupling reaches 4 - 1/k by building a G^k codebook, so the Viterbi scores 16x
more anchors per step.  This reaches 4 - 1/(2L) with the codebook untouched at
256, because the metric decomposes: the min-plus convolution over Z/4 is
associative, so L positions combine in a binary tree of 4x4 minima and the cost
stays LINEAR in L.  Free rate where the k axis charged exponentially.

Whether it is worth having is a different question from whether it exists, and
the honest test is the frontier, not the baseline: L=2 buys 0.25 bpp of payload,
so it must beat the log-linear interpolation between today's rung and the
zero-redundancy nearest-neighbour point at 4.5 bpp.  Beating rung 7 outright
would be no achievement -- it spends more.
"""
import argparse, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.trellis import TCQ, ConvCode
from tessera_free_codebook_trellis import colour4, lloyd, transitions
from tessera_memory_and_codebook import labels

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
CC = ConvCode(memory=6)


def subset_costs(x, cb, lab):
    """(T,B,4) best cost and (T,B,4) best code, per position per subset."""
    d = torch.cdist(x.reshape(-1, x.shape[-1]), cb).reshape(*x.shape[:2], len(cb)) ** 2
    groups = [(lab == u).nonzero().squeeze(1) for u in range(4)]
    return (torch.stack([d[:, :, g].min(-1).values for g in groups], -1),
            torch.stack([g[d[:, :, g].argmin(-1)] for g in groups], -1))


def combine(cost):
    """One level of the tree: adjacent positions, labels summed mod 4."""
    A, B = cost[0::2], cost[1::2]
    roll = torch.arange(4, device=cost.device)
    terms = torch.stack([A[..., u:u + 1] + B[..., (roll - u) % 4] for u in range(4)], -1)
    best, arg = terms.min(-1)
    return best, arg                      # (n,B,4), (n,B,4) -> left label


def viterbi_super(cost, code):
    """One conv-code bit per super-symbol; returns the chosen super-label."""
    T, B, _ = cost.shape
    S = code.states
    _, sub, pred, pbit = transitions(code)
    sub, pred, pbit = sub.to(cost.device), pred.to(cost.device), pbit.to(cost.device)
    metric = torch.full((B, S), float('inf'), device=cost.device)   # decoder starts at 0
    metric[:, 0] = 0.0
    back = torch.empty(T, B, S, dtype=torch.uint8, device=cost.device)
    ps, pu = pred.reshape(-1), sub[pred, pbit].reshape(-1)
    for t in range(T):
        cand = (metric[:, ps] + cost[t][:, pu]).reshape(B, S, 2)
        metric, arg = cand.min(-1)
        back[t] = arg.to(torch.uint8)
    state = metric.argmin(-1)
    out = torch.empty(T, B, dtype=torch.long, device=cost.device)
    for t in range(T - 1, -1, -1):
        j = back[t].gather(1, state[:, None]).squeeze(1).long()
        prev = pred[state, j]
        out[t] = sub[prev, pbit[state, j]]
        state = prev
    return out


def encode_multidim(x, cb, lab, code, L):
    cost, pick = subset_costs(x, cb, lab)
    args = []
    cur = cost
    for _ in range(L.bit_length() - 1):
        cur, arg = combine(cur)
        args.append(arg)
    labs = viterbi_super(cur, code)                       # (T/L, B)
    for arg in reversed(args):                            # descend the tree
        u = arg.gather(-1, labs[..., None]).squeeze(-1)
        labs = torch.stack([u, (labs - u) % 4], 1).reshape(-1, labs.shape[1])
    return cb[pick.gather(-1, labs[..., None]).squeeze(-1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="experiments/results/tessera_multidim_partition.json")
    a = ap.parse_args()

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(vals.abs().max())
    forest = build_forest(grid.rate_cap, grid=grid)
    coset = labels(TCQ(forest, CC).subsets, forest.anchors, grid.size, "cuda")

    acc = {}
    for layer in a.layers:
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name)[:a.rows, :a.cols].contiguous().cuda().float()
            R, C = w.shape
            nrm = torch.linalg.norm(w)
            s = (w.reshape(R // 32, 32, C).abs().amax(1, keepdim=True)
                 .clamp(min=1e-12) / peak)
            xn = (w.reshape(R // 32, 32, C) / s).reshape(R, C)
            seq = xn.reshape(R // 2, 2, C).permute(0, 2, 1).contiguous()
            rel = lambda q: float(torch.linalg.norm(
                w - (q.permute(0, 2, 1).reshape(R, C)
                     .reshape(R // 32, 32, C) * s).reshape(R, C)) / nrm)

            cb = lloyd(seq.reshape(-1, 2)[torch.randperm(R * C // 2)[:300000]], 256)
            lb = colour4(cb)
            for tag, book, lab in (("E2M1x2", vals, coset), ("Lloyd ", cb, lb)):
                for L in a.levels:
                    acc.setdefault((tag, L, 4 - 1 / (2 * L) + 0.5), []).append(
                        rel(encode_multidim(seq, book, lab, CC, L)))
                # r = 0: no trellis at all, 4.0 payload
                acc.setdefault((tag, 0, 4.5), []).append(
                    rel(book[torch.cdist(seq.reshape(-1, 2), book).argmin(1)]
                        .reshape(R // 2, C, 2)))
            print(f"  {layer:>3} {proj:<10} done", flush=True)

    m = {k: sum(v) / len(v) for k, v in acc.items()}
    print(f"\n{'alphabet':<9}{'L':>3}{'payload':>9}{'bpp':>7}{'rel err':>10}"
          f"{'vs L=1':>9}{'vs frontier':>13}")
    for tag in ("E2M1x2", "Lloyd "):
        one, zero = m[(tag, 1, 4.0)], m[(tag, 0, 4.5)]
        for L in sorted(a.levels) + [0]:
            bpp = 4.5 if L == 0 else 4 - 1 / (2 * L) + 0.5
            e = m[(tag, L, bpp)]
            # log-linear frontier through the two endpoints of the redundancy axis
            f = one * (zero / one) ** ((bpp - 4.0) / 0.5)
            note = "-" if L in (0, 1) else f"{f/e:>7.3f}x " + ("ABOVE" if e < f else "below")
            print(f"{tag:<9}{L:>3}{(4.0 if L==0 else 4-1/(2*L)):>9.4f}{bpp:>7.3f}"
                  f"{e:>10.5f}{one/e:>8.3f}x{note:>13}")
    json.dump({f"{k[0].strip()}_L{k[1]}_{k[2]}": v for k, v in acc.items()},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
