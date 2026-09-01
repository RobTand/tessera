"""The product structure of E2M1x2 costs more than the trellis earns.

Measured in tessera_overhead_budget.py, at a matched ~4.0 bpp:

    E2M1x2 grid, nearest-neighbour     0.10988
    free Lloyd 256-point 2D codebook   0.09134     1.203x better

The trellis is worth 1.134x.  The alphabet is worth 1.203x.  Tessera has been
paying a redundancy bit to win the smaller of the two while giving away the
larger for free, because ``tuple_grid(E2M1_GRID, 2)`` is a *tensor product*:
sixteen scalar levels crossed with themselves.  A product grid spends points on
the corners of a square where the weight density is a round blob, and no amount
of re-spacing the scalar ladder fixes that -- which is exactly why the earlier
learned-values experiment came back 1.006x and then inverted to 0.990x.  It
learned the levels while keeping the cross.

The question this answers is whether the two compose.  They are not obviously
independent: the trellis's gain comes from Ungerboeck set-partitioning, and set
partitioning on a product grid has a closed form (``sum of base ranks mod 4``)
that a free point set does not have.  So the free codebook has to earn its
partition too.

The generalisation used here is the criterion, not the formula.  Ungerboeck's
rule is "maximise the minimum distance within each subset"; on a line that is
stride-4 through the value order, and on a free 2D point set it is a greedy
4-colouring that places each centroid in whichever subset leaves it furthest
from its same-subset neighbours.  Same criterion, no product assumed.

Arm D is the control that makes the rest readable: the same standalone Viterbi
run on the E2M1x2 grid with stride partitioning must reproduce Tessera's own
encoder.  If it does not, nothing below means anything.
"""
import argparse, json
import torch
from safetensors import safe_open
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid, value_order
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
CC = ConvCode(memory=6)
EXL3_K4 = 0.05653


def transitions(code):
    """(next_state, subset) tables, and the inverse map the Viterbi walks."""
    S = code.states
    nxt = torch.zeros(S, 2, dtype=torch.long)
    sub = torch.zeros(S, 2, dtype=torch.long)
    for s in range(S):
        for b in (0, 1):
            n, u = code.step(s, b)
            nxt[s, b], sub[s, b] = n, u
    pred = torch.zeros(S, 2, dtype=torch.long)
    pbit = torch.zeros(S, 2, dtype=torch.long)
    fill = [0] * S
    for s in range(S):
        for b in (0, 1):
            n = int(nxt[s, b])
            pred[n, fill[n]], pbit[n, fill[n]] = s, b
            fill[n] += 1
    assert all(f == 2 for f in fill), fill
    return nxt, sub, pred, pbit


def colour4(c):
    """Ungerboeck's criterion on a free point set: put each centroid in the
    subset whose nearest same-subset member is furthest away."""
    D = torch.cdist(c, c)
    D.fill_diagonal_(float("inf"))
    order = D.min(1).values.argsort()          # densest region first, it is
    lab = torch.full((len(c),), -1, dtype=torch.long, device=c.device)
    for i in order.tolist():                   # the one with least freedom
        best, pick = -1.0, 0
        for u in range(4):
            mem = (lab == u).nonzero().squeeze(1)
            d = float(D[i, mem].min()) if len(mem) else float("inf")
            if d > best:
                best, pick = d, u
        lab[i] = pick
    # balance: the trellis needs |subset| = size/4 exactly
    for u in range(4):
        while int((lab == u).sum()) > len(c) // 4:
            mem = (lab == u).nonzero().squeeze(1)
            worst = mem[D[mem][:, mem].min(1).values.argmin()]
            cand = [v for v in range(4) if int((lab == v).sum()) < len(c) // 4]
            lab[worst] = max(cand, key=lambda v: float(
                D[worst, (lab == v).nonzero().squeeze(1)].min())
                if int((lab == v).sum()) else 1e9)
    return lab


def viterbi(x, cb, lab, code):
    """Rate-(log2|subset|+1) TCQ over an arbitrary codebook. x: (T, B, d)."""
    T, B, _ = x.shape
    S = code.states
    _, sub, pred, pbit = transitions(code)
    sub, pred, pbit = sub.to(x.device), pred.to(x.device), pbit.to(x.device)
    subsets = [(lab == u).nonzero().squeeze(1) for u in range(4)]

    # per position, the best point and its cost inside each of the four subsets
    d = torch.cdist(x.reshape(-1, x.shape[-1]), cb).reshape(T, B, len(cb)) ** 2
    cost = torch.stack([d[:, :, s].min(-1).values for s in subsets], -1)   # T,B,4
    pick = torch.stack([s[d[:, :, s].argmin(-1)] for s in subsets], -1)    # T,B,4

    # The decoder replays from state 0 (TCQ.decode), so the encoder must start
    # there too.  A free start (every state at 0) finds paths the decoder cannot
    # reproduce and reads ~0.3% low in SSE -- every standalone number before
    # 2026-09-01 carried that optimism, uniformly across arms.
    metric = torch.full((B, S), float('inf'), device=x.device)
    metric[:, 0] = 0.0
    back = torch.empty(T, B, S, dtype=torch.uint8, device=x.device)
    ps = pred.reshape(-1)                         # 2S
    pu = sub[pred, pbit].reshape(-1)              # subset on each incoming edge
    for t in range(T):
        cand = metric[:, ps] + cost[t][:, pu]     # B, 2S
        cand = cand.reshape(B, S, 2)
        metric, arg = cand.min(-1)
        back[t] = arg.to(torch.uint8)

    state = metric.argmin(-1)
    out = torch.empty(T, B, dtype=torch.long, device=x.device)
    for t in range(T - 1, -1, -1):
        j = back[t].gather(1, state[:, None]).squeeze(1).long()
        prev = pred[state, j]
        out[t] = pick[t].gather(1, sub[prev, pbit[state, j]][:, None]).squeeze(1)
        state = prev
    return cb[out]


def lloyd(x, k, iters=50, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        n = torch.zeros(k, device=x.device).index_add_(
            0, a, torch.ones(len(a), device=x.device))
        for d in range(x.shape[1]):
            s = torch.zeros(k, device=x.device).index_add_(0, a, x[:, d])
            c[:, d] = torch.where(n > 0, s / n.clamp(min=1), c[:, d])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--out", default="experiments/results/tessera_free_codebook_trellis.json")
    a = ap.parse_args()

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(vals.abs().max())
    forests = {grid.rate_cap: build_forest(grid.rate_cap, grid=grid)}
    stride = torch.zeros(grid.size, dtype=torch.long, device="cuda")
    stride[torch.tensor(list(value_order(grid)), device="cuda")] = \
        torch.arange(grid.size, device="cuda") % 4

    acc = {}
    for layer in a.layers:
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name)[:a.rows, :a.cols].contiguous().cuda().float()
            R, C = w.shape
            nrm = torch.linalg.norm(w)
            rel = lambda r: (torch.linalg.norm(w - r.float()) / nrm).item()
            s = (w.reshape(R // 32, 32, C).abs().amax(1, keepdim=True)
                 .clamp(min=1e-12) / peak)
            xn = (w.reshape(R // 32, 32, C) / s).reshape(R, C)
            un = lambda q: (q.reshape(R // 32, 32, C) * s).reshape(R, C)
            # (T, B, 2): T pair-positions down the rows, B columns -- the trellis
            # runs along the same axis Tessera's encoder does.
            seq = xn.reshape(R // 2, 2, C).permute(0, 2, 1).contiguous()
            flat = lambda q: q.permute(0, 2, 1).reshape(R, C)

            got = {}
            # A -- Tessera's own encoder, the number to beat
            u = encode_unit(w, forests, (grid.rate_cap,) * C, code=CC,
                            rotation=RotationState.NONE, completion=0,
                            group=32, half=16)
            got["A tessera encoder  E2M1x2 + coset"] = (4.0, rel(reconstruct_unit(u, forests, CC)))
            # D -- control: this file's Viterbi on the same grid + stride subsets
            got["D control          E2M1x2 + stride"] = (
                4.0, rel(un(flat(viterbi(seq, vals, stride, CC)))))
            # B -- free codebook, greedy-4-coloured, same trellis, same rate
            cb = lloyd(seq.reshape(-1, 2)[torch.randperm(R * C // 2)[:300000]], 256)
            got["B free codebook    Lloyd + colour4"] = (
                4.0, rel(un(flat(viterbi(seq, cb, colour4(cb), CC)))))
            # C -- free codebook, no trellis, spends the redundancy bit on rate
            got["C free codebook    Lloyd + NN"] = (
                4.5, rel(un(flat(cb[torch.cdist(seq.reshape(-1, 2), cb)
                                     .argmin(1)].reshape(R // 2, C, 2)))))
            for k, v in got.items():
                acc.setdefault(k, []).append(v)
            print(f"  {layer:>3} {proj:<10} " +
                  "  ".join(f"{k[0]}={v[1]:.5f}" for k, v in got.items()))

    base = sum(v[1] for v in acc["A tessera encoder  E2M1x2 + coset"]) / len(a.layers) / len(a.projs)
    print(f"\n{'arm':<36}{'bpp':>7}{'rel err':>10}{'vs A':>8}{'vs EXL3':>9}")
    for k in sorted(acc, key=lambda k: sum(v[1] for v in acc[k])):
        m = sum(v[1] for v in acc[k]) / len(acc[k])
        print(f"{k:<36}{acc[k][0][0]:>7.2f}{m:>10.5f}{base/m:>7.3f}x{m/EXL3_K4:>8.2f}x")
    print(f"{'EXL3 K=4 (reference)':<36}{4.0117:>7.2f}{EXL3_K4:>10.5f}"
          f"{base/EXL3_K4:>7.3f}x{1.0:>8.2f}x")
    json.dump({k: v for k, v in acc.items()}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
