"""Is the trellis's redundancy bit worth what it costs?

TCQ spends exactly one bit per code: the codebook is 2^(R+1) but the rate is R
(`cap = payload_bits - 1`).  That bit is why payload = 4 - 1/k on an E2M1 grid,
and why Tessera converts bytes to payload at 87.5% where EXL3 manages 99.7%.

Puncturing -- spending a FRACTION of a redundancy bit -- would interpolate
between two curves that can both be measured today:

  trellis(p)  R=p over the full 256-code book, 1 bit of redundancy, p bits spent
  NN(p)       nearest-neighbour over the best 2^p subset, 0 redundancy

At the same p they cost the same bits, so their vertical gap IS the value of
the redundancy bit at that rate.  And NN reaches p=8 (4.0 bpp payload) where
the trellis structurally cannot go.

The question puncturing turns on: does NN(p+1) beat trellis(p)?  If spending
the bit on payload beats spending it on shaping, the trellis is over-paying for
redundancy and a punctured code lands somewhere better than both.  If not,
puncturing is chasing a bit that is already well spent.
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


def medoids(x, v, k):
    """Best k-subset of the grid -- the strongest 0-redundancy codebook of that
    size, so the trellis is not being flattered by a lazy baseline."""
    D = torch.cdist(x, v)
    ch = [int(D.mean(0).argmin())]
    best = D[:, ch[0]].clone()
    while len(ch) < k:
        g = torch.minimum(D, best[:, None]).mean(0)
        g[torch.tensor(ch, device=D.device)] = float("inf")
        j = int(g.argmin()); ch.append(j); best = torch.minimum(best, D[:, j])
    return v[torch.tensor(ch, device=v.device)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="model.language_model.layers.10.mlp.experts.0.gate_proj.weight")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--out", default="experiments/results/tessera_redundancy_exchange.json")
    a = ap.parse_args()

    shard = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"][a.tensor]
    with safe_open(f"{SRC}/{shard}", framework="pt") as f:
        w = f.get_tensor(a.tensor)
    w = w[:a.rows, :a.cols].contiguous().cuda().float()
    rows, cols = w.shape
    rel = lambda r: (torch.linalg.norm(w - r.float()) / torch.linalg.norm(w)).item()

    grid = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(vals.abs().max())
    grp = w.reshape(rows // 32, 32, cols)
    scale = grp.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / peak
    pairs = (grp / scale).reshape(rows, cols).reshape(rows//2, 2, cols) \
                .permute(0, 2, 1).reshape(-1, 2)
    def unpair(q):
        q = q.reshape(rows // 2, cols, 2).permute(0, 2, 1).reshape(rows, cols)
        return (q.reshape(rows // 32, 32, cols) * scale).reshape(rows, cols)
    samp = pairs[torch.randperm(len(pairs))[:60000]]

    tre, nn = {}, {}
    print(f"{a.tensor.split('layers.')[1]}  {rows}x{cols}")
    print(f"  {'p bits':>7}{'bpp':>7}{'trellis':>10}{'NN':>10}{'redundancy worth':>18}")
    for p in range(3, 9):
        e_nn = None
        if 2**p <= grid.size:
            cb = medoids(samp, vals, 2**p)
            e_nn = rel(unpair(cb[torch.cdist(pairs, cb).argmin(1)]))
            nn[p] = e_nn
        e_tr = None
        if p <= grid.rate_cap:
            f = {p: build_forest(p, grid=grid)}
            u = encode_unit(w, f, (p,)*cols, code=CC,
                            rotation=RotationState.NONE, completion=0)
            e_tr = rel(reconstruct_unit(u, f, code=CC))
            tre[p] = e_tr
        worth = f"{e_nn/e_tr:.3f}x" if (e_nn and e_tr) else "-"
        print(f"  {p:7d}{p/2:7.2f}{(f'{e_tr:.5f}' if e_tr else '-'):>10}"
              f"{(f'{e_nn:.5f}' if e_nn else '-'):>10}{worth:>18}")

    print("\n  NN(p+1) vs trellis(p) -- SAME CODEBOOK, and NN spends one more")
    print("  bit.  Not a matched-rate comparison; the matched-rate one is the")
    print("  'redundancy worth' column above, which is clean only at p = cap.")
    verdict = []
    for p in sorted(tre):
        if p + 1 in nn:
            r = nn[p+1] / tre[p]
            verdict.append(r)
            print(f"    NN({p+1}) {nn[p+1]:.5f}  vs  trellis({p}) {tre[p]:.5f}"
                  f"   -> {r:.3f}x for +1 bit")
    json.dump({"trellis": tre, "nn": nn}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
