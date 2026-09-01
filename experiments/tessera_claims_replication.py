"""Replicate the load-bearing ratios across layer types.

Every number reported on 2026-09-01 came from one 2048x512 slice of one
routed-expert gate_proj.  Three claims carry weight and each is re-measured
here on a spread of real GLM-5.3 Linears:

  1. trellis gain      -- trellis over the grid vs nearest-neighbour on the
                          best 128-subset of the SAME grid, both at 3.5 bpp.
                          Claimed 1.131x.
  2. learned-values    -- a per-tensor Lloyd-256 built into a custom
                          PayloadGrid and run under Tessera's own trellis at
                          rung 7, vs the stock grid.  Claimed 1.006x, i.e.
                          learning the values buys ~nothing.
  3. selection penalty -- actual anchors vs the best k-subset of the grid at
                          a sub-cap rung.  Claimed 1.21-1.43x on E2M1x2.

A claim that does not replicate across layer types is not a result.
"""
import argparse, json, re, time
import torch
from safetensors import safe_open
from tessera.alphabet import (E2M1_GRID, PayloadGrid, build_forest, tuple_grid,
                              alphabet_size)
from tessera.encode import encode_unit
from tessera.decode import reconstruct_unit
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
CC = ConvCode(memory=6)


def lloyd(x, k, iters=60):
    c = x[torch.randperm(len(x))[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


def medoids(x, v, k):
    D = torch.cdist(x, v)
    ch = [int(D.mean(0).argmin())]
    best = D[:, ch[0]].clone()
    while len(ch) < k:
        g = torch.minimum(D, best[:, None]).mean(0)
        g[torch.tensor(ch, device=D.device)] = float("inf")
        j = int(g.argmin()); ch.append(j); best = torch.minimum(best, D[:, j])
    return v[torch.tensor(ch, device=v.device)]


def measure(w, stock, vals):
    rows, cols = w.shape
    rel = lambda r: (torch.linalg.norm(w.float()-r.float())
                     / torch.linalg.norm(w.float())).item()
    peak = float(vals.abs().max())
    grp = w.float().reshape(rows // 32, 32, cols)
    scale = grp.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / peak
    pairs = (grp / scale).reshape(rows, cols).reshape(rows//2, 2, cols) \
                .permute(0, 2, 1).reshape(-1, 2)

    def unpair(q):
        q = q.reshape(rows // 2, cols, 2).permute(0, 2, 1).reshape(rows, cols)
        return (q.reshape(rows // 32, 32, cols) * scale).reshape(rows, cols)

    f7 = {7: build_forest(7, grid=stock)}
    tess = rel(reconstruct_unit(
        encode_unit(w, f7, (7,)*cols, code=CC, rotation=RotationState.NONE,
                    completion=0), f7, code=CC))

    samp = pairs[torch.randperm(len(pairs))[:60000]]
    g128 = medoids(samp, vals, 128)
    nn = rel(unpair(g128[torch.cdist(pairs, g128).argmin(1)]))

    learned = lloyd(samp, 256).cpu()
    learned = learned[learned.norm(dim=1).argsort()]
    lg = PayloadGrid(name="LEARNED_K2", values=tuple(learned.reshape(-1).tolist()),
                     arity=2, partition=stock.partition)
    fl = {7: build_forest(7, grid=lg)}
    lt = rel(reconstruct_unit(
        encode_unit(w, fl, (7,)*cols, code=CC, rotation=RotationState.NONE,
                    completion=0), fl, code=CC))

    # selection penalty at a sub-cap rung (rung 4, 32 anchors)
    k = alphabet_size(4, stock.rate_cap)
    anch = vals[torch.tensor(build_forest(4, grid=stock).blocks, device=vals.device)[:, 0]]
    ea = ((samp - anch[torch.cdist(samp, anch).argmin(1)]) ** 2).sum(1).mean().sqrt()
    eb = ((samp - medoids(samp, vals, k)[torch.cdist(samp, medoids(samp, vals, k)).argmin(1)]) ** 2).sum(1).mean().sqrt()
    return {"trellis_gain": nn / tess, "learned_values": tess / lt,
            "selection_penalty": (ea / eb).item(), "tessera_rel_err": tess}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=2048)
    ap.add_argument("--max-cols", type=int, default=384)
    ap.add_argument("--out", default="experiments/results/tessera_claims_replication.json")
    a = ap.parse_args()

    ix = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    wanted = [
        "model.language_model.layers.10.self_attn.b_proj.weight",
        "model.language_model.layers.10.self_attn.f_b_proj.weight",
        "model.language_model.layers.20.self_attn.g_b_proj.weight",
        "model.language_model.layers.10.mlp.shared_experts.gate_proj.weight",
        "model.language_model.layers.20.mlp.shared_experts.down_proj.weight",
        "model.language_model.layers.10.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.30.mlp.experts.5.down_proj.weight",
        "model.language_model.layers.40.mlp.experts.9.up_proj.weight",
    ]
    stock = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([stock.vector(c) for c in range(stock.size)],
                        dtype=torch.float32, device="cuda")
    rows_out = []
    print(f"{'tensor':>52} {'shape':>13} {'trellis':>8} {'learned':>8} {'select':>7}")
    for name in wanted:
        if name not in ix:
            print(f"  (absent) {name}"); continue
        with safe_open(f"{SRC}/{ix[name]}", framework="pt") as f:
            w = f.get_tensor(name)
        r = min(a.max_rows, w.shape[0] // 32 * 32)
        c = min(a.max_cols, w.shape[1])
        if r < 64 or c < 32:
            print(f"  (too small) {name} {tuple(w.shape)}"); continue
        w = w[:r, :c].contiguous().cuda()
        m = measure(w, stock, vals)
        m["tensor"] = name; m["shape"] = [r, c]
        rows_out.append(m)
        short = re.sub(r"^model\.language_model\.", "", name).replace(".weight", "")
        print(f"{short:>52} {str((r,c)):>13} {m['trellis_gain']:7.3f}x "
              f"{m['learned_values']:7.3f}x {m['selection_penalty']:6.3f}x")
        del w
    json.dump({"rows": rows_out}, open(a.out, "w"), indent=1)
    if rows_out:
        for key, claim in (("trellis_gain", 1.131), ("learned_values", 1.006),
                           ("selection_penalty", 1.37)):
            v = [r[key] for r in rows_out]
            print(f"\n{key}: min {min(v):.3f}x  max {max(v):.3f}x  "
                  f"mean {sum(v)/len(v):.3f}x   (single-tensor claim was {claim}x)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
