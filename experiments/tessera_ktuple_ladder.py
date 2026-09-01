"""Does k=3 tupling beat k=2 at matched rate, or only raise the ceiling?

The trellis spends one redundancy bit per CODE, so it costs 1/k bpp:
`rate_cap = payload_bits - 1` gives a payload ceiling of `cap/k` = 3.500 at
k=2 and 3.667 at k=3.  That is a higher ceiling, not by itself a win --
k=3 at its cap is a BIGGER artifact than k=2 at its cap.

The real question is matched rate.  k=3's rungs bracket k=2's ceiling
(10/3 = 3.333 and 11/3 = 3.667), so if k=3 at 3.333 bpp beats k=2 at
3.500, it wins on strictly less rate and the amortised redundancy bit is
real money.  Same E2M1 crayon box throughout, so NVFP4 materialisation is
untouched either way.
"""
import argparse, json, time
import torch
from safetensors import safe_open
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.encode import encode_unit
from tessera.decode import reconstruct_unit
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="model.language_model.layers.10.mlp.experts.0.gate_proj.weight")
    ap.add_argument("--rows", type=int, default=2046, help="divisible by both 2 and 3")
    ap.add_argument("--cols", type=int, default=384)
    ap.add_argument("--out", default="experiments/results/tessera_ktuple_ladder.json")
    a = ap.parse_args()

    shard = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"][a.tensor]
    with safe_open(f"{SRC}/{shard}", framework="pt") as f:
        w = f.get_tensor(a.tensor)
    w = w[:a.rows, :a.cols].contiguous().cuda()
    rel = lambda r: (torch.linalg.norm(w.float()-r.float())
                     / torch.linalg.norm(w.float())).item()
    cc = ConvCode(memory=6)
    print(f"{a.tensor}  {tuple(w.shape)}")
    print(f"  {'k':>2} {'rung':>5} {'c':>2} {'payload bpp':>12} {'rel_err':>9} {'encode':>8}")

    rows = []
    for k, points in ((2, [(7, 0)]), (3, [(10, 0), (11, 0), (10, 1)])):
        grid = tuple_grid(E2M1_GRID, k)
        for rate, c in points:
            forests = {rate: build_forest(rate, grid=grid)}
            t0 = time.time()
            u = encode_unit(w, forests, (rate,)*a.cols, code=cc,
                            rotation=RotationState.NONE, completion=c)
            dt = time.time() - t0
            e = rel(reconstruct_unit(u, forests, code=cc))
            bpp = (rate + c) / k
            print(f"  {k:2d} {rate:5d} {c:2d} {bpp:12.4f} {e:9.5f} {dt:7.1f}s")
            rows.append({"k": k, "rung": rate, "completion": c,
                         "payload_bpp": bpp, "rel_err": e, "encode_s": dt})

    base = next(r for r in rows if r["k"] == 2)
    print(f"\n  k=2 baseline: {base['rel_err']:.5f} at {base['payload_bpp']:.4f} bpp")
    for r in rows:
        if r["k"] == 3:
            print(f"  k=3 rung {r['rung']} c={r['completion']}: "
                  f"{r['rel_err']/base['rel_err']:.3f}x error at "
                  f"{r['payload_bpp']/base['payload_bpp']:.3f}x rate"
                  f"{'   <-- wins on both' if r['rel_err'] < base['rel_err'] and r['payload_bpp'] <= base['payload_bpp'] else ''}")
    json.dump({"tensor": a.tensor, "shape": list(w.shape), "rows": rows},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
