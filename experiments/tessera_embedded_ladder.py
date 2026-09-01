"""Is Tessera's rate axis *embedded* -- can one encode serve every lower rate?

The wire says yes: ``planes.CANONICAL_PLANE_ORDER`` is "wire order, which is
also the truncation order", and ``build_forest``'s nesting guarantee is that
"truncating completion bits lands on an ancestor, which is a legal partial
map".  ``decode.reconstruct_unit`` takes a ``completion`` argument that can
only truncate, never reach past what was written.

So truncation is *legal*.  This asks what it is *worth*.  The trellis body is
Viterbi'd against ``_descendant_values(forest, completion)`` -- the anchor path
is chosen assuming the completion bits will be there.  Cut them and you decode
an optimal-for-deep path at a shallow level.  The question is how much that
costs against simply encoding at the shallow rate to begin with.

Two ladders to the same bpp, measured on a real GLM-5.3-Flash routed expert:

  * TRUNCATED -- encode once at rung R with full completion, decode at c<depth.
  * NATIVE    -- encode at (R, c) directly, one encode per point.

Both report payload bpp = (R + c) / arity.  Scale planes are excluded from the
axis because they are identical across the comparison and would only dilute it.
"""
import argparse, json, time
import torch
from safetensors import safe_open
from tessera.alphabet import (E2M1_GRID, E4M3_GRID, build_forest, tuple_grid,
                              completion_capacity)
from tessera.encode import encode_unit
from tessera.decode import reconstruct_unit
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"


def rel_err(a, b):
    return (torch.linalg.norm((a - b).float()) / torch.linalg.norm(a.float())).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="model.language_model.layers.10.mlp.experts.0.gate_proj.weight")
    ap.add_argument("--cols", type=int, default=512, help="column slice, to keep this cheap next to the fleet")
    ap.add_argument("--grid", default="E2M1x2", choices=["E2M1x2", "E4M3"],
                    help="E2M1x2 is the 4-bit family, E4M3 the 8-bit one")
    ap.add_argument("--out", default="experiments/results/tessera_embedded_ladder.json")
    a = ap.parse_args()

    shard = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"][a.tensor]
    with safe_open(f"{SRC}/{shard}", framework="pt") as f:
        w = f.get_tensor(a.tensor)
    w = w[:, : a.cols].contiguous().cuda()
    rows, cols = w.shape
    print(f"{a.tensor}  slice {rows}x{cols}  {w.dtype}")

    grid = tuple_grid(E2M1_GRID, 2) if a.grid == "E2M1x2" else E4M3_GRID
    cap, arity = grid.rate_cap, grid.arity
    print(f"grid {grid.name}: size {grid.size} cap {cap} arity {arity} "
          f"-> payload ceiling {cap / arity:.2f} bpp")
    cc = ConvCode(memory=6)
    rows_out = []

    for rate in range(1, cap + 1):
        depth = completion_capacity(rate, cap)
        forests = {rate: build_forest(rate, grid=grid)}
        rates = (rate,) * cols

        # One encode at full depth; every shallower level is a truncated decode.
        t0 = time.time()
        deep = encode_unit(w, forests, rates, code=cc,
                           rotation=RotationState.NONE, completion=depth)
        t_deep = time.time() - t0
        for c in range(depth + 1):
            rec = reconstruct_unit(deep, forests, code=cc, completion=c)
            rows_out.append({
                "ladder": "truncated", "rate": rate, "completion": c,
                "bpp": (rate + c) / arity, "rel_err": rel_err(w, rec),
                "encodes": 1, "encode_s": round(t_deep, 3),
            })

        # And the honest baseline: encode at each depth directly.
        for c in range(depth + 1):
            t0 = time.time()
            nat = encode_unit(w, forests, rates, code=cc,
                              rotation=RotationState.NONE, completion=c)
            t_nat = time.time() - t0
            rec = reconstruct_unit(nat, forests, code=cc)
            rows_out.append({
                "ladder": "native", "rate": rate, "completion": c,
                "bpp": (rate + c) / arity, "rel_err": rel_err(w, rec),
                "encodes": 1, "encode_s": round(t_nat, 3),
            })
        print(f"  rung {rate}: depth {depth}, deep encode {t_deep:.1f}s")

    out = {"tensor": a.tensor, "shape": [rows, cols], "grid": grid.name,
           "rate_cap": cap, "arity": arity, "rows": rows_out}
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    # The comparison that answers the question.
    print(f"\n{'bpp':>6} {'rung':>5} {'native':>10} {'trunc@rung1':>12} {'ratio':>7}")
    best_nat = {}
    for r in rows_out:
        if r["ladder"] == "native":
            k = r["bpp"]
            if k not in best_nat or r["rel_err"] < best_nat[k]["rel_err"]:
                best_nat[k] = r
    t1 = {r["bpp"]: r for r in rows_out if r["ladder"] == "truncated" and r["rate"] == 1}
    for bpp in sorted(best_nat):
        n = best_nat[bpp]
        t = t1.get(bpp)
        if t:
            print(f"{bpp:6.2f} {n['rate']:5d} {n['rel_err']:10.5f} {t['rel_err']:12.5f} "
                  f"{t['rel_err']/n['rel_err']:7.3f}x")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
