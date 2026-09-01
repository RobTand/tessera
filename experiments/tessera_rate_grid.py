"""Tessera's rate is two-dimensional, and until today the artifact hid one axis.

A column at body rate ``R`` may spend up to ``cap - R`` further bits choosing
among the descendants its trellis subset reaches.  ``encode_unit`` has always
honoured a ``completion`` limit -- ``level = min(completion, depth)`` -- but two
things downstream did not:

  * ``unit_artifact`` sized the COMPLETION plane from ``completion_capacity(R)``
    and packed it at that width, so a unit encoded at ``completion=0`` wrote a
    full-width plane of zeros.  ``sum(R) + sum(cap - R) = columns * cap`` is
    constant, which is exactly why every rung of a family weighed the same:
    the body shrank and the all-zero completion plane grew to match.
  * ``encode_linear`` hardcoded ``completion=0``, so the exporter could not
    reach the axis at all.

Both are fixed.  This sweep is the receipt: for each family it walks the real
grid ``(q256, completion)`` and reports **exact artifact bytes** from the
accountant that writes them, against output error on real activations.  If the
size column is not monotone in the rate, the fix did not take.

Sizes come from ``encode_linear(...).exact_bytes`` -- the accountant, on the
real bytes, with its own round-trip verification -- never from a formula.
"""
import json
import statistics as st
import sys

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.export import _plan_for, encode_linear
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

CC = ConvCode(memory=6)
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
OUT = "/home/rob/tessera/experiments/results/tessera_rate_grid.json"
LAYERS = (5, 20, 42)
PROJ = ("gate_proj", "up_proj")

# q256 is the per-POSITION body rate x256; the per-CODE rate is q256*arity/256
# and must not exceed the grid's cap.  Both families are swept over the same
# per-position band so the two ladders are directly comparable.
# The whole realisable band of each family, in q256.  The floor is the rate
# the shaped domain admits (R >= 1, so q256 >= 256/arity) and the ceiling is
# the grid's cap; everything between is legal now that the sub-cap partition
# is balanced.  Step 64 = a quarter bit per code.
FAMILIES = {
    "E2M1_K1": (E2M1_GRID, tuple(range(256, 769, 64)), (0, 1, 2, None)),
    "E2M1_K2": (tuple_grid(E2M1_GRID, 2, partition="coset"),
                tuple(range(128, 897, 64)), (0, 1, 2, None)),
    # The 8-bit ladder, on the same weights and the same accountant.  Only
    # c=0 and full: the two E2M1 families already answer the completion
    # question, and re-asking it over seven more rungs buys nothing.
    "E4M3_K1": (E4M3_GRID, tuple(range(256, 1793, 128)), (0, None)),
}


def main():
    index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    arms, sizes = {}, {}
    for layer in LAYERS:
        blob = torch.load(
            f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        x = xa[xa.shape[0] // 2:].contiguous()          # held out from any fit
        for proj in PROJ:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{MODEL}/{index[name]}", framework="pt") as handle:
                w = handle.get_tensor(name).cuda()
            wf = w.float()
            ref = x @ wf.T
            den = ref.norm()
            rows, cols = w.shape
            for fam, (grid, rungs, completions) in FAMILIES.items():
                for q256 in rungs:
                    rates, forests = _plan_for(grid, q256, cols)
                    for comp in completions:
                        tag = f"{fam}/q{q256}/c{'F' if comp is None else comp}"
                        unit = encode_unit(wf, forests, rates, CC,
                                           rotation=RotationState.NONE,
                                           completion=comp, group=32, half=16)
                        recon = reconstruct_unit(unit, forests, CC)
                        arms.setdefault(tag, []).append(
                            float((x @ recon.T - ref).norm() / den))
                        if tag not in sizes:
                            built = encode_linear(
                                w, grid=grid, q256=q256, name=tag, code=CC,
                                rotation=RotationState.NONE,
                                with_diagonals=False, completion=comp,
                                verify=True)
                            sizes[tag] = built.exact_bytes * 8 / (rows * cols)
            print(f"{layer:>3} {proj:<10} done", flush=True)
            del w, wf, ref
            torch.cuda.empty_cache()
        del x, xa
        torch.cuda.empty_cache()

    means = {k: st.mean(v) for k, v in arms.items()}
    for fam, (grid, rungs, completions) in FAMILIES.items():
        print(f"\n== {fam}  cap={grid.rate_cap} arity={grid.arity}  "
              f"payload ceiling {grid.rate_cap / grid.arity} b/wt")
        print(f"{'rung':>8} " + "".join(
            f"{('c=' + ('F' if c is None else str(c))):>19}" for c in completions))
        for q256 in rungs:
            cells = []
            for comp in completions:
                tag = f"{fam}/q{q256}/c{'F' if comp is None else comp}"
                cells.append(f"{sizes[tag]:8.4f}bpp {means[tag]:8.5f}"
                             if tag in sizes else " " * 19)
            print(f"{q256:>8} " + "".join(cells))
    json.dump(dict(means=means, sizes=sizes), open(OUT, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
