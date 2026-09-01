"""The scale plane costs 12.5% of Tessera's budget.  What does buying it back cost?

The head-to-head compared Tessera at 4.0000 bpp against EXL3 at 4.0117 and
found EXL3 1.72x better.  `exl3_rate_sweep.py` then compared them at matched
**payload** bits -- EXL3 K=3 at 3.0 payload is 0.11089, Tessera `E2M1_K1` at
3.0 payload is 0.12666 -- and the gap is only **1.142x**.  The two numbers
differ because the budgets are spent differently:

    Tessera E2M1_K2 @ 4.0000 bpp = 3.5 payload + 0.5000 scale plane
    EXL3    K=4     @ 4.0117 bpw = 4.0 payload + 0.0117 (fp16 suh/svh)

Tessera hands **12.5% of its bits to scale metadata**; EXL3 hands over 0.3%.
Body and completion sum to the grid's cap, so Tessera cannot move the saving
into the payload -- but it can move it off the artifact, and the size win in
the sub-4.5 band is the whole thesis.

`group` and `half` are manifest geometry, already on the wire, already
parameters of `encode_unit` and `encode_linear`.  So this is a sweep over
points that exist today, not a proposal.  The rank-1 diagonals are crossed in
because they are the thing a coarse group scale most obviously needs: they
remove the part of the channel imbalance that factorises, which is exactly what
a per-32 scale was doing by brute force, and they cost 0.0117 bpp instead of
0.375.

**Sizes come from the accountant that writes the bytes** -- `encode_linear`
builds the real artifact and returns `exact_bytes`, and it verifies the
round-trip while it is there.  A cost experiment that charges itself from a
formula is how the bit-trade's gain leg got refuted.
"""
import json
import statistics as st
import sys
from fractions import Fraction

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.diagonals import fit_diagonals
from tessera.encode import encode_unit
from tessera.export import encode_linear
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

CC = ConvCode(memory=6)
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
OUT = "/home/rob/tessera/experiments/results/scale_plane_geometry.json"
GEOMETRIES = ((32, 16), (64, 32), (128, 64), (256, 128))
EXL3 = {3.0117: 0.11089, 4.0117: 0.05653}


def main():
    tensors = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2, partition="coset")
    arms, sizes = {}, {}
    for layer in (5, 20, 42):
        blob = torch.load(
            f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        x = xa[xa.shape[0] // 2:].contiguous()
        for proj in ("gate_proj", "up_proj"):
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{MODEL}/{tensors[name]}", framework="pt") as handle:
                w = handle.get_tensor(name).cuda()
            wf = w.float()
            ref = x @ wf.T
            den = ref.norm()
            rows, cols = w.shape
            rates = bresenham_rate_schedule(Fraction(grid.rate_cap), cols,
                                            cap=grid.rate_cap)
            forests = {r: build_forest(r, grid=grid) for r in sorted(set(rates))}
            for group, half in GEOMETRIES:
                for diag in (False, True):
                    tag = f"g{group}/h{half}" + ("+d" if diag else "")
                    fit = fit_diagonals(wf) if diag else None
                    unit = encode_unit(wf, forests, rates, CC,
                                       rotation=RotationState.NONE,
                                       diagonals=fit, completion=0,
                                       group=group, half=half)
                    recon = reconstruct_unit(unit, forests, CC)
                    arms.setdefault(tag, []).append(
                        float(((x @ recon.T - ref).norm() / den)))
                    if tag not in sizes:
                        # The accountant, on the real bytes, with its own
                        # round-trip verification -- not a formula.
                        built = encode_linear(w, grid=grid, q256=896, name=tag,
                                              code=CC, group=group, half=half,
                                              rotation=RotationState.NONE,
                                              with_diagonals=diag, verify=True)
                        extra = 16 * (rows + cols) if diag else 0
                        sizes[tag] = (built.exact_bytes * 8 + extra) / (rows * cols)
            print(f"{layer:>3} {proj:<10} done", flush=True)
            del w, wf, ref
            torch.cuda.empty_cache()
        del x, xa
        torch.cuda.empty_cache()

    means = {k: st.mean(v) for k, v in arms.items()}
    print(f"\n{'geometry':>14} {'scale bpp':>10} {'total bpp':>10} {'rel_err':>9} "
          f"{'vs g32/h16':>11}")
    base = means["g32/h16"]
    for group, half in GEOMETRIES:
        for diag in (False, True):
            tag = f"g{group}/h{half}" + ("+d" if diag else "")
            scale = 8 / group + 4 / half
            print(f"{tag:>14} {scale:>10.4f} {sizes[tag]:>10.4f} "
                  f"{means[tag]:>9.5f} {base / means[tag]:>10.3f}x")
    print("\nEXL3 for reference: 3.0117 bpw -> 0.11089   4.0117 bpw -> 0.05653")
    json.dump(dict(means=means, sizes=sizes, exl3=EXL3), open(OUT, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
