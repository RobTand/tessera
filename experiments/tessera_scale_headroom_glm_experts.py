"""Ask the objective where the group scale should land, instead of a rule.

``_pack_scales`` sets a group's scale so its ``amax`` lands exactly on the top
of the payload grid.  That guarantees nothing clips.  It does **not** minimise
error, and the two are different questions: clipping the extremes buys finer
resolution for everything else, and whether that trade pays is a property of
the weight distribution and the grid, not something a rule can know.
Principle 2 -- no heuristics when an explicit exists.

This is the Tessera analogue of what JSO does for NVFP4, and JSO is the reason
PrismaQuant's NVFP4 render beats its own RTN by 1.258x where pure error
feedback (``compensation_block_and_coder.py``) buys only 1.076x.  It is
**encoder-side only**: whatever scale the encoder picks is written into the
segment-2b bytes, so any headroom is already a legal artifact and no wire,
manifest or decoder change is involved.  ``headroom=1.0`` reproduces every
existing byte exactly.

A global multiplier is the crude form -- the full search is per-group and
joint, because a group's scale spans 16 columns of one row while the trellis
runs down columns.  Crude is the point: if the crude form moves the number,
the rule is leaving quality on the table and the joint search is worth
building.  If it does not, the rule is already near-optimal and this lever is
closed.

Six GLM routed-expert projections, held-out eval half, same harness as
everything else in this series.
"""
import json
import statistics as st
import sys
from fractions import Fraction

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

CC = ConvCode(memory=6)
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
OUT = "/home/rob/tessera/experiments/results/scale_headroom_glm_experts.json"
HEADROOMS = (0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.1, 1.25)


def main():
    tensors = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2, partition="coset")
    arms = {}
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
            cols = w.shape[1]
            rates = bresenham_rate_schedule(Fraction(grid.rate_cap), cols,
                                            cap=grid.rate_cap)
            forests = {r: build_forest(r, grid=grid) for r in sorted(set(rates))}
            line = []
            for headroom in HEADROOMS:
                unit = encode_unit(wf, forests, rates, CC,
                                   rotation=RotationState.NONE,
                                   with_diagonals=False, completion=0,
                                   group=32, half=16, scale_headroom=headroom)
                recon = reconstruct_unit(unit, forests, CC)
                rel = float(((x @ recon.T - ref).norm() / den))
                arms.setdefault(headroom, []).append(rel)
                line.append(f"{headroom}:{rel:.5f}")
            print(f"{layer:>3} {proj:<10} " + "  ".join(line), flush=True)
            del w, wf, ref
            torch.cuda.empty_cache()
        del x, xa
        torch.cuda.empty_cache()

    means = {h: st.mean(v) for h, v in arms.items()}
    base = means[1.0]
    print(f"\n{'headroom':>9} {'rel_err':>9} {'vs rule':>9}")
    for headroom in HEADROOMS:
        mark = "  <- amax rule" if headroom == 1.0 else ""
        print(f"{headroom:>9} {means[headroom]:>9.5f} "
              f"{base / means[headroom]:>8.3f}x{mark}")
    best = min(means, key=means.get)
    print(f"\nbest headroom {best}: {means[best]:.5f}  ({base / means[best]:.3f}x "
          f"over the amax rule, at identical size)")
    json.dump(dict(means={str(k): v for k, v in means.items()},
                   raw={str(k): v for k, v in arms.items()}),
              open(OUT, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
