"""Two controls the compensated arm needs before its number means anything.

``tessera_compensated_glm_experts.py`` measured LDLQ moving Tessera from
0.09738 to 0.08952 -- a **1.088x** response, well short of the 1.258x that
PrismaQuant's NVFP4 render gains from GPTQ+JSO.  Read naively that says "the
trellis responds worse to calibration than a scalar format does", which is the
branch that reopens the coder question and redirects the roadmap.  Two things
have to be excluded first, and both are cheap.

**Control 1: block size.**  Block-LDL leaves the diagonal blocks
uncompensated, so a 256-wide block leaves 256 of 4096 input features
uncorrected within each block -- EXL3 uses **16**.  If the response is a
function of the block and not of the coder, then 1.088x is a statement about
my scheduling, not about Tessera.  The floor is the 32-column scale group: a
group's scale is fit to whatever target it ends up with and cannot be fit to
half of one.

**Control 2: the same machinery on a different coder.**  Comparing my LDLQ
against PrismaQuant's GPTQ+JSO compares two implementations as well as two
coders -- JSO is a scale search the trellis arm does not have.  Running *this*
compensation loop with an NVFP4-RTN encoder isolates the coder: identical
schedule, identical Hessian, identical block, one substitution.  Whatever
NVFP4 gains here is the response the same mechanism produces on a scalar
format, and the ratio of the two gains is the answer.

Same six tensors, same fit/eval split, scored on the disjoint eval half.
"""
import json
import statistics as st
import sys
from fractions import Fraction

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.compensate import block_ldl, compensated_targets, regularize_hessian
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

import prismaquant.format_registry as fr

CC = ConvCode(memory=6)
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
OUT = "/home/rob/tessera/experiments/results/compensation_block_and_coder.json"
BLOCKS = (32, 64, 128, 256, 1024)


def activations(layer):
    blob = torch.load(
        f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
        map_location="cpu", weights_only=False)
    x = blob["inputs"].float().cuda()
    half = x.shape[0] // 2
    return x[:half].contiguous(), x[half:].contiguous()


def main():
    tensors = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    nv = fr.get_format("NVFP4")
    arms = {}
    grid = tuple_grid(E2M1_GRID, 2, partition="coset")
    for layer in (5, 20, 42):
        x_fit, x = activations(layer)
        H = regularize_hessian((x_fit.T @ x_fit).double().float(),
                               count=x_fit.shape[0])
        for proj in ("gate_proj", "up_proj"):
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{MODEL}/{tensors[name]}", framework="pt") as handle:
                w = handle.get_tensor(name).cuda()
            wf = w.float()
            ref = x @ wf.T
            den = ref.norm()
            err = lambda recon: float(((x @ recon.to(wf.dtype).T - ref).norm() / den))
            cols = w.shape[1]
            rates = bresenham_rate_schedule(Fraction(grid.rate_cap), cols,
                                            cap=grid.rate_cap)
            forests = {r: build_forest(r, grid=grid) for r in sorted(set(rates))}

            def tessera(target, start, stop):
                unit = encode_unit(target, forests, rates[start:stop], CC,
                                   rotation=RotationState.NONE, with_diagonals=False,
                                   completion=0, group=32, half=16)
                return reconstruct_unit(unit, forests, CC)

            def nvfp4(target, start, stop):
                return nv.quantize_dequantize(target.to(w.dtype)).float()

            coders = {"tessera": tessera, "nvfp4_rtn": nvfp4}
            for tag, coder in coders.items():
                arms.setdefault(f"{tag}@plain", []).append(err(coder(wf, 0, cols)))
            for block in BLOCKS:
                L = block_ldl(H.clone(), block)
                for tag, coder in coders.items():
                    _, recon = compensated_targets(wf, L, coder, block=block)
                    arms.setdefault(f"{tag}@{block}", []).append(err(recon))
                del L
            print(f"{layer:>3} {proj:<10} done", flush=True)
            del w, wf, ref
            torch.cuda.empty_cache()
        del H, x, x_fit
        torch.cuda.empty_cache()

    means = {k: st.mean(v) for k, v in arms.items()}
    print(f"\n{'block':>7} {'tessera':>9} {'gain':>7}   {'nvfp4_rtn':>10} {'gain':>7}"
          f"   {'gain ratio':>10}")
    base_t, base_n = means["tessera@plain"], means["nvfp4_rtn@plain"]
    print(f"{'none':>7} {base_t:>9.5f} {1.0:>6.3f}x   {base_n:>10.5f} {1.0:>6.3f}x")
    for block in BLOCKS:
        t, n = means[f"tessera@{block}"], means[f"nvfp4_rtn@{block}"]
        print(f"{block:>7} {t:>9.5f} {base_t / t:>6.3f}x   {n:>10.5f} "
              f"{base_n / n:>6.3f}x   {(base_n / n) / (base_t / t):>9.3f}x")
    print("\nPrismaQuant GPTQ+JSO on the same six: 0.06595 (RTN 0.08294, gain 1.258x)")
    print("EXL3 @4.0117 bpw: 0.05653   Tessera @4.0000 uncompensated: 0.09738")
    json.dump(dict(means=means, blocks=list(BLOCKS)), open(OUT, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
