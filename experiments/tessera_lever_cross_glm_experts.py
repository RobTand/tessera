"""Every cheap lever Tessera already owns, crossed, against EXL3 at the same bpw.

The head-to-head ran Tessera with **no rotation, no diagonals and no
activations** -- three levers off -- against EXL3 with all three of its own on.
Two of the three are already implemented here and cost almost nothing:

* ``with_diagonals`` is segment 2a, and it is EXL3's ``suh``/``svh`` by
  construction (``diagonals.py`` cites ``quantize.py`` for it).  It costs
  ``16*(rows+cols)/(rows*cols)``, which on these shapes is **0.0117 bpp** --
  so **Tessera K2 + diagonals is 4.0117 bpp, EXL3 K=4's rate to four decimals.**
  That is the matched-size comparison with no rounding in it.
* ``R_IN_ONLY`` is the input-side blockwise Hadamard.  Tessera cannot offer the
  output side: S7 makes two-sided rotation a weight-space measurement state,
  not a serving branch, because its output basis needs an ``R_out^T`` inverse
  through every consumer.  So this arm is structurally half of EXL3's
  incoherence processing and is expected to under-deliver.
* LDLQ is ``compensate.py``, at the block ``compensation_block_and_coder.py``
  picks.

Crossed, because they are not independent: the diagonals remove the rank-1
magnitude field a per-32 scale cannot see, the rotation spreads what is left,
and compensation only helps to the extent the residual is still correlated.

Every arm is scored on the disjoint eval half; the Hessian comes from the fit
half only.  The bpp column is the accountant's, not a formula -- an arm that
cannot state its own size is not a comparison.
"""
import itertools
import json
import statistics as st
import sys
from fractions import Fraction

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.diagonals import Diagonals
from tessera.compensate import block_ldl, compensated_targets, regularize_hessian
from tessera.decode import reconstruct_unit
from tessera.diagonals import apply_rotation, diagonal_bits, fit_diagonals
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

CC = ConvCode(memory=6)
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
OUT = "/home/rob/tessera/experiments/results/lever_cross_glm_experts.json"
BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 32
EXL3_K4 = 0.05653


def activations(layer):
    blob = torch.load(
        f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
        map_location="cpu", weights_only=False)
    x = blob["inputs"].float().cuda()
    half = x.shape[0] // 2
    return x[:half].contiguous(), x[half:].contiguous()


def main():
    tensors = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2, partition="coset")
    arms, shape = {}, None
    levers = list(itertools.product((False, True), (False, True), (False, True)))
    for layer in (5, 20, 42):
        x_fit, x = activations(layer)
        H = regularize_hessian((x_fit.T @ x_fit).double().float(),
                               count=x_fit.shape[0])
        L = block_ldl(H.clone(), BLOCK)
        for proj in ("gate_proj", "up_proj"):
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{MODEL}/{tensors[name]}", framework="pt") as handle:
                w = handle.get_tensor(name).cuda()
            wf = w.float()
            shape = list(w.shape)
            ref = x @ wf.T
            den = ref.norm()
            cols = w.shape[1]
            rates = bresenham_rate_schedule(Fraction(grid.rate_cap), cols,
                                            cap=grid.rate_cap)
            forests = {r: build_forest(r, grid=grid) for r in sorted(set(rates))}
            for diag, rot, ldlq in levers:
                rotation = RotationState.R_IN_ONLY if rot else RotationState.NONE
                # Fit the rank-1 diagonals ONCE, on the whole rotated matrix.
                # ``sv`` is per output row and a row spans every column, so
                # letting each compensated slice fit its own would give the
                # sliced arms a fit no whole-matrix encode could reproduce --
                # optimistic, and not the artifact.  ``su`` is per column and
                # slices exactly.
                whole_fit = None
                if diag:
                    rotated, _ = apply_rotation(wf, rotation)
                    whole_fit = fit_diagonals(rotated)

                def run(target, start, stop):
                    fit = whole_fit and Diagonals(sv=whole_fit.sv,
                                                  su=whole_fit.su[start:stop])
                    unit = encode_unit(target, forests, rates[start:stop], CC,
                                       rotation=rotation, diagonals=fit,
                                       completion=0, group=32, half=16)
                    return reconstruct_unit(unit, forests, CC)

                if ldlq:
                    target, recon = compensated_targets(wf, L, run, block=BLOCK)
                    if not torch.equal(run(target, 0, cols), recon):
                        raise SystemExit(
                            f"{name}: compensated arm d={diag} r={rot} does not "
                            "re-encode from its own target -- the slice is not "
                            "the span and the arm is a confound")
                else:
                    recon = run(wf, 0, cols)
                tag = ("d" if diag else "-") + ("r" if rot else "-") + \
                      ("q" if ldlq else "-")
                arms.setdefault(tag, []).append(float((x @ recon.T - ref).norm() / den))
            print(f"{layer:>3} {proj:<10} done", flush=True)
            del w, wf, ref
            torch.cuda.empty_cache()
        del H, L, x, x_fit
        torch.cuda.empty_cache()

    rows, cols = shape
    extra = diagonal_bits(rows, cols) / (rows * cols)
    print(f"\nblock {BLOCK}   d=diagonals  r=R_IN_ONLY rotation  q=LDLQ")
    print(f"{'arm':>5} {'bpp':>8} {'rel_err':>9} {'vs plain':>9} {'vs EXL3':>9}")
    means = {k: st.mean(v) for k, v in arms.items()}
    base = means["---"]
    for tag in sorted(means, key=lambda t: means[t]):
        bpp = 4.0 + (extra if tag[0] == "d" else 0.0)
        print(f"{tag:>5} {bpp:>8.4f} {means[tag]:>9.5f} {base / means[tag]:>8.3f}x "
              f"{means[tag] / EXL3_K4:>8.3f}x")
    print(f"\nEXL3 K=4 @ {4 + extra:.4f} bpw: {EXL3_K4:.5f}")
    json.dump(dict(means=means, block=BLOCK, shape=shape, diag_bpp=extra),
              open(OUT, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
