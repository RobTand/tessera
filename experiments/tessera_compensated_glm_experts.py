"""Does an activation-aware Tessera encoder close the 1.72x gap to EXL3?

`docs/measurements/exl3-head-to-head-2026-09-01.md` measured EXL3 at 4.0117 bpw
against Tessera at 4.0000 on six GLM routed-expert projections and found EXL3
**1.72x** better (0.05653 vs 0.09738).  It also bounded the split: PrismaQuant's
NVFP4 render responds to activation-awareness by **1.258x** (RTN 0.08294 ->
GPTQ+JSO 0.06595), which credits calibration with ln(1.258)/ln(1.72) = 42% of
the gap and leaves <= 1.37x for the coder *if a trellis responds at least as
well as a scalar quantiser does*.  Nothing measured Tessera's response, because
Tessera had no activation-aware encoder.  This measures it.

**Acceptance, fixed before the run:** the baseline arm at **0.09738** moving to
**~0.077** is the NVFP4-like response, and confirms calibration as the bulk of
the gap.  Materially less movement is itself a finding -- the trellis responds
*worse* than a scalar format, the coder question reopens, and the rate-ceiling
work becomes the next lever instead.

**The mechanism is error feedback, not importance weighting.**  Diagonal-Hessian
weighting is a provable no-op on this encoder (see ``tessera/compensate.py``):
every position in a column shares one weight and Viterbi's argmins are
per-column, so anchors, completion bits, scale bytes and release order come out
bit-identical.  A probe built on weighting would have returned a null that meant
nothing.  What is built here is LDLQ -- EXL3's own scheduling, block-LDL of the
regularised Hessian, input-feature blocks quantised last-to-first with each
block's residual pushed into the ones still to come.

Same six tensors, same fit/eval token split, same disjoint eval half as both
prior harnesses, so the numbers are directly comparable to the table in the
head-to-head doc.  Every arm is scored on ``X_eval``; the Hessian is built from
``X_fit`` only.  Held out.

Scope carried over unchanged: gate_proj/up_proj only (the probe caches one
input per packed-expert entry at hidden dim, so down_proj is unpriced), expert
0 of layers 5/20/42, functional error on cached activations.  Principle 3: this
is a screen and selects nothing.
"""
import json
import statistics as st
import sys
import time
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

CC = ConvCode(memory=6)
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
OUT = "/home/rob/tessera/experiments/results/compensated_glm_experts.json"
ARITY = 2                     # E2M1_K2: the 4.0000 bpp rung, the only one at size
LDL_BLOCK = 256               # aligned to the 32-col scale group and the 128 rotation block
BASELINE = 0.09738            # the arm this has to move
EXL3 = 0.05653


def activations(layer):
    blob = torch.load(
        f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
        map_location="cpu", weights_only=False)
    x = blob["inputs"].float().cuda()
    half = x.shape[0] // 2
    return x[:half].contiguous(), x[half:].contiguous()


def plan(cols):
    grid = tuple_grid(E2M1_GRID, ARITY, partition="coset") if ARITY > 1 else E2M1_GRID
    rates = bresenham_rate_schedule(Fraction(grid.rate_cap), cols, cap=grid.rate_cap)
    return rates, {r: build_forest(r, grid=grid) for r in sorted(set(rates))}


def encoder(rates, forests, rotation):
    def run(w, start, stop):
        unit = encode_unit(w, forests, rates[start:stop], CC, rotation=rotation,
                           with_diagonals=False, completion=0, group=32, half=16)
        return reconstruct_unit(unit, forests, CC)
    return run


def main():
    tensors = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    rows, arms = [], {}
    for layer in (5, 20, 42):
        x_fit, x = activations(layer)
        H = None
        for proj in ("gate_proj", "up_proj"):
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{MODEL}/{tensors[name]}", framework="pt") as handle:
                w = handle.get_tensor(name).cuda()
            wf = w.float()
            ref = x @ wf.T
            den = ref.norm()
            err = lambda recon: float(((x @ recon.T - ref).norm() / den))

            rates, forests = plan(w.shape[1])
            if H is None:
                H = regularize_hessian((x_fit.T @ x_fit).double().float(),
                                       count=x_fit.shape[0])
                L = block_ldl(H.clone(), LDL_BLOCK)

            row = dict(layer=layer, proj=proj, shape=list(w.shape))
            for tag, rotation in (("plain", RotationState.NONE),
                                  ("rot", RotationState.R_IN_ONLY)):
                run = encoder(rates, forests, rotation)
                started = time.time()
                row[tag] = err(run(wf, 0, w.shape[1]))
                row[f"{tag}_s"] = round(time.time() - started, 1)

                started = time.time()
                target, recon = compensated_targets(wf, L, run, block=LDL_BLOCK)
                # Render identity: the stitched reconstruction must equal a
                # whole-matrix encode of the returned target, or the arm was
                # scored on something the exporter would not build.
                whole = run(target, 0, w.shape[1])
                if not torch.equal(whole, recon):
                    raise SystemExit(
                        f"{name}/{tag}: the compensated target does not re-encode to "
                        "the stitched reconstruction -- the slice alignment is wrong "
                        "and this arm is a confound, not a measurement"
                    )
                row[f"{tag}_ldlq"] = err(recon)
                row[f"{tag}_ldlq_s"] = round(time.time() - started, 1)

            rows.append(row)
            for key, value in row.items():
                if isinstance(value, float):
                    arms.setdefault(key, []).append(value)
            print(f"{layer:>3} {proj:<10} " + "  ".join(
                f"{k} {row[k]:.5f}" for k in
                ("plain", "plain_ldlq", "rot", "rot_ldlq")), flush=True)
            del w, wf, ref
            torch.cuda.empty_cache()
        del H, L, x, x_fit
        torch.cuda.empty_cache()

    print(f"\n{'arm':<12} {'mean rel_err':>12} {'vs baseline':>12} {'vs EXL3':>9}")
    for key in ("plain", "plain_ldlq", "rot", "rot_ldlq"):
        mean = st.mean(arms[key])
        print(f"{key:<12} {mean:>12.5f} {BASELINE / mean:>11.3f}x "
              f"{mean / EXL3:>8.3f}x")
    print(f"\nEXL3 @4.0117 bpw: {EXL3:.5f}   acceptance was plain_ldlq ~0.077")
    json.dump(dict(rows=rows, means={k: st.mean(v) for k, v in arms.items()},
                   baseline=BASELINE, exl3=EXL3, ldl_block=LDL_BLOCK, arity=ARITY),
              open(OUT, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
