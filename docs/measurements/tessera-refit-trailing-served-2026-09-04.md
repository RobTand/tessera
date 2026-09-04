# The trailing refit's objective, served (2026-09-04)

**PENDING — this file is written ahead of its own numbers and must not be
read until the placeholders below are gone.** Every `TBD` is a measurement
that has not returned.

**What this is.** tessera#75's fair pair, taken to the one leg a screen cannot
supply. The measurement half is on master (`experiments/refit_trailing_pair.py`,
merged `9add21d`): at the wire, matched pass count, swapping only the
**trailing** refit's objective to the full `H` is **0.9191x** on six dense
Qwen3-0.6B units (6 of 6) and **0.9999x** on the six GLM-5.3-Flash experts, so
it clears the GLM gate. `tessera.control.assert_plane_promotion` then refused
it on exactly one leg:

> the served KL measures arm None, not the promoted arm 'B-Jac ...' -- a served
> number for a different arm is not evidence for it

This is that leg, and nothing else. **No default moves here.**

## The blocker that stood between the screen and the serve

`ActivationSource.refit_objective_trailing` landed with tessera#103 — the
field, the exported config, the merge guard's `SHARED_ACTIVATION` — but no
exporter could set it. `experiments/export_tessera_serving.py` plumbed
`--refit-metric` alone, so the arm was expressible in a measurement script and
not in a checkpoint. `--refit-metric-trailing` is that flag; unset is the
uniform schedule, byte for byte the encode that was already there.

## The arms

| | A (incumbent) | B-Jac (the arm) |
|---|---|---|
| name | `ldlqH1` | `bjac` |
| inner refits | `h^1.0` x4 | `h^1.0` x3 |
| trailing refit | `h^1.0` | **full `H`, Jacobi** |
| wire | E2M1x2, `q256=896` (the 4-bit TCQ cap), LUT16 plane | same |
| LDLQ | sigma 1.0, block 32 | same |

Everything that is not the trailing refit's objective is held fixed: the same
source (`/home/rob/models/Qwen3-0.6B`), the same Hessian capture, the same
static A4 input scales, the same teacher payload, the same corpus contract,
the same pinned image, eager, one box.

## The matched pair, on the artifacts

TBD — `experiments/results/refit_trailing_bytes.json`.

## Served

TBD — `/mnt/shared/tessera-runs/refit-trailing/kl_bjac.json`.

## The gate, verbatim

TBD — `experiments/results/refit_trailing_pair_gate.json`.

## Scope

TBD.
