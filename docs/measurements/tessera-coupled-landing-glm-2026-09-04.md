# The coupled landing fails the GLM gate: it improves the objective it minimises on all six experts and loses held-out output error on all six (2026-09-04)

**Claim (measured; a screen, and a negative one).** #105 promoted nothing and
said so: the coupled LUT landing (`refit_coupled_landing`, issue #50) was
measured on six dense Qwen3-0.6B units, where it takes the trailing full-H
Gauss-Seidel arm from 0.8851x to **0.8037x** against the served `h^1.0`
control, and its receipt named the two gates still owed --- *"the GLM
six-expert geomean and a same-session serve."* This is the first of those two.
**It fails.** On the six GLM-5.3-Flash experts the `h^1.0` default was chosen
on, at the same E2M1x2 q896 cap wire, LDLQ 1.0/32, `scale_refit=4`, the
trailing arm with the coupled landing is **1.0160x** against the same control
--- above the 1x gate `tessera.control.assert_plane_promotion` reads, and
worse than the 1.0104x of the same arm without it.

| arm | GLM `out` | vs A | GLM `hfit` | vs A | Qwen `out` vs A (#105) |
|---|---|---|---|---|---|
| A `h^1.0 x4` (the served default) | 0.07393 | 1.0000 | 0.05502 | 1.0000 | 1.0000 |
| B-Jac `T R_h T R_h T R_h T R_H` | 0.07393 | 0.9999 | 0.05463 | 0.9929 | 0.9191 |
| B-GS `T R_h T R_h T R_h T R_H(GS)` | 0.07470 | 1.0104 | 0.05382 | 0.9783 | 0.8851 |
| **B-GS+CL** (+ coupled landing, trailing pass) | **0.07511** | **1.0160** | **0.05297** | **0.9627** | **0.8037** |
| C-Jac `T R_H x4` | 0.07827 | 1.0587 | 0.05684 | 1.0332 | 0.9868 |
| C-GS `T R_H x4 (GS)` | 0.07682 | 1.0391 | 0.05457 | 0.9920 | 0.9635 |
| C-GS+CL (+ coupled landing, every pass) | 0.07683 | 1.0392 | 0.05372 | 0.9765 | 0.9239 |

## The lever's own worth, holding schedule and metric fixed

B-GS -> B-GS+CL is the single-lever pair: same objective, same optimiser, same
pass count, the landing swapped. On GLM it is **1.0055x on `out` and 0.9841x
on `hfit`**, and it is unanimous in both directions --- six units of six worse
held-out, six of six better on the quadratic the refit actually minimises.

```
unit             A out    B-GS out  B-GS+CL   lever    | A hfit   B-GS hfit B-GS+CL   lever
L5.gate_proj     0.07871  0.07954   0.07987   1.0042   | 0.06247  0.06109   0.06030   0.9871
L5.up_proj       0.08146  0.08232   0.08266   1.0041   | 0.06456  0.06314   0.06232   0.9870
L20.gate_proj    0.07647  0.07724   0.07778   1.0071   | 0.05700  0.05582   0.05499   0.9851
L20.up_proj      0.08055  0.08143   0.08195   1.0064   | 0.06038  0.05915   0.05826   0.9850
L42.gate_proj    0.05649  0.05707   0.05740   1.0058   | 0.03825  0.03740   0.03667   0.9806
L42.up_proj      0.07318  0.07393   0.07434   1.0055   | 0.05222  0.05104   0.05001   0.9798
geomean                                       1.0055   |                              0.9841
```

On Qwen the same lever is **0.9081x** on `out` and 0.8719x on `hfit`. So the
mechanism is not broken and its sign has not flipped on its own objective: it
minimises the coupled quadratic better on both populations. What differs is
how much (12.8% on Qwen against 1.6% here) and, decisively, whether any of it
survives off the fit rows.

## What the split means, and why it is the reading that matters

**The two columns are the same functional on two different Hessians**, which
is what makes the split readable at all. `hfit` is
`sqrt(tr(E H E^T) / tr(W H W^T))` with `H` the fit-row Hessian --- the
quantity every refit here is monotone in, and the one the coupled landing is
an exact coordinate-descent minimiser of. `out` is
`||X_ev E^T||_F / ||X_ev W^T||_F`, and since `||X_ev E^T||_F^2 =
tr(E X_ev^T X_ev E^T)`, that is the *identical* expression with `X_ev^T X_ev`
in place of `H` (the row-count normalisation cancels in the ratio). So the two
numbers differ in one thing and one thing only: **which rows the Hessian was
built from.** There is no second reading available --- not a different norm,
not a different weighting, not a rendering difference. A mechanism that moves
them in opposite directions is telling you about `H`, and about nothing else.

On Qwen the two moved together, and #105's receipt said so in as many words: *"Both `out` (held out) and `hfit` improve
together, so there is no screen-inverts-on-held-out signature here."* That
sentence was true of Qwen and is false of GLM. Here they part company, 6/6 in
each direction: **the coupled landing is overfitting the fit-row Hessian.**

That is a coherent mechanism rather than noise --- and note there is no
run-to-run noise to appeal to: every arm is deterministic given code and
inputs, the drift control's floor is 0.0000%, and the sign is unanimous over
six units in both columns.

**The split is not the coupled landing's; it is the full-H objective's.** Read
the B family as a ladder in how hard each arm minimises `E H E^T`:

| arm | GLM `hfit` | GLM `out` |
|---|---|---|
| B-Jac --- one Jacobi step | 0.9929x | 0.9999x |
| B-GS --- Gauss--Seidel sweep | 0.9783x | 1.0104x |
| B-GS+CL --- and the coupled landing | 0.9627x | 1.0160x |

Monotone in both columns, in opposite directions: each increment of minimising
power buys `hfit` and pays for it in `out`. The coupled landing is the
strongest minimiser on the rung and therefore the worst held-out --- it is not
a defect of its own, it is the end of this ladder. The C family ends flat
rather than worse (`C-GS -> C-GS+CL` is 0.9845x `hfit` for 1.00005x `out`), so
the honest general statement is: **on GLM no arm here converts a full-H gain
into an `out` gain.**

The reason is the fit-row Hessian, not the landing rule. `H` is
`x_fit^T x_fit / n_fit` over the **fit** rows only --- the 1024-row tail `out`
is scored on is excluded from it, which is what makes `out` held-out at all ---
and that sample is small against the dimension it has to condition. A GLM
expert's rows are 4096-dimensional with `n_fit = 8192 - 1024 = 7168`, so
`p/n = 0.57`. The six Qwen units are 1024- or 3072-dimensional over 16384 fit
tokens: `p/n` from 0.06 to 0.19, three to nine times better conditioned. At
`p/n ~ 0.57` a sample Hessian's small eigenvalues are badly underestimated, so
an exact minimiser of `E H E^T` is free to spend its budget along directions
the held-out rows still have energy in. Nearest-in-linear landing is a weak
per-block rule that never looks past one block's own target: it cannot exploit
that fine structure and so cannot be hurt by it. The coupled landing looks at
the full quadratic and takes whatever `H` offers, sampling error included. That
an expert's rows are only the tokens the router sent it makes the fit/held-out
split harder still, on top of the aspect ratio.

**Do not read this lever on `plain`.** Weight space worsens on GLM as it does
on Qwen --- lever 1.0101x, composite 1.0261x --- and that is the mechanism
working, not the failure. The failure is `out`.

## The arms are on the wire and the pair is exact

All eight arms emit the **same `len(blob)` on every unit** (4195582--4195585
bytes, one distinct value per unit), and every arm carries
`sink_vs_wire_bit_identical=True` with `sink_vs_wire_rel=0.0`: the scored
reconstruction *is* `stock_dequant(materialize_stock(...))`. Same sixteen E4M3
table bytes, same four-bit indices, same byte count --- zero-byte on GLM as on
Qwen.

The B pair is exact in the strongest available sense: **`codes_sha256` is
equal between B-GS and B-GS+CL on all six units** while the reconstruction
differs, because the coupled landing runs in the trailing refit and no trellis
pass follows it. Nothing but the scale plane's assignment can be responsible
for the 1.0055x. The C pair's codes *do* differ (`codes_sha256` unequal on all
six), which is the effect #50's frozen-codes oracle structurally could not
see. That arm says the same thing more starkly: C-GS -> C-GS+CL is
**1.00005x** on `out` -- exactly nothing, three units either side of zero --
while `hfit` improves 0.9845x on all six. Letting the re-assignment feed the
next trellis pass neither rescues the held-out number nor costs more; the
objective just gets minimised better for free, and free is what it is worth.

The run's own drift control passes on every unit: arm A run first and again
last in one process is the **identical reconstruction**, `out` and `hfit`
+0.0000%.

## Gate verdicts, verbatim

`experiments/refit_trailing_pair_gate.py` over the seven arms, Qwen as the
screen and GLM as the cross-check, at the wire landing:

```
== B-GS+CL T R_h T R_h T R_h T R_H(GS,CL)   (trailing full-H, sweep, coupled landing)
   Qwen six-unit out geomean 0.8037x, wins 6/6   per-unit 0.8698 0.8770 0.9910 0.9131 0.3974 0.9826
   GLM six-expert out geomean 1.0160x against the 1x gate: FAILS
   assert_plane_promotion: PromotionRefusedError: tessera#75 the trailing-refit
   objective: GLM six-expert 1.016x is above the 1x gate -- the cross-check the
   coordinator's gate requires, and no screen overrules it

== C-GS+CL T R_H T R_H T R_H T R_H(GS,CL)   (full-H every pass, sweep, coupled landing)
   Qwen six-unit out geomean 0.9239x, wins 5/6   per-unit 0.8912 0.9033 0.9997 0.9321 0.8265 1.0032
   GLM six-expert out geomean 1.0392x against the 1x gate: FAILS
```

`B-Jac` remains the only arm in this family that clears GLM (0.9999x), and it
is refused for the other reason --- nothing in it is served. Nothing changes
about that here.

## What this settles for #105, and what it does not

- **`refit_coupled_landing` stays opt-in and stays research.** Its Qwen
  numbers are not withdrawn --- 0.9081x is real on that population, and the
  0.8037x composite it sits inside was predicted to 0.4% by a frozen-codes
  oracle written separately --- but a Qwen-only win is precisely the
  shape every prior Tessera lever had that inverted, and this one inverts on
  the first cross-population read. No default moves; the flag is off, and with
  it off the encode is byte for byte what it was (#105 proved that by
  re-encoding, not by reading a diff).
- **It does not clear the way to a serve.** The served A/B #105 owes is still
  owed and still blocked: `ActivationSource` gained
  `refit_objective_trailing` and `refit_gauss_seidel` in #103, but it has
  **no `refit_coupled_landing` field**, so no exportable artifact can express
  this arm. That gap is now cheaper to leave open than to close: the arm that
  would be exported fails its cross-population gate.
- **The interesting question it opens is not this lever.** A mechanism that
  improves the fit-row quadratic unanimously and loses held-out error
  unanimously is a statement about `H`, not about the landing. The same
  reading applies to any future exact minimiser of the same objective --- a
  better solver of a fit-row quadratic buys nothing when the quadratic is the
  thing that is wrong. There is a concrete
  asymmetry to start from, visible in this script's own schedule: LDLQ is
  given `regularize_hessian(H, sigma_reg=1.0)` at `refit_trailing_pair.py:261`,
  while the refit and the coupled landing are handed the **raw** `H`. One
  consumer of this Hessian is damped and the other is not, and it is the
  undamped one that overfits. Damping the refit's `H`, or scoring the refit on
  held-out rows, is where that goes.

## Provenance

`experiments/refit_trailing_pair.py --population glm`, one process, eight arms
including the first/last drift control, six GLM-5.3-Flash expert-0 tensors
(`gate_proj`/`up_proj` on layers 5/20/42), Hessian from the fit rows and the
score on the held-out 1024-row tail, fp32. Source
`/mnt/shared/models/GLM-5.3-Flash-BF16`, activations
`/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act` --- both
read-only, the same inputs `ad26a21`'s GLM leg used. Wire: E2M1x2 `q256=896`,
body TCQ, plane LUT16, `LUT_LANDING_WIRE`.

Data: `experiments/results/refit_trailing_pair_glm_cl.json`, gate verdicts
`experiments/results/refit_trailing_pair_gate_cl.json`. The Qwen column is
`experiments/results/refit_trailing_pair_qwen_cl.json` (#105, unchanged here
and not re-derived).

Box: sparky (GB10), through the PrismaBuild pool. This is a **quality**
measurement, not a timing one: the arms are deterministic given code and
inputs, the run's own drift control proves it per unit, and no number in this
receipt is read off the `s` column. The box was shared with other pool actions
and that is fine for exactly that reason.
