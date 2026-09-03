# The LUT landing's loss is in the assignment, not the table fit; a coupled landing recovers TBD_RECOV of it (2026-09-03)

**Claim, weight space and held-out activation space (measured; a screen and
not a result).** Issue #50 read `tessera-lut-refit-gauss-seidel-2026-09-03`'s
diagnostic sink -- the landing gives back 24-91% of the refit's step -- and
named the cause: `_fit_lut` chooses the sixteen table entries under a
**separable** model, `sum_b A_b (c_b - s*_b)^2`, with every cross-block term
of the metric dropped. The first half is true and was never in doubt: that is
the objective at `encode.py::_lut_cost`, and it is separable by construction.
The second half -- that the separable *table fit* is where the loss lives --
was measured and is **false**. On six dense Qwen3-0.6B units at the E2M1x2
q896 cap wire, over TBD_PASSES refit passes across three arms: the encoder's
accept guard took the re-fitted table on **TBD_NEWTABLE passes** and re-assigned
the table the unit already had on the rest; and an exact coordinate step over
the sixteen entries with **every** cross-block term kept, started from the
coupled assignment, moved **0 entries on every pass**. The table is already
optimal entry by entry under the coupled objective. What is not optimal is the
**assignment** (the issue's (c)): each block lands on the entry nearest its
*own* one-step target `s*_b`, which is the block's conditional minimiser given
its neighbours at *their* targets -- not where its neighbours land. Once they
have landed, TBD_BEFORE of the plane's quadratic is still available to a
block-by-block re-assignment (fixture, pre-change tree), and the sink's
`landed - continuous` was reading that.

**The coupled landing** (`_coupled_landing`, opt-in `refit_coupled_landing`)
re-assigns every block to the table entry minimising the **full** quadratic
given every other block where it now stands -- iterated conditional modes over
blocks, gradient field carried block to block, exact per-block gain, stop when
no block moves or a sweep lowers the cost by less than fp32 resolves on it. It
is separable-exact: under the served `h^1.0` (1-D) metric it is bit-identical
to the landing that was there (test), and under a coupled metric it is a
fourth candidate at or below the third. What it is worth, GS arm, six units:

- **Oracle, at the codes the wire holds** (refit replayed at the end of the
  encode; the pre-registered decision number): `out` geomean landed
  TBD_O_LANDED -> coupled TBD_O_COUPLED (**TBD_O_RATIO**), against a
  pre-registered bar of **1.38%** (#35's promotion margin): TBD_O_VERDICT.
  `hfit` TBD_O_HFIT.
- **In the encoder** (the coupled landing inside the four-pass alternation,
  where it also changes the codes the next trellis pass sees): GS + coupled vs
  GS `out` geomean TBD_E_RATIO; vs the served `h^1.0` control TBD_E_CTL.
- **Of the landing loss itself** (`landed - continuous`, pooled over passes,
  cost-weighted): the coupled landing recovers **TBD_RECOV_GS** on the GS arm
  and TBD_RECOV_JAC on the Jacobi arm; measured against the exact joint
  minimiser (`free`, fp64) instead of the one-step continuous point it is
  TBD_VSFREE_GS / TBD_VSFREE_JAC. The rest of the distance to `free` is the
  sixteen-entry budget: `free-e4m3` (any E4M3 value per block, 8 bits) reaches
  TBD_E4M3, so what the wire cannot have is mostly entry *count*, not grid.

**Nothing here is served.** Six units, one wire, weight-space `hfit` and a
held-out activation-space `out` screen. No KL, no GLM arm. The lever stays
opt-in, exactly as `refit_gauss_seidel` is, and inherits the same two gates
before any default moves: the GLM six-expert geomean and a same-session serve.

Commit: see `git log` for this file. Code: `src/tessera/encode.py`
(`_coupled_landing`, `_refit_scales_lut_metric(coupled_landing=)`,
`encode_unit(refit_coupled_landing=)`), `src/tessera/export.py`
(`encode_linear_planes(refit_coupled_landing=)`). Tests:
`tests/test_ldlq_lut_plane.py` (34 pass, 7 new). Data:
`/mnt/shared/tessera-runs/ldlq-lut/qwen_lut_landing_oracle.json`
(`experiments/lut_landing_oracle.py`) and
`/mnt/shared/tessera-runs/ldlq-lut/qwen_lut_coupled.json`
(`experiments/ldlq_window_sweep.py --coupled-landing`), read together by
`experiments/lut_landing_oracle_report.py`. Box: sparklina, out of `/mnt/shared`.

## What "separable by construction" is true of, and why it does not matter

`_refit_scales_lut_metric` does three approximate things after the step (its
docstring's (a), (b), (c)): the Jacobi/Gauss-Seidel step itself, the table
fit, and the landing. The issue's charge is against (b). Two measurements
close it:

1. **The accept guard.** Every pass scores three planes on the full
   quadratic: the plane the unit had, the old table re-assigned to the stepped
   targets, and the re-fitted table. `_fit_lut`'s output is the third. The
   candidate the guard chose is recorded per pass in the sink. Pooled over
   TBD_PASSES passes (six units x four passes x three arms), the re-fitted
   table won TBD_NEWTABLE_DETAIL. The table that ships is, on this recipe, the
   amax-fit table `_pack_scales_lut` wrote before the first refit.
2. **An exact entry step with the cross terms kept.** Moving entry `k` by
   `delta` changes the full quadratic by `Q_k delta^2 + 2 g_k delta` with
   `Q_k = sum_r V_rk H V_rk^T`, `g_k = -sum (G o V_k)`, `V_rk` the row's codes
   on blocks assigned to `k`. The oracle takes the best E4M3 byte for each
   entry in turn from that parabola -- exact, not a fit -- starting from the
   coupled assignment (`coupled-table` in the ladder). Over every pass of
   every arm it moved **0 entries** (checked live: a perturbed entry moves
   back, predicted and measured decrease agree to six figures,
   `.scratch/entry_pass_check.py`, not committed).

So the sixteen values are fine. The loss is that nearest-in-linear is the
exact minimiser of the *wrong* parabola: `A_b (c_b - s*_b)^2` is the block's
error given its neighbours at their continuous targets `s*`, and they land
elsewhere. The issue's suggested remedy for (c) -- curvature weighting --
would not help: the assignment is already weighted by `A_b`; what is missing
is the coupling, and `_coupled_landing` supplies exactly that and nothing else.

## Identity check: the sink's `landed` is the wire's `hfit`

On E2M1x2 the trailing refit's `landed` equals `hfit^2 x (W H W^T)` on all
twelve refit rows checked (ratio 1.000000), and the plane materialised from
the written wire equals the last refit's plane (`replay==wire` True on every
unit). The cost-frame numbers below and the `hfit` column are the same
quantity in two units; `out` is the held-out activation-space relative error
on 8192 rows the fit never saw.

## The matched pair on the landing alone (oracle)

Each refit call the encoder made is captured (inputs, metric, candidate,
flag) and replayed under **its own** metric, so the pair differs in the
landing only -- same codes, same step, same table, same accept guard, same
rows. Per pass: `before`, `stepped`, `continuous` (the sink's), `landed`
(the encoder's), `coupled-assign` (ICM re-assignment), `coupled-table` (+ the
exact entry step), `free` (exact per-row joint minimiser, fp64). Fractions of
the pass's starting cost, pooled cost-weighted over units and passes:

```
TBD_TABLE_PASS
```

The `noise` column is `replay_rel_discrepancy`: the replayed `landed` against
the sink's, max TBD_NOISE -- fp32 cancellation on `E H E^T`, the floor under
every cost-frame number here.

## The end-state ladder, at the codes the wire holds

The trailing refit replayed from the encoder's landed plane upward, GS arm.
`out` first (held-out activation space), then `hfit` (fit-row quadratic):

```
TBD_TABLE_LADDER
```

Same ladder, Jacobi arm and control, `out` geomean ratios to landed:

```
TBD_TABLE_LADDER2
```

## In the encoder

The oracle re-lands a finished encode. Inside `encode_unit` the coupled landing
also changes the codes the next trellis pass sees, so the in-encoder number is
the ship-relevant one and was run as its own matched pair (`ldlq_window_sweep
--coupled-landing --drift-control`): control, Jacobi, GS, Jacobi + coupled,
GS + coupled, same process, same rows, drift control first and last.

```
TBD_TABLE_ENCODER
```

TBD_ENCODER_PROSE

## Controls

- **Drift.** Both runs encode the served default first and last per unit:
  bytes IDENTICAL on every unit in both runs; `out` equal to the digit.
- **Reference digests.** The control, Jacobi and GS encodes reproduce the
  sha256 digests recorded by `qwen_lut_gs.json` (the #35 receipt's run) on
  every unit -- the arms here are that receipt's arms.
- **Byte identity with the flag off.** `experiments/gs_refit_byte_baseline.py`
  before the change against after it: **0 changed of 15** artifact blobs, 0
  refused (`sparklina:/home/rob/tmp/ts50/bytes_{before,after}.json`). The
  matrix includes the full-H arms, which run the branch the change edits.
- **Two treatments are not a control.** Jacobi + coupled is reported next to
  GS + coupled so the coupled landing's worth is read at both steps, and the
  "swap" arm (below) is labelled as what it is: one *more* refit, under a
  different objective, not a matched pair.

## Tests, and the before-number

Seven tests in `tests/test_ldlq_lut_plane.py`, "Issue #50: the coupled
landing": the property (the separable landing leaves > 0.1% of the cost
first-order available on the fixture; the coupled one leaves none beyond
fp32 resolution; strictly lower cost; both sweep orders), monotone and at or
below the separable landing (three seeds), flag off is the refit that was
there (bit-identical), a no-op under a 1-D metric (bit-identical), the sink
keeps `landed` next to `coupled`, and two CUDA tests (reaches the bytes and
leaves the wire alone; refused where the landing is already exact -- no
metric, separable metric, no table). 34 pass on the branch (sparklina, GPU).

Anti-vacuity, two pieces of evidence that are not the same thing: on the
pre-change tree (fresh detached worktree at `35f57b4`) the five CPU tests
fail with `TypeError: _refit_scales_lut() got an unexpected keyword argument
'coupled_landing'` -- the trivial kind, the kwarg is new -- and the
**substantive** before-number is the property itself measured on the
pre-change refit: the separable landing leaves **TBD_BEFORE** of the plane's
quadratic first-order available to a block re-assignment (fixture seed 11,
128 columns; 62.09% Jacobi / 61.81% GS). The coupled landing leaves
`<= eps x cost`.

## Scope limits

- Six dense Qwen3-0.6B units, one wire (E2M1x2, q896 cap, LUT16, span 2),
  one metric per arm. **Not served.** `out` is a held-out activation-space
  screen, not KL; under principle 3 nothing here is a result.
- **No GLM arm.** The `h^1.0` default was chosen on GLM experts; any move of
  a default inherits the GLM six-expert geomean gate. The lever is opt-in and
  not reachable from `ActivationSource`, exactly as `refit_gauss_seidel`.
- The `free` ceiling is the exact minimiser over *continuous* per-block
  scales at fixed codes; it is not a bound on what a different code choice
  could reach.
- Box: sparklina, two runs concurrent with each other; GPU power 20-25 W of
  the envelope -- the oracle's ICM is a Python loop over blocks and is a
  measurement harness, not a hot path.
- Two other agents worked #50 concurrently in their own worktrees
  (`/home/rob/tmp/i50-lut-ceiling`, `/home/rob/tmp/ts50-landing-ceiling`);
  nothing here reads or depends on theirs. The branch carries `f948088`, a
  snapshot commit of this agent's in-flight v1 oracle preserved at a session
  kill; the v2 in this commit supersedes it.

## Filed, not taken

TBD_SWAP
