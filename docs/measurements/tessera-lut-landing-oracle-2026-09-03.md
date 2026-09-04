# The LUT landing's loss is in the assignment, not the table fit; landing the blocks against each other takes the GS arm to 0.84x at frozen codes (2026-09-03)

**Claim, weight space and held-out activation space (measured; a screen and
not a result).** Issue #50 read `tessera-lut-refit-gauss-seidel-2026-09-03`'s
diagnostic sink -- the landing gives back 24-91% of the refit's step -- and
named the cause: `_fit_lut` chooses the sixteen table entries under a
**separable** model, `sum_b A_b (c_b - s*_b)^2`, with every cross-block term
of the metric dropped. The first half is true and was never in doubt: that is
the objective at `encode.py::_lut_cost`, and it is separable by construction.
The second half -- that the separable *table fit* is where the loss lives --
was measured and is **false**. On six dense Qwen3-0.6B units at the E2M1x2
q896 cap wire, over 72 refit passes across three arms (`h^1.0`, full-H
Jacobi, full-H Gauss-Seidel): the encoder's accept guard took the re-fitted
table on **16 passes** (4 of the 24 Gauss-Seidel ones) and re-assigned the
table the unit already had on the other 56; and an exact coordinate step over
the sixteen entries with **every** cross-block term kept, started from the
coupled assignment, moved **0 entries on every pass of five of the six
units** -- on `L2.down_proj`, the unit whose plane starts furthest from its
optimum, 6-19 entries a pass worth ~1% of that unit's error. The table is
optimal entry by entry under the coupled objective. What is not optimal is the
**assignment** (the issue's (c)): each block lands on the entry nearest its
*own* one-step target `s*_b`, which is the block's conditional minimiser given
its neighbours at *their* targets -- not where its neighbours land. Once they
have landed, **62%** of the plane's quadratic is still available to a
block-by-block re-assignment on the pre-change refit (fixture seed 11, 128
columns: 62.09% Jacobi / 61.81% GS), and the sink's `landed - continuous` was
reading that.

**The coupled landing** (`_coupled_landing`, opt-in `refit_coupled_landing`)
re-assigns every block to the table entry minimising the **full** quadratic
given every other block where it now stands -- iterated conditional modes over
blocks, gradient field carried block to block, exact per-block gain, the
refit's own revert rule kept (a block whose conditional optimum is non-positive
holds its scale), stop when no block moves or a sweep lowers the cost by less
than fp32 resolves on it. Under the served `h^1.0` (1-D) metric nearest-in-
linear already is the conditional minimiser, so the encoder refuses the flag
there and the oracle's replay of that arm moved **0 blocks** (one fp32 tie in
3.67 M block-passes); under a coupled metric it is a fourth candidate at or
below the third. What it is worth, GS arm, six units:

- **Oracle, at the codes the wire holds** (the trailing refit replayed with
  the coupled landing; the pre-registered decision number): `out` geomean
  landed **0.04922 -> 0.04124 (0.8377x, -16.2%)**, against a pre-registered
  bar of **1.38%** (#35's promotion margin): **clears it**. `hfit` 0.04620 ->
  0.03806 (0.8239x). Against the served `h^1.0` control (0.05113) the coupled
  GS arm is **0.8065x**. One unit carries most of it: `L2.down_proj` 0.02383
  -> 0.00993 (0.4167x); **without it the five-unit geomean is 0.9633x** (out)
  / 0.9548x (hfit), which still clears the bar, and no unit regresses (worst
  0.9964x on `L13.down_proj`, 0.9958x on `L27.o_proj`).
- **In the encoder** (the coupled landing inside the four-pass alternation,
  where it also changes the codes the next trellis pass sees): TBD_E_PROSE
- **Of the landing loss itself** (`landed - continuous`, pooled over passes,
  cost-weighted): the coupled landing recovers **102%** of it on the GS arm
  (it lands below the one-step continuous point, which is a step and not a
  minimiser) and 56% on the Jacobi arm; measured against the exact joint
  minimiser over continuous per-block scales (`free`, fp64) it recovers
  **76.3% / 43.3%** of the distance. Per unit at the trailing pass (GS, vs
  `free`): q_proj 45%, k_proj 33%, L13.down 12%, gate 37%, L2.down 86%,
  o_proj 20%. The rest of the distance is the sixteen-entry budget: `free-e4m3`
  (any E4M3 value per block, 8 bits) reaches 0.03803 against `free` 0.03219,
  so what the wire cannot have is mostly entry *count*, not grid.

**Nothing here is served.** Six units, one wire, weight-space `hfit` and a
held-out activation-space `out` screen. No KL, no GLM arm. The lever stays
opt-in, exactly as `refit_gauss_seidel` is, and inherits the same two gates
before any default moves: the GLM six-expert geomean and a same-session serve.

Commit: see `git log` for this file. Code: `src/tessera/encode.py`
(`_coupled_landing`, `_refit_scales_lut_metric(coupled_landing=)`,
`encode_unit(refit_coupled_landing=)` -- `True` every pass, `"trailing"` the
last refit only), `src/tessera/export.py`
(`encode_linear_planes(refit_coupled_landing=)`). Tests:
`tests/test_ldlq_lut_plane.py` (8 new; 41 pass on the merged branch). Data:
`/mnt/shared/tessera-runs/ldlq-lut/qwen_lut_landing_oracle.json`
(`experiments/lut_landing_oracle.py`) and
`/mnt/shared/tessera-runs/ldlq-lut/qwen_lut_coupled.json`
(`experiments/ldlq_window_sweep.py --coupled-landing --gauss-seidel
--drift-control`), read together by `experiments/lut_landing_oracle_report.py`.
Box: sparklina, out of `/mnt/shared`, both runs concurrent.

## What "separable by construction" is true of, and why it does not matter

`_refit_scales_lut_metric` does three approximate things after the step (its
docstring's (a), (b), (c)): the Jacobi/Gauss-Seidel step itself, the table
fit, and the landing. The issue's charge is against (b). Two measurements
close it:

1. **The accept guard.** Every pass scores three planes on the full
   quadratic: the plane the unit had, the old table re-assigned to the stepped
   targets, and the re-fitted table. `_fit_lut`'s output is the third. The
   candidate the guard chose is recorded per pass in the sink. Over the 72
   passes the re-fitted table won 16: 8 under `h^1.0`, 4 under Jacobi, 4 under
   Gauss-Seidel, and 12 of the 16 are `L2.down_proj` (all four passes of the
   `h^1.0` and Jacobi arms, three of the GS ones); the other four are first
   passes (q_proj x2, gate_proj, o_proj). On the other 56 the table that ships
   is the amax-fit table `_pack_scales_lut` wrote before the first refit,
   re-assigned. Filed as tessera#76.
2. **An exact entry step with the cross terms kept.** Moving entry `k` by
   `delta` changes the full quadratic by `Q_k delta^2 + 2 g_k delta` with
   `Q_k = sum_r V_rk H V_rk^T`, `g_k = -sum (G o V_k)`, `V_rk` the row's codes
   on blocks assigned to `k`. The oracle takes the best E4M3 byte for each
   entry in turn from that parabola -- exact, not a fit -- starting from the
   coupled assignment (`coupled-table` in the ladder; `coupled-assign` is the
   same plane before it). On five of the six units it moved **0 entries on
   every pass** of every arm (checked live: a perturbed entry moves back,
   predicted and measured decrease agree to six figures). On `L2.down_proj`
   it moved 6-19 entries a pass: `coupled-assign` 0.01002 -> `coupled-table`
   0.00993 out, 0.9% of a unit the assignment had already taken from 0.02383.

So the sixteen values are fine. The loss is that nearest-in-linear is the
exact minimiser of the *wrong* parabola: `A_b (c_b - s*_b)^2` is the block's
error given its neighbours at their continuous targets `s*`, and they land
elsewhere. The issue's suggested remedy for (c) -- curvature weighting --
would not help: the assignment is already weighted by `A_b`; what is missing
is the coupling, and `_coupled_landing` supplies exactly that and nothing else.

**The revert rule is kept, and it costs `L2.down_proj` 0.7%.** The refit
holds any block whose optimum is a non-positive scale (`valid`), because a
collapsed scale was measured to break the alternation's monotonicity in true
SSE (`_refit_scales_lut`). A first version of the sweep re-assigned those
blocks too, and the diagonal control caught it: under `h^1.0` the oracle moved
16-33 blocks a pass on `L2.down_proj` -- exactly its `reverted` count --
worth 0.7-1.9% of the cost, and nothing anywhere else. Masked, the control
moves nothing and `L2.down_proj`'s coupled trailing `out` is 0.00993 instead
of 0.00921; the other five units are unchanged to the digit. The first run
is kept as `qwen_lut_landing_oracle_unmasked.json` / `qwen_lut_coupled_
unmasked.json` for the record; every number in this receipt is from the
masked run.

## Identity check: the sink's `landed` is the wire's `hfit`

On E2M1x2 the trailing refit's `landed` equals `hfit^2 x (W H W^T)` on all
twelve refit rows checked (ratio 1.000000), and the plane materialised from
the written wire equals the last refit's plane (`replay==wire` True on every
unit). The cost-frame numbers below and the `hfit` column are the same
quantity in two units; `out` is the held-out activation-space relative error
on 8192 rows the fit never saw. Independently, master's landing-ceiling
instrument (`dabb615`, `experiments/lut_landing_ceiling.py`, its
`i50_ceiling_qwen.log`) read the same GS arm's oracle-table at **0.8239x**
out and `free` at 0.6540x -- a second implementation, the same codes, the
same digits on `free`, and 0.8239 against this receipt's unmasked 0.8273 /
masked 0.8377 (it does not hold the revert rule).

## The matched pair on the landing alone (oracle)

Each refit call the encoder made is captured (inputs, metric, candidate,
flag) and replayed under **its own** metric, so the pair differs in the
landing only -- same codes, same step, same table, same accept guard, same
rows. Per pass: `before`, `stepped`, `continuous` (the sink's), `landed`
(the encoder's), `coupled-assign` (ICM re-assignment), `coupled-table` (+ the
exact entry step), `free` (exact per-row joint minimiser, fp64). Fractions of
the pass's starting cost, pooled cost-weighted over units and passes, then per
unit and pass on the GS arm:

```
   fractions of the pass's starting cost; recov = share of (landed - continuous) the
   coupled landing gets back; vs free = the same share of (landed - free)
   arm          step  landing  coupled   recov  vs free  freegap entry moves new-table wins
   h^1.0     19.328%  14.725%   0.000%    0.0%     0.0%  14.760%           0        8/24   
   jacobi    61.199%  30.756%  17.330%   56.3%    43.3%  40.068%          39        4/24   
   gs        62.095%  36.579%  37.384%  102.2%    76.3%  49.023%          42        4/24   
   per unit and pass, GS arm:
   unit              p     step  landing  coupled   recov  vs free    candidate  moves entry    noise
   L0.q_proj         0  56.852%  29.933%  15.725%   52.5%    45.8%    new-table  20870     0  0.0e+00
   L0.q_proj         1  43.939%  34.266%  18.358%   53.6%    46.2%    old-table  16800     0  0.0e+00
   L0.q_proj         2  39.674%  34.246%  17.947%   52.4%    44.9%    old-table  16291     0  0.0e+00
   L0.q_proj         3  39.183%  34.016%  17.961%   52.8%    45.2%    old-table  16248     0  0.0e+00
   L1.k_proj         0  43.069%  19.604%   8.388%   42.8%    35.6%    old-table  12104     0  0.0e+00
   L1.k_proj         1  35.789%  21.321%   8.951%   42.0%    34.8%    old-table  11615     0  0.0e+00
   L1.k_proj         2  33.627%  21.997%   9.393%   42.7%    35.8%    old-table  11487     0  0.0e+00
   L1.k_proj         3  33.605%  20.301%   8.119%   40.0%    33.0%    old-table  10989     0  0.0e+00
   L13.down_proj     0  19.006%  15.478%   2.827%   18.3%    16.1%    old-table  20889     0  0.0e+00
   L13.down_proj     1  16.577%  14.892%   2.344%   15.7%    13.9%    old-table  17427     0  0.0e+00
   L13.down_proj     2  15.798%  14.547%   2.056%   14.1%    12.5%    old-table  15826     0  0.0e+00
   L13.down_proj     3  15.282%  14.222%   1.914%   13.5%    11.9%    old-table  14651     0  0.0e+00
   L14.gate_proj     0  37.831%  20.527%  10.361%   50.5%    41.3%    old-table  32432     0  0.0e+00
   L14.gate_proj     1  31.098%  20.449%   9.390%   45.9%    37.6%    old-table  29326     0  0.0e+00
   L14.gate_proj     2  29.077%  20.256%   9.154%   45.2%    37.1%    old-table  28345     0  0.0e+00
   L14.gate_proj     3  28.262%  19.988%   9.034%   45.2%    36.7%    old-table  27780     0  0.0e+00
   L2.down_proj      0  88.022%  20.800%  26.533%  127.6%    74.1%    new-table 334976     9  0.0e+00
   L2.down_proj      1  86.584%  59.514%  65.912%  110.8%    86.1%    new-table 243354    19  0.0e+00
   L2.down_proj      2  86.594%  59.141%  66.033%  111.7%    85.3%    old-table 218657     8  0.0e+00
   L2.down_proj      3  86.708%  63.121%  70.928%  112.4%    86.2%    new-table 188448     6  0.0e+00
   L27.o_proj        0  16.494%  11.510%   3.823%   33.2%    24.2%    old-table  35523     0  0.0e+00
   L27.o_proj        1  14.261%  11.365%   3.406%   30.0%    21.8%    old-table  32065     0  0.0e+00
   L27.o_proj        2  13.328%  11.045%   3.074%   27.8%    20.2%    old-table  29723     0  0.0e+00
   L27.o_proj        3  12.906%  10.914%   2.960%   27.1%    19.6%    old-table  29240     0  0.0e+00
```

`recov` above 100% is not an error: `continuous` is the line-searched step,
not the minimiser, and a re-assignment that keeps going lands below it. The
`noise` column is `replay_rel_discrepancy`, the replayed `landed` against the
sink's: max **4.6e-5** (on `L2.down_proj`) -- fp32 cancellation on `E H E^T`,
the floor under every cost-frame number here.

## The end-state ladder, at the codes the wire holds

The trailing refit replayed from the encoder's landed plane upward, GS arm.
`out` first (held-out activation space), then `hfit` (fit-row quadratic);
then the same ladder's `out` geomean ratios for the Jacobi arm, the control,
and the "swap" arm:

```
   unit               landed   assign    table     e4m3     free | assign/l  table/l   free/l |      ctl table/ctl
   -- out
   L0.q_proj         0.04296  0.03894  0.03894  0.03624  0.03347 |   0.9066   0.9066   0.7791 |  0.04375    0.8902
   L1.k_proj         0.04256  0.04082  0.04082  0.03955  0.03659 |   0.9591   0.9591   0.8596 |  0.04567    0.8939
   L13.down_proj     0.08404  0.08374  0.08374  0.08176  0.07848 |   0.9964   0.9964   0.9339 |  0.08348    1.0031
   L14.gate_proj     0.05123  0.04926  0.04926  0.04769  0.04483 |   0.9615   0.9615   0.8752 |  0.05304    0.9286
   L2.down_proj      0.02383  0.01002  0.00993  0.00721  0.00360 |   0.4207   0.4167   0.1511 |  0.02686    0.3696
   L27.o_proj        0.07585  0.07553  0.07553  0.07506  0.07177 |   0.9958   0.9958   0.9462 |  0.07519    1.0046
   GEOMEAN           0.04922  0.04130  0.04124  0.03803  0.03219 |   0.8391   0.8377   0.6540 |  0.05113    0.8065
   -- hfit
   L0.q_proj         0.04188  0.03770  0.03770  0.03482  0.03192 |   0.9003   0.9003   0.7623 |  0.04283    0.8803
   L1.k_proj         0.04137  0.03938  0.03938  0.03802  0.03501 |   0.9520   0.9520   0.8465 |  0.04477    0.8796
   L13.down_proj     0.07758  0.07682  0.07682  0.07450  0.07098 |   0.9903   0.9903   0.9150 |  0.07793    0.9857
   L14.gate_proj     0.04868  0.04622  0.04622  0.04453  0.04164 |   0.9495   0.9495   0.8554 |  0.05093    0.9076
   L2.down_proj      0.02340  0.00929  0.00922  0.00691  0.00335 |   0.3970   0.3942   0.1431 |  0.02677    0.3445
   L27.o_proj        0.06351  0.06254  0.06254  0.06189  0.05842 |   0.9848   0.9848   0.9198 |  0.06427    0.9731
   GEOMEAN           0.04620  0.03811  0.03806  0.03513  0.02940 |   0.8249   0.8239   0.6365 |  0.04855    0.7840

   LDLQ 1.0/32 + refit full-H (Jacobi)          landed 0.05043  assign 0.9244x  table 0.9238x  e4m3 0.7346x  free 0.6419x
   control [LDLQ 1.0/32 + refit h^1.0]          landed 0.05113  assign 1.0000x  table 1.0000x  e4m3 0.9919x  free 0.8291x
   control codes + one full-H GS refit          landed 0.04537  assign 0.9068x  table 0.9051x  e4m3 0.8157x  free 0.6849x
```

The control's ladder is the separability check on real units: `assign` and
`table` at 1.0000x, `free-e4m3` 0.9919x -- under a diagonal metric the
sixteen-entry table already sits within 0.8% of any E4M3 value per block, and
the whole distance to `free` (0.8291x) is what the E4M3 grid cannot name.

## In the encoder

The oracle re-lands a finished encode. Inside `encode_unit` the coupled landing
also changes the codes the next trellis pass sees, so the in-encoder number is
the ship-relevant one and was run as its own matched pair (`ldlq_window_sweep
--coupled-landing --gauss-seidel --drift-control`): control, Jacobi, GS,
Jacobi + coupled, GS + coupled, and both again with the coupled landing on the
**trailing refit only** (`refit_coupled_landing="trailing"`) -- the inner
passes then are the plain arms' passes to the float (test), and only the plane
that ships is re-landed. Same process, same rows, drift control first and last.

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
  refused, three times -- after the sweep was added, after the revert mask,
  and after the merge with master's landing-ceiling instrument
  (`sparklina:/home/rob/tmp/ts50/bytes_{before,after,after2,after3}.json`).
  The matrix includes the full-H arms, which run the branch the change edits.
- **Two treatments are not a control.** Jacobi + coupled is reported next to
  GS + coupled so the coupled landing's worth is read at both steps; the first
  sweep's un-doing of the revert leg was caught by the diagonal control and
  removed before any number was kept; and the "swap" arm (below) is labelled
  as what it is: one *more* refit under a different objective, not a matched
  pair.

## Tests, and the before-number

Eight tests in `tests/test_ldlq_lut_plane.py`, "Issue #50: the coupled
landing": the property (the separable landing leaves > 0.1% of the cost
first-order available on the fixture; the coupled one leaves none beyond
fp32 resolution, reverted blocks excluded; strictly lower cost; both sweep
orders), monotone and at or below the separable landing (three seeds), flag
off is the refit that was there (bit-identical), a no-op under a 1-D metric
(bit-identical), the sink keeps `landed` next to `coupled`, and three CUDA
tests (reaches the bytes and leaves the wire alone; `"trailing"` re-lands the
last refit only and an unknown mode is refused; refused where the landing is
already exact -- no metric, separable metric, no table). 41 pass on the
merged branch (sparklina, GPU); `tests/test_ldlq_window.py` +
`tests/test_merge_guard.py` 45 pass.

Anti-vacuity, two pieces of evidence that are not the same thing: on the
pre-change tree (fresh detached worktree at `35f57b4`) all eight fail with
`TypeError: ... got an unexpected keyword argument` -- the trivial kind, the
kwarg is new -- and the **substantive** before-number is the property itself
measured on the pre-change refit: the separable landing leaves **62%** of the
plane's quadratic first-order available to a block re-assignment (fixture
seed 11, 128 columns; 62.09% Jacobi / 61.81% GS). The coupled landing leaves
`<= eps x cost`. Note the property test's `<= eps x cost` holds because the
sweep terminates by "no block moved"; a sweep stopped by the fp32 rule with
moves still being taken could leave more, and would then be a test of the
stop rule, not of the landing.

## Scope limits

- Six dense Qwen3-0.6B units, one wire (E2M1x2, q896 cap, LUT16, span 2),
  one metric per arm. **Not served.** `out` is a held-out activation-space
  screen, not KL; under principle 3 nothing here is a result.
- **No GLM arm.** The `h^1.0` default was chosen on GLM experts; any move of
  a default inherits the GLM six-expert geomean gate. The lever is opt-in and
  not reachable from `ActivationSource`, exactly as `refit_gauss_seidel`.
- The `free` ceiling is the exact minimiser over *continuous* per-block
  scales at fixed codes, revert rule not applied; it is not a bound on what a
  different code choice could reach.
- The six-unit geomeans are carried by `L2.down_proj`; the five-unit geomeans
  are given next to them everywhere for that reason.
- Box: sparklina, two runs concurrent with each other; GPU power 20-25 W of
  the envelope -- the oracle's ICM is a Python loop over blocks and is a
  measurement harness, not a hot path. The encoder's sweep is the same loop;
  its cost is one `E @ H` per sweep in `nb` panels and 8-21 sweeps a pass on
  these units, unmeasured against the trellis and not claimed cheap.
- Two other agents worked #50 concurrently in their own worktrees; master's
  landing-ceiling instrument (`dabb615`) was merged into this branch, with
  the coupled landing refused under its non-`"table"` modes (no table to
  assign into). The branch also carries `f948088`, a snapshot commit of this
  agent's in-flight v1 oracle preserved at a session kill; the v2 in this
  commit supersedes it.

## Filed, not taken

- **tessera#75** -- the served default's codes plus **one** trailing full-H
  Gauss-Seidel refit with the coupled landing: `out` geomean **0.04106
  (0.8030x control)**, level with the full-H GS alternation re-landed the same
  way (0.04124) and ahead of it on five of six units. The full-H objective's
  worth looks to be in the trailing refit, not in steering the inner passes.
  Not a matched pair (five refits against four); the fair pair and the GLM
  arm are the ask.
- **tessera#76** -- `_fit_lut`'s re-fitted table is taken on 16 of 72 passes,
  12 of them on one unit; the amax table ships. Count it on GLM before
  removing it.
