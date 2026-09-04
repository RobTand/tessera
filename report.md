# #89 — the dyadic residue of the table sigma

**Status: reproduced exactly, mechanism found and stated as a predictor, and
the disposition is a *cost to record*.** #89 names the right parameter and
draws the wrong response from it. No proposal to move `default_channel_sigma`
follows; the measurement says that constant already sits at the minimum of the
curve #89 discovered.

Everything below is Qwen3-0.6B, E4M3 grid, window body, CHANNEL plane, L=14,
`window_seed=0`, weight-space, `h_diag.pt` as the diagonal Hessian. Unless a
line says otherwise the unit is `model.layers.2.mlp.down_proj` [1024, 3072] and
the rung is **R2048 (8 b/wt)** — which is where the whole effect lives; see §2.
Nothing here is a served number.

---

## 0. Defect or cost — the answer, and the evidence that separates them

**A cost to record.** Four measurements decide it, and each one is a thing a
defect would have failed.

1. **The residue is a single periodic curve with one minimum, and the shipped
   default sits in the minimising band.** Twenty-one sigmas across one binade
   (§4) put h at 1.35 / 1.37 / 1.18 / 1.09 / **1.00** / 1.08 / 1.32 across the
   seven bands the E4M3 grid admits — worse in *both* directions from the
   default's band. A constant sitting at an arbitrary point of a curve it does
   not model would be a defect. This one is at the bottom of it.
2. **The 37% is an 8-bit-rung effect.** At R1024 (4 b/wt) — where the E4M3
   window wire actually ships — the same unclamped sigmas read 1.000 / 1.001 /
   0.985. The residue is worth **0.1%** there, against 37% at R2048.
3. **Under the refit objective production runs, the sensitivity is gone.**
   `DEFAULT_REFIT_OBJECTIVE["channel"] == "hessian"`, and when export is given
   an `ActivationSource` that means the **full [cols, cols] Hessian**, not a
   diagonal. Measured on that arm (§5b): the ratio is **0.9677** — m=0.75 is
   *better* than the default, and the effect has changed sign, not just size.
4. **It is one unit.** Across eight dense units (§6) seven move ≤1.4% and one
   reads below 1.0. The outlier is the only unit whose *error* concentrates on
   six massive-activation columns (top-6 h share 0.999, 99% of the error).

**What is therefore worth recording, and is not hypothetical.** On a
**weights-only** E4M3 CHANNEL encode at 8 b/wt of a massive-activation unit,
the table sigma's dyadic residue is worth up to 37% Hessian-weighted weight
error. The `activation` argument of `export_checkpoint` is optional, and that
path is live: of the six E4M3 artifacts under `/mnt/shared/tessera-runs/`,
**four carry no `activation_aware` block at all** (`qwen3-0.6b-uniform-R750`,
`-R1006`, `-R1262`, `qwen3-0.6b-tessera-e4m3-reach-gridbook`) and a fifth
(`ldlq-gridbook`) had a Hessian and recorded `refit_metric: "plain"`. So the
metric-free refit is what most E4M3 bytes on this box were built with.

**My registered branch fired, and I am reporting that honestly.** §0 of the
earlier draft registered: *if `refit_metric=h` collapses the ratio, the refit
is a defect and the proposal is on the refit.* It did — 1.3667 → 0.9853. I then
ran the arm production actually runs (the full H) rather than resting the
disposition on a diagonal nobody ships, and it collapses too. The "proposal on
the refit" therefore reduces to *supply a Hessian*, which is already the
default whenever there is one. There is nothing left to propose, and that is
why the disposition lands on **record**, not on **fix**.

---

## 1. Reproduction — exact, both rungs

`experiments/ts89_dyadic_reach.py --stage repro`, results in
`experiments/results/ts89_repro.json`. The default arm ran first and again
last and was identical both times.

| arm | table sigma | reach | issue | measured | delta |
|---|---|---|---|---|---|
| m=0.5 | 47.09 | 192 | 1.000 | 1.000 | 0.0000 |
| m=1 | 94.18 | 384 | 1.000 | 1.000 | 0.0000 |
| rho=0.5 | 47.09 | 192 | 0.995 | 0.995 | −0.0003 |
| m=0.75 | 70.64 | 288 | 1.367 | 1.367 | −0.0003 |
| rho=0.75 | 70.64 | 288 | 1.362 | 1.362 | +0.0004 |
| m=1.25 | 117.73 | 448 | 1.313 | 1.313 | +0.0003 |
| m=1.5 | 141.27 | 448 | 1.341 | 1.341 | +0.0002 |
| m=2 | 188.36 | 448 | 1.411 | 1.411 | +0.0003 |

R1024 reproduced too (m=0.75 → 1.001, rho=0.75 → 0.985).

The encoder is deterministic, so an exact match is what a healthy repo should
give: this establishes **no drift since `430ca5e`**, not an independent
confirmation of the effect.

## 2. What the 1.367 is a statement about

Three scope facts, all measured, all narrowing the claim hard.

**h on this unit is six columns.** `h_max / h_median = 2.2e6`; columns 55, 128
and 1489 carry 91.9% of the metric and six carry 99.9%. These are Qwen3-0.6B's
known outlier input channels.

**In weight space the two sigmas are the same.** At every refit count the plain
`||e||/||w||` ratio is 1.0022–1.0027. The whole of #89's 1.367 is
Hessian-weighted error on six of 3072 columns.

**And it is an 8-b/wt effect.** The R1024 half of the same reproduction reads
1.001 (m=0.75) and 0.985 (rho=0.75) — the unclamped bands 192, 288 and 384 are
within 1.5% of each other at 4 b/wt. Every band number below is R2048. Whether
the curve reappears at 4 b/wt on some other unit is not measured.

## 3. Sigma is a gauge; the residue is its only coordinate (negative result)

**The sigma axis is exactly a gauge up to powers of two.** Every row's pre-fp16
scale is proportional to `1/sigma` — a row inside the reach starts at
`rms/sigma`, and a row past it starts at `reach·rms/amax`, which is again
proportional to `1/sigma` because `reach ∝ sigma`. `channel_global` then
returns `2^floor(log2(median scale))`, **a power of two**. So under
`sigma → 2^k·sigma` every stored fp16 row word is *bit-identical* and only the
global's exponent moves, while every table value scales by the same `2^k`. The
encode is exactly gauge-equivalent, not approximately, and sigma's only free
parameter is its **dyadic residue** `log2(σ) mod 1`. #89's parameterisation is
right.

The measurement says so to eight digits. At R2048 `m=1` and `m=0.5` read
`wt = 0.026461195201` and `0.026461195201` — identical to twelve decimals — and
`h = 0.010237647902` against `0.010237653232`, a relative difference of 5e-7,
which is the residue of the few table entries that hit the E4M3 floor
(`table(2σ) = 2·table(σ)` fails only for the innermost quantiles, which snap
onto the grid's smallest magnitude instead of halving through it: 12 entries of
16384 at σ=17.66, 2 at σ=94.18). The gauge is pinned by a test
(`test_the_channel_sigma_is_a_gauge_up_to_powers_of_two`), including where it
breaks.

**The alphabet's own error does not carry the effect.** For every unclamped
sigma the reach is proportional to sigma, so the two arms' reach-aware row
scales are exactly proportional and the two normalised tables differ only in
where the E4M3 snap lands. Snapped-against-ideal relative energy is `6.976e-4`
at the default and `7.031e-4` at m=0.75 — **a predicted ratio of 1.004 against
a measured 1.367**.

**And the first Viterbi pass does not carry it either.** `ts89_table_surgery.py`
replays production's **first pass alone** — exact, on the CPU, same
`trellis_weighting="scale"` branch weight — on the 64 columns holding 99.93% of
this unit's h. It reads **1.0369**, and grafting the default table's outermost
1/2/4/8/16/32/64/128 entries into the m=0.75 table moves it to 1.0368.

> **Scope, corrected.** That negative is about the **first pass**, not about the
> alphabet in general. §4 shows the converged alternation *is* strongly ordered
> by a property of the table. An earlier draft of this report generalised the
> surgery result to "no part of the table carries it"; that overreached, and
> the ladder is the arm that shows it.

## 4. The mechanism: the E4M3 mantissa the reach snaps to

`--stage ladder`, 21 sigmas across one binade, R2048, results in
`experiments/results/ts89_ladder.json`.

| reach (grid units) | ulp/reach | measured h ratio | n arms |
|---|---|---|---|
| 256 | 0.1250 | 1.348–1.357 | 3 |
| 288 | 0.1111 | 1.366–1.376 | 3 |
| 320 | 0.1000 | 1.174–1.185 | 3 |
| 352 | 0.0909 | 1.084–1.093 | 3 |
| **384 (the default)** | 0.0833 | **0.999–1.010** | 3 |
| 416 | 0.0769 | 1.078–1.080 | 3 |
| 448 | 0.0714 | 1.316–1.317 | 3 |

h is **flat to ~1% inside a band and steps between bands**, with the minimum at
the default's band and both directions worse. The two dyadic gauge arms
(σ = 35.3176 and 17.6588, reach 144 and 72) read `wt 0.026532`, `h 0.013992`,
ratio 1.367 — identical to σ = 70.6353 in every printed digit, as §3 requires.

**The controlling variable, stated so it can be used.** `reach = snap_E4M3(c·σ)`
with `c ≈ 4.0`, and E4M3 is floating point, so what survives the gauge is the
**mantissa** the reach snaps to: 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0 — seven
values, 7.0 being the grid's finite peak (448), so every residue from ≈0.73
upward pins there. That mantissa is a pure function of the dyadic residue and
is invariant under `σ → 2^k σ`, exactly as the measurement demands. **Given a
sigma, snap the reach, read the mantissa, read the band.**

**It predicts #89's own table.** m=0.5 → 192 = 6·32 → mantissa 6.0 → predicts
1.00, measures 1.000. m=1 → 384 = 6·64 → 1.00 / 1.000. rho=0.5 → 192 → 1.00 /
0.995. m=0.75 and rho=0.75 → 288 = 4.5·64 → mantissa 4.5 → predicts 1.37,
measures 1.367 and 1.362. m=1.25/1.5/2 → clamped at 448 → mantissa 7.0 →
predicts ≈1.32, measures 1.313 / 1.341 / 1.411 (the clamp adds its own spread
on top; see §8). Five arms exact, three approximate.

**This corrects #89's reading of its own evidence.** The issue reads the
`rho=0.5` arm — reach 192, no cost — as showing the effect is not about the
reach. But 192 is `6·32`: the *same mantissa* as the default's `6·64`, hence
the same point on the curve, by the gauge. It is not a counterexample to the
reach; it is a confirmation of the gauge. And the issue's dyadic-versus-
non-dyadic framing is a special case: the dyadic arms share the default's band
*by construction*. The real object is a periodic function of the residue with
one minimum and one maximum per binade, and "non-dyadic" spans everything from
0.999 (σ = 92.5) to 1.376 (σ = 74.5).

**Two readings the data refutes.** The 448 band is bad (1.32) with **2 of
16384 entries saturated** — essentially unclamped — so "only the clamping arms
are worse than the default" is false. And the curve is not monotone in the
reach: 448 is the largest unclamped reach and the second-worst band.

### 4b. What does *not* explain the ordering (a recorded failure)

The ladder printed a `predict` column: relative table resolution `ulp/reach`,
normalised to the default. **It fails, and the log should not be read as if it
did not.** It holds to 2.5% across bands 288→384 and then inverts, calling 448
the best band (0.857) where it measures second worst (1.32).

`experiments/ts89_band_predictors.py` (CPU, no encode) replays the table build
at all 21 sigmas and scores every obvious alphabet statistic by Spearman
against the measured ratio (`experiments/results/ts89_bands.json`):

| statistic | Spearman |
|---|---|
| snapped-vs-ideal energy, outer tail (above 2σ) | 0.778 |
| snapped-vs-ideal energy, whole table | 0.640 |
| `ulp/reach` (the ladder's own predictor) | 0.579 |
| −reach | 0.579 |
| −innermost quantile / E4M3 floor | 0.543 |
| worst relative snap | 0.243 |
| −distinct values | −0.427 |
| −distinct values in the outer tail | −0.543 |
| saturated entries | −0.543 |

The best of them, outer-tail snap energy, gets the gross trend and **mis-ranks
exactly the pairs that matter**: 352 (5.87e-4 mid) against 384 (5.94e-4), where
384 is measurably better; and 320 (6.25e-4) against 416 (6.44e-4), where 416 is
better. Distinct-value count is essentially constant (200–204) across the whole
ladder, so the bands are not a resolution story.

> **So: the band is a mechanism you can predict with, and not one I can derive.**
> Which band a sigma lands in is a two-line calculation that gets every arm of
> #89's table right. *Why* mantissa 6.0 beats 5.5 and 7.0 is not established by
> anything here, and no property of the alphabet alone orders them.

## 5. The refit is the amplifier, not the source

Walking `scale_refit` at both sigmas (`--stage refit`,
`experiments/results/ts89_refit2.json`):

| refits | h (m=1, band 384) | h (m=0.75, band 288) | h ratio | rows moved (1 / 0.75) |
|---|---|---|---|---|
| 0 | 0.01356 | 0.01405 | **1.0367** | 0 / 0 |
| 1 | 0.01239 | 0.01403 | **1.1327** | 258 / 340 |
| 2 | 0.01142 | 0.01402 | **1.2274** | 469 / 520 |
| 3 | 0.01054 | 0.01401 | **1.3296** | 471 / 521 |
| 4 | 0.01024 | 0.01399 | **1.3667** | 471 / 523 |

The `refit=0` arm reads 1.0367 against the reduced model's 1.0369 — a
prediction made before that arm ran, on a machine and a code path it was not
fit to, agreeing to four decimals.

**Read down the columns, not across.** Four refits buy the default's band
**1.324× in h** and buy the 288 band **1.0045× — nothing**. Both arms get the
same 1.003× in plain weight error. So #89's 1.367 is the size of a **benefit
the default's band receives**, not a penalty the alternative pays.

**And the refit is not being *rejected* at m=0.75.** `rows_moved` was added
precisely to separate the two readings of a null lever: at m=0.75 the refit
moves **more** rows than at the default (523 against 471) and buys nothing. It
runs, it accepts its steps, and it converges to a higher floor. The band is
what sets where the alternation can converge to.

### 5b. The 2×2 at `scale_refit=4` — both knobs, and production's own objective

`experiments/results/ts89_fullh.json`.

| refit arm | h (m=1) | h (m=0.75) | h ratio | wt (m=1) | rows@clip |
|---|---|---|---|---|---|
| *(none — #89's regime)* | 0.01024 | 0.01399 | **1.3667** | 0.02646 | 222 / 219 |
| `refit_metric = h` (diagonal) | 0.00983 | 0.00969 | **0.9853** | 0.02796 | 340 / 388 |
| **`refit_metric = H` (full — production's CHANNEL default)** | 0.01074 | 0.01039 | **0.9677** | 0.04804 | 428 / 426 |
| `refit_reach_floor=True` | 0.01394 | 0.01399 | **1.0035** | 0.02649 | 0 / 0 |
| `refit_metric = h` + floor | 0.01378 | 0.01369 | **0.9940** | 0.02718 | 0 / 0 |

Three things this settles.

* **A metric-aware refit removes the residue sensitivity, and production's own
  objective is metric-aware.** The full-H arm is exactly what
  `ActivationSource.encode_kwargs` passes on the CHANNEL plane
  (`export.py:392-394`). Its ratio is 0.9677: the default's band stops being
  special. The full H comes from a different capture than the scoring diagonal
  (`ldlq/h_full_qwen06b.pt` vs `bf16/refs/h_diag.pt`), which agree on this
  unit's top-six columns exactly and on the median diagonal to 7% — so it is an
  out-of-sample metric for the score, not teaching to the test. Its own diagonal
  score is slightly worse than the h-blind default's (0.01074 vs 0.01024) and
  its weight error much worse (0.048 vs 0.026), both expected for an objective
  scored under a different metric than the one it minimises.
* **`refit_reach_floor` explains where the default's 32% comes from and refuses
  to bank it.** With the floor on, `rows@clip` goes to 0 at both sigmas and the
  default's h reverts from 0.01024 to 0.01394 — i.e. the gain the h-blind refit
  buys at the default is bought by *shrinking rows into the trellis clip*. That
  is a real reconstruction, not an artefact, but it is worth knowing that the
  quantity #89 measures is the size of a clip-assisted gain.
* **The mechanism names a knob, and the knob is already set.** Nothing here is a
  wire change; every one of these is an argument `encode_linear_planes` already
  takes, and the shipping default already takes the branch that removes the
  effect whenever a Hessian is available.

## 6. Generality — eight dense units, R2048

`--units all --skip-2x2`, `experiments/results/ts89_refitall.json`. Ratio is
m=0.75 (band 288) over m=1 (band 384).

| unit | top-6 h share | error on top 6 | ratio @refit 0 | ratio @refit 4 |
|---|---|---|---|---|
| L2 `mlp.down_proj` | 0.9990 | 99.1% | 1.0367 | **1.3667** |
| L2 `q_proj` | 0.3494 | 19.2% | 1.0116 | 1.0140 |
| L2 `k_proj` | 0.3494 | 34.3% | 1.0062 | 1.0129 |
| L14 `gate_proj` | 0.3432 | 45.2% | 1.0053 | 1.0013 |
| L2 `up_proj` | 0.6160 | 10.6% | 1.0061 | 1.0047 |
| L14 `mlp.down_proj` | 0.0442 | 2.9% | 0.9993 | **0.9989** |
| L27 `o_proj` | 0.0967 | 2.2% | 1.0029 | 1.0034 |
| L14 `v_proj` | 0.4532 | 6.2% | 1.0028 | 1.0023 |

Seven of eight move ≤1.4%. The outlier is not the unit with the most
concentrated *Hessian* — several share that — but the only one whose
**error** concentrates on the same six columns (99.1%).

### 6b. The sign test, closed

An earlier section registered a directional prediction: under a *luck* reading
(each residue an arbitrary draw), at least one other unit should read the ratio
**below 1.0**. One does — L14 `mlp.down_proj` at 0.9989.

**I read that as noise, not as a coin landing tails**, and say so rather than
claiming the test came out clean. 0.9989 is 0.1%, inside the 0.2–1.4% spread of
the six units that read above 1.0, and the design that settles the question is
the ladder, not the sign: 21 sigmas, seven bands, a single minimum, repeatable
to 1% within each band, and a curve that behaves identically on the two dyadic
gauge arms. A lattice of arbitrary draws does not produce that.

The two observations reconcile cleanly: the **shape** of the curve is a property
of the grid (it is the same seven mantissas for every unit), while the **size**
of the effect is a property of the unit — it needs the error concentrated on
Hessian-heavy columns, which is why seven units barely see it.

## 7. What this says about `default_channel_sigma`

The brief asked me to re-derive the constant against the table's actual use if
the mechanism supported it. **It does not support a change, and I am not
proposing one.** The constant is already at the measured optimum: `94.1804`
puts the reach at 384, the band that minimises h on the very unit #89 is
worried about, with both neighbours worse.

**But the way it got there is a fragility worth writing down.**
`experiments/ts89_residue_ladder.py` replays `_default_sigma`'s forty-rung
quarter-binade ladder. On E4M3 those forty rungs collapse — by the §3 gauge —
to exactly **four** residues, and the two objectives rank them like this:

| residue | reach band | `_default_sigma`'s own RTN objective | measured h ratio |
|---|---|---|---|
| **0.557355** (chosen) | 384 | **best** | **1.000** |
| 0.807355 | 448 | +0.115% | ≈1.32 |
| 0.307355 | 320 | +0.323% | ≈1.18 |
| 0.057355 | 256 | +1.094% | ≈1.35 |

The two objectives agree on the winner and on the loser, and **swap the middle
two**. The margin that decided it was **0.115%** of a scalar-RTN nearest-value
error on a unit Gaussian; the h consequence of that decision is **32%** on a
massive-activation unit at 8 b/wt. The constant chose correctly, on a criterion
that does not see the quantity that made the choice matter, by a margin that
could not have justified it. That is a coincidence to record, not a defect to
repair — and it is the reason the honest answer to "re-derive it" is *not from
this evidence*: re-deriving one per-grid constant against a landscape shaped
like §4's would be fitting a global default to six columns of one dense unit at
a rung the E4M3 wire does not ship at.

**Bytes and `encoder_profile_id`, stated loudly because the brief asked.** No
change I have landed moves either. Recording the hazard for whoever does touch
this constant: `unit_artifact._normalize_reach` normalises
`channel_sigma == default_channel_sigma(grid)` to `None`, so the profile id
does **not** bind the numeric default. Changing `default_channel_sigma` would
change every default-built E4M3 artifact's bytes and decoded tensor **while
leaving its `encoder_profile_id` unchanged** — two artifacts carrying one id
and two renderings. That is a gate problem before it is a numerics problem.

## 8. #84, reconciled

Both tables are right; they measure different axes, and the coordinator's
reading that #89's `m>=1.25` arms share a realised reach is the wrong one.

* On the **ratio** axis (`rho`) `channel_sigma` is fixed at 94.18, so once the
  table clamps at 448 grid units `reach_rms = 448/94.18 = 4.7568` and never
  moves again. That is #84's flat column.
* On the **spread** axis (`m`) `channel_sigma` grows with the arm, so
  `reach_rms = 448/(m·94.18)` keeps falling: 3.8055, 3.1712, 2.3784. Those are
  #89's distinct values. Same clamp, different denominator.

**Inside the clamped regime the reach is not a sufficient statistic.** At ratios
1.75→4.0 the realised reach (4.7568) and the over-reach row fraction (0.2285)
are pinned identically, yet R2048 h runs 1.411 → 1.458 → 1.571 → 1.706 → 1.991.
The table's interior keeps deforming after the reach stops moving, and the
quantity that shows it is `saturated`: 4 / 36 / 358 / 4120 of 16384 entries on
the peak at ratios 1.25 / 1.5 / 2.0 / 4.0.

> **Scope, because §4 says the opposite about the unclamped regime.** Below the
> clamp the snapped reach — via its E4M3 mantissa — *is* the statistic that
> orders the arms, to ~1%. The two statements do not conflict: they are about
> the two sides of the peak. Above it the reach is frozen by construction and
> only the saturation count moves; below it the reach moves and the saturation
> count is 0–2 of 16384.

That last row also corrects #89's own parenthesis. The issue flags `m=2` as the
one clamping arm ("358 of 16384 entries pile on 448") and treats `m=1.25` and
`m=1.5` as clean; they are not — they clamp 4 and 36 entries. #87's agent
reached the same two counts independently. It does not change #89's conclusion
(the two 70.64 arms are genuinely clamp-free and are the ones carrying the
1.367), but of the eight arms in the issue's table only four are clean gauges:
`m=0.5`, `m=1`, `m=0.75` and `rho=0.5`. Any reading of the `m>=1.25` rows is a
reading of the residue *and* the clamp together.

**Landed on this branch** (commit `b85e233`, closes #84's reporting half):
`window_table_reach()` returns requested vs realised reach, `delivered`,
`saturated` and `saturated_fraction`; `EncodedUnit.table_reach` records it per
encode (diagnostic, never wire); `bf16_l_sigma_sweep.py`'s `reach_stats`
reports all of it beside the realised reach. Refusing the clamp rather than
reporting it moves behaviour and is Rob's call — not landed.

## 9. Does anything else cite #80's contaminated R2048 column?

**No.** The R2048 geomean row (m0.75 1.0447, oracle 0.9640) and the predictor
table (R2048 3/8, 1.0221) appear only in
`docs/measurements/tessera-per-unit-reach-2026-09-03.md` itself and in #89,
which is the issue flagging them. No other doc, test, experiment or open issue
reads them. (`0.9640` also occurs in two unrelated results JSONs as a
coincidence of value.)

## 10. What would be needed to ship anything here, and how far short I am

Nothing in this report is a ship claim, and the gap is large:

* Every number is **weight-space h**, and all the band numbers are **one unit at
  one rung**. Principle 3 wants exact full-vocab vLLM KL-vs-BF16 on the served
  artifact at matched bpp. I have no served arm.
* The metric is **six columns** of one Qwen3-0.6B dense unit.
* The effect is at **8 b/wt**; the E4M3 window wire ships at 4 b/wt, where the
  same sweep reads 0.1%. A serving-relevant version of this ticket would have to
  redo the ladder at R1024 on a unit that shows the effect there — I did not
  find one, and did not look beyond the R1024 half of the reproduction.
* A `default_channel_sigma` change would move bytes on every default-built E4M3
  artifact **without moving `encoder_profile_id`** (§7). Before such a change
  could ship, that normalisation would have to bind the constant, or the
  constant would have to be written into the profile explicitly.
* The one arm that would turn §5b into a proposal — "always refit under a
  metric, even weights-only" — has no metric to use when there is no Hessian,
  so it is not a proposal at all. If someone wants the weights-only path
  hardened, the measured options are the reach floor (h 0.01394, worse) or a
  Hessian.

---

## 11. Off-task fixes landed on this branch

Each is a separate commit so it can be taken or dropped independently.

| commit | what |
|---|---|
| `b85e233` | #84's reporting half: `window_table_reach()` returns requested vs realised reach, `delivered`, `saturated`, `saturated_fraction`; `EncodedUnit.table_reach` records it per encode (diagnostic, never wire); `bf16_l_sigma_sweep.py` reports it. Two new tests in `tests/test_window_body.py`. |
| `14880cd` | `scale_channel._default_sigma`'s docstring said the ladder was dyadic. It is a **quarter-binade** ladder (`peak · 2^(-k/4)`, forty rungs) minimising scalar-RTN nearest-value error — neither dyadic nor the objective #89 is about. Prose only, plus prose in `export.py`, `tests/test_bf16_route.py`, `experiments/tessera16_alphabet_floor.py`. |
| `4cf9a69` | `tests/test_window_body.py::test_the_channel_sigma_is_a_gauge_up_to_powers_of_two` — pins §3's gauge (bit-identical fp16 row words, exact table scaling) and characterises where it breaks at the E4M3 floor. Passes on master too: it is a pin, not evidence of a fix. |
| `a7f8ece` | `pbrun_result.txt` had ridden into git on a `git add -A`; the pool refuses to run an action whose declared result path already exists, so a restored copy failed two jobs. Untracked and excluded locally. |

## 12. Consultations

None. No `fable-*` agent was spawned. Every open question on this ticket was a
few-encode experiment or a CPU diagnostic, and the one reduced model I built
(`ts89_table_surgery.py`) was cheaper to write than to delegate.
