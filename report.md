# #89 — the dyadic residue of the table sigma

**Status: reproduced, mechanism found, and the mechanism is not the one the
issue's title names.** No proposal to move `default_channel_sigma` follows,
because the measurement says that constant is not where the effect lives.

Everything below is Qwen3-0.6B, E4M3 grid, window body, CHANNEL plane, L=14,
`window_seed=0`, weight-space, `h_diag.pt` as the diagonal Hessian. Unless a
line says otherwise the unit is `model.layers.2.mlp.down_proj` [1024, 3072] and
the rung is R2048 (8 b/wt). Nothing here is a served number.

---

## 0. Defect or cost — the question, and what decides it

The coordinator's question is whether the non-dyadic sigma is *a defect to fix*
or *a cost to record*. The walk in §4 has already moved it off sigma: the
sensitivity lives in `scale_refit`, not in the table. That relocates the
question rather than answering it, so the answer is registered here **before
the deciding arms ran**, against the 2×2 in `stage_refit`'s docstring
(`refit_metric=h`, `refit_reach_floor=True`, and both):

* **If `refit_metric=h` collapses the ratio toward 1.04** — the refit is a
  **defect**, and a nameable one. #89's whole table was measured with an
  h-blind per-row least squares and then scored on h; `refit_channel_scale`
  already takes a `metric`, and the encoder already has the diagonal Hessian
  in hand wherever a Hessian is supplied. The proposal is then on the refit,
  not on `default_channel_sigma`, and it is still a weight-space screen: served
  evidence remains owed in full (§8).
* **If neither knob moves the ratio**, and each arm's own h falls monotonically
  in the refit count while the *ratio between arms* grows — the documented
  behaviour of this alternation under `trellis_weighting="scale"`, no `ldl`,
  `metric=None` — then it is **basin sensitivity in a non-convex coordinate
  descent**. There is no smooth objective in the residue for a constant to be
  re-derived against, and the honest disposition is **a cost to record**: the
  refit's gain is residue-dependent, the default happens to sit on a residue
  where it gains 32%, and that is a fact about this unit to write down, not a
  defect to repair.

Either way `default_channel_sigma` does not move on this evidence. The two
branches differ in whether a *different* knob acquires a proposal.

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

Two scope facts, both measured, both narrowing the claim hard.

**h on this unit is six columns.** `h_max / h_median = 2.2e6`; columns 55, 128
and 1489 carry 91.9% of the metric and six carry 99.9%. These are Qwen3-0.6B's
known outlier input channels.

**In weight space the two sigmas are the same.** At every refit count the plain
`||e||/||w||` ratio is 1.0022–1.0027. The whole of #89's 1.367 is
Hessian-weighted error on six of 3072 columns.

## 3. The alphabet is not the mechanism (negative result)

**The sigma axis is a gauge up to powers of two.** `table(2σ) = 2·table(σ)`
exactly but for about 2 of 16384 entries — the innermost quantiles, where the
E4M3 grid's smallest-magnitude floor breaks its ×2 closure (12 entries differ
at σ=17.66, 2 at σ=94.18; no entry of any unclamped table lands below that
floor, so this is quantiles snapping onto the floor, not halving through it). So sigma's only free parameter is its **dyadic residue**
`log2(σ) mod 1`, and the issue's parameterisation is right. m=0.5 and m=1 share
residue 0.55735 and measure identically; m=0.75 and m=1.5 share 0.14232.

**But the residue barely moves the table's own error.** For every unclamped
sigma the reach is exactly `4.0773·σ`, so the reach-aware row scales at the two
sigmas are exactly proportional and the two normalised tables differ only in
where the snap lands. Snapped-against-ideal relative energy is `6.976e-4` at
the default and `7.031e-4` at m=0.75 — **a predicted h ratio of 1.004 against a
measured 1.367.** The table's own snapping error is not what sets the h error.

**And no part of the table carries it.** `ts89_table_surgery.py` grafts the
default table's outermost N entries (rescaled by 0.75, matched by trellis
state) into the m=0.75 table and sweeps N: 1, 2, 4, 8, 16, 32, 64, 128 entries
all read 1.0369 → 1.0368. Replacing the whole outer eighth-of-a-percent of the
alphabet with the default's own moves the reduced model by 1e-4. Whatever
distinguishes the two sigmas is not localised in the table's tail.

## 4. The mechanism: the scale refit, not the table

A reduced model isolates it. `experiments/ts89_table_surgery.py` replays
production's **first Viterbi pass alone** — exact, on the CPU, same
`trellis_weighting="scale"` branch weight — on the 64 columns that carry 99.93%
of this unit's h. It reads **1.0369**, not 1.367. What the reduction drops is
`scale_refit=4`.

Production agrees. Walking the refit count at both sigmas
(`--stage refit`, `experiments/results/ts89_refit.json`):

| refits | h (m=1) | h (m=0.75) | **h ratio** | wt ratio |
|---|---|---|---|---|
| 0 | 0.01356 | 0.01405 | **1.0367** | 1.0022 |
| 1 | 0.01239 | 0.01403 | **1.1327** | 1.0023 |
| 2 | 0.01142 | 0.01402 | **1.2274** | 1.0025 |
| 4 | 0.01024 | 0.01399 | **1.3667** | 1.0027 |

The `refit=0` arm reads 1.0367 against the reduced model's 1.0369 — a
prediction made before that arm ran, on a machine and a code path it was not
fit to, agreeing to four decimals.

**Read down the columns, not across.** Four refits buy the default sigma
**1.324× in h** and buy m=0.75 **1.0045× — nothing**. Both arms get the same
1.003× in plain weight error.

> The scale refit is a 32% h lever at the default's dyadic residue and a null
> lever at m=0.75's. #89's 1.367 is the size of a **benefit the default
> receives**, not a penalty the alternative pays.

That is a different claim from the issue's title, and it changes what a fix
would be: the quantity that responds to the residue is the alternation's
effectiveness, not the alphabet's fidelity.

## 5. What this says about `default_channel_sigma`

The brief asked me to re-derive the constant against the table's actual use if
the mechanism supported it. **It does not, and I am not proposing a change.**

`_default_sigma` picks `peak · 2^(-k/4)` over a quarter-binade ladder by
minimising the nearest-value error of a **scalar RTN quantiser** on a unit
Gaussian. #89 is right that this is the wrong question — the window body is not
a scalar RTN quantiser. But the measurement says the replacement question is
not "what spread does the table want" either. The sigma-sensitive quantity is
whether the h-blind least-squares refit happens to help the handful of columns
the Hessian cares about, and that is a property of the unit's outlier structure
as much as of the constant.

Re-deriving one per-grid constant against a landscape shaped like that would be
fitting a global default to six columns of one dense unit. **No proposal.**

**Bytes and `encoder_profile_id`, stated loudly because the brief asked.** No
change I have landed moves either. But recording the hazard for whoever does
touch this constant: `unit_artifact._normalize_reach` normalises
`channel_sigma == default_channel_sigma(grid)` to `None`, so the profile id
does **not** bind the numeric default. Changing `default_channel_sigma` would
change every default-built E4M3 artifact's bytes and decoded tensor **while
leaving its `encoder_profile_id` unchanged** — two artifacts carrying one id
and two renderings. That is a gate problem before it is a numerics problem.

## 6. #84, reconciled

Both tables are right; they measure different axes, and the coordinator's
reading that #89's `m>=1.25` arms share a realised reach is the wrong one.

* On the **ratio** axis (`rho`) `channel_sigma` is fixed at 94.18, so once the
  table clamps at 448 grid units `reach_rms = 448/94.18 = 4.7568` and never
  moves again. That is #84's flat column.
* On the **spread** axis (`m`) `channel_sigma` grows with the arm, so
  `reach_rms = 448/(m·94.18)` keeps falling: 3.8055, 3.1712, 2.3784. Those are
  #89's distinct values. Same clamp, different denominator.

The ratio axis then supplies a clean natural experiment. At ratios 1.75→4.0 the
realised reach (4.7568) and the over-reach row fraction (0.2285) are pinned
**identically**, yet R2048 h runs 1.411 → 1.458 → 1.571 → 1.706 → 1.991. **The
reach is not a sufficient statistic**; the table's interior keeps deforming
after the reach stops moving. This also corrects #84's own claim that a
clamp-blind sweep reads a flat error curve — the curve is not flat, and the
quantity that shows why is `saturated`: 4 / 36 / 358 / 4120 of 16384 entries on
the peak at ratios 1.25 / 1.5 / 2.0 / 4.0.

That last row also corrects #89's own parenthesis. The issue flags `m=2` as the
one clamping arm ("358 of 16384 entries pile on 448") and treats `m=1.25` and
`m=1.5` as clean; they are not — they clamp 4 and 36 entries. #87's agent
reached the same two counts independently. It does not change #89's conclusion
(the two 70.64 arms are genuinely clamp-free and are the ones carrying the
1.367), but it does mean that of the eight arms in the issue's table only four
are clean gauges: `m=0.5`, `m=1`, `m=0.75` and `rho=0.5`. Any reading of the
`m>=1.25` rows is a reading of the residue *and* the clamp together.

**Landed on this branch** (commit `b85e233`, closes #84's reporting half):
`window_table_reach()` returns requested vs realised reach, `delivered`,
`saturated` and `saturated_fraction`; `EncodedUnit.table_reach` records it per
encode (diagnostic, never wire); `bf16_l_sigma_sweep.py`'s `reach_stats`
reports all of it beside the realised reach. Refusing the clamp rather than
reporting it moves behaviour and is Rob's call — not landed.

## 7. Does anything else cite #80's contaminated R2048 column?

**No.** The R2048 geomean row (m0.75 1.0447, oracle 0.9640) and the predictor
table (R2048 3/8, 1.0221) appear only in
`docs/measurements/tessera-per-unit-reach-2026-09-03.md` itself and in #89,
which is the issue flagging them. No other doc, test, experiment or open issue
reads them. (`0.9640` also occurs in two unrelated results JSONs as a
coincidence of value.)

## 8. What would be needed to ship anything here, and how far short I am

Nothing in this report is a ship claim, and the gap is large:

* Every number is **weight-space h on one unit**. Principle 3 wants exact
  full-vocab vLLM KL-vs-BF16 on the served artifact at matched bpp. I have no
  served arm.
* The metric is **six columns**. Anything tuned against it is tuned against six
  outlier input channels of one Qwen3-0.6B dense unit.
* A `default_channel_sigma` change would move bytes on every default-built
  E4M3 artifact **without moving `encoder_profile_id`** (§5). Before such a
  change could ship, that normalisation would have to bind the constant, or the
  constant would have to be written into the profile explicitly.
* If the 2×2 gives the refit a proposal (§0), it would need the same ladder:
  a rung sweep on more than one unit, then a served A/B at matched bytes. A
  refit change moves no wire — `refit_metric` is an encoder argument, not a
  plane — so it is far cheaper to ship than a per-grid constant, but it is
  still a default and still owes served evidence.

---

## 9. Off-task fixes landed on this branch

Each is a separate commit so it can be taken or dropped independently.

| commit | what |
|---|---|
| `b85e233` | #84's reporting half: `window_table_reach()` returns requested vs realised reach, `delivered`, `saturated`, `saturated_fraction`; `EncodedUnit.table_reach` records it per encode (diagnostic, never wire); `bf16_l_sigma_sweep.py` reports it. Two new tests in `tests/test_window_body.py`. |
| `14880cd` | `scale_channel._default_sigma`'s docstring said the ladder was dyadic. It is a **quarter-binade** ladder (`peak · 2^(-k/4)`, forty rungs) minimising scalar-RTN nearest-value error — which is neither dyadic nor the objective #89 is about. Prose only. |
| `14880cd` | Prose in `export.py`, `tests/test_bf16_route.py`, `experiments/tessera16_alphabet_floor.py`. |

## 10. Consultations

None. No `fable-*` agent was spawned. Every open question on this ticket was a
four-encode experiment or a two-line diagnostic, and the one reduced model I
built (`ts89_table_surgery.py`) was cheaper to write than to delegate.
