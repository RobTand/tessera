# Issue #50 — the LUT table fit, bounded attempt

Branch `muse/ts-50-lutfit`, off `master` at `82cdf51`.

## Pre-registration (written before the arm ran)

Recorded here, and committed, before any arm number existed, because the stop
rule requires that no arm be selected after seeing its own score.

### The two funded mechanisms do not exist on the arm the ceiling was measured on

The decision funds, in order: (1) re-derive `landing=grid`; (2) a
curvature-weighted fit — "weight each block's assignment error by `A_b`"; (3) a
coordinate pass over the sixteen entries with the cross-block terms retained.
Reading the code, (2) and (3) are already answered, and (3) is answered by the
shipped objective being separable rather than by anything the fit does.

**(2) is already implemented, and the half of it that is not is a no-op.**
`_refit_scales_lut_metric` builds `weights = where(valid, A, 0)` and passes it
straight into `_fit_lut` as `flat_w`; `_lut_cost` is `sum_b w_b * gap_b^2`, and
the greedy's elimination step is an `index_add` of `w * (alt^2 - here^2)`. So
the *table* is already chosen under the block curvature. The *assignment* is
`_nearest`, i.e. nearest-in-linear — and weighting that by `A_b` cannot change
it: with the table fixed, `argmin_k A_b (t_b - c_k)^2 = argmin_k |t_b - c_k|`
for every `A_b > 0`. A per-block positive constant does not move a per-block
argmin.

**(3) has nothing to retain on the incumbent.** The shipped arm is
`refit h^1.0`, whose metric is 1-D, so `_refit_scales_lut_metric` takes the
`metric.ndim == 1` branch, where `cost(C) = sum(A*C^2 - 2*B*C)` — separable by
construction, one term per block, no `E @ H` and no cross-block term. Cross
terms exist only under the full-H arms (Jacobi, Gauss-Seidel), which
`tests/test_plane_promotion.py` pins as *not* the default. The ~5.0% ceiling in
the decision is the `h^1.0` row, so it is measured where neither funded
mechanism exists.

A corollary worth stating: because the incumbent's objective is separable,
`grid ⊇ table` makes the grid landing a **per-pass proof-level bound** on any
sixteen-entry table, not merely a close estimate. The decision's caveat
("defensible estimate, not a proof") was about cross terms, which the
incumbent does not have.

### The arm this substitutes, and why it is the honest one

What the shipped objective *does* leave is the **solver**.
`sum_b w_b (s_b - nearest(table, s_b))^2` over sixteen values drawn from a
sorted finite candidate set is a one-dimensional weighted k-median, which has
an exact dynamic program; `_fit_lut` answers it with greedy backward
elimination plus swap passes. That is a heuristic where an explicit exists.

`_fit_lut_exact` is the explicit (prefix sums of `w`, `w s`, `w s^2` in
float64; shortest path over the candidates; `O(16 n^2)`, `n <= 119`), and
because it is exact it splits the `table -> grid` gap with no fraction left to
argue about:

* whatever it returns is **the solver's** share;
* whatever remains is **the sixteen-entry budget**, i.e. the four-bit index —
  which is a wire change and outside this issue's premise.

**The arm:** `LDLQ 1.0/32 + refit h^1.0 + exact-16 fit`, `landing=table`, the
same six dense Qwen3-0.6B units, E2M1x2 `q256=896`, control triplicate in one
process. One arm. `--stage exact-fit` in `experiments/lut_landing_ceiling.py`.

### The prediction, and the number it is based on

A CPU-only screen on the six units' *initial pack* distribution (real Qwen
weights, `amax/peak` targets and energy weights — `_pack_scales_lut`'s own
inputs, not the refit's) says the exact solver captures a small share of that
distribution's `table -> grid` gap **in the fit objective**:

| unit | exact/greedy fit cost | solver share of `table -> grid` |
|---|---:|---:|
| `L0.q_proj` | 1.000000 | 0.0% |
| `L1.k_proj` | 0.959570 | 6.6% |
| `L13.down_proj` | 0.960161 | 7.2% |
| `L14.gate_proj` | 1.000000 | 0.0% |
| `L2.down_proj` | 0.962099 | 6.9% |
| `L27.o_proj` | 0.991945 | 2.4% |

So the greedy is *not* exact on real data — it leaves 0–4% of the fit
objective on four of six units — but it is a small share of the gap the ceiling
measures. **I expect the stop rule to fire**, and the finding to be that the
`table -> grid` gap is the sixteen-entry budget rather than the fit. This is a
screen on the wrong distribution (the pack's, not the refit's) and in the wrong
space (the fit objective, not held-out `out`), so it is a prediction and not
the result. The arm is run either way.

### What is deliberately not built

* **`_pack_scales_lut` keeps the greedy.** The exact fit is scoped to the
  refit, which is where `lut_landing` removes the landing, so the arm and the
  ceiling are measured at the same locus. Changing the initial pack would also
  move the trellis's starting scales and stop being a landing measurement.
* **A cross-block coordinate pass for the full-H arms.** It is a real
  mechanism *there* (their `cost` is `((Ec @ H) * Ec).sum()`), and their own
  `table -> grid` gap is 5.7%. But they are not the incumbent, and building it
  after the exact fit fires the stop rule would be exactly the rescue spend the
  decision rules out. It is a proposal in this report, not a build.

---

## Result

**The stop rule fires.** The exact sixteen-entry fit returns **−0.515%** of a
**4.94%** ceiling whose half is **2.47%** — not merely under the bar, but on
the wrong side of zero. No default is proposed and none should be.

### Step 1 — the `landing=grid` column, re-derived

`experiments/results/tessera_lut_landing_ceiling.json` (+ `.log`), eleven arms
× six dense Qwen3-0.6B units, E2M1x2 `q256=896`, LDLQ 1.0/32, held-out `out`
geomeans, control encoded first and last with bit-identical reconstructions on
every unit.

| arm | `landing=table` | `grid` | `none` |
|---|---:|---:|---:|
| `refit h^1.0` (shipped) | 1.0000 | **0.9506** | 0.7846 |
| `refit full-H` (Jacobi) | 0.9868 | 0.9470 | 0.7060 |
| `refit full-H` (Gauss-Seidel) | 0.9635 | 0.9090 | 0.7277 |

The decision's table reproduces to within 0.0008 on all six of its entries,
and the `grid` column it recorded from a transcript (~0.9501) lands at 0.9506.
So the split stands as decided and **the stop rule keeps the threshold it was
given**:

* **`table → grid` = 4.94%** — the ceiling for a cleverer sixteen-entry fit
* **`grid → none` = 17.46%** — the E4M3 value set itself, i.e. a wire change

For the record, the two non-default objectives have their own gaps:
Jacobi 4.03%, Gauss-Seidel 5.66%.

### Step 2 — the arm

`experiments/results/tessera_lut_exact_fit.json` (+ `.log`). Same six units,
same wire, `landing=table`, control encoded three times in one process.

| unit | ratio | |
|---|---:|---|
| `L0.self_attn.q_proj` | 1.03056x | table changed, **worse** |
| `L1.self_attn.k_proj` | 1.00000x | greedy already optimal, bit-identical |
| `L13.mlp.down_proj` | 1.00000x | bit-identical |
| `L14.mlp.gate_proj` | 1.00000x | bit-identical |
| `L2.mlp.down_proj` | 1.00075x | table changed, **worse** |
| `L27.self_attn.o_proj` | 1.00000x | bit-identical |
| **six-unit `out` geomean** | **1.00515x** | |

Control triplicate spread **0.0000%**, three bit-identical reconstructions, so
a 0.5% arm-to-control gap is comfortably readable at this scale.

### The mechanism, in the fit's own quadratic

`experiments/lut_landing_split.py` reads `refit_diagnostics`' third leg,
`continuous → landed` — the landing itself. Its costs are *negative* (the 1-D
metric's cost is `sum(A*C² − 2*B*C)`, the quadratic with its constant
dropped), so they must be **differenced, never divided**; the quantity with a
meaning is the loss the landing adds, `landed − continuous ≥ 0`.

On the **first refit pass** — the one pass where both arms enter with the same
`continuous` optimum, so the comparison is like for like:

| unit | control loss | exact loss | removed |
|---|---:|---:|---:|
| `L0.q_proj` | 5.8000 | 5.3809 | **7.23%** |
| the other five | — | unchanged | 0.00% |
| **total** | 13.8560 | 13.4368 | **3.03%** |

So the exact DP **strictly wins the objective `_fit_lut` states**, on identical
inputs — and held-out `out` still comes out 0.515% worse.

Later passes are excluded from that table on purpose: an arm that changes the
table in pass 1 hands pass 2 a different starting plane, so the summed columns
(`step won` 64.23 vs 63.30, `landing lost` 48.62 vs 48.21) are not comparable
pass for pass.

## What this retires

Not "the optimiser was not clever enough". The measurement says the objective
is the problem:

* On four of six units **the greedy already attains `_fit_lut`'s minimum** —
  the DP returns the same bytes. There is no solver slack to spend there at
  all.
* On the two where the DP finds a strictly better table under that objective,
  **held-out `out` goes up**, once by 3.06%.

`_fit_lut` minimises `sum_b A_b (c_b − s*_b)²`: a weight-space quadratic in
the scale plane, weighted by the refit's local curvature, with the trellis
re-encode that follows it **not represented at all**. Minimising it harder
moves away from the error that ships.

That retires the decision's step 3 as well. A coordinate pass over the sixteen
entries retaining cross-block terms is *the same objective optimised further
still* — and on the shipped `h^1.0` arm there are no cross-block terms to
retain, because a 1-D metric's cost is separable by construction. It is not
"unfunded because the stop rule fired"; it is measured to be pointed the wrong
way.

**The 4.94% `table → grid` gap is therefore not a fit problem.** What is left
inside it is the sixteen-entry budget — four bits of index — which is a wire
change and outside this issue's premise. The 17.46% below it is the E4M3 value
set, likewise a wire change.

## What would be worth doing instead (proposals, not builds)

1. **Make the landing's objective the one that ships.** The honest form of
   #50's question is not "fit sixteen numbers better" but "fit them to
   something that predicts held-out output error". The rendered-error scorer
   already exists elsewhere in the tree; scoring a candidate table by it,
   rather than by the separable weight-space proxy, is a different experiment
   with a different premise and needs its own decision.
2. **Leave `_fit_lut` alone.** It is optimal on most real units and cheap to
   be wrong on the rest, and its errors do not point the way the objective
   claims.

Neither is started here. The stop rule fired and the deliverable is the
negative result.

## The promotion gate

**`assert_plane_promotion` was not invoked, deliberately.** It has no
candidate to read: nothing here is proposed as a default, there is no GLM
six-expert measurement, and the gate's first leg is a GLM bar. Running it to
produce a refusal I already know the shape of would be theatre. The honest
statement is that no default is proposed, so the gate has nothing to judge.

## Byte and test evidence

* **Byte baseline** (`tessera_ts50_byte_baseline.log`): fifteen encodes,
  three shapes × five recipes, master against this branch with
  `refit_lut_exact` at its default of off — **0 changed of 15**. The exact
  solver is reachable only through an argument nothing in the tree passes.
* **If it ever became the default it would move bytes** on real units — two of
  six here. That is a wire-adjacent decision and Rob's to make, not something
  to land quietly. The tests assert the *shape* of that (smooth input →
  byte-identical, clustered input → differs) rather than being adjusted to
  match new bytes.
* **Targeted tests** (`tessera_ts50_targeted_tests.txt`), per AGENTS.md
  `1f7836c`: sixteen files — the two the diff touches plus the LUT / refit /
  landing / byte-baseline / grammar / promotion neighbours — **321 passed,
  exit 0**, no failures, so none needed the second run against master. The
  pre-fix failure line for each of the five added tests is recorded there;
  every one fails on master for the reason it exists.
* **Provenance** (`tessera_ts50_run_fingerprint.txt`): both runs came from one
  rsynced copy, nothing rsynced into it mid-run, base deliberately still
  `82cdf51` — rebasing onto `dc83ee1` (#100) mid-run would split the six units
  across two builds, which is what `require_same_build`'s new `image_digest`
  leg exists to refuse.

## Off-task fixes, one line each

* `d489387` — the stop-rule line printed `1 − ratio`, so a 0.5%-**worse** arm
  printed `-0.5155% against the control`; it now names the direction in words.
* `d489387` — `--verdict-only`: the step-1 JSON landed after the arm run, so no
  verdict was stamped though both numbers existed; the verdict is now
  computable from two committed JSONs with nothing re-encoded, through the one
  function both paths call.
* `a872ed1` — the first landing-split reader **divided** the diagnostics'
  negative costs, which is meaningless; it differences them now.

## An unhedged timing observation, deliberately not a claim

The DP ran the refit at roughly half the greedy's wall clock (`q_proj` 70.2 s
vs 133–142 s; `k_proj` 37.5 s vs 75 s), because `_fit_lut`'s elimination and
swap passes are a Python double loop calling `_lut_cost` many hundreds of
times. **This is a wall clock on a contended box with no profile either side,
so under principle 15 it is not a speed claim** — only a reason someone might
want to take one.

## Consultations

* **advisor** (stronger reviewer, full transcript) — asked how to spend the
  wait on the encode arms. It steered the targeted-test ordering, warned that
  an `!! IDENTICAL RECONSTRUCTION` on *some* units was expected rather than a
  bug, predicted a held-out regression was possible even under a monotone
  per-pass fit, and pointed at `refit_diagnostics`' third leg as the
  mechanism-level reading. All four held up against the measurement; the third
  leg became the decisive evidence. No Fable consultation was needed — the
  question turned out to be measurable rather than hard.
