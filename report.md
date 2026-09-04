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
