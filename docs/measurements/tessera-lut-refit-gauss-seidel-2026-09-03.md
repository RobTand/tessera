# The LUT refit's Jacobi step was the gap: a Gauss-Seidel sweep clears the screen (2026-09-03)

**Claim, weight space (measured, a screen and not a result).**
`tessera-ldlq-lut-plane-served-2026-09-02` found the exact full-Hessian refit
objective *losing* to the diagonal `h^1.0` one on the LUT plane -- on
`layers.0.self_attn.q_proj`, the unit its table shows, and on four of the six
units it swept; on the **six-unit geomean** it was 1.38% *ahead*, a margin
carried entirely by `L2.mlp.down_proj`. It ruled out the accept guard and
generalisation and named the optimiser: the vector step is **Jacobi**, every
16-column block stepping from the same residual. It filed the fix and did not
take it (issue #35). Taken, on the same six dense Qwen3-0.6B units in one
process at the E2M1x2 q896 cap wire: a **Gauss-Seidel sweep** takes the full-H
refit to **3.73% ahead** of the served `h^1.0` default on the out-space geomean
(**0.04922 against 0.05113**, where Jacobi reached 0.05043), ahead on `hfit` too
(**0.04620 against 0.04855**), and -- unlike Jacobi -- ahead on four of the six
units rather than on one.

**On the receipt's own unit the sign flips.** `layers.0.self_attn.q_proj` was
the case the mechanism was argued on -- full-H 0.04822 out / 0.04699 hfit
against `h^1.0`'s 0.04375 / 0.04283. Under Gauss-Seidel the same arm is
**0.04296 / 0.04188**: below the default on both columns, from 10.2% above it.

**Both roles that carry the remaining served gap improve.** The
196-Linear census (`tessera-bf16-gauge-and-dense4-residual-2026-09-02`) put the
dense 4-bit residual entirely in `q_proj` (1.0648) and `k_proj` (1.0976), with
the other five roles already winning. Both improve here: `q_proj` **0.9819x** and
`k_proj` **0.9320x** against the served default -- `k_proj` the second-largest
single-unit gain of the six, `q_proj` the fourth. Both are gains; neither is the
largest, which is `L2.mlp.down_proj` at 0.8869x.

**Nothing here is served.** This is a weight-space screen on six units. Under
principle 3 the gate is the serve, and the next step is a re-export at
byte-identical bytes against the same BF16 teacher in the **same** vLLM session
as a re-run of the current default, because KL drifts 4-8x across sessions.

Commit: see `git log` for this file. Code: `src/tessera/encode.py`
(`_refit_scales_lut_metric`, `refit_diagnostics`). Tests:
`tests/test_ldlq_lut_plane.py` (27 pass), `tests/test_ldlq_window.py` +
`tests/test_merge_guard.py` (45 pass). Data:
`/mnt/shared/tessera-runs/ldlq-lut/qwen_lut_gs.json`, read by
`experiments/gs_refit_report.py`.

## What changed, and what did not

`refit_gauss_seidel` solves block `b` against a residual that already carries
blocks `0..b-1`, instead of stepping every block from one `G`. It costs one
`E @ H`'s worth of flops split into `nb` panels, and it keeps the existing
exact per-row line search, computed against the gradient field **at `S`** so
that `t*` is the minimiser of the true line through `S` under either optimiser.
That is the only difference between the two arms: same LDLQ factor, same block,
same metric, same table fit, same accept guard, same wire. Two treatments in one
arm would have made the comparison say nothing.

**No byte of the wire moves.** With the option off the encode is byte for byte
what it was: `experiments/gs_refit_byte_baseline.py` hashes fifteen artifact
blobs -- the real `q_proj` plus two synthetic shapes, five arms each, including
the full-H arms that run the branch the change edits -- and reports **0 changed
of 15**, twice (before the change against after it, and again after the
diagnostic sink was added). A matrix covering only the plain and diagonal paths
would have proved nothing about the branch that moved.

It is refused where it would be the parallel step under another name: with no
`refit_metric`, under a 1-D one (the blocks decouple exactly, so a sequential
sweep computes the parallel numbers), and off the LUT plane.

> **Superseded 2026-09-04 (#103).** This paragraph used to end "it is **not**
> offered through `export.ActivationSource`, so no exporter can set it and no
> `tessera_config.json` can record a refit the merge guard has no field to
> compare." `refit_gauss_seidel` is now a field on `ActivationSource`, rides
> into the exported config, and the merge guard compares it -- the "adding
> both, in one change" this asked for. Default off, and off is byte for byte
> the encode that was already there. What has **not** changed is the evidence:
> this is still a weight-space screen with no serve behind it.

## The drift control

The served default ran as the **first** arm of every unit and again as the
**last**, after every other arm had run, in one process. Byte-identical on all
six, and 0.0000% on both columns:

| unit | out first | out last | hfit first | hfit last | bytes |
|---|---|---|---|---|---|
| `L0.self_attn.q_proj` | 0.04375 | 0.04375 | 0.04283 | 0.04283 | IDENTICAL |
| `L1.self_attn.k_proj` | 0.04567 | 0.04567 | 0.04477 | 0.04477 | IDENTICAL |
| `L13.mlp.down_proj` | 0.08348 | 0.08348 | 0.07793 | 0.07793 | IDENTICAL |
| `L14.mlp.gate_proj` | 0.05304 | 0.05304 | 0.05093 | 0.05093 | IDENTICAL |
| `L2.mlp.down_proj` | 0.02686 | 0.02686 | 0.02677 | 0.02677 | IDENTICAL |
| `L27.self_attn.o_proj` | 0.07519 | 0.07519 | 0.06427 | 0.06427 | IDENTICAL |

A **third** replicate of the same arm sits in the middle of every unit's table
(the sweep's own `LDLQ 1.0/32 + refit h^1.0`) and agrees to every digit. So the
noise floor of this table is zero and every gap below is a real difference, not
a re-run. The six-unit control geomean also reproduces the 2026-09-02 receipt's
numbers exactly -- `q_proj` 0.04375 / 0.04283, baseline 0.06053 / 0.06006 --
across sessions and across a code change.

`experiments/ldlq_lut_qwen_hfit.sh`, the issue's single-unit harness, was run
**separately** on `q_proj` and reproduces the six-unit run's every arm to five
decimals in a different process -- GS 0.04296 / 0.04188, Jacobi 0.04822 /
0.04699, control 0.04375 / 0.04283
(`/mnt/shared/tessera-runs/ldlq-lut/qwen_lut_hfit_gs.json`). Cross-process as
well as within-process.

## The arms

Six Qwen3-0.6B units, one process, E2M1x2 at `q256=896` (the 4.0 bpp TCQ cap
wire, LUT plane), LDLQ `sigma=1.0` `block=32`, `out` on the held-out eval slice
and `hfit` = `sqrt(E H E^T / W H W^T)` on the fit rows.

| arm | out geomean | hfit geomean |
|---|---|---|
| **LDLQ 1.0/32 + refit full-H (Gauss-Seidel)** | **0.04922** | **0.04620** |
| LDLQ 1.0/32 + refit full-H (Jacobi) | 0.05043 | 0.04759 |
| LDLQ 1.0/32 + refit h^1.0 *(served default, x3)* | 0.05113 | 0.04855 |
| LDLQ sigma=1.0 block=32 | 0.05348 | 0.05086 |
| refit full-H only (Gauss-Seidel) | 0.05795 | 0.05621 |
| refit full-H only (Jacobi) | 0.06004 | 0.05914 |
| refit h^1.0 only | 0.06029 | 0.05994 |
| baseline (no LDLQ, plain refit) | 0.06551 | 0.06519 |

## Per unit -- because a geomean is exactly the statistic that can hide this

The six units are one per role except `down_proj`, which appears twice; there is
no `up_proj` and no `v_proj` in this capture, so "per role" here is per unit.

| unit | ctl out | jac out | **gs out** | gs/ctl | gs/jac | ctl hfit | jac hfit | **gs hfit** | gs/jac |
|---|---|---|---|---|---|---|---|---|---|
| `L0.self_attn.q_proj` | 0.04375 | 0.04822 | **0.04296** | **0.9819** | 0.8908 | 0.04283 | 0.04699 | **0.04188** | 0.8911 |
| `L1.self_attn.k_proj` | 0.04567 | 0.04542 | **0.04256** | **0.9320** | 0.9370 | 0.04477 | 0.04432 | **0.04137** | 0.9332 |
| `L13.mlp.down_proj` | 0.08348 | 0.08596 | 0.08404 | 1.0067 | 0.9777 | 0.07793 | 0.07992 | **0.07758** | 0.9707 |
| `L14.mlp.gate_proj` | 0.05304 | 0.05473 | **0.05123** | 0.9657 | 0.9360 | 0.05093 | 0.05239 | **0.04868** | 0.9291 |
| `L2.mlp.down_proj` | 0.02686 | **0.02072** | 0.02383 | 0.8869 | 1.1501 | 0.02677 | **0.02038** | 0.02340 | 1.1480 |
| `L27.self_attn.o_proj` | 0.07519 | 0.07709 | 0.07585 | 1.0088 | 0.9840 | 0.06427 | 0.06532 | **0.06351** | 0.9723 |
| **geomean** | 0.05113 | 0.05043 | **0.04922** | **0.9627** | 0.9760 | 0.04855 | 0.04759 | **0.04620** | 0.9708 |

Three things the geomean would have hidden, in both directions:

* **The two roles that matter most both improve.** `q_proj` and `k_proj` are the
  only two roles the 196-Linear census has losing to NVFP4 GPTQ+JSO at 4.0 vs
  4.5 bpp, and both gain here: `k_proj` 0.9320x (the second-largest gain of the
  six) and `q_proj` 0.9819x (the fourth). The largest gain is `L2.mlp.down_proj`
  at 0.8869x, which is not one of the two. So the lever does land on the part of
  the residual that is left -- but it is not concentrated there, and the ordering
  does not say it prefers those roles.
* **Two units regress slightly against the served default on `out`** --
  `L13.mlp.down_proj` at 1.0067x and `L27.self_attn.o_proj` at 1.0088x -- while
  *improving* on `hfit` (0.9707x and 0.9723x against Jacobi, and below the
  control). Sub-1% and in the direction of a generalisation gap, not a guard
  failure: the refit lowers the quadratic it is solving and the held-out slice
  does not follow it all the way.
* **The prior six-unit full-H "win" was one unit.** In the 2026-09-02 sweep
  Jacobi full-H beat `h^1.0` on the geomean while losing on **four of six**
  units (it wins only `L1.self_attn.k_proj`, 0.9947x, and `L2.mlp.down_proj`,
  0.7714x); without `L2` its geomean *loses* by 3.6%. So the **1.38% span the
  promotion bar was set from is itself a single-unit artefact**, and the bar was
  low for that reason -- the Gauss-Seidel arm clears it by 2.7x. Two counts that
  must not be blurred: GS beats the **control** on `out` on **four of six**
  units, and beats **Jacobi** on **five of six**.

`L2.mlp.down_proj` is the one unit where Gauss-Seidel is **worse than Jacobi**
(1.1501x out, 1.1480x hfit) -- while still beating the served default at
0.8869x. See the mechanism below: it is the only unit with any
non-positive-target reverts, but that is **not** where it loses -- it loses at
the step.

## The promotion rule, as written, against the numbers

> GS goes to a serve only if it beats the served `h^1.0` default on the `out`
> geomean by more than the 1.38% the two refit objectives currently span, and
> stays monotone on `hfit`.

* **Margin.** Control `out` geomean 0.0511311; the bar is `control / 1.0138 =
  0.0504351`. (The multiplicative form `control x 0.9862 = 0.0504256` differs by
  0.02% and does not change the answer.) GS reaches **0.0492236**, i.e.
  **-3.73%** against the control. **Clears**, by 2.7x the required margin. The
  bar and the Jacobi arm's own geomean (0.0504342) agree to six significant
  figures -- that is arithmetic, not luck, because the 1.38% *was* the gap
  between those two arms -- so "clears the bar" and "strictly beats Jacobi" are
  the same test here, and GS passes it (0.9760x).
* **Monotone on `hfit`, reading (i) -- the guard.** Every refit arm at or below
  the baseline on `hfit`, per unit, which is the receipt's own definition and
  the property the accept test provides: **6 of 6, holds.** The GS arm is also
  below the control on `hfit` on 6 of 6.
* **Monotone on `hfit`, reading (ii) -- GS at or below Jacobi, per unit:**
  **5 of 6.** `L2.mlp.down_proj` fails at 1.1480x. Reading (ii) is not the rule
  as filed, and it is recorded here rather than averaged away.

**Verdict: the screen is cleared.** Under the rule as written the arm is
promoted to a serve. It is **not promoted to a default** by this document: the
2026-09-02 receipt's own per-plane decision rule requires a GLM six-expert
geomean no worse than 1.00x against the same wire without levers -- the E2M1
route's wins today are on GLM's experts and must not be paid for out of them --
and no GLM arm was run here. **A screen is not a result** (principle 3).

## The mechanism: the Jacobi step was real, and the landing is what is left

`refit_diagnostics()` splits every refit call three ways -- the **step**
(direction plus line search), the **revert** (blocks whose optimum is a
non-positive scale keep the one they have), and the **landing** (`_fit_lut`'s
separable model, which drops every cross-block term, plus nearest-in-linear
assignment into sixteen table entries). Fractions of each pass's starting cost;
`survives` is the share of the step's own gain still present after the other two.
Pass 0 of each unit, the two arms side by side:

| unit | arm | step | revert | landing | survives | reverted |
|---|---|---|---|---|---|---|
| `L0.q_proj` | jacobi | 44.47% | 0.00% | 27.27% | 38.68% | 0 |
| `L0.q_proj` | **gs** | **56.85%** | 0.00% | 29.93% | **47.35%** | 0 |
| `L1.k_proj` | jacobi | 37.24% | 0.00% | 22.48% | 39.64% | 0 |
| `L1.k_proj` | **gs** | **43.07%** | 0.00% | 19.60% | **54.48%** | 0 |
| `L13.down_proj` | jacobi | 11.53% | 0.00% | 10.47% | 9.18% | 0 |
| `L13.down_proj` | **gs** | **19.01%** | 0.00% | 15.48% | **18.57%** | 0 |
| `L14.gate_proj` | jacobi | 30.29% | 0.00% | 21.42% | 29.29% | 0 |
| `L14.gate_proj` | **gs** | **37.83%** | 0.00% | 20.53% | **45.74%** | 0 |
| `L2.down_proj` | jacobi | **94.67%** | 1.65% | 29.59% | 66.99% | 879 |
| `L2.down_proj` | gs | 88.02% | 4.40% | 20.80% | **71.37%** | 934 |
| `L27.o_proj` | jacobi | 7.32% | 0.00% | 6.38% | 12.84% | 0 |
| `L27.o_proj` | **gs** | **16.49%** | 0.00% | 11.51% | **30.21%** | 0 |

Three findings, and the third is the one worth keeping:

1. **The receipt's diagnosis was right.** The sweep removes strictly more error
   at the step on five of six units, and by a lot: `o_proj` 16.49% against
   7.32%, `down_proj` L13 19.01% against 11.53%, `q_proj` 56.85% against 44.47%.
   The Jacobi step was a real approximation and the standard fix fixes it.
2. **The revert is not a factor on real trellis codes.** It fires on exactly one
   of six units. (A synthetic fixture with random codes reverts ~40% of blocks
   and eats the whole step gain; that fixture over-reads it, and the unit test
   built on it asserts on the step alone for this reason.) The one unit that
   does revert -- `L2.mlp.down_proj`, 879-1896 blocks per pass, rising each
   pass under Jacobi and 934-1537 under GS -- is also the unit where GS loses to
   Jacobi. It is tempting to call that the cause. **The table says otherwise, so
   it is not claimed.** On `L2` the GS *step* is the **smaller** of the two on
   all four passes (88.02/86.58/86.59/86.71% against 94.67/93.54/93.38/93.12%),
   and GS reverts **fewer** blocks than Jacobi in three of the four (1006 v 1037,
   1252 v 1433, 1537 v 1896). GS loses on `L2` **at the step leg, before revert
   or landing** -- the single per-row line search resolves a summed sequential
   direction to one scalar, and on this unit it prefers Jacobi's
   diagonally-preconditioned direction, exactly the `(G.D)^2/(D H D)` collapse
   the implementation probe exposed. `L2` is also the unit where the *step* removes
   ~90% of each pass's starting cost, i.e. the plane starts furthest from its
   optimum --
   the regime where one step length per row is the poorest stand-in for a step
   length per block. What *is* true of the reverts is that GS's are individually
   costlier (4.40-7.57% of the pass cost against 1.65-3.01%): a secondary effect,
   coincident with the loss, not shown to cause it.
3. **The landing is now the dominant approximation, and it always was.** It
   takes back **24-91% of the step's gain** across the twelve pass-0 rows above
   (and up to 94% on a later pass), above half on nine of the twelve --
   `o_proj` gives back 6.38 of 7.32 points under Jacobi and 11.51 of 16.49 under
   GS; the two lowest are `L2.mlp.down_proj` at 31.3% and 23.6%, the unit whose
   plane starts furthest from optimum. The `_fit_lut` fit is separable by construction (`sum_b A_b (c_b -
   s*_b)^2`, diagonal-block curvatures only, cross-block terms dropped) and the
   assignment is nearest-in-linear into sixteen entries. So of the three
   approximations the 2026-09-02 receipt listed, **the one it named as the fix
   was worth 3.7% out-space, and the one left is several times larger.** A
   cross-block-aware table fit is the next lever on this plane, and it is a
   bigger one than the step ever was.

`survives` improves under GS on all six units **in pass 0** (the table above),
even where the landing takes more back in absolute terms, because the step it is
taking from is bigger. That does not hold for the whole schedule: in passes 1-3
GS survives *less* than Jacobi on two of the six -- `L0.q_proj` and
`L2.mlp.down_proj` -- which is what spending more of the available gain in the
first pass looks like, not a second effect.

## What this does not claim

* **Nothing is served.** No vLLM run, no KL, no PPL. Weight space only.
* **No default changes**, on any plane. The per-plane refit objective is
  untouched: `hessian` on CHANNEL, `h^1.0` on `lut16`. Gauss-Seidel is an
  opt-in encoder argument reachable only from `encode_linear`/
  `encode_linear_planes`, and an exporter cannot set it at all.
* **No GLM evidence.** Six dense Qwen3-0.6B units. The 2026-09-02 decision rule
  makes a GLM six-expert geomean the gate for a LUT-plane default, and no GLM
  arm was run. A dense-Qwen result does not transfer to experts and the reverse
  is equally false (`tessera-bf16-gauge-and-dense4-residual-2026-09-02` is
  explicit about this).
* **No `up_proj` and no `v_proj`**, and only one layer each of `q`, `k`, `gate`
  and `o`. The per-role reading here is six units, not the 196-Linear census.
* **One wire, one rung.** E2M1x2 at `q256=896`. Nothing about the sub-cap window
  body, the E4M3/CHANNEL wire, or any other rung.
* **One `(sigma, block)`**, 1.0 / 32. No interaction with the LDLQ regulariser
  was swept.
* **No cost claim.** The box was running several other agents' full test suites
  throughout; no wall-clock or throughput number from this run means anything
  and none is quoted. The `s` column in the log is recorded for provenance and
  is not a measurement.
* **The unit-test step numbers are not a quality claim.** The synthetic-fixture
  figures (13.67 against 14.65) are evidence that the option is Gauss-Seidel,
  nothing more.
