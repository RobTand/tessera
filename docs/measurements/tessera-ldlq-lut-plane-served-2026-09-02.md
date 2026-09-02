# The 4-bit route goes activation-aware: LDLQ and an H-solved block-scale refit on the LUT plane (2026-09-02)

**Claim.** *(filled from the served run below)*

Commit: see `git log` for this file. Tests: `tests/test_ldlq_lut_plane.py`,
`tests/test_ldlq_window.py`, `tests/test_merge_guard.py`.

## Why this was blocked, and why the block was wrong

`tessera-ldlq-window-served-2026-09-02` landed both encoder-side levers on the
FP8 route and said, in `encode_unit`, that they were CHANNEL-plane mechanisms:

> LDLQ is implemented for the CHANNEL scale plane; a block plane's
> per-column-span scales would have to be scheduled with it

That reason does not survive reading the loop it guards. The plane is read
**once per pass**, before the block loop (`scale = current_scale()`), and refit
**once after it**; every block of every pass quantises against the same fixed
plane whatever its column granularity. The schedule and the plane never
interleave, so there is nothing to schedule together. The refusal was
conservative, not load-bearing, and it was lifted by implementing the missing
half rather than by bypassing the check — the LUT plane's refit genuinely did
not read a metric, and that half is new code.

Both bodies the 4-bit grid ships are covered, and
`test_identity_factor_is_the_plain_pass_on_the_lut_plane` pins the claim on each
at blocks 32/64/128: a factor with no off-diagonal blocks reproduces the
ordinary whole-matrix pass **bit for bit** — codes, scale table, indices and the
reported sse. That is the statement that block-sequential encoding is a
*schedule* and not a second encoder, and it is the same test the CHANNEL plane
already carried.

## The block-scale refit: JSO's knob, solved with H instead of searched

The CHANNEL plane's refit had one scale per output row, so its optimum was one
scalar quadratic per row. A block plane has one scale per **sixteen input
columns of one row** — NVFP4's own layout — and those blocks are coupled by
every off-diagonal of `H`. The closed form is the same projection restricted to
a block's columns.

Row `r` reconstructs as `w_hat_r = sum_b s_rb u_rb`, where `u_rb` is the row's
unscaled grid values on block `b` and zero elsewhere. The proxy loss is

```
L = sum_r (w_r - w_hat_r) H (w_r - w_hat_r)^T
```

`H` couples input features, so *columns* interact and *rows* do not — which is
why every step below is exact per row. Differentiating in one block's scale
with the others held:

```
dL/ds_rb = -2 (w_r - w_hat_r) H u_rb^T
s*_rb    = (r_rb H u_rb^T) / (u_rb H u_rb^T),   r_rb = w_r - sum_{b' != b} s_rb' u_rb'
```

the projection of the block's residual onto its own codes in the `H` inner
product. Written incrementally — the form actually computed — with
`G = (W - S*U) H` the current gradient field and `A_rb = u_rb H u_rb^T`:

```
s*_rb = s_rb + (G_rb . u_rb) / A_rb
```

`A` needs only `H`'s **diagonal 16x16 blocks**; the off-diagonal coupling enters
through `G`. At `H = I` the cross terms vanish because blocks have disjoint
support, `A = <u,u>`, `G.u = <w,u> - s<u,u>`, and the form collapses to
`<w,u>/<u,u>` — the plain refit, exactly. `test_an_identity_metric_is_the_plain_refit`
pins that collapse.

**This is JSO's question with `H` in place of the grid search.** Joint scale
optimisation asks which per-block scale minimises an activation-weighted error
and answers it by trying a small ladder of levels; here the answer is the
stationary point of that same error, in closed form, with no ladder and no level
to choose (principle 2).

**Landing stays exact and needs no rounding rule of its own.** The plane's
sixteen table entries are exact E4M3 bytes, and with the codes fixed a block's
error is a parabola in its scale, so nearest-in-*linear* distance to `s*` is the
exact minimiser among them (`_nearest`) — not nearest in log distance, which is
what an E4M3 rounder would do. `land_at_least`'s round-up exists for a *floor*,
a CHANNEL-plane mechanism that raises a **row** scale so the row's loudest
weight stays inside the body's reach; a block scale already tracks its own
sixteen weights' amax, so there is no floor here and `refit_reach_floor` is
refused under a block plane rather than silently ignored.

### Three approximations, and the guard that charges for them

The step is a *coordinate* optimum, so honesty about what is not exact:

1. Every block moves at once — the vector step is Jacobi, not the joint
   minimiser. Corrected by an exact **per-row step length** `t*` along the
   direction, which is the true minimiser on that line (rows are independent
   under `H`) and is identically 1 when the metric is separable.
2. `_fit_lut` chooses the sixteen table entries under the separable
   second-order model `sum_b A_b (c_b - s*_b)^2` — the expansion around the
   current plane with cross-block terms dropped.
3. Assignment is nearest-in-linear to `s*`, the exact per-block optimum given
   the others.

The accept test is where the cross terms come back: the candidate planes are
scored on the **full quadratic**, and the plane the unit already has is one of
the candidates, so the step is monotone in the metric's own error and the
alternation with the trellis stays monotone
(`test_the_metric_refit_is_monotone_under_its_own_metric`).

**The failure this design had to avoid, and the test that would catch it.** A
Jacobi step that the guard rejects on every pass would leave the "+ refit" arm
encoding *identically* to the arm without it, raising nothing — the same class
of silent no-op the previous commit closed for dropped kwargs.
`test_the_metric_refit_moves_scales_and_lowers_the_metric_cost` asserts both
halves (indices move, and the exact `H`-cost strictly falls) on a Hessian with
real off-block coupling, and the sweep harness prints `!! IDENTICAL BYTES`
whenever two arms of one unit materialise the same tensor.

## The Hessian, and what it may not have seen

The same capture as the FP8 receipt — `experiments/capture_h_full.py`, fp32
accumulation of `x^T x` per Linear from a BF16 HF forward with hooks.

| field | value |
|---|---|
| source | wikitext-2-raw-v1 **train** (local `datasets` cache), 295,562 chars |
| text sha256 | `a5c5fd091a3486361e71eae1132fff141aeadd0ae51ebf688da4661752f853d3` |
| fit slice | 16,384 tokens, `seqlen` 512, ids sha256 `229c6f72307f7050…` |
| eval slice | 8,192 tokens from the next offset, ids sha256 `30ec7255e6934172…` |
| Linears | 196 (every `*_proj` under `model.layers`) |

The served KL corpus is `corpus_qwen_n8_s512.json` — wikitext-2 **test**
(`source_sha256 076d33efc447…`). Train and test are disjoint splits, so no arm
was fit on text it is graded on, and the weight-space sweep is scored on the
**eval** slice, disjoint from the fit slice the Hessian and the refit were built
from.

## The exact objective loses to the diagonal one, on its own quadratic

The CHANNEL plane's measured default is the **exact** `hessian` objective: on
the FP8 wire it beat every diagonal power (0.5982x vs h^1.0's 0.6376x,
`tessera-ldlq-window-served-2026-09-02.md`). On the LUT plane it is the other
way round, and the reason is not the one a reader would guess.

`experiments/ldlq_lut_qwen_hfit.sh`, `layers.0.self_attn.q_proj`, every arm
carrying both scores -- `out` is the held-out eval slice, `hfit` is
`sqrt(E H E^T / W H W^T)` on the **fit** rows, the quadratic the refit is
provably monotone in:

| arm | out | hfit | s |
|---|---|---|---|
| baseline (no LDLQ, plain refit) | 0.06053 | 0.06006 | 27 |
| LDLQ 1.0/32 | 0.05149 | 0.05025 | 179 |
| refit h^1.0 only | 0.05417 | 0.05393 | 29 |
| refit full-H only | 0.05584 | 0.05541 | 19 |
| **LDLQ 1.0/32 + refit h^1.0** | **0.04375** | **0.04283** | 173 |
| LDLQ 1.0/32 + refit full-H | 0.04822 | 0.04699 | 156 |

Two readings, and only one of them survives:

* **It is not the accept guard.** Every refit arm lands *below* the baseline on
  `hfit`, so nothing raises the quantity the guard scores. The guard holds.
* **It is not generalisation.** `hfit` and `out` agree to within 1-3% on every
  arm, i.e. the 16k-token Hessian transfers to the held-out slice; and the
  ordering is the same in both columns. The full-H refit is worse **on the fit
  rows' own quadratic** -- the objective it alone is solving exactly.

What is left is the optimiser, and the receipt's own list of approximations
already names it. Under a **diagonal** metric the sixteen-column blocks
decouple completely: the coordinate step is the joint minimiser (`t* = 1`
exactly), and `_fit_lut`'s separable model `sum_b A_b (c_b - s*_b)^2` is not a
model but the cost itself. Under the **full** H neither is true: the vector
step is Jacobi corrected by a single per-row step length, and the table fit
drops every cross-block term. So the alternation with the trellis -- which
minimises its own SSE, not this quadratic -- converges to a worse point of the
full-H objective than the diagonal path reaches of it.

The CHANNEL plane has no such gap: one scalar per row, rows independent under
H, so its full-H refit *is* the exact minimiser and there is no table to fit.
**The exactness that matters is the optimiser's, not the objective's** -- and
that is a property of the plane, which is why the default below is per plane
and not per repo.

**Open lever, not tried:** a Gauss-Seidel sweep (update `G` after each block
rather than stepping every block from the same residual) is the standard fix
for exactly this, and would plausibly make the full-H objective win here too.
The per-plane default records what was measured *with this optimiser*.

## The decision rule, written before the numbers

Three candidate recipes for "what an exporter does on the LUT plane when it is
handed a Hessian", chosen before the six-unit geomeans landed so that the rule
is not fitted to them:

| candidate | LDLQ | refit objective |
|---|---|---|
| `plain` | 1.0 / 32 | none (the weights-only refit) |
| `h^1.0` | 1.0 / 32 | the diagonal of H, normalised |
| `hessian` | 1.0 / 32 | the exact quadratic -- the CHANNEL plane's default |

**The rule.** The default is the candidate with the better Qwen six-unit
out-space geomean, *subject to* a GLM six-expert geomean no worse than 1.00x
against the same wire without levers -- the coordinator's gate, which exists
because the E2M1 route's wins today are on GLM's experts and must not be paid
for out of them. If the Qwen geomeans are within 1% (unit 2 already separated
two candidates by 0.5%), GLM breaks the tie. `plain` stays in the table even
where it loses, because it is the arm that most obviously cannot regress and a
reader needs to see the size of what the refit buys over it.

The served leg gates separately and on its own terms: **served KL better than
0.640** (the same recipe, weights-only, same A4 scales, same teacher) at
matched bytes.

## What an H-aware encode costs on this body

The FP8 receipt measured LDLQ at ~2x the plain encode on the window body. On
the **TCQ** body it is far worse, and it scales with the row's width:

| unit | cols | plain | + LDLQ 1.0/32 | factor |
|---|---|---|---|---|
| Qwen `layers.0.self_attn.q_proj` | 1024 | 18.0 s | 114.5 s | 6.4x |
| Qwen `layers.1.self_attn.k_proj` | 1024 | 14.4 s | 171.4 s | 11.9x |
| GLM `L5.gate_proj` | 4096 | 12 s | 519 s | 43x |

The refit is cheap by comparison (20 s on the GLM expert): the cost is the
**body**, not the plane, and the mechanism is the box. LDLQ splits a row into
`cols/block` sequential segments and calls the trellis once per segment
(`trellis_pass(span_cols=...)`), so a pass that made one `viterbi_columns` call
per distinct rate now makes one per rate *per segment* -- 32 times on a
1024-column row, 128 on a 4096-column one. The columns are independent, so the
total *work* is unchanged; what multiplies is the per-call launch overhead, and
this box is launch-bound (34.5 W of a ~140 W envelope at 96% "utilization",
whatever the job count). That is also why the window body pays only ~2x for the
same schedule: `viterbi_window` is the fused kernel, and the TCQ body's Viterbi
is not fused.

The consequence is concrete and it is a limit on the route, not a footnote.
Both exports walk the same 196 units in the same order, so their own progress
lines are the measurement:

| Qwen3-0.6B E2M1x2 @ q896, 196 units | at `[20/196]` | whole model |
|---|---|---|
| weights-only | 361 s | 3538 s (18 s/unit) |
| `--hessian --refit-metric h^1.0` | 3311 s | ~32,500 s extrapolated (166 s/unit) |

**~9.2x, i.e. ~9 h for a 0.6B model.** The factor grows with the row width
because LDLQ's cost is the segment count, so it is 43x on a 4096-column GLM
expert: `export_glm53_tessera.py --hessian` at block 32 on those experts is not
a practical whole-model export at all. Two follow-ups, neither
done here: shard the export across processes the way `export_glm53_tessera.py`
already does (the box is launch-bound, so N processes over disjoint layer
ranges should be close to N times faster, and the merge guard exists to make
the halves provably one artifact), or fuse the TCQ Viterbi as the window body's
already is.

## Weight space: the sweep

*(table)*

## Served (the gate) -- IN FLIGHT, NOT YET MEASURED

**Unmet at the time of writing.** The arm is exported and served by two
commands, and the first one takes ~9 h on this box (launched 16:52, `[20/196]`
at 3311 s, so it lands ~01:55) for the reason the section above measures. This section says exactly where it is so that the
leg can be finished by anyone, including after this session ends.

| what | value |
|---|---|
| arm | `ldlqH1` -- `export_tessera_serving.py <src> --grid E2M1x2 --q256 896 --input-scales scales_pqcal.safetensors --stock-twin ... --hessian h_full_qwen06b.pt --refit-metric h^1.0` |
| why the flag is explicit | the per-plane default was set by *this* receipt; naming the objective keeps the arm meaning the same before and after that change. `test_the_default_on_this_plane_is_the_arm_that_was_measured` pins that the default now writes the same bytes |
| launched by | `experiments/ldlq_lut_export_arms.sh` (it also launched a full-H arm, killed once the weight-space screen had chosen -- the task asks for ONE export) |
| log | `/mnt/shared/tessera-runs/ldlq-lut/export_ldlqH1.log` |
| output | `/mnt/shared/tessera-runs/ldlq-lut/ldlqH1-stock-twin` |
| to finish | `experiments/ldlq_lut_serve.sh ldlqH1` -> `/mnt/shared/tessera-runs/ldlq-lut/kl_ldlqH1.json` |
| already armed | a detached waiter on sparklina (`/home/rob/tmp/ldlq_arm_serve.sh`, log `serve_ldlqH1_chain.log`) blocks on the export pid and then runs exactly that command, so the leg lands without a session attached. It takes `serve_lock.sh` like every other serve on this box, on port 8001 under `TESSERA_KL_NAME=tessera-kl-ldlqlut`, and refuses rather than serving a partial twin |
| comparators | **0.640** (`kl_unrot-k2-w4a4-pqcal.json`, `all.kl_lower_mean`; same recipe, weights-only, same A4 scales, same teacher, same box) and **0.511** (PrismaQuant NVFP4 GPTQ+JSO at 4.5 bpp) |
| the gate | served KL better than 0.640 at matched bytes |

The baseline leg is already controlled: `base` (weights-only, the pre-merge
exporter) and `base2` (weights-only, the *arm's own* post-merge exporter) are
byte-compared against each other and against the served comparator, so the
served delta is attributable to the Hessian and not to the merge that landed
between them. That compare is measured, not assumed:

```
base2-stock-twin vs base-stock-twin           784 shared, 784 identical, 0 different
base2-stock-twin vs unrot-k2-w4a4-pqcal       784 shared, 784 identical, 0 different
```

so the served **0.640** is the baseline arm's own number under the arm's own
code, and byte identity for a weights-only encode survives every change in this
branch.

## GLM cross-check

*(table)*

## The gate

*(table)*

## Scope, and what is not measured

*(list)*

## Files

Code: `src/tessera/encode.py` (`_refit_scales_lut_metric`, the lifted LDLQ
refusal, the re-scoped `refit_metric`/`refit_reach_floor` refusals),
`src/tessera/export.py` (`ActivationSource.from_capture`, the per-plane
`DEFAULT_REFIT_OBJECTIVE` and `objective_for`, and the two library call sites
that now hand the encode the plane from the same resolved recipe),
`experiments/export_tessera_serving.py` and `experiments/export_glm53_tessera.py`
(`--hessian`, `--refit-metric` defaulting to the measured per-plane map rather
than to a constant).

Measurement: `experiments/ldlq_window_sweep.py` (`--grid`, the `hfit` column),
`experiments/tessera_window_wire.py` (the levers ride the TCQ arm, `--no-window`,
`hfit`), and the run scripts that pin each arm's exact flags --
`ldlq_lut_qwen_hfit.sh`, `ldlq_lut_glm_h1.sh`, `ldlq_lut_glm_hess.sh`,
`ldlq_lut_chain.sh`, `ldlq_lut_export_arms.sh`, `ldlq_lut_export_base2.sh`,
`ldlq_lut_serve.sh`.

Tests: `tests/test_ldlq_lut_plane.py` (21), `tests/test_ldlq_window.py` (the
defaults are the measured ones), `tests/test_merge_guard.py` (a per-plane map
that disagrees only off its own plane still refuses).
