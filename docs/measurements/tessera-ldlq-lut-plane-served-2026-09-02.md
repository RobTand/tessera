# The 4-bit route goes activation-aware: LDLQ and an H-solved block-scale refit on the LUT plane (2026-09-02)

**Claim, weight space (measured).** Both encoder-side levers now run on the
LUT plane, so the 4-bit wire can be encoded H-aware for the first time.
**LDLQ is the lever and the refit objective is second order**: on six Qwen
units LDLQ alone is **0.8164x** against the same wire with no levers, the
refit alone is 0.9203x, and the choice between the two refit objectives moves
1.38%. On six GLM experts **LDLQ alone closes 39% of the wire's out-space gap
to EXL3 K=4** -- 1.1779x to **1.0957x** -- at the same 4.0 bpp and the same
bytes.

**Claim, served (measured).** **The gate passes.** On Qwen3-0.6B at the
E2M1x2 q896 cap, served on vanilla vLLM against the same BF16 teacher, the
H-aware wire scores **KL 0.5310** against the weights-only wire's **0.6404**
at **identical bytes** (both 220,301,312 wire bytes, 4.0018 bpp) -- **0.829x**,
and 0.813x on the confident subset, with top-1 agreement up from 58.76% to
62.52%. Against PrismaQuant's NVFP4 GPTQ+JSO at 4.5 bpp (0.511) the H-aware
4.0 bpp wire is 1.039x behind on **11% fewer bits**, closing most of a gap that
was 1.253x before this work.

**Cost.** An H-aware encode is **19.2x** the weights-only one, measured as a
matched pair on a quiet box. Three earlier figures in this receipt -- 9.2x,
1.72x, 1.34x -- were each a ratio between two exports run at different times
under different background load, and all three are retracted below.

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
the candidates, so the **step** is monotone in the metric's own error
(`test_the_metric_refit_is_monotone_under_its_own_metric`, which asserts the
step and only the step). The *alternation* with the trellis is not monotone
here and this doc originally said it was: the Viterbi's branch metric never
sees `metric`, so an H-weighted refit and the trellis pass after it descend
different errors by construction. Corrected 2026-09-02 with the math audit's
§5; no number in this doc moves, because none of them was read off that
claim.

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

> **Tried, 2026-09-03 (issue #35) -- and the diagnosis above was right.**
> `tessera-lut-refit-gauss-seidel-2026-09-03` sweeps the blocks sequentially and
> the sign flips on this very unit: `layers.0.self_attn.q_proj` goes from
> **0.04822 out / 0.04699 hfit** to **0.04296 / 0.04188**, below the `h^1.0`
> default's 0.04375 / 0.04283. Over the same six units the full-H objective goes
> from **1.38% ahead of the default on the geomean but ahead on only one unit**
> (the `L2.mlp.down_proj` margin above) to **3.73% ahead, on four of six**
> (out geomean 0.04922 vs Jacobi's 0.05043 vs the default's 0.05113). It is a **weight-space screen**, opt-in and encoder-side, with no
> byte of the wire moved and no serve behind it, so **the per-plane default in
> the next section stands unchanged** until a served A/B and a GLM six-expert
> arm exist. That receipt also finds what the step was hiding: the separable
> `_fit_lut` table fit gives back 24-91% of *either* optimiser's gain (above half
> on nine of twelve pass-0 rows), and is now the larger of the two approximations
> by several times. It also finds that the one unit where the sweep loses to
> Jacobi loses **at the step**, not at the non-positive-target revert.

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

**19.2x, measured as a matched pair on a quiet box.** Three earlier figures in
this receipt -- 9.2x, then 1.72x, then 1.34x -- were all wrong, and all wrong
the same way: every one of them divided one export by a *different* export that
had run at a different time against a different background load. I never ran
the control. `base3`, which the coordinator scheduled after reading the 1.34x,
was the first weights-only export on a quiet box, and it broke everything
downstream of it.

The control, run afterwards on one box minutes apart, identical flags, two
layers (14 units), with the weights-only arm repeated *after* the H-aware one
so that box drift would show up as disagreement between the two rather than as
a factor:

| arm | total | per unit | power |
|---|---|---|---|
| weights-only (before) | 18 s | 1.29 s | 11.72 W |
| **`--hessian --refit-metric h^1.0`** | **345 s** | **24.64 s** | 11.77 W |
| weights-only (after) | 18 s | 1.29 s | 11.91 W |
| | | **19.2x** | |

The two weights-only arms agree exactly, so the box did not drift. A second,
independent route gives the same answer: `ldlqH1`'s own quiet stretch (units
81-180, 24.34 s/unit) against `base3`'s quiet 1.30 s/unit is **18.7x**. Two
routes, ~19x. Projected to the whole model, a quiet H-aware export of
Qwen3-0.6B is ~4900 s against the weights-only 255 s; the `ldlqH1` artifact
actually took 7949 s because parts of it were contended.

### What was actually wrong, twice

Two hypotheses were tested and **both are disproved**, which is why the answer
is the box after all:

- **Not `0739f33`** (the fused/reference rate crossover). One tree, one
  variable: `TESSERA_WINDOW_FUSED_MAX_RATE=999` restores the always-fused
  behaviour. Smokes 24 s vs 23 s -- identical. The code says the same:
  `viterbi_window` is reached only under `if body is BodyKind.WINDOW`
  (`encode.py:1378`), and `wire_recipe(E2M1x2, 896)` is `BodyKind.TCQ`.
- **Not the tree.** `693a3c2` -- the exact commit `base`, `base2` and `ldlqH1`
  ran on -- exported to a separate directory and run back to back against the
  current tree: old 17 s, new 18 s, old again 17 s, bytes identical 56/56. The
  old encoder runs at 1.2 s/unit today.

So `base`/`base2`'s 15.1-18.1 s/unit was three concurrent sweeps, a **~13x
penalty on the weights-only export**. Which refutes, flatly, a sentence I wrote
in the previous version of this section: *"the weights-only export barely
notices contention (`base` 3542 s and `base2` 2966 s under similar load)."*
They were similar to each other because **both were contended**. I compared two
treatments and reported it as a control.

That is also why 1.34x came out so small: it divided a 13x-inflated baseline by
a partly-inflated H-aware run. The inflation cancelled, and what survived
looked like a cheap lever. The lesson is not "beware contention" -- it is that
**a ratio between two runs is only a measurement if something pins the
conditions**, and back-to-back-plus-repeat is what pins them. Nothing in the
first three figures did.

One consequence worth stating for the route rather than for me: at 19.2x the
encode cost is a real constraint again. It is affordable on a 0.6B dense model
(~1.4 h), and the GLM expert cost -- withdrawn above as unmeasured -- has now
been measured. See the next section.

### The GLM expert cost, and the dial nobody had turned

Measured 2026-09-02 on **sparklina** (the second GB10), load 0.6-1.5 throughout,
nothing else on the box. One GLM-5.3-Flash expert, `L5.gate_proj` (2048x4096),
the shipping 4.0 bpp wire (`E2M1x2` TCQ q896). Every rung is listed twice in one
process so the first pass absorbs CUDA/Triton first-launch cost and the second
is the measurement; **both passes agreed exactly at every point**, and the
quality columns are bit-identical to the same arms run on sparky, so the encoder
is deterministic across boxes and only the seconds are in question.

**The denominators are not the same as 19.2x and must not be quoted together.**
19.2x is a whole *export*, factorisation included, over 14 mixed-width units.
The figures here are `encode_linear` + `read_unit_artifact` for ONE unit with the
LDL factor precomputed outside the timed region (`tessera_window_wire.py` caches
it in `factors`), while production refactors per tensor (`export.py:370`).

| cols | segments | plain | LDLQ sigma=1 | factor | per segment |
|---|---|---|---|---|---|
| 1024 | 32 | 1.01 s | 22.35 s | 22.1x | **0.699 s** |
| 2048 | 64 | 1.42 s | 44.66 s | 31.4x | **0.698 s** |
| 4096 | 128 | 2.56 s | 89.89 s | 35.1x | **0.702 s** |

Contention was worth ~15%: the same 4096 arm read 104 s on sparky at CPU user
80-86% (three audit workers), against 90 s here. Every earlier figure in this
receipt was taken under that load and is an upper bound.

**The factor widens with the tensor, and it is the denominator that moves.**
LDLQ is exactly linear in segments -- 0.699/0.698/0.702 s each, flat to 0.6%
across a 4x width range. The *plain* arm is sublinear (1.01 -> 2.56 s for 4x the
columns) because the fused Viterbi amortises its fixed cost. So "the cost grows
with the segment count", the sentence this section used to carry, is right about
LDLQ and wrong about what it implies: LDLQ costs what everything costs, linear
in the tensor. What grows is the ratio to a baseline that is getting better.

**And the block size is a free dial, which nobody had turned.** Same tensor,
same sigma, varying only `ldl_block` -- which changes how many segments the pass
uses without changing how many columns get encoded:

| `ldl_block` | segments | LDLQ | out-space | a4 (W4A4) | per segment |
|---|---|---|---|---|---|
| **32** (`DEFAULT_LDLQ_BLOCK`) | 128 | 89.89 s | 0.07867 | 0.12069 | 0.702 s |
| 64 | 64 | 45.16 s | 0.07868 | 0.12071 | 0.706 s |
| 128 | 32 | **23.24 s** | **0.07864** | **0.12066** | 0.726 s |

**3.9x faster at block 128, with quality flat -- marginally better, not worse.**

That table is also the mechanism experiment this receipt has been missing. A
block-128 segment encodes four times as many columns as a block-32 segment and
costs the **same 0.70-0.73 s**. Total device work is constant down the column
(all 4096 columns are encoded either way) while total time falls 3.9x. Fitting
the three points: **`time = 0.694 s x segments + 1.0 s`** -- so 96 extra
segments buy 66.6 s of nothing, and the per-segment cost is invariant to the
work inside the segment.

That is a controlled result, not a timing shape: the work per segment was varied
4x and the time per segment did not move. It establishes the *class* -- a fixed
cost per segment dominates, not the work -- and it **refutes the growing-operand
term** the Qwen profile proposed, which predicts wider segments costing more and
the total going super-linear in columns. Neither happens.

What it does **not** establish is what that fixed 0.694 s *is*. The GLM profile
was attempted and is not in this receipt: `torch.profiler` over a 128-segment
encode reached **121 GB of a shared 121 GB pool** and was killed before it could
OOM the box. Re-run it on a 512-column unit (16 segments) if the identity
matters. Until then this receipt names the class and not the culprit -- which is
the distinction the previous version of this section failed to make in the other
direction.

**What this changes.** `DEFAULT_LDLQ_BLOCK = 32` (`export.py:155`) appears to
cost ~4x for nothing on this shape. The open question on issue #13 is therefore
not "is 30x affordable" but "why is the default 32", and the answer needs (a)
the sweep repeated on a dense tensor and a second expert, and (b) a check that
`ldl_block` is not observable in any stored byte, fingerprint or cache key --
if it is, moving the default makes every existing receipt irreproducible and is
not a free perf change.

### Where the time goes -- profiled, and the hypothesis was wrong

The first version of this section asserted a mechanism: that LDLQ's cost is
per-call overhead rather than work, because the pass splits a row into
`cols/block` sequential segments and calls the trellis once per segment, so the
columns' independence keeps the total work fixed while the call count
multiplies. `experiments/ldlq_cost_profile.py` was written to check it.
**It is false.**

Qwen `layers.1.self_attn.k_proj` (1024x1024, 32 segments), quiet box, warmed,
`torch.profiler` with a power sampler alongside:

| arm | wall | device busy | GPU idle | device launches | power |
|---|---|---|---|---|---|
| weights-only | 3.09 s | 0.73 s | 76.3% | 197,436 | 18.6 W |
| refit `h^1.0` only | 2.85 s | 0.87 s | 69.5% | 223,294 | 16.1 W |
| LDLQ 1.0/32 | 71.36 s | **7.19 s** | 89.9% | 6,965,809 | 15.0 W |
| LDLQ + refit `h^1.0` | 68.26 s | 7.35 s | 89.2% | 4,875,235 | 15.3 W |

**Device-busy time goes up 9.85x.** The work is not unchanged: LDLQ does an
order of magnitude more GPU work, which kills the original "the columns are
independent, so the work is fixed" hypothesis outright.

But it does not account for the whole slowdown either, and the previous version
of this section claimed it did. That claim came from matching the profiled
device delta (+6.46 s on one unit) against a wall delta of +6.23 s/unit taken
from the contaminated export series -- two numbers that agreed to 4% and were
both being compared to the wrong thing. Against the **matched quiet pair**, the
wall delta is +23.35 s/unit (24.64 against 1.29), so:

| | per unit | share of the wall delta |
|---|---|---|
| extra device work | +6.46 s | ~28% |
| everything else (launch, host, sync) | ~+16.9 s | ~72% |

**Both mechanisms are real, and neither of my two single-cause stories was.**
The GPU does ~10x the work *and* the loop gets more launch-bound while doing it
-- GPU idle rises from 76.3% to 89.9%, on 35x the launches. A wall factor of
19.2x on a device factor of 9.85x is exactly what a widening idle fraction
looks like.

Where the extra device work goes is not the Viterbi. The LDLQ arm's top device
ops are gather-and-scatter: `aten::index` 591,135 calls / 1.146 s,
`index_elementwise_kernel` 328,745 / 0.726 s, `aten::copy_` 562,409 / 0.562 s,
`aten::min` 131,077 / 0.356 s, then `sub` and `remainder`. That is the
per-segment residual materialisation (`ldlq_target[:, start:stop] = base +
residual @ ldl_factor[stop:, start:stop]`, a matmul that grows with the segment
index) plus re-indexing the plane once per segment -- real work the plain pass
never does. Anyone optimising this should start there and not at the Viterbi,
and should expect to recover at most the 28%: the other 72% is launch count,
which is a fusion problem.

Two things the profile says that are **not** the cost story, recorded so they
are not misread. The profiled *wall* ratio is 23x; that is a profiler artifact
(7 million recorded events at ~10 microseconds of instrumentation each) and
must not be quoted as the encode's cost -- the matched pair's 19.2x is.
And the refit is free: 2.85 s against the weights-only 3.09 s, inside the noise.

Finally, the power column is its own finding. **Every arm draws 15-19 W of a
~140 W envelope** at 23-33% reported utilization, and the GPU sits idle 76-90%
of the wall in all four. The encoder -- levered or not -- uses about an eighth
of this box. The headroom is the other half of the 19.2x: three quarters of
the H-aware slowdown is idle GPU, and nothing -- not the fused window Viterbi,
which this body does not reach -- has closed the launch-bound gap here.

The 43x on a 4096-column GLM expert was measured inside a sweep with the same
contention problem. It has since been re-run on a quiet box and is **35.1x**
encode-only at `ldl_block=32`, falling to **9.0x** at block 128 with quality
flat -- see "The GLM expert cost, and the dial nobody had turned" above. The
43x is superseded, not merely withdrawn.

The `ldlqH1` artifact took 7949 s wall, but it was contended for part of that;
a quiet whole-model H-aware export projects to ~4900 s against the weights-only
255 s.


## Weight space: the sweep

Six Qwen3-0.6B units, E2M1x2 at the q896 cap (the 4.0 bpp wire), scored on
`out` -- output-space error through 8192 held-out eval rows the refit never
saw. Geomean over the six, and the ratio against the same wire with no levers:

| arm | out (geomean) | vs baseline |
|---|---|---|
| baseline (no LDLQ, plain refit) | 0.06551 | 1.0000x |
| LDLQ sigma=0.3 block=32 | 0.05277 | 0.8055x |
| **LDLQ sigma=1.0 block=32** | 0.05348 | 0.8164x |
| LDLQ sigma=3.0 block=32 | 0.05489 | 0.8380x |
| refit h^0.5 only | 0.06267 | 0.9566x |
| refit h^1.0 only | 0.06029 | 0.9203x |
| refit full-H only | 0.06004 | 0.9165x |
| LDLQ 1.0/32 + refit h^0.5 | 0.05261 | 0.8031x |
| LDLQ 1.0/32 + refit h^1.0 | 0.05113 | 0.7805x |
| LDLQ 1.0/32 + refit full-H | 0.05043 | **0.7699x** |

**LDLQ is the lever.** It is worth 0.82x on its own; the refit adds ~0.04x on
top of it and is worth 0.92x alone. Both together are 0.77-0.78x. `sigma=1.0`
is not the best sigma here (0.3 edges it, 0.8055x vs 0.8164x) but it is the
value the window receipt measured and the one carried forward; the difference
is 1.3% and re-tuning sigma was not this task.

### The geomean picks the arm four of six units reject

Per unit, `out` and -- because it is the tell -- `plain`, the *unweighted*
weight-space error:

| unit | LDLQ+h^1.0 out | LDLQ+full-H out | LDLQ+h^1.0 plain | LDLQ+full-H plain |
|---|---|---|---|---|
| `layers.0.self_attn.q_proj` | **0.04375** | 0.04822 | 0.10155 | 0.10274 |
| `layers.1.self_attn.k_proj` | 0.04567 | **0.04542** | 0.09842 | 0.10247 |
| `layers.13.mlp.down_proj` | **0.08348** | 0.08596 | 0.09788 | 0.09966 |
| `layers.14.mlp.gate_proj` | **0.05304** | 0.05473 | 0.09862 | 0.10109 |
| `layers.2.mlp.down_proj` | 0.02686 | **0.02072** | 0.10643 | **0.22229** |
| `layers.27.self_attn.o_proj` | **0.07519** | 0.07709 | 0.09729 | 0.10247 |

`h^1.0` wins four of six units and loses two: `k_proj` by 0.5% and
`layers.2.mlp.down_proj` by 23%. The full-H geomean win is carried entirely by
that second unit, where it is 23% better on
`out` -- and 2.3x **worse** on `plain`, against a 0.095-0.110 band every other
unit and every other arm sits in. The refit-alone arm on that unit is 3.0x
worse on `plain` (0.28714). Nothing else in the sweep moves `plain` by more
than 15%.

That is the signature of a scale set that has bet the unit's absolute accuracy
on the calibration Hessian being right about which directions matter. `out` is
held-out rows, so it is not fit-row overfitting; it is the stronger claim that
the eval rows agree with the fit rows about the loud directions, which is
exactly the assumption a served corpus is free to break.

## GLM cross-check

Six GLM-5.3 experts, same wire, same rung, each levered arm against the same
wire with no levers. The gate is <= 1.00x:

| arm | out | vs plain | a4 | vs plain | vs EXL3 K4 (out) |
|---|---|---|---|---|---|
| E2M1x2 TCQ q896, no levers | 0.07935 | 1.0000x | 0.11687 | 1.0000x | 1.1779x |
| + LDLQ 1.0/32 | 0.07381 | **0.9302x** | 0.11314 | 0.9681x | 1.0957x |
| + refit h^1.0 only | 0.07926 | 0.9989x | 0.11681 | 0.9995x | 1.1766x |
| + LDLQ 1.0/32 + refit h^1.0 | 0.07390 | **0.9313x** | 0.11319 | 0.9684x | 1.0970x |
| + refit full-H only | 0.08382 | **1.0564x** | 0.11988 | 1.0258x | 1.2443x |
| + LDLQ 1.0/32 + refit full-H | 0.07822 | 0.9858x | 0.11604 | 0.9928x | 1.1612x |

Read three ways:

- **LDLQ carries the GLM gate**, 0.9302x, and it closes 39% of the wire's gap
  to EXL3 K=4 on these experts (1.1779x -> 1.0957x out-space).
- **The refit is a no-op on GLM's experts** at `h^1.0`: 0.9989x alone, and it
  costs LDLQ 0.1% when stacked. That is the flat-`diag(H)` prediction -- these
  experts' inputs are near-Gaussian, a flat metric is the plain refit, and the
  same fact is why rotation measured dead here.
- **The full-H refit regresses GLM**, 1.0564x alone, and it gives back most of
  LDLQ's win when stacked (0.9858x against LDLQ's own 0.9302x). It passes the
  <= 1.00x gate only because LDLQ drags it under.

## The gate

| candidate | Qwen six-unit out | GLM six-expert out vs plain | GLM gate <= 1.00x | served KL vs 0.640 |
|---|---|---|---|---|
| LDLQ only | 0.8164x | 0.9302x | pass | not served |
| LDLQ + `h^1.0` | 0.7805x | 0.9313x | pass | **0.5310, pass (0.829x)** |
| LDLQ + `hessian` | **0.7699x** | 0.9858x | pass (barely) | not served, and will not be |

**Both of the coordinator's gates are met by the shipped arm**: served KL
0.5310 < 0.640 at matched bytes, and GLM six-expert 0.9313x <= 1.00x. The
default-when-H-supplied on the LUT plane stands on a serve and a cross-check,
not on a screen.

### The two verdicts, side by side

| | rule's pick | shipped arm |
|---|---|---|
| objective | `hessian` | `h^1.0` |
| Qwen six-unit `out` geomean | **0.7699x** | 0.7805x |
| Qwen units won, of six | 2 | **4** |
| worst unit's unweighted weight error (`plain`) | **0.22229** | 0.10643 |
| same, refit alone | **0.28714** | 0.09844 |
| every other unit and arm sits in | 0.093 -- 0.110 | 0.093 -- 0.110 |
| GLM refit alone, vs plain | **1.0564x** (regresses) | 0.9989x (no-op) |
| GLM stacked on LDLQ, vs plain | 0.9858x | **0.9313x** |
| GLM gate <= 1.00x | pass, on LDLQ's back | pass |
| served evidence | none, and none will exist | the gate below |

The weight-error column is the argument, and it is not in the rule. `hessian`
buys its 23% on that one unit by putting the unit's weights 2.3x further from
their true values and trusting the calibration Hessian to say that does not
matter. That is fitting the H's rows rather than the weights.

**Applied literally, the pre-registered rule selects `hessian`.** The Qwen
geomeans are 1.38% apart, which is outside the 1% band that would have sent the
tie to GLM, and both candidates clear the GLM constraint. I am recording that
verdict rather than the one I would now prefer, because the rule was written
before the numbers and re-writing it after them is the failure it exists to
prevent.

What the rule did not anticipate, and what the table above shows, is that its
deciding margin is one unit of six, that the same unit shows a 2.3x blowup on
the unweighted error, and that on GLM the objective the rule picks is the one
that regresses the route's existing win. A rule that reads a geomean cannot see
either of those.

**The export was `h^1.0`, not `hessian`** -- it was launched off a three-unit
partial screen while the last three units were still encoding, and the ordering
inverted when they landed. **The serve has since settled it in favour of what
shipped**: `h^1.0` clears the served gate at 0.5310 against 0.640, and it is
the only arm with served evidence. The code default (`lut16: h^1.0`) is now
carried by a serve rather than by a screen. Whether `hessian` would have served
better is unmeasured and stays unmeasured -- the coordinator's instruction was
not to relaunch.

### The deviation, owned

I did not relaunch, and the code default stays `lut16: h^1.0`. The reasoning,
so it can be overruled on a word rather than discovered later:

- The task budgeted **one** export and said not to tune the arm until it wins.
  Killing the export to swap the refit objective is both a second export and a
  tune. (I believed at the time that it cost 9 h; it costs 19.2x the
  weights-only encode, ~1.4 h whole-model. The cost was never the load-bearing part of this reason,
  and the coordinator's answer when asked was "do not relaunch", which settles
  it either way.)
- The default's gate is **served KL against 0.640**, not this screen. The rule
  above chose which candidate to *export*; it was never authorised to set a
  default by itself. A 1.38% weight-space edge cannot set a default here --
  only the serve can.
- So setting the code default to `hessian` would ship a default chosen on a
  screen, for an arm that will never be served, and would leave the arm that
  *does* carry served evidence undeclared. That is the worse of the two
  incoherences.
- And under either served outcome, relaunching buys nothing: if `h^1.0` misses
  0.640 the finding is that the H-aware LUT route does not clear its gate, and
  `hessian` -- which regresses GLM 1.0564x on its own -- is not the arm that
  rescues it.

This is a deviation from my own rule, not a reading of it. It is recorded here
rather than resolved, and one instruction reverses it.

## What this branch changes by default

## Served (the gate) -- MEASURED, PASSES

**Met.** Qwen3-0.6B, E2M1x2 at the q896 cap, exported with
`--hessian --refit-metric h^1.0`, stock twin served on vanilla vLLM (W4A4
NVFP4 kernels) against the same BF16 teacher, same corpus, same positions,
same metric identity as the comparator (`prismaquant.kl_compare/2`, top-1024,
teacher-student intersection, 4088 positions, corpus `076d33efc447`, tokenizer
`76f13c8e6e55`):

| arm | bpp | wire bytes | KL (all) | KL (confident) | top-1 agree |
|---|---|---|---|---|---|
| E2M1x2 q896, weights-only | 4.0018 | 220,301,312 | 0.640404 | 0.548641 | 58.76% |
| **+ LDLQ 1.0/32 + refit h^1.0** | 4.0018 | 220,301,312 | **0.531028** | **0.446066** | **62.52%** |
| ratio | -- | identical | **0.829x** | **0.813x** | +3.76 pp |
| PrismaQuant NVFP4 GPTQ+JSO | 4.5 | -- | 0.511 | -- | -- |

**The gate passes.** Better than 0.640 at matched bytes -- and the bytes are
matched exactly, not approximately: both exports write 220,301,312 wire bytes
at 4.001823 bpp, and the byte compare shows every quantised tensor the same
shape and dtype with only its contents changed (`q_proj.weight_packed`
differing fraction 0.3252, `k_proj.weight_scale` 0.2123, and so on). The
delta is the Hessian and nothing else: the weights-only baseline re-exported
by this arm's own code (`base2`) is byte-identical to the served comparator,
784/784.

Two readings worth separating. Against the **same wire without levers** the
H-aware encode is worth 1.206x, and that is this receipt's result. Against
**PrismaQuant's NVFP4 GPTQ+JSO at 4.5 bpp** (0.511) the 4.0 bpp wire is 1.039x
behind on 11% fewer bits -- it does not win, but the gap that opened this task
at 1.253x (0.640 vs 0.511) is now 1.039x, closed by the weight leg alone with
no change to the wire, the reader, or the activation leg.

**The screen predicted this one.** Weight space said 0.7805x on the six Qwen
units; the serve delivered 0.829x -- same direction, same rough size. That is
worth recording precisely because the opposite keeps happening on this project,
and one agreement does not make the screen trustworthy.

The arm is exported and served by two commands. The export was launched 16:52 and, once the box went quiet at 17:54,
finished in 7949 s at 19:05, not the ~01:55 the retracted 9.2x implied. This section says exactly where it is so that the
leg can be finished by anyone, including after this session ends.

| what | value |
|---|---|
| arm | `ldlqH1` -- `export_tessera_serving.py <src> --grid E2M1x2 --q256 896 --input-scales scales_pqcal.safetensors --stock-twin ... --hessian h_full_qwen06b.pt --refit-metric h^1.0` |
| why the flag is explicit | the per-plane default was set by *this* receipt; naming the objective keeps the arm meaning the same before and after that change. `test_the_default_on_this_plane_is_the_arm_that_was_measured` pins that the default now writes the same bytes |
| launched by | `experiments/ldlq_lut_export_arms.sh` (it also launched a full-H arm, killed once the weight-space screen had chosen -- the task asks for ONE export) |
| **arm mismatch** | the screen that killed the full-H arm was **three of six units**; the completed six-unit geomean inverted and selects `hessian` by 1.38%. The arm being served is therefore the one my pre-registered rule does *not* pick. It was not relaunched -- see "The deviation, owned" above -- so the served number below measures `h^1.0`, and no served number exists or will exist for `hessian` |
| log | `/mnt/shared/tessera-runs/ldlq-lut/export_ldlqH1.log` |
| output | `/mnt/shared/tessera-runs/ldlq-lut/ldlqH1-stock-twin` |
| to finish | `experiments/ldlq_lut_serve.sh ldlqH1` -> `/mnt/shared/tessera-runs/ldlq-lut/kl_ldlqH1.json` |
| already armed | a detached waiter on sparklina (`/home/rob/tmp/ldlq_arm_serve.sh`, log `serve_ldlqH1_chain.log`) blocks on the export pid and then runs exactly that command, so the leg lands without a session attached. It takes `serve_lock.sh` like every other serve on this box, on port 8001 under `TESSERA_KL_NAME=tessera-kl-ldlqlut`, and refuses rather than serving a partial twin |
| comparators | **0.640** (`kl_unrot-k2-w4a4-pqcal.json`, `all.kl_lower_mean`; same recipe, weights-only, same A4 scales, same teacher, same box) and **0.511** (PrismaQuant NVFP4 GPTQ+JSO at 4.5 bpp) |
| the gate | served KL better than 0.640 at matched bytes |

**What each outcome means, written before the number.** If `ldlqH1` beats
0.640, the H-aware LUT route clears its gate on the arm the weight-space screen
*second*-ranked, and the natural follow-up is whether `hessian` would have done
better -- a question this receipt cannot answer and should not guess at. If it
does not beat 0.640, the finding is that LDLQ plus a block-scale refit on the
LUT plane does not pay for itself on the serving metric at 4.0 bpp, the default
does not move, and a screen that says 0.78x in weight space has already failed to predict a
serve once on this project. The encode cost is no longer the argument against
the route on a 0.6B dense model -- ~1.4 h -- so if it misses the gate, it
misses on quality. On GLM's 4096-column experts the cost is now measured and is
also not the argument: 35.1x encode-only at the default block size, and **9.0x
at `ldl_block=128` with output-space error flat to marginally better**. The dial
that sets it had never been turned.

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

## What this branch changes by default

For the release tag, stated as a table rather than a paragraph, because the
question is "does the wire's default behaviour move" and the answer is
different for each plane:

| path | before | after |
|---|---|---|
| any encode with **no** Hessian | weights-only | **byte-identical** (784/784, twice) |
| CHANNEL plane + H (the E4M3 window route) | sigma 1.0, block 32, refit `hessian` | **identical** |
| LUT plane + H (E2M1, E2M1x2 -- the 4-bit wire) | `GrammarError`, refused | runs: sigma 1.0, block 32, refit `h^1.0` |
| S6b plane + H | `GrammarError`, refused | refit still refused; LDLQ now permitted |

So **no default moves on any path the previous code could execute.**
`DEFAULT_LDLQ_SIGMA` and `DEFAULT_LDLQ_BLOCK` are unchanged at 1.0 and 32.
`DEFAULT_REFIT_OBJECTIVE` changes *type* -- from the string `"hessian"` to a
per-plane map -- but it still evaluates to `"hessian"` on CHANNEL, and CHANNEL
was the only plane that could read it before. The two new rows are capability,
not policy: they are reachable only by passing a Hessian, which no recipe does
on its own.

Two things worth saying out loud about that table. The `h^1.0` on the LUT row
is **not** what the completed weight-space screen picks -- the six-unit geomean
selects `hessian` by 1.38%, on a margin carried by one unit. It is what the
**serve** picks: `h^1.0` clears the gate at 0.5310 against the weights-only
wire's 0.640 at matched bytes, and it is the only arm with served evidence.
The deviation from my own pre-registered rule, and why it was not relaunched,
is under "The deviation, owned" above. And the S6b row is narrower than it looks: across every `q256`
of all three serialisable grids, `wire_recipe` resolves only to LUT and
CHANNEL, so S6b is not on any exportable wire and its map entry exists to make
`objective_for` total over the enum.

## Scope, and what is not measured

- **The refit lever and the LDLQ lever do not live in the same place.** On
  dense Qwen the block-scale refit is worth real error; on the GLM experts it is
  a measured no-op (`refit-h^1.0` moves `out` by <0.1% on every expert). That is
  the same fact the rotation A/B found from the other side: the experts' inputs
  are close to Gaussian, so their `diag(H)` is nearly flat, and a metric that is
  flat is the plain refit. The GLM gate below is therefore carried by **LDLQ**,
  not by the refit. Read the two levers separately; do not assume either
  transfers.
- **One export, one serve.** The served leg is a single Qwen3-0.6B artifact at
  one rung (E2M1x2, q896, the TCQ cap) against one teacher on one box. It is
  the arm the weight-space screen chose, not a sweep of arms; a served
  comparison of `hessian` against `h^1.0` was deliberately not run, because the
  screen separated them and, at the time of the decision, each export was
  believed to cost ~9 h. It costs 19.2x the weights-only encode, ~1.4 h
  whole-model, so the *cost* half of that reasoning is weaker than I thought
  but not absent -- the ONE-export budget and
  the coordinator's instruction not to relaunch do.
- **`sigma` and `block` were not re-tuned for this plane.** They are the
  window-body receipt's values (1.0 / 32) carried over, and the sweep's
  three-sigma scan confirms 1.0 on this body rather than deriving it. Block 32
  in particular is the whole cost story above; a larger block would be cheaper
  and was not measured.
- **The Hessian is one capture.** 16k tokens of wikitext-2 *train*, disjoint
  from the wikitext-2 *test* KL corpus by construction and by hash (table
  above). Nothing here says how the levers behave under a calibration mismatch,
  which is the failure mode GPTQ is known for and the reason the weights-only
  wire remains the default when no H is supplied.
- **Weights-only encodes are unchanged, and that is tested, not asserted.**
  `base2` -- the weights-only baseline re-exported by the arm's own post-merge
  code -- is byte-identical to the pre-merge `base` and to the served 0.640
  comparator, 784/784 tensors. The wire schema, the recipes and the reader are
  untouched: both levers are encoder-side.

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
`ldlq_lut_serve.sh`, and the four behind the cost section:
`ldlq_cost_profile.py` (profiler + power trace), `ldlq_cost_matched_pair.sh`
(the control -- weights-only, H-aware, weights-only again), and
`ldlq_cost_knob_ab.sh` / `ldlq_cost_tree_ab.sh` (the two disproved
hypotheses, kept so the disproof is re-runnable).

Tests: `tests/test_ldlq_lut_plane.py` (21), `tests/test_ldlq_window.py` (the
defaults are the measured ones), `tests/test_merge_guard.py` (a per-plane map
that disagrees only off its own plane still refuses).

## 2026-09-03: the decision rule is now a gate (tessera#65)

The rule above lived in this receipt's prose, which is why the deviation it
records could happen with nothing refusing. It now lives in
`tessera.control.assert_plane_promotion`, beside the byte-match and
selection gates: a candidate promotes only on a strict majority of the
receipt's own units -- never on the geomean alone, which is derived from the
per-unit ratios and cannot arrive without them -- and only when the served KL
measures the promoted arm, still under this receipt's GLM six-expert gate
gate, which the gate restates without moving.

Run against this receipt's own record -- the six-unit `out` table in
`tessera-lut-refit-gauss-seidel-2026-09-03.md:122-128`, divided rather than
retyped -- the gate refuses the rule's literal pick at 2 of 6. It does not
"accept the shipped arm", and the distinction matters: head to head the two
arms refuse *each other*, `hessian` on the per-unit leg and `h^1.0` on the
geomean (1/0.9864 = 1.0138x). `h^1.0` stands because it is the incumbent and
the only candidate against it was refused -- an incumbent is what a
promotion is measured against, not something that promotes itself.

The served bar moved with that reading. This receipt's 0.640 is the *stock*
wire's served KL: the incumbent for "levers vs no levers", and for nothing
after `h^1.0` served 0.5310. The gate therefore takes the incumbent's served
KL as an argument with no default, so the leg reads the arm being replaced.
No default moves: the gate makes the next such decision checkable, it does
not remake this one.
