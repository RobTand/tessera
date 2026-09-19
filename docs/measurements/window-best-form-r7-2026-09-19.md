# Best-form vs front-form at the window body's shipping case (L=14, R=7)

**Date:** 2026-09-19 · **Issue:** #483 · **Box:** sparklina (GB10, sm121,
driver 595.91.07) · **Runner:** PrismaBuild only · **Action:** `30b3266d6c9e`
(`--measurement --host-class gb10 --gpu`, 70 s) · **Receipt:** CAS
`61f7c9ec3db0…` · **Raw JSON:** `/mnt/shared/agents-ts-enc-483/ab-r7.json`

Harness `experiments/window_viterbi_best_form_ab.py --mode time --configs R7`
(new R7 row this branch): L=14, R=7, arity 1, `[1792, 1024]` fp32 targets,
weighted, E4M3 production table -- the `[1792, 1024]` shape PR #386's L2 sweep
measured, at the production chunk 512. Three ABC-block arms in one process on
one tensor; energy is nvidia-smi 1 Hz trapezoidal joules over each block's own
interval (bracketed, coverage 1.0 on all nine blocks).

## Identity first

All three arms return the reference's bytes before any clock is read:

| arm | states equal | sse |
|---|---|---|
| front | true | `0x1.4609d50000000p+10` |
| best | true | same float |
| best@w32 | true | same float |

Byte-identical states and the identical `sse` float, so the timing compares
spellings of one answer. `encoder_fixture_id()` is untouched -- no byte moved.

## Timing (median of 3 blocks; power = mean over the arm's own joules)

| arm | width | batches | s/call | mean W | envelope | work/J |
|---|---|---:|---:|---:|---:|---:|
| front | 32 | 32 | 0.43911 | 79.07 | 56% | 865.7 M |
| best | 512 | 2 | 0.13163 | 66.41 | 47% | 3444.5 M |
| best@w32 | 32 | 32 | 1.13472 | 30.51 | 22% | 868.3 M |

Ratios against front: **best 3.3359x wall, 3.9787x work per joule**;
best@w32 0.387x wall, 1.003x work per joule. Registers (launched kernels,
no spills anywhere): front 40, best 96.

## Attribution, which is the point of the third arm

best@w32 holds the candidate to the front form's internal width (L2 budget
narrowed for that arm alone; chunk unchanged, so the epilogue is identical).
At equal width the candidate is *slower* per call (1.13 s vs 0.44 s) and
exactly as efficient per joule (1.003x): the store/front reduction alone buys
nothing. The whole win is the width the smaller resident set earns -- 512
columns and 2 batches instead of 32 and 32 -- i.e. fewer trips of the same
traffic, not a resident front. A per-column on-chip kernel (the issue's named
lever) would now have to beat best@512, not front@32.

## What this does not establish

- Synthetic `randn` targets and the E4M3 table, not the campaign's 32 LFM L18
  expert `w1` `[1792, 2048]` units at `BF16_K1@1792`. The traffic shape
  (L, R, rows, cols, weighted) matches; the values do not.
- One action, one box, one evening. No served KL (weight-space screen only),
  no campaign-unit bytes beyond the identity check above.
- The default is unchanged: promoting best-form moves the shipping schedule
  and is a decision only Rob prices. This measurement is the evidence for
  that decision, not the decision.

## Follow-up left in the issue

`want_sse=False` on `viterbi_columns` was explicitly scoped out here: that is
the TCQ path whose fused implementation is #486's live lane (read-only from
this branch). The window half (`viterbi_window`, `_run_joined`) already passes
it. The register-resident column kernel at L=12/R=4 stays a recorded negative
(`window_viterbi.py:66-74`); at L=14/R=7 the measured comparison is now
front@32 vs best@512 above.
