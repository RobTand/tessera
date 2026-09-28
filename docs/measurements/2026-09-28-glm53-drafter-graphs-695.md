# CUDA graphs for the GLM53 NoPE drafter (tessera#695), step 1 on the stub

Step 1 measures the MTP drafter's CUDA-graph paths on a four-layer stub, at
TP1 on sparky. It adds **no receipt**: `glm53_nope._SPECULATIVE_GRAPH_RECEIPTS`
stays empty, so every drafter graph path stays refused by name, and
speculative method `dflash` stays refused in every mode (tessera#700).
Admission is decided in step 2, on the release artifact at TP2: an eager
sweep over k = 1, 2, 3 picks k\*, then four eager and three graph serves at
k\* are compared with the serve as the unit (relabelling floor 1/35 = 0.029).
The stub results below are supporting depth for that decision, not the
admission.

## Setup

- **Model.** `/mnt/shared/tessera-runs/moe/glm53-4layer-a4-e2m1x2-q896-l2-mtp1`:
  the tessera#508 four-layer stub (KDA, MLA, KDA, MLA; routed experts on
  layers 1 and 3) plus the 45-layer export's MTP layer, renamed from
  `layers.45` to `layers.4` with its bytes copied verbatim
  (`experiments/glm53_695_drafter_qual/build_stub_mtp.py`). The drafter
  loads from `...-mtp1-draft`, which holds the MTP shard, the stub's first
  shard and a config.
- **Runtime.** Image
  `localhost/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b`
  (the V2 runner with vllm-project/vllm#57317 backported, tessera#508), vLLM
  `0.28.1rc1.dev397+gfd4a15126.d20260904`, TP1 on sparky (GB10, `sm_121`,
  140 W envelope). One arm, `eSk2`, ran eager on the stock image
  `f8dbe1a0`.
- **Serve.** `experiments/glm53_508_graph_qual/srv-508.sh` with #508's
  flags, and `--max-num-seqs 4` (the release value),
  `--gpu-memory-utilization 0.45`, `--no-enable-prefix-caching` and
  `--speculative-config '{"method": "mtp", "num_speculative_tokens": k,
  "attention_backend": "CUSTOM", "model": ".../glm53-4layer-a4-e2m1x2-q896-l2-mtp1-draft"}'`.
  A graph arm adds `--compilation-config` with its CUDA-graph mode and every
  capture size from 1 to 4 (1 + k): 1..8 at k = 1, 1..12 at k = 2, 1..16 at
  k = 3.
- **Measurement tree.** `claude/695-meas` on this repository, never merged.
  Its commit `1f5117fa7f` fills `_SPECULATIVE_GRAPH_RECEIPTS` with the
  candidate keys, each valued with a string, so the graph serves start and
  the gate withholds the equivalence claim. Every graph serve logged it:
  `this serve's outputs are not claimed equal to eager's; measure its
  quality on it: MEASUREMENT-ONLY admission (tessera#695 step 1)`. Arms ran
  at `ee893d6be6`, except `eBk2a`, `eSk2` and `fBk2a` (`1f5117fa7f`) and
  `eBk2b` (`f0042c9846`); every tree was clean.
- **Hooks** (`experiments/glm53_508_graph_qual/digest/usercustomize.py`, all
  outside the arithmetic): `T695_GC_BEFORE_DRAFTER=1` on every serve
  (section 8), and `T695_DRAFT_LOG` on the draft-identity arms (section 2).
- **Admission of each serve.** The queue (`t695/arm-queue-695.sh`) started an
  arm only with no container or GPU process resident, no PrismaBuild row
  claimed on or pinned to sparky, and MemAvailable minus 45 GB (75 GB under
  `compute-sanitizer`) above the box's 16 GiB watchdog floor.

Arm names: `e` eager, `f` FULL_DECODE_ONLY, `a` FULL_AND_PIECEWISE, `p`
PIECEWISE; `B` the backport image, `S` the stock image; `k` the draft
tokens; a trailing letter a repeat. `pr` arms are profiles, `mc` arms
memcheck.

## Verdict

On the stub, no k is consistent with admitting FULL_DECODE_ONLY on all three
criteria, and step 1 adds no receipt. Each reading applies the rule as it
was written down before the last arms finished (section 1).

| k | mode | eager + graph serves: identity; membership | floor | 1. not below eager | 2. `p_low` at position 0, position 1, vector | 3. largest graph run vs largest eager run | reading |
|---|---|---|---|---|---|---|---|
| 1 | FULL_DECODE_ONLY | 3 + 1; 3 + 1 | 0.25 | holds | 1.0 (cannot refuse) | 2.0 vs 1.0: fails | not admitted (criterion 3) |
| 2 | FULL_DECODE_ONLY | 5 + 4; 7 + 5 | 0.008 | holds | 0.429, 0.143, 0.135 | 4.14 vs 4.0: fails | not admitted (criterion 3) |
| 2 | FULL_AND_PIECEWISE | 5 + 1; 7 + 1 | 0.167 | holds | 0.333 at each (cannot refuse) | 2.29 vs 4.0: holds | consistent, one serve |
| 2 | PIECEWISE | 5 + 1; 7 + 1 | 0.167 | holds | 0.833, 1.0, 1.0 (cannot refuse) | 1.29 vs 4.0: holds | consistent, one serve |
| 3 | FULL_DECODE_ONLY | 4 + 2; 4 + 2 | 0.067 | holds | 0.133, **0.067**, 0.133 | 3.0 vs 3.0: holds | refused (criterion 2, position 1) |

- k = 1 and k = 2 fail criterion 3 by count, the headline. As an
  annotation, not a re-verdict: the non-members of the runs above the eager
  maximum (`fBk1`'s first pass, `fBk2b`'s second) differ from the nearest
  eager outcome by at most 0.07 in a top-20 logprob, about one bf16 step of
  a logit and the size of eager's own differences, while the eager run with
  the most non-members at k = 2 (`eBk2a`, 4) carries a move of 0.56 at a
  first token (sections 2 and 4).
- k = 3 is refused at position 1 at its floor: of the 15 ways to label two
  of the six serves "graph", the observed one has the lowest statistic.
- FULL_AND_PIECEWISE and PIECEWISE ran one serve each, whose floor cannot
  refuse.

Step 2 decides on the release at k\*, with four eager and three graph
serves (floor 0.029). Its criterion 3 is the serve-level relabelling of the
non-member counts (section 1), ruled before any release data; the stub
readings above stand as run, under the max rule.

## 1. Protocol

**Probes.** Each serve ran some of:

- `accept` (`accept-695.py`): the 64 U4 acceptance prompts
  (`accept-prompts-64x256.json`, sha256 `6a5d7313...`), 64 greedy tokens
  each, one at a time: 4032 drafts per serve.
- `eq` and `eq2` (`equal-508.py`): #508's equality suite restricted to
  batches of at most 4, 22 completions: batch 1 at prompt lengths 1, 2, 5,
  8, 17, 100, 500, 1500 and 2000; batches 2, 3 and 4; and the two repeat
  pairs. `eq2` repeats the suite later in the same serve.
- `long`: 3649-token prompts, a two-chunk prefill (2048 + 1601 tokens).
- The harness smoke: two repeats of a 5-token prompt, and one 3649-token
  prompt.

**The eager reference runs the same speculative configuration.**
Verification runs 1 + k query tokens per request, so a serve without the
drafter is a different computation (the gate's rule since tessera#700).

**Acceptance cannot judge the drafter on the stub.** The acceptance probe
accepted 0 of 4032 drafts in every serve that ran it. Where the equality
suite ran before it, the engine's counters show 1 accepted draft token in
`eBk2b` and 1 in `fBk2c`, and none in the others. The MTP layer follows a
four-layer body, not the 45 layers it was trained after.

**Draft identity is the instrument instead.** `T695_DRAFT_LOG` records, at
every serving step, each request's k drafts. A request's context at a step
is its prompt plus every token the target emitted through that step; the
drafter's input is that context, so drafts are compared only where two
serves reached the same context. For two serves, identity is the fraction
of shared contexts at which their draft token at position i agrees, per
position, and at which the whole draft vector agrees. Position 0 comes from
the drafter's first step, which runs with the target's verification;
positions 1 and later come from its decode steps, which are separate
graphs. The comparisons below use the contexts every compared serve
reached, each context weighted once (`draftlog-695.py`,
`serve-matrix-695.py`).

**Membership.** `eq-member-508.py` calls a completion a member when its
token ids and every top-20 logprob list equal some eager outcome in a pool
(#508 section 2). `member-pools-695.py` judges every run against pools of
the same size: with n eager serves, an eager run against the other n - 1,
and a graph run against every (n - 1)-subset, averaged. The eager counts
are then the noise floor for the graph counts. An earlier count judged each
eager second pass against a pool that held the run itself; the fix is
`9fdf34df2f`.

**Criteria, as run.**

1. *Not below eager.* The mode's mean graph~eager identity is not below the
   lowest eager~eager pair, at position 0, position 1 and the whole vector.
   The ruling first said "inside the eager spread"; a graph pair above the
   spread (`fBk2c` at position 0, `aBk2` at position 1) is not a failure,
   so it reads "not below". The lowest pair falls as eager serves are
   added, so each verdict states its eager count.
2. *Serve-level relabelling.* The statistic is the mean graph~eager identity
   minus the mean eager~eager identity. Under the null hypothesis the serves
   are exchangeable, so `p_low` is the fraction of the ways to label |graph|
   of the pooled serves "graph" whose statistic is at or below the observed
   one. A mode is refused on the stub when `p_low` < 0.10 at position 0,
   position 1 or the whole vector. The smallest attainable `p_low` is
   1 / C(eager + graph, graph), the floor; a set whose floor is above 0.10
   cannot refuse. At k = 3, position 2 is a diagnostic, not part of the
   rule.
3. *Membership.* No graph run's mean non-member count exceeds the largest
   eager run's count. The count is the headline; the signature of a
   non-member (token ids, and the size of the largest logprob change on an
   identical prefix) is an annotation.

**Criterion 3 in step 2.** Under exchangeable runs, the max rule refuses by
chance with probability up to about g / (g + e), for g graph and e eager
runs: about 0.4 at four eager and three graph serves. It cannot separate a
failure from noise. Step 2 therefore applies criterion 2's serve-level
relabelling to each serve's mean non-member count, with the pools
recomputed under each labelling, and refuses when `p_high` < 0.10 (floor
0.029 at four eager and three graph serves). The max-rule count is printed
beside it as a diagnostic and never decides. This was ruled on 2026-09-28,
before any release data; the stub readings in this document stand as run,
under the max rule.

**The criterion change, and why.** Criterion 2 began as a 95% bootstrap of
the paired difference over contexts, inside fixed serves. Two
FULL_DECODE_ONLY serves at k = 2 then split under it, on the 513 to 517
contexts every arm had reached, against the three eager serves `eBk2c`,
`eBk2d` and `eBk2e`:

| serve | position 1, paired graph minus eager, by reference serve |
|---|---|
| `fBk2b` | c +0.0053 [-0.0083, +0.0193]; d -0.0141 [-0.0240, -0.0043]; e -0.0184 [-0.0290, -0.0082] |
| `fBk2c` | c +0.0069 [-0.0041, +0.0176]; d -0.0030 [-0.0113, +0.0052]; e -0.0038 [-0.0138, +0.0059] |

`fBk2b` fails at position 1 and `fBk2c` passes at every position. The
context bootstrap ignores serve-to-serve variance, so its intervals are too
narrow whenever one serve is atypical, and one serve could admit or refuse
a mode. The test was changed to the serve-level relabelling of criterion 2,
and the queue was extended to five eager and four FULL_DECODE_ONLY serves
at k = 2 (126 labellings, floor 0.008). The per-context intervals remain in
the receipts as diagnostics. The final analysis inputs (serves per k and
mode, exclusions, pools) were written down at 22:18Z, before the last four
arms finished (`analysis/analysis-plan.txt`).

## 2. k = 2

Draft identity uses five eager serves (`eBk2c` to `eBk2g`), four
FULL_DECODE_ONLY serves (`fBk2b` to `fBk2e`, sizes 1..12), one
FULL_AND_PIECEWISE serve (`aBk2`) and one PIECEWISE serve (`pBk2`), on the
513 contexts that all eleven reached. `eBk2a` and `fBk2a` ran before the
drafter log existed, and `eBk2b`'s log is empty (a hook bug, fixed before
`eBk2c`), so these three count only for membership. `eSk2` (the stock
image) counts for neither.

| | position 0 | position 1 | whole vector |
|---|---|---|---|
| eager~eager, 10 pairs: mean (range) | 0.9598 (0.9575 to 0.9652) | 0.9393 (0.9296 to 0.9542) | 0.9279 (0.9193 to 0.9436) |
| FULL_DECODE_ONLY~eager, 20 pairs: mean | 0.9595 | 0.9360 | 0.9244 |
| `fBk2b`~eager, mean of 5 | 0.9556 | 0.9287 | 0.9163 |
| `fBk2c`~eager, mean of 5 | 0.9614 | 0.9364 | 0.9248 |
| `fBk2d`~eager, mean of 5 | 0.9595 | 0.9386 | 0.9274 |
| `fBk2e`~eager, mean of 5 | 0.9616 | 0.9404 | 0.9292 |
| FULL_DECODE_ONLY~FULL_DECODE_ONLY, 6 pairs: mean | 0.9583 | 0.9336 | 0.9215 |
| `aBk2`~eager (FULL_AND_PIECEWISE), mean of 5 | 0.9594 | 0.9380 | 0.9262 |
| `pBk2`~eager (PIECEWISE), mean of 5 | 0.9605 | 0.9423 | 0.9315 |

**FULL_DECODE_ONLY:**

- Criterion 1 holds with five eager serves: the mode's mean (0.9595, 0.9360,
  0.9244) is above the lowest eager pair (0.9575, 0.9296, 0.9193) at every
  column. `fBk2b` alone is below that pair at every column (0.9556, 0.9287,
  0.9163); the criterion is on the mode's mean.
- Criterion 2 does not refuse: the statistics are -0.0003, -0.0033 and
  -0.0035, with `p_low` 0.429, 0.143 and 0.135 over 126 labellings (floor
  0.008).
- Criterion 3 fails by count. Mean non-member completions of 22: each eager
  run against the other six eager serves, and each graph run against all
  seven pools of six.

| serve | mode | first pass | second pass (`eq2`) |
|---|---|---|---|
| `eBk2a` | eager | 4.0 | 1.0 |
| `eBk2b` | eager | 2.0 | 1.0 |
| `eBk2c` | eager | 2.0 | 2.0 |
| `eBk2d` | eager | 3.0 | 2.0 |
| `eBk2e` | eager | 2.0 | 1.0 |
| `eBk2f` | eager | 1.0 | 0.0 |
| `eBk2g` | eager | 2.0 | 3.0 |
| `fBk2a` | FULL_DECODE_ONLY | 0.14 | 0.29 |
| `fBk2b` | FULL_DECODE_ONLY | 3.29 | **4.14** |
| `fBk2c` | FULL_DECODE_ONLY | 2.29 | 2.0 |
| `fBk2d` | FULL_DECODE_ONLY | 1.14 | 1.14 |
| `fBk2e` | FULL_DECODE_ONLY | 1.14 | 0.29 |
| `aBk2` | FULL_AND_PIECEWISE | 2.29 | 1.0 |
| `pBk2` | PIECEWISE | 1.0 | 1.29 |

`fBk2b`'s second pass, at 4.14, exceeds the largest eager run (`eBk2a`'s
first pass, 4.0), so FULL_DECODE_ONLY fails criterion 3 at k = 2.

**Signature, as an annotation** (`nonmember-sig-695.py`, each completion
against the pool it was judged in). `fBk2b`'s second pass misses the same
four completions in all seven pools:

- `b1_len1` completion 0: the same token ids as 7 of the 14 eager outcomes;
  the nearest, `eBk2b`'s second pass, differs at 32 positions by at most
  0.061 in a top-20 logprob.
- `b2` completion 1: the same token ids as all 14; the nearest, `eBk2a`,
  differs at 24 positions by at most 0.062.
- `rep_len17_a` completion 0: the same token ids as all 14; the nearest,
  `eBk2a`'s second pass, differs at 32 positions by at most 0.065.
- `rep_len1_b` completion 0: no eager outcome has its token ids. The
  nearest, `eBk2d`'s second pass, first differs in token at position 21;
  up to there, 6 positions differ, by at most 0.062.

In the one pool without `eBk2a`, it also misses `b2` completion 0, which is
identical to `eBk2a`'s. Each of these moves is one bf16 step of a logit (a
logit change of 0.0625). `fBk2b`'s first pass misses, in every pool,
`b1_len1` completion 0 (tokens first differ at position 5; 0.062),
`b1_len2000` completion 0 (tokens first differ at position 31; 0.105, a
logit change of 0.094) and `b2` completion 1 (0.062). The largest eager run,
`eBk2a`'s first pass, misses `b1_len1` completion 0 with a first token that
differs from every other eager outcome's (a logprob change of 0.562, a
logit change of 1.06), and three same-token completions at 0.061 to 0.096.

**FULL_AND_PIECEWISE and PIECEWISE**, one serve each, diagnostics (6
labellings, floor 0.167, so neither can refuse):

- `aBk2` is not below the lowest eager pair. Its statistics are -0.0004,
  -0.0013 and -0.0017, with `p_low` 0.333 at each. Membership holds (2.29
  and 1.0, against the eager 4.0).
- `pBk2` is not below the lowest eager pair. Its statistics are +0.0006,
  +0.0029 and +0.0036, with `p_low` 0.833, 1.0 and 1.0. Membership holds
  (1.0 and 1.29).

## 3. The position-1 profile and the sparse index layout

Before `fBk2c` ran, `fBk2b`'s position-1 deficit was profiled (`prEk2`,
`prFk2`: batch-1 decode at k = 2, `torch.profiler` in the engine process,
section 9). In every profiled step of each arm the kernels are the same
multiset. The drafter's position-1 step runs the same 70 compute kernels in
both modes, with the same names, grids, blocks, shared memory and
registers; only copy kernels and the stream order differ. The difference is
upstream, in how the sparse index is built:

- In a FULL decode graph the indexer always runs its general path:
  `sm120_fp8_paged_mqa_logits`, then `persistent_topk`, then
  `_expand_pools_and_append_tail`, three times per step on the stub (the
  target's two indexed layers and the drafter's first step).
- Eager takes `_fill_short_decode_causal_indices` whenever
  `max_seq_len <= 2048`, a host-side branch in the image's
  `vllm/model_executor/layers/sparse_attn_indexer_kpool.py`; capture freezes
  the general path in.
- By reading the code (not measured), both paths select every token at
  these lengths but lay the list out differently: eager writes
  `[0..pos, -1, ...]`; the general path expands the history pools into
  columns 0 to 2047 and puts the up to 3 most recent tokens at 2048 to 2050
  (`kpool_compress.py`, `_expand_pools_and_append_tail_kernel`). The sparse
  attention therefore reduces in another grouping. With
  `index_share_for_mtp_iteration`, position 1 reuses position 0's index.

`fBk2c` then passed every criterion on the same configuration, so the
layout has not been separated from eager's own variation, here or in #508's
FULL_DECODE_ONLY receipts (the same image and indexer code; membership held,
`member-fdoN8.json`, `member-fdoF8.json`). Making eager and FULL graphs
build the same index belongs to the image's patch set and is filed as
RobTand/prismaquant#1631 [P2]. Its prefill scope was read, not measured:
FULL_DECODE_ONLY runs prefills eagerly, so the prefill fast path
(`kpool:423-441`) is evaluated at run time; under breakable PIECEWISE the
indexer carries `@eager_break_during_capture` (`kpool:259`) and runs live
between segments. So a served KL panel sees the same index as the release
graph serve, except a request of at most 1 + k tokens, which takes the
decode path (`kpool:727`); the panel's 512-token sequences never do.

## 4. k = 1

Three eager serves (`eBk1`, `eBk1b`, `eBk1c`) and one FULL_DECODE_ONLY
serve (`fBk1`, sizes 1..8), 4341 contexts that every serve reached. With one
draft token, position 0 is the whole vector.

| pair | identity |
|---|---|
| `eBk1`~`eBk1b` | 0.9912 |
| `eBk1`~`eBk1c` | 0.9916 |
| `eBk1b`~`eBk1c` | 0.9919 |
| `fBk1`~`eBk1` | 0.9919 |
| `fBk1`~`eBk1b` | 0.9928 |
| `fBk1`~`eBk1c` | 0.9938 |

- Criterion 1 holds: the graph~eager mean, 0.99285, is above the lowest
  eager pair, 0.99123 (three eager serves).
- Criterion 2 cannot refuse: four labellings, floor 0.25. The statistic is
  +0.0013, `p_low` 1.0. So k = 1 rests on criteria 1 and 3 only.
- Criterion 3 fails by count, the headline: `fBk1`'s first pass averages
  2.0 non-member completions of 22 over its three pools of two serves, and
  its second pass 0.33, against 0 to 1 for the six eager runs.
- The signature, as an annotation: both first-pass non-members (`b2`
  completion 1 and `b4` completion 2, in every pool) generate the same token
  ids as an eager outcome. `b2` differs at 24 positions by at most 0.0693 in
  a top-20 logprob, the size of one bf16 step of a logit (#508 section 3);
  `b4` differs at one position by 0.00098, a normalizer shift from a logit
  outside the top 20. The second pass's one non-member, `b1_len100`,
  differs by at most 0.065 with the same token ids.

So on the stub, k = 1 FULL_DECODE_ONLY is not below eager on drafts, but
two of its equality-suite outputs are not eager outcomes, by eager-sized
differences. Step 2 decides k = 1 only if the sweep picks it.

## 5. k = 3

Four eager serves (`eBk3`, `eBk3b`, `eBk3c`, `eBk3d`) and two
FULL_DECODE_ONLY serves (`fBk3`, `fBk3b`, sizes 1..16), on the 4160 contexts
that every serve reached. `eBk3d` and `fBk3b` were appended after the first
four, with the same configuration and admission gates.

| pair | position 0 | position 1 | position 2 | whole vector |
|---|---|---|---|---|
| `eBk3`~`eBk3b` | 0.9893 | 0.9832 | 0.9739 | 0.9718 |
| `eBk3`~`eBk3c` | 0.9859 | 0.9779 | 0.9671 | 0.9642 |
| `eBk3`~`eBk3d` | 0.9870 | 0.9791 | 0.9666 | 0.9630 |
| `eBk3b`~`eBk3c` | 0.9885 | 0.9812 | 0.9719 | 0.9703 |
| `eBk3b`~`eBk3d` | 0.9874 | 0.9788 | 0.9682 | 0.9642 |
| `eBk3c`~`eBk3d` | 0.9850 | 0.9718 | 0.9609 | 0.9565 |
| `fBk3`~`eBk3` | 0.9846 | 0.9759 | 0.9634 | 0.9604 |
| `fBk3`~`eBk3b` | 0.9843 | 0.9722 | 0.9616 | 0.9587 |
| `fBk3`~`eBk3c` | 0.9832 | 0.9718 | 0.9590 | 0.9558 |
| `fBk3`~`eBk3d` | 0.9830 | 0.9703 | 0.9581 | 0.9539 |
| `fBk3b`~`eBk3` | 0.9862 | 0.9747 | 0.9641 | 0.9592 |
| `fBk3b`~`eBk3b` | 0.9869 | 0.9758 | 0.9642 | 0.9606 |
| `fBk3b`~`eBk3c` | 0.9882 | 0.9793 | 0.9691 | 0.9667 |
| `fBk3b`~`eBk3d` | 0.9851 | 0.9696 | 0.9570 | 0.9510 |
| `fBk3`~`fBk3b` | 0.9842 | 0.9735 | 0.9609 | 0.9564 |
| mean eager~eager | 0.9872 | 0.9787 | 0.9681 | 0.9650 |
| mean graph~eager | 0.9852 | 0.9737 | 0.9621 | 0.9583 |

- Criterion 1 holds with four eager serves: the graph~eager mean is above
  the lowest eager pair, `eBk3c`~`eBk3d` (0.9850, 0.9718 and 0.9565 at
  position 0, position 1 and the whole vector). At position 0 the margin is
  0.00017 (0.98520 against 0.98503).
- Criterion 2 refuses at position 1. The statistic is -0.0050, and `p_low`
  is 0.067, the floor of 15 labellings: no other labelling of two serves as
  "graph" gives a statistic as low. Position 0 (-0.0020), the whole vector
  (-0.0067) and position 2 (-0.0060, a diagnostic) each have `p_low` 0.133.
- Criterion 3 holds, in pools of three: `fBk3` 2.5 and 1.25 non-members
  (first and second pass), `fBk3b` 3.0 and 1.0, against eager runs of 0 to
  3.0 (`eBk3` 3.0 and 3.0, `eBk3b` 1.0 and 0, `eBk3c` 1.0 and 2.0, `eBk3d` 0
  and 0).

So on the stub, k = 3 FULL_DECODE_ONLY is refused, at the smallest `p_low`
a four-plus-two set can reach. By the step-2 ruling, the stub is deepened
only if the release sweep picks a k\* other than 2.

## 6. Captured graphs

`prFk2` ran with `T508_CAPTURE_LOG` (FULL_DECODE_ONLY, k = 2, sizes 1..12,
`max_num_seqs` 4). vLLM captured only the uniform decode shapes of each
family, not every listed size:

| manager | graphs (tokens: requests x tokens per request) | capture wall | reserved |
|---|---|---|---|
| target (`ModelCudaGraphManager`), verification | 12: 4x3, 9: 3x3, 6: 2x3, 3: 1x3 | 18.126 s | 458.0 MiB |
| drafter, first step (`SpeculatorCudaGraphManager`) | 12, 9, 6, 3 (x3) | 0.081 s | 58.0 MiB |
| drafter, later steps (`SpeculatorCudaGraphManager`) | 4, 3, 2, 1 (x1) | 0.233 s | 56.0 MiB |

Every batch of 1 to 4 requests has its own graph in each family, which is
what `eager_equivalence_gap` checks (`_padded_families`): with every size
from 1 to 4 (1 + k) captured, no family pads. The other FULL_DECODE_ONLY
arms follow from the same configuration; PIECEWISE and FULL_AND_PIECEWISE
captures were not logged. The target's first graph (12.5 s) and first graph
under 8 tokens (5.5 s) are the one-time costs #508 section 7 measured.

## 7. Long prefills and faults

Every graph serve that sent a 3649-token prompt completed its two-chunk
prefill (2048 + 1601 tokens): 33 prefills over 11 graph serves. The smoke
sent one in each of 10 serves; the `long` probe sent two in `aBk2`, `fBk1`,
`fBk3` and `fBk3b` and ten in `fBk2a`; and `prFk2`'s latency case sent
five. No graph serve's log has a fault line (`CUDA error`, `illegal
memory`, `Xid`, `device-side assert`, `Segmentation fault` or `core dumped`;
`analysis/longcount.py`). The first generated token's logprob ranged from
-5.335 to -5.088 over the smoke's 10 prefills and from -4.930 to -4.700 over
the probe's 18. Prefills above 2048 tokens are not repeat-exact, because
the stock top-k kernel writes its selection in arrival order (#508
section 3), so these prefills answer the fault question only.

## 8. Memory

**The drafter loads on top of the target's load garbage.** The V2 runner
loads the target and the drafter inside one `DeviceMemoryProfiler` block,
and the target's load leaves memory collectable until that block exits.
The first two `eBk2a` loads ran without `T695_GC_BEFORE_DRAFTER`; the box's
memory watchdog removed both at its 16 GiB floor (MemAvailable 15988 MiB at
19:25:07Z and 15739 MiB at 19:32:14Z). The first of them also named no
draft model, so its drafter loaded the 38.72 GiB target checkpoint. With
the hook, which runs `gc.collect()` and `torch.cuda.empty_cache()` once
before `MTPSpeculator.load_model`, every later load completed. In each of
the 27 serve logs that print it, the hook raised MemAvailable by 41.8 to
43.7 GiB, from 29.9-35.6 to 72.9-78.2 GiB. The minima below hold with the
hook; the release serve does not carry it.

Each serve's memory, sampled at 1 Hz on sparky (`mem-sampler-695.sh`). The
footprint is MemAvailable at launch minus the in-run minimum, plus the
swap-out over the run (`mem-summary-695.py`):

| serve | MemAvailable at launch (GiB) | minimum (GiB), at (s) | footprint (GiB) | swap-out (GiB) |
|---|---|---|---|---|
| `eBk1` | 110.6 | 33.1, 66 | 83.4 | 5.9 |
| `eBk1b` | 110.4 | 33.7, 68 | 83.5 | 6.7 |
| `eBk1c` | 110.8 | 34.1, 70 | 83.1 | 6.4 |
| `eBk2a` | 113.3 | 32.2, 68 | 83.0 | 1.9 |
| `eBk2b` | 110.5 | 33.7, 74 | 85.1 | 8.3 |
| `eBk2c` | 106.2 | 33.5, 76 | 79.6 | 7.0 |
| `eBk2d` | 109.8 | 33.8, 74 | 82.3 | 6.3 |
| `eBk2e` | 108.8 | 32.4, 70 | 84.2 | 7.9 |
| `eBk2f` | 108.9 | 34.7, 64 | 81.8 | 7.5 |
| `eBk2g` | 109.6 | 29.9, 62 | 83.7 | 4.0 |
| `eBk3` | 112.3 | 30.4, 66 | 85.3 | 3.3 |
| `eBk3b` | 107.8 | 31.2, 81 | 83.1 | 6.6 |
| `eBk3c` | 107.5 | 29.5, 70 | 82.4 | 4.4 |
| `eBk3d` | 109.0 | 29.9, 81 | 83.1 | 4.0 |
| `eSk2` | 111.5 | 33.2, 64 | 82.6 | 4.3 |
| `fBk1` | 109.1 | 29.6, 68 | 85.3 | 5.8 |
| `fBk2a` | 112.1 | 33.6, 62 | 82.2 | 3.7 |
| `fBk2b` | 109.9 | 33.2, 77 | 82.4 | 5.6 |
| `fBk2c` | 111.6 | 31.9, 68 | 82.4 | 2.7 |
| `fBk2d` | 109.9 | 31.2, 68 | 83.2 | 4.5 |
| `fBk2e` | 109.0 | 31.3, 66 | 82.6 | 4.9 |
| `fBk3` | 106.3 | 32.7, 81 | 83.9 | 10.3 |
| `fBk3b` | 109.4 | 31.4, 68 | 84.7 | 6.7 |
| `aBk2` | 110.7 | 34.4, 76 | 84.1 | 7.7 |
| `pBk2` | 110.1 | 30.8, 63 | 83.0 | 3.7 |
| `prEk2` | 110.3 | 31.7, 74 | 84.1 | 5.5 |
| `prFk2` | 109.5 | 34.8, 78 | 82.8 | 8.0 |

Over the 27 serves, the minimum was 29.5 to 34.8 GiB, reached 62 to 81 s
after launch, during the load, where the hook's before values sit (29.9 to
35.6 GiB); the footprint was 79.6 to 85.3 GiB. On the stub the load
garbage sets the minimum, not the steady state.

**Memcheck: UNMEASURED.** `mcFk2` (FULL_DECODE_ONLY, k = 2, the whole
serve under `compute-sanitizer --tool memcheck` on the kpool kernels) did
not finish loading: the watchdog removed it at MemAvailable 15317 MiB,
22:01:31Z (`serve-arms/failed-load-3/`). Its eager counterpart `mcEk2` was
not run. So whether the drafter's kpool-tail row stays in bounds on the
backport image is not measured, and #508's drafter caveat stays open: vLLM's
speculator builds attention metadata without positions, so a drafter's tail
row keeps the runner's mapping. A memcheck serve on a box with more free
memory is a later follow-up, not part of step 1.

## 9. Performance (principle 15)

`prEk2` (eager) and `prFk2` (FULL_DECODE_ONLY, sizes 1..12), both k = 2,
without the drafter log. `lat-508.py`, client-side stream timestamps,
medians of 5 runs; 1 Hz `nvidia-smi` power cut per case:

| case | eager TTFT / ITL (ms) | FDO TTFT / ITL (ms) | eager power mean (W) | FDO power mean (W) |
|---|---|---|---|---|
| batch 1, 128-token prompt, 128 new | 80.9 / 43.51 | 85.5 / 43.33 | 36.8 | 36.2 |
| batch 1, 1024-token prompt | 203.4 / 44.30 | 204.3 / 43.65 | 39.1 | 38.4 |
| 3649-token prefill (two chunks) | 735.1 | 739.0 | 55.7 | 50.8 |

With acceptance at 0, every step emits one token, so ITL is the step time.
The first request of each case took about 5.5 s in both modes (TTFT p90
5.5 to 5.6 s); it is not examined here. Netdata's
`nvidia_smi.gpu_power_draw` (10 s samples) read 40 to 48 W in both modes'
decode windows, 29 to 34% of the envelope, plus one sample of 13 to 14 W in
each. `system.cpu` (18 s buckets) was 9 to 13% busy in the eager decode
windows and 6.6 to 8.5% in the graph ones, on a 20-core box; nothing else
loaded it.

**In-process profile** of batch-1 decode at k = 2 (`PROF=1`; 8 steps timed
on the host, the trace holds about 7):

| | eager | FULL_DECODE_ONLY |
|---|---|---|
| host time in `execute_model`, median of 8 steps | 13.85 ms | 3.31 ms |
| `cudaGraphLaunch` calls in the trace | 0 | 22 (three graphs per step) |
| GPU busy (union of kernel intervals) / kernel span | 313.3 / 318.2 ms (98.4%) | 309.7 / 313.3 ms (98.8%) |
| share of kernel time: BF16 `gemvx` / `cutlass_80_wmma_tensorop_bf16` GEMM / CUTLASS MoE GEMM | 33.6% / 33.9% / 13.5% | 33.4% / 33.8% / 14.2% |

The graph removes about 10.5 ms of host work per step, but the GPU is
already busy 98% of the step in eager, so ITL moves by 0.4 to 1.5% and power
by 0.6 to 0.7 W. A third of the kernel time in both modes is
`cutlass_80_wmma_tensorop_bf16` kernels launched from `aten::mm`: an sm80
schedule on `sm_121`, for BF16 GEMMs at 3 tokens per request (shapes were
not recorded). It is the same share in both modes, so it is a route fact
about the stub's BF16 GEMMs under verification, not a graph effect. On the
45-layer model, whose host and device work per step both grow, none of
this is measured.

## 10. Limitations

- One four-layer stub at TP1, with a drafter that no configuration
  accepts. Draft identity says whether graph and eager propose the same
  drafts; it is not acceptance.
- The drafter ran at TP1; the release plan serves it at
  `draft_tensor_parallel_size` 2, whose graphs are not measured here.
- Sequence-parallel padding at TP2 was not checked for padded replays.
- Eager pools are samples, so membership is a lower bound on agreement.
- The index layout difference (section 3) is read from code and not
  separated from eager's variation.
- The drafter log, the profile and the capture log each ran on different
  arms; no arm carries all three.
- The smoke's two 5-token repeats differed in `aBk2`, `eBk2d`, `eBk2e`,
  `eBk2f`, `eBk2g`, `eSk2`, `fBk2d` and `fBk3`, eager and graph alike,
  consistent with eager's own run-to-run variation (#508 section 3). The
  probes after the smoke ran normally.
- Criterion 3's max rule, as run on the stub, refuses by chance with
  probability up to about g / (g + e) under exchangeable runs, counting
  passes: 0.25 at k = 1, 0.42 at k = 2 FULL_DECODE_ONLY and 0.33 at k = 3.
  The stub readings stand as run; step 2 uses the relabelling form
  (section 1).

## Receipts

| what | where |
|---|---|
| serve arms (`<arm>.eq.*.json`, `<arm>.draft.<pid>.jsonl`, `<arm>.accept.json`, `<arm>.long.summary.json`, `<arm>.mem.json`, `engine-args-<arm>.txt`, logs, `prEk2.prof/`, `prFk2.prof/`, `prFk2.capture.jsonl`, `failed-load-1/` to `failed-load-3/`) | `/mnt/shared/tessera-runs/receipts/695-drafter-20260928/serve-arms/` |
| analysis (serve matrices, membership pools, draft identity per graph arm, `analysis-plan.txt`) | `/mnt/shared/tessera-runs/receipts/695-drafter-20260928/analysis/`; interim files from before the criterion change in `analysis/superseded/` |
| gate tests in the backport image (tessera#700) | PB `6e4990557249`, image `c2e75e03`, at `ccce32c0de`: 85 passed, 0 failed; `/mnt/shared/tessera-runs/receipts/695-drafter-20260928/intest-ccce32c0de/junit.xml` |
| tools | `experiments/glm53_695_drafter_qual/` (`build_stub_mtp.py`, `accept-695.py`, `draftlog-695.py`, `serve-matrix-695.py`, `member-pools-695.py`, `nonmember-sig-695.py`, `eq-table-695.py`, `mem-sampler-695.sh`, `mem-summary-695.py`); `experiments/glm53_508_graph_qual/` (`srv-508.sh`, `run-arm.sh`, `equal-508.py`, `eq-member-508.py`, `lat-508.py`, `prof-summary-508.py`, `digest/usercustomize.py`) |
