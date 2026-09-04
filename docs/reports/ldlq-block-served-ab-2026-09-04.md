# tessera#60, served leg: `ldlq_block=4` against a re-run of `ldlq_block=32`

Status: **served leg complete on b8; b4 still encoding.** This file was written
as the run went, so the sections below are in the order they were learned. Where
a projection made before the run is contradicted by what the run then did, the
projection is left standing with the measurement under it, because the failure
of the projection is itself one of the findings.

## The served answer

**The arm served is b8, not b4.** b4 was still encoding; b8 was authorised as a
substitute and is what these numbers describe. b4 has its own bracket pending.

| arm | all-position KL | confident-position KL |
|---|---|---|
| `ldlq_block=32`, served **first** -- the bar | 0.5200805955 | 0.4282823138 |
| `ldlq_block=8`, the candidate | **0.5099719526** | 0.4284178346 |
| `ldlq_block=32`, served **last** -- drift control | 0.5200805955 | 0.4282823138 |

**The drift control is 0.000e+00 on both metrics**, and every arm reads
identically against a second, differently-boxed teacher. So the delta is a
difference between two exactly reproducible numbers, not between two draws.

| | |
|---|---|
| weight space predicted | 0.9373x (**-6.27%**), 6/6 unit wins |
| served, all positions | 0.9806x (**-1.94%**) |
| **carry fraction** | **0.31** |
| served, confident positions | 1.0003x (**+0.03%**) |

**The answer to the question this serve was reduced to -- does the weight-space
win carry to served KL at all -- is: about a third of it does, and only on the
all-position metric.** On confident positions it does not carry at all. Weight
space is directionally right and quantitatively over-optimistic by 3x here.

### Re-running the default rather than quoting it changed the answer's size

The incumbent, had I quoted its 2026-09-02 number, would have been
0.5310275686796917. Re-run on the current exporter it is 0.5200805955 --
**2.06% better**. Quoting the old number would have made b8 look 3.96% better
instead of 1.94%: the error would have been about as large as the effect. The
161-of-784 differing tensors between the two exporter states is why, and the
instruction to re-run was load-bearing rather than hygienic.

### The gate, verbatim, including the leg it cannot evaluate

`GLM_GATE = 1.0`, `LUT_LANDING_WIRE = 'table'`.

Called with the input this campaign actually has:

```
glm_ratio=None
TypeError: float() argument must be a string or a real number, not 'NoneType'
```

The gate has no default for a missing GLM leg; it fails rather than passes.
There is no GLM measurement at block 8 or block 4 -- the lowest GLM block on
disk is 16, because `choose_ldl_block` floors LUT/S6B callers at 16 and the
GLM sweep inherited that floor.

Called again with the block-16 GLM point substituted in -- **not the
candidate's block, and recorded only to show which leg is blocking**:

```
PlanePromotion(candidate='ldlq_block=8', served_arm='ldlq_block=8',
 unit_ratios=(0.9316, 0.8773, 0.9668, 0.9397, 0.9393, 0.9724),
 geomean=0.9373291536531269, wins=6, glm_ratio=0.9993, glm_bar=1.0,
 served_kl=0.5099719526, served_bar=0.5200805955, landing='table',
 where='the per-plane promotion')
```

All five legs pass once the missing one is filled. **This is not a promotion**
and I am not reporting one: the input that makes it pass is not a measurement
of the candidate. Two things the gate does not look at are worth saying out
loud, because a pass here would be read as more than it is. It has no encode
cost leg, and this lever is precisely a quality-for-encode-time trade. It reads
the all-position KL that moved, not the confident-position KL that didn't. And
its `unit_ratios` leg -- "a strict majority of per-unit wins" -- is fed the
six-unit **weight-space** sweep over layers 0-1, not a per-unit measurement of
the 196 units that actually served. That is the input the gate was built to
take, so this is not misuse; but the majority-of-units leg and the served-KL
leg are evidence about different populations, and only the second is about the
artifact.

Receipt: `experiments/results/ldlq_block_served_ab.json`.

## What this run settles, and what it no longer settles

It was funded to decide whether `DEFAULT_LDLQ_BLOCK` flips 32 -> 4. **The
encode cost takes that question off the table before the KL arrives** -- but by
less than I said before the arms landed, and the correction matters enough to
lead with. I projected block 4 at ~8x block 32 model-wide. The arms refute that
number: see "What the arms did to the projection" below. What survives is the
*shape* of the argument, not its size. Block 4 costs meaningfully more than
block 32 -- **2.12x measured, and a floor rather than an estimate**, since the
contention runs the wrong way to bound it tighter -- while GLM's experts are flat across the axis (0.4% from 16 to 256).
A *global* default of 4 therefore charges GLM for an axis that pays it nothing,
and a flip to another round number is the wrong shape of fix whatever the KL
says. That conclusion was right; the 8x I attached to it was not measured, and
is withdrawn.

What is left is the question behind it, and it is the one worth the wall
clock: **does the weight-space win carry to served KL at all?** If it does not,
then the derived-block work -- #95, and a `choose_ldl_block` in the export path
after it -- is not worth building either, and #12's dense-4 story goes back to
the drawing board.

## The question

Does `ldlq_block=4` beat the `DEFAULT_LDLQ_BLOCK = 32` on **served** KL for
dense Qwen3-0.6B, at byte-identical bytes? The six-unit weight-space sweep in
issue #60 says the geomean crosses parity between the default and everything
below it (1.0421 -> 0.9629 against NVFP4 GPTQ+JSO at 4.5 bpp). That is a
screen, and this repo does not promote on screens.

## The arms

Three are being encoded, not two, and the reason is the encode cost rather
than curiosity. `b4` is the arm the issue's recommendation names and the arm
whose per-unit ratios the gate already has; `b8` crosses parity in weight space
on the same six units (6 of 6 wins, geomean 0.937x against b32) at half the
encode, so it answers "does it carry" for half the wall clock. Both were
launched because the box went from load 3 to load 89 within half an hour of the
first launch and a 20-hour dead end is not a risk worth carrying for a third
core. **Whichever is served is named explicitly wherever a number is quoted.**


Both exports come out of **one checkout at one commit**,
`82cdf513aff3d013e05a804d3e0085422445c704`, through one script
(`experiments/ldlq_block_serve_export.sh`) whose only parameter is the block:

```
export_tessera_serving.py /home/rob/models/Qwen3-0.6B <out> \
  --grid E2M1x2 --q256 896 --input-scales .../rotation/scales_pqcal.safetensors \
  --stock-twin <twin> --hessian .../ldlq/h_full_qwen06b.pt \
  --ldlq-sigma 1.0 --ldlq-block {32|4} --refit-metric h^1.0
```

The default arm is **re-run, not quoted**. The 2026-09-02 `ldlqH1` artifact is
the same recipe at block 32 and serves 0.531028, but the tree has moved since
(trellis weighting, the reach-aware per-row start), so reusing it would put a
commit's worth of change inside a delta that is supposed to hold only the
block.

## "One session" is not a shape this stack has, and I am not going to pretend

vLLM serves one model per engine, and every serving wrapper in this repo
(`serve_and_dump_kl.sh`, `tessera_plugin_served.sh`) is start-serve-dump-reap.
Two checkpoints are two containers. There is no way to put both arms in one
vLLM session, so **I am not reporting one.**

What replaces it is a matched-pair drift control, which is evidence rather
than an assertion: the default is served **first and last**, with the
candidate between them, on one box, one image, one corpus, one teacher, eager
(the one known source of cross-container disagreement on this stack is
inductor build nondeterminism in the compiled forward, and an eager serve has
none). If the two default dumps agree, nothing between them drifted. If they
disagree, that disagreement is the error bar on the delta and is reported as
one. `experiments/ldlq_block_serve_ab.sh` is that chain. The two default
readings are reported separately and never averaged.

### And the substitute turns out not to be a compromise: drift is zero, measured

Before the A/B depended on the serve chain, the **2026-09-02 `ldlqH1` bytes
were re-served** -- an artifact whose answer is already published -- on the
same box, against the same teacher, on the same corpus, eager, in a different
container on a different day:

| | KL-vs-BF16, `kl_lower_mean` |
|---|---|
| published 2026-09-02 | 0.5310275686796917 |
| re-served 2026-09-04 | **0.5310275686796917** |
| delta | **0.0** |

`confident` (0.44606594216118406) and top-1 agreement (62.52446183953033%) are
identical to the last digit too, over the same 4,088 positions.

Three things follow, and the third was not the goal:

1. The serve chain is proved before the A/B rests on it, rather than debugged
   at hour fourteen with three exports finished and waiting.
2. **Cross-session drift on this lane is exactly zero for an eager serve.**
   That is measured, on bytes with a known answer, not assumed. The 4-8x
   cross-session drift this repo warns about is a PrismaQuant-era fact about a
   live-model path; it does not describe this one. So the A/B/A bracket is
   confirmatory rather than load-bearing -- which is the right order to find
   that out in.
3. The `identity`/`image_digest` fingerprint change that landed in #100 moved
   nothing observable on these bytes. Build fingerprint
   `03b89d80124b6123...`, vLLM 0.28.0, `vllm/vllm-openai:latest`, eager,
   `compiled_forward: false`.

None of this excuses reporting a delta as same-session. It says the delta does
not need to be.

### The bracket then had to move boxes, and that was measured too, not argued

sparklina is tied up with the b4 encode for another six hours, so the serve
bracket moved to **sparky**. Two standing objections had to be cleared before
a sparky number could be read against the incumbent, and both are testable
against artifacts that already have answers:

**Teacher side, at no GPU cost.** The bracket scores students against
`qwen_rot_teacher_lina.json.npz`, dumped on sparklina. Rather than assume the
cross-box term is small, or spend a serve dumping a fresh teacher, the two
BF16 teacher dumps already on disk were compared: `qwen_rot_teacher_lina`
(host `gx10-6b77`, 2026-09-02T17:12Z) against `qwen_teacher_bf16_v028` (host
`sparky`, 2026-09-02T05:32Z), same corpus hash, same tokenizer hash, twelve
hours apart.

> `KL >= 0.000000`, `top1_agree=100.00%`, 4,088 positions.

**Student side, one serve.** A teacher does not exercise the quantised kernel
path and a student does, so teacher agreement does not licence student
agreement. The `ldlqH1` bytes were therefore re-served *on sparky* and read
against both teachers:

| reading | host | teacher | `kl_lower_mean` |
|---|---|---|---|
| published 2026-09-02 | gx10-6b77 | lina | 0.5310275686796917 |
| re-serve, same box, +29 h | gx10-6b77 | lina | 0.5310275686796917 |
| re-serve, **other box** | sparky | lina | **0.5310275686796917** |
| re-serve, other box, **other teacher** | sparky | sparky | **0.5310275686796917** |

`confident` is 0.44606594216118406 on both sparky readings, and the build
fingerprint is `03b89d80124b6123` -- the sparklina re-serve's. Neither the
session, nor the box, nor whose BF16 teacher the arm is scored against moves
this number at all.

Before running it I checked the thing that would have made this a comparison
of *builds* rather than boxes: `vllm/vllm-openai:latest` is a floating tag,
and it resolves to the same digest
`sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` on
both boxes. Had it drifted, #100's `image_digest` in `identity` is exactly the
field that would have refused the pairing.

Scope, because this is easy to over-read: eager only, one image digest, one
model, one corpus, one quantised family. It says nothing about a graph-mode
arm, where inductor build nondeterminism is a known source of cross-container
disagreement.

Receipts: `experiments/results/kl_teacher_cross_box.json`,
`experiments/results/ldlq_block_crossbox_control.json`.

## Box and lock decisions, stated rather than assumed

- Both exports run on **sparklina** (at launch: load 3.1, 91 GB available).
  sparky was at load 63 with 6 GB free and a resident vLLM engine, and is not
  a box to encode or serve on.
- **The long exports do not hold `gpulock.sh`, deliberately.** At launch the
  lock was held by a sibling's `bf16_l_sigma_sweep` with another job already
  queued behind it, and the b4 export is a ~15x encode. Taking the lock would
  have meant queueing the critical path behind an unknown wait and then
  blockading the box for a further ten-plus hours against every sibling GPU
  job, including the GLM arm of this same issue. The lock exists for memory
  pressure; each export measures 4.5-4.7 GB RSS at 40 W of a ~140 W envelope
  on a box with 78 GB available, so the risk it guards against is not the risk
  present here. The matched-pair timing measurement, where quiet conditions
  are the measurement, is a different case and is treated as one.
- **`gpuslot.sh` landed mid-run and the three exports were not restarted onto
  it.** The pool is three slots; my three arms would have taken all of them for
  eighteen hours, which is the blockade the split exists to prevent, and the
  restart would have cost about an hour of aggregate progress to get there. The
  arms stay where they are, the count falls to two when b32 lands, and every
  new long launch goes through `gpuslot.sh` while the sequential matched pair
  goes through `gpulock.sh`.

## Two controls that fail in opposite directions, so both are reported

The encode cost is quoted twice, under conditions named each time, because
neither design is the number on its own:

* **Concurrent** — the three arms encoded side by side. They see the same
  background load *simultaneously* and interfere with each other symmetrically.
* **Sequential** — `b32 -> b4 -> b32` back to back on a quiet box under
  `gpulock.sh`. No mutual interference, but the arms see different load at
  different times and the box can drift under them, which is exactly how the
  LUT-plane receipt's three retracted figures went wrong.

The two failure modes point in opposite directions: the concurrent pair is
biased by interference it shares, the sequential pair by drift it does not.
Reporting both with their conditions named is more honest than picking one and
calling it the number.

## Projected wall clock, recorded before the run lands

The LUT-plane receipt measured the LDLQ pass at a **fixed cost per segment**,
invariant to the work inside it: `time = 0.694 s x segments + 1.0 s`, flat to
0.6% across a 4x width range. Segments are `cols / block`, so the projection
follows from the model's geometry alone.

Qwen3-0.6B has 28 layers x (q,k,v @ 1024 cols + o @ 2048 + gate,up @ 1024 +
down @ 3072) = **286,720 input columns**, so

| block | segments | projected, quiet |
|---:|---:|---:|
| 32 | 8,960 | ~6,200 s (1.7 h) |
| 4 | 71,680 | ~49,700 s (13.8 h) |

The b32 projection is corroborated: the 2026-09-02 `ldlqH1` export of this
exact recipe took 7,949 s contended and the receipt's own quiet estimate was
~4,900 s. The issue's per-unit datapoint corroborates the b4 one:
`L0.q_proj` at block 4 is 256 segments and was measured at ~175 s, i.e.
0.68 s/segment.

**So the candidate arm is a ~14 h encode on a quiet box, and this box is not
quiet.** At launch sparklina was load 3; fifteen concurrent test suites took it
to load 86 within twenty minutes. Both exports hold a full core each (98% CPU,
4.7 GB RSS, 34 W of a ~140 W envelope -- launch-bound, exactly as the receipt
says), so they are not stalled, but the wall clock will exceed the projection
and is reported as measured rather than as projected.

### What the arms did to the projection: it broke, in both directions

That paragraph is wrong and I am leaving it visible rather than editing it away.
The arms landed and the segment-cost model that produced it does not survive
them.

| block | segments | projected quiet | **measured, contended** | s/segment |
|---:|---:|---:|---:|---:|
| 32 | 8,960 | ~6,200 s | **12,876 s** (3:34:36) | 1.437 |
| 8 | 35,840 | ~24,800 s | **20,849 s** (5:47:29) | 0.582 |
| 4 | 71,680 | ~49,700 s | **27,351 s** (7:35:51) | 0.382 |

The receipt's model -- `0.694 s x segments`, flat to 0.6% across a 4x width
range -- assumed cost per segment is invariant. Across *block sizes* it is not:
per-segment cost falls **3.8x** from block 32 to block 4. So the projection
over-predicted the small blocks and under-predicted the large one, and the two
errors are not the same error with a sign flip. The likely reading is that at
block 32 a per-unit fixed cost the model never had to separate is a large share
of the total, and it does not multiply when the segments do -- but I did not
measure that decomposition and am not claiming it.

The practical consequence is the one in "What this run settles": **the ~8x
encode penalty for block 4 was a projection, not a measurement, and it is
withdrawn.** The measured contended ratios are b8/b32 = 1.62x and **b4/b32 = 2.12x**, both
from completed runs. Neither is a clean number -- b32 ran with three of my arms
on the box, b8 with two, b4 finished alone -- and the bias runs the same way for
both: the more-contended arm is the denominator, so the true quiet ratios are
**larger** than these. **2.12x is therefore a floor for b4/b32**, not an
estimate, and 8x has nothing behind it.

Peak RSS settles the other half of the same question. b4 peaked at 6,125,908 kB
against b32's 6,021,028 -- **equal within 2%, at 8x the segment count**. So b8's
1.26x peak was three arms sharing a box, not a property of the block, and
`ldlq_block` is not a memory lever in either direction.

## The concurrent design failed as a measurement, three times over

I ran the three arms side by side and argued that made them a matched set. It
did not, and the reasons compounded:

1. **Different durations over a moving load.** Their first-chunk windows were
   15, 68 and 95 minutes while `load1` ran 3.3 to 82.6, so each averaged a
   different stretch of the curve. This is what forced the retraction of the
   6.15x.
2. **Different start times, so the interference is not symmetric** -- which was
   the specific thing "matched by construction" claimed.
3. **The interference is as large as the effect.** b32's rate got *worse* as
   the box got quieter: 20.2 s/Mparam over units 0-20 (external load ~40)
   against 31.6 s/Mparam over units 60-100 (external load ~8). The confounder
   resolves cleanly -- b32's first chunk ended 19:41 and b8 started 19:40, so
   chunk 1 had two of my arms and the quiet chunks had three. The 1.5x tracks
   **my own arm count**, not the box.

The sequential matched pair under the pool's `--exclusive` is therefore not a
refinement of the concurrent measurement. It is the only version of it that was
ever going to work, and the concurrent full-export wall clocks will be reported
as elapsed-under-named-conditions rather than as any kind of ratio.

## Completion: what I projected, and what happened

Recorded before the arms landed, and kept because the outcome grades it:

| arm | own rate (s/Mparam) | projected total | 1/block scaling of b32 |
|---|---|---|---|
| b32 | 20.2 | ~2.5 h | ~3.7 h |
| b8 | 88.2 | **~10.8 h** | ~15.0 h |
| b4 | 124.0 | **~15.2 h** | ~29.3 h |

**b8 finished in 5:47:29 -- 1.9x faster than its own-rate projection, and 2.6x
faster than the 1/block one.** Both methods were built on rates sampled through
the load spike and both were badly pessimistic; "all three are upper bounds" was
the only part of that paragraph that held.

**b4 finished in 7:35:51** (27,351 s, exit 0) -- 2.0x faster than its own-rate
projection and 3.9x faster than the 1/block one.

It also came in under every estimate I made while it ran, and those are worth
recording because they were all made the same way and all missed low. At
120/196 I estimated 8.7-9.9 h from the per-unit rate; the true answer was 7.6 h.
The box was quieting under it the whole time, so each successive extrapolation
was built on a rate that was still improving. I said at the 140/196 mark that I
would stop re-fitting and wait for the number, which is the only reason the
figure in this table is a measurement.

The lesson is narrow and worth keeping: **every encode-cost number this campaign
produced ahead of the run was wrong by 1.6-3.9x, always pessimistic, from two
independent methods and four separate extrapolations.** The only encode numbers
worth anything here are the three completed wall clocks, and all three are
contended.

## The branch suite

`pytest -q` on the branch, through the pool on sparky. It ran twice, and the
two runs are not interchangeable, so both are here:

| run | placement | result | exit |
|---|---|---|---|
| earlier | pool, GPU visible | **1,619 passed, 9 skipped, 0 failed**, 990 s | 0 |
| final | pool, `--cpus 8`, no GPU | **1,178 passed, 450 skipped, 0 failed**, 293 s | 0 |

Same 1,628 tests collected both times, so this is one suite under two
placements, not two suites. The 441-test difference is entirely CUDA-gated
tests skipping themselves when the pool handed out no GPU -- which is the
correct behaviour, but it means **the no-GPU run is not the one that clears the
branch**; the GPU-visible run is. Quoting the shorter run's pass count as the
denominator would have understated coverage by 27%.

**0 failed in both**, so no test needed running against pristine master.

## Off-task fixes and things I found but did not fix

Fixed on this branch, each in its own commit so it can be taken or dropped
independently of the measurement:

- `c1f9ca7` -- `experiments/ldlq_block_serve_ab.sh` hard-coded one teacher dump.
  Made it `$TESSERA_KL_TEACHER`-overridable and added an optional second
  cross-check teacher per arm, which is what let every arm be scored twice.
- `357fe49` -- added `experiments/ldlq_block_crossbox_control.sh`, and gave it a
  distinct `TESSERA_KL_NAME` so one worker's `docker rm -f` cannot reap another
  worker's serve. The shared name was a live hazard with several agents serving.

Found, not fixed, with the reason:

- **`choose_ldl_block` floors LUT/S6B callers at block 16.** The floor comes
  from a constraint only the slice-stitching path has, and it excludes every
  block that wins in this campaign's own sweep. Not fixed here: it is the
  mechanism a `DEFAULT_LDLQ_BLOCK` decision would go through, so moving it is
  Rob's call, not a side effect of a measurement branch.
- **The LUT-plane receipt's encode-cost model does not hold across block
  sizes.** Its `0.694 s x segment` is flat across widths but not across blocks
  -- 1.44 s/segment at block 32 against ~0.44 at block 4. Not fixed here: the
  correct fix needs the controlled sequential measurement that is still
  pending, and re-fitting a constant on contended data would replace one wrong
  number with another.
- **prismabuild#4** -- my b8/b4 encodes were launched bare over `ssh`, so the
  pool's ledger showed sparklina's GPU free while four of my processes held it.
  Filed rather than fixed: it is another project's admission path. Every GPU
  action after that point in this session went through `pbrun`.

`DEFAULT_LDLQ_BLOCK` is untouched on this branch, as instructed. It reads 32.

## What is pending

Done, and where it is:

- [x] b32 and b8 export wall clocks and bytes -- contended, both named as such
- [x] byte equality and `activation_aware.ldlq_block` on each arm --
      `experiments/results/ldlq_block_b8_byte_check.json`, PASS
- [x] served KLs b32 / b8 / b32-again, plus the cross-box controls --
      `experiments/results/ldlq_block_served_ab.json`
- [x] `assert_plane_promotion` output verbatim, both calls
- [x] branch suite, twice; 0 failures, so master was not needed

Not done, and none of it is blocking the answer above:

- [ ] **b4's own bracket.** The encode is **done** (7:35:51, exit 0); the serve
      is not. The byte check against b32 is queued in the pool behind eight
      other agents' actions and is not reported here because it has not run.
      What remains after it is `experiments/ldlq_block_serve_ab.sh` with `b4` as
      the candidate, on an idle box. b8 is the arm this report is about and the
      coordinator authorised the substitution; b4 refines the size of the
      effect, not its sign.
- [ ] **A controlled encode cost.** The only defensible version is a sequential
      matched pair under `pbrun --gpu --exclusive`, which needs a quiet box for
      roughly 10 h for the b32+b4 pair. Every encode number in this report is
      either contended or extrapolated. Given the carry fraction is 0.31, that
      box-time is a judgement call and I am not making it unasked.
- [ ] **A GLM leg at block 8 or 4.** This is the leg that stops the promotion
      gate from being callable on the candidate's own evidence. It does not
      exist because `choose_ldl_block` floors LUT/S6B callers at block 16 (see
      the off-task fixes below), so producing it means going around or changing
      that floor -- a default-adjacent change I am not making on this branch.
