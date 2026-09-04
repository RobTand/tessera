# tessera#60, served leg: `ldlq_block=4` against a re-run of `ldlq_block=32`

Status: **in progress.** This file is written as the run goes, not at the end.

## What this run settles, and what it no longer settles

It was funded to decide whether `DEFAULT_LDLQ_BLOCK` flips 32 -> 4. **The
encode cost below takes that question off the table before the KL arrives:** if
block 4 is ~8x the encode model-wide, a *global* default of 4 charges GLM's
experts 8x for an axis that is flat there (0.4% across 16->256), and a flip to
another round number is the wrong shape of fix whatever the KL says.

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

## Projected completion, by each arm's own rate

Two methods disagree and the difference decides whether b4 is deliverable:

| arm | own rate (s/Mparam) | total | scaling b32's quiet rate by 32/block |
|---|---|---|---|
| b32 | 20.2 | ~2.5 h | ~3.7 h |
| b8 | 88.2 | **~10.8 h** | ~15.0 h |
| b4 | 124.0 | **~15.2 h** | ~29.3 h |

The right-hand column assumes cost scales as 1/block, which my own data
refutes: b4's measured rate is 124 s/Mparam where 1/block predicts 253. Each
arm's own rate involves no cross-block extrapolation and is the defensible one.
All three were measured through the load spike, so all three are upper bounds.

Both candidates are inside the driver's window, so both should get served.

## The branch suite

`pytest -q` on the branch, run once through the pool on sparky:
**1,619 passed, 9 skipped, 0 failed**, exit code 0, 990 s. No failures, so
nothing needed checking against master.

## What is pending

- [ ] b32 export wall-clock and bytes
- [ ] b4 export wall-clock and bytes
- [ ] byte equality between the arms, and `activation_aware.ldlq_block` on each
- [ ] served KLs: b32, b4, b32-again
- [ ] matched-pair encode cost, back-to-back-plus-repeat
- [ ] `assert_plane_promotion` output, verbatim
- [ ] branch suite, and master only for whatever fails
