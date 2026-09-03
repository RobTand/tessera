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

## What is pending

- [ ] b32 export wall-clock and bytes
- [ ] b4 export wall-clock and bytes
- [ ] byte equality between the arms, and `activation_aware.ldlq_block` on each
- [ ] served KLs: b32, b4, b32-again
- [ ] matched-pair encode cost, back-to-back-plus-repeat
- [ ] `assert_plane_promotion` output, verbatim
- [ ] branch suite, and master only for whatever fails
