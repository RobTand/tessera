# Batched LDLQ encoder: closing acceptance box 2 of #385 (after-profile)

Status: **complete.** Both after-arms are harvested, box-level power is read over each arm's
own window, and the in-process profile of the TCQ/LUT body exists for the first time. #385
is closed. This file is the measurement record.

Date: 2026-09-13. Tree: `master` at `00d849dcb`. Box: GB10 / DGX Spark, 140 W envelope.

## What was already true before this session

The batch axis of #385 is **merged, not pending**. PR #386 ("Batch the LDLQ encoder along
the column axis, byte-identical per unit") merged 2026-09-07 as `382a1a97`, giving
`encode.encode_units` / `export.encode_linears` with `encode_unit` as the same driver at
`B=1`. The campaign has run on that encoder since. What was never done is acceptance
box 2: **the profiled after-delta**. The hand-off recorded "After: not measured" because
bench-head-02 (PB `4f54a467`) was withdrawn before admission.

So the open question on this issue is not "should we batch" — that shipped — but
"what did batching actually buy, in params/s and in work per joule".

## The before-profile of record

Measured 2026-09-06 on the unbatched encoder at `f8b87ce` (PB `a440faa5`, sparky
exclusive), on the campaign's own units: 32 LFM L18 expert `w1` `[1792, 2048]` plus the
L0 `w1` `[7168, 2048]` dense unit, routed Hessians. Fixture:
`/mnt/shared/tessera-measurements/tessera385-2026-09-06/units`.

| Arm | Throughput | Wall (32 experts) |
|---|---|---|
| BF16_K1@1792, experts | 0.949 Mparam/s | 123.8 s |
| BF16_K1@1792, dense | 0.929 Mparam/s | — |
| E2M1_K2@896, experts | 0.279 Mparam/s | 420.6 s |
| E2M1_K2@896, dense | 0.262 Mparam/s | — |

In-process profile (BF16, 8 experts): `_step` **3,670,016 launches at 7.6 us**,
`cudaStreamSynchronize` 4336 calls / 28.4 s. Box-level: **16 W of a 140 W envelope**
(11% of the envelope) with `gpu_utilization` reading 96%. On GB10 that utilization
figure is non-diagnostic; the power fraction is the signal, and it says the device was
running at roughly one ninth of its envelope while the CPU was parked in
`cuStreamSynchronize` draining a launch-bound stream.

## Two adjacent results that bound what batching had to beat

- **Quality-trade levers are not a substitute** (PQ #283 factor screen,
  `/mnt/shared/tessera-measurements/pq283-2026-09-06/README.md`). Turning the Hessian
  off is 6.0x on the TCQ body but costs 1.62x (E2M1_K2), 3.25x (E4M3) and 3.95x (BF16)
  in dloss; `ldl_block` 32->128 is 2.6x for +2.8% dloss, 32->512 is 4.6x for +12.7%;
  `scale_refit` 4->1 is ~3.2x for +5-25%. Batching is the only lever that keeps the
  shipping numerics, which is why box 2's target was >=5x at B=32 *at zero dloss change*.
- **The window body cannot be widened by batching** (L2 sweep, PB `f1f6d8a9`): one
  `viterbi_window` call on `[1792, 1024]`, L=14, R=7 took 0.436 s at width 32, 0.424 s
  at 64, 0.637 s at 128 and 0.779 s at 256, with identical states. That kernel is
  L2-traffic bound. The launch-bound body is the TCQ/LUT one (E2M1_K2), which is where
  the batch axis was expected to pay. **A near-1.0x BF16 result is therefore a
  prediction of this measurement, not a failure of the patch** — read the two families
  separately.

## The negative result that is already durable on this issue

Priority 2 (host-sync removal, PR #392) was **measured and rejected**, not superseded:
BF16 123.4105 -> 123.6967 s (0.9977x), E2M1 54.2373 -> 54.5267 s (0.9947x), work per
joule 1.0004x and 0.9971x, all 64 family/unit blobs byte-unchanged
(`docs/measurements/tessera392-host-syncs-2026-09-07.md`, recovered onto `master` with
this change — it had been stranded on the closed PR #392 branch at `8f7ab6522`).
The lesson worth carrying: **the `float()` host syncs were a symptom of the launch-bound
stream, not its cause.** Removing them bought nothing measurable. The batch axis was the
cause-level fix. The profile below now confirms that from the other instrument.

## Throughput, harvested

Both arms ran today's `master` (`00d849dcb`, post-#386) and compare `B=1` against `B=32`
*within the same binary*, so the delta is attributable to the batch axis alone.

| Body | unbatched B=1 | batched B=1 | batched B=32 | B32 / B1 |
|---|---:|---:|---:|---:|
| E2M1_K2@896 (TCQ/LUT) | 0.307 Mparam/s | 0.307 | **2.318** | **7.55x** |
| BF16_K1@1792 (window) | 0.972 Mparam/s | 0.971 | 0.981 | 1.01x |

**Box 2's >=5x bar is met on the TCQ/LUT body at 7.55x.** The window body's 1.01x is the
predicted result, not a regression: the power table below shows it already runs at 64% of
the envelope at `B=1`, so batching had nothing to recover there.

Per-unit blob identity holds across everything: `encoder_fixture_id`
`03bbc5b1c56d55e1d7f5f0d1baa1107e462d5bad18a412d0c232c78d04c95519` is unchanged from the
pre-#386 record, and the 32 mutually-distinct per-unit blob digests are identical across
all six arms within each family **and across all three actions** (32/32).

## Power and work per joule

Read from Netdata `nvidia_smi.gpu_power_draw` on sparklina, trapezoid-integrated over each
arm's own harness-recorded `start_epoch`/`end_epoch`. 140 W envelope, ~4.3 W idle floor,
117,440,512 quantizable parameters per arm. Tier-0 collection is `update_every: 10`, so
`n` is the number of native samples inside the window and is the honest resolution bound.

| Body | Arm | Window | n | Mean W | Envelope | params/J |
|---|---|---:|---:|---:|---:|---:|
| BF16_K1@1792 | unbatched B=1 | 120.8 s | 12 | 89.16 | 64% | 10,900 |
| BF16_K1@1792 | batched B=1 | 120.9 s | 12 | 85.26 | 61% | 11,393 |
| BF16_K1@1792 | batched B=32 | 119.7 s | 12 | 89.02 | 64% | 11,016 |
| E2M1_K2@896 | unbatched B=1 | 382.8 s | 38 | 18.02 | 13% | 17,023 |
| E2M1_K2@896 | batched B=1 | 382.6 s | 38 | 18.64 | 13% | 16,466 |
| E2M1_K2@896 | **batched B=32** | 307.1 s | 31 | **37.97** | **27%** | **60,425** |
| E2M1_K2@896 | unbatched B=1 (control) | 385.0 s | 39 | 17.64 | 13% | 17,292 |

**Work per joule: E2M1_K2 3.49x, BF16_K1 1.01x.** The TCQ body went 7.52x faster at 2.15x
the power, so the energy win is real but not free. Above the idle floor it is 2.98x.

Two independent cross-checks agree. The unbatched E2M1 arm read 18.02 W in one action and
17.64 W in another, reproducing this issue's opening "16 W of 140 W". And the PR #392
measurement, taken on **sparky** with a different instrument (the PB broker's own power
integration over each phase, 50-117 samples per phase, rather than a Netdata trapezoid over
an arm window), read BF16 at 88.58 W / ~63% of envelope against the 89.16 W / 64% here.

### Correction: a superseded work-per-joule number

An earlier read of the E2M1 batched-B32 arm gave **4.40x** work per joule. That is
superseded by **3.49x**. The first read integrated a 50.7 s window holding about 5 native
Netdata samples, with ramp-in and ramp-out inside it; the arm was re-run six times back to
back inside one exclusive process to give a contiguous 307.1 s span with 31 samples. The
short window **understated mean power by 7 W** (30.92 W vs 37.97 W) and so overstated work
per joule. The durable rule: at `update_every: 10`, a sub-60 s window is not a power
measurement, and the error is biased low because the ramp sits inside it.

## The in-process profile of the TCQ/LUT body

This is the instrument that was missing. The before-profile of record covers BF16 only:
the 2026-09-06 E2M1 capture was OOM-killed building its event table, and the 2026-09-13
one died inside the capture. Both are now explained and one of them is fixed.

Captured on sparklina, exclusive, `--only-profile` with CUDA activities only,
`key_averages()` summary, no chrome trace, `acc_events` left off, and `profile_warmup`
outside the profiled region. Widest reachable batch is `B=8`: `--profile-experts` caps at
8, so `B=32` itself cannot be profiled by this harness.

Per unit-column, at a matched 256-column span:

| Arm | launches / unit-col | device ms / unit-col | mean kernel |
|---|---:|---:|---:|
| unbatched `B=1` | 4,137 | 5.80 | 1.40 us |
| batched `B=2` | 2,168 | 3.28 | 1.51 us |
| batched `B=8` | 608 | 1.15 | 1.89 us |

Launch count per unit-column fits **`L(B) = 4033/B + 104`** — fit on `B=1` and `B=8`, and
the `B=2` point, which is not in the fit, checks to 2.2%. The `104` is the un-batchable
floor, 2.5% of the `B=1` count, so batching alone tops out near **18x fewer launches at
`B=32`**. Measured throughput at `B=32` was 7.55x, well short of that, and the mean kernel
grows 1.40 -> 1.89 us with batch: the launches are being amortized, and the kernels are
starting to do real arithmetic instead of just starting and stopping.

Where the device time goes at `B=8`, the closest reachable point to the shipping `B=32`:

| Share | Device s | Launches | Mean | Kernel family |
|---:|---:|---:|---:|---|
| 36.0% | 0.849 | 614,579 | 1.38 us | elementwise arithmetic |
| 28.1% | 0.664 | 154,280 | 4.30 us | reductions (scale / CHANNEL refit) |
| 27.6% | 0.651 | 264,669 | 2.46 us | index / gather (LUT + coset lookup) |
| 5.2% | 0.122 | 152,626 | 0.80 us | device-to-device copy |
| 0.1% | 0.002 | 224 | 7.12 us | GEMM |

**The body is still launch-bound after batching.** 1,245,109 device kernels for 2048
unit-columns, mean 1.89 us, exactly **two** kernel types averaging above 20 us, and GEMM is
0.1% of device time. Batching changed the launch *count* by roughly `1/B`; it did not
change the *mix*, which is the same four generic families in nearly the same proportions at
`B=1` and `B=8`. That is the expected signature of a change that is byte-identical per
unit, and it says plainly what the next lever is: these are generic PyTorch elementwise,
reduction and gather kernels, not a fused trellis kernel.

The profile also re-confirms the #392 negative from the other instrument.
`cudaStreamSynchronize` is called 23,668 times at `B=1` and 23,444 times at `B=8` — the
count barely moves — but the CPU time inside those calls collapses from **9.326 s to
1.374 s**. The syncs were waiting on a launch-bound stream. Removing them was always going
to buy nothing; widening the launches is what drained them.

### Instrument gap: the full span cannot be captured on this box

Recorded against principle 15 as an **instrument failure, not a negative result about the
encoder** — the two look identical in a log, so this section says which.

The profiler's host-side event table for this body is **linear in the captured span**, not
pathological. Four completed captures, all measured the same way — child `ru_maxrss` reported
by the rung runner, which is the whole-process host high-water mark:

| Capture | unit-columns | peak host RSS (`ru_maxrss`) | Outcome |
|---|---:|---:|---|
| 2 units x 64 cols, `B=32` | 128 | 6.347 GiB | completed |
| 2 units x 256 cols, `B=32` | 512 | 15.411 GiB | completed |
| 8 units x 64 cols, `B=8` | 512 | 15.904 GiB | completed |
| 8 units x 256 cols, `B=8` | 2048 | 47.473 GiB | completed |
| 2 units x 2048 cols, `B=32` | 4096 | ~100 GiB predicted | **OOM at 16 GiB, then at 64 GiB** |

Pooled across all four, `peak = 4.38 GiB + 21.6 MiB x unit-columns`, r^2 = **0.999**. Split by
configuration, the two 2-unit rungs — the same shape as the run that died — give
`peak = 3.33 GiB + 24.2 MiB x unit-columns`, and the two 8-unit rungs give
`5.38 GiB + 21.0 MiB x unit-columns`. **The trace is not unbounded; it is linear at ~21-24 MiB
per unit-column**, which is why the full 2 units x 2048 columns needs about **100 GiB** and
does not fit on a 128 GiB box.

Both deaths agree with that law, and they pin it down independently. At `mem_gb=16` the action
was OOM-killed at exactly 16.00 GiB after 123 s; at `mem_gb=64`, at exactly 64.00 GiB after
377 s. Both carry `oom_local: 1` with the box's own `psi_mem_full_avg10_max` at 0.0, so both
kills came from the action's own cgroup while the box itself was never under memory pressure.
Solving those two points for the two unknowns gives a capture fill rate of **193 MiB/s** and a
fixed pre-capture cost of **56 s** — and 193 MiB/s divided by the ladder's 24.2 MiB/unit-column
is 8.0 unit-columns per second, so the two instruments' numbers are the same number. The
16 GiB ceiling is reached 13% into the span and the 64 GiB ceiling 63% into it.

On GB10, CUDA bytes are charged to no memcg, so `mem_gb` governs host RSS only — which is
exactly what the event table consumes, and why `mem_gb` is the right knob and a `--gpu-memory-gb`
increase would have changed nothing.

Four times the reservation bought about three times the runtime and the same death. **A bigger
reservation is not the fix**; a bounded span is, and a bounded span answers the question.
Size any future capture of this body from the law above — not from a timed run's footprint, as
the timed run peaks at 3.141 GiB RSS plus 6.949 GiB CUDA, a different quantity and the number
that caused all three deaths.

## Correction: how the 2026-09-13 E2M1 action actually died

Earlier notes described PB `8a854e0ecbe7b026` as a lease loss during the trailing
`profile_warmup`. **The record says otherwise and the record wins:** `status: "timeout"`,
`execution_governed_by: "deadline"`, `execution_timeout_s: 1750.0`, `elapsed_s: 1766.996`.
It hit its own `--timeout-s`. The timed arms consumed 1741 s before the profiler started,
`profile_warmup.unbatched.B1.r0` got 25.0 s, and the deadline killed it inside the capture.
Every timed row had already been checkpointed, so the throughput data above stands. The
operational lesson is that a profile belongs in its own action with `--only-profile`, not
appended to a full timing run.

## Evidence

| What | PB action key | Record |
|---|---|---|
| BF16 after-arms | `4e389075087ca178c239eed8f2fedb5aa6540de7e1bdd4c8530ab33bbb39c2e4` | `done/` |
| E2M1 after-arms | `8a854e0ecbe7b026726aa3c145a0d9b4bed65c0306cd82626479644787aa2bf2` | `failed/` (deadline; rows checkpointed) |
| E2M1 B32 power window + control | `f64a907b15e5ba1beb2a5ea7d388ac710c3b2134c12e1307cadbcaeb06dbf294` | `done/` |
| Bounded profile, `B=1` / `B=2` | `b7930f92bc224c9635b80dc1253eff94e947e27cfbb056e06f339b443da32ec0` | `failed/` (rungs r1, r2 completed) |
| Full-span profile at 64 GiB | `04fc0c8ca1a206a98a043674944e623ff0be6c6e86ccdbb5f0891c952eec1a05` | `failed/` (OOM; the instrument-gap datum) |
| Profile, real `B=8` | `c9de78f653f10cfbb1db1e688221cf9ae1bdf34892e4bb538e82f841287a6a26` | `done/` |

Records are under `/mnt/shared/prismabuild-fleet/pb-queue/<state>/<key>.json`. Profile
tables and receipts are under
`/mnt/shared/tessera-measurements/tessera385-profile-2026-09-13/`, power receipts under
`/mnt/shared/tessera-measurements/tessera385-power-2026-09-13/`. The runners that produced
the bounded captures are `experiments/run_385_profile.sh` and
`experiments/run_385_profile_b8.sh`, added with this change.

## What remains, and where it lives

#385 is closed: all four acceptance items have a measured disposition, and box 2's bar is
met on the body the bar is scoped to. Two levers the measurements leave on the table are
carried forward rather than closed with it:

- **#483** — the window body's front-residency lever, plus the `want_sse=False` leftover
  from this issue's hand-off. Baseline to beat is 0.972 Mparam/s at 89.16 W / 64% of
  envelope, 10,900 params/J.
- **#486** — the TCQ body's own headroom. After batching it sits at 27% of the 140 W
  envelope, so the envelope claim in #385's title is only partly paid off. The profile above
  names the lever: 92% of device time is in elementwise, reduction and gather kernels
  averaging 1-4 us, so fusing the per-column refit reduction and the LUT/coset gather is a
  kernel change, not a scheduling change, and batching cannot reach it. Baseline to beat is
  37.97 W / 27% of envelope, 60,425 params/J, 608 launches per unit-column at `B=8`.
