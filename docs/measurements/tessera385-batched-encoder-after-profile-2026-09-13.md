# Batched LDLQ encoder: closing acceptance box 2 of #385 (after-profile)

Status: **in progress — the after-arm is submitted, not yet harvested.** This file records
what is established, what is submitted, and what remains open, so the next worker resumes
from receipts instead of re-deriving the setup.

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
(`docs/measurements/tessera392-host-syncs-2026-09-07.md`). The lesson worth carrying:
**the `float()` host syncs were a symptom of the launch-bound stream, not its cause.**
Removing them bought nothing measurable. The batch axis was the cause-level fix.

## What this session submitted

Both arms run today's `master` (`00d849dcb`, i.e. post-#386) and compare `B=1` against
`B=32` *within the same binary*, so the delta is attributable to the batch axis alone and
not to any other commit that landed between `f8b87ce` and today. `B=1` is also a
cross-check against the 2026-09-06 before-numbers above.

- BF16_K1@1792, 32 experts, `--batch-sizes 1,32 --arms both --profile`:
  PB action key `4e389075087ca178c239eed8f2fedb5aa6540de7e1bdd4c8530ab33bbb39c2e4`,
  done record `/mnt/shared/prismabuild-fleet/pb-queue/done/4e389075087c...json`,
  out-dir `/mnt/shared/tessera-measurements/tessera385-after-2026-09-13/bf16-b32`.
- E2M1_K2@896, same arms:
  PB action key `8a854e0ecbe7b026726aa3c145a0d9b4bed65c0306cd82626479644787aa2bf2`,
  out-dir `/mnt/shared/tessera-measurements/tessera385-after-2026-09-13/e2m1-b32`.

Both `--tag gb10 --exclusive --priority -10 --cpus 4 --demand mem_gb=64
--gpu-memory-gb 24`, each bounded under 30 min. The `mem_gb=64` demand is inherited from
the 2026-09-06 baseline action on the identical fixture and arms, not re-measured here.

## Open, with the exact resume

1. **Harvest both done-records** and read `results.json` from each out-dir. Report
   params/s at `B=1` and `B=32` per family, and the ratio. Box 2 passes at >=5x on
   E2M1_K2 with the per-unit blobs byte-identical across the two batch sizes (the bench
   records the blob digests; confirm them rather than trusting the ratio).
2. **Power per phase.** Run `/usr/bin/python3 /home/rob/tmp/tessera385/power_by_phase.py
   <out-dir>/results.json` for the host-aware Netdata power series, and report work per
   joule, not only params/s. The claim of this issue is an envelope claim: 16 W of 140 W
   before. If the after-arm does not move the power fraction, the batch axis did not
   recover the envelope regardless of what wall-clock did.
3. **Still unowned from the hand-off:** `want_sse=False` on `viterbi_window` /
   `viterbi_columns` with `_run_joined` passing it — the ~480 remaining per-call syncs on
   the shipping path. Given the #392 negative above, expect this to be worth little; it
   should be measured before it is implemented, not after.

## What this session did NOT do

- Did not harvest the after-numbers; both PB actions were still in `ready` at write time.
- Did not run a shape census of a campaign row. The batch axis is already merged, so the
  census is no longer load-bearing for the decision; it would only size the win.
- Did not touch the encoder. No code change is proposed here — this is a measurement
  record and a resume.
