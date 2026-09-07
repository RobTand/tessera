# tessera#385: the batched LDLQ encoder, measured on the campaign's own units

Status: measurement in progress (PrismaBuild actions queued 2026-09-06); the
tables below are filled from `results.json` / `power_by_phase.json` under
`/mnt/shared/tessera-measurements/tessera385-2026-09-06/` as each lands.

## What was measured

Workload: the PrismaQuant #275 LFM campaign's own units -- 32 layer-18 expert
`w1` projections `[1792, 2048]` with their routed-row Hessians (calibration
wikitext-2-raw-v1/train, nsamples 32, seqlen 512, seed 0; identity
`fit_ids_sha256 809883cb...`), plus the `[7168, 2048]` layer-0 dense `w1` the
#283 profile row ran, captured by `experiments/tessera385_dump_units.py`
(PrismaQuant side) into `units/units.safetensors`. Encoder kwargs come from
`ActivationSource.for_unit` exactly as `tessera_hessian.encoder_kwargs`
builds them (LDLQ sigma 1.0, block 32, `scale_refit` 4, CHANNEL refit
`hessian`, LUT refit `h^1.0`).

Arms, all through `experiments/tessera385_bench.py`:

* `unbatched`: `encode_linear` per unit (the campaign's current path);
* `batched B`: `encode_linears` over the 32 experts in chunks of B;
* before = the same script on `f8b87ce` (origin/master, no batch axis);
  after = this branch.

Instruments (AGENTS.md principle 9 / PrismaQuant principle 15): wall time and
params/s per arm from the harness; Netdata GPU power on the box the run
landed on, read for each phase's epoch window (`power_by_phase.py`);
`torch.profiler` kernel tables for one unbatched and one batched arm.

## Identity receipt

`encoder_fixture_id` before/after, and the per-unit blob sha256 across arms
(the harness refuses to continue on the first mismatch): see below.

## Results

### Before (origin/master f8b87ce, no batch axis) -- PB action a440faa5, sparky, exclusive

`encoder_fixture_id` = `03bbc5b1c56d55e1d7f5f0d1baa1107e462d5bad18a412d0c232c78d04c95519`.
Timed arms (unbatched `encode_linear`, one unit at a time):

| family@rung | units | wall s | Mparam/s |
|---|---|---|---|
| BF16_K1@1792 | 32 x [1792, 2048] experts | 123.8 | 0.949 |
| BF16_K1@1792 | 1 x [7168, 2048] dense | 15.8 | 0.929 |
| E2M1_K2@896 | 32 x [1792, 2048] experts | 420.6 | 0.279 |
| E2M1_K2@896 | 1 x [7168, 2048] dense | 56.0 | 0.262 |

`torch.profiler`, BF16_K1@1792, 8 experts unbatched: `_step` 3,670,016 launches
at 7.6 us mean (27.9 s device), `_traceback` 2048 at 318 us, `aten::mm` 2048 at
43.5 us, `cudaStreamSynchronize` 4336 calls / 28.4 s host. The E2M1 profile
table was lost: the action was OOM-killed at its 40 GB memory limit while the
profiler built the CPU event table for the coset trellis (millions of aten
ops); the bench now profiles CUDA activity only and checkpoints results after
every arm. Netdata power for these arms was not read: `results.json` was not
written before the kill (phase epochs reconstructable from `started_epoch`
1788745041.1 and the walls above; sparky showed 93-95 W / 67% of envelope
during the BF16 arm on the ambient vitals hook).

### L2-budget sweep of one wide window call -- PB action f1f6d8a9, sparky, exclusive

`viterbi_window` on [1792, 1024] fp32 targets, BF16 table L=14, R=7, weighted;
steady-state (persistent plan, third and fourth call) wall per call:

| TESSERA_WINDOW_L2_BYTES | width | batches | steady s |
|---|---|---|---|
| default (L2/6 = 4 MiB) | 32 | 32 | 0.436 |
| 4 MiB | 32 | 32 | 0.436 |
| 8 MiB | 64 | 16 | 0.424 |
| 16 MiB | 128 | 8 | 0.637 |
| 32 MiB | 256 | 4 | 0.779 |

States byte-identical at every width (one sha). A wider tile does not buy the
window body time: the step kernel is bound by front traffic through L2, and
past 64 columns it loses. So batching (PR1) cannot widen the window body's
launches; what it can remove there is per-call overhead only. The lever for
the window body is the front's residency (priority 3 of the issue: a
per-column kernel with the 2^L front on chip), and this sweep is the
measurement that says so.

### After (this branch) -- NOT MEASURED at hand-off

bench-head-02 (PB 4f54a467) and the resubmitted before-arm bench-base-02
(PB 5bf2c98d) were withdrawn at the 2026-09-06 22:19 EDT session wind-down
before admission. Resume: `/usr/bin/python3 /home/rob/tmp/tessera385/submit.py
/home/rob/tmp/tessera385/command-bench-base-02.json` and
`... command-bench-head-02.json`, then
`/usr/bin/python3 /home/rob/tmp/tessera385/power_by_phase.py <out-dir>/results.json`.
