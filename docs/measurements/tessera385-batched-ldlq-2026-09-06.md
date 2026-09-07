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

(pending)
