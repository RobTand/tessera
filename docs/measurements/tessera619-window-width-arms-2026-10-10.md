# Tessera#619: window Viterbi width arms at GLM shapes (GB10)

**Date:** 2026-10-10 · **Issue:** tessera#619 · **Runner:** PrismaBuild only
**Probe:** `experiments/ig619_viterbi_power.py` at branch `tessera-619`
commit `29904621ad` · **Shape:** 2048x4096, BF16 q256 1024 and E4M3 q256 896,
L=14 window body over the CHANNEL plane, weights-only fresh encode, 4 units
**Actions:** baseline `159f3d9d` (sparklina), wide-tile `--l2-div 2`
`f5ab2f20` (sparklina), best-form `--best-form 1` `20c2ad4c` (sparky).
Before-data: retry-4 `faa28dad` (sparklina) replicates the baseline exactly
(BF16 0.2111 vs 0.2116 u/s), so the measurement is stable.

## Identity first

Every arm writes the same bytes. Blob sha256 matches across baseline,
wide-tile and best-form at both rungs, and batch4 matches single inside
every arm. No arm moved a byte, so the clocks compare spellings of one
answer. The earlier `identical=False` (retry-2 `17035b5d`) was a probe
artifact: the single arm used the default unit name while the batch arm
used `u0..u3`, and the manifest binds the name.

## Throughput, power, energy (single-unit arms)

Power is the in-process nvidia-smi sampler over each arm's own window,
against the 140 W envelope. Netdata `nvidia_smi.gpu_power_draw` agrees
(BF16 single: 68.3 W vs 68.96 W; best-form single: 88.2 W vs 85.95 W).

| arm | BF16 u/s | BF16 W (frac) | BF16 J/unit | E4M3 u/s | E4M3 W (frac) | E4M3 J/unit |
|---|---|---:|---:|---:|---:|---:|
| front, L2/6 (baseline) | 0.2116 | 68.96 (0.493) | 325.9 | 0.2227 | 75.11 (0.536) | 337.3 |
| front, L2/2 (wide) | 0.0896 | 48.52 (0.347) | 541.4 | 0.0875 | 50.35 (0.360) | 575.1 |
| best-form | 0.5962 | 85.95 (0.614) | 144.2 | 0.5977 | 82.68 (0.590) | 138.3 |

Ratios against baseline (BF16): wide-tile 0.42x wall at 1.66x joules;
best-form **2.82x wall at 0.44x joules**. Batch4 matches single within
noise in every arm (BF16: 0.2110, 0.0895, 0.5903 u/s), so joining units
buys nothing on the window body, as `encode_units` already states.

## Plans and profiler (BF16 rung)

Arity 1, steps 2048, size 16384, chunk 512, rates R3+R4 mixed columns.

| arm | width | batches (R4/4096) | grid | top kernel | self ms | launches | gets per launch |
|---|---|---:|---|---|---:|---:|---:|
| baseline | 32 | 128 | [16, 16] | `_step` | 4583 | 1048576 | 4.37 us |
| wide | 96 | 48 | [16, 48] | `_step` | 11018 | 393216 | 28.0 us |
| best | 512 | 8 | [16, 256] | `_step_best` | 1552 | 65504 | 23.7 us |

`_traceback` costs ~37 ms in all arms. The wide tile runs 3x the columns
per launch at 6.4x the time: its 12.6 MB resident pair thrashes the 24 MB
L2 it shares with fronts, tables and the second rate stream, while the
box reports 89% mean utilization. Utilization is not saturation here.

## Reading

- The time sits in `_step`: ~1M launches of ~4 us each, serial along the
  2048-step trellis. Occupancy per launch is small (256 CTAs) and the
  chain cannot overlap with itself.
- A wider front-form tile loses: fewer launches, much slower launches.
  The sixth-of-L2 budget is near its optimum at these shapes.
- The best-form wins by removing the front from the recurrence: 16x the
  columns per batch at 1/16 the batches, byte-identical. This replicates
  the 2026-09-19 same-box receipt (3.34x wall, 3.98x work per joule).
- Box confound: best-form ran on sparky, the rest on sparklina. The 2.8x
  effect exceeds box variance, and the prior receipt is same-box.
- The diagnostic SSE sync and the rate streams are already mitigated
  (device-side accumulator, `want_sse=False` in `_run_joined`, one stream
  per rate). The remainder is the serial step chain itself.

## Conclusion

No default moves. Widening the budget is measured worse, and promoting
the best-form moves the shipping schedule, which only Rob prices
(tessera#483). This file is the evidence for that decision.
