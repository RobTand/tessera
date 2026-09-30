# Tessera-8 run tables: the kernel oracle, the chord test and the allowable rule

**Status:** oracle and chord receipts final; the time-penalty sweep is pending
(tessera#750). This receipt backs `formats[TESSERA_E4M3_K1].allowable_rungs`
in contract v50.

## What the rule publishes

`allowable_rungs` admits every rung in `[256, 2048]` at step 1 whose run
table, `rate_set(q256 * code_arity / 256)`, is one of the 15 tables below.
No table and no rung is excluded.

| Tables | Kind |
|---|---|
| `[1]` `[2]` `[3]` `[4]` `[5]` `[6]` `[7]` `[8]` | whole rates |
| `[1,2]` `[2,3]` `[3,4]` `[4,5]` `[5,6]` `[6,7]` `[7,8]` | adjacent pairs |

A table is admitted when the kernels decode it exactly in every launch mode and
a random mix inside it decodes exactly. The mix fraction and its placement
are runtime data, so one attested table covers every rung inside it.

## Oracle

The oracle ran on GB10 through PrismaBuild, in image
`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0…b5f5`.

| Job | Head | Result | Receipt |
|---|---|---|---|
| `946a6479` | `c62803168c` | 418 passed, 0 failed, 0 skipped | `/mnt/shared/tessera-measurements/t8r-speed-20260929/t8-tests-mix2-20260930T154041Z/junit.xml` |

The run covers the `e4m3` and `e4m3mma` fused libraries and the `value`
library, on both routed and dense launches:

- **Every rate exactly.** `test_fused_stages_decode_every_rate_exactly` (66)
  and `test_dense_forward_decodes_every_rate_exactly` (42) decode one-hot
  inputs bit-exactly at rates 1 to 8, on gate/up, down and dense.
- **Every pair, mixed.** `test_random_mixes_inside_every_pair_decode_exactly`
  (17, routed) and `test_dense_random_mixes_inside_every_pair_decode_exactly`
  (21, dense) draw random mix fractions and placements inside each of the
  seven pairs.
  - The routed test uses one placement per projection across experts,
    because a grouped stack holds one packed layout per projection.
  - Gate and up draw different placements.
- **Within bound on real shapes.** `test_fused_stages_at_every_rate_are_within_the_derived_bounds`
  (33), `test_dense_forward_at_every_rate_is_within_the_derived_bound` (36)
  and `test_dense_forward_on_the_glm_role_shapes` (12).
- **Graph capture.** 30 routed and 30 dense capture-and-replay tests.

T-16 attested the same tables on its own library in PB `ee6b92ed`
(297 passed, 0 failed, 0 skipped).

## Contract tests

| Job | Head | Result | Receipt |
|---|---|---|---|
| `3ba48683` | `c60ae71423` | 1144 tests, 0 failed, 17 skipped | `/mnt/shared/tessera-measurements/t8r-speed-20260929/t8-tests-contract3-20260930T155822Z/junit.xml` |

Sixteen skips are box artifacts absent on the runner, such as serve logs and
Qwen3-0.6B checkpoints. The seventeenth is the existing E2M1 case, which
publishes no reader range.

## Chord test: does a fractional rung buy accuracy per byte?

**Question.** Does a rung inside a pair have lower error per byte than a
whole-bit mix of the same bytes across units?

**Setup.**
- 36 GLM-5.3-Flash routed-expert units: experts 0 to 3 of layers 5, 20 and
  42, with gate, up and down each.
- 512 rows per unit, encoded by the production encoder at every rung of
  `{768, 800, …, 1024, 1088, …, 1280}`.
- Two metrics:
  - `w_mse`: weight error.
  - `y_mse`: output error on recorded activations, for 24 gate/up units.
- Errors are summed squared errors, so units of different sizes add.
- **Receipt:** PB `2ee9c9d9`, head `3d5f3714d7`,
  `/mnt/shared/tessera-measurements/t8r-speed-20260929/t8-chord2-20260930T154140Z/p0/rung_chord.json`.
  It was re-analysed with `rung_chord.py --reanalyse`; the original is
  `rung_chord.orig.json`.

**Results.**

| Metric | Pair | Units | Chord residual, median | Within-unit / across-unit |
|---|---|---|---|---|
| `w_mse` | 768–1024 | 36 | +0.0001 to +0.0007 | 1.010 to 1.032 |
| `w_mse` | 1024–1280 | 36 | +0.0023 to +0.0035 | 1.020 to 1.033 |
| `y_mse` | 768–1024 | 24 | −0.0008 to +0.0049 | 1.136 to 1.568 |
| `y_mse` | 1024–1280 | 24 | +0.0041 to +0.0076 | 1.326 to 1.508 |

**Column definitions.**
- **Chord residual.** `(error(q) − chord(q)) / (error(lo) − error(hi))`: the
  distance from the straight line between the pair's whole rates, as a
  fraction of the pair's error drop.
- **Within-unit / across-unit.** The total error of every unit at the
  fractional rung, over the total error of a byte-matched whole-bit mix
  across units. The mix raises whole units in order of gain per bit, which is
  the continuous-knapsack optimum.

**Reading.**
- **A fractional rung sits on the chord.** Its error is the linear
  interpolation of its two whole rates, within 1% of the pair's error drop.
- **So it buys no accuracy per byte over a whole-bit mix across units.** The
  ratio is at least 1 because units differ in gain per bit, and the ratio
  measures that spread.
- **Caveats.**
  - The across-unit mix assumes an allocator that knows each unit's gain.
  - The `y_mse` units span three layers with different output scales, which
    widens the spread.
  - These are proxy metrics on single units, not served KL.
- **What a fractional rung does buy is granularity.** A grouped stack holds
  one packed layout per projection across all 288 experts. A projection is
  therefore one unit of about 2.4 G weights (2048 × 4096 × 288), and a
  whole-bit step on it costs about 302 MB. The rule's step of 1 q256 is
  1/256 bit per weight, about 0.15 GB over the whole routed body
  (304.4 G weights).

## Time penalty of a pair against its whole rates

Pending: the protocol sweep, PB `ff133e6f`, a measurement job on sparklina.
It runs `bench_geometry.py` at every whole rate and at q256 offsets 64, 128
and 192 inside each pair, over the routed and dense protocol shapes, at
M = 1, 64, 512 and 8192. When it lands, this section records each pair's time
against the linear interpolation of its bracketing whole rates.

The exclusion rule is evaluated per run table (lead, #750, 2026-09-30):
exclude a two-run table `[r, r+1]` when the one-run table above it, `[r+1]`,
is at least as fast at every M, beyond the A/B's spread. No table is
excluded until a sweep shows that. The T8R/PACT release recipe runs `[4,5]`
and `[3,4]`, so excluding either is the coordinator's call.
