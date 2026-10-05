# Tessera-8 run tables: the kernel oracle, the chord test and the allowable rule

**Status:** oracle, chord and time-penalty receipts final at head
`44b98b5554`; the re-sweep at master `59cfe1b470` is pending (tessera#750).
This receipt backs `formats[TESSERA_E4M3_K1].allowable_rungs` in contract v50.

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

The protocol sweep ran `bench_geometry.py` at every q256 rung from 256 to
2048 in steps of 64: every whole rate, and offsets 64, 128 and 192 inside each
pair. Each rung was timed at M = 1, 64, 512 and 8192.

- **Launches:** routed gate/up and down (288 experts, balanced routing, plus
  recorded GLM-5.3 routing at M = 512 and 8192), and dense `o_proj`, `q_b`
  and `kda_in_12416`. Dense `kda_in` (12448 rows) needs the partial last
  row block, which this head does not have.
- **Source:** head `44b98b5554`, kernel sha `faf8f636`, image `f8dbe1a0`, the
  E4M3 instruction library, on one GB10. The head has the per-pair kernel and
  the descriptor ring, and not the staged stream history (#763).
- **Statistic:** the mean of the forward and reverse passes' graph-replay
  medians. Spread is |F − R| / mean: 0 to 1% for most cells, up to 10% for
  dense cells at M = 8192.
- **Receipt:** PB `ff133e6f`,
  `/mnt/shared/tessera-measurements/t8r-speed-20260929/t8-geo-protocol-20260930T151837Z/`
  (`p0/bench_geometry_routed.json`, `p1/bench_geometry_dense.json`).
  `experiments/t8r_speed/pair_time.py` reduces them to the tables below and
  evaluates the rule per table.

In each table, a cell holds the range of the pair's rungs' time against the
whole rate above it, `[r+1]`, over the pair's rungs and routings. After the
slash is the median time against the linear interpolation of the two whole
rates.

#### Routed gate/up

| Table | M=1 | M=64 | M=512 | M=8192 |
|---|---:|---:|---:|---:|
| `[1,2]` | +53..+56% / +52% | +53..+55% / +54% | +48..+52% / +52% | +37..+42% / +40% |
| `[2,3]` | +43..+48% / +50% | +40..+45% / +49% | +40..+45% / +47% | +24..+32% / +35% |
| `[3,4]` | +10..+13% / +25% | +17..+19% / +28% | +16..+18% / +27% | +24..+31% / +27% |
| `[4,5]` | +20..+22% / +18% | +4..+10% / +20% | +4..+8% / +16% | +16..+20% / +22% |
| `[5,6]` | −3..+3% / +11% | −5..+1% / +4% | −4..+4% / +6% | +27..+32% / +29% |
| `[6,7]` | −1..+5% / +7% | −6..+1% / +6% | −12..−3% / +2% | +11..+17% / +21% |
| `[7,8]` | +3..+8% / +9% | 0..+3% / +4% | −5..+2% / +1% | +12..+18% / +15% |

#### Routed down

| Table | M=1 | M=64 | M=512 | M=8192 |
|---|---:|---:|---:|---:|
| `[1,2]` | +49..+54% / +51% | +49..+54% / +53% | +48..+52% / +51% | +27..+40% / +35% |
| `[2,3]` | +40..+45% / +47% | +40..+45% / +48% | +36..+43% / +45% | +22..+33% / +32% |
| `[3,4]` | +52..+53% / +42% | +36..+38% / +37% | +33..+36% / +34% | +23..+26% / +23% |
| `[4,5]` | +24..+34% / +41% | +23..+32% / +34% | +20..+28% / +31% | +13..+18% / +20% |
| `[5,6]` | +42..+46% / +40% | +19..+23% / +28% | +17..+21% / +26% | +22..+31% / +26% |
| `[6,7]` | +3..+9% / +22% | +9..+13% / +17% | +8..+13% / +17% | +17..+27% / +24% |
| `[7,8]` | −21..−12% / −1% | −19..−10% / +0% | −15..−6% / +3% | +11..+12% / +16% |

#### Dense `o_proj`

| Table | M=1 | M=64 | M=512 | M=8192 |
|---|---:|---:|---:|---:|
| `[1,2]` | +40..+47% / +39% | +41..+49% / +39% | +41..+50% / +41% | −2..−1% / 0% |
| `[2,3]` | +35..+42% / +41% | +29..+36% / +41% | +35..+43% / +41% | −3..−2% / −2% |
| `[3,4]` | +42..+45% / +35% | +35..+38% / +30% | +41..+43% / +35% | −3..+3% / −2% |
| `[4,5]` | +26..+34% / +39% | +23..+31% / +34% | +25..+34% / +38% | −7..−3% / −3% |
| `[5,6]` | +39..+47% / +40% | +33..+41% / +35% | +39..+49% / +40% | −3..+1% / −4% |
| `[6,7]` | +30..+37% / +37% | +25..+33% / +35% | +29..+37% / +38% | −1..+1% / −1% |
| `[7,8]` | +41..+44% / +35% | +35..+36% / +30% | +35..+37% / +32% | −1..+1% / +1% |

#### Dense `q_b`

| Table | M=1 | M=64 | M=512 | M=8192 |
|---|---:|---:|---:|---:|
| `[1,2]` | +39..+44% / +38% | +37..+43% / +35% | +37..+45% / +38% | +32..+39% / +35% |
| `[2,3]` | +32..+42% / +39% | +28..+37% / +37% | +30..+40% / +38% | +28..+36% / +34% |
| `[3,4]` | +39..+42% / +33% | +34..+35% / +30% | +37..+40% / +32% | +32..+35% / +29% |
| `[4,5]` | +27..+33% / +37% | +21..+28% / +31% | +22..+31% / +34% | +22..+30% / +32% |
| `[5,6]` | +41..+46% / +40% | +32..+39% / +33% | +35..+42% / +36% | +33..+40% / +35% |
| `[6,7]` | +29..+36% / +37% | +24..+32% / +31% | +24..+32% / +33% | +21..+30% / +31% |
| `[7,8]` | +38..+42% / +33% | +30..+33% / +27% | +31..+34% / +27% | +27..+30% / +24% |

#### Dense `kda_in_12416`

| Table | M=1 | M=64 | M=512 | M=8192 |
|---|---:|---:|---:|---:|
| `[1,2]` | +46..+54% / +44% | +42..+51% / +41% | +41..+50% / +41% | −6..−3% / −3% |
| `[2,3]` | +38..+47% / +44% | +34..+44% / +41% | +36..+46% / +41% | −3..−2% / −2% |
| `[3,4]` | +46..+49% / +39% | +33..+37% / +31% | +36..+40% / +33% | +2..+3% / +0% |
| `[4,5]` | +18..+26% / +37% | +13..+20% / +29% | +27..+34% / +37% | −9..−2% / −2% |
| `[5,6]` | +8..+14% / +20% | +9..+13% / +18% | +37..+45% / +40% | −3..+1% / −3% |
| `[6,7]` | 0..+9% / +11% | −4..+8% / +10% | +26..+33% / +35% | −4..−3% / −1% |
| `[7,8]` | +2..+8% / +11% | +5..+11% / +13% | +29..+32% / +29% | −4..−2% / −4% |

### What the sweep shows

- **A pair costs more than its interpolation.** At M ≤ 512, a pair's median
  time is 16% to 54% above the interpolation of its two whole rates on most
  tables and launches. The exceptions are routed gate/up `[5,6]` to `[7,8]`
  (+1% to +11%), routed down `[7,8]` (−1% to +3%), and `kda_in_12416`
  `[6,7]` and `[7,8]` at M ≤ 64 (+10% to +13%).
- **The release tables are slower than the whole rate above them.** The
  T8R/PACT recipe runs `[4,5]` and `[3,4]` on its routed layers. At M = 1,
  `[4,5]` runs 20% to 22% slower than `[5]` on gate/up and 24% to 34% slower
  on down. `[3,4]` runs 10% to 13% slower than `[4]` on gate/up and 52% to
  53% slower on down. A uniform `[5]` stack therefore decodes faster than the
  `[4,5]` mix while carrying more bytes.
- **The cost belongs to the two-run path.** On dense `o_proj`, whole rate 4
  runs `routed_fused_kernel<..., 4, false, 64>` in 62 µs. At q1088, `[4,5]`
  runs `<..., 4, true, 64>` in 92 µs for 6% more bytes. The odd whole rates
  (1, 3, 5, 7) run within 15% of their even neighbours, so the unaligned
  8-byte tail is not the main cause.
- **Routed down at whole rate 8 is off the roofline.** At M = 1 it takes
  247 µs (0.59 of the byte roofline), against 173 µs (0.73) at rate 7. That
  is why `[7,8]` beats `[8]` on down by 6% to 21% at M ≤ 512.
- **Two dense shapes do not separate the rates at M = 8192.** On `o_proj`
  and `kda_in_12416`, every pair reads within −9% to +3% of `[r+1]`, and
  spreads there reach 10%. `q_b` still separates them (+21% to +39%).
- **Dense M = 1 cells are warm-L2.** The sweep replays one module's weights
  with no L2 flush. At M = 1 the `scaled_mm` reference on `o_proj` reads
  44 µs against a 72 µs DRAM floor for its 16.8 MB. Ratios within a launch
  compare like with like. The absolute dense M = 1 times are optimistic for
  serving; price those from `bench_dense_module.py --l2 cold`.

### The exclusion rule, applied

The rule is evaluated per run table (lead, #750, 2026-09-30): exclude a
two-run table `[r, r+1]` when `[r+1]` is at least as fast at every M, beyond
the spread.

- **Ties counted as "at least as fast":** `[3,4]` meets the rule. `[4]` is
  faster beyond the spread in 65 of its 72 cells, and the other 7 are ties.
  - `[1,2]`, `[2,3]` and `[4,5]` are blocked only by `o_proj` and
    `kda_in_12416` cells at M = 8192, which do not separate the rates.
  - `[5,6]`, `[6,7]` and `[7,8]` are blocked by routed cells at M ≤ 512,
    where `[r+1]` is itself slow. `[6,7]` and `[7,8]` are also blocked by
    `kda_in_12416` at M ≤ 64.
- **Strictly (faster beyond the spread in every cell):** no table is excluded.

The T8R/PACT release recipe runs `[4,5]` on 17 routed layers and `[3,4]` on
3, so excluding either is the coordinator's call. This receipt excludes
nothing, and `allowable_rungs` is unchanged. Until a kernel change lands,
price each rung from its measured time, not from the interpolation.

### Re-sweep at master

Pending: the same sweep at master `59cfe1b470`, PB `7e70e6d4`, which adds the
staged stream history (#763), output under
`/mnt/shared/tessera-measurements/t8r-speed-20260929/t8-geo-master-20260930T210113Z`.
