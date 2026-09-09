# The best-form step's tile was never measured.  Measured, it loses

> **RETRACTED, 2026-09-09, and being re-measured.** Every timed number below
> came from a nine-arm cycle against a plan cache that holds
> `_WINDOW_PLAN_CACHE = 8`. Nine distinct plan keys cycled in a fixed order is
> the pathological case for an LRU: the entry evicted is always the one about
> to be asked for, so every arm rebuilt its plan and re-captured its graph at
> the head of every block, inside the clock and inside the energy bracket.
> Root read that off the source; this file's own data confirms it, because
> each arm's block median exceeds the hot single call it recorded for the
> same arm by 0.02 to 0.23 s a block, and the amount differs per arm, so it
> does not cancel in a ratio. **No seconds, watt or work-per-joule figure
> below is a result, and none of them should be cited**, including the
> 90 to 93 W band.
>
> What survives, because it was not timed: every arm returned the reference's
> states and the identical `sse`, the launch geometry and the register and
> spill counts, and the separate three-arm profile actions, which fit the
> cache. The re-measurement runs four arms an action.
>
> *One clear and one warm call per arm establishes that an arm was warm once,
> not that it was warm when it was measured. The screen now counts plans built
> either side of every block and refuses to report a block that built one, and
> refuses at startup to run more arms than the cache holds.*

## Verdict (RETRACTED, see the banner)

**`_tile_best` returns `128x2` at 4 warps for both production shapes, and it is
not the right point in either.** `BL=64, BC=4, 2 warps` beats it on seconds and
on work per joule at R3 and at R4, exact to the byte in every arm:

| shape | vs the incumbent tile | vs the front form |
|---|---|---|
| L14 R3, 4096x192 | **1.127x seconds, 1.059x work/J**, 86.9 -> 92.5 W | 2.94x, up from 2.61x |
| L14 R4, 4096x64 | **1.029x seconds, 1.160x work/J**, 57.8 -> 51.2 W | 1.76x, up from 1.71x |

The incumbent was not chosen badly; it was chosen earlier. `_tile_best` sizes
a program by lanes rather than by the `FAN`-wide output `_tile` sizes by,
which is right, but it caps `bl` at 128 and then tests `bl * bc < 256`, so
`bc` can never exceed 2 and every shape with `low >= 128` lands on the same
point. That rule was written against the column counts `_layout` admitted
**before** the class-width resident set let it admit `FAN` times as many. The
width lever the best form buys is exactly what moved the tile's optimum, and
nothing re-derived it.

## R3: more programs, not fatter ones (timing RETRACTED)

| arm | grid | programs | elem/thread | regs | spills | s/call | W | work/J vs inc | s vs inc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front | 16x16 | 256 | 2 | 40 | 0 | 0.11262 | 52.67 | 0.632 | 0.384 |
| `128x2w4` (incumbent) | 16x96 | 1536 | 2 | 40 | 0 | 0.04321 | 86.94 | 1.000 | 1.000 |
| **`64x4w2`** | 32x48 | 1536 | 4 | 40 | 0 | **0.03834** | **92.51** | 1.059 | **1.127** |
| `256x2w4` | 8x96 | 768 | 4 | 39 | 0 | 0.03907 | 90.11 | **1.065** | 1.106 |
| `128x4w8` | 16x48 | 768 | 2 | 40 | 0 | 0.03995 | 93.63 | 1.004 | 1.082 |
| `128x4w4` | 16x48 | 768 | 4 | 40 | 0 | 0.04119 | 89.91 | 1.012 | 1.049 |
| `256x4w8` | 8x48 | 384 | 4 | 40 | 0 | 0.04189 | 88.54 | 1.011 | 1.032 |
| `128x8w4` | 16x24 | 384 | 8 | 48 | 0 | 0.04788 | 80.20 | 0.977 | 0.903 |
| `64x8w4` | 32x24 | 768 | 4 | 33 | 0 | 0.06408 | 73.29 | 0.798 | 0.674 |

Both arms that cut the program count to 384, and the one that put 8 elements on
a thread, went backwards. At this shape the step wants blocks, and the two arms
that keep 1536 of them are the two fastest.

## R4: the same tile wins for the opposite reason (timing RETRACTED)

| arm | grid | programs | elem/thread | regs | s/call | W | work/J vs inc | s vs inc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| front | 16x16 | 256 | 1 | 40 | 0.03917 | 51.07 | 0.661 | 0.585 |
| `128x2w4` (incumbent) | 8x32 | 256 | 2 | 37 | 0.02290 | 57.75 | 1.000 | 1.000 |
| **`64x4w2`** | 16x16 | 256 | 4 | 40 | 0.02226 | 51.20 | **1.160** | 1.029 |
| `128x4w8` | 8x16 | 128 | 2 | 33 | **0.02209** | 57.45 | 1.042 | **1.037** |
| `256x2w4` | 4x32 | 128 | 4 | 39 | 0.02425 | 48.52 | 1.123 | 0.944 |
| `128x4w4` | 8x16 | 128 | 4 | 40 | 0.02554 | 49.93 | 1.037 | 0.897 |
| `256x4w8` | 4x16 | 64 | 4 | 40 | 0.02855 | 45.71 | 1.012 | 0.802 |
| `128x8w4` | 8x8 | 64 | 8 | 48 | 0.03390 | 43.94 | 0.885 | 0.676 |
| `64x8w4` | 16x8 | 128 | 4 | 33 | 0.05020 | 38.64 | 0.681 | 0.456 |

At R4 the winner launches the **same 256 programs** as the incumbent. Nothing
about parallelism changed; only the shape of a program did, from 128 classes by
2 columns on 128 threads to 64 by 4 on 64 threads. Half the threads, 2.9% less
time and 16% more work per joule. The incumbent is over-threaded for the work
it has at this rate, and that costs energy rather than seconds.

Two arms therefore disagree across the shapes, which is why one shape would not
have settled it. `256x2w4` is the best work per joule at R3 (1.065) and **loses
outright on seconds at R4** (0.944). `128x4w8` is marginally the fastest at R4
(1.037 against 1.029) and gains almost nothing on energy at either. `64x4w2` is
the only arm that improves both metrics at both shapes.

## What was held fixed, and what that leaves attributable

The outer chunk stays 512, the production chunk: `chunk` is the OUTER loop, and
moving it moves the epilogue's `min`, its `sse` accumulation and its traceback
call count as well as the plan's width. Each arm runs the width its resident
set earns, 192 at R3 and 64 at R4, so `_layout` is not part of the comparison.
The scan unroll stays derived from `(fan, bl, bc, warps)`, so every tile is
screened as the code would configure it rather than under another tile's
unroll -- it resolves to 32 for all three profiled arms, so it is not a hidden
second variable here.

Every arm returned the reference's states by `torch.equal` and the reference's
`sse` as the identical float, checked before any clock was read. No arm
spilled.

## Where the time goes

Profiles are separate actions from the timing, one config each so no trace
overwrites another's path, front and incumbent and winner in one trace
separated by `record_function` markers. `_step_best` is one kernel name for
both candidate arms, so the marker totals are what splits them.

R3, action `bccfd8d021d7`, trace `b6877904e43c`, 65973 events: `arm:front`
113213.6 us with `_step` at 106998.7 us over 24576 calls; `arm:t128x2w4`
42162.1 us and `arm:t64x4w2` 39423.7 us, with `_step_best` at 73717.9 us over
the 8190 calls the two share, 4095 each. `_traceback` is 6395.5 us over 3
calls, one per arm, and is identical work in all three.

R4, action `28e42f5c626b`, trace `14e506b6e3b5`, 33140 events: `arm:front`
40901.2 us with `_step` at 36092.5 us over 8192 calls; `arm:t128x2w4` 24612.7
us and `arm:t64x4w2` 23946.7 us over 8190 shared `_step_best` calls.

The profiled ratios agree with the timed ones: 2.685x and 2.872x against the
front at R3 where the timed arms said 2.606 and 2.937, and 1.662x and 1.708x at
R4 against a timed 1.711 and 1.760.

## Receipts

| what | action | where |
|---|---|---|
| tests, 19 passed, 0 skipped, 6 on the device | `21929eb2a0ef` | sparklina |
| R3 screen, 9 arms | `0a490a3da366` | gb10, exclusive, measurement |
| R4 screen, 9 arms | `0183610f7a1e` | gb10, exclusive, measurement |
| R3 profile | `bccfd8d021d7` | sparklina, `--profile torch` |
| R4 profile | `28e42f5c626b` | sparklina, `--profile torch` |

Timing is 3 ABCD blocks an arm, each sized so a 1 Hz sampler sees it, with a
two second lead in and out; energy is integrated over each block's own
interval and every block here was fully bracketed, `covered` 1.0 throughout.
Block to block spread is 1.2 to 2.3% at R3 and under 0.5% at R4. The two
leading R3 arms do not overlap on seconds -- `64x4w2` ran [0.03798, 0.03834,
0.03856] against `256x2w4`'s [0.03885, 0.03907, 0.03932] -- so that ordering is
separable. Their work per joule differs by 0.6% and is **not**.

## What this does not establish

Arms run in list order inside each block, so `64x4w2` is measured eighth of
nine and `256x2w4` sixth. At R3 that is a thermal headwind the winner took and
still won on seconds; it also means its work per joule is measured at a hotter
point than the arm it nearly ties. The ordering is not randomised, and three
blocks would not resolve a 0.6% energy difference if it were.

**R4 cannot be brought into the 90 to 100 W band by tiling.** Its best arm
draws 57.8 W and the winner 51.2 W, against 92.5 W at R3. With 64 columns and
`low = 1024` the whole problem is 256 programs however it is cut, so the power
that shape can reach is a property of the shape, not of the tile. Reading that
as a tile failure would be reading it backwards.

Two shapes, one rate family, one box, `viterbi_window` alone rather than a real
encode. `E4-R1088` includes R5, and the dense shapes use other `L`. Nothing
here changes a default: `TESSERA_WINDOW_BEST_TILE` is unset by default and
unset is byte-identical to what `_tile_best` already returns. Making `64x4w2`
the derived default is a separate change, and it should wait for R5 and a dense
shape, because the rule that produces it has to be right for more than two
points.
