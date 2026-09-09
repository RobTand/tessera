# The best-form step's tile was never measured.  Measured, it loses

## Verdict

**`_tile_best` returns `128x2` at 4 warps for both production shapes, and it is
not the right point in either.** `BL=64, BC=4, 2 warps` beats it on seconds and
on work per joule at R3 and at R4, exact to the byte in every arm:

| shape | vs the incumbent tile | vs the front form | W |
|---|---|---|---|
| L14 R3, 4096x192 | **1.122x seconds, 1.062x work/J** | 2.93x, up from 2.61x | 88.7 -> 93.5 |
| L14 R4, 4096x64 | **1.029x seconds, 1.105x work/J** | 1.76x, up from 1.71x | 58.3 -> 54.3 |

The incumbent was not chosen badly; it was chosen earlier. `_tile_best` sizes
a program by lanes rather than by the `FAN`-wide output `_tile` sizes by,
which is right, but it caps `bl` at 128 and then tests `bl * bc < 256`, so
`bc` can never exceed 2 and every shape with `low >= 128` lands on the same
point. That rule was written against the column counts `_layout` admitted
**before** the class-width resident set let it admit `FAN` times as many. The
width lever the best form buys is exactly what moved the tile's optimum, and
nothing re-derived it.

## The measurement this replaces, and why it was wrong

A first version of this screen ran **nine** arms against a plan cache that
holds `_WINDOW_PLAN_CACHE = 8`. Nine distinct plan keys cycled in a fixed order
is the pathological case for an LRU: the entry evicted is always the one about
to be asked for, so every arm rebuilt its plan and re-captured its graph at the
head of every block, inside the clock and inside the energy bracket. Root read
that off the source. The run's own data agreed without a re-run -- each arm's
block median exceeded the hot single call recorded for the same arm by 0.08 to
0.23 s a block at R3, by different amounts per arm, so it did not cancel in a
ratio; and because inner repeats scale as the reciprocal of call time, the
faster arms amortised the capture better and the bias ran **toward** the
conclusion. Every timed number from that run is withdrawn.

*One clear and one warm call per arm establishes that an arm was warm once, not
that it was warm when it was measured.* The screen now refuses at startup to
run more arms than the cache holds -- refusing rather than detecting, because a
counter can only say a block was spoiled after the box has spent it -- and
counts `_WindowPlan` constructions either side of every timed block, refusing
to report any block that built one. Both runs below report **zero** plans built
inside any timed block, and the gap between an arm's block median and its own
hot single call has changed sign. In the withdrawn run every arm ran slow
against its own warm call, by 1.6 to 4.6% at R3 and 0.4 to 1.5% at R4. Here it
is at most 1.2% at R3 and it runs the other way at R4, where all four medians
sit 0.4 to 0.9% below the single call. That is what the first run could not
show.

The plan cache itself was not touched. A plan's traceback is
`nmax * steps * low` bytes and the bound of eight is there for that; raising it
would have been a memory-budget change wearing a screen's clothes.

## R3: 4096x192, front and incumbent and the two frontier candidates

| arm | grid | programs | elem/thread | regs | spills | s/call | W | work/J vs inc | s vs inc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front | 16x16 | 256 | 2 | 40 | 0 | 0.10975 | 58.26 | 0.583 | 0.384 |
| `128x2w4` (incumbent) | 16x96 | 1536 | 2 | 40 | 0 | 0.04210 | 88.72 | 1.000 | 1.000 |
| **`64x4w2`** | 32x48 | 1536 | 4 | 40 | 0 | **0.03751** | **93.49** | **1.062** | **1.122** |
| `256x2w4` | 8x96 | 768 | 4 | 39 | 0 | 0.03800 | 93.29 | 1.050 | 1.108 |

Five blocks an arm. The two candidates do not overlap on seconds -- `64x4w2`
ran [0.03737, 0.03739, 0.03751, 0.03769, 0.03778] against `256x2w4`'s
[0.03793, 0.03797, 0.03800, 0.03820, 0.03833] -- so that ordering is
separable. Their work per joule differs by 1.2% and their power by 0.2 W,
which is closer than five blocks resolve; the seconds are what separates them.

Both candidates land at 93.3 to 93.5 W, inside the 90 to 100 W band, against
the incumbent's 88.7 W and the front form's 58.3 W.

## R4: 4096x64, where speed and energy point at different arms

| arm | grid | programs | elem/thread | regs | s/call | W | work/J vs inc | s vs inc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| front | 16x16 | 256 | 1 | 40 | 0.03849 | 54.64 | 0.625 | 0.586 |
| `128x2w4` (incumbent) | 8x32 | 256 | 2 | 37 | 0.02254 | 58.29 | 1.000 | 1.000 |
| **`64x4w2`** | 16x16 | 256 | 4 | 40 | 0.02190 | **54.30** | **1.105** | 1.029 |
| `128x4w8` | 8x16 | 128 | 2 | 33 | **0.02156** | 59.45 | 1.025 | **1.046** |

At R4 the winner launches the **same 256 programs** as the incumbent. Nothing
about parallelism changed; only the shape of a program did, from 128 classes by
2 columns on 128 threads to 64 by 4 on 64 threads. Half the threads, 2.9% less
time, 6.8% less power and 10.5% more work per joule: the incumbent is
over-threaded for the work this rate has, and that costs energy rather than
seconds.

`128x4w8` is the one arm that is genuinely faster here, by 1.6%, and it pays
9.5% more power for it, so it is 7.8% behind on work per joule. Both orderings
are separable -- the blocks do not overlap on either axis. On GB10, where the
envelope and not the wall clock is what a long encode runs out of, work per
joule is the axis that decides, and `64x4w2` takes it.

## What was held fixed, and what that leaves attributable

The outer chunk stays 512, the production chunk: `chunk` is the OUTER loop, and
moving it moves the epilogue's `min`, its `sse` accumulation and its traceback
call count as well as the plan's width. Each arm runs the width its resident
set earns, 192 at R3 and 64 at R4, so `_layout` is not part of the comparison.
The scan unroll stays derived from `(fan, bl, bc, warps)`, so every tile is
screened as the code would configure it rather than under another tile's
unroll -- it resolves to 32 for every arm here, so it is not a hidden second
variable.

Every arm returned the reference's states by `torch.equal` and the reference's
`sse` as the identical float, checked before any clock was read. No arm
spilled.

## Where the time goes

Profiles are separate actions from the timing, one config each so no trace
overwrites another's path, front and incumbent and winner in one trace
separated by `record_function` markers. Three arms, inside the cache.
`_step_best` is one kernel name for both candidate arms, so the marker totals
are what splits them.

R3, action `bccfd8d021d7`, trace `b6877904e43c`, 65973 events: `arm:front`
113213.6 us with `_step` at 106998.7 us over 24576 calls; `arm:t128x2w4`
42162.1 us and `arm:t64x4w2` 39423.7 us, with `_step_best` at 73717.9 us over
the 8190 calls the two share, 4095 each. `_traceback` is 6395.5 us over 3
calls, one per arm, identical work in all three.

R4, action `28e42f5c626b`, trace `14e506b6e3b5`, 33140 events: `arm:front`
40901.2 us with `_step` at 36092.5 us over 8192 calls; `arm:t128x2w4` 24612.7
us and `arm:t64x4w2` 23946.7 us over 8190 shared `_step_best` calls.

Profiled and timed agree: 2.685x and 2.872x against the front at R3 where the
timed arms say 2.607 and 2.926, and 1.662x and 1.708x at R4 against a timed
1.708 and 1.758.

## Receipts

| what | action | where |
|---|---|---|
| tests, 19 passed, 0 skipped, 6 on the device | `21929eb2a0ef` | sparklina |
| R3 screen, 4 arms, 5 blocks | `530795d20399` | gb10, exclusive, measurement |
| R4 screen, 4 arms, 5 blocks | `6551e14e654a` | gb10, exclusive, measurement |
| R3 profile | `bccfd8d021d7` | sparklina, `--profile torch` |
| R4 profile | `28e42f5c626b` | sparklina, `--profile torch` |
| the withdrawn nine-arm run | `0a490a3da366`, `0183610f7a1e` | kept as negative evidence |

Energy is integrated over each block's own interval and every block here was
fully bracketed, `covered` 1.0 throughout, with a two second lead in and out.
Block to block spread within an arm is 0.06 to 0.16% at R4. At R3 it is 0.11%
on the front and 1.0 to 2.2% on the three tiled arms, whose blocks run a third
as long; the per call spans of those three do not overlap each other, so the
seconds ordering at R3 does not depend on which block is read.

## What this does not establish

Arms run in list order inside each block and the order is not randomised, so
each arm is measured at a slightly different thermal point. At R3 the winner
runs last of four and still wins on seconds, which is a headwind rather than a
tailwind; at R4 the blocks are tight enough that it does not signify. The 1.2%
work-per-joule gap between the two R3 candidates is inside that, and is not
claimed.

**R4 cannot be brought into the 90 to 100 W band by tiling.** Its highest arm
draws 59.5 W. With 64 columns and `low = 1024` the whole problem is 256
programs however it is cut, so the power that shape can reach is a property of
the shape, not of the tile. Reading that as a tile failure would be reading it
backwards.

Two shapes, one rate family, one box, `viterbi_window` alone rather than a real
encode. `E4-R1088` includes R5, and the dense shapes use other `L`. Nothing
here changes a default: `TESSERA_WINDOW_BEST_TILE` is unset by default and
unset is byte-identical to what `_tile_best` already returns. Making `64x4w2`
the derived default is a separate change, and it should wait for R5 and a dense
shape, because the rule that produces it has to be right for more than two
points.
