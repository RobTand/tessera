# Which `build_forest` cell reached 73 GB, and which allocation it was

**Date** 2026-09-05 · **Box** `sparky` (aarch64, Ubuntu 24.04.4, Linux
6.17.0-1032-nvidia, 20 CPUs, 121 GiB RAM), **CPU only** — `CUDA_VISIBLE_DEVICES=""`,
system `/usr/bin/python3` 3.12.3, no torch in the process (`tessera.alphabet`
imports only `.errors` and `.grammar`). · **Tree** `master` `b444631`, `src/`
rsynced to `/home/rob/audit-sync/ts-285/src`. The `build_forest` path is
byte-identical at `e78959e`, the commit the original observation was taken
near: `git diff e78959e master -- src/tessera/alphabet.py` is two additive
hunks, `PayloadGrid.hardware_byte` and `require_hardware_byte_grid`, neither
on this path. · **Answers** tessera#285.

## Verdict

**The cell is `BF16` at rate 11: 68,866,388 KB max RSS (65.7 GiB / 70.5 GB) in
170 s.** It is the widest BF16 cell that fits a 121 GiB box; rate 12 projects
to ~130 GiB from the measured slope and cannot run, so a script walking the
rates upward completes rate 11 and dies inside rate 12 — which is the ~73 GB
kill tessera#285 recorded.

**The allocation is not a distance matrix and not the k-d builder.** It is
`_mass_balanced_blocks`'s two Python containers, both live at once:

| object | line | shape | rate-7 footprint | bytes/element |
|---|---|---|---|---|
| `pairs` | `alphabet.py:1057-1061` | `sorted` list of `anchors x grid.size` `(distance, target, code)` tuples | 2,241,416 KB (2.14 GiB), 16,777,216 tuples | 136.8 |
| `ranked` | `alphabet.py:1072-1076` | dict of `grid.size - anchors` lists of `anchors` `(distance, target)` tuples | 1,711,060 KB (1.63 GiB), 16,711,680 tuples | 104.8 |

`pairs` is still referenced when `ranked` is built (`alphabet.py:1064` consumes
it, nothing drops it), so the peak is their **sum**, ~254.8 bytes per
`(anchor, code)` pair. That is the whole cost: brackets around those two
statements reconstruct 99.8% of the cell's measured peak (below).

**`E2M1x2` — the only production cell on the audit's filter — is free.** Every
rate 1-7 peaks at **34.4 MB and 0.10 s**, because arity > 1 routes to
`_build_forest_kd`, which never builds a cross product. There is nothing to
bound or stream on the shipping path.

**Nothing production builds the BF16 cells, and nothing could use one if it
did.** `export._build_plan` returns `(rates, grid)` before any forest when the
body is `BodyKind.WINDOW`, and `export.wire_recipe` gives BF16 a window recipe
at **every** rung; `AnchorForest._refuse_unserialisable` then refuses the
planes of any forest over 256 codes, so a BF16 forest has no wire even once
built. The one reach was `encode_linear_planes(..., grid=BF16_GRID,
body=BodyKind.TCQ)`, an explicit override `export._resolve_recipe:1664` honours
on any grid. That reach is now refused by name before the first allocation
(`alphabet.require_forest_grid`, same commit as this file's sibling).

## Method

One cell at a time, largest last, nothing else of mine on the box, `free -g`
checked before each (114 GiB available at the start; the rate-11 cell was run
only once ≥ 105 GiB was free, and left 43 GiB). Address space capped so a
runaway could not take the box:

```
cd /home/rob/audit-sync/ts-285
( ulimit -v 85000000            # 81 GiB, the box cap
  CUDA_VISIBLE_DEVICES="" /usr/bin/time -v /usr/bin/python3 -c '
import sys; sys.path.insert(0, "src")
from tessera.alphabet import BF16_GRID, build_forest
build_forest(11, grid=BF16_GRID)' )
```

Max RSS is `/usr/bin/time -v`'s `Maximum resident set size (kbytes)`; wall time
is its `Elapsed (wall clock) time`. `build_s` below is the same run's own
`time.monotonic` bracket around `build_forest` alone, i.e. without interpreter
start-up and grid construction (~0.09 s, ~32 MB, which is the floor every row
carries).

## Per-cell table

`samples` is the default at every cell — `build_forest` fills it from
`GROUP_SCALED_SOURCE(peak)`, 16384 points, which is also what
`export._plan_for` passes when `source_sigma is None`.

| grid | R | anchors | width | max RSS (KB) | max RSS | wall | `build_s` | dominating allocation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E2M1x2 | 1 | 4 | 64 | 34,324 | 33.5 MiB | 0:00.08 | 0.002 | none — `_build_forest_kd` |
| E2M1x2 | 2 | 8 | 32 | 34,284 | 33.5 MiB | 0:00.10 | 0.002 | none — `_build_forest_kd` |
| E2M1x2 | 3 | 16 | 16 | 34,316 | 33.5 MiB | 0:00.09 | 0.002 | none — `_build_forest_kd` |
| E2M1x2 | 4 | 32 | 8 | 34,196 | 33.4 MiB | 0:00.10 | 0.002 | none — `_build_forest_kd` |
| E2M1x2 | 5 | 64 | 4 | 34,320 | 33.5 MiB | 0:00.09 | 0.002 | none — `_build_forest_kd` |
| E2M1x2 | 6 | 128 | 2 | 34,344 | 33.5 MiB | 0:00.08 | 0.002 | none — `_build_forest_kd` |
| E2M1x2 | 7 | 256 | 1 | 34,388 | 33.6 MiB | 0:00.18 | 0.095 | none — `depth == 0` early return |
| BF16 | 1 | 4 | 16384 | 111,168 | 108.6 MiB | 0:18.23 | 18.118 | `pairs` + `ranked` |
| BF16 | 2 | 8 | 8192 | 170,700 | 166.7 MiB | 0:09.51 | 9.413 | `pairs` + `ranked` |
| BF16 | 3 | 16 | 4096 | 294,060 | 287.2 MiB | 0:05.25 | 5.141 | `pairs` + `ranked` |
| BF16 | 4 | 32 | 2048 | 540,600 | 528.0 MiB | 0:03.47 | 3.344 | `pairs` + `ranked` |
| BF16 | 5 | 64 | 1024 | 1,034,264 | 0.99 GiB | 0:03.33 | 3.155 | `pairs` + `ranked` |
| BF16 | 6 | 128 | 512 | 2,017,912 | 1.92 GiB | 0:04.65 | 4.378 | `pairs` + `ranked` |
| BF16 | 7 | 256 | 256 | 3,991,888 | 3.81 GiB | 0:07.73 | 7.378 | `pairs` + `ranked` |
| BF16 | 8 | 512 | 128 | 8,434,660 | 8.04 GiB | 0:15.53 | 14.768 | `pairs` + `ranked` |
| BF16 | 9 | 1024 | 64 | 17,310,472 | 16.51 GiB | 0:33.95 | 32.619 | `pairs` + `ranked` |
| BF16 | 10 | 2048 | 32 | 34,856,552 | 33.24 GiB | 1:11.16 | 68.713 | `pairs` + `ranked` |
| **BF16** | **11** | **4096** | **16** | **68,866,388** | **65.68 GiB** | **2:53.76** | **170.093** | **`pairs` + `ranked`** |
| BF16 | 12 | 8192 | 8 | *~136,900,000* | *~130.6 GiB* | — | — | **not run** — exceeds the box |
| BF16 | 13 | 16384 | 4 | *~272,900,000* | *~260 GiB* | — | — | **not run** — exceeds the box |
| BF16 | 14 | 32768 | 2 | *~544,900,000* | *~520 GiB* | — | — | **not run** — exceeds the box |
| BF16 | 15 | 65536 | 1 | 44,424 | 43.4 MiB | 0:00.23 | 0.128 | none — `depth == 0` early return |

The BF16 rows below the cap are a straight line in `anchors`: each rate doubles
`anchors` and doubles the peak, at 16,606 KB per anchor (from the measured
R=10 → R=11 delta), i.e. **254.8 bytes per `(anchor, code)` pair** with
`grid.size` fixed at 65536. Rates 12-14 are that slope extrapolated and are
italicised because **they were not run**: each exceeds 121 GiB, and the brief's
floor was to leave ≥ 40 GiB free on a shared box. Rate 15 is off the line
because `completion_capacity(15, 15) == 0` returns before
`_mass_balanced_blocks` is reached at all.

## Naming the allocation

`resource`/`/proc/self/statm` brackets around each statement of
`_mass_balanced_blocks`, transcribed verbatim and run in the module's own
order on the module's own inputs (`targets` from `_lloyd_levels`, `values`
from the grid), so each delta is the live footprint of the named object at the
point the module holds it:

| BF16 R | `pairs` (KB) | `ranked` (KB) | bracket peak (KB) | cell max RSS (KB) | agreement |
|---:|---:|---:|---:|---:|---:|
| 6 | 1,123,216 | 855,728 | 2,011,804 | 2,017,912 | 99.70% |
| 7 | 2,241,416 | 1,711,060 | 3,985,312 | 3,991,888 | 99.84% |
| 8 | 4,484,008 | 3,911,280 | 8,428,164 | 8,434,660 | 99.92% |

So the two containers **are** the cell: nothing else in `build_forest` is
within two orders of magnitude of them. Dropping `pairs` before `ranked` is
built cuts the R=7 peak from 3,985,312 KB to 2,013,004 KB — half, measured,
and still quadratic.

Time follows memory: at R=7 the two statements are 2.23 s and 3.45 s of the
cell's 7.38 s (77%).

An independent `tracemalloc` run agrees on the ranking and names the same two
lines (`alphabet.py:1058` 240,834,576 B high-water, `alphabet.py:1057`
111,619,216 B, next line down `alphabet.py:550` at 7,074,856 B — a 34x gap),
but its totals are not quotable: the sampling thread is GIL-starved against a
CPU-bound builder and its own traces pushed the R=6 cell from 1.92 GiB to
12.6 GiB. The brackets above are the number.

## What was not measured

- **BF16 rates 12, 13, 14.** Projected only, from the measured slope. Each
  needs more RAM than the box has.
- **The other two `SERIALISABLE_GRIDS` entries.** `E2M1` (16 codes) and `E4M3`
  (256 codes) fall below the audit script's `arity > 1 or size > 256` filter
  and were not the question; their `pairs` is at most 256 x 256 elements.
- **`E4M3x2`.** `tuple_grid(E4M3_GRID, 2)` is 65536 codes and *is* admitted by
  `tuple_grid` (its limit refuses `size > 1 << 16`, and 65536 is not greater
  than 65536), but it is arity 2, so it would route to `_build_forest_kd` and
  not to the statements measured here. It is not in `SERIALISABLE_GRIDS` and
  was not run.
- **Anything about wire bytes, quality or a serve.** No artifact was written
  and no weight was encoded. This is an offline build cost only; nothing here
  prices, writes or serves a byte.
- **The original 73 GB process.** tessera#285 records the number without a rate,
  so the identification above is by reconstruction — the measured 65.7 GiB at
  R=11 plus a rate-12 that cannot fit — not by re-running the auditor's script.
