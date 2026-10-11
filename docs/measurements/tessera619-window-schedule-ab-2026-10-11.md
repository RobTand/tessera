# Tessera#619: the window Viterbi's two schedules on one board

**Date:** 2026-10-11 · **Issue:** tessera#619 · **Runner:** PrismaBuild only
**Probe:** `experiments/ig619_viterbi_power.py`. Both runs executed it at commit
`affe0c1dd`. Two later commits add `--table` and the import-root fix. They change
no measurement code.
**Actions:** sparky `0ab6c0cdbd6921b09239cdd6ce661f36b03426cc69fcdc2e56f2ce295f32b26d`,
sparklina `ff64432c66248b151b4973491f379878e1a14dab48ba533b148115ba3e185efe`.
Each ran about 13.5 minutes on one GB10 and exited 0. Each is one action per
box, pinned by host tag, so each box holds its own matched pair.
**Receipts:** `tessera619-window-schedule-ab-2026-10-11-sparky.json` (SHA-256
`db26bfdc37897dbb79779b7feb2da835217143515bbb706be257a270e764e4ea`) and
`...-sparklina.json` (`f00fb5166059cffe041b254d98b0277c1fedbe0dd52a7cb3d3b3a5db279b59a0`)
in this directory. They are byte copies of the files under
`/mnt/shared/tessera-measurements/issue-619-ab-20261011/`. The tables below are
read from the receipts with the probe's `--table` option.

## Why this receipt exists

The 2026-10-10 receipt (`tessera619-window-width-arms-2026-10-10.md`) ran the
best-form schedule on sparky and the front schedule on sparklina. A box
difference could sit inside its 2.8x. It did not time two units in flight. Its
probe recorded `byte_identical` and exited 0 whatever the value.

This receipt runs both schedules in one process on one board, in ABBA order, on
each box. The probe refuses unequal bytes and exits 3. It also times two
controls, and it profiles one unit under each schedule.

## Identity first

Every arm wrote the same bytes.

- Each rung ran 12 units through 6 arms (BF16) or 8 arms (E4M3). All arms of a
  rung agree on all 12 unit digests, and the 12 digests differ from each other.
- Sparky and sparklina agree on all 24 digests.
- Units `u0` to `u3` carry the digests recorded in the three 2026-10-10 actions
  (`159f3d9d`, `20c2ad4c`, `f5ab2f20`), which also agree with each other.

So every ratio below compares spellings of one answer. No file under `src/`
changes. The wire, the recipe table and the byte audit stay as they were.

## Best-form against front-form, same board

Each schedule pools its two ABBA arms. Units/s is total units over total
seconds. J/unit is total joules over total units, with joules as mean watts
times seconds per arm. Units are synthetic 2048x4096 bf16 weights,
weights-only fresh encode, `verify=False`.

| box | rung | schedule | units/s | mean W | of 140 W | J/unit |
|---|---|---|---:|---:|---:|---:|
| sparky | BF16 q256 1024 | front | 0.2105 | 76.19 | 0.544 | 361.9 |
| sparky | BF16 q256 1024 | best | 0.5807 | 84.96 | 0.607 | 146.3 |
| sparky | E4M3 q256 896 | front | 0.2114 | 73.19 | 0.523 | 346.3 |
| sparky | E4M3 q256 896 | best | 0.5846 | 82.14 | 0.587 | 140.5 |
| sparklina | BF16 q256 1024 | front | 0.2104 | 74.56 | 0.533 | 354.4 |
| sparklina | BF16 q256 1024 | best | 0.5780 | 83.80 | 0.599 | 145.0 |
| sparklina | E4M3 q256 896 | front | 0.2144 | 73.09 | 0.522 | 340.9 |
| sparklina | E4M3 q256 896 | best | 0.5801 | 80.28 | 0.573 | 138.4 |

| box | rung | best over front, units/s | J/unit, best over front |
|---|---|---:|---:|
| sparky | BF16 | 2.758x | 0.404 |
| sparky | E4M3 | 2.766x | 0.406 |
| sparklina | BF16 | 2.748x | 0.409 |
| sparklina | E4M3 | 2.706x | 0.406 |

- Best-form is 2.71x to 2.77x faster and spends 0.40x to 0.41x the joules per
  unit. That is 2.44x to 2.48x the units per joule. All four boxes-and-rungs
  agree within 2.3%.
- A unit takes 4.656 to 4.774 s at the front schedule and 1.705 to 1.734 s at
  best. The issue measured 4.4 to 4.6 s per unit on real GLM stub weights.
- Drift is small. The second arm of a schedule runs at 0.988 to 0.996 of the
  first arm's units/s, in every box and rung.
- The 2026-10-10 figure, 2.82x at 0.44x joules, mixed two boxes. The same-board
  figure for BF16 is 2.75x to 2.76x at 0.40x to 0.41x. The earlier headline
  stands. It read a little high on speed and a little pessimistic on joules.

### Power, clock and the 140 W envelope

Power is the in-process sampler (500 ms) over each arm's own window.

- The front schedule draws 73 to 76 W (52% to 54% of 140 W). Best-form draws 80
  to 85 W (57% to 61%).
- The SM clock falls as power rises. Single-thread front arms hold 2430 to
  2486 MHz. Best arms hold 2067 to 2251 MHz. Idle is 208 MHz at 5.5 to 6 W.
- PrismaBuild's own box window recorded a peak of 92.4 W (sparky) and 91.9 W
  (sparklina). It recorded a peak board temperature of 87 and 86 C.
  PrismaBuild's declared reference text says the fleet's recorded maxima from
  2026-08-28 to 2026-09-21 were 114 W (sparky) and 106 W (sparklina).
- So 61% of 140 W is not 39% of free headroom. The clock falls while the board
  stays far below 140 W. That is consistent with a power or thermal limit. This
  receipt does not say which one. Read the fractions above as readings, not as
  spare capacity.

Netdata, the second instrument, agrees. Over 28 arm windows the median
difference from the in-process sampler is 1.2 W, and 25 windows differ by 5 W or
less. Three windows differ by more: sparklina E4M3 `best/2t` by 15.0 W,
sparklina BF16 `front#2` by 14.7 W and sparklina E4M3 `best#2` by 8.1 W. Each
holds four whole Netdata groups, 5 to 12 s long. This receipt does not explain
those three. Over each 15-group ABBA block the two instruments differ by 0.31,
0.87, 1.15 and 3.70 W. The single-thread ABBA arms, one row each:

| box | rung | arm | in-process W | of 140 W | SM MHz | Netdata W | groups |
|---|---|---|---:|---:|---:|---:|---:|
| sparky | BF16 | front#1 | 74.44 | 0.532 | 2486 | 74.70 | 4 |
| sparky | BF16 | best#1 | 85.99 | 0.614 | 2251 | 88.03 | 4 |
| sparky | BF16 | best#2 | 83.95 | 0.600 | 2213 | 83.95 | 4 |
| sparky | BF16 | front#2 | 77.93 | 0.557 | 2472 | 78.42 | 4 |
| sparky | E4M3 | front#1 | 71.75 | 0.512 | 2453 | 72.33 | 4 |
| sparky | E4M3 | best#1 | 82.65 | 0.590 | 2120 | 83.00 | 4 |
| sparky | E4M3 | best#2 | 81.64 | 0.583 | 2101 | 79.62 | 4 |
| sparky | E4M3 | front#2 | 74.63 | 0.533 | 2444 | 75.90 | 4 |
| sparklina | BF16 | front#1 | 73.15 | 0.522 | 2468 | 75.19 | 4 |
| sparklina | BF16 | best#1 | 83.89 | 0.599 | 2224 | 80.05 | 4 |
| sparklina | BF16 | best#2 | 83.72 | 0.598 | 2204 | 86.22 | 4 |
| sparklina | BF16 | front#2 | 75.95 | 0.542 | 2454 | 61.30 | 4 |
| sparklina | E4M3 | front#1 | 72.31 | 0.516 | 2441 | 72.78 | 5 |
| sparklina | E4M3 | best#1 | 80.95 | 0.578 | 2105 | 79.35 | 4 |
| sparklina | E4M3 | best#2 | 79.61 | 0.569 | 2067 | 87.75 | 4 |
| sparklina | E4M3 | front#2 | 73.86 | 0.528 | 2430 | 74.22 | 5 |

## Which kernel holds the time

One unit encoded under `torch.profiler` per schedule. `device/wall` is the
summed kernel time over the encode's wall time. The profiler adds its own
overhead to both.

| box | rung | schedule | wall s | device s | device/wall | top kernel | launches | us each |
|---|---|---|---:|---:|---:|---|---:|---:|
| sparky | BF16 | front | 4.911 | 4.796 | 0.977 | `_step` | 1048576 | 4.39 |
| sparky | BF16 | best | 1.715 | 1.738 | 1.013 | `_step_best` | 65504 | 23.81 |
| sparky | E4M3 | front | 4.897 | 5.355 | 1.094 | `_step` | 1048576 | 4.92 |
| sparky | E4M3 | best | 1.692 | 2.160 | 1.276 | `_step_best` | 98256 | 20.00 |
| sparklina | BF16 | front | 4.926 | 4.811 | 0.977 | `_step` | 1048576 | 4.41 |
| sparklina | BF16 | best | 1.723 | 1.747 | 1.014 | `_step_best` | 65504 | 23.93 |
| sparklina | E4M3 | front | 4.900 | 5.341 | 1.090 | `_step` | 1048576 | 4.91 |
| sparklina | E4M3 | best | 1.698 | 2.161 | 1.273 | `_step_best` | 98256 | 19.99 |

Plans, same on both boxes. Every plan has 2048 steps. A unit runs four trellis
passes (`scale_refit=4`, `encode.py:3663`), so launches are four times batches
times steps.

| rung | schedule | rate | width | batches | grid |
|---|---|---:|---:|---:|---|
| BF16 | front | 4 | 32 | 128 | [16, 16] |
| BF16 | best | 4 | 512 | 8 | [16, 256] |
| E4M3 | front | 3 | 32 | 64 | [16, 16] |
| E4M3 | front | 4 | 32 | 64 | [16, 16] |
| E4M3 | best | 3 | 256 | 8 | [16, 128] |
| E4M3 | best | 4 | 512 | 4 | [16, 256] |

Reading:

- **The device is busy.** At BF16 and the front schedule the summed kernel time
  is 97.7% of the unit's wall time. The device is idle for at most 2.3% of the
  unit in total. Any host-only phase, such as source reads or packing, fits
  inside that figure. The E4M3 rows sum to 109% because their two rate chains
  overlap, so they bound nothing. The BF16 rows carry the bound.
- **One kernel holds the time.** `_step` is 96% of device time at the front
  schedule, `_step_best` 90% at best-form. `_traceback` is the next kernel: 37 ms
  at BF16 and 41 to 42 ms at E4M3, under 1% of the unit.
- **The chain is serial along the trellis.** Step `s+1` reads step `s`. One unit
  is 1,048,576 dependent launches of about 4.4 us at the front schedule. Best-form
  makes 65,504 launches of 23.8 us. The launch count falls 16x, each launch takes
  5.4x as long, and the product falls 2.95x (4.60 s to 1.56 s).
- **Width comes from the L2 budget, not from the caller.** `_layout` sizes the two
  resident buffers against a sixth of L2 (`window_viterbi.py:826`). A front buffer
  holds 16384 floats a column, so 32 columns fit. A best-form buffer holds one
  float per class, 1024 at R4, so 512 columns fit. Joining units along the column
  axis adds batches and does not add width. The 2026-10-10 batch-of-four arm
  already showed that (0.2110 against 0.2116 units/s).

## Two units in flight

Two threads, each on its own CUDA stream with its own plans, the 12 units split
six and six. Rate streams are `TESSERA_WINDOW_RATE_STREAMS`: at a mixed rate
(E4M3 q256 896 is R3 and R4) the two rate chains already run on two streams.
`rs0` turns that off.

| box | rung | schedule | one thread, units/s | two threads, units/s | ratio | J/unit ratio |
|---|---|---|---:|---:|---:|---:|
| sparky | BF16 | front | 0.2105 | 0.1566 | 0.744 | 1.262 |
| sparky | BF16 | best | 0.5807 | 0.6248 | 1.076 | 0.870 |
| sparky | E4M3 | front | 0.2114 | 0.1357 | 0.642 | 1.361 |
| sparky | E4M3 | best | 0.5846 | 0.5692 | 0.974 | 0.967 |
| sparklina | BF16 | front | 0.2104 | 0.2164 | 1.029 | 0.968 |
| sparklina | BF16 | best | 0.5780 | 0.6087 | 1.053 | 0.879 |
| sparklina | E4M3 | front | 0.2144 | 0.1281 | 0.598 | 1.386 |
| sparklina | E4M3 | best | 0.5801 | 0.5551 | 0.957 | 0.975 |

| rung | schedule | rate streams on over off, sparky | sparklina |
|---|---|---:|---:|
| E4M3 | front | 0.999 | 1.015 |
| E4M3 | best | 1.057 | 1.059 |

- Two units in flight buy nothing at the front schedule. No reading is above
  1.03x. Three of four are 0.60x to 0.74x. At best-form the readings are 0.96x to
  1.08x.
- Rate streams overlap the two rate chains of one unit. They buy 0% to 1.5% at
  the front schedule and 5.7% to 5.9% at best-form. The same lever gave 18% to
  22% at the G2 LDLQ shapes (`tessera668-window-rate-streams-2026-09-27.md`).
  Those shapes call the Viterbi in many short calls. A weights-only unit calls it
  four times.
- The short profile run of two units, one per thread, explains the shape. At the
  front schedule the summed kernel time is 1.97x the wall time, so the two chains
  do overlap in time. But every `_step` takes 8.88 us (sparky) or 9.25 us
  (sparklina). Alone it takes 4.39 and 4.41 us. So the two units take 9.63 s and
  10.00 s. One after the other they would take 9.82 s and 9.85 s. At best-form the
  kernels stretch 1.22x to 1.24x. The pair takes 3.16 s and 3.21 s, against 3.43 s
  and 3.45 s in series. That is a 7% to 9% gain.
- A second chain therefore does not find spare device at the front schedule.
  That points at a resource the two chains share. This receipt does not say
  which: the L2, the issue slots or the launch path. The 2026-10-06 Nsight packet in
  `tessera-window-best-form-2026-09-09.md` reads the same front kernel (grid
  16x16, block 128). It shows 5.6% issue-active and 17.2% of L2 throughput, with
  long-scoreboard stalls first. It ran at rung 1088 with the cache flushed and
  base clocks, so it does not certify production residency.
- The long two-thread arms run slower than the short profile pair at the front
  schedule (0.60x to 0.74x against 1.0x on three of four). This receipt does not
  find the cause. The host threads share the GIL and the memory system, and
  neither was isolated.

## The diagnostic SSE sync

`encode.py:3808` reads `float((error ** 2).sum())` once per unit, after the last
refit. `_run_joined` passes `want_sse=False` (`encode.py:3912` to `3915`), so the
Viterbi calls queue behind each other. The SSE read is one of the points where
the host waits for work already queued. The profile bounds what that wait can
hide. At BF16 and the front schedule the device is busy 97.7% of the unit's wall
time. At best-form the sum is 101%, which I read as profiler attribution noise around
100%. So the sync opens no gap. Removing it could save at most the 2.3% the
device is idle in total.

## What this changes in the 2026-10-10 reading

- The 2026-10-10 reading said occupancy per launch is small (256 CTAs) and the
  chain cannot overlap with itself. Both statements hold. But a second chain does
  overlap in time, and the overlap does not turn into throughput at the front
  schedule. Small occupancy is not spare capacity.
- "A wider front-form tile loses" stands. This receipt does not repeat that arm.
- The box confound is closed. The ratio holds on one board, on each of two
  boards.

## What a default change would need

No default moves in this change. `AGENTS.md` (rule 11) and
`window_viterbi.py:728` to `730` reserve the default to Rob. The numbers above
are the evidence for that decision.

- The G2 manifest already seals `TESSERA_WINDOW_BEST_FORM=1` and
  `TESSERA_WINDOW_BEST_TILE=64,4,2` (`experiments/window_viterbi_best_form_ab.py:101`
  to `116`). The slow schedule reaches only a run that does not seal them, such
  as the issue's unconfigured export.
- The breadth is not measured. This receipt covers arity 1, `L=14`, R3 and R4.
  The 2026-09-19 receipt covers R7 at kernel level. Arity above 1, dense BF16 and
  E2M1, and other `L` are open in `tessera-window-best-form-2026-09-09.md`
  ("What is still open").
- The 54 test modules that reach the trellis through the encoder run the front
  schedule today (same doc). A flip changes what they cover. They need a GPU
  re-run.
- `export_routed_part.sh:311` records `window_best_form` as the raw environment
  value, `None` when unset, and `export_serving.py:3293` to `3296` prints the knob
  only when `--encode-batch` is above 1. After a flip, `None` would name the new
  default. Both would need to record the schedule that actually ran.
- Any new source digest is next-campaign-only (tessera#652).

A follow-on issue carries the decision, with the options and a recommendation.

## What this does not establish

- Real GLM weights. The units are N(0, 0.02^2) in bf16. They fix the shapes,
  rates and kernel sequence and not the values. The issue's real-weight units
  took 4.4 to 4.6 s at the front schedule against 4.66 to 4.77 s here. That
  compares two different setups, so values may move the time by a few percent.
  The bytes' identity was checked on these values only.
- Served quality. The bytes are identical, so there is nothing to price. No KL ran.
- One action per box, 12 units per arm, two repeats of each ABBA arm. No run on
  another day.
- The default best-form tile (`_tile_best`), not the G2 tile `64,4,2`. The sealed
  G2 configuration may run faster. tessera#652 owns that screen.
- Anything below the kernel: no Nsight counters ran here, so the shared resource
  of two chains in flight is not named.
- Any shape but 2048x4096 at BF16 q256 1024 and E4M3 q256 896.

## Reproduce

```
pbrun --cwd . --demand gpu=1,mem_gb=32 --tag sparky --cpus 4 \
  --priority 0 --max-attempts 1 --timeout-s 3000 -- \
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
  experiments/ig619_viterbi_power.py --publish-dir DIR
python experiments/ig619_viterbi_power.py --table DIR/receipt-sparky-*.json
```

The probe exits 0 when every arm matches and no phase failed. It exits 3 when
an arm writes different bytes, 1 when a phase fails, and 2 without a CUDA
device. The probe puts its own `src` first. A stale `tessera` in the
site-packages cannot stand in for the snapshot. It did so for the first three
arms of the 2026-10-10 attempt.
