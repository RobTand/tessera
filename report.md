# Issue #94 — the window trellis under LDLQ

Branch `muse/ts-94-window`, from master `82cdf51`.

**The defect is there.** `viterbi_window_fused` decided whether to capture from
`len(descs) >= 6`, and under LDLQ every call carries exactly one batch, so the
shipping E4M3 recipe ran the eager step loop on a freshly allocated,
freshly tiled, freshly warmed plan on every one of its several hundred calls.
The eager/captured split was a read on *whether the caller was compensating*,
which is not a property the launch stream has any business seeing.

*(Numbers below are filled in from the runs; see "Measurement".)*

## Why one batch, always

`_L2_BUDGET` is L2/6 = 4 MiB. A batch is `2 * size * cols * 4` bytes of front,
so at the E4M3 default (`window_bits = 14`, `size = 16384`)

    width = 4194304 // (2 * 16384 * 4) = 32 columns

LDLQ hands the trellis `ldl_block` columns at a time. At the served rung the
Bresenham arrangement splits each block again by rate, so the calls are
narrower still — measured on `model.layers.0.mlp.down_proj` [1024, 3072] at
`q256=1042` (root 521/128), `ldl_block=32`:

| call shape | occurrences per pass |
|---|---|
| rate 4, 30 columns | 72 blocks |
| rate 4, 29 columns | 24 blocks |
| rate 5, 2 columns  | 72 blocks |
| rate 5, 3 columns  | 24 blocks |

Four distinct shapes for the whole unit, each recurring dozens of times per
pass and again in each of the `scale_refit` passes — every one of them under
`width`, so `len(descs) == 1 < 6` on all of them, and none of them captured.
The same tensor encoded in one call (no LDLQ) has 96 batches and does capture.
Receipt: `shape_census.json`.

## The change

`src/tessera/window_viterbi.py`, modelled on `2903784`:

* the per-call allocate/tile/warm moves into a `_WindowPlan` that owns its
  fronts, its descriptor table and its traceback, so a repeated shape builds
  them once;
* plans are keyed by shape and kept **per thread** (PrismaBuild encodes units
  concurrently in one process, and a plan owns mutable buffers);
* a shape captures on its **second** sight, so a shape seen once never pays a
  capture it cannot replay;
* a call at or above `_GRAPH_MIN_BATCHES` keeps today's behaviour — per-call
  buffers, captured and dropped inside the call — because persisting one would
  pin the whole tensor's traceback (545 MiB at the E4M3 default over 3072
  columns) to buy nothing but the one capture that call already amortises
  over its own 96 batches;
* `TESSERA_WINDOW_GRAPH` is resolved **per call** instead of being bound to a
  module constant at import, which is what makes it usable as an A/B control
  in one process — and is why the pre-change tree had to be swept by rebinding
  the constant from the harness.

One knob semantic moves with this: `TESSERA_WINDOW_GRAPH=1` used to capture
per call and drop the graph with the call; it now persists a plan on the first
sight of a narrow shape. `0` and unset are unchanged in what they compute, and
all three return the reference's bytes.

The epilogue (transpose, `cost.min`, `sse += float(final.sum())`, traceback)
stays on the host stream and in its original order, so the returned `sse`
float is bit-identical and not merely close.

## Bit-exactness

`tests/test_window_graph.py`, 19 tests, all against `impl="reference"` — the
torch chain that defines the trellis, not another run of the same kernels.

## Tests

Full suite on the branch, sparky, `PYTHONPATH=src`, `-p no:randomly`:

    1638 passed, 9 skipped, 14 warnings in 2097.30s

No failures and no errors, so the per-failure comparison against a pristine
master checkout has nothing to compare — every test that master runs, this
branch runs and passes. The suite was launched under `nohup` without capturing
`$?`, so the exit status is reported as what the summary line implies rather
than as an observed number; a pass/skip-only summary is pytest's zero.

`tests/test_window_graph.py` (19) and `tests/test_window_viterbi_fast.py` (52)
were additionally run on their own against the same tree.

## Measurement

**Trees, pinned.** The decisive arms are `--levers 0,auto,1` in ONE process on
ONE checkout — branch `muse/ts-94-window`, and `0` there is the pre-change
machine (per-call plan, eager step loop) reproduced by the branch's own code
rather than by a second tree. That is deliberate: a lever sweep whose arms
differ only in the lever cannot be split across builds without reintroducing
the confound the interleaving removes. The separate master leg is a *control*
on the pristine snapshot of `82cdf51` (`ts-94-before`), and it answers one
question only — that the lever does nothing on the tree that has no plan cache.
Nothing was rebased between legs.

### Where it was taken

sparklina (`gx10-6b77`), through the PrismaBuild pool with `--exclusive`, on a
quiet box. Netdata over the profile window (01:14:01-01:16:30Z) says so rather
than my asserting it: **`system.load` 8.10-8.86**, **`mem.swapio` in max 13.8
KiB/s and out 0**, **`nvidia_smi.gpu_power_draw` avg 32.7 W (min 29, max 45) =
0.234 of the ~140 W envelope**. The harness's own `/proc/loadavg` per rep
agrees: 8.09-8.65 across all 21 timings in leg C.

This is the second pass. The first was taken while the box was swapping --
Netdata caught `mem.swapio` in peaking at **108,076 KiB/s**, and one spike
landed on the eager profile arm while the auto arm ran at 28 KiB/s, which is to
say the contamination pointed in the direction that flattered the change. Its
numbers (1.447x timing, 1.361x profile wall) are discarded, not averaged in.
The in-process profiler could not see any of that, which is the whole reason
principle 15 asks for both instruments.

### The timing pair (leg C)

Branch `ce811e8`, ONE process, ONE tree, 7 reps interleaved lever by lever.
`model.layers.0.mlp.down_proj` [1024, 3072], E4M3, `q256=1042`, `ldl_block=32`,
`sigma_reg=1.0`, `scale_refit` default.

| lever | B (LDLQ+refit) per rep, s | median |
|---|---|---|
| `0` eager -- the pre-change machine | 31.30 29.00 30.32 27.13 26.30 23.03 29.02 | **29.00** |
| unset -- the fix | 21.85 20.80 22.02 18.67 18.55 21.22 20.69 | **20.80** |
| `1` force capture | 25.11 23.73 21.91 19.02 20.07 21.93 21.65 | 21.91 |

**1.394x on the LDLQ arm.** Per-rep ratios 1.432 1.394 1.377 1.453 1.418
**1.085** 1.402 -- six of seven inside 1.38-1.45 and one outlier, which is why
the headline is a median and the outlier stays in the table.

Two controls, both of which had to hold for the number to mean anything:

* **A arm (weights-only, full width): eager/auto = 0.971.** Unchanged, as it
  must be -- that call runs 96 batches and captured on master already. A change
  here would have meant the lever was reaching something it should not.
* **`auto` vs `1`: 0.950.** Both replay cached plans in steady state, so they
  are the same machine and have to agree; they do, within a scatter that
  overlaps heavily (auto 18.55-22.02, graph 19.02-25.11).

### The master control (leg 1)

Pristine `82cdf51`, same unit, 3 reps interleaved:

| lever | B median | A median |
|---|---|---|
| `0` eager | 35.99 s | 12.26 s |
| unset | 35.13 s | 11.62 s |

**B ratio 1.02** -- on the tree with no plan cache, forcing eager and letting
the default decide are the same machine on the LDLQ path, because `len(descs)
== 1 < 6` either way. **A ratio 1.055** -- the lever is wired and does bite,
just not on any call LDLQ makes. That asymmetry is issue #94 as data.

### The profile (legs D and E), and where the time went

`--crop 1024x1024`, run twice in opposite arm order so arm order cannot be
doing the work. B arm (LDLQ+refit) both times:

| | D: eager | D: auto | E: eager | E: auto |
|---|---|---|---|---|
| `cuLaunchKernelEx` | **262,912** | **0** | **262,912** | **0** |
| `cudaGraphLaunch` | 0 | **256** | 0 | **256** |
| wall | 13.41 s | 9.42 s | 12.09 s | 8.71 s |
| device busy | 3.53 s | 3.01 s | 3.13 s | 2.75 s |
| device fraction | 0.264 | 0.320 | 0.259 | 0.316 |
| power | 31.6 W | 31.4 W | 33.6 W | 32.4 W |
| envelope fraction | 0.226 | 0.224 | 0.240 | 0.231 |
| **`gpu_utilization`** | **96.0%** | **96.0%** | **96.0%** | **96.0%** |
| weights per joule | 2477.0 | 3550.0 | 2581.6 | 3718.9 |

**262,912 individual kernel launches collapse to 256 graph launches**, byte for
byte identical in both orders. That count is the claim: it is an API call
tally inside one process, and no neighbour, no swap storm and no clock
behaviour changes it.

Wall **1.424x** forward and **1.388x** reversed; work per joule **1.433x**
forward and **1.441x** reversed. In #13's units: **424 J -> 296 J** for the
same 1,048,576 weights.

**`gpu_utilization` reads 96.0% on all eight arms** -- before and after, LDLQ
and not, in both orders -- across a 1.4x change. Power against the envelope is
the instrument that says something: both arms sit near a quarter of ~140 W, so
this is a launch-bound path throughout, and the fix shortens it rather than
loading the board harder. The device fraction moving 0.26 -> 0.32 is the same
fact from the other side: the device was idle three-quarters of the wall
waiting for launches, and is now idle slightly less.

The host breakdown names the mechanism directly. Eager: 262,912
`cuLaunchKernelEx` and 782 `cudaStreamSynchronize`. Captured: zero
`cuLaunchKernelEx`, 256 `cudaGraphLaunch`, 526 syncs. The remaining
`cudaStreamSynchronize` (~8 s of a ~9 s host total) is the epilogue blocking on
the device -- `sse += float(final.sum())` forces a sync per chunk -- and is
untouched by this change, deliberately, because that is what keeps the returned
`sse` float summed in the reference's order.

### Residency, measured

`peak_alloc_gib` 0.951 (master control) -> 1.024 (branch) = **+74.8 MiB**. The
prediction from the buffers was ~74 MiB for the unit's four shapes (two rate-4
plans at ~36 MiB, two rate-5 plans at ~1 MiB). The plan cache costs what it
was priced at.

### What this is not

**1.394x, not #13's 3.22x, and that is the expected shape of the result.** A
window step is ONE Triton kernel of a few microseconds; the coset trellis
launches dozens of tiny ones per position. Same defect, same fix, smaller
headroom in the step loop because there was less launch overhead per unit of
work to remove. Reporting #13's factor here would have been borrowing a number.

## Review corrections (this branch's own code — not separable)

`8a164a6` and the residency-pricing commit are defects in the mechanism this task adds, caught in
review before any number was taken. They are listed separately so the diff can
be read in stages, but none of them can be dropped without breaking the change
they correct.

* the end-to-end LDLQ test's reference arm did not pin `impl="reference"`; it
  compared fused against fused, and the `lever=None` case passed by running one
  configuration twice. Deliverable 3 was not asserted until this was fixed.
* `TESSERA_WINDOW_GRAPH=0` read the plan cache before it read the lever, so a
  thread that had cached a plan under `auto` and then asked for the eager
  control replayed a graph. The forced-eager branch now precedes the lookup,
  which is what makes `0` an A/B control at all — the measurement below depends
  on it.
* the plan cache held four shapes and the shipping schedule asks for exactly
  four; raised to eight, priced from the buffers (`bbd8f0f`) rather than from a
  millisecond figure nobody had measured.

## Off-task fixes

* `e85c6f9` — `experiments/ldlq_hcost_matched_pair.py` (issue #13's harness,
  not this branch's code) ran every rep of one lever before starting the next,
  so its arms were matched inside a lever and unmatched between them — and the
  arms that get compared are the ones across levers. Interleaved rep by rep,
  with per-rep load recorded. Separable: take or drop it independently.

## Not fixed, and why

`encode._TCQPlan` is the sibling machinery from #13. I checked it for both of
the defects I fixed above: its `forced == "0"` maps to `impl="reference"`
before any plan lookup, so it is correct, and its `_TCQ_PLAN_CACHE = 4` is a
*measured* choice (80 MiB a plan, 0.295 -> 0.375 GiB peak on the unit its
author measured). Raising it would be a tuning change on someone else's
measured decision on a path this task does not measure, so I left it. There is
no common table build or common capture helper to unify: the two plans own
different buffers, different kernels and different epilogues, and a shared
base would be a wrapper over two bodies with nothing in common but the LRU.

`docs/ARCHITECTURE.md` is unchanged, deliberately: it makes no claim about
graph capture, and neither `TESSERA_WINDOW_GRAPH` nor `TESSERA_TCQ_GRAPH` is
documented there. No normative claim moved — the wire, the bytes, the defaults
and the format menu are untouched.

## Consultations

* `advisor()` before writing the plan cache: steered me to check the LDLQ rate
  vector before choosing the plan key. Verified against the code — the census
  above is that check, and it is why exact-`cols` keying (rather than capacity
  padding) is right: padding the 2-column calls to width 32 would run 16x the
  device work they need.
* `advisor()` after the mechanism landed: found the three defects listed under
  "Off-task fixes" above. Each was verified by reading the line before fixing.
