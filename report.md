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

The epilogue (transpose, `cost.min`, `sse += float(final.sum())`, traceback)
stays on the host stream and in its original order, so the returned `sse`
float is bit-identical and not merely close.

## Bit-exactness

`tests/test_window_graph.py`, 19 tests, all against `impl="reference"` — the
torch chain that defines the trellis, not another run of the same kernels.

## Measurement

FILLED IN BELOW.

## Off-task fixes

Each is a separate commit.

* `8a164a6` — the end-to-end LDLQ test's reference arm did not pin
  `impl="reference"`; it compared fused against fused and the `lever=None` case
  passed by running one configuration twice.
* `8a164a6` — `TESSERA_WINDOW_GRAPH=0` read the plan cache before it read the
  lever, so a thread that had cached a plan under `auto` and then asked for the
  eager control replayed a graph. The forced-eager branch now precedes the
  lookup, which is what makes `0` an A/B control at all.
* `8a164a6` — the plan cache held four shapes and the shipping schedule asks
  for exactly four; raised to eight with the measured shape census recorded
  beside it.
* `e85c6f9` — the matched-pair harness ran every rep of one lever before
  starting the next, so its arms were matched inside a lever and unmatched
  between them. Interleaved rep by rep, with per-rep load recorded.

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
