# The window step does not need the front.  What that is worth, and why

**Date** 2026-09-09 · **Repo** `/home/rob/tmp/tessera-fusion`, branch
`claude/window-viterbi-two-step-fusion` off `07ad344c3` · **Status** candidate,
**default OFF**, behind `TESSERA_WINDOW_BEST_FORM=1`. No default flip is
proposed here: the GLM expert shape at R = 8, and a real encode arm, are owed
first.

`_step` writes a `2^L` front every step and the next step's first act is to
minimise it away over `2^R` predecessors. Substituting the branch cost into
that minimum removes the front from the recurrence:

    best_s[c]  = min over f of ( best_(s-1)[(f*LOW + c) >> R] + B_(s-1)(f*LOW + c) )
    back[s][c] = that argmin, first minimal f

`_step_best` is that line. It needs **no new exactness argument**: its scan
compares the same sums `_step` stores and then scans, in the same order, under
the same strict `<`, built from the same `_mul` and the same `(d*d) * w`
association. What is gone is a store and a load of `FAN` times as many floats
per step, and with them the `[BC, BL, FAN]` intermediate `_step` forms only to
store. The traceback is untouched: the choice is still keyed by `c` in
`[0, LOW)`.

## Verdict

**2.63x on seconds and 1.51x on work per joule at the R = 3 production shape,
1.72x and 1.56x at R = 4, exact to the byte.** The lever is **not** the bytes.
It is that `_layout` sizes its two resident arrays from the class width instead
of the front width, so the same L2 budget admits `2^R` times as many columns --
more blocks over an unchanged serial chain.

| shape | arm | width | batches | s/call | W | work/J | vs front |
|---|---|---:|---:|---:|---:|---:|---:|
| L14 R3, 4096x192 | front | 32 | 6 | 0.10960 | 51.1 | 2.30e9 | — |
| | **best** | 192 | 1 | **0.04172** | 88.7 | 3.48e9 | **2.63x / 1.51x** |
| | best@w32 | 32 | 6 | 0.07386 | 63.0 | 2.77e9 | 1.48x / 1.20x |
| L14 R4, 4096x64 | front | 32 | 2 | 0.03852 | 51.5 | 2.16e9 | — |
| | **best** | 64 | 1 | **0.02240** | 56.9 | 3.37e9 | **1.72x / 1.56x** |
| | best@w32 | 32 | 2 | 0.03883 | 43.2 | 2.56e9 | 0.99x / 1.18x |

`best@w32` is the attribution control: the candidate held to the front form's
**internal** width by narrowing `_L2_BUDGET` for that arm alone, which the
plan-cache key already binds. So the store and the front alone are worth
**1.48x at R = 3 and nothing at R = 4** (0.99x), and the width carries the
rest -- 1.77x and 1.73x respectively.

A first version of that control held the width by passing `chunk=32` instead.
That was wrong and its own record said so: `chunk` is the OUTER loop, so it
moved the epilogue's `min`, its `sse` accumulation and its traceback call count
as well as the width, and the arm's `sse` differed from the other two. Every arm
above runs the production chunk and matches the reference on **states and
`sse`**, byte for byte. *A control whose answer cannot be compared with the arms
it controls has changed more than the one thing it names.*

## Why the byte ratio was the wrong target

Front bytes at fixed `L` are provably rate invariant -- `width` is
`budget // (2 * size * 4)` and `steps` is `rows // arity`, neither carrying a
rate term -- yet `experiments/results/window_viterbi_bench_graph.jsonl` moves
R4 to R5 by +34% at L=12, +78% at L=14 and +75% at L=16, and draws 20 W more
doing it. So a 2x front-byte figure predicts neither number above, and did not:
at R = 4 the byte reduction is 14.3x and the matched-width arm is 0.99x.

## How it was measured

Three arms, one process, one tensor, the encoder's own table (`E4M3_GRID`
through `window_table`, sigma 1.0, seed 0), weighted, 4096 rows, production
chunk, graphs on. Identity is asserted **before any clock is read** -- a faster
wrong answer is not a result. Timing is ABC blocks, each sized to at least five
seconds so a 1 Hz sampler sees it, with the sampler running two seconds either
side so every block is bracketed; joules are integrated by trapezoid over each
block's **own** interval, and a block whose interval is not bracketed
contributes neither energy nor work. The figure is total work over total
joules. Profiling never shares a run with timing.

Two mistakes were made and are recorded because the numbers they produced were
plausible. Clearing the plan cache per repeat timed a graph capture and called
it a step (0.2327 s against 0.1095 s on the same arm). Integrating only over
the samples that fell inside a block, with the full work in the denominator,
reported a work per joule that was too good.

## Receipts

Correctness, PrismaBuild GPU on gb10, no skips:

| module | passed | action |
|---|---:|---|
| `test_window_viterbi_best_form.py` | 91 | `bc224223f21b` |
| `test_window_viterbi_fast.py` | 52 | `e5ecbd8eccba` |
| `test_window_graph.py` | 20 | `2c37d508d0ec` |
| `test_window_body.py` | 30 | `71298a18e364` |
| `test_audit_doc_claims.py` | 10 | `3c6e5da63af2` |
| `test_e4m3_ladder.py` | 6 | `b2f99d50a71e` |
| `test_window_viterbi_two_step.py` | 68 | `f7bebb11c5f1` |

The 54 modules that reach the trellis through the encoder, six shards on
sparklina, 1453 passed, 5 skipped, 1 xfailed, nothing failed:
`49e5e1610a6b`, `d0093945785e`, `bd404e128a59`, `883df5d02b81`,
`8f21727b8f85`, `60ef250585fc`. Those six commands set no
`TESSERA_WINDOW_BEST_FORM`, so they ran the default, which is `0`.

The empty-problem defect root found while reading call coverage, demonstrated
in an action of its own because before the fix it takes the CUDA context down:
pre-fix `52cb04a22264` (illegal memory access, raised from `_init_best`),
post-fix `e0872d56cd35` (the reference's answer). `experiments/
window_viterbi_zero_row_repro.py` is that action's script.

Those seven modules are every test that names `window_viterbi` or
`viterbi_window` directly. They are not every test that reaches it: the
encoder does, through `encode_unit` to `encode_units` to `_drive_in_step` to
`_run_joined`, and 54 modules call the encoder. Because those 54 ran on the
default `TESSERA_WINDOW_BEST_FORM=0`, what they cover is the front form, whose
positive-input path this branch does not change. They are a non-regression
receipt for the default and are not offered as best-form coverage; the best
form's coverage is the 91 above, which run both settings. The 91 cover both
production rates and R5 at L14, `R > L-R`, eager and captured, weighted and
not, duplicated table rows that force two predecessors to carry the identical
float, a coarse table whose distinct rows produce sums that ROUND to the same
float, an assertion that the tie family actually ties, and `cols=205` against
chunks 40, 96 and 130 so both `cols % chunk` and `m % BC` are non-zero, and
zero rows against positive columns in both graph modes and both spellings.

Timing `63eae68972d2` (sparklina, measurement, exclusive). Torch profiles, one
config per action so neither overwrites the other's trace: R3
`748070a53a96` (blob `feb3d43c8704`, 107003 events), R4 `d2da1454fefa` (blob
`5fd5e54cbe1e`, 41348 events). Netdata over the timing window has sparky
peaking 96 W against a 60.8 W mean while the other box held 3.2-4.0 W, so
nothing else was on the fleet.

Registers off the launched `CompiledKernel`: front 40 regs, 0 spills, 2048
shared; best 40/0/1024 at R3 and 37/0/1024 at R4. The candidate is not paying
for its speed in occupancy.

## What this does not establish

### The GLM shape, measured by root, not here

Root has since run the candidate on the real GLM expert shape at R = 8, in its
own pricing producer, and that run is the qualification this section says is
owed. Recording what it established and what it did not, since it is root's
measurement and not this branch's: 16 real layer-4 `down` experts against the
original 864-sequence capture, fixed B8, ABBA, front 77.4026 s and 77.4164 s
against best 37.3149 s and 37.4762 s, with **all six arms exact on wire SHA
and `dloss`**. Action `569ee819458cb3cf1c91a687ff66f1edbcd51a87a4f31813e36dc9fb2c572c2b`.

So the seconds and the byte identity hold on the shape that matters. **The
energy half does not follow from it.** Root rejected all four of that run's
CUDA traces, because torch 2.13 dynamic collection reported physically
impossible durations even across four fresh contexts, and owns a replacement
capture. No work-per-joule figure on the GLM shape is claimed here or there
until that lands, and the default stays off regardless.

### What is still open

One shape family, one box, `viterbi_window` alone. The encoder reaches it
through `_run_joined`, so a real column count varies with how many units join
and the width lever varies with it. `E4-R1088` includes R4 and R5, and dense
BF16/E2M1 use other `L` and arity; nothing here speaks for them. At arity above
1 the candidate still reads its trailing coordinates inside the `f` loop, so an
arity 2 timing would carry a load asymmetry these arity 1 numbers do not. The
GLM expert shape at R = 8 is the qualification that matters next, and the
default stays off until a real encode arm and a review say otherwise.
