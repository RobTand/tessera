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
plan-cache key already binds. Holding the width fixed leaves everything else
the candidate changes, which is not the store alone: it is the rewritten
recurrence together with materialising the front once at the end instead of
every step. Those two combined are worth **1.48x at R = 3 and nothing at
R = 4** (0.99x), and the width carries the rest -- 1.77x and 1.73x
respectively. The control separates width from the rest; it does not separate
the rest into its parts, and this doc does not claim a number for the store.

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
sparklina, 1453 passed, 5 skipped, 1 xfailed, nothing failed: `49e5e1610a6b`,
`d0093945785e`, `bd404e128a59`, `883df5d02b81`, `8f21727b8f85`,
`60ef250585fc`. Those six commands set no `TESSERA_WINDOW_BEST_FORM`, so they
ran the default, which is `0`. They also ran a tree that precedes the
empty-input guard below, differing from this head by that guard and by prose;
the guard is inert on positive input, which is the only kind those 54 modules
feed it, so they were not re-run. Root audited that question rather than
taking the claim, reading the CAS source behind all 13 receipts above
(`root-post-empty-fix-tests-cas-source-audit.json`): the six native modules on
the post-guard tree come to 209 passed with 156 of them allocating CUDA and no
skips, and the CPU reference module to 68 passed with no CUDA. Those are the
runs already in the table, verified, not run again. The five skips in the
broad run are real and named: one because E2M1 publishes no reader range, two
because a shape cannot be cut four ways, two because it cannot be cut eight.

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

## The GLM shape, measured by root, not here

Root has since run the candidate on the real GLM expert shape, in its own
pricing producer, and that run is the qualification the last section of this
doc named as owed. Recording it here because it settles that question, and
recording it as root's measurement rather than this branch's: 16 real layer 4
`down` experts, 4096x2048, artifact rung `R832`, fixed batch `B8`, against 864
prefetched entries from the original capture, ABBA. The rung is not the
recurrence rate; the actual recurrence over that rung mixes `R3` and `R4`, so
nothing here is an `R = 8` measurement and the two must not be read as one
number.

| arm | s/call | J | W | vs front |
|---|---:|---:|---:|---:|
| front | 77.409474 | 4516.04582 | 58.33970 | — |
| **best** | **37.395534** | **3078.12257** | **82.31257** | **2.070019x / 1.467143x** |

Those are the means of two ABBA pairs each: front 77.4026 and 77.4164, best
37.3149 and 37.4762. Candidate power peaks 91.44 W and 91.52 W. **All six arms
are exact on wire SHA and `dloss`**, and exact against the older `07ad` batch
screen as well. Action
`569ee819458cb3cf1c91a687ff66f1edbcd51a87a4f31813e36dc9fb2c572c2b`; root's
audit is `performance-best-form-ab-01/root-real-wall-energy-parity-audit.json`.

**Where the work per joule comes from matters, because the first four CUDA
traces did not survive.** Root rejected both front traces on physical checks,
after torch 2.13 dynamic collection reported impossible durations across four
fresh contexts. The ratio above never depended on them: it integrates the
continuous `pqteld` series between wall clock endpoints, which measures the box
rather than a kernel timeline. Root then re-captured, complete call and with no
dynamic toggling, and those profiles pass (PB `c86c6920`, CPU audit
`15a5ba58`). On the `R3` residual shape, 4096x192 weighted at `L14`: the front
takes 6 graph launches over 24576 steps for 98496.575 us of kernel in a
103884.535 us span; the candidate takes 1 launch over 4095 steps for 34566.569
us in a 37479.674 us span, with traceback 2159.72 us against 2159.758 us. Mean
step is 4.00784 us front against 8.44116 us candidate at six times the width,
and `best_step` is 93.55% of candidate kernel time. So the per step cost
roughly doubles and the step count falls sixfold, which is the width lever the
verdict above names, now visible in the kernel timeline and not only in the
wall clock.

The tree root measured is not this branch's head. It is `61da`, which is
`af1497c4b5` plus an identical prose fix, and precedes the empty input guard
added here; that guard is inert on positive input. The default stays off
regardless.

## What this does not establish

### What is still open

One shape family, one box, `viterbi_window` alone. The encoder reaches it
through `_run_joined`, so a real column count varies with how many units join
and the width lever varies with it. `E4-R1088` includes R4 and R5, and dense
BF16/E2M1 use other `L` and arity; nothing here speaks for them. At arity above
1 the candidate still reads its trailing coordinates inside the `f` loop, so an
arity 2 timing would carry a load asymmetry these arity 1 numbers do not. The
GLM expert shape above closes the qualification this section used to name as
next, on both halves: root's re-captured complete-call profiles supply the
paired kernel attribution the first four traces could not. What is still owed
is breadth, not a real encode -- the upper `R1088` endpoint, which root is
staging, and the dense shapes. The default stays off until those and a review
say otherwise.

## 2026-10-06: issue 652 joined-geometry before packet

This continuation reuses the retained geometry extension at
`0e1f33ae834600b2dfc713da7b5aa9a0ce85bc3c`. The four cases use window bits
14, arity one, eight units with 32 columns each, and the incumbent best-form
tile `64,4,2`: rung 1088 joins 192 columns at rate four and 64 at rate five;
rung 1152 joins 128 columns at each rate. Both 2048-row and 4096-row cases
are expressed. The production joined-call driver and its automatic graph
behavior run on **generated targets and weights**, not a captured G2
calibration block or an entire encoder unit.

The benchmark now has `--cpu-dry-run`, which exercises its own argument
parser, table construction and joined gathers on four CPU rows before a GPU
submission. It also fixes the G2 timing routine's undefined argument variable,
refuses actual numerical disagreement with the reference, and brackets the
warmed Nsight Compute capture. No production source or defaults changed.

Final benchmark source: `2b711e0e7498802954bf75cfdabe7b93c6f47737`.
The final-source CPU action
`06865752b875b962393dfa71adbcc13a4ff5d79daa52c87cfbac541d1c903f90`
executed on dl380g10 with return code zero. Seven entry-point invocations
covered all four geometry cases plus timing, Torch profiling and Nsight
argument modes. These dry runs did not execute CUDA kernels.

The first exclusive GB10 measurement was allowed to choose either Spark and
ran on sparklina. Action
`df7dc3ca89f1e763a11c80b06cf682330664104a874ea308d3d8657994d2d597`
returned zero in 39.49199 seconds. At rung 1088 with 2048 rows, both forms
returned reference-identical states and squared errors at both rates.

| Existing form | Median seconds per joined group | Branch evaluations per joule |
|---|---:|---:|
| Front counterfactual | 0.06914 | 1.6743915725 billion |
| Incumbent best form | 0.02289 | 4.0642715040 billion |

The work denominator is the benchmark's branch-evaluation count, not tokens,
encoded-model throughput or served quality. Energy uses each form's own
three timestamped power blocks; all six blocks were bracketed and fully
covered. Recomputing reductions from those blocks differed by at most
0.000001635 seconds per call and 0.034607 branch evaluations per joule,
within the report's rounding. These are existing-form comparisons, **not a
before/after optimization claim**.

The exact 19:30:02–19:30:42 UTC window was recovered from Netdata on both
Sparks through CPU action
`fc3f2841e6342b65f26e04fce36c1cfdc89c0ba619b445b8a83a15517aef44d6`
(return code zero). Sparky mean GPU power was 13 W; sparklina mean was
60.995 W, median 71.05 W and maximum 73 W. Native power collection cadence
was ten seconds; returned one-second buckets are not one-second sensor
samples and cannot qualify the benchmark's five-second blocks individually.
The first collection action returned zero but recorded DNS errors; its error
receipts remain retained, and recovery used verified numerical addresses.

Raw results and power responses are under
`/mnt/shared/tessera-measurements/issue-652-geometry-sol-20261006/`.
The timing receipt is
`/mnt/shared/prismabuild-fleet/cas/actions/v3/df/df7dc3ca89f1e763a11c80b06cf682330664104a874ea308d3d8657994d2d597.json`,
receipt digest
`40914fb09e48449ade9c8fc701d3e13486140e59bfeb266b83ff6ae9c262b9d3`.
The full action list, commands, completion clients and limits are in
`/home/rob/fleet/records/kernels-652-g2-geometry-sol-20261006.json`.

**Still incomplete:** the other three timing cases, the Torch and Python
sampling profiles, and the incumbent/front Nsight Compute profiles were
admitted with completion clients, but the observed scheduler holds were
`measurement_host_not_idle` and `transition_busy`. No profiling result,
cache-bandwidth roofline, limiting-resource conclusion, packed-wire equality,
G2 unit timing, production optimization or issue closure is claimed here.
Do not optimize production code from this partial packet.

### Later completion events on 2026-10-06

The five previously pending timing and host-profile actions subsequently
executed with return code zero, without resubmission or an admission bypass.
All four geometry cases now have reference-identical states and squared
errors for both forms and both rates.

| Rung | Rows | Front median seconds | Incumbent best median seconds |
|---|---:|---:|---:|
| 1088 | 2048 | 0.06914 | 0.02289 |
| 1088 | 4096 | 0.16390 | 0.04677 |
| 1152 | 2048 | 0.06942 | 0.02405 |
| 1152 | 4096 | 0.13084 | 0.04856 |

All 24 timing blocks have bracketed full power coverage. Across the eight
form/configuration reductions, the maximum mean-seconds rounding difference
is 0.000004680 and the maximum work-per-joule difference is 0.048344.

The Torch action
`a7fc61ae6ef7ab2e274839d15739e73537e8a6ca620ccb59a9213c6b99bc9b5b`
produced 82,825 trace events, retained as the 553,371-byte compressed blob
`605dae0b0197fbfc94275aa3906b5974d274f7006bfcf469f22c8f1a30d4defd`.
Its aggregate device times include 246.1048 milliseconds over 32,768 front
step launches and 84.5407 milliseconds over 8,190 best-form step launches.
These are profiled aggregate device times, not unprofiled wall latency.

The Python sampling action
`0b79337e87335b97de41650d776b5ac14bf8a69094758921e06152d1995e2ea6`
produced 2,376 samples with zero sampling errors using py-spy 0.4.2 at
100 samples per second. Blob
`e87f6315fdfe0953821b54ee13803de446d5ac4eea928cf08f5a3e4458915502`
contains the actual sampled stacks: 1,092 leaf samples in graph replay and
526 in CUDA synchronization. Sampling includes startup and correctness work;
it does not by itself quantify CPU compute overhead or identify a roofline.
Both profile blobs' bytes were checked against their recorded digests.

CPU action
`3a09f81f09161753983dd83d6b720a5fd856b863c21552ba08a8e7916d074049`
returned zero and retained both-Spark Netdata responses for every remaining
completed timing and host-profile window. Native power cadence remains ten
seconds. In particular, the nine-second Torch window cannot establish a
fresh per-kernel power reading; returned one-second buckets do not fix that.

The two Nsight Compute actions still lack observed completion evidence:
`925589c5cad78e160e34f1a3f2a366c677f31c3978f1f7ffcdd9e149c5abf889`
(incumbent) and
`5e12c1972a2583bb3574819cbce5c74a0c70db54a08f77e2808feb8cdcbfd5ab`
(front control). Their published completion client is retained. The last
admission diagnostic recorded actual host-idle/transition safety holds;
no new seal or permission requirement was introduced. Cache counters,
achieved resident bandwidth and a limiting-resource conclusion remain
unmeasured, so the full issue 652 acceptance is still open.

### Recovered Nsight endings and final generated-input packet

The two original actions completed on sparky without resubmission. The
incumbent action `925589c5cad78e160e34f1a3f2a366c677f31c3978f1f7ffcdd9e149c5abf889` returned zero in 20.84215 seconds;
the front control `5e12c1972a2583bb3574819cbce5c74a0c70db54a08f77e2808feb8cdcbfd5ab` returned zero in 24.46473 seconds.
Published action readers and `pbwait.py --wait-s 0` observed both endings.
Their receipt digests are, respectively,
`19b13e7fb676ef83d8b4e2c0ce3ec0657944ae7c63032ba54a297c2dccc67c8d` and
`46a04c3885bd9774e2ca20eef3b0a13972cbcc4b2f589ce2476fc0601064b4f1`.
Both local result claims passed payload-byte, digest and manifest-binding
checks; both stdout and stderr files matched their own recorded digests.

The reports were decoded **without running another GPU workload** by CPU
PrismaBuild action
`efe1f380091149a52c0322d0b1773657c6e82b69905f5130f719817400c847c3` on dl380g10 (return code zero,
44.99666 seconds; receipt digest
`74afa308176848911ade7ea710f9559db9820b0ca404cc9ff2b4b73d13794858`).
It used the public NVIDIA x86 reader, version 2025.3.1.0, build 36398880,
whose 322,049,776-byte distribution matched published digest
`d3c0a0402511034c58227b817cbfed599765f32dc662cdcda58922247b52dd7a`.
The temporary reader installation was removed on completion. An unexecuted
ARM reader was withdrawn from the ready queue; two failed CPU-only decoder
commands remain recorded as failures, not measurements or success receipts.
They failed on a quoted newline and an unsupported printing option,
respectively. No original timing or profile action was duplicated.

Each capture contains eight matching step launches at rung 1088 with 4096
rows. All incumbent launches have grid `(16,48,1)`, block `(64,1,1)` and
stream 13; all front launches have grid `(16,16,1)`, block `(128,1,1)` and
stream 13. Thus these are sampled, differently sized launches, **not equal
work per kernel or the complete two-rate joined group**. Each launch took
nineteen replay passes. The original command leaves Nsight's cache control
and clock control at their documented defaults: cache flush before each
replay and base clocks. See the [Nsight Compute 2025.3 command-line
contract](https://docs.nvidia.com/nsight-compute/2025.3/NsightComputeCli/index.html).
These counters cannot certify unflushed production cache residency.

The following are means across the eight retained launches in each report;
full per-launch values, units, ranges and sampled warp-state counts are in
the packet. Traffic rates multiply the measured sector counts by 32 bytes
and divide by that launch's measured duration. Gigabytes are decimal.

| Counter or derived rate | Incumbent best | Front control |
|---|---:|---:|
| Profiled duration, microseconds | 21.372 | 15.876 |
| Total L2 traffic, gigabytes per second | 133.593 | 285.144 |
| L2 sector hit rate, percent | 38.491 | 5.305 |
| L2 throughput, percent of sustained elapsed peak | 12.649 | 17.189 |
| System-memory fill proxy, gigabytes per second | 40.231 | 136.749 |
| System-memory write proxy, gigabytes per second | 46.428 | 134.717 |
| Issue active, percent of sustained elapsed peak | 23.026 | 5.612 |
| Active warps, percent of sustained active peak | 63.868 | 41.456 |
| Registers per thread | 40 | 40 |
| Total shared bytes per block, including driver allocation | 2048 | 3072 |
| Long-scoreboard not-issued samples, total | 4257 | 7602 |
| Drain not-issued samples, total | 1617 | 1080 |
| Memory-input/output throttle not-issued samples, total | 1253 | 85 |
| Short-scoreboard not-issued samples, total | 926 | 118 |

Recomputing traffic rates against the printed sector-per-nanosecond counters
differs by at most 0.000015879 gigabytes per second, within printed rounding.
Sampled stall counts are not percentages of wall time. Neither report
contains a direct `dram__` or `sys__` metric: L2 system-memory fills and
writes are retained **as proxies, not direct DRAM-controller counters**.
The L2 rates above are achieved rates in these cache-flushed sampled kernels,
not a measured resident-bandwidth ceiling or a remaining-limiter diagnosis.

Both-Spark Netdata power and CPU responses now bracket the actual profiler
action windows, including their fractional start and finish times:

| Action window in UTC | Sparky returned mean watts | Sparklina returned mean watts |
|---|---:|---:|
| Front: 20:04:01.166 through 20:04:26.611 | 16.6444 | 44.0000 |
| Incumbent: 20:05:29.989 through 20:05:52.065 | 14.9200 | 13.9760 |

The returned windows are 20:04:01–20:04:27 and 20:05:29–20:05:53,
respectively. They have 27 and 25 one-second buckets, but only three and four
native ten-second power points. No returned power point has an empty,
reset or partial annotation. Recomputed bucket means agree with Netdata's
view means within 0.000000045 watts. These short replay windows include
startup, reference checks and both forms; they cannot supply per-kernel
joules. The receipt's faster sampler separately records sparky mean/peak
22.164/60.220 watts for the incumbent and 21.051/55.130 watts for the front
control. These are different sampling instruments, not interchangeable
energy reductions. The existing four unprofiled timing and work-per-joule
rows above remain the comparison evidence and were reused unchanged.

Final machine-readable packet:
`/mnt/shared/tessera-measurements/issue-652-geometry-sol-20261006/issue652-before-profile-packet-final.json`,
SHA-256 `490b4fd62e20be1cbf832c7720ce6f4ab65119ab6dbc2743ec40c15f41f9567a`.
It retains all successful original receipts, raw counter exports and their
own digests, both-Spark windows, resource observations, timing reductions,
failed decoder attempts and exact missing acceptance. The benchmark source
remains `2b711e0e7498802954bf75cfdabe7b93c6f47737`; this addition changes
only documentation.

**Draft-source readiness, not issue closure:** the generated weighted-target
before packet is ready for parent source and receipt review. Direct DRAM
counters, an unflushed resident-bandwidth measurement, captured G2 inputs,
complete encoder-unit before/after seconds and joules, explicit tie-case and
packed-wire comparisons, a production candidate and a defensible remaining
limiter are still missing. No production source, default, wire, serving pin
or admission gate changed; no optimization or root-cause claim is made.
Pull request 1016 remains draft and issue 652 remains open. Hosted `pure`
was observed successful only on documentation head
`a35f0c93ae9bca5b1e268d6f992c6b6bf6d2b7ea`; parent approval, current-master
refresh and integration-head checks are not claimed for this later addition.
