# The window-GEMV decode-path latency A/B: the pair, and the box it needs (2026-09-04)

> **STATUS: the instrument is built and verified against the 2026-09-03 pair;
> the ratio itself is NOT taken.** Issue #109 asks for one number -- the
> window-GEMV kernel's decode-path latency against the route it replaces, as a
> ratio, on an otherwise idle box. This document does not contain that number.
> What it contains is (a) the pair harness master did not have, (b) the
> box-state readings that decided against taking the measurement tonight,
> (c) a re-reading of the 2026-09-03 arms that produces, for the first time,
> a *measured* control for why their 8x is not a lane result, and (d) the
> **exchange rate** between the served ratio and the lane, read off that pair's
> own trace: the lane owns 29.8% of a decode step's device time and `lm_head`
> owns 48.1%, so a served **X** implies a lane-bucket speedup of
> **1 + (X-1)/0.298**. Nothing here is a placeholder standing in for a number
> that was taken and disliked. **Section 7, added later on 2026-09-04**,
> records three things found while asking which box this can run on: the
> box-side instrument wrote nothing for the whole campaign and the driver
> discarded the reason (fixed, with the four windows recovered from
> Netdata); the quiet box is reachable from a `/mnt/shared` clone without
> cloning to it; and a loaded box stops being addressable at all.

## 1. What #109 owes, and what is missing

#83 shipped the window GEMV's census and its served KL and deferred the
latency. #109 restates the latency deliverable with the discipline it needs:
profile before and after, two instruments (in-process and box-side), power
against the ~140 W envelope rather than `gpu_utilization`, and an idle reading
taken *before the run starts* rather than asserted.

The 2026-09-03 campaign's own §4 records why its two arms are not that number:
they came out **8x apart** where the kernel difference cannot exceed ~2x,
because the box moved between them. That document argued the point from what
the kernel *could not* be. This one measures it -- see §3.

## 2. What was built

Master had `experiments/window_gemv_latency.sh`, which serves **one** arm and
does it well: it reads TTFT and TPOT off the engine's own histograms rather
than a client clock, enables vLLM 0.28's profiler through `--profiler-config`
(the env-var spelling silently no-ops), records host load at both ends of every
timed window, and refuses a serve with spec-decode active. What master did not
have is anything that turns two arms into a **pair**. Four pieces now do:

| piece | what it owns |
|---|---|
| `experiments/box_power_window.py` | GPU power, `gpu_utilization`, load, CPU and swap I/O for an explicit UTC window, off a box's own Netdata, into a receipt that carries the query beside the number |
| `experiments/window_gemv_latency_ab.sh` | the quiet-box gate, the settle, the crossover order, and one power window per arm cut to that arm's own marks |
| `experiments/window_gemv_latency_ratio.py` | the ratio, its provenance, and its refusal |
| `experiments/window_gemv_load.py` (extended) | swap *activity* beside swap in use; sub-second marks on the profiled sub-loads |
| `tests/test_window_gemv_latency_ratio.py` | the ratio's direction, its provenance stamps, and its refusal, pinned on the 2026-09-03 receipts' own bytes |

Three of those deserve their reason stated.

**The quiet-box gate is a measurement, not an intention.** #109 names #5's
failure directly: that report had to give a 5.0-9.94 s/unit *bracket* rather
than a number because Netdata showed 63-88 W on sparky for the seven minutes
before the process existed. So the chain reads the idle window off Netdata,
writes it to a receipt, and **refuses to run** when the box is not idle by that
reading (GPU peak over 30 W, load1 over half a runnable process per core, or
sustained paging -- see §4 for the rate thresholds and why they are rates).
`TESSERA_LAT_REQUIRE_QUIET=0` records the reading and runs anyway; the refusal
is the default because the alternative is a number nobody can use. Run against
tonight's two boxes it refuses both, which is §4.

**The order is crossed over.** Rep 1 runs A then B; rep 2 runs B then A. Drift
that is monotone across a session -- a box warming, another agent's job ramping
-- lands on the two arms in opposite directions and cancels in the mean of the
two ratios. Two runs in a fixed order cannot do that, and the 2026-09-03 pair
is what a fixed order looks like when the box moves.

**"Both arms in the same process" is refused by design, and the refusal is
this plugin's own.** #109's framing asks for the arms in one process and one
session. The session half is what the chain delivers. The process half cannot
be delivered and should not be engineered around:

- Whether the window-GEMV extension can be built is a **process** fact. Arm B
  is made by pointing `TORCH_EXTENSIONS_DIR` at a read-only mount, so
  `kernel_window_gemv._ext()`'s `os.makedirs` raises before nvcc is consulted
  and the route takes its published fallback -- the same state a rate-3 wire or
  a shard start state reaches. `experiments/window_gemv_served.sh` said this at
  the top of the file when the arms were built.
- The plugin **refuses in-run dispatch changes on purpose**.
  `src/tessera/serving/flags.py`'s `_latch` raises when a serving flag moves
  after dispatch is fixed: *"restart the process instead of changing serving
  behaviour within one run"*, precisely so a run's numbers describe one
  setting. A per-request toggle added to `fp8_gemv.streamed_apply` to make an
  in-process A/B possible would be a measurement lever against the contract
  that exists to keep measurements meaningful -- and it would not reach the
  compiled arm at all, where the dispatch is traced into the graph.

What is held equal instead: one checkpoint (arm A and arm B are `cp -al`
hardlinks, one inode, checked at run start and not merely when they were made),
one pinned image by digest, one box, one session, back to back, crossed over.

**The B side is the streamed fallback, not the resident route.** #83 item 3
says "against the resident route"; #109 says "the route it replaces". They are
different comparisons and the second is the one owed here: what the GEMV
replaced is the streamed route's own materialised path -- kernel-decode the
window into a tile, then `torch._scaled_mm` -- which `fp8_gemv.py`'s docstring
names and which arm B serves by refusing the lane. A resident arm changes the
residency *and* the lane and would price two differences as one.

**Swap activity, without relaxing the contention label.** sparky carries ~2 GiB
of resident swap tonight from an earlier stampede, and `window_gemv_load.py`
labels a receipt contended above 1 GiB in use. Lowering that threshold would
have made tonight's receipts look clean, which is the wrong repair. The
threshold is unchanged. What the receipt now also carries is `pswpin`/`pswpout`
differenced across each timed window, so it can say "2 GiB resident, nothing
moved" rather than leaving a reader to guess which of the two facts the label
meant. #83's armB was the other case -- 13 of 15 GiB resident *and* thrashing --
and only the second reading separates them.

## 3. The 2026-09-03 pair, re-read: its own control says the 8x is the box

Running the new ratio tool over the two eager receipts reproduces that
campaign's numbers and adds the thing it did not have.

```
decode.tpot   engaged  43.541 ms   fallback 349.367 ms   x8.024   (n 12/12)
decode.itl    engaged  43.541 ms   fallback 349.367 ms   x8.024   (n 1524/1524)
decode.ttft   engaged  70.339 ms   fallback 541.691 ms   x7.701   (n 12/12)
prefill.ttft  engaged  69.718 ms   fallback 299.054 ms   x4.290   (n 12/12)
VERDICT: NOT EVIDENCE about the lane
```

**The last row is the pair's own control.** The prefill window is a 512-row
forward: `fp8_gemv`'s `GEMV_MAX_M = 8` refuses it by name and **both** arms
serve it through the materialised path. Whatever the lane is worth, that row
should read 1.00x. It reads **4.29x**. A pair whose null arm moves by 4.3x
cannot price a lane at 8x, and this is now a measured statement rather than an
inference from what a kernel could plausibly do.

Two smaller facts fell out of the re-read and are recorded because they change
what the old receipts are good for:

- **The eager receipts' TPOT is recoverable, and only through `series_moved`.**
  Both eager receipts carry `tpot: None` and `itl: None` -- the driver was
  reading `vllm:time_per_output_token_seconds`, which vLLM 0.28 does not
  publish. The 43.541/349.367 numbers quoted in the 2026-09-03 document exist
  only because `moved()` dumps every vLLM series that changed over each window.
  The insurance paid; the ratio tool reads it and stamps `series_moved` as the
  source rather than presenting it as a published histogram.
  **This is an age, not a live defect, and the clocks say so:** the stem was
  corrected in `b8ef715`, committed `23:24:09Z`, while armA started
  `23:18:28Z` and armB `23:22:53Z` -- both arms were already running. A future
  receipt should carry `tpot` in its named field with `source: histogram`, and
  a `/2` receipt that still says `series_moved` there is a finding, not a
  formality. `tests/test_window_gemv_latency_ratio.py` pins the recovery
  arithmetic on those receipts' own bytes so the fallback keeps working
  whichever way that goes.
- **Those receipts cannot say whether the box was paging.** They predate the
  swap-activity field, so the tool reports "cannot say" rather than "nothing
  moved". They are `schema /1`; the new receipts are `/2`.

## 4. The box, tonight, measured

`nvidia_smi.gpu_power_draw` and friends over 2026-09-04 04:47:51 - 05:47:51 UTC,
one hour, read by `box_power_window.py` (receipts:
`/home/rob/tessera-runs/ts109/power-decision-{sparky,sparklina}.json`).

| box | GPU power med / max | envelope | load1 med (of 20 cores) | CPU user med | `gpu_utilization` |
|---|---|---|---|---|---|
| sparky | 5.0 W / 56 W | 3.6% / 40% | 2.46 | 8.85% | 0-96, med 0 |
| sparklina (`gx10-6b77`) | 16.0 W / 32 W | 11.4% / 22.9% | 4.10 | 20.20% | **96.00 flat** |

**sparklina is not available and its own utilization figure is the evidence for
principle 15 rather than against it.** `gpu_utilization` reads **96.00 for every
one of 359 samples** across the hour -- min, median and max identical -- while
power sits at the 16 W floor and never leaves it. A reader ranking boxes by
utilization would have called sparklina the busiest machine in the fleet; power
says its GPU did nothing for an hour. What sparklina *is* doing is steady CPU
work (20% of 20 cores, load ~4.1) for an out-of-pool campaign whose GPU claim
runs to roughly 07:00, and its host-side load alone disqualifies it for a
host-driven latency number.

**sparky was busy, then briefly free, then busy again.** A `serve_and_dump_kl.sh`
serve for #60 held the serve lock and port 8000 at 05:33 UTC and was gone by
05:47. The pool ledger
then showed both of sparky's GPU slots claimed by two other agents' jobs
(`/home/rob/tmp/wf105` and `/home/rob/tmp/wf75`, `needs_gpu: true`, claimed
05:38 and 05:42 UTC). Holding the GPU *lock* is not the same as loading the GPU,
and the 5 W median says the GPU was mostly idle -- but a claimed slot is another
agent's job about to run, and a latency number taken beside it is a number about
that job.

**The quiet gate, run against those two receipts, refuses both boxes.** That is
the gate doing its job rather than a formality, and it is worth reading as the
decision itself:

```
sparky      REFUSED  GPU peaked at 56 W in the idle window (>30 W is not idle)
                     paging in: median 14.152 KiB/s, max 18890.417 KiB/s
sparklina   REFUSED  GPU peaked at 32 W in the idle window (>30 W is not idle)
```

The paging clause was rewritten while doing this. Its first spelling refused on
`swap in max > 1 MiB/s` or **any** page-out, which on a box carrying ~2 GiB of
resident swap refuses on a single log flush touching a paged-out page -- an
event that costs a latency measurement nothing. Paging is a rate, so the median
now carries the refusal (>100 KiB/s sustained) and the max only catches a burst
large enough to be a stall in its own right (>10 MiB/s). sparky trips the burst
clause on a genuine 18.9 MiB/s spike, not on the resident 2 GiB. What was
deliberately **not** relaxed is the per-arm `contended` label in
`window_gemv_load.py`: loosening the threshold that decides whether my own
receipt is admissible is the self-serving edit, and the `/2` schema's per-window
`pswpin`/`pswpout` deltas are the honest way to distinguish "swap is resident"
from "pages moved while this arm was timed".

**And the queue is the other half of the refusal.** The submission below was
accepted at `05:55:47Z` as `66266919c4c2` and had not been claimed an hour
later, for a reason the ledger states plainly:

| | | |
|---|---|---|
| rank 1 | `1244c3e5db31` | **156 passes**, `gpu: 1`, `/home/rob/tmp/pb1` |
| rank 2 | `4e70f110eb97` | priority 30, CPU only |
| rank 3 | `66266919c4c2` | priority 5, `gpu: 2` -- this job |

`pool.claim` orders ready items by passes first, and an item at or past
`STARVATION_FLOOR = 3` that cannot acquire returns `None` from the scan rather
than letting smaller work overtake it (`pool.py:750-754`). A 156-pass item ahead
of this one therefore withholds sparky on every pass, correctly. Both of
sparky's GPU tokens are held (`reservations/sparky/held/`: `gpu-0000` by wf105,
`gpu-0001` by wf75), and `--exclusive` resolves to `gpu: 2` -- the box's whole
capacity -- so this job needs *both* back before it can run at all.

One pool defect was found on the way and is **not** fixed here, because it is
not this repo's code and the stale loops are not mine to restart: sparky
publishes its worker offer from **two runtime versions at once**, and the file
flickers every ~15 s between

```
{"capacity": {"cpu": 10, "gpu": 2, "mem_gb": 48}, "runtime_commit": "1caa8908..."}
{"capacity": {"gpu": 2, "mem_gb": 48}}                      # no cores, no commit
```

(reproduced live at 06:00:14-06:01:09Z, 4 of 12 samples in the second shape).
`pbrun`'s pre-submit fit check reads whichever shape it catches, and a demand
naming `cpu` is rejected outright against the second one -- "sparky's offer had
no cores dimension" -- which cost two submission attempts here before one
landed. The claim path is safe (it reads the reservation ledger, which keeps its
`cpu` tokens), so this is a submit-path flake rather than a scheduling bug, but
a `--cpus N` submission to sparky will fail intermittently until one of the two
worker loops is retired.

## 5. What remains, and on which box

The A/B itself, which is one command once a box is genuinely free -- and the
box is now an argument rather than a constant:

```
BOX=gx10-6b77 CWD=/mnt/shared/tessera-ts109 bash experiments/ts109_submit.sh
```

`BOX` names the tag the action requires; `CWD` is what makes naming it
possible at all. **Which box an action can run on is a fact about its
checkout's path, not a flag.** `pbrun`'s own rule: "Every box mounts
`/mnt/shared` at the same path, so a checkout underneath it is visible to all
of them and an action that runs there can run anywhere. A checkout outside it
exists on exactly one box." A worktree under `/home/rob/tmp` is therefore
pinned to the box that holds it, and `pbrun` says so on every submission from
one. So the answer to "must the branch be cloned onto the quiet box" is **no**:
one clone under `/mnt/shared` plus `--tag` reaches either GB10. It has to be a
`git clone`, not a `git worktree` -- a worktree writes a `.git` *file* pointing
into `/home/rob/tessera/.git`, which the other box cannot see, and the worker's
closure check refuses it.

What travels with the checkout is the code. What does not, and must already be
on the box named:

| | sparky | sparklina (`gx10-6b77`) |
|---|---|---|
| pinned serve image by digest (`61fc8a89…`) | yes | yes, as `vllm/vllm-openai:latest`'s `RepoDigest` |
| `/home/rob/dq-runs/venvs/prismaquant-cu130` | yes | yes |
| `$SRC/{armA,armB}`, one inode | yes | **no** -- 846 MB to copy, then `cp -al armA armB` |
| `$SRC/ext-A` | prebuilt | absent, and it does not matter |

`ext-A` does not matter because `kernel_window_gemv._ext()` builds at load
(`kernel_window_gemv.py:485`: "built (or found) at load, never on the first
call"), so an nvcc build on a cold box lands inside the 40-minute startup wait
and not inside a timed window. That is also what makes arm B's mechanism work
at all, so the two facts are the same fact.

`--exclusive` rather than `--gpu` is the point: a latency A/B needs the whole
box, not one of two slots, and the ledger turns a full-capacity demand into
exclusion. `SETTLE_S` holds the claim doing nothing so the idle window that
follows describes a box with nothing on it -- which is not queueing, because
that reading is itself one of #109's deliverables. `--timeout-s` is now 10800:
two eager reps ran 06:53Z -> 09:34Z with the *fixed* parser in the second half,
and 7200 was already a near miss.

Then `compiled` as a second pass. Eager first deliberately: the eager census
carries the `tessera_window_gemv::gemv` symbol at M1 shapes directly, so the
lane's engagement in the eager arm needs no trace to establish, while a compiled
record stamps a combined pair and proves dispatch rather than launch.

## 6. What the served ratio is made of, measured before it is taken

The 2026-09-03 document flagged a cuBLAS bf16 GEMV in armA's compiled trace --
263.9 ms over 50 launches, more device time than the entire window-GEMV bucket
-- and did not attribute it. It is attributed now, and it changes how the ratio
must be read.

**It is `lm_head`, and this is read off the trace rather than inferred from
shapes.** Each of the 50 `internal::gemvx::kernel<..., __nv_bfloat16, ...>`
kernels carries a correlation id; each id resolves to one `cudaLaunchKernel`;
each launch resolves to an enclosing `aten::mm` (50 of 50); and every one of
those 50 sits inside a `vllm/model_executor/layers/logits_processor.py(132):
_apply_head` frame, under `LogitsProcessor_0` and
`vllm/model_executor/models/qwen3.py(330): compute_logits`. 50 launches, 50
`_apply_head` frames. The arm model is Qwen3-0.6B: `vocab_size` 151936 against
`hidden_size` 1024, so the output projection is by a wide margin the largest
single matmul in a decode step and it is in bf16.

Segmenting that trace into engine steps (`_process_engine_step`, 62 of them)
and classifying each by which route it launched gives the decode step's real
composition:

| decode steps (46) | device ms | share |
|---|---|---|
| `lm_head` cuBLAS GEMV | 242.0 | **48.1%** |
| the lane (`window_gemv_kernel`) | 149.9 | **29.8%** |
| everything else | 111.2 | 22.1% |
| **total** | **503.1** | |

(The 4 prefill steps are a different mix -- 583.9 ms, of which 237.7 ms is
`window_decode_kernel` materialising for `_scaled_mm` and only 21.9 ms is
`lm_head` -- which is the same fact from the other side: `GEMV_MAX_M = 8`
refuses prefill, so the lane's *decode* kernel does not appear there at all.)

### The exchange rate, and the ceiling this is *not*

Write a decode step as `T = R + L`: `R = 353.2 ms` of work the lane cannot
touch (`lm_head`'s 242.0 ms and 111.2 ms of everything else) and `L` the lane
bucket, `L_e = 149.9 ms` in the engaged arm. The A/B is
`T_f/T_e = (R + L_f)/(R + L_e)`, and **it has no upper bound**: nothing stops
the fallback's `L_f` from being arbitrarily large. An earlier draft of this
section called `1/(1 - s_e) = T_e/R = 1.424x` "the ceiling the ratio can
reach", which is wrong and inverted -- that quantity is the *engaged* arm's own
remaining headroom, what armA would gain if its lane cost nothing, and it
bounds nothing about armB. The A/B's actual Amdahl ceiling is `1/(1 - s_f)`,
read off the **fallback** arm's lane share, and **no fallback decode trace
exists**: the 2026-09-03 latency receipts carry
`profiled_load: {"error": "HTTPError: HTTP Error 404: Not Found"}`. It will
exist when the pair runs, and the tool reports it from that arm.

What the engaged trace *does* give, and it is the useful thing, is the exchange
rate between the two numbers #109 asks about. With `s_e = 0.2979`, a served
ratio `X` implies a lane-bucket speedup of `k = 1 + (X - 1)/s_e`:

| served `X` | implied lane `k` | fallback step | the A/B ceiling that `k` implies |
|---|---|---|---|
| 1.15x | 1.50x | 578.6 ms | 1.638x |
| 1.30x | 2.01x | 654.0 ms | 1.852x |
| 1.50x | 2.68x | 754.7 ms | 2.137x |
| **8.024x** (the 09-03 pair) | **24.6x** | 4036.9 ms | 11.43x |

The last row is §3's control restated as arithmetic instead of as a
null-window observation, and the two agree: for that pair to be a lane result
the torch fallback would have to be **24.6x** slower than a hand-written CUDA
kernel on the same work. It is not; the box moved.

The first row is the one to carry into reading the real ratio. A served 1.15x
is not a disappointing kernel -- it is a kernel that beat the torch fallback
**1.50x** on the work it owns, on a step where `lm_head` alone is 48.1%. The
honest framing is that **Qwen3-0.6B is a poor ruler for this lane**: a
151936-row `lm_head` against 28 layers of 1024-wide Linears puts nearly half
the decode step in one bf16 kernel the lane will never touch. The ratio taken
on these arms is still the right thing to take -- it is the number #109 asks
for, on the arms whose KL and census are already published, and a lane that
cannot move a real serve is not worth its complexity -- but it prices the lane
*as deployed on this model*, and the per-bucket device time is what transfers
to a model whose `lm_head` is a smaller share. That is why
`window_gemv_latency_ratio.py` reports two things and calls neither the other's
headline: the served ratio, and the device time inside the profiled decode
window, cut from both arms' traces by one wall-clock rule. The cuBLAS GEMV now
has a bucket of its own in the shared summariser rather than sitting in
`other`, which is where a dilution hides: on this trace `other` falls from
297.3 ms to 33.4 ms once it is named.

### The lane cannot be compared across the arms by kernel name

One asymmetry in the profile half is worth stating before a receipt shows it.
The engaged arm's lane is one named CUDA kernel; the fallback's window decode
is **pure torch over packed bits** and lands in `at::native::*` elementwise and
index kernels that no lane pattern matches. Summing a `window_gemv|
window_decode|scaled_mm` name bucket therefore counts the *whole* engaged lane
against only the fallback's `_scaled_mm` -- flattering the built kernel, and
understating `s_f`, which is the share the A/B's ceiling is read from.

So the tool also attributes by **enclosing Python frame**: was this kernel
launched from inside `tessera/serving/fp8_gemv.py`? That is the same question
in both arms. On the #83 engaged trace it finds 542.65 ms under those frames
against 387.6 ms by name -- the difference being the torch work inside
`_materialised_path` and `_role_view` that the name buckets scatter into
`elementwise/triton`.

**And it is eager-only, measured rather than assumed.** On that same *compiled*
trace, none of the 542.65 ms is `window_gemv_kernel`: the matched frames are
`streamed_apply`, `_materialised_path` and `_role_view`, while the GEMV route
is traced into the inductor graph and launched from generated code with no
`fp8_gemv.py` frame above it. Under a compiled forward the number is a floor on
the lane, not the lane, and the receipt says so in a `note`. This is a second
reason the campaign runs eager first, beyond the one in §5.

This section was computed from `prof-armA-streamed-compiled/rank0.*.pt.trace.json.gz`
-- the #83 campaign's own compiled trace, which is whole-serve rather than cut
to a decode window, hence the step-classification above. The 2026-09-03 *latency*
receipts carry `profiled_load: {"error": "HTTPError: HTTP Error 404: Not Found"}`
and no trace of their own; the `/2` receipts written by the new driver carry
`profile_decode_start`/`profile_decode_end` marks so the same cut can be made
directly and identically in both arms.

## 7. Three things found while asking where this can run (2026-09-04, later)

### 7.1 The box-side instrument wrote nothing for the entire campaign, and the driver threw the reason away

Principle 15 asks for two instruments and names why: the in-process profiler
says where time went, the box series says whether the box was loaded, and no
in-process tool can see the second. The 2026-09-04 campaign produced four
latency receipts and four chrome traces and **zero `power-arm*.json`**. Not one
of its eight timed windows has a box-side reading.

The cause is one line. `box_power_window.py` split `--window` on the **first**
colon, while `window_gemv_latency_ab.sh` passes each arm's own marks:

```
--window=2026-09-04T09:16:12Z:2026-09-04T09:17:26Z
```

An ISO-8601 stamp carries two colons of its own, so the first split yields
`2026-09-04T09` and `16:12Z:2026-09-04T09:17:26Z`, neither of which is an
instant. The tool refused **its own docstring's usage line** (line 32) and
exited 1 -- and the driver ran it under `subprocess.run(..., check=False)` and
said nothing. A gate whose refusal nothing reads is the confession log
principle 9 names, and this one refused every arm of a two-rep campaign in
silence.

Both halves are fixed here. The separator is now *found* rather than assumed --
every colon is tried and the one where both halves parse is the separator, with
an ambiguous window refused rather than guessed at, because a box-side window
silently off by a minute describes a different box state than the arm was timed
in. And the driver now prints `!!! NO BOX-SIDE POWER RECEIPT` the way it already
prints `!!! NO TRACE`. `tests/test_box_power_window.py` (6 passed) pins the
driver's exact argument, the unmoved `-1800:0` form used by the idle windows,
the two refusals, and the first-colon split itself as the regression that must
not come back.

**The four windows are recoverable, and were recovered**, because Netdata's
10 s tier still holds the morning. Re-read with the fixed tool:

| arm | timed window | GPU power median | of the 140 W envelope | swap-in median |
|---|---|---|---|---|
| armA rep 1 | 55 s | 14.0 W | 10.0% | 0.36 KiB/s |
| armB rep 1 | 439 s | 43.0 W | 30.7% | 0.40 KiB/s |
| armB rep 2 | 445 s | 43.0 W | 30.7% | 2.40 KiB/s |
| armA rep 2 | 74 s | 72.1 W | 51.5% | 17.82 KiB/s |

Receipts: `/home/rob/tessera-runs/ts109/power-arm{A,B}-streamed-eager-rep{1,2}.json`,
labelled `*-decode-window-recovered` so nobody mistakes them for readings the
campaign took at the time.

Read them for what they can carry. Box power is the **whole box**, this arm's
own serve included, so no row here is the lane's power. What the table does
separate is the *same arm against itself*: armA's decode window is 55 s at
14 W in rep 1 and 74 s at 72.1 W in rep 2 -- identical work, 1.35x the wall
clock and 5x the box power -- while armB's is 439 s against 445 s, 1.4% apart.
That is the box moving under the campaign, seen from outside, and it agrees
with `max_load1_per_cpu` 0.059 against 0.463 seen from inside. It also says
which arm the contention lands on: the engaged arm is short and launch-bound,
so host load inflates it, which pushes the fallback-over-engaged ratio
**down**. A contended pair understates this lane rather than flattering it.

### 7.2 The box is a property of the checkout's path, so the quiet box is reachable without cloning to it

Covered in §5, recorded here as the answer to the question that was asked:
**no, the branch does not need to be cloned onto sparklina.** One `git clone`
under `/mnt/shared` plus `--tag gx10-6b77` places the same action on either
GB10; only the arms (846 MB, then `cp -al`) have to be staged.

**But sparklina could not take this action at 2026-09-04T10:30Z, for a reason
that reads as "stuck" and is not.** Its reservation ledger holds exactly **one**
GPU token, and `held/out-of-pool-ts60-encode-sparklina/gpu-0000` has held it
since 2026-09-03T20:10 local (pid 2003068, `tessera_window_wire.py --grids
E2M1x2 … glm_v2_b4.json`, ~2 h in at the time of reading). The offer file says
`capacity.gpu: 2`, and `--exclusive` reads *that* number, not
`observed_capacity.gpu: 1`. `pool.claim` skips an item whose demand exceeds the
ledger's total for a kind **without recording a pass** -- "never fits this box;
not this box's to hold" -- so an `--exclusive` item tagged `gx10-6b77` sits in
`ready` at **0 passes** and does not age until that encode exits and
`ensure_capacity` mints `gpu-0001`. Zero passes there is the queue being
correct, not the queue being broken.

### 7.3 A loaded box becomes unaddressable, which is why the submit script retries

`pbrun` refuses a submission outright, before queueing anything, when no *live*
offer carries the tag it names; an offer is live for `OFFER_TIMEOUT_S = 120`
seconds. Sampled every 12 s from 10:24:48Z, sparky's offer age climbed
monotonically to **260.5 s** before resetting to 3.6 s -- one announce every
~264 s at `load1 19.56`. So sparky publishes a stale offer for roughly 55% of
each cycle, and a submission naming it during that stretch is told "no live
worker can run this action" while four of its own actions are running.

This is a second, independent cause of the refusal §4 attributed to the
two-runtime shape flicker, and it bites hardest exactly when a box is busy
enough to be worth queueing behind. `experiments/ts109_submit.sh` retries the
submission for both; neither is routed around, and a refusal that survives the
whole loop is reported rather than worked past.

## 8. Rep 1's ratio, computed at last -- and what arm B actually serves

The 2026-09-04 campaign's rep 1 never had a ratio: its parse was the
`lane_by_frames` scan that spun for 55 minutes and was killed, so only rep 2's
receipt was written. Computed now on the same bytes with the indexed parser
(`ratio-streamed-eager-rep1.json`), it is the first pair in this issue's
history that has a **fallback trace at all** -- the 2026-09-03 receipts carry
`profiled_load: {"error": "HTTPError: HTTP Error 404"}` -- and that trace
withdraws the argument §3 built on the prefill row.

### The two reps, side by side

Fallback over engaged, from the engine's own histograms in both reps (the
`series_moved` recovery §3 describes is no longer needed; both receipts carry
`source: histogram/histogram`):

| | rep 1 (A then B) | rep 2 (B then A) |
|---|---|---|
| `decode.tpot` / `decode.itl` | 34.522 -> 282.381 ms, **8.180x** | 44.786 -> 286.369 ms, **6.394x** |
| `decode.ttft` | 67.737 -> 439.528 ms, 6.489x | 245.997 -> 442.605 ms, 1.799x |
| `prefill.ttft` | 69.783 -> 272.597 ms, 3.906x | 248.423 -> 275.580 ms, 1.109x |
| `max_load1_per_cpu`, engaged / fallback | 0.059 / 0.151 | 0.463 / 0.525 |
| box GPU power over the decode window (§7.1) | 14.0 W / 43.0 W | 72.1 W / 43.0 W |

**The fallback arm barely moves between reps and the engaged arm moves a lot.**
Fallback: 282.4 vs 286.4, 439.5 vs 442.6, 272.6 vs 275.6 -- every row within
1.4%. Engaged: 34.5 vs 44.8, 67.7 vs 246.0, 69.8 vs 248.4. The reason is in the
traces below: for the same 192 decode steps the fallback spends 50.6 s of
device time and the engaged arm spends 1.21 s, so the fallback is GPU-bound and
indifferent to a busy host while the engaged arm is neither. Contention lands
on the engaged arm, which is why the crossover exists and why rep 2's numbers
are the ones to distrust.

### Arm B does not serve the route arm A replaces; it serves no CUDA extension at all

§6 called the B side "kernel-decode the window into a tile, then
`torch._scaled_mm`", and §3 read the prefill row as a null window that "should
read 1.00x" because `GEMV_MAX_M = 8` refuses prefill in *both* arms. Both
statements are wrong about the arm that was built, and the fallback trace is
what says so.

`fp8_gemv._materialised_path` opens with `ext = kg._ext()` and decodes the tile
with `ext.window_decode(...)`. Arm B is made by pointing
`TORCH_EXTENSIONS_DIR` at a read-only mount so that `_ext()` raises -- so
`_materialised_path` cannot run either. What the route does instead it says
itself: *"the window GEMV lane did not prepare (…); serving streamed through
the torch window decode instead"* (`bf16_route.py:775`, `fp8_route.py:320`).

The traces agree exactly. Both cuts contain the same 192 decode steps -- 192
`cublas_gemv` (`lm_head`) launches, 10,752 attention, 21,504 `fp8_quant` in
each -- so this is step for step:

| bucket | engaged (armA rep 1) | fallback (armB rep 1) |
|---|---|---|
| `window_gemv` | 37,044 launches, 369.3 ms | -- |
| `window_decode` | 588 launches, 83.8 ms | **none** |
| `scaled_mm/cutlass` | 336 launches, 4.6 ms | 21,504 launches, 322.2 ms |
| `elementwise/triton` | 124,620 launches, 216.9 ms | **453,312 launches, 49,350.7 ms** |
| device ms in the cut | **1,211.7** | **50,569.0** |
| lane, by enclosing frame | 684.6 ms (56.5%) | 49,347.2 ms (97.6%) |

Arm B runs `_scaled_mm` -- 21,504 launches, one per unit per step, against the
engaged arm's 336 -- and has **no `window_decode` kernel at all**. The 453,312
elementwise launches are the torch window decode standing in for it.

So the prefill row is not a null window. In prefill both arms take the
materialised path, but only arm A has a CUDA kernel to materialise with, so
3.906x on the quiet rep is the **window-decode kernel's** own A/B at prefill
shapes -- a measurement, not a symptom. "A pair whose null arm moves by 4.3x
cannot price a lane at 8x" was reasoning from a control that was never null.
The box *did* move -- §7.1's power table and rep 2's engaged arm say so -- but
the prefill row is not the evidence for it, and the 2026-09-03 pair's 4.29x is
explained without invoking the box at all.

### The engaged arm is host-bound in eager, and that caps the served ratio

Per decode step, engaged: **6.31 ms of device time inside a 34.52 ms step**, so
18.3% device-busy. Fallback: 263.4 ms of device inside a 282.4 ms step, 93.3%
device-busy. The lane difference on the device is 41.7x per step; the served
ratio is 8.18x. The gap is 28 ms per step of host time in the engaged arm that
the lane cannot remove and the fallback's own slowness hides. That is a bound
on what any eager served ratio can show here, and it is the strongest argument
yet for taking the `compiled` pass §5 defers: a launch-bound arm is what CUDA
graphs are for (principle 10).

### What this means for what #109 asks

#109 wants the GEMV "against the route it replaces". As built, the pair prices
**the whole Tessera CUDA extension against no extension** -- the GEMV *and* the
CUDA window decode, against a torch decode. Two readings are open and the
choice is the owner's, not this document's:

* If "the route it replaces" is the streamed route as it shipped before the
  kernel lane -- pure torch over packed bits, which is what
  `tessera-lane-two-families` records it as -- then this pair is the right
  comparison, and 8.180x on the quieter rep is the answer, stated as pricing
  two kernels rather than one.
* If it is the materialised path `fp8_gemv.py`'s docstring names -- CUDA
  `window_decode` then `_scaled_mm` -- then arm B has to keep the extension and
  refuse only the GEMV. **There is no knob for that today**: `GEMV_MAX_M = 8`
  is a module constant (`kernel_window.py:78`) with no env spelling. Giving it
  one would be a *process*-level fact fixed at import, the same shape as
  `TORCH_EXTENSIONS_DIR`, so it would not fight `flags.py`'s in-run latch --
  but it is a serving-path edit made for a measurement and wants an owner's
  decision before anyone makes it.

Whichever it is, §6's exchange rate has to be re-read against these arms: the
engaged lane is 37.8% of the decode window by kernel-name bucket and 56.5% by
enclosing frame, and the fallback's is 97.6%, which puts this pair's Amdahl
ceiling at 1/(1 - 0.976) = 41x rather than anything near 1.4x.

### And the pair is still refused, on the one clause that has not moved

`ratio-streamed-eager-rep1.json` ends `VERDICT: NOT EVIDENCE about the lane`.
Its four reasons are resident swap (2.7 GiB on both arms, above
`window_gemv_load.py`'s 1 GiB threshold) and a non-zero page-in delta during
the timed windows (**10 pages** on the engaged arm, **224** on the fallback,
0.04 and 0.88 MiB). Load was 0.059 and 0.151 per core. So the clause that
refuses rep 1 is not the clause the campaign's own verdict quoted -- that one
quoted rep 2's 0.463 and 0.525 -- and on rep 1 it is a residue from an earlier
stampede plus a handful of pages.

That threshold is **deliberately not relaxed here.** §2 already names why:
loosening the rule that decides whether one's own receipt is admissible is the
self-serving edit, and the branch that wrote the gate declined to make it. What
is added is the reading a decision needs: rep 1 fails on resident swap and 234
pages, not on load, and the two reps' ratios differ by 28% (8.180 vs 6.394)
which is larger than any effect those pages could have. The number to take is
still the one from a box that passes the gate on its own terms.
