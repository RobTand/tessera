# The window-GEMV decode-path latency A/B: the pair, and the box it needs (2026-09-04)

> **STATUS: the instrument is built and verified against the 2026-09-03 pair;
> the ratio itself is NOT taken.** Issue #109 asks for one number -- the
> window-GEMV kernel's decode-path latency against the route it replaces, as a
> ratio, on an otherwise idle box. This document does not contain that number.
> What it contains is (a) the pair harness master did not have, (b) the
> box-state readings that decided against taking the measurement tonight, and
> (c) a re-reading of the 2026-09-03 arms that produces, for the first time,
> a *measured* control for why their 8x is not a lane result. Nothing here is a
> placeholder standing in for a number that was taken and disliked.

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

Three of those deserve their reason stated.

**The quiet-box gate is a measurement, not an intention.** #109 names #5's
failure directly: that report had to give a 5.0-9.94 s/unit *bracket* rather
than a number because Netdata showed 63-88 W on sparky for the seven minutes
before the process existed. So the chain reads the idle window off Netdata,
writes it to a receipt, and **refuses to run** when the box is not idle by that
reading (GPU peak over 30 W, load1 over half a runnable process per core, or
any page-out). `TESSERA_LAT_REQUIRE_QUIET=0` records the reading and runs
anyway; the refusal is the default because the alternative is a number nobody
can use.

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

## 5. What remains

The A/B itself, which is one command once a box is genuinely free:

```
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --exclusive --here --cpus 10 --demand mem_gb=48 --timeout-s 7200 \
  --cwd <worktree> --tag sparky \
  --env TESSERA_LAT_OTHER_BOXES=sparklina --env SETTLE_S=420 --env IDLE_S=360 \
  -- bash experiments/window_gemv_latency_ab.sh eager 2
```

`--exclusive` rather than `--gpu` is the point: a latency A/B needs the whole
box, not one of sparky's two slots, and the ledger turns a full-capacity demand
into exclusion. `SETTLE_S` holds the claim doing nothing so the idle window that
follows describes a box with nothing on it -- which is not queueing, because
that reading is itself one of #109's deliverables.

Then `compiled` as a second pass. Eager first deliberately: the eager census
carries the `tessera_window_gemv::gemv` symbol at M1 shapes directly, so the
lane's engagement in the eager arm needs no trace to establish, while a compiled
record stamps a combined pair and proves dispatch rather than launch.

**What is still unknown after the ratio lands**, and worth saying before it does
so it is not discovered as a disappointment: the served TPOT ratio prices a
whole decode step, and the lane owns only the Tessera Linears in it. armA's
compiled trace has a cuBLAS bf16 GEMV taking 263.9 ms over 50 launches -- more
device time than the entire window-GEMV bucket -- which the 2026-09-03 document
flagged and did not attribute. If that kernel is on the decode path, an
end-to-end TPOT ratio is diluted by work the lane does not touch and will sit
closer to 1 than the kernel does however good the kernel is. That is why
`window_gemv_latency_ratio.py` reports two things and calls neither the other's
headline: the served ratio, and the device time inside the profiled decode
window by bucket, cut from both arms' traces by one wall-clock rule.
