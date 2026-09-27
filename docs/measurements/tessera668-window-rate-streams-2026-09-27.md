# Window rate streams: tessera#668

Status: **measured on GB10; bit-exact.** At a mixed-rate window rung, the batched
LDLQ encode now runs each block's rate calls on separate CUDA streams. On the
PACT G2 w01b shapes (GLM-5.3 routed experts, BF16_K1 at q256 1088 and 1152, 16
units a batch), a batch encodes 18.5-22.4 % faster, with 1.18-1.23x the units per
joule. All 18 byte comparisons against `master` match: every unit's wire blob and
every `EncodedUnit` field. The GPU-gated test receipts are in the pull request
(tessera#669).

Date: 2026-09-27. Before: `master` `a3e83875d`. After: `64cc379a6`, on that
`master`. Box: GB10 / DGX Spark `sparklina`, NVIDIA driver 595.91.07, 140 W
envelope. Image: `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`
(torch 2.13.0+cu130). The PrismaBuild action is `ec986f8c09eeda27`: GPU-exclusive,
2 CPUs, 40 GB container cap, rc 0 in 2414 s, peak RSS 5.3 GB.

## What changed

`trellis_pass` used to yield one window Viterbi call per rate and wait for that
call's answer before it built the next. At BF16_K1 q1088 and q1152, every
32-column LDLQ block holds R4 and R5 columns, so each block ran its R4 and R5
Viterbi back to back on the caller's stream.

- **Rate tuple.** A window span yields all of its rate calls at once, as a tuple.
- **Rate streams.** `_drive_in_step` joins each rate across the batch's units.
  `_run_group` runs the rates on per-thread side streams that fork from the
  caller's stream and join back to it; `record_stream` orders the caching
  allocator's reuse.
- **Serial control.** `TESSERA_WINDOW_RATE_STREAMS=0` runs the group on the
  caller's stream, in rate order.
- **Index hoist.** The window table's `.long()` index view is computed once per
  unit instead of once per call.

No byte moves by construction. The same tensors reach the same persistent plans
(the plan key carries the rate) and the same kernels. Only the stream each call
is enqueued on changes. The residual matmul and its layout are untouched, so
cuBLAS picks the same algorithm.

## Method

The harness (`harness/bench.py`, commit `4a9391f`, stored with the receipts)
generates synthetic inputs once: Gaussian weights and correlated Hessians at the
two w01b expert shapes, gate/up `[2048, 4096]` and down `[4096, 2048]`. Every
arm loads the same bytes. Each arm is one container process on one source tree,
and it encodes through `export.encode_linears_planes` with the exporter
defaults: the window body over the CHANNEL plane, LDLQ at block 32,
`scale_refit=4`, `trellis_weighting="scale"` and `verify=False`. The Tessera
environment matches the G2 row actions: `TESSERA_WINDOW_BEST_FORM=1`,
`TESSERA_WINDOW_BEST_TILE=64,4,2`, `PYTORCH_ALLOC_CONF=expandable_segments:True`
and one native thread.

The arms interleave in three rounds: A (master) and B (the fix) at q1088, A and
B at q1152, then C (the fix with `TESSERA_WINDOW_RATE_STREAMS=0`) at q1088.
Each arm times one warm, sync-bracketed batch per shape and hashes two encodes.
Power comes from an in-process NVML sampler at 10 Hz. The first round adds a
`torch.profiler` window of 20 block steps and a replay of the joined Viterbi
calls alone, serial and on streams.

## Results

Wall seconds per 16-unit batch, over three rounds, as mean (min-max):

| Shape at rung | A: master | B: fix | C: fix, serial | B vs A | B units/kJ vs A |
|---|---|---|---|---|---|
| gate@1088 | 31.98 (31.95-32.01) | 25.32 (25.25-25.40) | 31.82 (31.81-31.83) | -20.8 % | 1.23x |
| down@1088 | 30.44 (30.34-30.49) | 23.61 (23.58-23.63) | 30.16 (30.13-30.20) | -22.4 % | 1.23x |
| gate@1152 | 33.61 (33.59-33.61) | 27.37 (27.33-27.42) | -- | -18.5 % | 1.18x |
| down@1152 | 31.94 (31.88-31.98) | 25.66 (25.62-25.70) | -- | -19.7 % | 1.21x |

- **The ranges never overlap.** Each B range lies below its A range by more than
  6 seconds; the widest spread in any cell is 0.15 seconds.
- **The streams carry the whole gain.** C is within 0.5-0.9 % of A, so the index
  hoist alone is worth under 1 %.
- **The Viterbi replay agrees.** On the first round's captured inputs, the
  joined Viterbi calls alone take 29.39 s serial and 22.54 s on streams at gate,
  and 27.42 s and 20.81 s at down. That's 6.85 s and 6.61 s saved, the size of
  the full-encode saving.
- **Power.** GPU power rises from 79-80 W (56-57 % of the envelope) to 82 W
  (59 %). Work per joule improves because the same energy per second buys more
  of the batch.

### Where the time goes

The master profile (gate@1088, 20 block steps) names the regime:

| Measure | A: master | B: fix |
|---|---|---|
| GPU busy fraction of the span | 99.0 % | 99.7 % |
| Gap fraction | 1.0 % | 0.27 % |
| Kernels per block step (Viterbi graph nodes) | 4971 (4473) | 4918 (4462) |
| Median kernel, all / Viterbi | 11.1 / 11.2 us | 13.8 / 14.7 us |
| Span per block step | 66.9 ms | 51.2 ms |
| Viterbi kernel time, summed | 1.14 s | 1.38 s |
| `cudaLaunchKernel`, calls / host time | 8680 / 0.97 s | 8040 / 0.69 s |

- **The device is busy; the host waits for it.** `_step_best` is 82 % of GPU
  time. The host spends about 112 us per `cudaLaunchKernel` because the launch
  queue is full, which is why the process reads 1.00 CPU seconds per wall
  second.
- **The chain is serial and the kernels are small.** A window call replays its
  L2-width tiles one after another, and each tile is a chain of one-step
  kernels of about 11 us. So the GPU is busy all the time but draws 57 % of the
  envelope.
- **Overlap lengthens each kernel and shortens the span.** Under the fix, the
  summed Viterbi kernel time rises by 21 % and the median kernel lengthens, yet
  the span per block step falls 24 %. Two rate chains now share the SMs; the
  extra summed time is overlap, not extra work. down@1088 shows the same shape:
  121.1 ms to 94.6 ms per block step.

### Clocks and the envelope

Under the fix the mean SM clock falls from about 2400 MHz to about 2260 MHz, at
82 W. A 10-second `nvidia-smi` sampler on the box read the SW power cap reason as
active in 14 of 156 samples, each at 80-85 W. Its cumulative SW-power-cap
counter did not advance over the run, so the two readings disagree and the cause
of the clock drop is not established. For planning, treat it as a ceiling: the
next concurrency lever, running a call's independent tiles at once, will likely
return less than the 41 % envelope gap suggests.

Netdata samples GPU power every 10 seconds, so it cannot resolve these arms,
which last 23-34 seconds each. The per-arm power above is the 10 Hz sampler's.
Netdata is the box-level view: the box sat at 10-11 % CPU for the whole run (the
2-CPU container and background), with no other GPU tenant, and GPU power
peaked at 95 W.

## What it means for a G2 row

The A arms match the production w01b anchor batches (row-0051, `encoding_batch_size`
16) within 2-4 %: 31.98 s against 32.9 s at gate@1088, 30.44 s against 31.2 s at
down@1088, 33.61 s against 34.6 s at gate@1152, and 31.94 s against 33.1 s at
down@1152. Weighting gate and up at 2/3 and down at 1/3 of each rung's
batches, the fix saves 21.4 % of the q1088 encode (about 376 of 1759.5 s) and
18.9 % of the q1152 encode (about 349 of 1844.3 s). That's about 725 of 3604 s per
row, or a 61-minute row in about 49 minutes. This is a **projection** from
per-batch measurements. It becomes a result when the first re-planned row runs
on this code.

## Limits

- **Synthetic inputs.** The weights and Hessians are generated, not GLM
  calibration data. They set the shapes, rates and LDLQ schedule, which fix the
  kernel sequence; the calibration against row-0051 above is the check.
- **Memory.** `record_stream` defers a block's return to the pool until the
  side stream's work finishes, and two rate plans' scratch is live at once.
  Peak CUDA memory reserved per process rises by 0.06-0.37 GB under the fix:
  10.34 against 10.28 GB at gate@1088, 12.12 against 11.83 GB at down@1088,
  10.26 against 10.12 GB at gate@1152, and 11.79 against 11.42 GB at down@1152
  (rounds 2 and 3; round 1's A arm also ran the profiler and the replay). Each
  arm encodes two or three batches; a G2 row runs 108, and memory over a full
  row is not measured.
- **One box.** The numbers are from `sparklina`. Sparky runs the same part and
  driver, and was not measured.

## Receipts

`/mnt/shared/tessera-measurements/claude-ldlq-perf-20260927/abab-20260927T182931Z-sparklina/`:
`result-{A,B,C}{1,2,3}-{1088,1152}.json` (per-arm timing, power, identity hashes
and the first round's profiles), `traces/` (the `torch.profiler` exports),
`analysis.txt`, `capwatch.csv` (the clock-event sampler), `netdata-arms.txt`,
`pbrun-client.log` and `harness/`.
