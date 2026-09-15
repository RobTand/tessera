# Fused coset TCQ trellis: stage 1 of #486

Status: **stage 1 complete; #486 stays open.** The coset TCQ trellis now runs as three
Triton launches a call and returns the reference's bytes. On the same binary the E2M1_K2
encoder is 1.96x faster at `B=32` and 3.07x faster at `B=8`. That is 1.82x and 2.47x the
parameters per joule. It still draws only 35-37% of the 140 W envelope, because the encoder
is now bound by the host loop in `_fit_lut`. That loop is stage 2.

Date: 2026-09-14 (the last receipt, the doc tests, finished 2026-09-15 00:49). Branch: `claude/486-fused-tcq`
at `ce87ceb13`, on `master` `afb71840c`. Boxes: GB10 / DGX Spark (`sparky`, `sparklina`),
140 W envelope, torch 2.11.0+cu130.

## What changed

`src/tessera/tcq_fused.py` runs `_TCQPlan.run`'s recurrence as three Triton kernels:

- `_minima` takes the best point per subset for every step, column and subset in one
  launch. It depends only on the targets and the tables, so it runs before the recurrence.
- `_forward` runs the fold and the state recurrence, one column per lane, with the
  super-steps looped inside the program.
- `_traceback` walks the states backwards, one column per lane.

`encode.viterbi_columns` gains `impl="fused"`. `auto` takes it on a CUDA device whenever
`fused_available()` holds and `tcq_fused_refusal` names no reason. That means float32,
float16 or bfloat16 targets and weights, at arity 1 or 2.

- Every other call keeps the captured-graph rule.
- `TESSERA_TCQ_FUSED=0` restores that rule for every call.
- An explicit `impl="fused"` raises on a call the fused path refuses.
- `impl="reference"` is unchanged.

The contract is identity with `_TCQPlan.run`: the same anchors, the same body field and the
same `sse` float, exact ties included. The kernels get there by construction:

- **No fused multiply-add.** Every launch passes `enable_fp_fusion=False`, and every multiply
  is the inline-asm `mul.f32` in `_mul`. On Blackwell the packed NVPTX multiply-add pair is
  contracted into an FMA even with fusion off. The bite demo below shows that on this box.
- **First index on ties.** Every minimum that returns an index is a strict-`<` scan in index
  order from an `inf` seed. That is the first minimal index, which `torch.min(dim)` returns.
- **The reference's association.** The squared distance is associated as the reference
  associates it. The sum over coordinates has one IEEE spelling at arity 1 and 2, which is
  why those are the only admitted arities.
- **The reference's epilogue.** The final `argmin`, the gather and the `sum` stay in torch,
  on a front of the reference's shape. So the `sse` is summed in the reference's order.

## Identity

### Tests

`tests/test_tcq_fused.py` asserts fused == reference, bitwise, over these axes:

- Rate, completion depth, span and the weighted branch metric.
- Every column width.
- Every admitted dtype.
- The campaign's call shape, against both the reference and the graph.
- Exact ties.
- Every launch and tile knob.
- Empty calls.
- The `auto` policy and `TESSERA_TCQ_FUSED=0`.
- Refusals by name.
- Concurrent calls.
- The torch primitives the kernels reproduce.
- No kernel spill on the shipping shapes.

The full suite ran on the rebased head `434426432` (stage 1 on `a5ee7c166`): 12 shards through
`pbtest.py`, `--tag gb10`, priority 1.

| Passed | Skipped | xfailed | Failed |
|---:|---:|---:|---:|
| 5,362 | 16 | 1 | 1 |

- **The one failure is pre-existing.** It is
  `tests/test_slice_unit.py::test_the_span2_kernel_lane_refuses_a_shard`, which fails the same
  way on `master` `44d20d670` (PB `39cac96ecf18`). This gate excludes it.
- **Where the trellis tests ran.** `tests/test_tcq_fused.py` ran in shard 2, which reported
  523 passed, 1 skipped and 1 xfailed. `tests/test_tcq_graph.py` ran in shard 3, which
  reported 630 passed, 5 skipped and 1 failed (the span2 test).
- **rc=74 shards.** Five shards (1, 5, 7, 9 and 11) ended with client rc=74, "unavailable pool
  outcome". Every one of those actions ran to completion, and its counts come from the
  attempt's stdout record.
- **Collection.** No module failed to collect.
- **The doc tests.** The 11 test files that read `docs/` ran again on this change's own
  documentation, `ce87ceb13` plus this file and the `docs/ARCHITECTURE.md` edit, in 3 shards on
  the x86 CPU lane (`--tag dl380g10`). They reported 284 passed, 0 skipped and 0 modules not
  collected (151, 109 and 24 by shard). Two of the three shards ended with client rc=74, and
  both of their actions are in `done/` with those counts.

The 16 skips, by reason (long reasons shortened; the shard records carry them in full):

| Count | Reason |
|---:|---|
| 4 | could not import 'vllm': No module named 'vllm' |
| 2 | box artifact absent: the PrismaQuant worktree carrying the continuous-rate branch (`/home/rob/pq-wt/tessera-continuous`) |
| 2 | box artifact absent: kl_tool.py and kl_estimator.py, the untracked served-KL instrument (`KL_TOOL_DIR`) |
| 2 | e2m1-tcq-lut-release does not cut 4 ways along columns |
| 2 | e2m1-tcq-lut-release does not cut 8 ways along columns |
| 2 | needs two CUDA devices |
| 1 | the fp4 activation quantizer is vLLM's operator |
| 1 | E2M1 publishes no reader range |

### Bite demo: mutate the driver

Each mutation changes one token of the kernel. Each ran all 204 tests of
`tests/test_tcq_fused.py` in its own action, on stage 1 before its commit (`44d20d670` plus
the change).

| Mutation | Kernel | Failed of 204 | PB |
|---|---|---:|---|
| `take = e < bst` to `<=` | `_minima`, point scan | 55 | `e0b24750e807` |
| `tk = t < cur` to `<=` | `_forward`, fold scan | 72 | `0f819416f05b` |
| `take = c1 < c0` to `<=` | `_forward`, branch choice | 13 | `9c75c5ab573a` |
| `_mul` to a plain `a * b` | every multiply | 2 | `6a6911e8f9c6` |

The last row is why `_mul` is inline asm. With a plain `a * b`, the anchors and the body field
still match, but two `sse` floats come out exactly one float32 ulp from the reference:

- 1129.551025390625 against 1129.5511474609375 (E2M1x2, rate 2, completion 2, span 2,
  unweighted).
- 145.63046264648438 against 145.63047790527344 (E2M1x2, rate 6, completion 1, span 3,
  weighted).

That is the FMA contraction, observed on this box, and the test sees it.

### Blob digests

- **Fixture id.** `encoder_fixture_id` is
  `03bbc5b1c56d55e1d7f5f0d1baa1107e462d5bad18a412d0c232c78d04c95519` in all eight bench actions
  below, on `44d20d670`, `121441039` and `ce87ceb13`. The old and the new id are the same.
- **LFM timing units.** The 32 LFM L18 expert `w1` blobs are mutually distinct. Their per-unit
  digests are identical across four actions: before (`44d20d670`, graph), after (`121441039`,
  fused), and both arms of the power A/B on `ce87ceb13`.
- **Profile units.** The eight profiled units' digests are identical before and after.

### GLM-5.3 census wires

The GLM check re-encodes 32 stored `TESSERA_E2M1_K2_R896` routed-expert units from the
`extension-e2m1-01` census workspace. It uses the BF16 source weight and each row's Hessian
references, calls `export.encode_linears` in chunks of 8 as the census does, and compares each
produced blob with the stored blob byte for byte:

- row-0045 (layer 10), 16 units:
  - `down_proj` `[4096, 2048]`, experts 0, 1, 10, 50, 100, 150, 200 and 287.
  - `gate_proj` `[2048, 4096]`, experts 0, 10, 100 and 200.
  - `up_proj`, experts 1, 50, 150 and 287.
- row-0046 (layer 11): 8 `gate_proj` and `up_proj` units.
- row-0047 (layer 12): 8 `down_proj` units.

The script is `receipts/glm_identity_reencode.py`.

| Tree | Arm | Byte-identical | Fused calls | Warm s/unit | PB |
|---|---|---:|---:|---:|---|
| `121441039` | control, `TESSERA_TCQ_FUSED=0` | 32/32 | 0 | 5.0153 | `a14175a56dcb` (sparky) |
| `121441039` | fused | 32/32 | 1,536 | 1.1639 | `0dd871a93300` (sparklina) |
| `434426432` | control, `TESSERA_TCQ_FUSED=0` | 32/32 | 0 | 5.0106 | `35be0e57bb36` (sparky) |
| `434426432` | fused | 32/32 | 1,536 | **1.1411** | `35be0e57bb36` (sparky) |

In every arm, all 32 checkpoint identities are equal except `encoder_source_sha256`, which
moves with any source change. On the GLM shape the fused path is **4.39x** per unit. That ratio
comes from the same-action pair on `434426432`. The `121441039` pair ran on two boxes.

## Which tree each receipt was taken on

Stage 1 moved through three bases:

1. It was written on `master` `44d20d670` as commit `121441039`.
2. It was rebased onto `a5ee7c166` as `434426432` when #510 and #511 landed.
3. It was rebased onto `afb71840c` as `ce87ceb13`, this change, when #513 landed.

`encode.py` and `tcq_fused.py` are identical in `121441039` and `ce87ceb13`. The only source
files `git diff 44d20d670 afb71840c -- src/` touches are `container.py`,
`serving/nvfp4_moe_route.py`, `serving/runtime_contract.json`, `slicing.py`, `unit_artifact.py`
and `wire.py`. None of them is on the encoder's hot path. So each receipt stands as follows:

- **Profiles.** The profiles, taken on `44d20d670` and `121441039`, measure the encoder code
  this change merges.
- **Identity receipts.** The full suite and the GLM A/B were re-taken after the first rebase,
  on `434426432`. They were in flight on `a5ee7c166` when #513 landed. #513 touches
  `wire.pack_levels` only, so the write-path bytes cannot differ, and they were not re-run on
  `afb71840c`.
- **Power.** The power A/B of record ran on `ce87ceb13` itself.

## In-process profile

This is `torch.profiler` with CUDA activities, from `experiments/tessera385_bench.py
--only-profile`. The workload is the #385 shape: `B=8`, 8 LFM experts x 256 columns, which is
2048 unit-columns.

- Before: `44d20d670`, PB `f56872def576`.
- After: `121441039`, PB `be03ec0fa10d`.

Both ran on sparky, exclusive.

| Kernel family | Before: per unit-col | Before: device s | After: per unit-col | After: device s |
|---|---:|---:|---:|---:|
| elementwise | 241.4 | 0.6620 | 45.4 | 0.1838 |
| reduction (min / sum / argmin) | 78.0 | 0.6447 | 22.0 | 0.2496 |
| index / gather / scatter | 129.2 | 0.6215 | 3.2 | 0.0202 |
| copy (D2D / H2D) | 157.2 | 0.3151 | 31.1 | 0.0553 |
| fill / memset | 1.6 | 0.0030 | 1.6 | 0.0031 |
| BLAS / solver | 0.1 | 0.0016 | 0.1 | 0.0016 |
| other | 0.4 | 0.0076 | 0.4 | 0.0077 |
| Triton `_forward` | none | none | 32 launches | 0.0406 |
| Triton `_traceback` | none | none | 32 launches | 0.0092 |
| Triton `_minima` | none | none | 32 launches | 0.0024 |
| Triton `_unpack_body_kernel` | 8 launches | 0.0001 | 8 launches | 0.0001 |
| **Total** | **608.0** | **2.2555** | **103.9** | **0.5734** |

| Per unit-column | Before | After | Change |
|---|---:|---:|---:|
| Device kernels | 608.0 (1,245,109 total) | 103.9 (212,885 total) | -82.9% |
| Device time | 1.101 ms | 0.280 ms | -74.6% |
| Mean device kernel | 1.81 us | 2.69 us | |
| Host launches | 103.9 (212,853 total) | 103.9 (212,885 total) | unchanged |
| Host syncs | 11.45 (23,445 total) | 11.45 (23,445 total) | unchanged |

The fused trellis removed about a million device kernels and 1.68 s of device time from this
span. It removed no host launches. The captured graph had already collapsed the trellis's host
launches into replays, so the 103.9 host launches per unit-column that remain belong to the
scale refit, the LUT fit and packing. Removing device work while the host launches stay put is
the signature of an encoder whose host is now the bound. The power table below says the same
thing from the box.

## Host profile after the change

This is py-spy at 100 Hz on `121441039`, PB `dab3cf9eb7e6` (sparklina), over a `B=8` bench
process: 3,988 samples.

| Frame | Share |
|---|---:|
| `_fit_lut`, inclusive | 57.3% |
| `encode.py:1566` `cost = float(_lut_cost(s, w, trial))`, self | 47.2% |
| `_lut_cost` launches (`encode.py:1331-1332`), self | 10.0% |
| `encode.py:1564-1565` trial clone and entry write, self | 5.2% |
| `encode.py:1545` `drop = int(loss.argmin())`, self | 4.0% |
| `viterbi_columns_fused` final `float(... .sum())` (`tcq_fused.py:494`), self | 5.0% |

- **By caller.** `_fit_lut` is reached through `_refit_scales_lut_metric` (46.8% inclusive) and
  `_pack_scales_lut` (23.7%).
- **Where the host waits.** Almost half the process's samples sit in one host sync per swap
  trial. The fused trellis itself shows up only where the host waits for its three launches.

## Throughput and power

### The same-binary A/B of record

- **Tree and action.** `ce87ceb13`, PB `0aed01c8a428`, sparky with exclusive GPU admission,
  priority 1.
- **Arms.** The `graph` arm sets `TESSERA_TCQ_FUSED=0`, the prior `auto` rule, which captures
  a graph once a shape repeats. The `fused` arm is the default.
- **Workload.** `experiments/tessera385_bench.py`, E2M1_K2@896, 32 LFM L18 experts
  (117,440,512 parameters an arm): three repeats at `B=32`, then three at `B=8`.
- **Power.** Netdata `nvidia_smi` `power_draw` on sparky, `update_every` 10 s, trapezoid
  integrated over the harness's own `start_epoch` and `end_epoch`
  (`receipts/power_window.py`). `n` is the number of native samples in the window.

| Arm | B | Window | n | Mparam/s | Mean W | Envelope | params/J |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph | 32 | 150.1 s | 15 | 2.351 | 48.26 | 34.5% | 48,715 |
| fused | 32 | 76.6 s | 8 | **4.610** | 52.10 | 37.2% | **88,452** |
| graph | 8 | 243.6 s | 24 | 1.447 | 39.77 | 28.4% | 36,392 |
| fused | 8 | 79.6 s | 8 | **4.437** | 49.30 | 35.2% | **90,012** |

| B | Throughput | params/J | Power |
|---:|---:|---:|---:|
| 32 | **1.96x** | **1.82x** | +3.8 W |
| 8 | **3.07x** | **2.47x** | +9.5 W |

- **Windows.** Every window is at least 60 s.
- **Per-arm spread.** Per-arm means are resolution-bound: an arm of 25-80 s holds 3-8 samples.
  The graph `B=8` arms read 33.7, 39.2 and 46.5 W at a flat 1.44-1.45 Mparam/s, and the fused
  `B=8` arms read 53.0, 50.4 and 44.5 W at a flat 4.44 Mparam/s. The work rate did not move,
  so the spread is the power state of a host-bound device, not work. The window mean is the
  number.

### Earlier reads, and a correction

Before the rebase, the before and after reads ran as two actions on sparky, priority 1:

- `44d20d670`, PB `7aaa2fb9da04`.
- `121441039`, PB `0e353fa9faee`.

| Tree | B | Window | n | Mparam/s | Mean W | Envelope | params/J |
|---|---:|---:|---:|---:|---:|---:|---:|
| `44d20d670` (graph) | 32 | 149.9 s | 15 | 2.354 | 47.22 | 33.7% | 49,834 |
| `44d20d670` (graph) | 8 | 162.8 s (2 arms) | 16 | 1.444 | 45.73 | 32.7% | 31,574 |
| `121441039` (fused) | 32 | 76.8 s | 8 | 4.600 | 45.85 | 32.8% | 100,305 |
| `121441039` (fused) | 8 | 53.8 s (2 arms) | 5 | 4.379 | 44.69 | 31.9% | 97,981 |

**Correction: the `121441039` work per joule is superseded.** The same-binary A/B's 88,452 and
90,012 params/J replace its 100,305 and 97,981.

- **Why.** Its `B=8` window was 53.8 s, under the 60 s floor. Both of its windows read 4.6-6.3 W
  below the same code's A/B windows.
- **What agrees.** Throughput agrees within 1.3% across the two reads (4.600 against 4.610,
  4.379 against 4.437). So the difference is in the power read, not in the work.
- **The claim of this stage** is therefore 1.82x at `B=32` and 2.47x at `B=8`.

## Batch width

The fused encoder barely cares about batch width. Fused `B=8` is 3.8% below fused `B=32`
(4.437 against 4.610 Mparam/s), where the graph arm was 1.62x apart (1.447 against 2.351).

- **Identity.** Width stays outside identity: `tests/test_batched_encode_identity.py` pins every
  unit's blob at any `B`.
- **Stage 2.** After `_fit_lut` is fused, stage 2 re-derives the width and the memory plan.

## Reseal

PQ #629 carries the campaign identity reseal proof; this change submits none.
`encoder_fixture_id` is `03bbc5b1...` before and after. The GLM census wires above re-encode
byte for byte on both trees they were taken on.

## What remains: `_fit_lut`

The `_fit_lut` statistics come from PB `520a27f7fff5` on `ce87ceb13` (sparklina). The workload
is the bench's `B=8` arm over 8 experts, with a wrapper that synchronizes the device around each
call. So these walls attribute time; they do not measure throughput.

| Caller | Calls | Wall | Trial costs |
|---|---:|---:|---:|
| `_refit_scales_lut_metric` (`encode.py:2097`) | 36 | 4.787 s | 19,085 |
| `_pack_scales_lut` (`encode.py:1771`) | 16 | 2.277 s | 11,378 |
| `_refit_scales_lut` (`encode.py:2387`) | 28 | 0.078 s | 5,792 |
| **Total** | **80** | **7.142 s of a 12.533 s run (57%)** | **36,255** |

- **Call shapes.** Live counts are `n = 128` (40 calls) and `n = 229,376` (40 calls). Brackets
  span 16-50 grid points. Swap passes: 1 in 36 calls, 2 in 41, 3 in 3.
- **Where the time is.** The bracket and the greedy elimination together take 0.65 s. The rest
  is the swap loop.
- **Cost of one trial.** Each trial in the reference is a clone, an entry write, six kernels of
  `_lut_cost` and one host sync.
- **The bar.** Fusing the swap loop has to keep the reference's accept/reject sequence and its
  first-minimum tie order bit for bit. That requires every trial cost to be the same float
  torch's `sum` returns.
- **The sum order is replicated.** A replica of torch's CUDA float32 full-`sum` order (the
  `ATen/native/cuda/Reduce.cuh` configuration, per-thread vectorised accumulators, lane tree
  and block split) matched `torch.sum` bitwise in 906 of 906 cases. It covered 151 sizes from
  128 to 33,333,331 elements, and 593 of those cases give a different float if summed
  sequentially (PB `197cbc785234`, sparklina, `receipts/stage2-sum-order-probe-cta.log`).

The probe and the statistics wrapper land with stage 2.

## Evidence

| What | Tree | PB action key | Record |
|---|---|---|---|
| Profile, before | `44d20d670` | `f56872def576555d18f90cff74bc3f5f8e427d63fea26bbb7ffbfdcfb35f76d6` | `done/` (sparky) |
| Power, before | `44d20d670` | `7aaa2fb9da04f08e9d23280b0088e5513a993f7f31079046fa828c86aae6b3f8` | `done/` (sparky) |
| Profile, after | `121441039` | `be03ec0fa10d0531be31143ab8973495a9189840014a9dd82db9cb5677d0352c` | `done/` (sparky) |
| Power, after (superseded) | `121441039` | `0e353fa9faee1e0d8cf81cdcc45bf9a8750cef8543fa5faa3a53d5ca005e85bf` | `done/` (sparky) |
| py-spy, after | `121441039` | `dab3cf9eb7e6e5c7dee4338c6b715f75cea31e2a5c8184b493af123197b4390e` | `done/` (sparklina) |
| GLM identity, control | `121441039` | `a14175a56dcbe959210d4f6b6dc8e64d96f1ae1ff54d638330f539a684ae4da7` | `done/` (sparky) |
| GLM identity, fused | `121441039` | `0dd871a933004bd64533af592c56af3f57d8f91963678ae5606b6cc2188765d5` | `done/` (sparklina) |
| GLM A/B, both arms | `434426432` | `35be0e57bb3663dd263e02b95b46eb775eeef34dd7fd3978023b76f5c6121c17` | `done/` (sparky) |
| Full suite, 12 shards | `434426432` | `9d0b5ba2ca85`, `901a75d7ad37`, `6d9fa84b7939`, `5f26b9d1a465`, `e4b59c149521`, `ce0febed6375`, `12b31922429f`, `390f5bc4336d`, `9aff8c4e2175`, `f1c5332b8d03`, `6635465fc536`, `d5bc1932f99b` | `done/`; shard 3 (`5f26b9d1a465`) `failed/` on the span2 test |
| Doc tests, 3 shards | `ce87ceb13` + these docs | `133e96009cf41a2de5aeaf5f13c42922f55ef0626e4c9c3dd2f887a1852e2fa6`, `b9197b1bbe4229357b0214ef77a1211a874f3c77ad9fb6b961f43ba9e4e49d9f`, `c88bdf34d6e25ee8b2951ac9a15314aa744569596244c7b6d2a9974f1ca4e618` | `done/` (dl380g10) |
| span2 failure on `master` | `44d20d670` | `39cac96ecf18a7b0954c8e8a0441153c361e56484c83674f8e537113eaac2c6a` | `failed/` (pre-existing) |
| Bite, `_minima` point scan | `44d20d670` + stage 1 | `e0b24750e8074f2223aebc49b62c0faf998d87efa769dc6fccac2223fa8d3396` | `failed/`, as intended |
| Bite, `_forward` fold scan | `44d20d670` + stage 1 | `0f819416f05ba2bae3c444064aaefc3ea38427d4f7820638161552803ee6f0fd` | `failed/`, as intended |
| Bite, `_forward` branch choice | `44d20d670` + stage 1 | `9c75c5ab573ab7848b44d1715bb42461bf58c9a617f954b39caf569b01c48b1c` | `failed/`, as intended |
| Bite, plain multiply | `44d20d670` + stage 1 | `6a6911e8f9c61b5369d21136343fa474305cfb7867af086f970700e09131f34d` | `failed/`, as intended |
| Power A/B of record | `ce87ceb13` | `0aed01c8a428aee10caf74dd21173c94025badc75ce5bf66b7bf6fcdfbc48569` | `done/` (sparky) |
| `_fit_lut` statistics | `ce87ceb13` | `520a27f7fff549cfe608a86b03bf2ad16c94f2c815869b5a2f78a2c2bab942f4` | `done/` (sparklina) |
| Sum-order replica | `ce87ceb13` | `197cbc785234a4b5c327d5f67c215198487cc90425832f0f44047dac5dc06944` | `done/` (sparklina) |

**Where things live**

- **Records.** Records are under `/mnt/shared/prismabuild-fleet/pb-queue/<state>/<key>.json`.
  A pbrun snapshot of a worktree with uncommitted edits records the commit it was taken from
  as its parent. The "Tree" column is that parent.
- **Receipts.** Bench results, profile tables and GLM results are under
  `/mnt/shared/tessera-measurements/tessera486-fused-tcq/`. Each `results.json` records its
  argv, environment and `encoder_fixture_id`.
- **Power windows and scripts.** `receipts/` in that directory holds the Netdata samples,
  the window integrals, `power_window.py`, `profile_families.py` and the GLM script.
