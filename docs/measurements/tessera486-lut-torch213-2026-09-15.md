# Torch 2.13 on the fused LUT swap passes

Status: **admitted in tessera#519; tessera#486 stays open.** The GLM census image runs torch
2.13.0+cu130, where `lut_swap_refusal` refused every fit, so census encodes ran the reference swap
passes. Torch 2.13 keeps torch 2.11's CUDA float32 `sum` order, and `lut_fused` now admits it. In
the census image the replica matches `torch.sum` bitwise in 906 of 906 cases, and
`tests/test_lut_fused.py` passes. The GLM census wires re-encode byte for byte with every fit on
the fused passes, 2.62x faster per unit than on the reference.

Date: 2026-09-15. Commit: `3538997634` on `claude/486-lut-torch213`, whose parent is `master`
`6aba6cbca` (tessera#518). Image: `prismaquant-glm-producer:content-qualified-20260908` (content
`eb8592ab`, image `sha256:cf3f7f83`), with torch 2.13.0+cu130 (git `cf30153c`), Triton 3.7.1 and
Python 3.12.3. Boxes: GB10 / DGX Spark (`sparky`, `sparklina`), NVIDIA driver 595.91.07, 140 W
envelope.

## What changed

`src/tessera/lut_fused.py` sets `_VERIFIED_TORCH = ("2.11.", "2.13.")`. `lut_swap_refusal` reads
the list at call time, so any other release still keeps the reference. `sum_plan` doesn't change.
`tests/test_lut_fused.py` adds `test_a_torch_release_nobody_checked_keeps_the_reference`, which
refuses 2.12 and admits 2.11 and 2.13.

The commit changes no other file, so it cherry-picks onto the census producer branch,
`census/d403-486-v9`. A trial pick onto `ba7c31cf7` applied cleanly, and the picked `src/tessera`
tree is `7b93de3a`.

## Torch 2.13 keeps the sum order

The comparison reads torch 2.11.0 (git `70d99e998b`, the host venv's release) against 2.13.0 (git
`cf30153c`, the census image's release). At each commit, GitHub's source equals the shipped header
apart from an install guard: one line at the top and four at the end. Line numbers below are
shipped-header lines.

- **`Reduce.cuh`.** 23 lines change, and none of them is on the CUDA order:
  - `block_x_reduce` spells the `warpSize` builtin as `C10_WARP_SIZE` (lines 639, 642 and 653).
    `C10_WARP_SIZE` is 32 on CUDA in both releases (`torch/headeronly/macros/Macros.h`), and the
    GB10 reports a warp size of 32.
  - The loop that walks offsets upward within a warp is now compiled only under `USE_ROCM`;
    before, `USE_ROCM` or `FBCODE_CAFFE2` selected it (lines 659-660). A CUDA wheel defines
    neither, so it keeps the downward loop.
  - `global_reduce` adds staging variants for ROCm only (lines 839-865). Its CUDA loop is
    unchanged.
  - `AccumulationBuffer() {}` became `= default` (line 980), and a comment changed (line 1128).
- **Unchanged.** `setReduceConfig`, `ReduceConfig` and `mnt_wrapper`, and every line of
  `ReduceSumProdKernel.cu`, `Reduce.cu`, `cuda/ReduceOps.h`, `block_reduce.cuh`, `CUDAContext.cpp`
  and `thread_constants.h`.
- **Around the kernel.** `MemoryAccess.cuh` changes only under `USE_ROCM`. The `sum_out` and
  `should_use_acc_buffer` bodies in `ReduceOps.cpp` are identical. The caching allocator is
  refactored, but `kMinBlockSize` (512) and `round_size` are unchanged.

## In the census image

Every run below used the census container. Its spec is the census anchor spec,
`extension-e2m1-01/anchor-spec.extension-e2m1-r896.reseal-57f12627c.json`, with the tree under test
at `/workspace` in place of the census producer source. The spec keeps the image, all four mounts,
every environment entry and the `PYTHONPATH` order; only `PYTHONPATH[2]` and `TESSERA_REPO` point
at `/workspace`.

### The sum-order probe

`experiments/tessera486_sum_order_probe.py` held the replica to `torch.sum` in the census image
(PB `080e269adbcb`, sparky). 906 of 906 cases are bitwise equal, over 151 sizes from 128 to
33,333,331 elements, and 596 of the cases are order-sensitive. On torch 2.11 the probe counted 593
order-sensitive cases (PB `197cbc785234`).

The environment probe (PB `8d40d7832fda`, sparklina) read what the gate and the replica depend on:

- a warp size of 32, and 48 multiprocessors with at most 1,536 threads each;
- the native caching allocator, under `PYTORCH_ALLOC_CONF=expandable_segments:True`;
- the tensors `_lut_cost` sums. At 10 sizes from 128 to 4,194,304 elements, every one of them
  started on a 512-byte boundary at storage offset 0.

### Tests

`tests/test_lut_fused.py --strict-cuda` ran on `3538997634` in both environments:

| Environment | PB | Box | Result |
|---|---|---|---|
| Census image, torch 2.13.0+cu130 | `1e13b4ac1e27` | sparklina | 94 passed, 0 skipped, 82 on the device |
| Host venv, torch 2.11.0+cu130 | `42734507e1c0` | sparklina | 94 passed, 0 skipped, 82 on the device |

The full suite didn't run on `3538997634`. The change only adds `"2.13."` to a tuple that keeps
`"2.11."`, so no behavior moves on torch 2.11, and the targeted file passed on both torch releases.
`tools/impacted_tests.py` can't narrow the set: a changed path reaches `tests/conftest.py`, so it
selects 245 test files.

### GLM census wires

The check is stage 2's (`tessera486-fused-lut-2026-09-15.md`, GLM-5.3 census wires). It takes the
32 stored `TESSERA_E2M1_K2_R896` routed-expert units of the `extension-e2m1-01` census workspace:
16 from row-0045 (layer 10), 8 from row-0046 (layer 11) and 8 from row-0047 (layer 12). It encodes
them in chunks of 8 and compares each blob with the stored blob byte for byte. The script is
`glm_identity_reencode.py` v2.

A wrapper, `glm_reencode_census_wrap.py`, runs v2 in its own process and records:

- every answer from `lut_swap_refusal` and `tcq_fused_refusal`;
- `lut_fused.STATS`;
- the window of each `export.encode_linears` call.

Both arms ran in the census container on sparklina with exclusive GPU admission.

| Tree | Arm | PB | Byte-identical | TCQ fused calls | LUT gate answers | LUT fused fits | Tripped | Warm s/unit |
|---|---|---|---:|---:|---|---:|---:|---:|
| `6aba6cbca` | before | `917b84a6a7eb` | 32/32 | 1,536 | 160 refused | 0 | 0 | 1.1822 |
| `3538997634` | after | `61ffd2c675de` | 32/32 | 1,536 | 160 admitted | 160 | 0 | **0.4519** |

- **Identities.** In both arms, every unit's checkpoint identities equal the stored ones except
  `encoder_source_sha256`, which moves with any source change.
- **Speed.** Warm encodes are **2.62x** faster per unit. The four encode windows, cold batches
  included, take 19.1 s after the change and 40.8 s before. In the host venv on torch 2.11, stage
  2's check read 1.1447 s/unit on the reference and 0.4593 s/unit fused (PB `3f4d21d40676`,
  driver 595.84).
- **Stage 1.** These are stage 1's first measurements in the census image. `tcq_fused_refusal`
  admitted all 1,536 calls in both arms.
- **The refusal.** Before the change, all 160 answers were "torch 2.13.0+cu130: the replicated
  sum order is checked on 2.11.x only".
- **Memory.** Peak CUDA allocation is 3,796,012,032 bytes before the change and 3,796,019,712
  bytes after.
- **The source.** The after arm recorded `lut_fused.py` at sha256 `8da23604`, the file's hash at
  `3538997634`.

### Profiles

A static py-spy 0.4.2 sampled each arm at 100 Hz inside the container. Each figure below counts
the samples whose stack holds the frame.

| Frame | Before, 5,447 samples | After, 3,227 samples |
|---|---:|---:|
| `export.encode_linears` | 3,957 (72.6%) | 1,761 (54.6%) |
| `encode._fit_lut` | 2,705 (49.7%) | 517 (16.0%) |
| `encode._lut_swap_passes_reference` | 2,453 (45.0%) | 6 (0.2%) |
| `lut_fused.swap_passes_fused` | 0 | 265 (8.2%) |
| `tcq_fused.viterbi_columns_fused` | 653 (12.0%) | 604 (18.7%) |

- **The six reference samples after the change.** All six sit under v2's
  `encoder_identity.encoder_fixture_id`, which encodes the encoder fixtures before the timed
  encodes. `_fixture_weight` builds the fixture matrix on the CPU, so `_lut_swap_passes` sends
  those fits to the reference without asking the gate. No reference sample falls under a GLM
  encode.
- **PB's sampler.** PB `--profile sample` doesn't see Python inside the container. On the
  environment probe it took 11 samples, all PrismaBuild relay, launcher and import frames. None
  landed in the probe's named busy loop (PrismaBuild issue 562). In the same run, py-spy inside
  the container put 841 of 841 samples in its marker loop.

### Power

| Arm | pqteld, whole action | Netdata, encode span |
|---|---|---|
| Before | 58.9 s: 29.95 W mean, 44.06 W peak (31.5% of 140 W) | 48.1 s: 33.5 W (23.9%), 1,612 J |
| After | 37.1 s: 17.50 W mean, 35.13 W peak (25.1%) | 26.0 s: 24.4 W (17.4%), 633 J |

Neither source isolates the encode:

- **Netdata.** Sparklina's Netdata GPU power chart updates every 10 s, so each encode span rests
  on 3 to 5 points.
- **pqteld.** The box window summarizes the whole action, including model loading and the fixture
  encodes.

So these readings are envelope checks, not a power delta. Both arms stay well below the 140 W
envelope. This run doesn't measure the change in work per joule.

## How a run shows which path ran

- **Stage 2.** `tessera.lut_fused.STATS` counts three kinds of fit: those the fused passes
  answered (`fused`), those the tripwire handed back (`tripped`), and nonfinite ones
  (`nonfinite`).
  - A refusal returns its reason, and `_lut_swap_passes` runs the reference without counting it.
  - A fit that never reaches the gate also runs the reference uncounted. That happens with a CPU
    tensor, `swaps == 0`, a replaced `_lut_cost`, or `TESSERA_LUT_FUSED=0`.
  - The only log line is a `RuntimeWarning` on the first trip.
- **Stage 1.** Nothing counts or logs `tcq_fused.viterbi_columns_fused` calls.
- **A proof arm.** `encode` imports both gates and both fused entry points at call time, so a
  harness can wrap them with no source change, as `glm_reencode_census_wrap.py` does. Scope the
  counts to the encode window, because the encoder fixture id also encodes fits that never reach
  the gate. Over the window, require all four:
  - `STATS["fused"]` above 0;
  - `STATS["tripped"]` at 0;
  - no refusal from `lut_swap_refusal`;
  - at least one `viterbi_columns_fused` call.

## Evidence

| What | Tree | PB action key | Record |
|---|---|---|---|
| Container environment | `6aba6cbca` | `8d40d7832fdad7f7c1bfdf8d9e3312031bd3f08ae3dad2221f52b01d0b2d0f5a` | `done/` (sparklina) |
| Sum-order probe | `6aba6cbca` | `080e269adbcb9e718ab688b339a9ed902711b734a5a8505cadb9d0586eadc650` | `done/` (sparky) |
| GLM re-encode, before | `6aba6cbca` | `917b84a6a7ebc8f2847a4e19d8a8c1939dfdc3df7558094a2db685d831d5609d` | `done/` (sparklina) |
| GLM re-encode, after | `3538997634` | `61ffd2c675de79a4b641f3e588e7233aa506d4c33c549b0da37d9c48925ef85e` | `done/` (sparklina) |
| `tests/test_lut_fused.py`, census image | `3538997634` | `1e13b4ac1e27d49fca1899ee2fa006345ef1319001d6a7f2e6721d837074003e` | `done/` (sparklina) |
| `tests/test_lut_fused.py`, host venv | `3538997634` | `42734507e1c0a21646ded981020c8c8f3f0eace455665da6f0116f74763b3471` | `done/` (sparklina) |

**Where things live**

- **Records.** Records are under `/mnt/shared/prismabuild-fleet/pb-queue/<state>/<key>.json`.
  - The "Tree" column is each snapshot's parent commit.
  - The container actions' snapshots also hold the untracked `experiments/torch213_admission/`:
    PrismaQuant's container launcher (`pq/tools/`), v2, the wrapper and the environment probe.
- **Receipts.** The receipts are under
  `/mnt/shared/tessera-measurements/tessera486-fused-lut/torch213/`:
  - `source-diff/`: GitHub's sources at both torch commits, the host venv's 2.11 headers, and the
    diffs. `Reduce.cuh.shipped-2.11-vs-2.13-image.diff` holds the whole `Reduce.cuh` change.
  - `env-6aba6cbca/`: `env.json`, the image's torch headers (`torch-include.tar.gz`) and the
    in-container py-spy check.
  - `probe-080e269adbcb.stdout.txt`: the probe report.
  - `glm-row0045-identity/before-6aba6cbca/` and `glm-row0045-identity/after-353899763/`: v2's
    `result.json`, the wrapper's `wrapper.json`, `pyspy.speedscope.json`, and the Netdata
    `power-arms.json` and `power-samples.json`.
  - `host-lut-353899763.json` and `host-lut-353899763.log`: the host venv test run.
  - `receipts/`: the container spec, its deviations from the census anchor spec, the wrapper and
    the environment probe.
  - `bin/py-spy`: the static py-spy 0.4.2 both arms ran.
