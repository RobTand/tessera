# T-16 decode-once dense prefill: KILL

## Decision

The prototype fails the fixed M=2048 speed bar. Its full lane takes 2.164870 times the same-wire BF16 control.
GO requires a ratio at most 1.15. KILL requires a ratio above 1.5. These thresholds did not change.

Do not select this lane for service. The code remains default-off and experimental.
This result does not authorize a default change, serving pin change, cell promotion or larger tuning build.

## Scope and protocol

The rank-local KDA cell has 12576 rows and 4096 columns. It includes q, k and v at 4096 rows each.
It also includes b at 32 rows and the replicated f_a and g_a roles at 128 rows each.
The experiment uses synthetic packed T-16 R2048 WINDOW words as one merged role. It does not load a checkpoint.
The BF16 control uses the independent reference decode of those same words, table, row scales and column order.

Each cell has 80 cold-L2 CUDA graph samples per order after ten warm-up samples.
The primary statistic is the mean of the forward and reverse medians. Spread is abs(F-R)/mean.
The L2 eviction buffer occupies 100663296 bytes. CUDA events exclude the eviction read.
The M=16 window split uses a separate two-order pilot. Prefill uses split one. No production split rule changes.

## Primary result

| Arm | M=16 ms | M=2048 ms | M=4096 ms |
|---|---:|---:|---:|
| T-16 R2048 window | 0.372600 | 6.306976 | 21.189696 |
| T-16 per-step decode plus BF16 GEMM | 3.580416 | 5.798336 | 8.582336 |
| cuBLAS, same-wire BF16 | 0.478632 | 2.678376 | 5.422176 |

| Full decode-once ratio to BF16 | M=16 | M=2048 | M=4096 |
|---|---:|---:|---:|
| Ratio | 7.480519 | 2.164870 | 1.582821 |

| Cell | Forward median ms | Reverse median ms | Spread % |
|---|---:|---:|---:|
| Window M16 | 0.374224 | 0.370976 | 0.871714 |
| Decode-once M16 | 3.565568 | 3.595264 | 0.829400 |
| BF16 M16 | 0.475952 | 0.481312 | 1.119860 |
| Window M2048 | 6.272288 | 6.341664 | 1.099988 |
| Decode-once M2048 | 5.788672 | 5.808000 | 0.333339 |
| BF16 M2048 | 2.670224 | 2.686528 | 0.608723 |
| Window M4096 | 21.198080 | 21.181312 | 0.079135 |
| Decode-once M4096 | 8.560992 | 8.603680 | 0.497392 |
| BF16 M4096 | 5.416432 | 5.427920 | 0.211870 |
| Decode only | 3.087376 | 3.087360 | 0.000521 |

Decode alone takes 3.087368 ms per step. The full lane includes that decode on every replay.
The independent CPU verifier recomputed all ten cell reductions and all six ratios. The maximum difference is zero.
It reproduced KILL from the raw samples. It did not rerun the measurement.

## Scratch and correctness

The shared KDA scratch occupies 103022592 bytes, or 98.25 MiB. One shape and device share one allocation.
Each module retains its packed bundles, not a private BF16 matrix. Native named_tensors exposes the shared scratch.
The experiment retains a separate 103022592-byte BF16 control. That control is experiment memory, not serving storage.
The packed BODY occupies 52428800 bytes. The window module owns 52550852 bytes of distinct storage.

The scratch equals the independent reference at all 51511296 positions. One-hot checks cover all 4096 input columns exactly.
The full-size numerical checks use the derived fp64 bound. The largest observed error-to-bound ratio is 0.308491.
The tests also cover mixed rates, initial state, role order, per-step refresh, shared storage, owner lifetime and eager stream order.
Graph replay passes under external serialization. This result does not qualify concurrent graph replay with shared scratch.

The targeted GPU run passed 78 tests, failed zero and skipped zero. It used two worksteal workers and strict-cuda.
Torch 2.13.0+cu130 reported one NVIDIA GB10 device. No module was uncollected. The skip-reason map is empty.
The allocator counter reports 66 calls with CUDA allocation. This is a targeted population, not the full CUDA surface.
The earlier CPU run passed 105 tests and skipped 23 with the verbatim reason `BF16 prefill requires CUDA`.
That CPU run had no CUDA device and predates the final empty-input and address fixes.

Pre-fix proof is retained:

- The absent-API GPU run failed all 29 cases with `ModuleNotFoundError`. It had no skips and no uncollected modules.
- No call in that absent-API run allocated on CUDA. It proves the missing API, not CUDA arithmetic.
- The address probe failed with `scratch addresses wrap: [[-2147483648, -2147479552], [-2147483647, -2147479551]]`.
- The address probe used CUDA and allocated during its call. The corrected probe passes in the 78-case run.
- All eight harness cases report errors before their source files exist. The completed CPU consumer run passes those cases.

The branch has two bounded corrections: it avoids an empty-step decode and uses 64-bit scratch addresses.
The address correction has its own pre-fix failure and passing regression test.

## Profiles, power and safety

The packet retains one Chrome trace per cell. Each trace covers three eager calls after the timing samples.
The direct decoder profile contains one `_window_bf16_decode_kernel` launch at 3112.512 us per call.
The full M2048 lane profile takes 5813.030 us per call and contains two launches.
These profiles identify the decoder cost. They are not the cold-L2 graph statistic.

Both-Spark Netdata covers the action:

| Host | Board power mean W | Board power max W | Returned points | Collection interval s |
|---|---:|---:|---:|---:|
| sparklina | 33.917 | 81.6 | 6 | 10 |
| sparky | 21.329 | 56.0 | 7 | 10 |

The action mean includes compile, reference and profile work. Short per-arm power windows do not establish qualified energy.
The packet retains NVML per-arm series and diagnostic calls per joule. Energy remains HOLD.
This result does not use GPU utilization as a saturation measure.

The GPU actions use class gb10, priority -10 and an exclusive device. They have ordinary task class, not PB measurement qualification.
Each action reserves 16 GiB of host memory. The main proof and screen reserve an 8 GiB GPU subset.
The screen uses an 8563892224-byte start estimate from the prior measured footprint plus additions and a 3 GiB margin.
That estimate does not isolate driver overhead. The in-run guard aborts below 2 GiB with SIGTERM, then SIGKILL.
The guard did not trigger. Minimum host MemAvailable was 119018209280 bytes.

The screen reports CUDA allocation peak 915406848 bytes, CUDA reservation peak 1004535808 bytes and host RSS peak 2022862848 bytes.
PB reports scope memory peak 4465795072 bytes for the screen and 7911354368 bytes for the parallel GPU proof.
Both PB observations are complete and report zero OOM events. No process remains live in either recorded scope.
The wrapper cannot read its end cgroup path after cleanup. The report uses the PB scope observation instead.
The guard emergency path remains unexercised.

## Source and receipts

- Master base: `fc267e47558af8dfe1f34d788a91b34faac399b4`.
- Measured implementation: `d6478a3f5bff9986f6cc0478ba9d0515bb1ab59c`.
- GPU proof snapshot: `507852c699bf8c771454fa30b5965d46bb409345`.
- Screen snapshot: `00c44fc57480af86069a8604d38663e3e2db16a2`.
- Image: `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`.
- GPU proof action: `28a2c0711404a2355de24aab1fabd7845d59acbb1e1e697e9cc6ee2dfec76233`.
- Screen action: `cf1e512682ea893a97c61efb8cfa51b4a19e58d3df629fa12878ba0a7cb2d6e4`.
- Address pre-fix action: `3e314714fefe802fef92b8117acfd513e6bc5cdde809b3f08643b6e952eb571d`.
- Independent reduction action: `663e4c1ef4f673b037a27b249d6e408aaf1504e8b3a279aaa97ce7311032a20e`.
- GPU proof receipt SHA256: `24c837fa9d32de3099e95708b4b56cfadba75d6fbec8d4d1d984335abc8bb236`.
- Screen receipt SHA256: `1ba831564a9a9acf912f89629663d1f2e977bae3456437c024a8f14661fb12ff`.

Raw data, traces, memory, power and verification use `/mnt/shared/tessera-measurements/eng-t16-decode-once-20261007/`.
`verification.json` contains the independently recomputed cells and the digest of each retained data file and trace.
CAS claim, payload and receipt consistency checks pass. The verifier does not verify the full worker attestation.

## Limits and integration

This is one synthetic shape, one wire seed and one GPU host for the screen. It is not a checkpoint or TP2 serve.
No served KL, full-model throughput, full-model scratch population or concurrent graph qualification is measured.
No default, serving pin, seal, encoder profile or allowable-rung table changes.

The impacted-test selector reports FULL. The targeted receipts do not replace that verdict.
Per the repository rule, ts-integrator owns one full suite on the merge result.
The source remains an opt-in coverage prototype. Its KILL result remains the decision input for lane selection.

## Source equivalence audit

Audit action `5041316c2bfcf68b1737e7f24d127be114f682b1b44f2f4c8abff9c74b9807c1` passed on x86 with exit zero.
It materialized both retained snapshots and checked each snapshot blob against its sealed input digest.
The published pbsnapshot owner verified each exact action with exit zero and `snapshot: true`.
It verified `.pbrun-closure.801655af83c45e80.json` for the GPU proof and `.pbrun-closure.dd65b6586bea90bf.json` for the screen.
Only those verified entries were excluded. No other tracked path, mode or content blob differs from `d6478a3f5bff9986f6cc0478ba9d0515bb1ab59c`.
`source-equivalence.json` records `effective_source_agree: true` and preserves both raw snapshot identifiers and generated-entry bindings.
The audit did not rerun a test or measurement. The later report and verifier files do not change the measured kernel source.

## Contract guard correction

The first PR head removed the #538 table-to-dispatch guard with its hardcoded roster. That removal was wrong.
The corrected test retains the equality invariant and derives its expected pairs from each live route owner.
The attestation test now checks disjoint producer/census partitions and known decoders across every declared route structure.
It does not pin a list of native pairs. Both tests retain their original names.

Fault action `b6ecfb4df582b02fe217e451761e1949187e1c1fbf31bc6b93c24047cb3aae0a` removes the served BF16 pair in a separate before tree.
The restored #538 guard fails with AssertionError at `tests/test_serving_contract.py:777`.
This is a deliberate faulty-table probe, not a runtime defect in the measured prototype.
The initial corrected contract run `7ae5e715d81ac58488e54d684a693c6f01d7324b2103df192e8f6ff2b029c046` passes 93 tests.
It reports no skips and no uncollected modules on Torch 2.11.0+cpu, with no CUDA device.

Selector correction: the retained PB receipt reports FULL because its unexcluded stamp is not a source path.
ts-integrator reports a clean-worktree narrowed result with 433 tests. The earlier FULL statement describes that stamped receipt only.
The integration coordinator still owns one full suite on the merge result. The KILL measurement and GPU numerical evidence remain unchanged.

## Offline capture correction and eligibility proofs

The pool review at `49c07e53` found a stale offline BF16 mirror and missing eligibility controls.
The mirror now includes the decode-once pair without Torch, vLLM or Tessera imports.
The contract controls cover residency, regime and dense extension lanes separately.

Each negative probe removes one production condition in its own before tree:

| Missing condition | Action | Observed failure |
|---|---|---|
| Offline BF16 pair | `6e7276f492bc86e344af80dbb77051e81d90abdc4a94970b7454d1e8b5b40951` | QualificationRefused names the BF16 decode-once pair. |
| Residency filter | `c6461e6e451efe1d236cccbf370b8527c56b486dbd74cc92dde8c98c3a5a6030` | AssertionError at test_serving_contract.py:808 exposes the pair in streamed mode. |
| Regime filter | `06dcf3fd0745260a4d96066cef216fe7e988cc30b723475b79c10536f76edbd6` | AssertionError at test_serving_contract.py:827 exposes the batch-only pair in decode mode. |
| Dense extension filter | `0184168efb7f6535c3e51a61cca49410642c3ceeb3db9e1bb58aeaa0210cf1f9` | AssertionError at test_serving_contract.py:841 exposes the fused pair with no extension lane. |

These actions exit one on dl380g10. Each has no skips or uncollected modules and reports no CUDA device.
They are deliberate negative probes, not failed measurements. Their terminal logs exist; their failed actions publish no CAS receipt.

The mixed-M regression then exposed a second defect in the actual capture consumer.
Action `d4c9b64166041cec39246d386eb083ff25efa45d64e6a756d313994647c6a15a` passes 146 cases and fails that case.
Its pre-fix line is:

`QualificationRefused: 2 modules dispatched on bf16_unquantized (TESSERA_BF16/dense), the artifact assigns 1`

The parser already grouped counts by M, but the merger summed each pair's independent maximum.
The corrected merger sums admissible pairs within each M, then takes the largest M group.
It preserves launch totals, per-pair records, name checks and unnamed-module refusals.

Action `29c74dbee65289a0f6f8623c6a3a036ab6078518fa408783e16bb20a67a6f94c` exits zero and passes all 147 cases.
Its population is Torch 2.11.0+cpu on dl380g10, with four worksteal workers, zero skips and zero uncollected modules.
Its skip-reason map is empty. It has no CUDA device and records zero device allocations.
The surface digest is `995e0d7643141636e287360816309b8b3156b379b43a2e1fd952e35d2965c0a5`.

Direct API smoke `b875600659abe6eb138ddfd109194070ce3cc6d5d78379d33930bf74f4620853` exits zero.
It reports one named module and two launches across M1 window and M2048 decode-once entries.
This synthetic metadata control proves the consumer path; it does not qualify a model serve.

Hosted pure passes on source head `46291a86fe1a40be537ab1462beecb0084ce56fa` in run `37574845112`.
The hosted interpreter has no Torch or CUDA. Its log records the complete skip histogram and 161 uncollected modules.
The code correction leaves the measured kernel, scratch size and KILL result unchanged.
The record preserves two failed revisions. The CEO holds custody and keeps the executive engineer as owner.

