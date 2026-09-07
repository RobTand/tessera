# Full B32 encoder host-synchronization review, 2026-09-07

Both full B32 timing pairs show essentially unchanged throughput. This
reviews the separately retained host-synchronization change against the
merged batched LDLQ encoder at `382a1a97`. It does not combine speedup ratios
with the earlier B8 measurements: those used a different PyTorch environment
and compared a different source baseline. The measured source is `15836ebc`;
later edits in this review are documentation only.

## Workload, environment, and identity

The workload is all 32 captured LFM2.5-8B-A1B layer-18 expert `w1` tensors,
each BF16 `[1792, 2048]`, with 117,440,512 quantizable parameters per arm.
Both before and after use B32, BF16_K1@1792 and E2M1_K2@896, the same captured
per-expert Hessians, unchanged default LDLQ/refit settings and the existing
resident export path. Calibration is WikiText-2 raw train, seed 0, 32 samples
of length 512 (16,384 fit positions). No input columns or expert units were
removed for profiling.

The producer Docker image on Sparky is
`sha256:cf3f7f83e6820fa75aae249393e8fa4840af4562192203a1aed3f2082f3ea2f9`,
resolved from `prismaquant-qwen38-producer:20260827-tf516-hf128` before launching
by immutable ID. It has Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0 and
Triton 3.7.1; the device is NVIDIA GB10, driver 595.84. All four benchmark
actions reserve five physical CPUs (four native threads plus the sampler),
32 GiB total shared-system memory, and one GPU through PrismaBuild
`--measurement`. PB pins and isolates these actions on Sparky; ordinary
portable test partitions run independently where PB admits them.

Each action performs one full warmup, one unprofiled measured repetition and
one separate full sampled-profile call. The profile uses py-spy native/Python
sampling at 100 Hz, rather than retaining every CUDA event. It authorizes its
owned same-UID sampler child, checks attachment before encoding, stops that
child, and verifies the profile/log hashes and a positive sample count. Native
sampling pauses the process; its wall time is not the performance claim.
The full benchmark is never reduced to qualify the profiler.

The common encoder fixture ID is
`03bbc5b1c56d55e1d7f5f0d1baa1107e462d5bad18a412d0c232c78d04c95519`.
The input manifest and safetensors SHA-256 values are respectively
`39a6c2fdbb53dbcf37c5b0468e796d1b7629783667872fca6bb3f7ed7dde0e7f` and
`8a79b042d880006782b60e45cd4560bfe8f564f4ad5ff2f55999a8d6a99ff80f`.
The identical harness hash is
`57a7fb0cbaa52564484d27f490d857fceef3369a8b47febca763e8080c286efd`.
Receipt environment records independently hash source and inputs at entry and
exit; PB source bundles and successful result payloads are audited separately.
All 64 family/unit blobs are byte-identical before and after, and each matches
its warmup, unprofiled and full sampled-profile copies. The independent audit
also rehashes both original input files and matches all four input seals.

## Timing, power, and sampled profiles

| Family | Before seconds | After seconds | Throughput ratio | Before / after GPU joules | Work/J ratio |
|---|---:|---:|---:|---:|---:|
| BF16_K1@1792 | 123.410480 | 123.696731 | 0.997686x | 10931.68 / 10927.28 | 1.000402x |
| E2M1_K2@896 | 54.237256 | 54.526747 | 0.994691x | 2679.53 / 2687.27 | 0.997121x |

These are effectively flat comparisons; the small differences do not establish
a regression or improvement. BF16 throughput is 0.952 to 0.949 Mparam/s, and
E2 throughput is 2.165 to 2.154 Mparam/s. Mean sampled GPU power is 88.58 to
88.34 W for BF16 (about 63% of the 140 W envelope), and 49.40 to 49.28 W for
E2 (about 35%). The process consumes approximately one CPU-second per wall
second in every unprofiled arm. Peak CUDA reservation is 7.758 GiB for BF16
and 6.629/6.627 GiB for E2; the 32 GiB reservation did not bind.

Whole-host Netdata busy CPU, excluding I/O wait, is 7.84/8.47% for before/after
BF16 and 8.00/11.68% for E2 on Sparky. Available memory stays above 102 GiB
in all four measured intervals. Sparklina's corresponding CPU series is
7.03/6.03% and 29.87/6.00%, recording its independent admitted workload.
The broker power integration uses 50–117 samples inside each phase, with a
maximum bracketing gap of 1.084 seconds.


Power is an estimate from the broker's approximately one-second GPU samples,
linearly integrated over each unprofiled timing interval. Every used sample is
complete and attributed to the one expected action, with no foreign GPU
process. Both Sparky and Sparklina Netdata CPU, memory, load, I/O and GPU-power
series are retained for every timing/profile action. Netdata's ten-second GPU
power samples provide host context; the finer broker series provides the
energy estimate. GPU utilization percentages are not used as saturation
measurements. One measured repetition per source/family provides no confidence
interval or broad performance qualification.

The BF16 negative result is informative: the sampled wait moves from the old
per-chunk `float(final.sum())` to the new final `float(sse)` readback. The
respective callsites contain 85.23 and 85.19 sampled seconds of synchronization;
total sampled stream synchronization is 85.48 versus 85.44 seconds. Fewer
synchronization opportunities did not accelerate this full B32 workload.
Sampled native frames locate host waits; they are not CUDA kernel event counts
or a measurement of GPU kernel durations.

For E2, sampled stream-synchronization time falls from 35.10 to 20.54 seconds,
while kernel-launch frames rise from 3.01 to 9.84 seconds and memcpy frames
from 1.09 to 6.35 seconds. These inclusive categories overlap. The TCQ final
SSE read still accounts for 18.98/18.85 sampled seconds. Both the profile
itself and its overhead differ across arms (63.113/58.352 seconds profiled
wall versus 54.237/54.527 seconds unprofiled), so the shorter profiled run
is not a throughput result. A future minimal change can address the unused
SSE read at `encode.py:_TCQPlan.sse` and `_run_joined`; it is not part of this
retained experiment.

## Validation and disposition

The exact targeted regression controls fail on baseline `382a1a97`: three failures,
zero skips/uncollected cases and three observed CUDA-allocating tests. The
candidate passes all 23 ordered-cost, byte-oracle and synchronization-warning
tests, with 15 CUDA allocations and zero skips/uncollected cases. The warning
counter checks a bounded regression; it does not enumerate every CUDA event.

PrismaBuild partitions the 185 selected test files into eight independent
four-worker actions. The original result is **3,624 passed, 92 failed,
4 errors, 7 skipped and 1 xfailed**, with 509 observed CUDA-allocating tests
and no missing collection. All eight original source digests agree at
`7870c7ce660aa07bfcd78db8ea720af6760778fab126f20d4d2d216ee4279219`.
The seven skips are two tests requiring two CUDA devices, one E2M1 reader-range
case and four RELEASE column-cut cases outside the format's contract.

The 92 native failures share a missing development-header cause in the producer
container; four CI setup errors need Git helper programs. The Git repair
passes all seven tests. Native header supplementation initially exposed a
compiler/runtime-header conflict; supplying only the existing image's
cuSPARSE/cuBLAS/cuSOLVER library headers, without pip's runtime/CRT headers,
qualifies an actual native kernel and then passes the affected files:
161 passed with two expected two-device skips, and 35 passed with no skips.
The container's nvcc is 13.0.88, reported in those retained logs. These corrected
populations resolve the original failures; they are not added to the broad
population as a deduplicated pass count. No production code changed for the
environment repairs.

The independent audit verifies terminal/log status, source-bundle hashes,
recomputed source digests, and successful CAS receipt/payload hashes and sizes.
One CI worker-share JSON (`gw1`) is absent; the controller's seven-pass
population and source agreement for all four workers are authoritative, while
that missing individual share remains an evidence limitation.

Setup attempts with absent container `python`, missing scoped xdist tooling,
NFS-root write refusal, absent git/source attestation, or incompatible py-spy
`--native --nonblocking` flags are retained as unsuccessful tooling attempts.
Their partial test output is not counted as a passing validation population.
Scoped xdist/execnet, git and py-spy dependencies are hashed in the receipt
root; containers run as UID/GID 1000 with explicit writable cache directories.

Additional cleanup is recorded separately: the retained test wrapper that
always exited zero was removed, and stale host-sync-counter/SSE-accounting
comments were corrected. No new format, runtime pin, served promotion or
quality threshold is introduced. This covers captured layer-18 `w1` tensors,
not a full-model export, another projection/model, a served KL comparison or
a density/format promotion. The remaining kernel and full-pipeline work in
[#385](https://github.com/RobTand/tessera/issues/385) remains open.

## Reproduction and receipts

Receipt root:
`/mnt/shared/tessera-measurements/tessera385-pr2-review-20260907`.
`b32-campaign.json` contains the exact PB actions, Docker invocation and flags;
`b32-submit.jsonl` records their action keys. Each `A-BF16`, `B-BF16`, `A-E2`,
and `B-E2` directory contains `results.json`, the speedscope sample profile,
its hash receipt/log, and `netdata-both-hosts.json`. `comparison.json`,
`profile-summary.json` and their retained receipt-root analysis scripts
summarize existing artifacts without rerunning the encoder. `artifact-audit`
JSON/Markdown and its script independently verify terminal states, source
bundles, test populations and successful CAS receipts/payloads.

| Arm | PB action key |
|---|---|
| A-BF16 | `d7d5c9ca99525cbfc51e6ab99d754ceb9d59745fd5b79f0bcf9042c66b7399fd` |
| B-BF16 | `56b5fc58dfdc1b058f603df10ec5cf5cb3a817939742b9b869d227ead1691bf9` |
| A-E2 | `b4e83db41f4dd4b45b899f6e999bcf798629adb3429052b60e57647e16e56300` |
| B-E2 | `d2ab4fb474ada45bc7f3f6c377fa33b5e1577574f3cfd6006a375fd7ff5f0a6a` |

Final repair actions: `467a8777becc` (seven CI tests), `6d576b2598b4`
(one native qualification), `9c65683e761a` (161 native/lane tests), and
`8f20be4b5592` (35 BF16 GEMV tests). Full keys, commands, populations and
hashes are in the audit and corresponding submission manifests.

Disposition: PR392 remains unmerged as a measured negative experiment, with
its branch and all artifacts retained. The canonical measured producer stays
at `382a1a97`; PR392 does not advance the producer freeze. Issue 385 remains
open for its actual performance and full-pipeline acceptance criteria.
