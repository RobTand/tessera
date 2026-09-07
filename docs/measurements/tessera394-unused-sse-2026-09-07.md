# Unused joined-TCQ SSE read, full B32 review, 2026-09-07

The named SSE read is removed, but this single full B32 pair does not establish
a reliable material speedup: E2M1 throughput is 1.01072x and work per GPU joule
is 1.00598x. The BF16 control is flat in time and varies more in energy. This
is bounded evidence for the mechanism and its small measured end-to-end delta,
not completion of the encoder performance work in #385.

This is the separate minimal TCQ scalar-read experiment in [PR394](https://github.com/RobTand/tessera/pull/394), based on
`382a1a97`; it does not include [PR392](https://github.com/RobTand/tessera/pull/392)'s host-synchronization rewrite. The
public `viterbi_columns` function still returns its scalar SSE. Its shared
private runner returns paths and the plan; `_run_joined` consumes only paths,
leaving the otherwise discarded diagnostic SSE on device. Every encoded unit
still computes its own reconstruction SSE.

Candidate production source is `3f9aae5c`. The later `ee1f5176` commit changes
only a cold-worker diagnostic test setup. The baseline is the attributable
full32 A-BF16/A-E2 pair from the earlier same-session B32 review; its workload,
input seals, container image, harness, B32 width and resource reservations are
the same. These baseline measurements precede the new candidate rather than
being adjacent interleaved arms. No confidence interval is claimed.

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
Triton 3.7.1; the device is NVIDIA GB10, driver 595.84. The retained baseline and new candidate benchmark
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

## Time, energy and host evidence

| Family | Before seconds | After seconds | Throughput ratio | Before / after GPU joules | Work/J ratio |
|---|---:|---:|---:|---:|---:|
| BF16_K1@1792 | 123.410480 | 123.533258 | 0.999006x | 10931.68 / 10762.62 | 1.015707x |
| E2M1_K2@896 | 54.237256 | 53.661918 | 1.010722x | 2679.53 / 2663.61 | 1.005979x |

The BF16 WINDOW path is unchanged and serves as the control. Mean GPU power
is 88.58/87.12 W before/after for BF16, and 49.40/49.64 W for E2M1: about
63% and 35% of GB10's 140 W envelope. Power is estimated by linearly
integrating complete, attributed broker samples for the expected action,
with no foreign GPU processes. Each measured interval contains 51–117 power
samples; the maximum bracketing gap is 1.078 seconds. The E2M1 energy delta
is smaller than the control's energy variation. GPU utilization percentages
are not used as saturation evidence.

Process CPU time is approximately one CPU-second per wall second in all
four timed arms. Peak CUDA reservation is unchanged: 7.758 GiB for BF16 and
6.629 GiB for E2M1, below the 32 GiB total reservation. Sparky's mean
available memory exceeds 102 GiB. Whole-host Netdata busy CPU excluding
I/O wait is 7.84/8.69% for BF16 and 8.00/7.62% for E2M1. Sparklina's
corresponding values are 7.03/2.74% and 29.87/1.10%, reflecting its separate
workloads. Raw CPU, load, RAM, I/O and GPU-power series on both hosts cover
every timed and sampled-profile interval; `host-summary.json` derives each
interval from those archived samples.

## What the full native profiles show

The full E2M1 sampled profiles contain 6,381/6,281 observations over separate
63.113/61.622-second calls. The baseline has 18.98 sampled seconds of stream
synchronization at `_TCQPlan.sse`; the candidate has no sampled SSE frame.
The joined runner's inclusive sampled time falls from 25.62 seconds to below
the top-30 function list. This confirms removal of the specific diagnostic
read identified in the regression.

The overall timing does not fall by those 18.98 seconds. Native kernel-launch
frames instead rise from 3.01 to 20.19 sampled seconds, including 15.36 seconds
at `trellis_pass`'s `completion_bits[:, which] = c_bits` assignment
(`encode.py:2861` in the candidate). Stream synchronization at `columns_of`'s
first device index-tensor creation grows from 0.15 to 4.86 sampled seconds.
The scale-fit trial-cost read still accounts for 14.54/14.19 sampled seconds.
Total stream-synchronization frames are 35.10/21.49 seconds, while
`trellis_pass`'s inclusive time grows from 2.06 to 24.74 seconds.

These observations locate time later in the same schedule after the unused
read is removed. They do not establish individual GPU kernel durations or
prove the device-side cause of a native launch's wait. Inclusive categories
overlap, and the sampler's overhead differs from an unprofiled call; the
unprofiled 1.01072x ratio remains the performance result. BF16 profiles show
the unchanged window final-cost read remains dominant, with 85.48/87.53
sampled seconds of stream synchronization across 12,623/12,706 samples.

## Targeted validation and source freeze

All tests and measurements run through PrismaBuild. The authoritative
baseline scalar-read regression is action `4d50e407fd74`: four failures at
`tests/test_joined_tcq_sse.py:34`, with the message “joined TCQ extracted
discarded SSE and drained the CUDA stream”, zero skips, zero uncollected
modules and four CUDA-allocating tests. It covers single/joined calls in
both eager and captured execution.

The candidate targeted action `7f137dbff749` runs the new regression,
TCQ graph checks, batch identity, LDLQ LUT-plane and byte-baseline audit
files with four pytest-xdist workers, `--dist worksteal`, native threads
bounded to one and `--strict-cuda`. It records 111 passed, one failed,
zero skips/uncollected modules and 69 CUDA-allocating tests. All new
regressions, public SSE comparisons and byte checks pass. The sole failure
is an existing cold-worker test setup: identity fixtures add four refits
to a diagnostic sink expecting only three experiment passes. The exact
case fails identically on pristine `382a1a97` in action `7952d8f9d32c`,
showing `[7, 3, 3]` where the assertion expects `[3, 3, 3]`.

The separate off-task fix `ee1f5176` resolves `encoder_fixture_id()` before
opening that diagnostic sink. Its cold isolated case passes in action
`592c613c4485` and its complete 55-test file passes in `8732100b093a`, both
with zero skips or missing collection. These repair counts are not added
to the original population as a new aggregate. PB action `d9a723fe5eee`
passes `py_compile` for the encoder, benchmark and both touched test files.
No further broad-suite run was repeated for this separate minimal patch.
The existing broad encoder/native validation remains recorded with the
previous experiment; this run's identity claims are its own receipts.

The independent artifact audit verifies terminal results, actual logs,
source bundles, successful CAS payload hashes, population files and sampler
outputs. The first regression attempt used the wrong container UID and
could not write its surface JSON after its four expected failures; it is
retained as an environment failure and superseded by the authoritative
UID-corrected run above. No other missing artifacts or unexpected skips
remain. Benchmark result and sampler JSONs are hashed at audit time; their
external file hashes were not emitted into the successful CAS logs.

Candidate package source SHA-256 is
`e95c5282a4475d6582199c7053e45d492fd22c935e47d95cc8e825eddf0f6b83`,
versus baseline
`081b47f5cbcd172c0709eb17114c4783166be0b4b51c0a7a90a696ce60de7128`.
The measured candidate's production Git tree
`977272638424089fd83144b2b8d3bf58a5facfe1` matches both `3f9aae5c` and
`ee1f5176`. Only `encode.py` changes in the package; `window_viterbi.py`
and `export.py` retain the baseline bytes. The full SHA-256 map is in
`candidate-source-identity.json`.

All 64 family/unit blobs match the retained baseline and every warmup,
unprofiled and full sampled-profile pass. Thus this population's bytes and
bpp are unchanged. No new served KL, second-model measurement, runtime pin
or release qualification is claimed. The performance issue #385 stays
open; this pair is not evidence of a material acceleration. The experiment
is retained unmerged, with its branch and evidence kept for recovery; the
canonical producer remains `382a1a97`. No further performance repeats are
part of this review.

## Reproduction and retained artifacts

All new artifacts are under
`/mnt/shared/tessera-measurements/tessera385-sse-20260907/`.
`b32-campaign.json` contains the exact two candidate commands and resource
reservations; `b32-submit.jsonl` contains their immutable action keys:

- BF16: `0a7c5cd11ef19cb8776e9328e2bad32eb5859764a1532c850cc85b94905a5a2a`.
- E2M1: `d4a881edd18821336bba132bd341a41e64fac6ed349a5130cf2222717c3f966d`.

`baseline-submit.jsonl` points to the retained same-session A arms under
`/mnt/shared/tessera-measurements/tessera385-pr2-review-20260907/`;
those original receipts and telemetry are unchanged. `comparison.json`,
`profile-summary.json`, `host-summary.json`, each candidate's
`netdata-both-hosts.json`, and `artifact-audit.json` carry the numerical
and identity evidence. Their analysis scripts are retained alongside them.
`targeted-campaign.json`, `cold-baseline-campaign.json` and
`repair-campaign.json`, with their submission JSONLs, retain the test
commands and populations. The owned read-only broker observer was stopped
after both measurements completed.
