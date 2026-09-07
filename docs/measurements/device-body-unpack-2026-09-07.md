# Device BODY field reconstruction — 2026-09-07

Status: correctness checks passed within the surfaces below; the original-14
paired measurement is pending. No speed or energy improvement is claimed yet.
Base: `a29dbec6cf2f720f1cf91eda26cbbe934a737fcf`.

`wire.unpack_body` retains its existing CPU length and canonical-padding
validation, then reconstructs byte-sized fields on the requested CUDA device.
It copies the original packed plane and derives the small rate/offset schedule.
The shared MSB-first word reader was extracted unchanged from `kernel_window`
into `kernel_bits`; importing it does not register the window custom operator.
CPU, wider-field, optional-Triton-free, and unsupported zero-rate/window cases
retain the original NumPy reconstruction. Wire bytes, render recipes and serving
gates are unchanged. Reader source identity changes.

## Functional evidence

All executions used PrismaBuild and the pinned container
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
Native threads were bounded to one; the broad CPU/GPU checks used four PB CPU
slots. These are correctness runs, not timings for a performance comparison.

| Action prefix | Mode and outcome |
| --- | --- |
| `171c518fddcc` | Baseline regression: CUDA request still called CPU `_from_bits`; expected failure. |
| `24e9aed86110` | CUDA surface: 169 passed, zero skipped/uncollected, 18 tests allocating CUDA. |
| `9be17c9756c1` | CPU surface: 148 passed, zero skipped/uncollected. |
| `a94c3f4d8251` | Existing native-window selection: 15 passed, zero skipped/uncollected, all 15 allocating CUDA. |
| `3941512fd573` | Optional-dependency regression: missing Triton raised before fallback; expected failure. |
| `c6512d14a018` | Optional-dependency fix: 22 passed, zero skipped/uncollected, 14 allocating CUDA. |
| `ff975c989710` | Final source including 64-bit flattened kernel indices: 22 passed, zero skipped/uncollected, 14 allocating CUDA. |

The native-window run spent about 207 seconds in existing CPU encoding for its
largest synthetic case, then passed; it does not measure decoder speed. Earlier
attempts `38bddcbe9aa1`, `5b16ef1c0383` and `fc9001a5785d` failed before collection
because the older scoped pytest installation lacked `py`. A new scoped pytest
8.4.2/xdist 3.8.0 installation fixed collection; none of those failed attempts is
counted as a test pass. Initial broad runs reported Torch deprecation and
read-only pytest-cache warnings; the final 22-case CUDA run had no warnings.

Terminal status, cleanup, exit codes and all passing CAS payload hashes were
independently checked. Full keys, snapshot identities and receipts are in
`/mnt/shared/tessera-clean-runtime-20260907/reader-device-unpack/verified-tests-02.json`.
The dependency-based selector reaches 191 test files through the shared wire
surface; this bounded evidence does not certify that entire selection.

## Paired experiment contract

Frozen candidate reader source SHA-256:
`83a398827f18c9d8f898ebf2103c712ebf8aa688ffba12f2f73f8f07ddd80ae2`.
Baseline reader source SHA-256:
`6215086c41553e82bbc9740da240cb09900252aea59815d25bdf1c5cebe5b0a6`.
Plan SHA-256:
`65740d96d8b00bd775bab303d3ae1822e286683815ae95aba4602da8ae5e3167`.

The existing original-14 harness retains the same two dense units, seven rungs,
wire files, original render/source/settings identities, calibration inputs and
production-cache prefetch. Both arms execute within one PB measurement action,
with three interleaved pairs, Torch CPU/CUDA profiles and a Python sampler.
GB10 class placement uses identical shared inputs and the pinned container;
the admitted worker must be read from the actual receipt. Both-host Netdata
CPU, RAM, I/O and GPU power will be retained for the measured interval. The
10-second power cadence cannot support energy attribution to a roughly
three-second phase, if that remains the observed duration.

The earlier host-pinned action `e57104b26b21` was withdrawn while READY, with no
execution, after PB gained qualified class placement. Replacement
`9e71c17716e5` uses the same frozen plan under `--measurement --host-class gb10`.
This paragraph records submission, not an experiment result.

## Paired result — 19:08 UTC, 2026-09-07

This result supersedes the pending status above. PB `46f61e769ec6` executed
both arms on Lina (GB10, driver 595.84, Torch 2.13.0+cu130, CPU affinity 5–8)
and exited zero, with complete scope cleanup. All 14 original records matched
exactly, all eight render/source/settings/wire mutations were refused, and the
original producer's source files and module objects remained unchanged.

| Warm pair | Before (s) | After (s) |
| --- | ---: | ---: |
| 0 | 2.892223 | 1.318281 |
| 1, reversed arm order | 2.907724 | 1.314013 |
| 2 | 2.897756 | 1.307653 |
| Median | 2.897756 | 1.314013 |

The observed warm verification speedup is **2.205×** for these two dense units
and their 14 original wires/renders. Every arm opened the same 28 files and
514,309,808 logical bytes; warm process physical read-byte deltas were zero.
Existing PWC prefetch retained 14 entries / 411,071,318 bytes with zero misses.
Peak GPU allocation was 923,015,168 bytes and reservation 968,884,224 bytes.
This is a reader-verification measurement, not model KL, serving throughput,
or full-model cost-generation performance.

The 50 Hz sampler's measured-arm call sites attribute 195 BODY `_from_bits`
samples to the baseline and zero to the candidate. Candidate BODY validation
still runs on CPU. The separate Torch CPU/CUDA traces contain 14 new BODY
unpack kernels totaling 6.825 ms; total recorded kernel duration changed from
233.237 ms to 241.369 ms, with kernel count 706 to 748. The improvement removes
CPU field reconstruction; it does not speed up the rest of the GPU decoder.
Remaining sampled work includes tensor identities, file digests and container
validation. First-parse times were 12.049 s and 6.473 s, retained separately;
those include initialization and are not claimed as disk-cold measurements.

Both-host Netdata covers the interval. Lina's mean host CPU busy fraction was
7.74%, with 11–13 W in four GPU power samples; Sparky had unrelated work
(22.19% CPU and 44–45 W). The 10-second GPU power cadence cannot attribute
energy to the 1.3–2.9-second warm arms. **No work-per-joule ranking or GPU
saturation claim is supported.** This bounded path remains partly CPU-bound.

The first class action `9e71c17716e5` was admitted correctly but failed before
reader qualification: the source-only candidate export lacked the exact
`pyproject.toml` used by Tessera's version resolver. Its failed outputs remain
unchanged. A new immutable sibling added base metadata SHA-256
`a98ba159a7d376f3fa931a3369cf02b209eea9f900bcc8abb2564bce66082e3a`,
identical to the baseline metadata. Candidate source SHA remained `83a39882…`.
Plan05 SHA is
`95c4acf165a50a7aa61de348d62e257041a967690b1de3a9db4c60246f45ee1c`;
its launch checks the metadata hash explicitly before invoking the unchanged
paired harness. Plan04 and its failed source-only sibling are retained as
superseded evidence. `packaging-repair-01.json` records this correction.

Evidence root:
`/mnt/shared/tessera-clean-runtime-20260907/reader-device-unpack/`.
`qualify-original14-02/results.json` contains every identity, refusal and phase;
that directory also contains both Torch traces/operator tables,
`netdata-both-hosts.json` and `profile-summary-01.json`.
`qualify-original14-sampler-02/` contains the Python profile and child exit.
`measurement-verified-01.json` independently binds both terminal records,
cleanup, CAS receipts and all result/profile/telemetry artifact hashes.
Passing CAS payload:
`25a2d52775ae50fea37640ca16ad00226fe30fc02bc85e25dd093fbfa522a26a`;
receipt:
`ea248afbf45edd19b2d2d8ac979b8276f29d703da8e559a116292aab2ef7f5e9`.

## Validation-wrapper gate correction — 2026-09-07

PR #419's first bytes-only CI run reported one failure in
`test_every_wrapper_that_starts_a_container_gates_and_names_no_digest`:
`device_unpack_check.sh` had not called the shared `runtime_image_require`
gate. Its explicit image digest alone did not satisfy the wrapper policy.
The wrapper now uses that existing gate and forwards its resolved image
metadata into the container. Its image, CPU/GPU mode handling and all reader
source files are unchanged; the completed functional and performance results
above were not rerun.

PB `f49766005c00` ran `tests/test_runtime_image_pin.py` on dl380g10 with four
xdist workers/native threads bounded to one: 36 passed, zero skipped and no
CUDA allocation. The CPU-only controller reported 71 Torch-dependent modules
not collected outside this targeted policy surface; this does not replace the
separate CPU/CUDA reader coverage. Terminal exit zero, cleanup and CAS payload
were independently verified in
`/mnt/shared/tessera-clean-runtime-20260907/reader-device-unpack/wrapper-policy-verified-01.json`.
CAS payload:
`84488e47613c49f5bd0c0b366fe9454c731db0fed643eab0afdb0368f56acec9`;
receipt:
`54b414cef4e4e1d633fcc4cd8b0b67b2b10254b3d00b6d2892a99fff71962655`.
