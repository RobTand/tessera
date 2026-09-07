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
