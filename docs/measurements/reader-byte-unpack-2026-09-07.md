# Byte-sized wire field decoding — 2026-09-07

`wire._from_bits` expanded every bit into an int64 word, shifted another
rows-by-width int64 matrix and reduced it. For fields of 1–8 bits, the reader
now packs each row into one byte, shifts its padding and widens once. Wider
fields retain their old path. The wire grammar, encoder profile IDs and
reconstructed values are unchanged; the reader source identity changes.

The candidate is isolated from the producer and from reader14df used by the
ongoing full preparation. This does not alter production defaults or qualify
a serving lane.

## Correctness and memory regression

All execution used PrismaBuild and the pinned CPU/GPU container
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
CPU pytest used four xdist workers, worksteal and one native thread each.

- Red `d678297724d97c45c2bea9dfe08fb004848065e1729b0cc000290d23dd4cb7a5`:
  1 expected failure, 18 passes. The 262,144 seven-bit-field case allocated
  **31,458,810 bytes against a 6,029,312-byte bound** at
  `tests/test_wire_unpack_memory.py:42` before the fix.
- Green `f62871c466668f9ef4e7f40496299fd09ea542354f364cb011be88647579d700`:
  109 passes in 8.98 seconds on Sparky (`test_wire_unpack_memory.py`,
  `test_wire.py`), including that allocation bound, integer bit-order goldens
  through width 64, read-only/non-contiguous inputs, empty fields and dirty
  slack/truncation refusals.
- Byte audit/legacy green
  `f1d3937f88c0a09ffba45df0d2ab8d2066fe20af3b0851fc454ceb60353fd0cd`:
  39 passes in 13.52 seconds on Sparklina (`test_audit_byte_baseline.py`,
  `test_ladder_wire.py`).
- Namespace qualification
  `f4b2737095ec9437f322cca0f81e1d4427bd93ef86e0fef3d821593894976908`:
  all 11 retained artifacts decode bit-exactly; all 44 original/new corruption
  cases refuse. The primary producer package remains bound to its original
  source and module objects.

Both CPU pytest runs report **0 skipped, 0 modules not collected**, Torch
2.13.0+cu130 with no visible CUDA device, and no CUDA surface coverage. These
are targeted tests; the combined full suite belongs to the coordinator.

## Isolated original-fourteen GPU A/B

PB `d2692d876acebbff144d7908d7ffc79d77bdd7f84696bf5b3297e7214a499a7e`
ran on Sparky GB10, affinity 5–8, CPU4/memory12 GiB/GPU8 GiB under measurement
admission. PQ base was `7fa2d6e8d16ef12926cf95af75fd09161ec6922d`; both arms
used identical checkpoint identity binding, original render hashes, PWC
transfers, source/H/recipe checks and original 14-wire records. Only the
namespaced reader changed:

- Before: `14df443217e2a6a1bc4857532f0b5ead7fa7f5755dbd61e96605659940a8a1bc`.
- After: `6215086c41553e82bbc9740da240cb09900252aea59815d25bdf1c5cebe5b0a6`.

Three interleaved measured pairs, in seconds:

| Arm | Pair 0 | Pair 1 | Pair 2 | Median |
| --- | ---: | ---: | ---: | ---: |
| Before | 3.171481 | 3.185037 | 3.169881 | 3.171481 |
| After | 2.897479 | 2.936563 | 2.926822 | 2.926822 |

The measured ratio is **1.083592**, or **7.71% less elapsed time**. All 14
verification records agree; render, source, settings and wire corruption
refuse in both arms (8 cases). This gate did not inject a separate Hessian
corruption case. The 411,071,318-byte original render population was resident
prefetched, with no misses. Every measured phase has the same exact 28-file
open roster, 514,309,808 logical opened-file bytes, 514,309,932 `/proc` rchar,
464 read calls and zero physical read bytes. Logical opened-file bytes sum
file sizes times opens; the independent kernel counters record actual I/O.

First parses (12.270/7.091 seconds) include startup/cache effects and are
excluded; the first before parse physically read 119,709,696 bytes and is not
an equal cold-cache A/B. The timed claim is resident verification, not full
preparation, model streaming, capture intake or a served throughput gain.

Torch CPU/CUDA profiles were retained for both arms. The py-spy stack sample
contains 228 NumPy `_sum` leaf samples below `before_verify` and zero below
`after_verify`; it includes first parsing/JIT and profiling, so these totals
are not exclusive timing attribution for the measured pairs. Byte packing
remains visible in the after samples. Netdata CPU, memory, I/O, load and GPU
power series were captured on both hosts. Power updates every 10 seconds,
which cannot resolve these ~3-second phases: **no per-arm energy or work per
joule claim is supported**, and GPU utilization is not used as a saturation
metric.

## Reproduction and retained evidence

Artifacts live at
`/mnt/shared/tessera-clean-runtime-20260907/reader-byte-unpack/`.
`verified-evidence.json` independently hashes terminal records, CAS payloads,
all result/profile/telemetry files and the immutable input plan.
`qualify-original14-01/results.json` SHA256 is
`e65394019a0ae3c3f4db369d1d2cff72882ed59a99e059e2faf8ebd2cfc93d88`.
Sampler and measured child both exited zero independently of the PB wrapper.
The five-line implementation is frozen separately at `reader-6215086c.../`;
the full existing reader14df and producer57809bff sources remain unchanged.

The exact PQ harness is retained in the separate
`codex/pq-reader-byte-unpack-ab` branch, files `experiments/reader_unpack_ab.py`
and `.sh`; it holds the bound PQ path constant and rejects unequal original
file open rosters. The submitted command was:

```sh
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/pq-reader-byte-unpack-ab --measurement \
  --gpu --gpu-memory-gb 8 --cpus 4 --demand mem_gb=12 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --timeout-s 600 --detach -- bash experiments/reader_unpack_ab.sh \
  /mnt/shared/tessera-clean-runtime-20260907/reader-byte-unpack/qualify-original14-sampler-01 \
  --plan /mnt/shared/tessera-clean-runtime-20260907/reader-byte-unpack/qualify-original14-plan-01.json \
  --plan-sha256 2b7e4cbf7ac67a1c0545d5a21ee2eee314e2bbd1f8a34c9edddf517640bc5aa9
```
