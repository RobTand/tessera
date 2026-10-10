# The 16-bit E4M3 activation prefetch does not reach SASS (tessera#739)

Date: 2026-10-10. Head: this branch. Toolchain: CUDA 13.0.88, sm_121.

## Question

The fused routed chunk loop waits on the activation global one chunk
ahead. #746 L1-prefetches that row on the E4M3 instruction's one-run
launches. Does the same hint work on the 16-bit E4M3 library?

## Answer

No. The branch put `prefetch.global.L1` on the 16-bit library's routed
one-run launches. Direct nvcc builds show 96 prefetch instructions in the
new PTX and 0 in the base PTX, but 0 in both SASS binaries. The ptxas
stage drops the hint. The branch is reverted below. #739 stays open for
the 16-bit library.

## Method

- A/B harness: `experiments/t8r_speed/f16_prefetch_ab.sh` over
  `bench_pairs.py` on the f16 library (`--library e4m3`). Base arm is
  master's source, new arm is this checkout. Cases `r4` (one-run R1024
  pair), `r4+q` (two-run R1088 pair) and `r3+q` (two-run R832 pair),
  modes 0 (gate/up) and 2 (down), M 1, 512 and 2048. Deterministic
  inputs, forward pass then reverse pass. Each cell hashes its output
  and records the wall median.
- PTX and SASS probe: `experiments/t8r_speed/f16_prefetch_ptx.sh`.
  It builds the kernel translation unit twice with nvcc under the f16
  library's own flags (`-DTESSERA_ROUTED_FUSED_FP8=1`,
  `-DTESSERA_ROUTED_FUSED_MMA8=0`, `-gencode arch=compute_121,code=sm_121`)
  and counts the prefetch in the PTX and in `cuobjdump -sass`.
- Oracle GPU tests: `tests/test_routed_fused_window.py` and
  `tests/test_dense_fused_window.py` with `--strict-cuda` in the pinned
  serving image.
- One GB10 each (sparky for the A/B runs, sparklina for the probe).

## Results

### Bitwise equality

All 54 timed cells (3 runs of 18) hash equal across master and branch.
The oracle run passes 724 tests with 0 failures, 709 tests on the
device, 0 skips, under `--strict-cuda`. The hint changes no byte.

### Timing (new over master, forward and reverse passes)

| Run | r4 m0 M512 | r4 m0 M2048 | r4 m2 M512 | r4 m2 M2048 | two-run range |
|---|---|---|---|---|---|
| `a28bed10` | 0.904 / 0.926 | 0.890 / 0.901 | 0.957 / 0.944 | 0.938 / 0.946 | 0.987-1.018 |
| `ba17c879` | 0.965 / 0.969 | 0.951 / 0.964 | 0.952 / 0.947 | 0.938 / 0.941 | 0.976-1.018 |
| `9c485e08` | 1.003 / 1.007 | 0.967 / 0.990 | 0.956 / 0.953 | 0.955 / 0.954 | 0.958-1.017 |

Mode 2 (down) reads 0.94-0.96 in all six passes. Mode 0 (gate/up)
reads 0.90-1.01 across sessions. Two-run pairs read 0.96-1.02 in
single passes with drift-symmetric means at 0.99-1.01; the base arm
itself drifts 1.7% between passes, so this band is noise.

### The hint never reaches the binary (PB `f7d34fc`)

| Arm | PTX prefetch lines | SASS prefetch lines |
|---|---|---|
| master source | 0 | 0 |
| branch source | 96 (`prefetch.global.L1 [%rd...]`) | 0 |

The 96 PTX prefetches die in ptxas. Both SASS binaries hold no
prefetch instruction (case-insensitive count over the full 90-91 MB
dumps; the torch-built libraries read the same 0 and 0). The timing
deltas above are scheduling side effects of dead PTX, not served
prefetching. They cannot meet the issue's R1024 or NCU criteria by
construction, and the mode-0 spread shows they do not even hold
across sessions.

### NCU

No verdict. The NCU steps ran green, but the profiler markers
captured no kernel data: each `ncu.csv` holds only the connect and
disconnect lines. The WarpStateStats table is empty. A PC-level
sampling run stays open work, but it would measure a binary that
holds no prefetch, so it is not run here.

## Revert

This branch reverts the gate (`PREFETCH_DISTANCE` back to master's
expression) and deletes its CPU test. The kernel matches master
again. The two harness scripts stay as the method record. Dense and
two-run launches never changed: the gate excluded them in every
revision, and their timing band above confirms it.

## What would reopen this

- A 16-bit activation lever that survives ptxas, such as a `cp.async`
  ring port of the MMA8 ring to the f16 library, with its own A/B.
- A newer ptxas that keeps the hint, with a repeat of the probe above
  before any timing claim.
