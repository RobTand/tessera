# Probe a newer ptxas for f16 prefetch hint retention (issue 1192)

Date: 2026-10-10. Parent: tessera#739. Deliverable: measurement. No kernel
change. No toolchain pin change.

## Question

The f16 library emits 96 `prefetch.global.L1` hints in PTX, and ptxas
13.0.88 drops all 96 before SASS. A newer ptxas that keeps the hints
would give the parent a cheap prefetch lever instead of the cp.async
ring. This record asks which ptxas versions the pool holds, and repeats
the f7d34fc probe on each one it finds.

## Method

The probe builds the fused routed kernel translation unit twice with
nvcc under the f16 library flags, then counts the hint in PTX and SASS:

```bash
IMG=localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a
FLAGS="-O3 -lineinfo -std=c++17 -DTESSERA_ROUTED_FUSED_FP8=1 -DTESSERA_ROUTED_FUSED_MMA8=0 -gencode arch=compute_121,code=sm_121"
nvcc $FLAGS $INC --ptx base.cu -o base.ptx
nvcc $FLAGS $INC -c base.cu -o base.o
cuobjdump -sass base.o > base.sass
grep -ci prefetch base.ptx
grep -ci prefetch base.sass
```

`base.cu` is the master source. `new.cu` is the same file with the
one-line prefetch gate (`FAMILY_FP8 ? A_PREFETCH`) plus its comment.
`$INC` is the torch plus Python include paths from the image.

The baseline run is PB action f7d34fc on sparklina in that image. Its
inputs match this checkout byte for byte: `base.cu` sha256 `8a1536df`
equals `src/tessera/serving/csrc/routed_fused_window.cu` here, and
`new.cu` sha256 `d8e7daa2` equals the prefetch source blob. The sweep
for newer toolchains reads `nvcc --version` and `ptxas --version` on
both Sparks, in every CUDA-bearing image and on the hosts.

## Results

One ptxas version exists on the pool, so the table has one version.
Both arms come from the f7d34fc log, which this writer read from the
pool archive:

| ptxas | arm | PTX prefetch lines | SASS prefetch lines |
|---|---|---|---|
| 13.0.88 (V13.0.88) | base source | 0 | 0 |
| 13.0.88 (V13.0.88) | prefetch source | 96 | 0 |

The 96 PTX hints die in ptxas. Both SASS binaries hold no prefetch
instruction. The baseline repeats the known negative result.

## Sweep: no newer ptxas is available

| Location | ptxas | Evidence |
|---|---|---|
| sparklina host `/usr/local/cuda`, `cuda-13`, `cuda-13.0` | 13.0.88 | PB 0749fb3a |
| sparky host `/usr/local/cuda`, `cuda-13`, `cuda-13.0` | 13.0.88 | PB 14a01780 |
| sparky `/usr/local/cuda-13.4` | none: `doc`, `gds`, `targets` only, no `bin` | PB 4d6290ba |
| sparklina images: spark-vllm nightly-20260929, f8dbe1a0-kpooltail1, 0afec8d4, glm-tp2-diag nccl230 and nccl229, lfm25-teacher, stage6-encoder, glm53-mia-sm121, vllm-openai v0.30.0 | 13.0.88 | PB 0749fb3a, d2d8e6a2 |
| sparky images: spark-vllm nightly-20260929, f8dbe1a0-kpooltail1, a5424378-mtpmap1, 0afec8d4, pinned-0afec8d4, vllm-openai latest, glm53-mia-sm121 | 13.0.88 | PB 14a01780 |
| dl380g10 host | none: no CUDA toolkit; torch 2.11.0+cpu only | PB 5ed91cbd |

Two fresh repeats were submitted and withdrawn: PB 9555a3fe waited 9
hours READY while the pool deferred CPU-only work behind GPU load, and
PB f94b6ab0 waited 25 minutes with the same deferral. The pool admits
no CPU-only compile on the Sparks under current GPU load, so the
baseline evidence rests on the archived f7d34fc run above.

## Verdict

The prefetch lever cannot survive on a newer toolchain: no newer
ptxas exists on the pool to test it. k2, the cp.async ring, is the
only path. When a newer ptxas arrives, repeat the probe above before
any timing claim. A change to the repo toolchain pin needs a decision
from Rob.

## What a hint count is not

A hint count is not a speed result. A hint count is not a wait result.
This record makes no timing claim and no claim about C2 or C4. It did
not measure kernel time, stall or wait fractions, profiler counters,
other libraries, other architectures, or other toolchains.
