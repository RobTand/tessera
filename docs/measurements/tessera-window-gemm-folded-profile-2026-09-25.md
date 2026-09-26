# The dense window GEMM on the folded BF16 arithmetic: what it costs (2026-09-25)

**Result.** Folding the row scale into each decoded weight
(`window_gemm`'s `arithmetic="folded"`, tessera#614) costs **+5.0 % kernel
time at the median** over 75 (shape, rung, M) points at the GLM-5.3 dense
shapes, range +1.4 % to +8.2 %. The cost is largest at decode (M = 1, median
+7.3 %) and smallest at M = 512 to 2048 (median +3.5 %). Adding the `FOLDED`
switch does not move the epilogue kernel: its time on the new build is
0.2 % of the old build's at the median (range -4.4 % to +3.9 %, run-to-run
noise).

**What this is not.** It is not a quality measurement; bf16_route's docstring
keeps the served KL comparison of the two arithmetics (#45). It is not an
attestation either: no census ran, and the dense BF16 scope stays unattested
at contract v37 until one does.

---

## 1. What was measured

| | |
|---|---|
| driver | `experiments/profile_window_gemm_arithmetic.py` |
| before | `8d3c234ca` (epilogue only: that build has no `arithmetic` argument) |
| after | `36d26d706` (both arithmetics, interleaved per point: epilogue first, then folded) |
| image | `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`; torch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0 |
| device | NVIDIA GB10 (sparky), capability `[12, 1]`, 48 SMs, 140 W envelope |
| placement | one pool measurement action per run, pinned to sparky: before 2026-09-25T22:24:49Z to 22:27:31Z, after 22:57:16Z to 23:02:28Z |
| units | synthetic value-family units: random codes at the rung's per-column rates, a random bf16 table, a random positive row scale. Kernel cost depends on geometry and rates, not values |
| shapes | 12288 x 4096, 4096 x 12288, 16384 x 1536, 4096 x 16384, 2048 x 4096 (rows x cols) |
| rungs | q256 832, 1024, 1088 |
| M | 1, 8, 64, 512, 2048 |
| per point | 10 warm-up calls; kernel time over 50 calls from `torch.profiler` (CUDA activity, `_window_gemm_kernel` events only); wall time over 50 calls from CUDA events; board power from `nvidia-smi` at 0.1 s over a loop of at least 1.5 s |

The two result files are run logs, not tracked files:

| file | sha256 |
|---|---|
| `before-8d3c234ca.json` | `2d8161f82ba1713e46787e4f97ed1d0e327cba6dce83f72db93c36099759f659` |
| `after-36d26d706.json` | `3d93ebaf15e70ccb4ae5e22b4cc47f746a95597644a9c225a1db421a47d529ab` |

Both are in
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/r13-stageb-20260923/ws-serve-research-20260925/profiles-614/`.

## 2. Kernel time

Kernel time is in microseconds at q256 = 1024. The ratio column is the range
over the three rungs.

| shape (rows x cols) | M | epilogue, before | epilogue, after | folded, after | folded / epilogue |
|---|---:|---:|---:|---:|---:|
| 12288 x 4096 | 1 | 451.7 | 450.0 | 480.5 | 1.068-1.073 |
| 12288 x 4096 | 8 | 454.6 | 455.9 | 487.1 | 1.053-1.068 |
| 12288 x 4096 | 64 | 521.8 | 514.0 | 539.8 | 1.034-1.050 |
| 12288 x 4096 | 512 | 4195.2 | 4228.0 | 4476.2 | 1.023-1.059 |
| 12288 x 4096 | 2048 | 16795.8 | 17088.3 | 17666.1 | 1.030-1.034 |
| 4096 x 12288 | 1 | 640.8 | 637.3 | 684.1 | 1.073-1.076 |
| 4096 x 12288 | 8 | 653.5 | 648.9 | 692.9 | 1.050-1.068 |
| 4096 x 12288 | 64 | 750.3 | 762.3 | 796.5 | 1.045-1.050 |
| 4096 x 12288 | 512 | 4359.7 | 4400.4 | 4553.1 | 1.031-1.042 |
| 4096 x 12288 | 2048 | 16970.5 | 17106.8 | 17833.4 | 1.036-1.042 |
| 16384 x 1536 | 1 | 234.8 | 237.0 | 255.0 | 1.076-1.079 |
| 16384 x 1536 | 8 | 238.8 | 238.4 | 254.5 | 1.050-1.068 |
| 16384 x 1536 | 64 | 272.0 | 272.3 | 284.3 | 1.044-1.047 |
| 16384 x 1536 | 512 | 1967.8 | 1961.9 | 2037.4 | 1.039-1.054 |
| 16384 x 1536 | 2048 | 7887.4 | 7878.2 | 8171.7 | 1.037-1.050 |
| 4096 x 16384 | 1 | 900.4 | 905.5 | 962.2 | 1.063-1.082 |
| 4096 x 16384 | 8 | 913.8 | 916.0 | 977.8 | 1.052-1.067 |
| 4096 x 16384 | 64 | 1031.5 | 1031.0 | 1080.1 | 1.032-1.048 |
| 4096 x 16384 | 512 | 6454.3 | 6564.4 | 6770.2 | 1.026-1.031 |
| 4096 x 16384 | 2048 | 25401.0 | 25933.1 | 26505.5 | 1.014-1.027 |
| 2048 x 4096 | 1 | 108.9 | 108.8 | 115.8 | 1.065-1.073 |
| 2048 x 4096 | 8 | 108.5 | 108.1 | 115.7 | 1.051-1.071 |
| 2048 x 4096 | 64 | 123.6 | 122.8 | 128.7 | 1.039-1.048 |
| 2048 x 4096 | 512 | 709.2 | 707.4 | 747.8 | 1.035-1.057 |
| 2048 x 4096 | 2048 | 2714.8 | 2690.5 | 2822.7 | 1.037-1.049 |

Median folded / epilogue by M: 1.073 (M = 1), 1.053 (8), 1.045 (64),
1.035 (512), 1.036 (2048).

## 3. Power

The box-level series (Netdata, sparky, `nvidia_smi.gpu_power_draw`) read an
average of 59.3 W over the before window (min 4, max 90) and 78.8 W over the
after window (min 7, max 95), against a 140 W envelope. Neither arithmetic
loads the GPU to its envelope at any point here. So the kernel is limited by
its decode and issue rate, not by power.

The per-point power means in the result files are not a fair comparison
between the two arms. Each arm's power loop lasts 1.5 s, and the epilogue arm
runs first after every change of shape or M. At M = 1 the epilogue arm reads
42 W to 76 W and the folded arm 64 W to 83 W on the same point, which is the
clock ramp, not the arithmetic. Kernel time is the comparison this document
claims.

## 4. A finding outside this change

At decode the dense window GEMM runs well below the memory system. At
12288 x 4096, q256 = 1024, M = 1 the kernel reads about 25 MB of packed weight
in 450 us, about 56 GB/s. The before build shows the same, so the fold did not
cause it. It is a property of the kernel's decode path at small M and is filed
separately.
