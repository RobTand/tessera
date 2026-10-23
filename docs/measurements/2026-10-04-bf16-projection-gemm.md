# GLM-5.3-Flash prefill: the BF16 projection GEMM budget on GB10

**Status:** attribution from the nominated serve traces, plus kernel screens on
one GB10 at the served shapes. These are operator timings with seeded random
weights. No serve was run and no served gain is claimed.

Goal 2 needs about 338 ms removed from each 2048-token chunk per rank
(1157 → 819 ms, L8192 c1, TP2, MNBT 2048). The BF16 projection GEMMs
cost 154–159 ms of that chunk (record `kernels-2500-gemm-attribution-20261004`).
This note asks how much of that time a kernel change can remove, and how much
a quantized projection could.

In short:

- **cuBLAS is already at its best kernel for every large projection.** For
  each shape, cuBLASLt's heuristic offers 3 to 7 algorithms, and the default
  is the fastest of them. The best of ten Triton tiles beats it only on the KDA
  input projection, by 5% (about 4 ms per chunk). Fusing the MLA
  `q_a+kv_a` GEMM with the two indexer-K GEMMs saves 1.3 ms per chunk.
  Fusing `q_b` with indexer-Q saves 0.3 ms. The ceiling for BF16 kernel work on
  the projections is about 5–6 ms per chunk.
- **The large BF16 projections run at 80–87 TFLOP/s**, which is 65–71% of
  the measured `mma.sync` BF16 peak (123.1 TFLOP/s). cuBLAS reaches 97 only at
  M = 8192. Without changing the arithmetic, little is left at M = 2048.
- **E4M3 W8A8 (`torch._scaled_mm`, row-wise scales) runs the same shapes
  2.1–2.3× faster.** Over every BF16 projection and shared-expert GEMM in the
  chunk, the GEMMs save 86 ms. With vLLM's per-token quantiser charged per
  call, the net saving is 70 ms per chunk. Fusing the quantiser into the
  preceding norm would bring it close to 86.
- **Today's Tessera T-8 dense lane is slower than BF16 at M = 2048.** It
  takes 4.3–6.1 ms on the KDA input projection against BF16's 2.56 ms, and
  1.4–2.0 ms on `o_proj` against 0.78 ms. It runs at 14–20% of its own
  arithmetic limit and draws about 40 W. Allocating T-8 to these projections
  today would add prefill time.
- **The MLA absorbed BMMs look slow but are memory-bound.** These are the
  `W_UK` and `W_UV` products, outside the projection bucket, at 16 ms per
  chunk. They run at 18 and 32 TFLOP/s, yet each moves about 109 MB per call,
  which takes about 0.45 ms at the measured 232 GB/s. `W_UV` already runs at
  81% of that floor. For `W_UK`, a strided Triton BMM is bitwise equal and
  takes 0.549 ms against 0.877 ms served, which saves about 3.6 ms per chunk.
- **On sm_121, kernel choice did not change bits.** Every non-split-K
  cuBLASLt algorithm and every Triton tile screened here produced output
  bitwise equal to the default kernel. These kernels all walk K in order with
  one fp32 accumulator in `mma.sync`. Only split-K variants differed.

## Attribution (A8SE752VB traces)

The traces are rank 0 on sparklina and rank 1 on sparky, the paths and hashes
in record `kernels-2500-gemm-attribution-20261004`. Each `cat=kernel` event
is joined by External id to its launching `aten::mm`/`aten::bmm`, with input
dims and strides. Times are ms per 2048-token chunk, the sum over 4 chunks
divided by 4. Rate is the per-call median against the shape's FLOPs.

| Role | x calls/chunk | Shape (M = 2048) | Kernel (cuBLAS's pick) | ms/chunk r0 / r1 | TFLOP/s r0 / r1 |
|---|---|---|---|---|---|
| KDA fused input (q,k,v,gates) | 34 | 4096→12576 | `nvjet_sm121_tst_mma_128x208x64_2` | 88.3 / 86.0 | 81.4 / 83.5 |
| KDA o_proj | 34 | 4096→4096 | `nvjet_sm121_tst_mma_192x144x64_2` | 27.7 / 27.0 | 84.5 / 87.0 |
| MLA o_proj | 11 | 8192→4096 | `nvjet_sm121_tst_mma_192x144x64_2` | 20.5 / 18.7 | 74.7 / 80.6 |
| MLA q_b | 11 | 1536→8192 | `cutlass_80 ... 128x256_32x3` | 6.1 / 6.2 | 92.3 / 91.0 |
| KDA low-rank b-proj pair | 68 | 128→4096 | `cutlass_80 ... 256x128_32x3` | 6.0 / 6.0 | 24 (output-write bound) |
| MLA q_a+kv_a | 11 | 4096→2048 | `nvjet_sm121_tst_mma_128x176x64_2` | 4.6 / 4.1 | 83.0 / 91.4 |
| MLA indexer q | 11 | 1536→4096 | `cutlass_80 ... 128x256_32x3` | 3.2 / 3.2 | 88.7 / 87.7 |
| MLA indexer k (160, 128) | 22 | 4096→160, 128 | `nvjet_sm121_tst_mma_64x128x64_4` | 3.1 / 2.8 | 15–17 |
| **Projections** | | | | **159.5 / 154.0** | |
| MLA absorbed `W_UK` bmm | 11 | 32 x (2048x256 @ 256x512) | `cutlass_80_wmma ... 32x32_32x2` | 9.9 / 9.7 | 19.1 / 19.6 |
| MLA absorbed `W_UV` bmm | 11 | 32 x (2048x512 @ 512x256) | `cutlass_80 ... 64x64_32x6` | 6.1 / 5.9 | 31.3 / 32.3 |

The large `nvjet` kernels run with two pipeline stages and one 256-thread
block per SM (87 KB shared memory, 255 registers).

## Screen: kernel and precision options at the served shapes

PB `7c9355f1b8f9` ran on sparky, exclusive measurement, image
`spark-vllm-nccl230@sha256:5be13705` (torch 2.13.0+cu130), branch head
`1113ce7e6b`. It ran `experiments/t8r_speed/bench_proj_gemm.py` and wrote
`/mnt/shared/tessera-measurements/bf16proj-20261004/screen3-20261004T220358Z/proj/proj_gemm.json`.
Each cell reports the median of 5 windows of 40 back-to-back calls. Weight
copies rotate past 64 MB, so weights stream from DRAM as they do in a serve.

| Shape | calls | default ms | best cuBLASLt | best Triton | E4M3 GEMM | per-token quant | default ms/chunk | Triton saves | E4M3 net saves |
|---|---|---|---|---|---|---|---|---|---|
| kda_in | 34 | 2.614 | 2.630 | 2.491 | 1.223 | 0.096 | 88.9 | 4.2 | 44.0 |
| kda_o | 34 | 0.796 | 0.826 | 0.891 | 0.426 | 0.089 | 27.1 | 0 | 9.5 |
| kda_aux | 68 | 0.093 | 0.091 | 0.090 | 0.089 | 0.010 | 6.3 | 0.2 | -0.4 |
| mla_qa_kva | 11 | 0.431 | 0.451 | 0.476 | 0.237 | 0.092 | 4.7 | 0 | 1.1 |
| mla_qb | 11 | 0.592 | 0.588 | 0.646 | 0.408 | 0.013 | 6.5 | 0 | 1.9 |
| mla_idx_q | 11 | 0.321 | 0.309 | 0.324 | 0.200 | 0.013 | 3.5 | 0.1 | 1.2 |
| mla_idx_k160 | 11 | 0.107 | 0.102 | 0.103 | 0.031 | 0.092 | 1.2 | 0.1 | -0.2 |
| mla_idx_k128 | 11 | 0.103 | 0.095 (split-K) | 0.100 | 0.018 | 0.096 | 1.1 | 0.1 | -0.1 |
| mla_o | 11 | 1.698 | 1.741 | 1.828 | 0.758 | 0.211 | 18.7 | 0 | 8.0 |
| shared_gate_up | 33 | 0.412 | 0.431 | 0.467 | 0.238 | 0.097 | 13.6 | 0 | 2.6 |
| shared_down | 40 | 0.231 | 0.226 | 0.215 | 0.159 | 0.012 | 9.2 | 0.6 | 2.4 |
| **total** | | | | | | | **180.9** | **5.3** | **70.1** |

The bench's default timings reproduce the traces to within 1–3%. For
example, kda_in is 2.614 ms here against the trace's 2.591 / 2.526 ms
medians.

- "Per-token quant" is vLLM's `scaled_fp8_quant` (per-token, dynamic) on the
  GEMM's input. The net E4M3 column charges it once per call. Inputs shared
  between GEMMs, such as `q_b` and indexer-Q, could share one quant.
- The E4M3 weights carry one fp32 scale per output channel. That is the
  `_scaled_mm` row-wise ceiling, not a Tessera wire. Which projections may
  run at 8 bits is campaign's accuracy decision. This note prices only the
  speed side.
- Same-input concatenation, timed as one GEMM against the sum of its parts:
  `q_a+kv_a` + indexer-K 160 + indexer-K 128 (N = 2336) takes 0.527 ms
  against 0.641 ms separate, which saves 1.25 ms per chunk. `q_b` + indexer-Q
  (N = 12288) takes 0.883 ms against 0.913 ms, which saves 0.33 ms per chunk.

### The Tessera T-8 dense lane at M = 2048

PB `9bbcfe431ba1` (sparky) ran step 2 with
`experiments/t8r_speed/bench_dense_module.py --modules kda_in,o_proj --ms 2048 --refs`
and wrote
`/mnt/shared/tessera-measurements/bf16proj-20261004/screen1-20261004T213619Z/t8/bench_dense_module.json`.
It used seeded Gaussian weights and timed CUDA-graph replay, forward and
reverse.

| Module | BF16 `F.linear` | E4M3 `_scaled_mm` | T-8 fused q1024 | T-8 fused q1088 |
|---|---|---|---|---|
| kda_in (12448 x 4096) | 2.560 ms | 1.150 ms | 4.326 ms | 6.086 ms |
| o_proj (4096 x 4096) | 0.775 ms | 0.373 ms | 1.385 ms | 1.950 ms |

The fused lane reaches 14–20% of its own floor (`roof_frac`). That kernel
decodes the wire inside the GEMM, and it is decode-bound at this M, as the
routed kernel is (2026-09-30-fp8-prefill-roofline.md).

### The MLA absorbed BMMs

The same screen (`bmms`) timed both products at the served strides. The
inputs are head-major views of token-major tensors, and the weight's batch
stride is 2·K·N.

| Product | served | contiguous operands | best cuBLASLt (strided) |
|---|---|---|---|
| `W_UK` | 0.937 ms, 18.3 TFLOP/s, `wmma 32x32_32x2` | 0.896 ms, same kernel | 0.901 ms |
| `W_UV` | 0.536 ms, 32.1 TFLOP/s, `64x64_32x6` | 0.468 ms | 0.537 ms |

The rate is the wrong yardstick for these two products. Per call, `W_UK`
reads 33.5 MB of `q_nope` and 8.4 MB of weight and writes 67.1 MB. `W_UV`
reads 67.1 MB and writes 33.5 MB. That is 109 MB either way, against
17.2 GFLOP: 158 FLOP per byte. GB10's ridge point is
123.1 TFLOP/s ÷ 232.2 GB/s = 530 FLOP per byte. The byte floor is about
0.45 ms per call, and the FLOP floor is 0.14 ms.

PB `ef0152941e95` ran on sparky with the same image, branch head `b0aa7cf638`,
using `experiments/t8r_speed/bench_mla_bmm.py`. It wrote
`/mnt/shared/tessera-measurements/bf16proj-20261004/bmm1-20261004T221148Z/bmm/mla_bmm.json`.
Each cell reports the median of 7 windows of 40 calls. Each arm is checked
against the served call's output:

| Product | served | weight column-major per head | strided Triton BMM, best of 10 tiles | byte floor |
|---|---|---|---|---|
| `W_UK` | 0.877 ms | 0.702 ms, bitwise equal | 0.549 ms (128x128x64, 3 stages), bitwise equal | ~0.45 ms |
| `W_UV` | 0.556 ms | 0.553 ms, bitwise equal | 0.521 ms (64x128x64, 4 stages), bitwise equal | ~0.45 ms |

All ten Triton tiles were bitwise equal to the served call on both products.
At 11 calls per chunk, the `W_UK` saving is 3.6 ms per chunk with Triton, or
1.9 ms with only the weight layout changed. The `W_UV` saving is 0.4 ms.

## What a kernel change can remove, and what it cannot

Estimates are ms per 2048-token chunk per rank, from the operator screens
above. None of these is a served measurement.

| Option | Removable | Bitwise | Confidence | Note |
|---|---|---|---|---|
| (a) Better BF16 kernels: cuBLASLt algorithm choice | 0 | — | high | the default is the best algorithm the heuristic offers on every large shape |
| (a) Triton tile for kda_in | ~4 | yes | medium (one screen, 5% margin) | needs a Tessera-owned GEMM route for one Linear shape |
| (c) Same-input concatenation (MLA a-side, q_b + indexer-Q) | ~1.6 | per column, same K order | medium | needs a weight-load change in the MLA module |
| MLA `W_UK` strided BMM (outside the projection bucket) | ~3.6 (1.9 by layout only) | yes | medium | the call sits inside stock `MLAAttention.forward_impl`, which also serves decode capture |
| (b) E4M3 W8A8 projections and shared experts (`_scaled_mm`) | ~70 net (86 for the GEMMs alone) | no | medium-high for the speed side | accuracy and allocation are campaign's decision |
| (b) Through today's Tessera T-8 dense lane | negative (adds 81–160 on kda_in + kda_o alone) | no | high at these two shapes | the lane is decode-bound at M = 2048 |

The BF16 kernel options together remove about 9 ms, under 3% of the 338 ms
the target needs. No bitwise kernel change on this budget is a material
lever for Goal 2. The material lever is 8-bit arithmetic on the projections.
It needs two things: campaign must accept the accuracy, and the serve needs an
8-bit dense lane that runs at the E4M3 GEMM rate at prefill. The current T-8
fused dense lane takes 3.7–5.3× the `_scaled_mm` time on the same shapes.

No serving override was built. The largest bitwise win, `W_UK` at about
3.6 ms, would have to intercept `torch.bmm` inside stock vLLM's
`forward_impl`, on the same path that decode CUDA-graph capture uses (#702's
territory). The kernel and its bitwise checks stay in
`experiments/t8r_speed/bench_mla_bmm.py` for whoever next owns that forward
pass.

## Reproduce

```bash
# from a GB10 checkout (PB refuses class-scoped measurements submitted from celestia)
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py --cwd . --measurement --host-class gb10 \
  --exclusive --cpus 4 --demand mem_gb=24 \
  --container-image content:sha256:a0b85c050cdd73a00488f46e1f5a436d5fd31abbf09a0c23b3e51be54a176918 \
  --env ORACLE_IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a \
  -- bash experiments/t8r_speed/proj_gemm_action.sh OUT   # PROJ_STEPS="proj t8 bmm"
```
