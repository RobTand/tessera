# Fused mHC post/pre for GLM-5.3 prefill (#783)

Status 2026-10-04: source landed default-off; **no GPU result yet** (the
bitwise gate and the timing are queued through PrismaBuild). Nothing below
the "Measured" heading is a result until its receipt is named there.

## What it costs today

Nominated baseline (A8SE752VB, `608bbdf0`/v53/image `5be13705`, artifact
`glm53-a8-bf16menu-20260930`, SP on), rank-1 torch profile over 4 nominal
2048-token chunks
(`/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/A8SE752VB-val787-20261001/run/profile/`,
kernels attributed to `vllm::mhc_fused_post_pre_tilelang` by External id):

| Kernel | ms / 4 chunks | ms / chunk / rank | per call |
|---|---|---|---|
| `mhc_post_tilelang_kernel` | 118.075 | 29.52 | 332 µs |
| `sm120_tf32_hc_prenorm_gemm_impl` (split 1) | 59.658 | 14.91 | 168 µs |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 56.976 | 14.24 | 160 µs |
| **site total, 89 sites / chunk** | **234.709** | **58.68** | 660 µs |

The record's 86.069/87.196 ms is the op's nested "CUDA total" column, not
kernel time. 58.68 ms is the removable-at-most budget, and the floor is not
zero.

Each SP call is 1024 local tokens run at the full 2048-token batch's split (1),
which `SplitForcer` forces for exactness.

## Floor

Per token per site, DRAM must read x (8 KiB) and the old residual (32 KiB),
and write the new residual (32 KiB) and the layer input (8 KiB): **80 KiB**,
plus 80 B of mixes. At 273 GB/s that is 0.307 ms per 1024-token site, or
27.3 ms per chunk per rank over 89 sites. The stock passes move 144 KiB per
token: 72 for post, 32 for the GEMM and 40 for pre. Kernels already run at
0.74-0.87 of peak (#783 timed row `86920f49`), so bytes are the lever.

## Why not the issue's option 2

Option 2 runs the stock kernels over L2-sized token tiles. It was measured on
2026-10-01 (`probe-mhctile-20261001T154046Z-p0`, sparklina) and loses at the
served shape. At 1024 tokens and split 1, stock is 0.638 ms and tile 512 is
0.673 ms. The tiled GEMM's grid collapses: BLOCK_M 128 at split 1 gives 4 to
8 CTAs on 48 SMs. Tiling the stock kernels cannot fix the grid; one kernel per
tile can.

## The stock arithmetic the kernel reproduces

Sources are the image's own: vLLM `af5b4857e` modules, sha256-equal to the
`_INTERFACES` pins, and DeepGEMM `sm120_tf32_hc_prenorm_gemm.cuh` instantiated
as `<24, 16384, 128, 32, 64, S, 4, 256, 128>`. The TileLang device source and
cubins came from the probe cache. SASS was read with `cuobjdump` 13.4.

| Stage | Stock (evidence) | Fused |
|---|---|---|
| post | `v = post[i]*x; v = fma(comb[j][i], res[j], v)` for j = 0..3, bf16 RN. SASS: 32 FMUL + 128 FFMA per 32 outputs, no FADD. | `__fmul_rn`, then four `__fmaf_rn` in j order |
| GEMM | `mma.sync m16n8k8 .tf32` on the fp32 bits of bf16 A and the raw fp32 B bits. K blocks of 64 in order within the split, 8 k-steps. Fragments: a0 = A[g][t], a1 = A[g+8][t], a2 = A[g][t+4], a3 = A[g+8][t+4]. sqrsum per lane is `+= a0*a0 + a2*a2`, then xor 2, xor 1. | The same instruction (`HMMA.1688.F32.TF32` in both SASS), fragments and order. One warp per (split, n-tile). |
| pre | Split partials summed from 0 in split order. `rsqrtf(rms/16384 + eps)`. IEEE division and full `expf` (no fast math: 17 MUFU.RCP + slow-path calls). Sinkhorn reductions are butterflies: row xor 2,1; column xor 8,4; max xor 2,1. The layer input is `fma(pre, x, +0)` chains; sumsq is FFMA across chunks, then rv order (rv&1)*8 + (rv>>1); the 64-thread sum is xor 32 via smem, then xor 16..1; output is `(bf16(ol)*r)*w`. | The same C expressions, compiled `-O3` without fast math (TileLang's default), the same thread-to-position map and the same reduction trees. |

## Design

Each CTA (256 threads) processes 16-token tiles, one m16 MMA tile, in a
persistent loop:

1. **post** streams x and the old residual (`ld.global.cs`) and writes the
   new residual with an L2 `evict_last` policy.
2. **GEMM** re-reads the new residual from L2 in the split's K order. The
   order is stream-major, but post produces all four streams per h, so the
   tile must be buffered: 512 KiB per tile. fn comes from L2 (1.5 MiB,
   shared). Partials go to a `[S, T, 24]` workspace.
3. **pre** runs one warp per token for the mixes and Sinkhorn, then
   64-thread groups compute the layer input, reading the new residual from L2
   a final time.

Grid: as many tiles as half the L2 holds (`L2/2 / 512 KiB`, 24 on GB10), at
most one per SM. Numerics do not depend on the grid.

Known first-version limits, to be measured before optimizing:

- At split 1 only 3 of 8 warps run the GEMM, and its chain is about 2048
  dependent MMAs per tile.
- Phases are sequential within a CTA; overlap comes only from other CTAs.
- fn L2 traffic is 1.5 MiB per 16 tokens.

## Measured

None yet. Pending PrismaBuild:

- `experiments/mhc/mhc_fused_probe.py --parts bitwise,timing` (served
  checkpoint, both sites, 15 shape cases × realistic/adversarial, graph
  replay, interleaved stock/fused arms, grid sweep, power).
- `tests/test_mhc_fusion_cuda.py`.
