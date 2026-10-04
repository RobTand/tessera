# Fused mHC post/pre for GLM-5.3 prefill (#783)

Status 2026-10-04: default-off; **bitwise to stock** on GB10 (GPU probe,
60/60 cases at every tile height); **2.6-3.0 ms per 2048-token chunk per
rank faster** at the served SP shape in two matched microbenchmarks, not yet
served.
Receipts are under "Measured".

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
| post | Source `v = post[i]*x; v += comb[j][i]*res[j]`, j = 0..3, which nvcc contracted as `v = comb[0][i]*res[0]` (the FMUL), `v = fma(post[i], x, v)`, then `fma(comb[j][i], res[j], v)` for j = 1..3, bf16 RN. Read by register provenance in the SASS (comb from the 256-bit loads, post from the 128-bit one, x from its own pointer); the other contraction differs on ~1.6e-5 of outputs (first GPU gate, PB `f068e908`). | `__fmul_rn` on comb[0]·res[0], then `__fmaf_rn` in that order |
| GEMM | `mma.sync m16n8k8 .tf32` on the fp32 bits of bf16 A and the raw fp32 B bits. K blocks of 64 in order within the split, 8 k-steps. Fragments: a0 = A[g][t], a1 = A[g+8][t], a2 = A[g][t+4], a3 = A[g+8][t+4]. sqrsum per lane is `+= a0*a0 + a2*a2`, then xor 2, xor 1. | The same instruction (`HMMA.1688.F32.TF32` in both SASS), fragments and order. One warp per (split, n-tile). |
| pre | Split partials summed from 0 in split order. `rsqrtf(rms/16384 + eps)`. IEEE division and full `expf` (no fast math: 17 MUFU.RCP + slow-path calls). Sinkhorn reductions are butterflies: row xor 2,1; column xor 8,4; max xor 2,1. The layer input is `fma(pre, x, +0)` chains; sumsq is FFMA across chunks, then rv order (rv&1)*8 + (rv>>1); the 64-thread sum is xor 32 via smem, then xor 16..1; output is `(bf16(ol)*r)*w`. | The same C expressions, compiled `-O3` without fast math (TileLang's default), the same thread-to-position map and the same reduction trees. |

## Design

Each CTA (256 threads) runs a persistent loop over token tiles of 16 or 32
rows (one or two m16 MMA tiles):

1. **post** streams x and the old residual (`ld.global.cs`) and writes the new
   residual with an L2 `evict_last` policy.
2. **GEMM** re-reads the new residual from L2 through a 4-stage cp.async ring
   that every thread fills (fn comes from L2 too), in the split's K order. The
   order is stream-major while post produces all four streams per h, so the
   tile must be buffered. Every (m-tile, n-tile) chain of the tile runs at
   once, up to two per warp; each chain is DeepGEMM's arithmetic. Partials go
   to a `[S, T, 24]` workspace.
3. **pre** runs one warp per token for the mixes and Sinkhorn, then 64-thread
   groups compute the layer input, reading the new residual from L2 once
   more.

Grid: one CTA per SM, which is what the 218 registers admit. Tile height:
32 rows exactly when that, and not 16, fits the site in one wave; else 16.
Calls stock runs at split > 1 stay stock. Neither choice changes a bit, and
both were set from the measurements below.

**What bounds it.** A tile's GEMM is about 2048 dependent TF32 MMAs. On GB10
that is about 67 ns per step, about 140 µs per chain, and it does not change
with ring depth (4 vs 10 stages, PB `5bb4c705`). Stock DeepGEMM pays the same
per step (2048 steps in 167 µs). Stock pays it once per site; the fused
kernel pays it once per wave of tiles, during which that CTA moves no DRAM.
At the served shape the site is one wave:

- post about 330 µs, DRAM-bound at about 228 GB/s;
- GEMM about 173 µs;
- pre about 55 µs.

So the fused site sits at about 1.9× the floor.

Tried and removed, as measured losers:
- **Overlapping** a tile's GEMM with the next tile's post on separate warps:
  slower everywhere (5.38 vs 4.05 ms at 8192).
- **A 10-stage ring:** no change.
- **Tiles of 48 or 64 rows:** GEMM 315–548 µs per tile, because the CTA's
  TF32 chains stop overlapping.

## Measured

All on GB10, image `5be13705`, served checkpoint layer-1 `hc_attn`/`hc_ffn`.
Times are medians of graph replay over L2-defeating input copies, with stock
and fused arms interleaved round by round.

**Bitwise**: PASSED, 60/60 cases × tile heights 16/32/48/64, plus graph
replay and determinism (PB `fa23b165`, sparky,
`/mnt/shared/tessera-measurements/mhc-fusion-783/bitwise-c52a7602/mhc_fused_probe.json`;
also `e50d9a9f` and `5f108ffe`). The cases: attn and ffn × 15 shapes,
including the SP shards 1024@2048 and 4096@8192, and ragged 2049 ×
realistic/adversarial. The kernel source is unchanged since.

**Timing** (PB `286a7d3b`, sparky, exclusive GPU,
`timing-c52a7602/mhc_fused_probe.json`), ms per site:

| Site, tokens@batch (split) | Stock | Fused (current rule) | Floor at 273 GB/s |
|---|---|---|---|
| attn 1024@2048 (1) — served SP | 0.605 | 0.583 (tile 32) | 0.308 |
| ffn 1024@2048 (1) — served SP | 0.615 | 0.579 (tile 32) | 0.308 |
| attn 2048 (1) | 1.302 | 1.206 (tile 16) | 0.616 |
| attn 4096@8192 (1) | 2.615 | 2.389 (tile 16) | 1.232 |
| attn 8192 (1) | 5.540 | 4.557 (tile 16) | 2.463 |
| attn 512 (6) | 0.222 | 0.343 → declines to stock | 0.154 |

**At the served shape**, 44 attn and 45 ffn fused sites per chunk:
44 × 0.022 + 45 × 0.036 ≈ **2.6 ms per 2048-token chunk per rank**
(3.0 ms in the confirmation run at the final head, PB `5f7e12ed`; tables in
`docs/measurements/2026-10-04-mhc-fused-783.md`), about 0.2%
of the 1157 ms chunk. The fused site is about 1.9× the floor; stock is about
2.0×.

**Not measured:** served end to end, TR3/KL, both-Spark power, and work per
joule. These belong to the kernels lead.

## Next levers (not done)

- **Stream the GEMM during post.** Produce the new residual stream-major
  (re-reading the old tile from L2 per stream) so the split-1 chain starts on
  stream 0 while streams 1–3 are produced. The GEMM tail would drop from
  about 173 µs to about a quarter of that. This is the only route to the floor
  at the one-wave served shape.
- **Profile the TF32 step cost with Nsight Compute.** At about 67 ns it is far
  above Ampere-class mma.sync latency. If it is an issue-rate limit, it bounds
  stock DeepGEMM too.
