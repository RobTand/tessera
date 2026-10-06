// One-pass mHC post + pre-norm GEMM + pre for GLM-5.3 prefill (#783).
//
// Replaces the stock split-k path of vLLM's mhc_fused_post_pre_tilelang
// (mhc_post_tilelang_kernel -> deep_gemm sm120_tf32_hc_prenorm_gemm_impl ->
// mhc_pre_big_fuse_with_norm_tilelang_kernel) for hc_mult 4, hidden 4096.
// Every output is required to be bitwise equal to that sequence; each phase
// below restates the stock arithmetic it reproduces, read from the stock
// source and its SASS (docs/design/mhc-fusion-783.md):
//
//  post  stock source: out = post[i]*x; out += comb[j][i]*res[j], j = 0..3.
//        nvcc contracted it as out = comb[0][i]*res[0] (the FMUL);
//        out = fma(post[i], x, out); out = fma(comb[j][i], res[j], out) for
//        j = 1..3; bf16 RN.  Read off mhc_post_tilelang_kernel's SASS by
//        register provenance (comb in the 256-bit loads, post in the 128-bit
//        one, x from its own pointer); the other contraction differs in
//        about 1.6e-5 of outputs.
//  gemm  per split s: K blocks of 64 in order, eight m16n8k8 TF32 MMAs per
//        block on the fp32 bits of the bf16 residual and TF32-rounded fn
//        values, fragments laid out as DeepGEMM's; sqrsum per lane over
//        columns t and t+4, then xor-2, xor-1 lane sums.
//  pre   the TileLang kernel's mixes, Sinkhorn and RMSNorm arithmetic and
//        reduction trees, per token (split partials summed in split order).
//
// The tile's new residual is written once, then read by the GEMM (whose K
// order is stream-major) and pre. The design assumes those rereads hit L2.
// Read x and the old residual, write the new residual and layer input: an
// 80 KiB/token theoretical external-memory floor, not measured DRAM traffic.
#include <cuda_bf16.h>
#include <math_constants.h>
#ifndef __CUDACC_RTC__  // NVRTC (device-only compile checks) has neither header.
#include <cuda_runtime.h>
#include <stdint.h>
#else
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif

namespace tessera_mhc {

constexpr int HC = 4;
constexpr int HIDDEN = 4096;
constexpr int K = HC * HIDDEN;          // 16384
constexpr int NMIX = HC * 2 + HC * HC;  // 24
constexpr int TM_MAX = 64;              // most tokens per tile: four m16 MMA tiles
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int BLOCK_K = 64;
constexpr int KBLOCKS = K / BLOCK_K;    // 256
constexpr int NTILES = NMIX / 8;        // 3 (DeepGEMM's fourth, zero-B tile is never stored)
constexpr int GROUPS = THREADS / 64;    // 64-thread layer-input groups

struct Params {
  const __nv_bfloat16* x;         // [T, HIDDEN]
  const __nv_bfloat16* residual;  // [T, HC, HIDDEN]
  const float* post;              // [T, HC]
  const float* comb;              // [T, HC, HC]
  const float* fn;                // [NMIX, K]
  const float* scale;             // [3]
  const float* base;              // [NMIX]
  const __nv_bfloat16* norm_w;    // [HIDDEN]
  __nv_bfloat16* residual_out;    // [T, HC, HIDDEN]
  float* post_mix;                // [T, HC]
  float* comb_mix;                // [T, HC*HC]
  __nv_bfloat16* layer_input;     // [T, HIDDEN]
  float* part;                    // [splits, T, NMIX] workspace
  float* sqrsum;                  // [splits, T] workspace
  int tokens;
  int splits;
  float rms_eps, pre_eps, sinkhorn_eps, post_mult, norm_eps;
  int sinkhorn_repeat;
  unsigned long long* clocks;     // [tiles, 5] %globaltimer ns at phase boundaries, or null
  int tm;                         // tokens per tile: a multiple of 16, at most TM_MAX
};

// Diagnostic only: when the caller passes a buffer, thread 0 stamps each tile's
// phase boundaries: start, post done, GEMM done, mixes done, layer input done.
__device__ __forceinline__ void stamp(const Params& p, int tile, int k) {
  if (p.clocks != nullptr && threadIdx.x == 0) {
    unsigned long long ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
    p.clocks[tile * 5 + k] = ns;
  }
}

__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__device__ __forceinline__ void st_evict_last(void* ptr, uint4 v, uint64_t policy) {
  asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;"
               :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(policy) : "memory");
}

// Stock loads fn through a TFLOAT32 tensor map. Our cp.async copy retains
// FP32, so reproduce that conversion before supplying B to the TF32 MMA.
// Already-representable checkpoint values are unchanged; arbitrary FP32
// weights must not silently lose their low mantissa bits by truncation.
__device__ __forceinline__ uint32_t tf32_bits(float value) {
  uint32_t bits;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value));
  return bits;
}

__device__ __forceinline__ void tf32_mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  // DeepGEMM mma::sm120::tf32_mma, verbatim.
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void unpack8(uint4 v, float (&f)[8]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 p = __bfloat1622float2(h[i]);
    f[2 * i] = p.x;
    f[2 * i + 1] = p.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  uint4 v;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __float22bfloat162_rn(make_float2(f[2 * i], f[2 * i + 1]));
  return v;
}

__device__ __forceinline__ void named_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(threads) : "memory");
}

// ---------------------------------------------------------------- post
__device__ void post_phase(const Params& p, int t0, int tm, uint64_t keep) {
  __shared__ float s_post[TM_MAX][HC];
  __shared__ float s_comb[TM_MAX][HC * HC];
  for (int i = threadIdx.x; i < tm * HC * HC; i += THREADS) {
    int tok = t0 + i / (HC * HC);
    s_comb[i / (HC * HC)][i % (HC * HC)] = tok < p.tokens ? p.comb[(int64_t)tok * HC * HC + i % (HC * HC)] : 0.f;
  }
  for (int i = threadIdx.x; i < tm * HC; i += THREADS) {
    int tok = t0 + i / HC;
    s_post[i / HC][i % HC] = tok < p.tokens ? p.post[(int64_t)tok * HC + i % HC] : 0.f;
  }
  __syncthreads();
  constexpr int H8 = HIDDEN / 8;              // 512 eight-element groups per stream
  const int ITEMS = tm * H8;
  constexpr int UNROLL = 4;
  for (int base = threadIdx.x; base < ITEMS; base += THREADS * UNROLL) {
    uint4 xv[UNROLL], rv[UNROLL][HC];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int item = base + u * THREADS;
      int tl = item / H8, h8 = item % H8, tok = t0 + tl;
      if (item < ITEMS && tok < p.tokens) {
        xv[u] = __ldcs(reinterpret_cast<const uint4*>(p.x + (int64_t)tok * HIDDEN) + h8);
#pragma unroll
        for (int j = 0; j < HC; ++j)
          rv[u][j] = __ldcs(reinterpret_cast<const uint4*>(p.residual + ((int64_t)tok * HC + j) * HIDDEN) + h8);
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int item = base + u * THREADS;
      int tl = item / H8, h8 = item % H8, tok = t0 + tl;
      if (item >= ITEMS || tok >= p.tokens) continue;
      float d[8], b[HC][8];
      unpack8(xv[u], d);
#pragma unroll
      for (int j = 0; j < HC; ++j) unpack8(rv[u][j], b[j]);
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        float o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float v = __fmul_rn(s_comb[tl][i], b[0][e]);
          v = __fmaf_rn(s_post[tl][i], d[e], v);
#pragma unroll
          for (int j = 1; j < HC; ++j) v = __fmaf_rn(s_comb[tl][j * HC + i], b[j][e], v);
          o[e] = v;
        }
        st_evict_last(reinterpret_cast<uint4*>(p.residual_out + ((int64_t)tok * HC + i) * HIDDEN) + h8,
                      pack8(o), keep);
      }
    }
  }
}

// ---------------------------------------------------------------- pre-norm GEMM
// DeepGEMM's math warp, per chain: one (m-tile, n-tile) pair of the tile, fed
// by a cp.async ring every thread fills.  K blocks run 0..255 in order; split
// s owns the contiguous range DeepGEMM gives it, and each chain's accumulator
// (and, on n-tile 0, its m-tile's sqrsum) is flushed and reset at every
// split's end, so each split's chain is DeepGEMM's: same blocks, same k-steps,
// same order.  A chain is ~2048 dependent TF32 MMAs (~135 us on GB10, PB
// 5bb4c705 phase clocks, at any ring depth), so a tile runs all of its chains
// at once -- up to two per warp, interleaved like DeepGEMM's n-tiles -- and
// the caller sizes tiles so a site is one wave.
constexpr int NSTAGE = 4;
constexpr int MT_MAX = TM_MAX / 16;
constexpr int CHAINS_PER_WARP = (MT_MAX * NTILES + WARPS - 1) / WARPS;  // 2
constexpr int A_LD = BLOCK_K + 8;   // bf16 per staged A row (144 B: 16B-aligned, conflict-free)
constexpr int B_LD = BLOCK_K + 4;   // fp32 per staged B row (272 B)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  // src-size 0 zero-fills: rows past the batch read as zero, as DeepGEMM's TMA fills them.
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :: "r"(dst), "l"(gmem), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_stages() {
  asm volatile("cp.async.wait_group %0;" :: "n"(NSTAGE - 2) : "memory");
}

struct GemmSmem {
  __nv_bfloat16 a[NSTAGE][TM_MAX][A_LD];
  float b[NSTAGE][NMIX][B_LD];
};

__device__ __forceinline__ void stage_block(const Params& p, GemmSmem& sm, int t0, int tm, int kb, int st) {
  // tm rows x 8 chunks of A, then 24 rows x 16 chunks of B, 16 bytes each.
  for (int c = threadIdx.x; c < tm * 8 + NMIX * 16; c += THREADS) {
    if (c < tm * 8) {
      const int r = c / 8, q = c % 8, tok = t0 + r;
      const bool ok = tok < p.tokens;
      cp_async16(&sm.a[st][r][q * 8], p.residual_out + (int64_t)(ok ? tok : 0) * K + kb * BLOCK_K + q * 8, ok);
    } else {
      const int cb = c - tm * 8, n = cb / 16, q = cb % 16;
      cp_async16(&sm.b[st][n][q * 4], p.fn + (int64_t)n * K + kb * BLOCK_K + q * 4, true);
    }
  }
}

__device__ void gemm_phase(const Params& p, GemmSmem& sm, int t0, int tm) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int chains = (tm / 16) * NTILES;
  const int per = KBLOCKS / p.splits, rem = KBLOCKS % p.splits;
  bool on[CHAINS_PER_WARP];
  int mt[CHAINS_PER_WARP], nt[CHAINS_PER_WARP];
  float acc[CHAINS_PER_WARP][4], sq0[CHAINS_PER_WARP], sq1[CHAINS_PER_WARP];
#pragma unroll
  for (int j = 0; j < CHAINS_PER_WARP; ++j) {
    const int c = warp + j * WARPS;
    on[j] = c < chains;
    mt[j] = c / NTILES;
    nt[j] = c % NTILES;
    acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    sq0[j] = sq1[j] = 0.f;
  }
  int s = 0, split_end = per + (0 < rem);
#pragma unroll
  for (int st = 0; st < NSTAGE - 1; ++st) {
    stage_block(p, sm, t0, tm, st, st);
    cp_async_commit();
  }
  for (int kb = 0; kb < KBLOCKS; ++kb) {
    cp_async_wait_stages();
    __syncthreads();  // block kb is resident; every warp is done with the stage refilled below
    const int nk = kb + NSTAGE - 1;
    if (nk < KBLOCKS) stage_block(p, sm, t0, tm, nk, nk % NSTAGE);
    cp_async_commit();
    const int st = kb % NSTAGE;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int k0 = ks * 8 + t;
#pragma unroll
      for (int j = 0; j < CHAINS_PER_WARP; ++j) {
        if (!on[j]) continue;
        const int ra = mt[j] * 16 + g, nb = nt[j] * 8 + g;
        float fa0 = __bfloat162float(sm.a[st][ra][k0]);
        float fa1 = __bfloat162float(sm.a[st][ra + 8][k0]);
        float fa2 = __bfloat162float(sm.a[st][ra][k0 + 4]);
        float fa3 = __bfloat162float(sm.a[st][ra + 8][k0 + 4]);
        if (nt[j] == 0) {
          sq0[j] += fa0 * fa0 + fa2 * fa2;
          sq1[j] += fa1 * fa1 + fa3 * fa3;
        }
        uint32_t a[4] = {__float_as_uint(fa0), __float_as_uint(fa1), __float_as_uint(fa2), __float_as_uint(fa3)};
        uint32_t b[2] = {tf32_bits(sm.b[st][nb][k0]), tf32_bits(sm.b[st][nb][k0 + 4])};
        tf32_mma(acc[j], a, b);
      }
    }
    if (kb + 1 == split_end) {
#pragma unroll
      for (int j = 0; j < CHAINS_PER_WARP; ++j) {
        const int r0 = t0 + mt[j] * 16 + g, r1 = r0 + 8;
        const bool v0 = on[j] && r0 < p.tokens, v1 = on[j] && r1 < p.tokens;
        if (on[j] && nt[j] == 0) {
          // DeepGEMM math::warp_reduce_sum<4>: xor 2, then xor 1.
          float r0s = sq0[j], r1s = sq1[j];
          r0s = r0s + __shfl_xor_sync(0xffffffffu, r0s, 2);
          r0s = r0s + __shfl_xor_sync(0xffffffffu, r0s, 1);
          r1s = r1s + __shfl_xor_sync(0xffffffffu, r1s, 2);
          r1s = r1s + __shfl_xor_sync(0xffffffffu, r1s, 1);
          if (t == 0) {
            if (v0) p.sqrsum[(int64_t)s * p.tokens + r0] = r0s;
            if (v1) p.sqrsum[(int64_t)s * p.tokens + r1] = r1s;
          }
          sq0[j] = sq1[j] = 0.f;
        }
        const int col = nt[j] * 8 + t * 2;
        if (v0) *reinterpret_cast<float2*>(p.part + ((int64_t)s * p.tokens + r0) * NMIX + col) = make_float2(acc[j][0], acc[j][1]);
        if (v1) *reinterpret_cast<float2*>(p.part + ((int64_t)s * p.tokens + r1) * NMIX + col) = make_float2(acc[j][2], acc[j][3]);
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      }
      ++s;
      split_end += per + (s < rem);
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// ---------------------------------------------------------------- pre: mixes
// One warp per token: mhc_pre_big_fuse_with_norm's warp 0 (post mix, Sinkhorn
// comb mix) and its threads 32..35 (pre mix), the same expressions.
__device__ void mixes_token(const Params& p, int tok, float* s_mix, float* s_pre) {
  const int lane = threadIdx.x % 32;
  float rms = 0.f;
  for (int i_split = 0; i_split < p.splits; ++i_split)
    rms = (rms + p.sqrsum[(int64_t)i_split * p.tokens + tok]);
  rms = rsqrtf(((rms / 16384.f) + p.rms_eps));
  if (lane < NMIX) {
    float mixes = 0.f;
    for (int i_split = 0; i_split < p.splits; ++i_split)
      mixes = (mixes + p.part[((int64_t)i_split * p.tokens + tok) * NMIX + lane]);
    s_mix[lane] = (mixes * rms);
  }
  __syncwarp();
  const float* scale = p.scale;
  const float* base = p.base;
  if (lane < 4)
    p.post_mix[(int64_t)tok * HC + lane] =
        ((1.f / (1.f + expf((0.f - ((s_mix[lane + 4] * scale[1]) + base[lane + 4]))))) * p.post_mult);
  float cm = ((s_mix[(lane & 15) + 8] * scale[2]) + base[(lane & 15) + 8]);
  float row_max = -CUDART_INF_F;
  row_max = max(row_max, cm);
  row_max = max(row_max, __shfl_xor_sync(0xffffffffu, row_max, 2));
  row_max = max(row_max, __shfl_xor_sync(0xffffffffu, row_max, 1));
  cm = expf((cm - row_max));
  float row_sum = 0.f;
  row_sum = (row_sum + cm);
  row_sum = row_sum + __shfl_xor_sync(0xffffffffu, row_sum, 2);
  row_sum = row_sum + __shfl_xor_sync(0xffffffffu, row_sum, 1);
  cm = ((cm / row_sum) + p.sinkhorn_eps);
  float col_sum = 0.f;
  col_sum = (col_sum + cm);
  col_sum = col_sum + __shfl_xor_sync(0xffffffffu, col_sum, 8);
  col_sum = col_sum + __shfl_xor_sync(0xffffffffu, col_sum, 4);
  cm = (cm / (col_sum + p.sinkhorn_eps));
  for (int it = 0; it < p.sinkhorn_repeat - 1; ++it) {
    row_sum = 0.f;
    row_sum = (row_sum + cm);
    row_sum = row_sum + __shfl_xor_sync(0xffffffffu, row_sum, 2);
    row_sum = row_sum + __shfl_xor_sync(0xffffffffu, row_sum, 1);
    cm = (cm / (row_sum + p.sinkhorn_eps));
    col_sum = 0.f;
    col_sum = (col_sum + cm);
    col_sum = col_sum + __shfl_xor_sync(0xffffffffu, col_sum, 8);
    col_sum = col_sum + __shfl_xor_sync(0xffffffffu, col_sum, 4);
    cm = (cm / (col_sum + p.sinkhorn_eps));
  }
  if ((lane >> 4) == 0) p.comb_mix[(int64_t)tok * HC * HC + (lane & 15)] = cm;
  if (lane >= 16 && lane < 16 + HC) {
    const int q = lane - 16;
    s_pre[q] = ((1.f / (1.f + expf((0.f - ((s_mix[q] * scale[0]) + base[q]))))) + p.pre_eps);
  }
  __syncwarp();
}

// ---------------------------------------------------------------- pre: layer input
// One 64-thread group per token: thread q is the stock kernel's thread 32+q.
// Position of ol[i] in chunk c: c*1024 + (i>>3)*512 + q*8 + (i&7).
__device__ void layer_input_token(const Params& p, int tok, const float* pre_mix, int group, float* red) {
  const int q = threadIdx.x % 64;
  const __nv_bfloat16* res = p.residual_out + (int64_t)tok * K;
  float sumsq_per_pos[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) sumsq_per_pos[i] = 0.f;
  uint4 olb[4][2];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    float xl[64];
#pragma unroll
    for (int j = 0; j < HC; ++j)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float f[8];
        unpack8(__ldcs(reinterpret_cast<const uint4*>(res + j * HIDDEN + c * 1024 + half * 512 + q * 8)), f);
#pragma unroll
        for (int v = 0; v < 8; ++v) xl[j * 16 + half * 8 + v] = f[v];
      }
    float ol[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) ol[i] = 0.f;
    for (int i_hc = 0; i_hc < HC; ++i_hc) {
      float pre = pre_mix[i_hc];
#pragma unroll
      for (int i = 0; i < 16; ++i) ol[i] = (ol[i] + (pre * xl[((i_hc * 16) + i)]));
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) sumsq_per_pos[i] = __fmaf_rn(ol[i], ol[i], sumsq_per_pos[i]);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float f[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) f[v] = ol[half * 8 + v];
      olb[c][half] = pack8(f);
    }
  }
  float sumsq = 0.f;
#pragma unroll
  for (int rv = 0; rv < 16; ++rv) sumsq = (sumsq + sumsq_per_pos[(((rv & 1) * 8) + (rv >> 1))]);
  // tl::AllReduce<SumOp, 64, 1, 32, NamedBarrier<64>>: xor 32 through shared memory, then shuffles.
  named_sync(1 + group, 64);
  red[q] = sumsq;
  named_sync(1 + group, 64);
  sumsq = sumsq + red[q ^ 32];
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) sumsq = sumsq + __shfl_xor_sync(0xffffffffu, sumsq, off);
  const float rsqrt_norm = rsqrtf(((sumsq / 4096.f) + p.norm_eps));
#pragma unroll
  for (int c = 0; c < 4; ++c)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int h = c * 1024 + half * 512 + q * 8;
      float o[8], w[8];
      unpack8(olb[c][half], o);
      unpack8(*reinterpret_cast<const uint4*>(p.norm_w + h), w);
#pragma unroll
      for (int v = 0; v < 8; ++v) o[v] = __fmul_rn(__fmul_rn(o[v], rsqrt_norm), w[v]);
      *reinterpret_cast<uint4*>(p.layer_input + (int64_t)tok * HIDDEN + h) = pack8(o);
    }
}

__global__ void __launch_bounds__(THREADS) mhc_fused_post_pre_kernel(Params p) {
  __shared__ float s_mix[TM_MAX][32];
  __shared__ float s_pre[TM_MAX][HC];
  __shared__ float s_red[GROUPS][64];
  extern __shared__ __align__(16) unsigned char dyn_smem[];  // GemmSmem, > 48 KiB
  GemmSmem& s_gemm = *reinterpret_cast<GemmSmem*>(dyn_smem);
  const uint64_t keep = policy_evict_last();
  const int warp = threadIdx.x / 32;
  const int tm = p.tm;
  const int ntiles = (p.tokens + tm - 1) / tm;
  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    const int t0 = tile * tm;
    stamp(p, tile, 0);
    post_phase(p, t0, tm, keep);
    __syncthreads();
    stamp(p, tile, 1);
    gemm_phase(p, s_gemm, t0, tm);
    __syncthreads();
    stamp(p, tile, 2);
    for (int tl = warp; tl < tm; tl += WARPS)
      if (t0 + tl < p.tokens) mixes_token(p, t0 + tl, s_mix[tl], s_pre[tl]);
    __syncthreads();
    stamp(p, tile, 3);
    const int group = threadIdx.x / 64;
    for (int tl = group; tl < tm; tl += GROUPS)
      if (t0 + tl < p.tokens) layer_input_token(p, t0 + tl, s_pre[tl], group, s_red[group]);
    __syncthreads();
    stamp(p, tile, 4);
  }
}

}  // namespace tessera_mhc

#ifndef __CUDACC_RTC__
extern "C" int tessera_mhc_fused_post_pre(const tessera_mhc::Params* params, int grid, cudaStream_t stream) {
  if (params->tm < 16 || params->tm > tessera_mhc::TM_MAX || params->tm % 16) return static_cast<int>(cudaErrorInvalidValue);
  constexpr int smem = sizeof(tessera_mhc::GemmSmem);
  static const cudaError_t attr = cudaFuncSetAttribute(
      tessera_mhc::mhc_fused_post_pre_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (attr != cudaSuccess) return static_cast<int>(attr);
  tessera_mhc::mhc_fused_post_pre_kernel<<<grid, tessera_mhc::THREADS, smem, stream>>>(*params);
  return static_cast<int>(cudaGetLastError());
}
#endif
