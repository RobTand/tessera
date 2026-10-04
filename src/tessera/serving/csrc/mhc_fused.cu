// One-pass mHC post + pre-norm GEMM + pre for GLM-5.3 prefill (#783).
//
// Replaces the stock split-k path of vLLM's mhc_fused_post_pre_tilelang
// (mhc_post_tilelang_kernel -> deep_gemm sm120_tf32_hc_prenorm_gemm_impl ->
// mhc_pre_big_fuse_with_norm_tilelang_kernel) for hc_mult 4, hidden 4096.
// Every output is required to be bitwise equal to that sequence; each phase
// below restates the stock arithmetic it reproduces, read from the stock
// source and its SASS (docs/design/mhc-fusion-783.md):
//
//  post  out = post[i]*x; out = fma(comb[j][i], res[j], out) for j = 0..3;
//        bf16 RN.  (FMUL then four FFMA, mhc_post_tilelang_kernel.)
//  gemm  per split s: K blocks of 64 in order, eight m16n8k8 TF32 MMAs per
//        block on the fp32 bits of the bf16 residual and the raw fp32 fn
//        bits, fragments laid out as DeepGEMM's; sqrsum per lane over
//        columns t and t+4, then xor-2, xor-1 lane sums.
//  pre   the TileLang kernel's mixes, Sinkhorn and RMSNorm arithmetic and
//        reduction trees, per token (split partials summed in split order).
//
// The tile's new residual is written once and read back from L2 twice (by
// the GEMM, whose K order is stream-major, and by the pre), so DRAM moves
// the floor: read x and the residual, write the residual and the layer input.
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
constexpr int TM = 16;                  // tokens per tile: one m16 MMA tile
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
};

__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__device__ __forceinline__ void st_evict_last(void* ptr, uint4 v, uint64_t policy) {
  asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;"
               :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(policy) : "memory");
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
__device__ void post_phase(const Params& p, int t0, uint64_t keep) {
  __shared__ float s_post[TM][HC];
  __shared__ float s_comb[TM][HC * HC];
  for (int i = threadIdx.x; i < TM * HC * HC; i += THREADS) {
    int tok = t0 + i / (HC * HC);
    s_comb[i / (HC * HC)][i % (HC * HC)] = tok < p.tokens ? p.comb[(int64_t)tok * HC * HC + i % (HC * HC)] : 0.f;
  }
  for (int i = threadIdx.x; i < TM * HC; i += THREADS) {
    int tok = t0 + i / HC;
    s_post[i / HC][i % HC] = tok < p.tokens ? p.post[(int64_t)tok * HC + i % HC] : 0.f;
  }
  __syncthreads();
  constexpr int H8 = HIDDEN / 8;              // 512 eight-element groups per stream
  constexpr int ITEMS = TM * H8;              // 8192 per tile
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
          float v = __fmul_rn(s_post[tl][i], d[e]);
#pragma unroll
          for (int j = 0; j < HC; ++j) v = __fmaf_rn(s_comb[tl][j * HC + i], b[j][e], v);
          o[e] = v;
        }
        st_evict_last(reinterpret_cast<uint4*>(p.residual_out + ((int64_t)tok * HC + i) * HIDDEN) + h8,
                      pack8(o), keep);
      }
    }
  }
}

// ---------------------------------------------------------------- pre-norm GEMM
// Task (split s, n-tile nt) on one warp, exactly as DeepGEMM's math warp for
// rows t0..t0+15 of split s, restricted to n-tile nt.
__device__ void gemm_task(const Params& p, int t0, int s, int nt) {
  const int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int per = KBLOCKS / p.splits, rem = KBLOCKS % p.splits;
  const int kb0 = s * per + min(s, rem);
  const int nblocks = per + (s < rem);
  const int r0 = t0 + g, r1 = t0 + g + 8;
  const bool v0 = r0 < p.tokens, v1 = r1 < p.tokens;
  const __nv_bfloat16* a0 = p.residual_out + (int64_t)(v0 ? r0 : 0) * K;
  const __nv_bfloat16* a1 = p.residual_out + (int64_t)(v1 ? r1 : 0) * K;
  const float* bn = p.fn + (int64_t)(nt * 8 + g) * K;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  float sqr_sum_acc_0 = 0.f, sqr_sum_acc_1 = 0.f;
  const __nv_bfloat16 zero = __float2bfloat16_rn(0.f);
  for (int blk = 0; blk < nblocks; ++blk) {
    const int kbase = (kb0 + blk) * BLOCK_K;
    __nv_bfloat16 av[8][4];
    float bv[8][2];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int k0 = kbase + ks * 8 + t;
      av[ks][0] = v0 ? a0[k0] : zero;
      av[ks][1] = v1 ? a1[k0] : zero;
      av[ks][2] = v0 ? a0[k0 + 4] : zero;
      av[ks][3] = v1 ? a1[k0 + 4] : zero;
      bv[ks][0] = __ldg(bn + k0);
      bv[ks][1] = __ldg(bn + k0 + 4);
    }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      float fa0 = __bfloat162float(av[ks][0]);
      float fa1 = __bfloat162float(av[ks][1]);
      float fa2 = __bfloat162float(av[ks][2]);
      float fa3 = __bfloat162float(av[ks][3]);
      if (nt == 0) {
        sqr_sum_acc_0 += fa0 * fa0 + fa2 * fa2;
        sqr_sum_acc_1 += fa1 * fa1 + fa3 * fa3;
      }
      uint32_t a[4] = {__float_as_uint(fa0), __float_as_uint(fa1), __float_as_uint(fa2), __float_as_uint(fa3)};
      uint32_t b[2] = {__float_as_uint(bv[ks][0]), __float_as_uint(bv[ks][1])};
      tf32_mma(acc, a, b);
    }
  }
  if (nt == 0) {
    // DeepGEMM math::warp_reduce_sum<4>: xor 2, then xor 1.
    float r0s = sqr_sum_acc_0, r1s = sqr_sum_acc_1;
    r0s = r0s + __shfl_xor_sync(0xffffffffu, r0s, 2);
    r0s = r0s + __shfl_xor_sync(0xffffffffu, r0s, 1);
    r1s = r1s + __shfl_xor_sync(0xffffffffu, r1s, 2);
    r1s = r1s + __shfl_xor_sync(0xffffffffu, r1s, 1);
    if (t == 0) {
      if (v0) p.sqrsum[(int64_t)s * p.tokens + r0] = r0s;
      if (v1) p.sqrsum[(int64_t)s * p.tokens + r1] = r1s;
    }
  }
  const int col = nt * 8 + t * 2;
  if (v0) *reinterpret_cast<float2*>(p.part + ((int64_t)s * p.tokens + r0) * NMIX + col) = make_float2(acc[0], acc[1]);
  if (v1) *reinterpret_cast<float2*>(p.part + ((int64_t)s * p.tokens + r1) * NMIX + col) = make_float2(acc[2], acc[3]);
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
  __shared__ float s_mix[TM][32];
  __shared__ float s_pre[TM][HC];
  __shared__ float s_red[GROUPS][64];
  const uint64_t keep = policy_evict_last();
  const int warp = threadIdx.x / 32;
  const int ntiles = (p.tokens + TM - 1) / TM;
  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    const int t0 = tile * TM;
    post_phase(p, t0, keep);
    __syncthreads();
    for (int task = warp; task < p.splits * NTILES; task += WARPS)
      gemm_task(p, t0, task / NTILES, task % NTILES);
    __syncthreads();
    for (int tl = warp; tl < TM; tl += WARPS)
      if (t0 + tl < p.tokens) mixes_token(p, t0 + tl, s_mix[tl], s_pre[tl]);
    __syncthreads();
    const int group = threadIdx.x / 64;
    for (int tl = group; tl < TM; tl += GROUPS)
      if (t0 + tl < p.tokens) layer_input_token(p, t0 + tl, s_pre[tl], group, s_red[group]);
    __syncthreads();
  }
}

}  // namespace tessera_mhc

#ifndef __CUDACC_RTC__
extern "C" int tessera_mhc_fused_post_pre(const tessera_mhc::Params* params, int grid, cudaStream_t stream) {
  tessera_mhc::mhc_fused_post_pre_kernel<<<grid, tessera_mhc::THREADS, 0, stream>>>(*params);
  return static_cast<int>(cudaGetLastError());
}
#endif
