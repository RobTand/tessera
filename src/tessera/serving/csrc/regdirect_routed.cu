// The register-direct routed window kernel (eng-regdirect-build, stage 1).
//
// WHAT IT COMPUTES.  T-8 (E4M3 grid, 14-bit window) routed gate/up with the
// SwiGLU epilogue (MODE 0) and routed down (MODE 2), from the FRAGMENT-ORDER
// wire that ``tessera.fragment_wire`` builds at load from the unit's on-disk
// column streams.  Weight w[n][k] is table[state], where state is the last 14
// bits of column k's MSB-first stream (rate R bits per row) ending after row n,
// starting from the column's stored start state.
//
// HOW IT DECODES.  The weight is the A operand of mma.sync m16n8k32 E4M3.  A
// warp owns 16 rows (tile rows g and g + 8 are rows 2g and 2g + 1) and walks K;
// lane (g, t) decodes its two rows of eight columns straight into its A
// fragment.  A row's window reaches into the two rows before it: lanes g - 1
// and g - 2 (shuffles), and for g < 2 the previous 16-row block's lanes 6 and 7
// (one load of that unit, or of the history unit in front of tile 0).  No
// decoded weight passes through shared memory; there is no producer warp and
// no barrier per k-step.  See docs/design/register-direct-routed.md.
//
// UNIT GROUPS.  A fragment unit carries two pair groups p in {0, 1}.  MODE 0:
// p is the projection (gate, up) over the SAME 32 columns.  MODE 2: p is a
// second group of 32 columns of the one down projection, and the two groups'
// products add into one output.  Either way a unit k-step is one rate.
//
// RATES.  Each expert has a rate profile: k-step slots [0, ksa) at RA, then
// [ksa, KS) at RB (RA == RB for a whole-bit expert).  The slots are the
// original k-steps sorted by rate; ``kperm`` maps a slot back, and the kernel
// gathers each activation chunk through it.  A task selects its expert's rate
// paths with a warp-uniform branch, so one launch serves every class.
//
// SCHEDULE.  A work unit is (superblock item, 128-row tile, K part).  The
// launch covers the absolute interval [item_off[e0] * U, item_off[e1] * U)
// with U = NT * P, split statically by CTA index (no claim counter).  With
// P > 1 each part writes fp32 partials and the last part to arrive sums them
// in part order (deterministic) and resets its counter.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <limits>

namespace {

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int BLOCK_ROWS = 16;
constexpr int TILE = WARPS * BLOCK_ROWS;        // 128 rows per task
constexpr int KSTEP = 32;                       // columns per pair group per unit k-step
constexpr int WIN = 14;
constexpr int TAB = 1 << WIN;
constexpr int HIST_LANES = 8;                   // a history unit keeps lanes (6, t) and (7, t)
constexpr int KS_MAX = 128;                     // unit k-steps per expert (K = 4096 gate/up)
constexpr int XKC = 4;                          // prefill: unit k-steps per staged activation chunk
constexpr uint32_t FULL = 0xffffffffu;

struct Params {
    const uint32_t* wire;          // fragment words, all experts
    const int64_t* expert_word0;   // [E] word offset of the expert's tile 0, slot 0
    const uint32_t* hist;          // history words, all experts
    const int64_t* expert_hist0;   // [E] word offset of the expert's history block
    const int32_t* rate;           // [E] RA | RB << 4 | ksa << 8
    const int16_t* kperm;          // [E][KS * NG] slot group -> original 32-column group
    const uint8_t* table;          // [E][NTAB][TAB]
    const float* wscale;           // [E][NTAB][N]
    const uint8_t* x;              // activations [rows + 1][Kx], row xzero / Kx is zero
    const float* a_scale;          // [rows]
    const int32_t* offsets;        // [E + 1] absolute route offsets
    const int32_t* sorted;         // [routes] flat route ids, expert-sorted
    const float* rw;               // [routes] route weights in sorted order (MODE 2)
    const int32_t* item_off;       // [E + 1] superblock prefix
    uint16_t* out;                 // MODE 0: [routes][N] at sorted pos; MODE 2: at flat id
    float* part;                   // [work units][2][TILE][SB]
    int32_t* arrive;               // [items * NT]
    uint8_t* dump;                 // DUMP: [E][NTAB][N][Kx]
    const uint8_t* zeros;          // >= 4 KB of zeros
    int e0, e1;                    // the launch's experts [e0, e1)
    int E, Kx, N, KS, NT, top_k, P, out_stride, a_row_mode, mul_weight, l2_hint, xzero;
    float limit;
};

template <int MODE> struct Mode {
    static constexpr int NG = MODE == 0 ? 1 : 2;      // activation column groups per unit k-step
    static constexpr int NTAB = MODE == 0 ? 2 : 1;    // tables (and scale rows) per expert
};

__device__ __forceinline__ void mma16832_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}
__device__ __forceinline__ uint16_t bf16_bits_rn(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bf16_bits_to_f32(uint32_t bits) { return __uint_as_float(bits << 16); }

template <int R, int LEN>
__device__ __forceinline__ uint32_t field(const uint32_t (&W)[R], int off) {
    const int wi = off >> 5, bo = off & 31;
    constexpr uint32_t MASK = (1u << LEN) - 1u;
    if (bo + LEN <= 32) return (W[wi] >> (32 - bo - LEN)) & MASK;
    return __funnelshift_r(W[wi + 1], W[wi], 64 - bo - LEN) & MASK;
}

// L2 eviction policies: the wire is streamed once (evict_first); the activation
// rows are re-read by every row tile of an expert (evict_last).
__device__ __forceinline__ uint64_t l2_policy(int kind) {
    uint64_t pol;
    if (kind == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    else if (kind == 2) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    else asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}
__device__ __forceinline__ uint32_t ld_nc(const uint32_t* q, uint64_t pol) {
    uint32_t v;
    asm volatile("ld.global.nc.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(v) : "l"(q), "l"(pol));
    return v;
}
__device__ __forceinline__ uint2 ld_nc2(const uint8_t* q, uint64_t pol) {
    uint2 v;
    asm volatile("ld.global.nc.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;" : "=r"(v.x), "=r"(v.y) : "l"(q), "l"(pol));
    return v;
}
// No L2::cache_hint on cp.async: that form raised cudaErrorIllegalInstruction
// on GB10 (eng-regdirect-killtest, PB 103c919a52f0).
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gsrc) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    const uint64_t s = static_cast<uint64_t>(__cvta_generic_to_global(gsrc));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(s) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// Both pair groups of one unit k-step: the 24-bit context pp | p1 | own of each
// column gives the windows of rows 2g (c >> R) and 2g + 1 (c), looked up in the
// group's table.
template <int R, int PG>
__device__ __forceinline__ void decode_group(const uint32_t (&o)[R], const uint32_t (&p1)[R], const uint32_t (&pp)[R],
                                             const uint8_t* tab, uint32_t (&a)[4]) {
    uint32_t v0[8], v1[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = PG * 8 + j;
        uint32_t c;
        if constexpr (R == 4) {
            const int w = q >> 2, b = 3 - (q & 3);     // a pair is one byte, MSB-first
            const uint32_t lo = __byte_perm(o[w], p1[w], b | ((4 + b) << 4));
            c = __byte_perm(lo, pp[w], 0x0010 | ((4 + b) << 8));
        } else if constexpr (R >= 5) {
            // A 14-bit window spans at most 3 codes at R >= 5: row 2g reaches back into p1 only.
            // The 4R-bit context p1 | own fits a register up to R = 8 and covers 14 + R bits.
            const int off = q * 2 * R;
            c = (field<R, 2 * R>(p1, off) << (2 * R)) | field<R, 2 * R>(o, off);
        } else {
            const int off = q * 2 * R;
            c = (field<R, 2 * R>(pp, off) << (4 * R)) | (field<R, 2 * R>(p1, off) << (2 * R)) | field<R, 2 * R>(o, off);
        }
        v0[j] = tab[(c >> R) & (TAB - 1)];
        v1[j] = tab[c & (TAB - 1)];
    }
    a[0] = pack4(v0[0], v0[1], v0[2], v0[3]);
    a[1] = pack4(v1[0], v1[1], v1[2], v1[3]);
    a[2] = pack4(v0[4], v0[5], v0[6], v0[7]);
    a[3] = pack4(v1[4], v1[5], v1[6], v1[7]);
}

// The wire prefetch depth per rate (decode and prefill rings): the ring holds depth x R words per lane, and one
// kernel holds every rate's path, so high rates keep the ring within the R4 budget (D x 4 words)
// instead of raising the kernel's register count for every rate.
template <int R, int D>
struct DecodeDepth { static constexpr int value = R <= 4 ? D : (R == 5 ? (D < 3 ? D : 3) : (D < 2 ? D : 2)); };

// The T-8 code rates the kernel instantiates (R768 .. R2048): one template per rate.
template <class F>
__device__ __forceinline__ void by_rate(int r, F&& f) {
    switch (r) {
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        case 7: f(std::integral_constant<int, 7>{}); break;
        default: f(std::integral_constant<int, 8>{}); break;
    }
}

// The history pairs of one unit k-step: pp from lane g - 2 and p1 from lane
// g - 1, or (g < 2) from the boundary words bb, which hold the previous 16-row
// block's lane g + 6.
template <int R>
__device__ __forceinline__ void history(const uint32_t (&o)[R], const uint32_t (&bb)[R], int g,
                                        uint32_t (&p1)[R], uint32_t (&pp)[R]) {
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        const uint32_t up4 = __shfl_up_sync(FULL, o[i], 4);
        const uint32_t dn4 = __shfl_down_sync(FULL, bb[i], 4);
        if constexpr (R < 5) {                      // decode_group reads pp only below R = 5
            const uint32_t up8 = __shfl_up_sync(FULL, o[i], 8);
            pp[i] = g < 2 ? bb[i] : up8;
        } else {
            pp[i] = 0u;
        }
        p1[i] = g == 0 ? dn4 : up4;
    }
}

// The same for words [W0, W1] only: a split warp's projection lies in those words.
template <int R, int W0, int W1>
__device__ __forceinline__ void history_range(const uint32_t (&o)[R], const uint32_t (&bb)[R], int g,
                                              uint32_t (&p1)[R], uint32_t (&pp)[R]) {
    #pragma unroll
    for (int i = W0; i <= W1; ++i) {
        const uint32_t up4 = __shfl_up_sync(FULL, o[i], 4);
        const uint32_t dn4 = __shfl_down_sync(FULL, bb[i], 4);
        if constexpr (R < 5) {                      // decode_group reads pp only below R = 5
            const uint32_t up8 = __shfl_up_sync(FULL, o[i], 8);
            pp[i] = g < 2 ? bb[i] : up8;
        } else {
            pp[i] = 0u;
        }
        p1[i] = g == 0 ? dn4 : up4;
    }
}

// Where a lane reads its own words and its boundary words in one rate segment.
struct SegPtr {
    const uint32_t* own;
    const uint32_t* bnd;
    long ostride, bstride;   // words per unit k-step
    int bistride;            // words between word i and i + 1 of the boundary lane
};
template <int R>
__device__ __forceinline__ SegPtr seg_ptr(const Params& p, const uint32_t* seg, long seg_tile_words,
                                          const uint32_t* hseg, int T, int warp, int lane, int g, int t) {
    constexpr int UNIT = WARPS * R * 32;
    SegPtr s;
    s.own = seg + (long)T * seg_tile_words + warp * R * 32 + lane;
    s.ostride = UNIT;
    s.bnd = reinterpret_cast<const uint32_t*>(p.zeros) + lane;
    s.bstride = 0;
    s.bistride = 32;
    if (g < 2) {
        const int bl = (g + 6) * 4 + t;
        if (warp > 0) { s.bnd = seg + (long)T * seg_tile_words + (warp - 1) * R * 32 + bl; s.bstride = UNIT; }
        else if (T > 0) { s.bnd = seg + (long)(T - 1) * seg_tile_words + (WARPS - 1) * R * 32 + bl; s.bstride = UNIT; }
        else { s.bnd = hseg + g * 4 + t; s.bstride = R * HIST_LANES; s.bistride = HIST_LANES; }
    }
    return s;
}

// ---------------------------------------------------------------- decode path
// RT route tiles in registers, activations prefetched with the wire (depth D).
template <int R, int MODE, int D, bool DUMP>
__device__ __forceinline__ void kseg_reg(const Params& p, const SegPtr& sp, int s0, int s1, int slot0,
                                         const int16_t* kp, int xrow, uint64_t pol_w, uint64_t pol_x,
                                         int g, int t, int e, int n0, const uint8_t* tab, float (&acc)[2][4]) {
    constexpr int NG = Mode<MODE>::NG;
    if (s0 >= s1) return;
    uint32_t wb[D][R], bb[D][R];
    uint2 xb[D][NG];
    // unconditional loads: a conditional load (or a select on its result) makes
    // ptxas copy the register at once, which waits for the load and voids the ring
    auto load = [&](int d, int s) {
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            wb[d][i] = ld_nc(sp.own + (long)s * sp.ostride + i * 32, pol_w);
            bb[d][i] = ld_nc(sp.bnd + (long)s * sp.bstride + i * sp.bistride, pol_w);
        }
        #pragma unroll
        for (int pg = 0; pg < NG; ++pg)
            xb[d][pg] = ld_nc2(p.x + xrow + (int)kp[(slot0 + s) * NG + pg] * KSTEP + 8 * t, pol_x);
    };
    #pragma unroll
    for (int d = 0; d < D; ++d) load(d, min(s0 + d, s1 - 1));
    for (int s = s0; s < s1; s += D) {
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const bool live = s + d < s1;
            uint32_t o[R], p1[R], pp[R], bv[R];
            #pragma unroll
            for (int i = 0; i < R; ++i) { o[i] = wb[d][i]; bv[i] = bb[d][i]; }
            history<R>(o, bv, g, p1, pp);
            uint2 xs[NG];
            #pragma unroll
            for (int pg = 0; pg < NG; ++pg) xs[pg] = xb[d][pg];
            load(d, min(s + d + D, s1 - 1));
            if (!live) continue;
            uint32_t a[2][4];
            decode_group<R, 0>(o, p1, pp, tab, a[0]);
            decode_group<R, 1>(o, p1, pp, tab + (MODE == 0 ? TAB : 0), a[1]);
            if constexpr (DUMP) {
                #pragma unroll
                for (int pg = 0; pg < 2; ++pg) {
                    const int proj = MODE == 0 ? pg : 0;
                    const int cg = kp[(slot0 + s + d) * NG + (MODE == 0 ? 0 : pg)];
                    uint8_t* row0 = p.dump + (((long)e * Mode<MODE>::NTAB + proj) * p.N + n0 + 2 * g) * p.Kx + cg * KSTEP + 8 * t;
                    *reinterpret_cast<uint2*>(row0) = make_uint2(a[pg][0], a[pg][2]);
                    *reinterpret_cast<uint2*>(row0 + p.Kx) = make_uint2(a[pg][1], a[pg][3]);
                }
            }
            mma16832_e4m3(acc[0], a[0], xs[0].x, xs[0].y);
            // MODE 2: both column groups add into one output (the prefill path's order too)
            mma16832_e4m3(acc[MODE == 0 ? 1 : 0], a[1], xs[NG - 1].x, xs[NG - 1].y);
        }
    }
}

// The epilogue of one lane: rows n0 + 2g (c = h) and n0 + 2g + 1 (c = 2 + h)
// at route 2t + h of route tile rt.
template <int MODE>
__device__ __forceinline__ void epilogue(const Params& p, int e, int r0, int cnt, int n0, int g, int t, int rt,
                                         const float (&a0)[4], const float (&a1)[4]) {
    const float* ws0 = p.wscale + ((long)e * Mode<MODE>::NTAB) * p.N;
    const float* ws1 = ws0 + (MODE == 0 ? p.N : 0);
    const float s00 = ws0[n0 + 2 * g], s01 = ws0[n0 + 2 * g + 1];
    const float s10 = ws1[n0 + 2 * g], s11 = ws1[n0 + 2 * g + 1];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int i = rt * 8 + 2 * t + h;
        if (i >= cnt) continue;
        const int pos = r0 + i;
        const int flat = p.sorted[pos];
        const int arow = p.a_row_mode == 1 ? pos : (p.a_row_mode == 0 ? flat / p.top_k : flat);
        const float a_s = p.a_scale[arow];
        uint32_t packed = 0;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            if constexpr (MODE == 0) {
                float gv = __fmul_rn(__fmul_rn(a0[2 * r + h], a_s), r ? s01 : s00);
                float uv = __fmul_rn(__fmul_rn(a1[2 * r + h], a_s), r ? s11 : s10);
                float gf = bf16_bits_to_f32(bf16_bits_rn(gv));
                float uf = bf16_bits_to_f32(bf16_bits_rn(uv));
                gf = fminf(gf, p.limit);
                uf = fmaxf(fminf(uf, p.limit), -p.limit);
                const float act = __fmul_rn(gf / (1.0f + expf(-gf)), uf);
                packed |= (uint32_t)bf16_bits_rn(act) << (16 * r);
            } else {
                float y = __fmul_rn(__fmul_rn(a0[2 * r + h], a_s), r ? s01 : s00);   // both groups are in a0
                if (p.mul_weight) y = __fmul_rn(y, p.rw[pos]);
                packed |= (uint32_t)bf16_bits_rn(y) << (16 * r);
            }
        }
        const long row = MODE == 0 ? (long)pos : (long)flat;
        *reinterpret_cast<uint32_t*>(p.out + row * p.out_stride + n0 + 2 * g) = packed;
    }
}

// Task decomposition shared by both paths.
struct Task { int e, item, T, part, r0, cnt; };
__device__ __forceinline__ Task task_of(const Params& p, long u, int sb) {
    Task k;
    k.part = (int)(u % p.P);
    const long rest = u / p.P;
    k.T = (int)(rest % p.NT);
    k.item = (int)(rest / p.NT);
    int lo = p.e0, hi = p.e1;                 // largest e in [e0, e1) with item_off[e] <= item
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (p.item_off[mid] <= k.item) lo = mid; else hi = mid;
    }
    k.e = lo;
    k.r0 = p.offsets[k.e] + (k.item - p.item_off[k.e]) * sb;
    k.cnt = min(sb, p.offsets[k.e + 1] - k.r0);
    return k;
}

// The expert's tables (2 x 16 KB or 16 KB) and k-step permutation into shared memory.
template <int MODE>
__device__ __forceinline__ void load_expert(const Params& p, int e, uint8_t* smem_tab, int16_t* smem_kp) {
    constexpr int NTAB = Mode<MODE>::NTAB, NG = Mode<MODE>::NG;
    const uint4* src = reinterpret_cast<const uint4*>(p.table + (long)e * NTAB * TAB);
    uint4* dst = reinterpret_cast<uint4*>(smem_tab);
    #pragma unroll 4
    for (int i = threadIdx.x; i < NTAB * TAB / 16; i += THREADS) dst[i] = __ldg(src + i);
    for (int i = threadIdx.x; i < p.KS * NG; i += THREADS) smem_kp[i] = p.kperm[(long)e * p.KS * NG + i];
}

__device__ __forceinline__ void rate_of(const Params& p, int e, int& ra, int& rb, int& ksa) {
    const int v = p.rate[e];
    ra = v & 15;
    rb = (v >> 4) & 15;
    ksa = v >> 8;
}

template <int MODE, bool DUMP>
__global__ void __launch_bounds__(THREADS, 2) rd_decode(Params p) {
    constexpr int D = 4, NTAB = Mode<MODE>::NTAB;    // D: regdirect_routed.DECODE_DEPTH
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int16_t s_kp[KS_MAX * 2];
    __shared__ int s_last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const long U = (long)p.NT * p.P;
    const long u_lo = (long)p.item_off[p.e0] * U, u_hi = (long)p.item_off[p.e1] * U;
    const long t0 = u_lo + (long)blockIdx.x * (u_hi - u_lo) / gridDim.x;
    const long t1 = u_lo + (long)(blockIdx.x + 1) * (u_hi - u_lo) / gridDim.x;
    int cur_e = -1;
    const uint64_t pol_w = l2_policy(p.l2_hint ? 1 : 0), pol_x = l2_policy(p.l2_hint ? 2 : 0);
    for (long u = t0; u < t1; ++u) {
        const Task k = task_of(p, u, 8);
        const int ks0 = (int)((long)k.part * p.KS / p.P);
        const int ks1 = (int)((long)(k.part + 1) * p.KS / p.P);
        const int n0 = k.T * TILE + warp * BLOCK_ROWS;
        int ra, rb, ksa;
        rate_of(p, k.e, ra, rb, ksa);
        const int i = g;                      // route g of the one route tile
        const int pos = k.r0 + i;
        const int arow = i < k.cnt
            ? (p.a_row_mode == 1 ? pos : (p.a_row_mode == 0 ? p.sorted[pos] / p.top_k : p.sorted[pos])) : -1;
        const int xrow = arow >= 0 ? arow * p.Kx : p.xzero;
        if (k.e != cur_e) {
            __syncthreads();
            load_expert<MODE>(p, k.e, smem, s_kp);
            __syncthreads();
            cur_e = k.e;
        }
        float acc[2][4] = {};
        const uint32_t* w0 = p.wire + p.expert_word0[k.e];
        const uint32_t* h0 = p.hist + p.expert_hist0[k.e];
        // segment A: slots [0, ksa) at ra; segment B: [ksa, KS) at rb
        const long tileA = (long)ksa * WARPS * ra * 32, tileB = (long)(p.KS - ksa) * WARPS * rb * 32;
        const long tile_words = tileA + tileB;
        auto run = [&](auto RC, const uint32_t* seg, const uint32_t* hseg, int a, int b, int slot0) {
            constexpr int RR = decltype(RC)::value;
            const SegPtr sp = seg_ptr<RR>(p, seg, tile_words, hseg, k.T, warp, lane, g, t);
            kseg_reg<RR, MODE, DecodeDepth<RR, D>::value, DUMP>(p, sp, a, b, slot0, s_kp, xrow, pol_w, pol_x, g, t, k.e, n0, smem, acc);
        };
        {
            const int a = ks0, b = min(ks1, ksa);
            by_rate(ra, [&](auto RC) { run(RC, w0, h0, a, b, 0); });
        }
        if (ksa < p.KS) {
            const int a = max(ks0, ksa) - ksa, b = ks1 - ksa;
            const uint32_t* wB = w0 + tileA;
            const uint32_t* hB = h0 + (long)ksa * ra * HIST_LANES;
            by_rate(rb, [&](auto RC) { run(RC, wB, hB, a, b, ksa); });
        }

        bool finish = true;
        if (p.P > 1) {
            const long tb = (long)k.item * p.NT + k.T;
            float* mine = p.part + (tb * p.P + k.part) * 2 * TILE * 8;
            #pragma unroll
            for (int pg = 0; pg < 2; ++pg)
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int ri = 2 * t + (c & 1);
                    const int rl = warp * BLOCK_ROWS + 2 * g + (c >> 1);
                    if (ri < k.cnt) __stcg(mine + (pg * TILE + rl) * 8 + ri, acc[pg][c]);
                }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                s_last = atomicAdd(p.arrive + tb, 1) == p.P - 1;
                if (s_last) p.arrive[tb] = 0;
            }
            __syncthreads();
            finish = s_last;
            if (finish) {
                __threadfence();
                const float* all = p.part + tb * p.P * 2 * TILE * 8;
                #pragma unroll
                for (int pg = 0; pg < 2; ++pg)
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const int ri = 2 * t + (c & 1);
                        const int rl = warp * BLOCK_ROWS + 2 * g + (c >> 1);
                        float v = 0.f;
                        if (ri < k.cnt)
                            for (int q = 0; q < p.P; ++q) v += __ldcg(all + ((long)q * 2 * TILE + pg * TILE + rl) * 8 + ri);
                        acc[pg][c] = v;
                    }
            }
        }
        if (finish) epilogue<MODE>(p, k.e, k.r0, k.cnt, n0, g, t, 0, acc[0], acc[1]);
    }
    (void)NTAB;
}

// --------------------------------------------------------------- prefill path
// 8 * RT-route superblocks; the CTA stages each XKC-k-step chunk of its routes'
// activation rows in shared memory once (cp.async, one barrier per chunk).
// SPLIT (gate/up at RT = 16): each warp owns one projection of one 16-row block,
// so it holds RT x 4 accumulators; a task is 64 rows; gate and up meet in shared
// memory after the K walk.  MODE 2 adds both column groups into one set.
template <int MODE, int RT, bool SPLIT> struct PF {
    static_assert(!SPLIT || MODE == 0, "only gate/up splits by projection");
    static constexpr int NG = Mode<MODE>::NG;
    static constexpr int SB = 8 * RT;                               // routes per superblock
    static constexpr int NACC = (MODE == 0 && !SPLIT) ? 2 : 1;      // accumulator sets per warp
    static constexpr int BLOCKS = SPLIT ? 4 : WARPS;                // 16-row blocks per task
    static constexpr int ROWS = BLOCKS * BLOCK_ROWS;
    // Bytes per staged row: the data plus 32 B, so the row stride is 8 words mod 32.  A lane
    // (g, t) reads 8 bytes at row g, word 2t: bank 8g + 2t, which no two lanes of a half-warp
    // share (a 16 B pad gave 4g + 2t: 2-way conflicts; NCU gpu-ncu-01, 450 M conflicts at M=4096).
    static constexpr int ROW = XKC * NG * KSTEP + 32;
    static constexpr int TABB = Mode<MODE>::NTAB * TAB;
    static constexpr int XBYTES = 2 * SB * ROW;
    static constexpr int SMEM = TABB + XBYTES;
    static_assert(!SPLIT || 4 * RT * 4 * 32 * 4 <= XBYTES, "the up exchange fits the staging buffers");
};

// Walks unit k-steps [s0, s1) of one rate segment with a depth-D wire ring.
// ``enter(slot)`` runs on every live slot in order: at a chunk start it waits
// for the chunk, syncs the CTA and stages the next chunk; it returns the staged
// chunk.  Every warp of the CTA walks the same slots, so the barrier is uniform.
// PRC: the split warp's projection (0 gate, 1 up), compile-time; its pairs lie in words
// [W0, W1] of the lane, and the warp loads and shuffles only those (-1: all words).
template <int R, int MODE, int RT, bool SPLIT, int D, bool DUMP, int PRC, class Enter>
__device__ __forceinline__ void kseg_xs(const Params& p, const SegPtr& sp, int s0, int s1, int slot0, Enter&& enter,
                                        uint64_t pol_w, int g, int t, int e, int n0, const uint8_t* tab,
                                        int rt_live, const int16_t* kp, float (&acc)[PF<MODE, RT, SPLIT>::NACC][RT][4]) {
    using G = PF<MODE, RT, SPLIT>;
    constexpr int NG = G::NG;
    constexpr int W0 = PRC < 0 ? 0 : (PRC * 16 * R) / 32;
    constexpr int W1 = PRC < 0 ? R - 1 : ((PRC + 1) * 16 * R - 1) / 32;
    if (s0 >= s1) return;
    uint32_t wb[D][R], bb[D][R];
    auto load = [&](int d, int s) {
        #pragma unroll
        for (int i = W0; i <= W1; ++i) {
            wb[d][i] = ld_nc(sp.own + (long)s * sp.ostride + i * 32, pol_w);
            bb[d][i] = ld_nc(sp.bnd + (long)s * sp.bstride + i * sp.bistride, pol_w);
        }
    };
    #pragma unroll
    for (int d = 0; d < D; ++d) load(d, min(s0 + d, s1 - 1));
    for (int s = s0; s < s1; s += D) {
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const bool live = s + d < s1;
            uint32_t o[R] = {}, bv[R] = {}, p1[R] = {}, pp[R] = {};
            #pragma unroll
            for (int i = W0; i <= W1; ++i) { o[i] = wb[d][i]; bv[i] = bb[d][i]; }
            history_range<R, W0, W1>(o, bv, g, p1, pp);
            load(d, min(s + d + D, s1 - 1));
            if (!live) continue;
            const int S = slot0 + s + d;
            const uint8_t* xc = enter(S);
            const uint8_t* xk = xc + (S % XKC) * NG * KSTEP + 8 * t;
            if constexpr (SPLIT) {
                constexpr int PR = PRC < 0 ? 0 : PRC;
                uint32_t a[4];
                decode_group<R, PR>(o, p1, pp, tab + PR * TAB, a);
                if constexpr (DUMP) {
                    const int cg = kp[S];
                    uint8_t* row0 = p.dump + (((long)e * 2 + PR) * p.N + n0 + 2 * g) * p.Kx + cg * KSTEP + 8 * t;
                    *reinterpret_cast<uint2*>(row0) = make_uint2(a[0], a[2]);
                    *reinterpret_cast<uint2*>(row0 + p.Kx) = make_uint2(a[1], a[3]);
                }
                #pragma unroll
                for (int rt = 0; rt < RT; ++rt) {
                    if (rt < rt_live) {
                        const uint2 b0 = *reinterpret_cast<const uint2*>(xk + (rt * 8 + g) * G::ROW);
                        mma16832_e4m3(acc[0][rt], a, b0.x, b0.y);
                    }
                }
            } else {
                uint32_t a[2][4];
                decode_group<R, 0>(o, p1, pp, tab, a[0]);
                decode_group<R, 1>(o, p1, pp, tab + (MODE == 0 ? TAB : 0), a[1]);
                if constexpr (DUMP) {
                    #pragma unroll
                    for (int pg = 0; pg < 2; ++pg) {
                        const int proj = MODE == 0 ? pg : 0;
                        const int cg = kp[S * NG + (MODE == 0 ? 0 : pg)];
                        uint8_t* row0 = p.dump + (((long)e * Mode<MODE>::NTAB + proj) * p.N + n0 + 2 * g) * p.Kx + cg * KSTEP + 8 * t;
                        *reinterpret_cast<uint2*>(row0) = make_uint2(a[pg][0], a[pg][2]);
                        *reinterpret_cast<uint2*>(row0 + p.Kx) = make_uint2(a[pg][1], a[pg][3]);
                    }
                }
                #pragma unroll
                for (int rt = 0; rt < RT; ++rt) {
                    if (rt < rt_live) {
                        const uint8_t* xr = xk + (rt * 8 + g) * G::ROW;
                        const uint2 b0 = *reinterpret_cast<const uint2*>(xr);
                        const uint2 b1 = *reinterpret_cast<const uint2*>(xr + (NG - 1) * KSTEP);
                        mma16832_e4m3(acc[0][rt], a[0], b0.x, b0.y);
                        mma16832_e4m3(acc[G::NACC - 1][rt], a[1], b1.x, b1.y);
                    }
                }
            }
        }
    }
}

template <int MODE, int RT, bool SPLIT, bool DUMP>
__global__ void __launch_bounds__(THREADS, 1) rd_prefill(Params p) {
    using G = PF<MODE, RT, SPLIT>;
    constexpr int NG = G::NG;
    constexpr int CPR = XKC * NG * 2;                 // 16-byte copies per staged row
    constexpr int COPIES = G::SB * CPR / THREADS;     // per thread per chunk
    static_assert(G::SB * CPR % THREADS == 0, "copies split evenly");
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int16_t s_kp[KS_MAX * 2];
    uint8_t* xbase = smem + G::TABB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int blk = SPLIT ? warp >> 1 : warp, pr = SPLIT ? warp & 1 : 0;   // warp-uniform role
    const int NTT = p.N / G::ROWS;                    // task tiles per item; P == 1 on this path
    const long u_lo = (long)p.item_off[p.e0] * NTT, u_hi = (long)p.item_off[p.e1] * NTT;
    const long t0 = u_lo + (long)blockIdx.x * (u_hi - u_lo) / gridDim.x;
    const long t1 = u_lo + (long)(blockIdx.x + 1) * (u_hi - u_lo) / gridDim.x;
    int cur_e = -1;
    const uint64_t pol_w = l2_policy(p.l2_hint ? 1 : 0);
    const int nch = (p.KS + XKC - 1) / XKC;
    for (long u = t0; u < t1; ++u) {
        Task k;                                       // (item, task tile), task tile fastest
        {
            k.T = (int)(u % NTT);
            k.item = (int)(u / NTT);
            int lo = p.e0, hi = p.e1;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (p.item_off[mid] <= k.item) lo = mid; else hi = mid;
            }
            k.e = lo;
            k.part = 0;
            k.r0 = p.offsets[k.e] + (k.item - p.item_off[k.e]) * G::SB;
            k.cnt = min(G::SB, p.offsets[k.e + 1] - k.r0);
        }
        const int row0 = k.T * G::ROWS + blk * BLOCK_ROWS;    // this warp's 16 rows
        const int T128 = row0 / TILE, w128 = (row0 % TILE) / BLOCK_ROWS;
        int ra, rb, ksa;
        rate_of(p, k.e, ra, rb, ksa);
        int xrow[COPIES];
        #pragma unroll
        for (int c = 0; c < COPIES; ++c) {
            const int i = (threadIdx.x + c * THREADS) / CPR;
            const int pos = k.r0 + i;
            const int arow = i < k.cnt
                ? (p.a_row_mode == 1 ? pos : (p.a_row_mode == 0 ? p.sorted[pos] / p.top_k : p.sorted[pos])) : -1;
            xrow[c] = arow >= 0 ? arow * p.Kx : p.xzero;
        }
        const int rt_live = (k.cnt + 7) >> 3;
        __syncthreads();                       // the previous task no longer reads the buffers
        if (k.e != cur_e) { load_expert<MODE>(p, k.e, smem, s_kp); cur_e = k.e; }
        __syncthreads();                       // s_kp is visible before the first gather
        auto stage = [&](int c) {
            uint8_t* b = xbase + (c & 1) * G::SB * G::ROW;
            #pragma unroll
            for (int q = 0; q < COPIES; ++q) {
                const int idx = threadIdx.x + q * THREADS;
                const int i = idx / CPR, rem = idx % CPR;
                const int d = rem / (NG * 2), pg = (rem / 2) % NG, h = rem & 1;
                const int s = min(c * XKC + d, p.KS - 1);
                const int cg = s_kp[s * NG + pg];
                cp_async16(b + i * G::ROW + (d * NG + pg) * KSTEP + h * 16, p.x + xrow[q] + cg * KSTEP + h * 16);
            }
            cp_async_commit();
        };
        stage(0);
        float acc[G::NACC][RT][4] = {};
        const uint32_t* w0 = p.wire + p.expert_word0[k.e];
        const uint32_t* h0 = p.hist + p.expert_hist0[k.e];
        const long tileA = (long)ksa * WARPS * ra * 32, tileB = (long)(p.KS - ksa) * WARPS * rb * 32;
        const long tile_words = tileA + tileB;
        auto enter = [&](int S) -> const uint8_t* {
            const int c = S / XKC;
            if (S % XKC == 0) {
                cp_async_wait_all();
                __syncthreads();               // chunk c visible; chunk c - 1 no longer read
                if (c + 1 < nch) stage(c + 1);
            }
            return xbase + (c & 1) * G::SB * G::ROW;
        };
        auto run = [&](auto RC, const uint32_t* seg, const uint32_t* hseg, int a, int b, int slot0) {
            constexpr int RR = decltype(RC)::value;
            const SegPtr sp = seg_ptr<RR>(p, seg, tile_words, hseg, T128, w128, lane, g, t);
            if constexpr (SPLIT) {
                if (pr == 0) kseg_xs<RR, MODE, RT, SPLIT, DecodeDepth<RR, XKC>::value, DUMP, 0>(p, sp, a, b, slot0, enter, pol_w, g, t, k.e, row0,
                                                                       smem, rt_live, s_kp, acc);
                else kseg_xs<RR, MODE, RT, SPLIT, DecodeDepth<RR, XKC>::value, DUMP, 1>(p, sp, a, b, slot0, enter, pol_w, g, t, k.e, row0,
                                                               smem, rt_live, s_kp, acc);
            } else {
                kseg_xs<RR, MODE, RT, SPLIT, DecodeDepth<RR, XKC>::value, DUMP, -1>(p, sp, a, b, slot0, enter, pol_w, g, t, k.e, row0,
                                                           smem, rt_live, s_kp, acc);
            }
        };
        by_rate(ra, [&](auto RC) { run(RC, w0, h0, 0, ksa, 0); });
        if (ksa < p.KS) {
            const uint32_t* wB = w0 + tileA;
            const uint32_t* hB = h0 + (long)ksa * ra * HIST_LANES;
            by_rate(rb, [&](auto RC) { run(RC, wB, hB, 0, p.KS - ksa, ksa); });
        }
        if constexpr (SPLIT) {
            // up warps hand their accumulators to the gate warp of the same block
            cp_async_wait_all();
            __syncthreads();                   // no warp reads the staging buffers any more
            float* xw = reinterpret_cast<float*>(xbase) + blk * RT * 4 * 32;
            if (pr == 1) {
                #pragma unroll
                for (int rt = 0; rt < RT; ++rt)
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) xw[(rt * 4 + c) * 32 + lane] = acc[0][rt][c];
            }
            __syncthreads();
            if (pr == 0) {
                #pragma unroll
                for (int rt = 0; rt < RT; ++rt) {
                    if (rt < rt_live) {
                        float up[4];
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) up[c] = xw[(rt * 4 + c) * 32 + lane];
                        epilogue<MODE>(p, k.e, k.r0, k.cnt, row0, g, t, rt, acc[0][rt], up);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int rt = 0; rt < RT; ++rt)
                if (rt < rt_live) epilogue<MODE>(p, k.e, k.r0, k.cnt, row0, g, t, rt, acc[0][rt], acc[G::NACC - 1][rt]);
        }
    }
}

template <int MODE, bool DUMP>
void launch_decode(const Params& p, int grid, cudaStream_t st) {
    rd_decode<MODE, DUMP><<<grid, THREADS, Mode<MODE>::NTAB * TAB, st>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int MODE, int RT, bool SPLIT, bool DUMP>
void launch_prefill(const Params& p, int grid, cudaStream_t st) {
    auto k = rd_prefill<MODE, RT, SPLIT, DUMP>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, PF<MODE, RT, SPLIT>::SMEM));
    k<<<grid, THREADS, PF<MODE, RT, SPLIT>::SMEM, st>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int MODE, int RT, bool SPLIT, bool DUMP>
int occupancy_prefill() {
    auto k = rd_prefill<MODE, RT, SPLIT, DUMP>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, PF<MODE, RT, SPLIT>::SMEM));
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, THREADS, PF<MODE, RT, SPLIT>::SMEM));
    return n;
}
template <int MODE, bool DUMP>
int occupancy_decode() {
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, rd_decode<MODE, DUMP>, THREADS, Mode<MODE>::NTAB * TAB));
    return n;
}

// One dispatch for both entries: route tiles RT (1 = the decode path, 8 or 16 = prefill).
// Gate/up at RT = 16 is the SPLIT variant.  ``p == nullptr`` returns the occupancy.
template <bool DUMP>
int run(int64_t mode, int64_t rt, const Params* p, int grid, cudaStream_t st) {
    const bool L = p != nullptr;
#define RD_DEC(M_) if (mode == M_ && rt == 1) { if (L) { launch_decode<M_, DUMP>(*p, grid, st); return 0; } return occupancy_decode<M_, DUMP>(); }
#define RD_PF(M_, RT_, S_) if (mode == M_ && rt == RT_) { if (L) { launch_prefill<M_, RT_, S_, DUMP>(*p, grid, st); return 0; } return occupancy_prefill<M_, RT_, S_, DUMP>(); }
    RD_DEC(0) RD_DEC(2) RD_PF(0, 8, false) RD_PF(0, 16, true) RD_PF(2, 8, false) RD_PF(2, 16, false)
#undef RD_DEC
#undef RD_PF
    TORCH_CHECK(false, "no instantiation for mode ", mode, " route tiles ", rt);
    return 0;
}

}  // namespace

int64_t blocks_per_sm(int64_t mode, int64_t route_tiles, bool dump) {
    return dump ? run<true>(mode, route_tiles, nullptr, 0, 0) : run<false>(mode, route_tiles, nullptr, 0, 0);
}

// The register-direct launch.  The routing front end owns the sorted routes,
// prefixes and the stream; this entry owns the geometry.  ``e0, e1`` bound the
// experts of the launch (one class, or every class); the work interval is
// [item_off[e0], item_off[e1]) x NT x P.
void forward(int64_t mode, int64_t route_tiles, bool dump,
             torch::Tensor wire, torch::Tensor expert_word0, torch::Tensor hist, torch::Tensor expert_hist0,
             torch::Tensor rate, torch::Tensor kperm, torch::Tensor table, torch::Tensor wscale,
             torch::Tensor x, torch::Tensor a_scale, torch::Tensor offsets, torch::Tensor sorted,
             torch::Tensor rw, torch::Tensor item_off, int64_t e0, int64_t e1,
             torch::Tensor out, torch::Tensor part, torch::Tensor arrive, torch::Tensor dump_buf,
             torch::Tensor zeros, int64_t KS, int64_t top_k, int64_t P, int64_t a_row_mode, bool mul_weight,
             double limit, bool l2_hint, int64_t grid) {
    const at::cuda::CUDAGuard guard(wire.device());
    TORCH_CHECK(mode == 0 || mode == 2, "mode is 0 (gate/up) or 2 (down)");
    const int ntab = mode == 0 ? 2 : 1, ng = mode == 0 ? 1 : 2;
    TORCH_CHECK(wire.dtype() == torch::kInt32 && hist.dtype() == torch::kInt32, "wire and history: int32");
    TORCH_CHECK(expert_word0.dtype() == torch::kInt64 && expert_hist0.dtype() == torch::kInt64, "offsets: int64");
    TORCH_CHECK(rate.dtype() == torch::kInt32 && kperm.dtype() == torch::kInt16, "rate int32, kperm int16");
    TORCH_CHECK(table.dtype() == torch::kUInt8 && table.dim() == 3 && table.size(1) == ntab && table.size(2) == TAB,
                "table [E, NTAB, 16384] u8");
    const int E = table.size(0), N = wscale.size(2), Kx = x.size(1);
    TORCH_CHECK(wscale.size(1) == ntab, "wscale [E, NTAB, N]");
    TORCH_CHECK(N % TILE == 0, "rows: a multiple of 128");
    TORCH_CHECK(KS >= 1 && KS <= KS_MAX && (long)KS * ng * KSTEP == Kx, "K = KS x groups x 32");
    TORCH_CHECK(kperm.numel() == (long)E * KS * ng, "kperm [E, KS * groups]");
    TORCH_CHECK(x.element_size() == 1 && x.is_contiguous(), "x: e4m3 [rows + 1, K], the last row zero");
    TORCH_CHECK(x.numel() < (1L << 31), "x: 32-bit offsets");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.size(1) == N, "out [routes, N] bf16");
    TORCH_CHECK(route_tiles == 1 || P == 1, "the prefill path has no K parts");
    TORCH_CHECK(P >= 1 && P <= KS, "P");
    TORCH_CHECK(0 <= e0 && e0 <= e1 && e1 <= E, "expert range");
    TORCH_CHECK(!dump || dump_buf.numel() == (long)E * ntab * N * Kx, "dump [E, NTAB, N, K]");
    TORCH_CHECK(zeros.dtype() == torch::kUInt8 && zeros.numel() >= 4096, "zeros: >= 4096 u8");
    Params p;
    p.wire = reinterpret_cast<const uint32_t*>(wire.data_ptr<int32_t>());
    p.expert_word0 = expert_word0.data_ptr<int64_t>();
    p.hist = reinterpret_cast<const uint32_t*>(hist.data_ptr<int32_t>());
    p.expert_hist0 = expert_hist0.data_ptr<int64_t>();
    p.rate = rate.data_ptr<int32_t>();
    p.kperm = kperm.data_ptr<int16_t>();
    p.table = table.data_ptr<uint8_t>();
    p.wscale = wscale.data_ptr<float>();
    p.x = reinterpret_cast<const uint8_t*>(x.data_ptr());
    p.a_scale = a_scale.data_ptr<float>();
    p.offsets = offsets.data_ptr<int32_t>();
    p.sorted = sorted.data_ptr<int32_t>();
    p.rw = rw.numel() ? rw.data_ptr<float>() : nullptr;
    p.item_off = item_off.data_ptr<int32_t>();
    p.out = reinterpret_cast<uint16_t*>(out.data_ptr());
    p.part = part.data_ptr<float>();
    p.arrive = arrive.data_ptr<int32_t>();
    p.dump = dump ? dump_buf.data_ptr<uint8_t>() : nullptr;
    p.zeros = zeros.data_ptr<uint8_t>();
    p.e0 = (int)e0; p.e1 = (int)e1;
    p.E = E; p.Kx = Kx; p.N = N; p.KS = (int)KS; p.NT = N / TILE;
    p.top_k = (int)top_k; p.P = (int)P; p.out_stride = (int)out.stride(0);
    p.a_row_mode = (int)a_row_mode; p.mul_weight = mul_weight ? 1 : 0; p.l2_hint = l2_hint ? 1 : 0;
    p.xzero = (int)((x.size(0) - 1) * x.size(1));
    p.limit = std::isfinite(limit) ? (float)limit : std::numeric_limits<float>::infinity();
    TORCH_CHECK(!p.mul_weight || p.rw, "route weights required");
    auto st = at::cuda::getCurrentCUDAStream();
    if (dump) run<true>(mode, route_tiles, &p, (int)grid, st); else run<false>(mode, route_tiles, &p, (int)grid, st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.def("blocks_per_sm", &blocks_per_sm);
}
