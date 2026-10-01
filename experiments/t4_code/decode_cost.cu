// Decode cost per weight on sm_121 for the two T-4 bodies, in the register
// form a fused FP4 kernel would use (the FP4 MMA's B fragments, no smem tile).
//
// Both decoders produce, per thread and per 64-column chunk, the eight B
// registers of four m16n8k64 n8-tiles: thread (g, t) of a warp decodes rows
// 4g..4g+3 of the warp's 32-row slice at columns k0+8t..k0+8t+7 and
// k0+32+8t..k0+32+8t+7 -- the N permutation "MMA column g + 8j <-> row 4g + j"
// puts one decoded row in each n8-tile, so no transpose is needed.  The
// planes sit in shared memory as MSB-first 32-bit words (what a loader repack
// hands the kernel), so the number is the decode's ALU/LDS ceiling, not the
// staging.  Identical nibble placement and the same LUT16 scale plane are
// common to both bodies and are included in both.
//
//   TCQ span-2 (R = F + 1, one rate per unit): per (column, pair) a select
//     window (memory + 1 bits) -> label LUT; the stored 2-bit label; two
//     F-bit points; two code-table bytes (rows 4p,4p+1 and 4p+2,4p+3).
//   Window (L bits, rate r bits per 2-row tuple): per (column, two tuples)
//     one 32-bit read covering both L-bit windows, two table bytes.
//   Window over a two-run table [r, r + 1]: the same, with each column's
//     rate read from its block's high-rate mask (``win_decode2``).
//
// MODE 0 XOR-reduces the fragments (decode only).  MODE 1 feeds them to the
// block-scaled FP4 MMA (kind::mxf4nvf4, UE4M3 per 16) MREP times, the A-side
// reuse of a 16*MREP-row tile, to show whether decode hides under the MMA.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -o decode_cost decode_cost.cu
// (``-arch=sm_121a`` also embeds compute_121 PTX, which ptxas refuses for
// the block-scaled MMA.)
// Run:   decode_cost [iters] > decode_cost.json
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

constexpr int ROWS = 256;           // rows per block slice: 8 warps x 32
constexpr int K = 256;              // columns staged per block
constexpr int CHUNKS = K / 64;
constexpr int PAIRS = ROWS / 4;     // TCQ: pairs of 2-row codes
constexpr int STEPS = ROWS / 2;     // 2-row tuples per column
constexpr int MEM = 6;              // DEFAULT_CODE memory
constexpr int THREADS = 256;

__device__ __forceinline__ uint32_t rd32(const uint32_t* w, uint32_t bit) {
    const uint32_t i = bit >> 5, s = bit & 31;
    return __funnelshift_l(w[i + 1], w[i], s);        // 32 stream bits from `bit`, MSB-first
}

__device__ __forceinline__ void place(uint32_t (&b)[2][4], int half, int i, uint32_t c0, uint32_t c1) {
    // c0: rows 4g (lo nibble), 4g+1 (hi); c1: rows 4g+2, 4g+3 -> n8-tiles j = 0..3
    b[half][0] |= (c0 & 0xFu) << (4 * i);
    b[half][1] |= ((c0 >> 4) & 0xFu) << (4 * i);
    b[half][2] |= (c1 & 0xFu) << (4 * i);
    b[half][3] |= ((c1 >> 4) & 0xFu) << (4 * i);
}

__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                        uint32_t sfa, uint32_t sfb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

// LUT16 scale plane: one nibble per (row, 16 columns), [groups][rows], two
// rows per byte, even row high.  The thread's four rows at one k16 group are
// sixteen contiguous bits, so one read serves them; each nibble indexes the
// unit's 16-entry E4M3 table held in four registers.  The result is the SFB
// register of each of the four n8-tiles (one byte per k16 group of the
// chunk).  Identical for both bodies.
__device__ __forceinline__ uint32_t lut_byte(const uint32_t (&lut)[4], uint32_t n) {
    const uint32_t lo = __byte_perm(lut[0], lut[1], n & 7);
    const uint32_t hi = __byte_perm(lut[2], lut[3], n & 7);
    return ((n & 8) ? hi : lo) & 0xFFu;
}
__device__ __forceinline__ void sf4(const uint32_t* nib, const uint32_t (&lut)[4], int kc, int row0,
                                    uint32_t (&sfb)[4]) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) sfb[j] = 0;
    #pragma unroll
    for (int kg = 0; kg < 4; ++kg) {
        const uint32_t z = rd32(nib, 4 * (((kc >> 4) + kg) * ROWS + row0));
        #pragma unroll
        for (int j = 0; j < 4; ++j) sfb[j] |= lut_byte(lut, (z >> (28 - 4 * j)) & 15u) << (8 * kg);
    }
}

template <int MODE, int MREP>
__device__ __forceinline__ void consume(uint32_t (&b)[2][4], const uint32_t (&sfb)[4], uint32_t& acc,
                                        float (&d)[4], const uint32_t (&a)[4]) {
    if constexpr (MODE == 0) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc ^= b[0][j] ^ b[1][j] ^ sfb[j];
    } else {
        #pragma unroll
        for (int m = 0; m < MREP; ++m)
            #pragma unroll
            for (int j = 0; j < 4; ++j) mma_fp4(d, a, b[0][j], b[1][j], 0x38383838u, sfb[j]);
    }
}

// ------------------------------------------------------------------ TCQ span 2
template <int F, int MODE, int MREP>
__global__ void __launch_bounds__(THREADS) tcq_decode(const uint32_t* __restrict__ g, int words,
                                                      int iters, uint32_t* out, long long* cyc) {
    extern __shared__ uint32_t sm[];
    for (int i = threadIdx.x; i < words; i += THREADS) sm[i] = g[i];
    __syncthreads();
    constexpr int SEL_W = 4;                            // (PAIRS + 8) = 72 bits, padded to 4 words
    constexpr int LAB_W = PAIRS * 2 / 32 + 1;           // 4 words + 1 slack
    constexpr int PT_W = F ? (STEPS * F / 32 + 1) : 1;  // 4F words + 1 slack
    constexpr int POINTS = 1 << F;
    const uint32_t* sel = sm;
    const uint32_t* lab = sel + K * SEL_W;
    const uint32_t* pt = lab + K * LAB_W;
    const uint32_t* nib = pt + K * PT_W;                // [K/16 groups][ROWS nibbles] + slack
    const uint8_t* code = reinterpret_cast<const uint8_t*>(nib + (K / 16) * (ROWS / 8) + 1);
    const uint8_t* llut = code + 4 * POINTS;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gq = lane >> 2, t = lane & 3;
    const int p = warp * 8 + gq;                        // the thread's pair
    const uint32_t q_sel = 8 - MEM + p, q_lab = 2 * p, q_pt = 2 * F * p;
    uint32_t lut[4] = {0x38403c34u, 0x48444c42u, 0x30283420u, 0x50545856u};
    uint32_t a[4] = {0x2468aceu ^ lane, 0x13579bdu, 0xfdb97531u, 0x0eca8642u};
    uint32_t acc = 0;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    const long long c0 = clock64();
    #pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        const int kc = (it % CHUNKS) * 64;
        uint32_t b[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int k = kc + 32 * h + 8 * t + i;
                const uint32_t win = rd32(sel + k * SEL_W, q_sel) >> (31 - MEM);
                const uint32_t ell = llut[win];
                const uint32_t stored = rd32(lab + k * LAB_W, q_lab) >> 30;
                const uint32_t lab0 = (ell - stored) & 3u;
                uint32_t pt0 = 0, pt1 = 0;
                if constexpr (F > 0) {
                    const uint32_t z = rd32(pt + k * PT_W, q_pt);
                    pt0 = z >> (32 - F);
                    pt1 = (z >> (32 - 2 * F)) & (POINTS - 1);
                }
                const uint32_t cA = code[lab0 * POINTS + pt0];
                const uint32_t cB = code[stored * POINTS + pt1];
                place(b, h, i, cA, cB);
            }
        }
        uint32_t sfb[4];
        sf4(nib, lut, kc, 4 * p, sfb);
        consume<MODE, MREP>(b, sfb, acc, d, a);
        asm volatile("" ::: "memory");
    }
    const long long c1 = clock64();
    out[blockIdx.x * THREADS + threadIdx.x] = acc ^ __float_as_uint(d[0] + d[1] + d[2] + d[3]);
    if (threadIdx.x == 0) cyc[blockIdx.x] = c1 - c0;
}

// ------------------------------------------------------------------ window
template <int L, int R, int MODE, int MREP>
__global__ void __launch_bounds__(THREADS) win_decode(const uint32_t* __restrict__ g, int words,
                                                      int iters, uint32_t* out, long long* cyc) {
    extern __shared__ uint32_t sm[];
    for (int i = threadIdx.x; i < words; i += THREADS) sm[i] = g[i];
    __syncthreads();
    constexpr int COL_W = 1 + STEPS * R / 32 + 1;       // pad word (history) + 4R words + slack
    const uint32_t* str = sm;
    const uint32_t* nib = str + K * COL_W;
    const uint8_t* table = reinterpret_cast<const uint8_t*>(nib + (K / 16) * (ROWS / 8) + 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gq = lane >> 2, t = lane & 3;
    const int s0 = warp * 16 + 2 * gq;                  // the thread's first tuple (rows 4g, 4g+1)
    // bits of tuples s0 and s0+1 end at (s0+1)R and (s0+2)R; one read covers both windows
    const uint32_t q = 32 + (s0 + 1) * R - L;
    uint32_t lut[4] = {0x38403c34u, 0x48444c42u, 0x30283420u, 0x50545856u};
    uint32_t a[4] = {0x2468aceu ^ lane, 0x13579bdu, 0xfdb97531u, 0x0eca8642u};
    uint32_t acc = 0;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    constexpr uint32_t MASK = (1u << L) - 1;
    const long long c0 = clock64();
    #pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        const int kc = (it % CHUNKS) * 64;
        uint32_t b[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int k = kc + 32 * h + 8 * t + i;
                const uint32_t z = rd32(str + k * COL_W, q);
                const uint32_t w0 = z >> (32 - L);
                const uint32_t w1 = (z >> (32 - L - R)) & MASK;
                place(b, h, i, table[w0], table[w1]);
            }
        }
        uint32_t sfb[4];
        sf4(nib, lut, kc, 32 * warp + 4 * gq, sfb);
        consume<MODE, MREP>(b, sfb, acc, d, a);
        asm volatile("" ::: "memory");
    }
    const long long c1 = clock64();
    out[blockIdx.x * THREADS + threadIdx.x] = acc ^ __float_as_uint(d[0] + d[1] + d[2] + d[3]);
    if (threadIdx.x == 0) cyc[blockIdx.x] = c1 - c0;
}

// ------------------------------------------------------------------ window, two-run table
// The same register form over a two-run table [RL, RL + 1]: the eight k of a
// B register are consecutive ORIGINAL columns, so their rates differ per
// column.  Each column's rate comes from its block's high-rate mask (bit c of
// a 32-bit word per 32 columns, as a lane would read it beside the staged
// words); its words sit in a slot sized for RL + 1.  The window position and
// the second field's shift become runtime values.
template <int L, int RL, int MODE, int MREP>
__global__ void __launch_bounds__(THREADS) win_decode2(const uint32_t* __restrict__ g, int words,
                                                       int iters, uint32_t* out, long long* cyc) {
    extern __shared__ uint32_t sm[];
    for (int i = threadIdx.x; i < words; i += THREADS) sm[i] = g[i];
    __syncthreads();
    constexpr int COL_W = 1 + STEPS * (RL + 1) / 32 + 1;
    const uint32_t* str = sm;
    const uint32_t* hi_mask = str + K * COL_W;                  // K / 32 words
    const uint32_t* nib = hi_mask + K / 32;
    const uint8_t* table = reinterpret_cast<const uint8_t*>(nib + (K / 16) * (ROWS / 8) + 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gq = lane >> 2, t = lane & 3;
    const int s0 = warp * 16 + 2 * gq;
    uint32_t lut[4] = {0x38403c34u, 0x48444c42u, 0x30283420u, 0x50545856u};
    uint32_t a[4] = {0x2468aceu ^ lane, 0x13579bdu, 0xfdb97531u, 0x0eca8642u};
    uint32_t acc = 0;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    constexpr uint32_t MASK = (1u << L) - 1;
    const int q_lo = 32 + (s0 + 1) * RL - L, q_hi = q_lo + s0 + 1;
    const int i_lo = q_lo >> 5, i_hi = q_hi >> 5;
    const uint32_t s_lo = q_lo & 31, s_hi = q_hi & 31;
    constexpr uint32_t f_lo = 32 - L - RL, f_hi = 32 - L - RL - 1;
    const long long c0 = clock64();
    #pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        const int kc = (it % CHUNKS) * 64;
        uint32_t b[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t m = hi_mask[(kc >> 5) + h] >> (8 * t);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int k = kc + 32 * h + 8 * t + i;
                const bool hi = (m >> i) & 1u;
                // both rates' window positions are per-thread constants; a
                // column selects one (word, shift) pair and the second field's shift
                const uint32_t* w = str + k * COL_W + (hi ? i_hi : i_lo);
                const uint32_t z = __funnelshift_l(w[1], w[0], hi ? s_hi : s_lo);
                const uint32_t w0 = z >> (32 - L);
                const uint32_t w1 = (z >> (hi ? f_hi : f_lo)) & MASK;
                place(b, h, i, table[w0], table[w1]);
            }
        }
        uint32_t sfb[4];
        sf4(nib, lut, kc, 32 * warp + 4 * gq, sfb);
        consume<MODE, MREP>(b, sfb, acc, d, a);
        asm volatile("" ::: "memory");
    }
    const long long c1 = clock64();
    out[blockIdx.x * THREADS + threadIdx.x] = acc ^ __float_as_uint(d[0] + d[1] + d[2] + d[3]);
    if (threadIdx.x == 0) cyc[blockIdx.x] = c1 - c0;
}

// ------------------------------------------------------------------ host
struct Res { double ms, weights, gw_per_s, sm_cycles_per_kweight; int blocks_per_sm; };

template <typename Kern>
Res run(Kern kern, int words, int iters, int sms, const uint32_t* dg, uint32_t* dout, long long* dcyc) {
    const int smem = words * 4;
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, THREADS, smem));
    const int blocks = sms * per_sm;                    // all co-resident
    kern<<<blocks, THREADS, smem>>>(dg, words, 8, dout, dcyc);   // warm
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        CK(cudaEventRecord(e0));
        kern<<<blocks, THREADS, smem>>>(dg, words, iters, dout, dcyc);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best) best = ms;
    }
    std::vector<long long> cyc(blocks);
    CK(cudaMemcpy(cyc.data(), dcyc, blocks * sizeof(long long), cudaMemcpyDeviceToHost));
    double mean = 0; for (auto c : cyc) mean += (double)c; mean /= blocks;
    const double w_block = (double)ROWS * 64 * iters;   // weights one block decodes
    Res r;
    r.ms = best;
    r.weights = w_block * blocks;
    r.gw_per_s = r.weights / (best * 1e-3) / 1e9;
    r.sm_cycles_per_kweight = mean / (w_block * per_sm) * 1000.0;
    r.blocks_per_sm = per_sm;
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return r;
}

template <int F, int MODE, int MREP>
void tcq_case(int iters, int sms, uint32_t* dg, uint32_t* dout, long long* dcyc, bool& first) {
    const int words = K * 4 + K * (PAIRS * 2 / 32 + 1) + K * (F ? STEPS * F / 32 + 1 : 1)
                      + (K / 16) * (ROWS / 8) + 1 + (4 * (1 << F) + (1 << (MEM + 1)) + 3) / 4;
    Res r = run(tcq_decode<F, MODE, MREP>, words, iters, sms, dg, dout, dcyc);
    printf("%s{\"body\":\"tcq\",\"rate\":%d,\"q256\":%d,\"mode\":%d,\"mrep\":%d,\"smem\":%d,"
           "\"blocks_per_sm\":%d,\"ms\":%.4f,\"gweights_per_s\":%.2f,\"sm_cycles_per_kweight\":%.2f}\n",
           first ? "" : ",", F + 1, 128 * (F + 1), MODE, MREP, words * 4, r.blocks_per_sm, r.ms,
           r.gw_per_s, r.sm_cycles_per_kweight);
    first = false;
}

template <int L, int R, int MODE, int MREP>
void win_case(int iters, int sms, uint32_t* dg, uint32_t* dout, long long* dcyc, bool& first) {
    const int words = K * (1 + STEPS * R / 32 + 1) + (K / 16) * (ROWS / 8) + 1 + (1 << L) / 4;
    Res r = run(win_decode<L, R, MODE, MREP>, words, iters, sms, dg, dout, dcyc);
    printf("%s{\"body\":\"window\",\"L\":%d,\"rate\":%d,\"q256\":%d,\"mode\":%d,\"mrep\":%d,\"smem\":%d,"
           "\"blocks_per_sm\":%d,\"ms\":%.4f,\"gweights_per_s\":%.2f,\"sm_cycles_per_kweight\":%.2f}\n",
           first ? "" : ",", L, R, 128 * R, MODE, MREP, words * 4, r.blocks_per_sm, r.ms, r.gw_per_s,
           r.sm_cycles_per_kweight);
    first = false;
}

template <int L, int RL, int MODE, int MREP>
void win2_case(int iters, int sms, uint32_t* dg, uint32_t* dout, long long* dcyc, bool& first) {
    const int words = K * (1 + STEPS * (RL + 1) / 32 + 1) + K / 32 + (K / 16) * (ROWS / 8) + 1 + (1 << L) / 4;
    Res r = run(win_decode2<L, RL, MODE, MREP>, words, iters, sms, dg, dout, dcyc);
    printf("%s{\"body\":\"window\",\"L\":%d,\"rate\":%d,\"run_table\":[%d,%d],\"q256\":%d,\"mode\":%d,"
           "\"mrep\":%d,\"smem\":%d,\"blocks_per_sm\":%d,\"ms\":%.4f,\"gweights_per_s\":%.2f,"
           "\"sm_cycles_per_kweight\":%.2f}\n",
           first ? "" : ",", L, RL, RL, RL + 1, 128 * RL + 64, MODE, MREP, words * 4, r.blocks_per_sm, r.ms,
           r.gw_per_s, r.sm_cycles_per_kweight);
    first = false;
}

template <int MODE, int MREP>
void all_cases(int iters, int sms, uint32_t* dg, uint32_t* dout, long long* dcyc, bool& first) {
    tcq_case<0, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    tcq_case<1, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    tcq_case<2, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    tcq_case<3, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    tcq_case<4, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    tcq_case<5, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    tcq_case<6, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<12, 1, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<12, 2, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<12, 3, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<12, 4, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<12, 5, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<12, 6, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<12, 7, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<12, 8, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<14, 4, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win_case<14, 7, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win2_case<12, 1, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win2_case<12, 3, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win2_case<12, 4, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win2_case<12, 7, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
    win2_case<14, 4, MODE, MREP>(iters, sms, dg, dout, dcyc, first);
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 20000;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const int sms = prop.multiProcessorCount;
    const size_t max_words = 1 << 16;                   // 256 KB of random plane words
    std::vector<uint32_t> h(max_words);
    std::mt19937 rng(1234);
    for (auto& v : h) v = rng();
    uint32_t* dg; uint32_t* dout; long long* dcyc;
    CK(cudaMalloc(&dg, max_words * 4));
    CK(cudaMemcpy(dg, h.data(), max_words * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dout, 4096 * THREADS * 4));
    CK(cudaMalloc(&dcyc, 4096 * sizeof(long long)));
    printf("{\"device\":\"%s\",\"sm_count\":%d,\"clock_khz_attr\":%d,\"iters\":%d,\"rows\":%d,"
           "\"k\":%d,\"results\":[\n", prop.name, sms, clock_khz, iters, ROWS, K);
    bool first = true;
    all_cases<0, 1>(iters, sms, dg, dout, dcyc, first);
    all_cases<1, 1>(iters, sms, dg, dout, dcyc, first);
    all_cases<1, 4>(iters / 4, sms, dg, dout, dcyc, first);
    printf("]}\n");
    return 0;
}
