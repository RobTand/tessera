// The fused routed window MoE kernel (tessera#640): one persistent,
// warp-specialised CUDA kernel serves a routed expert stack's gate/up
// projection WITH its SwiGLU epilogue, and a second launch of the same kernel
// serves the down projection with a deterministic route-sorted output.
//
// WHAT IT COMPUTES.  The same functions of the wire as the Triton grouped
// window GEMM (``tessera.window_gemm_grouped``): the 14-bit window state of
// row ``n`` in column ``k`` is the last 14 bits of that column's rate-4
// MSB-first bit stream ending after row ``n``, looked up in the expert's
// table.  The BF16 (value) family is FOLDED -- ``bf16(table[state] *
// row_scale[n])`` before the dot, no epilogue scale -- and the E4M3 family
// runs the epilogue arithmetic ``(acc * a_scale[row]) * row_scale[n]``, both
// in the exact fp32 operation order of the legacy kernel.  The E4M3 table
// entry ``native[codes[state]]`` is composed on the host into an f16 table
// (exact: e4m3 -> f16 is lossless) and the fp8 activation is converted to f16
// on the way into shared memory (also exact), so the E4M3 stack runs on the
// f16 tensor-core instruction with f32 accumulation.
//
// HOW IT IS SCHEDULED.  Work items are ``(expert, n-block, route superblock)``
// triples; their count is a function of the routing, computed on the device
// (``item_off`` = prefix sum of ceil(routes_e / 64)) and claimed through one
// device counter, so the grid is the SM count and no host synchronisation
// exists.  Eight producer warps stream the packed words with ``cp.async``,
// decode them through the shared-memory table into a swizzled 16-bit B tile
// and stage the A tile; eight consumer warps run ``ldmatrix`` + ``mma.sync``
// on a double-buffered stage behind named barriers, and write the epilogue.
// Every weight is decoded once per (item, superblock) and reused across the
// superblock's 64 routes.
//
// DETERMINISM.  The down projection writes each route's bf16-rounded,
// route-weighted row into a ``[routes, H]`` buffer indexed by the route's
// original position; ``token_sum`` then adds each token's ``top_k`` rows in
// fixed route order in fp32 and rounds once.  No atomics anywhere, so two
// runs are bitwise equal.  The device work counter is zeroed by the caller
// inside the same stream (a graph-captured memset), so a captured forward
// replays.
//
// The Python owner is ``tessera.routed_fused``; the contract publishes this
// file as two ``native_extensions`` entries (one per family, see ``ext``).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <limits>

#ifndef TESSERA_ROUTED_FUSED_FP8
#error "compile with -DTESSERA_ROUTED_FUSED_FP8=0 (the value family's library) or =1 (the E4M3 family's)"
#endif

namespace {

// One library per family (``ext.NATIVE_EXTENSIONS`` publishes two entries,
// each with its own lane decoder), so a library never holds a kernel it does
// not serve and the family is part of the library's name.
constexpr bool FAMILY_FP8 = TESSERA_ROUTED_FUSED_FP8 != 0;

constexpr int THREADS = 512;
constexpr int PRODUCER_THREADS = 256;
constexpr int BM = 64;                              // routes per superblock
constexpr int BN = 128;                             // B columns per item (two halves)
constexpr int HALF = 64;
constexpr int BK = 32;                              // k columns per chunk
constexpr int RATE = 4;
constexpr int ROWS_PER_WORD = 32 / RATE;            // 8
constexpr int WORDS_PER_HALF = HALF / ROWS_PER_WORD; // 8
constexpr int CHUNK_WORDS = 16 * RATE;              // words per column per 512-row tile
constexpr int TILE_ROWS = 512;
constexpr int WINDOW_BITS = 14;
constexpr int TABLE_ENTRIES = 1 << WINDOW_BITS;
constexpr int STAGES = 2;
constexpr int WORD_STAGES = 3;

constexpr int TABLE_BYTES = 2 * TABLE_ENTRIES * 2;              // 65536
constexpr int B_STAGE_BYTES = BK * BN * 2;                      // 8192
constexpr int A_STAGE_BYTES = BM * BK * 2;                      // 4096
constexpr int W_STAGE_INTS = 2 * BK * WORDS_PER_HALF;           // 512 ints
constexpr int W_STAGE_BYTES = W_STAGE_INTS * 4;                 // 2048
constexpr int WSCALE_FLOATS = 2 * BN;                           // two item slots
constexpr int DESC_INTS = 2 * 8;
constexpr int OFF_TABLES = 0;
constexpr int OFF_B = OFF_TABLES + TABLE_BYTES;
constexpr int OFF_A = OFF_B + STAGES * B_STAGE_BYTES;
constexpr int OFF_W = OFF_A + STAGES * A_STAGE_BYTES;
constexpr int OFF_WSCALE = OFF_W + WORD_STAGES * W_STAGE_BYTES;
constexpr int OFF_DESC = OFF_WSCALE + WSCALE_FLOATS * 4;
constexpr int OFF_CLAIM = OFF_DESC + DESC_INTS * 4;
constexpr int SMEM_BYTES = OFF_CLAIM + 16;                      // 97,616

constexpr int BAR_FULL0 = 1;
constexpr int BAR_EMPTY0 = 3;
constexpr int BAR_PROD = 5;

__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
template <bool F16>
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
    if constexpr (F16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t v) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
}
__device__ __forceinline__ uint16_t bf16_bits_rn(float f) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(f));
}
__device__ __forceinline__ float bf16_bits_to_f32(uint32_t bits) {
    return __uint_as_float(bits << 16);
}
// The B tile is [k][n] 16-bit with n contiguous (16 chunks of 16 B per k row).
// The swizzle keeps ldmatrix's eight k-rows of one logical chunk on eight bank
// groups and the decoder's eight chunks of one k-row on eight bank groups too.
__device__ __forceinline__ int bswz(int chunk, int k) {
    return (chunk & 8) | ((chunk ^ (k & 7) ^ ((chunk >> 3) << 1)) & 7);
}
// The A tile is [row][k] 16-bit with k contiguous (4 chunks of 16 B per row).
__device__ __forceinline__ int aswz(int chunk, int row) {
    return chunk ^ ((row >> 1) & 3);
}

struct Params {
    const void* x;                 // [rows_x, K] bf16 (value) or e4m3 (fp8)
    const float* a_scale;          // [rows_x] fp32 (fp8) or nullptr
    const int32_t* words0;         // [E, words_stride] gate (mode 0/1) or down (mode 2)
    const int32_t* words1;         // [E, words_stride] up (mode 0/1) or nullptr
    const uint16_t* table0;        // [E, 16384]
    const uint16_t* table1;
    const int32_t* init0;          // [E, K]
    const int32_t* init1;
    const int32_t* has_init0;      // [E]
    const int32_t* has_init1;
    const float* wscale0;          // [E, N]
    const float* wscale1;
    long words_stride;
    int tile_words;
    int K;
    int N;                         // rows per projection
    int E;
    const int32_t* offsets;        // [E + 1]
    const int32_t* flat_sorted;    // [P]
    const float* rw_sorted;        // [P]
    const int32_t* item_off;       // [E + 1] cumulative superblocks
    int32_t* counter;
    int n_blocks;                  // items per superblock
    int top_k;
    int a_row_mode;                // 0 token = flat / top_k, 1 sorted position, 2 flat
    int mul_weight;
    float limit;
    void* out;                     // bf16
    long out_stride;
    int inter;                     // I (mode 1: the up half's column offset)
};

template <bool FP8, int MODE>
__global__ void __launch_bounds__(THREADS, 1) routed_fused_kernel(const Params p) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint16_t* tab = reinterpret_cast<uint16_t*>(smem + OFF_TABLES);
    uint8_t* Bs = smem + OFF_B;
    uint8_t* As = smem + OFF_A;
    int32_t* Ws = reinterpret_cast<int32_t*>(smem + OFF_W);
    float* wsc = reinterpret_cast<float*>(smem + OFF_WSCALE);
    int32_t* desc = reinterpret_cast<int32_t*>(smem + OFF_DESC);
    int32_t* claim = reinterpret_cast<int32_t*>(smem + OFF_CLAIM);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int nk = p.K / BK;
    const int total_items = p.item_off[p.E] * p.n_blocks;
    unsigned gc = 0;          // global chunk counter: the stage is gc & 1
    unsigned item_idx = 0;    // the descriptor slot is item_idx & 1

    if (tid < PRODUCER_THREADS) {
        // ------------------------------------------------------------ producers
        const int col = tid >> 3;   // k column within the chunk this thread decodes
        const int j = tid & 7;      // word (8 rows) within the half
        for (;;) {
            bar_sync(BAR_PROD, PRODUCER_THREADS);   // every producer is done with the last item's smem
            if (tid == 0) claim[0] = atomicAdd(p.counter, 1);
            bar_sync(BAR_PROD, PRODUCER_THREADS);
            const int item = claim[0];
            const int slot = item_idx & 1;
            if (item >= total_items) {
                if (tid == 0) desc[slot * 8 + 0] = -1;
                __threadfence_block();
                bar_arrive(BAR_FULL0 + (gc & 1), THREADS);
                return;
            }
            // item -> (expert, n-block, superblock); items of one expert are
            // contiguous and ordered (n-block, superblock) so that neighbouring
            // items read the same words.
            const int sbg = item / p.n_blocks;
            int lo = 0, hi = p.E;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (p.item_off[mid] <= sbg) lo = mid; else hi = mid;
            }
            const int e = lo;
            const int nsb = p.item_off[e + 1] - p.item_off[e];
            const int local = item - p.item_off[e] * p.n_blocks;
            const int nb = local / nsb;
            const int sb = local - nb * nsb;
            const int start = p.offsets[e];
            const int end = p.offsets[e + 1];
            const int pos0 = start + sb * BM;
            const int mb = min(BM, end - pos0);
            if (tid == 0) {
                desc[slot * 8 + 0] = e;
                desc[slot * 8 + 1] = nb;
                desc[slot * 8 + 2] = sb;
                desc[slot * 8 + 3] = pos0;
                desc[slot * 8 + 4] = mb;
            }
            const int n0 = (MODE == 2) ? nb * BN : nb * HALF;
            // Row scales for the item's 128 B columns, in B-column order.
            if (tid < BN) {
                const int c = tid;
                float v;
                if (MODE == 2) {
                    v = p.wscale0[(long)e * p.N + n0 + c];
                } else {
                    const int q = c >> 5, r = c & 31;
                    const int h = r >> 4;
                    const float* ws = h ? p.wscale1 : p.wscale0;
                    v = ws[(long)e * p.N + n0 + q * 16 + (r & 15)];
                }
                wsc[slot * BN + c] = v;
            }
            // The expert's table(s): 32 KB each, asynchronously.
            {
                const uint16_t* t0 = p.table0 + (long)e * TABLE_ENTRIES;
                for (int i = tid; i < TABLE_ENTRIES / 8; i += PRODUCER_THREADS)
                    cp_async16(tab + i * 8, t0 + i * 8);
                if (MODE != 2) {
                    const uint16_t* t1 = p.table1 + (long)e * TABLE_ENTRIES;
                    for (int i = tid; i < TABLE_ENTRIES / 8; i += PRODUCER_THREADS)
                        cp_async16(tab + TABLE_ENTRIES + i * 8, t1 + i * 8);
                }
            }
            // Word geometry per half: half h decodes rows n_h0 .. n_h0 + 63 of
            // its projection (down: the two halves of one 128-row block; gate/up:
            // the same 64 intermediate rows of two projections).
            const int32_t* wbase[2];
            int g_h[2], wr0_h[2];
            const int32_t* init_h[2];
            int hasinit_h[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nh0 = (MODE == 2) ? n0 + HALF * h : n0;
                const int g = nh0 / TILE_ROWS;
                const int t = nh0 - g * TILE_ROWS;
                const bool second = (MODE != 2) && h == 1;
                g_h[h] = g;
                wr0_h[h] = t / ROWS_PER_WORD;
                wbase[h] = (second ? p.words1 : p.words0) + (long)e * p.words_stride
                           + (long)g * p.tile_words + wr0_h[h];
                init_h[h] = (second ? p.init1 : p.init0) + (long)e * p.K;
                hasinit_h[h] = (second ? p.has_init1 : p.has_init0)[e];
            }
            // The A row this thread stages (-1: a zero row past the superblock).
            long arow = -1;
            {
                const int r = FP8 ? (tid >> 1) : (tid >> 2);
                if ((!FP8 || tid < 128) && r < mb) {
                    const int pos = pos0 + r;
                    const int flat = p.flat_sorted[pos];
                    arow = (p.a_row_mode == 1) ? pos : (p.a_row_mode == 0 ? flat / p.top_k : flat);
                }
            }
            auto issue_words = [&](int kc) {
                if (tid < 128) {
                    const int h = tid >> 6;
                    const int c = (tid >> 1) & 31;
                    const int q = tid & 1;
                    const int32_t* src = wbase[h] + (long)(kc * BK + c) * CHUNK_WORDS + q * 4;
                    cp_async16(Ws + (kc % WORD_STAGES) * W_STAGE_INTS + (h * BK + c) * WORDS_PER_HALF + q * 4, src);
                }
            };
            auto load_prev = [&](int kc, int32_t (&pv)[2]) {
                if (j == 0) {
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const long kcol = (long)kc * BK + col;
                        int32_t v;
                        if (wr0_h[h] > 0) v = wbase[h][kcol * CHUNK_WORDS - 1];
                        else if (g_h[h] > 0) v = wbase[h][kcol * CHUNK_WORDS - p.tile_words + CHUNK_WORDS - 1];
                        else v = hasinit_h[h] ? init_h[h][kcol] : 0;
                        pv[h] = v;
                    }
                }
            };
            auto load_a = [&](int kc, uint4& a) {
                if (arow >= 0) {
                    if constexpr (FP8) {
                        const uint8_t* src = reinterpret_cast<const uint8_t*>(p.x) + arow * p.K + kc * BK + (tid & 1) * 16;
                        a = *reinterpret_cast<const uint4*>(src);
                    } else {
                        const uint16_t* src = reinterpret_cast<const uint16_t*>(p.x) + arow * p.K + kc * BK + (tid & 3) * 8;
                        a = *reinterpret_cast<const uint4*>(src);
                    }
                }
            };
            auto store_a = [&](int stage, const uint4& a) {
                uint8_t* A = As + stage * A_STAGE_BYTES;
                if constexpr (FP8) {
                    if (tid >= 128) return;
                    const int row = tid >> 1;
                    const int c16 = tid & 1;
                    uint4 lo, hi;
                    if (arow >= 0) {
                        lo.x = e4m3x2_to_f16x2((uint16_t)(a.x & 0xFFFF)); lo.y = e4m3x2_to_f16x2((uint16_t)(a.x >> 16));
                        lo.z = e4m3x2_to_f16x2((uint16_t)(a.y & 0xFFFF)); lo.w = e4m3x2_to_f16x2((uint16_t)(a.y >> 16));
                        hi.x = e4m3x2_to_f16x2((uint16_t)(a.z & 0xFFFF)); hi.y = e4m3x2_to_f16x2((uint16_t)(a.z >> 16));
                        hi.z = e4m3x2_to_f16x2((uint16_t)(a.w & 0xFFFF)); hi.w = e4m3x2_to_f16x2((uint16_t)(a.w >> 16));
                    } else {
                        lo = make_uint4(0, 0, 0, 0); hi = lo;
                    }
                    *reinterpret_cast<uint4*>(A + row * (BK * 2) + (aswz(2 * c16, row) << 4)) = lo;
                    *reinterpret_cast<uint4*>(A + row * (BK * 2) + (aswz(2 * c16 + 1, row) << 4)) = hi;
                } else {
                    const int row = tid >> 2;
                    const int c = tid & 3;
                    const uint4 v = (arow >= 0) ? a : make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(A + row * (BK * 2) + (aswz(c, row) << 4)) = v;
                }
            };

            issue_words(0);
            cp_async_commit();                     // group 0: the tables and chunk 0's words
            if (nk > 1) issue_words(1);
            cp_async_commit();                     // group 1
            int32_t prev_cur[2] = {0, 0}, prev_nxt[2] = {0, 0};
            uint4 a_cur = make_uint4(0, 0, 0, 0), a_nxt = make_uint4(0, 0, 0, 0);
            load_prev(0, prev_cur);
            load_a(0, a_cur);
            for (int kc = 0; kc < nk; ++kc, ++gc) {
                if (kc + 1 < nk) { load_prev(kc + 1, prev_nxt); load_a(kc + 1, a_nxt); }
                cp_async_wait<1>();                // chunk kc's words (and the tables) have landed
                bar_sync(BAR_PROD, PRODUCER_THREADS);   // ... for every producer; chunk kc-1's stage is free
                if (kc + 2 < nk) issue_words(kc + 2);
                cp_async_commit();
                const int stage = gc & 1;
                if (gc >= 2) bar_sync(BAR_EMPTY0 + stage, THREADS);
                store_a(stage, a_cur);
                const int32_t* W = Ws + (kc % WORD_STAGES) * W_STAGE_INTS;
                uint8_t* B = Bs + stage * B_STAGE_BYTES;
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t w = (uint32_t)W[(h * BK + col) * WORDS_PER_HALF + j];
                    uint32_t v = (uint32_t)__shfl_up_sync(0xffffffffu, (int)w, 1);
                    if (j == 0) v = (uint32_t)prev_cur[h];
                    const uint16_t* T = tab + ((MODE == 2) ? 0 : h * TABLE_ENTRIES);
                    const int chunk = (MODE == 2) ? (8 * h + j) : (4 * (j >> 1) + 2 * h + (j & 1));
                    const float* ws = wsc + slot * BN + chunk * 8;
                    uint32_t packed[4];
                    #pragma unroll
                    for (int r = 0; r < 8; r += 2) {
                        uint32_t s0 = __funnelshift_r(w, v, 28 - 4 * r) & 0x3FFFu;
                        uint32_t s1 = __funnelshift_r(w, v, 28 - 4 * (r + 1)) & 0x3FFFu;
                        uint32_t t0 = T[s0], t1 = T[s1];
                        if constexpr (!FP8) {
                            // FOLDED: one bf16 rounding of value * row_scale, before the dot
                            t0 = bf16_bits_rn(__fmul_rn(bf16_bits_to_f32(t0), ws[r]));
                            t1 = bf16_bits_rn(__fmul_rn(bf16_bits_to_f32(t1), ws[r + 1]));
                        }
                        packed[r >> 1] = t0 | (t1 << 16);
                    }
                    *reinterpret_cast<uint4*>(B + col * (BN * 2) + (bswz(chunk, col) << 4)) =
                        make_uint4(packed[0], packed[1], packed[2], packed[3]);
                }
                bar_arrive(BAR_FULL0 + stage, THREADS);
                prev_cur[0] = prev_nxt[0]; prev_cur[1] = prev_nxt[1];
                a_cur = a_nxt;
            }
            ++item_idx;
        }
    } else {
        // ------------------------------------------------------------ consumers
        const int cw = (tid - PRODUCER_THREADS) >> 5;
        const int mw = cw >> 2;      // 32 rows
        const int nw = cw & 3;       // 32 B columns
        for (;;) {
            const int slot = item_idx & 1;
            int stage = gc & 1;
            bar_sync(BAR_FULL0 + stage, THREADS);
            const int e = desc[slot * 8 + 0];
            if (e < 0) return;
            const int nb = desc[slot * 8 + 1];
            const int pos0 = desc[slot * 8 + 3];
            const int mb = desc[slot * 8 + 4];
            float acc[2][4][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) acc[mi][nt][i] = 0.f;
            for (int kc = 0; kc < nk; ++kc, ++gc) {
                stage = gc & 1;
                if (kc > 0) bar_sync(BAR_FULL0 + stage, THREADS);
                const uint8_t* A = As + stage * A_STAGE_BYTES;
                const uint8_t* B = Bs + stage * B_STAGE_BYTES;
                #pragma unroll
                for (int s = 0; s < BK / 16; ++s) {
                    uint32_t a[2][4];
                    const int q = lane >> 3;
                    #pragma unroll
                    for (int mi = 0; mi < 2; ++mi) {
                        const int row = 32 * mw + 16 * mi + 8 * (q & 1) + (lane & 7);
                        const int kch = 2 * s + (q >> 1);
                        ldmatrix_x4(a[mi], A + row * (BK * 2) + (aswz(kch, row) << 4));
                    }
                    uint32_t b[4][2];
                    #pragma unroll
                    for (int pair = 0; pair < 2; ++pair) {
                        const int rk = 16 * s + 8 * (q & 1) + (lane & 7);
                        const int ch = 4 * nw + 2 * pair + (q >> 1);
                        uint32_t r[4];
                        ldmatrix_x4_trans(r, B + rk * (BN * 2) + (bswz(ch, rk) << 4));
                        b[2 * pair][0] = r[0]; b[2 * pair][1] = r[1];
                        b[2 * pair + 1][0] = r[2]; b[2 * pair + 1][1] = r[3];
                    }
                    #pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) mma16816<FP8>(acc[mi][nt], a[mi], b[nt]);
                }
                bar_arrive(BAR_EMPTY0 + stage, THREADS);
            }
            // ------------------------------------------------------ epilogue
            const int n0 = (MODE == 2) ? nb * BN : nb * HALF;
            uint16_t* out = reinterpret_cast<uint16_t*>(p.out);
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                #pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int r = 32 * mw + 16 * mi + 8 * hr + (lane >> 2);
                    if (r >= mb) continue;
                    const int pos = pos0 + r;
                    const int flat = p.flat_sorted[pos];
                    float a_s = 1.f, rw = 1.f;
                    if constexpr (FP8) {
                        const long arow = (p.a_row_mode == 1) ? pos : (p.a_row_mode == 0 ? flat / p.top_k : flat);
                        a_s = p.a_scale[arow];
                    }
                    if (MODE == 2 && p.mul_weight) rw = p.rw_sorted[pos];
                    if constexpr (MODE == 0) {
                        // gate n-tiles 0,1 pair with up n-tiles 2,3 in the same registers
                        #pragma unroll
                        for (int nt = 0; nt < 2; ++nt) {
                            const int cb = 32 * nw + 8 * nt + 2 * (lane & 3);      // gate B column
                            uint32_t two = 0;
                            #pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                float g = acc[mi][nt][2 * hr + i];
                                float u = acc[mi][nt + 2][2 * hr + i];
                                if constexpr (FP8) {
                                    g = __fmul_rn(__fmul_rn(g, a_s), wsc[slot * BN + cb + i]);
                                    u = __fmul_rn(__fmul_rn(u, a_s), wsc[slot * BN + cb + 16 + i]);
                                }
                                // the bf16 GEMM output, widened for the fp32 activation
                                float gf = bf16_bits_to_f32(bf16_bits_rn(g));
                                float uf = bf16_bits_to_f32(bf16_bits_rn(u));
                                gf = fminf(gf, p.limit);
                                uf = fmaxf(fminf(uf, p.limit), -p.limit);
                                const float act = __fmul_rn(gf / (1.0f + expf(-gf)), uf);
                                two |= (uint32_t)bf16_bits_rn(act) << (16 * i);
                            }
                            const long col = n0 + 16 * nw + 8 * nt + 2 * (lane & 3);
                            *reinterpret_cast<uint32_t*>(out + (long)pos * p.out_stride + col) = two;
                        }
                    } else if constexpr (MODE == 1) {
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const int cb = 32 * nw + 8 * nt + 2 * (lane & 3);
                            const int h = nt >> 1;
                            const int nl = 16 * nw + 8 * (nt & 1) + 2 * (lane & 3);
                            uint32_t two = 0;
                            #pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                float y = acc[mi][nt][2 * hr + i];
                                if constexpr (FP8) y = __fmul_rn(__fmul_rn(y, a_s), wsc[slot * BN + cb + i]);
                                two |= (uint32_t)bf16_bits_rn(y) << (16 * i);
                            }
                            const long col = (long)h * p.inter + n0 + nl;
                            *reinterpret_cast<uint32_t*>(out + (long)flat * p.out_stride + col) = two;
                        }
                    } else {
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const int cb = 32 * nw + 8 * nt + 2 * (lane & 3);
                            uint32_t two = 0;
                            #pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                float y = acc[mi][nt][2 * hr + i];
                                if constexpr (FP8) y = __fmul_rn(__fmul_rn(y, a_s), wsc[slot * BN + cb + i]);
                                if (p.mul_weight) y = __fmul_rn(y, rw);
                                two |= (uint32_t)bf16_bits_rn(y) << (16 * i);
                            }
                            const long col = n0 + cb;
                            *reinterpret_cast<uint32_t*>(out + (long)flat * p.out_stride + col) = two;
                        }
                    }
                }
            }
            ++item_idx;
        }
    }
}

// out[t, :] = bf16( sum_{j < top_k} f32(routed[t * top_k + j, :]) ), fixed order.
__global__ void token_sum_kernel(const uint16_t* __restrict__ routed, uint16_t* __restrict__ out,
                                 long tokens, int top_k, long width) {
    const long vec = (long)blockIdx.x * blockDim.x + threadIdx.x;   // one 8-column vector
    const long vecs_per_row = width / 8;
    if (vec >= tokens * vecs_per_row) return;
    const long t = vec / vecs_per_row;
    const long c = (vec - t * vecs_per_row) * 8;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    for (int j = 0; j < top_k; ++j) {
        const uint4 v = *reinterpret_cast<const uint4*>(routed + (t * top_k + j) * width + c);
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[2 * i] += bf16_bits_to_f32(w[i] & 0xFFFFu);
            acc[2 * i + 1] += bf16_bits_to_f32(w[i] >> 16);
        }
    }
    uint4 o;
    o.x = bf16_bits_rn(acc[0]) | ((uint32_t)bf16_bits_rn(acc[1]) << 16);
    o.y = bf16_bits_rn(acc[2]) | ((uint32_t)bf16_bits_rn(acc[3]) << 16);
    o.z = bf16_bits_rn(acc[4]) | ((uint32_t)bf16_bits_rn(acc[5]) << 16);
    o.w = bf16_bits_rn(acc[6]) | ((uint32_t)bf16_bits_rn(acc[7]) << 16);
    *reinterpret_cast<uint4*>(out + t * width + c) = o;
}

template <bool FP8, int MODE>
void launch(const Params& p, int grid, cudaStream_t stream) {
    static bool attributed = false;
    if (!attributed) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(routed_fused_kernel<FP8, MODE>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        attributed = true;
    }
    routed_fused_kernel<FP8, MODE><<<grid, THREADS, SMEM_BYTES, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

const int32_t* i32_ptr(const torch::Tensor& t) { return t.data_ptr<int32_t>(); }
const float* f32_ptr(const torch::Tensor& t) { return t.data_ptr<float>(); }
const uint16_t* u16_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const uint16_t*>(t.data_ptr<int16_t>());
}

}  // namespace

// One launch of the persistent kernel.  ``mode``: 0 gate/up + SwiGLU (out is
// the sorted [routes, I] activation), 1 gate/up preserved (out is [T, top_k,
// 2I] by route), 2 down (out is [routes, H] by route, bf16-rounded, weighted).
void routed_fused_forward(
    int64_t mode, bool fp8,
    torch::Tensor x, torch::Tensor a_scale,
    torch::Tensor words0, torch::Tensor words1,
    torch::Tensor table0, torch::Tensor table1,
    torch::Tensor init0, torch::Tensor init1,
    torch::Tensor has_init0, torch::Tensor has_init1,
    torch::Tensor wscale0, torch::Tensor wscale1,
    int64_t tile_words,
    torch::Tensor offsets, torch::Tensor flat_sorted, torch::Tensor rw_sorted,
    torch::Tensor item_off, torch::Tensor counter,
    int64_t top_k, int64_t a_row_mode, bool mul_weight, double limit,
    torch::Tensor out, int64_t grid) {
    TORCH_CHECK(mode >= 0 && mode <= 2, "mode must be 0, 1 or 2");
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous 2-D CUDA tensor");
    const int64_t K = x.size(1);
    TORCH_CHECK(K % BK == 0 && K >= 4 * BK, "K must be a multiple of ", BK, " and at least ", 4 * BK);
    if (fp8) {
        TORCH_CHECK(x.scalar_type() == torch::kFloat8_e4m3fn, "the E4M3 family takes an e4m3 x");
        TORCH_CHECK(a_scale.is_cuda() && a_scale.scalar_type() == torch::kFloat32
                    && a_scale.is_contiguous() && a_scale.numel() == x.size(0),
                    "a_scale must be fp32 [rows_x]");
    } else {
        TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "the value family takes a bf16 x");
    }
    const bool two = mode != 2;
    const int64_t E = wscale0.size(0);
    const int64_t N = wscale0.size(1);
    TORCH_CHECK(words0.is_cuda() && words0.dim() == 2 && words0.size(0) == E
                && words0.scalar_type() == torch::kInt32 && words0.is_contiguous(), "words0 must be int32 [E, W]");
    TORCH_CHECK(table0.dim() == 2 && table0.size(0) == E && table0.size(1) == TABLE_ENTRIES
                && table0.scalar_type() == torch::kInt16 && table0.is_contiguous(), "table0 must be int16 [E, 16384]");
    TORCH_CHECK(init0.dim() == 2 && init0.size(0) == E && init0.size(1) == K
                && init0.scalar_type() == torch::kInt32 && init0.is_contiguous(), "init0 must be int32 [E, K]");
    TORCH_CHECK(has_init0.numel() == E && has_init0.scalar_type() == torch::kInt32, "has_init0 must be int32 [E]");
    TORCH_CHECK(wscale0.scalar_type() == torch::kFloat32 && wscale0.is_contiguous(), "wscale0 must be fp32 [E, N]");
    if (two) {
        TORCH_CHECK(words1.sizes() == words0.sizes() && words1.scalar_type() == torch::kInt32 && words1.is_contiguous(),
                    "words1 must match words0");
        TORCH_CHECK(table1.sizes() == table0.sizes() && table1.scalar_type() == torch::kInt16 && table1.is_contiguous(),
                    "table1 must match table0");
        TORCH_CHECK(init1.sizes() == init0.sizes() && init1.scalar_type() == torch::kInt32 && init1.is_contiguous(),
                    "init1 must match init0");
        TORCH_CHECK(has_init1.numel() == E && has_init1.scalar_type() == torch::kInt32, "has_init1 must be int32 [E]");
        TORCH_CHECK(wscale1.sizes() == wscale0.sizes() && wscale1.scalar_type() == torch::kFloat32 && wscale1.is_contiguous(),
                    "wscale1 must match wscale0");
        TORCH_CHECK(N % HALF == 0, "the intermediate size must be a multiple of ", HALF);
    } else {
        TORCH_CHECK(N % BN == 0, "the down rows must be a multiple of ", BN);
    }
    TORCH_CHECK(offsets.numel() == E + 1 && offsets.scalar_type() == torch::kInt32 && offsets.is_contiguous(),
                "offsets must be int32 [E + 1]");
    TORCH_CHECK(item_off.numel() == E + 1 && item_off.scalar_type() == torch::kInt32 && item_off.is_contiguous(),
                "item_off must be int32 [E + 1]");
    TORCH_CHECK(flat_sorted.scalar_type() == torch::kInt32 && flat_sorted.is_contiguous(), "flat_sorted must be int32");
    TORCH_CHECK(rw_sorted.scalar_type() == torch::kFloat32 && rw_sorted.is_contiguous()
                && rw_sorted.numel() == flat_sorted.numel(), "rw_sorted must be fp32 [P]");
    TORCH_CHECK(counter.scalar_type() == torch::kInt32 && counter.numel() >= 1, "counter must be int32");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.is_contiguous(), "out must be contiguous bf16");
    const int64_t P = flat_sorted.numel();
    const int64_t out_stride = out.size(-1);
    if (mode == 0) {
        TORCH_CHECK(out.dim() == 2 && out.size(0) == P && out.size(1) == N, "mode 0 out must be [P, I]");
    } else if (mode == 1) {
        TORCH_CHECK(out.dim() == 3 && out.size(0) * out.size(1) == P && out.size(1) == top_k && out.size(2) == 2 * N,
                    "mode 1 out must be [T, top_k, 2I]");
    } else {
        TORCH_CHECK(out.dim() == 2 && out.size(0) == P && out.size(1) == N, "mode 2 out must be [P, H]");
    }
    TORCH_CHECK(tile_words == K * CHUNK_WORDS, "tile_words must be cols * ", CHUNK_WORDS, " (rate 4)");
    TORCH_CHECK(grid >= 1, "grid must be positive");

    Params p{};
    p.x = x.data_ptr();
    p.a_scale = fp8 ? f32_ptr(a_scale) : nullptr;
    p.words0 = i32_ptr(words0);
    p.words1 = two ? i32_ptr(words1) : nullptr;
    p.table0 = u16_ptr(table0);
    p.table1 = two ? u16_ptr(table1) : nullptr;
    p.init0 = i32_ptr(init0);
    p.init1 = two ? i32_ptr(init1) : nullptr;
    p.has_init0 = i32_ptr(has_init0);
    p.has_init1 = two ? i32_ptr(has_init1) : nullptr;
    p.wscale0 = f32_ptr(wscale0);
    p.wscale1 = two ? f32_ptr(wscale1) : nullptr;
    p.words_stride = words0.size(1);
    p.tile_words = (int)tile_words;
    p.K = (int)K;
    p.N = (int)N;
    p.E = (int)E;
    p.offsets = i32_ptr(offsets);
    p.flat_sorted = i32_ptr(flat_sorted);
    p.rw_sorted = f32_ptr(rw_sorted);
    p.item_off = i32_ptr(item_off);
    p.counter = counter.data_ptr<int32_t>();
    p.n_blocks = (int)(two ? N / HALF : N / BN);
    p.top_k = (int)top_k;
    p.a_row_mode = (int)a_row_mode;
    p.mul_weight = mul_weight ? 1 : 0;
    p.limit = std::isfinite(limit) ? (float)limit : std::numeric_limits<float>::infinity();
    p.out = out.data_ptr();
    p.out_stride = out_stride;
    p.inter = (int)N;

    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int g = (int)grid;
    TORCH_CHECK(fp8 == FAMILY_FP8, "this library serves the ", FAMILY_FP8 ? "E4M3" : "value",
                " family only; the other family's library is a separate native extension");
    if (mode == 0) launch<FAMILY_FP8, 0>(p, g, stream);
    else if (mode == 1) launch<FAMILY_FP8, 1>(p, g, stream);
    else launch<FAMILY_FP8, 2>(p, g, stream);
}

void token_sum(torch::Tensor routed, torch::Tensor out, int64_t top_k) {
    TORCH_CHECK(routed.is_cuda() && routed.dim() == 2 && routed.scalar_type() == torch::kBFloat16
                && routed.is_contiguous(), "routed must be contiguous bf16 [P, H]");
    TORCH_CHECK(out.dim() == 2 && out.scalar_type() == torch::kBFloat16 && out.is_contiguous()
                && out.size(1) == routed.size(1) && out.size(0) * top_k == routed.size(0),
                "out must be bf16 [T, H] with T * top_k == P");
    TORCH_CHECK(routed.size(1) % 8 == 0, "H must be a multiple of 8");
    const long tokens = out.size(0);
    const long width = out.size(1);
    const long vecs = tokens * (width / 8);
    if (vecs == 0) return;
    const c10::cuda::CUDAGuard guard(routed.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const long blocks = (vecs + threads - 1) / threads;
    token_sum_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(routed.data_ptr()), reinterpret_cast<uint16_t*>(out.data_ptr()),
        tokens, (int)top_k, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("routed_fused_forward", &routed_fused_forward);
    m.def("token_sum", &token_sum);
    m.attr("BM") = BM;
    m.attr("BN") = BN;
    m.attr("HALF") = HALF;
    m.attr("BK") = BK;
    m.attr("RATE") = RATE;
    m.attr("WINDOW_BITS") = WINDOW_BITS;
    m.attr("SMEM_BYTES") = SMEM_BYTES;
    m.attr("THREADS") = THREADS;
    m.attr("FAMILY_FP8") = FAMILY_FP8;
}
