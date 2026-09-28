// The fused routed window MoE kernel (tessera#640): one persistent,
// warp-specialised CUDA kernel serves a routed expert stack's gate/up
// projection WITH its SwiGLU epilogue, and a second launch of the same kernel
// serves the down projection with a deterministic route-sorted output.
//
// WHAT IT COMPUTES.  The same functions of the wire as the Triton grouped
// window GEMM (``tessera.window_gemm_grouped``): the 14-bit window state of
// row ``n`` in column ``k`` is the last 14 bits of that column's MSB-first bit
// stream (``rate`` bits per row, any rate 1..8, per column -- the run table's
// one or two rates) ending after row ``n``, looked up in the expert's
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
// THE DENSE CASE.  A dense Linear is the E = 1, top_k = 1, unweighted case of
// the down projection: ``routed_fused_kernel<FP8, 2, DENSE=true, SPLIT>`` reads
// the same words/table/scale planes for one "expert", takes row ``m`` of ``x``
// as route ``m`` (no routing tables are read), and writes ``y[m, n]`` straight
// into a column slice of the module's output (``out_stride`` is the slice's row
// stride), so a merged Linear's roles are one launch each and no concatenation
// follows.  When the item count ``ceil(M / 64) * N / 128`` would leave SMs idle
// (decode: M <= 64 on a 4096-row role is 32 items for 48 SMs) the K range is
// split ``S`` ways (``k_split``); each split accumulates its chunk range into
// an fp32 workspace ``[S, M, N]`` and ``dense_reduce_kernel`` sums the ``S``
// partials in fixed order and applies the epilogue -- deterministic, and the
// same fp32 operation order as the unsplit epilogue once the sum is formed.
// The dense identity is ``tessera::fused_window_dense`` (``serving.native_
// window``), decoders ``native_fused_window_dense`` / ``..._folded``.
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
// Column rates the decode reads (bits per code): a column's 512-row chunk is
// ``16 * rate`` int32 words, a 64-row half of it ``2 * rate`` words.  Rates 4
// and 8 put a lane's eight rows on word boundaries; every other rate re-aligns
// the lane's window once per chunk (``decode_rows``).  The run table's rates
// are the grammar's -- at most the two bracketing the root
// (``grammar.rate_set``) -- and any of 1..8 is read.
constexpr int RATE_MIN = 1;
constexpr int RATE_MAX = 8;
constexpr int BDESC_INTS = 12;                      // per-32-column descriptor (see ``col_map``)
constexpr int TILE_ROWS = 512;
constexpr int WINDOW_BITS = 14;
constexpr int TABLE_ENTRIES = 1 << WINDOW_BITS;
constexpr int STAGES = 2;
constexpr int WORD_STAGES = 3;

constexpr int TABLE_BYTES = TABLE_ENTRIES * 2;                  // 32768, one table
constexpr int B_STAGE_BYTES = BK * BN * 2;                      // 8192
constexpr int A_STAGE_BYTES = BM * BK * 2;                      // 4096
constexpr int WSCALE_FLOATS = 2 * BN;                           // two item slots
constexpr int DESC_INTS = 2 * 8;
// The shared-memory layout.  The word stages come LAST and are sized at
// launch by the stack's rates: ``Params::slot_words`` int32 words per (half,
// column) slot -- at least 2 * rate (+1 where a lane's window over-reads one
// word past the column's 2 * rate words at a rate whose 8 * rate is not a
// multiple of 32), rounded to a multiple of 4 so the 16-byte copies stay
// aligned.  A block on sm_121 may opt in to 101,376 B of dynamic shared
// memory; the two-table gate/up launch needs 91,216 + 768 * slot_words, so it
// fits slots up to 12 words (rates <= 5) and not the 16-word slot of rates
// 6..8; the one-table down/dense launch (MODE 2) fits every rate.  The host
// entries check the launch against the device's own limit.
template <int MODE> struct Layout {
    static constexpr int TABLES = (MODE == 2) ? 1 : 2;
    static constexpr int OFF_TABLES = 0;
    static constexpr int OFF_B = OFF_TABLES + TABLES * TABLE_BYTES;
    static constexpr int OFF_A = OFF_B + STAGES * B_STAGE_BYTES;
    static constexpr int OFF_WSCALE = OFF_A + STAGES * A_STAGE_BYTES;
    static constexpr int OFF_DESC = OFF_WSCALE + WSCALE_FLOATS * 4;
    static constexpr int OFF_CLAIM = OFF_DESC + DESC_INTS * 4;
    static constexpr int OFF_W = OFF_CLAIM + 16;                // 91,216 (two tables) / 58,448 (one)
};
constexpr int SLOT_WORDS_MAX = 2 * RATE_MAX;                    // 16: the rate-8 slot
__host__ __device__ constexpr int w_stage_ints(int slot_words) { return 2 * BK * slot_words; }
__host__ __device__ constexpr int smem_bytes(int mode, int slot_words) {
    return (mode == 2 ? Layout<2>::OFF_W : Layout<0>::OFF_W) + WORD_STAGES * w_stage_ints(slot_words) * 4;
}
// The slot one column at ``rate`` needs: its 2 * rate words plus the one word
// a lane's re-aligned window reads past them when 8 * rate is not a multiple
// of 32 (the bits it holds enter no field).
__host__ __device__ constexpr int slot_words_for_rate(int rate) {
    return 2 * rate + (((8 * rate) % 32) != 0 ? 1 : 0);
}

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
// 8-byte copies serve the odd rates: a 64-row half at rate r is 8 * r bytes
// at an 8 * r * (rows / 64)-byte offset, 16-byte aligned only for even r.
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(s), "l"(gmem) : "memory");
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

// The run table, as the kernel reads it.  The wire's columns are sorted by
// (rate, column) into one or two contiguous runs of the PERMUTED column order
// (``kernel_window_gemv.repack_window_body``); the kernel keeps A and B in
// ORIGINAL column order and maps each original column to its run and its rank
// within the run.  ``runs`` is int32 [E, 8]: (r_lo, 0, n_lo, 0, r_hi, n_lo,
// n_hi, w_hi) with n_hi = 0 when the unit is one run.  ``bdesc`` is int32
// [E, K / 32, BDESC_INTS] per 32-column block: words 0..7 hold 32 bytes, the
// original in-block position of the block's low-rate columns in order, then
// of its high-rate columns; word 8 is the low-rate column count before the
// block, word 9 the block's low-rate column count.  Lane group ``m`` of a
// chunk decodes the block's m-th column in THAT order, so a warp's four
// columns share a rate except in the one warp straddling the two.
struct RunPair { int r_lo, n_lo, r_hi, n_hi, w_hi; };
struct ColMap {
    int rate;   // the column's rate
    int cib;    // its original position within the block: the B-tile column
    int p;      // its permuted index (the run table's column order): init[p]
    int cw0;    // the first word of its chunk within a 512-row tile
};
__device__ __forceinline__ RunPair load_runs(const int32_t* runs, int e) {
    const int4 a = *reinterpret_cast<const int4*>(runs + (long)e * 8);
    const int4 b = *reinterpret_cast<const int4*>(runs + (long)e * 8 + 4);
    RunPair r;
    r.r_lo = a.x; r.n_lo = a.z; r.r_hi = b.x; r.n_hi = b.z; r.w_hi = b.w;
    return r;
}
__device__ __forceinline__ ColMap col_map(const int32_t* bdesc, const RunPair& rp, int kc, int m) {
    const int32_t* blk = bdesc + (long)kc * BDESC_INTS;
    const int cib = (blk[m >> 2] >> (8 * (m & 3))) & 0xFF;
    const int2 counts = *reinterpret_cast<const int2*>(blk + 8);   // (n_lo_before, cnt_lo)
    const bool lo = m < counts.y;
    const int rank = lo ? counts.x + m : (kc * BK - counts.x) + (m - counts.y);
    ColMap c;
    c.rate = lo ? rp.r_lo : rp.r_hi;
    c.cib = cib;
    c.p = lo ? rank : rp.n_lo + rank;
    c.cw0 = lo ? rank * 16 * rp.r_lo : rp.w_hi + rank * 16 * rp.r_hi;
    return c;
}

// Eight rows of one column of one half, from the half's word slot: the
// window state of row ``n`` is the last 14 bits of the column's MSB-first
// stream ending after row ``n``.  Lane ``j`` holds rows 8j..8j+7, whose bits
// start at stream bit ``8 * j * R`` of the half: word ``b`` of the slot, bit
// ``u`` into it.  The lane re-aligns a three-word window on ``u`` once
// (``Z0 | Z1 | Z2`` = the 96 stream bits from 32 before its first row), after
// which every row's field sits at a compile-time position; at rates 4 and 8
// ``u`` is 0 and the window is the slot's words themselves -- the rate-4 path
// is exactly the original kernel's constant shifts.  ``prev`` is the word
// before the half's first word (the previous 64 rows, the previous tile's
// last word of the column, the cut's start state, or zero).
template <bool FP8, int R>
__device__ __forceinline__ void decode_rows(const int32_t* Wc, uint32_t prev, int j,
                                            const uint16_t* T, const float* ws,
                                            uint32_t (&packed)[4]) {
    constexpr bool ALIGNED = (8 * R) % 32 == 0;
    constexpr bool WIDE = 8 * R > 32;                 // rows reach past 32 bits: Z2 is read
    const int bits0 = 8 * j * R;
    const int b = bits0 >> 5;
    const uint32_t wm1 = (b > 0) ? (uint32_t)Wc[b - 1] : prev;
    const uint32_t w0 = (uint32_t)Wc[b];
    uint32_t Z0, Z1, Z2 = 0;
    if constexpr (ALIGNED) {
        Z0 = wm1;
        Z1 = w0;
        if constexpr (WIDE) Z2 = (uint32_t)Wc[b + 1];
    } else {
        const int u = bits0 & 31;                     // 1..31 here, 0 only for j = 0
        const uint32_t w1 = (uint32_t)Wc[b + 1];
        // the 32 stream bits starting u into (hi:lo), MSB-first; u = 0 gives hi
        Z0 = __funnelshift_rc(w0, wm1, 32 - u);
        Z1 = __funnelshift_rc(w1, w0, 32 - u);
        if constexpr (WIDE) {
            const uint32_t w2 = (uint32_t)Wc[b + 2];
            Z2 = __funnelshift_rc(w2, w1, 32 - u);
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r += 2) {
        uint32_t s[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int e = (r + i + 1) * R;            // the field ends here, in the window
            const int k1 = (e - 1) >> 5;              // 0: (Z0:Z1), 1: (Z1:Z2)
            const int shift = 32 * (k1 + 1) - e;
            const uint32_t lo = k1 ? Z2 : Z1;
            const uint32_t hi = k1 ? Z1 : Z0;
            s[i] = __funnelshift_r(lo, hi, shift) & 0x3FFFu;
        }
        uint32_t t0 = T[s[0]], t1 = T[s[1]];
        if constexpr (!FP8) {
            // FOLDED: one bf16 rounding of value * row_scale, before the dot
            t0 = bf16_bits_rn(__fmul_rn(bf16_bits_to_f32(t0), ws[r]));
            t1 = bf16_bits_rn(__fmul_rn(bf16_bits_to_f32(t1), ws[r + 1]));
        }
        packed[r >> 1] = t0 | (t1 << 16);
    }
}
template <bool FP8>
__device__ __forceinline__ void decode_rows_at(int rate, const int32_t* Wc, uint32_t prev, int j,
                                               const uint16_t* T, const float* ws,
                                               uint32_t (&packed)[4]) {
    switch (rate) {
        case 1: decode_rows<FP8, 1>(Wc, prev, j, T, ws, packed); break;
        case 2: decode_rows<FP8, 2>(Wc, prev, j, T, ws, packed); break;
        case 3: decode_rows<FP8, 3>(Wc, prev, j, T, ws, packed); break;
        case 4: decode_rows<FP8, 4>(Wc, prev, j, T, ws, packed); break;
        case 5: decode_rows<FP8, 5>(Wc, prev, j, T, ws, packed); break;
        case 6: decode_rows<FP8, 6>(Wc, prev, j, T, ws, packed); break;
        case 7: decode_rows<FP8, 7>(Wc, prev, j, T, ws, packed); break;
        default: decode_rows<FP8, 8>(Wc, prev, j, T, ws, packed); break;
    }
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
    const int32_t* runs0;          // [E, 8] run pairs (see RunPair)
    const int32_t* runs1;
    const int32_t* bdesc0;         // [E, K / 32, BDESC_INTS] block descriptors (see col_map)
    const int32_t* bdesc1;
    long words_stride;
    int tile_words;
    int slot_words;                // int32 words per (half, column) word-stage slot (see Layout)
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
    int rows_x;                    // DENSE: M, the rows of x (and of out)
    int k_split;                   // DENSE: S, the K-range splits per (n-block, superblock)
    float* partial;                // DENSE && SPLIT: fp32 [S, M, N] raw accumulators
};

template <bool FP8, int MODE, bool DENSE, bool SPLIT>
__global__ void __launch_bounds__(THREADS, 1) routed_fused_kernel(const Params p) {
    static_assert(!DENSE || MODE == 2, "the dense case is the single-projection (down) mode");
    static_assert(!SPLIT || DENSE, "a K split is a dense scheduling device");
    using L = Layout<MODE>;
    extern __shared__ __align__(128) uint8_t smem[];
    uint16_t* tab = reinterpret_cast<uint16_t*>(smem + L::OFF_TABLES);
    uint8_t* Bs = smem + L::OFF_B;
    uint8_t* As = smem + L::OFF_A;
    int32_t* Ws = reinterpret_cast<int32_t*>(smem + L::OFF_W);
    float* wsc = reinterpret_cast<float*>(smem + L::OFF_WSCALE);
    int32_t* desc = reinterpret_cast<int32_t*>(smem + L::OFF_DESC);
    int32_t* claim = reinterpret_cast<int32_t*>(smem + L::OFF_CLAIM);
    const int w_stage = w_stage_ints(p.slot_words);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int nk = p.K / BK;
    const int dense_nsb = (p.rows_x + BM - 1) / BM;      // DENSE: superblocks of x
    const int total_items = DENSE ? dense_nsb * p.k_split * p.n_blocks
                                  : p.item_off[p.E] * p.n_blocks;
    unsigned gc = 0;          // global chunk counter: the stage is gc & 1
    unsigned item_idx = 0;    // the descriptor slot is item_idx & 1

    if (tid < PRODUCER_THREADS) {
        // ------------------------------------------------------------ producers
        const int m = tid >> 3;     // lane group: the chunk's m-th column in the block's
                                    // (low-rate, high-rate) order -- see col_map
        const int j = tid & 7;      // eight rows (8j..8j+7) within the half
        int last_e = -1;            // the expert whose table(s) shared memory holds
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
            // items read the same words.  DENSE: one expert, the superblocks
            // of x in order, and a K-split index innermost so the S splits of
            // one (n-block, superblock) run on neighbouring SMs.
            int e, nb, sb, pos0, mb, kc0, nkc, ks;
            if constexpr (DENSE) {
                const int per_nb = dense_nsb * p.k_split;
                nb = item / per_nb;
                const int rem = item - nb * per_nb;
                sb = rem / p.k_split;
                ks = rem - sb * p.k_split;
                e = 0;
                pos0 = sb * BM;
                mb = min(BM, p.rows_x - pos0);
                kc0 = (int)(((long)ks * nk) / p.k_split);
                nkc = (int)(((long)(ks + 1) * nk) / p.k_split) - kc0;
            } else {
                const int sbg = item / p.n_blocks;
                int lo = 0, hi = p.E;
                while (hi - lo > 1) {
                    const int mid = (lo + hi) >> 1;
                    if (p.item_off[mid] <= sbg) lo = mid; else hi = mid;
                }
                e = lo;
                const int nsb = p.item_off[e + 1] - p.item_off[e];
                const int local = item - p.item_off[e] * p.n_blocks;
                nb = local / nsb;
                sb = local - nb * nsb;
                const int start = p.offsets[e];
                const int end = p.offsets[e + 1];
                pos0 = start + sb * BM;
                mb = min(BM, end - pos0);
                kc0 = 0;
                nkc = nk;
                ks = 0;
            }
            if (tid == 0) {
                desc[slot * 8 + 0] = e;
                desc[slot * 8 + 1] = nb;
                desc[slot * 8 + 2] = sb;
                desc[slot * 8 + 3] = pos0;
                desc[slot * 8 + 4] = mb;
                desc[slot * 8 + 5] = kc0;
                desc[slot * 8 + 6] = nkc;
                desc[slot * 8 + 7] = ks;
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
            // The expert's table(s): 32 KB each, asynchronously -- unless shared
            // memory already holds this expert's, which it does for every item
            // after the first of one expert (always, in the dense case).  Safe
            // because every producer passed the barrier above after its last
            // lookup of the previous item, and consumers never read the table.
            if (e != last_e) {
                const uint16_t* t0 = p.table0 + (long)e * TABLE_ENTRIES;
                for (int i = tid; i < TABLE_ENTRIES / 8; i += PRODUCER_THREADS)
                    cp_async16(tab + i * 8, t0 + i * 8);
                if (MODE != 2) {
                    const uint16_t* t1 = p.table1 + (long)e * TABLE_ENTRIES;
                    for (int i = tid; i < TABLE_ENTRIES / 8; i += PRODUCER_THREADS)
                        cp_async16(tab + TABLE_ENTRIES + i * 8, t1 + i * 8);
                }
                last_e = e;
            }
            // Word geometry per half: half h decodes rows n_h0 .. n_h0 + 63 of
            // its projection (down: the two halves of one 128-row block; gate/up:
            // the same 64 intermediate rows of two projections).  A column's
            // words for the half start ``2 * rate * t64`` words into its chunk
            // of tile g (``t64`` = the half's 64-row index within the tile).
            const int32_t* tbase_h[2];
            int g_h[2], t64_h[2];
            const int32_t* init_h[2];
            int hasinit_h[2];
            RunPair rp_h[2];
            const int32_t* bdesc_h[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nh0 = (MODE == 2) ? n0 + HALF * h : n0;
                const int g = nh0 / TILE_ROWS;
                const int t = nh0 - g * TILE_ROWS;
                const bool second = (MODE != 2) && h == 1;
                g_h[h] = g;
                t64_h[h] = t / HALF;
                tbase_h[h] = (second ? p.words1 : p.words0) + (long)e * p.words_stride
                             + (long)g * p.tile_words;
                init_h[h] = (second ? p.init1 : p.init0) + (long)e * p.K;
                hasinit_h[h] = (second ? p.has_init1 : p.has_init0)[e];
                rp_h[h] = load_runs(second ? p.runs1 : p.runs0, e);
                bdesc_h[h] = (second ? p.bdesc1 : p.bdesc0) + (long)e * nk * BDESC_INTS;
            }
            // The run pair must tile K and the wire's tile_words exactly: the
            // Python owner checks it per stack; a mismatch here would address
            // outside the expert's words, so it traps rather than reads.
            if (tid == 0) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const RunPair& rp = rp_h[h];
                    if (rp.n_lo + rp.n_hi != p.K
                        || 16 * (rp.n_lo * rp.r_lo + rp.n_hi * rp.r_hi) != p.tile_words
                        || rp.r_lo < RATE_MIN || rp.r_lo > RATE_MAX
                        || slot_words_for_rate(rp.r_lo) > p.slot_words
                        || (rp.n_hi > 0 && (rp.r_hi <= rp.r_lo || rp.r_hi > RATE_MAX
                                            || rp.w_hi != 16 * rp.n_lo * rp.r_lo
                                            || slot_words_for_rate(rp.r_hi) > p.slot_words)))
                        __trap();
                }
            }
            // The A row this thread stages (-1: a zero row past the superblock).
            long arow = -1;
            {
                const int r = FP8 ? (tid >> 1) : (tid >> 2);
                if ((!FP8 || tid < 128) && r < mb) {
                    const int pos = pos0 + r;
                    if constexpr (DENSE) {
                        arow = pos;                       // row m of x is route m
                    } else {
                        const int flat = p.flat_sorted[pos];
                        arow = (p.a_row_mode == 1) ? pos : (p.a_row_mode == 0 ? flat / p.top_k : flat);
                    }
                }
            }
            // The half this thread issues words for (threads 0..127; fixed per
            // item) -- selected once, so the per-half tables stay in registers
            // instead of becoming a runtime-indexed local array.
            const int ih = (tid >> 6) & 1;
            const int32_t* tbase_i = ih ? tbase_h[1] : tbase_h[0];
            const int t64_i = ih ? t64_h[1] : t64_h[0];
            const int32_t* bdesc_i = ih ? bdesc_h[1] : bdesc_h[0];
            RunPair rp_i;
            rp_i.r_lo = ih ? rp_h[1].r_lo : rp_h[0].r_lo;
            rp_i.n_lo = ih ? rp_h[1].n_lo : rp_h[0].n_lo;
            rp_i.r_hi = ih ? rp_h[1].r_hi : rp_h[0].r_hi;
            rp_i.n_hi = ih ? rp_h[1].n_hi : rp_h[0].n_hi;
            rp_i.w_hi = ih ? rp_h[1].w_hi : rp_h[0].w_hi;
            // The words of chunk kc for half ih, lane group mm: 2 * rate words per
            // column, 16-byte copies at even rates, 8-byte at odd (alignment).
            auto issue_words = [&](int kc) {
                if (tid < 128) {
                    const int mm = (tid >> 1) & 31;
                    const int q = tid & 1;
                    const ColMap c = col_map(bdesc_i, rp_i, kc, mm);
                    const int32_t* src = tbase_i + c.cw0 + 2 * c.rate * t64_i;
                    int32_t* dst = Ws + (kc % WORD_STAGES) * w_stage + (ih * BK + mm) * p.slot_words;
                    if (c.rate & 1) {
                        for (int k = q; k < c.rate; k += 2) cp_async8(dst + 2 * k, src + 2 * k);
                    } else {
                        for (int k = q; k < c.rate / 2; k += 2) cp_async16(dst + 4 * k, src + 4 * k);
                    }
                }
            };
            // The 32 stream bits before the half's first word, for every row
            // group whose window starts inside that word (8 * j * rate < 32:
            // j = 0 at any rate, j <= 3 at rate 1, j = 1 at rates 2 and 3) --
            // a field's 14-bit window reaches up to 13 bits before it, and at
            // rate 1 the second group's does reach past the word (tessera#694:
            // loading it for j = 0 alone left rows 8..12 of every half after
            // the first reading a zero history at rate 1).
            auto load_prev = [&](int kc, int32_t (&pv)[2]) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const ColMap c = col_map(bdesc_h[h], rp_h[h], kc, m);
                    if (8 * j * c.rate >= 32) continue;
                    const int wr0 = 2 * c.rate * t64_h[h];
                    const int32_t* wcol = tbase_h[h] + c.cw0;
                    int32_t v;
                    if (wr0 > 0) v = wcol[wr0 - 1];
                    else if (g_h[h] > 0) v = wcol[16 * c.rate - 1 - p.tile_words];
                    else v = hasinit_h[h] ? init_h[h][c.p] : 0;
                    pv[h] = v;
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

            issue_words(kc0);
            cp_async_commit();                     // group 0: the tables and the first chunk's words
            if (nkc > 1) issue_words(kc0 + 1);
            cp_async_commit();                     // group 1
            int32_t prev_cur[2] = {0, 0}, prev_nxt[2] = {0, 0};
            uint4 a_cur = make_uint4(0, 0, 0, 0), a_nxt = make_uint4(0, 0, 0, 0);
            load_prev(kc0, prev_cur);
            load_a(kc0, a_cur);
            for (int ic = 0; ic < nkc; ++ic, ++gc) {
                const int kc = kc0 + ic;
                if (ic + 1 < nkc) { load_prev(kc + 1, prev_nxt); load_a(kc + 1, a_nxt); }
                cp_async_wait<1>();                // chunk kc's words (and the tables) have landed
                bar_sync(BAR_PROD, PRODUCER_THREADS);   // ... for every producer; chunk kc-1's stage is free
                if (ic + 2 < nkc) issue_words(kc + 2);
                cp_async_commit();
                const int stage = gc & 1;
                if (gc >= 2) bar_sync(BAR_EMPTY0 + stage, THREADS);
                store_a(stage, a_cur);
                const int32_t* W = Ws + (kc % WORD_STAGES) * w_stage;
                uint8_t* B = Bs + stage * B_STAGE_BYTES;
                ColMap c0;
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    // MODE 2 reads one projection: both halves map alike.
                    const ColMap c = (MODE == 2 && h == 1) ? c0 : col_map(bdesc_h[h], rp_h[h], kc, m);
                    if (h == 0) c0 = c;
                    const int32_t* Wc = W + (h * BK + m) * p.slot_words;
                    const uint16_t* T = tab + ((MODE == 2) ? 0 : h * TABLE_ENTRIES);
                    const int chunk = (MODE == 2) ? (8 * h + j) : (4 * (j >> 1) + 2 * h + (j & 1));
                    const float* ws = wsc + slot * BN + chunk * 8;
                    uint32_t packed[4];
                    decode_rows_at<FP8>(c.rate, Wc, (uint32_t)prev_cur[h], j, T, ws, packed);
                    *reinterpret_cast<uint4*>(B + c.cib * (BN * 2) + (bswz(chunk, c.cib) << 4)) =
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
            const int nkc = desc[slot * 8 + 6];
            const int ks = desc[slot * 8 + 7];
            float acc[2][4][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) acc[mi][nt][i] = 0.f;
            for (int ic = 0; ic < nkc; ++ic, ++gc) {
                stage = gc & 1;
                if (ic > 0) bar_sync(BAR_FULL0 + stage, THREADS);
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
                    const int flat = DENSE ? pos : p.flat_sorted[pos];
                    float a_s = 1.f, rw = 1.f;
                    if constexpr (FP8 && !SPLIT) {
                        const long arow = DENSE ? (long)pos
                            : ((p.a_row_mode == 1) ? pos : (p.a_row_mode == 0 ? flat / p.top_k : flat));
                        a_s = p.a_scale[arow];
                    }
                    if (MODE == 2 && !DENSE && p.mul_weight) rw = p.rw_sorted[pos];
                    if constexpr (SPLIT) {
                        // The raw fp32 accumulator of this K range; the reduce
                        // kernel forms the sum in split order and applies the
                        // epilogue once.
                        float* part = p.partial + ((long)ks * p.rows_x + pos) * p.N + n0;
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const int cb = 32 * nw + 8 * nt + 2 * (lane & 3);
                            *reinterpret_cast<float2*>(part + cb) =
                                make_float2(acc[mi][nt][2 * hr], acc[mi][nt][2 * hr + 1]);
                        }
                    } else if constexpr (MODE == 0) {
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
                                if (!DENSE && p.mul_weight) y = __fmul_rn(y, rw);
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

int max_dynamic_smem_bytes(int device) {
    int v = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return v;
}

template <bool FP8, int MODE, bool DENSE = false, bool SPLIT = false>
void launch(const Params& p, int grid, cudaStream_t stream) {
    const int smem = smem_bytes(MODE, p.slot_words);
    static int attributed = 0;     // the largest dynamic size this instantiation was granted
    if (smem > attributed) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(routed_fused_kernel<FP8, MODE, DENSE, SPLIT>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attributed = smem;
    }
    routed_fused_kernel<FP8, MODE, DENSE, SPLIT><<<grid, THREADS, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The slot and shared-memory checks both host entries make.
void check_slot(int mode, int64_t slot_words, const torch::Tensor& on) {
    TORCH_CHECK(slot_words % 4 == 0 && slot_words >= 4 && slot_words <= SLOT_WORDS_MAX,
                "slot_words must be a multiple of 4 in [4, ", SLOT_WORDS_MAX, "]");
    const int need = smem_bytes(mode, (int)slot_words);
    const int have = max_dynamic_smem_bytes(on.device().index());
    TORCH_CHECK(need <= have, "the ", (mode == 2 ? "down/dense" : "gate/up"), " launch at ", slot_words,
                "-word slots needs ", need, " bytes of dynamic shared memory per block; device ",
                (int)on.device().index(), " allows ", have);
}

// DENSE && SPLIT: out[m, n] = epilogue( sum_{s < S} partial[s, m, n] ), the sum
// in fixed split order in fp32, then the same operation order as the unsplit
// epilogue: ``(acc * a_scale[m]) * w_scale[n]`` for the E4M3 family, the bare
// accumulator for the folded value family, one bf16 rounding.
template <bool FP8>
__global__ void dense_reduce_kernel(const float* __restrict__ partial, const float* __restrict__ a_scale,
                                    const float* __restrict__ wscale, uint16_t* __restrict__ out,
                                    long out_stride, int S, long M, long N) {
    const long quad = (long)blockIdx.x * blockDim.x + threadIdx.x;   // four consecutive columns
    const long quads_per_row = N / 4;
    if (quad >= M * quads_per_row) return;
    const long m = quad / quads_per_row;
    const long n = (quad - m * quads_per_row) * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < S; ++s) {
        const float4 v = *reinterpret_cast<const float4*>(partial + ((long)s * M + m) * N + n);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float y[4] = {acc.x, acc.y, acc.z, acc.w};
    if constexpr (FP8) {
        const float a_s = a_scale[m];
        #pragma unroll
        for (int i = 0; i < 4; ++i) y[i] = __fmul_rn(__fmul_rn(y[i], a_s), wscale[n + i]);
    }
    uint2 o;
    o.x = bf16_bits_rn(y[0]) | ((uint32_t)bf16_bits_rn(y[1]) << 16);
    o.y = bf16_bits_rn(y[2]) | ((uint32_t)bf16_bits_rn(y[3]) << 16);
    *reinterpret_cast<uint2*>(out + m * out_stride + n) = o;
}

const int32_t* i32_ptr(const torch::Tensor& t) { return t.data_ptr<int32_t>(); }
const float* f32_ptr(const torch::Tensor& t) { return t.data_ptr<float>(); }
const uint16_t* u16_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const uint16_t*>(t.data_ptr<int16_t>());
}
// The run pair and the block descriptors of one projection: shapes and dtypes
// only; their contents are checked per expert by the kernel (``__trap`` on a
// pair that does not tile K into tile_words) and per stack by the Python owner.
void check_run_tables(const torch::Tensor& runs, const torch::Tensor& bdesc, int64_t E, int64_t K,
                      const char* runs_name, const char* bdesc_name) {
    TORCH_CHECK(runs.is_cuda() && runs.dim() == 2 && runs.size(0) == E && runs.size(1) == 8
                && runs.scalar_type() == torch::kInt32 && runs.is_contiguous(),
                runs_name, " must be int32 [E, 8]");
    TORCH_CHECK(bdesc.is_cuda() && bdesc.dim() == 3 && bdesc.size(0) == E && bdesc.size(1) == K / BK
                && bdesc.size(2) == BDESC_INTS && bdesc.scalar_type() == torch::kInt32 && bdesc.is_contiguous(),
                bdesc_name, " must be int32 [E, K / ", BK, ", ", BDESC_INTS, "]");
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
    torch::Tensor runs0, torch::Tensor runs1,
    torch::Tensor bdesc0, torch::Tensor bdesc1,
    int64_t tile_words, int64_t slot_words,
    torch::Tensor offsets, torch::Tensor flat_sorted, torch::Tensor rw_sorted,
    torch::Tensor item_off, torch::Tensor counter,
    int64_t top_k, int64_t a_row_mode, bool mul_weight, double limit,
    torch::Tensor out, int64_t grid) {
    TORCH_CHECK(mode >= 0 && mode <= 2, "mode must be 0, 1 or 2");
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous 2-D CUDA tensor");
    const int64_t K = x.size(1);
    TORCH_CHECK(K % BK == 0 && K >= 4 * BK, "K must be a multiple of ", BK, " and at least ", 4 * BK);
    check_run_tables(runs0, bdesc0, wscale0.size(0), K, "runs0", "bdesc0");
    if (mode != 2) check_run_tables(runs1, bdesc1, wscale0.size(0), K, "runs1", "bdesc1");
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
    TORCH_CHECK(tile_words >= K * 16 * RATE_MIN && tile_words <= K * 16 * RATE_MAX && tile_words % 16 == 0,
                "tile_words must be 16 * (sum of the column rates), rates ", RATE_MIN, "..", RATE_MAX);
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
    p.runs0 = i32_ptr(runs0);
    p.runs1 = two ? i32_ptr(runs1) : nullptr;
    p.bdesc0 = i32_ptr(bdesc0);
    p.bdesc1 = two ? i32_ptr(bdesc1) : nullptr;
    p.words_stride = words0.size(1);
    p.tile_words = (int)tile_words;
    check_slot((int)mode, slot_words, x);
    p.slot_words = (int)slot_words;
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
    p.rows_x = (int)x.size(0);
    p.k_split = 1;
    p.partial = nullptr;

    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int g = (int)grid;
    TORCH_CHECK(fp8 == FAMILY_FP8, "this library serves the ", FAMILY_FP8 ? "E4M3" : "value",
                " family only; the other family's library is a separate native extension");
    if (mode == 0) launch<FAMILY_FP8, 0>(p, g, stream);
    else if (mode == 1) launch<FAMILY_FP8, 1>(p, g, stream);
    else launch<FAMILY_FP8, 2>(p, g, stream);
}

// The dense launch: one role of a dense Linear (E = 1, route m = row m, no
// weight), ``out`` a ``[M, N]`` view whose rows may be strided (a column slice
// of the module's merged output).  ``k_split`` > 1 accumulates each split's K
// range into ``partial`` ([S, M, N] fp32) and reduces it in fixed order.  The
// caller zeroes ``counter`` in-stream before the call.
void dense_forward(
    bool fp8, torch::Tensor x, torch::Tensor a_scale,
    torch::Tensor words, torch::Tensor table, torch::Tensor init, torch::Tensor has_init,
    torch::Tensor wscale, torch::Tensor runs, torch::Tensor bdesc, int64_t tile_words,
    int64_t slot_words, torch::Tensor counter, int64_t k_split, torch::Tensor partial, torch::Tensor out, int64_t grid) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous 2-D CUDA tensor");
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    TORCH_CHECK(K % BK == 0 && K >= 4 * BK, "K must be a multiple of ", BK, " and at least ", 4 * BK);
    check_run_tables(runs, bdesc, 1, K, "runs", "bdesc");
    if (fp8) {
        TORCH_CHECK(x.scalar_type() == torch::kFloat8_e4m3fn, "the E4M3 family takes an e4m3 x");
        TORCH_CHECK(a_scale.is_cuda() && a_scale.scalar_type() == torch::kFloat32
                    && a_scale.is_contiguous() && a_scale.numel() == M, "a_scale must be fp32 [M]");
    } else {
        TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "the value family takes a bf16 x");
    }
    TORCH_CHECK(wscale.dim() == 2 && wscale.size(0) == 1 && wscale.scalar_type() == torch::kFloat32
                && wscale.is_contiguous(), "wscale must be fp32 [1, N]");
    const int64_t N = wscale.size(1);
    TORCH_CHECK(N % BN == 0 && N > 0, "the role's rows must be a multiple of ", BN);
    TORCH_CHECK(words.is_cuda() && words.dim() == 2 && words.size(0) == 1
                && words.scalar_type() == torch::kInt32 && words.is_contiguous(), "words must be int32 [1, W]");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == 1 && table.size(1) == TABLE_ENTRIES
                && table.scalar_type() == torch::kInt16 && table.is_contiguous(), "table must be int16 [1, 16384]");
    TORCH_CHECK(init.dim() == 2 && init.size(0) == 1 && init.size(1) == K
                && init.scalar_type() == torch::kInt32 && init.is_contiguous(), "init must be int32 [1, K]");
    TORCH_CHECK(has_init.numel() == 1 && has_init.scalar_type() == torch::kInt32, "has_init must be int32 [1]");
    TORCH_CHECK(tile_words >= K * 16 * RATE_MIN && tile_words <= K * 16 * RATE_MAX && tile_words % 16 == 0,
                "tile_words must be 16 * (sum of the column rates), rates ", RATE_MIN, "..", RATE_MAX);
    TORCH_CHECK(counter.scalar_type() == torch::kInt32 && counter.numel() >= 1, "counter must be int32");
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.scalar_type() == torch::kBFloat16
                && out.size(0) == M && out.size(1) == N && out.stride(1) == 1 && out.stride(0) >= N
                && (out.stride(0) % 2) == 0,
                "out must be a bf16 [M, N] view with unit column stride and an even row stride");
    const int nk = (int)(K / BK);
    TORCH_CHECK(k_split >= 1 && k_split <= nk, "k_split must be in [1, K / ", BK, "]");
    if (k_split > 1) {
        TORCH_CHECK(partial.is_cuda() && partial.scalar_type() == torch::kFloat32 && partial.is_contiguous()
                    && partial.numel() == k_split * M * N, "partial must be contiguous fp32 [S, M, N]");
        // the reduce writes four bf16 at a time (uint2) at column offsets that
        // are multiples of 4: the row stride must keep those 8-byte aligned
        TORCH_CHECK((out.stride(0) % 4) == 0,
                    "a split-K launch needs out's row stride to be a multiple of 4 elements");
    }
    TORCH_CHECK(grid >= 1, "grid must be positive");
    TORCH_CHECK(fp8 == FAMILY_FP8, "this library serves the ", FAMILY_FP8 ? "E4M3" : "value",
                " family only; the other family's library is a separate native extension");

    Params p{};
    p.x = x.data_ptr();
    p.a_scale = fp8 ? f32_ptr(a_scale) : nullptr;
    p.words0 = i32_ptr(words);
    p.words1 = nullptr;
    p.table0 = u16_ptr(table);
    p.table1 = nullptr;
    p.init0 = i32_ptr(init);
    p.init1 = nullptr;
    p.has_init0 = i32_ptr(has_init);
    p.has_init1 = nullptr;
    p.wscale0 = f32_ptr(wscale);
    p.wscale1 = nullptr;
    p.runs0 = i32_ptr(runs);
    p.runs1 = nullptr;
    p.bdesc0 = i32_ptr(bdesc);
    p.bdesc1 = nullptr;
    p.words_stride = words.size(1);
    p.tile_words = (int)tile_words;
    check_slot(2, slot_words, x);
    p.slot_words = (int)slot_words;
    p.K = (int)K;
    p.N = (int)N;
    p.E = 1;
    p.offsets = nullptr;
    p.flat_sorted = nullptr;
    p.rw_sorted = nullptr;
    p.item_off = nullptr;
    p.counter = counter.data_ptr<int32_t>();
    p.n_blocks = (int)(N / BN);
    p.top_k = 1;
    p.a_row_mode = 2;
    p.mul_weight = 0;
    p.limit = std::numeric_limits<float>::infinity();
    p.out = out.data_ptr();
    p.out_stride = out.stride(0);
    p.inter = (int)N;
    p.rows_x = (int)M;
    p.k_split = (int)k_split;
    p.partial = k_split > 1 ? partial.data_ptr<float>() : nullptr;

    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int g = (int)grid;
    if (k_split > 1) {
        launch<FAMILY_FP8, 2, true, true>(p, g, stream);
        const long quads = M * (N / 4);
        const int threads = 256;
        const long blocks = (quads + threads - 1) / threads;
        dense_reduce_kernel<FAMILY_FP8><<<(unsigned)blocks, threads, 0, stream>>>(
            p.partial, p.a_scale, p.wscale0, reinterpret_cast<uint16_t*>(p.out), p.out_stride,
            (int)k_split, M, N);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        launch<FAMILY_FP8, 2, true, false>(p, g, stream);
    }
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
    m.def("dense_forward", &dense_forward);
    m.def("token_sum", &token_sum);
    m.attr("BM") = BM;
    m.attr("BN") = BN;
    m.attr("HALF") = HALF;
    m.attr("BK") = BK;
    m.attr("RATE_MIN") = RATE_MIN;
    m.attr("RATE_MAX") = RATE_MAX;
    m.attr("SLOT_WORDS_MAX") = SLOT_WORDS_MAX;
    m.attr("WORD_STAGES") = WORD_STAGES;
    m.attr("SMEM_FIXED_GATE_UP") = Layout<0>::OFF_W;
    m.attr("SMEM_FIXED_DOWN") = Layout<2>::OFF_W;
    m.attr("BDESC_INTS") = BDESC_INTS;
    m.attr("WINDOW_BITS") = WINDOW_BITS;
    m.def("smem_bytes", [](int64_t mode, int64_t slot_words) { return (int64_t)smem_bytes((int)mode, (int)slot_words); },
          "dynamic shared memory the launch of ``mode`` needs at ``slot_words``-word slots");
    m.def("max_dynamic_smem_bytes", [](int64_t device) { return (int64_t)max_dynamic_smem_bytes((int)device); },
          "cudaDevAttrMaxSharedMemoryPerBlockOptin of the device");
    m.attr("THREADS") = THREADS;
    m.attr("FAMILY_FP8") = FAMILY_FP8;
}
