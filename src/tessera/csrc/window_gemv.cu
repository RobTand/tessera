// Tessera window-body GEMV: the ~4 bits/weight wire read directly at M<=8.
//
// Layout (built once at load by ``kernel_window_gemv.repack_window_body``, a
// bijection of the BODY plane's bits -- see the Python docstring):
//
//   * rows are padded to a multiple of TILE_ROWS (512) with zero codes; a zero
//     code appended after the last row never changes an earlier state, so the
//     padded stream decodes identically on the rows that exist;
//   * columns are stably sorted by rate (``perm``), so every rate is one
//     contiguous run; the kernel reads ``x`` through ``perm``;
//   * tile ``g`` holds, for each permuted column ``c`` in order, the 512*R_c
//     bits of rows [512g, 512g+512) of column ``c``, MSB-first in stream order,
//     packed into little-endian u32 words with the stream's first bit at bit
//     31 of the first word (the wire's bytes, reversed within each 4-byte
//     word).  A column's chunk is 16*R_c words; a tile is ``tile_words``;
//   * the L-bit pad that opens every wire column is not stored: it is all
//     zeros by definition (state_{-1} = 0) and the kernel supplies it.
//
// One warp reads one column's tile chunk per iteration: lane ``l`` owns rows
// [l*RPL, l*RPL+RPL) of the 512-row tile (RPL=16: 8 bytes at R=4, one uint2;
// RPL=8: 4 bytes) -- a contiguous, coalesced 64*R-byte read per warp.  The
// L-R bits of history a lane's first window needs come from the previous
// lane by shuffle, and for lane 0 from the word before the chunk (previous
// tile's last word, or the zero pad on tile 0).  Every state is then a
// funnel shift with an immediate, one mask, one shared-memory table lookup
// and MT fused multiply-adds.  Warps in a block split an item's columns and
// share its 512 rows, reduce through shared memory, then one coalesced
// atomicAdd per (m, row) into the fp32 output.  The table (2^L bf16 or fp32
// values) is staged in shared memory once per block; blocks loop over items
// so the staging is paid ~once per SM, not once per tile.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int TILE_ROWS = 512;
constexpr int RED_STRIDE = 33;   // padded [16][33] reduction rows: conflict-free lane writes
constexpr int MAX_ITEM_COLS = 256;

struct __align__(16) Item {
    int tile;    // 512-row tile index
    int rate;    // bits per code for every column of the item
    int col0;    // first permuted column
    int ncols;   // columns in the item (<= MAX_ITEM_COLS)
    int word0;   // word offset of col0's chunk inside its tile
    int pad[3];
};

struct Params {
    const uint32_t* __restrict__ words;
    const Item* __restrict__ items;
    int n_items;
    long tile_words;
    const int* __restrict__ perm;        // permuted column -> original column
    const __nv_bfloat16* __restrict__ x; // [M, K] bf16, original column order
    int K;
    int M;
    int rows;
    const float* __restrict__ scale;     // [rows] fp32 (ones for the value family)
    float* __restrict__ out;             // [M, rows] fp32, pre-zeroed
    const void* __restrict__ table;      // [2^L] bf16 or fp32 values
};

template <typename TBL> __device__ __forceinline__ float table_value(const TBL* tbl, uint32_t state);
template <> __device__ __forceinline__ float table_value<uint16_t>(const uint16_t* tbl, uint32_t state) {
    return __uint_as_float(((uint32_t)tbl[state]) << 16);   // bf16 -> fp32 is a shift
}
template <> __device__ __forceinline__ float table_value<float>(const float* tbl, uint32_t state) {
    return tbl[state];
}

// Which reduction slot row ``r`` of the tile lands in for this RPL.
template <int RPL> __device__ __forceinline__ int red_index(int r) {
    // RPL=16: r = lane*16 + t            -> slot t*33 + lane
    // RPL=8 : r = half*256 + lane*8 + t  -> slot (half*8 + t)*33 + lane
    if (RPL == 16) return (r & 15) * RED_STRIDE + (r >> 4);
    return (((r >> 8) << 3) + (r & 7)) * RED_STRIDE + ((r & 255) >> 3);
}

// A hash standing in for a wire word under ABL_LOAD (keeps the arithmetic, drops the read).
__device__ __forceinline__ uint32_t fake_word(uint32_t a, uint32_t b) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u);
    h ^= h >> 15; h *= 0x85EBCA6Bu; h ^= h >> 13;
    return h;
}

// Process one column chunk for this lane: NB = RPL*R chunk bits, held as
// (hi, lo) words (NB=64), one word (NB=32) or a 16-bit value (NB=16), with
// ``prev`` the bits immediately before the chunk.  Accumulates into acc.
template <int L, int RPL, int R, int MT, typename TBL, bool ABL_GATHER, bool ABL_FMA>
__device__ __forceinline__ void consume_chunk(
    uint32_t hi, uint32_t lo, uint32_t prev, const TBL* __restrict__ tbl,
    const float* __restrict__ xv, float (&acc)[RPL][MT], uint32_t& junk)
{
    constexpr int NB = RPL * R;
    constexpr uint32_t MASK = (1u << L) - 1u;
    static_assert(NB == 16 || NB == 32 || NB == 64, "chunk width");
    static_assert(L <= 32, "window wider than a word");
    static_assert(NB != 16 || L <= 16 + R, "a 16-bit chunk needs L <= 16 + R");
#pragma unroll
    for (int t = 0; t < RPL; ++t) {
        uint32_t state;
        if (NB == 64) {
            // codes 0..RPL/2-1 end inside ``hi``; the rest inside ``lo``
            constexpr int half = RPL / 2;
            if (t < half) state = __funnelshift_r(hi, prev, R * (half - 1 - t)) & MASK;
            else          state = __funnelshift_r(lo, hi, R * (RPL - 1 - t)) & MASK;
        } else if (NB == 32) {
            state = __funnelshift_r(lo, prev, R * (RPL - 1 - t)) & MASK;
        } else {
            uint32_t v = (prev << 16) | lo;
            state = (v >> (R * (RPL - 1 - t))) & MASK;
        }
        float v;
        if (ABL_GATHER) v = __uint_as_float(0x3f000000u | state);   // no table read, a normal float
        else v = table_value<TBL>(tbl, state);
        if (ABL_FMA) { junk ^= __float_as_uint(v); }
        else {
#pragma unroll
            for (int m = 0; m < MT; ++m) acc[t][m] = fmaf(v, xv[m], acc[t][m]);
        }
    }
}

template <int L, int RPL, int R, int MT, typename TBL, bool ABL_GATHER, bool ABL_LOAD, bool ABL_FMA>
__device__ __forceinline__ void run_item(
    const Params& p, const Item& it, const TBL* __restrict__ tbl,
    float* __restrict__ red, const float* __restrict__ xs,
    int warp, int nwarps, int lane)
{
    constexpr int NB = RPL * R;
    constexpr int CHUNK_WORDS = 16 * R;                 // words per column chunk (512 rows)
    // RPL=16 -> a warp covers the tile; RPL=8 -> two warps split it, interleaved by column.
    const int half  = (RPL == 8) ? (warp & 1) : 0;
    const int wcol  = (RPL == 8) ? (warp >> 1) : warp;
    const int wstep = (RPL == 8) ? (nwarps >> 1) : nwarps;
    const long tile_base = (long)it.tile * p.tile_words + it.word0;
    const uint32_t* __restrict__ words = p.words;

    float acc[RPL][MT];
#pragma unroll
    for (int t = 0; t < RPL; ++t)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[t][m] = 0.f;
    uint32_t junk = 0;

    // Lane's chunk position inside the column chunk (in words / halfwords).
    // NB=64: words [2*lane, 2*lane+2); NB=32: word half*(CHUNK_WORDS/2) + lane;
    // NB=16: halfword lane inside word half*(CHUNK_WORDS/2) + lane/2 (high half first).
    const int lane_word = (NB == 64) ? 2 * lane
                        : (NB == 32) ? half * (CHUNK_WORDS / 2) + lane
                        : half * (CHUNK_WORDS / 2) + (lane >> 1);
    // The word holding the bits before lane 0's chunk: inside the chunk for
    // half=1, the previous tile's last word otherwise, or the zero pad on tile 0.
    long prev_word = -1;
    if (lane == 0) {
        if (lane_word > 0) prev_word = tile_base + lane_word - 1;
        else if (it.tile > 0) prev_word = tile_base - p.tile_words + CHUNK_WORDS - 1;
    }

    auto load = [&](int jj, uint32_t& hi, uint32_t& lo, uint32_t& pv) {
        const long base = tile_base + (long)jj * CHUNK_WORDS;
        if (ABL_LOAD) {
            hi = fake_word(jj, lane * 2); lo = fake_word(jj, lane * 2 + 1); pv = fake_word(jj, 77 + lane);
            if (NB == 16) { lo &= 0xffffu; pv &= 0xffffu; }
            return;
        }
        if (NB == 64) {
            uint2 v = __ldg(reinterpret_cast<const uint2*>(words + base + lane_word));
            hi = v.x; lo = v.y;
        } else {
            uint32_t w = __ldg(words + base + lane_word);
            if (NB == 32) { hi = 0; lo = w; }
            else { hi = 0; lo = (lane & 1) ? (w & 0xffffu) : (w >> 16); }
        }
        uint32_t p0 = 0;
        if (lane == 0 && prev_word >= 0) {
            uint32_t w = __ldg(words + prev_word + (long)jj * CHUNK_WORDS);
            // NB=16: lane 0's chunk is the high half of its word, so its
            // history is the low half of the word before.
            p0 = (NB == 16) ? (w & 0xffffu) : w;
        }
        // Other lanes take the previous lane's last word (or halfword).
        uint32_t up = __shfl_up_sync(0xffffffffu, lo, 1);
        pv = (lane == 0) ? p0 : up;
    };

    int jj = wcol;
    uint32_t hi = 0, lo = 0, pv = 0;
    if (jj < it.ncols) load(jj, hi, lo, pv);
    for (; jj < it.ncols; jj += wstep) {
        uint32_t nhi = 0, nlo = 0, npv = 0;
        const int jn = jj + wstep;
        if (jn < it.ncols) load(jn, nhi, nlo, npv);
        float xv[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) xv[m] = xs[jj * MT + m];
        consume_chunk<L, RPL, R, MT, TBL, ABL_GATHER, ABL_FMA>(hi, lo, pv, tbl, xv, acc, junk);
        hi = nhi; lo = nlo; pv = npv;
    }
    if (ABL_FMA) acc[0][0] = __uint_as_float(junk & 0x007fffffu);  // keep the gathers alive

    // Reduce across the block's warps through shared memory (conflict-free slots).
#pragma unroll
    for (int t = 0; t < RPL; ++t) {
        const int slot = ((RPL == 16) ? t : (half * 8 + t)) * RED_STRIDE + lane;
#pragma unroll
        for (int m = 0; m < MT; ++m) atomicAdd(red + m * (16 * RED_STRIDE) + slot, acc[t][m]);
    }
}

template <int L, int RPL, int MT, typename TBL, bool ABL_GATHER, bool ABL_LOAD, bool ABL_FMA>
__global__ void __launch_bounds__(512, 2) window_gemv_kernel(Params p)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    TBL* tbl = reinterpret_cast<TBL*>(smem_raw);
    constexpr int TABLE_BYTES = (1 << L) * (int)sizeof(TBL);
    float* red = reinterpret_cast<float*>(smem_raw + TABLE_BYTES);           // [MT][16*33]
    float* xs = red + MT * 16 * RED_STRIDE;                                  // [MAX_ITEM_COLS][MT]

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31, nwarps = nthreads >> 5;

    // Stage the table once per block (16-byte copies).
    {
        const uint4* src = reinterpret_cast<const uint4*>(p.table);
        uint4* dst = reinterpret_cast<uint4*>(smem_raw);
        for (int i = tid; i < TABLE_BYTES / 16; i += nthreads) dst[i] = __ldg(src + i);
    }

    for (int item_idx = blockIdx.x; item_idx < p.n_items; item_idx += gridDim.x) {
        const Item it = p.items[item_idx];
        __syncthreads();   // previous item's reduction fully drained
        for (int i = tid; i < MT * 16 * RED_STRIDE; i += nthreads) red[i] = 0.f;
        for (int i = tid; i < it.ncols * MT; i += nthreads) {
            const int jj = i / MT, m = i - jj * MT;
            const int col = p.perm[it.col0 + jj];
            xs[i] = __bfloat162float(p.x[(long)m * p.K + col]);
        }
        __syncthreads();
        switch (it.rate) {
            case 4: run_item<L, RPL, 4, MT, TBL, ABL_GATHER, ABL_LOAD, ABL_FMA>(p, it, tbl, red, xs, warp, nwarps, lane); break;
            case 2: run_item<L, RPL, 2, MT, TBL, ABL_GATHER, ABL_LOAD, ABL_FMA>(p, it, tbl, red, xs, warp, nwarps, lane); break;
            default:
                if (RPL == 16) run_item<L, 16, 1, MT, TBL, ABL_GATHER, ABL_LOAD, ABL_FMA>(p, it, tbl, red, xs, warp, nwarps, lane);
                break;  // RPL=8 at R=1 is refused on the host
        }
        __syncthreads();
        // One coalesced atomicAdd per (m, row) with the row scale folded in.
        const int row0 = it.tile * TILE_ROWS;
        for (int i = tid; i < MT * TILE_ROWS; i += nthreads) {
            const int m = i / TILE_ROWS, r = i - m * TILE_ROWS;
            const int row = row0 + r;
            if (row < p.rows) {
                const float v = red[m * (16 * RED_STRIDE) + red_index<RPL>(r)] * p.scale[row];
                atomicAdd(p.out + (long)m * p.rows + row, v);
            }
        }
    }
}

// Decode: every (row, col) -> the L-bit state -> ``of_state[state]``: the
// table's grid code (u8, the E4M3 family) or its raw bf16 value (u16, the
// value family -- the row scale is NOT folded in: it goes to the GEMM epilogue).
template <int L, int R, typename T>
__global__ void window_decode_kernel(
    const uint32_t* __restrict__ words, long tile_words, int word0, int col0, int ncols,
    const int* __restrict__ perm, const T* __restrict__ of_state,
    int rows, int cols, T* __restrict__ out)
{
    constexpr int RPL = 16, CHUNK_WORDS = 16 * R, NB = RPL * R;
    constexpr uint32_t MASK = (1u << L) - 1u;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const int tile = blockIdx.y;
    const int g_warp = blockIdx.x * nwarps + warp;   // one warp per column of this tile
    if (g_warp >= ncols) return;
    const int jj = g_warp;
    const long base = (long)tile * tile_words + word0 + (long)jj * CHUNK_WORDS;
    const int lane_word = (NB == 64) ? 2 * lane : (NB == 32) ? lane : (lane >> 1);
    uint32_t hi = 0, lo = 0;
    if (NB == 64) { uint2 v = *reinterpret_cast<const uint2*>(words + base + lane_word); hi = v.x; lo = v.y; }
    else { uint32_t w = words[base + lane_word]; lo = (NB == 32) ? w : ((lane & 1) ? (w & 0xffffu) : (w >> 16)); }
    uint32_t p0 = 0;
    if (lane == 0) {
        if (lane_word > 0) p0 = words[base + lane_word - 1];
        else if (tile > 0) p0 = words[base - tile_words + CHUNK_WORDS - 1];
        if (NB == 16) p0 &= 0xffffu;
    }
    uint32_t up = __shfl_up_sync(0xffffffffu, lo, 1);
    const uint32_t prev = (lane == 0) ? p0 : up;
    const int col = perm[col0 + jj];
#pragma unroll
    for (int t = 0; t < RPL; ++t) {
        uint32_t state;
        if (NB == 64) {
            constexpr int half = RPL / 2;
            if (t < half) state = __funnelshift_r(hi, prev, R * (half - 1 - t)) & MASK;
            else          state = __funnelshift_r(lo, hi, R * (RPL - 1 - t)) & MASK;
        } else if (NB == 32) {
            state = __funnelshift_r(lo, prev, R * (RPL - 1 - t)) & MASK;
        } else {
            state = (((prev << 16) | lo) >> (R * (RPL - 1 - t))) & MASK;
        }
        const int row = tile * TILE_ROWS + lane * RPL + t;
        if (row < rows) out[(long)row * cols + col] = of_state[state];
    }
}

template <typename T>
void decode_typed(const uint32_t* w, long tile_words, int n_tiles, torch::Tensor runs_cpu, torch::Tensor perm,
                  const T* of_state, int rows, int cols, T* out, cudaStream_t stream)
{
    for (int r = 0; r < runs_cpu.size(0); ++r) {
        const int rate = runs_cpu[r][0].item<int>(), col0 = runs_cpu[r][1].item<int>();
        const int ncols = runs_cpu[r][2].item<int>(), word0 = runs_cpu[r][3].item<int>();
        const int warps = 8;
        dim3 grid((ncols + warps - 1) / warps, n_tiles);
        switch (rate) {
            case 4: window_decode_kernel<14, 4, T><<<grid, warps * 32, 0, stream>>>(w, tile_words, word0, col0, ncols, perm.data_ptr<int>(), of_state, rows, cols, out); break;
            case 2: window_decode_kernel<14, 2, T><<<grid, warps * 32, 0, stream>>>(w, tile_words, word0, col0, ncols, perm.data_ptr<int>(), of_state, rows, cols, out); break;
            case 1: window_decode_kernel<14, 1, T><<<grid, warps * 32, 0, stream>>>(w, tile_words, word0, col0, ncols, perm.data_ptr<int>(), of_state, rows, cols, out); break;
            default: TORCH_CHECK(false, "rate ", rate, " has no kernel");
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <int L, int RPL, int MT, typename TBL, bool AG, bool AL, bool AF>
void launch_typed(const Params& p, int blocks, int threads, size_t smem, cudaStream_t stream) {
    auto k = window_gemv_kernel<L, RPL, MT, TBL, AG, AL, AF>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    k<<<blocks, threads, smem, stream>>>(p);
}

template <int L, int RPL, int MT, typename TBL>
void launch_abl(const Params& p, int blocks, int threads, size_t smem, cudaStream_t stream, int ablation) {
    switch (ablation) {
        case 0: launch_typed<L, RPL, MT, TBL, false, false, false>(p, blocks, threads, smem, stream); break;
        case 1: launch_typed<L, RPL, MT, TBL, true, false, false>(p, blocks, threads, smem, stream); break;   // no gather
        case 2: launch_typed<L, RPL, MT, TBL, false, true, false>(p, blocks, threads, smem, stream); break;   // no wire read
        case 3: launch_typed<L, RPL, MT, TBL, false, false, true>(p, blocks, threads, smem, stream); break;   // no FMA
        case 4: launch_typed<L, RPL, MT, TBL, true, true, false>(p, blocks, threads, smem, stream); break;    // neither read
        default: TORCH_CHECK(false, "unknown ablation ", ablation);
    }
}

template <int L, typename TBL>
void launch_mt(const Params& p, int rpl, int mt, int blocks, int threads, size_t smem, cudaStream_t stream, int ablation) {
    if (rpl == 16) {
        switch (mt) {
            case 1: launch_abl<L, 16, 1, TBL>(p, blocks, threads, smem, stream, ablation); return;
            case 2: launch_abl<L, 16, 2, TBL>(p, blocks, threads, smem, stream, ablation); return;
        }
    } else if (rpl == 8) {
        switch (mt) {
            case 1: launch_abl<L, 8, 1, TBL>(p, blocks, threads, smem, stream, ablation); return;
            case 2: launch_abl<L, 8, 2, TBL>(p, blocks, threads, smem, stream, ablation); return;
            case 4: launch_abl<L, 8, 4, TBL>(p, blocks, threads, smem, stream, ablation); return;
            case 8: launch_abl<L, 8, 8, TBL>(p, blocks, threads, smem, stream, ablation); return;
        }
    }
    TORCH_CHECK(false, "no kernel for rpl=", rpl, " mt=", mt);
}

}  // namespace

// out [M, rows] fp32 must be zeroed by the caller.
void window_gemv(
    torch::Tensor words, torch::Tensor items, long tile_words, torch::Tensor perm,
    torch::Tensor table, torch::Tensor scale, torch::Tensor x, torch::Tensor out,
    int window_bits, int rpl, int warps, int blocks, int ablation)
{
    TORCH_CHECK(words.is_cuda() && words.dtype() == torch::kInt32 && words.is_contiguous());
    TORCH_CHECK(items.is_cuda() && items.dtype() == torch::kInt32 && items.is_contiguous() && items.size(1) == 8);
    TORCH_CHECK(perm.is_cuda() && perm.dtype() == torch::kInt32 && perm.is_contiguous());
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous() && x.dim() == 2);
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat32 && out.is_contiguous() && out.dim() == 2);
    TORCH_CHECK(scale.is_cuda() && scale.dtype() == torch::kFloat32 && scale.is_contiguous());
    TORCH_CHECK(table.is_cuda() && table.is_contiguous() && table.numel() == (1L << window_bits));
    TORCH_CHECK(window_bits == 14, "this build instantiates L=14 only");
    const int M = (int)x.size(0), K = (int)x.size(1), rows = (int)out.size(1);
    TORCH_CHECK(out.size(0) == M && perm.numel() == K && scale.numel() == rows);
    TORCH_CHECK(M >= 1 && M <= 8);
    const int mt = (M <= 1) ? 1 : (M <= 2) ? 2 : (M <= 4) ? 4 : 8;
    TORCH_CHECK(mt == M, "M must be 1, 2, 4 or 8 here (the host pads)");
    TORCH_CHECK(warps >= 1 && warps <= 16 && (rpl != 8 || (warps % 2) == 0));

    Params p;
    p.words = reinterpret_cast<const uint32_t*>(words.data_ptr<int>());
    p.items = reinterpret_cast<const Item*>(items.data_ptr<int>());
    p.n_items = (int)items.size(0);
    p.tile_words = tile_words;
    p.perm = perm.data_ptr<int>();
    p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    p.K = K; p.M = M; p.rows = rows;
    p.scale = scale.data_ptr<float>();
    p.out = out.data_ptr<float>();
    p.table = table.data_ptr();

    const int threads = warps * 32;
    const size_t tbl_bytes = (size_t)table.numel() * table.element_size();
    const size_t smem = tbl_bytes + (size_t)mt * 16 * RED_STRIDE * 4 + (size_t)MAX_ITEM_COLS * mt * 4;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int grid = std::min(blocks, p.n_items);
    if (table.dtype() == torch::kBFloat16) launch_mt<14, uint16_t>(p, rpl, mt, grid, threads, smem, stream, ablation);
    else if (table.dtype() == torch::kFloat32) launch_mt<14, float>(p, rpl, mt, grid, threads, smem, stream, ablation);
    else TORCH_CHECK(false, "table must be bf16 or fp32");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Decode [rows, cols] from the repacked words through ``of_state``: u8 grid
// codes (E4M3 family) or u16 raw bf16 values (value family, scale separate).
// ``runs`` rows are (rate, col0, ncols, word0) in permuted-column order.
void window_decode(
    torch::Tensor words, long tile_words, int n_tiles, torch::Tensor runs, torch::Tensor perm,
    torch::Tensor of_state, int window_bits, torch::Tensor out)
{
    TORCH_CHECK(words.is_cuda() && words.dtype() == torch::kInt32);
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2);
    TORCH_CHECK(of_state.is_cuda() && of_state.is_contiguous() && of_state.numel() == (1L << window_bits));
    TORCH_CHECK(window_bits == 14, "this build instantiates L=14 only");
    TORCH_CHECK(runs.dtype() == torch::kInt32 && runs.dim() == 2 && runs.size(1) == 4);
    auto runs_cpu = runs.to(torch::kCPU).contiguous();
    const int rows = (int)out.size(0), cols = (int)out.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* w = reinterpret_cast<const uint32_t*>(words.data_ptr<int>());
    if (of_state.dtype() == torch::kUInt8) {
        TORCH_CHECK(out.dtype() == torch::kUInt8);
        decode_typed<uint8_t>(w, tile_words, n_tiles, runs_cpu, perm, of_state.data_ptr<uint8_t>(), rows, cols, out.data_ptr<uint8_t>(), stream);
    } else if (of_state.dtype() == torch::kBFloat16) {
        TORCH_CHECK(out.dtype() == torch::kBFloat16);
        decode_typed<uint16_t>(w, tile_words, n_tiles, runs_cpu, perm, reinterpret_cast<const uint16_t*>(of_state.data_ptr()), rows, cols, reinterpret_cast<uint16_t*>(out.data_ptr()), stream);
    } else {
        TORCH_CHECK(false, "of_state must be u8 codes or bf16 values");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("window_gemv", &window_gemv, "Tessera window-body GEMV (fp32 out, atomics)");
    m.def("window_decode", &window_decode, "Tessera window-body decode to grid codes (u8) or raw bf16 values");
}
