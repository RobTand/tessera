// Oracle kernels for the E2M1x2 window decode (``e2m1_window_decode.cuh``).
//
// ``decode``: the wire's tile words, the run pair, the fused lane's block
// descriptors, the window table and the LUT16 plane -> the B operand of the
// FP4 MMA as a linear ``[rows, K/2]`` packed-E2M1 tensor (row n, byte j:
// k = 2j low nibble, 2j + 1 high) and its UE4M3 scales ``[rows, K/16]``.
// One block per (64-k chunk, 64-tuple half): the half's words are staged per
// column into shared memory in ORIGINAL column order through the fused lane's
// column map (``col_map`` below is the fused kernel's, restated), exactly the
// addressing the lane uses, and every thread decodes two tuples at 2 x 8
// consecutive k.
//
// ``mma``: the decoded tile through the FP4 MMA from a shared-memory B tile
// in the layout the lane's consumer will read (``[2 k-halves][n][16 bytes]``,
// ldmatrix.x4, no swizzle needed at a 16-byte row stride), the scales in the
// SFB register (lane (g, t = 0), byte b = column g, k16 group b), and a
// one-hot A (row m hot at k = kb + m, E2M1 1.0 = code 0x2, UE4M3 1.0 = 0x38):
// D[m][n] = value(code(n, kb + m)) * scale(n, (kb + m) / 16), which is exact
// in fp32, for every (n, k).  A permutation of k between the decode and the
// MMA, or a scale byte on the wrong 16-group, shows up here as a mismatch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

#include "e2m1_window_decode.cuh"

namespace {

constexpr int BK = 32;            // the fused lane's block-descriptor width
constexpr int CHUNK = 64;         // k per FP4 MMA step
constexpr int HALF_T = 64;        // tuples per half
constexpr int TILE_T = 512;       // tuples per tile
constexpr int BDESC_INTS = 12;
constexpr int SLOT = 2 * 8 + 4;   // words per staged half: 2R at R <= 8, plus slack past the half

struct ColMap { int rate; bool lo; int cib; int p; int cw0; };

// The fused kernel's ``col_map`` (routed_fused_window.cu), restated.
template <int RL, bool TWO>
__device__ __forceinline__ ColMap col_map(const int32_t* blk, int n_lo, int w_hi, int kc, int m) {
    ColMap c;
    if constexpr (!TWO) {
        c.rate = RL; c.lo = true; c.cib = m; c.p = kc * BK + m; c.cw0 = c.p * 16 * RL;
    } else {
        const int cib = (blk[m >> 2] >> (8 * (m & 3))) & 0xFF;
        const int n_before = blk[8], cnt_lo = blk[9];
        c.lo = m < cnt_lo;
        const int rank = c.lo ? n_before + m : (kc * BK - n_before) + (m - cnt_lo);
        c.rate = c.lo ? RL : RL + 1;
        c.cib = cib;
        c.p = c.lo ? rank : n_lo + rank;
        c.cw0 = c.lo ? rank * 16 * RL : w_hi + rank * 16 * (RL + 1);
    }
    return c;
}

template <int L, int RL, bool TWO>
__global__ void __launch_bounds__(128) decode_kernel(
        const int32_t* __restrict__ words, int tile_words, const int32_t* __restrict__ pair,
        const int32_t* __restrict__ bdesc, const int32_t* __restrict__ init, int has_init,
        const uint8_t* __restrict__ table, const uint8_t* __restrict__ plane, const uint8_t* __restrict__ lut16,
        int rows, int K, uint8_t* __restrict__ B, uint8_t* __restrict__ SF) {
    __shared__ int32_t Ws[CHUNK][SLOT];
    __shared__ uint32_t prev_s[CHUNK];
    __shared__ int rate_s[CHUNK];
    const int kc = blockIdx.x;                     // 64-k chunk
    const int hh = blockIdx.y;                     // global half index
    const int g = hh / (TILE_T / HALF_T), t64 = hh % (TILE_T / HALF_T);
    const int n_lo = pair[2], w_hi = pair[7];
    const int tid = threadIdx.x;
    // Stage: thread c < 64 maps column c of the chunk's two 32-column blocks
    // and copies its half into the slot of its ORIGINAL position.
    if (tid < CHUNK) {
        const int b = 2 * kc + (tid >> 5), m = tid & 31;
        const ColMap c = col_map<RL, TWO>(bdesc + (long)b * BDESC_INTS, n_lo, w_hi, b, m);
        const int slot = (tid & 32) + c.cib;
        const int32_t* wcol = words + (long)g * tile_words + c.cw0;
        const int wr0 = 2 * c.rate * t64;
        for (int w = 0; w < 2 * c.rate; ++w) Ws[slot][w] = wcol[wr0 + w];
        for (int w = 2 * c.rate; w < SLOT; ++w) Ws[slot][w] = 0x5A5A5A5A;   // slack: must never matter
        uint32_t pv;
        if (wr0 > 0) pv = (uint32_t)wcol[wr0 - 1];
        else if (g > 0) pv = (uint32_t)wcol[16 * c.rate - 1 - tile_words];
        else pv = has_init ? (uint32_t)init[c.p] : 0u;
        prev_s[slot] = pv;
        rate_s[slot] = c.rate;
    }
    __syncthreads();
    uint32_t lut[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        lut[i] = e2m1w::pack4(lut16[4 * i], lut16[4 * i + 1], lut16[4 * i + 2], lut16[4 * i + 3]);
    // Decode: thread (tp, t) takes tuples 2tp, 2tp + 1 (rows 4tp..4tp + 3 of
    // the half) at k = 32h + 8t + i, h = 0, 1.
    const int tp = tid >> 2, t = tid & 3;
    const int s0 = 2 * tp;
    const int n0 = g * 2 * TILE_T + t64 * 2 * HALF_T + 4 * tp;   // weight row of tuple s0's first row
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t b0[8], b1[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = 32 * h + 8 * t + i;
            const int R = TWO ? rate_s[k] : RL;
            uint32_t st0, st1;
            e2m1w::states2<L>(Ws[k], prev_s[k], s0, R, st0, st1);
            b0[i] = table[st0];
            b1[i] = table[st1];
        }
        uint32_t r[4];
        e2m1w::rows2(b0, r[0], r[1]);
        e2m1w::rows2(b1, r[2], r[3]);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + j;
            if (n < rows)
                *reinterpret_cast<uint32_t*>(B + (long)n * (K / 2) + kc * 32 + 16 * h + 4 * t) = r[j];
        }
    }
    // Scales: thread (tp, t) writes rows 4tp..4tp + 3 at k16 group 4kc + t.
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n = n0 + j;
        if (n < rows) {
            const int grp = 4 * kc + t;
            SF[(long)n * (K / 16) + grp] = (uint8_t)e2m1w::lut_byte(lut, e2m1w::scale_nibble(plane, rows, n, grp));
        }
    }
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
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

// One block per (128-row n tile, 64-k chunk), four warps; warp w takes n8
// tiles 4w..4w+3 (two ldmatrix.x4, each two n8 tiles x both k-halves).
__global__ void __launch_bounds__(128) mma_kernel(const uint8_t* __restrict__ B, const uint8_t* __restrict__ SF,
                                                  int rows, int K, float* __restrict__ out) {
    __shared__ __align__(16) uint8_t Bs[2][128][16];
    __shared__ uint32_t SFs[128];
    const int nt = blockIdx.x, kc = blockIdx.y;
    const int tid = threadIdx.x;
    for (int i = tid; i < 2 * 128; i += 128) {
        const int c = i >> 7, n = i & 127, row = nt * 128 + n;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (row < rows) v = *reinterpret_cast<const uint4*>(B + (long)row * (K / 2) + kc * 32 + 16 * c);
        *reinterpret_cast<uint4*>(Bs[c][n]) = v;
    }
    {
        const int row = nt * 128 + tid;
        SFs[tid] = (row < rows) ? *reinterpret_cast<const uint32_t*>(SF + (long)row * (K / 16) + 4 * kc) : 0u;
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31, gq = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        const int j0 = 4 * warp + 2 * pair;           // n8 tiles j0, j0 + 1
        // matrix q = lane >> 3: n8 tile j0 + (q >> 1), k-half q & 1, row lane & 7
        const int q = lane >> 3;
        uint32_t bf[4];
        ldmatrix_x4(bf, Bs[q & 1][8 * (j0 + (q >> 1)) + (lane & 7)]);
        #pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            const int j = j0 + jj;
            const uint32_t sfb = (t == 0) ? SFs[8 * j + gq] : 0u;
            #pragma unroll
            for (int kb = 0; kb < 64; kb += 16) {
                // one-hot A: row m hot at k = kb + m
                uint32_t a[4];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int m = gq + 8 * (r & 1);
                    const int k = kb + m;
                    const int k0 = 8 * t + 32 * (r >> 1);
                    a[r] = (k >= k0 && k < k0 + 8) ? (0x2u << (4 * (k - k0))) : 0u;
                }
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                mma_fp4(d, a, bf[2 * jj], bf[2 * jj + 1], 0x38383838u, sfb);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int m = gq + 8 * (e >> 1), n = 8 * j + 2 * t + (e & 1);
                    const int row = nt * 128 + n;
                    if (row < rows) out[(long)row * K + kc * 64 + kb + m] = d[e];
                }
            }
        }
    }
}

template <int L>
void launch_decode_l(int rl, bool two, dim3 grid, cudaStream_t st, const int32_t* words, int tile_words,
                     const int32_t* pair, const int32_t* bdesc, const int32_t* init, int has_init,
                     const uint8_t* table, const uint8_t* plane, const uint8_t* lut16, int rows, int K,
                     uint8_t* B, uint8_t* SF) {
#define E2M1_ORACLE_CASE(R)                                                                           \
    case R:                                                                                            \
        if (two) decode_kernel<L, R, true><<<grid, 128, 0, st>>>(words, tile_words, pair, bdesc, init, \
                     has_init, table, plane, lut16, rows, K, B, SF);                                   \
        else decode_kernel<L, R, false><<<grid, 128, 0, st>>>(words, tile_words, pair, bdesc, init,    \
                     has_init, table, plane, lut16, rows, K, B, SF);                                   \
        break;
    switch (rl) {
        E2M1_ORACLE_CASE(1) E2M1_ORACLE_CASE(2) E2M1_ORACLE_CASE(3) E2M1_ORACLE_CASE(4)
        E2M1_ORACLE_CASE(5) E2M1_ORACLE_CASE(6) E2M1_ORACLE_CASE(7) E2M1_ORACLE_CASE(8)
        default: TORCH_CHECK(false, "rate ", rl, " outside 1..8");
    }
#undef E2M1_ORACLE_CASE
}

}  // namespace

std::vector<torch::Tensor> decode(torch::Tensor words, int64_t tile_words, int64_t n_tiles, torch::Tensor pair,
                                  torch::Tensor bdesc, torch::Tensor init, bool has_init, torch::Tensor table,
                                  torch::Tensor plane, torch::Tensor lut16, int64_t rows, int64_t K, int64_t L) {
    TORCH_CHECK(K % CHUNK == 0, "K must be a multiple of 64");
    TORCH_CHECK(rows % 2 == 0, "rows must pair into tuples");
    auto p = pair.cpu();
    const int rl = p[0].item<int>();
    const bool two = p[6].item<int>() > 0;
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(words.device());
    auto B = torch::zeros({rows, K / 2}, opts);
    auto SF = torch::zeros({rows, K / 16}, opts);
    dim3 grid(K / CHUNK, n_tiles * (TILE_T / HALF_T));
    auto st = at::cuda::getCurrentCUDAStream();
    const int32_t* ip = has_init ? init.data_ptr<int32_t>() : nullptr;
    auto args = std::make_tuple(words.data_ptr<int32_t>(), (int)tile_words, pair.data_ptr<int32_t>(),
                                bdesc.data_ptr<int32_t>(), ip, (int)has_init, table.data_ptr<uint8_t>(),
                                plane.data_ptr<uint8_t>(), lut16.data_ptr<uint8_t>(), (int)rows, (int)K,
                                B.data_ptr<uint8_t>(), SF.data_ptr<uint8_t>());
    auto go = [&](auto fn) {
        std::apply([&](auto... a) { fn(rl, two, grid, st, a...); }, args);
    };
    switch (L) {
        case 12: go(launch_decode_l<12>); break;
        case 14: go(launch_decode_l<14>); break;
        default: TORCH_CHECK(false, "window bits ", L, " not instantiated (12, 14)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {B, SF};
}

torch::Tensor mma(torch::Tensor B, torch::Tensor SF, int64_t rows, int64_t K) {
    auto out = torch::full({rows, K}, std::nanf(""), torch::TensorOptions().dtype(torch::kFloat32).device(B.device()));
    dim3 grid((rows + 127) / 128, K / CHUNK);
    mma_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(B.data_ptr<uint8_t>(), SF.data_ptr<uint8_t>(),
                                                                  (int)rows, (int)K, out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode);
    m.def("mma", &mma);
}
