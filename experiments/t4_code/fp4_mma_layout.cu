// The block-scaled FP4 MMA on sm_121a, pinned at the PTX level (tessera#750).
//
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//       .row.col.f32.e2m1.e2m1.f32.ue4m3
//
// Nothing in the tree exercises this instruction below Triton's
// ``tl.dot_scaled``, and the fused E2M1 lane will issue it from inline PTX.
// Two stages, one binary, JSON on stdout:
//
// 1. DISCOVERY of the scale registers.  Each experiment sets ONE byte of ONE
//    lane's SFA (or SFB) register to 2.0 (ue4m3 0x40) over a 1.0 background
//    (0x38), with the A (or B) operand's codes at 1.0 in ONE k16 group and 0
//    elsewhere, and reads which output rows (columns) doubled.  Over every
//    (lane, byte, group) that maps each register byte to the (row, group) or
//    (column, group) it scales, or to nothing.  The A, B and D fragment
//    layouts are the sm80 int4 m16n8k64 ones (lane = 4g + t):
//      A  a[r] nibble i: row g + 8 (r & 1), k = 8t + 32 (r >> 1) + i
//      B  b[r] nibble i: col g,              k = 8t + 32 r + i
//      D  d[j]:          row g + 8 (j >> 1), col 2t + (j & 1)
//    and stage 2 checks them.
// 2. VERIFICATION.  Random E2M1 codes and random UE4M3 scales, loaded by the
//    discovered map, against a double-precision reference.  Codes and scales
//    are drawn so every partial sum is exactly representable in fp32, so the
//    comparison is bitwise whatever the hardware's summation order.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -o fp4_mma_layout fp4_mma_layout.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(3); } } while (0)

__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                        uint32_t sfa, uint32_t sfb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

// Codes are [16][64] (A, row-major) and [8][64] (B, one row per column n),
// one e2m1 code per byte.  The fragment builders follow the layout above.
__device__ void load_a(const uint8_t* A, uint32_t (&a)[4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = g + 8 * (r & 1), k0 = 8 * t + 32 * (r >> 1);
        uint32_t v = 0;
        for (int i = 0; i < 8; ++i) v |= uint32_t(A[row * 64 + k0 + i] & 15) << (4 * i);
        a[r] = v;
    }
}
__device__ void load_b(const uint8_t* B, uint32_t (&b)[2]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int k0 = 8 * t + 32 * r;
        uint32_t v = 0;
        for (int i = 0; i < 8; ++i) v |= uint32_t(B[g * 64 + k0 + i] & 15) << (4 * i);
        b[r] = v;
    }
}
__device__ void store_d(float* D, const float (&d)[4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int j = 0; j < 4; ++j) D[(g + 8 * (j >> 1)) * 8 + 2 * t + (j & 1)] = d[j];
}

// Stage 1: experiment e = ((op * 32 + lane) * 4 + byte) * 4 + group.
__global__ void discover(float* out) {
    const int e = blockIdx.x;
    const int group = e & 3, byte = (e >> 2) & 3, tl = (e >> 4) & 31, op = e >> 9;
    __shared__ uint8_t A[16 * 64], B[8 * 64];
    for (int i = threadIdx.x; i < 16 * 64; i += 32) {
        const int k = i & 63;
        A[i] = (op == 1 || (k >> 4) == group) ? 0x2 : 0x0;       // e2m1 0x2 = 1.0
    }
    for (int i = threadIdx.x; i < 8 * 64; i += 32) {
        const int k = i & 63;
        B[i] = (op == 0 || (k >> 4) == group) ? 0x2 : 0x0;
    }
    __syncwarp();
    uint32_t a[4], b[2];
    load_a(A, a);
    load_b(B, b);
    const int lane = threadIdx.x & 31;
    const uint32_t one = 0x38383838u;                            // ue4m3 0x38 = 1.0
    const uint32_t hit = (lane == tl) ? (one & ~(0xFFu << (8 * byte))) | (0x40u << (8 * byte)) : one;
    uint32_t sfa = op == 0 ? hit : one, sfb = op == 1 ? hit : one;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_fp4(d, a, b, sfa, sfb);
    store_d(out + size_t(e) * 128, d);
}

// Stage 2: trial ``blockIdx.x``.  ``sfa_idx``/``sfb_idx`` [32][4] give, per
// (lane, byte), the index into the trial's SFA [16][4] / SFB [8][4] arrays
// (row * 4 + group), or -1 for a byte the instruction ignores (set to 1.0).
__global__ void verify(const uint8_t* codesA, const uint8_t* codesB, const uint8_t* sfA,
                       const uint8_t* sfB, const int* sfa_idx, const int* sfb_idx, float* out) {
    const int trial = blockIdx.x, lane = threadIdx.x & 31;
    uint32_t a[4], b[2];
    load_a(codesA + size_t(trial) * 16 * 64, a);
    load_b(codesB + size_t(trial) * 8 * 64, b);
    uint32_t sfa = 0, sfb = 0;
    for (int by = 0; by < 4; ++by) {
        const int ia = sfa_idx[lane * 4 + by], ib = sfb_idx[lane * 4 + by];
        sfa |= uint32_t(ia >= 0 ? sfA[size_t(trial) * 64 + ia] : 0x38) << (8 * by);
        sfb |= uint32_t(ib >= 0 ? sfB[size_t(trial) * 32 + ib] : 0x38) << (8 * by);
    }
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_fp4(d, a, b, sfa, sfb);
    store_d(out + size_t(trial) * 128, d);
}

static double e2m1_value(int c) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (c & 8) ? -mag[c & 7] : mag[c & 7];
}
static double ue4m3_value(int c) {                               // e4m3 without sign, bias 7
    const int e = (c >> 3) & 15, m = c & 7;
    return e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
}

int main(int argc, char** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 4096;
    // ---------------------------------------------------------------- stage 1
    const int NE = 2 * 32 * 4 * 4;
    float* dout;
    CK(cudaMalloc(&dout, sizeof(float) * NE * 128));
    discover<<<NE, 32>>>(dout);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> h(size_t(NE) * 128);
    CK(cudaMemcpy(h.data(), dout, sizeof(float) * h.size(), cudaMemcpyDeviceToHost));
    std::vector<int> sfa_idx(128, -1), sfb_idx(128, -1);
    int anomalies = 0;
    for (int op = 0; op < 2; ++op)
        for (int tl = 0; tl < 32; ++tl)
            for (int by = 0; by < 4; ++by) {
                int found = -1, hits = 0;
                for (int grp = 0; grp < 4; ++grp) {
                    const float* d = &h[size_t(((op * 32 + tl) * 4 + by) * 4 + grp) * 128];
                    // Baseline: one k16 group of 1.0 x 1.0 at unit scales = 16 per output.
                    std::vector<int> changed;
                    for (int m = 0; m < 16; ++m)
                        for (int n = 0; n < 8; ++n) {
                            const float v = d[m * 8 + n];
                            if (v == 32.f) changed.push_back(m * 8 + n);
                            else if (v != 16.f) ++anomalies;
                        }
                    if (changed.empty()) continue;
                    // A scale must change exactly one row (all 8 columns); a B scale one column.
                    int line = -1;
                    bool ok = true;
                    if (op == 0) {
                        line = changed[0] / 8;
                        ok = changed.size() == 8;
                        for (int c : changed) ok = ok && c / 8 == line;
                    } else {
                        line = changed[0] % 8;
                        ok = changed.size() == 16;
                        for (int c : changed) ok = ok && c % 8 == line;
                    }
                    if (!ok) { ++anomalies; continue; }
                    found = line * 4 + grp;
                    ++hits;
                }
                if (hits > 1) ++anomalies;
                (op == 0 ? sfa_idx : sfb_idx)[tl * 4 + by] = hits == 1 ? found : -1;
            }
    // Coverage: every (row, group) of A and (col, group) of B named by some byte.
    std::vector<int> cov_a(64, 0), cov_b(32, 0);
    for (int i = 0; i < 128; ++i) {
        if (sfa_idx[i] >= 0) ++cov_a[sfa_idx[i]];
        if (sfb_idx[i] >= 0) ++cov_b[sfb_idx[i]];
    }
    int a_missing = 0, b_missing = 0;
    for (int v : cov_a) a_missing += v == 0;
    for (int v : cov_b) b_missing += v == 0;
    // ---------------------------------------------------------------- stage 2
    std::mt19937 rng(20260930);
    // Codes: every e2m1 code.  Scales: ue4m3 in [0.25, 4] with at most two
    // significant bits (mantissa 0 or 4), so each product carries at most
    // 2+2+2+2 significant bits and every partial sum of 64 is a multiple of
    // 2^-8 below 2^16: exact in fp32 in any order.
    const uint8_t scale_codes[] = {0x28, 0x2C, 0x30, 0x34, 0x38, 0x3C, 0x40, 0x44, 0x48};
    std::vector<uint8_t> cA(size_t(trials) * 1024), cB(size_t(trials) * 512),
        sA(size_t(trials) * 64), sB(size_t(trials) * 32);
    for (auto& v : cA) v = rng() & 15;
    for (auto& v : cB) v = rng() & 15;
    for (auto& v : sA) v = scale_codes[rng() % 9];
    for (auto& v : sB) v = scale_codes[rng() % 9];
    uint8_t *dA, *dB, *dsA, *dsB;
    int *dia, *dib;
    float* dv;
    CK(cudaMalloc(&dA, cA.size()));
    CK(cudaMalloc(&dB, cB.size()));
    CK(cudaMalloc(&dsA, sA.size()));
    CK(cudaMalloc(&dsB, sB.size()));
    CK(cudaMalloc(&dia, 128 * sizeof(int)));
    CK(cudaMalloc(&dib, 128 * sizeof(int)));
    CK(cudaMalloc(&dv, sizeof(float) * size_t(trials) * 128));
    CK(cudaMemcpy(dA, cA.data(), cA.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, cB.data(), cB.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsA, sA.data(), sA.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsB, sB.data(), sB.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dia, sfa_idx.data(), 128 * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dib, sfb_idx.data(), 128 * sizeof(int), cudaMemcpyHostToDevice));
    verify<<<trials, 32>>>(dA, dB, dsA, dsB, dia, dib, dv);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> got(size_t(trials) * 128);
    CK(cudaMemcpy(got.data(), dv, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));
    long mismatches = 0, not_exact_ref = 0;
    double max_abs = 0.0;
    for (int tr = 0; tr < trials; ++tr)
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 8; ++n) {
                double ref = 0.0;
                for (int k = 0; k < 64; ++k)
                    ref += e2m1_value(cA[size_t(tr) * 1024 + m * 64 + k])
                           * e2m1_value(cB[size_t(tr) * 512 + n * 64 + k])
                           * ue4m3_value(sA[size_t(tr) * 64 + m * 4 + (k >> 4)])
                           * ue4m3_value(sB[size_t(tr) * 32 + n * 4 + (k >> 4)]);
                if (double(float(ref)) != ref) ++not_exact_ref;
                const double g = got[size_t(tr) * 128 + m * 8 + n];
                if (g != ref) ++mismatches;
                max_abs = std::fmax(max_abs, std::fabs(g - ref));
            }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("{\"device\": \"%s\", \"cc\": \"%d.%d\", \"discovery_anomalies\": %d, "
           "\"sfa_rows_groups_missing\": %d, \"sfb_cols_groups_missing\": %d, ",
           prop.name, prop.major, prop.minor, anomalies, a_missing, b_missing);
    printf("\"sfa_map\": [");
    for (int i = 0; i < 128; ++i) printf("%s%d", i ? ", " : "", sfa_idx[i]);
    printf("], \"sfb_map\": [");
    for (int i = 0; i < 128; ++i) printf("%s%d", i ? ", " : "", sfb_idx[i]);
    printf("], \"trials\": %d, \"outputs\": %ld, \"reference_not_fp32_exact\": %ld, "
           "\"mismatches\": %ld, \"max_abs_err\": %.9g}\n",
           trials, long(trials) * 128, not_exact_ref, mismatches, max_abs);
    return (anomalies || a_missing || b_missing || mismatches) ? 1 : 0;
}
