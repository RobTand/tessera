// Target the actual scale_vec::4X FP4 instruction, not a library GEMM.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include <stdexcept>

struct Input {
    uint8_t a[1024], b[512], sa[64], sb[32];
    uint32_t c[128];
};
static_assert(sizeof(Input) == 2144, "probe input layout");

#define CK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); return 3; } } while (0)

__global__ void probe(const Input* inputs, uint32_t* output) {
    const Input& in = inputs[blockIdx.x];
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4] = {}, b[2] = {};
    for (int r = 0; r < 4; ++r) {
        const int row = g + 8 * (r & 1), k0 = 8 * t + 32 * (r >> 1);
        for (int i = 0; i < 8; ++i) a[r] |= uint32_t(in.a[row * 64 + k0 + i]) << (4 * i);
    }
    for (int r = 0; r < 2; ++r)
        for (int i = 0; i < 8; ++i) b[r] |= uint32_t(in.b[g * 64 + 8 * t + 32 * r + i]) << (4 * i);
    uint32_t sa = 0, sb = 0;
    for (int i = 0; i < 4; ++i) {
        sa |= uint32_t(t < 2 ? in.sa[(g + 8 * t) * 4 + i] : 0x38) << (8 * i);
        sb |= uint32_t(t == 0 ? in.sb[g * 4 + i] : 0x38) << (8 * i);
    }
    float d[4];
    for (int j = 0; j < 4; ++j) d[j] = __uint_as_float(in.c[(g + 8 * (j >> 1)) * 8 + 2 * t + (j & 1)]);
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
    for (int j = 0; j < 4; ++j)
        output[blockIdx.x * 128 + (g + 8 * (j >> 1)) * 8 + 2 * t + (j & 1)] = __float_as_uint(d[j]);
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % sizeof(Input)) return 2;
    const size_t count = size_t(bytes) / sizeof(Input);
    std::vector<Input> host(count);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(host.data()), bytes)) return 2;
    Input* device_input;
    uint32_t* device_output;
    CK(cudaMalloc(&device_input, size_t(bytes)));
    CK(cudaMalloc(&device_output, count * 128 * sizeof(uint32_t)));
    CK(cudaMemcpy(device_input, host.data(), size_t(bytes), cudaMemcpyHostToDevice));
    probe<<<count, 32>>>(device_input, device_output);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> output(count * 128);
    CK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::ofstream result(argv[2], std::ios::binary);
    result.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(uint32_t));
    if (!result) return 2;
    cudaDeviceProp device;
    CK(cudaGetDeviceProperties(&device, 0));
    int driver, runtime;
    CK(cudaDriverGetVersion(&driver));
    CK(cudaRuntimeGetVersion(&runtime));
    std::printf("{\"device\":\"%s\",\"compute_capability\":[%d,%d],\"driver\":%d,\"runtime\":%d,\"cases\":%zu}\n",
                device.name, device.major, device.minor, driver, runtime, count);
    CK(cudaFree(device_input));
    CK(cudaFree(device_output));
    return 0;
}
