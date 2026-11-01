// Scalar operations at the actual single-precision and bfloat16 boundaries.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

struct Input { uint32_t value, scale; };
struct Output { uint32_t product, bfloat16; };
#define CK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); return 3; } } while (0)

__global__ void boundary(const Input* input, Output* output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = __uint_as_float(input[i].value);
    const float scale = __uint_as_float(input[i].scale);
    const float product = __fmul_rn(value, scale);
    output[i] = {__float_as_uint(product), uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(product))) << 16};
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
    Output* device_output;
    CK(cudaMalloc(&device_input, size_t(bytes)));
    CK(cudaMalloc(&device_output, count * sizeof(Output)));
    CK(cudaMemcpy(device_input, host.data(), size_t(bytes), cudaMemcpyHostToDevice));
    boundary<<<(count + 127) / 128, 128>>>(device_input, device_output, count);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<Output> output(count);
    CK(cudaMemcpy(output.data(), device_output, count * sizeof(Output), cudaMemcpyDeviceToHost));
    std::ofstream result(argv[2], std::ios::binary);
    result.write(reinterpret_cast<const char*>(output.data()), count * sizeof(Output));
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
