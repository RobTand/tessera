#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <vector>

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 3; } } while (0)

extern "C" __global__ void swiglu_probe(const float* gf, float* d, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = gf[i] / (1.0f + expf(-gf[i]));
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
    auto bytes = in.tellg();
    if (bytes <= 0 || bytes % 4) return 2;
    std::vector<float> h(size_t(bytes) / 4);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(h.data()), bytes)) return 2;
    float *a, *d;
    CK(cudaMalloc(&a, size_t(bytes))); CK(cudaMalloc(&d, size_t(bytes)));
    CK(cudaMemcpy(a, h.data(), size_t(bytes), cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    swiglu_probe<<<(h.size() + 255) / 256, 256>>>(a, d, h.size());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), d, size_t(bytes), cudaMemcpyDeviceToHost));
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<char*>(h.data()), bytes);
    CK(cudaFree(a)); CK(cudaFree(d));
    return out ? 0 : 2;
}
