// Tile-order window words -> register-direct fragment words, on the device (eng-regdirect-build).
//
// The routed construction bundles hold each expert's BODY in tile order (kernel_window_gemv
// layout): per 512-row tile, per rate run, per column, a chunk of 16*R words carrying that
// column's MSB-first code stream.  The register-direct kernel reads the fragment order
// (tessera.fragment_wire): per expert, per 128-row tile, per slot (rate-sorted 32-column group;
// down: a pair), per warp (16 rows), 32 lanes x R words, word i of lane L at i*32+L; lane (g,t)
// holds pair q = p*8+j at bit q*2R: rows 2g, 2g+1 of column 8t+j of group p.  History units
// (lanes (6,t), (7,t), rows -4..-1 from the start state) precede nothing: they are a plane.
//
// This is a pure bit permutation (plus the history fields the start state implies).  The test
// is bitwise equality with regdirect_routed.layer_stacks, the CPU-verified torch path.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {

struct Args {
    const uint32_t* words[2];      // tile-order words of projection p (gate, up) or [down, down]
    const int64_t* tile_word0[2];  // [E] each expert's first tile word in words[p]
    const int32_t* tile_words[2];  // [E] words per 512-row tile
    const int32_t* colbase[2];     // [E, cols] chunk word offset of original column c in a tile
    const int32_t* start[2];       // [E, cols] start state (0 when none), original column order
    const int16_t* kperm;          // [E, ks*ng] slot -> original 32-column group
    const int32_t* rate;           // [E] packed profile: ra | rb<<4 | ksa<<8
    const int64_t* wire0;          // [E] fragment wire word offset
    const int64_t* hist0;          // [E] history word offset
    uint32_t* wire;
    uint32_t* hist;
    int E, rows, cols, ks, ng, nt;  // nt = 128-row tiles
};

__device__ __forceinline__ uint32_t read_code(const Args& a, int p, int e, int n, int c, int r) {
    if (n >= a.rows) return 0u;                                   // padded tile rows hold zero codes
    const int64_t base = a.tile_word0[p][e] + (int64_t)(n >> 9) * a.tile_words[p][e]
                         + a.colbase[p][(int64_t)e * a.cols + c];
    const int bit = (n & 511) * r;
    const uint32_t* w = a.words[p] + base + (bit >> 5);
    const int off = bit & 31;
    const uint64_t pair = ((uint64_t)w[0] << 32) | (off + r > 32 ? (uint64_t)w[1] : 0ull);
    return (uint32_t)((pair >> (64 - off - r)) & ((1u << r) - 1u));
}

__device__ __forceinline__ void slot_rate(int prof, int s, int& r, int64_t& unit0, int64_t& hunit0) {
    const int ra = prof & 15, rb = (prof >> 4) & 15, ksa = prof >> 8;
    if (s < ksa) { r = ra; unit0 = (int64_t)s * 256 * ra; hunit0 = (int64_t)s * 8 * ra; }
    else { r = rb; unit0 = (int64_t)ksa * 256 * ra + (int64_t)(s - ksa) * 256 * rb;
           hunit0 = (int64_t)ksa * 8 * ra + (int64_t)(s - ksa) * 8 * rb; }
}

// One thread per (expert, 128-row tile, slot, warp, lane): that lane's R words of one unit.
__global__ void units(Args a) {
    const int64_t total = (int64_t)a.E * a.nt * a.ks * 256;
    for (int64_t id = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; id < total;
         id += (int64_t)gridDim.x * blockDim.x) {
        const int lane = id & 31, w = (id >> 5) & 7;
        int64_t rest = id >> 8;
        const int s = rest % a.ks; rest /= a.ks;
        const int T = rest % a.nt; const int e = (int)(rest / a.nt);
        const int prof = a.rate[e];
        int r; int64_t unit0, hunit0;
        slot_rate(prof, s, r, unit0, hunit0);
        const int ra = prof & 15, rb = (prof >> 4) & 15, ksa = prof >> 8;
        const int64_t tile_frag = (int64_t)ksa * 256 * ra + (int64_t)(a.ks - ksa) * 256 * rb;
        uint32_t* out = a.wire + a.wire0[e] + T * tile_frag + unit0 + (int64_t)w * 32 * r;
        const int g = lane >> 2, t = lane & 3;
        uint32_t cur = 0; int fill = 0, word = 0;
        for (int q = 0; q < 16; ++q) {
            const int p = q >> 3, j = q & 7;
            const int grp = a.kperm[(int64_t)e * a.ks * a.ng + s * a.ng + (a.ng == 1 ? 0 : p)];
            const int proj = a.ng == 1 ? p : 0;
            const int c = grp * 32 + 8 * t + j;
            for (int row = 0; row < 2; ++row) {
                const uint32_t f = read_code(a, proj, e, T * 128 + w * 16 + 2 * g + row, c, r);
                const int room = 32 - fill;                       // append r bits, MSB-first
                if (r <= room) {
                    cur |= f << (room - r); fill += r;
                } else {
                    cur |= f >> (r - room);
                    out[word * 32 + lane] = cur; ++word;
                    cur = f << (32 - (r - room)); fill = r - room;
                }
                if (fill == 32) { out[word * 32 + lane] = cur; ++word; cur = 0; fill = 0; }
            }
        }
    }
}

// One thread per (expert, slot, history lane): rows -4..-1 of lanes (6,t), (7,t) from the start state.
__global__ void history(Args a) {
    const int64_t total = (int64_t)a.E * a.ks * 8;
    for (int64_t id = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; id < total;
         id += (int64_t)gridDim.x * blockDim.x) {
        const int l = id & 7; int64_t rest = id >> 3;
        const int s = rest % a.ks; const int e = (int)(rest / a.ks);
        int r; int64_t unit0, hunit0;
        slot_rate(a.rate[e], s, r, unit0, hunit0);
        uint32_t* out = a.hist + a.hist0[e] + hunit0;
        const int g = 6 + (l >> 2), t = l & 3;
        uint32_t cur = 0; int fill = 0, word = 0;
        for (int q = 0; q < 16; ++q) {
            const int p = q >> 3, j = q & 7;
            const int grp = a.kperm[(int64_t)e * a.ks * a.ng + s * a.ng + (a.ng == 1 ? 0 : p)];
            const int proj = a.ng == 1 ? p : 0;
            const uint32_t st = (uint32_t)a.start[proj][(int64_t)e * a.cols + grp * 32 + 8 * t + j];
            for (int row = 0; row < 2; ++row) {
                const int k = 4 - ((g - 6) * 2 + row);              // this field is row -k
                const uint32_t f = (st >> ((k - 1) * r)) & ((1u << r) - 1u);
                const int room = 32 - fill;
                if (r <= room) {
                    cur |= f << (room - r); fill += r;
                } else {
                    cur |= f >> (r - room);
                    out[word * 8 + l] = cur; ++word;
                    cur = f << (32 - (r - room)); fill = r - room;
                }
                if (fill == 32) { out[word * 8 + l] = cur; ++word; cur = 0; fill = 0; }
            }
        }
    }
}

}  // namespace

void transcode(std::vector<torch::Tensor> words, std::vector<torch::Tensor> tile_word0,
               std::vector<torch::Tensor> tile_words, std::vector<torch::Tensor> colbase,
               std::vector<torch::Tensor> start, torch::Tensor kperm, torch::Tensor rate,
               torch::Tensor wire0, torch::Tensor hist0, torch::Tensor wire, torch::Tensor hist,
               int64_t rows, int64_t ks, int64_t ng) {
    TORCH_CHECK(words.size() == 2 && tile_word0.size() == 2 && tile_words.size() == 2 && colbase.size() == 2
                && start.size() == 2, "two projection slots");
    Args a{};
    for (int p = 0; p < 2; ++p) {
        a.words[p] = reinterpret_cast<const uint32_t*>(words[p].data_ptr<int32_t>());
        a.tile_word0[p] = tile_word0[p].data_ptr<int64_t>();
        a.tile_words[p] = tile_words[p].data_ptr<int32_t>();
        a.colbase[p] = colbase[p].data_ptr<int32_t>();
        a.start[p] = start[p].data_ptr<int32_t>();
    }
    a.kperm = kperm.data_ptr<int16_t>();
    a.rate = rate.data_ptr<int32_t>();
    a.wire0 = wire0.data_ptr<int64_t>();
    a.hist0 = hist0.data_ptr<int64_t>();
    a.wire = reinterpret_cast<uint32_t*>(wire.data_ptr<int32_t>());
    a.hist = reinterpret_cast<uint32_t*>(hist.data_ptr<int32_t>());
    a.E = (int)rate.size(0); a.rows = (int)rows; a.cols = (int)colbase[0].size(1);
    a.ks = (int)ks; a.ng = (int)ng; a.nt = (int)((rows + 127) / 128);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int64_t n_units = (int64_t)a.E * a.nt * a.ks * 256;
    const int blocks = (int)std::min<int64_t>((n_units + threads - 1) / threads, 65535 * 8);
    units<<<blocks, threads, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t n_hist = (int64_t)a.E * a.ks * 8;
    history<<<(int)std::min<int64_t>((n_hist + threads - 1) / threads, 65535), threads, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transcode", &transcode);
}
