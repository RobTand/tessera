// cuBLASLt with a named algorithm, for the BF16 projection GEMMs of the stock
// unquantized Linear (experiments/t8r_speed/gemm_algo_bitwise.py, tessera#806).
//
// out[M, N] = x[M, K] @ w[N, K]^T, bf16 in and out, fp32 compute and scale:
// the operation torch.nn.functional.linear runs, in the same column-major view
// (C(N x M) = op_T(W)(N x K) * X(K x M)). ``heuristics`` and ``exhaustive``
// enumerate algorithms (the sweep);
// ``check`` and ``run`` execute one by its 64-byte descriptor. The sweep and the
// serving lever must run this same file, so the bitwise evidence is for these descriptors.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

#define LT_CHECK(x) do { cublasStatus_t s_ = (x); \
  TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #x " failed: status ", (int)s_); } while (0)

namespace {
cublasLtHandle_t lt() {
  static cublasLtHandle_t h = [] { cublasLtHandle_t x; LT_CHECK(cublasLtCreate(&x)); return x; }();
  return h;
}
struct Descs { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; };
// torch linear: out[M,N] = x[M,K] w[N,K]^T. Column-major view: C(N x M) = op_T(W)(N x K) * X(K x M).
Descs& descs(int64_t M, int64_t N, int64_t K) {
  static std::map<std::tuple<int64_t, int64_t, int64_t>, Descs> cache;
  auto key = std::make_tuple(M, N, K);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  Descs d;
  LT_CHECK(cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&d.a, CUDA_R_16BF, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&d.b, CUDA_R_16BF, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&d.c, CUDA_R_16BF, N, M, N));
  return cache.emplace(key, d).first->second;
}
torch::Tensor pack(const cublasLtMatmulAlgo_t& algo, size_t ws, float waves, int state) {
  static_assert(sizeof(cublasLtMatmulAlgo_t) == 64, "algo blob is 8 x uint64");
  auto t = torch::zeros({11}, torch::kInt64);
  std::memcpy(t.data_ptr<int64_t>(), &algo, sizeof(algo));
  t[8] = (int64_t)ws; t[9] = (int64_t)(waves * 1000.0f); t[10] = state;
  return t;
}
cublasLtMatmulAlgo_t unpack(const torch::Tensor& t) {
  cublasLtMatmulAlgo_t a;
  std::memcpy(&a, t.contiguous().data_ptr<int64_t>(), sizeof(a));
  return a;
}
std::vector<uint32_t> cap_u32(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
  size_t need = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, nullptr, 0, &need) != CUBLAS_STATUS_SUCCESS || need == 0)
    return {};
  std::vector<uint32_t> v((need + 3) / 4);
  size_t got = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, v.data(), v.size() * 4, &got) != CUBLAS_STATUS_SUCCESS)
    return {};
  v.resize(got / 4);
  return v;
}
}  // namespace

std::vector<torch::Tensor> heuristics(int64_t M, int64_t N, int64_t K, int64_t ws, int64_t max_n) {
  auto& d = descs(M, N, K);
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  uint64_t wsb = (uint64_t)ws;
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb)));
  std::vector<cublasLtMatmulHeuristicResult_t> res((size_t)max_n);
  int got = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt(), d.op, d.a, d.b, d.c, d.c, pref, (int)max_n, res.data(), &got));
  cublasLtMatmulPreferenceDestroy(pref);
  std::vector<torch::Tensor> out;
  for (int i = 0; i < got; ++i)
    if (res[i].state == CUBLAS_STATUS_SUCCESS)
      out.push_back(pack(res[i].algo, res[i].workspaceSize, res[i].wavesCount, (int)res[i].state));
  return out;
}

std::vector<torch::Tensor> exhaustive(int64_t M, int64_t N, int64_t K, int64_t ws, int64_t max_n) {
  auto& d = descs(M, N, K);
  std::vector<int> ids(512);
  int n_ids = 0;
  LT_CHECK(cublasLtMatmulAlgoGetIds(lt(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                    CUDA_R_16BF, CUDA_R_16BF, (int)ids.size(), ids.data(), &n_ids));
  std::vector<torch::Tensor> out;
  for (int i = 0; i < n_ids && (int64_t)out.size() < max_n; ++i) {
    cublasLtMatmulAlgo_t base;
    if (cublasLtMatmulAlgoInit(lt(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                               CUDA_R_16BF, ids[i], &base) != CUBLAS_STATUS_SUCCESS)
      continue;
    auto tiles = cap_u32(base, CUBLASLT_ALGO_CAP_TILE_IDS);
    if (tiles.empty()) tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
    auto stages = cap_u32(base, CUBLASLT_ALGO_CAP_STAGES_IDS);
    if (stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
    for (uint32_t tile : tiles) {
      for (uint32_t stage : stages) {
        for (uint32_t swz = 0; swz < 2; ++swz) {
          if ((int64_t)out.size() >= max_n) break;
          cublasLtMatmulAlgo_t a = base;
          uint32_t splitk = 1, red = CUBLASLT_REDUCTION_SCHEME_NONE, custom = 0;
          if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stage, sizeof(stage)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof(custom)))
            continue;
          cublasLtMatmulHeuristicResult_t r;
          if (cublasLtMatmulAlgoCheck(lt(), d.op, d.a, d.b, d.c, d.c, &a, &r) != CUBLAS_STATUS_SUCCESS) continue;
          if (r.workspaceSize > (size_t)ws) continue;
          out.push_back(pack(a, r.workspaceSize, r.wavesCount, (int)r.state));
        }
      }
    }
  }
  return out;
}

std::vector<int64_t> describe(torch::Tensor t) {
  auto a = unpack(t);
  const cublasLtMatmulAlgoConfigAttributes_t attrs[] = {
      CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
      CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_STAGES_ID,
      CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
  std::vector<int64_t> v;
  for (auto attr : attrs) {
    size_t need = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&a, attr, nullptr, 0, &need) != CUBLAS_STATUS_SUCCESS ||
        need == 0 || need > 8) { v.push_back(-1); continue; }
    uint64_t buf = 0;
    size_t got = 0;
    v.push_back(cublasLtMatmulAlgoConfigGetAttribute(&a, attr, &buf, need, &got) == CUBLAS_STATUS_SUCCESS
                    ? (int64_t)buf : -1);
  }
  return v;
}

int64_t check(torch::Tensor t, int64_t M, int64_t N, int64_t K) {
  auto& d = descs(M, N, K);
  auto a = unpack(t);
  cublasLtMatmulHeuristicResult_t r;
  if (cublasLtMatmulAlgoCheck(lt(), d.op, d.a, d.b, d.c, d.c, &a, &r) != CUBLAS_STATUS_SUCCESS) return -1;
  return (int64_t)r.workspaceSize;
}

void run(torch::Tensor t, torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws) {
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  auto& d = descs(M, N, K);
  auto a = unpack(t);
  const float alpha = 1.f, beta = 0.f;
  LT_CHECK(cublasLtMatmul(lt(), d.op, &alpha, w.data_ptr(), d.a, x.data_ptr(), d.b, &beta, out.data_ptr(), d.c,
                          out.data_ptr(), d.c, &a, ws.data_ptr(), (size_t)ws.numel(),
                          at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("heuristics", &heuristics);
  m.def("exhaustive", &exhaustive);
  m.def("describe", &describe);
  m.def("check", &check);
  m.def("run", &run);
}
