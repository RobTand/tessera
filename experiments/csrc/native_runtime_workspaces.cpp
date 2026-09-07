// Read existing Torch BLAS workspace maps after Torch/CUDA initialization.
// Build and execute only through PrismaBuild. This observer never creates,
// clears, resizes or substitutes a serving workspace.
#include <ATen/cuda/CUDAContextLight.h>
#include <shared_mutex>
#include <sstream>
#include <string>

namespace {
thread_local std::string result;
void append(std::ostringstream& out, at::cuda::WorkspaceMapWithMutex& state,
            const char* owner, bool& first) {
  std::shared_lock<std::shared_mutex> lock(state.mutex);
  for (const auto& entry : state.map) {
    if (!first) out << ',';
    first = false;
    const auto& pointer = entry.second.first;
    out << "{\"owner\":\"" << owner << "\",\"handle\":"
        << reinterpret_cast<uintptr_t>(std::get<0>(entry.first))
        << ",\"stream\":" << reinterpret_cast<uintptr_t>(std::get<1>(entry.first))
        << ",\"address\":" << reinterpret_cast<uintptr_t>(pointer.get())
        << ",\"bytes\":" << entry.second.second
        << ",\"device_id\":" << int(pointer.device().index())
        << ",\"device_type\":\"" << (pointer.device().is_cuda() ? "cuda" : "unsupported") << "\"}";
  }
}
}

extern "C" const char* tessera_blas_workspace_snapshot() {
  std::ostringstream out;
  out << "{\"schema\":\"tessera.torch_blas_workspace_observation.v1\",\"workspaces\":[";
  bool first = true;
  append(out, at::cuda::cublas_handle_stream_to_workspace(), "torch.cublas", first);
  append(out, at::cuda::cublaslt_handle_stream_to_workspace(), "torch.cublaslt", first);
  out << "]}";
  result = out.str();
  return result.c_str();
}
