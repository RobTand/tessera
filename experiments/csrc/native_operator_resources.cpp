// CUDA 13 CUPTI Memory4 observer. Build and execute only through PrismaBuild.
// One collection per fresh process; no CUDA initialization or allocation here.
#include <cupti.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::mutex mu;
std::vector<std::string> events, apis, arguments, markers, drops, errors, configuration;
CUpti_SubscriberHandle subscriber;
bool subscribed = false;
const CUpti_CallbackId argument_callbacks[] = {
  CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020, CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020,
  CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020, CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020,
  CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020, CUPTI_RUNTIME_TRACE_CBID_cudaHostGetDevicePointer_v3020
};
bool started = false, stopped = false;
uint64_t begin_ns = 0, end_ns = 0;
uint32_t version = 0;
unsigned buffers = 0, pools = 0;
std::string quote(const char* s) {
  std::ostringstream o; o << '"';
  for (const unsigned char c : std::string(s ? s : "")) {
    if (c == '"' || c == '\\') o << '\\' << char(c);
    else if (c < 32) { const char* h = "0123456789abcdef"; o << "\\u00" << h[c >> 4] << h[c & 15]; }
    else o << char(c);
  }
  return o.str() + '"';
}
bool check(CUptiResult code, const char* operation) {
  if (code == CUPTI_SUCCESS) return true;
  std::lock_guard<std::mutex> lock(mu);
  errors.push_back("{\"operation\":" + quote(operation) + ",\"code\":" + std::to_string(code) + "}");
  return false;
}
void configured(CUptiResult code, const char* operation) {
  check(code, operation);
  std::lock_guard<std::mutex> lock(mu);
  configuration.push_back("{\"operation\":" + quote(operation) + ",\"code\":" + std::to_string(code) + "}");
}
void CUPTIAPI argument_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* data) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
  const auto* info = static_cast<const CUpti_CallbackData*>(data);
  if (info->callbackSite != CUPTI_API_EXIT) return;
  if (!info->functionParams || !info->functionReturnValue) {
    check(CUPTI_ERROR_INVALID_PARAMETER, "missing_argument_callback_data"); return;
  }
  const int result = *static_cast<const cudaError_t*>(info->functionReturnValue);
  uint64_t timestamp = 0;
  if (!check(cuptiGetTimestamp(&timestamp), "argument_timestamp")) return;
  std::ostringstream o;
  o << "{\"name\":" << quote(info->functionName) << ",\"process_id\":" << getpid()
    << ",\"correlation_id\":" << info->correlationId << ",\"callback_id\":" << cbid
    << ",\"timestamp_ns\":" << timestamp << ",\"site\":\"exit\",\"return_value\":" << result;
  switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020: {
      const auto* p = static_cast<const cudaMalloc_v3020_params*>(info->functionParams);
      o << ",\"bytes\":" << p->size << ",\"device_address\":"
        << (result == 0 && p->devPtr ? reinterpret_cast<uintptr_t>(*p->devPtr) : 0); break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020: {
      const auto* p = static_cast<const cudaFree_v3020_params*>(info->functionParams);
      o << ",\"device_address\":" << reinterpret_cast<uintptr_t>(p->devPtr); break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020: {
      const auto* p = static_cast<const cudaHostAlloc_v3020_params*>(info->functionParams);
      o << ",\"bytes\":" << p->size << ",\"flags\":" << p->flags << ",\"host_address\":"
        << (result == 0 && p->pHost ? reinterpret_cast<uintptr_t>(*p->pHost) : 0); break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020: {
      const auto* p = static_cast<const cudaMallocHost_v3020_params*>(info->functionParams);
      o << ",\"bytes\":" << p->size << ",\"host_address\":"
        << (result == 0 && p->ptr ? reinterpret_cast<uintptr_t>(*p->ptr) : 0); break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020: {
      const auto* p = static_cast<const cudaFreeHost_v3020_params*>(info->functionParams);
      o << ",\"host_address\":" << reinterpret_cast<uintptr_t>(p->ptr); break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaHostGetDevicePointer_v3020: {
      const auto* p = static_cast<const cudaHostGetDevicePointer_v3020_params*>(info->functionParams);
      o << ",\"host_address\":" << reinterpret_cast<uintptr_t>(p->pHost) << ",\"flags\":" << p->flags
        << ",\"device_address\":" << (result == 0 && p->pDevice ? reinterpret_cast<uintptr_t>(*p->pDevice) : 0); break;
    }
    default: check(CUPTI_ERROR_INVALID_PARAMETER, "unexpected_argument_callback"); return;
  }
  o << '}';
  std::lock_guard<std::mutex> lock(mu); arguments.push_back(o.str());
}
void dropped(CUcontext ctx, uint32_t stream) {
  size_t n = 0;
  const auto code = cuptiActivityGetNumDroppedRecords(ctx, stream, &n);
  check(code, "dropped_records");
  std::ostringstream o;
  o << "{\"context_handle\":" << reinterpret_cast<uintptr_t>(ctx)
    << ",\"stream_id\":" << stream << ",\"count\":" << n << ",\"code\":" << code << "}";
  std::lock_guard<std::mutex> lock(mu); drops.push_back(o.str());
}
void CUPTIAPI request(uint8_t** buffer, size_t* size, size_t* max_records) {
  *size = 1024 * 1024; *max_records = 0;
  *buffer = static_cast<uint8_t*>(std::malloc(*size));
  if (!*buffer) { *size = 0; check(CUPTI_ERROR_OUT_OF_MEMORY, "buffer_allocate"); }
}
void CUPTIAPI complete(CUcontext ctx, uint32_t stream, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* record = nullptr;
  for (;;) {
    auto result = cuptiActivityGetNextRecord(buffer, valid, &record);
    if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
    if (!check(result, "next_record")) break;
    std::ostringstream o;
    if (record->kind == CUPTI_ACTIVITY_KIND_MEMORY2) {
      auto* m = reinterpret_cast<CUpti_ActivityMemory4*>(record);
      o << "{\"timestamp_ns\":" << m->timestamp << ",\"process_id\":" << m->processId
        << ",\"device_id\":" << m->deviceId << ",\"context_id\":" << m->contextId
        << ",\"stream_id\":" << m->streamId << ",\"address\":" << m->address
        << ",\"bytes\":" << m->bytes << ",\"correlation_id\":" << m->correlationId
        << ",\"operation\":" << quote(m->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION ? "allocate" :
             m->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE ? "free" : "unsupported")
        << ",\"device_memory\":" << (m->memoryKind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE ? "true" : "false")
        << ",\"memory_kind\":" << m->memoryKind << ",\"async\":" << (m->isAsync ? "true" : "false")
        << ",\"pool_type\":" << m->memoryPoolConfig.memoryPoolType
        << ",\"source\":" << quote(m->source) << "}";
      std::lock_guard<std::mutex> lock(mu); events.push_back(o.str());
    } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME || record->kind == CUPTI_ACTIVITY_KIND_DRIVER) {
      auto* a = reinterpret_cast<CUpti_ActivityAPI*>(record);
      const char* name = nullptr;
      check(cuptiGetCallbackName(record->kind == CUPTI_ACTIVITY_KIND_RUNTIME ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API,
                                a->cbid, &name), "callback_name");
      o << "{\"name\":" << quote(name) << ",\"start_ns\":" << a->start << ",\"end_ns\":" << a->end
        << ",\"process_id\":" << a->processId << ",\"thread_id\":" << a->threadId
        << ",\"correlation_id\":" << a->correlationId << ",\"return_value\":" << a->returnValue << "}";
      std::lock_guard<std::mutex> lock(mu); apis.push_back(o.str());
    } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMORY_POOL) {
      std::lock_guard<std::mutex> lock(mu); ++pools;
    } else { check(CUPTI_ERROR_INVALID_KIND, "unexpected_record"); }
  }
  dropped(ctx, stream);
  { std::lock_guard<std::mutex> lock(mu); ++buffers; }
  std::free(buffer);
}
void array(std::ofstream& out, const char* name, const std::vector<std::string>& rows) {
  out << ',' << quote(name) << ":[";
  for (size_t i = 0; i < rows.size(); ++i) { if (i) out << ','; out << rows[i]; }
  out << ']';
}
}
extern "C" int tessera_memory_start() {
  if (started) return -1;
  started = true;
  check(cuptiGetVersion(&version), "version");
  check(cuptiGetTimestamp(&begin_ns), "start_timestamp");
  configured(cuptiActivityRegisterCallbacks(request, complete), "register_callbacks");
  configured(cuptiActivityEnableAllocationSource(1), "allocation_source");
  configured(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2), "enable_memory2");
  configured(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY_POOL), "enable_memory_pool");
  configured(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME), "enable_runtime");
  configured(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER), "enable_driver");
  const auto subscription = cuptiSubscribe(&subscriber, argument_callback, nullptr);
  configured(subscription, "subscribe_arguments");
  subscribed = subscription == CUPTI_SUCCESS;
  if (subscribed) for (auto cbid : argument_callbacks) {
    const std::string name = "enable_argument_callback_" + std::to_string(cbid);
    configured(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, cbid), name.c_str());
  }
  std::lock_guard<std::mutex> lock(mu);
  return errors.empty() ? 0 : -2;
}
extern "C" uint64_t tessera_memory_mark(const char* label) {
  uint64_t ns = 0;
  if (!started || stopped) return 0;
  check(cuptiGetTimestamp(&ns), "marker_timestamp");
  std::lock_guard<std::mutex> lock(mu);
  markers.push_back("{\"name\":" + quote(label) + ",\"timestamp_ns\":" + std::to_string(ns) + "}");
  return ns;
}
extern "C" int tessera_memory_stop(const char* path) {
  if (!started || stopped) return -1;
  configured(cuptiActivityFlushAll(0), "flush_before_disable");
  for (auto kind : {CUPTI_ACTIVITY_KIND_MEMORY2, CUPTI_ACTIVITY_KIND_MEMORY_POOL,
                    CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER})
    check(cuptiActivityDisable(kind), "disable_activity");
  configured(cuptiActivityFlushAll(0), "flush_after_disable");
  if (subscribed) {
    configured(cuptiUnsubscribe(subscriber), "unsubscribe_arguments");
    subscribed = false;
  }
  dropped(nullptr, 0);
  check(cuptiGetTimestamp(&end_ns), "stop_timestamp");
  stopped = true;
  std::lock_guard<std::mutex> lock(mu);
  std::ofstream out(path);
  out << "{\"schema\":\"tessera.cupti_memory_trace.v1\",\"process_id\":" << getpid()
      << ",\"cupti_version\":" << version << ",\"start_ns\":" << begin_ns << ",\"end_ns\":" << end_ns
      << ",\"completed_buffers\":" << buffers << ",\"pool_records\":" << pools;
  array(out, "memory_events", events); array(out, "api_events", apis); array(out, "markers", markers);
  out << ",\"argument_schema\":\"tessera.cuda_memory_api_arguments.v1\"";
  array(out, "api_argument_events", arguments);
  array(out, "dropped_records", drops); array(out, "errors", errors);
  array(out, "configuration", configuration);
  out << "}\n"; out.close();
  return out.fail() ? -3 : errors.empty() ? 0 : -2;
}
