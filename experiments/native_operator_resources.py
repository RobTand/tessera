"""CUPTI allocation evidence for one warmed eager operator, not engine memory.

The bound sums independent Torch live-allocation and external CUDA peaks. It
includes the returned output, so adding input/output activation bytes later is
conservative. Context/library startup, model fixed resources, KV, allocator
reservation slack and graph pools require the separate full-engine receipt.
Unsupported allocation domains or incomplete collection never become zero.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re

TRACE_SCHEMA = "tessera.cupti_memory_trace.v1"


class NativeMemoryCollector:
    """Start in a fresh subprocess before importing Torch or CUDA libraries.

    Build the library with CUDA 13 headers and libcupti; its hash belongs in
    the pinned runtime manifest. Do not run alongside another CUPTI profiler.
    The caller synchronizes the device before markers and before finish().
    """

    def __init__(self, library):
        path = Path(library).resolve(strict=True)
        maps = Path("/proc/self/maps").read_text()
        if re.search(r"/(?:libcuda\.so|libcudart|libtorch_cuda)", maps):
            raise RuntimeError("collector must start before CUDA libraries are loaded")
        self.library_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self._lib = ctypes.CDLL(str(path))
        self._lib.tessera_memory_start.restype = ctypes.c_int
        self._lib.tessera_memory_mark.argtypes = [ctypes.c_char_p]
        self._lib.tessera_memory_mark.restype = ctypes.c_uint64
        self._lib.tessera_memory_stop.argtypes = [ctypes.c_char_p]
        self._lib.tessera_memory_stop.restype = ctypes.c_int
        self.start_code = self._lib.tessera_memory_start()
        self._finished = False

    def mark(self, name):
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("marker requires a nonempty NUL-free string")
        timestamp = self._lib.tessera_memory_mark(name.encode())
        if not timestamp:
            raise RuntimeError("CUPTI marker failed or collector is closed")
        return timestamp

    def finish(self, path):
        if self._finished:
            raise RuntimeError("collector is already closed")
        path = Path(path)
        stop_code = self._lib.tessera_memory_stop(os.fsencode(path))
        self._finished = True
        raw = json.loads(path.read_text())
        raw["capture"] = {"started_before_cuda_libraries": True,
                          "collector_library_sha256": self.library_sha256,
                          "start_code": self.start_code, "stop_code": stop_code}
        path.write_text(json.dumps(raw, sort_keys=True) + "\n")
        return raw

    def observe_apply(self, apply, name, *, device=0):
        """Collect Torch allocation counters around one synchronized invocation."""
        import torch
        if self.start_code != 0:
            raise RuntimeError("CUPTI collection initialization failed")
        driver = ctypes.CDLL("libcuda.so.1")
        context = ctypes.c_void_p()
        if driver.cuCtxGetCurrent(ctypes.byref(context)) != 0 or not context.value:
            raise RuntimeError("no current CUDA context")
        cupti = ctypes.CDLL("libcupti.so")
        get_id = cupti.cuptiGetContextId
        get_id.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        context_id = ctypes.c_uint32()
        if get_id(context, ctypes.byref(context_id)) != 0:
            raise RuntimeError("CUPTI context identity unavailable")
        def segments():
            return [{"address": r["address"], "total_size": r["total_size"]}
                    for r in torch.cuda.memory_snapshot() if r["device"] == device]
        torch.cuda.synchronize(device)
        before = segments()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        self.mark(name + ":begin")
        output = apply()
        torch.cuda.synchronize(device)
        self.mark(name + ":end")
        observation = {"device_id": device, "context_id": context_id.value,
                       "allocator_backend": torch.cuda.get_allocator_backend(),
                       "before_allocated_bytes": baseline,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "before_segments": before, "after_segments": segments()}
        return output, observation


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid nonnegative integer in resource evidence")
    return value


def _segments(observation, name):
    rows = observation[name]
    if not isinstance(rows, list):
        raise ValueError("allocator segments must be an explicit list")
    result = sorted((_integer(r["address"], 1), _integer(r["total_size"], 1)) for r in rows)
    if any(a + n > b for (a, n), (b, _) in zip(result, result[1:])):
        raise ValueError("allocator segments overlap")
    return result


def analyze_trace(trace, *, interval, torch_observation):
    """Validate actual collector records and derive a conservative byte bound.

    ``torch_observation``: device_id, context_id, allocator_backend="native",
    before_allocated_bytes, peak_allocated_bytes, before_segments, after_segments.
    Segment rows use Torch memory_snapshot's address/total_size fields. The
    caller takes snapshots and resets the peak before ``interval:begin``;
    synchronizes apply before ``interval:end`` and then reads the peak/snapshot.
    All snapshots must concern the same device and this exact invocation.

    v1 requires a warmed stable allocator: any CUDA allocation/free touching
    its segments during apply refuses composition, preventing pointer-reuse
    ambiguities when separating external memory from Torch suballocations.
    Preexisting DEVICE_STATIC records have a separate startup ledger. CUDA 13
    can report those with raw context/correlation/stream zero; this is observed
    metadata, not an assertion that zero is CUPTI_INVALID_CONTEXT_ID or that
    the storage belongs to the operator's context. Static changes during apply
    still refuse the bound, and startup bytes never substitute for full-model
    fixed-resource evidence.
    """
    result = {"schema": "tessera.native_operator_resource_bound.v1", "status": "incomplete",
              "scope": "warmed_eager_operator_incremental_live_allocations",
              "composition": "sum_of_independent_peaks_including_output",
              "peak_scratch_bytes": None, "external_native_peak_bytes": None,
              "torch_peak_increment_bytes": None, "reasons": [],
              "startup_static": None,
              "full_model_fixed_resources_complete": False}
    reasons = result["reasons"]
    try:
        if trace["schema"] != TRACE_SCHEMA:
            raise ValueError("unsupported CUPTI trace schema")
        capture = trace["capture"]
        if (capture["started_before_cuda_libraries"] is not True
                or type(capture["start_code"]) is not int or capture["start_code"] != 0
                or type(capture["stop_code"]) is not int or capture["stop_code"] != 0
                or not re.fullmatch("[0-9a-f]{64}", capture["collector_library_sha256"])):
            raise ValueError("capture bootstrap/completion evidence failed")
        if trace["errors"] or _integer(trace["pool_records"]) != 0:
            raise ValueError("CUPTI errors or unsupported pool records")
        expected_configuration = {"register_callbacks", "allocation_source", "enable_memory2",
                                  "enable_memory_pool", "enable_runtime", "enable_driver",
                                  "flush_before_disable", "flush_after_disable"}
        configuration = trace["configuration"]
        if (len(configuration) != len(expected_configuration)
                or {r["operation"] for r in configuration} != expected_configuration
                or any(_integer(r["code"]) != 0 for r in configuration)):
            raise ValueError("missing successful CUPTI configuration/flush evidence")
        if _integer(trace["completed_buffers"], 1) + 1 != len(trace["dropped_records"]):
            raise ValueError("missing per-buffer/final dropped-record observation")
        if any(_integer(r["code"]) != 0 or _integer(r["count"]) != 0 for r in trace["dropped_records"]):
            raise ValueError("CUPTI dropped records or failed dropped-record query")
        pid = _integer(trace["process_id"], 1)
        starts = [r["timestamp_ns"] for r in trace["markers"] if r["name"] == interval + ":begin"]
        ends = [r["timestamp_ns"] for r in trace["markers"] if r["name"] == interval + ":end"]
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError("exactly one paired apply interval required")
        begin, end = _integer(starts[0], 1), _integer(ends[0], 1)
        if not _integer(trace["start_ns"], 1) < begin < end < _integer(trace["end_ns"], 1):
            raise ValueError("apply interval is outside collection")
        if torch_observation["allocator_backend"] != "native":
            raise ValueError("only the native Torch caching allocator is supported")
        device = _integer(torch_observation["device_id"])
        context = _integer(torch_observation["context_id"])
        segments = _segments(torch_observation, "before_segments")
        if segments != _segments(torch_observation, "after_segments"):
            raise ValueError("allocator reservations changed; warmup is incomplete")
        base = _integer(torch_observation["before_allocated_bytes"])
        peak = _integer(torch_observation["peak_allocated_bytes"])
        if peak < base:
            raise ValueError("Torch peak precedes baseline; counter interval mismatch")
        # Callback names are emitted by CUPTI itself. Versions identify the
        # ABI; strip only that suffix when comparing documented CUDA APIs.
        supported = {"cudaMalloc", "cudaFree", "cuMemAlloc", "cuMemFree"}
        ownership = re.compile(
            r"^(cudaMalloc|cudaFree|cudaHost|cudaMemPool|cudaImportExternalMemory|"
            r"cudaDestroyExternalMemory|cudaExternalMemory|cudaIpcOpenMemHandle|"
            r"cuMemAlloc|cuMemFree|cuMemHost|cuMemPool|cuMemCreate|cuMemMap|"
            r"cuMemUnmap|cuMemRelease|cuMemImport|cuMemAddress|cuImportExternalMemory|"
            r"cuDestroyExternalMemory|cuExternalMemory|cuIpcOpenMemHandle|cuArray|cuMipmappedArray)")
        if not trace["api_events"]:
            raise ValueError("no observed CUDA API records")
        by_correlation, required_operations = {}, {}
        for api in trace["api_events"]:
            name = re.sub(r"_v\d+$", "", api["name"])
            key = (_integer(api["process_id"], 1), _integer(api["correlation_id"]))
            api_start, api_end = _integer(api["start_ns"], 1), _integer(api["end_ns"], 1)
            if api_end < api_start:
                raise ValueError("CUDA API timestamps are reversed")
            by_correlation.setdefault(key, []).append((name, api_start, api_end))
            overlaps = api_start <= end and api_end >= begin
            if overlaps and ownership.match(name):
                if name not in supported or _integer(api["return_value"]) != 0:
                    raise ValueError("unsupported or failed allocation API: " + name)
                if key[0] != pid or not begin <= api_start <= api_end <= end:
                    raise ValueError("allocation API crosses the apply process/interval")
                if key in required_operations:
                    raise ValueError("ambiguous allocation API correlation")
                required_operations[key] = ("allocate" if name in {"cudaMalloc", "cuMemAlloc"} else "free")
            if overlaps and name.startswith(("cudaGraph", "cuGraph")):
                raise ValueError("CUDA graph execution is outside eager resource scope")
        events = sorted(trace["memory_events"], key=lambda r: r["timestamp_ns"])
        if not events:
            raise ValueError("no observed allocation records; collection not demonstrated")
        live, new_live, static_live, observed_operations = {}, {}, {}, set()
        external_peak = 0
        for row in events:
            t = _integer(row["timestamp_ns"], 1)
            if t > end:
                break
            if row["process_id"] != pid:
                raise ValueError("allocation belongs to another process")
            # Pageable/pinned host allocations are outside device-byte scope.
            if row["memory_kind"] in (1, 2):
                continue
            if row["memory_kind"] not in (3, 6) or row["async"] is not False or row["pool_type"] != 0:
                raise ValueError("unsupported managed/async/pool/unknown memory domain")
            address, size = _integer(row["address"], 1), _integer(row["bytes"], 1)
            inside = begin <= t <= end
            if row["memory_kind"] == 6:
                if inside:
                    raise ValueError("module/static allocation or free occurred during apply")
                # DEVICE_STATIC is a distinct CUPTI memory domain. Preserve
                # the raw coordinates observed on the qualified CUDA 13
                # runtime; never grant context-zero tolerance to DEVICE rows.
                raw_context = _integer(row["context_id"])
                known_static_coordinates = (trace["cupti_version"] == 130001
                                            and raw_context == 0
                                            and row["correlation_id"] == 0
                                            and row["stream_id"] == 0)
                if row["device_id"] != device or not (raw_context == context or known_static_coordinates):
                    raise ValueError("startup static allocation has unsupported device/context coordinates")
                key = (device, raw_context, address)
                if row["operation"] == "allocate":
                    if not isinstance(row["source"], str) or not row["source"]:
                        raise ValueError("startup static allocation source is unknown")
                    if key in static_live:
                        raise ValueError("duplicate live startup static allocation")
                    static_live[key] = (size, row["source"])
                elif row["operation"] == "free":
                    allocation = static_live.pop(key, None)
                    if allocation is None or allocation[0] != size:
                        raise ValueError("static free lacks matching startup allocation bytes/context")
                else:
                    raise ValueError("unknown startup static memory operation")
                continue
            if row["device_id"] != device or row["context_id"] != context:
                raise ValueError("allocation device/context differs from operator")
            key = (device, context, address)
            if inside:
                correlation = (pid, _integer(row["correlation_id"]))
                apis = by_correlation.get(correlation, [])
                if len(apis) != 1 or correlation not in required_operations:
                    raise ValueError("memory operation has no unambiguous in-apply API correlation")
                _, api_start, api_end = apis[0]
                if not api_start <= t <= api_end:
                    raise ValueError("memory operation timestamp is outside its API")
                if row["operation"] != required_operations[correlation]:
                    raise ValueError("allocation operation disagrees with its API")
                if correlation in observed_operations:
                    raise ValueError("multiple memory operations share an API correlation")
                observed_operations.add(correlation)
                if any(address < a + n and a < address + size for a, n in segments):
                    raise ValueError("allocator segment changed during apply")
            if row["operation"] == "allocate":
                if key in live:
                    raise ValueError("duplicate live allocation address")
                live[key] = size
                if inside:
                    new_live[key] = size
                    external_peak = max(external_peak, sum(new_live.values()))
            elif row["operation"] == "free":
                if live.pop(key, None) != size:
                    raise ValueError("free lacks matching allocation bytes/context")
                if inside:
                    if key not in new_live:
                        raise ValueError("preexisting allocation freed during apply")
                    del new_live[key]
            else:
                raise ValueError("unknown memory operation")
        # Coverage is reciprocal: API records without their MEMORY2 rows may
        # hide an entire transient lifetime even when startup memory, enables
        # and dropped-record queries look valid. Such gaps must remain unknown.
        if observed_operations != set(required_operations):
            raise ValueError("allocation API lacks its matching memory operation")
        if new_live:
            raise ValueError("native allocation retained after apply; fixed ownership unresolved")
        startup_sources = {}
        for size, source in static_live.values():
            entry = startup_sources.setdefault(source, {"source": source, "live_bytes": 0,
                                                        "live_allocation_count": 0})
            entry["live_bytes"] += size
            entry["live_allocation_count"] += 1
        startup = {"scope": "observed_live_device_static_before_apply",
                   "live_bytes": sum(size for size, _ in static_live.values()),
                   "live_allocation_count": len(static_live),
                   "raw_context_ids": sorted({key[1] for key in static_live}),
                   "sources": [startup_sources[source] for source in sorted(startup_sources)],
                   "operator_context_attributed": False,
                   "full_model_fixed_resources_complete": False}
        result.update(status="complete_operator_bound", peak_scratch_bytes=peak - base + external_peak,
                      external_native_peak_bytes=external_peak, torch_peak_increment_bytes=peak - base,
                      startup_static=startup,
                      interval={"name": interval, "begin_ns": begin, "end_ns": end},
                      process_id=pid, device_id=device, context_id=context)
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append(str(exc))
    return result


def _qualify(library, output):
    """Tiny real observer regression, invoked only inside an admitted PB GPU job."""
    collector = NativeMemoryCollector(library)
    import torch
    torch.cuda.init()
    warm = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    del warm
    torch.cuda.synchronize()
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    expected = 123456
    def apply():
        tensor = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
        pointer = ctypes.c_void_p()
        if cudart.cudaMalloc(ctypes.byref(pointer), expected) != 0:
            raise RuntimeError("qualification cudaMalloc failed")
        if cudart.cudaFree(pointer) != 0:
            raise RuntimeError("qualification cudaFree failed")
        return tensor
    tensor, observation = collector.observe_apply(apply, "qualification")
    trace = collector.finish(output)
    result = analyze_trace(trace, interval="qualification", torch_observation=observation)
    payload = {"trace": trace, "torch_observation": observation, "result": result,
               "expected_external_bytes": expected, "torch_version": torch.__version__}
    Path(str(output) + ".qualification.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    if (result["status"] != "complete_operator_bound"
            or result["external_native_peak_bytes"] != expected
            or result["torch_peak_increment_bytes"] != tensor.untyped_storage().nbytes()):
        raise RuntimeError("real allocation observer qualification failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--qualification-output", required=True)
    args = parser.parse_args()
    _qualify(args.library, args.qualification_output)
