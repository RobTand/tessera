"""Raw same-run native-unit/event partitions, never planner timing admission.

CUDA events measure each adjacent fixed gap directly. A separate observer
stream joins the main completion event and the stock asynchronous copy event;
the engine's streams never wait on the observer. The profiler's launch and
stream evidence is checked independently. Missing evidence leaves prices null.
"""
from collections import Counter
from contextlib import contextmanager
import ctypes
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from experiments.full_engine_timing_boundaries import observe_apply_boundaries, observe_tp2_owner_boundaries


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("timing observation must be finite and nonnegative")
    return value


#: The per-arm raw records never price. The same-run timing observation
#: (``full_engine_timing_observation``) derives the partition's terms from the
#: arms under its own qualification, and the resource report carries them as
#: ``derived.timing_terms`` once ``timing_partition`` closes on it.
PRICING_SCOPE = ("the raw timing arm records events and their partition; timing terms and their "
                 "admission are derived by full_engine_timing_observation from every arm of "
                 "the run under its qualification, and carried by the resource report")


def analyze_profile_partition(capture, profile):
    """Recompute launch ownership and explicit stream coverage from trace bytes.

This does not certify Kineto's collection completeness or observer overhead.
No kernel-duration sum is subtracted from CUDA-event elapsed time.
    """
    result = {"schema": ("tessera.full_engine_timing_partition.v2"
                         if capture.get("schema") == "tessera.full_engine_timing_capture.v2"
                         else "tessera.full_engine_timing_partition.v1"), "status": "incomplete",
              "timings": None, "admission": None, "pricing_scope": PRICING_SCOPE, "issues": [], "steps": [],
              "qualification_gaps": ["profiler collection health qualification", "observer overhead qualification",
                                     "bound reference assignment and runtime admission", "full-engine resource closure"]}
    try:
        composite = capture["schema"] == "tessera.full_engine_timing_capture.v2"
        if capture["schema"] not in {"tessera.full_engine_timing_capture.v1",
                                     "tessera.full_engine_timing_capture.v2"}:
            raise ValueError("unsupported timing capture schema")
        if capture["arm"] != "partition":
            raise ValueError("control arm has no candidate-unit partition")
        if capture["errors"]:
            raise ValueError("capture errors: " + repr(capture["errors"]))
        expected = capture["canonical_unit_ids"]
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("canonical unit roster is empty or duplicated")
        ranges = {name: [] for name in capture["ranges"]}
        launches, gpu = {}, []
        for event in profile["traceEvents"]:
            if (event.get("ph") == "X" and event.get("cat") == "user_annotation"
                    and event.get("name") in ranges):
                ranges[event["name"]].append(event)
            category = event.get("cat", "")
            if category in {"cuda_runtime", "cuda_driver"} and "correlation" in event.get("args", {}):
                launches.setdefault(event["args"]["correlation"], []).append(event)
            if category in {"kernel", "gpu_memcpy", "gpu_memset"}:
                gpu.append(event)
        if not gpu or any(len(rows) != 1 for rows in ranges.values()):
            raise ValueError("missing GPU operations or unique CPU observation ranges")
        by_step = {}
        for step in capture["steps"]:
            sid = step["step_id"]
            if sid in by_step:
                raise ValueError("duplicate worker step")
            unit_ids = [row["unit_id"] for row in step["units"]]
            if Counter(unit_ids) != Counter(expected):
                raise ValueError("worker step does not execute every canonical native unit exactly once")
            if step["phase"] not in {"prefill", "decode"} or step["scheduled_tokens"] != (512 if step["phase"] == "prefill" else 1):
                raise ValueError("step differs from canonical 512-token prefill/one-token decode")
            if step["copy_event_observed"] is not True or step["completion_join"] != "main_event_and_stock_copy_event":
                raise ValueError("asynchronous engine output has no explicit event join")
            segments = step.get("segments") if composite else step["units"]
            if composite:
                expected_segments = [(unit, role) for unit in expected
                                     for role in (("apply", "final_reduce") if unit.startswith("s:") else ("apply",))]
                observed_segments = [(segment.get("unit_id"), segment.get("role")) for segment in segments]
                if Counter(observed_segments) != Counter(expected_segments):
                    raise ValueError("TP2 owner segments omit or duplicate an apply or final reduction")
                pending = None
                for segment in segments:
                    unit_id, role = segment["unit_id"], segment["role"]
                    if role == "apply":
                        if pending is not None:
                            raise ValueError("TP2 final reduction follows another native apply")
                        pending = unit_id if unit_id.startswith("s:") else None
                    elif role == "final_reduce":
                        if pending != unit_id:
                            raise ValueError("TP2 final reduction is outside its routed owner")
                        pending = None
                    name = segment["range_name"]
                    if capture["ranges"].get(name) != {"kind": "owner_segment", "step_id": sid,
                                                       "unit_id": unit_id, "role": role}:
                        raise ValueError("TP2 owner segment range does not bind its role")
                    if role == "final_reduce":
                        reduction = segment.get("reduction") or {}
                        calls = reduction.get("collective_calls")
                        if (reduction.get("tp_size") != 2
                                or reduction.get("skip_final_all_reduce") is not False
                                or reduction.get("is_sequence_parallel") is not False
                                or reduction.get("effective_output_is_reduced") is not False
                                or reduction.get("trunc_size") is not None
                                or type(calls) is not list or len(calls) != 1
                                or calls[0].get("input_shape") != reduction.get("input_shape")
                                or calls[0].get("input_dtype") != reduction.get("input_dtype")
                                or reduction.get("output_shape") != reduction.get("input_shape")
                                or reduction.get("output_dtype") != reduction.get("input_dtype")):
                            raise ValueError("TP2 final reduction differs from the native owner collective")
                if pending is not None:
                    raise ValueError("TP2 routed apply has no final reduction")
                previous_end = None
                step_names = [name for name, description in capture["ranges"].items()
                              if description.get("kind") == "step" and description.get("step_id") == sid]
                if len(step_names) != 1:
                    raise ValueError("TP2 owner has no unique raw engine step range")
                step_span = ranges[step_names[0]][0]
                for segment in segments:
                    occurrences = ranges.get(segment["range_name"], [])
                    if len(occurrences) != 1:
                        raise ValueError("TP2 owner segment lacks one raw CPU range")
                    span = occurrences[0]
                    if previous_end is not None and span["ts"] < previous_end:
                        raise ValueError("TP2 owner segment CPU ranges overlap or reorder")
                    if (span["ts"] < step_span["ts"]
                            or span["ts"] + span["dur"] > step_span["ts"] + step_span["dur"]):
                        raise ValueError("TP2 owner segment lies outside its engine step")
                    previous_end = span["ts"] + span["dur"]
                if [segment["range_name"] for segment in segments] != list(dict.fromkeys(
                        segment["range_name"] for segment in segments)):
                    raise ValueError("TP2 owner segment ranges alias")
                by_unit = {unit["unit_id"]: unit for unit in step["units"]}
                for unit in expected:
                    parts = [segment["elapsed_ms"] for segment in segments if segment["unit_id"] == unit]
                    if not math.isclose(math.fsum(parts), by_unit[unit]["elapsed_ms"], rel_tol=2**-22, abs_tol=0):
                        raise ValueError("TP2 owner elapsed time disagrees with its segments")
            if len(step["fixed_gaps_ms"]) != len(segments) + 1:
                raise ValueError("adjacent fixed gap coverage is incomplete")
            components = [*step["fixed_gaps_ms"], *[row["elapsed_ms"] for row in segments]]
            for value in [*components, step["whole_step_ms"]]:
                _finite(value)
            # CUDA elapsed_time returns binary32 measurements, not exact real
            # numbers. Bound only their accumulated representation rounding;
            # this is not an acceptance allowance for unobserved GPU work.
            rounding_bound = math.fsum(math.ulp(float(value)) + abs(value) * 2**-23 for value in components)
            if abs(math.fsum(components) - step["whole_step_ms"]) > rounding_bound + abs(step["whole_step_ms"]) * 2**-23:
                raise ValueError("same-event interval partition does not recompose")
            row = {"step_id": sid, "phase": step["phase"], "whole_step_ms": step["whole_step_ms"],
                   "fixed_gap_samples_ms": step["fixed_gaps_ms"], "fixed_gap_sum_ms": math.fsum(step["fixed_gaps_ms"]),
                   "units": step["units"], "gpu_operations": [], "observed_stream_ids": []}
            by_step[sid] = (step, row)
            if composite:
                row["segments"] = segments
            result["steps"].append(row)
        if [step["phase"] for step in result["steps"]] != ["prefill", "decode"]:
            raise ValueError("timing workload must contain exactly prefill then decode")
        for operation in gpu:
            args = operation.get("args", {})
            paired = launches.get(args.get("correlation"), [])
            if len(paired) != 1:
                raise ValueError("GPU operation has no unique runtime/driver launch")
            launch = paired[0]
            matched = []
            for name, rows in ranges.items():
                span = rows[0]
                if (launch.get("pid") == span.get("pid") and launch.get("tid") == span.get("tid")
                        and span["ts"] <= launch["ts"] <= launch["ts"] + launch["dur"] <= span["ts"] + span["dur"]):
                    matched.append(capture["ranges"][name])
            step_ranges = [row for row in matched if row["kind"] == "step"]
            unit_ranges = [row for row in matched if row["kind"] in ({"unit", "owner_segment"} if composite else {"unit"})]
            if len(step_ranges) != 1 or len(unit_ranges) > 1:
                raise ValueError("GPU launch lies outside a unique step or inside overlapping units")
            sid = step_ranges[0]["step_id"]
            step, row = by_step[sid]
            if any(unit["step_id"] != sid for unit in unit_ranges):
                raise ValueError("native launch scope belongs to another worker step")
            stream = args.get("stream")
            if type(stream) is not int or args.get("device") != capture["device_id"]:
                raise ValueError("GPU operation has missing stream/device evidence")
            unit_id = unit_ranges[0]["unit_id"] if unit_ranges else None
            if stream not in (step["main_stream_id"], step["copy_stream_id"]):
                raise ValueError("GPU operation uses an unjoined stream")
            if unit_id is not None and stream != step["main_stream_id"]:
                raise ValueError("native unit has GPU work outside its measured stream")
            if stream == step["copy_stream_id"]:
                tail_segment = step["segments"][-1] if composite else step["units"][-1]
                tail = ranges[tail_segment["range_name"]][0]
                if launch["ts"] < tail["ts"] + tail["dur"]:
                    raise ValueError("copy-stream work overlaps native units; sequential gaps do not cover it")
            row["gpu_operations"].append({"name": operation["name"], "category": operation["cat"],
                "correlation": args["correlation"], "stream": stream, "unit_id": unit_id,
                "timestamp_us": operation["ts"], "duration_us": operation["dur"]}
                | ({"role": unit_ranges[0].get("role") if unit_ranges else None} if composite else {}))
        for _, row in by_step.values():
            observed_units = {op["unit_id"] for op in row["gpu_operations"] if op["unit_id"] is not None}
            if observed_units != set(expected):
                raise ValueError("canonical unit has no attributed GPU operation")
            if composite:
                # A TP2 final collective cannot be credited merely because a
                # Python hook was called. Its trace must contain GPU work in
                # the exact second owner span, on the joined main stream.
                reduced = {op["unit_id"] for op in row["gpu_operations"]
                           if op["role"] == "final_reduce"}
                if reduced != {unit for unit in expected if unit.startswith("s:")}:
                    raise ValueError("TP2 final reduction has no attributed GPU operation")
            row["observed_stream_ids"] = sorted({op["stream"] for op in row["gpu_operations"]})
        result["status"] = "observed_same_run_partition"
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        result["issues"].append(str(exc))
    return result


def _stream_id(stream):
    from experiments.bench_native_operator import _mapped_shared_libraries
    paths = [path for path in _mapped_shared_libraries() if path.name.startswith("libcupti.so")]
    if len(paths) != 1:
        raise RuntimeError("timing profiler must load one verifiable CUPTI library")
    library = ctypes.CDLL(str(paths[0]))
    get_id = library.cuptiGetStreamIdEx
    get_id.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint32)]
    get_id.restype = ctypes.c_int
    value = ctypes.c_uint32()
    handle = stream.cuda_stream
    # CUPTI cannot resolve CUDA's null/default stream without its context.
    # Query the active driver context and let CUPTI verify stream membership.
    driver = ctypes.CDLL("libcuda.so.1")
    get_context = driver.cuCtxGetCurrent
    get_context.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    get_context.restype = ctypes.c_int
    context = ctypes.c_void_p()
    code = get_context(ctypes.byref(context))
    if code or not context.value:
        raise RuntimeError(f"active CUDA context lookup failed: {code}")
    code = get_id(context, ctypes.c_void_p(handle), int(handle == 2), ctypes.byref(value))
    if code:
        raise RuntimeError(f"CUPTI stream identity lookup failed: {code}")
    return value.value


def _profiler_drop_witness(stream_ids):
    """Read post-stop counters for diagnosis, never collection qualification.

    CUPTI resets each queue's count on read; pinned Kineto can read inside a
    buffer callback in verbose mode. These post-stop values omit the global
    queue and other streams, including NCCL. Even all zero is not an owned
    cumulative loss witness. No consumer may promote this v1 diagnostic.
    """
    try:
        from experiments.bench_native_operator import _mapped_shared_libraries
        paths = [path for path in _mapped_shared_libraries() if path.name.startswith("libcupti.so")]
        if len(paths) != 1:
            raise RuntimeError("CUPTI library identity is ambiguous")
        library = ctypes.CDLL(str(paths[0]))
        driver = ctypes.CDLL("libcuda.so.1")
        context = ctypes.c_void_p()
        get_context = driver.cuCtxGetCurrent
        get_context.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        get_context.restype = ctypes.c_int
        code = get_context(ctypes.byref(context))
        if code or not context.value:
            raise RuntimeError(f"CUDA context lookup failed: {code}")
        dropped = library.cuptiActivityGetNumDroppedRecords
        dropped.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t)]
        dropped.restype = ctypes.c_int
        rows = []
        for stream_id in sorted(stream_ids):
            count = ctypes.c_size_t()
            code = dropped(context, stream_id, ctypes.byref(count))
            if code:
                raise RuntimeError(f"CUPTI dropped-record query failed for stream {stream_id}: {code}")
            rows.append({"stream_id": stream_id, "dropped_records": count.value})
        return {"schema": "tessera.cupti_drop_diagnostic.v1", "available": False,
                "reason": "post-stop CUPTI counts reset on read; Kineto callback consumption and global/NCCL queues are unobserved",
                "diagnostic_end_counters": rows,
                "counter_semantics": "reset_on_read",
                "scope": "diagnostic main/copy/join streams only; no cumulative profiler collection-health claim"}
    except Exception as exc:  # noqa: BLE001 -- absence is evidence, not success
        return {"schema": "tessera.cupti_drop_diagnostic.v1", "available": False,
                "reason": f"{type(exc).__name__}: {exc}", "diagnostic_end_counters": None,
                "counter_semantics": "reset_on_read"}


class FullEngineTimingRecorder:
    def __init__(self, boundaries, *, arm, output, device_id=0, tp2_composite=False):
        import torch
        if arm not in {"control", "partition"}:
            raise ValueError("unknown timing observation arm")
        self.torch, self.boundaries, self.arm = torch, boundaries, arm
        self.tp2_composite = tp2_composite
        self.started_unix_ns = time.time_ns()
        self.output, self.device = Path(output), device_id
        self.output.mkdir(parents=True, exist_ok=False)
        self.steps, self.ranges, self.errors, self.current = [], {}, [], None
        self.join_stream = torch.cuda.Stream(device=device_id)
        self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        self.profiler.start()
        self.patches = (observe_tp2_owner_boundaries(boundaries, self.unit, self.final_reduce,
                                                     self.output_collective)
                        if tp2_composite else observe_apply_boundaries(boundaries, self.unit)) if arm == "partition" else None
        if self.patches:
            self.patches.__enter__()

    def event(self, stream):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    @contextmanager
    def housekeeping(self, scheduler):
        """Retain stock request cleanup; any GPU launch here still refuses."""
        if self.current is not None or scheduler.total_num_scheduled_tokens != 0:
            raise RuntimeError("housekeeping overlaps a timed step or schedules tokens")
        name = "tessera.engine.housekeeping." + str(len(self.ranges))
        self.ranges[name] = {"kind": "housekeeping", "scheduled_tokens": 0,
                             "finished_request_ids": sorted(scheduler.finished_req_ids)}
        with self.torch.profiler.record_function(name):
            yield

    def begin_step(self, scheduler, main_stream, copy_stream):
        if self.current is not None or len(self.steps) >= 2:
            raise RuntimeError("overlapping or excess timed engine steps")
        count = scheduler.total_num_scheduled_tokens
        phase = "prefill" if not self.steps else "decode"
        if count != (512 if phase == "prefill" else 1):
            raise RuntimeError("timing step differs from the cold 512/1 workload")
        sid = str(len(self.steps))
        name = "tessera.engine.step." + sid
        self.ranges[name] = {"kind": "step", "step_id": sid}
        scope = self.torch.profiler.record_function(name)
        scope.__enter__()
        self.current = {"step_id": sid, "phase": phase, "scheduled_tokens": count,
                        "main_stream_id": _stream_id(main_stream), "copy_stream_id": _stream_id(copy_stream),
                        "main_stream": main_stream, "copy_stream": copy_stream,
                        "scope": scope, "thread_id": threading.get_ident(), "units": [], "segments": [],
                        "pending_reduce": None, "active_unit": None, "active_reduce": None,
                        "start_event": self.event(main_stream)}

    @contextmanager
    def unit(self, boundary, call):
        step = self.current
        if step is None:
            raise RuntimeError("canonical native unit ran outside an armed engine step")
        if (step["active_unit"] is not None or step["pending_reduce"] is not None
                or threading.get_ident() != step["thread_id"]):
            raise RuntimeError("native unit scopes overlap or cross worker threads")
        if self.torch.cuda.current_stream(self.device) != step["main_stream"]:
            raise RuntimeError("canonical native unit changed its execution stream")
        index = len(step["units"])
        name = f"tessera.engine.unit.{step['step_id']}.{index}"
        self.ranges[name] = {"kind": "owner_segment" if self.tp2_composite else "unit",
                             "step_id": step["step_id"], "unit_id": boundary["unit_id"],
                             "role": "apply"}
        step["active_unit"] = boundary["unit_id"]
        with self.torch.profiler.record_function(name):
            begin = self.event(step["main_stream"])
            try:
                yield
            finally:
                end = self.event(step["main_stream"])
                step["units"].append({"unit_id": boundary["unit_id"], "range_name": name,
                                      "begin_event": begin, "end_event": end})
                if self.tp2_composite:
                    step["segments"].append({"unit_id": boundary["unit_id"], "role": "apply",
                                              "range_name": name, "begin_event": begin, "end_event": end})
                    if boundary["unit_id"].startswith("s:"):
                        step["pending_reduce"] = boundary["unit_id"]
                step["active_unit"] = None

    @contextmanager
    def final_reduce(self, boundary, call):
        step = self.current
        unit_id = boundary["unit_id"]
        if (not self.tp2_composite or step is None or step["pending_reduce"] != unit_id
                or step["active_unit"] is not None or threading.get_ident() != step["thread_id"]):
            raise RuntimeError("TP2 final reduction is missing, duplicated or overlaps an owner")
        if self.torch.cuda.current_stream(self.device) != step["main_stream"]:
            raise RuntimeError("TP2 final reduction changed its execution stream")
        index = len(step["segments"])
        name = f"tessera.engine.owner_segment.{step['step_id']}.{index}.final_reduce"
        self.ranges[name] = {"kind": "owner_segment", "step_id": step["step_id"],
                             "unit_id": unit_id, "role": "final_reduce"}
        step["active_unit"] = unit_id
        calls = []
        step["active_reduce"] = calls
        with self.torch.profiler.record_function(name):
            begin = self.event(step["main_stream"])
            try:
                yield
            finally:
                end = self.event(step["main_stream"])
                args = call["arguments"]
                kwargs = call["keywords"]
                states = args[0] if args else kwargs.get("states")
                trunc_size = args[1] if len(args) > 1 else kwargs.get("trunc_size")
                output_is_reduced = (args[2] if len(args) > 2 else kwargs.get("output_is_reduced"))
                runner = boundary["runner"]
                config = runner.moe_config
                effective_reduced = (runner._fused_output_is_reduced
                                     if output_is_reduced is None else output_is_reduced)
                reduction = {"input_shape": list(states.shape), "input_dtype": str(states.dtype),
                             "output_shape": list(call["result"].shape) if call["result"] is not None else None,
                             "output_dtype": str(call["result"].dtype) if call["result"] is not None else None,
                             "trunc_size": trunc_size, "output_is_reduced": output_is_reduced,
                             "effective_output_is_reduced": bool(effective_reduced),
                             "tp_size": int(config.tp_size),
                             "skip_final_all_reduce": bool(config.skip_final_all_reduce),
                             "is_sequence_parallel": bool(config.is_sequence_parallel),
                             "collective_calls": calls,
                             "collective_site": "vllm.model_executor.layers.fused_moe.runner.moe_runner.tensor_model_parallel_all_reduce"}
                step["segments"].append({"unit_id": unit_id, "role": "final_reduce",
                                          "range_name": name, "begin_event": begin, "end_event": end,
                                          "reduction": reduction})
                step["pending_reduce"] = None
                step["active_unit"] = None
                step["active_reduce"] = None

    @contextmanager
    def output_collective(self, states):
        step = self.current
        if (step is not None and step.get("active_reduce") is not None
                and threading.get_ident() == step["thread_id"]):
            step["active_reduce"].append({"input_shape": list(states.shape), "input_dtype": str(states.dtype)})
        yield

    def end_step(self, result):
        step = self.current
        if (step is None or step["active_unit"] is not None or step["pending_reduce"] is not None
                or threading.get_ident() != step["thread_id"]):
            raise RuntimeError("unclosed or missing timed worker step")
        copy_event = getattr(result, "copy_event", None)
        if not isinstance(copy_event, self.torch.cuda.Event):
            raise RuntimeError("stock async output has no CUDA copy-completion event")
        main_end = self.event(step["main_stream"])
        self.join_stream.wait_event(main_end)
        self.join_stream.wait_event(copy_event)
        step["end_event"] = self.event(self.join_stream)
        step["copy_event_observed"] = True
        step["completion_join"] = "main_event_and_stock_copy_event"
        step["scope"].__exit__(None, None, None)
        self.steps.append(step)
        self.current = None

    def finish(self, *, identity, runtime, exclusivity=None):
        """Close the arm: drain the device, stop the profiler, write the capture.

        ``exclusivity`` is the worker's device-process samples at arm and at
        finish (``full_engine_timing_worker``): carried in the capture for the
        timing observation's device-exclusivity witness, never read here.
        """
        if self.current is not None:
            raise RuntimeError("timing finish encountered an unclosed engine step")
        if self.patches:
            self.patches.__exit__(None, None, None)
        self.torch.cuda.synchronize(self.device)
        self.profiler.stop()
        stream_ids = {stream_id for step in self.steps
                      for stream_id in (step["main_stream_id"], step["copy_stream_id"])}
        try:
            stream_ids.add(_stream_id(self.join_stream))
            profiler_health = _profiler_drop_witness(stream_ids)
        except Exception as exc:  # noqa: BLE001 -- unavailable is not zero
            profiler_health = {"available": False, "reason": f"{type(exc).__name__}: {exc}", "streams": None}
        finished_unix_ns = time.time_ns()
        profile_path = self.output / "profile.json"
        self.profiler.export_chrome_trace(str(profile_path))
        profile_bytes = profile_path.read_bytes()
        profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
        profile = json.loads(profile_bytes)
        del profile_bytes
        rows = []
        for step in self.steps:
            previous = step["start_event"]
            units, gaps = [], []
            segments = step["segments"] if self.tp2_composite else step["units"]
            measured_segments = []
            for segment in segments:
                gaps.append(previous.elapsed_time(segment["begin_event"]))
                elapsed = segment["begin_event"].elapsed_time(segment["end_event"])
                measured_segments.append({"unit_id": segment["unit_id"], "range_name": segment["range_name"],
                                          "role": segment.get("role", "apply"), "elapsed_ms": elapsed}
                                         | ({"reduction": segment["reduction"]} if "reduction" in segment else {}))
                previous = segment["end_event"]
            gaps.append(previous.elapsed_time(step["end_event"]))
            for unit in step["units"]:
                parts = [segment["elapsed_ms"] for segment in measured_segments
                         if segment["unit_id"] == unit["unit_id"]]
                units.append({"unit_id": unit["unit_id"], "range_name": unit["range_name"],
                              "elapsed_ms": math.fsum(parts)})
            rows.append({key: step[key] for key in ("step_id", "phase", "scheduled_tokens", "main_stream_id",
                          "copy_stream_id", "copy_event_observed", "completion_join")}
                        | {"whole_step_ms": step["start_event"].elapsed_time(step["end_event"]),
                           "units": units, "fixed_gaps_ms": gaps}
                        | ({"segments": measured_segments} if self.tp2_composite else {}))
        capture = {"schema": ("tessera.full_engine_timing_capture.v2" if self.tp2_composite
                               else "tessera.full_engine_timing_capture.v1"), "arm": self.arm,
                   "identity": identity, "runtime": runtime() if callable(runtime) else runtime, "device_id": self.device,
                   "canonical_unit_ids": [row["unit_id"] for row in self.boundaries],
                   "native_boundaries": [{key: row[key] for key in ("unit_id", "module", "boundary", "includes_router")}
                                         | ({"segments": ["apply", "final_reduce"] if row["unit_id"].startswith("s:")
                                             else ["apply"]} if self.tp2_composite else {})
                                         for row in self.boundaries],
                   "steps": rows, "ranges": self.ranges, "errors": self.errors,
                   "observer_span": {"started_unix_ns": self.started_unix_ns,
                                     "finished_unix_ns": finished_unix_ns},
                   "exclusivity": exclusivity,
                   "profiler_collection_health": profiler_health if self.tp2_composite else None,
                   "profile_sha256": profile_sha256,
                   "timings": None, "admission": None, "pricing_scope": PRICING_SCOPE}
        (self.output / "capture.json").write_text(json.dumps(capture, sort_keys=True, indent=2) + "\n")
        partition = analyze_profile_partition(capture, profile)
        (self.output / "partition.json").write_text(json.dumps(partition, sort_keys=True, indent=2) + "\n")
        return {"directory": str(self.output), "partition": partition}
