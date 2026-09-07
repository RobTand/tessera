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

from experiments.full_engine_timing_boundaries import observe_apply_boundaries


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("timing observation must be finite and nonnegative")
    return value


def analyze_profile_partition(capture, profile):
    """Recompute launch ownership and explicit stream coverage from trace bytes.

This does not certify Kineto's collection completeness or observer overhead.
No kernel-duration sum is subtracted from CUDA-event elapsed time.
    """
    result = {"schema": "tessera.full_engine_timing_partition.v1", "status": "incomplete",
              "timings": None, "admission": "not_implemented", "issues": [], "steps": [],
              "qualification_gaps": ["profiler collection health qualification", "observer overhead qualification",
                                     "bound reference assignment and runtime admission", "full-engine resource closure"]}
    try:
        if capture["schema"] != "tessera.full_engine_timing_capture.v1":
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
            if event.get("ph") == "X" and event.get("name") in ranges:
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
            if len(step["fixed_gaps_ms"]) != len(expected) + 1:
                raise ValueError("adjacent fixed gap coverage is incomplete")
            components = [*step["fixed_gaps_ms"], *[row["elapsed_ms"] for row in step["units"]]]
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
            unit_ranges = [row for row in matched if row["kind"] == "unit"]
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
                tail = ranges[step["units"][-1]["range_name"]][0]
                if launch["ts"] < tail["ts"] + tail["dur"]:
                    raise ValueError("copy-stream work overlaps native units; sequential gaps do not cover it")
            row["gpu_operations"].append({"name": operation["name"], "category": operation["cat"],
                "correlation": args["correlation"], "stream": stream, "unit_id": unit_id,
                "timestamp_us": operation["ts"], "duration_us": operation["dur"]})
        for _, row in by_step.values():
            observed_units = {op["unit_id"] for op in row["gpu_operations"] if op["unit_id"] is not None}
            if observed_units != set(expected):
                raise ValueError("canonical unit has no attributed GPU operation")
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
    code = get_id(None, ctypes.c_void_p(handle), int(handle == 2), ctypes.byref(value))
    if code:
        raise RuntimeError(f"CUPTI stream identity lookup failed: {code}")
    return value.value


class FullEngineTimingRecorder:
    def __init__(self, boundaries, *, arm, output, device_id=0):
        import torch
        if arm not in {"control", "partition"}:
            raise ValueError("unknown timing observation arm")
        self.torch, self.boundaries, self.arm = torch, boundaries, arm
        self.started_unix_ns = time.time_ns()
        self.output, self.device = Path(output), device_id
        self.output.mkdir(parents=True, exist_ok=False)
        self.steps, self.ranges, self.errors, self.current = [], {}, [], None
        self.join_stream = torch.cuda.Stream(device=device_id)
        self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        self.profiler.start()
        self.patches = observe_apply_boundaries(boundaries, self.unit) if arm == "partition" else None
        if self.patches:
            self.patches.__enter__()

    def event(self, stream):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

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
                        "scope": scope, "thread_id": threading.get_ident(), "units": [], "active_unit": None,
                        "start_event": self.event(main_stream)}

    @contextmanager
    def unit(self, boundary, call):
        step = self.current
        if step is None:
            raise RuntimeError("canonical native unit ran outside an armed engine step")
        if step["active_unit"] is not None or threading.get_ident() != step["thread_id"]:
            raise RuntimeError("native unit scopes overlap or cross worker threads")
        if self.torch.cuda.current_stream(self.device) != step["main_stream"]:
            raise RuntimeError("canonical native unit changed its execution stream")
        index = len(step["units"])
        name = f"tessera.engine.unit.{step['step_id']}.{index}"
        self.ranges[name] = {"kind": "unit", "step_id": step["step_id"], "unit_id": boundary["unit_id"]}
        step["active_unit"] = boundary["unit_id"]
        with self.torch.profiler.record_function(name):
            begin = self.event(step["main_stream"])
            try:
                yield
            finally:
                end = self.event(step["main_stream"])
                step["units"].append({"unit_id": boundary["unit_id"], "range_name": name,
                                      "begin_event": begin, "end_event": end})
                step["active_unit"] = None

    def end_step(self, result):
        step = self.current
        if step is None or step["active_unit"] is not None or threading.get_ident() != step["thread_id"]:
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

    def finish(self, *, identity, runtime):
        if self.current is not None:
            raise RuntimeError("timing finish encountered an unclosed engine step")
        if self.patches:
            self.patches.__exit__(None, None, None)
        self.torch.cuda.synchronize(self.device)
        self.profiler.stop()
        finished_unix_ns = time.time_ns()
        profile_path = self.output / "profile.json"
        self.profiler.export_chrome_trace(str(profile_path))
        rows = []
        for step in self.steps:
            previous = step["start_event"]
            units, gaps = [], []
            for unit in step["units"]:
                gaps.append(previous.elapsed_time(unit["begin_event"]))
                units.append({"unit_id": unit["unit_id"], "range_name": unit["range_name"],
                              "elapsed_ms": unit["begin_event"].elapsed_time(unit["end_event"])})
                previous = unit["end_event"]
            gaps.append(previous.elapsed_time(step["end_event"]))
            rows.append({key: step[key] for key in ("step_id", "phase", "scheduled_tokens", "main_stream_id",
                          "copy_stream_id", "copy_event_observed", "completion_join")}
                        | {"whole_step_ms": step["start_event"].elapsed_time(step["end_event"]),
                           "units": units, "fixed_gaps_ms": gaps})
        capture = {"schema": "tessera.full_engine_timing_capture.v1", "arm": self.arm,
                   "identity": identity, "runtime": runtime() if callable(runtime) else runtime, "device_id": self.device,
                   "canonical_unit_ids": [row["unit_id"] for row in self.boundaries],
                   "native_boundaries": [{key: row[key] for key in ("unit_id", "module", "boundary", "includes_router")}
                                         for row in self.boundaries],
                   "steps": rows, "ranges": self.ranges, "errors": self.errors,
                   "observer_span": {"started_unix_ns": self.started_unix_ns,
                                     "finished_unix_ns": finished_unix_ns},
                   "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                   "timings": None, "admission": "not_implemented"}
        (self.output / "capture.json").write_text(json.dumps(capture, sort_keys=True, indent=2) + "\n")
        partition = analyze_profile_partition(capture, json.loads(profile_path.read_text()))
        (self.output / "partition.json").write_text(json.dumps(partition, sort_keys=True, indent=2) + "\n")
        return {"directory": str(self.output), "partition": partition}
