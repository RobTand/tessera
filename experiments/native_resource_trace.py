"""Continuous-trace window analyzer for native resource measurement lanes.

A finished, terminal ``tessera.cupti_memory_trace.v1`` is the resource
authority. This module indexes one trace and derives per-marker rate windows
(baseline live set, allocation-request multiset, transient peak) without
inventing events: original rows are retained, all memory history is validated
through stop, and original trace bytes remain the receipt identity. It is a
published analyzer: clients consume derived windows as attested measurements
or invoke this module as a subprocess against a trace file; they never
reimplement CUPTI row semantics.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_REPOSITORY_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in _sys.path:
    # Allow this analyzer to run as a script (python experiments/native_resource_trace.py)
    # from any working directory, e.g. as a subprocess of a downstream consumer.
    _sys.path.insert(0, _REPOSITORY_ROOT)

import copy
import hashlib
import json
from bisect import bisect_left, bisect_right
from collections import Counter

from experiments.native_operator_resources import (
    MEMORY_API_OPERATIONS,
    _integer,
    api_base_name,
    memory_row_domain,
    ownership_operation,
    validate_cupti_capture,
)

WINDOW_SCHEMA = "tessera.native_rate_resource_window.v1"
CLI_SCHEMA = "tessera.native_resource_trace_windows.v1"

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def _requests(rows):
    counts = Counter((r["bytes"], r["source"], r["memory_kind"]) for r in rows
                     if r["operation"] == "allocate")
    return [{"bytes": size, "source": source, "memory_kind": kind, "count": count}
            for (size, source, kind), count in sorted(counts.items())]


class ContinuousTrace:
    """Index a finished trace once; retain original rows, never invented events.

    All memory history is validated through stop, including events outside a
    requested window. Starts snapshot the exact live allocation records, and
    window requests retain their sources. ``view`` only folds fully matched
    historical lifetimes; it is for the unchanged warmed-apply scalar analyzer,
    not an alternate completeness rule. Original trace bytes remain the receipt
    identity. Build/use this index on a terminal immutable trace.
    """

    def __init__(self, trace, *, device_id, context_id):
        self.pid = validate_cupti_capture(trace)
        self.device = _integer(device_id)
        self.context = _integer(context_id)
        self.trace = trace
        self.trace_sha256 = digest(trace)
        start, stop = _integer(trace["start_ns"], 1), _integer(trace["end_ns"], 1)
        if stop <= start:
            raise ValueError("continuous trace stop precedes start")
        paired = {}
        for marker in trace["markers"]:
            name, separator, side = marker["name"].rpartition(":")
            if not separator or side not in ("begin", "end"):
                raise ValueError("continuous trace marker must name begin or end")
            point = _integer(marker["timestamp_ns"], 1)
            if not start < point < stop or side in paired.setdefault(name, {}):
                raise ValueError("duplicate/out-of-range continuous trace marker")
            paired[name][side] = point
        if not paired or any(set(p) != {"begin", "end"} or p["begin"] >= p["end"]
                             for p in paired.values()):
            raise ValueError("exactly one paired interval required per marker")
        self.intervals = {name: (p["begin"], p["end"]) for name, p in paired.items()}
        ordered = sorted(self.intervals.values())
        for index in range(len(ordered) - 1):
            if ordered[index][1] >= ordered[index + 1][0]:
                raise ValueError("rate windows overlap in the continuous trace")
        self.first_begin = min(begin for begin, _ in self.intervals.values())
        self.apis = trace["api_events"]
        if not self.apis:
            raise ValueError("no observed CUDA API records")
        self.correlations = {}
        api_order = []
        for i, api in enumerate(self.apis):
            a, b = _integer(api["start_ns"], 1), _integer(api["end_ns"], 1)
            if b < a or not isinstance(api["name"], str):
                raise ValueError("CUDA API timestamps are reversed or name is invalid")
            key = (_integer(api["process_id"], 1), _integer(api["correlation_id"]))
            self.correlations.setdefault(key, []).append(i)
            api_order.append((a, b, i))
        # Sweep API intervals against all marker points once. Overlapping
        # windows are refused above, nesting included; each API row is priced
        # by the one window that contains it, without rescanning the prefix.
        api_order.sort()
        self.window_apis = {name: set() for name in self.intervals}
        active, opened, cursor = {}, set(), 0
        points = sorted((t, side, name) for name, (begin, end) in self.intervals.items()
                        for t, side in ((begin, 0), (end, 1)))
        for point, side, name in points:
            while cursor < len(api_order) and api_order[cursor][0] <= point:
                a, b, i = api_order[cursor]
                for interval in opened:
                    self.window_apis[interval].add(i)
                if b >= point:
                    active[i] = b
                cursor += 1
            active = {i: b for i, b in active.items() if b >= point}
            if side == 0:
                self.window_apis[name].update(active)
                opened.add(name)
            else:
                opened.remove(name)
        self.events = sorted(trace["memory_events"], key=lambda r: r["timestamp_ns"])
        if not self.events:
            raise ValueError("no observed allocation records; collection not demonstrated")
        self.times = []
        self.device_times = []
        for row in self.events:
            t = _integer(row["timestamp_ns"], 1)
            if not start <= t <= stop:
                raise ValueError("allocation lies outside continuous collection")
            self.times.append(t)
            if memory_row_domain(row) != "host":
                # Host rows are outside the device-byte scope, so they cannot
                # be an unpriced device gap either; the gap check reads these.
                self.device_times.append(t)
        self.starts, self.ends, self.peaks = {}, {}, {}
        # No memory operation may hide between rate windows: the pricing prefix
        # is the whole persistent process, not only the windows a rate asked for.
        # Host rows are excluded: they are out of scope for device bytes and
        # are skipped by the row application, so a pinned host allocation
        # between two windows must not refuse the trace.
        begins = [begin for begin, _ in ordered]
        last_end = ordered[-1][1]
        for point in self.device_times:
            if self.first_begin <= point <= last_end:
                index = bisect_right(begins, point) - 1
                if index < 0 or point > ordered[index][1]:
                    raise ValueError(
                        f"memory operation at {point} lies outside rate windows (unpriced gap)")
        live, opened, cursor = {}, set(), 0
        for point, side, name in points:
            # A begin snapshot precedes records exactly on its boundary; an end
            # snapshot follows them, matching analyze_trace's inclusive window.
            while cursor < len(self.events) and (self.times[cursor] < point or
                                                (side == 1 and self.times[cursor] == point)):
                row = self.events[cursor]
                self._apply(live, row)
                for interval in opened:
                    self.peaks[interval] = max(self.peaks[interval], sum(r["bytes"] for r in live.values()))
                cursor += 1
            if side == 0:
                self.starts[name] = tuple(live.values())
                self.peaks[name] = sum(r["bytes"] for r in live.values())
                opened.add(name)
            else:
                self.ends[name] = tuple(live.values())
                opened.remove(name)
        # A malformed tail must not disappear because no rate asked for it.
        for row in self.events[cursor:]:
            self._apply(live, row)
        first = min(self.intervals, key=lambda name: self.intervals[name][0])
        self.initial_live = self.starts[first]

    def _apply(self, live, row):
        if row["process_id"] != self.pid:
            raise ValueError("allocation belongs to another process")
        # One home classifies the row's domain: host rows are outside the
        # device-byte scope, and anything unsupported is refused right here.
        kind = memory_row_domain(row)
        if kind == "host":
            return  # host allocations are outside the established device-byte scope
        address, size = _integer(row["address"], 1), _integer(row["bytes"], 1)
        context = _integer(row["context_id"])
        static_zero = (kind == "static" and self.trace["cupti_version"] == 130001 and
                       context == row["correlation_id"] == row["stream_id"] == 0)
        if row["device_id"] != self.device or not (context == self.context or static_zero):
            raise ValueError("allocation device/context differs from operator")
        key = (kind, context, address)
        if row["operation"] == "allocate":
            if not isinstance(row["source"], str) or not row["source"]:
                raise ValueError("allocation request source is unknown")
            if key in live:
                raise ValueError("duplicate live allocation address")
            live[key] = row
        elif row["operation"] == "free":
            previous = live.pop(key, None)
            if previous is None or previous["bytes"] != size:
                raise ValueError("free lacks matching allocation bytes/context")
        else:
            raise ValueError("unknown memory operation")

    def _rows(self, interval):
        begin, end = self.intervals[interval]
        return self.events[bisect_left(self.times, begin):bisect_right(self.times, end)]

    def _apis(self, interval):
        indices = set(self.window_apis[interval])
        for row in self._rows(interval):
            indices.update(self.correlations.get((row["process_id"], row["correlation_id"]), ()))
        return [self.apis[i] for i in sorted(indices)]

    def _coverage(self, interval):
        begin, end = self.intervals[interval]
        required, observed = {}, set()
        for api in self._apis(interval):
            name = api_base_name(api["name"])
            a, b = api["start_ns"], api["end_ns"]
            if a > end or b < begin:
                continue
            if name.startswith(("cudaGraph", "cuGraph")):
                raise ValueError("CUDA graph execution is outside eager resource scope")
            operation = ownership_operation(name, api["return_value"])
            if operation == "unsupported":
                raise ValueError("unsupported or failed allocation API: " + name)
            if operation is not None:
                if api["process_id"] != self.pid or not begin <= a <= b <= end:
                    raise ValueError("allocation API crosses the rate process/interval")
                key = (self.pid, api["correlation_id"])
                if len(self.correlations[key]) != 1 or key in required:
                    raise ValueError("ambiguous allocation API correlation")
                required[key] = api
        for row in self._rows(interval):
            if memory_row_domain(row) == "host":
                continue
            if row["memory_kind"] == 6:
                # analyze_trace refuses module/static activity inside apply;
                # the persistent window must refuse it with the same words.
                raise ValueError("module/static allocation or free occurred during apply")
            key = (self.pid, row["correlation_id"])
            api = required.get(key)
            if api is None or key in observed:
                raise ValueError("memory operation has no unambiguous in-window API correlation")
            name = api_base_name(api["name"])
            if not api["start_ns"] <= row["timestamp_ns"] <= api["end_ns"]:
                raise ValueError("memory operation timestamp is outside its API")
            if row["operation"] != MEMORY_API_OPERATIONS[name]:
                raise ValueError("allocation operation disagrees with its API")
            observed.add(key)
        if observed != set(required):
            raise ValueError("allocation API lacks its matching memory operation")

    def window(self, interval):
        self._coverage(interval)
        baseline, final = self.starts[interval], self.ends[interval]
        if baseline != self.initial_live or final != baseline:
            raise ValueError("rate changed the one-time baseline or retained allocations after eviction")
        begin, end = self.intervals[interval]
        base = sum(row["bytes"] for row in baseline)
        rows = [row for row in self._rows(interval) if row["memory_kind"] in (3, 6)]
        return {"schema": WINDOW_SCHEMA, "status": "observed",
                "trace_sha256": self.trace_sha256, "process_id": self.pid,
                "collection_start_ns": self.trace["start_ns"], "interval": interval,
                "begin_ns": begin, "end_ns": end, "baseline_live": list(baseline),
                "end_live": list(final), "baseline_bytes": base,
                "window_peak_bytes": self.peaks[interval],
                "transient_peak_bytes": self.peaks[interval] - base,
                "allocation_requests": _requests(rows),
                "initialization_requests": _requests(self.initial_live)}



def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="experiments.native_resource_trace",
        description="Derive rate windows from a finished continuous trace file (published analyzer).")
    parser.add_argument("trace", help="path to a tessera.cupti_memory_trace.v1 JSON file")
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--context-id", type=int, required=True)
    parser.add_argument("--interval", action="append", required=True,
                        help="marker interval to derive, e.g. rate:TESSERA_BF16_K1_R256 (repeatable)")
    args = parser.parse_args(argv)
    try:
        with open(args.trace, encoding="utf-8") as handle:
            trace = json.load(handle)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"cannot read trace file {args.trace!r}: {error}") from error
    index = ContinuousTrace(trace, device_id=args.device_id, context_id=args.context_id)
    unknown = [i for i in args.interval if i not in index.intervals]
    if unknown:
        raise SystemExit(f"unknown intervals in trace: {unknown}")
    windows = {interval: index.window(interval) for interval in args.interval}
    report = {"schema": CLI_SCHEMA, "trace_sha256": index.trace_sha256,
              "device_id": args.device_id, "context_id": args.context_id,
              "intervals": windows}
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()

