"""Cap one routed gate role and retain its CPU profile, including interrupted phases."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import cProfile
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import pstats
import resource
import signal
import sys
import time

from experiments.numerics_rows.common import write_json


class ProfileEnd(BaseException):
    pass


def main():
    from experiments.numerics_rows import cases, scale_index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1500)
    args, remaining = parser.parse_known_args()
    if not 0 < args.seconds <= 1500:
        parser.error("the profile budget must be in [1, 1500] seconds")
    remaining = remaining[1:] if remaining[:1] == ["--"] else remaining
    if "--sample-expert" not in remaining:
        parser.error("one routed expert is required")
    args.out.mkdir(parents=True, exist_ok=True)
    phases, current, reads = [], {}, {}
    arithmetic = {"calls": 0, "seconds": 0.0}
    packages = {}
    for name in ("gmpy2", "flint"):
        present = importlib.util.find_spec(name) is not None
        packages[name] = {"present": present}
        if present:
            packages[name]["version"] = importlib.metadata.version("python-flint" if name == "flint" else name)
    events = (args.out / "phases.jsonl").open("w", buffering=1)
    def emit(record):
        events.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    def phase(name, function, metadata=lambda *_a, **_k: {}):
        def measured(*a, **k):
            record = {"phase": name, **metadata(*a, **k), "finished": False}
            started = time.monotonic()
            emit({"start": record})
            try:
                result = function(*a, **k)
                record["finished"] = True
                return result
            finally:
                record.update(seconds=time.monotonic() - started, peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                phases.append(record)
                emit({"end": record})
        return measured
    for name in ("checkpoint_routed", "decode_roles", "column_coverage", "routed_separation", "exact_chunks"):
        setattr(cases, name, phase(name, getattr(cases, name)))
    original_values = scale_index.scale_values
    class Values(Sequence):
        def __init__(self, entry):
            self.values = original_values(entry)
            self.entry = {k: v for k, v in entry.items() if k != "values"}
        def __len__(self):
            return len(self.values)
        def __getitem__(self, index):
            return self.values[index]
        def __iter__(self):
            # Only joint_coverage's shared grouping step iterates the whole array.
            record = {"phase": "scale_grouping", **self.entry, "finished": False}
            started = time.monotonic()
            emit({"start": record})
            try:
                yield from self.values
                record["finished"] = True
            finally:
                record["seconds"] = time.monotonic() - started
                phases.append(record)
                emit({"end": record})
    scale_index.scale_values = Values
    original_evaluate = scale_index.evaluate
    def evaluate(case, wrong, values, table=None):
        key = tuple(case["read"])
        if current.get("read") != key:
            if "read" in current:
                reads[current["plan"]]["completed_reads"] += 1
            current["read"] = key
        started = time.monotonic()
        try:
            return original_evaluate(case, wrong, values, table)
        finally:
            arithmetic["calls"] += 1
            arithmetic["seconds"] += time.monotonic() - started
    scale_index.evaluate = evaluate
    original_plan = scale_index.make_plan
    def plan(scale, n, read_cases, **kwargs):
        key = str(len(reads))
        current.clear()
        current["plan"] = key
        reads[key] = {"scale": scale, "n": n, "reads": len(read_cases), "completed_reads": 0, "array": kwargs.get("metadata", {})}
        result = phase("make_plan", original_plan, lambda *_a, **_k: reads[key])(scale, n, read_cases, **kwargs)
        reads[key]["completed_reads"] = result["reads"]
        current.clear()
        return result
    scale_index.make_plan = plan
    original_index = cases.scale_index_manifest
    def index(*a, **k):
        result = phase("scale_index_manifest", original_index)(*a, **k)
        write_json(args.out / "role-scale-index.json", result)
        raise ProfileEnd("ONE_ROLE_FINISHED")
    cases.scale_index_manifest = index
    def capped(_signum, _frame):
        raise ProfileEnd("CAPPED")
    previous = signal.signal(signal.SIGALRM, capped)
    profiler = cProfile.Profile()
    started, status, error = time.monotonic(), "ERROR", None
    sys.argv = ["cases", *remaining]
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    profiler.enable()
    try:
        cases.main()
    except ProfileEnd as end:
        status = str(end)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        profiler.disable()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        profiler.dump_stats(args.out / "c1.pstats")
        with (args.out / "c1-profile.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(100)
        write_json(args.out / "profile.json", {"measurement_only": True, "generation_finished": False,
            "status": status, "error": error, "wall_seconds": time.monotonic() - started, "budget_seconds": args.seconds,
            "peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "phases": phases,
            "plans": list(reads.values()), "arithmetic": arithmetic, "packages": packages,
            "scope": "one routed gate role, Fraction backend, one native thread"})
        events.close()


if __name__ == "__main__":
    main()
