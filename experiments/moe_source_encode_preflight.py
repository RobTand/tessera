#!/usr/bin/env python3
"""Measure one real source expert before scheduling a full MoE encode.

The exporter owns source naming and geometry. This probe reads its plan for
the first expert in the first routed layer, encodes and verifies each of that
expert's projections, and profiles a repeat of the first projection. It is a
weight-space preparation measurement, never a served-quality gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import threading
import time

import torch
from safetensors import safe_open

import export_tessera_serving as exporter
from moe_encode_rate_profile import Counters, encode_one, power_w
from tessera.alphabet import E4M3_GRID


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--q256", type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA-visible PrismaBuild allocation is required")
    shards, _shapes, _packed, routed = exporter.quantizable(args.src)
    stacks = exporter.expert_stacks(routed)
    stack = min(stacks, key=lambda name: (exporter.body_layer(name), name))
    plan = exporter.plan_expert_stack(
        stack, stacks[stack], E4M3_GRID, args.q256,
        config=json.loads((args.src / "config.json").read_text()))
    expert = min(unit["expert"] for unit in plan["units"])
    selected = [unit for unit in plan["units"] if unit["expert"] == expert]
    weight_map = {name: shard for shard, names in shards.items() for name in names}
    units = []
    for unit in selected:
        name = unit["source_tensor"]
        with safe_open(str(args.src / weight_map[name]), framework="pt") as handle:
            units.append((name, handle.get_tensor(name)))
    record = {
        "schema": "tessera.moe-source-encode-preflight.v1",
        "container_hostname": socket.gethostname(), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0), "source": str(args.src),
        "stack": stack, "expert": expert, "q256": args.q256,
        "source_stacks": len(stacks), "experts_in_stack": plan["experts"],
        "arm": "weights_only", "verify": True,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    counters = Counters()
    counters.install()
    stop = threading.Event()
    samples = []

    def sample():
        while not stop.is_set():
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                if key in {"MemAvailable", "SwapFree", "SwapTotal"}:
                    memory[key + "_kib"] = int(value.strip().split()[0])
            samples.append({"monotonic_s": time.monotonic(), "power_w": power_w(), **memory})
            stop.wait(1)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    try:
        for iteration, (name, weight) in enumerate([units[0], *units]):
            counters.reset()
            torch.cuda.synchronize()
            started = time.monotonic()
            encoded = encode_one(name, weight, args.q256, True)
            torch.cuda.synchronize()
            row = {"tensor": name, "shape": list(weight.shape),
                   "warmup": iteration == 0, "seconds": time.monotonic() - started,
                   "wire_bytes": int(encoded.exact_bytes), "bpp": float(encoded.bpp),
                   "counters": counters.snapshot()}
            rows.append(row)
            print(json.dumps(row), flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            encode_one(*units[0], args.q256, True)
        events = sorted((event for event in prof.key_averages() if event.self_device_time_total > 0),
                        key=lambda event: -event.self_device_time_total)
        record["profiler_top"] = [{"name": event.key, "self_cuda_us": event.self_device_time_total,
                                    "calls": event.count} for event in events[:8]]
    finally:
        stop.set()
        sampler.join()
    record.update({"units": rows, "telemetry": samples,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "telemetry"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
