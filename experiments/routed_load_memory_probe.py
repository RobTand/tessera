#!/usr/bin/env python3
"""Per-stack load memory of the compact routed window intake, on real wires.

The question this answers: what does one rank's device allocator hold while a
routed FP8/BF16 expert stack loads, and after it finalizes -- split into what
the prepared stack retains (``resident``), what the allocator has handed out
(``allocated``) and what it has reserved from the device (``reserved``, the
figure a unified-memory box pays).  A load whose ``reserved`` climbs while
``allocated`` stays flat is holding dead allocator segments, not weights.

It drives the load path vLLM calls, ``moe_route._RankLocalPackedIntake``, one
projection per callback in expert order, then ``finish`` and ``adapter()`` --
the fused routed lane or the compact adapter, whichever the stack admits.  No
vLLM engine is constructed.  Run it under the runtime's load context
(``PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20``, which ``gpu_worker`` scopes
around ``load_model``; ``docs/measurements/tessera-a4-loader-staging-20260916.md``).

``--history N`` records the allocator's own event trace for the first ``N``
callbacks and summarises every ``segment_alloc`` by its allocating stack: that
is the in-process attribution of any reserved growth.  ``--digest`` hashes
every retained tensor of the finished stack so two source trees can be
compared byte for byte.

``--replay-shards FIRST:LAST`` replays the checkpoint's own load ORDER instead:
the shard files FIRST..LAST (1-based, inclusive) in vLLM's natural order, and
each file's routed wires in its sorted key order, the order
``safetensors_weights_iterator`` yields them.  Every routed stack met gets its
intake, and all of them stay live until the window ends, as vLLM holds them
until ``process_weights_after_loading``.  Allocator stats are recorded per
shard.  A slab stranded by allocator state -- which holes the earlier stacks
left -- only shows up in this mode.  A single stack in a fresh process has
none of that history (tessera#724).  Only the routed wires are replayed:
dense and passthrough tensors are not constructed.

  routed_load_memory_probe.py --data CKPT --layers 11 --rank 0 --out OUT.json
  routed_load_memory_probe.py --data CKPT --replay-shards 88:104 --rank 1 --out OUT.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench_routed_load import MOE_UNITS, _load_wires, _schemes, _target  # noqa: E402


def _stats(torch) -> dict:
    s = torch.cuda.memory_stats()
    return {
        "allocated": int(s.get("allocated_bytes.all.current", 0)),
        "reserved": int(s.get("reserved_bytes.all.current", 0)),
        "peak_allocated": int(s.get("allocated_bytes.all.peak", 0)),
        "peak_reserved": int(s.get("reserved_bytes.all.peak", 0)),
        "inactive_split": int(s.get("inactive_split_bytes.all.current", 0)),
        "segments": int(s.get("segment.all.current", 0)),
        "large_segments": int(s.get("segment.large_pool.current", 0)),
        "alloc_retries": int(s.get("num_alloc_retries", 0)),
    }


def _frames(entry, limit=6):
    out = []
    for frame in entry.get("frames") or []:
        name = str(frame.get("filename", ""))
        if "/tessera/" not in name and "routed_load_memory_probe" not in name:
            continue
        out.append(f"{name.split('/tessera/')[-1]}:{frame.get('line')}:{frame.get('name')}")
        if len(out) >= limit:
            break
    return tuple(out)


def _history_summary(snapshot) -> dict:
    """``segment_alloc`` events grouped by allocating stack and size."""
    by_stack = collections.Counter()
    bytes_by_stack = collections.Counter()
    actions = collections.Counter()
    for trace in snapshot.get("device_traces", []):
        for entry in trace:
            action = entry.get("action")
            actions[action] += 1
            if action == "segment_alloc":
                key = (_frames(entry), int(entry.get("size", 0)))
                by_stack[key] += 1
                bytes_by_stack[key] += int(entry.get("size", 0))
    rows = [{"stack": list(stack), "segment_bytes": size, "count": n,
             "total_bytes": bytes_by_stack[(stack, size)]}
            for (stack, size), n in by_stack.most_common(20)]
    return {"actions": dict(actions), "segment_allocs": rows}


def _digest(torch, prepared) -> dict:
    out = {}
    for name, tensor in prepared.named_tensors():
        flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        out[name] = hashlib.sha256(flat.cpu().numpy().tobytes()).hexdigest()
    return out


def _dead_slabs(torch, size=20 << 20) -> dict:
    """Wholly inactive large-pool segments of exactly ``size`` bytes: the
    ``max_split_size_mb=20`` context's stranded slabs."""
    dead = 0
    for seg in torch.cuda.memory_snapshot():
        if seg.get("segment_type") != "large" or int(seg.get("total_size", 0)) != size:
            continue
        if all(b.get("state") == "inactive" for b in seg.get("blocks", [])):
            dead += 1
    return {"dead_20mib_slabs": dead, "dead_20mib_bytes": dead * size}


_WIRE_KEY = None


def _routed_wire(key):
    """``(layer, expert, proj)`` for a routed expert wire key, else ``None``."""
    global _WIRE_KEY
    import re

    if _WIRE_KEY is None:
        _WIRE_KEY = re.compile(
            r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
            r"(gate_proj|up_proj|down_proj)\.wire$")
    m = _WIRE_KEY.match(key)
    return None if m is None else (int(m.group(1)), int(m.group(2)), m.group(3))


def replay(args) -> int:
    """Stream the shard window FIRST..LAST through live intakes, in load order."""
    import torch
    import tessera
    from safetensors import safe_open
    from tessera.serving import moe_route

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    device = torch.device("cuda", torch.cuda.current_device())
    data = Path(args.data)
    first, last = (int(x) for x in args.replay_shards.split(":"))
    index = json.loads((data / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(index.values()))  # model-000NN-of-000MM: natural = lexicographic
    window = files[first - 1:last]
    groups = json.loads((data / "config.json").read_text())["quantization_config"]["config_groups"]
    routed = {g["targets"][0] for g in groups.values()
              if g.get("scheme", {}).get("structure") == "routed_moe"}
    proj_unit = {proj: (group, idx) for group, idx, proj in MOE_UNITS}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    record = {"tessera_file": tessera.__file__, "torch": torch.__version__,
              "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
              "data": str(data), "replay_shards": [first, last], "files": window,
              "rank": args.rank, "tp": args.tp, "experts": args.experts,
              "baseline": _stats(torch), "shards": []}
    intakes, lengths, declared, filled = {}, {}, {}, collections.Counter()
    callbacks = 0
    load_start = time.perf_counter()
    for number, name in enumerate(window, start=first):
        t_shard = time.perf_counter()
        met = collections.Counter()
        with safe_open(str(data / name), framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                hit = _routed_wire(key)
                if hit is None:
                    continue
                layer, expert, proj = hit
                if _target(layer) not in routed:
                    continue
                if layer not in intakes:
                    declared[layer] = _schemes(data, [layer], args.experts)[layer]
                    intakes[layer] = moe_route._RankLocalPackedIntake(
                        declared[layer], _target(layer), device, args.rank, args.tp)
                    lengths[layer] = (torch.zeros(args.experts, 2, dtype=torch.long),
                                      torch.zeros(args.experts, dtype=torch.long))
                group, idx = proj_unit[proj]
                wire = handle.get_tensor(key)
                intakes[layer].load(group, idx, expert, wire, device=device)
                if group == "w13":
                    lengths[layer][0][expert, idx] = wire.numel()
                else:
                    lengths[layer][1][expert] = wire.numel()
                filled[layer] += 1
                met[layer] += 1
                callbacks += 1
        torch.cuda.synchronize()
        record["shards"].append({
            "shard": number, "file": name, "seconds": time.perf_counter() - t_shard,
            "units": {str(k): v for k, v in sorted(met.items())},
            "resident": int(sum(i.resident_bytes() for i in intakes.values() if i is not None)),
            **_stats(torch), **_dead_slabs(torch)})
        s = record["shards"][-1]
        print(json.dumps({"shard": number, "units": s["units"], "resident": s["resident"],
                          "allocated": s["allocated"], "reserved": s["reserved"],
                          "dead_20mib_slabs": s["dead_20mib_slabs"]}), flush=True)
    record["load_seconds"] = time.perf_counter() - load_start
    record["callbacks"] = callbacks
    record["after_load"] = {"resident": int(sum(i.resident_bytes() for i in intakes.values())),
                            **_stats(torch), **_dead_slabs(torch)}
    complete = sorted(layer for layer, n in filled.items() if n == 3 * args.experts)
    record["complete_layers"] = complete
    record["partial_layers"] = {str(k): v for k, v in sorted(filled.items())
                                if v != 3 * args.experts}
    prepared = {}
    for layer in complete:
        prepared[layer] = intakes[layer].finish(*lengths[layer])
        intakes[layer] = None
    torch.cuda.synchronize()
    record["after_finish"] = {"resident": int(sum(p.resident_bytes() for p in prepared.values())),
                              **_stats(torch), **_dead_slabs(torch)}
    if args.digest:
        record["digest"] = {layer: _digest(torch, p) for layer, p in prepared.items()}
    Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True, default=str))
    a = record["after_load"]
    print(json.dumps({"replay": [first, last], "rank": args.rank, "callbacks": callbacks,
                      "complete_layers": complete, "resident": a["resident"],
                      "allocated": a["allocated"], "reserved": a["reserved"],
                      "peak_reserved": a["peak_reserved"],
                      "dead_20mib_slabs": a["dead_20mib_slabs"],
                      "load_seconds": round(record["load_seconds"], 1)}), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--layers", help="comma-separated routed layers")
    ap.add_argument("--replay-shards", help="FIRST:LAST shard files (1-based) in load order")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--every", type=int, default=48, help="record allocator stats every N callbacks")
    ap.add_argument("--history", type=int, default=0, help="trace the first N callbacks")
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.replay_shards:
        return replay(args)
    if not args.layers:
        ap.error("--layers or --replay-shards is required")

    import torch
    import tessera
    from tessera.serving import moe_route

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    device = torch.device("cuda", torch.cuda.current_device())
    data = Path(args.data)
    layers = [int(x) for x in args.layers.split(",")]
    t0 = time.time()
    wires = _load_wires(data, layers, args.experts)
    declared = _schemes(data, layers, args.experts)
    read_s = time.time() - t0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    record = {"tessera_file": tessera.__file__, "torch": torch.__version__,
              "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
              "data": str(data), "layers": layers, "rank": args.rank, "tp": args.tp,
              "experts": args.experts, "read_seconds": read_s,
              "q256": {layer: {g: int(declared[layer]["groups"][g]["q256"])
                               for g in ("w13", "w2")} for layer in layers},
              "baseline": _stats(torch), "samples": []}
    if args.history:
        torch.cuda.memory._record_memory_history(max_entries=500000)
    intakes, lengths = {}, {}
    callbacks = 0
    load_start = time.perf_counter()
    for layer in layers:
        intakes[layer] = moe_route._RankLocalPackedIntake(
            declared[layer], _target(layer), device, args.rank, args.tp)
        lengths[layer] = (torch.zeros(args.experts, 2, dtype=torch.long),
                          torch.zeros(args.experts, dtype=torch.long))
        for expert in range(args.experts):
            for group, index, proj in MOE_UNITS:
                wire = wires[layer][expert][proj]
                intakes[layer].load(group, index, expert, wire, device=device)
                if group == "w13":
                    lengths[layer][0][expert, index] = wire.numel()
                else:
                    lengths[layer][1][expert] = wire.numel()
                callbacks += 1
                if args.history and callbacks == args.history:
                    torch.cuda.synchronize()
                    snap = torch.cuda.memory._snapshot()
                    torch.cuda.memory._record_memory_history(enabled=None)
                    record["history"] = {"callbacks": callbacks, **_history_summary(snap)}
                if callbacks % args.every == 0:
                    torch.cuda.synchronize()
                    record["samples"].append({
                        "callbacks": callbacks, "layer": layer,
                        "resident": int(sum(i.resident_bytes() for i in intakes.values())),
                        **_stats(torch)})
    torch.cuda.synchronize()
    record["load_seconds"] = time.perf_counter() - load_start
    record["callbacks"] = callbacks
    record["after_load"] = {"resident": int(sum(i.resident_bytes() for i in intakes.values())),
                            **_stats(torch)}
    prepared = {}
    for layer in layers:
        prepared[layer] = intakes[layer].finish(*lengths[layer])
        intakes[layer] = None
    torch.cuda.synchronize()
    record["after_finish"] = {"resident": int(sum(p.resident_bytes() for p in prepared.values())),
                              **_stats(torch)}
    if not args.no_adapter:
        lanes = {}
        for layer, bundles in prepared.items():
            adapter = bundles.adapter()
            lanes[layer] = list(adapter.launch_pair)
        torch.cuda.synchronize()
        record["lanes"] = lanes
        record["after_adapter"] = {
            "resident": int(sum(p.resident_bytes() for p in prepared.values())), **_stats(torch)}
    if args.digest:
        record["digest"] = {layer: _digest(torch, p) for layer, p in prepared.items()}
    Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True, default=str))
    last = record.get("after_adapter", record["after_finish"])
    print(json.dumps({"layers": layers, "rank": args.rank, "callbacks": callbacks,
                      "resident": last["resident"], "allocated": last["allocated"],
                      "reserved": last["reserved"], "peak_allocated": last["peak_allocated"],
                      "peak_reserved": last["peak_reserved"], "segments": last["segments"],
                      "load_seconds": round(record["load_seconds"], 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
