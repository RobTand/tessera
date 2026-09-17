#!/usr/bin/env python3
"""Routed-expert load bench: each family's own expert intake on real wires.

This is the instrument for tessera#501. It calls the load path vLLM calls --
``moe_route._RankLocalPackedIntake`` at TP2, ``prepare_tessera_packed_moe_experts``
at TP1 -- on fixed expert wires read from a Tessera checkpoint, with no vLLM,
and records what the load costs in time and in device memory.

TWO ARMS, DISPATCHED ON THE SIDECAR'S FAMILY, because the two routes hold
different things.  The research-selected packed intake (``TESSERA_FP8`` /
``TESSERA_BF16``) keeps rank-local wire planes on an expert axis.  The NVFP4
routed builder (``TESSERA_NVFP4``, tessera#492/#507) preallocates the stock
modelopt tile per layer in ``create_weights`` and decodes each expert's wire
into its slot, dropping the planes -- so its footprint is the tile set, and
``--layers all`` over a whole body answers what one rank's routed experts
occupy.  The NVFP4 arm streams a layer at a time with a bounded read-ahead (a
42-layer GLM-5.3-Flash A4 body is ~142 GiB of wire), guards the host
``MemAvailable`` floor and memory PSI so a load that cannot fit reports
instead of hanging the box, and stops before
``process_weights_after_loading``: it measures the INTAKE, not the runtime's
kernel-format swizzle, and not an engine, a KV cache or an activation peak.

Two modes:

``parent``
    Runs one subprocess per (repeat, config, arm), interleaving the arms and
    alternating their order each repeat. Each arm is a ``src`` tree on its own
    ``PYTHONPATH``, so a before and an after tree never share a process.
    Samples GPU power beside the children, then runs the profile children and
    compares every prepared tensor the arms produced.

``child``
    Loads the named layers' expert wires into RAM first, so the timing is the
    intake alone, then feeds them through the intake in vLLM's order: every
    unit of every layer, then ``finish`` per layer. After each unit it records
    wall seconds (after a device synchronize), wire bytes, and the caching
    allocator's statistics. After ``finish`` it digests every tensor and field
    of the prepared owners, and saves a subset of expert slots for an exact
    ``torch.equal`` comparison.

Memory attribution follows from three differences, all recorded per unit:
``reserved - allocated`` with ``inactive_split_bytes`` is caching-allocator
fragmentation; ``allocated - resident`` is retained temporaries; ``resident``
against wire bytes is prepared state that is legitimately larger than the wire.

Profiles: ``--profile torch`` records ``torch.profiler`` over the first units
(operator and kernel-launch counts); ``--profile sample`` runs an in-process
stack sampler over a whole layer at 100 Hz, which needs no ptrace and so runs
on any worker, and reports self and inclusive frame shares.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

MOE_UNITS = (("w13", 0, "gate_proj"), ("w13", 1, "up_proj"), ("w2", 0, "down_proj"))

#: The NVFP4 routed arm loads through the runtime's OWN shard vocabulary
#: (``scheme.MOE_GROUP_SHARDS``: ``w13`` is ``("w1", "w3")``, ``w2`` is
#: ``("w2",)``), because that is what ``nvfp4_moe_route._load_wire`` resolves
#: through ``SHARD_TO_GROUP``.  Each entry is ``(shard_id, projection)``.
NVFP4_SHARDS = (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj"))

#: The stock modelopt parameter set ``create_weights`` allocates.  These tensors
#: ARE the NVFP4 route's resident state: the tile is preallocated per layer and
#: each expert's wire is decoded into its slot, so the planes never accumulate.
NVFP4_TILES = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
               "w13_weight_scale_2", "w2_weight_scale_2", "w13_input_scale", "w2_input_scale")

_EXPERT_WIRE = re.compile(r"layers\.(\d+)\.mlp\.experts\.\d+\.gate_proj\.wire$")


# --------------------------------------------------------------------------
# child
# --------------------------------------------------------------------------

def _target(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts"


def _load_wires(data: Path, layers, experts):
    """``{layer: {expert: {proj: uint8 CPU tensor}}}``, read fully into RAM."""
    import torch
    from safetensors import safe_open

    index = json.loads((data / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {}
    for layer in layers:
        for expert in range(experts):
            for _group, _index, proj in MOE_UNITS:
                key = f"{_target(layer)}.{expert}.{proj}.wire"
                wanted.setdefault(index[key], []).append((layer, expert, proj, key))
    wires = {layer: {e: {} for e in range(experts)} for layer in layers}
    for name, keys in sorted(wanted.items()):
        with safe_open(str(data / name), framework="pt", device="cpu") as handle:
            for layer, expert, proj, key in keys:
                tensor = handle.get_tensor(key).contiguous().clone()
                if tensor.dtype != torch.uint8:
                    raise SystemExit(f"{key}: expected uint8 wire, found {tensor.dtype}")
                wires[layer][expert][proj] = tensor
    return wires


def _schemes(data: Path, layers, experts):
    from tessera.serving.scheme import validate_tessera_moe_scheme

    groups = json.loads((data / "config.json").read_text())["quantization_config"]["config_groups"]
    out = {}
    for layer in layers:
        target = _target(layer)
        match = [g for g in groups.values() if g.get("targets") == [target]]
        if len(match) != 1:
            raise SystemExit(f"{target}: {len(match)} config groups")
        declared = dict(validate_tessera_moe_scheme(match[0]["scheme"], target))
        if experts != int(declared["experts"]):
            declared["experts"] = experts
        out[layer] = declared
    return out


def _allocator(torch):
    stats = torch.cuda.memory_stats()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
        "inactive_split": stats.get("inactive_split_bytes.all.current", 0),
        "alloc_retries": stats.get("num_alloc_retries", 0),
        "segments": stats.get("segment.all.current", 0),
        "allocations": stats.get("allocation.all.current", 0),
    }


def _module_resident(module) -> int:
    return int(module.wire_bytes_resident()) + 4 * int(module.rows)


def _tensor_record(torch, tensor):
    flat = tensor.detach().contiguous().reshape(-1)
    digest = hashlib.sha256(flat.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    return {"kind": "tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "stride": list(tensor.stride()), "device_type": tensor.device.type,
            "contiguous": bool(tensor.is_contiguous()), "sha256": digest}


def _walk(torch, obj, path, records, tensors, dump_experts, experts):
    """Every field of a prepared owner, by attribute path, mangled names included."""
    if isinstance(obj, torch.Tensor):
        records[path] = _tensor_record(torch, obj)
        if obj.ndim and obj.shape[0] == experts and dump_experts:
            tensors[path] = obj[:dump_experts].detach().cpu().clone()
        elif obj.numel() <= (1 << 22):
            tensors[path] = obj.detach().cpu().clone()
        return
    if isinstance(obj, torch.device):
        records[path] = {"kind": "device", "type": obj.type, "str": str(obj)}
        return
    if obj is None or isinstance(obj, (bool, int, float, str)):
        records[path] = {"kind": "scalar", "type": type(obj).__name__, "value": obj}
        return
    if isinstance(obj, (tuple, list)):
        records[path] = {"kind": type(obj).__name__, "len": len(obj)}
        for i, item in enumerate(obj):
            _walk(torch, item, f"{path}[{i}]", records, tensors, dump_experts, experts)
        return
    if isinstance(obj, dict):
        records[path] = {"kind": "dict", "keys": sorted(map(str, obj))}
        for key in sorted(obj, key=str):
            _walk(torch, obj[key], f"{path}[{key!r}]", records, tensors, dump_experts, experts)
        return
    fields = {}
    for cls in type(obj).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            name = (f"_{cls.__name__.lstrip('_')}{slot}"
                    if slot.startswith("__") and not slot.endswith("__") else slot)
            if hasattr(obj, name):
                fields[name] = getattr(obj, name)
    fields.update(getattr(obj, "__dict__", {}))
    records[path] = {"kind": "object", "type": type(obj).__name__, "fields": sorted(fields)}
    for name in sorted(fields):
        if "fingerprint" in name:
            continue  # data pointers and version counters: storage identity, not content
        _walk(torch, fields[name], f"{path}.{name}", records, tensors, dump_experts, experts)


# --------------------------------------------------------------------------
# the NVFP4 routed arm (tessera#492, tessera#507)
# --------------------------------------------------------------------------

def _checkpoint_index(data: Path, layers):
    """``{layer: (directory, weight_map)}`` for a merged checkpoint or a parts root."""
    direct = data / "model.safetensors.index.json"
    if direct.is_file():
        weight_map = json.loads(direct.read_text())["weight_map"]
        return {layer: (data, weight_map) for layer in layers}
    wanted, found = set(layers), {}
    for entry in sorted(data.iterdir()):
        index = entry / "model.safetensors.index.json"
        if not index.is_file():
            continue
        weight_map = json.loads(index.read_text())["weight_map"]
        present = {int(m.group(1)) for key in weight_map for m in [_EXPERT_WIRE.search(key)] if m}
        for layer in present & wanted:
            found[layer] = (entry, weight_map)
    return found


def _moe_layers(data: Path):
    """Every layer whose experts carry Tessera wires, ascending.

    Read off the checkpoint rather than off ``num_hidden_layers``: a passthrough
    expert stack has no wire to load and is not this route's intake.  GLM-5.3-
    Flash A4 has 42 of them (layers 3 to 44); its MTP layer 45 carries stock
    ``.weight`` experts and is named in the config's ``ignore``.
    """
    direct = data / "model.safetensors.index.json"
    sources = ([direct] if direct.is_file()
               else [entry / "model.safetensors.index.json" for entry in sorted(data.iterdir())])
    layers = set()
    for index in sources:
        if not index.is_file():
            continue
        for key in json.loads(index.read_text())["weight_map"]:
            match = _EXPERT_WIRE.search(key)
            if match:
                layers.add(int(match.group(1)))
    return sorted(layers)


def _quantization_config(directory: Path):
    """A merged checkpoint spells it ``config.json``, a part ``tessera_part_config.json``."""
    for name in ("config.json", "tessera_part_config.json"):
        path = directory / name
        if path.is_file():
            return json.loads(path.read_text())["quantization_config"]
    raise SystemExit(f"{directory}: no config.json or tessera_part_config.json")


def _expert_scheme(directory: Path, layer: int):
    """``(raw scheme, validated declaration)`` for one expert stack."""
    from tessera.serving.scheme import validate_tessera_moe_scheme

    target = _target(layer)
    groups = _quantization_config(directory)["config_groups"]
    match = [group for group in groups.values() if group.get("targets") == [target]]
    if len(match) != 1:
        raise SystemExit(f"{target}: {len(match)} config groups")
    return match[0]["scheme"], dict(validate_tessera_moe_scheme(match[0]["scheme"], target))


def _family(data: Path, layers):
    """The family the first named layer's sidecar declares."""
    if not layers:
        raise SystemExit("no layers to load")
    index = _checkpoint_index(data, layers[:1])
    if layers[0] not in index:
        raise SystemExit(f"layer {layers[0]}: no expert wires under {data}")
    return _expert_scheme(index[layers[0]][0], layers[0])[1]["family"]


def _read_layer(directory: Path, weight_map, layer: int, experts: int):
    """One layer's expert wires and A-side scales, read into RAM.

    A layer at a time, never the whole model: the 42-layer A4 body is ~142 GiB
    of wire, and the point of the measurement is to watch device memory while
    the layers arrive.
    """
    import torch
    from safetensors import safe_open

    target = _target(layer)
    wanted = {}
    for expert in range(experts):
        for shard, projection in NVFP4_SHARDS:
            for suffix in ("wire", "input_global_scale"):
                key = f"{target}.{expert}.{projection}.{suffix}"
                if key not in weight_map:
                    raise SystemExit(f"{key}: not in the checkpoint index")
                wanted.setdefault(weight_map[key], []).append((expert, shard, suffix, key))
    held, wire_bytes = {}, 0
    for name, keys in sorted(wanted.items()):
        with safe_open(str(directory / name), framework="pt", device="cpu") as handle:
            for expert, shard, suffix, key in keys:
                tensor = handle.get_tensor(key).contiguous().clone()
                if suffix == "wire":
                    if tensor.dtype != torch.uint8:
                        raise SystemExit(f"{key}: expected uint8 wire, found {tensor.dtype}")
                    wire_bytes += int(tensor.numel())
                held[(expert, shard, suffix)] = tensor
    return held, wire_bytes


def _layer_stream(index, layers, experts, read_ahead: int):
    """Yield ``(layer, held, wire_bytes)`` with a bounded read-ahead.

    The reader fills the next layer while the current one decodes, so the load
    never waits on the mount and never holds more than ``read_ahead + 1``
    layers of wire.
    """
    pending: "queue.Queue" = queue.Queue(maxsize=max(1, read_ahead))

    def produce():
        try:
            for layer in layers:
                directory, weight_map = index[layer]
                pending.put((layer, *_read_layer(directory, weight_map, layer, experts)))
        except BaseException as exc:            # noqa: BLE001 -- re-raised in the consumer
            pending.put(exc)
        else:
            pending.put(None)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        item = pending.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _host_pressure():
    """``(MemAvailable bytes, memory PSI full avg10)``.

    On GB10 the GPU allocates out of host memory, so this is the number that
    decides whether a load finishes or takes the box down with it: in
    tessera#501 an A8 load reached 112 of 166 shards and hung both Sparks.
    """
    available = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    stalled = 0.0
    try:
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            if line.startswith("full"):
                stalled = float(line.split("avg10=")[1].split()[0])
    except OSError:
        pass
    return available, stalled


def _stub_vllm_oracle():
    """vLLM's ``oracle.nvfp4`` seam, stubbed as the route's own tests stub it.

    WHAT RUNS FOR REAL AND WHAT DOES NOT.  Everything the route owns runs:
    ``create_weights`` allocates the stock modelopt parameter set,
    ``_load_wire`` parses each full container, cuts it to this rank through
    ``sharding.shard_parsed_roles`` and decodes it into the expert's slot.
    What is stubbed is the runtime's: the backend oracle, the kernel, and the
    finalize-time ``convert_to_nvfp4_moe_kernel_format`` swizzle.  So this arm
    measures the INTAKE and stops before ``process_weights_after_loading``.
    Vendoring the runtime is forbidden (AGENTS.md), which is why the seam is a
    stub here and the load-and-execute contract is measured on the pinned image
    by ``experiments/nvfp4_moe_route_load_probe.py`` instead.
    """
    import enum
    import types

    import torch

    names = ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fused_moe",
             "vllm.model_executor.layers.fused_moe.fused_moe_method_base",
             "vllm.model_executor.layers.fused_moe.oracle",
             "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
             "vllm.model_executor.layers.quantization",
             "vllm.model_executor.layers.quantization.utils",
             "vllm.model_executor.layers.quantization.utils.quant_utils",
             "vllm.model_executor.utils")
    already = [name for name in names if name in sys.modules]
    modules = {name: types.ModuleType(name) for name in names}
    sys.modules.update(modules)
    base = modules["vllm.model_executor.layers.fused_moe.fused_moe_method_base"]

    class Base:
        def __init__(self, moe):
            self.moe, self.moe_kernel, self.moe_quant_config = moe, None, None

        @property
        def is_monolithic(self):
            return False

    base.FusedMoEMethodBase = Base
    oracle = modules["vllm.model_executor.layers.fused_moe.oracle.nvfp4"]
    oracle.NvFp4MoeBackend = enum.Enum("NvFp4MoeBackend", ["FLASHINFER_CUTLASS", "MARLIN"])
    experts_cls = types.SimpleNamespace(is_monolithic=lambda: False)
    oracle.select_nvfp4_moe_backend = lambda **kwargs: (
        oracle.NvFp4MoeBackend.FLASHINFER_CUTLASS, experts_cls)
    oracle.convert_to_nvfp4_moe_kernel_format = lambda **kwargs: tuple(
        kwargs[key] for key in ("w13", "w13_scale", "w13_scale_2", "a13_scale",
                                "w2", "w2_scale", "w2_scale_2", "a2_scale"))
    oracle.make_nvfp4_moe_quant_config = lambda **kwargs: kwargs
    oracle.make_nvfp4_moe_kernel = lambda **kwargs: types.SimpleNamespace(
        fused_experts=types.SimpleNamespace(process_weights_after_loading=lambda layer: None))
    quant = modules["vllm.model_executor.layers.quantization.utils.quant_utils"]
    quant.kNvfp4Static, quant.kNvfp4Dynamic = object(), object()
    utils = modules["vllm.model_executor.utils"]
    utils.set_weight_attrs = lambda param, attrs: [setattr(param, k, v) for k, v in attrs.items()]
    utils.replace_parameter = lambda layer, name, value: setattr(
        layer, name, torch.nn.Parameter(value, requires_grad=False))
    return {"stubbed": sorted(names), "was_already_imported": already}


def _nvfp4_layer(rank: int, tp_size: int, experts: int, topk: int, swiglu_limit: float):
    """The ``RoutedExperts`` surface the route reads, as the route's tests build it."""
    import types

    import torch

    layer = torch.nn.Module()
    layer.moe_config = types.SimpleNamespace(
        is_act_and_mul=True, experts_per_token=topk,
        moe_parallel_config=types.SimpleNamespace(
            tp_size=tp_size, tp_rank=rank, ep_size=1, dp_size=1, pcp_size=1, sp_size=1,
            use_ep=False, enable_eplb=False))
    layer.activation = "silu"
    layer.global_num_experts = experts
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    layer._expert_routing_tables = lambda: None
    layer.swiglu_limit = swiglu_limit
    return layer


def _nvfp4_resident(layer) -> int:
    """The stock tile set this layer holds: the route's resident state."""
    return sum(getattr(layer, name).numel() * getattr(layer, name).element_size()
               for name in NVFP4_TILES)


def _child_nvfp4(args, torch, device, layers, experts, out) -> int:
    """Load every named layer's experts through the NVFP4 routed builder.

    Each layer's prepared state is the stock tile set, held exactly as vLLM
    holds it until ``process_weights_after_loading``.  Nothing is finalized and
    no layer is dropped, so ``resident_final`` is what one rank's routed
    experts occupy once the last layer has loaded.
    """
    from tessera.serving import nvfp4_moe_route

    seam = _stub_vllm_oracle()
    data = Path(args.data)
    index = _checkpoint_index(data, layers)
    missing = [layer for layer in layers if layer not in index]
    if missing:
        raise SystemExit(f"no expert wires for layer(s) {missing[:8]} under {data}")
    tp_size = 1 if args.config == "tp1" else 2
    rank = 1 if args.config == "tp2r1" else 0
    floor = int(args.guard_floor_gib * (1 << 30))

    units, held_layers = [], {}
    resident = wire_total = 0
    stopped = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = _allocator(torch)
    sampler = _StackSampler() if args.profile == "sample" else None
    profiler, profiler_state, profiled = None, "off", 0
    if args.profile == "torch":
        from torch.profiler import ProfilerActivity, profile

        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        profiler_state = "armed"
    load_start = time.time()
    if sampler is not None:
        sampler.start()
    for layer, wires, wire_bytes in _layer_stream(index, layers, experts, args.read_ahead):
        wire_total += wire_bytes
        directory, _map = index[layer]
        scheme, declared = _expert_scheme(directory, layer)
        if int(declared["experts"]) != experts:
            raise SystemExit(f"layer {layer}: the sidecar declares {declared['experts']} "
                             f"experts, the run asked for {experts}")
        hidden, inter = int(declared["hidden_size"]), int(declared["intermediate_size"])
        holder = _nvfp4_layer(rank, tp_size, experts, args.topk, args.swiglu_limit)
        method = nvfp4_moe_route.build_tessera_nvfp4_moe_method(
            scheme, _target(layer), "resident", holder)
        torch.cuda.synchronize()
        start = time.perf_counter()
        # vLLM builds its model under a default-device context; without one,
        # ``create_weights``' bare ``torch.zeros`` lands the tiles on the host,
        # every unit still "succeeds", and the footprint reads as nearly zero.
        with torch.device(device):
            method.create_weights(holder, experts, hidden, inter // tp_size, torch.bfloat16)
        torch.cuda.synchronize()
        create_seconds = time.perf_counter() - start
        for name in ("w13_weight", "w2_weight"):
            if getattr(holder, name).device.type != "cuda":
                raise SystemExit(f"layer {layer}: {name} was allocated on "
                                 f"{getattr(holder, name).device}, not {device}")
        held_layers[layer] = holder
        layer_resident = _nvfp4_resident(holder)
        resident += layer_resident
        units.append({"layer": layer, "phase": "create_weights", "seconds": create_seconds,
                      "layer_resident": int(layer_resident),
                      "resident_cumulative": int(resident), **_allocator(torch)})
        for expert in range(experts):
            available, stalled = _host_pressure()
            if available < floor or stalled >= args.guard_psi:
                stopped = {"why": "host memory guard", "layer": layer, "expert": expert,
                           "mem_available": available, "psi_full_avg10": stalled,
                           "floor_bytes": floor, "psi_limit": args.guard_psi}
                break
            for shard, _projection in NVFP4_SHARDS:
                group = "w2" if shard == "w2" else "w13"
                wire = wires.pop((expert, shard, "wire"))
                scale = wires.pop((expert, shard, "input_global_scale"))
                param = holder.w2_wire if group == "w2" else holder.w13_wire
                if profiler_state == "armed":
                    profiler.__enter__()
                    profiler_state = "running"
                torch.cuda.synchronize()
                start = time.perf_counter()
                param.weight_loader(param, wire, "wire", shard, expert, return_success=True)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                if profiler_state == "running":
                    profiled += 1
                    if profiled == args.profile_units:
                        profiler.__exit__(None, None, None)
                        _export_torch_profile(profiler, out, profiled)
                        profiler_state = "exported"
                scale_param = (holder.w2_input_global_scale if group == "w2"
                               else holder.w13_input_global_scale)
                scale_param.weight_loader(scale_param, scale, "input_global_scale", shard, expert)
                units.append({"layer": layer, "expert": expert, "group": group, "shard": shard,
                              "wire_bytes": int(wire.numel()), "seconds": seconds,
                              "resident_cumulative": int(resident), **_allocator(torch)})
                del wire, scale
            if args.stop_after and len(units) >= args.stop_after:
                stopped = {"why": "stop_after", "layer": layer, "expert": expert}
                break
        del wires
        if stopped is not None:
            break
    load_seconds = time.time() - load_start
    if sampler is not None:
        sampler.finish(out / "sample.collapsed")
    if profiler_state == "running":
        profiler.__exit__(None, None, None)
        _export_torch_profile(profiler, out, max(profiled, 1))

    # The decode wrote real bytes, or it did not: a zero tile is what a stubbed
    # or misrouted decode leaves behind, and the joined per-expert global is the
    # multiplier the kernel would be handed.
    evidence = {}
    for layer in dict.fromkeys([layers[0], layers[-1]]):
        holder = held_layers.get(layer)
        if holder is None:
            continue
        evidence[str(layer)] = {
            "w13_weight_nonzero_fraction": float((holder.w13_weight[0] != 0).float().mean()),
            "w2_weight_nonzero_fraction": float((holder.w2_weight[0] != 0).float().mean()),
            "w13_weight_scale_2_expert0": holder.w13_weight_scale_2[0].tolist(),
            "w2_weight_scale_2_expert0": float(holder.w2_weight_scale_2[0]),
            "w13_input_global_scale_all_finite": bool(
                torch.isfinite(holder.w13_input_global_scale).all()),
            "tile_shapes": {name: list(getattr(holder, name).shape) for name in NVFP4_TILES},
            "tile_devices": sorted({str(getattr(holder, name).device) for name in NVFP4_TILES}),
        }
    available, stalled = _host_pressure()
    summary = {
        "arm": "nvfp4_routed_intake", "family": "TESSERA_NVFP4", "config": args.config,
        "src": str(Path(args.src).resolve()), "data": str(data),
        "layers_requested": layers, "layers_loaded": sorted(held_layers),
        "experts": experts, "tp_size": tp_size, "tp_rank": rank,
        "wire_bytes_loaded": int(wire_total), "load_seconds": load_seconds,
        "baseline": baseline, "final": _allocator(torch),
        "resident_final": int(resident),
        "resident_per_layer": int(resident // max(len(held_layers), 1)),
        "unit_seconds": _unit_seconds(units), "stopped_early": stopped,
        "vllm_seam": seam, "evidence": evidence,
        "scope": ("Expert intake only: create_weights and _load_wire, on one rank. "
                  "NOT process_weights_after_loading (the runtime's kernel-format "
                  "swizzle), no engine, no KV cache, no activation peak, no "
                  "non-expert weights, no collective buffers."),
        "mem_available_end": available, "psi_full_avg10_end": stalled,
        "host": platform.node(), "torch": torch.__version__,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "t_start": load_start, "t_end": time.time(),
    }
    (out / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return 1 if (stopped or {}).get("why") == "host memory guard" else 0


class _StackSampler(threading.Thread):
    """Samples the main thread's Python stack at a fixed rate, in process."""

    def __init__(self, hz: int = 100):
        super().__init__(daemon=True)
        self.interval = 1.0 / hz
        self.counts: "collections.Counter[str]" = collections.Counter()
        self.halt = threading.Event()
        self.target = threading.main_thread().ident

    @staticmethod
    def _frame_name(frame):
        code = frame.f_code
        filename = code.co_filename
        if "/src/tessera/" in filename:
            filename = "tessera/" + filename.split("/src/tessera/", 1)[1]
        return f"{code.co_name} ({filename}:{frame.f_lineno})"

    def run(self):
        while not self.halt.wait(self.interval):
            frame = sys._current_frames().get(self.target)
            stack = []
            while frame is not None:
                stack.append(self._frame_name(frame))
                frame = frame.f_back
            if stack:
                self.counts[";".join(reversed(stack))] += 1

    def finish(self, path: Path):
        self.halt.set()
        self.join()
        path.write_text("".join(f"{stack} {count}\n" for stack, count in self.counts.items()))


def _unit_seconds(units):
    by_group = collections.defaultdict(list)
    for unit in units:
        if "group" in unit:
            by_group[unit["group"]].append(unit["seconds"])
            by_group["all"].append(unit["seconds"])
    out = {}
    for group, values in by_group.items():
        ordered = sorted(values)
        out[group] = {"n": len(values), "total": sum(values), "mean": statistics.fmean(values),
                      "median": statistics.median(values),
                      "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]}
    return out


def child(args) -> int:
    src = Path(args.src).resolve()
    sys.path.insert(0, str(src))
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    import tessera
    from tessera.serving import moe_route

    if not Path(tessera.__file__).resolve().is_relative_to(src):
        raise SystemExit(f"tessera imported from {tessera.__file__}, not {src}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    layers = (_moe_layers(data) if args.layers == "all"
              else [int(x) for x in args.layers.split(",")])
    experts = int(args.experts)
    device = torch.device("cuda", torch.cuda.current_device())
    # Each family's own builder owns its intake: the NVFP4 route decodes into a
    # preallocated stock tile, the research-selected packed route keeps wire
    # planes on an expert axis.  Dispatch on what the sidecar declares.
    if _family(data, layers) == "TESSERA_NVFP4":
        return _child_nvfp4(args, torch, device, layers, experts, out)
    t_read = time.time()
    wires = _load_wires(data, layers, experts)
    read_seconds = time.time() - t_read
    wire_total = sum(t.numel() for layer in wires.values() for e in layer.values() for t in e.values())
    declared = _schemes(data, layers, experts)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = _allocator(torch)
    units = []
    prepared = {}
    profiler = None
    profiler_state = "off"
    if args.profile == "torch":
        from torch.profiler import ProfilerActivity, profile

        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        profiler_state = "armed"
    sampler = _StackSampler() if args.profile == "sample" else None
    config = args.config
    load_start = time.time()
    if sampler is not None:
        sampler.start()
    if config == "tp1":
        for layer in layers:
            blobs = {"w13": [[wires[layer][e]["gate_proj"].numpy().tobytes(),
                              wires[layer][e]["up_proj"].numpy().tobytes()] for e in range(experts)],
                     "w2": [[wires[layer][e]["down_proj"].numpy().tobytes()] for e in range(experts)]}
            torch.cuda.synchronize()
            start = time.perf_counter()
            prepared[layer] = moe_route.prepare_tessera_packed_moe_experts(
                blobs, declared[layer], _target(layer), device=device, tp_rank=0, tp_size=1)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            del blobs
            units.append({"layer": layer, "phase": "prepare_layer", "wires": 3 * experts,
                          "wire_bytes": sum(t.numel() for e in wires[layer].values() for t in e.values()),
                          "seconds": seconds, "resident_cumulative": sum(
                              p.resident_bytes() for p in prepared.values()), **_allocator(torch)})
        finish_seconds = 0.0
    else:
        rank = int(config[-1])
        intakes, lengths = {}, {}
        profiled = 0
        base_resident = 0
        stop = False
        for layer in layers:
            if stop:
                break
            intakes[layer] = moe_route._RankLocalPackedIntake(declared[layer], _target(layer),
                                                             device, rank, 2)
            lengths[layer] = (torch.zeros(experts, 2, dtype=torch.long),
                              torch.zeros(experts, dtype=torch.long))
            for expert in range(experts):
                if stop:
                    break
                for group, index, proj in MOE_UNITS:
                    wire = wires[layer][expert][proj]
                    if profiler_state == "armed":
                        profiler.__enter__()
                        profiler_state = "running"
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    intakes[layer].load(group, index, expert, wire, device=device)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    if profiler_state == "running":
                        profiled += 1
                        if profiled == args.profile_units:
                            profiler.__exit__(None, None, None)
                            _export_torch_profile(profiler, out, profiled)
                            profiler_state = "exported"
                    if group == "w13":
                        lengths[layer][0][expert, index] = wire.numel()
                    else:
                        lengths[layer][1][expert] = wire.numel()
                    intake = intakes[layer]
                    if hasattr(intake, "resident_bytes"):
                        resident = sum(i.resident_bytes() for i in intakes.values())
                    else:
                        base_resident += _module_resident(intake.prepared[group][expert][index])
                        resident = base_resident
                    units.append({"layer": layer, "expert": expert, "group": group, "index": index,
                                  "wire_bytes": int(wire.numel()), "seconds": seconds,
                                  "resident_cumulative": int(resident), **_allocator(torch)})
                    if args.stop_after and len(units) >= args.stop_after:
                        stop = True
                        break
        finish_seconds = 0.0
        if not args.stop_after:
            for layer in layers:
                torch.cuda.synchronize()
                start = time.perf_counter()
                prepared[layer] = intakes[layer].finish(*lengths[layer])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                finish_seconds += seconds
                intakes[layer] = None
                units.append({"layer": layer, "phase": "finish", "seconds": seconds,
                              "resident_cumulative": sum(p.resident_bytes() for p in prepared.values()),
                              **_allocator(torch)})
    load_seconds = time.time() - load_start
    if sampler is not None:
        sampler.finish(out / "sample.collapsed")
    summary = {
        "config": config, "src": str(src), "tessera_file": tessera.__file__,
        "layers": layers, "experts": experts, "read_seconds": read_seconds,
        "wire_bytes_loaded": int(wire_total), "load_seconds": load_seconds,
        "finish_seconds": finish_seconds, "baseline": baseline, "final": _allocator(torch),
        "resident_final": int(sum(p.resident_bytes() for p in prepared.values())),
        "unit_seconds": _unit_seconds(units),
        "torch": torch.__version__, "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "host": platform.node(), "t_start": load_start, "t_end": time.time(),
    }
    (out / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    if args.dump and prepared:
        records, tensors = {}, {}
        for layer in layers:
            _walk(torch, prepared[layer], f"L{layer}", records, tensors, args.dump_experts, experts)
        (out / "prepared.json").write_text(json.dumps(records, indent=1, sort_keys=True))
        torch.save(tensors, out / "prepared_subset.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def _export_torch_profile(prof, out: Path, units: int):
    table = prof.key_averages()
    (out / "torch_profile_self_cpu.txt").write_text(
        table.table(sort_by="self_cpu_time_total", row_limit=45))
    (out / "torch_profile_count.txt").write_text(table.table(sort_by="count", row_limit=45))
    counts = {e.key: int(e.count) for e in table}
    keep = {k: v for k, v in counts.items()
            if k in ("cudaLaunchKernel", "aten::_local_scalar_dense", "aten::item", "cudaMemcpy",
                     "aten::copy_", "cudaSynchronize", "aten::index_put_", "aten::index_select",
                     "aten::empty", "aten::zeros", "aten::cat", "aten::stack", "aten::clone")}
    (out / "torch_profile_counts.json").write_text(json.dumps(
        {"units": units, "counts": keep, "per_unit": {k: v / units for k, v in keep.items()},
         "all_events": sum(counts.values())},
        indent=1, sort_keys=True))


# --------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------

class PowerSampler(threading.Thread):
    def __init__(self, path: Path, interval_ms: int = 250):
        super().__init__(daemon=True)
        self.path, self.interval_ms = path, interval_ms
        self.samples: "list[tuple[float, float]]" = []
        self.proc = None

    def run(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                 f"-lms={self.interval_ms}"], stdout=subprocess.PIPE, text=True)
        except OSError as exc:
            self.path.write_text(f"nvidia-smi unavailable: {exc}\n")
            return
        with self.path.open("w") as handle:
            for line in self.proc.stdout:
                now = time.time()
                try:
                    watts = float(line.strip())
                except ValueError:
                    continue
                self.samples.append((now, watts))
                handle.write(f"{now:.3f},{watts}\n")

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()

    def window(self, t0: float, t1: float):
        inside = [(t, w) for t, w in self.samples if t0 <= t <= t1]
        if len(inside) < 2:
            return {"samples": len(inside)}
        watts = [w for _, w in inside]
        joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(inside, inside[1:]))
        return {"samples": len(inside), "mean_w": statistics.fmean(watts), "max_w": max(watts),
                "p50_w": statistics.median(watts), "joules": joules,
                "envelope_fraction_mean": statistics.fmean(watts) / 140.0}


def _tree_identity(src: Path):
    tree = src.parent
    ident = {"src": str(src)}
    commit = tree / "COMMIT"
    if commit.exists():
        ident["commit_file"] = commit.read_text().strip()
    for key, cmd in (("head", ["git", "-C", str(tree), "rev-parse", "HEAD"]),
                     ("dirty", ["git", "-C", str(tree), "status", "--porcelain", "--", "src"])):
        try:
            ident[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:  # noqa: BLE001 -- identity is recorded, not required
            ident[key] = f"unavailable: {exc}"
    digest = hashlib.sha256()
    for path in sorted(src.rglob("*.py")):
        digest.update(str(path.relative_to(src)).encode())
        digest.update(path.read_bytes())
    ident["src_py_sha256"] = digest.hexdigest()
    return ident


def _run_child(args, arm, src, config, out, *, dump, profile="none", profile_units=0,
               layers=None, experts=None, stop_after=0):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    cmd = [sys.executable, str(Path(__file__).resolve()), "child", "--src", str(src),
           "--data", args.data, "--layers", layers or args.layers,
           "--experts", str(experts or args.experts), "--config", config, "--out", str(out),
           "--dump-experts", str(args.dump_experts), "--profile", profile,
           "--profile-units", str(profile_units), "--stop-after", str(stop_after)]
    if dump:
        cmd.append("--dump")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (out / "child.log").open("w") as log:
        rc = subprocess.run(cmd, env=env, cwd=str(out), stdout=log, stderr=subprocess.STDOUT).returncode
    return rc, t0, time.time()


def _compare(out: Path, arms, configs):
    import torch

    verdict = {}
    first = arms[0][0]
    for config in configs:
        ref_dir = out / f"rep0-{config}-{first}"
        if not (ref_dir / "prepared.json").exists():
            verdict[config] = {"error": f"no dump for {first}"}
            continue
        ref = json.loads((ref_dir / "prepared.json").read_text())
        ref_t = torch.load(ref_dir / "prepared_subset.pt")
        for arm, _src in arms[1:]:
            other_dir = out / f"rep0-{config}-{arm}"
            if not (other_dir / "prepared.json").exists():
                verdict[f"{config}:{arm}"] = {"error": "no dump"}
                continue
            got = json.loads((other_dir / "prepared.json").read_text())
            got_t = torch.load(other_dir / "prepared_subset.pt")
            mismatches, notes = [], []
            for path in sorted(set(ref) | set(got)):
                a, b = ref.get(path), got.get(path)
                if a is None or b is None:
                    mismatches.append({"path": path, "why": "missing", "ref": a, "got": b})
                    continue
                if a.get("kind") == "device":
                    if a["type"] != b.get("type"):
                        mismatches.append({"path": path, "why": "device kind", "ref": a, "got": b})
                    elif a["str"] != b.get("str"):
                        notes.append({"path": path, "device_str": [a["str"], b.get("str")]})
                    continue
                if a != b:
                    mismatches.append({"path": path, "why": "record", "ref": a, "got": b})
            equal_checks = 0
            for path in sorted(set(ref_t) | set(got_t)):
                x, y = ref_t.get(path), got_t.get(path)
                if x is None or y is None:
                    mismatches.append({"path": path, "why": "subset missing"})
                    continue
                equal_checks += 1
                if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                    mismatches.append({"path": path, "why": "torch.equal/dtype/shape"})
            verdict[f"{config}:{arm}"] = {"records": len(ref), "torch_equal_checks": equal_checks,
                                          "tensors_digested": sum(1 for r in ref.values()
                                                                  if r.get("kind") == "tensor"),
                                          "mismatches": mismatches[:50],
                                          "mismatch_count": len(mismatches), "notes": notes[:10]}
    return verdict


def _summarise_stacks(path: Path):
    if not path.exists():
        return None
    self_c, incl, total = collections.Counter(), collections.Counter(), 0
    for line in path.read_text().splitlines():
        stack, _, n = line.rstrip().rpartition(" ")
        try:
            n = int(n)
        except ValueError:
            continue
        frames = stack.split(";")
        total += n
        self_c[frames[-1]] += n
        for frame in set(frames):
            incl[frame] += n
    if not total:
        return {"samples": 0}
    return {"samples": total,
            "self": [[round(c / total, 4), f] for f, c in self_c.most_common(30)],
            "inclusive_tessera": [[round(c / total, 4), f] for f, c in incl.most_common(400)
                                  if f.split(" (", 1)[-1].startswith("tessera/")][:45]}


def parent(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = []
    for spec in args.arms.split(","):
        name, _, src = spec.partition("=")
        arms.append((name, Path(src).resolve()))
    configs = args.configs.split(",")
    meta = {"argv": sys.argv, "arms": {n: _tree_identity(s) for n, s in arms},
            "host": platform.node(), "python": sys.executable,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "t_start": time.time()}
    try:
        meta["nvidia_smi_L"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                              text=True, timeout=30).stdout
    except Exception as exc:  # noqa: BLE001
        meta["nvidia_smi_L"] = str(exc)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    sampler = PowerSampler(out / "power.csv")
    sampler.start()
    runs = []
    failed = False
    for rep in range(args.repeats):
        order = arms if rep % 2 == 0 else list(reversed(arms))
        for config in configs:
            for arm, src in order:
                run_dir = out / f"rep{rep}-{config}-{arm}"
                rc, t0, t1 = _run_child(args, arm, src, config, run_dir, dump=(rep == 0))
                row = {"rep": rep, "config": config, "arm": arm, "rc": rc, "t0": t0, "t1": t1,
                       "power": sampler.window(t0, t1)}
                summary_path = run_dir / "summary.json"
                if rc == 0 and summary_path.exists():
                    row["summary"] = json.loads(summary_path.read_text())
                    lo = row["summary"]["t_start"]
                    hi = lo + row["summary"]["load_seconds"]
                    row["power_load"] = sampler.window(lo, hi)
                else:
                    failed = True
                runs.append(row)
                print(json.dumps({k: row[k] for k in ("rep", "config", "arm", "rc")},
                                 default=str), flush=True)
                (out / "runs.json").write_text(json.dumps(runs, indent=1, default=str))
    if args.profile:
        first_layer = args.layers.split(",")[0]
        for arm, src in arms:
            run_dir = out / f"profile-torch-{arm}"
            rc, t0, t1 = _run_child(args, arm, src, "tp2r1", run_dir, dump=False, profile="torch",
                                    profile_units=args.profile_units, layers=first_layer,
                                    stop_after=args.profile_units)
            runs.append({"profile": "torch", "arm": arm, "rc": rc, "t0": t0, "t1": t1})
            failed = failed or bool(rc)
            run_dir = out / f"profile-sample-{arm}"
            rc, t0, t1 = _run_child(args, arm, src, "tp2r1", run_dir, dump=False, profile="sample",
                                    layers=first_layer)
            runs.append({"profile": "sample", "arm": arm, "rc": rc, "t0": t0, "t1": t1,
                         "power": sampler.window(t0, t1),
                         "stacks": _summarise_stacks(run_dir / "sample.collapsed")})
            failed = failed or bool(rc)
            (out / "runs.json").write_text(json.dumps(runs, indent=1, default=str))
    sampler.stop()
    verdict = _compare(out, arms, configs)
    (out / "compare.json").write_text(json.dumps(verdict, indent=1, default=str))
    bad = any(v.get("mismatch_count", 1) for v in verdict.values())
    meta["t_end"] = time.time()
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"failed_children": failed, "compare_mismatch": bad,
                      "compare": {k: v.get("mismatch_count", v.get("error")) for k, v in verdict.items()}}),
          flush=True)
    return 1 if (failed or bad) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("parent")
    p.add_argument("--arms", required=True, help="name=src[,name=src]; the first is the reference")
    p.add_argument("--configs", default="tp2r0,tp2r1,tp1")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile", action="store_true", help="also run torch.profiler and sampler children")
    p.add_argument("--profile-units", type=int, default=48)
    c = sub.add_parser("child")
    c.add_argument("--src", required=True)
    c.add_argument("--config", required=True, choices=("tp1", "tp2r0", "tp2r1"))
    c.add_argument("--dump", action="store_true")
    c.add_argument("--profile", default="none", choices=("none", "torch", "sample"))
    c.add_argument("--profile-units", type=int, default=0)
    c.add_argument("--stop-after", type=int, default=0)
    for q in (p, c):
        q.add_argument("--data", required=True)
        q.add_argument("--layers", default="3,40",
                       help="comma-separated layer indices, or 'all' for every layer whose "
                            "experts carry Tessera wires")
        q.add_argument("--experts", type=int, default=288)
        q.add_argument("--dump-experts", type=int, default=4)
        q.add_argument("--out", required=True)
        # NVFP4 routed arm
        q.add_argument("--read-ahead", type=int, default=1,
                       help="layers of expert wire read ahead of the decode (NVFP4 arm)")
        q.add_argument("--guard-floor-gib", type=float, default=16.0,
                       help="stop and report when host MemAvailable falls below this")
        q.add_argument("--guard-psi", type=float, default=20.0,
                       help="stop and report when memory PSI full avg10 reaches this")
        q.add_argument("--topk", type=int, default=8,
                       help="experts per token the layer stub declares (GLM-5.3-Flash: 8)")
        q.add_argument("--swiglu-limit", type=float, default=10.0,
                       help="the clamp the layer stub carries (GLM-5.3-Flash: 10.0)")
    args = parser.parse_args(argv)
    return parent(args) if args.mode == "parent" else child(args)


if __name__ == "__main__":
    raise SystemExit(main())
