#!/usr/bin/env python3
"""Load small projection artifacts and compare eager or CUDA graph forwards.

The CPU dry run reads every input and validates its wire, roles, and shapes.
It does not claim a device forward. Device runs use the current vLLM classes,
the public install hook, and each parameter's real loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fused_bound as fb

from tools.tessera_construction_census import _checkpoint_name, _free_port

# These are bounded test shapes, not a serving construction contract.
CASES = (
    ("language_model.model.layers.0.self_attn.in_proj_qkvbfg_a", "kda", 256,
     (("q_proj", 256), ("k_proj", 256), ("v_proj", 256), ("b_proj", 64), ("f_a_proj", 128), ("g_a_proj", 128))),
    ("language_model.model.layers.0.self_attn.f_b_proj", "column", 128, (("f_b_proj", 256),)),
    ("language_model.model.layers.0.self_attn.g_b_proj", "column", 128, (("g_b_proj", 256),)),
    ("language_model.model.layers.0.self_attn.o_proj", "row", 256, (("o_proj", 256),)),
    ("language_model.model.layers.1.self_attn.fused_qkv_a_proj", "mla_input", 256,
     (("q_a_proj", 256), ("kv_a_proj_with_mqa", 256))),
    ("language_model.model.layers.1.self_attn.q_b_proj", "column", 256, (("q_b_proj", 256),)),
    ("language_model.model.layers.2.self_attn.q_proj", "column", 256, (("q_proj", 256),)),
    ("language_model.model.layers.1.self_attn.o_proj", "row", 256, (("o_proj", 256),)),
    ("language_model.model.layers.1.self_attn.kv_b_proj", "mla", 256, (("kv_b_proj", 512),)),
    ("language_model.model.layers.1.self_attn.indexer.wq_b", "replicated", 256, (("wq_b", 256),)),
    ("language_model.model.layers.1.self_attn.indexer.wk_weights_proj", "indexer", 256,
     (("wk", 128), ("weights_proj", 32))),
    ("language_model.model.layers.1.mlp.gate", "router", 256, (("gate", 32),)),
    ("visual.blocks.0.attn.qkv", "qkv", 256, (("q", 256), ("k", 256), ("v", 256))),
    ("visual.blocks.0.attn.proj", "row", 256, (("proj", 256),)),
    ("visual.blocks.0.mlp.gate_up_proj", "merged", 256, (("gate_proj", 256), ("up_proj", 256))),
    ("visual.blocks.0.mlp.down_proj", "row", 256, (("down_proj", 256),)),
    ("visual.merger.proj", "column", 256, (("proj", 256),)),
    ("visual.merger.gate_up_proj", "merged", 256, (("gate_proj", 256), ("up_proj", 256))),
    ("visual.merger.down_proj", "row", 256, (("down_proj", 256),)),
)
FAMILIES = {"T-8": ("TESSERA_FP8", "E4M3"), "T-16": ("TESSERA_BF16", "BF16")}


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _unit(rows, columns, grid, seed):
    """Construct a real current-format window unit without a weight fit."""
    import torch
    from tessera.encode import EncodedUnit, window_table
    from tessera.manifest import BodyKind, ScalePlaneKind

    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 16, (rows, columns), generator=generator, dtype=torch.uint8)
    states = torch.empty((rows, columns), dtype=torch.int64)
    state = torch.zeros(columns, dtype=torch.int64)
    for row in range(rows):
        state = ((state << 4) | bits[row].long()) & ((1 << 14) - 1)
        states[row] = state
    table = window_table(grid, 14, seed=0)
    empty_byte = torch.empty(0, dtype=torch.uint8)
    empty_long = torch.empty(0, dtype=torch.int64)
    return EncodedUnit(
        rates=(4,) * columns, anchors=states, codes=table.long()[states],
        body_bits=bits, completion_bits=torch.zeros_like(states),
        scale_base=empty_byte, scale_refine=empty_byte.clone(),
        release_index=empty_long, release_code=empty_long.clone(), sse=0.0,
        body=BodyKind.WINDOW, window_bits=14, window_codes=table,
        scale_plane=ScalePlaneKind.CHANNEL,
        scale_rows=torch.linspace(0.0125, 0.0375, rows).half(), scale_global=1.0,
        completion_limit=0,
    )


def prepare(root):
    """Write deterministic test artifacts through the production wire writer."""
    import torch
    from safetensors.torch import save_file
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.fused import pack_fused
    from tessera.unit_artifact import build_unit_artifact

    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "tessera.projection-smoke.v1", "modules": []}
    for index, (prefix, kind, columns, roles) in enumerate(CASES):
        name = f"module-{index:02d}"
        tensors = {}
        entry = {"prefix": prefix, "kind": kind, "columns": columns,
                 "roles": [list(role) for role in roles], "file": name + ".safetensors", "routes": {}}
        for label, (family, grid_name) in FAMILIES.items():
            grid = E4M3_GRID if grid_name == "E4M3" else BF16_GRID
            parts = []
            for role_index, (role, rows) in enumerate(roles):
                unit = _unit(rows, columns, grid, 211 + index * 17 + role_index)
                _, _, blob = build_unit_artifact(unit, role, grid, 1024, fixture_id=None)
                parts.append((role, rows, blob))
            wire = pack_fused(parts)
            tensors[label] = torch.frombuffer(bytearray(wire), dtype=torch.uint8).clone()
            entry["routes"][label] = {
                "sha256": hashlib.sha256(wire).hexdigest(),
                "scheme": {"family": family, "grid": grid_name, "body": "WINDOW",
                           "plane": "CHANNEL", "q256": 1024, "rows": sum(r for _, r in roles),
                           "columns": columns, "wire_bytes": len(wire), "roles": entry["roles"]},
            }
        if prefix.startswith("visual."):
            tensors["bias"] = torch.linspace(-0.25, 0.25, sum(r for _, r in roles)).bfloat16()
        tensors["input"] = torch.randn(3, columns, generator=torch.Generator().manual_seed(index + 31)).bfloat16()
        save_file(tensors, str(root / entry["file"]))
        manifest["modules"].append(entry)
    _json(root / "smoke.json", manifest)
    return root / "smoke.json"


def read_inputs(path, m):
    import torch
    from safetensors import safe_open
    from tessera.serving.scheme import parse_tessera_blob_for_scheme, validate_tessera_scheme

    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "tessera.projection-smoke.v1":
        raise ValueError("The artifact manifest has an unsupported schema")
    expected = {prefix: (kind, columns, [list(r) for r in roles]) for prefix, kind, columns, roles in CASES}
    modules = manifest["modules"]
    if len(modules) != len(expected) or {row["prefix"] for row in modules} != set(expected):
        raise ValueError("The artifact must contain every projection exactly once")
    loaded = []
    for row in modules:
        if (row["kind"], row["columns"], row["roles"]) != expected[row["prefix"]]:
            raise ValueError(f"{row['prefix']}: the input shape or role order differs")
        file = path.parent / row["file"]
        if file.resolve().parent != path.parent.resolve():
            raise ValueError("The input file must remain in the artifact directory")
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            x = handle.get_tensor("input")
            if x.dtype != torch.bfloat16 or tuple(x.shape) != (3, row["columns"]):
                raise ValueError(f"{row['prefix']}: invalid input tensor")
            item = {"row": row, "input": x[:m].contiguous(), "wires": {}, "parsed": {}}
            if row["prefix"].startswith("visual."):
                bias = handle.get_tensor("bias")
                if bias.dtype != torch.bfloat16 or tuple(bias.shape) != (sum(r for _, r in row["roles"]),):
                    raise ValueError(f"{row['prefix']}: invalid vision bias")
                if not bool(bias.abs().sum() > 0):
                    raise ValueError("The vision bias control must be nonzero")
                item["bias"] = bias
            for label, (family, grid) in FAMILIES.items():
                route = row["routes"][label]
                declared = validate_tessera_scheme(route["scheme"], row["prefix"])
                if (declared["family"], declared["grid"], declared["columns"], list(map(list, declared["roles"]))) != (family, grid, row["columns"], row["roles"]):
                    raise ValueError(f"{row['prefix']}: the wire declaration differs from the input")
                tensor = handle.get_tensor(label)
                if tensor.dtype != torch.uint8 or tensor.ndim != 1:
                    raise ValueError("The wire must be a byte vector")
                wire = bytes(tensor.numpy().tobytes())
                if hashlib.sha256(wire).hexdigest() != route["sha256"]:
                    raise ValueError(f"{row['prefix']}: wire byte integrity failed")
                item["wires"][label] = wire
                item["parsed"][label] = parse_tessera_blob_for_scheme(wire, route["scheme"], row["prefix"], device="cpu")
        loaded.append(item)
    return loaded


def _constructor(row):
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.models.glm5next.common.kda import _Glm5NextMergedColumnParallelLinear
    from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear
    return {"column": linear.ColumnParallelLinear, "mla": linear.ColumnParallelLinear,
            "row": linear.RowParallelLinear, "replicated": linear.ReplicatedLinear,
            "merged": linear.MergedColumnParallelLinear, "indexer": linear.MergedColumnParallelLinear,
            "qkv": linear.QKVParallelLinear, "router": GateLinear,
            "kda": _Glm5NextMergedColumnParallelLinear, "mla_input": DeepSeekV2FusedQkvAProjLinear}[row["kind"]]


def _construct(row, world, rank):
    import torch
    cls = _constructor(row)
    columns, sizes = row["columns"], [r for _, r in row["roles"]]
    kw = {"bias": row["prefix"].startswith("visual."), "params_dtype": torch.bfloat16,
          "quant_config": None, "prefix": row["prefix"]}
    kind = row["kind"]
    if kind == "mla_input":
        from tools.tessera_construction_census import _set_default_torch_dtype
        with _set_default_torch_dtype()(torch.bfloat16):
            return cls(columns, sizes, quant_config=None, prefix=row["prefix"])
    if kind == "router":
        kw["out_dtype"] = torch.float32
    if kind == "kda":
        return cls(columns, sizes, replicated_shard_ids=(4, 5), tp_size=world, **kw)
    if kind == "qkv":
        return cls(columns, 128, 2, **kw)
    if kind in ("merged", "indexer"):
        if kind == "indexer":
            kw["disable_tp"] = True
        return cls(columns, sizes, **kw)
    if kind == "row":
        kw["input_is_parallel"] = False
    return cls(columns, sum(sizes), **kw)


def _dense_weight(parsed, family):
    import torch
    from tessera.serving import bf16_route, fp8_route
    if family == "TESSERA_FP8":
        module = fp8_route.prepare_tessera_fp8_module(parsed, device="cpu")
        return module.decode().view(torch.float8_e4m3fn).float() * module.row_scale().float()[:, None]
    module = bf16_route.prepare_tessera_bf16_module(parsed, device="cpu")
    return (module.decode().float() * module.row_scale().float()[:, None]).bfloat16().float()


def compare(got, expected, *, name, dtype, exact=False, bound=None):
    import torch
    if got.shape != expected.shape or got.dtype != dtype:
        raise AssertionError(f"{name}: output shape or dtype differs")
    if not bool(torch.isfinite(got).all()):
        raise AssertionError(f"{name}: the output contains a nonfinite value")
    ratio = None
    if exact:
        if not torch.equal(got.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
            raise AssertionError(f"{name}: output bits changed")
    else:
        if bound is None or bound.shape != expected.shape or expected.dtype != torch.float64:
            raise ValueError(f"{name}: the numeric check needs an FP64 reference and a derived bound")
        if not bool(torch.isfinite(bound).all()) or not bool((bound >= 0).all()):
            raise ValueError(f"{name}: the derived bound is invalid")
        ratio = fb.check_within(got, expected, bound, name)
    return {"shape": list(got.shape), "dtype": str(got.dtype),
            "max_abs_error": float((got.double() - expected.double()).abs().max()),
            "exact": exact, "worst_bound_ratio": ratio}


def _reference_weight(parsed, family, device):
    import torch
    from tessera.serving import fp8_route
    if family == "TESSERA_FP8":
        module = fp8_route.prepare_tessera_fp8_module(parsed, device="cpu")
        values = module.decode().view(torch.float8_e4m3fn).double().to(device)
        return values * module.row_scale().double().to(device)[:, None]
    return _dense_weight(parsed, family).double().to(device)


def _bias_bound(reference, error, bias):
    """Charge one FP32 add and one BF16 output cast after the dense cast."""
    absolute_sum = reference.abs() + error + bias.abs()
    added = reference + bias
    before_cast = error + (fb.gamma(1, fb.U32) + fb.gamma(1, fb.U64)) * absolute_sum
    return added, before_cast + 0.5 * fb.bf16_ulp(added.abs() + before_cast)


def _dense_reference(item, label, layer, x):
    import torch
    from tessera.serving.sharding import AXIS_COLUMNS, shard_parsed_roles
    family = FAMILIES[label][0]
    plan = layer.tessera_shard_plan
    parsed = shard_parsed_roles(item["parsed"][label], plan)
    local_input = x
    if plan.axis == AXIS_COLUMNS:
        ranges = {(role.lo, role.hi) for role in plan.roles}
        if len(ranges) != 1:
            raise ValueError("The row-parallel roles must share one input column range")
        lo, hi = ranges.pop()
        local_input = x[:, lo:hi].contiguous()
    if family == "TESSERA_FP8":
        from tessera.serving.native_ops import native_fp8_quant
        codes, scale = native_fp8_quant(local_input)
        left = codes.double() * scale.double().reshape(-1, 1)
    else:
        left = local_input.double()
    weight = _reference_weight(parsed, family, x.device)
    k = int(local_input.shape[-1])
    # K nonempty partials cover every valid prepared split and accumulation order.
    reference, error = fb.dense_bound("e4m3" if family == "TESSERA_FP8" else "value",
                                      left, weight, k, k)
    if "bias" in item and (plan.axis != AXIS_COLUMNS or plan.tp_rank == 0):
        reference, error = _bias_bound(reference, error, layer.bias.double())
    if plan.axis == AXIS_COLUMNS:
        peers = [None] * plan.tp_size
        torch.distributed.all_gather_object(peers, (reference.cpu(), error.cpu()))
        references = torch.stack([pair[0] for pair in peers], dim=1).to(x.device)
        errors = torch.stack([pair[1] for pair in peers], dim=1).to(x.device)
        reference, error = fb.route_sum_bound(references, errors, top_k_dim=1)
    return reference, error, {"owner": "tests/fused_bound.py", "K": k, "S_upper": k,
                               "rank_partials": plan.tp_size if plan.axis == AXIS_COLUMNS else 1,
                               "bias_additions": int("bias" in item)}



def _call(layer, x):
    value = layer(x)
    return value[0] if isinstance(value, tuple) else value


def resident_tensors(layer):
    tensors = list(layer.named_parameters()) + list(layer.named_buffers())
    native = getattr(layer, "tessera_native", None)
    if native is not None:
        tensors += [("native." + name, tensor) for name, tensor in native.named_tensors()]
    seen, total, rows = set(), {}, []
    for name, tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        unique = key not in seen
        if unique:
            seen.add(key)
            total[str(tensor.device)] = total.get(str(tensor.device), 0) + storage.nbytes()
        rows.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                     "device": str(tensor.device), "storage_bytes": storage.nbytes(), "unique_storage": unique})
    return {"bytes_by_device": total, "tensors": rows}

def _execute(call, x, mode):
    import torch
    eager = call(x)
    if mode == "eager":
        return eager, None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(x)
    torch.cuda.current_stream().wait_stream(stream)
    static = x.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call(static)
    graph.replay()
    first = captured.clone()
    static.copy_(x * 0.5)
    graph.replay()
    second = captured.clone()
    compare(first, eager, name="graph replay", dtype=eager.dtype, exact=True)
    compare(second, call(x * 0.5), name="changed graph input", dtype=eager.dtype, exact=True)
    return first, {"replays": 2, "changed_input": True}


def _load(layer, name, tensor):
    parameter = getattr(layer, name)
    parameter.weight_loader(parameter, tensor)


def _load_stock_weight(layer, row, weight):
    if row["kind"] not in ("kda", "merged", "indexer"):
        _load(layer, "weight", weight)
        return
    offset = 0
    for shard_id, (_, rows) in enumerate(row["roles"]):
        layer.weight.weight_loader(layer.weight, weight[offset:offset + rows], shard_id)
        offset += rows


def _stock_control(item, config, world, rank, original=None, mode="eager"):
    import torch
    from vllm.config import set_current_vllm_config
    row = item["row"]
    config.quant_config = None
    with set_current_vllm_config(config, check_compile=False), torch.device("cuda"):
        layer = _construct(row, world, rank)
    weight = _dense_weight(item["parsed"]["T-16"], "TESSERA_BF16").bfloat16().cuda()
    _load_stock_weight(layer, row, weight)
    if "bias" in item:
        _load(layer, "bias", item["bias"].cuda())
    layer.quant_method.process_weights_after_loading(layer)
    x = item["input"].cuda()
    got, graph = _execute(lambda value: _call(layer, value), x, mode)
    if original is None:
        return got, type(layer.quant_method), layer.prefix, layer.params_dtype
    expected, method, prefix, dtype = original
    if (type(layer.quant_method), layer.prefix, layer.params_dtype) != (method, prefix, dtype) or layer.quant_config is not None:
        raise AssertionError(f"{row['prefix']}: install changed the stock method")
    return {**compare(got, expected, name=row["prefix"], dtype=expected.dtype, exact=True), "graph": graph}

def _local_ip():
    """Return the host address other ranks can dial (no packet leaves)."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _tp2_master(world, rank, node_rank):
    """Share rank 0's reachable rendezvous address with the other rank.

    The actual vLLM image ignores the passed init method when nnodes > 1
    and dials ParallelConfig master_addr/master_port instead, so both ranks
    must use rank 0's address. Rank 0 writes it to the shared rendezvous
    file; rank 1 polls it. Single-node runs keep loopback.
    """
    import socket
    import time
    nnodes = int(os.environ.get("NNODES", "1"))
    if world < 2 or nnodes < 2:
        return os.environ.get("MASTER_ADDR", "127.0.0.1"), int(os.environ.get("MASTER_PORT", str(_free_port())))
    rendezvous = os.environ.get("TESSERA_TP2_RENDEZVOUS", "")
    if not rendezvous:
        raise ValueError("A two-node run needs TESSERA_TP2_RENDEZVOUS on a shared filesystem")
    if node_rank == 0:
        host = os.environ.get("MASTER_ADDR", "")
        if not host or host.startswith("127."):
            try:
                host = _local_ip()
            except OSError:
                host = socket.gethostbyname(socket.gethostname())
        port = int(os.environ.get("MASTER_PORT", str(_free_port())))
        Path(rendezvous).parent.mkdir(parents=True, exist_ok=True)
        Path(rendezvous).write_text(f"{host} {port}\n")
        print(f"[tp2] rank {rank}/{world} serves rendezvous {host}:{port}", flush=True)
        return host, port
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        try:
            host, port = Path(rendezvous).read_text().split()
            print(f"[tp2] rank {rank}/{world} joins rendezvous {host}:{port}", flush=True)
            return host, int(port)
        except (OSError, ValueError):
            time.sleep(2.0)
    raise TimeoutError("Timed out waiting for the rank-0 rendezvous file")


def run_device(inputs, mode, *, distributed_init_method=None):
    import torch
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from tessera.serving.config import TesseraConfig
    from tessera.serving.projection_routes import install
    from tessera.serving import telemetry

    if not torch.cuda.is_available():
        raise RuntimeError("The device smoke needs real CUDA")
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    print(f"[tp2] rank {rank}/{world} node {node_rank} starts {mode}", flush=True)
    torch.cuda.set_device(0)
    master_host, master_port = _tp2_master(world, rank, node_rank)
    config = VllmConfig(parallel_config=ParallelConfig(
        tensor_parallel_size=world, nnodes=int(os.environ.get("NNODES", "1")),
        node_rank=node_rank, master_addr=master_host, master_port=master_port))
    port = os.environ.get("MASTER_PORT", str(master_port))
    host = os.environ.get("MASTER_ADDR", master_host)
    print(f"[tp2] rank {rank}/{world} rendezvous {master_host}:{master_port}", flush=True)
    init_method = distributed_init_method or f"tcp://{host}:{port}"
    with set_current_vllm_config(config, check_compile=False):
        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank,
                                     distributed_init_method=init_method, backend="gloo")
        print(f"[tp2] rank {rank}/{world} joined process group", flush=True)
        initialize_model_parallel(world, 1)
        print(f"[tp2] rank {rank}/{world} joined model parallel", flush=True)
    if world not in (1, 2):
        raise ValueError("The small artifact supports one or two tensor-parallel ranks")
    import vllm
    runtime = {"source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "torch": torch.__version__, "vllm": vllm.__version__,
               "image": os.environ.get("TESSERA_CENSUS_RUNTIME_IMAGE")}
    if world > 1:
        context = {"mode": mode,
                   "inputs": [{"row": item["row"],
                               "input": hashlib.sha256(item["input"].view(torch.uint8).numpy().tobytes()).hexdigest(),
                               "bias": hashlib.sha256(item["bias"].view(torch.uint8).numpy().tobytes()).hexdigest() if "bias" in item else None}
                              for item in inputs]}
        peers = [None] * world
        torch.distributed.all_gather_object(peers, context)
        if any(peer != context for peer in peers):
            raise ValueError("The two ranks must use the same mode and artifact inputs")
    report = {"mode": mode, "world_size": world, "rank": rank, "runtime": runtime, "modules": []}
    # All stock controls precede the first installed selected configuration.
    originals = [_stock_control(item, config, world, rank, mode=mode) for item in inputs]
    install()
    controls = [_stock_control(item, config, world, rank, original, mode)
                for item, original in zip(inputs, originals)]
    for item, control in zip(inputs, controls):
        row = item["row"]
        result = {"prefix": row["prefix"], "stock": control, "routes": {}}
        for label, (family, _) in FAMILIES.items():
            scheme = row["routes"][label]["scheme"]
            from vllm.models.glm5next.common.model import Glm5NextForConditionalGeneration
            from tessera.serving.weights_mapper import module_name_mapper
            mapper = module_name_mapper(Glm5NextForConditionalGeneration.hf_to_vllm_mapper)
            checkpoint = _checkpoint_name(row["prefix"], mapper)
            quant = TesseraConfig({"smoke": {"targets": [checkpoint], "scheme": scheme}}, (), {"tp_agnostic": True})
            quant.apply_vllm_mapper(mapper)
            config.quant_config = quant
            with set_current_vllm_config(config, check_compile=False), torch.device("cuda"):
                layer = _construct(row, world, rank)
            if not hasattr(layer, "wire_bytes"):
                raise AssertionError(f"{row['prefix']}: the selected constructor did not create wire_bytes")
            wire = torch.frombuffer(bytearray(item["wires"][label]), dtype=torch.uint8).cuda()
            _load(layer, "wire_bytes", wire)
            if "bias" in item:
                _load(layer, "bias", item["bias"].cuda())
            layer.quant_method.process_weights_after_loading(layer)
            layer.update_param_tp_status()
            from tessera.serving.scheme import FUSED_WINDOW_DENSE_SYMBOL
            expected_decoder = (telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_E4M3MMA
                                if family == "TESSERA_FP8" else telemetry.DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED)
            if tuple(layer.tessera_native.launch_pair) != (FUSED_WINDOW_DENSE_SYMBOL, expected_decoder):
                raise AssertionError(f"{row['prefix']}: the requested native dense path was not prepared")
            x = item["input"].cuda()
            weight = _dense_weight(item["parsed"][label], family).cuda()
            if world > 1:
                from tessera.serving.sharding import shard_parsed_roles
                parsed = shard_parsed_roles(item["parsed"][label], layer.tessera_shard_plan)
                if row["kind"] == "kda":
                    expected_sizes = [r if index in (4, 5) else r // world
                                      for index, (_, r) in enumerate(row["roles"])]
                    if [int(parsed_unit.unit.codes.shape[0]) for _, parsed_unit in parsed] != expected_sizes:
                        raise AssertionError("The KDA shard divided or reordered a replicated role")
                local_weight = _dense_weight(parsed, family).cuda()
            expected, error, bound_facts = _dense_reference(item, label, layer, x)
            got, graph = _execute(lambda value: _call(layer, value), x, mode)
            output_dtype = torch.float32 if row["kind"] == "router" else torch.bfloat16
            numeric = compare(got, expected, name=f"{row['prefix']} {label}", dtype=output_dtype, bound=error)
            numeric["bound"] = bound_facts
            route = telemetry.read_route(layer)
            if route is None or route.get("state") != "served":
                raise AssertionError(f"{row['prefix']}: no served route exists")
            raw = {}
            if row["kind"] == "indexer":
                # This is the stock indexer operation, not a Linear forward.
                tail = layer.tessera_indexer_weights
                raw_call = lambda value: torch.mm(value.float(), tail)
                raw_got, raw_graph = _execute(raw_call, x, mode)
                raw_expected = torch.mm(x.float(), weight[128:, :].t().contiguous())
                raw["head_gate"] = compare(raw_got, raw_expected, name="indexer FP32 head gate", dtype=torch.float32, exact=True)
                raw["graph"] = raw_graph
                if tail.dtype != torch.float32 or tail.shape != (row["columns"], 32):
                    raise AssertionError("The indexer buffer must preserve its FP32 tail")
                raw["fp32_tail_bytes"] = tail.numel() * tail.element_size()
                raw["resident_buffers"] = {name: value.numel() * value.element_size()
                                           for name, value in layer.named_buffers()}
            if row["kind"] == "mla":
                from vllm.model_executor.layers.attention.mla_attention import split_kv_b_proj
                heads = 2 // world
                uk, uv = split_kv_b_proj(layer, torch.bfloat16, 256, heads, 128, 128)
                reference = local_weight if world > 1 else weight
                ref_uk, ref_uv = reference.bfloat16().t().reshape(256, heads, 256).split([128, 128], -1)
                compare(uk, ref_uk, name="MLA split key", dtype=torch.bfloat16, exact=True)
                compare(uv, ref_uv, name="MLA split value", dtype=torch.bfloat16, exact=True)
                query = torch.arange(heads * 128, device="cuda", dtype=torch.float32).reshape(1, heads, 128).bfloat16() / 256
                query_call = lambda value: torch.bmm(value.transpose(0, 1), uk.permute(1, 2, 0)).transpose(0, 1)
                absorbed_q, query_graph = _execute(query_call, query, mode)
                ref_q = torch.bmm(query.transpose(0, 1), ref_uk.permute(1, 2, 0)).transpose(0, 1)
                raw["absorbed_query"] = compare(absorbed_q, ref_q, name="MLA absorbed query BMM", dtype=torch.bfloat16, exact=True)
                raw["query_graph"] = query_graph
                value_call = lambda value: torch.bmm(value.transpose(0, 1), uv.permute(1, 0, 2)).transpose(0, 1)
                latent = x.reshape(-1, 1, 256).expand(-1, heads, -1).contiguous()
                raw_got, raw_graph = _execute(value_call, latent, mode)
                ref_out = torch.bmm(latent.transpose(0, 1), ref_uv.permute(1, 0, 2)).transpose(0, 1)
                raw["absorbed_value"] = compare(raw_got, ref_out, name="MLA absorbed value BMM", dtype=torch.bfloat16, exact=True)
                raw["value_graph"] = raw_graph
                retained = layer.tessera_projection_weight
                raw["decoded_weight_bytes"] = retained.numel() * retained.element_size()
            result["routes"][label] = {"numeric": numeric, "graph": graph, "route": route, "raw": raw,
                                        "method": type(layer.quant_method).__name__,
                                        "resident_storage": resident_tensors(layer)}
        report["modules"].append(result)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--prepare", action="store_true", help="write small test artifacts in --artifact")
    ap.add_argument("--dry-run", action="store_true", help="validate all inputs on CPU; do not claim a device forward")
    ap.add_argument("--mode", choices=("eager", "graph"), required=True)
    ap.add_argument("--m", choices=(1, 3), type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--distributed-init-method",
                    help="the Torch rendezvous URL for the admitted ranks")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TESSERA_SERVE_MODE", "resident")
    from tessera import routed_fused
    os.environ.setdefault(routed_fused.ENV_E4M3_MMA, "e4m3")
    import torch
    torch.set_num_threads(1)
    path = prepare(args.artifact) if args.prepare else args.artifact / "smoke.json"
    inputs = read_inputs(path, args.m)
    if args.dry_run:
        result = {"schema": "tessera.projection-smoke-result.v1", "status": "cpu-input-proof",
                  "mode_requested": args.mode, "device_forward": False, "modules": [
                      {"prefix": item["row"]["prefix"], "columns": item["row"]["columns"],
                       "roles": item["row"]["roles"], "input_shape": list(item["input"].shape),
                       "wire_bytes": {name: len(blob) for name, blob in item["wires"].items()}}
                      for item in inputs]}
    else:
        result = {"schema": "tessera.projection-smoke-result.v1", "status": "device-forward-proof",
                  **run_device(inputs, args.mode, distributed_init_method=args.distributed_init_method)}
    out = args.out
    if result.get("world_size", 1) > 1:
        out = out.with_name(out.stem + f".rank-{result['rank']}" + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    _json(out, result)
    print(json.dumps({"status": result["status"], "modules": len(result["modules"]), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
