"""Original-wire whole-expert research receipts, never summed leaf timings.

The supported owner is one complete 32-expert E4M3 K1 R1024 stack. Actual
captured routing tensors enter the existing resident/eager/TP1/EP1 method
unchanged; selecting experts is outside this operator's measured scope.
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
from enum import Enum
import hashlib
import json
import math
from pathlib import Path

from experiments import bench_native_operator as dense

PANEL_SCHEMA = "tessera.native_moe_panel.v1"
RECEIPT_SCHEMA = "tessera.native_moe_operator_receipt.v1"
REQUEST_SCHEMA = "tessera.native_moe_request.v1"
RUNTIME_SCHEMA = "tessera.native_moe_runtime.v1"
WORKSPACE_SCHEMA = "tessera.native_moe_workspace.v1"
FORMAT = "TESSERA_E4M3_K1_R1024"
ROLE_ORDER = ("w1", "w3", "w2")
EXECUTION = {"owner_kind": "complete_routed_moe", "mode": "resident",
             "execution_mode": "eager", "tensor_parallel": 1, "expert_parallel": 1,
             "include_router": False, "topk_selection": "external",
             "shared_experts": False, "monolithic": False, "bias": False}
PHASES = dense.PHASES
TENSOR_KEYS = ("input", "topk_ids", "topk_weights", "reference_qdq", "reference_output")
DTYPE_BYTES = {"torch.bfloat16": 2, "torch.float32": 4, "torch.int32": 4, "torch.int64": 8}


def tensor_record(record, shape, dtypes, where):
    dense._fields(record, ("shape", "dtype", "logical_bytes", "content_sha256"), where)
    if record["shape"] != shape or record["dtype"] not in dtypes:
        raise ValueError(where + ": tensor shape or dtype differs from its contract")
    dense._integer(record["logical_bytes"], where + ".logical_bytes")
    if record["logical_bytes"] != math.prod(shape) * DTYPE_BYTES[record["dtype"]]:
        raise ValueError(where + ": logical byte count differs from tensor geometry")
    dense._sha(record["content_sha256"], where)


def validate_routing(routing):
    dense._fields(routing, ("activation", "scoring_func", "renormalize", "routed_scaling_factor",
        "apply_router_weight_on_input", "expert_map", "input_dtype", "topk_weights_dtype",
        "topk_ids_dtype", "device", "weights_contract", "source_protocol"), "routing")
    if (routing["activation"] != "silu" or routing["scoring_func"] != "sigmoid"
            or type(routing["renormalize"]) is not bool
            or type(routing["routed_scaling_factor"]) not in (int, float)
            or routing["routed_scaling_factor"] != 1.0
            or routing["apply_router_weight_on_input"] is not False
            or routing["expert_map"] is not None
            or routing["input_dtype"] != "torch.bfloat16"
            or routing["topk_weights_dtype"] not in ("torch.bfloat16", "torch.float32")
            or routing["topk_ids_dtype"] not in ("torch.int32", "torch.int64")
            or routing["device"] != "cuda:0"
            or routing["weights_contract"] != "post_renormalization_and_routed_scaling"):
        raise ValueError("routing is outside the captured external-topk LFM scope")
    protocol = routing["source_protocol"]
    dense._fields(protocol, ("router_class", "router_source_sha256", "selection_bias",
                            "normalization_epsilon", "expert_bias_affects"), "source routing protocol")
    if not isinstance(protocol["router_class"], str) or not protocol["router_class"]:
        raise ValueError("source router class is missing")
    dense._sha(protocol["router_source_sha256"], "router source")
    if (type(protocol["normalization_epsilon"]) is not float
            or protocol["normalization_epsilon"] != 1e-6
            or protocol["expert_bias_affects"] != "selection_only"):
        raise ValueError("unsupported source routing normalization or bias protocol")
    if protocol["selection_bias"] is not None:
        tensor_record(protocol["selection_bias"], [32], ("torch.bfloat16", "torch.float32"), "selection bias")
    dense.identity_sha256(routing)
    return routing


def validate_transport(record, supplied, *, key):
    """Verify raw captured hashes by exact dtype reconstruction and roundtrip."""
    import torch
    dense._fields(record, ("source", "supplied", "operation"), key + " transport")
    allowed = ("torch.int32", "torch.int64") if key == "topk_ids" else ("torch.bfloat16", "torch.float32")
    for name in ("source", "supplied"):
        tensor_record(record[name], list(supplied.shape), allowed, key + "." + name)
    if dense.tensor_identity(supplied) != record["supplied"]:
        raise ValueError(key + ": supplied tensor differs from transport identity")
    expected_operation = "identity" if record["source"]["dtype"] == record["supplied"]["dtype"] else "lossless_dtype_conversion"
    if record["operation"] != expected_operation:
        raise ValueError(key + ": transport operation disagrees with the actual dtype change")
    if record["operation"] == "identity":
        if record["source"] != record["supplied"]:
            raise ValueError(key + ": identity transport changed tensor bytes or dtype")
    elif record["operation"] == "lossless_dtype_conversion":
        source_dtype = getattr(torch, record["source"]["dtype"].removeprefix("torch."))
        reconstructed = supplied.to(source_dtype)
        if (dense.tensor_identity(reconstructed) != record["source"]
                or not torch.equal(reconstructed.to(supplied.dtype), supplied)):
            raise ValueError(key + ": transport is not a lossless reconstruction of captured bytes")
    else:
        raise ValueError(key + ": unsupported transport operation")


def _require_cuda(value):
    import torch
    if (not isinstance(value, torch.Tensor)
            or value.device != torch.device("cuda", torch.cuda.current_device())
            or value.device.index != 0 or torch.cuda.is_current_stream_capturing()
            or torch.compiler.is_compiling()):
        raise ValueError("whole native receipt requires actual eager CUDA:0 tensors")


def validate_phase_values(phase_tensors, shape, routing, *, require_references):
    import torch
    dense._fields(phase_tensors, PHASES, "phases")
    for phase, values in phase_tensors.items():
        keys = TENSOR_KEYS if require_references else ("input", "topk_ids", "topk_weights")
        if set(values) != set(keys) and not (not require_references and set(values) == set(TENSOR_KEYS)):
            raise ValueError(phase + ": unknown or missing phase tensors")
        x = values["input"]
        _require_cuda(x)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != shape["hidden_size"]:
            raise ValueError(phase + ": input must be nonempty [M,hidden_size]")
        for key, value in values.items():
            _require_cuda(value)
            expected_shape = (x.shape[0], shape["top_k"] if key in ("topk_ids", "topk_weights") else shape["hidden_size"])
            expected_dtype = routing[key + "_dtype"] if key in ("input", "topk_ids", "topk_weights") else "torch.bfloat16"
            if tuple(value.shape) != expected_shape or str(value.dtype) != expected_dtype:
                raise ValueError(phase + ": phase tensor shape/dtype differs from routing contract")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(phase + ": nonfinite phase tensor")
        ids, weights = values["topk_ids"], values["topk_weights"]
        if bool((ids < 0).any()) or bool((ids >= shape["experts"]).any()) or bool((weights < 0).any()):
            raise ValueError(phase + ": expert IDs or routing weights are out of range")
        sorted_ids = ids.sort(dim=-1).values
        if sorted_ids.shape[1] > 1 and bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
            raise ValueError(phase + ": duplicate expert selection in a token")
        # Captured BF16 routing can have sum errors after its sum+epsilon
        # normalization. Never renormalize it or substitute a sum==1 axiom.


def validate_shape(shape):
    dense._fields(shape, ("experts", "hidden_size", "intermediate_size", "top_k"), "shape")
    for key, value in shape.items():
        dense._integer(value, "shape." + key)
    if shape["experts"] != 32 or shape["top_k"] > shape["experts"]:
        raise ValueError("only a complete 32-expert owner with valid top-k is supported")
    return dict(shape)


def validate_execution(execution, role_order):
    if dense.identity_sha256(execution) != dense.identity_sha256(EXECUTION):
        raise ValueError("only complete routed eager resident TP1/EP1 with external top-k is supported")
    if role_order != list(ROLE_ORDER):
        raise ValueError("explicit profile role order must be w1 gate, w3 up, w2 down")


def validate_member_order(members, shape):
    """The declared sequence is semantic; sorting member names is forbidden."""
    validate_shape(shape)
    if not isinstance(members, list) or len(members) != shape["experts"] * len(ROLE_ORDER):
        raise ValueError("the owner must bind all 96 expert-role members")
    units = set()
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValueError("member must be an object")
        if type(member.get("expert")) is not int or member["expert"] != index // len(ROLE_ORDER):
            raise ValueError("expert members are missing, reordered or duplicated")
        if member.get("role") != ROLE_ORDER[index % len(ROLE_ORDER)]:
            raise ValueError("member role differs from the explicit profile order")
        unit = member.get("unit")
        if not isinstance(unit, str) or not unit or unit in units:
            raise ValueError("member units must be nonempty and unique")
        units.add(unit)
        if member.get("format") != FORMAT:
            raise ValueError("whole routed receipt supports E4M3 K1 R1024 only")
    return members


def _member_shape(shape, role):
    n, k = shape["intermediate_size"], shape["hidden_size"]
    return [k, n] if role == "w2" else [n, k]


def _plain(value):
    """Observe configuration values without unstable repr/address fallbacks."""
    import torch
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, Enum):
        return {"enum": type(value).__module__ + "." + type(value).__qualname__, "value": _plain(value.value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite native configuration")
        return value
    if isinstance(value, torch.Tensor):
        return dense.tensor_identity(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_plain(item) for item in value]
        return {type(value).__name__: sorted(items, key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False))}
    raise ValueError("unobserved native configuration type: " + type(value).__qualname__)


def _storage_layout(tensor):
    import torch
    if tensor.device != torch.device("cuda", torch.cuda.current_device()):
        raise ValueError("native storage differs from the observed CUDA device")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
            "storage_bytes": tensor.untyped_storage().nbytes(),
            "logical_bytes": tensor.numel() * tensor.element_size(),
            "stride": list(tensor.stride()), "storage_offset": tensor.storage_offset()}


def observe_workspace():
    """Observe the runtime's existing owner, including storage outside the layer."""
    from vllm.v1.worker.workspace import current_workspace_manager
    manager = current_workspace_manager()
    if manager._num_ubatches != 1 or manager._num_lanes != 1 or not manager.is_locked():
        raise ValueError("workspace must be warmed and locked at one ubatch/lane")
    slots, storages, pointers = [], {}, []
    for index, tensor in enumerate(manager._current_workspaces):
        if tensor is None:
            slots.append({"index": index, "allocation": None})
            pointers.append(None)
            continue
        layout = _storage_layout(tensor)
        pointer = tensor.untyped_storage().data_ptr()
        slots.append({"index": index, **layout})
        storages[(str(tensor.device), pointer)] = layout["storage_bytes"]
        pointers.append(pointer)
    result = {"schema": WORKSPACE_SCHEMA, "owner": "vllm.WorkspaceManager",
              "num_ubatches": 1, "num_lanes": 1, "locked": True,
              "slots": slots, "resident_bytes": sum(storages.values())}
    return result, tuple(pointers)


@contextmanager
def native_runtime_context(vllm_config):
    """The actual TP1 context plus the engine's ordinary workspace owner."""
    import torch
    with dense.native_runtime_context(vllm_config):
        from vllm.v1.worker.workspace import (
            init_workspace_manager, is_workspace_manager_initialized, reset_workspace_manager)
        if is_workspace_manager_initialized():
            raise ValueError("routed receipt requires a fresh workspace owner")
        init_workspace_manager(torch.device("cuda", torch.cuda.current_device()), num_ubatches=1, num_lanes=1)
        try:
            yield
        finally:
            reset_workspace_manager()


def _native_config(layer):
    method = layer.quant_method
    fields = ("activation", "renormalize", "scoring_func", "routed_scaling_factor",
              "apply_router_weight_on_input", "expert_map", "global_num_experts", "local_num_experts",
              "top_k", "use_grouped_topk", "num_expert_group", "topk_group", "custom_routing_function",
              "e_score_correction_bias", "swiglu_limit", "swiglu_alpha", "swiglu_beta",
              "expert_placement_strategy", "is_fused_checkpoint_transposed")
    return {"layer": {key: _plain(getattr(layer, key)) for key in fields},
            "moe_config": _plain(layer.moe_config), "quant_config": _plain(method.moe_quant_config),
            "backend": str(getattr(method.fp8_backend, "value", method.fp8_backend)),
            "experts_class": method.experts_cls.__module__ + "." + method.experts_cls.__qualname__,
            "kernel_class": type(method.moe_kernel).__module__ + "." + type(method.moe_kernel).__qualname__,
            "is_monolithic": bool(method.is_monolithic)}


def resolve_serving_config(path, runtime_image):
    """Derive the factory context from explicit versioned engine settings.

    This config supplies the whole-operator factory only. It does not claim
    to instantiate or measure the complete engine or its KV allocations.
    """
    import os
    from vllm.config import (VllmConfig, CacheConfig, ParallelConfig,
        SchedulerConfig, KernelConfig, CompilationConfig)
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema") != "tessera.first_model_serving_config.v1":
        raise ValueError("versioned serving configuration is required")
    if document.get("runtime_image") != runtime_image:
        raise ValueError("serving configuration image differs from requested runtime")
    args = document["engine_args"]
    required = {"data_parallel_size", "dtype", "enable_chunked_prefill", "enable_expert_parallel",
        "enable_prefix_caching", "enforce_eager", "gpu_memory_utilization", "kernel_config",
        "max_model_len", "max_num_batched_tokens", "max_num_seqs", "pipeline_parallel_size",
        "tensor_parallel_size"}
    dense._fields(args, required, "serving engine arguments")
    if (args["dtype"] != "bfloat16" or args["enforce_eager"] is not True
            or args["enable_expert_parallel"] is not False
            or any(type(args[key]) is not int or args[key] != 1 for key in
                   ("data_parallel_size", "pipeline_parallel_size", "tensor_parallel_size"))
            or args["kernel_config"] != {"moe_backend": "auto"}
            or document["environment"] != {"TESSERA_SERVE_MODE": "resident"}
            or os.environ.get("TESSERA_SERVE_MODE") != "resident"):
        raise ValueError("serving configuration is outside resident eager BF16 TP1/EP1 scope")
    for key in ("max_model_len", "max_num_batched_tokens", "max_num_seqs"):
        dense._integer(args[key], key)
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    config = VllmConfig(
        scheduler_config=SchedulerConfig(max_model_len=args["max_model_len"], is_encoder_decoder=False,
            max_num_batched_tokens=args["max_num_batched_tokens"], max_num_seqs=args["max_num_seqs"],
            enable_chunked_prefill=args["enable_chunked_prefill"]),
        cache_config=CacheConfig(gpu_memory_utilization=args["gpu_memory_utilization"],
            enable_prefix_caching=args["enable_prefix_caching"]),
        parallel_config=ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1,
            data_parallel_size=1, enable_expert_parallel=False),
        kernel_config=KernelConfig(**args["kernel_config"]),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE))
    return config, {"file_sha256": hashlib.sha256(raw).hexdigest(), "document": document,
        "scope": "standalone_factory_context_not_full_engine",
        "resolved": {key: _plain(getattr(config, key)) for key in
                     ("scheduler_config", "cache_config", "parallel_config", "kernel_config",
                      "compilation_config", "device_config")}}


def verify_routing_bias(routing, bias):
    import torch
    expected = routing["source_protocol"]["selection_bias"]
    if expected is None:
        if bias is not None:
            raise ValueError("unexpected routing bias")
        return None
    if bias is None or bias.dtype != torch.float32 or list(bias.shape) != [32] or not bool(torch.isfinite(bias).all()):
        raise ValueError("captured selection bias must supply actual FP32 factory values")
    source_dtype = getattr(torch, expected["dtype"].removeprefix("torch."))
    original = bias.to(source_dtype)
    if dense.tensor_identity(original) != expected or not torch.equal(original.float(), bias):
        raise ValueError("factory selection bias differs from the captured source")
    return bias


def verify_native_configuration(layer, shape, routing, max_tokens):
    """Validate the factory's observed owner instead of trusting its arguments."""
    expected = {"renormalize": routing["renormalize"], "scoring_func": routing["scoring_func"],
        "routed_scaling_factor": routing["routed_scaling_factor"], "apply_router_weight_on_input": False,
        "expert_map": None, "global_num_experts": shape["experts"], "local_num_experts": shape["experts"],
        "top_k": shape["top_k"], "use_grouped_topk": True, "num_expert_group": 1, "topk_group": 1,
        "custom_routing_function": None, "swiglu_limit": None, "swiglu_alpha": None, "swiglu_beta": None,
        "is_fused_checkpoint_transposed": False, "tessera_mode": "resident", "tessera_family": "TESSERA_FP8"}
    for key, value in expected.items():
        if dense.identity_sha256(_plain(getattr(layer, key))) != dense.identity_sha256(value):
            raise ValueError("factory actual " + key + " differs from supported routed owner")
    if getattr(layer.activation, "value", layer.activation) != routing["activation"]:
        raise ValueError("factory actual activation differs from captured operator")
    verify_routing_bias(routing, layer.e_score_correction_bias)
    config = layer.moe_config
    config_expected = {"num_experts": shape["experts"], "num_local_experts": shape["experts"],
        "num_logical_experts": shape["experts"], "experts_per_token": shape["top_k"],
        "hidden_dim": shape["hidden_size"], "intermediate_size": shape["intermediate_size"],
        "intermediate_size_per_partition": shape["intermediate_size"], "max_num_tokens": max_tokens,
        "has_bias": False, "is_lora_enabled": False}
    for key, value in config_expected.items():
        if dense.identity_sha256(_plain(getattr(config, key))) != dense.identity_sha256(value):
            raise ValueError("factory actual MoE " + key + " differs from serving configuration")
    import torch
    device = torch.device(config.device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if str(config.in_dtype) != routing["input_dtype"] or device.type != "cuda" or device_index != 0:
        raise ValueError("factory actual dtype/device differs from captured operator")
    for key in ("tp_size", "ep_size", "dp_size", "pcp_size", "sp_size"):
        value = getattr(config.moe_parallel_config, key)
        if type(value) is not int or value != 1:
            raise ValueError("factory actual parallel configuration is outside TP1/EP1")
    if layer.quant_method.is_monolithic:
        raise ValueError("factory selected unsupported monolithic execution")


def _build_layer(scheme, shape, routing, *, unit, device, bias):
    """Use the same FusedMoEFactory arguments as the stock LFM2-MoE model."""
    import torch
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from tessera.serving.config import TesseraConfig
    config = TesseraConfig.from_config({"quant_method": "tessera", "format": "tessera", "ignore": [],
        "config_groups": {"native_receipt_experts": {"format": "TESSERA", "targets": [unit], "scheme": scheme}}})
    runner = FusedMoEFactory(num_experts=shape["experts"], top_k=shape["top_k"],
        hidden_size=shape["hidden_size"], intermediate_size=shape["intermediate_size"],
        params_dtype=torch.bfloat16, renormalize=routing["renormalize"], quant_config=config,
        use_grouped_topk=True, num_expert_group=1, topk_group=1, prefix=unit,
        enable_eplb=False, num_redundant_experts=0, scoring_func=routing["scoring_func"],
        e_score_correction_bias=bias, routed_scaling_factor=routing["routed_scaling_factor"],
        ckpt_names=("w1", "w2", "w3"))
    return runner.routed_experts.to(device)


def prepare_native_moe_operator(member_inputs, tensors, phase_tensors, *, unit, shape, routing,
                                runtime_image, execution, profile_role_order, serving_config, routing_capture_sha256,
                                phase_transport, routing_bias=None, warmup_iterations=8):
    """Verify all original PWC wires, then load one complete native owner."""
    import torch
    from tessera.cached_unit import verify_cached_unit, tensor_identity as producer_tensor_identity
    from tessera.fused import pack_fused
    from tessera.serving.scheme import MOE_SHARD_PROJECTIONS, TESSERA_FP8, validate_tessera_moe_scheme
    from tessera.unit_artifact import read_unit_artifact
    from vllm.v1.worker.workspace import lock_workspace
    validate_execution(execution, profile_role_order)
    shape = validate_shape(shape)
    validate_member_order(member_inputs, shape)
    validate_routing(routing)
    dense._integer(warmup_iterations, "warmup_iterations")
    dense._sha(routing_capture_sha256, "routing_capture_sha256")
    dense._fields(phase_transport, PHASES, "phase transport")
    for phase in PHASES:
        dense._fields(phase_transport[phase], ("topk_ids", "topk_weights"), "routing transport")
        for key in ("topk_ids", "topk_weights"):
            validate_transport(phase_transport[phase][key], phase_tensors[phase][key], key=key)
    bias = verify_routing_bias(routing, routing_bias)
    if bias is not None:
        _require_cuda(bias)
    validate_phase_values(phase_tensors, shape, routing, require_references=False)
    expected_tensor_names = {f"{kind}/{member['unit']}" for member in member_inputs
                             for kind in ("source_weight", "rendered_weight")}
    if set(tensors) != expected_tensor_names:
        raise ValueError("source/render tensor roster differs from the complete owner")
    members, wires, strides = [], [], {"w13": 0, "w2": 0}
    for member in member_inputs:
        dense._fields(member, ("unit", "expert", "role", "format", "blob", "record"), "member input")
        original, record = member["blob"], member["record"]
        source = tensors["source_weight/" + member["unit"]]
        rendered = tensors["rendered_weight/" + member["unit"]]
        for value in (source, rendered):
            dense._require_cuda_tensor(value)
            if list(value.shape) != _member_shape(shape, member["role"]):
                raise ValueError("expert-role tensor geometry differs from its owner")
        identity = record["identity"]
        if identity["unit"] != member["unit"].removesuffix(".weight"):
            raise ValueError("wire producer unit differs from its declared member")
        if identity["source"] != producer_tensor_identity(source):
            raise ValueError("wire source differs from the actual source member")
        accepted = verify_cached_unit(original, record, identity)
        recipe, manifest = identity["recipe"], accepted.manifest
        if (recipe["grid"] != "E4M3" or recipe["q256"] != 1024
                or manifest.body.name != "WINDOW" or manifest.scale_plane.kind.name != "CHANNEL"):
            raise ValueError("wire is outside the E4M3 K1 R1024 WINDOW/CHANNEL scope")
        decoded = read_unit_artifact(original, device=str(source.device)).bfloat16()
        if dense.tensor_identity(decoded) != dense.tensor_identity(rendered):
            raise ValueError("original wire decode differs from the actual PWC member render")
        projection = MOE_SHARD_PROJECTIONS[member["role"]]
        container = pack_fused([(projection, source.shape[0], original)])
        group = "w2" if member["role"] == "w2" else "w13"
        strides[group] = max(strides[group], len(container))
        wires.append((f"{member['expert']}.{member['role']}.wire",
                      torch.frombuffer(bytearray(container), dtype=torch.uint8).clone()))
        members.append({key: member[key] for key in ("unit", "expert", "role", "format")})
        members[-1].update(shape=list(source.shape), source_weight=dense.tensor_identity(source),
            rendered_weight=dense.tensor_identity(rendered), wire_sha256=hashlib.sha256(original).hexdigest(),
            wire_record_sha256=dense.identity_sha256(record))
        del decoded
    n, k = shape["intermediate_size"], shape["hidden_size"]
    scheme = validate_tessera_moe_scheme({"family": TESSERA_FP8, "structure": "routed_moe",
        "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL", "experts": shape["experts"],
        "groups": {"w13": {"rows": 2*n, "columns": k, "q256": 1024, "wire_stride": strides["w13"],
                              "roles": [[MOE_SHARD_PROJECTIONS[role], n] for role in ROLE_ORDER[:2]]},
                   "w2": {"rows": k, "columns": n, "q256": 1024, "wire_stride": strides["w2"],
                            "roles": [[MOE_SHARD_PROJECTIONS["w2"], k]]}}}, unit)
    device = phase_tensors["prefill"]["input"].device
    if max(v["input"].shape[0] for v in phase_tensors.values()) > serving_config["document"]["engine_args"]["max_num_batched_tokens"]:
        raise ValueError("actual phase exceeds the explicit serving scheduler token limit")
    layer = _build_layer(scheme, shape, routing, unit=unit, device=device, bias=bias)
    with torch.no_grad():
        loaded = list(layer.load_weights(wires))
        if len(loaded) != len(wires):
            raise ValueError("native RoutedExperts loader did not consume every member")
        layer.quant_method.process_weights_after_loading(layer)
    del wires
    verify_native_configuration(layer, shape, routing,
        serving_config["document"]["engine_args"]["max_num_batched_tokens"])
    # Warm actual phases before the panel freezes lazy native libraries and
    # the runtime's preexisting workspace. No output comparison or price here.
    initial_tensors = phase_identities(phase_tensors)
    native_before = dense._native_tensors(layer)
    with torch.inference_mode():
        for phase in PHASES:
            for _ in range(warmup_iterations):
                apply_whole(layer, phase_tensors[phase])
        torch.cuda.synchronize()
    if phase_identities(phase_tensors) != initial_tensors or dense._native_tensors(layer) != native_before:
        raise ValueError("native preparation mutated input or loaded expert tensors")
    lock_workspace()
    workspace, pointers = observe_workspace()
    config = _native_config(layer)
    if config["is_monolithic"]:
        raise ValueError("native runtime selected unsupported monolithic execution")
    operator = {"members": members, "shape": shape, "routing": routing,
        "profile_role_order": list(profile_role_order), "routing_capture_sha256": routing_capture_sha256,
        "serving_config_sha256": serving_config["file_sha256"], "serving_config": serving_config,
        "phases": {phase: {"transport": phase_transport[phase]} for phase in PHASES},
        "scheme": scheme, "scheme_sha256": dense.identity_sha256(scheme), "native_tensors": native_before,
        "config": config, "config_sha256": dense.identity_sha256(config),
        "declared_route": {"kind": "moe", "policy": "TESSERA_FP8:resident",
            "symbol": "vllm.fused_moe.modular_kernel:" + layer.tessera_backend,
            "decoder": layer.tessera_decoder, "contract": layer.tessera_activation_contract}}
    runtime = dense.observe_runtime(runtime_image)
    runtime.update(schema=RUNTIME_SCHEMA, execution=dict(EXECUTION))
    runtime["source"]["routed_harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"layer": layer, "operator": operator, "runtime": runtime,
            "workspace": workspace, "workspace_pointers": pointers}


def apply_whole(layer, tensors):
    return layer.quant_method.apply(layer, tensors["input"], tensors["topk_weights"],
                                    tensors["topk_ids"], None, None)


def phase_identities(phase_tensors):
    return {phase: {key: dense.tensor_identity(value) for key, value in values.items()}
            for phase, values in phase_tensors.items()}

def validate_panel(panel):
    """Check the independently frozen 96-member join before CUDA execution."""
    dense._fields(panel, ("schema", "unit", "format", "shape", "members", "profile_role_order",
        "routing", "routing_capture_sha256", "source_sha256", "calibration_sha256", "cost_sha256",
        "probe_identity_sha256", "probe_scope", "runtime_binding", "execution", "runtime", "native_tensors_sha256",
        "scheme_sha256", "config_sha256", "serving_config_sha256", "workspace", "workspace_sha256",
        "numerics", "phases"), "panel")
    if panel["schema"] != PANEL_SCHEMA or panel["format"] != FORMAT:
        raise ValueError("whole MoE panel schema or format unsupported")
    if not isinstance(panel["unit"], str) or not panel["unit"]:
        raise ValueError("whole owner unit is missing")
    shape = validate_shape(panel["shape"])
    validate_execution(panel["execution"], panel["profile_role_order"])
    validate_member_order(panel["members"], shape)
    validate_routing(panel["routing"])
    for key in ("routing_capture_sha256", "source_sha256", "calibration_sha256", "cost_sha256",
                "probe_identity_sha256", "native_tensors_sha256", "scheme_sha256", "config_sha256",
                "serving_config_sha256", "workspace_sha256"):
        dense._sha(panel[key], key)
    scope = panel["probe_scope"]
    if scope is not None:
        dense._fields(scope, ("schema", "parent_calibration_sha256", "subset_calibration_sha256",
                              "sample_indices", "scope"), "probe scope")
        if (scope["schema"] != "prismaquant.native_probe_subset.v1"
                or scope["scope"] != "first_sequence_integration_screen"
                or dense.identity_sha256(scope["sample_indices"]) != dense.identity_sha256([0])
                or scope["subset_calibration_sha256"] != panel["calibration_sha256"]):
            raise ValueError("probe subset scope differs from the explicit first-sequence screen")
        for key in ("parent_calibration_sha256", "subset_calibration_sha256"):
            dense._sha(scope[key], key)
    binding = panel["runtime_binding"]
    dense._fields(binding, ("member_formats", "member_operator_identity_sha256", "member_shapes", "operator_route"), "runtime binding")
    names = {member["unit"] for member in panel["members"]}
    for key in ("member_formats", "member_operator_identity_sha256", "member_shapes"):
        if not isinstance(binding[key], dict) or set(binding[key]) != names:
            raise ValueError("runtime binding must include exactly every original member")
    for member in panel["members"]:
        dense._fields(member, ("unit", "expert", "role", "format", "shape", "source_weight",
                               "rendered_weight", "activation", "wire"), "panel member")
        geometry = _member_shape(shape, member["role"])
        if (member["unit"] != f"{panel['unit']}.{member['expert']}.{member['role']}"
                or member["shape"] != geometry or binding["member_shapes"][member["unit"]] != geometry
                or binding["member_formats"][member["unit"]] != FORMAT):
            raise ValueError("member name/shape/format differs from its explicit owner role")
        dense._sha(binding["member_operator_identity_sha256"][member["unit"]], "member joint identity")
        for key in ("source_weight", "rendered_weight"):
            tensor_record(member[key], geometry, ("torch.bfloat16",), key)
        if member["activation"].get("clip_enabled") is not False or member["activation"].get("input_global_scale") is not None:
            raise ValueError("whole MoE requires dynamic unclipped member activations")
        wire = member["wire"]
        dense._fields(wire, ("blob_sha256", "blob_bytes", "record"), "member wire")
        dense._sha(wire["blob_sha256"], "member wire")
        dense._integer(wire["blob_bytes"], "member wire bytes")
        if wire["record"]["blob_sha256"] != wire["blob_sha256"] or wire["record"]["blob_bytes"] != wire["blob_bytes"]:
            raise ValueError("member wire record differs from wire identity")
    workspace = panel["workspace"]
    dense._fields(workspace, ("schema", "owner", "num_ubatches", "num_lanes", "locked", "slots", "resident_bytes"), "workspace")
    if (workspace["schema"] != WORKSPACE_SCHEMA or workspace["owner"] != "vllm.WorkspaceManager"
            or workspace["locked"] is not True or any(type(workspace[k]) is not int or workspace[k] != 1
                for k in ("num_ubatches", "num_lanes")) or not isinstance(workspace["slots"], list)
            or dense.identity_sha256(workspace) != panel["workspace_sha256"]):
        raise ValueError("workspace is not the frozen single-lane runtime owner")
    dense._integer(workspace["resident_bytes"], "workspace bytes", 0)
    for index, slot in enumerate(workspace["slots"]):
        if type(slot.get("index")) is not int or slot["index"] != index:
            raise ValueError("workspace slots are missing or reordered")
        if slot == {"index": index, "allocation": None}:
            continue
        dense._fields(slot, ("index", "shape", "dtype", "device", "storage_bytes", "logical_bytes", "stride", "storage_offset"), "workspace slot")
        if slot["device"] != "cuda:0" or len(slot["shape"]) != len(slot["stride"]):
            raise ValueError("workspace device/geometry unsupported")
        for n in (*slot["shape"], *slot["stride"], slot["storage_bytes"], slot["logical_bytes"], slot["storage_offset"]):
            dense._integer(n, "workspace dimension", 0)
    runtime = panel["runtime"]
    if runtime.get("schema") != RUNTIME_SCHEMA or dense.identity_sha256(runtime.get("execution")) != dense.identity_sha256(EXECUTION):
        raise ValueError("runtime manifest differs from whole MoE execution")
    dense._fields(panel["numerics"], ("atol", "rtol"), "numerics")
    for key, value in panel["numerics"].items():
        dense._number(value, key)
    dense._fields(panel["phases"], PHASES, "phases")
    for phase, item in panel["phases"].items():
        dense._fields(item, ("m", *TENSOR_KEYS, "transport", "expected_route"), phase)
        dense._integer(item["m"], phase + ".m")
        for key in TENSOR_KEYS:
            width = shape["top_k"] if key in ("topk_ids", "topk_weights") else shape["hidden_size"]
            dtype = panel["routing"][key + "_dtype"] if key in ("input", "topk_ids", "topk_weights") else "torch.bfloat16"
            tensor_record(item[key], [item["m"], width], (dtype,), phase + "." + key)
        dense._fields(item["transport"], ("topk_ids", "topk_weights"), "transport")
        for key, record in item["transport"].items():
            dense._fields(record, ("source", "supplied", "operation"), "transport record")
            allowed = ("torch.int32", "torch.int64") if key == "topk_ids" else ("torch.bfloat16", "torch.float32")
            tensor_record(record["source"], item[key]["shape"], allowed, "captured routing source")
            operation = "identity" if record["source"]["dtype"] == record["supplied"]["dtype"] else "lossless_dtype_conversion"
            if (record["supplied"] != item[key] or record["operation"] != operation
                    or operation == "identity" and record["source"] != record["supplied"]):
                raise ValueError("phase transport does not preserve captured routing identity")
        route = item["expected_route"]
        dense._fields(route, dense.ROUTE_KEYS, "route")
        if (route["kind"] != "moe" or route["policy"] != "TESSERA_FP8:resident"
                or route["decoder"] != "torch_materialize_stock" or route["contract"] != "fp8_per_token_dynamic"
                or not isinstance(route["symbol"], str) or not route["symbol"].startswith("vllm.fused_moe.modular_kernel:")
                or route["symbol"] == "vllm.fused_moe.modular_kernel:"
                or route["symbol"] != binding["operator_route"]):
            raise ValueError("route differs from native whole MoE binding")
    if panel["phases"]["prefill"]["expected_route"] != panel["phases"]["decode"]["expected_route"]:
        raise ValueError("whole MoE phases must use the same resident route")
    return json.loads(json.dumps(panel, allow_nan=False))


def _check_phase_tensors(panel, phase_tensors):
    validate_phase_values(phase_tensors, panel["shape"], panel["routing"], require_references=True)
    for phase in PHASES:
        for key, value in phase_tensors[phase].items():
            if dense.tensor_identity(value) != panel["phases"][phase][key]:
                raise ValueError(phase + ": actual " + key + " differs from independent panel")
        for key in ("topk_ids", "topk_weights"):
            validate_transport(panel["phases"][phase]["transport"][key], phase_tensors[phase][key], key=key)


def _check_prepared(prepared, panel):
    dense._require_eager_context()
    operator, layer = prepared["operator"], prepared["layer"]
    if prepared["runtime"] != panel["runtime"] or dense.observe_arithmetic() != panel["runtime"]["arithmetic"]:
        raise ValueError("actual runtime/arithmetic differs from independent panel")
    dense._check_native_library_scope(panel["runtime"])
    for key in ("shape", "routing", "profile_role_order", "routing_capture_sha256", "serving_config_sha256"):
        if operator[key] != panel[key]:
            raise ValueError("native " + key + " differs from independent panel")
    for key in ("native_tensors", "scheme", "config"):
        if dense.identity_sha256(operator[key]) != panel[key + "_sha256"]:
            raise ValueError("native " + key + " identity differs from independent panel")
    if dense._native_tensors(layer) != operator["native_tensors"] or _native_config(layer) != operator["config"]:
        raise ValueError("native expert state/configuration changed after preparation")
    if layer.tessera_mode != "resident" or layer.quant_method.is_monolithic:
        raise ValueError("native execution changed from resident modular MoE")
    expected_members = [{**{key: member[key] for key in ("unit", "expert", "role", "format", "shape", "source_weight", "rendered_weight")},
                         "wire_sha256": member["wire"]["blob_sha256"],
                         "wire_record_sha256": dense.identity_sha256(member["wire"]["record"])} for member in panel["members"]]
    if operator["members"] != expected_members:
        raise ValueError("native original-wire members differ from independent panel")
    workspace, pointers = observe_workspace()
    if workspace != panel["workspace"] or pointers != prepared["workspace_pointers"]:
        raise ValueError("runtime workspace layout or process storage changed after preparation")
    for phase in PHASES:
        if operator["phases"][phase]["transport"] != panel["phases"][phase]["transport"]:
            raise ValueError("native routing transport differs from independent panel")
        if operator["declared_route"] != panel["phases"][phase]["expected_route"]:
            raise ValueError("native backend differs from independent panel")


def measure_prepared_operator(prepared, panel, phase_tensors, *, warmup_iterations, iterations,
                              resource_collector=None):
    """Gate BOTH phases/QDQ before any timings; retain explicit resource gaps."""
    import torch
    from tessera.serving.telemetry import read_route
    panel = validate_panel(panel)
    dense._integer(warmup_iterations, "warmup_iterations")
    dense._integer(iterations, "iterations", 3)
    dense._fields(phase_tensors, PHASES, "phase tensors")
    _check_phase_tensors(panel, phase_tensors)
    _check_prepared(prepared, panel)
    layer = prepared["layer"]
    observations, resource_phases = {}, {}
    with torch.inference_mode():
        for phase in PHASES:
            tensors, expected = phase_tensors[phase], panel["phases"][phase]
            _check_phase_tensors(panel, phase_tensors)
            x = tensors["input"]
            qdq = dense.represented_native_input(layer, x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            output = apply_whole(layer, tensors)
            torch.cuda.synchronize()
            peak = max(0, torch.cuda.max_memory_allocated() - before)
            route = read_route(layer)
            wanted = expected["expected_route"]
            if (not isinstance(route, dict) or any(route.get(k) != v for k, v in wanted.items())
                    or route.get("state") != "served" or route.get("reason") is not None
                    or route.get("shape") != f"M{expected['m']}:N{2*panel['shape']['intermediate_size']}:K{panel['shape']['hidden_size']}"):
                raise ValueError(f"{phase}: actual route differs from independent panel")
            qdq_error = dense.compare_tensors(qdq, tensors["reference_qdq"], **panel["numerics"])
            error = dense.compare_tensors(output, tensors["reference_output"], **panel["numerics"])
            if not bool(torch.isfinite(x).all()):
                for record in (qdq_error, error):
                    record.update(status="failed", finite=False)
            observations[phase] = {"m": expected["m"], "input": dense.tensor_identity(x),
                "topk_ids": dense.tensor_identity(tensors["topk_ids"]),
                "topk_weights": dense.tensor_identity(tensors["topk_weights"]),
                "transport": expected["transport"],
                "reference_qdq": dense.tensor_identity(tensors["reference_qdq"]), "qdq": dense.tensor_identity(qdq),
                "qdq_numerics": qdq_error, "reference_output": dense.tensor_identity(tensors["reference_output"]),
                "output": dense.tensor_identity(output), "route": route, "numerics": error, "measurement": None}
            resource_phases[phase] = {"input_bytes": x.untyped_storage().nbytes(),
                "output_bytes": output.untyped_storage().nbytes(), "torch_peak_increment_bytes": peak}
            del output, qdq
            _check_phase_tensors(panel, phase_tensors)
        passed = all(observations[p][key]["status"] == "passed" for p in PHASES for key in ("numerics", "qdq_numerics"))
        if passed:
            _check_prepared(prepared, panel)
            for phase in PHASES:
                _check_prepared(prepared, panel)
                _check_phase_tensors(panel, phase_tensors)
                if resource_collector is None:
                    observations[phase]["measurement"] = dense.time_apply(
                        lambda phase=phase: apply_whole(layer, phase_tensors[phase]),
                        warmup_iterations=warmup_iterations, iterations=iterations)
                else:
                    # Warm the allocator with real applies, but do not price
                    # calls while CUPTI memory/API collection is active.
                    for _ in range(warmup_iterations):
                        apply_whole(layer, phase_tensors[phase])
                    torch.cuda.synchronize()
                _check_prepared(prepared, panel)
                _check_phase_tensors(panel, phase_tensors)
                observed = read_route(layer)
                if observed != observations[phase]["route"]:
                    raise ValueError(f"{phase}: native route changed during timing")
            if resource_collector is not None:
                for phase in PHASES:
                    _check_prepared(prepared, panel)
                    _check_phase_tensors(panel, phase_tensors)
                    output, allocation = resource_collector.observe_apply(
                        lambda phase=phase: apply_whole(layer, phase_tensors[phase]),
                        phase, device=phase_tensors[phase]["input"].device.index)
                    error = dense.compare_tensors(output, phase_tensors[phase]["reference_output"], **panel["numerics"])
                    if error["status"] != "passed" or read_route(layer) != observations[phase]["route"]:
                        raise ValueError(f"{phase}: resource invocation numerical/route mismatch")
                    resource_phases[phase]["torch_observation"] = allocation
                    resource_phases[phase]["numerics"] = error
                    del output
                    _check_prepared(prepared, panel)
                    _check_phase_tensors(panel, phase_tensors)
        _check_prepared(prepared, panel)
    status = ("resources_observed" if resource_collector is not None else "timing_admissible") if passed else "numerical_refused"
    return {"schema": RECEIPT_SCHEMA, "status": status,
            "panel": panel, "panel_sha256": dense.identity_sha256(panel), "runtime": prepared["runtime"],
            "runtime_sha256": dense.identity_sha256(prepared["runtime"]), "operator": prepared["operator"],
            "phases": observations, "resources": {"status": "incomplete", "scope": "torch_allocator_observation",
                "resident_bytes": dense._resident_bytes(layer), "phases": resource_phases,
                "workspace_resident_bytes": prepared["workspace"]["resident_bytes"],
                "workspace_sha256": dense.identity_sha256(prepared["workspace"]),
                "unknown": ["native_and_library_scratch_outside_torch_allocator", "fixed_and_full_model_resources"]}}


def time_after_resource_collection(prepared, panel, phase_tensors, receipt, *, collector,
                                   warmup_iterations, iterations):
    """Price only after allocation profiling is closed and its bound passed."""
    from tessera.serving.telemetry import read_route
    if not collector._finished:
        raise ValueError("resource collector must be closed before decision timing")
    if (receipt["status"] != "resources_observed"
            or receipt["resources"]["status"] != "complete_operator_bound"):
        raise ValueError("complete operator resource bounds required before decision timing")
    if receipt["panel_sha256"] != dense.identity_sha256(panel):
        raise ValueError("timing panel differs from resource panel")
    import torch
    with torch.inference_mode():
        for phase in PHASES:
            _check_prepared(prepared, panel)
            _check_phase_tensors(panel, phase_tensors)
            receipt["phases"][phase]["measurement"] = dense.time_apply(
                lambda phase=phase: apply_whole(prepared["layer"], phase_tensors[phase]),
                warmup_iterations=warmup_iterations, iterations=iterations)
            _check_prepared(prepared, panel)
            _check_phase_tensors(panel, phase_tensors)
            if read_route(prepared["layer"]) != receipt["phases"][phase]["route"]:
                raise ValueError(f"{phase}: native route changed during decision timing")
    receipt["status"] = "timing_admissible"
    receipt["timing_scope"] = "cuda_events_after_resource_collector_stop"
    return receipt


attach_resource_trace = dense.attach_resource_trace


def profile_prepared_operator(prepared, panel, phase_tensors, *, output_prefix,
                              warmup_iterations, iterations):
    """Separate-process profiling evidence; never a resource or price receipt."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    gate = measure_prepared_operator(prepared, panel, phase_tensors,
        warmup_iterations=warmup_iterations, iterations=iterations)
    result = {"schema": "tessera.native_moe_profile.v1", "status": "numerical_refused",
              "panel_sha256": gate["panel_sha256"], "runtime_sha256": gate["runtime_sha256"],
              "scope": "separate_process_torch_profiler_replay_not_resource_or_price_receipt",
              "phases": {}}
    if gate["status"] != "timing_admissible":
        return result
    for phase in PHASES:
        # Both phases already passed the strict unprofiled gate. Profiler
        # instrumentation may map extra libraries of its own after entry.
        if dense._native_tensors(prepared["layer"]) != prepared["operator"]["native_tensors"]:
            raise ValueError("native tensors changed before profiling")
        if dense.observe_arithmetic() != panel["runtime"]["arithmetic"]:
            raise ValueError("arithmetic changed before profiling")
        _check_phase_tensors(panel, phase_tensors)
        with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                             record_shapes=True, profile_memory=True) as prof:
            output = apply_whole(prepared["layer"], phase_tensors[phase])
            torch.cuda.synchronize()
        numerics = dense.compare_tensors(output, phase_tensors[phase]["reference_output"], **panel["numerics"])
        if numerics["status"] != "passed":
            raise ValueError(f"{phase}: profiled invocation failed numerical gate")
        del output
        _check_phase_tensors(panel, phase_tensors)
        events = prof.key_averages()
        kernels = [{"name": e.key, "calls": e.count, "self_device_us": e.self_device_time_total}
                   for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
        if not kernels or sum(e["self_device_us"] for e in kernels) <= 0:
            raise ValueError(f"{phase}: profiler recorded no CUDA kernel work")
        trace = Path(str(output_prefix) + f".{phase}.trace.json")
        prof.export_chrome_trace(str(trace))
        result["phases"][phase] = {"numerics": numerics, "qdq_numerics": gate["phases"][phase]["qdq_numerics"],
            "kernels": sorted(kernels, key=lambda row: -row["self_device_us"]),
            "cpu": [{"name": e.key, "calls": e.count, "self_cpu_us": e.self_cpu_time_total}
                    for e in sorted(events, key=lambda e: -e.self_cpu_time_total)
                    if e.device_type == torch.autograd.DeviceType.CPU][:20],
            "trace_file": trace.name, "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}
    result["status"] = "profiled"
    result["instrumentation_libraries"] = sorted(
        {str(path) for path in dense._mapped_shared_libraries()} - set(panel["runtime"]["native_libraries"]))
    return result



def main(argv=None):
    """Explicit artifact transport; preflight never silently becomes a panel."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--panel", type=Path, help="independent frozen panel; omit only with --prepare")
    parser.add_argument("--prepare", action="store_true", help="emit untimed native/runtime facts for panel preparation")
    parser.add_argument("--profile", action="store_true", help="separate-process Torch profiler replay; no resource collector or price receipt")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warmup-iterations", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--resource-library", type=Path, help="CUPTI collector built before this fresh process starts")
    args = parser.parse_args(argv)
    if args.prepare == (args.panel is not None):
        parser.error("supply exactly one of --prepare or --panel")
    if args.profile and args.prepare:
        parser.error("--profile requires --panel")
    request = json.loads(args.request.read_text())
    dense._fields(request, ("schema", "unit", "members", "shape", "routing", "execution", "runtime_image",
        "tensors_path", "profile_role_order", "routing_capture_sha256", "phases", "serving_config_path"), "request")
    if request["schema"] != REQUEST_SCHEMA:
        raise ValueError("native request schema unsupported")
    def artifact(key):
        path = Path(request[key])
        return path if path.is_absolute() else args.request.parent / path
    protected = {args.request.resolve(), *(artifact(key).resolve() for key in
                  ("serving_config_path", "tensors_path"))}
    for member in request["members"]:
        for key in ("wire_path", "wire_record_path"):
            path = Path(member[key])
            protected.add((path if path.is_absolute() else args.request.parent / path).resolve())
    if args.panel is not None:
        protected.add(args.panel.resolve())
    if args.out.resolve() in protected:
        raise ValueError("receipt output would overwrite an input artifact")
    if args.profile and any(Path(str(args.out) + f".{phase}.trace.json").resolve() in protected for phase in PHASES):
        raise ValueError("profile output would overwrite an input artifact")
    collector = None
    collector_finished = False
    collector_library_sha256 = None
    profiling_library = None
    trace_path = args.out.with_suffix(args.out.suffix + ".memory.json")
    if args.resource_library:
        protected.add(args.resource_library.resolve())
        if trace_path.resolve() in protected or args.out.resolve() in protected:
            raise ValueError("resource output would overwrite an input artifact")
        from experiments.native_operator_resources import NativeMemoryCollector
        if args.profile:
            # Load the exact observer binary for runtime identity, but never
            # register its CUPTI callbacks while Kineto owns collection.
            import ctypes
            profiling_library = ctypes.CDLL(str(args.resource_library.resolve(strict=True)))
            collector_library_sha256 = hashlib.sha256(args.resource_library.read_bytes()).hexdigest()
        else:
            collector = NativeMemoryCollector(args.resource_library)
            collector_library_sha256 = collector.library_sha256
    def run():
        nonlocal collector_finished
        from safetensors.torch import load_file
        tensors = load_file(str(artifact("tensors_path")), device="cuda")
        phase_tensors = {phase: {key: tensors[f"{phase}.{key}"] for key in TENSOR_KEYS} for phase in PHASES}
        expected_names = {f"{phase}.{key}" for phase in PHASES for key in TENSOR_KEYS}
        expected_names.update(f"{kind}/{m['unit']}" for m in request["members"] for kind in ("source_weight", "rendered_weight"))
        if request["routing"]["source_protocol"]["selection_bias"] is not None:
            expected_names.add("routing_bias")
        if set(tensors) != expected_names:
            raise ValueError("request safetensors roster differs from the complete routed owner")
        member_inputs = []
        for member in request["members"]:
            dense._fields(member, ("unit", "expert", "role", "format", "wire_path", "wire_record_path"), "request member")
            def member_path(key):
                path = Path(member[key])
                return path if path.is_absolute() else args.request.parent / path
            member_inputs.append({**{key: member[key] for key in ("unit", "expert", "role", "format")},
                "blob": member_path("wire_path").read_bytes(),
                "record": json.loads(member_path("wire_record_path").read_text())})
        dense._fields(request["phases"], PHASES, "request phases")
        for phase in PHASES:
            dense._fields(request["phases"][phase], ("transport",), "request phase")
        prepared = prepare_native_moe_operator(member_inputs,
            {key: value for key, value in tensors.items() if key.startswith(("source_weight/", "rendered_weight/"))},
            phase_tensors, unit=request["unit"], shape=request["shape"], routing=request["routing"],
            runtime_image=request["runtime_image"], execution=request["execution"],
            profile_role_order=request["profile_role_order"], serving_config=serving_config,
            routing_capture_sha256=request["routing_capture_sha256"], routing_bias=tensors.get("routing_bias"),
            phase_transport={phase: request["phases"][phase]["transport"] for phase in PHASES},
            warmup_iterations=args.warmup_iterations)
        if collector_library_sha256 is not None:
            prepared["runtime"]["resource_collector"] = {
                "library_sha256": collector_library_sha256,
                "analysis_source_sha256": hashlib.sha256(
                    Path(__file__).with_name("native_operator_resources.py").read_bytes()).hexdigest()}
        if args.prepare:
            result = {"schema": "tessera.native_moe_preflight.v1", "status": "untimed_preparation",
                      "operator": prepared["operator"], "runtime": prepared["runtime"],
                      "native_tensors_sha256": dense.identity_sha256(prepared["operator"]["native_tensors"]),
                      "scheme_sha256": prepared["operator"]["scheme_sha256"],
                      "workspace": prepared["workspace"],
                      "workspace_sha256": dense.identity_sha256(prepared["workspace"]),
                      "runtime_sha256": dense.identity_sha256(prepared["runtime"])}
        else:
            if args.profile:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                result = profile_prepared_operator(prepared, json.loads(args.panel.read_text()), phase_tensors,
                    output_prefix=args.out, warmup_iterations=args.warmup_iterations, iterations=args.iterations)
            else:
                result = measure_prepared_operator(prepared, json.loads(args.panel.read_text()), phase_tensors,
                    warmup_iterations=args.warmup_iterations, iterations=args.iterations,
                    resource_collector=collector)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if collector is not None:
            collector_finished = True
            trace = collector.finish(trace_path)
            if not args.prepare:
                attach_resource_trace(result, trace)
                if result["status"] == "resources_observed" and result["resources"]["status"] == "complete_operator_bound":
                    time_after_resource_collection(prepared, json.loads(args.panel.read_text()), phase_tensors,
                        result, collector=collector, warmup_iterations=args.warmup_iterations, iterations=args.iterations)
        args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        # pbrun's CAS result is stdout. Bind the separately retained artifact
        # bytes to that result, rather than relying on an external path alone.
        published = {"schema": "tessera.native_moe_publication.v1",
                     "status": result["status"], "receipt_path": str(args.out),
                     "receipt_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
        if collector is not None:
            published["memory_trace_path"] = str(trace_path)
            published["memory_trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        print(json.dumps(published, sort_keys=True), flush=True)
        admissible = (result["status"] == "profiled" if args.profile else result["status"] == "timing_admissible" and
                      (collector is None or result["resources"]["status"] == "complete_operator_bound"))
        return 0 if args.prepare or admissible else 2
    try:
        vllm_config, serving_config = resolve_serving_config(artifact("serving_config_path"), request["runtime_image"])
        with native_runtime_context(vllm_config):
            return run()
    finally:
        # Refused preparation/measurement keeps the raw failure trace too.
        if collector is not None and not collector_finished:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            collector.finish(trace_path)


if __name__ == "__main__":
    raise SystemExit(main())
