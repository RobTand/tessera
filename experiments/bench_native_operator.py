"""Research receipts for one actual eager/resident/TP1 dense serving operator.

Preparation is separate from measurement: callers freeze its native/runtime
identities in an independent panel before collecting timings. This is not a
vLLM engine benchmark, format admission, or complete resource price. An optional
fresh-process CUPTI collector proves a conservative operator scratch bound;
fixed/full-model resources still require independent engine evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from contextlib import contextmanager

PANEL_SCHEMA = "tessera.native_dense_panel.v1"
RECEIPT_SCHEMA = "tessera.native_dense_operator_receipt.v1"
RUNTIME_SCHEMA = "tessera.native_dense_runtime.v1"
EXECUTION = {"owner_kind": "single_dense", "mode": "resident",
             "execution_mode": "eager", "tensor_parallel": 1, "bias": False}
PHASES = ("prefill", "decode")
ROUTE_KEYS = {"kind", "policy", "symbol", "decoder", "contract"}


@contextmanager
def native_runtime_context(vllm_config=None):
    """Fresh-process vLLM TP1 context required by real BasevLLMParameter."""
    import tempfile
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (init_distributed_environment,
        ensure_model_parallel_initialized, destroy_model_parallel, destroy_distributed_environment)
    if torch.distributed.is_initialized():
        raise ValueError("standalone native receipt requires a fresh distributed context")
    with tempfile.TemporaryDirectory(prefix="tessera-native-tp1-") as temporary:
        rendezvous = Path(temporary) / "rendezvous"
        with set_current_vllm_config(VllmConfig() if vllm_config is None else vllm_config):
            try:
                init_distributed_environment(world_size=1, rank=0,
                    distributed_init_method=rendezvous.as_uri(),
                    local_rank=torch.cuda.current_device(), backend="gloo")
                ensure_model_parallel_initialized(1, 1)
                yield
            finally:
                destroy_model_parallel()
                destroy_distributed_environment()


def identity_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name}: missing or unknown fields")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: integer >= {minimum} required")


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name}: lowercase SHA256 required")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: finite nonnegative number required")


def _tensor_record(value, shape, name):
    _fields(value, ("shape", "dtype", "logical_bytes", "content_sha256"), name)
    if value["shape"] != shape or value["dtype"] != "torch.bfloat16":
        raise ValueError(f"{name}: exact BF16 shape {shape} required")
    _integer(value["logical_bytes"], name + ".logical_bytes")
    if value["logical_bytes"] != math.prod(shape) * 2:
        raise ValueError(f"{name}: logical byte count differs from shape/dtype")
    _sha(value["content_sha256"], name)


def validate_panel(panel):
    """Validate the frozen narrow scope before importing CUDA/vLLM."""
    _fields(panel, ("schema", "unit", "format", "shape", "source_sha256", "calibration_sha256",
                    "cost_sha256", "probe_identity_sha256", "joint_operator_identity_sha256",
                    "joint_operator_identity", "wire", "execution", "runtime",
                    "native_tensors_sha256", "scheme_sha256", "numerics", "phases"), "panel")
    if panel["schema"] != PANEL_SCHEMA:
        raise ValueError("panel schema unsupported")
    # Equality alone admits True == 1 and 0 == False.
    if identity_sha256(panel["execution"]) != identity_sha256(EXECUTION):
        raise ValueError("execution: only single dense eager resident TP1 without bias is supported")
    for key in ("unit", "format"):
        if not isinstance(panel[key], str) or not panel[key]:
            raise ValueError(f"{key}: nonempty string required")
    shape = panel["shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("shape: [N,K] required")
    for n in shape:
        _integer(n, "shape")
    for key in ("source_sha256", "calibration_sha256", "cost_sha256", "probe_identity_sha256",
                "joint_operator_identity_sha256", "native_tensors_sha256", "scheme_sha256"):
        _sha(panel[key], key)
    joint = panel["joint_operator_identity"]
    if identity_sha256(joint) != panel["joint_operator_identity_sha256"]:
        raise ValueError("joint operator identity SHA256 mismatch")
    if (joint.get("qname") != panel["unit"] or joint.get("format") != panel["format"]
            or joint.get("probe_identity_sha256") != panel["probe_identity_sha256"]):
        raise ValueError("joint operator unit/format/probe identity mismatch")
    for key in ("source_weight", "rendered_weight"):
        _tensor_record(joint[key], shape, key)
    if joint["activation"].get("clip_enabled") is not False:
        raise ValueError("native operator has no extra calibrated activation preclip")
    wire = panel["wire"]
    _fields(wire, ("blob_sha256", "blob_bytes", "record"), "wire")
    _sha(wire["blob_sha256"], "wire blob")
    _integer(wire["blob_bytes"], "wire blob bytes")
    record = wire["record"]
    _fields(record, ("file", "blob_sha256", "blob_bytes", "identity"), "wire record")
    if record["blob_sha256"] != wire["blob_sha256"] or record["blob_bytes"] != wire["blob_bytes"]:
        raise ValueError("wire record and panel blob identity disagree")
    if record["identity"]["source"]["shape"] != shape:
        raise ValueError("wire source shape differs from panel")
    _fields(panel["numerics"], ("atol", "rtol"), "numerics")
    for key, value in panel["numerics"].items():
        _number(value, key)
    runtime = panel["runtime"]
    if not isinstance(runtime, dict) or runtime.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("runtime manifest schema unsupported")
    if identity_sha256(runtime.get("execution")) != identity_sha256(EXECUTION):
        raise ValueError("runtime execution differs from panel")
    identity_sha256(runtime)  # refuse non-JSON/nonfinite coordinates
    _fields(panel["phases"], PHASES, "phases")
    for phase, item in panel["phases"].items():
        _fields(item, ("m", "input", "reference_qdq", "reference_output", "expected_route"), phase)
        _integer(item["m"], phase + ".m")
        for key, width in (("input", shape[1]), ("reference_qdq", shape[1]), ("reference_output", shape[0])):
            _tensor_record(item[key], [item["m"], width], phase + "." + key)
        route = item["expected_route"]
        _fields(route, ROUTE_KEYS, "expected route")
        if route["kind"] != "dense" or not route["policy"].endswith(":resident"):
            raise ValueError("expected route must be dense/resident")
        if any(not isinstance(v, str) or not v for v in route.values()):
            raise ValueError("expected route fields must be nonempty strings")
    if panel["phases"]["prefill"]["expected_route"] != panel["phases"]["decode"]["expected_route"]:
        raise ValueError("resident v1 phases require the same operator route")
    return json.loads(json.dumps(panel, allow_nan=False))


def tensor_identity(tensor):
    """The raw contiguous-byte convention used by PQ's actual PWC row."""
    import torch
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "logical_bytes": value.numel() * value.element_size(),
            "content_sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def compare_tensors(actual, expected, *, atol, rtol):
    """Explicit elementwise tolerance; nonfinite errors are JSON null failures."""
    import torch
    _number(atol, "atol")
    _number(rtol, "rtol")
    if actual.shape != expected.shape:
        raise ValueError("numerical comparison shape mismatch")
    a, b = actual.double(), expected.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a - b).abs()
    denominator = atol + rtol * b.abs()
    ratio = torch.where(delta == 0, torch.zeros_like(delta), delta / denominator)
    error = float(delta.max())
    normalized = float(ratio.max())
    passed = finite and math.isfinite(normalized) and normalized <= 1
    return {"status": "passed" if passed else "failed", "atol": atol, "rtol": rtol,
            "finite": finite, "max_normalized_error": normalized if math.isfinite(normalized) else None,
            "max_abs_error": error if math.isfinite(error) else None, "numel": actual.numel()}


def time_apply(apply, *, warmup_iterations, iterations):
    """One complete apply per event pair. No loop-averaged observations."""
    import torch
    _integer(warmup_iterations, "warmup_iterations")
    _integer(iterations, "iterations", 3)
    for _ in range(warmup_iterations):
        apply()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = apply()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("CUDA event sample must be finite and positive")
        samples.append(elapsed)
        del result
    return {"method": "cuda_events", "sample_unit": "single_apply",
            "warmup_iterations": warmup_iterations, "samples_ms": samples}


def _require_cuda_tensor(tensor):
    import torch
    if tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
        raise ValueError("native receipt requires actual 2-D CUDA BF16 tensors")
    if tensor.device.index != torch.cuda.current_device():
        raise ValueError("native receipt tensor must use the current CUDA device")


def _require_eager_context():
    import torch
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        raise ValueError("execution requires eager calls outside compilation/CUDA graph capture")


def observe_arithmetic():
    import torch
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}


def _mapped_shared_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/") and ".so" in parts[5]:
            paths.add(Path(parts[5]).resolve())
    return paths


def _check_native_library_scope(runtime):
    declared = set(runtime["native_libraries"])
    unobserved = {str(path) for path in _mapped_shared_libraries()} - declared
    if unobserved:
        raise ValueError("native libraries loaded after preparation: " + ", ".join(sorted(unobserved)))


def _native_tensors(layer):
    import torch
    result = {name: tensor_identity(value) for name, value in
              list(layer.named_parameters()) + list(layer.named_buffers())}
    # NVFP4's externally applied epilogue is a Python scalar, not a buffer.
    for name in ("tessera_epilogue_scale", "tessera_global_scale_real"):
        if hasattr(layer, name):
            result["$scalar." + name] = tensor_identity(torch.tensor(getattr(layer, name), dtype=torch.float64))
    if not result:
        raise ValueError("native operator has no observed resident tensors")
    return result


def _resident_bytes(layer):
    storages = {}
    for _name, value in list(layer.named_parameters()) + list(layer.named_buffers()):
        if value.device.type != "cuda":
            raise ValueError("loaded native storage is not CUDA resident")
        storage = value.untyped_storage()
        storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def represented_native_input(layer, x):
    """Decode the real native quantizer's values, without adding a preclip."""
    import torch
    from tessera.serving import native_ops
    from tessera.serving.scheme import TESSERA_BF16, TESSERA_FP8, TESSERA_NVFP4
    if layer.tessera_family == TESSERA_BF16:
        return x.to(torch.bfloat16)
    if layer.tessera_family == TESSERA_FP8:
        codes, scales = native_ops.native_fp8_quant(x.contiguous())
        return (codes.float() * scales).to(torch.bfloat16)
    if layer.tessera_family != TESSERA_NVFP4:
        raise ValueError("unknown native activation owner")
    from tessera.alphabet import E2M1_VALUES
    from tessera.serving.nvfp4_route import blocked_scales, GROUP_SIZE
    g = layer.trellis_input_global_scale.data.reshape(())
    packed, blocked = native_ops.native_fp4_quant(x.contiguous(), g)
    m, k = x.shape
    # Ask the existing owner for the permutation; do not restate its layout.
    ids = torch.arange(1, m * (k // GROUP_SIZE) + 1, device=x.device).reshape(m, -1)
    permutation = blocked_scales(ids)
    valid = permutation > 0
    scales = torch.empty(ids.numel(), dtype=torch.float32, device=x.device)
    scales[permutation[valid] - 1] = blocked.reshape(-1).view(torch.float8_e4m3fn).float()[valid]
    packed = packed.view(torch.uint8).reshape(m, k // 2)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(m, k).long()
    levels = torch.tensor(E2M1_VALUES, device=x.device, dtype=torch.float32)
    return (levels[codes].reshape(m, -1, GROUP_SIZE) * scales.reshape(m, -1, 1) / g).reshape(m, k).to(torch.bfloat16)


def observe_runtime(runtime_image):
    """Observe loaded binaries and source after native preparation, before timing."""
    import importlib.metadata
    import subprocess
    import sys
    from uuid import UUID
    import torch
    import tessera
    from tessera.cached_unit import encoder_source_sha256
    from tessera.serving.runtime_image import declared_reference
    declaration = declared_reference(runtime_image)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    # Torch's _CUuuid omits NVIDIA-SMI's GPU- prefix. Join the actual UUID,
    # never a visible index (CUDA_VISIBLE_DEVICES can remap that index).
    uuid = "GPU-" + str(UUID(str(getattr(props, "uuid", "")).removeprefix("GPU-")))
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version",
                                     "--format=csv,noheader"], text=True).splitlines()
    drivers = {parts[0].strip(): parts[1].strip() for line in lines
               if len(parts := line.split(",")) == 2}
    if uuid not in drivers:
        raise ValueError("runtime driver observation does not identify the CUDA UUID")
    paths = {Path(p).resolve() for p in torch.ops.loaded_libraries}
    paths.add(Path(torch._C.__file__).resolve())
    paths.update(p.resolve() for p in (Path(torch.__file__).parent / "lib").glob("*cuda*.so*"))
    # Include dynamically loaded CUDA/cuBLAS dependencies, not only Python's
    # torch.ops registration list. This is a Linux/CUDA research harness.
    paths.update(_mapped_shared_libraries())
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("vllm") and filename and ".so" in filename:
            paths.add(Path(filename).resolve())
    # CUDA's Python binding packages can load distinct extensions with the
    # same basename. Preserve their canonical paths and bytes independently.
    libraries = {}
    for path in sorted(paths):
        with path.open("rb") as stream:
            libraries[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    package = Path(tessera.__file__).parent
    encoder_source_sha256.cache_clear()
    return {"schema": RUNTIME_SCHEMA, "image": runtime_image, "image_declaration": declaration,
            "execution": dict(EXECUTION), "arithmetic": observe_arithmetic(),
            "versions": {"torch": torch.__version__, "vllm": importlib.metadata.version("vllm"),
                         "cuda": torch.version.cuda},
            "gpu": {"name": props.name, "uuid": uuid, "capability": [props.major, props.minor],
                    "total_memory": props.total_memory, "driver_version": drivers[uuid]},
            "source": {"tessera_package_sha256": encoder_source_sha256(),
                       "runtime_contract_sha256": hashlib.sha256((package / "serving/runtime_contract.json").read_bytes()).hexdigest(),
                       "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "native_libraries": libraries}


def prepare_native_operator(blob, record, source_weight, rendered_weight, *, unit, format_name,
                            runtime_image, input_global_scale=None, execution=None):
    """Actual create/load/process lifecycle; returns facts for freezing a panel.

    Original unit bytes are framed once through the existing single-role fused
    container owner. Multiple roles and already fused input blobs are refused.
    This preparation emits no timing or full-resource admission.
    """
    import torch
    from tessera.cached_unit import verify_cached_unit, tensor_identity as producer_tensor_identity
    from tessera.fused import pack_fused
    from tessera.serving.lane import build_tessera_method
    from tessera.serving.scheme import ROUTES, TESSERA_NVFP4, validate_tessera_scheme, launch_pairs, STRUCTURE_DENSE
    from tessera.unit_artifact import read_unit_artifact
    if identity_sha256(execution if execution is not None else EXECUTION) != identity_sha256(EXECUTION):
        raise ValueError("only single dense eager resident TP1 preparation is supported")
    _require_cuda_tensor(source_weight)
    _require_cuda_tensor(rendered_weight)
    if source_weight.shape != rendered_weight.shape:
        raise ValueError("source/render shape mismatch")
    expected = record["identity"]
    if expected["source"] != producer_tensor_identity(source_weight):
        raise ValueError("wire producer source differs from actual source weight")
    if expected["unit"] != unit.removesuffix(".weight"):
        raise ValueError("wire producer unit differs from requested unit")
    accepted = verify_cached_unit(blob, record, expected)
    decoded = read_unit_artifact(blob, device=str(rendered_weight.device)).to(torch.bfloat16)
    if tensor_identity(decoded) != tensor_identity(rendered_weight):
        raise ValueError("original wire bytes-only decode differs from actual PWC render")
    recipe = expected["recipe"]
    families = [family for family, route in ROUTES.items() if recipe["grid"] in route["grids"]]
    if len(families) != 1:
        raise ValueError("wire has no unique dense serving owner")
    family = families[0]
    launches = launch_pairs(family, structure=STRUCTURE_DENSE, mode="resident")
    if len(launches) != 1:
        raise ValueError("resident dense declaration does not identify one native route")
    declared_symbol, declared_decoder = next(iter(launches))
    rows, columns = source_weight.shape
    container = pack_fused([("weight", rows, blob)])
    manifest = accepted.manifest
    scheme = validate_tessera_scheme({"family": family, "grid": recipe["grid"],
        "body": manifest.body.name, "plane": manifest.scale_plane.kind.name,
        "q256": recipe["q256"], "rows": rows, "columns": columns,
        "wire_bytes": len(container), "roles": [["weight", rows]]}, unit)
    # A format name is a separate panel binding; verify its actual grid/arity/rung.
    grid_name = recipe["grid"].split("x", 1)[0]
    arity = recipe["grid"].split("x", 1)[1] if "x" in recipe["grid"] else "1"
    if format_name != f"TESSERA_{grid_name}_K{arity}_R{recipe['q256']}":
        raise ValueError("format differs from original wire recipe")
    method = build_tessera_method(scheme, unit, "resident")
    layer = torch.nn.Module()
    layer.tp_rank, layer.tp_size = 0, 1
    method.create_weights(layer, input_size_per_partition=columns, output_partition_sizes=[rows],
                          input_size=columns, output_size=rows, params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(container), dtype=torch.uint8).clone()
    if family == TESSERA_NVFP4:
        if type(input_global_scale) not in (int, float) or not math.isfinite(input_global_scale) or input_global_scale <= 0:
            raise ValueError("NVFP4 requires the independently calibrated positive input_global_scale")
        layer.trellis_input_global_scale.data.fill_(input_global_scale)
    elif input_global_scale is not None:
        raise ValueError("dynamic/identity activation route must not carry static input_global_scale")
    layer.to(source_weight.device)
    # PreparedWindow seals tensor version counters during loading. Inference
    # tensors have no version counter; the loader runs with gradients disabled.
    with torch.no_grad():
        method.process_weights_after_loading(layer)
    actual_g = float(layer.trellis_input_global_scale.reshape(())) if family == TESSERA_NVFP4 else None
    operator = {"wire_sha256": hashlib.sha256(blob).hexdigest(), "wire_record_sha256": identity_sha256(record),
                "rendered_weight": tensor_identity(decoded), "activation_contract": layer.tessera_activation_contract,
                "input_global_scale": actual_g, "clip_enabled": False,
                "scheme": json.loads(json.dumps(scheme)), "scheme_sha256": identity_sha256(scheme),
                "declared_route": {"kind": "dense", "policy": f"{family}:resident",
                    "symbol": declared_symbol, "decoder": declared_decoder,
                    "contract": ROUTES[family]["activation_contract"]},
                "native_tensors": _native_tensors(layer)}
    operator["source_weight"] = tensor_identity(source_weight)
    return {"method": method, "layer": layer, "operator": operator, "runtime": observe_runtime(runtime_image)}


def _check_prepared(prepared, panel):
    operator, layer = prepared["operator"], prepared["layer"]
    _require_eager_context()
    if (type(getattr(layer, "tp_size", None)) is not int or layer.tp_size != 1
            or type(getattr(layer, "tp_rank", None)) is not int or layer.tp_rank != 0
            or layer.tessera_mode != "resident"
            or layer.tessera_family != operator["scheme"]["family"]):
        raise ValueError("actual native execution differs from single dense resident TP1 panel")
    if observe_arithmetic() != panel["runtime"].get("arithmetic"):
        raise ValueError("actual arithmetic settings differ from independent panel")
    _check_native_library_scope(panel["runtime"])
    if prepared["runtime"] != panel["runtime"]:
        raise ValueError("observed runtime differs from independent panel")
    if identity_sha256(operator["native_tensors"]) != panel["native_tensors_sha256"]:
        raise ValueError("native tensor identity differs from independent panel")
    if _native_tensors(layer) != operator["native_tensors"]:
        raise ValueError("native tensor state changed after preparation")
    if (identity_sha256(operator["scheme"]) != panel["scheme_sha256"]
            or operator["scheme_sha256"] != panel["scheme_sha256"]):
        raise ValueError("scheme identity differs from independent panel")
    scheme = operator["scheme"]
    if (scheme["rows"], scheme["columns"]) != tuple(panel["shape"]) or scheme["roles"] != [["weight", panel["shape"][0]]]:
        raise ValueError("scheme is not the declared single dense owner/shape")
    if operator["wire_sha256"] != panel["wire"]["blob_sha256"] or operator["wire_record_sha256"] != identity_sha256(panel["wire"]["record"]):
        raise ValueError("wire identity differs from independent panel")
    joint = panel["joint_operator_identity"]
    for key in ("source_weight", "rendered_weight"):
        if operator[key] != joint[key]:
            raise ValueError(f"{key} identity differs from joint cost")
    if operator["input_global_scale"] != joint["activation"]["input_global_scale"] or operator["clip_enabled"] is not False:
        raise ValueError("native activation scale/policy differs from joint cost")
    if operator["activation_contract"] != panel["phases"]["prefill"]["expected_route"]["contract"]:
        raise ValueError("native activation contract differs from panel")


def _check_phase_tensors(panel, phase_tensors):
    for phase in PHASES:
        tensors, expected = phase_tensors[phase], panel["phases"][phase]
        _fields(tensors, ("input", "reference_qdq", "reference_output"), phase + " tensors")
        for name, tensor in tensors.items():
            _require_cuda_tensor(tensor)
            if tensor_identity(tensor) != expected[name]:
                raise ValueError(f"{phase}: actual {name} differs from independent panel")


def measure_prepared_operator(prepared, panel, phase_tensors, *, warmup_iterations, iterations,
                              resource_collector=None):
    """Gate BOTH phases/QDQ before any timings; retain explicit resource gaps."""
    import torch
    from tessera.serving.telemetry import read_route
    panel = validate_panel(panel)
    _integer(warmup_iterations, "warmup_iterations")
    _integer(iterations, "iterations", 3)
    _fields(phase_tensors, PHASES, "phase tensors")
    _check_phase_tensors(panel, phase_tensors)
    _check_prepared(prepared, panel)
    layer, method = prepared["layer"], prepared["method"]
    observations, resource_phases = {}, {}
    with torch.inference_mode():
        for phase in PHASES:
            tensors, expected = phase_tensors[phase], panel["phases"][phase]
            _check_phase_tensors(panel, phase_tensors)
            x = tensors["input"]
            qdq = represented_native_input(layer, x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            output = method.apply(layer, x)
            torch.cuda.synchronize()
            peak = max(0, torch.cuda.max_memory_allocated() - before)
            route = read_route(layer)
            wanted = expected["expected_route"]
            if (not isinstance(route, dict) or any(route.get(k) != v for k, v in wanted.items())
                    or route.get("state") != "served" or route.get("reason") is not None
                    or route.get("shape") != f"M{expected['m']}:N{panel['shape'][0]}:K{panel['shape'][1]}"):
                raise ValueError(f"{phase}: actual route differs from independent panel")
            qdq_error = compare_tensors(qdq, tensors["reference_qdq"], **panel["numerics"])
            error = compare_tensors(output, tensors["reference_output"], **panel["numerics"])
            if not bool(torch.isfinite(x).all()):
                for record in (qdq_error, error):
                    record.update(status="failed", finite=False)
            observations[phase] = {"m": expected["m"], "input": tensor_identity(x),
                "reference_qdq": tensor_identity(tensors["reference_qdq"]), "qdq": tensor_identity(qdq),
                "qdq_numerics": qdq_error, "reference_output": tensor_identity(tensors["reference_output"]),
                "output": tensor_identity(output), "route": route, "numerics": error, "measurement": None}
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
                    observations[phase]["measurement"] = time_apply(
                        lambda phase=phase: method.apply(layer, phase_tensors[phase]["input"]),
                        warmup_iterations=warmup_iterations, iterations=iterations)
                else:
                    # Warm the allocator with real applies, but do not price
                    # calls while CUPTI memory/API collection is active.
                    for _ in range(warmup_iterations):
                        method.apply(layer, phase_tensors[phase]["input"])
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
                        lambda phase=phase: method.apply(layer, phase_tensors[phase]["input"]),
                        phase, device=phase_tensors[phase]["input"].device.index)
                    error = compare_tensors(output, phase_tensors[phase]["reference_output"], **panel["numerics"])
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
            "panel": panel, "panel_sha256": identity_sha256(panel), "runtime": prepared["runtime"],
            "runtime_sha256": identity_sha256(prepared["runtime"]), "operator": prepared["operator"],
            "phases": observations, "resources": {"status": "incomplete", "scope": "torch_allocator_observation",
                "resident_bytes": _resident_bytes(layer), "phases": resource_phases,
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
    if receipt["panel_sha256"] != identity_sha256(panel):
        raise ValueError("timing panel differs from resource panel")
    import torch
    with torch.inference_mode():
        for phase in PHASES:
            _check_prepared(prepared, panel)
            _check_phase_tensors(panel, phase_tensors)
            receipt["phases"][phase]["measurement"] = time_apply(
                lambda phase=phase: prepared["method"].apply(prepared["layer"], phase_tensors[phase]["input"]),
                warmup_iterations=warmup_iterations, iterations=iterations)
            _check_prepared(prepared, panel)
            _check_phase_tensors(panel, phase_tensors)
            if read_route(prepared["layer"]) != receipt["phases"][phase]["route"]:
                raise ValueError(f"{phase}: native route changed during decision timing")
    receipt["status"] = "timing_admissible"
    receipt["timing_scope"] = "cuda_events_after_resource_collector_stop"
    return receipt


def attach_resource_trace(receipt, trace):
    """Compose operator bounds only; never infer engine fixed/KV resources."""
    from experiments.native_operator_resources import analyze_trace
    resources = receipt["resources"]
    for phase in PHASES:
        observed = resources["phases"][phase]
        if "torch_observation" in observed:
            observed["bound"] = analyze_trace(trace, interval=phase,
                                               torch_observation=observed["torch_observation"])
    resources["trace_sha256"] = identity_sha256(trace)
    complete = all(resources["phases"][phase].get("bound", {}).get("status") == "complete_operator_bound"
                   for phase in PHASES)
    if complete:
        resources.update(status="complete_operator_bound", scope="warmed_eager_operator",
                         unknown=["fixed_and_full_model_resources"])
    return receipt


def profile_prepared_operator(prepared, panel, phase_tensors, *, output_prefix,
                              warmup_iterations, iterations):
    """Separate-process profiling evidence; never a resource or price receipt."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    gate = measure_prepared_operator(prepared, panel, phase_tensors,
        warmup_iterations=warmup_iterations, iterations=iterations)
    result = {"schema": "tessera.native_dense_profile.v1", "status": "numerical_refused",
              "panel_sha256": gate["panel_sha256"], "runtime_sha256": gate["runtime_sha256"],
              "scope": "separate_process_torch_profiler_replay_not_resource_or_price_receipt",
              "phases": {}}
    if gate["status"] != "timing_admissible":
        return result
    for phase in PHASES:
        # Both phases already passed the strict unprofiled gate. Profiler
        # instrumentation may map extra libraries of its own after entry.
        if _native_tensors(prepared["layer"]) != prepared["operator"]["native_tensors"]:
            raise ValueError("native tensors changed before profiling")
        if observe_arithmetic() != panel["runtime"]["arithmetic"]:
            raise ValueError("arithmetic changed before profiling")
        _check_phase_tensors(panel, phase_tensors)
        with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                             record_shapes=True, profile_memory=True) as prof:
            output = prepared["method"].apply(prepared["layer"], phase_tensors[phase]["input"])
            torch.cuda.synchronize()
        numerics = compare_tensors(output, phase_tensors[phase]["reference_output"], **panel["numerics"])
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
        {str(path) for path in _mapped_shared_libraries()} - set(panel["runtime"]["native_libraries"]))
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
    _fields(request, ("schema", "unit", "format", "wire_path", "wire_record_path", "tensors_path",
                      "runtime_image", "input_global_scale", "execution"), "request")
    if request["schema"] != "tessera.native_dense_request.v1":
        raise ValueError("native request schema unsupported")
    def artifact(key):
        path = Path(request[key])
        return path if path.is_absolute() else args.request.parent / path
    protected = {args.request.resolve(), *(artifact(key).resolve() for key in
                  ("wire_path", "wire_record_path", "tensors_path"))}
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
        prepared = prepare_native_operator(artifact("wire_path").read_bytes(),
            json.loads(artifact("wire_record_path").read_text()), tensors["source_weight"], tensors["rendered_weight"],
            unit=request["unit"], format_name=request["format"], runtime_image=request["runtime_image"],
            input_global_scale=request["input_global_scale"], execution=request["execution"])
        if collector_library_sha256 is not None:
            prepared["runtime"]["resource_collector"] = {
                "library_sha256": collector_library_sha256,
                "analysis_source_sha256": hashlib.sha256(
                    Path(__file__).with_name("native_operator_resources.py").read_bytes()).hexdigest()}
        if args.prepare:
            result = {"schema": "tessera.native_dense_preflight.v1", "status": "untimed_preparation",
                      "operator": prepared["operator"], "runtime": prepared["runtime"],
                      "native_tensors_sha256": identity_sha256(prepared["operator"]["native_tensors"]),
                      "scheme_sha256": prepared["operator"]["scheme_sha256"],
                      "runtime_sha256": identity_sha256(prepared["runtime"])}
        else:
            phase_tensors = {phase: {key: tensors[f"{phase}.{key}"] for key in
                             ("input", "reference_qdq", "reference_output")} for phase in PHASES}
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
        published = {"schema": "tessera.native_dense_publication.v1",
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
        with native_runtime_context():
            return run()
    finally:
        # Refused preparation/measurement keeps the raw failure trace too.
        if collector is not None and not collector_finished:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            collector.finish(trace_path)


if __name__ == "__main__":
    raise SystemExit(main())
