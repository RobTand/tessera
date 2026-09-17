"""Read-only resolved KV observations for stock worker RPCs.

Importing this module does not construct a worker or start a resource recorder.
The assertions verify an explicit selected capacity, not full resource admission.
"""
import dataclasses
import enum
import hashlib
import json
import math
from fractions import Fraction


def kv_config_value(value):
    """Explicit JSON form for the pinned runtime's resolved KV dataclasses."""
    import torch
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, enum.Enum):
        return {"enum_type": kind, "value": kv_config_value(value.value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": kind, "fields": {field.name: kv_config_value(getattr(value, field.name))
                                          for field in dataclasses.fields(value)}}
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value)}
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (tuple, list)):
        return [kv_config_value(item) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: kv_config_value(value[key]) for key in sorted(value)}
    raise TypeError(f"unsupported resolved KV configuration value: {kind}")


def kv_configuration_observation(worker, received):
    runner = getattr(worker, "model_runner", None)
    resolved = getattr(runner, "kv_cache_config", None)
    cache = getattr(getattr(worker, "vllm_config", None), "cache_config", None)
    policy_fields = ("gpu_memory_utilization", "kv_cache_memory_bytes", "num_gpu_blocks_override",
                     "block_size", "user_specified_block_size", "cache_dtype",
                     "mamba_block_size", "user_specified_mamba_block_size", "mamba_cache_dtype",
                     "mamba_ssm_cache_dtype", "mamba_cache_mode", "enable_prefix_caching")
    values = {name: kv_config_value(getattr(cache, name)) for name in policy_fields if hasattr(cache, name)}
    groups = getattr(resolved, "kv_cache_groups", [])
    # The original descriptor projection remains readable by earlier consumers.
    return {"num_blocks": received.num_blocks,
            "tensors": [{"size": tensor.size, "layers": tensor.layers,
                         "layer_stride": tensor.layer_stride, "block_stride": tensor.block_stride,
                         "offset": tensor.offset} for tensor in received.kv_cache_tensors],
            "received": kv_config_value(received) if dataclasses.is_dataclass(received) else None,
            "runner_resolved": kv_config_value(resolved),
            "group_page_size_bytes": [kv_config_value(group.kv_cache_spec.page_size_bytes) for group in groups],
            "kernel_block_sizes": kv_config_value(getattr(runner, "kernel_block_sizes", None)),
            "capacity_policy": {"values": values, "missing_fields": sorted(set(policy_fields) - values.keys())},
            "physical_backing_bytes": None, "runtime_admission": False,
            "scope": "actual stock resolved KV descriptors and policy; physical storage is independently deduplicated"}


def kv_storage_observation(tensors):
    """Deduplicate actual KV storages; reject partially overlapping backings."""
    views, storages = [], {}

    def visit(value, name):
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from visit(item, f"{name}[{index}]")
        elif isinstance(value, dict):
            for key, item in sorted(value.items()):
                yield from visit(item, f"{name}[{key!r}]")
        else:
            yield name, value

    for name, tensor in visit(tensors, "kv_caches"):
        storage = tensor.untyped_storage()
        address, size = int(storage.data_ptr()), int(storage.nbytes())
        device = (tensor.device.type, tensor.device.index)
        offset = tensor.storage_offset() * tensor.element_size()
        shape, stride = list(tensor.shape), list(tensor.stride())
        extent = 0 if 0 in shape else (1 + sum((n - 1) * s for n, s in zip(shape, stride))) * tensor.element_size()
        if size < 0 or offset < 0 or any(s < 0 for s in stride) or offset + extent > size:
            raise ValueError("KV view exceeds its physical backing storage")
        row = {"owner": name, "device_type": device[0], "device_id": device[1],
               "address": address, "bytes": size, "storage_offset_bytes": offset,
               "view_extent_bytes": extent, "shape": shape, "stride": stride, "dtype": str(tensor.dtype)}
        views.append(row)
        if not size:
            continue
        if device[0] != "cuda" or device[1] is None or address <= 0:
            raise ValueError("KV backing is not an observed CUDA storage")
        key = (*device, address)
        entry = storages.setdefault(key, {"device_type": device[0], "device_id": device[1],
                                         "address": address, "bytes": size, "owners": []})
        if entry["bytes"] != size:
            raise ValueError("aliased KV storage has inconsistent physical size")
        entry["owners"].append(name)
    ordered = [storages[key] for key in sorted(storages)]
    for first, second in zip(ordered, ordered[1:]):
        if (first["device_type"], first["device_id"]) == (second["device_type"], second["device_id"]):
            if first["address"] + first["bytes"] > second["address"]:
                raise ValueError("distinct KV storages overlap physically")
    return {"views": views, "storages": ordered,
            "unique_physical_storage_bytes": sum(row["bytes"] for row in ordered),
            "scope": "actual shared CUDA backing storage, deduplicated by device/address/size"}


def inspect_worker_kv(worker, expectations=None, *, received=None):
    """Supported worker-RPC callback; does not alter the worker or KV state."""
    runner = worker.model_runner
    config = runner.kv_cache_config
    result = kv_configuration_observation(worker, config if received is None else received)
    result["received_argument_scope"] = ("post-initialization runner observation" if received is None
                                         else "actual initialization argument")
    result["storage"] = kv_storage_observation(runner.kv_caches)
    vllm_config = worker.vllm_config
    details = []
    for group in config.kv_cache_groups:
        spec = group.kv_cache_spec
        page = spec.page_size_bytes
        resident = spec.max_memory_usage_bytes(vllm_config)
        kind = {"MambaSpec": "recurrent", "FullAttentionSpec": "full_attention"}.get(type(spec).__name__, "unsupported")
        details.append({"kind": kind, "spec_type": f"{type(spec).__module__}.{type(spec).__qualname__}",
                        "layer_names": list(group.layer_names), "page_size_bytes": page,
                        "max_memory_usage_bytes": resident,
                        "resident_pages_per_request": (resident + page - 1) // page,
                        "block_table_entries_per_request": spec.max_num_blocks_per_req(vllm_config, vllm_config.model_config.max_model_len),
                        "block_size": spec.block_size,
                        "num_speculative_blocks": getattr(spec, "num_speculative_blocks", None),
                        "num_prefill_checkpoint_blocks": getattr(spec, "num_prefill_checkpoint_blocks", None),
                        "mamba_cache_mode": getattr(spec, "mamba_cache_mode", None),
                        "num_kv_heads": getattr(spec, "num_kv_heads", None),
                        "head_size": getattr(spec, "head_size", None),
                        "dtype": str(getattr(spec, "dtype", None))})
    result["group_details"] = details
    result["resolved_limits"] = {
        "max_model_len": vllm_config.model_config.max_model_len,
        "max_num_seqs": vllm_config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": vllm_config.scheduler_config.max_num_batched_tokens,
        "tensor_parallel_size": vllm_config.parallel_config.tensor_parallel_size}
    if expectations is not None:
        result["capacity_assertions"] = check_kv_capacity(result, expectations)
    return result


KV_OBSERVATION_SCHEMA = "tessera.full_engine_kv_observation.v1"


def read_only_kv_observation(worker):
    """Supported worker RPC: the engine's own KV descriptors and backings.

    Runs in the worker process with **no** resource recorder attached and takes
    no synchronized snapshot, so this is the read-only pass of the two-pass
    pair. It returns its own process id beside the observation because the pass
    evidence has to name the process that observed, not the launcher that asked
    -- the launcher's pid is the same for every rank and would make two ranks'
    records indistinguishable.
    """
    import os

    return {"process_id": os.getpid(), "observation": inspect_worker_kv(worker)}

#: The digests the two passes of one report must share. They are the run
#: identity's own fields, so a read-only KV pass and the intrusive resource pass
#: are the same run only if these agree; everything else about the two passes
#: (their process ids, their capture digests, their pointers) is expected to
#: differ and is never compared.
COMMON_RUN_DIGESTS = ("assignment_sha256", "canonical_units_sha256",
                      "configuration_sha256", "model_sha256",
                      "runtime_manifest_sha256", "workload_sha256")


def admission_evidence(*, mode, process_id, recorder_attached, snapshot_count):
    """What a KV observation's pass actually did, as the reason for its flag.

    ``runtime_admission`` is not a field the producer gets to assert: the
    intrusive resource pass synchronizes the device for allocator snapshots,
    which alters execution and is explicitly timing- and admission-ineligible,
    while the read-only worker RPC attaches nothing and takes none.  The caller
    reports what its own pass did and this decides the flag, so a record cannot
    claim admission eligibility without naming the read-only pass that earned
    it.
    """
    snapshot_count = int(snapshot_count)
    if snapshot_count < 0:
        raise ValueError("snapshot_count must not be negative")
    read_only = not recorder_attached and snapshot_count == 0
    return {"schema": "tessera.full_engine_kv_pass_evidence.v1",
            "mode": str(mode), "process_id": int(process_id),
            "recorder_attached": bool(recorder_attached),
            "snapshot_count": snapshot_count, "read_only": read_only,
            "reason": ("read-only worker RPC: no resource recorder attached and no allocator "
                       "snapshots taken" if read_only else
                       "intrusive observation pass: the resource recorder is attached and "
                       "synchronized allocator snapshots alter execution, so this record is "
                       "timing- and admission-ineligible")}


def kv_observation_record(observed, *, evidence, rank, world_size, run_identity, scope):
    """One rank's KV observation, in the shape the report consumer recomputes from.

    The record is the observer's OWN resolved descriptors and deduplicated
    physical backings, projected only where the consumer names a coordinate:
    ``num_blocks``, one ``group_page_size_bytes`` per cache group,
    ``resolved_limits``, the ``storage`` block with each backing's ``owners``,
    and the pass evidence.  Every other observed coordinate travels as evidence.

    The record binds the run it belongs to (rank, world size, the run identity's
    digests) because a per-rank charge is only meaningful for the run that
    measured it, and ``runtime_admission`` comes from the evidence rather than
    from this call's arguments.
    """
    if not isinstance(observed, dict) or "storage" not in observed:
        raise ValueError("kv observation requires the observer's own record")
    rank, world_size = int(rank), int(world_size)
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"kv observation rank {rank} is outside world {world_size}")
    missing = [name for name in COMMON_RUN_DIGESTS if name not in (run_identity or {})]
    if missing:
        raise ValueError(f"kv observation run identity is missing {sorted(missing)}")
    storage = observed["storage"]
    rows = []
    for row in storage["storages"]:
        # The consumer reads exactly these five fields per backing; a row that
        # carries anything else is a second spelling of the same pool.
        if set(row) != {"address", "bytes", "device_id", "device_type", "owners"}:
            raise ValueError("kv storage row is not the observer's own deduplicated backing")
        rows.append({name: row[name] for name in
                     ("address", "bytes", "device_id", "device_type", "owners")})
    limits = observed["resolved_limits"]
    return {
        "schema": KV_OBSERVATION_SCHEMA,
        "rank": rank, "world_size": world_size,
        "run_identity": {name: run_identity[name] for name in COMMON_RUN_DIGESTS},
        "process_id": int(evidence["process_id"]),
        # Derived from the evidence, never passed beside it.
        "runtime_admission": bool(evidence["read_only"]),
        "admission_evidence": dict(evidence),
        "num_blocks": int(observed["num_blocks"]),
        "group_page_size_bytes": [int(page) for page in observed["group_page_size_bytes"]],
        "resolved_limits": {"max_num_batched_tokens": int(limits["max_num_batched_tokens"]),
                            "max_num_seqs": int(limits["max_num_seqs"]),
                            "max_model_len": int(limits["max_model_len"]),
                            "tensor_parallel_size": int(limits["tensor_parallel_size"])},
        "scope": scope,
        "storage": {"storages": rows,
                    "unique_physical_storage_bytes": int(storage["unique_physical_storage_bytes"]),
                    "scope": storage.get("scope")},
        # Evidence the consumer keeps but does not recompute a term from.
        "tensors": observed.get("tensors"),
        "received": observed.get("received"),
        "runner_resolved": observed.get("runner_resolved"),
        "kernel_block_sizes": observed.get("kernel_block_sizes"),
        "capacity_policy": observed.get("capacity_policy"),
        "group_details": observed.get("group_details"),
        "capacity_assertions": observed.get("capacity_assertions"),
        "received_argument_scope": observed.get("received_argument_scope"),
    }


def check_kv_capacity(observed, expected):
    """Check the external first-model capacity contract without admitting prices."""
    if expected.get("schema") != "tessera.first_model_kv_capacity_expectations.v1":
        raise ValueError("unsupported KV capacity expectation schema")
    checks = []

    def check(name, actual, wanted):
        checks.append({"field": name, "actual": actual, "expected": wanted, "passed": actual == wanted})

    groups = observed["group_details"]
    kinds = [group["kind"] for group in groups]
    check("group_kinds", kinds, expected["group_kinds"])
    check("group_layer_counts", [len(group["layer_names"]) for group in groups], expected["group_layer_counts"])
    check("resident_pages_per_request_by_group", [group["resident_pages_per_request"] for group in groups], expected["resident_pages_per_request_by_group"])
    check("num_blocks", observed["num_blocks"], expected["num_blocks"])
    size = observed["storage"]["unique_physical_storage_bytes"]
    check("unique_physical_storage_bytes", size, expected["unique_physical_storage_bytes"])
    check("physical_pool_block_bytes", size // observed["num_blocks"] if observed["num_blocks"] and size % observed["num_blocks"] == 0 else None, expected["physical_pool_block_bytes"])
    for name, actual in observed["resolved_limits"].items():
        check(name, actual, expected[name])
    check("kv_cache_memory_bytes", observed["capacity_policy"]["values"].get("kv_cache_memory_bytes"), expected["unique_physical_storage_bytes"])
    for index, group in enumerate(groups):
        check(f"group[{index}].page_size_bytes", group["page_size_bytes"], expected["page_size_bytes"])
        if group["kind"] == "recurrent":
            for actual_name, expected_name in (("mamba_cache_mode", "recurrent_cache_mode"),
                    ("num_speculative_blocks", "num_speculative_blocks"),
                    ("num_prefill_checkpoint_blocks", "num_prefill_checkpoint_blocks")):
                check(f"group[{index}].{actual_name}", group[actual_name], expected[expected_name])
        elif group["kind"] == "full_attention":
            for actual_name, expected_name in (("block_size", "attention_block_size"),
                    ("num_kv_heads", "attention_num_kv_heads"), ("head_size", "attention_head_size"),
                    ("dtype", "attention_dtype")):
                check(f"group[{index}].{actual_name}", group[actual_name], expected[expected_name])
    # This is a mathematical capacity check, not a served concurrency result.
    usable = observed["num_blocks"] - expected["null_blocks"]
    needed = observed["resolved_limits"]["max_num_seqs"] * sum(group["resident_pages_per_request"] for group in groups)
    check("capacity_after_reserved_null_blocks", usable >= needed, True)
    return {"schema": "tessera.observed_kv_capacity_checks.v1", "checks": checks,
            "expectations_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "passed": all(row["passed"] for row in checks), "served_concurrency_verified": False,
            "full_model_fixed_resources_complete": False}
