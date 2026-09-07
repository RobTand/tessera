"""Argument-bound CUDA host lifetimes and null-free witnesses.

This is a raw domain ledger, not a process/UMA peak or a fixed-resource price.
Old activity-only captures receive no argument-based exemptions.
"""
import re


ARGUMENT_SCHEMA = "tessera.cuda_memory_api_arguments.v1"
# CUDA runtime callback IDs are part of the v1 collector ABI.
ARGUMENT_CALLBACK_IDS = frozenset({20, 22, 25, 26, 27, 28})
ARGUMENT_CONFIGURATION = {"subscribe_arguments", "unsubscribe_arguments"} | {
    f"enable_argument_callback_{cbid}" for cbid in ARGUMENT_CALLBACK_IDS}
CALLBACK_APIS = {"cudaMalloc", "cudaFree", "cudaHostAlloc", "cudaMallocHost",
                 "cudaFreeHost", "cudaHostGetDevicePointer"}
HOST_APIS = {"cudaHostAlloc": "allocate", "cudaMallocHost": "allocate", "cudaFreeHost": "free"}


def normalized_name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("missing CUDA API name")
    return re.sub(r"_v\d+$", "", value)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def key(row):
    return (integer(row["process_id"], "process", 1), integer(row["correlation_id"], "correlation"))


def analyze_memory_api_arguments(trace):
    result = {"status": "unavailable", "issues": [], "host_allocations": [],
              "host_mappings": [], "null_device_frees": [], "handled_api_keys": [],
              "scope": "observed pinned-host lifetimes and explicit pointer mappings; no whole-host/UMA closure"}
    if "argument_schema" not in trace:
        return result
    if trace["argument_schema"] != ARGUMENT_SCHEMA:
        raise ValueError("unsupported CUDA memory argument schema")
    pid = integer(trace["process_id"], "capture process", 1)
    begin, end = trace["start_ns"], trace["end_ns"]
    config = {row["operation"]: row["code"] for row in trace["configuration"]}
    if config.get("subscribe_arguments") != 0 or config.get("unsubscribe_arguments") != 0:
        raise ValueError("CUDA argument callback lifecycle was not successful")
    enabled = {int(name.removeprefix("enable_argument_callback_")) for name, status in config.items()
               if name.startswith("enable_argument_callback_") and status == 0}
    apis = {}
    for row in trace["api_events"]:
        apis.setdefault(key(row), []).append(row)
    args = {}
    for row in trace["api_argument_events"]:
        ident = key(row)
        if ident[0] != pid or ident in args:
            raise ValueError("duplicate or foreign CUDA argument callback")
        matches = apis.get(ident, [])
        if len(matches) != 1:
            raise ValueError("CUDA argument callback has no unique activity API")
        api = matches[0]
        name = normalized_name(row["name"])
        if (name not in CALLBACK_APIS or name != normalized_name(api["name"])
                or row["site"] != "exit" or row["callback_id"] not in enabled
                or type(row["return_value"]) is not int or type(api["return_value"]) is not int
                or row["return_value"] != api["return_value"]
                or not begin <= integer(row["timestamp_ns"], "callback timestamp", 1) <= end):
            raise ValueError("CUDA argument callback disagrees with its API/configuration")
        for field in ({"cudaMalloc": ("device_address", "bytes"), "cudaFree": ("device_address",),
                "cudaHostAlloc": ("host_address", "bytes", "flags"),
                "cudaMallocHost": ("host_address", "bytes"), "cudaFreeHost": ("host_address",),
                "cudaHostGetDevicePointer": ("host_address", "device_address", "flags")}[name]):
            integer(row[field], "CUDA argument " + field)
        args[ident] = row
    memory = {}
    for row in trace["memory_events"]:
        memory.setdefault(key(row), []).append(row)
    issues, handled = result["issues"], set()
    # Every subscribed API observed by activities needs its unique callback.
    for ident, rows in apis.items():
        if any(normalized_name(row["name"]) in CALLBACK_APIS for row in rows) and ident not in args:
            issues.append(f"CUDA allocation API has no argument witness: {ident}")
    live, generations = {}, {}
    for row in sorted(trace["memory_events"], key=lambda item: item["timestamp_ns"]):
        if row["memory_kind"] not in (1, 2):
            continue
        ident = key(row)
        argument = args.get(ident)
        api_rows = apis.get(ident, [])
        if argument is None or len(api_rows) != 1:
            issues.append(f"host memory has no unique argument/API witness: {ident}")
            continue
        api = api_rows[0]
        name = normalized_name(api["name"])
        address, size = integer(row["address"], "host address", 1), integer(row["bytes"], "host bytes", 1)
        if (ident[0] != pid or row["memory_kind"] != 2 or row["async"] is not False or row["pool_type"] != 0
                or api["return_value"] != 0 or HOST_APIS.get(name) != row["operation"]
                or len(memory[ident]) != 1 or argument.get("host_address") != address
                or not api["start_ns"] <= row["timestamp_ns"] <= api["end_ns"]):
            issues.append(f"host memory disagrees with allocation arguments/API: {ident}")
            continue
        if row["operation"] == "allocate":
            if address in live or argument.get("bytes") != size:
                issues.append(f"host allocation lifetime/size disagreement: {ident}")
                continue
            if any(address < other["address"] + other["bytes"] and other["address"] < address + size for other in live.values()):
                issues.append(f"live pinned-host allocations overlap: {ident}")
                continue
            generations[address] = generations.get(address, 0) + 1
            lifetime = {"allocation_id": f"host:{pid}:{address}:{generations[address]}",
                        "address": address, "bytes": size, "allocated_ns": row["timestamp_ns"],
                        "freed_ns": None, "source": row["source"], "device_id": row["device_id"],
                        "context_id": row["context_id"], "allocation_correlation_id": ident[1]}
            live[address] = lifetime
            result["host_allocations"].append(lifetime)
        elif row["operation"] == "free":
            if address not in live or live[address]["bytes"] != size:
                issues.append(f"host free has no matching allocation lifetime: {ident}")
                continue
            live.pop(address)["freed_ns"] = row["timestamp_ns"]
        handled.add(ident)
    for ident, argument in args.items():
        api = apis[ident][0]
        name = normalized_name(api["name"])
        if api["return_value"] != 0:
            issues.append(f"CUDA memory API failed: {ident}")
            continue
        if name == "cudaFree" and argument.get("device_address") == 0:
            if memory.get(ident):
                issues.append(f"null cudaFree unexpectedly owns memory records: {ident}")
                continue
            result["null_device_frees"].append(argument)
            handled.add(ident)
        elif name in ("cudaMalloc", "cudaFree"):
            matches = memory.get(ident, [])
            if (len(matches) != 1 or matches[0]["memory_kind"] != 3
                    or matches[0]["address"] != argument.get("device_address")
                    or matches[0]["operation"] != ("allocate" if name == "cudaMalloc" else "free")
                    or not api["start_ns"] <= matches[0]["timestamp_ns"] <= api["end_ns"]
                    or (name == "cudaMalloc" and matches[0]["bytes"] != argument.get("bytes"))):
                issues.append(f"device allocation arguments disagree with memory record: {ident}")
        elif name in HOST_APIS and ident not in handled:
            issues.append(f"host API has no matching memory lifetime: {ident}")
        elif name == "cudaHostGetDevicePointer":
            host = integer(argument["host_address"], "mapping host address", 1)
            device = integer(argument["device_address"], "mapping device address", 1)
            matches = [row for row in result["host_allocations"]
                       if row["address"] <= host < row["address"] + row["bytes"]
                       and row["allocated_ns"] <= api["start_ns"]
                       and (row["freed_ns"] is None or row["freed_ns"] > api["end_ns"])]
            if len(matches) != 1 or memory.get(ident):
                issues.append(f"host mapping has no unique live backing: {ident}")
                continue
            allocation = matches[0]
            result["host_mappings"].append({"host_allocation_id": allocation["allocation_id"],
                "host_address": host, "device_address": device,
                "bytes_from_queried_offset": allocation["address"] + allocation["bytes"] - host,
                "mapped_ns": api["end_ns"], "freed_ns": allocation["freed_ns"],
                "device_id": allocation["device_id"], "context_id": allocation["context_id"],
                "correlation_id": ident[1]})
            handled.add(ident)
    result["handled_api_keys"] = [list(ident) for ident in sorted(handled)]
    result["status"] = "incomplete" if issues else "observed_argument_domains"
    return result


def match_host_owner(owner, checkpoint_ns, domains, device):
    """Join an observed storage range to one physical pinned allocation lifetime."""
    address = integer(owner["address"], "owner address", 1)
    size = integer(owner["bytes"], "owner bytes", 1)
    allocations = {row["allocation_id"]: row for row in domains["host_allocations"]
                   if row["allocated_ns"] <= checkpoint_ns
                   and (row["freed_ns"] is None or checkpoint_ns < row["freed_ns"])}
    matches = set()
    if owner["device_type"] == "cpu":
        for ident, row in allocations.items():
            if row["address"] <= address and address + size <= row["address"] + row["bytes"]:
                matches.add(ident)
    elif owner["device_type"] == "cuda" and owner["device_id"] == device:
        for row in domains["host_mappings"]:
            if (row["host_allocation_id"] in allocations and row["device_id"] == device
                    and row["mapped_ns"] <= checkpoint_ns
                    and row["device_address"] <= address
                    and address + size <= row["device_address"] + row["bytes_from_queried_offset"]):
                matches.add(row["host_allocation_id"])
    if len(matches) > 1:
        raise ValueError("storage matches multiple live pinned-host allocations")
    return allocations[next(iter(matches))] if matches else None
