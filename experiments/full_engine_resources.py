"""Raw full-engine allocation ledger; never a complete fixed-resource receipt.

This extends the native CUPTI collector with Torch suballocation history and
explicit storage owners. Startup integration, stream/timing attribution, cache
capacity policy and external/context closure are separate qualification work.
CPU parsing of a synthetic capture cannot establish any of those properties.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass
import hashlib
import ctypes
import json
import os
from pathlib import Path
import re
import threading
from time import perf_counter_ns

from experiments.native_operator_resources import (
    MEMORY_API_OPERATIONS, MEMORY_OWNERSHIP_API, NativeMemoryCollector, validate_cupti_capture,
)
from experiments.full_engine_native_owners import checkpoint_site_owners
from experiments.full_engine_cuda_domains import analyze_memory_api_arguments, match_host_owner
from experiments.full_engine_snapshot_codec import (
    CanonicalHistoryPrefix, SnapshotFramePool, canonical_snapshot_digest,
    compact_capture, expand_capture,
)

CAPTURE_SCHEMA = "tessera.full_engine_resource_capture.v1"
LEDGER_SCHEMA = "tessera.full_engine_raw_resource_ledger.v1"
IDENTITY_SCHEMA = "tessera.full_engine_resource_identity.v1"
IDENTITY_HASHES = ("model_sha256", "configuration_sha256", "runtime_manifest_sha256",
                   "assignment_sha256", "canonical_units_sha256", "workload_sha256")
OWNER_CATEGORIES = {"fixed", "candidate", "kv", "shared", "unknown"}

# The six qualification gaps, in the order ``full_engine_resource_partition``
# states them as domains. The legacy prose spellings are kept as the ledger's
# own ``qualification_gaps`` so an existing reader is unchanged; the derivation
# consumes the domain names.
QUALIFICATION_GAPS = ("worker startup integration", "qualified Torch/CUPTI history join",
                      "external/context/host closure", "runtime provenance admission",
                      "cache capacity policy", "full-engine timing partition")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def process_memory_observation():
    result = {"scope": "Linux worker process RSS/high-water observation, not aggregate admitted memory",
              "rss_bytes": None, "high_water_bytes": None, "errors": []}
    try:
        fields = {line.split(":", 1)[0]: line.split(":", 1)[1].split()
                  for line in Path("/proc/self/status").read_text().splitlines() if ":" in line}
        for source, target in (("VmRSS", "rss_bytes"), ("VmHWM", "high_water_bytes")):
            value, unit = fields[source]
            if unit != "kB":
                raise ValueError("unexpected process memory unit")
            result[target] = int(value) * 1024
    except (OSError, KeyError, ValueError) as exc:
        result["errors"].append(str(exc))
    return result


def _int(value, where, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where}: expected integer >= {minimum}")
    return value


def _text(value, where):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{where}: expected nonempty NUL-free text")
    return value


def _identity(value):
    if not isinstance(value, dict) or value.get("schema") != IDENTITY_SCHEMA:
        raise ValueError("identity: unsupported full-engine identity schema")
    for name in IDENTITY_HASHES:
        if not isinstance(value.get(name), str) or not re.fullmatch("[0-9a-f]{64}", value[name]):
            raise ValueError(f"identity.{name}: expected SHA-256")
    _int(value.get("device_id"), "identity.device_id")
    _text(value.get("device_uuid"), "identity.device_uuid")
    return json.loads(_json_bytes(value))


def _torch():
    import torch
    return torch


def history_revision(previous, current):
    """Retain exact observed prefix changes; this does not qualify a join.

    Real Torch snapshots can revise history annotations and append their own
    marker after returning the snapshot. Keep the old and new field values so
    a refusal is diagnosable from the raw artifact, without excluding fields
    from its existing prefix hash or guessing why they changed.
    """
    changes = []
    for index, (before, after) in enumerate(zip(previous, current)):
        if before == after:
            continue
        fields = {}
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after or before[key] != after[key]:
                fields[key] = {"before_present": key in before, "before": before.get(key),
                               "after_present": key in after, "after": after.get(key)}
        if fields:
            changes.append({"index": index, "fields": fields})
    return {"previous_length": len(previous), "current_length": len(current),
            "missing_previous_rows": previous[len(current):], "changes": changes,
            "scope": "observed raw history revisions; prefix admission remains strict"}


def _verify_history_prefixes(history, checkpoints):
    """Verify each exact recorded prefix, rewinding observed timestamp revisions.

    Only ``time_us`` annotation revisions are supported. Allocation, pool,
    stream and stack identities cannot change. Revisions never make Torch's
    timestamps a qualified clock or price a unit's runtime.
    """
    if not isinstance(checkpoints, list) or not all(isinstance(row, dict) for row in checkpoints):
        raise ValueError("invalid allocator checkpoint records")
    if not all(isinstance(row, dict) for row in history):
        raise ValueError("invalid allocator history records")
    modes = {checkpoint.get("history_boundary", "included_snapshot_marker") for checkpoint in checkpoints}
    prefixes = CanonicalHistoryPrefix()
    if modes == {"included_snapshot_marker"}:
        for checkpoint in checkpoints:
            index = _int(checkpoint["trace_index"], "trace_index", 1)
            if (index > len(history) or history[index - 1]["action"] != "snapshot"
                    or checkpoint["history_prefix_sha256"] != prefixes.observe(history[:index])["history_prefix_sha256"]):
                raise ValueError("checkpoint history prefix is missing, changed or truncated")
        return {"snapshot_marker_placement": "included", "revised_time_fields": 0,
                "timing_eligible": False}
    if modes != {"before_current_snapshot_marker"}:
        raise ValueError("unknown or mixed allocator snapshot boundary convention")
    working = list(history)
    revised = 0
    for number in range(len(checkpoints) - 1, -1, -1):
        checkpoint = checkpoints[number]
        index = _int(checkpoint["trace_index"], "trace_index", 1)
        if index != len(working) or checkpoint["history_prefix_sha256"] != prefixes.observe(working)["history_prefix_sha256"]:
            raise ValueError("exact checkpoint history prefix cannot be reconstructed")
        if number == len(checkpoints) - 1:
            if checkpoint["label"] != "capture_end" or index != len(history):
                raise ValueError("terminal returned snapshot boundary is missing")
        elif history[index]["action"] != "snapshot":
            raise ValueError("returned snapshot has no following history marker")
        if not number:
            if "previous_history_revision" in checkpoint:
                raise ValueError("initial checkpoint cannot revise an absent predecessor")
            break
        previous_index = _int(checkpoints[number - 1]["trace_index"], "previous trace_index", 1)
        revision = checkpoint["previous_history_revision"]
        if (not previous_index < index or _int(revision["previous_length"], "previous revision length") != previous_index
                or _int(revision["current_length"], "current revision length") != index
                or revision["missing_previous_rows"]):
            raise ValueError("history revision lengths/order are incomplete")
        seen = set()
        for change in revision["changes"]:
            row_index = _int(change["index"], "revision index")
            if row_index >= previous_index or row_index in seen or set(change["fields"]) != {"time_us"}:
                raise ValueError("duplicate, out-of-range or unsupported history field revision")
            seen.add(row_index)
            value = change["fields"]["time_us"]
            before = _int(value["before"], "previous time_us")
            after = _int(value["after"], "current time_us")
            if (value["before_present"] is not True or value["after_present"] is not True
                    or before == after or working[row_index].get("time_us") != after):
                raise ValueError("history timestamp revision disagrees with observed bytes")
            working[row_index] = {**working[row_index], "time_us": before}
            revised += 1
        working = working[:previous_index]
    return {"snapshot_marker_placement": "appended_after_snapshot_return",
            "revised_time_fields": revised, "timing_eligible": False}


@dataclass(frozen=True)
class TensorOwner:
    """Caller-named ownership observation, not an independently admitted claim."""

    owner_id: str
    category: str
    tensor: object
    provenance: str


@dataclass(frozen=True)
class NativeStorageOwner:
    """Actual native storage-map entry, without manufacturing a Tensor view."""
    owner_id: str
    category: str
    device_type: str
    device_id: int
    address: int
    bytes: int
    provenance: str
    native_binding: dict


class BlasWorkspaceObserver:
    """Read Torch's existing mutex-protected maps through the pinned extension."""
    def __init__(self, library, expected_sha256):
        path = Path(library).resolve(strict=True)
        self.library_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.library_sha256 != expected_sha256:
            raise ValueError("BLAS workspace observer library identity changed")
        if not _torch().cuda.is_initialized():
            raise RuntimeError("BLAS workspace observer must load after Torch CUDA initialization")
        self._lib = ctypes.CDLL(str(path))
        self._lib.tessera_blas_workspace_snapshot.restype = ctypes.c_char_p
        self._lib.tessera_blas_workspace_snapshot.argtypes = []

    def owners(self):
        raw = json.loads(self._lib.tessera_blas_workspace_snapshot())
        if raw.get("schema") != "tessera.torch_blas_workspace_observation.v1":
            raise ValueError("unsupported Torch BLAS workspace observation")
        for row in raw["workspaces"]:
            yield NativeStorageOwner(
                f"{row['owner']}:handle={row['handle']}:stream={row['stream']}", "shared",
                row["device_type"], row["device_id"], row["address"], row["bytes"],
                "existing Torch BLAS handle/stream workspace map; selected-assignment dependence unresolved",
                {"handle": row["handle"], "stream": row["stream"], "observer_library_sha256": self.library_sha256})


def _owner_row(owner):
    _text(owner.owner_id, "owner_id")
    _text(owner.provenance, "owner provenance")
    if owner.category not in OWNER_CATEGORIES:
        raise ValueError("unsupported owner category")
    if isinstance(owner, NativeStorageOwner):
        size = _int(owner.bytes, "native storage bytes")
        return {"owner_id": owner.owner_id, "category": owner.category,
                "provenance": owner.provenance, "device_type": owner.device_type,
                "device_id": owner.device_id, "address": owner.address, "bytes": size,
                "storage_offset_bytes": 0, "view_extent_bytes": size,
                "observation_kind": "native_storage_map", "native_binding": owner.native_binding}
    tensor = owner.tensor
    storage = tensor.untyped_storage()
    itemsize = tensor.element_size()
    shape, stride = list(tensor.shape), list(tensor.stride())
    if any(s < 0 for s in stride):
        raise ValueError("negative-stride ownership is unsupported")
    extent = 0 if 0 in shape else (1 + sum((n - 1) * s for n, s in zip(shape, stride))) * itemsize
    return {"owner_id": owner.owner_id, "category": owner.category,
            "provenance": owner.provenance, "device_type": tensor.device.type,
            "device_id": tensor.device.index, "address": storage.data_ptr(),
            "bytes": storage.nbytes(), "storage_offset_bytes": tensor.storage_offset() * itemsize,
            "view_extent_bytes": extent, "element_size": itemsize,
            "shape": shape, "stride": stride, "dtype": str(tensor.dtype)}


class FullEngineResourceRecorder:
    """Standalone worker-side surface; install before CUDA initialization.

    Checkpoints synchronize the selected device and copy allocator snapshots;
    this is an intrusive resource pass, never a timing pass. ``max_checkpoints``
    must cover the caller's planned boundaries plus the terminal checkpoint.
    Per-snapshot host time, encoded/reused row counts, shared frame storage and
    process memory observations record observer cost. Logical history size is
    not an estimate of Python's resident memory. No vLLM worker
    hooks or configuration changes are installed here. ``finish`` preserves
    capture failures and raw inputs, with full fixed-resource admission false.
    """

    def __init__(self, library, identity, *, max_checkpoints, max_history_entries=1_000_000):
        self.identity = _identity(identity)
        self.max_checkpoints = _int(max_checkpoints, "max_checkpoints", 1)
        self.max_history_entries = _int(max_history_entries, "max_history_entries", 1)
        self.process_id = os.getpid()
        self.device = self.identity["device_id"]
        self._thread = threading.get_ident()
        self._closed = False
        self._errors, self._checkpoints, self._intervals, self._stack = [], [], [], []
        self._last_snapshot = None
        self._frame_pool = SnapshotFramePool()
        self._history_prefix = CanonicalHistoryPrefix(self._frame_pool)
        self._observer_cost = []
        self._context_id = None
        self._collector = NativeMemoryCollector(library)
        try:
            self._torch = _torch()
            self._early = not self._torch.cuda.is_initialized()
            self._backend = self._torch.cuda.get_allocator_backend()
            self._torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="all", max_entries=max_history_entries,
                clear_history=True, skip_actions=[])
        except Exception as exc:
            self._errors.append(f"history initialization failed: {type(exc).__name__}: {exc}")
            self._early, self._backend = False, "unknown"

    def _open(self):
        if self._closed:
            raise RuntimeError("resource recorder is closed")
        if os.getpid() != self.process_id or threading.get_ident() != self._thread:
            raise RuntimeError("resource recorder crossed its process/thread boundary")

    def snapshot(self, label, *, owners=()):
        if label == "capture_end":
            raise ValueError("capture_end is reserved for finish")
        return self._snapshot(label, owners=owners)

    def _snapshot(self, label, *, owners=(), closing=False, terminal=False):
        self._open()
        _text(label, "checkpoint label")
        if any(c["label"] == label for c in self._checkpoints):
            raise ValueError("duplicate checkpoint label")
        reserved = len(self._stack) - int(closing) + int(not terminal)
        if len(self._checkpoints) + 1 + reserved > self.max_checkpoints:
            self._errors.append("planned checkpoint budget exhausted before snapshot: " + label)
            raise RuntimeError("planned checkpoint budget exhausted")
        started = perf_counter_ns()
        cost = {"label": label, "serialized_snapshot_bytes": None,
                "serialized_history_prefix_bytes": None}
        try:
            self._torch.cuda.synchronize(self.device)
            context_id = self._collector.current_context_id()
            if self._context_id is not None and self._context_id != context_id:
                raise ValueError("CUDA context changed during engine capture")
            self._context_id = context_id
            rows = [_owner_row(owner) for owner in owners]
            # Torch materializes a fresh Python snapshot. Retain that owned
            # value and share exact frame arrays across snapshots. Reuse row
            # chunks while hashing the unchanged expanded canonical bytes.
            raw = self._torch.cuda.memory._snapshot()
            intern_started = perf_counter_ns()
            self._frame_pool.intern_value(raw)
            cost["frame_interning_elapsed_ns"] = perf_counter_ns() - intern_started
            trace = raw["device_traces"][self.device]
            hash_started = perf_counter_ns()
            history_encoding = self._history_prefix.observe(trace, already_interned=True)
            cost["history_encoding_elapsed_ns"] = perf_counter_ns() - hash_started
            cost["serialized_history_prefix_bytes"] = history_encoding["serialized_history_prefix_bytes"]
            cost["history_row_encoding"] = history_encoding
            timestamp = self._collector.mark(label)
            checkpoint = {"label": label, "trace_index": len(trace),
                          "history_boundary": "before_current_snapshot_marker",
                          "cupti_timestamp_ns": timestamp,
                          "history_prefix_sha256": history_encoding["history_prefix_sha256"],
                          "segments": raw["segments"], "owners": rows}
            if self._last_snapshot is not None:
                checkpoint["previous_history_revision"] = history_revision(
                    self._last_snapshot["device_traces"][self.device], trace)
            self._checkpoints.append(checkpoint)
            self._last_snapshot = raw
            return checkpoint
        except Exception as exc:
            self._errors.append(f"snapshot {label}: {type(exc).__name__}: {exc}")
            raise
        finally:
            cost["process_memory"] = process_memory_observation()
            cost["host_observer_elapsed_ns"] = perf_counter_ns() - started
            self._observer_cost.append(cost)

    @contextmanager
    def unit_scope(self, unit_id, *, owners=None):
        self._open()
        _text(unit_id, "unit_id")
        if len(self._checkpoints) + len(self._stack) + 3 > self.max_checkpoints:
            self._errors.append("planned checkpoint budget cannot cover unit: " + unit_id)
            raise RuntimeError("planned checkpoint budget cannot cover unit boundaries")
        invocation_id = f"unit:{len(self._intervals)}"
        begin, end = invocation_id + ":begin", invocation_id + ":end"
        self.snapshot(begin, owners=owners() if owners is not None else ())
        interval = {"invocation_id": invocation_id, "unit_id": unit_id,
                    "begin_checkpoint": begin, "end_checkpoint": None}
        self._intervals.append(interval)
        self._stack.append(invocation_id)
        try:
            yield
        except Exception as exc:
            self._errors.append(f"unit {unit_id}: {type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                self._snapshot(end, closing=True, owners=owners() if owners is not None else ())
                interval["end_checkpoint"] = end
            finally:
                self._stack.pop()

    def finish(self, directory, *, owners=(), native_ownership_evidence=(), measured_runtime_sha256=None):
        self._open()
        if self._stack:
            raise RuntimeError("cannot finish inside an open unit scope")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            self._snapshot("capture_end", owners=owners, terminal=True)
        except Exception:
            pass  # The failed checkpoint is retained in capture.errors.
        try:
            cupti = self._collector.finish(directory / "cupti-memory.json")
        except Exception as exc:
            self._errors.append(f"CUPTI finish failed: {type(exc).__name__}: {exc}")
            cupti = None
        try:
            self._torch.cuda.memory._record_memory_history(enabled=None)
        except Exception as exc:
            self._errors.append(f"history stop failed: {type(exc).__name__}: {exc}")
        self._closed = True
        raw = {"schema": CAPTURE_SCHEMA, "identity": self.identity,
               "process_id": self.process_id, "context_id": self._context_id,
               "capture": {"history_started_before_cuda_initialization": self._early,
                           "allocator_backend": self._backend,
                           "max_history_entries": self.max_history_entries,
                           "max_checkpoints": self.max_checkpoints, "errors": self._errors},
               "observer_cost": {"scope": "synchronized_resource_observer_host_wall_time",
                                 "gpu_timing_eligible": False, "snapshot_attempts": self._observer_cost},
               "torch_snapshot": self._last_snapshot, "checkpoints": self._checkpoints,
               "unit_intervals": self._intervals, "cupti_trace": cupti,
               "native_ownership_evidence": list(native_ownership_evidence),
               "measured_runtime_sha256": measured_runtime_sha256}
        finalization_started = perf_counter_ns()
        memory_before = process_memory_observation()
        compact = compact_capture(raw, self._frame_pool)
        encoded = _json_bytes(compact)
        (directory / "capture.json").write_bytes(encoded + b"\n")
        serialized_bytes = len(encoded) + 1
        del encoded
        write_elapsed = perf_counter_ns() - finalization_started
        analysis_started = perf_counter_ns()
        receipt = analyze_engine_resource_ledger(compact)
        (directory / "finalization.json").write_bytes(_json_bytes({
            "scope": "resource observer finalization; not engine latency or Python resident-memory estimate",
            "capture_encoding": compact["schema"], "serialized_capture_bytes": serialized_bytes,
            "unique_frame_values": len(self._frame_pool.entries),
            "unique_frame_encoded_bytes": self._frame_pool.encoded_bytes,
            "history_cache_memory": self._history_prefix.retained_memory(),
            "process_memory_before": memory_before,
            "process_memory_after": process_memory_observation(),
            "compact_encode_and_write_elapsed_ns": write_elapsed,
            "ledger_analysis_elapsed_ns": perf_counter_ns() - analysis_started}) + b"\n")
        artifacts = {}
        for name in ("capture.json", "cupti-memory.json", "finalization.json"):
            path = directory / name
            if path.exists():
                content = path.read_bytes()
                artifacts[name] = {"path": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        receipt["artifacts"] = artifacts
        (directory / "receipt.json").write_bytes(_json_bytes(receipt) + b"\n")
        return receipt


def _checkpoint_blocks(checkpoint, live, segments, device):
    """Reconcile the positive snapshot state with replayed allocator events."""
    observed_segments, observed_active = {}, {}
    for segment in checkpoint["segments"]:
        if segment["device"] != device:
            if segment["total_size"]:
                raise ValueError("snapshot contains another device's allocator segment")
            continue
        address, size = _int(segment["address"], "segment address", 1), _int(segment["total_size"], "segment size", 1)
        if segment.get("segment_pool_id") not in ([0, 0], (0, 0)):
            raise ValueError("nondefault allocator pool is outside raw ledger scope")
        if address in observed_segments:
            raise ValueError("duplicate allocator snapshot segment")
        observed_segments[address] = size
        cursor = address
        for block in segment["blocks"]:
            if block["address"] != cursor:
                raise ValueError("allocator snapshot blocks have a gap or overlap")
            block_size = _int(block["size"], "block size", 1)
            cursor += block_size
            if block["state"] in ("active_allocated", "active_awaiting_free"):
                requested = _int(block["requested_size"], "requested block size", 1)
                if requested > block_size:
                    raise ValueError("requested allocation exceeds rounded allocator block")
                observed_active[block["address"]] = (requested, block["state"])
                if block["address"] in live:
                    live[block["address"]]["allocator_block_bytes_observed"].add(block_size)
            elif block["state"] != "inactive":
                raise ValueError("unknown allocator block state")
        if cursor != address + size:
            raise ValueError("allocator blocks do not cover the segment")
    expected_active = {address: (row["bytes"], "active_awaiting_free" if row["free_requested_index"] is not None else "active_allocated")
                       for address, row in live.items()}
    if observed_segments != segments or observed_active != expected_active:
        raise ValueError("allocator snapshot disagrees with replayed lifetimes")


def _checkpoint_owners(checkpoint, live, device, issues, domains=None):
    storages, host_storages, owner_ids, unmatched = {}, {}, set(), []
    for owner in checkpoint["owners"]:
        owner_id = _text(owner["owner_id"], "owner_id")
        _text(owner["provenance"], "owner provenance")
        if owner_id in owner_ids:
            raise ValueError("duplicate owner ID in a checkpoint")
        owner_ids.add(owner_id)
        category = owner["category"]
        if category not in OWNER_CATEGORIES or category == "unknown":
            issues.append(f"owner {owner_id}: unknown category")
        size = _int(owner["bytes"], "storage bytes")
        if size == 0:
            continue
        address = _int(owner["address"], "storage address", 1)
        offset, extent = _int(owner["storage_offset_bytes"], "view offset"), _int(owner["view_extent_bytes"], "view extent")
        if offset + extent > size:
            raise ValueError("owner view exceeds its backing storage")
        torch_match = (owner["device_type"] == "cuda" and owner["device_id"] == device
                       and address in live and size <= live[address]["bytes"])
        host = match_host_owner(owner, checkpoint["cupti_timestamp_ns"], domains, device) if domains else None
        if torch_match and host is not None:
            raise ValueError("owner ambiguously matches Torch device and pinned-host storage")
        if host is not None:
            ident = host["allocation_id"]
            entry = host_storages.setdefault(ident, {"allocation_id": ident, "address": host["address"],
                "bytes": host["bytes"], "category": category, "owners": [], "views": []})
            if entry["category"] != category:
                issues.append("aliased pinned-host storage has conflicting categories: " + ident)
                entry["category"] = "unknown"
            entry["owners"].append(owner_id)
            entry["views"].append(owner)
            continue
        if not torch_match:
            issues.append(f"owner {owner_id}: no matching live Torch or pinned-host backing allocation")
            unmatched.append(owner)
            continue
        allocation = live[address]
        key = allocation["allocation_id"]
        entry = storages.setdefault(key, {"allocation_id": key, "address": address,
                                         "bytes": size, "category": category, "owners": [],
                                         "owner_categories": {}})
        if entry["bytes"] != size:
            raise ValueError("aliased storage has conflicting backing sizes")
        if entry["category"] != category:
            issues.append("aliased storage has conflicting category ownership: " + key)
            entry["category"] = "unknown"
        entry["owners"].append(owner_id)
        entry["owner_categories"][owner_id] = category
        allocation["observed_owners"].add(owner_id)
        allocation["observed_categories"].add(category)
    for entry in storages.values():
        entry["owners"].sort()
    return {"label": checkpoint["label"], "trace_index": checkpoint["trace_index"],
            "owner_count": len(owner_ids), "unique_owned_storage_bytes": sum(r["bytes"] for r in storages.values()),
            "unmatched_storage_observations": unmatched,
            "pinned_host_storages": [host_storages[k] for k in sorted(host_storages)],
            "unique_pinned_host_backing_bytes": sum(row["bytes"] for row in host_storages.values()),
            "storages": [storages[k] for k in sorted(storages)]}


def _cupti_coverage(raw, segment_operations, issues):
    trace = raw["cupti_trace"]
    pid = validate_cupti_capture(trace)
    if pid != raw["process_id"]:
        raise ValueError("CUPTI capture belongs to another process")
    argument_domains = analyze_memory_api_arguments(trace)
    issues.extend(argument_domains["issues"])
    handled_arguments = {tuple(key) for key in argument_domains["handled_api_keys"]}
    device, context = raw["identity"]["device_id"], raw["context_id"]
    begin, end = _int(trace["start_ns"], "CUPTI start", 1), _int(trace["end_ns"], "CUPTI end", 1)
    markers = {}
    for marker in trace["markers"]:
        markers.setdefault(marker["name"], []).append(marker["timestamp_ns"])
    for checkpoint in raw["checkpoints"]:
        timestamp = _int(checkpoint["cupti_timestamp_ns"], "checkpoint CUPTI timestamp", 1)
        if markers.get(checkpoint["label"]) != [timestamp] or not begin < timestamp < end:
            raise ValueError("checkpoint is not bound to its CUPTI marker/collection interval")
    apis, required, observed = {}, set(), set()
    if not trace["api_events"]:
        issues.append("no CUDA API activity was observed")
    for api in trace["api_events"]:
        name = re.sub(r"_v\d+$", "", api["name"])
        key = (_int(api["process_id"], "API process", 1), _int(api["correlation_id"], "API correlation"))
        first, last = _int(api["start_ns"], "API start", 1), _int(api["end_ns"], "API end", 1)
        if not begin <= first <= last <= end:
            raise ValueError("CUDA API is outside collection or has reversed timestamps")
        apis.setdefault(key, []).append((name, first, last, api["return_value"]))
        if MEMORY_OWNERSHIP_API.match(name):
            if key in handled_arguments:
                continue
            elif name not in MEMORY_API_OPERATIONS or type(api["return_value"]) is not int or api["return_value"] != 0:
                issues.append("unsupported or failed CUDA allocation API: " + name)
            else:
                if key in required or key[0] != pid:
                    issues.append("ambiguous allocation API process/correlation")
                required.add(key)
        if name.startswith(("cudaGraph", "cuGraph")):
            issues.append("CUDA graph API is outside eager raw ledger scope")
    expected = deque(segment_operations)
    unknown = []
    for row in sorted(trace["memory_events"], key=lambda r: r["timestamp_ns"]):
        if row["process_id"] != pid:
            raise ValueError("CUPTI memory record belongs to another process")
        if row["memory_kind"] in (1, 2):
            continue  # Retained in raw input; no host-memory completeness claim.
        if row["memory_kind"] == 3:
            key = (pid, _int(row["correlation_id"], "memory correlation"))
            matches = apis.get(key, [])
            if len(matches) != 1 or key not in required:
                issues.append("memory operation has no unambiguous successful CUDA API")
            else:
                name, first, last, _ = matches[0]
                if (MEMORY_API_OPERATIONS[name] != row["operation"]
                        or not first <= row["timestamp_ns"] <= last or key in observed):
                    issues.append("memory operation disagrees with its CUDA API/lifetime")
                observed.add(key)
        operation = (row["operation"], row["address"], row["bytes"])
        supported = (row["memory_kind"] == 3 and row["device_id"] == device
                     and row["context_id"] == context and row["async"] is False and row["pool_type"] == 0)
        # This is a coordinate/order consistency join, not an ownership proof
        # across different timestamp domains. Any unmatched record stays raw.
        if supported and expected and operation == expected[0]:
            expected.popleft()
        else:
            unknown.append(row)
    if expected:
        issues.append("Torch segment history has no matching CUPTI memory operations")
    if required != observed:
        issues.append("allocation API lacks its reciprocal memory operation")
    if unknown:
        issues.append("unattributed external/static/unsupported CUDA memory records")
    return unknown, argument_domains


def analyze_engine_resource_ledger(raw):
    """Replay captured allocation events; never construct fixed resource prices.

    ``observed_raw_ledger`` means only this restricted raw ledger reconciled.
    Every result explicitly leaves engine admission, timing and external/context
    closure unavailable. Additional supplied peak/time fields are not prices.
    """
    result = {"schema": LEDGER_SCHEMA, "scope": "full_engine_raw_resource_ledger",
              "status": "incomplete", "admission": "not_implemented",
              "fixed_resources": None, "timings": None,
              "full_model_fixed_resources_complete": False,
              "external_native_peak_bytes": None,
              "issues": [], "torch_allocations": [], "checkpoints": [],
              "escaping_allocation_ids": [], "unattributed_external_records": [],
              "fixture_provenance": None,
              "qualification_gaps": list(QUALIFICATION_GAPS)}
    issues = result["issues"]
    try:
        raw = expand_capture(raw)
        if raw["schema"] != CAPTURE_SCHEMA:
            raise ValueError("unsupported full-engine capture schema")
        # A capture that says on its face that it is synthetic keeps saying so.
        result["fixture_provenance"] = raw.get("fixture_provenance")
        identity = _identity(raw["identity"])
        result["identity"] = identity
        result["capture_sha256"] = canonical_snapshot_digest(raw)
        _int(raw["process_id"], "process_id", 1)
        _int(raw["context_id"], "context_id", 1)
        capture = raw["capture"]
        if capture["errors"]:
            issues.extend(str(error) for error in capture["errors"])
        if capture["history_started_before_cuda_initialization"] is not True:
            raise ValueError("Torch allocator history did not start before CUDA initialization")
        if capture["allocator_backend"] != "native":
            raise ValueError("unsupported Torch allocator backend")
        if len(raw["checkpoints"]) > _int(capture["max_checkpoints"], "max_checkpoints", 1):
            raise ValueError("capture exceeded its planned checkpoint budget")
        device = identity["device_id"]
        traces = raw["torch_snapshot"]["device_traces"]
        history = traces[device]
        capacity = _int(capture["max_history_entries"], "max_history_entries", 1)
        if not history or len(history) >= capacity:
            raise ValueError("Torch history is missing or may have reached its ring capacity")
        if any(trace for index, trace in enumerate(traces) if index != device):
            raise ValueError("Torch history contains another device")
        result["history_join"] = _verify_history_prefixes(history, raw["checkpoints"])
        by_index, by_label = {}, {}
        for checkpoint in raw["checkpoints"]:
            label = _text(checkpoint["label"], "checkpoint label")
            index = _int(checkpoint["trace_index"], "trace_index", 1)
            if label in by_label or index in by_index or index > len(history):
                raise ValueError("checkpoint label/index is duplicate or outside history")
            by_index[index], by_label[label] = checkpoint, index
        if not by_index or max(by_index) != len(history):
            raise ValueError("final allocator checkpoint/history boundary is missing")
        if by_index[len(history)]["segments"] != raw["torch_snapshot"]["segments"]:
            raise ValueError("final raw snapshot disagrees with the final checkpoint")
        intervals, invocation_ids = [], set()
        for interval in raw["unit_intervals"]:
            invocation = _text(interval["invocation_id"], "invocation_id")
            unit = _text(interval["unit_id"], "unit_id")
            if invocation in invocation_ids:
                raise ValueError("duplicate unit invocation")
            invocation_ids.add(invocation)
            begin, end = by_label[interval["begin_checkpoint"]], by_label[interval["end_checkpoint"]]
            if begin >= end:
                raise ValueError("unit interval is reversed or unclosed")
            intervals.append((begin, end, invocation, unit))
        intervals.sort(key=lambda item: (item[0], -item[1]))
        for index, (begin, end, _, _) in enumerate(intervals):
            if any(begin < other_begin < end < other_end for other_begin, other_end, _, _ in intervals[index + 1:]):
                raise ValueError("unit intervals cross instead of nesting")
        domains = analyze_memory_api_arguments(raw["cupti_trace"])
        live, generations, segments = {}, {}, {}
        rows, segment_operations, checkpoint_live = [], [], set()
        peak = 0
        for index, event in enumerate(history):
            action = event["action"]
            if event.get("pool_id") not in ([0, 0], (0, 0)):
                raise ValueError("missing or unsupported allocator trace pool")
            if action == "snapshot":
                pass
            elif action in ("segment_alloc", "segment_free"):
                address, size = _int(event["addr"], "segment address", 1), _int(event["size"], "segment bytes", 1)
                if action == "segment_alloc":
                    if any(address < a + n and a < address + size for a, n in segments.items()):
                        raise ValueError("allocator segments overlap")
                    segments[address] = size
                else:
                    if segments.pop(address, None) != size or any(address <= a < address + size for a in live):
                        raise ValueError("segment free lacks an empty matching segment")
                segment_operations.append(("allocate" if action == "segment_alloc" else "free", address, size))
            elif action == "alloc":
                address, size = _int(event["addr"], "allocation address", 1), _int(event["size"], "allocation bytes", 1)
                if not any(a <= address and address + size <= a + n for a, n in segments.items()):
                    raise ValueError("allocation has no known allocator segment")
                if any(address < a + row["bytes"] and a < address + size for a, row in live.items()):
                    raise ValueError("allocation overlaps a live generation")
                generation = generations.get(address, 0) + 1
                generations[address] = generation
                scopes = [r for r in intervals if r[0] <= index < r[1]]
                row = {"allocation_id": f"{device}:{address}:{generation}", "address": address,
                       "generation": generation, "bytes": size, "allocate_index": index,
                       "free_requested_index": None, "free_completed_index": None,
                       "unit_invocation": scopes[0][2] if scopes else None,
                       "scope_stack": [r[3] for r in scopes],
                       "allocator_block_bytes_observed": set(),
                       "observed_owners": set(), "observed_categories": set()}
                live[address] = row
                rows.append(row)
                peak = max(peak, sum(r["bytes"] for r in live.values()))
            elif action in ("free_requested", "free_completed"):
                address, size = _int(event["addr"], "free address", 1), _int(event["size"], "free bytes", 1)
                if address not in live or live[address]["bytes"] != size:
                    raise ValueError("free lacks matching allocation generation/bytes")
                row = live[address]
                if action == "free_requested":
                    if row["free_requested_index"] is not None:
                        raise ValueError("duplicate free request")
                    row["free_requested_index"] = index
                else:
                    if row["free_requested_index"] is None:
                        raise ValueError("free completed without a free request")
                    row["free_completed_index"] = index
                    del live[address]
            else:
                raise ValueError("unsupported allocator history action: " + str(action))
            if index + 1 in by_index:
                checkpoint = by_index[index + 1]
                _checkpoint_blocks(checkpoint, live, segments, device)
                checkpoint = dict(checkpoint, owners=list(checkpoint["owners"]))
                for evidence in raw.get("native_ownership_evidence", []):
                    checkpoint["owners"].extend(checkpoint_site_owners(checkpoint, live, history, evidence, device))
                result["checkpoints"].append(_checkpoint_owners(checkpoint, live, device, issues, domains))
                checkpoint_live.update(row["allocation_id"] for row in live.values())
        for row in rows:
            interval = next((r for r in intervals if r[2] == row["unit_invocation"]), None)
            end = row["free_completed_index"]
            row["lifetime_scope"] = ("outside_units" if interval is None else
                                     "inside_unit" if end is not None and end < interval[1] else "escapes_unit")
            if row["lifetime_scope"] == "escapes_unit":
                result["escaping_allocation_ids"].append(row["allocation_id"])
            if row["allocation_id"] in checkpoint_live and not row["observed_owners"] and row["lifetime_scope"] != "inside_unit":
                issues.append("unowned allocation live at checkpoint: " + row["allocation_id"])
            if len(row["observed_categories"]) > 1:
                issues.append("allocation ownership category changed: " + row["allocation_id"])
            row["observed_owners"] = sorted(row["observed_owners"])
            row["observed_categories"] = sorted(row["observed_categories"])
            row["allocator_block_bytes_observed"] = sorted(row["allocator_block_bytes_observed"])
        result["torch_allocations"] = rows
        result["torch_observed_live_peak_bytes"] = peak
        result["torch_observed_live_peak_scope"] = "requested_allocation_bytes_excluding_allocator_rounding"
        result["unattributed_external_records"], result["cuda_argument_domains"] = _cupti_coverage(raw, segment_operations, issues)
        result["status"] = "incomplete" if issues else "observed_raw_ledger"
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        issues.append(str(exc))
    return result
