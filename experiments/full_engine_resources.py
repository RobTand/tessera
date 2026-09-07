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
import json
import os
from pathlib import Path
import re
import threading
from time import perf_counter_ns

from experiments.native_operator_resources import (
    MEMORY_API_OPERATIONS, MEMORY_OWNERSHIP_API, NativeMemoryCollector, validate_cupti_capture,
)

CAPTURE_SCHEMA = "tessera.full_engine_resource_capture.v1"
LEDGER_SCHEMA = "tessera.full_engine_raw_resource_ledger.v1"
IDENTITY_SCHEMA = "tessera.full_engine_resource_identity.v1"
IDENTITY_HASHES = ("model_sha256", "configuration_sha256", "runtime_manifest_sha256",
                   "assignment_sha256", "canonical_units_sha256", "workload_sha256")
OWNER_CATEGORIES = {"fixed", "candidate", "kv", "shared", "unknown"}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


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


@dataclass(frozen=True)
class TensorOwner:
    """Caller-named ownership observation, not an independently admitted claim."""

    owner_id: str
    category: str
    tensor: object
    provenance: str


def _owner_row(owner):
    _text(owner.owner_id, "owner_id")
    _text(owner.provenance, "owner provenance")
    if owner.category not in OWNER_CATEGORIES:
        raise ValueError("unsupported owner category")
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
    Per-snapshot host elapsed time and serialized bytes record observer cost;
    serialized size is not an estimate of Python's resident memory. No vLLM worker
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
        cost = {"label": label, "serialized_snapshot_bytes": None}
        try:
            self._torch.cuda.synchronize(self.device)
            context_id = self._collector.current_context_id()
            if self._context_id is not None and self._context_id != context_id:
                raise ValueError("CUDA context changed during engine capture")
            self._context_id = context_id
            rows = [_owner_row(owner) for owner in owners]
            encoded = _json_bytes(self._torch.cuda.memory._snapshot())
            cost["serialized_snapshot_bytes"] = len(encoded)
            raw = json.loads(encoded)
            trace = raw["device_traces"][self.device]
            timestamp = self._collector.mark(label)
            checkpoint = {"label": label, "trace_index": len(trace),
                          "history_boundary": "before_current_snapshot_marker",
                          "cupti_timestamp_ns": timestamp, "history_prefix_sha256": _sha(trace),
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
            cost["host_observer_elapsed_ns"] = perf_counter_ns() - started
            self._observer_cost.append(cost)

    @contextmanager
    def unit_scope(self, unit_id):
        self._open()
        _text(unit_id, "unit_id")
        if len(self._checkpoints) + len(self._stack) + 3 > self.max_checkpoints:
            self._errors.append("planned checkpoint budget cannot cover unit: " + unit_id)
            raise RuntimeError("planned checkpoint budget cannot cover unit boundaries")
        invocation_id = f"unit:{len(self._intervals)}"
        begin, end = invocation_id + ":begin", invocation_id + ":end"
        self.snapshot(begin)
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
                self._snapshot(end, closing=True)
                interval["end_checkpoint"] = end
            finally:
                self._stack.pop()

    def finish(self, directory, *, owners=()):
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
               "unit_intervals": self._intervals, "cupti_trace": cupti}
        (directory / "capture.json").write_bytes(_json_bytes(raw) + b"\n")
        receipt = analyze_engine_resource_ledger(raw)
        artifacts = {}
        for name in ("capture.json", "cupti-memory.json"):
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
                observed_active[block["address"]] = (block_size, block["state"])
            elif block["state"] != "inactive":
                raise ValueError("unknown allocator block state")
        if cursor != address + size:
            raise ValueError("allocator blocks do not cover the segment")
    expected_active = {address: (row["bytes"], "active_awaiting_free" if row["free_requested_index"] is not None else "active_allocated")
                       for address, row in live.items()}
    if observed_segments != segments or observed_active != expected_active:
        raise ValueError("allocator snapshot disagrees with replayed lifetimes")


def _checkpoint_owners(checkpoint, live, device, issues):
    storages, owner_ids = {}, set()
    for owner in checkpoint["owners"]:
        owner_id = _text(owner["owner_id"], "owner_id")
        _text(owner["provenance"], "owner provenance")
        if owner_id in owner_ids:
            raise ValueError("duplicate owner ID in a checkpoint")
        owner_ids.add(owner_id)
        category = owner["category"]
        if category not in OWNER_CATEGORIES or category == "unknown":
            issues.append(f"owner {owner_id}: unknown category")
        if owner["device_type"] != "cuda" or owner["device_id"] != device:
            issues.append(f"owner {owner_id}: host/other-device storage is outside device census")
            continue
        size = _int(owner["bytes"], "storage bytes")
        if size == 0:
            continue
        address = _int(owner["address"], "storage address", 1)
        offset, extent = _int(owner["storage_offset_bytes"], "view offset"), _int(owner["view_extent_bytes"], "view extent")
        if offset + extent > size:
            raise ValueError("owner view exceeds its backing storage")
        if address not in live or size > live[address]["bytes"]:
            raise ValueError("owner has no matching live backing allocation")
        allocation = live[address]
        key = allocation["allocation_id"]
        entry = storages.setdefault(key, {"allocation_id": key, "address": address,
                                         "bytes": size, "category": category, "owners": []})
        if entry["bytes"] != size or entry["category"] != category:
            raise ValueError("aliased storage has conflicting size/category ownership")
        entry["owners"].append(owner_id)
        allocation["observed_owners"].add(owner_id)
        allocation["observed_categories"].add(category)
    for entry in storages.values():
        entry["owners"].sort()
    return {"label": checkpoint["label"], "trace_index": checkpoint["trace_index"],
            "owner_count": len(owner_ids), "unique_owned_storage_bytes": sum(r["bytes"] for r in storages.values()),
            "storages": [storages[k] for k in sorted(storages)]}


def _cupti_coverage(raw, segment_operations, issues):
    trace = raw["cupti_trace"]
    pid = validate_cupti_capture(trace)
    if pid != raw["process_id"]:
        raise ValueError("CUPTI capture belongs to another process")
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
            if name not in MEMORY_API_OPERATIONS or type(api["return_value"]) is not int or api["return_value"] != 0:
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
    return unknown


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
              "qualification_gaps": ["worker startup integration", "qualified Torch/CUPTI history join",
                                     "external/context/host closure", "runtime provenance admission",
                                     "cache capacity policy", "full-engine timing partition"]}
    issues = result["issues"]
    try:
        if raw["schema"] != CAPTURE_SCHEMA:
            raise ValueError("unsupported full-engine capture schema")
        identity = _identity(raw["identity"])
        result["identity"] = identity
        result["capture_sha256"] = _sha(raw)
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
        by_index, by_label = {}, {}
        for checkpoint in raw["checkpoints"]:
            label = _text(checkpoint["label"], "checkpoint label")
            index = _int(checkpoint["trace_index"], "trace_index", 1)
            if label in by_label or index in by_index or index > len(history):
                raise ValueError("checkpoint label/index is duplicate or outside history")
            if history[index - 1]["action"] != "snapshot" or checkpoint["history_prefix_sha256"] != _sha(history[:index]):
                raise ValueError("checkpoint history prefix is missing, changed or truncated")
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
                result["checkpoints"].append(_checkpoint_owners(checkpoint, live, device, issues))
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
        result["torch_allocations"] = rows
        result["torch_observed_live_peak_bytes"] = peak
        result["unattributed_external_records"] = _cupti_coverage(raw, segment_operations, issues)
        result["status"] = "incomplete" if issues else "observed_raw_ledger"
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        issues.append(str(exc))
    return result
