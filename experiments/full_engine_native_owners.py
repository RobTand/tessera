"""Finite allocation-site ownership observations, without a native getter claim."""
import hashlib
from pathlib import Path


def capture_rule_evidence(rule, mapped_libraries):
    sources = {}
    for name in rule["required_source_files"]:
        path = Path(name)
        sources[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    return {"rule": rule, "sources": sources, "mapped_libraries": mapped_libraries}


def validate_rule_evidence(evidence):
    rule = evidence["rule"]
    if (rule["schema"] != "tessera.native_allocation_site_owner_rule.v1"
            or rule["category"] != "shared" or not rule["required_allocation_frames_in_order"]
            or not rule["required_source_files"]):
        raise ValueError("unsupported native allocation-site owner rule")
    for path, expected in rule["required_source_files"].items():
        if evidence["sources"].get(path, {}).get("sha256") != expected["sha256"]:
            raise ValueError("native owner source identity differs: " + path)
    library = rule["required_mapped_library"]
    actual = evidence["mapped_libraries"].get(library["path"])
    if actual != {"sha256": library["sha256"], "bytes": library["bytes"]}:
        raise ValueError("native owner mapped library identity differs")
    return rule


def checkpoint_site_owners(checkpoint, live, history, evidence, device):
    rule = validate_rule_evidence(evidence)
    owners = []
    active = {block["address"]: block["requested_size"] for segment in checkpoint["segments"]
              if segment["device"] == device for block in segment["blocks"]
              if block["state"] == "active_allocated"}
    for address, allocation in live.items():
        if allocation["free_requested_index"] is not None or active.get(address) != allocation["bytes"]:
            continue
        frames = iter(history[allocation["allocate_index"]].get("frames", []))
        if not all(any(all(frame.get(key) == value for key, value in expected.items())
                       for frame in frames) for expected in rule["required_allocation_frames_in_order"]):
            continue
        owners.append({"owner_id": "allocation-site:" + rule["rule_id"] + ":" + allocation["allocation_id"],
            "category": "shared", "address": address, "bytes": allocation["bytes"],
            "device_type": "cuda", "device_id": device,
            "storage_offset_bytes": 0, "view_extent_bytes": allocation["bytes"],
            "observation_kind": "allocation_site_source_library_stack_lifetime",
            "provenance": rule["evidence_kind"] + "; " + rule["assignment_dependence"],
            "native_binding": {"rule_id": rule["rule_id"], "allocation_id": allocation["allocation_id"],
                               "native_getter_observed": False}})
    return owners
