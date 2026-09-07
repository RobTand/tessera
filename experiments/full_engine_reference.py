"""Bind a selected original-wire reference checkpoint to an engine plan.

This verifies bytes and the export statement. It is not exporter/PB provenance
admission, a served quality result, or a planner-selected assignment.
"""
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_reference_checkpoint(proof_path, source, roster, census_sha256):
    path = Path(proof_path).resolve(strict=True)
    proof = json.loads(path.read_text())
    if proof.get("schema") != "tessera.full_model_original_wire_checkpoint.v1" or proof.get("status") != "passed":
        raise ValueError("unsupported or unsuccessful original-wire checkpoint proof")
    root, checkpoint = path.parent, path.parent / "checkpoint"
    records = {}
    for name, field in (("originals.json", "originals_sha256"), ("plan.json", "plan_sha256"),
                        ("source-identity.json", "source_identity_sha256")):
        if digest(root / name) != proof[field]:
            raise ValueError("reference proof input digest disagrees: " + name)
        records[name] = json.loads((root / name).read_text())
    if canonical_hash(records["source-identity.json"]) != canonical_hash(source):
        raise ValueError("reference checkpoint was exported from another source identity")
    expected = {member for row in roster for member in row["members"]}
    originals = records["originals.json"]
    if originals["census_sha256"] != census_sha256:
        raise ValueError("reference checkpoint was exported against another canonical census")
    if not expected or set(proof["wires"]) != expected or set(originals["units"]) != expected:
        raise ValueError("reference checkpoint does not cover the complete canonical member census")
    if originals["anchors_sha256"] != proof["anchors_sha256"]:
        raise ValueError("reference checkpoint names different campaign anchors")
    for unit in sorted(expected):
        wire_sha = proof["wires"][unit]["sha256"]
        if not isinstance(wire_sha, str) or not re.fullmatch("[0-9a-f]{64}", wire_sha):
            raise ValueError("reference wire digest is invalid: " + unit)
        if wire_sha != originals["units"][unit]["original_blob_sha256"]:
            raise ValueError("reference wire differs from the selected campaign original: " + unit)
    if not proof["checkpoint_files"] or set(proof["checkpoint_files"]) != {p.name for p in checkpoint.iterdir()}:
        raise ValueError("reference checkpoint file roster is not closed")
    for name, record in proof["checkpoint_files"].items():
        file = checkpoint / name
        if (Path(name).name != name or file.is_symlink() or not file.is_file()
                or type(record["bytes"]) is not int or file.stat().st_size != record["bytes"]
                or digest(file) != record["sha256"]):
            raise ValueError("reference checkpoint file bytes disagree: " + name)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("quantization_config", {}).get("quant_method") != "tessera":
        raise ValueError("original-wire reference checkpoint does not select Tessera")
    assignment = {"schema": "tessera.full_engine_original_wire_assignment.v1",
                  "source_sha256": canonical_hash(source), "export_plan_sha256": proof["plan_sha256"],
                  "units": {row["unit_id"]: {member: proof["wires"][member]["sha256"] for member in row["members"]}
                            for row in roster}}
    return {"checkpoint": str(checkpoint), "checkpoint_files": proof["checkpoint_files"],
            "checkpoint_sha256": canonical_hash(proof["checkpoint_files"]), "proof_path": str(path),
            "proof_sha256": digest(path), "assignment": assignment,
            "scope": "hash-verified selected export statement; owning PB/export evidence still required for admission"}
