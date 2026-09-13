"""Canonical roster and assignment of a served Tessera artifact, for the full-engine observers.

The source-BF16 launcher derives its roster from a canonical census. A Tessera
artifact carries its own: ``tessera_serving_manifest.json`` names every
quantized module by its vLLM module path and every HF tensor each one fuses,
so the roster, the assignment and the model identity are read from the bytes
the engine will load, and nothing here is declared twice.
"""
import hashlib
import json
from pathlib import Path

ASSIGNMENT_SCHEMA = "tessera.artifact_observer_assignment.v1"
SOURCE_SCHEMA = "tessera.artifact_observer_source.v1"
MANIFEST_NAME = "tessera_serving_manifest.json"
#: Files whose bytes are the artifact's identity: the engine loads exactly these.
IDENTITY_FILES = ("config.json", MANIFEST_NAME)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def weight_files(model):
    index = model / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return ["model.safetensors.index.json", *names]
    if (model / "model.safetensors").exists():
        return ["model.safetensors"]
    raise ValueError("artifact carries neither model.safetensors nor a safetensors index")


def read_tessera_artifact(model):
    """Return ``(source, roster, assignment)`` for one Tessera artifact directory.

    ``source`` digests every file the identity is made of; ``roster`` is one
    row per manifest module, ``g:`` for a fused module with several HF roles
    and ``l:`` for a single-role module, sorted by unit id like the census
    roster; ``assignment`` maps each unit to the family the manifest serves it
    in. Refused when the checkpoint is not a Tessera artifact or when the
    manifest names a module whose roles are empty or duplicated.
    """
    model = Path(model)
    config = json.loads((model / "config.json").read_text())
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "tessera":
        raise ValueError("artifact observation requires a checkpoint whose quantization_config names tessera")
    manifest = json.loads((model / MANIFEST_NAME).read_text())
    modules = manifest["modules"]
    if not isinstance(modules, dict) or not modules:
        raise ValueError("tessera serving manifest declares no modules")
    roster, seen_members = [], set()
    for name in sorted(modules):
        entry = modules[name]
        members = [role["tensor"] for role in entry["roles"]]
        if not members or len(set(members)) != len(members) or seen_members & set(members):
            raise ValueError("manifest module roles are empty or duplicated: " + name)
        seen_members.update(members)
        roster.append({"unit_id": ("g:" if len(members) > 1 else "l:") + name, "module": name,
                       "members": members, "family": entry["family"]})
    roster.sort(key=lambda row: row["unit_id"])
    files = {name: _digest(model / name) for name in (*IDENTITY_FILES, *weight_files(model))}
    # The directory name is not part of the identity: the bytes are.
    source = {"schema": SOURCE_SCHEMA, "files": files,
              "manifest_totals": manifest.get("totals"), "ignore": list(quantization.get("ignore", []))}
    assignment = {"schema": ASSIGNMENT_SCHEMA, "source_sha256": canonical_hash(source),
                  "units": {row["unit_id"]: row["family"] for row in roster}}
    return source, roster, assignment


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()
