"""Canonical roster and assignment of a served Tessera artifact, for the full-engine observers.

The source-BF16 launcher derives its roster from a canonical census. A Tessera
artifact carries its own: ``tessera_serving_manifest.json`` names every
quantized module by its vLLM module path and every HF tensor each one fuses,
so the roster, the assignment and the model identity are read from the bytes
the engine will load, and nothing here is declared twice.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

ASSIGNMENT_SCHEMA = "tessera.artifact_observer_assignment.v1"
SOURCE_SCHEMA = "tessera.artifact_observer_source.v1"
FILE_ATTESTATION_SCHEMA = "tessera.artifact_file_stat_attestation.v1"
MANIFEST_NAME = "tessera_serving_manifest.json"
#: Files whose bytes are the artifact's identity: the engine loads exactly these.
IDENTITY_FILES = ("config.json", MANIFEST_NAME)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _attested_digest(path, *, capture_bytes=False):
    """Hash a regular file through one no-follow FD and retain its stat seal."""
    path = Path(path)
    if path.name != str(path).split("/")[-1] or path.is_symlink():
        raise ValueError("artifact identity file must be a regular non-symlink basename")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("artifact identity file is not regular")
        fields = lambda value: {"size": int(value.st_size),
                                "mtime_ns": int(value.st_mtime_ns),
                                "ctime_ns": int(value.st_ctime_ns),
                                "st_dev": int(value.st_dev), "st_ino": int(value.st_ino)}
        initial = fields(before)
        hashed = hashlib.sha256()
        chunks = [] if capture_bytes else None
        while block := os.read(descriptor, 8 * 1024 * 1024):
            hashed.update(block)
            if chunks is not None:
                chunks.append(block)
        final = fields(os.fstat(descriptor))
        current = fields(path.stat(follow_symlinks=False))
        if initial != final or final != current:
            raise ValueError("artifact identity file changed during its full hash")
        return (hashed.hexdigest(), final, b"".join(chunks)) if capture_bytes else (hashed.hexdigest(), final)
    finally:
        os.close(descriptor)


def weight_files(model):
    index = model / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return ["model.safetensors.index.json", *names]
    if (model / "model.safetensors").exists():
        return ["model.safetensors"]
    raise ValueError("artifact carries neither model.safetensors nor a safetensors index")


def read_tessera_artifact(model, *, with_file_attestation=False):
    """Return ``(source, roster, assignment)`` for one Tessera artifact directory.

    ``source`` digests every file the identity is made of; ``roster`` is one
    row per manifest module, ``g:`` for a fused module with several HF roles
    and ``l:`` for a single-role module, sorted by unit id like the census
    roster; ``assignment`` maps each unit to the family the manifest serves it
    in. Refused when the checkpoint is not a Tessera artifact or when the
    manifest names a module whose roles are empty or duplicated.
    """
    model = Path(model)
    if with_file_attestation:
        # Parse the exact bytes attested by the same no-follow FD/hash pass.
        config_hash = _attested_digest(model / "config.json", capture_bytes=True)
        manifest_hash = _attested_digest(model / MANIFEST_NAME, capture_bytes=True)
        config = json.loads(config_hash[2])
        manifest = json.loads(manifest_hash[2])
        index_path = model / "model.safetensors.index.json"
        if index_path.exists() or index_path.is_symlink():
            index_hash = _attested_digest(index_path, capture_bytes=True)
            weight_names = sorted(set(json.loads(index_hash[2])["weight_map"].values()))
            weight_names = ["model.safetensors.index.json", *weight_names]
            prehashed = {"config.json": config_hash[:2], MANIFEST_NAME: manifest_hash[:2],
                         "model.safetensors.index.json": index_hash[:2]}
        elif (model / "model.safetensors").exists():
            weight_names = ["model.safetensors"]
            prehashed = {"config.json": config_hash[:2], MANIFEST_NAME: manifest_hash[:2]}
        else:
            raise ValueError("artifact carries neither model.safetensors nor a safetensors index")
    else:
        config = json.loads((model / "config.json").read_text())
        manifest = json.loads((model / MANIFEST_NAME).read_text())
        weight_names = weight_files(model)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "tessera":
        raise ValueError("artifact observation requires a checkpoint whose quantization_config names tessera")
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
    names = (*IDENTITY_FILES, *weight_names)
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("artifact identity files must be distinct basenames")
    if with_file_attestation:
        hashed = {name: prehashed.get(name) or _attested_digest(model / name) for name in names}
        # The early small-file parses and later weight hashes form one source
        # attestation only if every named path still denotes the same file.
        for name, (_digest_value, stamp) in hashed.items():
            path = model / name
            if path.is_symlink():
                raise ValueError("artifact identity file changed after its full hash")
            current = path.stat(follow_symlinks=False)
            if {"size": current.st_size, "mtime_ns": current.st_mtime_ns,
                "ctime_ns": current.st_ctime_ns, "st_dev": current.st_dev,
                "st_ino": current.st_ino} != stamp:
                raise ValueError("artifact identity file changed after its full hash")
        files = {name: value[0] for name, value in hashed.items()}
        attestation = {"schema": FILE_ATTESTATION_SCHEMA,
                       "files": {name: value[1] for name, value in hashed.items()}}
    else:
        files = {name: _digest(model / name) for name in names}
    # The directory name is not part of the identity: the bytes are.
    source = {"schema": SOURCE_SCHEMA, "files": files,
              "manifest_totals": manifest.get("totals"), "ignore": list(quantization.get("ignore", []))}
    assignment = {"schema": ASSIGNMENT_SCHEMA, "source_sha256": canonical_hash(source),
                  "units": {row["unit_id"]: row["family"] for row in roster}}
    return (source, roster, assignment, attestation) if with_file_attestation else (source, roster, assignment)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()
