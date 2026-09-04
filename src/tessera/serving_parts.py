"""Whole-layer export ownership and checked assembly of serving checkpoints.

A part has no config.json, so it cannot masquerade as a loadable checkpoint.
The encoded containers are copied unchanged; only their index and summaries
are assembled. This module deliberately needs no tensor runtime.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path

SCHEMA = "tessera.serving-part.v1"
BODY_LAYER = re.compile(r"^model\.(?:[^.]+\.)*layers\.(\d+)\.")


def parse_partition(value: str) -> tuple[int, int]:
    try:
        index, count = map(int, value.split("/"))
        if count < 1 or not 0 <= index < count:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("partition must be INDEX/COUNT with 0 <= INDEX < COUNT") from None
    return index, count


def partition_owner(name: str, count: int) -> int:
    """A fused dense module and every expert of a stack share one owner."""
    match = BODY_LAYER.match(name)
    return int(match.group(1)) % count if match else 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _leaf(name: str) -> str:
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"shard must be a local filename: {name!r}")
    return name


def tensor_names(path: Path) -> set[str]:
    """Read the safetensors header without materializing its tensor payload."""
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(handle.read(length))
    return set(header) - {"__metadata__"}


def source_identity(source: Path) -> dict:
    source = Path(source)
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        tensors = json.loads(index_path.read_text())["weight_map"]
        files = sorted({_leaf(name) for name in tensors.values()})
        actual = {}
        for name in files:
            for tensor in tensor_names(source / name):
                if tensor in actual:
                    raise ValueError(f"source tensor appears twice: {tensor}")
                actual[tensor] = name
        if actual != tensors:
            raise ValueError("source index does not exactly cover the source tensor files")
    else:
        files = sorted(path.name for path in source.glob("*.safetensors"))
        tensors = {}
        for name in files:
            for tensor in tensor_names(source / name):
                if tensor in tensors:
                    raise ValueError(f"source tensor appears twice: {tensor}")
                tensors[tensor] = name
    if not tensors:
        raise ValueError("source has no safetensors tensors")
    auxiliary = sorted({p for pattern in ("*.json", "*.txt", "*.jinja", "*.model")
                        for p in source.glob(pattern)})
    return {"config_sha256": sha256_file(source / "config.json"),
            "auxiliary_sha256": {p.name: sha256_file(p) for p in auxiliary},
            "files": {name: sha256_file(source / name) for name in files},
            "tensors": tensors}


def export_identity(source: Path, options: dict, runtime_image: str, root: Path) -> dict:
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", runtime_image or ""):
        raise ValueError("partition runtime image must be an exact repository@sha256 digest")
    digest = hashlib.sha256()
    paths = sorted([*(p for p in root.joinpath("src").rglob("*")
                       if p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}),
                    *root.joinpath("experiments").glob("*.py"),
                    root / "src/tessera/serving/runtime_contract.json"])
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"source": source_identity(source), "code_sha256": digest.hexdigest(),
            # The dispatch command pins this image. Its observed runtime identity
            # belongs to the PrismaBuild receipt, not a self-attestation here.
            "runtime_image": runtime_image, "options": options}


def summarize_modules(modules: dict, passthrough_bytes: int, checkpoint_bytes: int) -> dict:
    roles = [role for module in modules.values() for role in module["roles"]]
    params = sum(r["rows"] * r["cols"] for r in roles)
    wire = sum(m["wire_bytes"] for m in modules.values())
    containers = sum(m["container_bytes"] for m in modules.values())
    resident = sum(m["resident_bytes_resident_mode"] for m in modules.values())
    families = {}
    for family in sorted({m["family"] for m in modules.values()}):
        members = [m for m in modules.values() if m["family"] == family]
        family_roles = [r for m in members for r in m["roles"]]
        p = sum(r["rows"] * r["cols"] for r in family_roles)
        w = sum(m["wire_bytes"] for m in members)
        r = sum(m["resident_bytes_resident_mode"] for m in members)
        families[family] = {"modules": len(members), "units": len(family_roles),
                            "quantized_params": p, "wire_bytes": w,
                            "wire_bpp": w * 8 / p, "resident_mode_bytes": r,
                            "resident_mode_bpp": r * 8 / p}
    return {"quantized_params": params, "modules": len(modules), "units": len(roles),
            "wire_bytes": wire, "wire_bpp": wire * 8 / params if params else None,
            "on_disk_bytes": containers, "on_disk_bpp": containers * 8 / params if params else None,
            "resident_mode_bytes": resident,
            "resident_mode_bpp": resident * 8 / params if params else None,
            "streamed_mode_note": "the prepared planes (~wire bytes + per-unit tables) plus one transient decoded tile per forward",
            "by_family": families, "passthrough_bytes": passthrough_bytes,
            "checkpoint_bytes": checkpoint_bytes}


def _expected_outputs(owned: set[str], modules: dict) -> set[str]:
    consumed, outputs = set(), set()
    for name, module in modules.items():
        for role in module["roles"]:
            consumed.add(role.get("source_tensor", role["tensor"]))
        if module.get("structure") == "routed_moe":
            outputs.update(r["tensor"].removesuffix(".weight") + ".wire"
                           for r in module["roles"])
        else:
            outputs.add(name + ".wire_bytes")
            if module["family"] == "TESSERA_NVFP4":
                outputs.add(name + ".trellis_input_global_scale")
    if not consumed <= owned:
        raise ValueError(f"module consumes source tensors outside its partition: {sorted(consumed - owned)[:3]}")
    return (owned - consumed) | outputs


def merge_serving_parts(paths, out: Path, source: Path, *, move=False) -> dict:
    """Prove identities, ownership and written tensor coverage before publishing."""
    out, source = Path(out), Path(source)
    if out.exists():
        raise ValueError(f"merge output already exists: {out}")
    loaded = []
    for path in map(Path, paths):
        manifest = json.loads((path / "tessera_serving_manifest.json").read_text())
        part = manifest.get("export_partition", {})
        if part.get("schema") != SCHEMA:
            raise ValueError(f"{path}: unsupported serving partition schema")
        config = json.loads((path / "tessera_part_config.json").read_text())
        index = json.loads((path / "model.safetensors.index.json").read_text())
        loaded.append((part["index"], path, part, manifest, config, index))
    if not loaded:
        raise ValueError("no serving partitions supplied")
    loaded.sort(key=lambda row: row[0])
    count = loaded[0][2]["count"]
    if (type(count) is not int or count < 1 or
            [r[0] for r in loaded] != list(range(count)) or
            any(r[2]["count"] != count for r in loaded)):
        raise ValueError("partition coverage must contain each INDEX in 0..COUNT-1 exactly once")
    identity = loaded[0][2]["identity"]
    if any(row[2]["identity"] != identity for row in loaded[1:]):
        raise ValueError("serving partition identity mismatch (source, plan, encoder or runtime)")
    if source_identity(source) != identity["source"]:
        raise ValueError("source identity changed since partition export")
    expected_source = set(identity["source"]["tensors"])
    source_config = json.loads((source / "config.json").read_text())
    source_config.pop("quantization_config", None)
    base_format = loaded[0][4]["quantization_config"]["format"]
    modules, groups, weight_map, copies = {}, {}, {}, []
    ignore, covered = set(), set()
    passthrough_bytes = 0
    for rank, path, part, manifest, config, index in loaded:
        owned = set(part["source_tensors"])
        expected = {n for n in expected_source if partition_owner(n, count) == rank}
        if owned != expected or len(owned) != len(part["source_tensors"]) or covered & owned:
            raise ValueError(f"partition {rank}: source tensor coverage disagrees with ownership")
        covered.update(owned)
        qconfig = config["quantization_config"]
        if {k: v for k, v in config.items() if k != "quantization_config"} != source_config:
            raise ValueError(f"partition {rank}: model config disagrees with source identity")
        if qconfig["quant_method"] != "tessera" or qconfig["format"] != base_format:
            raise ValueError("partition quantization config disagrees")
        declarations = {target for group in qconfig["config_groups"].values() for target in group["targets"]}
        if declarations != set(manifest["modules"]):
            raise ValueError(f"partition {rank}: config targets disagree with manifest modules")
        if modules.keys() & manifest["modules"].keys() or groups.keys() & qconfig["config_groups"].keys():
            raise ValueError("module or config group is claimed by two partitions")
        modules.update(manifest["modules"])
        groups.update(qconfig["config_groups"])
        ignore.update(qconfig["ignore"])
        local_map = index["weight_map"]
        if set(local_map) != _expected_outputs(owned, manifest["modules"]):
            raise ValueError(f"partition {rank}: written tensor coverage disagrees with source and encoded modules")
        files = set(local_map.values())
        if files != set(part["output_sha256"]):
            raise ValueError(f"partition {rank}: output sha256 coverage disagrees with index")
        actual = {}
        for filename in sorted(files):
            filename = _leaf(filename)
            payload = path / filename
            if sha256_file(payload) != part["output_sha256"][filename]:
                raise ValueError(f"partition {rank}: output sha256 mismatch: {filename}")
            for name in tensor_names(payload):
                if name in actual:
                    raise ValueError(f"tensor appears in two files: {name}")
                actual[name] = filename
            target = f"part-{rank:05d}-{filename}"
            copies.append((payload, target))
        if actual != local_map:
            raise ValueError(f"partition {rank}: index disagrees with actual tensor headers")
        if weight_map.keys() & local_map.keys():
            raise ValueError("tensor appears in two partitions")
        weight_map.update({n: f"part-{rank:05d}-{s}" for n, s in local_map.items()})
        passthrough_bytes += manifest["totals"]["passthrough_bytes"]
    if covered != expected_source:
        raise ValueError("source tensor coverage is incomplete")
    if ignore & modules.keys():
        raise ValueError("a merged target is also declared ignored")
    config = copy.deepcopy(source_config)
    config["quantization_config"] = {"quant_method": "tessera", "format": base_format,
                                      "config_groups": groups, "ignore": sorted(ignore)}
    manifest = copy.deepcopy(loaded[0][3])
    manifest.pop("export_partition")
    manifest["modules"] = modules
    manifest["merged_from"] = [{"index": r[0], "path": str(r[1]),
                                 "output_sha256": r[2]["output_sha256"]} for r in loaded]
    manifest["export_identity"] = identity
    moe = manifest["routed_moe"]
    for field in ("modules", "quantized_stacks"):
        moe[field] = sorted({name for row in loaded for name in row[3]["routed_moe"][field]})
    for field in ("packed_source_tensors", "unpacked_source_tensors", "quantized_source_tensors", "quantized_logical_units"):
        moe[field] = sum(row[3]["routed_moe"][field] for row in loaded)
    moe["disposition"] = ("quantized" if moe["quantized_stacks"] and not moe["modules"]
                          else "mixed" if moe["quantized_stacks"] else "passed_through_bf16")
    size = sum(payload.stat().st_size for payload, _ in copies)
    manifest["totals"] = summarize_modules(modules, passthrough_bytes, size)
    # No output directory or loadable config exists until every proof above passes.
    out.mkdir(parents=True)
    transfer = shutil.move if move else shutil.copy2
    for payload, name in copies:
        transfer(str(payload), str(out / name))
    for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
        for aux in source.glob(pattern):
            if aux.name not in ("config.json", "model.safetensors.index.json", "tessera_serving_manifest.json", "tessera_part_config.json"):
                shutil.copy2(aux, out / aux.name)
    (out / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": size}, "weight_map": weight_map}, indent=2))
    (out / "tessera_serving_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return manifest
