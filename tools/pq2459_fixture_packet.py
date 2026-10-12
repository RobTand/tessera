#!/usr/bin/env python3
"""Read real exported checkpoints and publish the authorized input geometry."""
from __future__ import annotations

import argparse
from itertools import product
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCOPE = ROOT / "experiments/pq2459/scope.json"


def bound_file(path):
    from tessera.serving_parts import sha256_file
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_blob(root, inventory, name):
    from safetensors import safe_open
    with safe_open(root / inventory[name], framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
        if str(tensor.dtype) != "torch.uint8" or tensor.ndim != 1:
            raise ValueError(f"{name}: wire tensor must be a uint8 vector")
        return tensor.numpy().tobytes()


def rank_geometry(target, declared, parsed, *, output_cut):
    from tessera.compact_prep import require_compact_cut
    from tessera.serving.sharding import plan_shard
    rows, columns = declared["rows"], declared["columns"]
    roles = declared["roles"]
    ranks = []
    for rank in range(2):
        plan = plan_shard(
            target, roles=roles, columns=columns,
            out_partitions=[r // 2 if output_cut else r for _, r in roles],
            in_size=columns if output_cut else columns // 2,
            tp_rank=rank, tp_size=2, input_size=columns, output_size=rows)
        cuts = []
        for name, wire in parsed:
            role = plan.role(name)
            cut = (require_compact_cut(wire, rows=(role.lo, role.hi))
                   if output_cut else require_compact_cut(wire, cols=(role.lo, role.hi)))
            cuts.append({"role": name, "rows": list(cut[0]), "columns": list(cut[1])})
        ranks.append({"rank": rank, "nk": [plan.shard_rows, plan.shard_columns],
                      "cuts": cuts})
    return ranks


def wire_record(artifact, target, tensor, blob, rung, ranks, *, projection=None):
    record = {"artifact": artifact, "target": target, "tensor": tensor,
              "q256": rung, "rank_local_nk": ranks[0]["nk"], "ranks": ranks,
              "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    if projection is not None:
        record["projection"] = projection
        record["shape_owner"] = "w13 group; gate and up form the rank-local output"
    return record


def artifact_wires(key, root, scope):
    from tessera.serving_parts import source_inventory, read_serving_manifest
    from tessera.serving.contract import PAYLOAD_FAMILY_BY_ROUTE
    from tessera.serving.scheme import (
        validate_tessera_scheme, parse_compact_blob_for_scheme,
        expert_role_declarations, parse_compact_tessera_expert_blob, MOE_GROUP_PROJECTIONS)
    config = json.loads((root / "config.json").read_bytes())
    inventory = source_inventory(root)
    manifest = read_serving_manifest(root / "tessera_serving_manifest.json")
    wires = []
    excluded = []
    for group in config["quantization_config"]["config_groups"].values():
        if len(group["targets"]) != 1:
            raise ValueError("fixture groups must name one exact target")
        target = group["targets"][0]
        if ".45." in target:
            raise ValueError("draft layer 45 cannot enter an additive fixture")
        declared = validate_tessera_scheme(group["scheme"], target)
        family = PAYLOAD_FAMILY_BY_ROUTE.get(declared["family"])
        structure = declared["structure"]
        if family not in scope["rungs"]:
            excluded.append({"target": target, "reason": "family outside the authorization"})
            continue
        if structure == "dense":
            rungs = set(declared["role_q256"])
            if len(rungs) != 1 or not rungs <= set(scope["rungs"][family][structure]):
                excluded.append({"target": target, "reason": "rung outside the authorization"})
                continue
            name = target + ".wire_bytes"
            blob = read_blob(root, inventory, name)
            parsed = parse_compact_blob_for_scheme(blob, group["scheme"], target, device="cpu")
            roles = tuple(name for name, _rows in declared["roles"])
            if roles not in tuple(MOE_GROUP_PROJECTIONS.values()):
                raise ValueError(f"{target}: fixture roles are not an MLP projection")
            ranks = rank_geometry(target, declared, parsed,
                                  output_cut=roles == MOE_GROUP_PROJECTIONS["w13"])
            record = wire_record(key, target, name, blob, next(iter(rungs)), ranks)
            record.update(family=family, structure=structure,
                          roles=[{"name": n, **w.role_facts} for n, w in parsed])
            wires.append(record)
        else:
            # Bind the whole checkpoint below. Small reads exercise both w13 roles
            # through the actual expert parser, not synthetic trace records.
            group_declared = declared["groups"]["w13"]
            rungs = set(group_declared["role_q256"])
            if len(rungs) != 1 or not rungs <= set(scope["rungs"][family][structure]):
                excluded.append({"target": target, "reason": "rung outside the authorization"})
                continue
            role_blobs = []
            all_parsed = []
            for role in expert_role_declarations(group_declared, expert=0):
                projection = role["roles"][0][0]
                name = f"{target}.0.{projection}.wire"
                blob = read_blob(root, inventory, name)
                parsed = parse_compact_tessera_expert_blob(blob, role, target, device="cpu")
                all_parsed.extend(parsed)
                role_blobs.append((projection, name, blob))
            ranks = rank_geometry(target, group_declared, all_parsed, output_cut=True)
            for projection, name, blob in role_blobs:
                record = wire_record(key, target, name, blob, next(iter(rungs)), ranks,
                                     projection=projection)
                record.update(family=family, structure=structure, experts=declared["experts"])
                wires.append(record)
    return wires, excluded, manifest, inventory


def build_packet(artifacts, *, export_requests, historical_package, historical_commit,
                 source_info, threads):
    from tessera.cached_unit import encoder_source_sha256
    from tessera.serving.source_identity import serving_source_sha256
    from tessera.serving_parts import sha256_files
    scope = json.loads(SCOPE.read_bytes())
    packet = {"schema": "tessera.qualification.fixture_inputs.v1", **scope,
              "scope_sha256": bound_file(SCOPE)["sha256"], "artifacts": {}, "cells": []}
    package_root = Path(source_info["src"])
    packet["serving"] = {
        "commit": source_info["commit"], "source_algorithm": "tessera.package_source.v1",
        "source_sha256": serving_source_sha256(package_root),
        "contract": bound_file(package_root / "tessera/serving/runtime_contract.json")}
    packet["source_input"] = source_info
    packet["exports"] = {}
    for name, request_path in export_requests.items():
        request = json.loads(request_path.read_bytes())
        packet["exports"][name] = {"action": request["action_key"], "request": bound_file(request_path),
                                   "snapshot": request["params"]["checkout_snapshot"]}
    from tools.pq2459_fixture_workloads import read_panel_inputs
    inputs = scope["token_inputs"]
    _panel, _tokens, token_receipts = read_panel_inputs(
        Path(inputs["panel"]), Path(inputs["arrays_root"]))
    packet["token_inputs"] = {**inputs, "receipt": token_receipts}
    packet["entrypoints"] = {
        profile: {"module": "tools.pq2459_fixture_workloads",
                  "profile_argument": profile,
                  "source": bound_file(ROOT / "tools/pq2459_fixture_workloads.py"),
                  "required_arguments": ["packet", "model", "panel", "arrays_root", "output",
                                         "tensor_parallel_size", "source_archive", "source_archive_sha256"],
                  **({"scorer": inputs["scorer"],
                      "additional_arguments": ["scorer_bundle", "scorer_bundle_sha256", "teacher", "teacher_sha256"]}
                     if profile == "tr3_batch" else {})}
        for profile in scope["profiles"]}
    all_wires = []
    for key, root in artifacts.items():
        root = root.resolve()
        wires, excluded, manifest, inventory = artifact_wires(key, root, scope)
        shard_names = sorted(set(inventory.values()))
        digests = sha256_files([root / name for name in shard_names], workers=threads)
        historical = manifest.get("cached_units", manifest.get("cached_expert_units", {}))
        fresh = key in export_requests
        producer = {
            "commit": source_info["commit"] if fresh else historical_commit,
            "source_algorithm": "tessera.encoder_source.v1",
            "source_sha256": encoder_source_sha256() if fresh else historical["historical_producer"]["source_sha256"],
            "contract": bound_file(package_root / "tessera/serving/runtime_contract.json"
                                   if fresh else historical_package / "serving/runtime_contract.json")}
        packet["artifacts"][key] = {
            "path": str(root), "config": bound_file(root / "config.json"),
            "index": bound_file(root / "model.safetensors.index.json"),
            "export_manifest": bound_file(root / "tessera_serving_manifest.json"),
            "producer": producer, "assembler_commit_observed": manifest["git"],
            "shards": [{"name": name, "bytes": (root / name).stat().st_size, "sha256": digest}
                       for name, digest in zip(shard_names, digests)],
            "excluded_modules": excluded,
            "cached_producer": historical.get("historical_producer"),
            "small_read_wires": len(wires)}
        all_wires.extend(wires)
    for family, by_structure in scope["rungs"].items():
        for structure, rungs in by_structure.items():
            for regime in ("batch", "decode"):
                profiles = [name for name, profile in scope["profiles"].items()
                            if profile["regime"] == regime]
                shapes = scope["profiles"][profiles[0]][structure + "_nk"]
                wires = [wire for wire in all_wires if wire["family"] == family
                         and wire["structure"] == structure
                         and wire["rank_local_nk"] in shapes]
                required = set(product(rungs, map(tuple, shapes)))
                observed = {(wire["q256"], tuple(wire["rank_local_nk"])) for wire in wires}
                if not required <= observed:
                    raise ValueError(f"{family}/{structure}: missing wire geometry {required - observed}")
                name = family.lower()
                cell_id = f"{name}_{structure}_sm121_{regime}_resident" + scope["cell_runtime_suffix"]
                packet["cells"].append({"id": cell_id, "family": family, "structure": structure,
                                        "regime": regime, "q256": rungs, "profiles": profiles,
                                        "wires": wires,
                                        "runtime": {**scope["runtime"], "tessera_commit": source_info["commit"],
                                                    "serving_source_sha256": packet["serving"]["source_sha256"]}})
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--export-request", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--historical-package", required=True, type=Path)
    parser.add_argument("--historical-commit", required=True)
    from tools.pq2459_source import add_source_arguments, activate_source
    add_source_arguments(parser)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    source_info = activate_source(args.source_archive, args.source_archive_sha256)
    if args.output.exists():
        parser.error("packet output exists; retain the immutable packet")
    artifacts = {}
    for entry in args.artifact:
        name, sep, path = entry.partition("=")
        if not sep or not name or name in artifacts:
            parser.error("artifact must be a unique NAME=PATH")
        artifacts[name] = Path(path)
    export_requests = {}
    for entry in args.export_request:
        name, sep, path = entry.partition("=")
        if not sep or not name or name in export_requests:
            parser.error("export request must be a unique NAME=PATH")
        export_requests[name] = Path(path)
    for name in artifacts:
        if name != "B" and name not in export_requests:
            parser.error(f"fresh artifact {name} needs its export request")
    packet = build_packet(artifacts, export_requests=export_requests,
                          historical_package=args.historical_package,
                          historical_commit=args.historical_commit,
                          source_info=source_info, threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps({"packet": bound_file(args.output), "cells": len(packet["cells"]),
                      "qualified_cells": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
