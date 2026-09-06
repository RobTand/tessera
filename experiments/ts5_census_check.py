#!/usr/bin/env python3
"""Fail closed on an incomplete dense/routed campaign census; replay current cells.

This is a receipt/sidecar gate, not a wire audit or a quality measurement. Run
under PrismaBuild against the same merged checkpoint mounted by the census.
Host and container checkpoint names may differ; exact sidecar hashes bind
the supplied config/manifest to the serve, and owner evidence joins schemes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tessera.serving.contract import (
    CENSUS_PHASE_REGIMES, PAYLOAD_FAMILY_BY_ROUTE, construction_entry, output_partitions,
    load_serving_contract, require_runtime_image, vllm_module_name)
from tessera.serving.scheme import (
    ROUTES, STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE, expert_role_declarations, launch_pairs,
    moe_census_symbol_base as census_symbol_base,
    validate_tessera_scheme)
from tessera.serving.census import STRUCTURE_BY_RECORD_KIND
from tessera.serving.dense_ownership import fused_module, partition_members
from tessera.serving_parts import sha256_file, validate_explicit_plan
from tools.tessera_route_census import (
    CHECKPOINT_SIDECAR_NAMES, all_structure_agreement, checkpoint_sidecar_hashes,
    declared_rung, join_records_to_declared, parse_eager_shape, phase_shape_problems)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result
    with Path(path).open() as stream:
        return json.load(stream, object_pairs_hook=unique)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mapping(value, where):
    _require(isinstance(value, dict), f"{where} must be an object")
    return value


def _same_roster(actual, expected, where):
    _require(isinstance(actual, (list, tuple)) and all(isinstance(x, str) for x in actual),
             f"{where} must be a list of names")
    _require(Counter(actual) == Counter(expected), f"{where} is not the exact planned population")


def _population(plan, config, manifest, entry):
    _mapping(plan, "plan")
    planned = {}
    for target, spec in plan.items():
        if spec in ("PASSTHROUGH", "BF16"):
            continue
        _require(isinstance(target, str), "plan targets must be names")
        _mapping(spec, f"plan {target}")
        _require(isinstance(spec.get("grid"), str) and type(spec.get("q256")) is int,
                 f"plan {target}: grid and integer q256 are required")
        planned[target] = spec
    _require(bool(planned), "plan has no quantized targets")
    identity = _mapping(manifest.get("export_identity"), "manifest.export_identity")
    _require(_mapping(identity.get("options"), "export_identity.options").get("plan") == plan,
             "manifest export identity does not bind the supplied common plan")
    _require("export_partition" not in manifest, "checkpoint is a partition, not a merged artifact")
    quant = _mapping(config.get("quantization_config"), "config.quantization_config")
    _require(quant.get("quant_method") == "tessera", "config is not Tessera")
    groups = _mapping(quant.get("config_groups"), "config_groups")
    schemes = {}
    for key, group in groups.items():
        _mapping(group, f"config_groups.{key}")
        targets = group.get("targets")
        _require(isinstance(targets, list) and targets, f"config_groups.{key} has no targets")
        _require(group.get("format") == "TESSERA", f"config_groups.{key} is not TESSERA")
        for target in targets:
            _require(isinstance(target, str) and target not in schemes,
                     f"config_groups.{key}: invalid or duplicate target {target!r}")
            raw = _mapping(group.get("scheme"), f"scheme {target}")
            schemes[target] = validate_tessera_scheme(raw, target)
    modules = _mapping(manifest.get("modules"), "manifest.modules")
    _same_roster(list(modules), list(schemes), "manifest modules/config targets")
    # The exporter/assembler owns source-leaf coverage and passthrough
    # obligations. A dense plan names weight tensors, while config targets
    # name their fused serving owners; the manifest's roles join the two.
    validate_explicit_plan(plan, modules, groups)
    units = routed_units = 0
    dense_tensors, routed_targets = [], []
    for target, scheme in schemes.items():
        module = _mapping(modules[target], f"manifest {target}")
        if scheme["structure"] == STRUCTURE_DENSE:
            units += _dense_population(target, scheme, module, planned, dense_tensors, entry)
            continue
        routed_targets.append(target)
        _require(target in planned, f"scheme {target}: unplanned routed target")
        _require(scheme["grid"] == planned[target]["grid"] and
                 declared_rung(scheme) == planned[target]["q256"],
                 f"scheme {target}: grid/rung differs from plan")
        expected = {"structure": STRUCTURE_ROUTED_MOE, "family": scheme["family"],
                    "grid": planned[target]["grid"], "q256": planned[target]["q256"],
                    "experts": scheme["experts"]}
        for field, value in expected.items():
            _require(module.get(field) == value, f"manifest {target}.{field} disagrees with scheme")
        expected_roles = {}
        for expert in range(scheme["experts"]):
            for group, declaration in scheme["groups"].items():
                for role in expert_role_declarations(declaration):
                    expected_roles[(expert, group, role["roles"][0][0])] = {
                        "rows": role["rows"], "cols": role["columns"], "q256": role["q256"],
                        "grid": expected["grid"], "family": role["family"]}
        roles = module.get("roles")
        _require(isinstance(roles, list), f"manifest {target}.roles must be a list")
        seen = set()
        for role in roles:
            _mapping(role, f"manifest {target} role")
            key = (role.get("expert"), role.get("group"), role.get("role"))
            _require(key in expected_roles and key not in seen,
                     f"manifest {target}: extra or duplicate projection {key}")
            seen.add(key)
            for field, value in expected_roles[key].items():
                _require(role.get(field) == value,
                         f"manifest {target} projection {key}.{field} disagrees with scheme")
        _require(seen == set(expected_roles), f"manifest {target}: missing expert projections")
        units += len(expected_roles)
        routed_units += len(expected_roles)
    _same_roster(dense_tensors, [t for t in planned if t not in routed_targets],
                 "dense planned tensor population")
    summary = _mapping(manifest.get("routed_moe"), "manifest.routed_moe")
    _same_roster(summary.get("quantized_stacks"), routed_targets, "routed_moe.quantized_stacks")
    _require(summary.get("quantized_logical_units") == routed_units, "routed_moe logical unit count differs")
    totals = _mapping(manifest.get("totals"), "manifest.totals")
    _require(totals.get("modules") == len(schemes) and totals.get("units") == units,
             "manifest totals differ from the complete planned projection population")
    return planned, schemes, units


def _dense_population(target, scheme, module, planned, dense_tensors, entry):
    """Join exact source leaves to the reader's ordered fused-role geometry.

    The roles of a dense owner are what ``partition_members`` derives from the
    runtime's attested output partitions (tessera#377): one role per member
    tensor, or -- for a ``MergedColumnParallelLinear`` built from ONE source
    tensor -- one role per partition, each a row window of it.  A manifest
    whose roles are any other sequence disagrees with the serve's geometry and
    is refused here, where the disagreement names the module, rather than at
    load.
    """
    _require(module.get("structure", STRUCTURE_DENSE) == STRUCTURE_DENSE,
             f"manifest dense {target}: structure disagrees with scheme")
    expected = {"family": scheme["family"], "grid": scheme["grid"],
                "q256": scheme["q256"], "rows": scheme["rows"], "cols": scheme["columns"],
                "container_bytes": scheme["wire_bytes"]}
    for field, value in expected.items():
        _require(module.get(field) == value,
                 f"manifest dense {target}.{field} disagrees with scheme")
    roles = module.get("roles")
    _require(isinstance(roles, list) and roles and len(roles) == len(scheme["roles"]),
             f"manifest dense {target}: role population disagrees with scheme")
    for role in roles:
        _mapping(role, f"manifest dense {target} role")
        tensor = role.get("tensor")
        _require(isinstance(tensor, str) and tensor in planned,
                 f"manifest dense {target}: unplanned source tensor {tensor!r}")
        _require(tensor.endswith(".weight"), f"manifest dense {target}: source tensor is not a weight")
    tensors = list(dict.fromkeys(r["tensor"] for r in roles))
    owner, members = fused_module(tensors[0]) or (tensors[0].removesuffix(".weight"), (tensors[0],))
    _require(owner == target and tensors == list(members),
             f"manifest dense {target}: source tensor {tensors[0]!r} belongs to owner {owner!r} "
             f"with ordered members {members}, manifest names {tensors}")
    # A whole-tensor role written before tessera#377 carries neither field;
    # its window is the tensor.
    tensor_rows = {r["tensor"]: int(r.get("source_rows", r.get("rows", 0))) for r in roles}
    try:
        expected_members = partition_members(target, members, tensor_rows,
                                             output_partitions(entry, target))
    except ValueError as exc:
        _require(False, f"manifest dense {target}: {exc}")
    got = [(r["tensor"], r.get("role"), int(r.get("row_offset", 0)), r.get("rows")) for r in roles]
    want = [(m.tensor, m.role, m.row_offset, m.rows) for m in expected_members]
    _require(got == want,
             f"manifest dense {target}: roles {got} are not the runtime's partition {want}")
    for role, (name, rows), rung in zip(roles, scheme["roles"], scheme["role_q256"]):
        tensor = role["tensor"]
        expected = {"role": name, "rows": rows, "cols": scheme["columns"],
                    "grid": scheme["grid"], "q256": rung, "family": scheme["family"]}
        for field, value in expected.items():
            _require(role.get(field) == value,
                     f"manifest dense {target} role {tensor}.{field} disagrees with scheme")
    # The roster is of planned source tensors. A row-sliced owner (tessera#377)
    # emits several roles for one tensor, so each tensor is counted once.
    dense_tensors.extend(dict.fromkeys(r["tensor"] for r in roles))
    return len(roles)


def check_census(plan, config, manifest, census, *, runtime_image, checkpoint, checkpoint_sidecars,
                 contract=None, require_attested=False):
    """Validate the complete raw population and replay the current contract."""
    runtime_image = require_runtime_image(runtime_image)
    contract = load_serving_contract() if contract is None else contract
    for value, name in ((config, "config"), (manifest, "manifest"), (census, "census")):
        _mapping(value, name)
    _mapping(checkpoint_sidecars, "supplied checkpoint sidecars")
    _require(set(checkpoint_sidecars) == set(CHECKPOINT_SIDECAR_NAMES) and
             all(isinstance(value, str) and value for value in checkpoint_sidecars.values()),
             "campaign requires both supplied checkpoint sidecar file hashes")
    _require(census.get("checkpoint_sidecars") == checkpoint_sidecars,
             "raw census checkpoint sidecar hashes differ from supplied files")
    entry = construction_entry(config.get("architectures"), contract)
    _require(entry is not None, "no attested construction name mapper for checkpoint architecture")
    planned, schemes, units = _population(plan, config, manifest, entry)
    _require(census.get("schema") == "tessera.serving.route_census/2", "raw census schema must be v2")
    runtime = {"image": runtime_image, "execution_mode": "eager"}
    _require(census.get("runtime") == runtime and census.get("compiled") is False,
             "census must record the requested exact runtime image and eager execution")
    _require(_mapping(census.get("env"), "census.env").get("TESSERA_SERVE_MODE") == "resident",
             "census must record resident execution")
    _require(census.get("verdict") == "served" and census.get("problems") == [],
             "raw census must be served with no problems")
    mapping = {target: vllm_module_name(entry, target) for target in schemes}
    _require(len(set(mapping.values())) == len(mapping), "construction maps targets to duplicate owners")
    stored_mapping = census.get("declared_name_mapping")
    if stored_mapping is None:
        _require(all(k == v for k, v in mapping.items()) and
                 census.get("declared_names_mapped_to_module_space") is False,
                 "census omitted a required construction name mapping")
    else:
        _require(stored_mapping == mapping and census.get("declared_names_mapped_to_module_space") is True,
                 "census declared name mapping differs from current construction mapping")
    expected_owners = sorted(mapping.values())
    runtime_schemes = {mapping[target]: scheme for target, scheme in schemes.items()}
    records = _mapping(census.get("records"), "census.records")
    _same_roster(list(records), list(CENSUS_PHASE_REGIMES), "census driven phases")
    stored_owners = _mapping(census.get("record_owner"), "census.record_owner")
    _same_roster(list(stored_owners), list(CENSUS_PHASE_REGIMES), "census owner phases")
    owners, owner_records, symbols = {}, {}, {}
    for phase, regime in CENSUS_PHASE_REGIMES.items():
        observed = _mapping(records[phase], f"census.records.{phase}")
        for name, record in observed.items():
            _mapping(record, f"{phase} {name}")
        owner, problems = join_records_to_declared(observed, runtime_schemes)
        _require(not problems and set(owner) == set(observed),
                 f"{phase}: unowned or ambiguous route records: {problems}")
        _same_roster(list(owner.values()), expected_owners, f"{phase} owner bijection")
        _require(stored_owners[phase] == owner, f"{phase}: recorded owner map differs from replay")
        owners[phase] = owner
        owner_records[phase] = {owner[name]: record for name, record in observed.items()}
        for name, record in observed.items():
            scheme = runtime_schemes[owner[name]]
            family = scheme["family"]
            structure = scheme["structure"]
            _, rows, columns = parse_eager_shape(record.get("shape"))
            # Routed create_weights records the w13 tile; dense records its
            # fused Linear shape. Both come from the validated owner scheme.
            geometry = ((scheme["groups"]["w13"]["rows"], scheme["hidden_size"])
                        if structure == STRUCTURE_ROUTED_MOE
                        else (scheme["rows"], scheme["columns"]))
            _require((rows, columns) == geometry,
                     f"{phase}: {name} shape N/K disagrees with declared {structure} tile")
            _require(STRUCTURE_BY_RECORD_KIND.get(record.get("kind")) == structure and
                     record.get("state") == "served" and
                     record.get("policy") == f"{family}:resident" and
                     record.get("contract") == ROUTES[family]["activation_contract"],
                     f"{phase}: {name} kind/state/policy/activation disagrees with declared {structure} route")
            symbol = (census_symbol_base(record.get("symbol"))
                      if structure == STRUCTURE_ROUTED_MOE else record.get("symbol"))
            pair = (symbol, record.get("decoder"))
            _require(pair in launch_pairs(family, structure=structure,
                                          regime=regime, mode="resident"),
                     f"{phase}: {name} launch pair {pair} is not owned by its declared {structure} route")
        symbols[phase] = sorted({record["symbol"] for record in observed.values()})
    shape_problems = phase_shape_problems(owner_records, phase_regimes=CENSUS_PHASE_REGIMES,
                                         require_each_owner=True)
    _require(not shape_problems, "; ".join(shape_problems))
    capability = _mapping(census.get("device"), "census.device").get("capability")
    _require(isinstance(capability, list) and len(capability) == 2 and
             all(type(x) is int and x >= 0 for x in capability), "census device capability is missing")
    platform = f"sm_{capability[0]}{capability[1]}"
    agreement, problems = all_structure_agreement(
        records, cells=contract["lane_eligibility"]["cells"], phase_regimes=CENSUS_PHASE_REGIMES,
        platform=platform, declared_rungs={mapping[t]: declared_rung(schemes[t]) for t in schemes},
        record_owners=owners, families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=runtime_image, execution_mode="eager")
    _require(not problems, f"current contract launch disagreement: {problems}")
    if require_attested:
        for structure, count in Counter(s["structure"] for s in schemes.values()).items():
            block = agreement["structures"][structure]
            _require(block["agrees"] is True and all(
                phase["covered_by_cell"] == count and phase["unattested"] == 0
                for phase in block["phases"].values()),
                f"current contract does not attest every planned {structure} owner in both driven phases: "
                f"{block['phases']}")
    return {"schema": "tessera.ts5-census-check/1", "verdict": "passed",
            "require_attested": require_attested, "runtime": runtime, "residency": "resident",
            "checkpoint": str(checkpoint), "census_checkpoint": census.get("checkpoint"),
            "checkpoint_sidecars": dict(checkpoint_sidecars),
            "expected_owners": expected_owners, "expected_projection_units": units,
            "declared_name_mapping": mapping, "record_owner": owners, "symbols": symbols,
            "owner_shapes": {p: {n: r["shape"] for n, r in rs.items()} for p, rs in owner_records.items()},
            "export_identity": manifest["export_identity"], "cell_launch_agreement": agreement,
            "current_contract_sha256": hashlib.sha256(json.dumps(
                contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "checkpoint", "census", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--require-attested", action="store_true")
    args = parser.parse_args(argv)
    paths = {"plan": args.plan, "config": args.checkpoint / "config.json",
             "manifest": args.checkpoint / "tessera_serving_manifest.json", "census": args.census}
    try:
        data = {name: read_json(path) for name, path in paths.items()}
        result = check_census(**data, runtime_image=args.runtime_image, checkpoint=args.checkpoint,
                              checkpoint_sidecars=checkpoint_sidecar_hashes(args.checkpoint),
                              require_attested=args.require_attested)
        result["inputs"] = {name: {"path": str(path), "sha256": sha256_file(path)}
                            for name, path in paths.items()}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = {"schema": "tessera.ts5-census-check/1", "verdict": "REFUSED", "problems": [str(exc)]}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"verdict": result["verdict"], "out": str(args.out),
                      "problems": result.get("problems", [])}))
    return 0 if result["verdict"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
