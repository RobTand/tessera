"""Join every signed module once. A module never stands for another module."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.numerics_rows.common import OUTCOMES, Invalid, sha256, write_json


def rule_unit_id(unit):
    """Project rows section 2 launch parameters; M and tensor ranges are evidence."""
    config = unit["m_config"]
    fields = (("triton_shape",) if unit["decoder"] == "native_window_gemm" else
              ("superblock_width",) if unit.get("structure") == "routed_moe" else ("k_split", "last_tile_partial"))
    return [unit[k] for k in ("cell", "symbol", "decoder", "rung")] + [
        {k: config[k] for k in fields}, {k: unit["tp"][k] for k in ("size", "rank")}, unit.get("launch_state_sha256")]


def execution_identity(receipt):
    from experiments.numerics_rows.launch_state import check_fields
    check_fields({**receipt["runtime"], "launch_levers_sha256": receipt["inputs"]["launch_levers_sha256"]})
    if receipt["rule"].get("launch_levers_sha256") != receipt["inputs"]["launch_levers_sha256"]:
        raise Invalid("row launch lever file differs from its frozen rule")
    fields = {"tool_commit": receipt["tool"]["commit"], "tool_tree_sha256": receipt["tool"]["tree_sha256"],
              "image_digest": receipt["runtime"]["image_digest"], "source_commit": receipt["tessera"]["commit"],
              "contract_sha256": receipt["tessera"]["contract_sha256"], "build_sha256": receipt["inputs"]["build_sha256"],
              "h1_sha256": receipt["rule"]["h1_sha256"], "rule_sha256": receipt["rule"]["sha256"],
              "q1_table_sha256": receipt["inputs"]["q1_table_sha256"],
              "launch_levers_sha256": receipt["inputs"]["launch_levers_sha256"],
              "launch_state_sha256": receipt["runtime"]["launch_state_sha256"]}
    if any(not isinstance(v, str) or not v for v in fields.values()):
        raise Invalid("join execution identity is incomplete")
    return fields


def join(population, digest, manifests, receipts, contract=None):
    """Reject malformed evidence; missing evidence contributes per-key UNATTESTED."""
    try:
        return _join(population, digest, manifests, receipts, contract)
    except Invalid:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise Invalid(f"malformed join input: {error}") from error


def _join(population, digest, manifests, receipts, contract):
    import hashlib
    from experiments.numerics_rows import scope
    from experiments.numerics_rows.common import IMAGES
    from experiments.numerics_rows.population import select_modules
    from experiments.numerics_rows.row import planned_units, unit_identity
    from experiments.numerics_rows.launch_state import check_binding, check_math_binding, check_fields

    contract, contract_hash = scope.contract_document(contract)
    image = IMAGES[population["image"]]["image_digest"]
    scope.check_population_cells(contract, image, population)
    launch = ({**{k: receipts[0][1]["runtime"][k] for k in ("launch_state_names", "launch_state_sha256", "launch_selectors", "start_environment_spec_sha256")},
               "launch_levers_sha256": receipts[0][1]["inputs"]["launch_levers_sha256"]} if receipts else
              {k: manifests[0][1][k] for k in ("launch_state_names", "launch_state_sha256", "launch_selectors", "start_environment_spec_sha256", "launch_levers_sha256")} if manifests else None)
    if launch is not None:
        check_fields(launch)
        scope.check_selectors(contract, population, launch["launch_selectors"])
    state = launch["launch_state_sha256"] if launch is not None else None
    names = [e["name"] for e in population["modules"]]
    if not names or len(set(names)) != len(names):
        raise Invalid("signed population module names are not unique")
    declarations = {m["name"]: m for m in scope.declared_modules(population)}
    required, allowed = {}, {}
    pack = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    for entry in population["modules"]:
        for cut in declarations[entry["name"]]["cuts"]:
            for config in cut["m_configs"]:
                for unit in scope.unit_scopes(contract, image, entry, config, cut["tp"]):
                    unit["launch_state_sha256"] = state
                    if unit["role"] != "unit" or unit["admitted_by_v68"] is not True:
                        continue
                    key = pack(rule_unit_id(unit))
                    required.setdefault(key, set()).add(entry["name"])
                    allowed.setdefault((entry["name"], key), set()).add(pack(unit_identity(unit)))
    if not required:
        raise Invalid("signed population has no required attestation keys")
    cases, case_modules, case_identity = {}, set(), None
    for case_sha, case in manifests:
        check_binding(case, launch)
        check_math_binding(case, case)
        selection = case["selected_modules"]
        if case_sha in cases or len(selection) != len(set(selection)) or not selection or set(selection) - set(names) or case_modules.intersection(selection):
            raise Invalid("duplicate or unknown C1 module selection")
        if case["population_sha256"] != digest:
            raise Invalid("join input belongs to a different signed population")
        binding = tuple(case[k] for k in ("source_commit", "contract_sha256", "q1_table_sha256"))
        if case["contract_sha256"] != contract_hash or case_identity is not None and binding != case_identity:
            raise Invalid("joined C1 and row execution identities differ")
        case_identity = binding
        # Validate all roles and cuts with the same policy used by the row.
        planned_units(contract, image, select_modules(population, selection), case)
        for module in case["modules"]:
            expected = declarations[module["name"]]
            if any(module[k] != expected[k] for k in ("cell_ids", "structure", "q256")):
                raise Invalid("C1 geometry differs from signed wires")
            for cut in module["cuts"]:
                declared, = [c for c in expected["cuts"] if c["tp"] == cut["tp"]]
                if any(cut[k] != declared[k] for k in ("launch_geometry", "m_configs")):
                    raise Invalid("C1 omitted or changed an expected M configuration from signed wires")
        case_modules.update(selection)
        cases[case_sha] = case
    evidence, execution, seen_receipts, seen_full, row_modules = {}, None, set(), set(), set()
    key_math = {}
    for receipt_sha, receipt in receipts:
        if receipt_sha in seen_receipts:
            raise Invalid("duplicate row receipt")
        seen_receipts.add(receipt_sha)
        if receipt.get("canary") or receipt["row"].get("canary") or receipt["rule"].get("canary"):
            raise Invalid("a canary cannot supply attestation")
        frozen = receipt["rule"].get("frozen_utc")
        if not isinstance(frozen, str) or not frozen.strip():
            raise Invalid("join requires a frozen rule in every receipt")
        if not receipt["row"].get("utc_end") or type(receipt["row"].get("exit_status")) is not int or receipt["row"].get("finalization_finished") is not True:
            raise Invalid("join requires finished row receipts")
        identity = execution_identity(receipt)
        check_math_binding(receipt["runtime"], receipt["runtime"])
        check_binding({**receipt["runtime"], "launch_levers_sha256": receipt["inputs"]["launch_levers_sha256"]}, launch)
        if execution is not None and execution != identity:
            raise Invalid("joined rows have different execution identities")
        if identity["contract_sha256"] != contract_hash or identity["image_digest"] != image:
            raise Invalid("joined rows have different execution identities from the signed scope")
        if case_identity is not None and tuple(identity[k] for k in ("source_commit", "contract_sha256", "q1_table_sha256")) != case_identity:
            raise Invalid("joined C1 and row execution identities differ")
        execution = identity
        inputs = receipt["inputs"]
        selection = inputs["selected_modules"]
        if inputs["population_sha256"] != digest or not selection or len(selection) != len(set(selection)) or set(selection) - set(names):
            raise Invalid("row population or selected modules differ")
        if row_modules.intersection(selection):
            raise Invalid("duplicate row module selection")
        row_modules.update(selection)
        case = cases.get(inputs["case_manifest_sha256"])
        if case is not None:
            check_math_binding(receipt["runtime"], case)
            if case["selected_modules"] != selection:
                raise Invalid("row and C1 selected modules differ")
            if any(case.get(k) != identity[k] for k in ("source_commit", "contract_sha256", "q1_table_sha256")):
                raise Invalid("joined C1 and row execution identities differ")
        for unit in receipt["units"]:
            if unit.get("launch_state_sha256") != state:
                raise Invalid("row unit launch state differs from its worker")
            if unit["unit_id"] != unit_identity(unit) or unit["outcome"] not in OUTCOMES or unit["role"] not in ("unit", "observation") or type(unit["admitted_by_v68"]) is not bool:
                raise Invalid("row unit key or outcome is invalid")
            if unit["role"] != "unit" or unit["admitted_by_v68"] is not True:
                continue
            if unit["outcome"] == "PASS":
                from experiments.numerics_rows.dependency_code import check_unit_code
                check_unit_code(unit, receipt["runtime"]["image_digest"])
            check_math_binding(unit, receipt["runtime"])
            module = unit.get("module")
            if module is None and len(selection) == 1:
                module = selection[0]
            key = pack(rule_unit_id(unit))
            check_math_binding(unit, key_math.setdefault(key, unit))
            full = pack(unit_identity({**unit, "tp": {k: unit["tp"][k] for k in ("size", "rank")}}))
            if module not in selection or full not in allowed.get((module, key), ()):
                raise Invalid("row unit is outside the required signed scope")
            marker = (module, full)
            if marker in seen_full:
                raise Invalid("duplicate module unit evidence")
            seen_full.add(marker)
            pair = (module, key)
            record = evidence.setdefault(pair, {"outcomes": [], "receipts": set(), "c1_present": True})
            record["outcomes"].append(unit["outcome"])
            record["receipts"].add(receipt_sha)
            record["c1_present"] &= case is not None
    units = []
    for key, modules in sorted(required.items()):
        joined = {"unit_id": json.loads(key), "modules": []}
        for module in sorted(modules):
            record = evidence.get((module, key))
            outcomes = list(record["outcomes"]) if record else []
            missing = (not record or not record["c1_present"] or
                       any((module, full) not in seen_full for full in allowed[(module, key)]))
            if missing:
                outcomes.append("UNATTESTED")
            joined["modules"].append({"module": module, "outcome": min(outcomes, key=OUTCOMES.index),
                                      "receipt_sha256s": sorted(record["receipts"]) if record else [],
                                      "missing_evidence": bool(missing)})
        joined["outcome"] = min((m["outcome"] for m in joined["modules"]), key=OUTCOMES.index)
        units.append(joined)
    declared = {"population_sha256": digest, "modules": list(declarations.values()), "launch": launch,
                "contract_sha256": contract_hash, "image_digest": image}
    return {"schema": "tessera.numerics_join.v2", "population_sha256": digest, "execution": execution,
            "required_scope": declared, "required_scope_sha256": hashlib.sha256(pack(declared).encode()).hexdigest(),
            "selected_modules": sorted(names), "verdict": min((u["outcome"] for u in units), key=OUTCOMES.index), "units": units}


def sample_rate(first, second):
    """Report the lead's two-sample mean, including each sample's write time."""
    for field in ("population_sha256", "q1_table_sha256", "source_commit", "contract_sha256"):
        if first[field] != second[field]:
            raise Invalid("routed samples have different inputs")
    if any(m.get("generation_finished") is not True or not m.get("sample") or len(m["items"]) != 3 or len(m["modules"]) != 1 for m in (first, second)):
        raise Invalid("two finished three-role samples are required")
    if first["modules"][0]["name"] != second["modules"][0]["name"] or first["modules"][0]["experts"] != second["modules"][0]["experts"]:
        raise Invalid("routed samples cover different modules")
    if {first["sample"]["expert"], second["sample"]["expert"]} != {0, first["modules"][0]["experts"] // 2}:
        raise Invalid("samples must cover expert zero and the middle expert")
    mean = sum(m["reference_seconds"] / len(m["items"]) for m in (first, second)) / 2
    count = first["modules"][0]["experts"] * 3 * 3
    return {"sample_scope_only": True, "samples": [m["sample"] for m in (first, second)],
            "mean_seconds_per_item": mean, "module_items": count, "projected_module_seconds": mean * count,
            "full_manifest_serialization_measured": False}


def main():
    from experiments.numerics_rows.population import load_signed_population

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--manifest", type=Path, action="append", default=[])
    p.add_argument("--row", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    population, digest = load_signed_population(args.population)
    load = lambda paths: [(sha256(path), json.loads(path.read_text())) for path in paths]
    write_json(args.out, join(population, digest, load(args.manifest), load(args.row), args.contract))


if __name__ == "__main__":
    main()
