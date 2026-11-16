"""Join every signed module once. A module never stands for another module."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.numerics_rows.common import Invalid, sha256, write_json

OUTCOMES = ("FAIL", "INVALID", "STOPPED", "UNATTESTED", "PASS")


def rule_unit_id(unit):
    """Project rows section 2 launch parameters; M and tensor ranges are evidence."""
    config = unit["m_config"]
    fields = (("triton_shape",) if unit["decoder"] == "native_window_gemm" else
              ("superblock_width",) if unit.get("structure") == "routed_moe" else ("k_split", "last_tile_partial"))
    return [unit[k] for k in ("cell", "symbol", "decoder", "rung")] + [
        {k: config[k] for k in fields}, {k: unit["tp"][k] for k in ("size", "rank")}]


def execution_identity(receipt):
    fields = {"tool_commit": receipt["tool"]["commit"], "tool_tree_sha256": receipt["tool"]["tree_sha256"],
              "image_digest": receipt["runtime"]["image_digest"], "source_commit": receipt["tessera"]["commit"],
              "contract_sha256": receipt["tessera"]["contract_sha256"], "build_sha256": receipt["inputs"]["build_sha256"],
              "h1_sha256": receipt["rule"]["h1_sha256"], "rule_sha256": receipt["rule"]["sha256"],
              "q1_table_sha256": receipt["inputs"]["q1_table_sha256"]}
    if any(not isinstance(v, str) or not v for v in fields.values()):
        raise Invalid("join execution identity is incomplete")
    return fields


def join(population, digest, manifests, receipts):
    from experiments.numerics_rows.row import unit_identity

    expected = Counter(e["name"] for e in population["modules"])
    if not expected or any(n != 1 for n in expected.values()):
        raise Invalid("signed population module names are not unique")
    for documents, is_case in ((manifests, True), (receipts, False)):
        selected = Counter()
        for _, doc in documents:
            scope = doc if is_case else doc["inputs"]
            if scope["population_sha256"] != digest:
                raise Invalid("join input belongs to a different signed population")
            selected.update(scope["selected_modules"])
            if is_case and (doc.get("generation_finished") is not True or doc.get("sample") or
                            Counter(m["name"] for m in doc["modules"]) != Counter(scope["selected_modules"])):
                raise Invalid("join requires finished, full module manifests")
        if selected != expected:
            raise Invalid("join requires every signed module exactly once")
    cases = dict(manifests)
    units, execution = {}, None
    for receipt_sha, receipt in receipts:
        if not receipt["row"].get("utc_end") or type(receipt["row"].get("exit_status")) is not int:
            raise Invalid("join requires finished row receipts")
        identity = execution_identity(receipt)
        if execution is not None and execution != identity:
            raise Invalid("joined rows have different execution identities")
        execution = identity
        scope = receipt["inputs"]
        case = cases.get(scope["case_manifest_sha256"])
        if case is None or case["selected_modules"] != scope["selected_modules"] or not receipt["units"]:
            raise Invalid("row has no matching C1 manifest or unit receipts")
        if any(case.get(k) != identity[k] for k in ("source_commit", "contract_sha256", "q1_table_sha256")):
            raise Invalid("joined C1 and row execution identities differ")
        for unit in receipt["units"]:
            if unit["unit_id"] != unit_identity(unit) or unit["outcome"] not in OUTCOMES:
                raise Invalid("row unit key or outcome is invalid")
            projected = rule_unit_id(unit)
            key = json.dumps(projected, sort_keys=True, separators=(",", ":"))
            joined = units.setdefault(key, {"unit_id": projected, "outcome": "PASS", "modules": []})
            joined["outcome"] = min((joined["outcome"], unit["outcome"]), key=OUTCOMES.index)
            joined["modules"].append({"modules": scope["selected_modules"], "outcome": unit["outcome"], "receipt_sha256": receipt_sha, "source_unit_id": unit["unit_id"]})
    return {"schema": "tessera.numerics_join.v1", "population_sha256": digest, "execution": execution, "selected_modules": sorted(expected),
            "verdict": min((u["outcome"] for u in units.values()), key=OUTCOMES.index),
            "units": [units[k] for k in sorted(units)]}


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
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--row", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    population, digest = load_signed_population(args.population)
    load = lambda paths: [(sha256(path), json.loads(path.read_text())) for path in paths]
    write_json(args.out, join(population, digest, load(args.manifest), load(args.row)))


if __name__ == "__main__":
    main()
