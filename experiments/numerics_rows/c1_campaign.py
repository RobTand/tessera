"""List every signed routed TP2 item, in complete-module order. Submit nothing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.numerics_rows.common import Invalid, write_json

ROLES = ("gate_proj", "up_proj", "down_proj")


def items(population):
    modules = population["modules"]
    if not modules or len({e["name"] for e in modules}) != len(modules) or any(
            e["structure"] != "routed_moe" or e["family"] != "TESSERA_FP8" or type(e["experts"]) is not int or e["experts"] < 1 for e in modules):
        raise Invalid("campaign requires unique signed routed E4M3 modules with expert counts")
    for module, entry in enumerate(modules):
        for role in ROLES:
            for rank in range(2):
                for expert in range(entry["experts"]):
                    yield {"job_name": f"tessera-1717-C1-R-M{module:02d}-E{expert:03d}-{role}-TP2-{rank}",
                           "module": entry["name"], "module_order": module, "expert": expert,
                           "role": role, "tp": {"size": 2, "rank": rank},
                           "pass_name": f"tessera-1717-C1-R-M{module:02d}-{role}-TP2-{rank}"}


def main():
    from experiments.numerics_rows.population import load_signed_population

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    population, digest = load_signed_population(args.population)
    args.out.mkdir(parents=True, exist_ok=True)
    counts, passes = {}, {}
    with (args.out / "items.jsonl").open("w") as stream:
        for item in items(population):
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
            counts[item["module"]] = counts.get(item["module"], 0) + 1
            key = item["pass_name"]
            if key not in passes:
                passes[key] = {k: v for k, v in item.items() if k not in ("job_name", "expert")}
                passes[key]["experts"] = 0
            passes[key]["experts"] += 1
    write_json(args.out / "passes.json", list(passes.values()))
    write_json(args.out / "campaign.json", {"population_sha256": digest, "modules": counts,
        "items": sum(counts.values()), "passes": len(passes), "cores": 1, "memory_gib": 4,
        "parallel_per_spark": 8, "parallel_total": 16, "submission_enabled": False,
        "wall_hours_at_item_seconds": "items * seconds_per_item / (16 * 3600)"})


if __name__ == "__main__":
    main()
