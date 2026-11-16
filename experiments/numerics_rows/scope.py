"""Derive expected launch keys from signed wires, without reading C1 evidence."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.numerics_rows.cases import TP_CUTS, admitted_units, configurations, rate_layout
from experiments.numerics_rows.common import Invalid


def contract_document(path=None):
    import tessera

    path = Path(path) if path else Path(tessera.__file__).parent / "serving/runtime_contract.json"
    from experiments.numerics_rows.common import sha256

    return json.loads(path.read_text()), sha256(path)


def wire_geometries(entry):
    """Read and shard wire metadata with the serving parser; do not decode weights."""
    from experiments.numerics_rows.population import store_for
    from experiments.numerics_rows.cases import checkpoint_routed, projection_scheme
    from tessera.routed_fused import dense_module_launch_enabled
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    from tessera.serving.sharding import plan_shard, shard_parsed_roles

    store = store_for(entry)
    scheme = store.schemes[entry["name"]]
    if entry["structure"] == "dense":
        blob = store.wire(entry["name"]).contiguous().numpy().tobytes()
        declaration = scheme
    else:
        scheme, blobs, _ = checkpoint_routed(store, entry, [0])
        blob = blobs["gate_proj"][0]
        declaration = projection_scheme(scheme, "gate_proj", blob)
    parsed = parse_tessera_blob_for_scheme(blob, declaration, entry["name"], device="cpu")
    roles = declaration["roles"]
    down = entry["structure"] == "dense" and entry["name"].endswith("down_proj")
    merged = (entry["structure"] == "dense" and entry["family"] == "TESSERA_FP8"
              and entry["library"] != "triton" and dense_module_launch_enabled())
    result = []
    for size, rank in TP_CUTS:
        plan = plan_shard(entry["name"], roles=roles, columns=declaration["columns"],
                          out_partitions=[int(r) if down else int(r) // size for _, r in roles],
                          in_size=declaration["columns"] // size if down else declaration["columns"],
                          tp_rank=rank, tp_size=size, input_size=declaration["columns"], output_size=sum(int(r) for _, r in roles))
        local = shard_parsed_roles(parsed, plan)
        rows = [int(p.manifest.geometry.rows) for _, p in local]
        first = local[0][1].unit
        geometry = {"rows": sum(rows) if merged else rows[0], "cols": int(local[0][1].manifest.geometry.columns),
                    "tile_words": rate_layout(first.rates)["tile_words"],
                    "blocks": sum(-(-r // 128) for r in rows) if merged else -(-rows[0] // 128), "merged": merged}
        result.append(({"size": size, "rank": rank}, geometry))
    return result


def declared_modules(population):
    """Configuration headers depend only on signed wires and dispatcher functions."""
    result = []
    for entry in population["modules"]:
        cuts = [{"tp": tp, "launch_geometry": geometry,
                 "m_configs": configurations(**geometry, library=entry["library"], sms=population["sms"], structure=entry["structure"])}
                for tp, geometry in wire_geometries(entry)]
        result.append({"name": entry["name"], "cell_ids": entry["cell_ids"], "structure": entry["structure"], "q256": entry["q256"], "cuts": cuts})
    return result


def check_population_cells(contract, image, population):
    """Signed scope names full source-contract IDs, including both regimes."""
    for entry in population["modules"]:
        expected = sorted({s["cell"] for m in (1, 2) for s in admitted_units(
            contract, image, entry["contract_family"], entry["structure"], entry["q256"], entry["pair"], m)
            if s["role"] == "unit"})
        if not expected or entry.get("cell_ids") != expected:
            raise Invalid("signed population cell IDs differ from the source contract")


def check_selectors(contract, population, mapping):
    selected = {name for entry in population["modules"] for name in entry["cell_ids"]}
    for cell in contract["lane_eligibility"]["cells"]:
        if cell["id"] not in selected:
            continue
        if "eager" not in cell.get("runtime", {}).get("execution_modes", ["eager"]):
            raise Invalid("source cell does not admit the eager row")
        for selector in cell.get("requires_serve_flags", []):
            name, value = selector.split("=", 1)
            if mapping.get(name) != value:
                raise Invalid(f"required selector differs from the source cell: {name}")


def unit_scopes(contract, image, entry, config, cut):
    scopes = admitted_units(contract, image, entry["contract_family"], entry["structure"], entry["q256"], entry["pair"], config["M"])
    if not scopes:
        raise Invalid("population pair has no matching contract unit")
    return [{**scope, "module": entry["name"], "symbol": entry["pair"][0], "decoder": entry["pair"][1],
             "rung": entry["q256"], "structure": entry["structure"], "m_config": config, "tp": cut,
             **({"role": "observation", "admitted_by_v68": False} if cut["size"] > 1 and entry["contract_family"] != "TESSERA_E4M3_K1" else {})}
            for scope in scopes]
