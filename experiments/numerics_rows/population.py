"""Draft the specified populations, then read their wire stamps in a CPU job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from experiments.numerics_rows.common import check_source, write_json

BASE = Path("/mnt/shared/tessera-measurements")
STUB = Path("/mnt/shared/tessera-runs/moe/u1-stubs-20260926/stub-B")
CACHE = BASE / "glm-canonical-census-20260908/activation-runtime-allocation-20260911/union-a4a8a16-01/cache/wire"
OPTION_C = Path("/mnt/shared/flux-jobs/tessera-option-c-r1-5c2ef6de-20261115T1023Z/artifact-5c2ef6de")
BF16 = Path("/mnt/shared/flux-jobs/tessera-bf16-only-5c2ef6de-20261115T2027Z/artifact-5c2ef6de")
FAMILY = {"TESSERA_FP8": "TESSERA_E4M3_K1", "TESSERA_NVFP4": "TESSERA_E2M1_K2", "TESSERA_BF16": "TESSERA_BF16_K1"}


def module_key(entry):
    return tuple(json.dumps(entry.get(key), sort_keys=True) for key in
                 ("root", "name", "wire_path", "wire_root", "wire_family", "layer"))


def store_for(entry):
    """Use the served Store interface for checkpoint and canonical dense wires."""
    from experiments.dense_fused_oracle import Store

    store = Store(entry["root"])
    if "wire_path" not in entry:
        return store
    import torch
    from tessera import fused
    from tessera.unit_artifact import parse_unit_metadata

    blob = Path(entry["wire_path"]).read_bytes()
    template = store.schemes[entry.get("scheme_template", entry["name"])]
    if blob[:8] != fused.FUSED_MAGIC:
        meta = parse_unit_metadata(blob)
        name, = [name for name, _ in template["roles"]]
        blob = fused.pack_fused([(name, int(meta.rows), blob)])
    member, = fused.parse_fused(blob)
    meta = parse_unit_metadata(member.blob)
    store.schemes[entry["name"]] = {**template, "grid": meta.grid.name, "body": meta.body.name,
        "plane": meta.manifest.scale_plane.kind.name, "q256": int(meta.manifest.branch.root_q256) // int(meta.grid.arity),
        "rows": int(meta.rows), "columns": int(meta.columns), "roles": [[member.name, int(meta.rows)]], "wire_bytes": len(blob)}
    get = store.get
    store.get = lambda name: torch.frombuffer(bytearray(blob), dtype=torch.uint8) if name == entry["name"] + ".wire_bytes" else get(name)
    return store


def entries(root, *, family=None, structure=None):
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    result = []
    for group in config["quantization_config"]["config_groups"].values():
        scheme = group["scheme"]
        if family and scheme["family"] != family or structure and scheme.get("structure", "dense") != structure:
            continue
        for name in group["targets"]:
            routed = scheme.get("structure") == "routed_moe"
            source = scheme["groups"]["w13"] if routed else scheme
            names = [f"{name}.{e}.{projection}.wire" for e in range(scheme["experts"]) for projection in ("gate_proj", "up_proj", "down_proj")] if routed else [name + ".wire_bytes"]
            paths = sorted({str(root / index[tensor]) for tensor in names})
            if not all(Path(path).is_file() for path in paths):
                raise ValueError(f"missing wire shard for {root}: {name}")
            result.append({"root": str(root), "name": name, "contract_family": FAMILY[scheme["family"]],
                "family": scheme["family"], "structure": "routed_moe" if routed else "dense",
                "layer": int(name.split("layers.")[1].split(".")[0]), "q256": source["q256"],
                "rows": source["rows"], "cols": source["columns"], "roles": source["roles"],
                "experts": scheme.get("experts", 1), "wire_sources": paths,
                "paths_exist": True, "wire_stamp": None, "runs_table": None})
    return result


def arm(name, modules, library, decoder, *, routed=False, selector=None):
    symbol = ("tessera.routed_fused_e2m1.FusedRoutedE2M1MoE.__call__" if library == "e2m1" else
              "tessera.routed_fused.FusedRoutedWindowMoE.__call__") if routed else "tessera.routed_fused_e2m1.dense_forward_quantized" if library == "e2m1" else "tessera::window_gemm_dense" if library == "triton" else "tessera::fused_window_dense"
    return {"schema": "tessera.numerics_population.v1", "arm": name, "sms": None,
            "reviewed": False, "wire_stamps_read": False,
            "modules": [{**entry, "library": library, "pair": [symbol, decoder], "selector_env": selector or {}}
                        for entry in modules]}


def draft(out):
    dense = entries(BASE / "t8-coverage-20260930/stubs/stub-T8D1", family="TESSERA_FP8", structure="dense")
    dense += entries(STUB, family="TESSERA_FP8", structure="dense")
    template, = [entry for entry in dense if entry["root"] == str(STUB) and entry["name"] == "model.language_model.layers.5.mlp.shared_experts.down_proj"]
    for rung in (864, 928):
        name = "model.language_model.layers.7.mlp.shared_experts.down_proj"
        wire = CACHE / (name.replace(".", "__") + f"__TESSERA_E4M3_K1_R{rung}.tessera")
        if not wire.is_file():
            raise ValueError(f"missing canonical wire: {wire}")
        dense.append({**template, "name": name, "scheme_template": template["name"], "layer": 7, "q256": rung,
                      "wire_path": str(wire), "wire_sources": [str(wire)]})
    routed = entries(BASE / "ig1080-base-export-20261009-r2", structure="routed_moe")
    routed += entries(STUB, family="TESSERA_FP8", structure="routed_moe")
    for layer, rung in ((3, 832), (3, 960), (4, 864), (3, 944)):
        template, = [entry for entry in routed if entry["root"] == str(STUB) and entry["layer"] == layer and "wire_root" not in entry]
        paths = [str(CACHE / (f"model__language_model__layers__{layer}__mlp__experts__{e}__{projection}__TESSERA_E4M3_K1_R{rung}.tessera"))
                 for e in range(288) for projection in ("gate_proj", "up_proj", "down_proj")]
        if not all(Path(path).is_file() for path in paths):
            raise ValueError(f"missing canonical routed wires at layer {layer}, R{rung}")
        routed.append({**template, "q256": rung, "wire_root": str(CACHE),
                       "wire_family": {"payload": "TESSERA_E4M3_K1", "rung": f"R{rung}", "family": "TESSERA_FP8"}, "wire_sources": paths})
    populations = [arm("N-E4M3-R", routed, "e4m3mma", "native_routed_window_classes_e4m3mma", routed=True)]
    for library, suffix, decoder, selector in (("e4m3mma", "mma", "native_fused_window_dense_e4m3mma", {}),
        ("e4m3", "f16", "native_fused_window_dense", {"TESSERA_FUSED_E4M3_MMA": "f16"}),
        ("triton", "tri", "native_window_gemm", {"TESSERA_DENSE_FUSED": "0"})):
        populations.append(arm("N-E4M3-D-" + suffix, dense, library, decoder, selector=selector))
    for name, root, library, dense_decoder, routed_decoder in (("N-E2M1", OPTION_C, "e2m1", "native_fused_window_dense_e2m1", "native_routed_fused_window_e2m1"),
        ("N-BF16", BF16, "value", "native_fused_window_dense_bf16", "native_routed_window_classes_bf16")):
        population = arm(name, entries(root, structure="dense"), library, dense_decoder)
        population["modules"] += arm(name, entries(root, structure="routed_moe"), library, routed_decoder, routed=True)["modules"]
        populations.append(population)
    out.mkdir(parents=True, exist_ok=True)
    for population in populations:
        write_json(out / (population["arm"] + ".json"), population)


def inspect(population):
    import torch
    from experiments.numerics_rows.cases import checkpoint_routed, projection_scheme, rate_layout, wire_stamp
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    torch.set_num_threads(1)
    retained, tables = [], {}
    population["duplicate_run_tables"] = []
    for entry in population["modules"]:
        store = store_for(entry)
        records = []
        if entry["structure"] == "routed_moe":
            scheme, blobs, _ = checkpoint_routed(store, entry)
            wires = [(expert, projection, wire, projection_scheme(scheme, projection, wire))
                     for projection, items in blobs.items() for expert, wire in enumerate(items)]
        else:
            wires = [(0, "dense", store.wire(entry["name"]).numpy().tobytes(), store.schemes[entry["name"]])]
        for expert, projection, wire, declaration in wires:
            for role, parsed in parse_tessera_blob_for_scheme(wire, declaration, entry["name"], device="cpu"):
                stamp = wire_stamp(parsed)
                if stamp["q256"] != entry["q256"]:
                    raise ValueError(f"wire rung differs from population: {entry['name']}, {expert}, {role}")
                records.append({"expert": expert, "role": role, "projection": projection,
                    "wire_sha256": hashlib.sha256(wire).hexdigest(), "wire_stamp": stamp,
                    **rate_layout(parsed.unit.rates)})
        entry["wire_stamp"] = records
        entry["runs_table"] = [record["runs_table"] for record in records]
        if entry["root"] == str(BASE / "t8-coverage-20260930/stubs/stub-T8D1"):
            key = tuple(sorted({tuple(tuple(run) for run in record["runs_table"]) for record in records}))
            if key in tables:
                population["duplicate_run_tables"].append({"name": entry["name"], "covered_by": tables[key]})
                continue
            tables[key] = entry["name"]
        retained.append(entry)
    population["modules"] = retained
    population["wire_stamps_read"] = True
    return population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("draft", "inspect"))
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.mode == "draft":
        draft(args.out)
        return
    check_source(args.source)
    sys.path.insert(0, str(args.source / "src"))
    args.out.mkdir(parents=True, exist_ok=True)
    errors = []
    for path in sorted(args.population_dir.glob("*.json")):
        population = json.loads(path.read_text())
        try:
            inspect(population)
        except Exception as error:
            errors.append({"arm": population["arm"], "error": str(error), "traceback": traceback.format_exc()})
        write_json(args.out / path.name, population)
        write_json(args.out / "errors.json", errors)
    raise SystemExit(3 if errors else 0)


if __name__ == "__main__":
    main()
