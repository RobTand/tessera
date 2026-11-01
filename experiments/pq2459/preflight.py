#!/usr/bin/env python3
"""Exercise the fixture parsers and the sealed token inputs on CPU."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--arrays-root", type=Path, required=True)
    parser.add_argument("--scorer-bundle", type=Path)
    parser.add_argument("--scorer-bundle-sha256")
    from tools.pq2459_source import add_source_arguments, activate_source
    add_source_arguments(parser)
    args = parser.parse_args(argv)
    source_info = activate_source(args.source_archive, args.source_archive_sha256)
    from tools.pq2459_fixture_packet import artifact_wires
    from tools.pq2459_fixture_workloads import read_panel_inputs, speed_cases, engine_kwargs
    scope = json.loads((ROOT / "experiments/pq2459/scope.json").read_bytes())
    records = {}
    for entry in args.artifact:
        name, _, path = entry.partition("=")
        wires, excluded, manifest, inventory = artifact_wires(name, Path(path), scope)
        records[name] = {"parsed_wires": len(wires), "indexed_tensors": len(inventory),
                         "producer_observed": manifest["git"], "excluded": excluded,
                         "geometry": sorted({(wire["family"], wire["structure"], wire["q256"],
                                              tuple(wire["rank_local_nk"])) for wire in wires})}
    panel, inputs, receipts = read_panel_inputs(args.panel, args.arrays_root)
    dependencies = [importlib.import_module(name).__file__ for name in
                    ("numpy", "safetensors", "tessera.export_serving", "tessera.serving.scheme",
                     "tessera.compact_prep", "tessera.serving.sharding", "tessera.serving.topology")]
    plan = {name: {"required_token_rows": profile["token_rows"],
                   "engine_kwargs": engine_kwargs(scope, profile, Path(args.artifact[0].split("=", 1)[1]),
                                                   {"tensor_parallel_size": 2}),
                   "cases": speed_cases(profile) if name.startswith("speed_") else "source-owned TR3 scorer"}
            for name, profile in scope["profiles"].items()}
    if args.scorer_bundle is not None:
        import subprocess
        from tools.pq2459_scorer_source import prepare_scorer_source
        scorer = prepare_scorer_source(args.scorer_bundle, args.scorer_bundle_sha256)
        script = Path(scorer["root"]) / "experiments/measure_glm_tr3_vllm.py"
        subprocess.run([sys.executable, str(script), "--help"], check=True)
    print(json.dumps({"artifacts": records, "token_inputs": receipts,
                      "token_windows": len(inputs), "vocab_size": panel["vocab_size"],
                      "dependencies": dependencies, "workloads": plan,
                      "source": source_info, "qualified_cells": 0, "device": "cpu"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
