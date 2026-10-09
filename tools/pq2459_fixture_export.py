#!/usr/bin/env python3
"""Export additive fixtures through the existing serving exporter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# The A/B fixture union lacks these family/rung/shape combinations.
# Existing routed units stay on their original source and rung.
DENSE = {
    "0.mlp.gate_proj": ("BF16", 880),
    "0.mlp.up_proj": ("BF16", 880),
    "0.mlp.down_proj": ("BF16", 880),
    "1.mlp.gate_proj": ("BF16", 1088),
    "1.mlp.up_proj": ("BF16", 1088),
    "1.mlp.down_proj": ("BF16", 1024),
    "2.mlp.gate_proj": ("E4M3", 1024),
    "2.mlp.up_proj": ("E4M3", 1024),
    "2.mlp.down_proj": ("E4M3", 1088),
    "3.mlp.shared_experts.gate_proj": ("BF16", 880),
    "3.mlp.shared_experts.up_proj": ("BF16", 880),
}


def dense_only_supplement(original):
    plan = {name: "PASSTHROUGH" for name in original}
    for suffix, (grid, q256) in DENSE.items():
        name = f"model.language_model.layers.{suffix}.weight"
        if name not in plan:
            raise ValueError(f"source plan lacks {name}")
        plan[name] = {"grid": grid, "q256": q256}
    return plan


def supplemental_plan(original):
    plan = {name: value if name.endswith(".experts") else "PASSTHROUGH"
            for name, value in original.items()}
    for suffix, (grid, q256) in DENSE.items():
        name = f"model.language_model.layers.{suffix}.weight"
        if name not in plan:
            raise ValueError(f"source plan lacks {name}")
        plan[name] = {"grid": grid, "q256": q256}
    return plan


def scoped_reuse_inputs(args, original, *, publish):
    from tessera.export_serving import ROUTED_EXPERT_2D, grid_for
    from tessera.serving.contract import PAYLOAD_FAMILY_BY_ROUTE
    from tessera.serving_plan import family_for
    scope = json.loads((Path(__file__).resolve().parents[1] /
                        "experiments/pq2459/scope.json").read_bytes())
    reference_plan = json.loads(args.routed_reference_plan.read_bytes())
    plan = dict(original)
    plan.update({name: spec for name, spec in reference_plan.items() if name.endswith(".experts")})
    for name, spec in list(plan.items()):
        if not isinstance(spec, dict):
            continue
        family = PAYLOAD_FAMILY_BY_ROUTE[family_for(grid_for(spec["grid"]))]
        structure = "routed_moe" if name.endswith(".experts") else "dense"
        if family not in scope["rungs"] or spec["q256"] not in scope["rungs"][family][structure]:
            plan[name] = "PASSTHROUGH"
    if not publish:
        return plan, None
    primary = json.loads(args.cached_all.read_bytes())
    routed = json.loads(args.routed_reference_cache.read_bytes())
    if primary["schema"] != "tessera.cached_units.v1" or routed["schema"] != primary["schema"]:
        raise ValueError("the fixture inputs require the original v1 unit receipts")
    if primary["source"] != routed["source"]:
        raise ValueError("the dense and routed inputs name different source bytes")
    selected = {}
    for document, path, expert_only in ((primary, args.cached_all, False),
                                        (routed, args.routed_reference_cache, True)):
        for name, record in document["units"].items():
            match = ROUTED_EXPERT_2D.fullmatch(name + ".weight")
            if bool(match) != expert_only:
                continue
            owner = match.group("moe") + ".experts" if match else name + ".weight"
            if not isinstance(plan.get(owner), dict):
                continue
            selected[name] = (record, path.parent / record["file"])
    cache_dir = args.plan_output.with_suffix(".cache")
    cache_dir.mkdir(parents=True, exist_ok=False)
    for record, source in selected.values():
        leaf = record["file"]
        if Path(leaf).name != leaf or source.is_symlink():
            raise ValueError("cached input must name a regular leaf file")
        destination = cache_dir / leaf
        if not destination.exists():
            os.link(source, destination)
    cache_path = cache_dir / "cached_units.v1.json"
    cache_path.write_text(json.dumps({"schema": primary["schema"], "source": primary["source"],
                                     "units": {name: record for name, (record, _source) in selected.items()}},
                                    indent=2) + "\n")
    return plan, cache_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original-plan", type=Path, required=True)
    cache = parser.add_mutually_exclusive_group(required=False)
    cache.add_argument("--cached-routed", type=Path)
    cache.add_argument("--cached-all", type=Path)
    parser.add_argument("--dense-only", action="store_true",
                        help="fresh-encode only the eleven supplement dense units; no routed intake")
    parser.add_argument("--routed-reference-plan", type=Path)
    parser.add_argument("--routed-reference-cache", type=Path)
    parser.add_argument("--producer-package", type=Path, required=True)
    parser.add_argument("--producer-source-sha256", required=True)
    parser.add_argument("--producer-authority", type=Path, required=True)
    parser.add_argument("--hessian", type=Path, required=True)
    parser.add_argument("--source-digest-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--intake-threads", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight", action="store_true")
    from tools.pq2459_source import add_source_arguments, activate_source
    add_source_arguments(parser)
    args = parser.parse_args(argv)
    source_info = activate_source(args.source_archive, args.source_archive_sha256)
    if args.output.exists():
        parser.error("output already exists; use a fresh additive directory")
    original = json.loads(args.original_plan.read_bytes())
    if args.dense_only and (args.cached_all is not None or args.cached_routed is not None
                            or args.routed_reference_plan is not None or args.routed_reference_cache is not None):
        parser.error("--dense-only takes no cached or routed reference inputs")
    if args.dense_only:
        plan = dense_only_supplement(original)
        cached_all = None
    if bool(args.routed_reference_plan) != bool(args.routed_reference_cache):
        parser.error("routed reference plan and cache must be supplied together")
    cached_all = args.cached_all
    if args.routed_reference_plan is not None:
        if args.cached_all is None:
            parser.error("routed reference inputs require --cached-all")
        plan, prepared = scoped_reuse_inputs(args, original, publish=not args.preflight)
        if prepared is not None:
            cached_all = prepared
    if args.preflight:
        from tessera.export_serving import quantizable, load_producer_authority, check_recipe, grid_for
        from tessera.serving_plan import validate_serving_plan
        validate_serving_plan(plan)
        if not args.dense_only:
            load_producer_authority(args.producer_authority.resolve())
        for target, spec in plan.items():
            if isinstance(spec, dict):
                check_recipe(grid_for(spec["grid"]), spec["q256"], target,
                             structure="routed_moe" if target.endswith(".experts") else "dense")
        shards, shapes, _, routed = quantizable(args.source)
        selected = {name for name, value in plan.items()
                    if isinstance(value, dict) and name.endswith(".weight")}
        if not selected <= shapes.keys():
            raise ValueError("source lacks a selected dense tensor")
        print(json.dumps({"dense_units": len(selected), "source_shards": len(shards),
                          "routed_source_units": len(routed),
                          "source": source_info, "qualified_cells": 0}, sort_keys=True))
        return 0
    if args.plan_output.exists():
        parser.error("plan output already exists; retain the previous input")
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, indent=2) + "\n")
    args.source_digest_cache.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "tessera.export_serving", str(args.source),
               str(args.output), "--device", args.device, "--plan-json", str(args.plan_output)]
    if not args.dense_only:
        command += ["--cached-units" if cached_all is not None else "--cached-expert-units",
                    str(cached_all if cached_all is not None else args.cached_routed),
                    "--cached-producer-package", str(args.producer_package),
                    "--cached-producer-source-sha256", args.producer_source_sha256,
                    "--producer-authority", str(args.producer_authority.resolve()),
                    "--hessian", str(args.hessian), "--cached-hessian-identity", "committed",
                    "--cached-intake-threads", str(args.intake_threads)]
    else:
        command += ["--producer-authority", str(args.producer_authority.resolve()),
                    "--hessian", str(args.hessian)]
    command += ["--source-digest-cache", str(args.source_digest_cache),
                "--fit-tp-size", "2"]
    if args.device == "cuda":
        command[1:3] = [str(ROOT / "tools/pq2459_export_progress.py")]
    print(json.dumps({"source": source_info, "qualified_cells": 0}, sort_keys=True), flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
