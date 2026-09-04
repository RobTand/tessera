#!/usr/bin/env python3
"""Write the ``--plan-json`` that puts a checkpoint's routed-MoE stacks on a rung.

The plan names STACKS -- ``<moe>.experts`` -- because vLLM builds one method
per ``RoutedExperts`` module and the sidecar declares one scheme for it.  It is
BUILT from the checkpoint rather than typed, so a layer that is not sparse
cannot appear in it and a hand-typed index cannot drift from the file.

    moe_stack_plan.py SRC OUT.json --grid E4M3 --q256 1024 [--layers N]

``--layers N`` keeps only stacks in decoder layers below N, matching the
exporter's own ``--layers`` smoke bound: the exporter REFUSES a planned stack
outside that bound rather than silently skipping it, so the two have to agree
and this is where they are made to.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _exporter():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        "export_tessera_serving", str(HERE / "export_tessera_serving.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--grid", default="E4M3")
    ap.add_argument("--q256", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=None)
    args = ap.parse_args()

    exporter = _exporter()
    _shards, _shapes, _packed, routed = exporter.quantizable(args.src)
    stacks = exporter.expert_stacks(routed)
    if not stacks:
        raise SystemExit(f"{args.src} carries no unpacked routed-expert leaves; there is no "
                         "stack to plan (a packed 3-D source has no export -- see the exporter)")
    plan = {}
    for stack in sorted(stacks):
        leaf = next(iter(stacks[stack].values()))["gate_proj"][0]
        if args.layers is not None and exporter.body_layer(leaf) >= args.layers:
            continue
        plan[stack] = {"grid": args.grid, "q256": args.q256}
    if not plan:
        raise SystemExit(f"--layers {args.layers} excludes every routed stack in {args.src}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=1) + "\n")
    units = sum(len(experts) * 3 for stack, experts in stacks.items() if stack in plan)
    print(f"plan: {len(plan)} stack(s), {units} units -> {args.out}")
    for stack in plan:
        print(f"  {stack}  ({len(stacks[stack])} experts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
