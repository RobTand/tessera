#!/usr/bin/env python
"""Emit the producer's explicit expert projection without encoding or serving.

PrismaQuant calls this named tool as a subprocess, keeping Tessera's serving
imports in the producer process and its source grammar in the exporter.
The source seal hashes the actual checkpoint; the geometry reads its headers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_tessera_serving import project_expert_plan, quantizable
from tessera.cached_unit import read_manifest
from tessera.serving_parts import source_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("--stack-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _shards, dense, packed, routed = quantizable(args.src)
    projected = project_expert_plan({**dense, **packed, **routed},
                                    json.loads((args.src / "config.json").read_text()),
                                    read_manifest(args.stack_plan))
    projected["source"] = source_identity(args.src)
    args.out.write_text(json.dumps(projected, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
