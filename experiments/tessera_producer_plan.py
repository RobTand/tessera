#!/usr/bin/env python
"""Emit the producer's explicit expert projection without encoding or serving.

PrismaQuant calls this named tool as a subprocess, keeping Tessera's serving
imports in the producer process and its source grammar in the exporter.
The source seal binds the checkpoint; the geometry reads its headers.
Optional stat-bound caching reuses fenced shard hashes and records their receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_tessera_serving import project_expert_plan, quantizable
from tessera.cached_unit import read_manifest
from tessera.serving_parts import source_identity
from tessera.source_digest_cache import SourceDigestCache


def _producer_projection(src: Path, stack_plan: Path, *, digest_cache=None) -> dict:
    """Assemble the producer-owned geometry and whole-source seal."""
    _shards, dense, packed, routed = quantizable(src)
    projected = project_expert_plan({**dense, **packed, **routed},
                                    json.loads((src / "config.json").read_text()),
                                    read_manifest(stack_plan))
    projected["source"] = source_identity(src, digest_cache=digest_cache)
    return projected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("--stack-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-digest-cache", type=Path,
                        help="existing trusted shard-digest directory; record reuse in output")
    args = parser.parse_args(argv)
    cache = (SourceDigestCache(args.source_digest_cache, source=args.src)
             if args.source_digest_cache is not None else None)
    projected = _producer_projection(args.src, args.stack_plan, digest_cache=cache)
    if cache is not None:
        projected["source_digest_cache"] = cache.receipt()
    args.out.write_text(json.dumps(projected, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
