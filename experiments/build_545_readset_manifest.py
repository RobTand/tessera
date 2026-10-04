#!/usr/bin/env python3
"""Build the PrismaBuild data manifest (readset) for one 545 A/B half.

usage: build_545_readset_manifest.py h1|h2 OUT.json

The manifest names, under /mnt/shared, every byte the half's action reads:
the fixed artifact files and the shared wikitext corpus.  Entry digests stay
null here (hashing 22 GiB of stub shards twice buys nothing); the driver
records the sha256 of every file it actually reads beside each result, which
is where the readset proof lives.
"""
import json
import sys
from pathlib import Path

HALVES = {
    "h1": [
        "/mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024",
        "/mnt/shared/tessera-runs/bf16/qwen0.6b-bf16-r7-plugin",
        "/mnt/shared/tessera-kl/wikitext_test.txt",
    ],
    "h2": [
        "/mnt/shared/tessera-runs/moe/u1-stubs-20260926/stub-D",
        "/mnt/shared/tessera-kl/wikitext_test.txt",
    ],
}


def main() -> int:
    half, dest = sys.argv[1], Path(sys.argv[2])
    roots = HALVES[half]
    entries = []
    for root in roots:
        p = Path(root)
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if not f.is_file():
                continue
            entries.append({
                "path": str(f),
                "offset": 0,
                "bytes": f.stat().st_size,
                "sha256": None,
            })
    manifest = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {
            "tool": "experiments/build_545_readset_manifest.py",
            "issue": "RobTand/tessera#545 step-3 served A/B",
            "half": half,
        },
        "annotations": {},
        "mount_prefix": "/mnt/shared",
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
    }
    dest.write_text(json.dumps(manifest, indent=1))
    print(f"{half}: {len(entries)} entries, {manifest['total_bytes']} bytes -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
