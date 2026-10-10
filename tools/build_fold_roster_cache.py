#!/usr/bin/env python3
"""Build the audit roster and digest cache for the tessera#1204 fold arm.

Reads the actual export bytes of the nominated artifact, hashes every
file, and writes two derived records into the run dir:
roster.json (comparison_arm_identity audit roster) and
candidate-digest-cache.sparklina.json (prismaquant digest-cache schema
for the TR3 scorer). Both describe the same bytes; neither confers
quality, admission or serve authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ARTIFACT = Path("/mnt/shared/tessera-runs/moe/glm53-x-picks-full-6be66bc/exported")
CHUNK = 1 << 26


def sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(CHUNK)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    artifact = args.artifact
    names = sorted(p.name for p in artifact.iterdir() if p.is_file())
    assert "config.json" in names and "tessera_serving_manifest.json" in names
    assert "model.safetensors.index.json" in names
    try:
        from prismabuild.progress import commit as pb_commit
    except ImportError:
        pb_commit = None
    files = []
    entries = []
    done = 0
    for name in names:
        path = artifact / name
        sha, size = sha_file(path)
        files.append({"name": name, "bytes": size, "sha256": sha})
        stat = path.stat()
        entries.append({"fingerprint": {
            "ctime_ns": stat.st_ctime_ns, "device": stat.st_dev,
            "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
            "path": str(path), "size": size}, "sha256": sha})
        done += 1
        if pb_commit is not None:
            pb_commit(1, "hash")
        print(f"hashed {done}/{len(names)} {name}", flush=True)
    by_name = {row["name"]: row for row in files}
    manifest_raw = (artifact / "tessera_serving_manifest.json").read_bytes()
    assert (len(manifest_raw) == by_name["tessera_serving_manifest.json"]["bytes"]
            and hashlib.sha256(manifest_raw).hexdigest()
            == by_name["tessera_serving_manifest.json"]["sha256"])
    manifest = json.loads(manifest_raw)
    index = json.loads((artifact / "model.safetensors.index.json").read_bytes())
    weights = {n: r for n, r in by_name.items() if n.endswith(".safetensors")}
    assert set(index["weight_map"].values()) == set(weights)
    weight_bytes = sum(r["bytes"] for r in weights.values())
    total = sum(r["bytes"] for r in files)
    assert weight_bytes == manifest["totals"]["checkpoint_bytes"], "shard/index byte mismatch"
    assert total != 0
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    roster = {"artifact": str(artifact), "files": files,
              "manifest_sha256": by_name["tessera_serving_manifest.json"]["sha256"],
              "all_files_bytes": total}
    (out / "roster.json").write_text(json.dumps(roster, indent=2) + "\n")
    cache = {"schema": "prismaquant.source_checkpoint.digest_cache.v1",
             "artifact": str(artifact), "entries": entries}
    (out / "candidate-digest-cache.sparklina.json").write_text(json.dumps(cache) + "\n")
    summary = {"files": len(files), "all_files_bytes": total,
               "weight_bytes": weight_bytes,
               "export_commit": manifest.get("git"),
               "contract_version": manifest.get("serving_gate", {}).get("contract_version"),
               "roster_sha256": hashlib.sha256(
                   (out / "roster.json").read_bytes()).hexdigest(),
               "cache_sha256": hashlib.sha256(
                   (out / "candidate-digest-cache.sparklina.json").read_bytes()).hexdigest()}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
