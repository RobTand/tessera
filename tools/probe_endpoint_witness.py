#!/usr/bin/env python3
"""Probe one live listener and publish its endpoint runtime witness (tessera#1056).

The serving producer boundary: read the served alias from the listener's
``/v1/models`` reply, the launch attempt and rank set from the invocation,
each rank's loaded bytes from its served directory, and the server
tokenizer facts from the served artifact. Join them through
``tessera.endpoint_witness``, prove them against the served directory, and
publish one self-contained JSON receipt with its sha256 sidecar.

This probe needs no Tessera serving import, no torch, and no vLLM: it reads
HTTP replies and file bytes with the standard library. Exit 0 publishes one
receipt. Exit 4 refuses the observation by name. Any other exit is a tool
failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tessera import endpoint_observer as eo  # noqa: E402
from tessera import endpoint_witness as ew  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True, help="listener address, e.g. http://10.100.96.2:8142")
    ap.add_argument("--attempt-id", required=True, help="launch attempt identity")
    ap.add_argument("--ranks", required=True, help="comma-separated rank ids, e.g. 0,1")
    ap.add_argument("--artifact-dir", required=True, help="served artifact directory each rank loaded")
    ap.add_argument("--rank-dirs", default=None,
                    help="per-rank served directories as rank=dir pairs, e.g. 0=/a,1=/b")
    ap.add_argument("--lifetime-id", required=True, help="observation lifetime all reads share")
    ap.add_argument("--publish-root", required=True, help="directory that receives the receipt")
    ap.add_argument("--rank-observations", default=None,
                    help="comma-separated rank observation JSON files; with it, --artifact-dir only proves bytes")
    args = ap.parse_args(argv)
    try:
        ranks = [int(part) for part in args.ranks.split(",") if part != ""]
        if not ranks or len(set(ranks)) != len(ranks):
            raise ValueError("empty or repeated")
    except ValueError:
        print(f"REFUSED: --ranks is not a rank list: {args.ranks!r}", file=sys.stderr)
        return 4
    try:
        listener = eo.observe_listener(args.base_url, lifetime_id=args.lifetime_id)
        launch = eo.observe_launch(args.attempt_id, ranks, lifetime_id=args.lifetime_id)
        if args.rank_observations is not None:
            artifacts = eo.collect_rank_observations(args.rank_observations.split(","))
            got = sorted(item["rank"] for item in artifacts)
            if got != sorted(ranks):
                raise ValueError(f"rank observations {got} do not cover launch ranks {sorted(ranks)}")
        elif args.rank_dirs is not None:
            mapping = {}
            for pair in args.rank_dirs.split(","):
                rank_text, _, directory = pair.partition("=")
                mapping[int(rank_text)] = directory
            if sorted(mapping) != sorted(ranks):
                raise ValueError(f"rank directories {sorted(mapping)} do not cover ranks {sorted(ranks)}")
            artifacts = [eo.observe_rank_bytes(rank, mapping[rank], lifetime_id=args.lifetime_id)
                         for rank in ranks]
        else:
            artifacts = [eo.observe_rank_bytes(rank, args.artifact_dir, lifetime_id=args.lifetime_id)
                         for rank in ranks]
        tokenizer = eo.observe_server_tokenizer(args.artifact_dir, lifetime_id=args.lifetime_id)
        witness = ew.build_witness(listener=listener, launch=launch,
                                   artifacts=artifacts, tokenizer=tokenizer)
        stamped = ew.stamp_byte_proof(witness, args.artifact_dir)
        receipt = eo.publish_witness(args.publish_root, witness=stamped)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4
    print(f"witness ok: endpoint {stamped['listener']['endpoint']} "
          f"alias {stamped['listener']['served_alias']} "
          f"attempt {stamped['launch']['attempt_id']} "
          f"ranks {stamped['launch']['ranks']} "
          f"receipt {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
