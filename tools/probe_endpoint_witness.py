#!/usr/bin/env python3
"""Probe one live listener and publish its endpoint runtime witness (tessera#1056).

The serving producer boundary. The serving workers observe themselves
through the serving-owned runtime module: rank identity from the live
distributed group, model path and served names from the worker's own
model config, loaded wire digests from resident module state, and the
vocabulary length from the engine's initialized tokenizer. Each worker
writes one JSON observation file. This probe reads those files, reads the
served alias from the listener's live ``/v1/models`` reply while the
listener still answers, joins them through ``tessera.endpoint_witness``,
proves the bytes against the served directory, and publishes one
self-contained JSON receipt with its sha256 sidecar.

The probe itself needs no Tessera serving import, no torch, and no vLLM:
it reads worker observation files, HTTP replies and file bytes with the
standard library. The runtime facts come from inside the serve, never
from the invocation. Exit 0 publishes one receipt. Exit 4 refuses the
observation by name. Any other exit is a tool failure.
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
    ap.add_argument("--worker-observations", required=True,
                    help="comma-separated worker observation JSON files, one per serving rank")
    ap.add_argument("--artifact-dir", required=True, help="served artifact directory to prove bytes against")
    ap.add_argument("--lifetime-id", required=True, help="observation lifetime all reads share")
    ap.add_argument("--publish-root", required=True, help="directory that receives the receipt")
    args = ap.parse_args(argv)
    try:
        workers = eo.collect_worker_observations(args.worker_observations.split(","),
                                                 lifetime_id=args.lifetime_id)
        listener = eo.observe_listener(args.base_url, lifetime_id=args.lifetime_id)
        eo.check_listener_owned(listener, workers)
        launch = eo.observe_launch(args.attempt_id, workers, lifetime_id=args.lifetime_id)
        artifacts = eo.observe_rank_bytes(workers, args.artifact_dir,
                                          lifetime_id=args.lifetime_id)
        tokenizer = eo.observe_server_tokenizer(workers, args.artifact_dir,
                                                lifetime_id=args.lifetime_id)
        witness = ew.build_witness(listener=listener, launch=launch,
                                   artifacts=artifacts, tokenizer=tokenizer)
        stamped = ew.stamp_byte_proof(witness, args.artifact_dir)
        receipt = eo.publish_witness(args.publish_root, witness=stamped,
                                     served_dir=args.artifact_dir)
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
