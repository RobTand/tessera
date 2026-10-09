#!/usr/bin/env python3
"""Verify artifact bytes and the current listener without serving imports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tessera.endpoint_observer import fetch_witness  # noqa: E402
from tessera.endpoint_witness import verify_witness  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("witness")
    parser.add_argument("--served-dir", required=True)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--expect-ranks")
    args = parser.parse_args(argv)
    try:
        receipt = json.loads(Path(args.witness).read_bytes())
        endpoint = receipt["listener"]["endpoint"]
        ranks = None if args.expect_ranks is None else [int(p) for p in args.expect_ranks.split(",")]
        live = fetch_witness(endpoint)["receipt"]
        reason = verify_witness(receipt, ranks=ranks, served_dir=args.served_dir,
                                tokenizer_dir=args.tokenizer_dir, live=live)
        if reason is not None:
            raise ValueError(reason)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4
    print(f"runtime witness valid: {receipt['launch']['attempt_id']} ranks={receipt['launch']['ranks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
