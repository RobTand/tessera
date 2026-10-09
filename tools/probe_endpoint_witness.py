#!/usr/bin/env python3
"""Publish a receipt from the live serving producer; accept no detached observations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tessera.endpoint_observer import fetch_witness  # noqa: E402
from tessera.endpoint_witness import verify_witness  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--tokenizer-dir")
    args = parser.parse_args(argv)
    try:
        envelope = fetch_witness(args.base_url, publish=True)
        receipt = envelope["receipt"]
        live = fetch_witness(args.base_url)["receipt"]
        reason = verify_witness(receipt, served_dir=args.artifact_dir,
                                tokenizer_dir=args.tokenizer_dir, live=live)
        if reason is not None:
            raise ValueError(reason)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4
    print(envelope["public_receipt_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
