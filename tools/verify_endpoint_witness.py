#!/usr/bin/env python3
"""Verify a task endpoint runtime witness without Tessera serving imports (tessera#1056).

The D50 task adapter is a consumer: it must not import serving modules or
manage rank lifecycles. This tool is the standalone verifier the receipt
publishes beside itself. It reads one witness JSON file, re-derives the
join through ``tessera.endpoint_witness`` alone, and refuses missing,
incomplete or inconsistent evidence by name.

Exit 0: the witness binds its runtime join. Exit 4: the witness is refused
and the reason is on stderr. Any other exit is a tool failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tessera import endpoint_witness as ew  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("witness", help="the witness JSON file to verify")
    ap.add_argument("--expect-ranks", default=None,
                    help="comma-separated rank ids the witness must cover, e.g. 0,1")
    args = ap.parse_args(argv)
    try:
        witness = json.loads(Path(args.witness).read_text())
    except (OSError, ValueError) as exc:
        print(f"REFUSED: cannot read witness: {exc}", file=sys.stderr)
        return 4
    ranks = None
    if args.expect_ranks is not None:
        try:
            ranks = [int(part) for part in args.expect_ranks.split(",") if part != ""]
        except ValueError:
            print(f"REFUSED: --expect-ranks is not a rank list: {args.expect_ranks!r}",
                  file=sys.stderr)
            return 4
    reason = ew.verify_witness(witness, ranks=ranks)
    if reason is not None:
        print(f"REFUSED: {reason}", file=sys.stderr)
        return 4
    print(f"witness ok: endpoint {witness['listener']['endpoint']} "
          f"alias {witness['listener']['served_alias']} "
          f"attempt {witness['launch']['attempt_id']} "
          f"ranks {witness['launch']['ranks']} "
          f"fingerprint {witness['fingerprint'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
