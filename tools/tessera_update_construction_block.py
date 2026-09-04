#!/usr/bin/env python3
"""Regenerate ``runtime_contract.json``'s ``construction`` block from the receipts.

The block is DERIVED, never hand-edited: every row is
``contract.construction_entry_from_receipt`` applied to one receipt under
``docs/measurements/construction/``, and ``tests/test_serving_construction.py``
re-derives the block and refuses any drift.  When a census is re-taken (a new
image, a new architecture), drop the receipt in that directory and run this;
the test is what proves the two agree.

Idempotent: running it twice writes the same bytes, and it does not touch the
changelog unless ``--changelog`` is given with an entry to prepend.

    tools/tessera_update_construction_block.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tessera.serving.contract import (  # noqa: E402
    CONSTRUCTION_SCHEMA, construction_entry_from_receipt)

RECEIPTS = ROOT / "docs/measurements/construction"
CONTRACT = ROOT / "src/tessera/serving/runtime_contract.json"

NOTE = (
    "Which Linears the pinned runtime OFFERS to a quant config, per architecture. "
    "LinearBase.__init__ takes UnquantizedLinearMethod() in the quant_config is None "
    "branch WITHOUT calling get_quant_method, so a model that builds a projection with "
    "quant_config=None takes vLLM's own BF16 method and this plugin is never asked -- it "
    "cannot refuse, warn, or see the prefix. A wire written there deletes the "
    "<module>.weight the runtime wants and puts bytes in its place that nothing decodes. "
    "offered/never_offered are vLLM MODULE patterns with repeat indices collapsed to '*'; "
    "a producer translates its checkpoint names with hf_to_vllm_mapper_unstacked first "
    "(the same table configure_quant_config hands this plugin) and normalises the same "
    "way. A name in neither list is a module the runtime does not build. Every row is "
    "DERIVED from a receipt under docs/measurements/construction/ by "
    "contract.construction_entry_from_receipt and re-derived in "
    "tests/test_serving_construction.py, so the table cannot drift from the observation; "
    "the observation itself is tools/tessera_construction_census.py, which builds the "
    "model the way the loader does and records every prefix a probe quant config is "
    "offered. An architecture with no row is UNCENSUSED, which is an honest gap and not a "
    "claim that everything is routed.")


def build_block() -> dict:
    entries = []
    for path in sorted(RECEIPTS.glob("*.json")):
        entry = construction_entry_from_receipt(json.loads(path.read_text()))
        entry["receipt"] = str(path.relative_to(ROOT))
        entries.append(entry)
    entries.sort(key=lambda e: e["architecture"])
    return {"schema": CONSTRUCTION_SCHEMA, "note": NOTE, "architectures": entries}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed block is not what the receipts derive")
    args = ap.parse_args()

    contract = json.loads(CONTRACT.read_text())
    block = build_block()
    if args.check:
        if contract.get("construction") == block:
            print("construction block matches the receipts")
            return 0
        print("construction block DRIFTED from docs/measurements/construction/; "
              "run tools/tessera_update_construction_block.py", file=sys.stderr)
        return 1
    if contract.get("construction") == block:
        print("construction block already current; nothing written")
        return 0
    contract["construction"] = block
    CONTRACT.write_text(json.dumps(contract, indent=1) + "\n")
    print("wrote", CONTRACT, "architectures",
          [e["architecture"] for e in block["architectures"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
