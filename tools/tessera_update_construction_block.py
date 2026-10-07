#!/usr/bin/env python3
"""Derive the active construction block from its named census receipts.

The contract names one active receipt per architecture. Historical receipts
remain unchanged. Use --receipt to replace one active architecture.

    tools/tessera_update_construction_block.py --receipt <new receipt>
    tools/tessera_update_construction_block.py --check
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

CONTRACT = ROOT / "src/tessera/serving/runtime_contract.json"

NOTE = (
    "The active receipts record which modules reach a quant config. "
    "An explicit Tessera target can use the selective constructor override. "
    "Unselected modules retain their stock method. "
    "The offered and never_offered lists use normalized runtime module names. "
    "The producer applies hf_to_vllm_mapper_unstacked before the lookup. "
    "Each row derives from its named active receipt. Historical receipts remain unchanged. "
    "Construction proves reachability, not weight loading or a forward result.")


def build_block(contract, replacements=()) -> dict:
    entries = {}
    paths = [ROOT / entry["receipt"] for entry in contract["construction"]["architectures"]]
    for path in [*paths, *replacements]:
        entry = construction_entry_from_receipt(json.loads(path.read_text()))
        entry["receipt"] = str(path.relative_to(ROOT))
        entries[entry["architecture"]] = entry
    return {"schema": CONSTRUCTION_SCHEMA, "note": NOTE,
            "architectures": sorted(entries.values(), key=lambda entry: entry["architecture"])}


def splice(raw: str, block: dict) -> str:
    """The block, written into ``raw`` without reformatting anything else.

    ``runtime_contract.json`` is hand-formatted and several branches edit it at
    once; a whole-file ``json.dumps`` would reformat hundreds of lines it does
    not change and turn every concurrent edit into a conflict.  So the block is
    rendered at the indent its siblings use and spliced in place.
    """
    lines = raw.splitlines(keepends=True)
    anchor = next((i for i, l in enumerate(lines)
                   if l.lstrip().startswith('"changelog"')), None)
    if anchor is None:
        raise SystemExit("runtime_contract.json has no changelog key to anchor on")
    at = next((i for i, l in enumerate(lines)
               if l.lstrip().startswith('"construction"')), None)
    if at is None:
        at = end = anchor                  # first write: insert before changelog
    else:
        depth = 0
        end = None
        for i in range(at, len(lines)):
            depth += lines[i].count("{") - lines[i].count("}")
            if depth == 0:
                end = i + 1      # also the one-line case, where depth never rises
                break
        if end is None:
            raise SystemExit("runtime_contract.json's construction block does not close")
    pad = lines[anchor][: len(lines[anchor]) - len(lines[anchor].lstrip())]
    n = len(pad) or 1
    inner = json.dumps({"construction": block}, indent=n).splitlines()[1:-1]
    rendered = [pad + l[n:] + "\n" for l in inner]
    rendered[-1] = rendered[-1].rstrip("\n") + ",\n"
    return "".join(lines[:at] + rendered + lines[end:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, action="append", default=[],
                    help="replace the active receipt for its measured architecture")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed block is not what the receipts derive")
    args = ap.parse_args()

    raw = CONTRACT.read_text()
    contract = json.loads(raw)
    block = build_block(contract, [path.resolve() for path in args.receipt])
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
    out = splice(raw, block)
    assert json.loads(out)["construction"] == block, "the splice did not produce the block"
    CONTRACT.write_text(out)
    print("wrote", CONTRACT, "architectures",
          [e["architecture"] for e in block["architectures"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
