#!/usr/bin/env python
"""Point an existing Tessera-wire checkpoint at Tessera's own vLLM plugin.

The wires do not change when the runtime that reads them changes.  A checkpoint
written for Gridbook's Tessera lane carries ``quant_method: "gridbook"`` and
exactly the schemes ``tessera.serving.scheme`` validates, so retargeting it is a
config edit and nothing else.  This script proves that: the weight files are
HARDLINKED (one inode, byte-identical by construction -- there is no copy to get
wrong), and only ``config.json`` is rewritten.

That is what makes the served comparison meaningful.  The plugin's KL against
the Gridbook lane's dump is then a statement about two runtimes over ONE set of
bytes, not about two encodes.

usage::

    retarget_checkpoint_to_plugin.py <src-checkpoint> <out-checkpoint>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.serving.scheme import (  # noqa: E402
    STRUCTURE_DENSE, is_tessera_scheme, validate_tessera_scheme)

QUANT_METHOD = "tessera"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--copy", action="store_true",
                    help="copy the weight files instead of hardlinking them (use only across "
                         "filesystems; a copy is not the identity argument a hardlink is)")
    args = ap.parse_args()

    config = json.loads((args.src / "config.json").read_text())
    qc = config.get("quantization_config")
    if not isinstance(qc, dict) or not qc.get("config_groups"):
        raise SystemExit(f"{args.src}/config.json carries no quantization_config.config_groups")
    groups = {}
    for name, group in qc["config_groups"].items():
        scheme = dict(group.get("scheme") or {})
        if not is_tessera_scheme(scheme):
            raise SystemExit(f"config group {name!r} is not a Tessera scheme; this is not a "
                             "Tessera-wire checkpoint")
        # ``structure`` is optional on the wire (a checkpoint written before the
        # field existed is dense by construction) and stated here so the copy
        # exercises the field a routed-MoE export will use.
        scheme.setdefault("structure", STRUCTURE_DENSE)
        for target in group.get("targets", ()):
            validate_tessera_scheme(scheme, str(target))
        groups[name] = {**group, "scheme": scheme}
    config["quantization_config"] = {**qc, "quant_method": QUANT_METHOD, "config_groups": groups}

    args.out.mkdir(parents=True, exist_ok=True)
    linked, copied = [], []
    for entry in sorted(args.src.iterdir()):
        if not entry.is_file() or entry.name == "config.json":
            continue
        dst = args.out / entry.name
        if dst.exists():
            dst.unlink()
        if entry.suffix == ".safetensors" and not args.copy:
            os.link(entry, dst)
            linked.append(entry.name)
        else:
            shutil.copy2(entry, dst)
            copied.append(entry.name)
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    same = all((args.src / n).stat().st_ino == (args.out / n).stat().st_ino for n in linked)
    print(json.dumps({
        "src": str(args.src), "out": str(args.out),
        "quant_method": QUANT_METHOD, "config_groups": len(groups),
        "hardlinked": linked, "same_inode": same, "copied": copied,
        "ignore": qc.get("ignore"),
    }, indent=1))
    return 0 if (same or args.copy) else 1


if __name__ == "__main__":
    raise SystemExit(main())
