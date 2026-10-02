#!/usr/bin/env python3
"""Bind banked conv numerics to unchanged cubin text after adding a control kernel."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from cubin_cmp import sections


def main() -> int:
    out = Path(sys.argv[1])
    paths = [list((out / name).glob("*.cubin")) for name in ("old", "new")]
    if any(len(p) != 1 for p in paths):
        raise SystemExit(f"expected one extracted cubin per module: {paths}")
    old, new = [sections(p[0]) for p in paths]
    names = sorted(n for n in old["sections"] if n.startswith(".text.") and "kda_conv_ref" in n)
    rows = [{"section": n, "old": old["sections"][n], "new": new["sections"].get(n),
             "equal": old["sections"][n] == new["sections"].get(n)} for n in names]
    result = {"schema": "tessera.kda_conv_kernel_identity.v1", "kernels": rows,
              "all_banked_conv_text_equal": bool(rows) and all(r["equal"] for r in rows),
              "cubins": [{"path": str(p[0]), "sha256": d["sha256"]} for p, d in zip(paths, (old, new))]}
    for name in ("old", "new"):
        module = Path((out / f"{name}-module-path.txt").read_text().strip())
        result[f"{name}_module"] = {"path": str(module),
                                     "sha256": hashlib.sha256(module.read_bytes()).hexdigest()}
    (out / "kernel_identity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_banked_conv_text_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
