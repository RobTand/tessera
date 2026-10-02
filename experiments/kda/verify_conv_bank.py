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
    # Admission of a composed evidence packet does not rewrite the v1 ending.
    # Device text is checked above; the current helper rechecks the raw control
    # and current source/module/SASS identities. All old output/state witnesses
    # remain actual recorded observations.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mhc"))
    import mhc_probe as probe

    bank_path, control_path = map(Path, sys.argv[2:4])
    bank_record = json.loads(bank_path.read_text())
    control_record = json.loads(control_path.read_text())
    bank = bank_record["kdaptx"]
    stock = bank["source_bindings"]
    stock_binding_ok = (hashlib.sha256(Path(stock["stock_conv_file"]).read_bytes()).hexdigest()
                        == stock["stock_conv_sha256"])
    composed = dict(bank, gate_contract=probe.KDA_GATE_CONTRACT,
                    ex2_equivalence=control_record["kdaex2"])
    errors = probe.kdaptx_gate_errors(composed)
    if not result["all_banked_conv_text_equal"] or not stock_binding_ok:
        errors.append("banked convolution device text or installed stock source differs")
    result.update({"gate_contract": probe.KDA_GATE_CONTRACT, "gate_passed": not errors, "errors": errors,
                   "stock_binding_ok": stock_binding_ok, "numerical_bank": {
                       "path": str(bank_path), "sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                       "action_key": bank_record["meta"]["pb_action"], "cases": len(bank["cases"])},
                   "intermediate_control": {"path": str(control_path),
                       "sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
                       "action_key": control_record["meta"]["pb_action"]}})
    (out / "kernel_identity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
