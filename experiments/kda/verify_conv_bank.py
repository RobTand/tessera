#!/usr/bin/env python3
"""Bind banked conv numerics to unchanged cubin text after adding a control kernel."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from cubin_cmp import sections


def main() -> int:
    out = Path(sys.argv[1])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mhc"))
    import mhc_probe as probe

    expected_old_module_sha = sys.argv[6]
    paths = [list((out / name).glob("*.cubin")) for name in ("old", "new")]
    if any(len(p) != 1 for p in paths):
        raise SystemExit(f"expected one extracted cubin per module: {paths}")
    old, new = [sections(p[0]) for p in paths]
    names = sorted(n for n in old["sections"] if n.startswith(".text.") and "kda_conv_ref" in n)
    new_names = sorted(n for n in new["sections"] if n.startswith(".text.") and "kda_conv_ref" in n)

    def modes(names):
        matches = [re.match(r"\.text\._Z12kda_conv_refILi(\d+)EE", n) for n in names]
        return [int(m[1]) if m else None for m in matches]

    expected_modes = set(probe.KDA_PTX_MODES)
    roster_ok = (len(names) == len(new_names) == len(expected_modes) and
                 set(modes(names)) == set(modes(new_names)) == expected_modes)
    rows = [{"section": n, "old": old["sections"][n], "new": new["sections"].get(n),
             "equal": old["sections"][n] == new["sections"].get(n)} for n in names]
    result = {"schema": "tessera.kda_conv_kernel_identity.v1", "kernels": rows,
              "expected_modes": sorted(expected_modes), "exact_mode_roster": roster_ok,
              "all_banked_conv_text_equal": roster_ok and all(r["equal"] for r in rows),
              "cubins": [{"path": str(p[0]), "sha256": d["sha256"]} for p, d in zip(paths, (old, new))]}
    for name in ("old", "new"):
        module = Path((out / f"{name}-module-path.txt").read_text().strip())
        result[f"{name}_module"] = {"path": str(module),
                                     "sha256": hashlib.sha256(module.read_bytes()).hexdigest()}
    # Admission of a composed evidence packet does not rewrite the v1 ending.
    # Device text is checked above; the current helper rechecks the raw control
    # and current source/module/SASS identities. All old output/state witnesses
    # remain actual recorded observations.
    bank_path, control_path = map(Path, sys.argv[2:4])
    expected_bank_sha, expected_control_sha = sys.argv[4:6]
    if (hashlib.sha256(bank_path.read_bytes()).hexdigest() != expected_bank_sha or
            hashlib.sha256(control_path.read_bytes()).hexdigest() != expected_control_sha):
        raise SystemExit("numerical bank or control differs from the sealed expected digest")
    bank_record = json.loads(bank_path.read_text())
    control_record = json.loads(control_path.read_text())
    bank = bank_record["kdaptx"]
    stock = bank["source_bindings"]
    stock_binding_ok = (hashlib.sha256(Path(stock["stock_conv_file"]).read_bytes()).hexdigest()
                        == stock["stock_conv_sha256"])
    composed = dict(bank, gate_contract=probe.KDA_GATE_CONTRACT,
                    ex2_equivalence=control_record["kdaex2"])
    errors = probe.kdaptx_gate_errors(composed)
    if result["old_module"]["sha256"] != expected_old_module_sha:
        errors.append("OLD module differs from the sealed banked module identity")
    if result["new_module"]["sha256"] != control_record["kdaex2"]["compiled_module_sha256"]:
        errors.append("NEW compared module differs from the control's compiled module identity")
    if not roster_ok:
        errors.append("convolution kernel roster has extra or missing modes")
    if not result["all_banked_conv_text_equal"] or not stock_binding_ok:
        errors.append("banked convolution device text or installed stock source differs")
    result.update({"gate_contract": probe.KDA_GATE_CONTRACT, "gate_passed": not errors, "errors": errors,
                   "stock_binding_ok": stock_binding_ok, "numerical_bank": {
                       "path": str(bank_path), "sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                       "action_key": bank_record["meta"]["pb_action"], "cases": len(bank["cases"])},
                   "intermediate_control": {"path": str(control_path),
                       "sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
                       "action_key": control_record["meta"]["pb_action"]}})
    result["expected_old_module_sha256"] = expected_old_module_sha
    (out / "kernel_identity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
