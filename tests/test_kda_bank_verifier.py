"""The compared device modules must be the bank and control's actual modules."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from test_kda_probe_gate import _screen, probe


@pytest.mark.parametrize("fault", [None, "old", "new", "both", "subset", "extra"])
def test_module_binding_and_exact_mode_roster(probe, monkeypatch, tmp_path, fault):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "experiments/kda"))
    spec = importlib.util.spec_from_file_location("kda_bank_verifier_test", root / "experiments/kda/verify_conv_bank.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    out = tmp_path / "audit"
    for name in ("old", "new"):
        (out / name).mkdir(parents=True)
        (out / name / "device.cubin").write_bytes(b"ELF sections are mocked in this CPU policy test")
    s = _screen(probe, tmp_path)
    old_module = tmp_path / "bank.so"
    old_module.write_bytes(b"banked module identity fixture")
    old_sha = hashlib.sha256(old_module.read_bytes()).hexdigest()
    new_module = Path(s["ex2_equivalence"]["compiled_module"])
    other = tmp_path / "other.so"
    other.write_bytes(b"unrelated module identity fixture")
    (out / "old-module-path.txt").write_text(str(other if fault in ("old", "both") else old_module))
    (out / "new-module-path.txt").write_text(str(other if fault in ("new", "both") else new_module))
    modes = set(probe.KDA_PTX_MODES)
    if fault == "subset":
        modes.remove(max(modes))
    elif fault == "extra":
        modes.add(99)
    sections = {f".text._Z12kda_conv_refILi{mode}EEfixture": {"type": 1, "size": 64, "sha256": "a" * 64}
                for mode in modes}
    monkeypatch.setattr(verifier, "sections", lambda _: {"sections": sections, "sha256": "b" * 64})
    stock = tmp_path / "stock.py"
    stock.write_bytes(b"CPU stock-source identity fixture")
    s["source_bindings"] = {"stock_conv_file": str(stock), "stock_conv_sha256": hashlib.sha256(stock.read_bytes()).hexdigest()}
    bank = tmp_path / "bank.json"
    control = tmp_path / "control.json"
    bank.write_text(json.dumps({"meta": {"pb_action": "CPU fixture bank"}, "kdaptx": s}))
    control.write_text(json.dumps({"meta": {"pb_action": "CPU fixture control"}, "kdaex2": s["ex2_equivalence"]}))
    monkeypatch.setattr(sys, "argv", ["verify_conv_bank.py", str(out), str(bank), str(control),
                                      hashlib.sha256(bank.read_bytes()).hexdigest(),
                                      hashlib.sha256(control.read_bytes()).hexdigest(), old_sha])
    assert (verifier.main() == 0) is (fault is None)
