"""A failed or vacuous numerical screen must fail the admitted action."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "experiments/mhc/mhc_probe.py"
    spec = importlib.util.spec_from_file_location("kda_probe_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda _: "CPU gate test")
    sampler = SimpleNamespace(start=lambda: None, source="test", stop_flag=False)
    monkeypatch.setattr(module.rpo, "PowerSampler", lambda: sampler)
    monkeypatch.setattr(module.rpo, "netdata_window", lambda *_: {})
    return module


@pytest.mark.parametrize("failed_field", ["served", "fma", "ex2", "div", None])
def test_main_propagates_numerical_admission_failure(probe, monkeypatch, tmp_path, failed_field):
    result = {
        "served_bit_equal_all": failed_field != "served",
        "mutants_seen": {
            "mutant_two_roundings_per_tap": failed_field != "fma",
            "mutant_ex2_ftz": failed_field != "ex2",
            "mutant_div_rn": failed_field != "div",
        },
    }
    monkeypatch.setattr(probe, "part_kdaptx", lambda _: result)
    monkeypatch.setattr(sys, "argv", ["mhc_probe.py", "--out", str(tmp_path),
                                      "--parts", "kdaptx", "--numerics-only"])
    rc = probe.main()
    saved = json.loads((tmp_path / "mhc_probe_numerics.json").read_text())
    assert saved["kdaptx"]["served_bit_equal_all"] == result["served_bit_equal_all"]
    assert saved["kdaptx"]["mutants_seen"] == result["mutants_seen"]
    assert (rc == 0) is (failed_field is None)
