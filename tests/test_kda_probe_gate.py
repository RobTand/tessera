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


@pytest.mark.parametrize("failed_field", ["served", "fma", "ex2", "div", "missing", None])
def test_main_propagates_numerical_admission_failure(probe, monkeypatch, tmp_path, failed_field):
    result = {
        "served_bit_equal_all": failed_field != "served",
        "mutants_seen": {name: True for mode, name in probe.KDA_PTX_MODES.items() if mode},
    }
    bad = {"fma": "mutant_two_roundings_per_tap", "ex2": "mutant_ex2_ftz", "div": "mutant_div_rn"}
    if failed_field in bad:
        result["mutants_seen"][bad[failed_field]] = False
    elif failed_field == "missing":
        result["mutants_seen"].pop("mutant_ex2_ftz")
    monkeypatch.setattr(probe, "part_kdaptx", lambda _: result)
    monkeypatch.setattr(sys, "argv", ["mhc_probe.py", "--out", str(tmp_path),
                                      "--parts", "kdaptx", "--numerics-only"])
    rc = probe.main()
    saved = json.loads((tmp_path / "mhc_probe_numerics.json").read_text())
    assert saved["kdaptx"]["served_bit_equal_all"] == result["served_bit_equal_all"]
    assert saved["kdaptx"]["mutants_seen"] == result["mutants_seen"]
    assert (rc == 0) is (failed_field is None)


def test_failed_screen_stops_later_timing(probe, monkeypatch, tmp_path):
    result = {"served_bit_equal_all": False, "mutants_seen": {}}
    monkeypatch.setattr(probe, "part_kdaptx", lambda _: result)
    monkeypatch.setattr(probe, "part_kdafwd", lambda *_: pytest.fail("timing ran after failed numerics"))
    monkeypatch.setattr(sys, "argv", ["mhc_probe.py", "--out", str(tmp_path), "--parts", "kdaptx,kdafwd"])
    assert probe.main() != 0
    saved = json.loads((tmp_path / "mhc_probe.json").read_text())
    assert "kdafwd" not in saved


def _ex2_control():
    return {"finite_accumulator_count": 29, "all_accumulators_finite": True,
            "ex2": {"bit_equal": False, "bits_differing": 5},
            "denominator": {"bit_equal": True, "bits_differing": 0},
            "output_fp32": {"bit_equal": True, "bits_differing": 0},
            "output_bf16": {"bit_equal": True, "bits_differing": 0},
            "input_subnormal_count": 4, "input_subnormal_ex2_is_one_both": True,
            "changed_ex2_is_subnormal_flushed_to_positive_zero": True}


def test_ex2_intermediate_equivalence_is_distinct_from_output_mutation(probe, monkeypatch, tmp_path):
    """The authorized v2 contract needs an active, erased exponential mutation."""
    result = {"gate_contract": "tessera.kda_conv_screen.v2", "served_bit_equal_all": True,
              "mutants_seen": {name: mode != 2 for mode, name in probe.KDA_PTX_MODES.items() if mode},
              "ex2_equivalence": _ex2_control()}
    monkeypatch.setattr(probe, "part_kdaptx", lambda _: result)
    monkeypatch.setattr(sys, "argv", ["mhc_probe.py", "--out", str(tmp_path),
                                      "--parts", "kdaptx", "--numerics-only"])
    assert probe.main() == 0
    saved = json.loads((tmp_path / "mhc_probe_numerics.json").read_text())
    assert saved["kdaptx"]["mutants_seen"]["mutant_ex2_ftz"] is False
    assert saved["kdaptx"]["ex2_equivalence"]["ex2"]["bits_differing"] == 5
