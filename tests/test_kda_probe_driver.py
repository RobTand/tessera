"""Driver controls use the existing scientific dependency population."""
import torch  # Collection declares the dependency; no blanket skips.
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_kda_probe_gate import _screen
from test_kda_probe_durability import observe_publication

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
    result = _screen(probe, tmp_path)
    result["served_bit_equal_all"] = failed_field != "served"
    bad = {"fma": "mutant_two_roundings_per_tap", "div": "mutant_div_rn"}
    if failed_field in bad:
        result["mutants_seen"][bad[failed_field]] = False
    elif failed_field == "missing":
        result["mutants_seen"].pop("mutant_ex2_ftz")
    elif failed_field == "ex2":
        result["ex2_equivalence"]["raw_fp32_bits"][2][3] = 7053949
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


def test_ex2_intermediate_equivalence_is_distinct_from_output_mutation(probe, monkeypatch, tmp_path):
    """The authorized v2 contract needs an active, erased exponential mutation."""
    result = _screen(probe, tmp_path)
    monkeypatch.setattr(probe, "part_kdaptx", lambda _: result)
    monkeypatch.setattr(sys, "argv", ["mhc_probe.py", "--out", str(tmp_path),
                                      "--parts", "kdaptx", "--numerics-only"])
    assert probe.main() == 0
    saved = json.loads((tmp_path / "mhc_probe_numerics.json").read_text())
    assert saved["kdaptx"]["mutants_seen"]["mutant_ex2_ftz"] is False
    assert saved["kdaptx"]["ex2_equivalence"]["ex2"]["bits_differing"] == 1


def test_publish_progress_waits_for_complete_durable_result(probe, monkeypatch, tmp_path):
    events = observe_publication(probe, monkeypatch)
    path = tmp_path / "mhc_probe.json"
    monkeypatch.setattr(probe, "part_kdafwd", lambda *_: {"cells": [{"completed_calls": 7}]})
    monkeypatch.setattr(sys, "argv", ["mhc_probe.py", "--out", str(tmp_path), "--parts", "kdafwd"])

    def commit(units, phase):
        assert units == 1 and phase == "publish"
        saved = json.loads(path.read_text())
        assert saved["meta"]["utc_end"] >= saved["meta"]["utc_start"]
        assert saved["meta"]["power_sampler"] == "test"
        assert saved["kdafwd"]["cells"] == [{"completed_calls": 7}]
        assert "netdata" in saved
        assert events == ["file", "replace", "directory"]
        events.append("progress")

    monkeypatch.setattr(probe.runpy, "run_path", lambda _: {"commit": commit})
    assert probe.main() == 0
    assert events[-1] == "progress"
