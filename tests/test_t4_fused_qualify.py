"""Behavior tests for completeness, safety, byte matching, and ordered evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
source = Path(os.environ.get("T4_QUALIFY_SOURCE", ROOT / "experiments/t4_code/t4_fused_qualify.py"))
spec = importlib.util.spec_from_file_location("t4_qualify_under_test", source)
qual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qual)


def report(mode="correctness"):
    cell = dict(key="a", serving_owner=True, eager_ok=True, graph_equal=True,
                numeric={"ok": True}, register_resources=[{"REG": 124}])
    return dict(mode=mode, head="test-source", requested_population=["a"], cells=[cell],
                skips=[], resources={"functions": [{"REG": 124}]},
                population={"cells": 1, "requested_cells": 1})


def run_control(tmp_path, monkeypatch, data, mode="correctness"):
    target = "mode_correctness" if mode == "correctness" else "mode_quality"
    monkeypatch.setattr(qual, target, lambda _args: data)
    return qual.main(["--mode", mode, "--out", str(tmp_path / "receipt.json"),
                      "--q256", "896", "--ms", "1"])


@pytest.mark.before_control
def test_no_observed_cells_cannot_succeed(tmp_path, monkeypatch):
    data = report()
    data["cells"] = []
    assert run_control(tmp_path, monkeypatch, data) == 1


@pytest.mark.before_control
def test_missing_graph_is_a_failure(tmp_path, monkeypatch):
    data = report()
    data["cells"][0]["graph_equal"] = None
    assert run_control(tmp_path, monkeypatch, data) == 1


@pytest.mark.before_control
def test_numeric_mismatch_is_a_failure(tmp_path, monkeypatch):
    data = report()
    data["cells"][0]["numeric"] = {"ok": False, "mismatch": 1}
    assert run_control(tmp_path, monkeypatch, data) == 1


@pytest.mark.before_control
def test_mandatory_skip_is_a_failure(tmp_path, monkeypatch):
    data = report()
    data["skips"] = [{"reason": "actual route rejected its input"}]
    assert run_control(tmp_path, monkeypatch, data) == 1


@pytest.mark.before_control
def test_unmatched_quality_screen_cannot_claim_completion(tmp_path, monkeypatch):
    data = report("quality-screen")
    data["cells"] = [{"tensor": "real.weight", "complete": False,
                      "exact_byte_match": False, "unmatched_slack_bytes": 512}]
    assert run_control(tmp_path, monkeypatch, data, "quality-screen") == 1


@pytest.mark.before_control
def test_unknown_memory_is_not_safe(monkeypatch):
    monkeypatch.setattr(qual, "mem_available_gib", lambda: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        qual.check_mem_guard()


def test_guard_exact_boundary_and_below(monkeypatch):
    monkeypatch.setattr(qual, "mem_available_gib", lambda: 2.0)
    assert qual.check_mem_guard()["available_gib"] == 2.0
    monkeypatch.setattr(qual, "mem_available_gib", lambda: math.nextafter(2.0, 0))
    with pytest.raises(RuntimeError, match="below"):
        qual.check_mem_guard()


def test_correctly_completed_population_succeeds(tmp_path, monkeypatch):
    assert run_control(tmp_path, monkeypatch, report()) == 0
    saved = json.loads((tmp_path / "receipt.json").read_text())
    assert saved["complete"] is True and saved["failures"] == []


def test_duplicate_or_omitted_keys_cannot_succeed(tmp_path, monkeypatch):
    data = report()
    data["requested_population"] = ["a", "b"]
    data["cells"].append(copy.deepcopy(data["cells"][0]))
    assert run_control(tmp_path, monkeypatch, data) == 1


def test_false_serving_label_cannot_succeed(tmp_path, monkeypatch):
    data = report()
    data["cells"][0]["serving_owner"] = False
    assert run_control(tmp_path, monkeypatch, data) == 1


def test_gamma_domain_refuses_unbounded_contraction():
    with pytest.raises(ValueError, match="domain"):
        qual.conditional_gamma(2 ** 23)


def test_reference_rejects_a_finite_wrong_output():
    reference = torch.tensor([1.0], dtype=torch.float64)
    wrong = torch.tensor([2.0], dtype=torch.bfloat16)
    exact = qual.compare_projection(wrong, reference, torch.zeros_like(reference), True)
    random = qual.compare_projection(wrong, reference, torch.zeros_like(reference), False)
    assert not exact["ok"] and not random["ok"]


def test_relative_sse_is_not_relative_l2():
    source = torch.tensor([1.0, 2.0])
    decoded = torch.tensor([0.0, 2.0])
    got = qual.weight_error(source, decoded)
    assert got["relative_sse"] == .2
    assert got["relative_l2"] == math.sqrt(.2)
    assert got["squared_error"] == 1


def test_t8_wrong_bytes_or_geometry_is_not_a_pair():
    cell = dict(kind="mode0", m=16, hidden=4096, inter=1024, experts=288, top_k=8,
                wire_bytes=20000, tp_size=1, tp_rank=0, seed=7,
                input_sha256="a", routing_sha256="b", routing_weights_sha256="c",
                source_weight_seeds={"gate": 18}, source_weight_sha256={"gate": "d"},
                weight_bytes_scope=["gate", "up"],
                timing={"eager": {"median_ms": 3.0}, "graph": {"median_ms": 2.0}})
    baseline = dict(cell, format="T8")
    for field, wrong in (("wire_bytes", 20001), ("experts", 4), ("seed", 8)):
        bad = dict(baseline, **{field: wrong})
        with pytest.raises(ValueError, match="equal actual"):
            qual.compare_t8_cell(cell, bad)
    result = qual.compare_t8_cell(cell, baseline)
    assert result["graph"]["ratio"] == 1.0 and result["graph"]["pass_kill"]


def test_raw_events_keep_acquisition_order(monkeypatch):
    samples = iter([3.0, 1.0, 2.0])
    class Event:
        def __init__(self, **_kwargs):
            pass
        def record(self):
            pass
        def elapsed_time(self, _other):
            return next(samples)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    got = qual.time_callable(lambda: None, 0, 3)
    assert got["samples_ms"] == [3.0, 1.0, 2.0]
    assert got["median_ms"] == 2.0


def test_d41_resource_parse_keeps_each_register_record():
    raw = """Function : native_rate7
    REG:124 STACK:0 SHARED:1024 LOCAL:16
    Function : native_rate8
    REG:132 STACK:8 SHARED:2048 LOCAL:0
    """
    got = qual.parse_resource_usage(raw)
    assert [(r["REG"], r["LOCAL"]) for r in got] == [(124, 16), (132, 0)]
    with pytest.raises(ValueError, match="no register"):
        qual.parse_resource_usage("a tool printed no resource records")


def test_resource_pair_requires_the_requested_template():
    data = {"functions": [dict(rate=7, mode=0, dense=False, mixed=False, split=False, REG=124)]}
    assert qual.measured_registers(data, 896, "mode0")[0]["REG"] == 124
    with pytest.raises(ValueError, match="missing"):
        qual.measured_registers(data, 1024, "mode0")


def test_small_source_read_uses_real_safetensor_slice(tmp_path):
    from safetensors.torch import save_file

    weight = torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256)
    path = tmp_path / "source.safetensors"
    save_file({"actual.weight": weight}, str(path))
    metadata = dict(path=str(path), tensor="actual.weight", shape=[128, 256])
    got = qual.read_weight(metadata, small=True)
    assert torch.equal(got, weight[:2, :16])


def test_budget_search_prices_overhead_without_padding():
    rows, cols = 128, 256
    # A known byte-price target. The search must return this exact candidate,
    # not add zero padding to some cheaper artifact.
    base = qual.plane_bytes(128, rows, cols, "tcq") + 900
    budget = qual.plane_bytes(384, rows, cols, "tcq") + 900
    candidates = qual.budget_candidates(budget, rows, cols, base, 128)
    assert candidates[0] == 384


def test_prepare_inputs_declares_exact_ranges(tmp_path):
    from safetensors.torch import save_file

    model = tmp_path / "model"
    model.mkdir()
    key = "real.dense.weight"
    save_file({key: torch.arange(32 * 256, dtype=torch.float32).reshape(32, 256)},
              str(model / "one.safetensors"))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "one.safetensors"}}))
    args = qual.build_parser().parse_args(["--mode", "prepare-inputs", "--out", str(tmp_path / "prepared"),
                                         "--model-root", str(model), "--source-keys", key])
    result = qual.prepare_inputs(args)
    manifest = json.loads(Path(result["data_manifest"]).read_text())
    unit = result["units"][0]
    assert manifest["entry_count"] == len(manifest["entries"])
    assert manifest["total_bytes"] == sum(e["bytes"] for e in manifest["entries"])
    assert unit["bytes"] == 32 * 256 * 4
    declared = [e for e in manifest["entries"] if e["path"] == unit["path"] and e["offset"] == unit["offset"]]
    assert len(declared) == 1 and declared[0]["bytes"] == unit["bytes"]


def test_in_run_memory_guard_terminates_only_its_owned_child(tmp_path, monkeypatch):
    # The deterministic counter crosses the floor AFTER a real child starts.
    # The existing Envelope must terminate that exact child process group.
    output = tmp_path / "guard.json"
    values = iter([9.0, 1.5])
    monkeypatch.setattr(qual, "mem_available_gib", lambda: next(values))
    monkeypatch.setattr(sys, "argv", [str(source), "--mode", "research-fixture",
        "--out", str(output), "--q256", "128", "--seconds", "5"])
    with pytest.raises(RuntimeError, match="below"):
        qual.guarded_cli()
    receipt = json.loads(output.with_suffix(".memory.json").read_text())
    stopped = receipt["terminations"]
    assert len(stopped) == 1
    assert stopped[0]["pid"] > 0
    assert stopped[0]["returncode"] < 0
    assert stopped[0]["signals"][0]["signal"] == "SIGTERM"
    assert receipt["returncode"] is None


def test_t8_selector_uses_actual_bytes_not_a_prediction():
    from experiments.t4_code.t4_t8_baseline import select_actual_rate
    costs = {256: 10, 257: 20, 258: 30}
    result = select_actual_rate(20, 5, list(costs), costs.__getitem__,
                               {256: 20, 257: 999, 258: 30})
    assert result["exact_match"] and result["selected_q256"] == 257
    assert any(r["actual_serialized_bytes"] == 20 for r in result["measured"])


def test_t8_scalar_floor_never_calls_a_fake_cost_producer():
    from experiments.t4_code.t4_t8_baseline import select_actual_rate
    def forbidden(_rate):
        raise AssertionError("A proved scalar floor must not emit a timing candidate")
    result = select_actual_rate(10, 11, [256], forbidden, {256: 12})
    assert result["status"] == "unattainable_scalar_floor"
    assert not result["exact_match"] and result["measured"] == []


def test_t8_unmatched_bytes_do_not_receive_padding():
    from experiments.t4_code.t4_t8_baseline import select_actual_rate
    costs = {256: 10, 257: 20}
    result = select_actual_rate(15, 5, list(costs), costs.__getitem__, costs)
    assert not result["exact_match"] and result["selected_q256"] is None
    assert {r["actual_serialized_bytes"] for r in result["measured"]} == {10, 20}
    assert result["status"] == "no_exact_match_found"


def test_unattainable_t8_keeps_valid_t4_rows():
    data = report("timing")
    cell = data["cells"][0]
    cell.update(case_id="routed:gate_up", q256=128, wire_bytes=10,
                timing={"eager": {"samples_ms": [1.0]}, "graph": {"samples_ms": [1.0]}})
    baseline = {"cells": [], "byte_plan": [dict(case_id="routed:gate_up", t4_q256=128,
        actual_t4_serialized_bytes=10, exact_match=False, status="unattainable_scalar_floor",
        t8_plane_lower_bound=11)]}
    qual.attach_t8_comparison(cell, baseline)
    args = qual.build_parser().parse_args(["--mode", "timing", "--out", "/tmp/unused.json",
                                         "--compare-json", "baseline.json"] )
    complete, failures = qual.validate_report(data, args)
    assert complete and failures == []
    assert cell["competitive_status"] == "unattainable_scalar_floor"
    assert "t8_comparison" not in cell


def test_baseline_plan_cannot_emit_an_unmatched_timing_cell():
    from experiments.t4_code.t4_t8_baseline import validate_baseline
    args = SimpleNamespace(part="routed", dense_shapes=[], q256=[128],
                           plan_only=False, ms=[1], tp_cuts=False)
    plans = [dict(case_id="routed:" + group, group=group, t4_q256=128, exact_match=False,
                  status="unattainable_scalar_floor", t8_plane_lower_bound=11, target_bytes=10)
             for group in ("gate_up", "down", "chain")]
    complete, _ = validate_baseline({"byte_plan": plans, "cells": [], "skips": []}, args)
    assert complete
    complete, failures = validate_baseline({"byte_plan": plans, "cells": [{}], "skips": []}, args)
    assert not complete and failures


def test_component_weight_bytes_are_not_the_total_resident_context():
    row = dict(kind="mode0", projection_serialized_bytes={"gate": 10, "up": 12, "down": 15},
               total_serialized_bytes=37)
    qual.prepare_comparison_cell(row)
    assert row["wire_bytes"] == 22 and row["total_serialized_bytes"] == 37
    assert row["weight_bytes_scope"] == ["gate", "up"]
