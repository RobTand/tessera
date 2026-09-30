"""CPU planning must never look like device qualification (#689)."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tessera.serving import scheme


def planner():
    return importlib.import_module("tessera.serving.census_plan")


def request(**changes):
    value = {
        "route": scheme.TESSERA_FP8, "grid": "E4M3", "q256": 896,
        "structure": scheme.STRUCTURE_DENSE, "mode": "resident",
        "execution_mode": "eager", "regime": "decode", "tp_degree": 1,
        "requested_platform": "sm_121", "shape": {"M": 1, "N": 128, "K": 128},
    }
    value.update(changes)
    return value


def test_plan_is_explicitly_unmeasured():
    result = planner().build_census_plan([request()])
    assert result["schema"] == "tessera.native_census_plan.v1"
    assert result["status"] == "not_executed"
    assert result["gpu_executed"] is False
    assert result["rows"][0]["measurement"] is None
    assert result["rows"][0]["qualification"] == "not_measured"
    assert "lane_eligibility" not in result
    assert "runtime_image" not in result  # No supplied identity masquerading as evidence.


def test_launch_pairs_come_from_shared_dispatch_registry(monkeypatch):
    owner = scheme.launch_pairs
    scope = {"structure": scheme.STRUCTURE_DENSE,
             "regime": "decode", "mode": "resident"}
    expected = owner(scheme.TESSERA_FP8, **scope)
    calls = []

    def observed(route, **filters):
        calls.append((route, filters))
        return owner(route, **filters)

    # Observe the real owner without making the packaged contract contradict
    # its dispatch table. A fictional pair would fail contract validation first.
    monkeypatch.setattr(scheme, "launch_pairs", observed)
    result = planner().build_census_plan([request()])
    assert (scheme.TESSERA_FP8, scope) in calls
    assert result["rows"][0]["admissible_launch_pairs"] == [list(p) for p in sorted(expected)]


def test_scope_ids_are_order_independent_but_bind_every_requested_axis():
    first = request()
    second = request(q256=912, execution_mode="compiled", mode="streamed")
    a = planner().build_census_plan([first, second])
    b = planner().build_census_plan([dict(reversed(list(second.items()))), first])
    assert a == b
    assert len({row["id"] for row in a["rows"]}) == 2
    assert a["rows"][0]["scope"] != a["rows"][1]["scope"]


@pytest.mark.parametrize("exists", [False, True])
def test_unreadable_or_invalid_contract_refuses_by_name(tmp_path, monkeypatch, exists):
    path = tmp_path / "runtime_contract.json"
    if exists:
        path.write_text("not JSON")
    monkeypatch.setattr(planner(), "contract_path", lambda: path)
    with pytest.raises(ValueError, match="packaged runtime contract"):
        planner().build_census_plan([request()])


def test_duplicate_scopes_refuse_instead_of_silently_deduplicating():
    with pytest.raises(ValueError, match="duplicate request scope"):
        planner().build_census_plan([request(), request()])


@pytest.mark.parametrize("field,value", [
    ("q256", True), ("q256", 0), ("q256", 896.0),
    ("tp_degree", -1), ("execution_mode", "invented"),
    ("mode", "invented"), ("requested_platform", ""),
    ("route", "invented"), ("grid", "invented"),
])
def test_invalid_selectors_refuse_by_field(field, value):
    with pytest.raises(ValueError, match=field):
        planner().build_census_plan([request(**{field: value})])


@pytest.mark.parametrize("shape", [{"M": 0, "N": 128, "K": 128},
                                    {"M": True, "N": 128, "K": 128},
                                    {"M": 1, "N": 128},
                                    {"M": 1, "N": 128, "K": 128, "typo": 1}])
def test_shape_is_closed_and_strictly_integer(shape):
    with pytest.raises(ValueError, match="shape"):
        planner().build_census_plan([request(shape=shape)])


def test_regime_must_match_shared_problem_shape_rule():
    with pytest.raises(ValueError, match="regime"):
        planner().build_census_plan([request(shape={"M": 3, "N": 128, "K": 128})])


def test_routed_geometry_requires_experts_and_topk():
    with pytest.raises(ValueError, match="shape"):
        planner().build_census_plan([request(structure=scheme.STRUCTURE_ROUTED_MOE)])
    result = planner().build_census_plan([
        request(structure=scheme.STRUCTURE_ROUTED_MOE,
                shape={"M": 1, "N": 128, "K": 128, "experts": 4, "topk": 2})])
    row = result["rows"][0]
    assert row["reader_scope"] == "dense_range_only_not_routed_admission"
    assert row["qualification"] == "not_measured"


def test_requested_unattested_rungs_do_not_promote_or_mutate_contract():
    from tessera.serving.contract import load_serving_contract
    before = copy.deepcopy(load_serving_contract())
    result = planner().build_census_plan([request(q256=880), request(q256=912)])
    assert all(row["qualification"] == "not_measured" for row in result["rows"])
    assert load_serving_contract() == before


def test_cli_writes_an_unmeasured_plan(tmp_path):
    requests, output = tmp_path / "requests.json", tmp_path / "plan.json"
    requests.write_text(json.dumps([request()]))
    tool = Path(__file__).resolve().parents[1] / "tools" / "plan_native_census.py"
    result = subprocess.run([sys.executable, str(tool), "--requests", str(requests),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["gpu_executed"] is False
    assert "not executed or qualified" in result.stdout


def test_cli_invalid_request_does_not_overwrite_output(tmp_path):
    requests, output = tmp_path / "requests.json", tmp_path / "plan.json"
    requests.write_text(json.dumps([request(q256=True)]))
    output.write_text("sentinel")
    tool = Path(__file__).resolve().parents[1] / "tools" / "plan_native_census.py"
    result = subprocess.run([sys.executable, str(tool), "--requests", str(requests),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "q256" in result.stderr
    assert output.read_text() == "sentinel"
