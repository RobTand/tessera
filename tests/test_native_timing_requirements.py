"""A timing evidence request cannot be mistaken for a measurement (#688)."""
from __future__ import annotations

import copy

import pytest

from tessera.serving import census_plan, scheme


def plan():
    return census_plan.build_census_plan([{
        "route": scheme.TESSERA_FP8, "grid": "E4M3", "q256": 896,
        "structure": scheme.STRUCTURE_DENSE, "mode": "resident",
        "execution_mode": "eager", "regime": "decode", "tp_degree": 1,
        "requested_platform": "sm_121", "shape": {"M": 1, "N": 128, "K": 128},
    }])


def test_timing_requirements_contain_no_measurement_or_image_assertion():
    value = census_plan.build_timing_requirements(plan())
    assert value["schema"] == "tessera.kernel_timing_requirements.v1"
    assert value["status"] == "not_executed"
    assert value["minimum_samples"] == 3  # Issue #688's explicit acceptance.
    assert value["method"] == "cuda_events"
    assert value["rows"][0]["measurement"] is None
    assert "runtime_image" not in value
    assert set(value["required_evidence"]) == {
        "cuda_event_samples", "torch_profiler", "netdata", "observed_runtime_identity",
        "wire_identity", "native_preparation", "route_census",
    }


def test_timing_requirements_bind_the_complete_plan_without_mutation():
    original = plan()
    before = copy.deepcopy(original)
    value = census_plan.build_timing_requirements(original)
    assert original == before
    assert value["rows"][0]["scope_id"] == original["rows"][0]["id"]
    assert len(value["plan_sha256"]) == 64


@pytest.mark.parametrize("field,value", [("status", "measured"),
                                         ("gpu_executed", True),
                                         ("contract_sha256", "0" * 64)])
def test_forged_or_stale_plan_refuses(field, value):
    original = plan()
    original[field] = value
    with pytest.raises(ValueError, match="plan"):
        census_plan.build_timing_requirements(original)


def test_populated_timing_is_not_laundered_into_a_request():
    original = plan()
    original["rows"][0]["measurement"] = {"median_ms": 0.1}
    with pytest.raises(ValueError, match="plan"):
        census_plan.build_timing_requirements(original)
