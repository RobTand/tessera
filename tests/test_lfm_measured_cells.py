"""Published cells must cover the real sealed LFM receipt, only in its scope.

The fixture is the original PB census with one trailing newline added for the
repository, not a synthetic positive observation. Its original bytes are hashed.
The q1024 plan and exact artifact binding are in the accompanying campaign
receipt; replay here does not replace the full source/plan/sidecar collector.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tessera.serving.contract import CENSUS_PHASE_REGIMES, load_serving_contract

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/measurements/census/lfm25-8b-a1b-served-r4.json"
RAW_SHA256 = "825157292db88bd3791d59d867743ddf8e37e68dd24c63e93cfc919d927a2028"


def _replay(**overrides):
    raw = RECEIPT.read_bytes().removesuffix(b"\n")
    assert hashlib.sha256(raw).hexdigest() == RAW_SHA256
    receipt = json.loads(raw)
    spec = importlib.util.spec_from_file_location("lfm_census_replay", ROOT / "tools/tessera_route_census.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    owners = {owner for phase in receipt["record_owner"].values() for owner in phase.values()}
    kwargs = {
        "cells": load_serving_contract()["lane_eligibility"]["cells"],
        "phase_regimes": CENSUS_PHASE_REGIMES, "platform": "sm_121",
        "declared_rungs": {owner: 1024 for owner in owners},
        "record_owners": receipt["record_owner"],
        "families_by_route": {"TESSERA_FP8": "TESSERA_E4M3_K1"},
        "runtime_image": receipt["runtime"]["image"],
        "execution_mode": receipt["runtime"]["execution_mode"],
    }
    records = overrides.pop("records", receipt["records"])
    kwargs.update(overrides)
    return tool.all_structure_agreement(records, **kwargs)


def _positive_control():
    block, problems = _replay()
    assert not problems
    assert block["agrees"] is True, "measured LFM routes have no matching promoted cells"
    return block


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_real_lfm_receipt_is_covered_in_each_measured_phase(phase):
    block = _positive_control()["structures"]["routed_moe"]["phases"][phase]
    assert block["modules"] == block["covered_by_cell"] > 0
    assert block["unattested"] == 0


@pytest.mark.parametrize("scope", [
    {"runtime_image": None},
    {"runtime_image": "other/runtime@sha256:" + "a" * 64},
    {"execution_mode": "compiled"},
    {"execution_mode": None},
    {"platform": "sm_90"},
    {"record_owners": {}},
])
def test_other_runtime_or_unowned_records_cannot_borrow_the_measured_cells(scope):
    _positive_control()
    block, problems = _replay(**scope)
    assert not problems and block["agrees"] is None
    for phase in block["structures"]["routed_moe"]["phases"].values():
        assert phase["covered_by_cell"] == 0 and phase["unattested"] == phase["modules"]


@pytest.mark.parametrize("field,value", [("policy", "TESSERA_FP8:streamed"), ("kind", "dense")])
def test_other_structure_or_residency_cannot_borrow_the_measured_cells(field, value):
    _positive_control()
    records = json.loads(RECEIPT.read_text())["records"]
    for phase in records.values():
        for record in phase.values():
            record[field] = value
    block, problems = _replay(records=records)
    assert not problems and block["agrees"] is None


def test_an_unmeasured_rung_is_not_promoted_by_a_measured_stack_name():
    _positive_control()
    receipt = json.loads(RECEIPT.read_text())
    owners = {owner for phase in receipt["record_owner"].values() for owner in phase.values()}
    block, problems = _replay(declared_rungs={owner: 896 for owner in owners})
    assert not problems and block["agrees"] is None


@pytest.mark.parametrize("field,value", [("symbol", "wrong.kernel"), ("decoder", "wrong.decoder")])
def test_a_covered_scope_still_refuses_a_wrong_observed_launch(field, value):
    _positive_control()
    records = copy.deepcopy(json.loads(RECEIPT.read_text())["records"])
    next(iter(records["decode"].values()))[field] = value
    block, problems = _replay(records=records)
    assert block["agrees"] is False and problems
