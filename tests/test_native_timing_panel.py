"""CPU evidence controls; synthetic records do not claim GPU qualification."""
from __future__ import annotations
import copy
import gzip
import json
import importlib.abc
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tessera.serving import timing_panel as tp, census_plan, contract, scheme
from native_timing_panel_support import (
    save, canonical_wire, panel,
)


def rewrite(panel, key, fn):
    bound = panel["evidence"][key]
    value = tp.json_bytes(tp.read_bound(bound))
    fn(value)
    panel["evidence"][key] = save(Path(bound["path"]).parent, Path(bound["path"]).name, value)


def test_true_even_median_and_positive_binding(panel):
    result = tp.validate_panel(panel, expected_runtime=copy.deepcopy(panel["runtime"]))
    assert result["timing"]["median_ms"] == 2.5
    assert result["energy_status"] == "hold"
    assert panel["plan"]["gpu_executed"] is False


@pytest.mark.parametrize("fault", ["producer_as_runtime", "wrong_cell_source", "missing_code", "pair", "state",
    "shape", "activation", "platform", "wire", "native", "raw_samples", "fake_median",
    "nan", "sample_bool", "sample_count", "prep_wire", "prep_geometry", "prep_roles", "no_trace",
    "single_box", "missing_context", "wrong_interval", "no_fast_power", "energy", "duplicate_row", "extra_field", "noncanonical_wire", "cpu_only_trace", "float_count", "numeric_claim", "empty_netdata", "error_netdata", "missing_scope", "wrong_trace_type"])
def test_positive_receipt_refuses_false_evidence(panel, fault):
    expected = copy.deepcopy(panel["runtime"])
    if fault == "producer_as_runtime": panel["runtime"]["tessera_commit"] = "3" * 40
    elif fault in ("wrong_cell_source", "missing_code"):
        def change(doc):
            cell = next(c for c in doc["lane_eligibility"]["cells"] if c["id"] == panel["rows"][0]["cell_id"])
            if fault == "missing_code":
                cell["runtime"].pop("tessera_commit")
            else: cell["runtime"]["serving_source_sha256"] = "5" * 64
        rewrite(panel, "contract", change)
        panel["runtime"]["contract_sha256"] = panel["evidence"]["contract"]["sha256"]
        expected["contract_sha256"] = panel["runtime"]["contract_sha256"]
        rewrite(panel, "runtime", lambda v: v.update(panel["runtime"]))
        if fault == "missing_code":
            with pytest.raises(ValueError): census_plan.build_census_plan([panel["plan"]["rows"][0]["scope"]], raw_contract=tp.read_bound(panel["evidence"]["contract"]))
            return
        panel["plan"] = census_plan.build_census_plan([panel["plan"]["rows"][0]["scope"]], raw_contract=tp.read_bound(panel["evidence"]["contract"]))
        panel["rows"][0]["scope_id"] = panel["plan"]["rows"][0]["id"]
    elif fault in ("pair", "state", "shape", "activation", "platform"):
        field = {"pair": "symbol", "state": "state", "shape": "shape", "activation": "contract", "platform": "platform"}[fault]
        rewrite(panel, "routes", lambda v: v["records"][0].update({field: "wrong"}))
    elif fault in ("wire", "native", "raw_samples", "no_trace"):
        key = {"native": "native_binary", "raw_samples": "samples", "no_trace": "trace"}.get(fault, fault)
        Path(panel["evidence"][key]["path"]).write_bytes(b"tampered")
    elif fault == "float_count": panel["rows"][0]["timing"]["n"] = 4.0
    elif fault == "numeric_claim": panel["claims"]["certifies_placement"] = 0
    elif fault == "empty_netdata": rewrite(panel, "telemetry", lambda v: v["netdata"]["sparky"][next(iter(tp.NETDATA_CONTEXTS))]["raw_response"].update(result={"labels": [], "data": []}))
    elif fault == "error_netdata": rewrite(panel, "telemetry", lambda v: v["netdata"]["sparky"][next(iter(tp.NETDATA_CONTEXTS))]["raw_response"].update(result={"error": "unknown context"}))
    elif fault == "missing_scope": panel["plan"]["rows"][0].pop("scope")
    elif fault == "wrong_trace_type":
        bound = panel["evidence"]["trace"];panel["evidence"]["trace"] = save(Path(bound["path"]).parent, "trace.json.gz", gzip.compress(tp.canonical([])), True)
    elif fault == "fake_median": panel["rows"][0]["timing"]["median_ms"] = 3
    elif fault in ("nan", "sample_bool", "sample_count"):
        rewrite(panel, "samples", lambda v: v.update(samples_ms=[True, 1, 2] if fault == "sample_bool" else [1, 2]))
        if fault == "nan": panel["rows"][0]["timing"]["median_ms"] = float("nan")
    elif fault.startswith("prep_"):
        field, value = {"prep_wire": ("wire_sha256", "5" * 64), "prep_geometry": ("shape", {}),
                        "prep_roles": ("roles", [])}[fault]
        rewrite(panel, "preparation", lambda v: v.update({field: value}))
    elif fault in ("single_box", "missing_context", "wrong_interval", "no_fast_power"):
        def change(v):
            if fault == "single_box": v["netdata"].pop("sparky")
            elif fault == "missing_context": v["netdata"]["sparky"].pop(next(iter(tp.NETDATA_CONTEXTS)))
            elif fault == "wrong_interval": v["interval_unix"] = [9, 10]
            else: v["fast_power_samples"] = []
        rewrite(panel, "telemetry", change)
    elif fault == "energy": panel["energy"]["status"] = "qualified"
    elif fault == "duplicate_row": panel["rows"].append(copy.deepcopy(panel["rows"][0]))
    elif fault == "extra_field": panel["rows"][0]["default_topology"] = True
    elif fault == "noncanonical_wire":
        bound = panel["evidence"]["wire"];raw = bytearray(tp.read_bound(bound));raw[-1] ^= 1
        panel["evidence"]["wire"] = save(Path(bound["path"]).parent, "wire.bin", bytes(raw), True)
        rewrite(panel, "preparation", lambda v: v.update(wire_sha256=panel["evidence"]["wire"]["sha256"]))
    elif fault == "cpu_only_trace":
        bound = panel["evidence"]["trace"]
        panel["evidence"]["trace"] = save(Path(bound["path"]).parent, "trace.json.gz", gzip.compress(tp.canonical({"traceEvents": [{"cat": "cpu_op"}]})), True)
    with pytest.raises((ValueError, OSError)):
        tp.validate_panel(panel, expected_runtime=expected)


def test_legacy_cell_uses_independently_bound_runtime_and_record_proof(panel, monkeypatch):
    def change(doc):
        cell=next(c for c in doc["lane_eligibility"]["cells"] if c["id"]==panel["rows"][0]["cell_id"])
        cell["runtime"].pop("tessera_commit");cell["runtime"].pop("serving_source_sha256")
    rewrite(panel,"contract",change)
    panel["runtime"]["contract_sha256"]=panel["evidence"]["contract"]["sha256"]
    rewrite(panel,"runtime",lambda v:v.update(panel["runtime"]))
    panel["plan"]=census_plan.build_census_plan([panel["plan"]["rows"][0]["scope"]],raw_contract=tp.read_bound(panel["evidence"]["contract"]))
    panel["rows"][0]["scope_id"]=panel["plan"]["rows"][0]["id"]
    assert tp.validate_panel(panel,expected_runtime=copy.deepcopy(panel["runtime"]))["cell_id"]==panel["rows"][0]["cell_id"]
    rewrite(panel,"runtime_origins",lambda v:v["installation"].update(installed_commit="9"*40))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE","0")
    with pytest.raises(ValueError,match="RECORD"):tp.validate_panel(panel,expected_runtime=panel["runtime"])


def test_legacy_cell_record_proof_stamps_in_dev_mode(panel, monkeypatch, capsys):
    """D32: the installed-record proof is a seal on the external panel path;
    default dev stamps the mismatch and the legacy cell still validates."""
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)

    def change(doc):
        cell = next(c for c in doc["lane_eligibility"]["cells"] if c["id"] == panel["rows"][0]["cell_id"])
        cell["runtime"].pop("tessera_commit");cell["runtime"].pop("serving_source_sha256")
    rewrite(panel, "contract", change)
    panel["runtime"]["contract_sha256"] = panel["evidence"]["contract"]["sha256"]
    rewrite(panel, "runtime", lambda v: v.update(panel["runtime"]))
    panel["plan"] = census_plan.build_census_plan([panel["plan"]["rows"][0]["scope"]], raw_contract=tp.read_bound(panel["evidence"]["contract"]))
    panel["rows"][0]["scope_id"] = panel["plan"]["rows"][0]["id"]
    assert tp.validate_panel(panel, expected_runtime=copy.deepcopy(panel["runtime"]))["cell_id"] == panel["rows"][0]["cell_id"]
    rewrite(panel, "runtime_origins", lambda v: v["installation"].update(installed_commit="9" * 40))
    result = tp.validate_panel(panel, expected_runtime=panel["runtime"])
    out = capsys.readouterr().out
    assert out.count("[DEV-MODE]") == 1 and "installed RECORD proof" in out
    assert result["cell_id"] == panel["rows"][0]["cell_id"]


def test_panel_validator_stays_the_local_entry_point_and_the_handoff_is_versioned(panel):
    result = tp.validate_panel(panel, expected_runtime=copy.deepcopy(panel["runtime"]))
    # validate_panel remains the passive summary; the bound handoff is a
    # separate, versioned artifact the caller must gate on an installed CPU
    # preflight before it can exist.
    assert set(result) == {"scope_id", "cell_id", "kernel_lane", "timing", "energy_status", "claims"}
    assert tp.OBSERVATION_SCHEMA == "tessera.shape_time_observation.v1"
    assert tp.SAMPLE_UNIT == "single_apply"
    assert "not end-to-end serving evidence" in tp.OPERATOR_PROJECTION


def test_source_pin_difference_stamps_and_continues_in_dev_mode(panel, monkeypatch, capsys):
    """D32: the old-source-pin refusal stamps once and the panel validates.

    This is the regression for the reported failure: a panel measured on one
    Tessera tree against a frozen expected context naming another was refused
    outright. Default dev mode stamps one [DEV-MODE] line naming both digests
    and continues with the stored data.
    """
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    expected = copy.deepcopy(panel["runtime"])
    expected["serving_source_sha256"] = "0" * 64
    result = tp.validate_panel(panel, expected_runtime=expected)
    out = capsys.readouterr().out
    assert out.count("[DEV-MODE]") == 1
    assert "0" * 64 in out and "2" * 64 in out
    assert result["timing"]["median_ms"] == 2.5


def test_certified_zero_keeps_the_source_pin_refusal(panel, monkeypatch):
    """Certified mode keeps the verbatim refusal the site always raised."""
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    expected = copy.deepcopy(panel["runtime"])
    expected["serving_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="observed runtime differs from independent expected context"):
        tp.validate_panel(panel, expected_runtime=expected)


def test_panel_byte_integrity_still_refuses_in_dev_mode(panel, monkeypatch):
    """Dev mode never weakens integrity: bound bytes that moved still refuse."""
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    Path(panel["evidence"]["samples"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file evidence differs"):
        tp.validate_panel(panel, expected_runtime=copy.deepcopy(panel["runtime"]))


def test_execution_semantics_refuse_in_dev_mode(panel, monkeypatch):
    """D32 boundary: a different execution case refuses even in default dev."""
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    expected = copy.deepcopy(panel["runtime"])
    expected["serve_flags"] = {"TESSERA_SERVE_MODE": "streamed"}  # valid, but the wrong case
    with pytest.raises(ValueError, match="observed runtime differs from independent expected context"):
        tp.validate_panel(panel, expected_runtime=expected)
