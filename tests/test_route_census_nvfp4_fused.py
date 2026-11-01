"""The route census recognizes the native paired WINDOW W4A4 entry points.

The route owners publish their actual dispatch pairs. This test checks the
census reader against that shared declaration. It does not attest a serve.
Historical TCQ cells cannot qualify the current WINDOW bytes.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "tessera_route_census.py"
CONTRACT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "tessera" / "serving" / "runtime_contract.json"
)


def _tool():
    spec = importlib.util.spec_from_file_location("tessera_route_census", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # its top level imports stdlib only
    return module


def test_nvfp4_dense_census_expected_owns_the_fused_pair():
    """The dense NVFP4 expectation is the route's own dispatch, fused pair in."""
    pytest.importorskip("torch")
    from tessera.serving import nvfp4_route
    from tessera.serving.scheme import (
        FUSED_WINDOW_DENSE_E2M1_SYMBOL,
        STRUCTURE_DENSE,
        TESSERA_NVFP4,
        launch_pairs,
    )
    from tessera.serving.telemetry import DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1

    fused = (FUSED_WINDOW_DENSE_E2M1_SYMBOL, DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1)
    got = nvfp4_route.census_expected(compiled=False)
    for regime in ("decode", "batch"):
        assert got[regime] == launch_pairs(
            TESSERA_NVFP4, structure=STRUCTURE_DENSE, regime=regime,
            include_experimental=True)
        assert fused in got[regime], regime


def test_nvfp4_routed_census_expected_owns_the_fused_pair():
    """The NVFP4 expert-stack expectation already opts into its fused pair."""
    pytest.importorskip("torch")
    from tessera.serving import nvfp4_moe_route
    from tessera.serving.scheme import (
        ROUTED_FUSED_WINDOW_E2M1_SYMBOL,
        STRUCTURE_ROUTED_MOE,
        TESSERA_NVFP4,
        launch_pairs,
    )
    from tessera.serving.telemetry import DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1

    fused = (ROUTED_FUSED_WINDOW_E2M1_SYMBOL, DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1)
    got = nvfp4_moe_route.census_expected(compiled=False)
    for regime in ("decode", "batch"):
        assert fused in got[regime], regime
        assert got[regime] <= launch_pairs(
            TESSERA_NVFP4, structure=STRUCTURE_ROUTED_MOE, regime=regime,
            mode="resident", include_experimental=True)


def test_route_census_tool_accepts_the_fused_nvfp4_pairs():
    """The tool grades fused NVFP4 records against the routes' own sets.

    Before the fix the tool compared every NVFP4 dense module against the
    single retired pair ``(torch._scaled_mm, native_span2)`` and every NVFP4
    expert stack against the FP8 stack's pair -- so a serve of master's fused
    dispatch mismatched on every NVFP4 module.  The expectation lives one
    place (the route modules); the tool reads it there.
    """
    pytest.importorskip("torch")
    from tessera.serving import nvfp4_moe_route, nvfp4_route
    from tessera.serving.scheme import (
        FUSED_WINDOW_DENSE_E2M1_SYMBOL,
        ROUTED_FUSED_WINDOW_E2M1_SYMBOL,
        TESSERA_NVFP4,
    )
    from tessera.serving.telemetry import (
        DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1,
        DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1,
    )

    tool = _tool()
    for regime in ("decode", "batch"):
        dense_want = tool.expected_pairs(
            TESSERA_NVFP4, regime, "dense", compiled=False, platform=None)
        assert dense_want == nvfp4_route.census_expected(compiled=False)[regime]
        assert (FUSED_WINDOW_DENSE_E2M1_SYMBOL, DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1) in dense_want
        moe_want = tool.expected_pairs(
            TESSERA_NVFP4, regime, "moe", compiled=False, platform=None)
        assert moe_want == nvfp4_moe_route.census_expected(compiled=False)[regime]
        assert (ROUTED_FUSED_WINDOW_E2M1_SYMBOL, DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1) in moe_want


def test_experimental_launches_require_their_own_runtime_and_rung_receipt():
    """A scoped receipt must not qualify an operation on another runtime or table."""
    from tessera.serving.contract import CENSUS_PHASE_REGIMES, cell_executes, format_entry
    from tessera.serving.scheme import EXPERIMENTAL_LAUNCHES

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    results = CONTRACT.parents[3] / "experiments/results"
    receipt = json.loads((results / "glm53_r768_stub_base_tp2_eager_census.json").read_text())
    config = json.loads((results / "glm53_r768_stub_base_config.json").read_text())
    tool = _tool()
    rungs = {tool.declared_rung(group["scheme"])
             for group in config["quantization_config"]["config_groups"].values()}
    experimental = set(EXPERIMENTAL_LAUNCHES)
    for cell in contract["lane_eligibility"]["cells"]:
        entry = format_entry(cell["family"], contract)
        phase = next(phase for phase, regime in CENSUS_PHASE_REGIMES.items()
                     if regime == cell["regime"])
        recorded = {(record["symbol"], record["decoder"])
                    for record in receipt["ranks"][0]["records"][phase].values()}
        for rung in cell["rungs_q256"]:
            scope_matches = (
                cell["platform"] == receipt["device"]["platform_token"]
                and cell["runtime"]["image"] == receipt["runtime"]["image"]
                and cell["runtime"]["execution_modes"] == [receipt["runtime"]["execution_mode"]]
                and cell["structure"] == "routed_moe" and cell["family"] == "TESSERA_E4M3_K1"
                and cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident"]
                and rung in rungs)
            observed = cell_executes(cell, q256=rung, entry=entry) & experimental
            assert observed == (recorded & experimental if scope_matches else set()), (
                cell["id"], rung, observed)
