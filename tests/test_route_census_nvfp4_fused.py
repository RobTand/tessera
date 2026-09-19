"""The served census must recognise the fused NVFP4 launches (tessera#545).

Master serves the fused pairs -- ``nvfp4_route.apply`` stamps
``(a4_span2_gemm, native_span2_gemm)`` on every dense module and the native
expert stack stamps ``(a4_span2_grouped_gemm, native_span2_grouped)`` -- while
no ``lane_eligibility`` cell attests either (both sit in
``scheme.EXPERIMENTAL_LAUNCHES``).  Step 1 of #545 is a served census that
records the fused pairs actually dispatching, and the census tool is what
takes it -- so the tool must accept what the dispatch stamps, or every fused
module reads as a mismatch and the census cannot run at all.

These pin the tool's expectation to the routes that own the dispatch
(``nvfp4_route.census_expected`` / ``nvfp4_moe_route.census_expected``,
derived from ``scheme.ROUTE_LAUNCHES`` -- never a second spelling in the
tool), and pin the attestation side: no packaged cell names an experimental
pair, so the pin cannot move to a runtime carrying those cells until a
receipt earns them.  That last test is the red-before-green partner of the
attestation itself: it goes red the day a cell names a fused launch, and the
commit that turns it green is the one carrying the receipt.
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
        A4_DENSE_GEMM_SYMBOL,
        STRUCTURE_DENSE,
        TESSERA_NVFP4,
        launch_pairs,
    )
    from tessera.serving.telemetry import DECODER_NATIVE_SPAN2_GEMM

    fused = (A4_DENSE_GEMM_SYMBOL, DECODER_NATIVE_SPAN2_GEMM)
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
        A4_GROUPED_GEMM_SYMBOL,
        STRUCTURE_ROUTED_MOE,
        TESSERA_NVFP4,
        launch_pairs,
    )
    from tessera.serving.telemetry import DECODER_NATIVE_SPAN2_GROUPED

    fused = (A4_GROUPED_GEMM_SYMBOL, DECODER_NATIVE_SPAN2_GROUPED)
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
        A4_DENSE_GEMM_SYMBOL,
        A4_GROUPED_GEMM_SYMBOL,
        TESSERA_NVFP4,
    )
    from tessera.serving.telemetry import (
        DECODER_NATIVE_SPAN2_GEMM,
        DECODER_NATIVE_SPAN2_GROUPED,
    )

    tool = _tool()
    for regime in ("decode", "batch"):
        dense_want = tool.expected_pairs(
            TESSERA_NVFP4, regime, "dense", compiled=False, platform=None)
        assert dense_want == nvfp4_route.census_expected(compiled=False)[regime]
        assert (A4_DENSE_GEMM_SYMBOL, DECODER_NATIVE_SPAN2_GEMM) in dense_want
        moe_want = tool.expected_pairs(
            TESSERA_NVFP4, regime, "moe", compiled=False, platform=None)
        assert moe_want == nvfp4_moe_route.census_expected(compiled=False)[regime]
        assert (A4_GROUPED_GEMM_SYMBOL, DECODER_NATIVE_SPAN2_GROUPED) in moe_want


def test_no_packaged_cell_attests_an_experimental_launch():
    """Pin guard, torch-free: no cell names a launch no receipt has earned.

    The fused pairs leave ``EXPERIMENTAL_LAUNCHES`` when a served census earns
    them cells (#545 step 2), and the pin bump (step 3) follows the cells --
    never precedes them.  A cell naming a fused pair while this test still
    asserts absence is the signal to update the test beside the attestation,
    not to bump the pin past it.
    """
    from tessera.serving.scheme import EXPERIMENTAL_LAUNCHES

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    experimental = set(EXPERIMENTAL_LAUNCHES)
    for cell in contract["lane_eligibility"]["cells"]:
        pairs = {(entry["symbol"], entry["decoder"]) for entry in cell["executes"]}
        assert not (pairs & experimental), cell["id"]
