"""The census and the contract must name the two problem shapes once.

``runtime_contract.json`` declares ``lane_eligibility.regimes = ["decode",
"batch"]`` and keys every cell by one of them.  ``tools/tessera_route_census.py``
drives the same two shapes and calls the many-row one ``prefill``, because its
receipt is keyed by the shape it drove and several served receipts quote those
keys.  Two vocabularies for one axis cost nothing while nothing reads the axis
-- and the moment something does (a per-``(family, regime)`` expectation, which
is what issues #10 and #47 need per the #42 decision) they cost either a
``KeyError`` in the per-module loop with two loaded models behind it, or a
guard that is vacuously true on half the matrix.

So the pair is written once, in ``contract.CENSUS_PHASE_REGIMES``, and these
tests are what make a divergence a test failure:

1. a regime declared in the contract that the census never drives is refused at
   contract load;
2. a rename on one side only is refused there too;
3. every phase the census drives joins to a real cell, for every family the
   contract publishes -- the join #10 and #47 are about to build;
4. the census resolves its phase names *through* the table rather than writing
   them a second time.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from tessera.serving.census import cell_launch_agreement
from tessera.serving.contract import (
    CENSUS_PHASE_REGIMES,
    PAYLOAD_FAMILY_BY_ROUTE,
    contract_path,
    load_serving_contract,
    validate_serving_contract,
)

from withdrawn_cells import withdrawn_v39_cells

ROOT = Path(__file__).resolve().parent.parent
CENSUS = ROOT / "tools" / "tessera_route_census.py"


def _contract() -> dict:
    """The packaged contract, parsed but not yet validated."""
    return json.loads(contract_path().read_text(encoding="utf-8"))


def test_the_packaged_contract_declares_exactly_the_regimes_the_census_drives():
    regimes = set(load_serving_contract()["lane_eligibility"]["regimes"])
    assert regimes == set(CENSUS_PHASE_REGIMES.values())


def test_a_declared_regime_the_census_never_drives_is_refused():
    """The quiet failure: a cell no observation can ever join to."""
    contract = _contract()
    contract["lane_eligibility"]["regimes"].append("chunked_prefill")
    with pytest.raises(ValueError, match="regimes"):
        validate_serving_contract(contract)


def test_renaming_a_regime_on_the_contract_side_only_is_refused():
    """The loud failure, moved from the per-module loop to contract load."""
    contract = _contract()
    block = contract["lane_eligibility"]
    block["regimes"] = ["batched" if r == "batch" else r for r in block["regimes"]]
    for cell in block["cells"]:
        if cell["regime"] == "batch":
            cell["regime"] = "batched"
    with pytest.raises(ValueError, match="regimes"):
        validate_serving_contract(contract)




#: The two ranks' route traces from the two-rank GLM-5.3-Flash 4-layer stub
#: serve (#506), committed byte for byte from the serve's log directory.
TP2_STUB_TRACES = (ROOT / "experiments/results/glm53_a4_stub_tp2_route_trace_rank0.json",
                   ROOT / "experiments/results/glm53_a4_stub_tp2_route_trace_rank1.json")


def test_every_route_the_two_rank_stub_served_joins_to_a_cell():
    """The two-rank stub's records join the cells they were minted from.

    Since contract v39 its E2M1_K2 records, routed and dense, join only the
    WITHDRAWN cells (``tests/withdrawn_cells.withdrawn_v39_cells``): they name
    launches this build no longer makes.

    Each rank's ``tessera.route_trace/1`` entry is joined to the sm_121 cells
    by what it names -- family (through the route), structure (through the
    record kind), regime (through the forward's own M, one row for decode),
    residency, activation contract and the launch pair -- so a cell that stops
    matching what the serve executed fails here.  The routed launch carries
    the backend the runtime picked (``:FLASHINFER_CUTLASS``); the comparison
    removes it through ``scheme.moe_census_symbol_base``, as the census does.

    The runtime image is NOT part of this join.  The stub served every family
    on one image, and the dense cells name the vanilla vLLM pin their own
    receipts ran; a dense record here shows the route executed at a world of
    two, not that the dense cells cover this image.  The routed E2M1_K2 cells'
    image is tied to the receipt in ``tests/test_serving_contract.py``.
    """
    from tessera.serving.contract import cell_residency_modes
    from tessera.serving.scheme import moe_census_symbol_base

    cells = [cell for cell in load_serving_contract()["lane_eligibility"]["cells"]
             if cell["platform"] == "sm_121"]
    joined = set()
    per_rank = []
    for path in TP2_STUB_TRACES:
        trace = json.loads(path.read_text(encoding="utf-8"))
        assert trace["schema"] == "tessera.route_trace/1", path.name
        assert trace["entries"], f"{path.name} records no route"
        seen = set()
        for entry in trace["entries"]:
            route, _, mode = entry["policy"].partition(":")
            family = PAYLOAD_FAMILY_BY_ROUTE[route]
            structure = "routed_moe" if entry["kind"] == "moe" else "dense"
            rows = entry["shape"].split(":", 1)[0]
            regime = "decode" if rows == "M1" else "batch"
            symbol = (moe_census_symbol_base(entry["symbol"])
                      if structure == "routed_moe" else entry["symbol"])
            launch = {"symbol": symbol, "decoder": entry["decoder"]}
            matched = [cell["id"] for cell in cells
                       if cell["family"] == family and cell["structure"] == structure
                       and cell["regime"] == regime
                       and cell["activation_contract"] == entry["contract"]
                       and mode in cell_residency_modes(cell)
                       and launch in cell["executes"]]
            if family == "TESSERA_E2M1_K2":
                # Contract v39 withdrew the E2M1 cells these records were
                # minted from (tessera#604, second half): the routed ones name
                # the materialising launch nvfp4_moe_route no longer makes,
                # the dense ones the ``(torch._scaled_mm, native_span2)``
                # launch nvfp4_route no longer makes.  The records still join
                # the withdrawn cells, quoted outside the published document,
                # and join nothing the contract ships.
                assert entry["decoder"] == {"routed_moe": "torch_materialize_stock",
                                            "dense": "native_span2"}[structure], entry
                assert not matched, f"{path.name}: {entry} joins to {matched}"
                withdrawn = [cell["id"] for cell in withdrawn_v39_cells()
                             if cell["regime"] == regime and cell["structure"] == structure
                             and launch in cell["executes"]
                             and mode in cell_residency_modes(cell)
                             and cell["activation_contract"] == entry["contract"]]
                assert withdrawn, f"{path.name}: {entry} joins no withdrawn cell"
                seen.add((family, structure, regime))
                continue
            if structure == "dense" and family in ("TESSERA_E4M3_K1", "TESSERA_BF16_K1"):
                # Contract v31 withdrew these families' dense cells with the
                # dispatch they attested (tessera#538), so the stub's dense
                # records join to nothing.  Asserted rather than skipped: the
                # day a dense cell is published for either family, this branch
                # stops being the one that runs.
                assert not matched, f"{path.name}: {entry} joins to {matched}"
                continue
            assert matched, f"{path.name}: {entry} joins to no sm_121 cell"
            seen.add((family, structure, regime))
        per_rank.append(seen)
        joined |= seen
    assert per_rank[0] == per_rank[1], "the two ranks served different routes"
    assert {("TESSERA_E2M1_K2", structure, regime)
            for structure in ("dense", "routed_moe")
            for regime in ("decode", "batch")} <= joined, joined


def test_the_census_drives_every_regime_the_table_names():
    source = CENSUS.read_text()
    driven = re.search(r"^DRIVEN_REGIMES = \(([^)]*)\)", source, re.MULTILINE)
    assert driven, f"{CENSUS} no longer states which regimes it drives"
    names = set(re.findall(r'"([^"]+)"', driven.group(1)))
    undrivable = sorted(set(CENSUS_PHASE_REGIMES.values()) - names)
    assert not undrivable, (
        f"the table names the regime(s) {undrivable}, which the census drives no forward for"
    )


def test_the_census_writes_no_phase_name_of_its_own():
    """Anti-vacuity: one table, not a table plus a copy.

    A literal phase key in the tool is the second spelling this test exists to
    prevent -- it is what lets the contract's vocabulary move while the
    census's does not.  The phase names live in ``CENSUS_PHASE_REGIMES`` and
    the tool reads them from there; prose in a docstring or a message is free.
    """
    code = [
        line for line in CENSUS.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    literals = [
        line.strip() for line in code
        if re.search(r"phases\[\s*[\"']", line)
        or re.search(r"^\s*(?:batch_phase|decode_phase)\s*=\s*[\"']", line)
    ]
    assert not literals, (
        "the census keys a phase by a literal instead of through "
        f"contract.CENSUS_PHASE_REGIMES:\n  " + "\n  ".join(literals)
    )


# --- a phase attests the regime its forward RAN, not the one it is named (#207)
#
# The phase label is what the census asked for; the record's ``M`` is what the
# machine did.  ``cell_launch_agreement`` selected the cell from the label
# alone, so an eight-row forward recorded under the decode phase was counted as
# a covered, agreeing decode observation -- and resident FP8 publishes the same
# launch pair in both regimes, so nothing downstream could notice.  The census's
# own generic shape check could not catch it either: it asked only whether the
# two phases' shape strings differed in aggregate.  A decode attestation needs
# an M=1 observation; a multi-row forward is prefill evidence.

_MODULE = "model.layers.0.mlp.down_proj"


def _tool():
    """The census tool, loaded by path: its top level is stdlib-only by design."""
    spec = importlib.util.spec_from_file_location("tessera_route_census", CENSUS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(m, **over):
    """A current FP8 dense record isolates the shape and regime rules."""
    return dict({"kind": "dense", "policy": "TESSERA_FP8:resident",
                 "symbol": "tessera::window_gemm_dense", "decoder": "native_window_gemm",
                 "shape": f"M{m}:N64:K64", "state": "served",
                 "contract": "fp8_per_token_dynamic"}, **over)


def _records(batch_m, decode_m):
    phases = {regime: m for regime, m in (("batch", batch_m), ("decode", decode_m))}
    return {phase: {_MODULE: _record(phases[regime])}
            for phase, regime in CENSUS_PHASE_REGIMES.items()}


def _agreement(records):
    contract = load_serving_contract()
    return cell_launch_agreement(
        records, cells=contract["lane_eligibility"]["cells"],
        phase_regimes=CENSUS_PHASE_REGIMES, platform="sm_121",
        rungs_by_module={_MODULE: 1024}, families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=_dense_image(contract), execution_mode="eager")


def _dense_image(contract):
    return next(cell["runtime"]["image"] for cell in contract["lane_eligibility"]["cells"]
                if (cell["family"], cell["structure"], cell["regime"]) ==
                ("TESSERA_E4M3_K1", "dense", "batch") and "eager" in cell["runtime"]["execution_modes"])


def _decode_phase():
    return next(p for p, regime in CENSUS_PHASE_REGIMES.items() if regime == "decode")


def test_a_multi_row_forward_cannot_be_counted_as_a_decode_observation():
    """The shared cell matcher, which the generic census and ts111 replay share."""
    block, problems = _agreement(_records(64, 8))
    decode = block["phases"][_decode_phase()]
    assert problems and "M8" in problems[0], problems
    assert block["agrees"] is False
    assert decode["covered_by_cell"] == 0


def test_a_record_with_no_concrete_shape_attests_no_regime():
    records = _records(64, 1)
    records[_decode_phase()][_MODULE].pop("shape")
    block, problems = _agreement(records)
    assert problems and "shape" in problems[0], problems
    assert block["agrees"] is False
    assert block["phases"][_decode_phase()]["covered_by_cell"] == 0


def test_the_matched_shapes_are_the_control():
    block, problems = _agreement(_records(64, 1))
    assert problems == []
    assert block["agrees"] is True
    assert block["phases"][_decode_phase()]["covered_by_cell"] == 1


def test_the_generic_shape_check_reads_each_record_against_its_own_phase():
    """Not an aggregate difference: every counted record must exercise its regime."""
    tool = _tool()
    assert tool.phase_shape_problems(
        _records(64, 1), phase_regimes=CENSUS_PHASE_REGIMES) == []
    problems = tool.phase_shape_problems(_records(64, 8), phase_regimes=CENSUS_PHASE_REGIMES)
    assert problems and all(_MODULE in p for p in problems), problems
    # ...and the case the aggregate check could see is still refused.
    assert tool.phase_shape_problems(_records(64, 64), phase_regimes=CENSUS_PHASE_REGIMES)


def test_a_compiled_census_keeps_its_symbolic_records():
    """Eager M parsing is not imposed on a shape-polymorphic trace."""
    tool = _tool()
    records = {phase: {_MODULE: _record("*")} for phase in CENSUS_PHASE_REGIMES}
    assert tool.phase_shape_problems(
        records, phase_regimes=CENSUS_PHASE_REGIMES, compiled=True) == []
