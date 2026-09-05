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


def test_every_phase_the_census_drives_joins_to_a_cell_of_every_family():
    """The per-(family, regime) expectation, exercised on the real table.

    Keyed the way an implementer would key it -- the census's phase name
    mapped through the table, against the cells' own ``(family, regime)`` --
    so a divergence shows up here as a missing pair rather than as a
    ``KeyError`` on a loaded box, and a vacuous half of the matrix shows up as
    an absent cell rather than as a guard that passed.
    """
    contract = load_serving_contract()
    block = contract["lane_eligibility"]
    cells = {(cell["family"], cell["regime"]) for cell in block["cells"]}
    families = {entry["family"] for entry in contract["formats"]}
    assert families, "no family is published; the join below would be vacuous"
    missing = sorted(
        (family, phase, CENSUS_PHASE_REGIMES[phase])
        for family in families
        for phase in CENSUS_PHASE_REGIMES
        if (family, CENSUS_PHASE_REGIMES[phase]) not in cells
    )
    assert not missing, (
        "the census drives a phase whose regime has no cell for these families: "
        f"{missing}; a per-(family, regime) expectation would be vacuous there"
    )


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
    """One served resident-FP8 dense record whose forward ran ``m`` rows."""
    return dict({"kind": "dense", "policy": "TESSERA_FP8:resident",
                 "symbol": "torch._scaled_mm", "decoder": "torch_window",
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
        runtime_image=contract["versions"]["default_serve_image"], execution_mode="eager")


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
