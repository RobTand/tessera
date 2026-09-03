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

import json
import re
from pathlib import Path

import pytest

from tessera.serving.contract import (
    CENSUS_PHASE_REGIMES,
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
