"""Allowable rungs are a RULE over run tables (lane schema v11, tessera#750).

A window-grammar rung mixes at most two adjacent column rates -- its run
table, ``grammar.rate_set`` of its root -- and the fused kernels are
instantiated per run table, the mix fraction and the placement being runtime
data.  So the kernel oracle attests a VARIANT (``tests/test_routed_fused_window.
py::test_random_mixes_inside_every_pair_decode_exactly`` and its dense twin),
and ``formats[].allowable_rungs`` states which rungs that makes allowable.  A
cell covers the run tables of its census rungs (``run_tables``, derived).

These pin: the rule the E4M3 row publishes and what it admits; the one
coverage predicate every consumer reads (``contract.cell_covers_rung``) and
its three consumers -- the export gate, the census join and ``attested_by``'s
predicate; that cell run tables are derived and refused when they drift; and
the rule's refusals by name.  Torch-free.
"""
from __future__ import annotations

import copy

import pytest

from tessera.serving.contract import (ALLOWABLE_RULES, cell_covers_rung, derived_cell_run_tables,
                                      format_entry, load_serving_contract, rung_allowable,
                                      rung_rates, validate_serving_contract)

E4M3 = "TESSERA_E4M3_K1"
ROUTED = "tessera_e4m3_k1_routed_moe_sm121_batch_resident"


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


def _cell(contract, cell_id):
    return next(c for c in contract["lane_eligibility"]["cells"] if c["id"] == cell_id)


def _row(contract, family=E4M3):
    return next(e for e in contract["formats"] if e["family"] == family)


def _rederive(contract):
    """Recompute every cell's ``run_tables`` after a rule mutation, the way a
    contract author would, so a test can isolate the rule's own refusal."""
    rows = {e["family"]: e for e in contract["formats"]}
    for cell in contract["lane_eligibility"]["cells"]:
        tables = derived_cell_run_tables(cell, rows[cell["family"]])
        if tables:
            cell["run_tables"] = tables
        else:
            cell.pop("run_tables", None)
    return contract


def test_the_e4m3_rule_admits_every_rung_of_its_attested_run_tables(contract):
    row = _row(contract)
    rule = row["allowable_rungs"]
    assert rule["rule"] == "window_rate_set" and rule["rule"] in ALLOWABLE_RULES
    assert rule["code_arity"] == 1
    assert rule["range_q256"] == row["reader_rate_range_q256"] and rule["step_q256"] == 1
    tables = [tuple(t) for t in rule["run_tables"]]
    # every one-run rate 1..8 and every adjacent pair: the oracle's variants
    assert tables == sorted({(r,) for r in range(1, 9)} | {(r, r + 1) for r in range(1, 8)})
    assert rule["excluded_run_tables"] == [] and rule["excluded_q256"] == []
    admitted = [q for q in range(256, 2049) if rung_allowable(row, q)]
    assert len(admitted) == 2048 - 256 + 1
    # the rule's wire is the one every attested E4M3 rung is stamped with
    for item in row["attested_wire"]:
        assert {k: v for k, v in item.items() if k != "q256"} == rule["wire"]


BF16 = "TESSERA_BF16_K1"


def test_the_bf16_rule_admits_every_rate_the_dense_launch_reads(contract):
    """Contract v51: Tessera-16's rule attests every one-run rate 1..14 and
    every adjacent pair, the rates the value library's dense launch reads,
    over 256..3584.  The range stops where the wire does: above rate 14 the
    exporter widens the window table to the rate, so 3585 would be cut on
    another wire (and rates 15 and 16 are excluded by geometry)."""
    row = _row(contract, BF16)
    rule = row["allowable_rungs"]
    assert rule["rule"] == "window_rate_set" and rule["code_arity"] == 1
    assert rule["range_q256"] == [256, 3584] and rule["step_q256"] == 1
    tables = [tuple(t) for t in rule["run_tables"]]
    assert tables == sorted({(r,) for r in range(1, 15)} | {(r, r + 1) for r in range(1, 14)})
    assert rule["excluded_run_tables"] == [] and rule["excluded_q256"] == []
    assert [q for q in range(256, 4097) if rung_allowable(row, q)] == list(range(256, 3585))
    assert rung_rates(row, 3584) == (14,) and rung_rates(row, 3585) == (14, 15)
    for item in row["attested_wire"]:
        assert {k: v for k, v in item.items() if k != "q256"} == rule["wire"]


def test_bf16_rungs_are_admitted_by_the_rule_but_covered_by_no_cell(contract):
    """The rule is family-wide; coverage is per cell.  Contract v59
    withdraws every BF16 cell, so the rule still admits 256..3584 and no
    cell covers any of it: a rung the rule admits is covered only by a
    cell of a structure whose census reached its table, and BF16
    publishes no cell."""
    row = _row(contract, BF16)
    by_name = {e["module_name_prefix"]: e for e in contract["native_extensions"]}
    requires = by_name["tessera_routed_fused_value"]["lane"]["requires"]
    dense_rates = set(requires["column_rates"])
    routed_rates = set(requires["column_rates_routed_moe"])
    assert dense_rates == set(range(1, 15)) and routed_rates == set(range(1, 9))
    assert {r for t in row["allowable_rungs"]["run_tables"] for r in t} == dense_rates
    assert [q for q in range(256, 4097) if rung_allowable(row, q)] == list(range(256, 3585))
    cells = [c for c in contract["lane_eligibility"]["cells"] if c["family"] == BF16]
    assert cells == []
    assert [q for q in range(256, 4097)
            if any(cell_covers_rung(cell, q, row) for cell in cells)] == []


def test_a_rung_resolves_to_its_run_table():
    row = {"grid": "E4M3", "native_terminal_q256": 2048}
    assert rung_rates(row, 1024) == (4,)
    assert rung_rates(row, 1025) == (4, 5)
    assert rung_rates(row, 1279) == (4, 5)
    assert rung_rates(row, 1280) == (5,)
    assert rung_rates(row, 256) == (1,) and rung_rates(row, 2048) == (8,)


def test_a_cell_covers_its_census_rungs_and_the_allowable_rungs_of_their_run_tables(contract):
    """The routed E4M3 cell's census rungs are 832..1088; their run tables are
    [3, 4], [4] and [4, 5], so it covers every rung of 769..1279 and nothing
    at rate 3 alone (768) or at rate 5 and above."""
    row, cell = _row(contract), _cell(contract, ROUTED)
    assert cell["run_tables"] == [[3, 4], [4], [4, 5]]
    covered = [q for q in range(256, 2049) if cell_covers_rung(cell, q, row)]
    assert covered == list(range(769, 1280))
    for q in (768, 1280, 1300, 1536, 2048, 256):
        assert not cell_covers_rung(cell, q, row), q
    # without its family's row only the census rungs are covered
    assert [q for q in range(256, 2049) if cell_covers_rung(cell, q, None)] == cell["rungs_q256"]


def test_the_e4m3_dense_cells_cover_every_rung_of_the_rule(contract):
    """Contract v53 (#750, the T-8 dense census): stub t8d1 carries one E4M3
    dense rung of every run table the rule admits, [1]..[8] and the seven
    pairs between them, so the two GLM-image dense resident cells derive all
    fifteen tables and cover the rule's whole range, 256..2048, where v52
    covered 769..1279.  The routed cells stay at stub B's tables, and the
    dense cells' nightly-runtime twins, which name a runtime this census did
    not serve, do not move."""
    row = _row(contract)
    by_id = {c["id"]: c for c in contract["lane_eligibility"]["cells"]}
    for regime in ("decode", "batch"):
        dense = by_id[f"tessera_e4m3_k1_dense_sm121_{regime}_resident"]
        assert dense["run_tables"] == row["allowable_rungs"]["run_tables"]
        assert [q for q in range(128, 2561)
                if cell_covers_rung(dense, q, row)] == list(range(256, 2049))
        (twin,) = [c for c in contract["lane_eligibility"]["cells"]
                   if c["id"].startswith(f"tessera_e4m3_k1_dense_sm121_{regime}_resident_runtime_")]
        assert twin["run_tables"] == [[3, 4], [4], [4, 5]]
        routed = by_id[f"tessera_e4m3_k1_routed_moe_sm121_{regime}_resident"]
        assert routed["run_tables"] == [[3, 4], [4], [4, 5]]


def test_every_cell_publishes_exactly_its_derived_run_tables(contract):
    rows = {e["family"]: e for e in contract["formats"]}
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell.get("run_tables", []) == derived_cell_run_tables(cell, rows[cell["family"]]), \
            cell["id"]


def test_a_cell_run_table_with_no_census_rung_is_refused(contract):
    bad = copy.deepcopy(contract)
    _cell(bad, ROUTED)["run_tables"].append([5])
    with pytest.raises(ValueError, match="run_tables"):
        validate_serving_contract(bad)


def test_a_cell_that_omits_the_run_tables_its_census_covers_is_refused(contract):
    bad = copy.deepcopy(contract)
    del _cell(bad, ROUTED)["run_tables"]
    with pytest.raises(ValueError, match="run_tables"):
        validate_serving_contract(bad)


def test_an_excluded_run_table_leaves_its_census_rungs_covered_and_nothing_else(contract):
    """Exclusion is data: excluding [4, 5] drops it from the derived cell
    tables, the census rung 1088 stays covered by enumeration, and 1100 is
    no longer covered anywhere."""
    table = _rederive(copy.deepcopy(contract))
    rule = _row(table)["allowable_rungs"]
    rule["run_tables"].remove([4, 5])
    rule["excluded_run_tables"] = [[4, 5]]
    _rederive(table)
    validate_serving_contract(table)
    row, cell = _row(table), _cell(table, ROUTED)
    assert cell["run_tables"] == [[3, 4], [4]]
    assert cell_covers_rung(cell, 1088, row)
    assert not cell_covers_rung(cell, 1100, row) and not rung_allowable(row, 1100)


def test_an_excluded_rung_is_not_allowable(contract):
    table = copy.deepcopy(contract)
    _row(table)["allowable_rungs"]["excluded_q256"] = [1100]
    validate_serving_contract(table)
    row, cell = _row(table), _cell(table, ROUTED)
    assert not rung_allowable(row, 1100) and not cell_covers_rung(cell, 1100, row)
    assert cell_covers_rung(cell, 1101, row)


def test_an_unknown_rule_admits_nothing_and_is_refused(contract):
    row = copy.deepcopy(_row(contract))
    row["allowable_rungs"]["rule"] = "every_rung"
    assert not any(rung_allowable(row, q) for q in (769, 1024, 1100))
    bad = copy.deepcopy(contract)
    _row(bad)["allowable_rungs"]["rule"] = "every_rung"
    with pytest.raises(ValueError, match="rule"):
        validate_serving_contract(bad)


@pytest.mark.parametrize("field,value,match", [
    ("run_tables", [[3], [3, 5]], "adjacent"),
    ("run_tables", [[4], [3]], "ascending"),
    ("run_tables", [[9]], "adjacent"),
    ("excluded_run_tables", [[4]], "both attested and excluded"),
    ("range_q256", [128, 2048], "reader"),
    ("excluded_q256", [99999], "excluded_q256"),
    ("evidence", ["/abs/path.md"], "repository path"),
    ("evidence", [], "evidence"),
    ("code_arity", 2, "code_arity"),
])
def test_a_malformed_rule_is_refused_by_name(contract, field, value, match):
    bad = copy.deepcopy(contract)
    _row(bad)["allowable_rungs"][field] = value
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(bad)


def test_a_rule_wire_that_disagrees_with_a_covered_stamp_is_refused(contract):
    bad = copy.deepcopy(contract)
    _row(bad)["allowable_rungs"]["wire"]["seed"] = 7
    with pytest.raises(ValueError, match="one wire"):
        validate_serving_contract(bad)


def test_a_rule_wire_the_route_does_not_decode_is_refused(contract):
    bad = copy.deepcopy(contract)
    _row(bad)["allowable_rungs"]["wire"]["body"] = "tcq"
    with pytest.raises(ValueError, match="does not decode"):
        validate_serving_contract(bad)


def test_a_run_table_no_rung_of_the_range_reaches_is_refused(contract):
    bad = copy.deepcopy(contract)
    _row(bad)["allowable_rungs"]["range_q256"] = [768, 1280]
    with pytest.raises(ValueError, match="no rung"):
        validate_serving_contract(bad)


def test_the_routed_export_gate_admits_every_rung_the_cells_cover(contract):
    """The gate reads ``cell_covers_rung``: a routed E4M3 stack at q256 1100
    (run table [4, 5], census rung 1088) is admitted, 1300 ([5, 6], no
    census) and 768 ([3], no census) are refused by the cells' ids."""
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, refuse_unserveable_wire

    def gate(q):
        return refuse_unserveable_wire("E4M3", q, "WINDOW", "CHANNEL", family="TESSERA_FP8",
                                       span=1, target="stack.probe",
                                       structure=STRUCTURE_ROUTED_MOE, contract=contract)

    for q in (769, 1000, 1100, 1279):
        assert gate(q) == "TESSERA_FP8", q
    for q in (768, 1300):
        with pytest.raises(ValueError) as caught:
            gate(q)
        assert ROUTED in str(caught.value) and "run tables" in str(caught.value)


def test_the_census_join_covers_a_record_at_a_rule_rung(contract):
    """A served record at q256 1100 joins the routed cell through its run
    table; with the rule's row withheld the same record is unattested."""
    from tessera.serving.census import cell_launch_agreement

    cell = copy.deepcopy(_cell(contract, ROUTED))
    pair = (cell["executes"][0]["symbol"], cell["executes"][0]["decoder"])
    records = {"prefill": {"model.layers.3.mlp.experts": {
        "kind": "moe", "policy": "TESSERA_FP8:resident", "symbol": pair[0],
        "decoder": pair[1], "shape": "M64:N4096:K4096"}}}
    kwargs = dict(cells=[cell], phase_regimes={"prefill": "batch"}, platform="sm_121",
                  structure="routed_moe", rungs_by_module={"model.layers.3.mlp.experts": 1100},
                  families_by_route={"TESSERA_FP8": E4M3},
                  runtime_image=cell["runtime"]["image"], execution_mode="eager")
    block, problems = cell_launch_agreement(records, formats=contract["formats"], **kwargs)
    assert problems == [] and block["phases"]["prefill"]["covered_by_cell"] == 1, block
    block, problems = cell_launch_agreement(records, formats=[], **kwargs)
    assert block["phases"]["prefill"]["covered_by_cell"] == 0
    assert block["phases"]["prefill"]["unattested"] == 1


def test_format_entry_finds_the_row(contract):
    assert format_entry(E4M3, contract) is _row(contract)
    assert format_entry("TESSERA_NOPE", contract) is None


def test_a_covered_rung_reaches_the_lanes_of_the_census_rung_that_shares_its_run_table(contract):
    """The run table is a sound unit only if a cell's launches are a function
    of it: a rung the rule lets a cell cover must reach exactly the lanes the
    census rung of the same table reaches, so covering it widens no cell's
    ``executes``.  Lane reach is decided on the rates and the wire
    (``contract._lanes_a_rung_reaches``); the rule's wire is validated equal
    to every census stamp it covers, and this holds the rest."""
    from tessera.serving.contract import _FAMILY_TO_ROUTE, _lanes_a_rung_reaches

    rows = {e["family"]: e for e in contract["formats"]}
    checked = 0
    for cell in contract["lane_eligibility"]["cells"]:
        row = rows[cell["family"]]
        rule = row.get("allowable_rungs")
        if not cell.get("run_tables") or not isinstance(rule, dict):
            continue
        route = _FAMILY_TO_ROUTE[cell["family"]]
        stamps = {int(w["q256"]): w for w in row["attested_wire"]}
        census = {}
        for rung in cell["rungs_q256"]:
            census.setdefault(rung_rates(row, rung), _lanes_a_rung_reaches(
                route, contract, stamps[int(rung)], rung_rates(row, rung), row["grid"],
                cell["structure"]))
        low, high = rule["range_q256"]
        for q in range(low, high + 1, rule["step_q256"]):
            if not cell_covers_rung(cell, q, row):
                continue
            rates = rung_rates(row, q)
            assert _lanes_a_rung_reaches(route, contract, rule["wire"], rates, row["grid"],
                                         cell["structure"]) == census[rates], (cell["id"], q)
            checked += 1
    assert checked >= 4 * 511, checked
