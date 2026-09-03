"""The runtime contract this plugin packages, and what it may not be made to say.

Principle 14: a claim about what a serving runtime DOES is derived from a
machine-readable table the runtime publishes.  This package IS that runtime for
Tessera bytes, so the table travels inside it and a producer reads it through
``importlib.resources``.  Two layers of test, because one alone is not enough:

* the packaged VALIDATOR refuses cells that are structurally wrong -- a rung the
  family does not publish, an activation contract that is not the route's, a
  cell with no serve flag, a cell that forgets it is plugin-gated, a structure
  the dispatch refuses;
* a LAWS TABLE below pins the four cells field for field, because no generic
  rule knows which rungs a receipt covered.  ``rungs_q256: [512]`` is a
  perfectly well-formed cell and a false claim.
"""
from __future__ import annotations

import copy

import pytest

from tessera.serving.contract import (
    CONTRACT_SCHEMA,
    LANE_ELIGIBILITY_SCHEMA,
    REQUIRES_PLUGIN,
    contract_path,
    load_serving_contract,
    validate_serving_contract,
)

#: The four cells the 2026-09-02 GB10 Tessera receipts cover, field for field:
#: Qwen3-0.6B on the E2M1x2 cap wire (q256 = 896) and on the E4M3 window wire
#: (q256 = 1024), both residency modes, eager and compiled, every dense Linear
#: on ``torch._scaled_mm``.  Widening ANY value here without a new receipt is
#: the failure this pins.
_CELL_LAWS: dict[str, dict[str, object]] = {
    "tessera_e2m1_k2_dense_sm121_decode_scaled_mm_w4a4": {
        "platform": "sm_121", "family": "TESSERA_E2M1_K2", "structure": "dense",
        "regime": "decode", "rungs_q256": [896],
        "activation_contract": "e2m1_group16_ue4m3_static",
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"],
        "predicates": [],
    },
    "tessera_e2m1_k2_dense_sm121_batch_scaled_mm_w4a4": {
        "platform": "sm_121", "family": "TESSERA_E2M1_K2", "structure": "dense",
        "regime": "batch", "rungs_q256": [896],
        "activation_contract": "e2m1_group16_ue4m3_static",
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"],
        "predicates": [],
    },
    "tessera_e4m3_k1_dense_sm121_decode_scaled_mm_w8a8": {
        "platform": "sm_121", "family": "TESSERA_E4M3_K1", "structure": "dense",
        "regime": "decode", "rungs_q256": [1024],
        "activation_contract": "fp8_per_token_dynamic",
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"],
        "predicates": [],
    },
    "tessera_e4m3_k1_dense_sm121_batch_scaled_mm_w8a8": {
        "platform": "sm_121", "family": "TESSERA_E4M3_K1", "structure": "dense",
        "regime": "batch", "rungs_q256": [1024],
        "activation_contract": "fp8_per_token_dynamic",
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"],
        "predicates": [],
    },
}

#: The rungs each family ATTESTS -- rungs a container receipt covers.
#: ``TESSERA_BF16_K1`` attests none: the route exists and the reader takes a
#: continuous range (below), but no served census and no served KL against the
#: exporter's plain-BF16 twin exist yet, so every BF16 rung resolves
#: ``unattested``.  An empty list here is the honest state and is deliberately
#: not the same thing as an absent family.
_FAMILY_RUNGS = {"TESSERA_E2M1_K2": [896], "TESSERA_E4M3_K1": [1024],
                 "TESSERA_BF16_K1": []}


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


def _cells(contract):
    return {cell["id"]: cell for cell in contract["lane_eligibility"]["cells"]}


# --- what the packaged table says --------------------------------------------

def test_the_packaged_contract_loads_and_validates(contract):
    assert contract["schema"] == CONTRACT_SCHEMA
    assert contract["lane_eligibility"]["schema"] == LANE_ELIGIBILITY_SCHEMA
    assert contract["quant_method"]["canonical"] == REQUIRES_PLUGIN


def test_the_file_is_reachable_through_importlib_resources():
    """A producer resolves it by package, never by repo-root arithmetic --
    and without importing torch (``contract.py`` imports only ``json``; the
    import graph itself is pinned in ``test_no_gridbook_import``)."""
    path = contract_path()
    assert path.is_file()
    assert path.name == "runtime_contract.json"


def test_three_families_and_what_each_one_attests(contract):
    formats = {entry["family"]: entry for entry in contract["formats"]}
    assert sorted(formats) == ["TESSERA_BF16_K1", "TESSERA_E2M1_K2", "TESSERA_E4M3_K1"]
    for family, rungs in _FAMILY_RUNGS.items():
        assert formats[family]["attested_rungs_q256"] == rungs
        assert formats[family]["residency_modes"] == ["resident", "streamed"]


def test_a_family_with_no_receipt_publishes_no_cell(contract):
    """Absence resolves ``unattested``; it is never invented into a cell.

    ``TESSERA_BF16_K1`` is a served ROUTE with no served MEASUREMENT, and the
    two are different claims.  Publishing a cell for it would attest a route
    status nobody has observed -- which is the failure principle 14 exists for,
    and the one a route module is most tempting to commit the day it is
    written.
    """
    attested = {c["family"] for c in contract["lane_eligibility"]["cells"]}
    unattested = {f for f, rungs in _FAMILY_RUNGS.items() if not rungs}
    assert unattested, "this test is vacuous if every family is attested"
    assert not (attested & unattested), (
        f"{sorted(attested & unattested)} publish a lane_eligibility cell but attest no rung")


#: MEASURED, not chosen: each rate was encoded and taken through the route's own
#: load path (``parse_tessera_blob_for_scheme`` then ``prepare_*``), and this is
#: the accepted set.  Two mechanisms bound it -- the trellis grammar's shaped
#: domain at both ends on E4M3, and on E2M1x2 the grammar above plus the native
#: decoder's span-2-TCQ-only support below.
_READER_RATES = {
    "TESSERA_E2M1_K2": ("E2M1x2", [896, 896], 1),
    "TESSERA_E4M3_K1": ("E4M3", [256, 2048], 1),
    # ``experiments/bf16_reader_rate_range.py``: 25 rungs, every integer rate
    # 1..16 plus nine of the non-integer rungs a Bresenham schedule makes, each
    # encoded and taken through the route's load path.  The one candidate that
    # refused (777 over 128 columns) refused on GEOMETRY -- "it needs 9/2
    # columns at rate 4" -- and loads at 512 columns, so it bounds nothing.
    "TESSERA_BF16_K1": ("BF16", [256, 4096], 1),
}


def test_the_reader_range_is_what_the_decoder_takes(contract):
    formats = {entry["family"]: entry for entry in contract["formats"]}
    for family, (grid, span, step) in _READER_RATES.items():
        entry = formats[family]
        assert entry["grid"] == grid, family
        assert entry["reader_rate_range_q256"] == span, (
            f"{family}: the published reader range is not the measured one")
        assert entry["reader_rate_step_q256"] == step, family
        assert entry["reader_rate_bound"], f"{family}: no mechanism named for the bound"


def test_the_deprecated_alias_is_carried_and_must_agree(contract):
    """``candidate_rungs_q256`` is kept so the rename stays ADDITIVE.

    PrismaQuant reads this packaged file through ``importlib.resources`` and its
    ``load_published_formats`` was written against schema v1; dropping a key it
    reads by name, while the ``schema`` string still says v1, would be the same
    "current and wrong" fault this change exists to close.  So the alias stays
    until the schema moves, and it may not disagree with the field it aliases.
    """
    for entry in contract["formats"]:
        assert entry["candidate_rungs_q256"] == entry["attested_rungs_q256"], (
            f"{entry['family']}: the alias has drifted from what it aliases")

    broken = copy.deepcopy(contract)
    broken["formats"][0]["candidate_rungs_q256"] = [
        broken["formats"][0]["attested_rungs_q256"][0] + 1]
    with pytest.raises(ValueError, match="DEPRECATED ALIAS"):
        validate_serving_contract(broken)

    # and it is genuinely OPTIONAL: a document without it still validates
    without = copy.deepcopy(contract)
    for entry in without["formats"]:
        entry.pop("candidate_rungs_q256")
    validate_serving_contract(without)


def test_an_attested_rung_outside_the_reader_range_is_refused(contract):
    """An attested rung is one that WAS served; it cannot be unreadable."""
    broken = copy.deepcopy(contract)
    unreadable = [broken["formats"][0]["reader_rate_range_q256"][1] + 1]
    broken["formats"][0]["attested_rungs_q256"] = unreadable
    # the alias moves with it, or the alias check fires first and this stops
    # testing the range at all
    broken["formats"][0]["candidate_rungs_q256"] = list(unreadable)
    with pytest.raises(ValueError, match="not one the reader accepts"):
        validate_serving_contract(broken)


def test_an_empty_or_stepless_reader_range_is_refused(contract):
    for field, value, message in (("reader_rate_range_q256", [2048, 256], "which is empty"),
                                  ("reader_rate_step_q256", 0, "must be >= 1")):
        broken = copy.deepcopy(contract)
        broken["formats"][1][field] = value
        with pytest.raises(ValueError, match=message):
            validate_serving_contract(broken)


def test_the_reader_grid_resolves_by_route_AND_grid(contract):
    """``TESSERA_NVFP4`` holds two grids and the contract describes one.

    Resolving by route alone would hand an ``E2M1`` checkpoint the ``E2M1x2``
    numbers, which is exactly the kind of near-miss this contract exists to
    stop.
    """
    from tessera.serving.contract import reader_accepts, reader_rate_grid

    assert reader_rate_grid("TESSERA_FP8", "E4M3", contract) == (
        "TESSERA_E4M3_K1", 256, 2048, 1)
    assert reader_rate_grid("TESSERA_NVFP4", "E2M1x2", contract) == (
        "TESSERA_E2M1_K2", 896, 896, 1)
    assert reader_rate_grid("TESSERA_NVFP4", "E2M1", contract) is None, (
        "the arity-1 E2M1 grid has no published range and must not borrow one")
    assert reader_rate_grid("TESSERA_FP8", "E2M1x2", contract) is None

    assert reader_accepts(256, 256, 2048, 1) and reader_accepts(2048, 256, 2048, 1)
    assert not reader_accepts(255, 256, 2048, 1)
    assert not reader_accepts(2049, 256, 2048, 1)
    assert reader_accepts(1024, 256, 2048, 128)
    assert not reader_accepts(1025, 256, 2048, 128), "the step is part of the set"


def test_the_four_cells_are_pinned_field_for_field(contract):
    cells = _cells(contract)
    assert sorted(cells) == sorted(_CELL_LAWS)
    for cell_id, laws in _CELL_LAWS.items():
        got = cells[cell_id]
        for field, value in laws.items():
            assert got[field] == value, f"{cell_id}.{field}"


def test_every_cell_is_backed_with_a_serve_flag_and_plugin_gated(contract):
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["route_status"] == "backed_with_serve_flag"
        assert cell["qualification"] == "device_qualified"
        assert cell["requires_plugin"] == "tessera"
        assert cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident|streamed"]


def test_the_table_is_dense_only(contract):
    """No cell mentions routed MoE, and there is no expert-parallel claim.

    A cell is attested by a served measurement, and no served measurement
    covers routed-MoE experts: the expert route is not built, and the dispatch
    refuses such a layer by name rather than returning ``None``.  So the table
    says ``dense`` and nothing else, and the absence resolves ``unattested``
    rather than ``denied``.
    """
    block = contract["lane_eligibility"]
    assert block["structures"] == ["dense"]
    assert {cell["structure"] for cell in block["cells"]} == {"dense"}
    assert "routed_moe" not in repr(contract).lower()
    assert contract["expert_parallel"]["units"] == []


def test_each_cell_executes_the_contract_its_route_module_exposes(contract):
    """The cell's ``activation_contract`` is the route's own constant.

    Imported lazily: the route modules import torch, and the contract half of
    this file must stay readable where it is not installed.
    """
    from tessera.serving import bf16_route, fp8_route, nvfp4_route

    by_family = {
        "TESSERA_E2M1_K2": nvfp4_route.ACTIVATION_CONTRACT,
        "TESSERA_E4M3_K1": fp8_route.ACTIVATION_CONTRACT,
        "TESSERA_BF16_K1": bf16_route.ACTIVATION_CONTRACT,
    }
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["activation_contract"] == by_family[cell["family"]]


def test_no_unit_admits_a_world_size_above_one(contract):
    units = contract["tensor_parallel"]["units"]
    assert units, "the contract makes a tensor-parallel claim"
    assert {u["unit"] for u in units} == set(_FAMILY_RUNGS)
    for unit in units:
        assert unit["max_world_size"] == 1


# --- what the validator refuses ----------------------------------------------

def _mutated(contract, mutate):
    copy_ = copy.deepcopy(contract)
    mutate(copy_)
    return copy_


def _set_routed_moe_structure(c):
    c["lane_eligibility"]["structures"] = ["dense", "routed_moe"]


def _empty_serve_flags(c):
    c["lane_eligibility"]["cells"][0]["requires_serve_flags"] = []


def _wrong_activation_contract(c):
    c["lane_eligibility"]["cells"][0]["activation_contract"] = "fp8_per_token_dynamic"


def _drop_requires_plugin(c):
    del c["lane_eligibility"]["cells"][0]["requires_plugin"]


def _foreign_requires_plugin(c):
    c["lane_eligibility"]["cells"][0]["requires_plugin"] = "gridbook"


def _unpublished_rung(c):
    c["lane_eligibility"]["cells"][0]["rungs_q256"] = [512]


@pytest.mark.parametrize("mutate, match", [
    (_set_routed_moe_structure, "routed-MoE"),
    (_empty_serve_flags, "declared residency"),
    (_wrong_activation_contract, "route executes"),
    (_drop_requires_plugin, r"missing \['requires_plugin'\]"),
    (_foreign_requires_plugin, "plugin-gated"),
    (_unpublished_rung, "the family does not publish"),
])
def test_the_validator_refuses_a_contract_this_package_would_not_honour(
        contract, mutate, match):
    bad = _mutated(contract, mutate)
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(bad)


def test_a_cell_may_not_declare_a_structure_the_dispatch_refuses(contract):
    """Both halves of the routed-MoE refusal: the structure list, and a cell
    naming a structure that is not in it."""
    bad = _mutated(contract,
                   lambda c: c["lane_eligibility"]["cells"][0].__setitem__("structure", "routed_moe"))
    with pytest.raises(ValueError, match="is not declared"):
        validate_serving_contract(bad)


def test_an_expert_parallel_claim_is_refused(contract):
    bad = _mutated(contract, lambda c: c["expert_parallel"]["units"].append(
        {"unit": "TESSERA_E2M1_K2", "kind": "tessera_wire_family", "max_world_size": 2}))
    with pytest.raises(ValueError, match="expert_parallel.units must be empty"):
        validate_serving_contract(bad)


def test_a_sharded_unit_is_refused(contract):
    bad = _mutated(contract,
                   lambda c: c["tensor_parallel"]["units"][0].__setitem__("max_world_size", 2))
    with pytest.raises(ValueError, match="per-rank wires"):
        validate_serving_contract(bad)
