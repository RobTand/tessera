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

_FAMILY_RUNGS = {"TESSERA_E2M1_K2": [896], "TESSERA_E4M3_K1": [1024]}


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


def test_two_families_at_one_rung_each(contract):
    formats = {entry["family"]: entry for entry in contract["formats"]}
    assert sorted(formats) == ["TESSERA_E2M1_K2", "TESSERA_E4M3_K1"]
    for family, rungs in _FAMILY_RUNGS.items():
        assert formats[family]["candidate_rungs_q256"] == rungs
        assert formats[family]["reader_rate_range_q256"] == [rungs[0], rungs[0]]
        assert formats[family]["residency_modes"] == ["resident", "streamed"]


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
    from tessera.serving import fp8_route, nvfp4_route

    by_family = {
        "TESSERA_E2M1_K2": nvfp4_route.ACTIVATION_CONTRACT,
        "TESSERA_E4M3_K1": fp8_route.ACTIVATION_CONTRACT,
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
