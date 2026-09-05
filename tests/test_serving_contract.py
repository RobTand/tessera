"""The runtime contract this plugin packages, and what it may not be made to say.

Runtime attestation: a claim about what a serving runtime DOES is derived from a
machine-readable table the runtime publishes.  This package IS that runtime for
Tessera bytes, so the table travels inside it and a producer reads it through
``importlib.resources``.  Two layers of test, because one alone is not enough:

* the packaged VALIDATOR refuses cells that are structurally wrong -- a rung the
  family does not publish, an activation contract that is not the route's, a
  cell with no serve flag, a cell that forgets it is plugin-gated, a structure
  the dispatch refuses;
* a LAWS TABLE below pins the measured cells field for field, because no generic
  rule knows which rungs a receipt covered.  ``rungs_q256: [512]`` is a
  perfectly well-formed cell and a false claim.
"""
from __future__ import annotations

import copy
from pathlib import Path
import re

import pytest

from tessera.serving.contract import (
    CENSUS_PHASE_REGIMES,
    CONTRACT_SCHEMA,
    LANE_ELIGIBILITY_SCHEMA,
    REQUIRES_PLUGIN,
    contract_path,
    load_serving_contract,
    validate_serving_contract,
)

ROOT = Path(__file__).resolve().parents[1]


#: Placeholder the LAWS table carries for the dense image; the pinning test
#: resolves it through :func:`_dense_runtime_image` at test time, so a
#: checkout without ``docs/`` fails one test instead of collection.
_DENSE_IMAGE_FROM_RECEIPT = "<the image the migration receipt records>"


def _dense_runtime_image() -> str:
    """The vanilla vLLM image the eight dense cells were measured on, read
    from the receipt that records it rather than copied here: the pin lives
    in ``runtime_contract.json`` and ``tests/test_runtime_image_pin.py``
    refuses a second copy of its digest under ``tests/``.  The migration
    receipt is exempt (it records), so the LAWS table reads it from there."""
    receipt = ROOT / "docs/measurements/runtime-scope-migration-2026-09-04.md"
    found = sorted(set(re.findall(r"vllm/vllm-openai@sha256:[0-9a-f]{64}", receipt.read_text())))
    assert len(found) == 1, found
    return found[0]


def _resolved(laws: dict[str, object]) -> dict[str, object]:
    runtime = laws["runtime"]
    if runtime["image"] is _DENSE_IMAGE_FROM_RECEIPT:
        laws = {**laws, "runtime": {**runtime, "image": _dense_runtime_image()}}
    return laws


#: The toolchain the dense receipts record, verbatim:
#: ``tessera-window-gemv-served-2026-09-03.md`` :75 and
#: ``tessera-bf16-route-served-2026-09-02.md`` :39 (vLLM 0.28.0, torch
#: 2.13.0+cu130).  The v5 global block wrote ``2.13.0`` without the suffix.
_DENSE_RUNTIME = {"image": _DENSE_IMAGE_FROM_RECEIPT, "execution_modes": ["eager", "compiled"],
                  "vllm": "0.28.0", "torch": "2.13.0+cu130"}

#: The eight preserved dense cells the served Tessera receipts cover:
#: Qwen3-0.6B on the E2M1x2 cap wire (q256 = 896), on the E4M3 window wire
#: (q256 = 1024) and on the BF16 window wire (q256 = 1792), every dense Linear,
#: eager and compiled -- the 2026-09-02 receipts for the six that were here
#: before, plus ``/home/rob/tessera-runs/ts104/census-R1024-readable.json``
#: (2026-09-03, streamed, eager) for what the E4M3 wire executes once the
#: window-GEMV lane is reachable.  Widening ANY value here without a new
#: receipt is the failure this pins.
#:
#: The E4M3 family carries FOUR cells because its launches differ by RESIDENCY:
#: both window routes set ``layer.tessera_gemv = None`` in ``resident``, so the
#: lane exists in ``streamed`` alone and the decode regime executes two
#: different things under one rung (#111).  The other two families execute one
#: launch in both residencies and keep one cell per regime -- and BF16 keeps
#: the torch decode because its attested rung 1792 is root 7, outside the
#: lane's rates, so its own GEMV lane is unreachable there.
_CELL_LAWS: dict[str, dict[str, object]] = {
    "tessera_e2m1_k2_dense_sm121_decode": {
        "platform": "sm_121", "family": "TESSERA_E2M1_K2", "structure": "dense",
        "regime": "decode", "rungs_q256": [896],
        "activation_contract": "e2m1_group16_ue4m3_static",
        "executes": [{"symbol": "torch._scaled_mm", "decoder": "native_span2"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"],
        "predicates": [],
        "runtime": _DENSE_RUNTIME,
    },
    "tessera_e2m1_k2_dense_sm121_batch": {
        "platform": "sm_121", "family": "TESSERA_E2M1_K2", "structure": "dense",
        "regime": "batch", "rungs_q256": [896],
        "activation_contract": "e2m1_group16_ue4m3_static",
        "executes": [{"symbol": "torch._scaled_mm", "decoder": "native_span2"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"],
        "predicates": [],
        "runtime": _DENSE_RUNTIME,
    },
    "tessera_e4m3_k1_dense_sm121_decode_resident": {
        "platform": "sm_121", "family": "TESSERA_E4M3_K1", "structure": "dense",
        "regime": "decode", "rungs_q256": [1024],
        "activation_contract": "fp8_per_token_dynamic",
        "executes": [{"symbol": "torch._scaled_mm", "decoder": "torch_window"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
        "predicates": [],
        "runtime": _DENSE_RUNTIME,
    },
    # THE CELL THE CENSUS BOUGHT.  Before #111 this rung's decode regime
    # published the materialised pair, in every case, on a document whose own
    # receipt records ``tessera_window_gemv::gemv`` on 112 of 112 modules.
    "tessera_e4m3_k1_dense_sm121_decode_streamed": {
        "platform": "sm_121", "family": "TESSERA_E4M3_K1", "structure": "dense",
        "regime": "decode", "rungs_q256": [1024],
        "activation_contract": "fp8_per_token_dynamic",
        "executes": [{"symbol": "tessera_window_gemv::gemv", "decoder": "window_gemv"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=streamed"],
        "predicates": [],
        "runtime": _DENSE_RUNTIME,
    },
    "tessera_e4m3_k1_dense_sm121_batch_resident": {
        "platform": "sm_121", "family": "TESSERA_E4M3_K1", "structure": "dense",
        "regime": "batch", "rungs_q256": [1024],
        "activation_contract": "fp8_per_token_dynamic",
        "executes": [{"symbol": "torch._scaled_mm", "decoder": "torch_window"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
        "predicates": [],
        "runtime": _DENSE_RUNTIME,
    },
    # TWO LAUNCHES, because the batch regime is every M > 1 forward and not
    # only a first prefill (``contract.CENSUS_PHASE_REGIMES`` says so in its
    # own words).  The census drives a 64-row prefill, where the tile comes
    # off the lane's kernel decode under ``torch._scaled_mm`` (112 of 112 in
    # the R1024 receipt) -- but the same regime holds the 2-to-8-row forwards,
    # where the lane serves its own ``gemv`` exactly as it does at one row.
    # Publishing the prefill launch alone was #111's defect one regime over:
    # true of the shape the census drove, false of the runtime.
    "tessera_e4m3_k1_dense_sm121_batch_streamed": {
        "platform": "sm_121", "family": "TESSERA_E4M3_K1", "structure": "dense",
        "regime": "batch", "rungs_q256": [1024],
        "activation_contract": "fp8_per_token_dynamic",
        "executes": [{"symbol": "tessera_window_gemv::gemv", "decoder": "window_gemv"},
                     {"symbol": "torch._scaled_mm", "decoder": "window_gemv"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=streamed"],
        "predicates": [],
        "runtime": _DENSE_RUNTIME,
    },
    "tessera_bf16_k1_dense_sm121_decode": {
        "platform": "sm_121", "family": "TESSERA_BF16_K1", "structure": "dense",
        "regime": "decode", "rungs_q256": [1792],
        "activation_contract": "bf16_unquantized",
        "executes": [{"symbol": "torch.mm", "decoder": "torch_window"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"],
        "predicates": [],
        "runtime": _DENSE_RUNTIME,
    },
    "tessera_bf16_k1_dense_sm121_batch": {
        "platform": "sm_121", "family": "TESSERA_BF16_K1", "structure": "dense",
        "regime": "batch", "rungs_q256": [1792],
        "activation_contract": "bf16_unquantized",
        "executes": [{"symbol": "torch.mm", "decoder": "torch_window"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera",
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"],
        "predicates": [],
        "runtime": _DENSE_RUNTIME,
    },
}

#: The rungs each family ATTESTS -- rungs a container receipt covers.  Each
#: family's reader takes a far wider range (below); attestation is the narrower
#: claim and the only one a cell may make.  ``TESSERA_BF16_K1`` attested none
#: until 2026-09-02, when four route censuses and a served KL against the
#: exporter's plain-BF16 twin covered q256 = 1792
#: (``docs/measurements/tessera-bf16-route-served-2026-09-02.md``); an empty
#: list here remains the honest state for a family without a receipt, and is
#: deliberately not the same thing as an absent family.
_FAMILY_RUNGS = {"TESSERA_E2M1_K2": [896], "TESSERA_E4M3_K1": [1024],
                 "TESSERA_BF16_K1": [1792]}

# The full LFM receipt pins this measured pair, not a capability-derived roster.
for _regime in ("decode", "batch"):
    _CELL_LAWS[f"tessera_e4m3_k1_routed_moe_sm121_{_regime}_resident"] = {
        "platform": "sm_121", "family": "TESSERA_E4M3_K1", "structure": "routed_moe",
        "regime": _regime, "rungs_q256": [1024],
        "activation_contract": "fp8_per_token_dynamic",
        "executes": [{"symbol": "vllm.fused_moe.modular_kernel", "decoder": "torch_materialize_stock"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera", "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
        "predicates": [], "runtime": {
            "image": "eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c",
            "execution_modes": ["eager"],
            # docs/measurements/census/lfm25-8b-a1b-served-r4.json ``versions``;
            # tests/test_lfm_measured_cells.py ties the cells to that receipt.
            "vllm": "0.28.1rc1.dev397+gfd4a15126.d20260904", "torch": "2.13.0+cu130"}}


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
    from tessera.serving.lane import MODES

    formats = {entry["family"]: entry for entry in contract["formats"]}
    assert sorted(formats) == ["TESSERA_BF16_K1", "TESSERA_E2M1_K2", "TESSERA_E4M3_K1"]
    for family, rungs in _FAMILY_RUNGS.items():
        assert formats[family]["attested_rungs_q256"] == rungs
        # Derived from the tuple the serve gates on, not restated: a row may
        # claim a subset (a family served in one residency only), never
        # anything outside it -- and the validator enforces exactly that.
        assert sorted(formats[family]["residency_modes"]) == sorted(MODES)


def test_a_cell_exists_exactly_where_a_receipt_does(contract):
    """Both directions of principle 14, on the shipped file.

    Until 2026-09-02 this test asserted only one of them -- that
    ``TESSERA_BF16_K1``, then receiptless, published no cell.  That was true
    only while its receipt was missing and went vacuous the moment one
    existed.  The pair below does not: a family that attests rungs MUST carry a
    cell (or the receipt bought nothing), and a family that attests none MUST
    NOT (or a route status nobody observed is published).  The second branch is
    the old test, now reached exactly when a family is unattested -- which is
    where it belongs, because it is the failure a route module is most tempting
    to commit the day it is written.
    """
    for family, rungs in _FAMILY_RUNGS.items():
        cells = [c for c in contract["lane_eligibility"]["cells"] if c["family"] == family]
        if rungs:
            assert cells, f"{family} attests {rungs} but publishes no lane_eligibility cell"
        else:
            assert not cells, (
                f"{family} publishes a lane_eligibility cell but attests no rung; absence "
                "resolves unattested and is never invented into a cell")


def test_a_cell_cannot_attest_a_rung_the_family_does_not_publish(contract):
    """The mechanism, not the state: ``attested_rungs_q256`` bounds every cell.

    A rung must not be able to acquire a route status by being written into a
    cell, so ``validate_serving_contract`` refuses a cell naming anything the
    family does not publish.  This exercises that refusal on every family that
    has a cell to mutate; it passes on any tree where the mechanism is intact
    (``contract.py``'s ``unknown_rungs`` check), which is the point.
    """
    seen = 0
    for family, rungs in _FAMILY_RUNGS.items():
        if not any(c["family"] == family for c in contract["lane_eligibility"]["cells"]):
            continue
        seen += 1
        broken = copy.deepcopy(contract)
        target = next(c for c in broken["lane_eligibility"]["cells"] if c["family"] == family)
        invented = max(rungs) + 1
        assert invented not in set(rungs)
        target["rungs_q256"] = [invented]
        with pytest.raises(ValueError, match="the family does not publish"):
            validate_serving_contract(broken)
    assert seen, "no family publishes a cell; the mutation above never ran"


def test_every_cell_names_a_rung_its_family_attests(contract):
    """The same bound, read off the shipped file rather than a mutation."""
    formats = {entry["family"]: entry for entry in contract["formats"]}
    for cell in contract["lane_eligibility"]["cells"]:
        attested = set(formats[cell["family"]]["attested_rungs_q256"])
        assert set(cell["rungs_q256"]) <= attested, (
            f"{cell['id']} attests {sorted(set(cell['rungs_q256']) - attested)}, "
            f"which {cell['family']} does not publish")


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


def test_the_cells_are_pinned_field_for_field(contract):
    cells = _cells(contract)
    assert sorted(cells) == sorted(_CELL_LAWS)
    for cell_id, laws in _CELL_LAWS.items():
        got = cells[cell_id]
        for field, value in _resolved(laws).items():
            assert got[field] == value, f"{cell_id}.{field}"


def test_every_cell_is_backed_with_a_serve_flag_and_plugin_gated(contract):
    """Every route is plugin-gated and reached through a NAMED residency.

    The flag was pinned to the literal ``resident|streamed`` until #111.  That
    read as a formatting rule and was really a claim -- that both residencies
    execute the same thing -- which is false on the E4M3 wire.  The rule is
    that a cell names residencies from ``lane.MODES``, parsed rather than
    matched; which ones it names is the LAWS table's business.
    """
    from tessera.serving.contract import cell_residency_modes
    from tessera.serving.lane import MODES

    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["route_status"] == "backed_with_serve_flag"
        assert cell["qualification"] == "device_qualified"
        assert cell["requires_plugin"] == "tessera"
        modes = cell_residency_modes(cell)
        assert modes and set(modes) <= set(MODES)


def test_the_table_adds_only_the_measured_moe_scope_without_expert_parallelism(contract):
    """The LFM receipt adds one family/rung/residency/runtime pair of regimes."""
    block = contract["lane_eligibility"]
    assert block["structures"] == ["dense", "routed_moe"]
    moe = [cell for cell in block["cells"] if cell["structure"] == "routed_moe"]
    assert {cell["regime"] for cell in moe} == {"decode", "batch"}
    assert len(moe) == 2
    for cell in moe:
        assert cell["family"] == "TESSERA_E4M3_K1" and cell["rungs_q256"] == [1024]
        assert cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident"]
        assert cell["runtime"]["execution_modes"] == ["eager"]
    assert contract["expert_parallel"]["units"] == []


def test_each_cell_executes_the_contract_its_route_module_exposes(contract):
    """The cell's ``activation_contract`` is the route's own constant.

    Imported lazily: the route modules import torch, and the contract half of
    this file must stay readable where it is not installed.
    """
    pytest.importorskip("torch")
    from tessera.serving import bf16_route, fp8_route, nvfp4_route

    by_family = {
        "TESSERA_E2M1_K2": nvfp4_route.ACTIVATION_CONTRACT,
        "TESSERA_E4M3_K1": fp8_route.ACTIVATION_CONTRACT,
        "TESSERA_BF16_K1": bf16_route.ACTIVATION_CONTRACT,
    }
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["activation_contract"] == by_family[cell["family"]]


def test_the_launch_table_is_spelled_in_the_vocabulary_the_serve_stamps():
    """``ROUTE_LAUNCHES`` is a table about telemetry, so it uses telemetry's words.

    It lives in ``scheme`` (torch-free, because the contract validator reads it
    on a producer box) and ``telemetry`` imports torch, so the decoder strings
    are literals there.  That is exactly the drift ``loader_axes`` vs
    ``ROUTE_TP_AXES`` is tied against, and this is the same tie.
    """
    pytest.importorskip("torch")
    from tessera.serving import telemetry
    from tessera.serving.scheme import LAUNCH_FIELDS, ROUTE_LAUNCHES, ROUTES

    for route, launches in ROUTE_LAUNCHES.items():
        assert route in ROUTES
        assert launches, f"{route} launches nothing"
        for launch in launches:
            assert set(launch) == set(LAUNCH_FIELDS), launch
            assert launch["decoder"] in telemetry.DECODERS
            assert set(launch["regimes"]) <= set(CENSUS_PHASE_REGIMES.values())


def test_the_launch_tables_lane_is_the_published_extension():
    """A launch may only name a lane this build publishes an extension for."""
    from tessera.serving import ext
    from tessera.serving.scheme import ROUTE_LAUNCHES

    published = {e["module_name_prefix"] for e in ext.NATIVE_EXTENSIONS if e.get("lane")}
    for route, launches in ROUTE_LAUNCHES.items():
        for launch in launches:
            if launch["lane"] is not None:
                assert launch["lane"] in published, launch
                # And the extension must say it serves that route.  The
                # window GEMV published TESSERA_FP8 alone while
                # ``bf16_route`` loads and dispatches on it too, so a BF16
                # serve with the extension and one without fingerprinted
                # alike -- and no BF16 cell could ever derive a GEMV launch.
                assert route in next(
                    e["routes"] for e in ext.NATIVE_EXTENSIONS
                    if e["module_name_prefix"] == launch["lane"]), (route, launch["lane"])


def test_the_routes_census_expectation_is_the_launch_table():
    """The routes' own ``census_expected`` and a cell's ``executes`` are one table.

    Two spellings of "what this route may launch" is how the census tool and
    the contract came to disagree about one runtime.  Both sides read
    ``scheme.ROUTE_LAUNCHES`` now, so this asserts the derivation rather than
    a copy of the answer.

    The lane's op is in BOTH regimes.  It used to be pinned to ``decode``
    alone, which is the KERNEL's word for M <= ``GEMV_MAX_M`` read into a
    table written in the CONTRACT's, where ``decode`` is the one-row forward
    and ``batch`` is every M > 1 -- so the two-row tile, which the lane serves,
    fell in a regime the table said the lane never touched.
    """
    pytest.importorskip("torch")
    from tessera.serving import bf16_route, fp8_gemv
    from tessera.serving.scheme import TESSERA_BF16, TESSERA_FP8, launch_pairs

    for module, route in ((fp8_gemv, TESSERA_FP8), (bf16_route, TESSERA_BF16)):
        eager = module.census_expected(compiled=False)
        for regime in ("decode", "batch"):
            assert eager[regime] == launch_pairs(route, regime=regime), (route, regime)
            assert (module.GEMV_SYMBOL, "window_gemv") in eager[regime], (route, regime)
        # ...and the launch only the materialised path makes is batch-only:
        # ``decode_is_gemv`` is unconditionally true at one row, so a one-row
        # forward on a prepared lane cannot report the kernel-decoded tile
        # under the stock GEMM.
        assert (module.GEMM_SYMBOL, "window_gemv") in eager["batch"]
        assert (module.GEMM_SYMBOL, "window_gemv") not in eager["decode"]


def test_the_launch_tables_regimes_are_the_routes_own_dispatch():
    """``ROUTE_LAUNCHES`` regimes, derived from ``decode_is_gemv`` and nothing else.

    THE DEFECT THIS PINS.  The table is hand-written and the dispatch does not
    read it, so until this test the only thing tying the two together was a
    serve -- and a serve only ever drove two shapes, one row and sixty-four.
    Both of the table's first errors lived in the gap: a launch conditioned on
    a rate set (unreachable at one row, so dead) and a batch regime published
    without the GEMV (unreachable at sixty-four rows, so invisible).  Both are
    the failure #111 is about, and both are ruled out by quantifying over
    every M the dispatch distinguishes rather than the two anyone drove.

    ``decode_is_gemv`` depends on M only through ``_m_tile(M)`` (1, 2, 4, 8)
    and the ``M > GEMV_MAX_M`` test, so ``range(1, GEMV_MAX_M + 2)`` visits
    every class it has.  ``rate_one`` is the unit's other axis and both values
    are tried; the assertion is EXACT equality per regime, so a launch the
    dispatch cannot make fails as loudly as one it can.
    """
    pytest.importorskip("torch")
    import types

    from tessera.serving import bf16_route, ext, fp8_gemv, telemetry
    from tessera.serving.scheme import (TESSERA_BF16, TESSERA_FP8, launch_pairs,
                                        regime_of_m)

    for module, route in ((fp8_gemv, TESSERA_FP8), (bf16_route, TESSERA_BF16)):
        seen: dict[str, set] = {"decode": set(), "batch": set()}
        for m in range(1, module.GEMV_MAX_M + 2):
            for rate_one in (False, True):
                holder = types.SimpleNamespace(rate_one=rate_one)
                # The two pairs ``apply``'s lane branch stamps, by the same
                # predicate it stamps them on.
                pair = ((module.GEMV_SYMBOL, telemetry.DECODER_WINDOW_GEMV)
                        if module.decode_is_gemv(holder, m)
                        else (module.GEMM_SYMBOL, telemetry.DECODER_WINDOW_GEMV))
                seen[regime_of_m(m)].add(pair)
        for regime, pairs in seen.items():
            assert pairs == launch_pairs(
                route, regime=regime, mode="streamed",
                lanes=(ext.WINDOW_GEMV_MODULE_NAME,)), (route, regime)


def test_every_cell_executes_a_launch_its_route_can_make(contract):
    """The shipped table, read against the launch table rather than mutated."""
    from tessera.serving.contract import cell_executes, cell_residency_modes
    from tessera.serving.scheme import launch_pairs

    by_family = {"TESSERA_E2M1_K2": "TESSERA_NVFP4", "TESSERA_E4M3_K1": "TESSERA_FP8",
                 "TESSERA_BF16_K1": "TESSERA_BF16"}
    for cell in contract["lane_eligibility"]["cells"]:
        route = by_family[cell["family"]]
        admissible = set()
        for mode in cell_residency_modes(cell):
            admissible |= launch_pairs(route, structure=cell["structure"],
                                       regime=cell["regime"], mode=mode)
        assert cell_executes(cell) <= admissible, cell["id"]


def test_launch_table_structures_follow_the_dispatch_builders():
    from tessera.serving.scheme import (
        MOE_BUILDERS, ROUTE_LAUNCHES, ROUTES, STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE,
        STRUCTURES)

    actual = {(route, structure) for route, launches in ROUTE_LAUNCHES.items()
              for launch in launches for structure in launch["structures"]}
    expected = ({(route, STRUCTURE_DENSE) for route in ROUTES}
                | {(route, STRUCTURE_ROUTED_MOE) for route in MOE_BUILDERS})
    assert actual == expected
    for launches in ROUTE_LAUNCHES.values():
        for launch in launches:
            assert launch["structures"]
            assert set(launch["structures"]) <= set(STRUCTURES)


@pytest.mark.parametrize("regime", sorted(set(CENSUS_PHASE_REGIMES.values())))
def test_moe_launches_are_structure_specific_and_resident_only(regime):
    pytest.importorskip("torch")
    from tessera.serving import fp8_gemv, moe_route
    from tessera.serving.scheme import (
        MOE_BUILDERS, ROUTES, STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE,
        TESSERA_FP8, launch_pairs)

    # Existing callers keep their dense meaning. A requested expert structure
    # cannot borrow a dense launch, even at the same family and rate.
    dense = launch_pairs(TESSERA_FP8, regime=regime)
    assert dense == launch_pairs(TESSERA_FP8, structure=STRUCTURE_DENSE, regime=regime)
    assert dense == fp8_gemv.census_expected(compiled=False)[regime]
    moe = launch_pairs(TESSERA_FP8, structure=STRUCTURE_ROUTED_MOE,
                       regime=regime, mode="resident", lanes=())
    assert moe == moe_route.census_expected(compiled=False)[regime]
    assert moe and moe.isdisjoint(dense)
    assert not launch_pairs(TESSERA_FP8, structure=STRUCTURE_ROUTED_MOE,
                            regime=regime, mode="streamed")
    for unsupported in set(ROUTES) - set(MOE_BUILDERS):
        assert not launch_pairs(unsupported, structure=STRUCTURE_ROUTED_MOE,
                                regime=regime, mode="resident")


def test_moe_census_expectation_is_derived_from_the_shared_launch_table(monkeypatch):
    pytest.importorskip("torch")
    from tessera.serving import moe_route, scheme

    # Perturb the shared value rather than restating its current symbol. A
    # route-owned duplicate would keep returning its private spelling.
    pair = ("test.changed_moe_launch", "torch_materialize_stock")
    regimes = tuple(CENSUS_PHASE_REGIMES.values())
    monkeypatch.setitem(scheme.ROUTE_LAUNCHES, scheme.TESSERA_FP8, (
        {"symbol": pair[0], "decoder": pair[1], "regimes": regimes,
         "modes": ("resident",), "lane": None, "when_lane_absent": True,
         "structures": (scheme.STRUCTURE_ROUTED_MOE,)},))
    for compiled in (False, True):
        assert moe_route.census_expected(compiled=compiled) == {
            regime: {pair} for regime in regimes}


@pytest.mark.parametrize("regime", sorted(set(CENSUS_PHASE_REGIMES.values())))
def test_cell_launch_derivation_uses_the_cells_structure(contract, regime):
    from tessera.serving.contract import _validate_cell_executes

    entry = next(row for row in contract["formats"]
                 if row["family"] == "TESSERA_E4M3_K1")
    synthetic = {
        "structure": "routed_moe", "regime": regime, "rungs_q256": [1024],
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
        "executes": [{"symbol": "vllm.fused_moe.modular_kernel",
                      "decoder": "torch_materialize_stock"}]}
    # A synthetic execution claim checks the derivation without publishing
    # a receipt-bearing cell in the packaged contract.
    _validate_cell_executes(synthetic, "TESSERA_FP8", entry, contract, "synthetic")
    synthetic["executes"] = [{"symbol": "torch._scaled_mm", "decoder": "torch_window"}]
    with pytest.raises(ValueError, match="executes"):
        _validate_cell_executes(synthetic, "TESSERA_FP8", entry, contract, "synthetic")


def test_a_cell_that_names_a_launch_its_route_cannot_make_is_refused(contract):
    broken = copy.deepcopy(contract)
    broken["lane_eligibility"]["cells"][0]["executes"] = [
        {"symbol": "torch.mm", "decoder": "window_gemv"}]
    with pytest.raises(ValueError, match="executes"):
        validate_serving_contract(broken)


def test_a_cell_with_no_launch_is_refused(contract):
    broken = copy.deepcopy(contract)
    broken["lane_eligibility"]["cells"][0]["executes"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        validate_serving_contract(broken)


def test_a_cell_id_may_not_name_a_launch(contract):
    """The id is the SCOPE.  ``..._decode_scaled_mm_w8a8`` is the defect (#111)."""
    broken = copy.deepcopy(contract)
    cell = broken["lane_eligibility"]["cells"][0]
    cell["id"] = cell["id"] + "_scaled_mm_w4a4"
    with pytest.raises(ValueError, match="a cell id is its SCOPE"):
        validate_serving_contract(broken)


def test_no_unit_attests_a_world_size_above_one(contract):
    """``max_world_size`` is an ATTESTATION, and nothing has been measured.

    It is 1 not because the bytes cannot shard -- they can, and the loader cuts
    a unit at load -- but because no multi-rank serve has been run.  Raising it
    takes a two-rank serve with a per-rank census and a KL against the
    single-rank arm.
    """
    units = contract["tensor_parallel"]["units"]
    assert units, "the contract makes a tensor-parallel claim"
    assert {u["unit"] for u in units} == set(_FAMILY_RUNGS)
    for unit in units:
        assert unit["max_world_size"] == 1


def test_loader_axes_is_the_table_the_routes_gate_on(contract):
    """The published per-axis answer IS ``sharding.ROUTE_TP_AXES``.

    Two documents about one runtime is how the plugin came to ship a refusal
    saying the unit slicer was absent from a build that had it.  This is the
    same check ``activation_contract`` gets: the value a gate reads is compared
    against the constant the code itself uses, not against prose.
    """
    from tessera.serving.contract import _FAMILY_TO_ROUTE
    from tessera.serving.sharding import AXES, ROUTE_TP_AXES

    for unit in contract["tensor_parallel"]["units"]:
        axes = unit["loader_axes"]
        route = _FAMILY_TO_ROUTE[unit["unit"]]
        assert sorted(axes) == sorted(AXES)
        for axis in AXES:
            assert axes[axis]["status"] == ROUTE_TP_AXES[route][axis]


def test_the_published_axes_are_the_ones_the_seam_can_serve(contract):
    """Named, not merely derived: E4M3 cuts both axes, E2M1x2 cuts columns only.

    The row axis is the body's answer, not the tile's -- the window body's
    L-bit pad IS ``state_{-1}``, and the span-2 TCQ decoders supply
    ``state_{-1} = 0`` themselves -- so this is the assertion that would fail if
    a future edit quietly widened the NVFP4 route's claim without teaching its
    packer a start state.
    """
    axes = {u["unit"]: {a: v["status"] for a, v in u["loader_axes"].items()}
            for u in contract["tensor_parallel"]["units"]}
    assert axes["TESSERA_E4M3_K1"] == {"row": "sharded", "column": "sharded"}
    assert axes["TESSERA_E2M1_K2"] == {"row": "refused", "column": "sharded"}
    refused = [u for u in contract["tensor_parallel"]["units"]
               if u["unit"] == "TESSERA_E2M1_K2"][0]["loader_axes"]["row"]
    assert "INITIAL_STATE" in refused["reason"]


# --- what the validator refuses ----------------------------------------------

def _mutated(contract, mutate):
    copy_ = copy.deepcopy(contract)
    mutate(copy_)
    return copy_


def _remove_cells_for_a_declared_structure(c):
    """A declared structure without any receipt-bearing cell is not attested."""
    structure = c["lane_eligibility"]["structures"][-1]
    c["lane_eligibility"]["cells"] = [cell for cell in c["lane_eligibility"]["cells"]
                                    if cell["structure"] != structure]


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
    (_remove_cells_for_a_declared_structure, "where served facts go"),
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


def test_a_cell_may_not_name_a_structure_absent_from_the_published_axis(contract):
    """A cell's structure must be present in the published projection."""
    bad = _mutated(contract,
                   lambda c: c["lane_eligibility"]["structures"].pop(0))
    with pytest.raises(ValueError, match="is not declared"):
        validate_serving_contract(bad)


def test_a_new_dispatch_structure_is_not_attested_without_a_served_cell(
        contract, monkeypatch):
    """Dispatch capability cannot mint an attestation by itself.

    A structure enters ``scheme.STRUCTURES`` when this build can execute it;
    it enters ``lane_eligibility.structures`` only when a published cell says
    which served receipt covers it.  Growing the former must therefore leave
    the latter fail-closed until that cell exists.
    """
    from tessera.serving import scheme

    future = "future_dispatch_structure"
    monkeypatch.setattr(scheme, "STRUCTURES", (*scheme.STRUCTURES, future))
    bad = _mutated(
        contract, lambda c: c["lane_eligibility"]["structures"].append(future))

    with pytest.raises(ValueError, match="no receipt-bearing cell"):
        validate_serving_contract(bad)


def test_the_attested_structure_axis_is_a_canonical_list(contract):
    """Set-equivalent spellings are not equivalent published contracts."""
    bad = _mutated(
        contract, lambda c: c["lane_eligibility"].__setitem__(
            "structures", ["dense", "dense"]))

    with pytest.raises(ValueError, match="non-empty list of distinct strings"):
        validate_serving_contract(bad)


def test_an_expert_parallel_claim_is_refused(contract):
    bad = _mutated(contract, lambda c: c["expert_parallel"]["units"].append(
        {"unit": "TESSERA_E2M1_K2", "kind": "tessera_wire_family", "max_world_size": 2}))
    with pytest.raises(ValueError, match="expert_parallel.units must be empty"):
        validate_serving_contract(bad)


def test_an_unmeasured_world_size_is_refused(contract):
    """And the refusal no longer gives a false reason.

    It used to say a sharded form needs per-rank wires.  It does not: the
    artifact is TP-agnostic, one whole unit per role, and the rank cuts its own
    shard at load.  What is missing is the measurement, and that is what the
    message now says.
    """
    bad = _mutated(contract,
                   lambda c: c["tensor_parallel"]["units"][0].__setitem__("max_world_size", 2))
    with pytest.raises(ValueError, match="ATTESTATION") as excinfo:
        validate_serving_contract(bad)
    assert "per-rank wires" not in str(excinfo.value)


def test_a_loader_axis_that_disagrees_with_the_code_is_refused(contract):
    """The document may not widen what the loader does."""
    bad = _mutated(contract, lambda c: c["tensor_parallel"]["units"][0]["loader_axes"]["row"]
                   .update({"status": "sharded", "reason": None}))
    with pytest.raises(ValueError, match="ROUTE_TP_AXES"):
        validate_serving_contract(bad)


def test_a_refused_axis_must_carry_a_reason(contract):
    bad = _mutated(contract,
                   lambda c: c["tensor_parallel"]["units"][0]["loader_axes"]["row"]
                   .__setitem__("reason", None))
    with pytest.raises(ValueError, match="carries no reason"):
        validate_serving_contract(bad)


def test_a_unit_without_loader_axes_is_refused(contract):
    bad = _mutated(contract,
                   lambda c: c["tensor_parallel"]["units"][0].pop("loader_axes"))
    with pytest.raises(ValueError, match=r"missing \['loader_axes'\]"):
        validate_serving_contract(bad)


def test_every_published_family_states_the_a_side_a_gate_can_read():
    """``activation_contract`` is on the FORMATS row, not only on the cells.

    A family is decodable before any receipt covers it -- TESSERA_BF16_K1
    landed exactly that way -- and while the field lived only on
    ``lane_eligibility.cells`` such a family published its A-side contract in
    changelog prose and nowhere a consumer could reach.  That is the failure
    principle 14 names: a producer could not tell "unquantised by design" from
    "nobody filled it in", and PrismaQuant's lane preflight hit it deriving an
    executes glob for a family whose A side it could not price.

    The two claims stay INDEPENDENT once a receipt lands.  This test originally
    demonstrated that with ``TESSERA_BF16_K1``'s empty ``attested_rungs_q256``,
    which stopped being empty at v5; pinning that emptiness would have made the
    test a hostage to the next receipt rather than a statement about the field.
    What it pins instead is the invariant that does not move: every row's A side
    is its route's, and every cell's A side is its row's -- the row says what the
    decoder feeds the GEMM, the cell says a receipt covered it.
    """
    from tessera.serving.contract import _FAMILY_TO_ROUTE, load_serving_contract
    from tessera.serving.scheme import ROUTES

    contract = load_serving_contract()
    rows = {row["family"]: row for row in contract["formats"]}
    assert rows, "the packaged contract publishes no formats[] rows"
    for family, row in rows.items():
        assert row["activation_contract"] == ROUTES[_FAMILY_TO_ROUTE[family]]["activation_contract"]
    # The one this test exists for: an A side published as a value, whatever the
    # attestation state beside it happens to be.
    assert rows["TESSERA_BF16_K1"]["activation_contract"] == "bf16_unquantized"
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["activation_contract"] == rows[cell["family"]]["activation_contract"], (
            f"{cell['id']} executes a different A side from its own family row")


def test_a_row_whose_a_side_disagrees_with_its_route_is_refused():
    """The route is the authority.  A row that priced an A side the runtime does
    not execute is the currency error that moved an 87 GB allocation once."""
    import copy

    from tessera.serving.contract import load_serving_contract, validate_serving_contract

    contract = copy.deepcopy(load_serving_contract())
    contract["formats"][0]["activation_contract"] = "fp8_per_token_dynamic"
    with pytest.raises(ValueError, match="route executes"):
        validate_serving_contract(contract)


def test_a_route_status_nothing_defines_is_refused(contract):
    """The accepted vocabulary equals the published one: backed,
    backed_with_serve_flag, unbacked.  A fourth value no cell uses and no
    consumer defines was accepted by the validator and would have reached
    every reader's ``else`` branch."""
    import copy
    doc = copy.deepcopy(contract)
    doc["lane_eligibility"]["cells"][0]["route_status"] = "fallback"
    with pytest.raises(ValueError, match="route_status"):
        validate_serving_contract(doc)


# --- the cell predicate grammar (#134) ---------------------------------------
#
# ``predicates`` was a required cell key that no line of this package read: a
# cell could publish ``["anything"]`` and validate.  The grammar is the closed
# ``{fact, op, value}`` vocabulary the lane-eligibility receipt records and
# PrismaQuant's reader resolves; the publisher refuses what the reader could
# not read, so the first gate to consult the field never inherits a
# never-validated one.

def _with_predicates(contract, predicates):
    return _mutated(contract, lambda c: c["lane_eligibility"]["cells"][0].__setitem__(
        "predicates", predicates))


def test_every_published_cell_states_no_predicate(contract):
    """Every cell today is unconditional over its scope; a predicate appearing
    here is a narrowing no receipt has measured."""
    from tessera.serving.contract import cell_predicates
    for cell in _cells(contract).values():
        assert cell["predicates"] == []
        assert cell_predicates(cell, cell["id"]) == ()


def test_a_predicate_that_is_not_the_grammar_is_refused(contract):
    with pytest.raises(ValueError, match=r"predicates\[0\] must be a JSON object"):
        validate_serving_contract(_with_predicates(contract, ["anything"]))
    with pytest.raises(ValueError, match=r"predicates\[0\] is missing \['op', 'value'\]"):
        validate_serving_contract(_with_predicates(contract, [{"fact": "k"}]))
    with pytest.raises(ValueError, match="must be a JSON array"):
        validate_serving_contract(_with_predicates(contract, "k multiple_of 16"))


def test_a_well_formed_predicate_row_is_read_back(contract):
    from tessera.serving.contract import cell_predicates
    rows = [
        {"fact": "k", "op": "multiple_of", "value": 16},
        {"fact": "payload_family", "op": "in", "value": ["E4M3", "E2M1x2"]},
        {"fact": "in_features", "op": "at_least", "value": 1024},
        {"fact": "role_split", "op": "equals", "value": "column"},
    ]
    doc = _with_predicates(contract, rows)
    validate_serving_contract(doc)
    assert cell_predicates(doc["lane_eligibility"]["cells"][0]) == (
        ("k", "multiple_of", 16),
        ("payload_family", "in", ["E4M3", "E2M1x2"]),
        ("in_features", "at_least", 1024),
        ("role_split", "equals", "column"),
    )


@pytest.mark.parametrize("row, match", [
    ({"fact": "layer_index", "op": "equals", "value": 3}, "not a structural fact"),
    ({"fact": "k", "op": "greater_than", "value": 3}, r"op 'greater_than' is not one of"),
    ({"fact": "k", "op": "multiple_of", "value": 0}, "positive integer"),
    ({"fact": "k", "op": "multiple_of", "value": "16"}, "positive integer"),
    ({"fact": "k", "op": "at_least", "value": 3.5}, "takes an integer"),
    ({"fact": "k", "op": "at_most", "value": True}, "takes an integer"),
    ({"fact": "payload_family", "op": "in", "value": []}, "non-empty list"),
    ({"fact": "payload_family", "op": "in", "value": "E4M3"}, "non-empty list"),
    ({"fact": "payload_family", "op": "in", "value": [["E4M3"]]}, "non-empty list"),
    ({"fact": "role_split", "op": "equals", "value": ["column"]}, "takes a scalar"),
    ({"fact": "k", "op": "equals", "value": 16, "note": "x"}, r"unknown field\(s\) \['note'\]"),
], ids=lambda x: x if isinstance(x, str) else x.get("op") + ":" + repr(x.get("value")))
def test_a_predicate_outside_the_closed_grammar_is_refused(contract, row, match):
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(_with_predicates(contract, [row]))


def test_one_bound_per_fact_and_op(contract):
    rows = [{"fact": "k", "op": "at_least", "value": 16},
            {"fact": "k", "op": "at_least", "value": 32}]
    with pytest.raises(ValueError, match="repeats"):
        validate_serving_contract(_with_predicates(contract, rows))


def test_a_cell_that_narrows_itself_is_refused_until_a_consumer_reads_it(contract):
    """The pre-fix failure this test was written for::

        Failed: DID NOT RAISE ValueError

    The grammar is published and validated; nothing EVALUATES it.
    ``scheme.attested_cells`` selects by family and structure, the census
    matcher by platform, structure, runtime scope, residency and rung, and a
    predicate is exactly the part of a cell no such key carries -- so the
    first cell to state a narrowing would have been read as unconditional by
    the export gate and by the census, which is the failure the closed
    grammar was written to prevent.  A narrowed cell is a legal DOCUMENT and
    a refused one for a consumer that cannot resolve it.
    """
    from tessera.serving.scheme import attested_cells

    doc = _with_predicates(contract, [{"fact": "k", "op": "multiple_of", "value": 16}])
    validate_serving_contract(doc)
    narrowed = doc["lane_eligibility"]["cells"][0]
    with pytest.raises(ValueError, match="no consumer in this build evaluates them"):
        attested_cells(narrowed["family"], narrowed["structure"], doc)
    # Every other pair still reads: the refusal is the cell's, not the table's.
    other = next(cell for cell in doc["lane_eligibility"]["cells"]
                 if (cell["family"], cell["structure"])
                 != (narrowed["family"], narrowed["structure"]))
    assert attested_cells(other["family"], other["structure"], doc)


def test_the_grammar_is_the_one_the_receipt_and_the_consumer_name(contract):
    """The closed sets are exported so a consumer can equate its own."""
    from tessera.serving.contract import CELL_PREDICATE_FACTS, CELL_PREDICATE_OPS
    assert CELL_PREDICATE_FACTS == ("payload_family", "k", "n_sub", "rate_q256",
                                    "role_split", "in_features", "out_features")
    assert CELL_PREDICATE_OPS == ("equals", "in", "multiple_of", "at_least", "at_most")
