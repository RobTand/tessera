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
import json
from pathlib import Path
import re

import pytest

from withdrawn_cells import withdrawn_cells

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

#: The dense cells the served Tessera receipts still cover: Qwen3-0.6B on the
#: E2M1x2 cap wire (q256 = 896), every dense Linear, eager and compiled
#: (2026-09-02 receipts).  Widening ANY value here without a new receipt is the
#: failure this pins.
#:
#: The E4M3 and BF16 dense cells stood here until contract v31 and are listed in
#: ``_WITHDRAWN_CELL_IDS`` below with the reason.  The E2M1 pair survives because
#: its launch never was the window-GEMV lane's: ``nvfp4_route`` decodes span-2 at
#: load and runs ``torch._scaled_mm`` on the materialised tile, which is what it
#: still does.
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
}

#: The rungs each family ATTESTS -- rungs a container receipt covers.  Each
#: family's reader takes a far wider range (below); attestation is the narrower
#: claim and the only one a cell may make.  ``TESSERA_BF16_K1`` attested none
#: until 2026-09-02, when four route censuses and a served KL against the
#: exporter's plain-BF16 twin covered q256 = 1792
#: (``docs/measurements/tessera-bf16-route-served-2026-09-02.md``); an empty
#: list here remains the honest state for a family without a receipt, and is
#: deliberately not the same thing as an absent family.
_FAMILY_RUNGS = {"TESSERA_E2M1_K2": [128, 256, 384, 512, 640, 768, 896], "TESSERA_E4M3_K1": [1024],
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

#: The two-rank GLM-5.3-Flash 4-layer stub serve (#506).  Its receipt names the
#: image, the vLLM build its rank logs print, and the torch build
#: ``experiments/results/nvfp4_moe_route_load_probe.json`` records for the same
#: image digest.  Eager only: the GLM NoPE attention backend refuses graph mode
#: (tessera#508), so no compiled arm exists to attest.
TP2_STUB_RECEIPT = "docs/measurements/tessera-glm53-a4-stub-tp2-served-2026-09-14.md"
for _regime in ("decode", "batch"):
    _CELL_LAWS[f"tessera_e2m1_k2_routed_moe_sm121_{_regime}_resident"] = {
        "platform": "sm_121", "family": "TESSERA_E2M1_K2", "structure": "routed_moe",
        "regime": _regime, "rungs_q256": [128, 256, 384, 512, 640, 768, 896],
        "activation_contract": "e2m1_group16_ue4m3_static",
        "executes": [{"symbol": "vllm.fused_moe.modular_kernel", "decoder": "torch_materialize_stock"}],
        "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
        "requires_plugin": "tessera", "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
        "predicates": [], "runtime": {
            "image": "localhost/prismaquant/spark-vllm-nccl230@sha256:"
                     "a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb",
            "execution_modes": ["eager"],
            "vllm": "0.28.1rc1.dev397+gfd4a15126.d20260904", "torch": "2.13.0+cu130"}}

#: RE-EARNED at contract v34 (tessera#545): the dense native window GEMM, the
#: one launch ``fp8_route.apply`` and ``bf16_route.apply`` have made since
#: ``1b767a207``.  Four route censuses on the ``sm_121`` platform's own
#: ``serve_image`` put all 112 declared modules on
#: ``tessera::window_gemm_dense``/``native_window_gemm`` in BOTH regimes and
#: both residencies, for ``TESSERA_E4M3_K1`` at q256 1024 and
#: ``TESSERA_BF16_K1`` at q256 1792
#: (``docs/measurements/tessera-window-gemm-census-2026-09-21.md``).
#:
#: EAGER ONLY, and that is the receipt's own limit rather than a narrowing:
#: ``tools/tessera_route_census.py`` does not support compiled dense launch
#: agreement -- a compiled trace combines launches as ``a+b`` for a graph
#: serving every M -- so a compiled arm could not have joined a dense cell.
#: The v5-era dense cells published ``["eager", "compiled"]``; that scope is
#: not re-asserted here, which is why these four carry their own runtime
#: rather than ``_DENSE_RUNTIME``.
WINDOW_GEMM_RECEIPT = "docs/measurements/tessera-window-gemm-census-2026-09-21.md"
_WINDOW_GEMM_RUNTIME = {"image": _DENSE_IMAGE_FROM_RECEIPT, "execution_modes": ["eager"],
                        "vllm": "0.28.0", "torch": "2.13.0+cu130"}
for _family, _contract, _rung in (("TESSERA_E4M3_K1", "fp8_per_token_dynamic", 1024),
                                  ("TESSERA_BF16_K1", "bf16_unquantized", 1792)):
    for _regime in ("decode", "batch"):
        _CELL_LAWS[f"{_family.lower()}_dense_sm121_{_regime}"] = {
            "platform": "sm_121", "family": _family, "structure": "dense",
            "regime": _regime, "rungs_q256": [_rung],
            "activation_contract": _contract,
            "executes": [{"symbol": "tessera::window_gemm_dense",
                          "decoder": "native_window_gemm"}],
            "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
            "requires_plugin": "tessera",
            "requires_serve_flags": ["TESSERA_SERVE_MODE=resident|streamed"],
            "predicates": [], "runtime": _WINDOW_GEMM_RUNTIME}

#: WITHDRAWN at contract v31 (tessera#538).  These eight cells attested the
#: dense window-GEMV lane: ``torch._scaled_mm``/``torch_window``,
#: ``tessera_window_gemv::gemv``/``window_gemv`` and ``torch.mm``/``torch_window``.
#: ``1b767a207`` left ``fp8_route.apply`` and ``bf16_route.apply`` making one
#: launch each -- ``tessera::window_gemm_dense``/``native_window_gemm``, raising
#: rather than falling back -- so from that commit the build could not make the
#: arithmetic they published, and their own receipts (2026-09-02, -09-03,
#: -09-13) were all taken before it.  Retagging them would claim a measurement
#: that never ran, so they are removed rather than re-pointed.  The ids are kept
#: here, not only in the changelog, because a test reads them: a withdrawal must
#: be a deliberate, named act, and a cell that quietly comes back under one of
#: these ids has to face this list.
_WITHDRAWN_CELL_IDS = frozenset({
    "tessera_e4m3_k1_dense_sm121_decode_resident",
    "tessera_e4m3_k1_dense_sm121_decode_streamed",
    "tessera_e4m3_k1_dense_sm121_batch_resident",
    "tessera_e4m3_k1_dense_sm121_batch_streamed",
    "tessera_bf16_k1_dense_sm121_decode",
    "tessera_bf16_k1_dense_sm121_batch",
    "tessera_bf16_k1_dense_gfx1201_decode",
    "tessera_bf16_k1_dense_gfx1201_batch",
})

#: RE-EARNED at contract v34 (tessera#545), on the census receipt
#: ``WINDOW_GEMM_RECEIPT`` names.  A subset of the withdrawn ids, because an id
#: is a SCOPE and re-attesting a scope reuses it; the CLAIM is new, and
#: ``test_no_withdrawn_cell_has_come_back_with_its_withdrawn_claim`` checks the
#: launch rather than the spelling.  The other six stay withdrawn.
_REEARNED_CELL_IDS = frozenset({
    "tessera_bf16_k1_dense_sm121_decode",
    "tessera_bf16_k1_dense_sm121_batch",
})


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
    from tessera.serving.contract import _FAMILY_TO_ROUTE
    from tessera.serving.scheme import STRUCTURES, launch_pairs

    for family, rungs in _FAMILY_RUNGS.items():
        cells = [c for c in contract["lane_eligibility"]["cells"] if c["family"] == family]
        # THE THIRD CASE, added with contract v31 (tessera#538).  A rung is
        # attested by a container receipt -- the wire loads and reads back --
        # while a CELL states what the runtime executes on it.  The two came
        # apart when the dense window-GEMV dispatch was retired: the BF16
        # family's 2026-09-02 receipt still covers q256 1792, and there is no
        # attested launch left for a dense cell to name.  So "attests a rung"
        # buys a cell only where the route makes an attested launch, and where
        # it makes none a cell is not merely unnecessary, it is refused by
        # ``_validate_cell_executes``.
        route = _FAMILY_TO_ROUTE[family]
        launchable = any(launch_pairs(route, structure=structure)
                         for structure in STRUCTURES)
        if rungs and launchable:
            assert cells, f"{family} attests {rungs} but publishes no lane_eligibility cell"
        elif not rungs:
            assert not cells, (
                f"{family} publishes a lane_eligibility cell but attests no rung; absence "
                "resolves unattested and is never invented into a cell")
        elif not cells:
            # The state the withdrawal left BF16 in: an attested wire, no
            # attested dispatch, and therefore no cell.  Asserted rather than
            # passed over, so the day a launch returns this branch stops being
            # the one that runs.
            assert not any(launch_pairs(route, structure=structure)
                           for structure in STRUCTURES), family


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
    # tessera#506 leg 2: the full trellis-shaped domain. Whole-rate rungs of
    # rate 1..7 over the arity-2 grid -- q256 128..896 step 128 (one forest
    # per span-2 unit; off-step rungs like 448 resolve to a mixed-rate
    # schedule the preparer refuses by name, so the step is the honest bound).
    "TESSERA_E2M1_K2": ("E2M1x2", [128, 896], 128),
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
        "TESSERA_E2M1_K2", 128, 896, 128)
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


def test_the_routed_e2m1_cells_name_the_runtime_their_receipt_records(contract):
    """The image and vLLM build on the two routed E2M1_K2 cells are the ones
    the two-rank stub receipt records, read from the receipt rather than
    trusted from the LAWS table alone (#506)."""
    receipt = (ROOT / TP2_STUB_RECEIPT).read_text(encoding="utf-8")
    images = sorted(set(re.findall(
        r"localhost/prismaquant/spark-vllm-nccl230@sha256:[0-9a-f]{64}", receipt)))
    assert len(images) == 1, images
    for regime in ("decode", "batch"):
        cell = _cells(contract)[f"tessera_e2m1_k2_routed_moe_sm121_{regime}_resident"]
        assert cell["runtime"]["image"] == images[0], cell["id"]
        assert cell["runtime"]["vllm"] in receipt, cell["id"]
        assert cell["runtime"]["torch"] in receipt, cell["id"]


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
    """Two receipts, two (family, rung, runtime) pairs of regimes, and no more.

    The LFM receipt adds the FP8 family at q1024; the two-rank GLM stub serve
    (v28, #506) adds the E2M1x2 cap wire at q896.  Each pair is resident and
    eager, and each names its own receipt's image.
    """
    block = contract["lane_eligibility"]
    assert block["structures"] == ["dense", "routed_moe"]
    moe = [cell for cell in block["cells"] if cell["structure"] == "routed_moe"]
    by_family: dict = {}
    for cell in moe:
        by_family.setdefault((cell["family"], tuple(cell["rungs_q256"]),
                              cell["runtime"]["image"]), set()).add(cell["regime"])
        assert cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident"]
        assert cell["runtime"]["execution_modes"] == ["eager"]
    assert len(moe) == 4
    assert sorted((family, rungs) for family, rungs, _ in by_family) == [
        ("TESSERA_E2M1_K2", (128, 256, 384, 512, 640, 768, 896)),
        ("TESSERA_E4M3_K1", (1024,))]
    assert all(regimes == {"decode", "batch"} for regimes in by_family.values())
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
    """A launch may only name a lane this build publishes an extension for.

    No launch names one since contract v31 (tessera#538): the only launch that
    ever did was the dense window-GEMV lane's, and the dispatch that made it
    was retired by ``1b767a207``.  The loop below is therefore vacuous on this
    tree and would silently stay vacuous if the rule broke, so it is stated
    both ways -- the rule, and the fact that today there is nothing to apply it
    to.  A lane launch returning without an extension entry fails here; a lane
    launch returning at all has to change the count.
    """
    from tessera.serving import ext
    from tessera.serving.scheme import ROUTE_LAUNCHES

    published = {e["module_name_prefix"] for e in ext.NATIVE_EXTENSIONS if e.get("lane")}
    lane_launches = 0
    for route, launches in ROUTE_LAUNCHES.items():
        for launch in launches:
            if launch["lane"] is not None:
                lane_launches += 1
                assert launch["lane"] in published, launch
                # And the extension must say it serves that route.  The
                # window GEMV published TESSERA_FP8 alone while
                # ``bf16_route`` loads and dispatches on it too, so a BF16
                # serve with the extension and one without fingerprinted
                # alike -- and no BF16 cell could ever derive a GEMV launch.
                assert route in next(
                    e["routes"] for e in ext.NATIVE_EXTENSIONS
                    if e["module_name_prefix"] == launch["lane"]), (route, launch["lane"])
    assert lane_launches == 0, (
        "a launch names an extension lane again; the rule above now has "
        "something to say and this count has to be raised deliberately")
    assert published, "ext still publishes lane-bearing extensions; only the LAUNCH went"


def test_the_dense_launch_table_is_the_launch_apply_makes(monkeypatch):
    """THE DEFECT tessera#538 IS ABOUT, and the check that catches it.

    ``scheme.ROUTE_LAUNCHES`` is hand-written and the dispatch does not read
    it, so the only thing that ever tied the two together was a second table.
    Until this test that second table was ``fp8_gemv.census_expected`` and
    ``decode_is_gemv`` -- and ``fp8_route`` does not import ``fp8_gemv``, so
    when ``1b767a207`` retired the decode-to-global branches and left ``apply``
    making one launch, the check compared a table against a table in a module
    the dispatch no longer runs.  It agreed.  The published ``lane_eligibility``
    cells derived from that table went on naming
    ``torch._scaled_mm``/``torch_window`` and
    ``tessera_window_gemv::gemv``/``window_gemv``, arithmetic the build cannot
    launch, and nothing refused them.

    The tie is now to the LIVE route module: each of ``fp8_route`` and
    ``bf16_route`` owns a ``DENSE_LAUNCH`` pair which its ``apply`` unpacks at
    its one ``emit_route`` call, so a route cannot stamp a launch the constant
    does not name, and this asserts the table's dense entry IS that set --
    equality, so a launch the dispatch cannot make fails as loudly as a missing
    one.

    ON ``origin/master`` THIS FAILS: the table carried the retired lane's three
    dense launches beside the native GEMM while ``apply`` made only the GEMM.
    """
    pytest.importorskip("torch")
    from tessera.serving import bf16_route, fp8_route, scheme, telemetry
    from tessera.serving.scheme import (STRUCTURE_DENSE, TESSERA_BF16, TESSERA_FP8,
                                        WINDOW_GEMM_SYMBOL, launch_pairs)

    for module, route in ((fp8_route, TESSERA_FP8), (bf16_route, TESSERA_BF16)):
        assert module.DENSE_LAUNCH == (WINDOW_GEMM_SYMBOL,
                                       telemetry.DECODER_NATIVE_WINDOW_GEMM)
        assert launch_pairs(route, structure=STRUCTURE_DENSE,
                            include_experimental=True) == {module.DENSE_LAUNCH}, route
        for regime in ("decode", "batch"):
            for mode in ("resident", "streamed"):
                assert launch_pairs(route, structure=STRUCTURE_DENSE, regime=regime,
                                    mode=mode, include_experimental=True) == {
                    module.DENSE_LAUNCH}, (route, regime, mode)

    # ...and it BITES.  Put back a launch no ``apply`` makes -- the shape the
    # table was in before this commit -- and the check fails.  The DRIVER is
    # mutated; nothing about ``DENSE_LAUNCH`` or the assertion above moves.
    revived = dict(scheme.ROUTE_LAUNCHES)
    revived[TESSERA_FP8] = scheme.ROUTE_LAUNCHES[TESSERA_FP8] + (
        {"symbol": "torch._scaled_mm", "decoder": "torch_window",
         "regimes": ("batch", "decode"), "modes": ("resident", "streamed"),
         "lane": None, "structures": (STRUCTURE_DENSE,), "when_lane_absent": True},)
    monkeypatch.setattr(scheme, "ROUTE_LAUNCHES", revived)
    assert launch_pairs(TESSERA_FP8, structure=STRUCTURE_DENSE,
                        include_experimental=True) != {fp8_route.DENSE_LAUNCH}


def test_a_cell_naming_a_launch_the_build_cannot_make_is_refused(contract):
    """The document half of the same rule, on the packaged file.

    A cell's ``executes`` is derived from the launch table, so once the table
    stops carrying the retired lane the validator refuses any cell that still
    names it -- which is what makes the v31 withdrawal fail closed rather than
    linger.  Driving it with the exact pair the withdrawn E4M3 cells published
    shows the refusal is about the LAUNCH and not about a well-formedness rule
    the mutation happens to trip.
    """
    revived = copy.deepcopy(contract)
    cell = next(c for c in revived["lane_eligibility"]["cells"]
                if c["family"] == "TESSERA_E4M3_K1" and c["structure"] == "routed_moe"
                and c["regime"] == "decode")
    cell["structure"] = "dense"
    cell["id"] = "tessera_e4m3_k1_dense_sm121_decode_resident"
    cell["executes"] = [{"symbol": "torch._scaled_mm", "decoder": "torch_window"}]
    with pytest.raises(ValueError, match="but the TESSERA_FP8 route makes"):
        validate_serving_contract(revived)


def test_no_published_dense_cell_names_a_launch_the_build_cannot_make(contract):
    """The published cells read against the DISPATCH, not against the table.

    ``test_every_cell_executes_a_launch_its_route_can_make`` compares cells to
    ``ROUTE_LAUNCHES``; #538 is the case where BOTH drifted together, so a
    second reference is needed.  This one is the literal pair ``apply`` emits.
    It names nothing this branch introduced, so it runs to this assertion on
    ``master`` too -- where it fails, listing the eight stale cells.
    """
    pytest.importorskip("torch")
    from tessera.serving.scheme import TESSERA_BF16, TESSERA_FP8, WINDOW_GEMM_SYMBOL
    from tessera.serving.telemetry import DECODER_NATIVE_WINDOW_GEMM

    window = {"TESSERA_E4M3_K1": TESSERA_FP8, "TESSERA_BF16_K1": TESSERA_BF16}
    made = {(WINDOW_GEMM_SYMBOL, DECODER_NATIVE_WINDOW_GEMM)}
    stale = {}
    for cell in contract["lane_eligibility"]["cells"]:
        if cell["structure"] != "dense" or cell["family"] not in window:
            continue
        pairs = {(e["symbol"], e["decoder"]) for e in cell["executes"]}
        if pairs - made:
            stale[cell["id"]] = sorted(pairs - made)
    assert not stale, stale


def test_no_withdrawn_cell_has_come_back_with_its_withdrawn_claim(contract):
    """A withdrawal is a named act, not a gap somebody can refill quietly.

    ``_WITHDRAWN_CELL_IDS`` records what contract v31 removed and why, and the
    question a returning cell has to answer is "on which receipt?".

    That question cannot be asked by id alone, and pretending otherwise would
    make this test wrong in the other direction.  A cell id IS its scope --
    ``validate_serving_contract`` derives it from (family, structure, platform,
    regime) and refuses any other spelling -- so re-attesting a scope reuses
    its id by construction.  Two of the eight, the ``sm_121`` BF16 dense pair,
    came back at contract v34 on the tessera#545 census, and they are named in
    ``_REEARNED_CELL_IDS`` with that receipt.  What must not come back is the
    withdrawn CLAIM, so this checks the launch: a returning cell executes the
    native window GEMM the v34 receipt measured, never the window-GEMV
    arithmetic ``1b767a207`` retired.

    The other six stay absent, and for two different reasons: the four
    ``tessera_e4m3_k1_dense_sm121_*_{resident,streamed}`` ids split a scope by
    residency, which the v34 cells do not (one cell covers both), and the two
    ``gfx1201`` ids have no ROCm census of this launch.
    """
    present = {cell["id"] for cell in contract["lane_eligibility"]["cells"]}
    assert not (present & (_WITHDRAWN_CELL_IDS - _REEARNED_CELL_IDS))
    assert not (set(_CELL_LAWS) & (_WITHDRAWN_CELL_IDS - _REEARNED_CELL_IDS))
    assert _REEARNED_CELL_IDS <= _WITHDRAWN_CELL_IDS
    assert _REEARNED_CELL_IDS <= present, "a re-earned id that is not shipped is a stale record"
    withdrawn_launches = {(e["symbol"], e["decoder"])
                          for cell in withdrawn_cells(sorted(_REEARNED_CELL_IDS))
                          for e in cell["executes"]}
    for cell in contract["lane_eligibility"]["cells"]:
        if cell["id"] not in _REEARNED_CELL_IDS:
            continue
        pairs = {(e["symbol"], e["decoder"]) for e in cell["executes"]}
        assert pairs == {("tessera::window_gemm_dense", "native_window_gemm")}, cell["id"]
        assert not (pairs & withdrawn_launches), cell["id"]


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


def test_the_native_route_pairs_are_registered_experimental_and_censusable():
    """The four native lanes, by the constants route owners import.

    Each pair is in ``ROUTE_LAUNCHES`` for the structure it serves, spelled in
    ``telemetry.DECODERS``' vocabulary, absent from the contract validator's
    default view, and reachable through the documented census opt-in
    (``experimental_launch_pairs`` / ``include_experimental=True``).  Nothing
    here promotes a cell: the only pairs in ``EXPERIMENTAL_LAUNCHES`` are ones
    a table entry actually makes, so a candidate pair cannot dangle.
    """
    pytest.importorskip("torch")
    from tessera.serving import telemetry
    from tessera.serving.scheme import (A4_DENSE_GEMM_SYMBOL, A4_GROUPED_GEMM_SYMBOL,
                                        EXPERIMENTAL_LAUNCHES, ROUTE_LAUNCHES,
                                        STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE,
                                        TESSERA_BF16, TESSERA_FP8, TESSERA_NVFP4,
                                        WINDOW_GEMM_SYMBOL, WINDOW_MOE_COMPACT_SYMBOL,
                                        experimental_launch_pairs, launch_pairs)

    expected = {
        (A4_DENSE_GEMM_SYMBOL, telemetry.DECODER_NATIVE_SPAN2_GEMM):
            (TESSERA_NVFP4, STRUCTURE_DENSE),
        (A4_GROUPED_GEMM_SYMBOL, telemetry.DECODER_NATIVE_SPAN2_GROUPED):
            (TESSERA_NVFP4, STRUCTURE_ROUTED_MOE),
        (WINDOW_MOE_COMPACT_SYMBOL, telemetry.DECODER_NATIVE_WINDOW_MOE_COMPACT):
            (TESSERA_FP8, STRUCTURE_ROUTED_MOE),
    }
    assert set(expected) == set(EXPERIMENTAL_LAUNCHES)
    # LEFT at contract v34 (tessera#545), and this is the other half of that
    # move: the dense window GEMM is attested now, so it is NOT experimental
    # and it IS in the validator's default view -- which is what lets the four
    # v34 cells name it, since ``_validate_cell_executes`` derives ``executes``
    # from ``launch_pairs`` with ``include_experimental=False``.
    promoted = (WINDOW_GEMM_SYMBOL, telemetry.DECODER_NATIVE_WINDOW_GEMM)
    assert promoted not in EXPERIMENTAL_LAUNCHES
    assert promoted not in experimental_launch_pairs(TESSERA_FP8, structure=STRUCTURE_DENSE)
    assert promoted in launch_pairs(TESSERA_FP8, structure=STRUCTURE_DENSE)
    assert promoted in launch_pairs(TESSERA_BF16, structure=STRUCTURE_DENSE)
    for pair, (route, structure) in expected.items():
        assert pair[1] in telemetry.DECODERS, pair
        assert pair in experimental_launch_pairs(route, structure=structure), pair
        assert pair not in launch_pairs(route, structure=structure), (
            f"{pair} leaked into the attested dispatch")
        entries = [launch for launch in ROUTE_LAUNCHES[route]
                   if (launch["symbol"], launch["decoder"]) == pair
                   and structure in launch["structures"]]
        assert entries, (pair, route, structure)
        for launch in entries:
            assert launch["lane"] is None and not launch["when_lane_absent"], launch
    # A route owner's census opt-in is the union, not a new table.
    for route, structure in ((TESSERA_NVFP4, STRUCTURE_DENSE),
                             (TESSERA_NVFP4, STRUCTURE_ROUTED_MOE),
                             (TESSERA_FP8, STRUCTURE_DENSE),
                             (TESSERA_FP8, STRUCTURE_ROUTED_MOE)):
        assert experimental_launch_pairs(route, structure=structure) <= launch_pairs(
            route, structure=structure, include_experimental=True)


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
    # cannot borrow a dense launch, even at the same family and rate.  The
    # dense side is compared with the experimental launches in: the routes'
    # census expectation knows the packed native lane, and the MoE side below
    # must still not see it.
    dense = launch_pairs(TESSERA_FP8, regime=regime, include_experimental=True)
    assert dense == launch_pairs(TESSERA_FP8, structure=STRUCTURE_DENSE,
                                 regime=regime, include_experimental=True)
    assert dense == fp8_gemv.census_expected(compiled=False)[regime]
    non_experimental = launch_pairs(TESSERA_FP8, regime=regime)
    assert non_experimental <= dense
    assert not any(pair in non_experimental
                   for pair in dense - non_experimental), "experimental leaked"
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


def test_a_world_size_above_one_is_attested_by_a_receipt(contract):
    """``max_world_size`` is an ATTESTATION, and above 1 it names its receipt.

    It is not a statement about whether the bytes can shard -- they can, and
    the loader cuts a unit at load -- but about which world a served receipt
    covers.  A unit at 1 names nothing.  A unit above 1 names a
    ``tensor_parallel.world_size_receipts`` entry at exactly its world whose
    route traces executed it, carrying a two-rank serve record and a KL
    against the single-rank arm.  The packaged contract validates with the
    receipts in place, and every raised unit is refused, by name, once its
    receipt is removed.
    """
    units = contract["tensor_parallel"]["units"]
    assert units, "the contract makes a tensor-parallel claim"
    assert {u["unit"] for u in units} == set(_FAMILY_RUNGS)
    receipts = {r["id"]: r for r in contract["tensor_parallel"].get("world_size_receipts", [])}
    raised = [i for i, unit in enumerate(units) if unit["max_world_size"] > 1]
    for unit in units:
        if unit["max_world_size"] == 1:
            assert "world_size_receipt" not in unit, unit["unit"]
            continue
        receipt = receipts[unit["world_size_receipt"]]
        assert receipt["world_size"] == unit["max_world_size"], unit["unit"]
        assert unit["unit"] in receipt["executed_units"], unit["unit"]
        assert receipt["single_rank_kl"]["arms"], unit["unit"]
        assert all(len(serve["route_traces"]) == receipt["world_size"]
                   for serve in receipt["serves"]), unit["unit"]
        assert receipt["grade"] == "route_only", unit["unit"]

    validate_serving_contract(copy.deepcopy(contract))
    for i in raised:
        bad = _mutated(contract,
                       lambda c, i=i: c["tensor_parallel"]["units"][i].pop("world_size_receipt"))
        with pytest.raises(ValueError, match="names no world_size_receipt") as excinfo:
            validate_serving_contract(bad)
        assert units[i]["unit"] in str(excinfo.value)


# --- KV-head replication: the silence #330 gated, and v29's exit from it ------
#
# Until v29 ``tensor_parallel`` published ``max_world_size`` and ``loader_axes``
# and said NOTHING about KV-head replication, although the loader enforces
# vLLM's rule (``sharding.layer_replicas`` reads ``KV_REPLICAS_ATTRIBUTE`` off
# the layer and the plan hands rank ``tp_rank`` the shard at
# ``tp_rank // replicas``).  That was deliberate: at ``max_world_size == 1`` a
# published replication rule would have been a claim a consumer PINS on the
# strength of no served evidence.  The silence was only honest while the world
# was one, so it was GATED rather than commented -- and v29, which attests a
# world of two on a receipt, takes the gate's publish exit.  The gate below is
# unchanged; the tests read the other way.

def _packaged_tensor_parallel() -> dict:
    """The shipped ``tensor_parallel`` block, read WITHOUT the validator.

    Since v29 the validator refuses a raised world with no published rule
    itself (``_validate_tensor_parallel``), so on a tree that drops the rule the
    ``contract`` fixture raises before the gate can speak.  Reading the
    packaged bytes through the contract module's own ``contract_path`` keeps
    this test-side gate an independent second reading of the same bytes.
    """
    return json.loads(contract_path().read_text(encoding="utf-8"))["tensor_parallel"]


def _unpublished_replication_above_one(tensor_parallel: dict) -> str | None:
    """What #330 decided, as a gate.  Both sides derived, neither typed.

    The attested world is the contract's own ``max_world_size``; "says nothing
    about replication" is the absence of ``sharding.KV_REPLICAS_ATTRIBUTE`` --
    the loader's own constant, the one name vLLM publishes the rule under --
    anywhere in the published block.  Returns the reader's instructions, or
    ``None`` when there is nothing to answer for.
    """
    from tessera.serving.sharding import KV_REPLICAS_ATTRIBUTE

    attested = max(int(unit["max_world_size"]) for unit in tensor_parallel["units"])
    if attested <= 1:
        return None
    if KV_REPLICAS_ATTRIBUTE in json.dumps(tensor_parallel, sort_keys=True):
        return None
    return (
        f"runtime_contract.json attests tensor_parallel max_world_size {attested}, and its "
        f"tensor_parallel block still says nothing about KV-head replication: the name "
        f"{KV_REPLICAS_ATTRIBUTE!r} (tessera.serving.sharding.KV_REPLICAS_ATTRIBUTE) appears "
        f"nowhere in it. Above one rank that rule DECIDES which rows of a GQA/MQA layer each "
        f"rank loads -- sharding.layer_replicas reads it off the layer and the plan gives rank "
        f"tp_rank the shard at index tp_rank // {KV_REPLICAS_ATTRIBUTE} -- so a consumer that "
        f"pins this contract can no longer be left to infer it. The silence was a DECISION "
        f"(tessera#330, docs/ARCHITECTURE.md 3.8) and it was conditioned on this number being "
        f"1: while no multi-rank serve is attested, publishing the rule would assert a contract "
        f"on the strength of no served evidence. Do one of two things, and not this third: "
        f"PUBLISH the rule in tensor_parallel, derived from "
        f"tessera.serving.sharding.KV_REPLICAS_ATTRIBUTE rather than typed, with the loader's "
        f"index arithmetic beside it; or PUT THE ATTESTATION BACK to 1, because a world size "
        f"above 1 needs a two-rank serve with a per-rank census and a KL against the "
        f"single-rank arm anyway.")


def test_the_contract_publishes_the_replication_rule_its_attested_world_owes():
    """The #330 gate on the SHIPPED bytes, inverted: the world is above one,
    so the rule is published, and it is the loader's own.

    It passes because v29 took the publish exit, not because the world is 1:
    the first assertion is what would fail if a later version put every unit
    back to 1 without also withdrawing the rule, which the validator refuses
    separately (``test_a_replication_rule_at_a_world_of_one_is_refused``).
    """
    from tessera.serving.sharding import KV_REPLICAS_ATTRIBUTE

    tensor_parallel = _packaged_tensor_parallel()
    assert max(int(unit["max_world_size"]) for unit in tensor_parallel["units"]) > 1
    problem = _unpublished_replication_above_one(tensor_parallel)
    assert problem is None, problem
    rule = tensor_parallel["kv_head_replication"]
    assert rule["attribute"] == KV_REPLICAS_ATTRIBUTE
    assert rule["shard_index"] == f"tp_rank // {KV_REPLICAS_ATTRIBUTE}"
    assert rule["exercised_by_receipt"] is False


@pytest.mark.parametrize("raised", [2, 8])
def test_withdrawing_the_rule_above_one_is_refused_twice(contract, raised):
    """The gate still has teeth: drop the rule in a copy, not in the file.

    The test-side gate names both exits, and the validator -- which now owns
    the rule -- refuses the same copy on its own reading.
    """
    from tessera.serving.sharding import KV_REPLICAS_ATTRIBUTE

    tensor_parallel = _packaged_tensor_parallel()
    tensor_parallel["units"][0]["max_world_size"] = raised
    del tensor_parallel["kv_head_replication"]

    problem = _unpublished_replication_above_one(tensor_parallel)
    assert problem is not None, "a world above one with no replication rule must be refused"
    assert str(raised) in problem
    assert KV_REPLICAS_ATTRIBUTE in problem
    assert "#330" in problem
    assert "PUBLISH" in problem and "PUT THE ATTESTATION BACK" in problem

    bad = _mutated(contract, lambda c: c["tensor_parallel"].pop("kv_head_replication"))
    with pytest.raises(ValueError, match="publishes no kv_head_replication"):
        validate_serving_contract(bad)


def test_publishing_the_rule_is_the_exit_the_shipped_block_took():
    """Not a tripwire that only ever says "go back": the packaged rule clears
    the gate, removing it trips the gate, and restoring it clears it again."""
    tensor_parallel = _packaged_tensor_parallel()
    rule = tensor_parallel.pop("kv_head_replication")
    assert _unpublished_replication_above_one(tensor_parallel) is not None
    tensor_parallel["kv_head_replication"] = rule
    assert _unpublished_replication_above_one(tensor_parallel) is None


def test_a_replication_rule_that_is_not_the_loaders_is_refused(contract):
    """``attribute`` and ``shard_index`` are compared with the constant the
    loader reads, as ``loader_axes`` is compared with ``ROUTE_TP_AXES``."""
    renamed = _mutated(contract, lambda c: c["tensor_parallel"]["kv_head_replication"]
                       .__setitem__("attribute", "num_kv_heads"))
    with pytest.raises(ValueError, match="KV_REPLICAS_ATTRIBUTE"):
        validate_serving_contract(renamed)
    reindexed = _mutated(contract, lambda c: c["tensor_parallel"]["kv_head_replication"]
                         .__setitem__("shard_index", "tp_rank"))
    with pytest.raises(ValueError, match="index arithmetic"):
        validate_serving_contract(reindexed)


def test_a_replication_rule_at_a_world_of_one_is_refused(contract):
    """At one rank the rule describes no served path: #330's silence holds."""
    def back_to_one(c):
        tp = c["tensor_parallel"]
        for unit in tp["units"]:
            unit["max_world_size"] = 1
            unit.pop("world_size_receipt")
        tp.pop("world_size_receipts")

    bad = _mutated(contract, back_to_one)
    with pytest.raises(ValueError, match="while every unit is at 1"):
        validate_serving_contract(bad)
    bad["tensor_parallel"].pop("kv_head_replication")
    validate_serving_contract(bad)


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
    """Named, not merely derived: every declared family cuts both axes.

    The row axis is the body's answer, not the tile's -- the window body's
    L-bit pad IS ``state_{-1}``, and since v26 (tessera#492) the span-2 TCQ
    packer threads a shard's register into its decoders' select pad
    (``tests/test_span2_start_state.py``) -- so this is the assertion that
    would fail if a future edit quietly narrowed the NVFP4 route's claim back,
    or widened a new family's without teaching its packer a start state.
    """
    axes = {u["unit"]: {a: v["status"] for a, v in u["loader_axes"].items()}
            for u in contract["tensor_parallel"]["units"]}
    both = {"row": "sharded", "column": "sharded"}
    assert axes["TESSERA_E4M3_K1"] == both
    assert axes["TESSERA_E2M1_K2"] == both
    assert axes["TESSERA_BF16_K1"] == both
    for u in contract["tensor_parallel"]["units"]:
        assert all(v["reason"] is None for v in u["loader_axes"].values()), u["unit"]


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
    c["lane_eligibility"]["cells"][0]["rungs_q256"] = [700]


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
    """A world wider than the named receipt covers is refused, and so is a
    raised unit that names no receipt -- with the reason being the missing
    measurement, never the old false one that a sharded form needs per-rank
    wires (the artifact is TP-agnostic and the rank cuts its shard at load).
    """
    wider = _mutated(contract,
                     lambda c: c["tensor_parallel"]["units"][0].__setitem__("max_world_size", 4))
    with pytest.raises(ValueError, match="covers a world of 2"):
        validate_serving_contract(wider)

    bare = _mutated(contract,
                    lambda c: c["tensor_parallel"]["units"][0].pop("world_size_receipt"))
    with pytest.raises(ValueError, match="ATTESTATION") as excinfo:
        validate_serving_contract(bare)
    assert "per-rank wires" not in str(excinfo.value)


# --- world-size receipts (v29, tessera#506, tessera#514) ----------------------
#
# ``max_world_size`` above 1 names an entry of ``tensor_parallel
# .world_size_receipts``.  The validator checks every value's grammar and the
# joins inside the document; the tests below also hold the entry to the files
# it names -- which the wheel does not ship -- by DERIVING what the entry
# states from them: the executed units from the committed route traces, the
# KL arms from the committed table.

_REPO = Path(__file__).resolve().parent.parent


def _world_size_receipts(contract) -> dict:
    return {r["id"]: r for r in contract["tensor_parallel"]["world_size_receipts"]}


@pytest.mark.parametrize("index", [0, 1, 2])
def test_a_raised_unit_without_its_receipt_is_refused_by_name(contract, index):
    unit = contract["tensor_parallel"]["units"][index]
    assert unit["max_world_size"] > 1, "the premise: every packaged unit is raised"
    bad = _mutated(contract,
                   lambda c: c["tensor_parallel"]["units"][index].pop("world_size_receipt"))
    with pytest.raises(ValueError, match="names no world_size_receipt") as excinfo:
        validate_serving_contract(bad)
    assert unit["unit"] in str(excinfo.value)


def test_a_receipt_the_block_does_not_publish_is_refused(contract):
    bad = _mutated(contract, lambda c: c["tensor_parallel"]["units"][0]
                   .__setitem__("world_size_receipt", "some_other_serve"))
    with pytest.raises(ValueError, match="some_other_serve"):
        validate_serving_contract(bad)


def test_a_receipt_whose_traces_did_not_execute_the_unit_is_refused(contract):
    family = contract["tensor_parallel"]["units"][1]["unit"]
    bad = _mutated(contract, lambda c: c["tensor_parallel"]["world_size_receipts"][0]
                   ["executed_units"].remove(family))
    with pytest.raises(ValueError, match="not this unit") as excinfo:
        validate_serving_contract(bad)
    assert family in str(excinfo.value)


def test_a_serve_record_at_another_world_is_refused(contract):
    def one_rank(c):
        flags = c["tensor_parallel"]["world_size_receipts"][0]["serves"][0]["flags"]
        flags[flags.index("--tensor-parallel-size") + 1] = "1"

    with pytest.raises(ValueError, match="covers a world of 2"):
        validate_serving_contract(_mutated(contract, one_rank))


def test_a_serve_needs_one_route_trace_per_rank(contract):
    bad = _mutated(contract, lambda c: c["tensor_parallel"]["world_size_receipts"][0]
                   ["serves"][1]["route_traces"].pop())
    with pytest.raises(ValueError, match="one route trace per rank"):
        validate_serving_contract(bad)


def test_a_typed_excess_that_the_arms_do_not_derive_is_refused(contract):
    bad = _mutated(contract, lambda c: c["tensor_parallel"]["world_size_receipts"][0]
                   ["single_rank_kl"]["excess_over_control"].__setitem__("abs_dlogprob_p50", 1.0))
    with pytest.raises(ValueError, match="derive"):
        validate_serving_contract(bad)


def test_a_single_rank_kl_missing_an_arm_is_refused(contract):
    bad = _mutated(contract, lambda c: c["tensor_parallel"]["world_size_receipts"][0]
                   ["single_rank_kl"]["arms"].pop("control"))
    with pytest.raises(ValueError, match="control"):
        validate_serving_contract(bad)


@pytest.mark.parametrize("grade", ["kl_lower_bound", "kl_full_vocab"])
def test_a_world_size_receipt_grades_route_only(contract, grade):
    bad = _mutated(contract, lambda c: c["tensor_parallel"]["world_size_receipts"][0]
                   .__setitem__("grade", grade))
    with pytest.raises(ValueError, match="route_only"):
        validate_serving_contract(bad)


def test_a_receipt_no_unit_names_is_refused(contract):
    def orphan(c):
        extra = copy.deepcopy(c["tensor_parallel"]["world_size_receipts"][0])
        extra["id"] = "unnamed_serve"
        c["tensor_parallel"]["world_size_receipts"].append(extra)

    with pytest.raises(ValueError, match="unnamed_serve"):
        validate_serving_contract(_mutated(contract, orphan))


def test_every_file_a_world_size_receipt_names_is_in_the_tree(contract):
    for receipt in _world_size_receipts(contract).values():
        named = [receipt["receipt"], receipt["single_rank_kl"]["table"]]
        named += [t for serve in receipt["serves"] for t in serve["route_traces"]]
        missing = [path for path in named if not (_REPO / path).is_file()]
        assert missing == [], f"{receipt['id']} names files that are not here: {missing}"


def test_executed_units_are_what_every_rank_of_every_serve_traced(contract):
    """Scope from the evidence: a unit is raised exactly when its route is on
    every rank's trace, and ``executed_units`` is derived here, not trusted."""
    import hashlib

    from tessera.serving.contract import PAYLOAD_FAMILY_BY_ROUTE

    units = contract["tensor_parallel"]["units"]
    for name, receipt in _world_size_receipts(contract).items():
        receipt_text = (_REPO / receipt["receipt"]).read_text(encoding="utf-8")
        per_rank = []
        for serve in receipt["serves"]:
            for path in serve["route_traces"]:
                raw = (_REPO / path).read_bytes()
                trace = json.loads(raw)
                assert trace["schema"] == "tessera.route_trace/1", path
                per_rank.append({PAYLOAD_FAMILY_BY_ROUTE[e["policy"].partition(":")[0]]
                                 for e in trace["entries"]})
                assert hashlib.sha256(raw).hexdigest() in receipt_text, (
                    f"{receipt['receipt']} does not quote the sha256 of {path}")
        on_every_rank = set.intersection(*per_rank)
        assert set(receipt["executed_units"]) == on_every_rank, (name, per_rank)
        raised = {u["unit"] for u in units if u.get("world_size_receipt") == name}
        assert raised == on_every_rank


def test_the_single_rank_kl_block_is_the_committed_table(contract):
    for receipt in _world_size_receipts(contract).values():
        block = receipt["single_rank_kl"]
        table = json.loads((_REPO / block["table"]).read_text(encoding="utf-8"))
        assert table["issue"] == block["issue"]
        assert table["instrument"] == block["instrument"]
        assert table["read"] == block["read"]
        for role, arm in block["arms"].items():
            assert arm["metrics"] == table["arms"][arm["id"]]["metrics"], role
            assert arm["scope"] == table["arms"][arm["id"]]["scope"], role
        assert table["excess_over_control"]["quantized"] == block["arms"]["quantized"]["id"]
        assert table["excess_over_control"]["control"] == block["arms"]["control"]["id"]
        for metric, ratio in block["excess_over_control"].items():
            assert table["excess_over_control"][metric] == ratio
        assert receipt["grade"] == "route_only"
        text = (_REPO / receipt["receipt"]).read_text(encoding="utf-8")
        assert block["issue"] in text and "route_only" in text


def test_a_loader_axis_that_disagrees_with_the_code_is_refused(contract):
    """The document may neither widen nor narrow what the loader does: a row
    refusal the code no longer makes (units[0] is TESSERA_E2M1_K2, whose row
    cut the loader has taken since v26) is a claim about a runtime that does
    not exist."""
    assert contract["tensor_parallel"]["units"][0]["unit"] == "TESSERA_E2M1_K2"
    bad = _mutated(contract, lambda c: c["tensor_parallel"]["units"][0]["loader_axes"]["row"]
                   .update({"status": "refused", "reason": "a reason the code does not give"}))
    with pytest.raises(ValueError, match="ROUTE_TP_AXES"):
        validate_serving_contract(bad)


def test_a_refused_axis_must_carry_a_reason(contract, monkeypatch):
    """Provoked through the code table, since no shipping route refuses an
    axis today: with the loader refusing the NVFP4 row cut, a document that
    agrees but says nothing about why is a wall, not a contract."""
    from tessera.serving import sharding
    from tessera.serving.scheme import TESSERA_NVFP4

    monkeypatch.setitem(sharding.ROUTE_TP_AXES, TESSERA_NVFP4,
                        {sharding.AXIS_ROWS: sharding.TP_REFUSED,
                         sharding.AXIS_COLUMNS: sharding.TP_SHARDED})
    assert contract["tensor_parallel"]["units"][0]["unit"] == "TESSERA_E2M1_K2"
    bad = _mutated(contract,
                   lambda c: c["tensor_parallel"]["units"][0]["loader_axes"]["row"]
                   .update({"status": "refused", "reason": None}))
    with pytest.raises(ValueError, match="carries no reason"):
        validate_serving_contract(bad)
    # ... and the same refusal with its reason is the document the code asks for.
    good = _mutated(contract,
                    lambda c: c["tensor_parallel"]["units"][0]["loader_axes"]["row"]
                    .update({"status": "refused", "reason": "the pad cannot start it"}))
    validate_serving_contract(good)


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


def test_format_structures_follow_the_dispatch_builders(contract):
    """v27 (tessera#492): each format row names the structures the plugin
    dispatches for its family, and the validator holds the row to
    ``scheme.MOE_BUILDERS`` -- a dispatch fact, distinct from the cells."""
    import copy

    from tessera.serving.contract import validate_serving_contract
    from tessera.serving.scheme import (
        MOE_BUILDERS, STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE)

    by_family = {"TESSERA_E2M1_K2": "TESSERA_NVFP4", "TESSERA_E4M3_K1": "TESSERA_FP8",
                 "TESSERA_BF16_K1": "TESSERA_BF16"}
    for entry in contract["formats"]:
        route = by_family[entry["family"]]
        expected = [STRUCTURE_DENSE] + ([STRUCTURE_ROUTED_MOE] if route in MOE_BUILDERS else [])
        assert entry["structures"] == expected, entry["family"]
    assert {entry["family"] for entry in contract["formats"]
            if STRUCTURE_ROUTED_MOE in entry["structures"]} == {"TESSERA_E2M1_K2", "TESSERA_E4M3_K1"}
    # A row that offers a structure its route does not dispatch, or hides one
    # it does, is refused by the validator rather than read.
    for family, wrong in (("TESSERA_BF16_K1", [STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE]),
                          ("TESSERA_E2M1_K2", [STRUCTURE_DENSE]),
                          ("TESSERA_E4M3_K1", [STRUCTURE_ROUTED_MOE, STRUCTURE_DENSE])):
        broken = copy.deepcopy(contract)
        for entry in broken["formats"]:
            if entry["family"] == family:
                entry["structures"] = wrong
        with pytest.raises(ValueError, match="structures"):
            validate_serving_contract(broken)
    # The field is optional: a v26 document without it still validates.
    older = copy.deepcopy(contract)
    for entry in older["formats"]:
        del entry["structures"]
    validate_serving_contract(older)
