"""What the serve executed, against what the contract says it executes (#111).

THE DEFECT THIS PINS.  A ``lane_eligibility`` cell used to say which A-side
contract ran and which rungs a receipt covered.  The LAUNCH -- which GEMM, off
which decoder -- appeared only in the cell's ``id``: the E4M3 family published
``..._decode_scaled_mm_w8a8`` for both regimes and no cell named the
window-GEMV lane at all.  That was accidentally true while the lane was
unreachable (#104: every allocated checkpoint carried a column rate outside
``kernel_window_gemv.SUPPORTED_RATES``) and false the moment a rate-constrained
artifact was served.  On ``qwen3-0.6b-uniform-R1024`` -- q256 1024, root rate 4
exactly -- the census recorded ``tessera_window_gemv::gemv`` on 112 of 112
modules in the decode regime while the contract said the materialised FP8 pair,
in every case.

THE TWO HALVES, AND WHY BOTH.  ``contract.validate_serving_contract`` derives
each cell's ``executes`` from ``scheme.ROUTE_LAUNCHES`` -- the table the routes'
own ``census_expected`` is built from -- which proves the DOCUMENT agrees with
the CODE.  Only a serve proves the code agrees with the machine, and this is
that join: every served route record against the cell that covers its
``(platform, family, structure, regime, residency, rung)``.

THE RECORDS BELOW ARE REAL.  They are copied verbatim from
``/home/rob/tessera-runs/ts104/census-R1024-readable.json`` (GB10, vLLM 0.28,
``TESSERA_SERVE_MODE=streamed``, eager, 112 of 112 modules in both phases), so
the fixture is the observation the cell was attested from rather than a
plausible-looking invention.  The full 224-record replay is recorded in
``docs/measurements/tessera-lane-eligibility-executes-2026-09-04.md``.

THE FAIL-BEFORE.  On master this module does not import: ``cell_launch_agreement``
does not exist, and neither does the ``executes`` key it reads.
"""
from __future__ import annotations

import copy

import pytest

from tessera.serving.census import cell_launch_agreement
from tessera.serving.contract import (
    CENSUS_PHASE_REGIMES, PAYLOAD_FAMILY_BY_ROUTE, load_serving_contract)

#: Two modules of the R1024 census, both phases, verbatim -- including
#: ``kind``, which the first trim of this fixture dropped and the source
#: receipt carries on every record.  It is not decoration: ``kind`` is what
#: says a record is a DENSE layer, and a block keyed to one structure is only
#: closed over the records whose structure it covers
#: (``census.STRUCTURE_BY_RECORD_KIND``).
_SERVED = {
    "decode": {
        "model.layers.0.mlp.down_proj": {
            "policy": "TESSERA_FP8:streamed", "symbol": "tessera_window_gemv::gemv",
            "decoder": "window_gemv", "contract": "fp8_per_token_dynamic",
            "state": "served", "shape": "M1:N1024:K3072", "kind": "dense"},
        "model.layers.0.mlp.gate_up_proj": {
            "policy": "TESSERA_FP8:streamed", "symbol": "tessera_window_gemv::gemv",
            "decoder": "window_gemv", "contract": "fp8_per_token_dynamic",
            "state": "served", "shape": "M1:N6144:K1024", "kind": "dense"},
    },
    "prefill": {
        "model.layers.0.mlp.down_proj": {
            "policy": "TESSERA_FP8:streamed", "symbol": "torch._scaled_mm",
            "decoder": "window_gemv", "contract": "fp8_per_token_dynamic",
            "state": "served", "shape": "M64:N1024:K3072", "kind": "dense"},
        "model.layers.0.mlp.gate_up_proj": {
            "policy": "TESSERA_FP8:streamed", "symbol": "torch._scaled_mm",
            "decoder": "window_gemv", "contract": "fp8_per_token_dynamic",
            "state": "served", "shape": "M64:N6144:K1024", "kind": "dense"},
    },
}
_RUNGS = {name: 1024 for name in _SERVED["decode"]}


def _agree(cells, records=None, rungs=None):
    return cell_launch_agreement(
        records if records is not None else _SERVED,
        cells=cells, phase_regimes=CENSUS_PHASE_REGIMES, platform="sm_121",
        rungs_by_module=rungs if rungs is not None else _RUNGS,
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE)


@pytest.fixture(scope="module")
def cells():
    return load_serving_contract()["lane_eligibility"]["cells"]


def test_the_served_records_agree_with_the_cells_that_cover_them(cells):
    block, problems = _agree(cells)
    assert problems == []
    assert block["agrees"] is True
    assert block["phases"]["decode"]["cells"] == {
        "tessera_e4m3_k1_dense_sm121_decode_streamed": 2}
    assert block["phases"]["prefill"]["cells"] == {
        "tessera_e4m3_k1_dense_sm121_batch_streamed": 2}
    assert block["phases"]["decode"]["unattested"] == 0


def test_the_pre_111_claim_would_have_been_caught_by_these_records(cells):
    """The exact table #111 was filed on, replayed against the serve it missed.

    Before this change the E4M3 decode cell published the materialised pair.
    Put that claim back and the same 2026-09-03 records refuse it -- which is
    the whole content of the issue, as a check rather than as an argument.
    """
    stale = copy.deepcopy(cells)
    cell = next(c for c in stale if c["id"] == "tessera_e4m3_k1_dense_sm121_decode_streamed")
    cell["executes"] = [{"symbol": "torch._scaled_mm", "decoder": "torch_window"}]
    block, problems = _agree(stale)
    assert block["agrees"] is False
    assert len(problems) == 2
    assert "tessera_window_gemv::gemv" in problems[0]
    assert "torch._scaled_mm" in problems[0]


def test_a_small_batch_gemv_record_is_covered_by_the_batch_cell(cells):
    """The 2-to-8-row forward: batch regime, lane's own op, and NOT a problem.

    This record is CONSTRUCTED, and says so -- no census we hold drove a
    four-row forward, because the tool drives one prompt and a 64-row prefill.
    That is exactly why the cell was wrong: the batch cell published the
    materialised launch alone, which is what a 64-row prefill takes, while the
    contract's batch regime is every M > 1 and the lane serves its own ``gemv``
    from two rows up.  The DISPATCH fact behind this record is not constructed:
    ``test_the_launch_tables_regimes_are_the_routes_own_dispatch`` derives it
    from the routes' own ``decode_is_gemv`` over every M.

    Before the correction this record was a refusal on a serve that had done
    nothing wrong, which is worse than the stale value #111 was filed on.
    """
    records = {"prefill": {name: dict(rec, symbol="tessera_window_gemv::gemv",
                                      shape=rec["shape"].replace("M64", "M4"))
                           for name, rec in _SERVED["prefill"].items()}}
    block, problems = _agree(cells, records=records)
    assert problems == []
    assert block["agrees"] is True
    assert block["phases"]["prefill"]["cells"] == {
        "tessera_e4m3_k1_dense_sm121_batch_streamed": 2}

    # ...and the pre-correction cell refuses it, once per module.
    stale = copy.deepcopy(cells)
    cell = next(c for c in stale if c["id"] == "tessera_e4m3_k1_dense_sm121_batch_streamed")
    cell["executes"] = [{"symbol": "torch._scaled_mm", "decoder": "window_gemv"}]
    stale_block, stale_problems = _agree(stale, records=records)
    assert stale_block["agrees"] is False
    assert len(stale_problems) == 2


def test_a_rung_no_cell_covers_is_unattested_and_not_a_problem(cells):
    """Absence is the only negative signal a closed-world table has.

    A serve at a rung nobody attested must not be laundered into a verdict --
    in either direction.  It is counted, and it is not a failure.
    """
    block, problems = _agree(cells, rungs={n: 1006 for n in _RUNGS})
    assert problems == []
    assert block["agrees"] is None
    assert block["phases"]["decode"]["unattested"] == 2
    assert block["phases"]["decode"]["covered_by_cell"] == 0


def test_a_fused_module_with_no_single_rung_is_unattested(cells):
    """The census resolves a per-role ``q256`` list to a rung only when it is one.

    A cell is keyed by a rung, and a mixed-rung group has none; borrowing one
    member's would attest the whole module on a receipt that covered part of it.
    """
    block, problems = _agree(cells, rungs={n: None for n in _RUNGS})
    assert problems == []
    assert block["agrees"] is None
    assert block["phases"]["decode"]["unattested"] == 2


def test_the_residency_is_part_of_the_join(cells):
    """The same records under ``resident`` resolve to the resident cell and refuse.

    The residency is the axis the two E4M3 decode cells are told apart by
    (``resident`` sets ``layer.tessera_gemv = None``, so there is no lane), and
    a join that ignored it would read either cell as covering both.
    """
    records = copy.deepcopy(_SERVED)
    for phase in records.values():
        for rec in phase.values():
            rec["policy"] = "TESSERA_FP8:resident"
    block, problems = _agree(cells, records=records)
    assert block["agrees"] is False
    assert block["phases"]["decode"]["cells"] == {
        "tessera_e4m3_k1_dense_sm121_decode_resident": 2}
    assert any("tessera_window_gemv::gemv" in p for p in problems)


def test_a_record_from_a_route_the_table_does_not_know_is_unattested(cells):
    records = copy.deepcopy(_SERVED)
    for phase in records.values():
        for rec in phase.values():
            rec["policy"] = "TESSERA_NOT_A_ROUTE:streamed"
    block, problems = _agree(cells, records=records)
    assert problems == []
    assert block["agrees"] is None


#: One routed-expert record, verbatim from the first served census of a Tessera
#: MoE checkpoint (``/mnt/shared/tessera-runs/ts5/served/census.json``, GB10,
#: vLLM 0.28, ``TESSERA_SERVE_MODE=resident``, eager, 16-expert cut of
#: GLM-5.3-Flash-4layer).  Note the module name: the record is written at
#: ``<prefix>.routed_experts`` while the checkpoint declares ``<prefix>``.
_MOE_RECORD = {
    "policy": "TESSERA_FP8:resident", "symbol": "vllm.fused_moe.modular_kernel:TRITON",
    "decoder": "torch_materialize_stock", "contract": "fp8_per_token_dynamic",
    "state": "served", "shape": "M64:N4096:K4096", "kind": "moe", "tile_m": 0,
    "reason": None}


def test_a_routed_expert_record_is_unattested_under_a_dense_block(cells):
    """A served MoE stack is counted, never covered, while no cell publishes one.

    THE DEFECT THIS PINS.  Everything else about this record resolves: the
    policy names a route the table knows, the residency is a cell's residency,
    the regime is a cell's regime, and the rung below is the rung the E4M3
    cells are attested at.  What does NOT match is the only thing that decides
    the launch -- the structure.  The stack executes vLLM's modular fused-MoE
    kernel over a materialised tile; every cell in this contract is
    ``structure: "dense"`` and publishes ``torch._scaled_mm``.  So a join that
    ignored ``kind`` would report a disagreement on a serve that did nothing
    wrong, and a join that ignored it in the other direction (a dense cell
    whose executes happened to match) would attest a launch no cell publishes.

    Today the rung lookup misses anyway, because the record's name carries the
    ``.routed_experts`` suffix the declaration does not -- which is an accident
    of the join, not a check.  Passing the rung here is the point: it is what
    a caller that resolved that join would supply.
    """
    records = {"prefill": {"model.layers.1.mlp.experts.routed_experts": dict(_MOE_RECORD)}}
    block, problems = _agree(cells, records=records,
                             rungs={"model.layers.1.mlp.experts.routed_experts": 1024})
    assert problems == []
    assert block["agrees"] is None
    assert block["phases"]["prefill"]["unattested"] == 1
    assert block["phases"]["prefill"]["covered_by_cell"] == 0


def test_a_record_that_does_not_say_its_kind_is_not_covered(cells):
    """A reader that lost the field leaves a hole, and a hole is not a dense layer.

    The conservative direction on purpose: the cost of refusing to classify is
    an ``unattested`` count a reader can see, and the cost of guessing is a
    published agreement about a launch nobody observed the structure of.
    """
    records = {"decode": {name: {k: v for k, v in rec.items() if k != "kind"}
                          for name, rec in _SERVED["decode"].items()}}
    block, problems = _agree(cells, records=records)
    assert problems == []
    assert block["agrees"] is None
    assert block["phases"]["decode"]["unattested"] == 2
    assert block["phases"]["decode"]["covered_by_cell"] == 0
