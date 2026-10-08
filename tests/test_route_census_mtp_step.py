"""A one-step MTP draft census attests the k+1 generation step, not a decode it never runs.

THE DEFECT THIS PINS.  ``tools/tessera_route_census.py --draft-routes`` accepts
only ``num_speculative_tokens = 1``, and at k = 1 every generation step is M =
k+1 = 2 for the target (it verifies the sampled token and the draft token) and
for the draft (its one forward per step runs over the same scheduled tokens).
No forward is one row.  The census nevertheless asked its decode phase for the
contract's one-row decode regime and read the draft's decode phase from the
observer's M1 bucket, which is unsatisfiable by construction.  Served run
``u4-BAL-20260928T0540Z-2c-r5`` (GLM-5.3 PACT BAL, TP2 GB10, eager) was
REFUSED on 533 problems while every route was correct: the body's M2 records
took the batch launch pair, and the draft expert stack served
``native_window_moe_compact_folded`` on both ranks.

``census_phase_plan`` derives the phase -> regime table and the exact M per
phase from the speculative config -- prefill = the prompt, generation = k+1 --
without touching ``contract.CENSUS_PHASE_REGIMES``.  An M1 target under k = 1
means the draft was not engaged and still refuses; an M1 draft call is
inconsistent with k = 1 and refuses.  ``decode_regime_served: false`` is
stamped rather than passed or refused.

The fixture is trimmed from the r5 receipt by tests/fixtures/generate_route_census_mtp_r5.py.
The replay uses the original TCQ cells and the current census phase rules.
These historical receipts do not attest the production WINDOW owner.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tessera.serving.contract import CENSUS_PHASE_REGIMES

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "tessera_route_census.py"
FIXTURE = ROOT / "tests" / "fixtures" / "route_census_mtp_k1_r5.json"
PHASE_OF = {regime: phase for phase, regime in CENSUS_PHASE_REGIMES.items()}
PREFILL, GENERATION = PHASE_OF["batch"], PHASE_OF["decode"]


def _tool():
    spec = importlib.util.spec_from_file_location("census_mtp_step_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _k1_plan(tool, fixture):
    return tool.census_phase_plan(fixture["draft"]["speculative_config"],
                                  prompt_tokens=fixture["prompt_tokens"])


def _replay(fixture, *, plan, with_draft=True):
    """``main``'s validation call, fed from the stored observations."""
    pytest.importorskip("torch")  # the routes that own the launch sets import torch
    tool = _tool()
    from tessera.serving import bf16_route, fp8_route, moe_route, nvfp4_route
    from tessera.serving.contract import PAYLOAD_FAMILY_BY_ROUTE, load_serving_contract
    from tessera.serving.scheme import (
        ROUTES, TESSERA_BF16, TESSERA_FAMILIES, TESSERA_FP8, TESSERA_NVFP4)

    config = fixture["config"]
    groups = config["quantization_config"]["config_groups"]
    declared = {t: g["scheme"]["family"] for g in groups.values() for t in g["targets"]}
    rungs = {t: tool.declared_rung(g["scheme"]) for g in groups.values() for t in g["targets"]}
    body, draft_declared = tool.partition_glm_mtp_targets(config, declared)
    mapping = fixture["declared_name_mapping"]
    ranks = fixture["ranks"]
    platform = fixture["platform"]
    draft = fixture["draft"]
    # Replay against the actual historical TCQ receipts, not current WINDOW admission.
    current = load_serving_contract()["lane_eligibility"]["cells"]
    archive = json.loads((ROOT / "tests/fixtures/t4_tcq_cells_historical.json").read_text())
    cells = [cell for cell in current if cell["family"] != "TESSERA_E2M1_K2"] + archive["cells"]
    return tool.validate_census_observations(
        phases_by_rank={phase: [rank["records"][phase] for rank in ranks]
                        for phase in (PREFILL, GENERATION)},
        identities=[rank["identity"] for rank in ranks],
        refusals_by_rank=[rank["lane_refusals"] for rank in ranks],
        declared={mapping[t]: family for t, family in body.items()},
        declared_rungs={mapping[t]: rungs[t] for t in body},
        phase_plan=plan, mode=fixture["serve_mode"], platform=platform,
        runtime_image=fixture["runtime"]["image"],
        execution_mode=fixture["runtime"]["execution_mode"], compiled=False,
        cells=cells,
        contract_for={TESSERA_NVFP4: nvfp4_route.ACTIVATION_CONTRACT,
                      TESSERA_FP8: fp8_route.ACTIVATION_CONTRACT,
                      TESSERA_BF16: bf16_route.ACTIVATION_CONTRACT},
        expected=lambda family, regime, kind: {
            (launch["symbol"], launch["decoder"])
            for cell in cells if cell["family"] == PAYLOAD_FAMILY_BY_ROUTE[family]
            and cell["regime"] == regime
            and cell["structure"] == ("routed_moe" if kind == "moe" else "dense")
            for launch in cell["executes"]},
        symbol_for={family: ROUTES[family]["gemm_symbol"] for family in TESSERA_FAMILIES},
        symbol_base=moe_route.census_symbol_base,
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        policy_prefixes=tuple(f"{family}:" for family in TESSERA_FAMILIES),
        draft=({"arms": draft["arms"], "declared": draft_declared,
                "rungs": {t: rungs[t] for t in draft_declared},
                "source_to_module": draft["source_to_module"],
                "speculative_config": draft["speculative_config"],
                "batch_prompts": draft["requested_batch_prompts"],
                "native_decoder": moe_route.native_decoder,
                "supported_families": {TESSERA_FP8, TESSERA_BF16}}
               if with_draft else None))


# --- the plan ---------------------------------------------------------------

def test_without_a_draft_the_plan_is_the_contract_table_unchanged():
    plan = _tool().census_phase_plan(None, prompt_tokens=64)
    assert plan["phase_regimes"] == dict(CENSUS_PHASE_REGIMES)
    assert plan["expected_m"] is None
    assert plan["decode_regime_served"] is True
    assert plan["num_speculative_tokens"] is None and plan["generation_step_m"] is None


def test_the_k1_plan_expects_the_prompt_and_k_plus_one_on_batch_cells():
    plan = _k1_plan(_tool(), _fixture())
    assert plan["phase_regimes"] == {PREFILL: "batch", GENERATION: "batch"}
    assert plan["expected_m"] == {PREFILL: 64, GENERATION: 2}
    assert plan["decode_regime_served"] is False
    assert plan["num_speculative_tokens"] == 1 and plan["generation_step_m"] == 2


@pytest.mark.parametrize("config,prompt", [
    ({"method": "mtp", "num_speculative_tokens": 2}, 64),
    ({"method": "mtp", "num_speculative_tokens": 1}, 2),
    ({"method": "mtp", "num_speculative_tokens": 1}, None),
])
def test_the_plan_keeps_k1_only_and_refuses_an_indistinguishable_prompt(config, prompt):
    with pytest.raises(ValueError):
        _tool().census_phase_plan(config, prompt_tokens=prompt)


def test_a_collapsed_table_names_its_phases_through_the_contract():
    """Both phases are batch under k=1; the shape check must not StopIteration."""
    tool = _tool()
    plan = _k1_plan(tool, _fixture())
    record = {"shape": "M64:N8:K8"}
    records = {PREFILL: {"m": record}, GENERATION: {"m": dict(record, shape="M2:N8:K8")}}
    assert tool.driven_phase_pair(plan["phase_regimes"]) == (PREFILL, GENERATION)
    assert tool.phase_shape_problems(records, phase_regimes=plan["phase_regimes"],
                                     phase_plan=plan, require_each_owner=True) == []


# --- the served records -----------------------------------------------------

def test_the_fixture_reproduces_r5_under_the_contract_table():
    """The one-row plan still rejects M2 under every current cell.

    The BF16 cell withdrawal removes its cell-specific shape checks.
    The general shape checks remain. Old folded decoder records also refuse.
    """
    tool = _tool()
    fixture = _fixture()
    assert fixture["source"]["verdict"] == "REFUSED"
    checked = _replay(fixture, plan=tool.census_phase_plan(None), with_draft=False)
    from tessera.serving.contract import load_serving_contract

    active_cells = {cell["id"] for cell in load_serving_contract()["lane_eligibility"]["cells"]}
    original = [p for p in fixture["original_problems_for_kept_modules"]
                if not p.startswith("draft")
                and ("keyed to cell '" not in p
                     or any(f"keyed to cell '{cell}'" in p for cell in active_cells))]
    shape = [p for p in checked["problems"] if "shape M2 is a batch-regime forward" in p]
    assert original and sorted(shape) == sorted(original)
    extra = [p for p in checked["problems"] if p not in shape]
    assert extra, "v59 must refuse the folded BF16 records under the current dispatch"
    assert all("folded" in p for p in extra), extra
    # ...and the draft side: no observed draft call was one row, so the M1
    # bucket the old census read for the draft's decode phase was empty.
    for row in fixture["draft"]["arms"]["decode_arm"]:
        assert "decode" not in row["by_regime"]
        assert row["by_regime"]["batch"]["latest_m"] == 2


def test_the_r5_records_replay_clean_under_the_k1_plan():
    """Current agreement: r5's records replay with only the BF16 withdrawal refused.

    Contract v62 withdrew the BF16 cells and the folded launches. The folded
    BF16 body and draft records the fixture kept have no current cell and no
    current native decode coverage. Everything else -- plan, shapes, sources,
    the launch the draft's own coverage names -- replays as r5 ran it. The
    folded decoder below is the receipt's own word for what ran, not a claim
    about the current dispatch.
    """
    tool = _tool()
    fixture = _fixture()
    plan = _k1_plan(tool, fixture)
    checked = _replay(fixture, plan=plan)
    problems = checked["problems"]
    assert problems, "v59 must refuse the folded BF16 records under the current dispatch"
    names = set()
    for rank_i, rank in enumerate(fixture["ranks"]):
        for phase_records in rank["records"].values():
            for name, record in phase_records.items():
                if record["policy"].partition(":")[0] == "TESSERA_BF16":
                    names.add(name)
                    names.add(f"rank{rank_i}/{name}")
    for module in fixture["draft"]["source_to_module"].values():
        names.add(module)
        for rank_i in range(len(fixture["ranks"])):
            names.add(f"rank{rank_i}/{module}")
    for problem in problems:
        assert ("folded" in problem or "compact_bf16" in problem
                or any(name in problem for name in names)), problem
    assert plan["decode_regime_served"] is False
    assert checked["histogram"][GENERATION]["regime"] == "batch"
    draft = checked["draft"]
    assert draft["phase_sources"] == {GENERATION: "decode_arm/latest_batch",
                                      PREFILL: "decode_arm/first_batch"}
    assert draft["phase_expected_m"] == {PREFILL: 64, GENERATION: 2}
    shapes = {phase: {r["shape"] for r in records.values()}
              for phase, records in draft["records"].items()}
    assert shapes == {PREFILL: {"M64:N2048:K4096"}, GENERATION: {"M2:N2048:K4096"}}
    assert draft["decoder_coverage"]["phases"][GENERATION]["decoders"] == {
        "native_window_moe_compact_folded": 2}


def test_a_one_row_target_under_k1_is_refused_as_the_draft_not_engaged():
    tool = _tool()
    fixture = _fixture()
    for rank in fixture["ranks"]:
        for record in rank["records"][GENERATION].values():
            record["shape"] = "M1:" + record["shape"].split(":", 1)[1]
    problems = _replay(fixture, plan=_k1_plan(tool, fixture))["problems"]
    engaged = [p for p in problems if "the MTP draft was not engaged" in p]
    assert len(engaged) == 2 * len(fixture["ranks"][0]["records"][GENERATION]), problems


def test_a_one_row_draft_call_under_k1_is_refused_as_inconsistent():
    tool = _tool()
    fixture = _fixture()
    for row in fixture["draft"]["arms"]["decode_arm"]:
        batch = row["by_regime"]["batch"]
        latest = copy.deepcopy(batch["latest"])
        for record in latest.values():
            record["shape"] = "M1:" + record["shape"].split(":", 1)[1]
        row["by_regime"]["decode"] = {"calls": 1, "first": latest, "first_m": 1,
                                      "latest": latest, "latest_m": 1}
        row["calls"] += 1
    problems = _replay(fixture, plan=_k1_plan(tool, fixture))["problems"]
    inconsistent = [p for p in problems
                    if "inconsistent with num_speculative_tokens=1" in p]
    assert len(inconsistent) == len(fixture["draft"]["arms"]["decode_arm"]), problems


def test_one_draft_call_cannot_stand_for_both_phases():
    tool = _tool()
    fixture = _fixture()
    for row in fixture["draft"]["arms"]["decode_arm"]:
        batch = row["by_regime"]["batch"]
        batch.update(calls=1, latest=batch["first"], latest_m=batch["first_m"])
    problems = _replay(fixture, plan=_k1_plan(tool, fixture))["problems"]
    assert any("would quote the same call" in p for p in problems), problems
    assert any("not the k+1=M2 generation step" in p for p in problems), problems


def test_a_prefill_that_is_not_the_prompt_is_refused():
    tool = _tool()
    fixture = _fixture()
    for rank in fixture["ranks"]:
        for record in rank["records"][PREFILL].values():
            record["shape"] = "M32:" + record["shape"].split(":", 1)[1]
    problems = _replay(fixture, plan=_k1_plan(tool, fixture))["problems"]
    assert any("M32 is not the M64 of one full prefill" in p for p in problems), problems


def test_the_speculative_scope_is_stamped_into_the_receipt():
    """The plan's scope fields reach the receipt, gated to the draft census."""
    source = TOOL.read_text()
    gate = source.index("    if args.draft_routes:\n        # THE SCOPE TRAVELS WITH THE RECEIPT.")
    block = source[gate:source.index("        })", gate)]
    for key, field in (("decode_regime_served", "decode_regime_served"),
                       ("num_speculative_tokens", "num_speculative_tokens"),
                       ("generation_step_m", "generation_step_m"),
                       ("phase_regimes", "phase_regimes"),
                       ("phase_expected_m", "expected_m")):
        assert f'"{key}": phase_plan["{field}"],' in block, key
