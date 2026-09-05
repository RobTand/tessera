"""A cell says what GRADE of evidence it rests on, in a field a gate can read (#133).

Until lane-eligibility schema v6 the two ``routed_moe`` cells were
machine-indistinguishable from the eight dense cells: same ``route_status``,
same ``qualification``, same key set.  What separated them lived in changelog
prose -- a top-1,024 KL lower bound on one model, a repetitive greedy smoke,
no decode-regime KL -- which no consumer can read (principle 14).

The correction that this field forces into the open: EVERY served KL this
repository holds, dense and MoE alike, is a ``kl_tool`` top-1024
teacher/student-intersection lower bound (``tessera-bf16-route-served
-2026-09-02.md`` :38, ``tessera-compiled-decode-kl-r6-2026-09-04.md`` :18,
``tessera-lfm-campaign-2026-09-04.md`` §8).  The axes that actually separate
the cells are which REGIME the bound was scored in, under which execution
modes, and whether a greedy smoke was recorded -- so those are what the field
carries, receipt by receipt, and ``grade`` is DERIVED from them the way
``executes`` is derived from the route table: stored so a reader with no
derivation can read it, checked so it cannot drift from the entries.

No number lives in a cell.  A cell names the receipt; the receipt holds the
number, its bounds and its caveats.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tessera.serving.contract import (
    EVIDENCE_CONTROL_OUTCOMES,
    EVIDENCE_CONTROL_REFERENCES,
    EVIDENCE_GRADES,
    EVIDENCE_KL_KINDS,
    EVIDENCE_RECEIPT_ROOT,
    EVIDENCE_SMOKE_ATTRIBUTIONS,
    EVIDENCE_SMOKE_STATUSES,
    cell_evidence,
    derive_evidence_grade,
    derive_smoke_attribution,
    load_serving_contract,
    validate_serving_contract,
)

ROOT = Path(__file__).resolve().parents[1]

PLUGIN = "docs/measurements/tessera-serving-plugin-2026-09-02.md"
WINDOW_GEMV = "docs/measurements/tessera-window-gemv-served-2026-09-03.md"
DECODE_EAGER = "docs/measurements/tessera-decode-regime-kl-2026-09-03.md"
DECODE_COMPILED = "docs/measurements/tessera-compiled-decode-kl-r6-2026-09-04.md"
BF16 = "docs/measurements/tessera-bf16-route-served-2026-09-02.md"
LFM = "docs/measurements/tessera-lfm-campaign-2026-09-04.md"
MOE_DEBT = "docs/measurements/moe-evidence-debt-2026-09-04.md"


def _bound(regime, modes, receipt):
    return {"kind": "topk_intersection_lower_bound", "top_k": 1024, "regime": regime,
            "execution_modes": modes, "receipt": receipt}


_NO_SMOKE = {"status": "not_recorded", "receipt": None,
             "attribution": "unattributed", "control": None}


def _uncontrolled(status, receipt):
    """A smoke nobody ran a reference against: the shape every dense cell has."""
    return {"status": status, "receipt": receipt,
            "attribution": "unattributed", "control": None}


#: The control section 7 of the MoE evidence-debt receipt ran: the same prompt,
#: byte for byte, against the BF16 SOURCE on the same pinned image, eager --
#: and the completion came back identical character for character.
_BF16_CONTROL = {"reference": "bf16_source", "outcome": "identical_completion",
                 "receipt": MOE_DEBT}
_ROUTE_ONLY = {"grade": "route_only", "kl": [], "smoke": _NO_SMOKE}

#: What each published cell rests on, receipt by receipt.  A KL entry attests
#: the cell's WHOLE scope -- every residency the cell names, under the listed
#: execution modes, in the cell's own regime -- and only a serve whose route
#: is the cell's ``executes`` counts: the 2026-09-02 plugin receipt's
#: ``e8-streamed`` arms ran the torch window decode under ``streamed`` (its
#: census table, :131/:133), a launch set no streamed E4M3 cell publishes since
#: #111, so they attest nothing here; the window-GEMV receipt's arm B ran the
#: refused-lane fallback under ``streamed`` and attests nothing either.
_EVIDENCE = {
    # E2M1x2 (q896): prefill KL, eager and compiled, both residencies
    # (``k2-resident``, ``k2-streamed``, ``k2-*-graph``, PLUGIN §3).  No
    # decode-regime KL and no greedy smoke on record.
    "tessera_e2m1_k2_dense_sm121_decode": _ROUTE_ONLY,
    "tessera_e2m1_k2_dense_sm121_batch": {
        "grade": "kl_lower_bound", "kl": [_bound("batch", ["eager", "compiled"], PLUGIN)],
        "smoke": _NO_SMOKE},
    # E4M3 (q1024).  The decode regime has a KL only where the window-GEMV
    # lane was under test -- the streamed cell -- eager on 2026-09-03 and
    # compiled in the r6 population; the resident decode cell has the census
    # alone.  The resident batch cell rests on the 2026-09-02 ``e8-resident``
    # arms; the streamed batch cell on the window-GEMV receipt's arm A.
    "tessera_e4m3_k1_dense_sm121_decode_resident": _ROUTE_ONLY,
    "tessera_e4m3_k1_dense_sm121_decode_streamed": {
        "grade": "kl_lower_bound",
        "kl": [_bound("decode", ["eager"], DECODE_EAGER),
               _bound("decode", ["compiled"], DECODE_COMPILED)],
        "smoke": _NO_SMOKE},
    "tessera_e4m3_k1_dense_sm121_batch_resident": {
        "grade": "kl_lower_bound", "kl": [_bound("batch", ["eager", "compiled"], PLUGIN)],
        "smoke": _NO_SMOKE},
    "tessera_e4m3_k1_dense_sm121_batch_streamed": {
        "grade": "kl_lower_bound",
        "kl": [_bound("batch", ["eager", "compiled"], WINDOW_GEMV)],
        "smoke": _NO_SMOKE},
    # BF16 (q1792): prefill KL under an ``--enforce-eager`` serve, both
    # residencies; the receipt records an identical greedy continuation from
    # all four census arms and does not grade it.
    "tessera_bf16_k1_dense_sm121_decode": {
        "grade": "route_only", "kl": [], "smoke": _uncontrolled("recorded", BF16)},
    "tessera_bf16_k1_dense_sm121_batch": {
        "grade": "kl_lower_bound", "kl": [_bound("batch", ["eager"], BF16)],
        "smoke": _uncontrolled("recorded", BF16)},
    # Routed MoE (LFM2.5-8B-A1B, q1024): prefill top-1024 bound, eager,
    # resident; the decode regime has the census alone.  The greedy smoke was
    # repetitive AND the BF16 source repeats identically on the same prompt
    # (MOE_DEBT section 7), so the symptom is shared with the reference and the
    # cells say so where a gate reads rather than in prose (#195).
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident": {
        "grade": "route_only", "kl": [],
        "smoke": {"status": "repetitive", "receipt": LFM,
                  "attribution": "shared_with_reference", "control": _BF16_CONTROL}},
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident": {
        "grade": "kl_lower_bound", "kl": [_bound("batch", ["eager"], LFM)],
        "smoke": {"status": "repetitive", "receipt": LFM,
                  "attribution": "shared_with_reference", "control": _BF16_CONTROL}},
}


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


def _cells(contract):
    return {cell["id"]: cell for cell in contract["lane_eligibility"]["cells"]}


def _with_evidence(contract, cell_id, evidence):
    doc = copy.deepcopy(contract)
    # These mutations exercise KL/smoke grammar; artifact scope has its own
    # regression matrix in test_evidence_artifact.py.
    _cells(doc)[cell_id]["evidence"] = {
        "artifact": _cells(doc)[cell_id]["evidence"]["artifact"], **evidence}
    return doc


# --- what the packaged table says --------------------------------------------

def test_every_cell_states_its_evidence_receipt_for_receipt(contract):
    cells = _cells(contract)
    assert sorted(cells) == sorted(_EVIDENCE)
    for cell_id, expected in _EVIDENCE.items():
        assert {k: v for k, v in cells[cell_id]["evidence"].items()
                if k != "artifact"} == expected, cell_id


def test_the_stored_grade_is_the_derived_one(contract):
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["evidence"]["grade"] == derive_evidence_grade(cell), cell["id"]


def test_no_cell_claims_full_vocabulary_kl(contract):
    """The premise correction, read off the table: every KL this repository
    serves is a top-1024 intersection lower bound, so no cell may grade above
    ``kl_lower_bound`` until a full-vocabulary measurement exists."""
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["evidence"]["grade"] != "kl_full_vocab"
        for entry in cell["evidence"]["kl"]:
            assert entry["kind"] == "topk_intersection_lower_bound"
            assert entry["top_k"] == 1024


def test_the_routed_moe_cells_are_now_distinguishable_from_the_dense_ones(contract):
    """The defect #133 names, closed on the shipped file: a gate reading
    ``evidence`` alone tells the MoE decode cell (census only, repetitive
    smoke) from a dense batch cell (a bound in its own regime)."""
    cells = _cells(contract)
    moe_decode = cells["tessera_e4m3_k1_routed_moe_sm121_decode_resident"]["evidence"]
    moe_batch = cells["tessera_e4m3_k1_routed_moe_sm121_batch_resident"]["evidence"]
    assert moe_decode["grade"] == "route_only"
    assert moe_batch["grade"] == "kl_lower_bound"
    assert moe_decode["smoke"]["status"] == moe_batch["smoke"]["status"] == "repetitive"
    assert moe_batch["kl"][0]["execution_modes"] == ["eager"]


def test_only_the_streamed_e4m3_decode_cell_has_a_decode_regime_bound(contract):
    """Regime coverage is the axis that separates the decode cells, and the
    honest count is one: the window-GEMV lane is the only route a decode-regime
    KL was ever scored against (eager 2026-09-03, compiled r6)."""
    with_decode_kl = sorted(
        cell["id"] for cell in contract["lane_eligibility"]["cells"]
        if cell["regime"] == "decode" and cell["evidence"]["kl"])
    assert with_decode_kl == ["tessera_e4m3_k1_dense_sm121_decode_streamed"]


def test_every_named_receipt_is_in_the_tree(contract):
    """The wheel does not ship docs, so the validator checks the grammar of a
    receipt path and this tree test checks the file exists."""
    for cell in contract["lane_eligibility"]["cells"]:
        paths = [entry["receipt"] for entry in cell["evidence"]["kl"]]
        smoke = cell["evidence"]["smoke"]
        if smoke["receipt"] is not None:
            paths.append(smoke["receipt"])
        if smoke["control"] is not None:
            paths.append(smoke["control"]["receipt"])
        for path in paths:
            assert path.startswith(EVIDENCE_RECEIPT_ROOT)
            assert (ROOT / path).is_file(), f"{cell['id']} names a missing receipt {path}"


def test_the_grammar_is_exported_for_a_consumer(contract):
    assert EVIDENCE_KL_KINDS == ("topk_intersection_lower_bound", "full_vocab")
    assert EVIDENCE_SMOKE_STATUSES == ("recorded", "repetitive", "not_recorded")
    assert EVIDENCE_GRADES == ("route_only", "kl_lower_bound", "kl_full_vocab")
    assert EVIDENCE_RECEIPT_ROOT == "docs/measurements/"
    assert EVIDENCE_SMOKE_ATTRIBUTIONS == (
        "unattributed", "shared_with_reference", "not_shared_with_reference")
    assert EVIDENCE_CONTROL_REFERENCES == ("bf16_source",)
    assert EVIDENCE_CONTROL_OUTCOMES == ("identical_completion", "different_completion")


# --- the validator refuses what a gate could not read ------------------------

BATCH = "tessera_e4m3_k1_dense_sm121_batch_resident"
DECODE = "tessera_e4m3_k1_dense_sm121_decode_resident"


def test_a_cell_without_evidence_is_refused(contract):
    doc = copy.deepcopy(contract)
    del _cells(doc)[BATCH]["evidence"]
    with pytest.raises(ValueError, match=r"missing \['evidence'\]"):
        validate_serving_contract(doc)


def test_evidence_is_a_closed_object(contract):
    good = _EVIDENCE[BATCH]
    with pytest.raises(ValueError, match=r"evidence carries unknown field\(s\) \['detail'\]"):
        validate_serving_contract(_with_evidence(
            contract, BATCH, {**good, "detail": "screen-grade, prefill only"}))
    with pytest.raises(ValueError, match=r"evidence is missing \['smoke'\]"):
        validate_serving_contract(_with_evidence(
            contract, BATCH, {"grade": good["grade"], "kl": good["kl"]}))
    with pytest.raises(ValueError, match=r"evidence\.kl\[0\] carries unknown field\(s\) \['value'\]"):
        validate_serving_contract(_with_evidence(
            contract, BATCH, {**good, "kl": [{**good["kl"][0], "value": 0.466}]}))


def test_the_grade_cannot_be_asserted_above_or_below_its_entries(contract):
    over = {**_EVIDENCE[DECODE], "grade": "kl_lower_bound"}
    with pytest.raises(ValueError, match="grade is 'kl_lower_bound' but its kl entries derive 'route_only'"):
        validate_serving_contract(_with_evidence(contract, DECODE, over))
    under = {**_EVIDENCE[BATCH], "grade": "route_only"}
    with pytest.raises(ValueError, match="grade is 'route_only' but its kl entries derive 'kl_lower_bound'"):
        validate_serving_contract(_with_evidence(contract, BATCH, under))
    with pytest.raises(ValueError, match="grade 'measured' is not one of"):
        validate_serving_contract(_with_evidence(
            contract, BATCH, {**_EVIDENCE[BATCH], "grade": "measured"}))


def test_a_bound_in_another_regime_is_not_this_cells_evidence(contract):
    """The confusion #133 is about, refused at the bytes: a prefill bound
    written into a decode cell."""
    borrowed = {"grade": "kl_lower_bound",
                "kl": [_bound("batch", ["eager", "compiled"], PLUGIN)], "smoke": _NO_SMOKE}
    with pytest.raises(ValueError, match="regime 'batch' is not the cell's regime 'decode'"):
        validate_serving_contract(_with_evidence(contract, DECODE, borrowed))


@pytest.mark.parametrize("entry, match", [
    ({"kind": "screen", "top_k": 1024, "regime": "batch", "execution_modes": ["eager"],
      "receipt": PLUGIN}, "kind 'screen' is not one of"),
    ({"kind": "topk_intersection_lower_bound", "top_k": None, "regime": "batch",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "top_k must be a positive integer"),
    ({"kind": "topk_intersection_lower_bound", "top_k": 0, "regime": "batch",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "top_k must be a positive integer"),
    ({"kind": "full_vocab", "top_k": 1024, "regime": "batch",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "top_k must be null"),
    ({"kind": "topk_intersection_lower_bound", "top_k": 1024, "regime": "prefill",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "regime 'prefill'"),
    ({"kind": "topk_intersection_lower_bound", "top_k": 1024, "regime": "batch",
      "execution_modes": [], "receipt": PLUGIN}, "execution_modes"),
    ({"kind": "topk_intersection_lower_bound", "top_k": 1024, "regime": "batch",
      "execution_modes": ["eager", "eager"], "receipt": PLUGIN}, "execution_modes"),
    ({"kind": "topk_intersection_lower_bound", "top_k": 1024, "regime": "batch",
      "execution_modes": ["graph"], "receipt": PLUGIN}, "execution_modes"),
    ({"kind": "topk_intersection_lower_bound", "top_k": 1024, "regime": "batch",
      "execution_modes": ["eager"], "receipt": "/an/absolute/path/x.json"},
     "receipt must be a repository path under 'docs/measurements/'"),
    ({"kind": "topk_intersection_lower_bound", "top_k": 1024, "regime": "batch",
      "execution_modes": ["eager"], "receipt": "docs/measurements/../x.md"},
     "receipt must be a repository path under 'docs/measurements/'"),
    ({"kind": "topk_intersection_lower_bound", "top_k": 1024, "regime": "batch",
      "execution_modes": ["eager"], "receipt": None},
     "receipt must be a repository path under 'docs/measurements/'"),
], ids=lambda x: x if isinstance(x, str) else "entry")
def test_a_kl_entry_outside_the_grammar_is_refused(contract, entry, match):
    evidence = {"grade": "kl_lower_bound", "kl": [entry], "smoke": _NO_SMOKE}
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(_with_evidence(contract, BATCH, evidence))


def test_a_kl_entry_may_not_claim_an_execution_mode_the_cell_does_not_cover(contract):
    moe = "tessera_e4m3_k1_routed_moe_sm121_batch_resident"
    evidence = {"grade": "kl_lower_bound", "kl": [_bound("batch", ["eager", "compiled"], LFM)],
                "smoke": _EVIDENCE[moe]["smoke"]}
    with pytest.raises(ValueError, match=r"execution_modes \['compiled'\] the cell does not cover"):
        validate_serving_contract(_with_evidence(contract, moe, evidence))


def test_the_same_receipt_and_mode_cannot_be_counted_twice(contract):
    entry = _bound("batch", ["eager", "compiled"], PLUGIN)
    evidence = {"grade": "kl_lower_bound", "kl": [entry, entry], "smoke": _NO_SMOKE}
    with pytest.raises(ValueError, match="repeats"):
        validate_serving_contract(_with_evidence(contract, BATCH, evidence))


@pytest.mark.parametrize("smoke, match", [
    (_uncontrolled("coherent", BF16), "smoke.status 'coherent' is not one of"),
    (_uncontrolled("recorded", None), "names no receipt"),
    (_uncontrolled("repetitive", None), "names no receipt"),
    (_uncontrolled("not_recorded", BF16), "not_recorded names a receipt"),
    (_uncontrolled("recorded", "/an/absolute/path/x.md"),
     "receipt must be a repository path under 'docs/measurements/'"),
    ({**_uncontrolled("recorded", BF16), "text": "France is"},
     r"smoke carries unknown field\(s\) \['text'\]"),
    ({"receipt": BF16, "attribution": "unattributed", "control": None},
     r"smoke is missing \['status'\]"),
    ({"status": "recorded", "receipt": BF16},
     r"smoke is missing \['attribution', 'control'\]"),
], ids=lambda x: x if isinstance(x, str) else "smoke")
def test_a_smoke_outside_the_grammar_is_refused(contract, smoke, match):
    evidence = {**_EVIDENCE[BATCH], "smoke": smoke}
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(_with_evidence(contract, BATCH, evidence))


# --- the control a smoke was compared against (#195, schema v7) --------------

MOE_DECODE = "tessera_e4m3_k1_routed_moe_sm121_decode_resident"


def _smoke(status=None, control=None, **over):
    """A smoke record whose attribution is the DERIVED one, so a test that is
    not about the derivation cannot fail on it."""
    status = status or "repetitive"
    record = {"status": status, "receipt": LFM, "control": control}
    record["attribution"] = derive_smoke_attribution(record)
    return {**record, **over}


def test_a_control_tells_a_shared_symptom_from_a_route_specific_one(contract):
    """The defect #195 names, closed where a gate reads it.

    Both routed-MoE cells record `repetitive`, and the BF16 SOURCE returns the
    identical completion on the identical prompt, so the repetition is the
    model and the prompt.  A consumer refusing the lane on `status` alone was
    right under its rule and wrong about the runtime; `attribution` is the
    field that carries the decision.
    """
    for cell in contract["lane_eligibility"]["cells"]:
        smoke = cell["evidence"]["smoke"]
        if cell["structure"] != "routed_moe":
            assert smoke["control"] is None and smoke["attribution"] == "unattributed", cell["id"]
            continue
        assert smoke["status"] == "repetitive"
        assert smoke["attribution"] == "shared_with_reference", cell["id"]
        assert smoke["control"] == {"reference": "bf16_source",
                                    "outcome": "identical_completion",
                                    "receipt": MOE_DEBT}, cell["id"]


def test_the_stored_attribution_is_the_derived_one(contract):
    for cell in contract["lane_eligibility"]["cells"]:
        smoke = cell["evidence"]["smoke"]
        assert smoke["attribution"] == derive_smoke_attribution(smoke), cell["id"]


def test_no_cell_claims_a_control_that_differed(contract):
    """Read off the table, not asserted: the one control this repository has
    run came back identical.  A cell claiming `different_completion` would be
    claiming a measurement nobody took."""
    for cell in contract["lane_eligibility"]["cells"]:
        control = cell["evidence"]["smoke"]["control"]
        assert control is None or control["outcome"] == "identical_completion", cell["id"]


def test_a_control_that_differs_attributes_the_symptom_to_this_route(contract):
    """The other branch of the derivation, so the field is not a constant: a
    reference that answers differently leaves the symptom unexplained by the
    model and the prompt."""
    control = {"reference": "bf16_source", "outcome": "different_completion",
               "receipt": MOE_DEBT}
    evidence = {**_EVIDENCE[MOE_DECODE], "smoke": _smoke(control=control)}
    doc = _with_evidence(contract, MOE_DECODE, evidence)
    validate_serving_contract(doc)
    smoke = _cells(doc)[MOE_DECODE]["evidence"]["smoke"]
    assert smoke["attribution"] == "not_shared_with_reference"
    assert cell_evidence(_cells(doc)[MOE_DECODE], MOE_DECODE)["smoke"]["control"] == control


def test_an_attribution_cannot_be_asserted_beside_its_control(contract):
    """`attribution` is derived from `control` the way `grade` is derived from
    `kl`: stored so a reader needs no derivation, checked so it cannot drift."""
    over = {**_EVIDENCE[MOE_DECODE],
            "smoke": _smoke(control=_BF16_CONTROL, attribution="unattributed")}
    with pytest.raises(ValueError,
                       match="attribution is 'unattributed' but its control derives "
                             "'shared_with_reference'"):
        validate_serving_contract(_with_evidence(contract, MOE_DECODE, over))
    under = {**_EVIDENCE[BATCH],
             "smoke": {**_NO_SMOKE, "attribution": "shared_with_reference"}}
    with pytest.raises(ValueError,
                       match="attribution is 'shared_with_reference' but its control derives "
                             "'unattributed'"):
        validate_serving_contract(_with_evidence(contract, BATCH, under))


@pytest.mark.parametrize("control, match", [
    ({"reference": "the bf16 model", "outcome": "identical_completion", "receipt": MOE_DEBT},
     "control.reference 'the bf16 model' is not one of"),
    ({"reference": "bf16_source", "outcome": "looked fine", "receipt": MOE_DEBT},
     "control.outcome 'looked fine' is not one of"),
    ({"reference": "bf16_source", "outcome": "identical_completion",
      "receipt": "/home/somebody/smoke.log"},
     "receipt must be a repository path under 'docs/measurements/'"),
    ({"reference": "bf16_source", "outcome": "identical_completion", "receipt": MOE_DEBT,
      "note": "same prompt"},
     r"control carries unknown field\(s\) \['note'\]"),
    ({"reference": "bf16_source", "receipt": MOE_DEBT},
     r"control is missing \['outcome'\]"),
], ids=["reference", "outcome", "receipt", "closed", "required"])
def test_a_control_outside_the_grammar_is_refused(contract, control, match):
    evidence = {**_EVIDENCE[MOE_DECODE],
                "smoke": {"status": "repetitive", "receipt": LFM,
                          "attribution": "shared_with_reference", "control": control}}
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(_with_evidence(contract, MOE_DECODE, evidence))


def test_a_smoke_nobody_ran_cannot_carry_a_control(contract):
    """`not_recorded` means no completion came back, so there is nothing for a
    reference to have matched."""
    evidence = {**_EVIDENCE[BATCH],
                "smoke": {"status": "not_recorded", "receipt": None,
                          "attribution": "shared_with_reference", "control": _BF16_CONTROL}}
    with pytest.raises(ValueError, match="status not_recorded names a control"):
        validate_serving_contract(_with_evidence(contract, BATCH, evidence))


def test_a_full_vocabulary_bound_grades_above_a_top_k_one(contract):
    """The grammar is ready for the measurement #133 asks for, so a future
    full-vocabulary KL is a receipt and an entry, never a schema change."""
    full = {"kind": "full_vocab", "top_k": None, "regime": "batch",
            "execution_modes": ["eager"], "receipt": PLUGIN}
    evidence = {"grade": "kl_full_vocab",
                "kl": [_bound("batch", ["eager", "compiled"], PLUGIN), full], "smoke": _NO_SMOKE}
    doc = _with_evidence(contract, BATCH, evidence)
    validate_serving_contract(doc)
    cell = _cells(doc)[BATCH]
    assert derive_evidence_grade(cell) == "kl_full_vocab"
    parsed = cell_evidence(cell, BATCH)
    assert parsed["grade"] == "kl_full_vocab" and len(parsed["kl"]) == 2
