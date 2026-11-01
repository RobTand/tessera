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
import importlib.util
import json
from pathlib import Path

import pytest

from glm_nightly_cells import nightly_ids

from tessera.serving.contract import (
    EVIDENCE_CONTROL_OUTCOMES,
    EVIDENCE_CONTROL_REFERENCES,
    EVIDENCE_GRADES,
    EVIDENCE_KL_KINDS,
    EVIDENCE_RECEIPT_ROOT,
    EVIDENCE_SMOKE_ATTRIBUTIONS,
    EVIDENCE_SMOKE_FORMS,
    EVIDENCE_SMOKE_INTERFACES,
    EVIDENCE_SMOKE_STATUSES,
    cell_evidence,
    derive_evidence_grade,
    derive_smoke_attribution,
    derive_smoke_status,
    load_serving_contract,
    smoke_status_is_derived,
    validate_serving_contract,
)

ROOT = Path(__file__).resolve().parents[1]

#: The instrument that owns the per-completion rule and the scoring, loaded from
#: the tree the way ``tests/test_moe_greedy_smoke_rule.py`` loads it: the wheel
#: does not ship ``experiments/``, so what the contract quotes from it is a TREE
#: test's business, exactly like the existence of a named receipt.
INSTRUMENT = "experiments/moe_greedy_smoke.py"
_SPEC = importlib.util.spec_from_file_location("moe_greedy_smoke", ROOT / INSTRUMENT)
instrument = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(instrument)

PLUGIN = "docs/measurements/tessera-serving-plugin-2026-09-02.md"
WINDOW_GEMV = "docs/measurements/tessera-window-gemv-served-2026-09-03.md"
DECODE_EAGER = "docs/measurements/tessera-decode-regime-kl-2026-09-03.md"
DECODE_COMPILED = "docs/measurements/tessera-compiled-decode-kl-r6-2026-09-04.md"
BF16 = "docs/measurements/tessera-bf16-route-served-2026-09-02.md"
LFM = "docs/measurements/tessera-lfm-campaign-2026-09-04.md"
MOE_DEBT = "docs/measurements/moe-evidence-debt-2026-09-04.md"
MOE_SMOKE = "docs/measurements/moe-smoke-recorded-2026-09-05.md"
GFX1201 = "docs/measurements/tessera-gfx1201-bf16-k1-served-2026-09-13.md"


def _bound(regime, modes, receipt):
    return {"kind": "topk_intersection_lower_bound", "top_k": 1024, "regime": regime,
            "execution_modes": modes, "receipt": receipt}


#: The open gap tessera#702 records: no CUDA-graph serve of GLM-5.3 computes
#: what eager computes, so no decode cell claims a compiled KL (tessera#1152).
GRAPH_GAP = "docs/measurements/2026-09-30-glm-nightly-cells-and-graph-equivalence.md"


def _graph_equality(regime, modes, receipt=GRAPH_GAP):
    return {"kind": "graph_equality", "top_k": None, "regime": regime,
            "execution_modes": modes, "receipt": receipt}

#: The ``kl`` kinds that bound quality and raise the grade. ``graph_equality``
#: records the open graph gap and never does (tessera#1152).
_BOUND_KINDS = ("topk_intersection_lower_bound", "full_vocab")


_NO_SMOKE = {"status": "not_recorded", "receipt": None,
             "attribution": "unattributed", "control": None, "record": None}


def _uncontrolled(status, receipt):
    """A smoke nobody ran a reference against and that carries no record: the
    shape every dense cell has, and an ASSERTED status (schema v9, #327) --
    a word read off a receipt that nothing here can check."""
    return {"status": status, "receipt": receipt,
            "attribution": "unattributed", "control": None, "record": None}


#: Section 6 of ``moe-smoke-recorded-2026-09-05.md``, transcribed verdict by
#: verdict: ``(prompt, form) -> (Tessera student, BF16 source)``.  P0-P3 are raw
#: continuations sent to ``/v1/completions``; P4-P6 carry the same content
#: through the checkpoint's own ``chat_template.jinja``.  The order is the
#: instrument's own (campaign first, then pure_greedy, each by prompt id).
_MOE_SMOKE_TABLE = {
    ("P0", "campaign"): ("repetitive", "repetitive"),
    ("P1", "campaign"): ("repetitive", "repetitive"),
    ("P2", "campaign"): ("recorded", "repetitive"),
    ("P3", "campaign"): ("repetitive", "repetitive"),
    ("P4", "campaign"): ("recorded", "recorded"),
    ("P5", "campaign"): ("recorded", "recorded"),
    ("P6", "campaign"): ("recorded", "recorded"),
    ("P0", "pure_greedy"): ("repetitive", "repetitive"),
    ("P1", "pure_greedy"): ("repetitive", "repetitive"),
    ("P2", "pure_greedy"): ("repetitive", "repetitive"),
    ("P3", "pure_greedy"): ("repetitive", "repetitive"),
    ("P4", "pure_greedy"): ("recorded", "recorded"),
    ("P5", "pure_greedy"): ("recorded", "recorded"),
    ("P6", "pure_greedy"): ("recorded", "recorded"),
}
_RAW_PROMPTS = ("P0", "P1", "P2", "P3")

#: Section 4.3 of ``tessera-gfx1201-bf16-k1-served-2026-09-13.md``, the same
#: instrument on the same fourteen (prompt, form) pairs, against the same BF16
#: reference served on the gfx1201 image.  Nothing on the route's arm cycles;
#: the reference cycles on P1 in both forms, which is why the derived
#: attribution is ``unattributed`` and not ``not_shared_with_reference``: the
#: attribution is about a symptom on THIS arm, and there is none.  That the
#: route did not reproduce the reference's P1 cycle is not a quality claim --
#: it is one prompt on a 0.6B model and the receipt says so.
_GFX1201_SMOKE_TABLE = {
    (prompt, form): ("recorded", "repetitive" if prompt == "P1" else "recorded")
    for form in ("campaign", "pure_greedy")
    for prompt in ("P0", "P1", "P2", "P3", "P4", "P5", "P6")
}


def _smoke_record(table):
    """An ``evidence.smoke.record`` from a (prompt, form) -> (arm, reference)
    table.  ``rule`` is the instrument's own string, not a copy."""
    return {"instrument": INSTRUMENT, "rule": instrument.RULE, "reference": "bf16_source",
            "rows": [{"prompt": prompt, "form": form,
                      "interface": ("raw_completion" if prompt in _RAW_PROMPTS
                                    else "chat_template"),
                      "status": arm, "reference_status": reference}
                     for (prompt, form), (arm, reference) in table.items()]}


def _moe_smoke_record():
    """The ``evidence.smoke.record`` the two routed-MoE cells carry.

    ``rule`` is the instrument's own string, not a copy: a restatement here
    would pass on the day the two disagree, which is the whole reason the
    contract quotes it rather than paraphrasing it.
    """
    return _smoke_record(_MOE_SMOKE_TABLE)


def _recorded_against_reference(receipt, record):
    """A smoke whose word is DERIVED from its record, not asserted beside it."""
    smoke = {"status": None, "receipt": receipt, "control": None, "record": record}
    smoke["status"] = derive_smoke_status(smoke)
    smoke["attribution"] = derive_smoke_attribution(smoke)
    return smoke


#: The control section 7 of the MoE evidence-debt receipt ran: the same prompt,
#: byte for byte, against the BF16 SOURCE on the same pinned image, eager --
#: and the completion came back identical character for character.
_BF16_CONTROL = {"reference": "bf16_source", "outcome": "identical_completion",
                 "receipt": MOE_DEBT}
_ROUTE_ONLY = {"grade": "route_only", "kl": [], "smoke": _NO_SMOKE}

# The grammar tests use current cell scopes, not a second evidence roster.
_GRAMMAR_CONTRACT = load_serving_contract()
_EVIDENCE = {cell["id"]: {key: value for key, value in cell["evidence"].items()
                        if key != "artifact"}
             for cell in _GRAMMAR_CONTRACT["lane_eligibility"]["cells"]}

#: The evidence the pin-image dense E2M1 pair carried until contract v39
#: withdrew it, as v38 published it.  The 2026-09-02 receipt is real and stays
#: in the tree; what v39 retracted is the claim that this build executes the
#: launch it measured.
_WITHDRAWN_V39_EVIDENCE = {
    "tessera_e2m1_k2_dense_sm121_decode": _ROUTE_ONLY,
    "tessera_e2m1_k2_dense_sm121_batch": {
        "grade": "kl_lower_bound",
        "kl": [{**_bound("batch", ["eager", "compiled"], PLUGIN), "q256": 896}],
        "smoke": _NO_SMOKE},
}

#: The evidence the two routed E4M3 cells carried until contract v38 withdrew
#: them (tessera#604), as the v37 table published it.  The receipts are real
#: and stay in the tree; what v38 retracted is the claim that this build serves
#: the arithmetic they measured.  Quoted rather than dropped, because the smoke
#: record is the only real one this repository holds, and the grammar tests
#: below need a real record, not an invented one.
_WITHDRAWN_V38_FIXTURE = ROOT / "tests" / "fixtures" / "lane_eligibility_cells_withdrawn_v38.json"
_WITHDRAWN_V38 = {cell["id"]: cell for cell in
                  json.loads(_WITHDRAWN_V38_FIXTURE.read_text(encoding="utf-8"))["cells"]}
_WITHDRAWN_V38_EVIDENCE = {
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident": {
        "grade": "route_only", "kl": [],
        "smoke": _recorded_against_reference(MOE_SMOKE, _moe_smoke_record())},
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident": {
        "grade": "kl_lower_bound", "kl": [_bound("batch", ["eager"], LFM)],
        "smoke": _recorded_against_reference(MOE_SMOKE, _moe_smoke_record())},
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



def test_the_stored_grade_is_the_derived_one(contract):
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["evidence"]["grade"] == derive_evidence_grade(cell), cell["id"]


def test_no_cell_claims_full_vocabulary_kl(contract):
    """The premise correction, read off the table: every KL this repository
    serves is a top-1024 intersection lower bound, so no cell may grade above
    ``kl_lower_bound`` until a full-vocabulary measurement exists. A
    ``graph_equality`` record is not a KL bound: it cites the open graph gap
    (tessera#1152) and carries no top-K number."""
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["evidence"]["grade"] != "kl_full_vocab"
        for entry in cell["evidence"]["kl"]:
            if entry["kind"] == "graph_equality":
                assert entry["top_k"] is None
            else:
                assert entry["kind"] == "topk_intersection_lower_bound"
                assert entry["top_k"] == 1024


def test_the_routed_moe_cells_are_now_distinguishable_from_the_dense_ones(contract):
    """The defect #133 names, closed on the shipped file: a gate reading
    ``evidence`` alone tells the MoE decode cell (census only, a recorded
    smoke) from a dense batch cell (a bound in its own regime) and from the
    six dense E4M3 cells that never ran a smoke."""
    cells = _WITHDRAWN_V38  # the withdrawn pair, quoted (contract v38)
    moe_decode = cells["tessera_e4m3_k1_routed_moe_sm121_decode_resident"]["evidence"]
    moe_batch = cells["tessera_e4m3_k1_routed_moe_sm121_batch_resident"]["evidence"]
    assert moe_decode["grade"] == "route_only"
    assert moe_batch["grade"] == "kl_lower_bound"
    assert moe_decode["smoke"]["status"] == moe_batch["smoke"]["status"] == "recorded"
    assert moe_decode["smoke"]["receipt"] == moe_batch["smoke"]["receipt"] == MOE_SMOKE
    assert moe_batch["kl"][0]["execution_modes"] == ["eager"]


def test_the_quoted_withdrawn_evidence_is_what_v37_published_and_still_validates(contract):
    """The v38 quote is the v37 evidence, and the grammar still accepts it.

    Injected into the shipped cell that now holds the scope, the quoted smoke
    record validates: withdrawing the cells retracted a route claim, not the
    record's well-formedness, and the validator still has a real record to run
    on.  (The quoted KL entry cannot be re-injected: it names a batch-regime
    receipt the v38 cell's rung and image never measured, so it is only
    compared here.)
    """
    assert sorted(_WITHDRAWN_V38) == sorted(_WITHDRAWN_V38_EVIDENCE)
    for cell_id, expected in _WITHDRAWN_V38_EVIDENCE.items():
        quoted = _WITHDRAWN_V38[cell_id]
        assert {k: v for k, v in quoted["evidence"].items() if k != "artifact"} == expected
        assert quoted["executes"] == [{"symbol": "vllm.fused_moe.modular_kernel",
                                       "decoder": "torch_materialize_stock"}]
        shipped = _cells(contract)[cell_id]
        assert shipped["runtime"]["image"] != quoted["runtime"]["image"]
        assert shipped["evidence"]["smoke"] == _NO_SMOKE
        validate_serving_contract(_with_evidence(contract, cell_id, {
            "grade": "route_only", "kl": [], "smoke": quoted["evidence"]["smoke"]}))


def test_the_stored_smoke_status_is_the_derived_one(contract):
    """The defect #327 names, closed: the word is COMPUTED from the record.

    ``status`` is to ``smoke.record`` what ``grade`` is to ``kl`` -- stored so a
    reader needs no derivation, derived so it cannot be asserted, checked so it
    cannot drift.  Before this, the step from fourteen recorded completions to
    one published word was stated only in a dated measurement file and verified
    by substring-matching two of its markdown lines, so re-running the smoke
    with a different result changed no test.
    """
    for cell in contract["lane_eligibility"]["cells"]:
        smoke = cell["evidence"]["smoke"]
        assert smoke["status"] == derive_smoke_status(smoke), cell["id"]


def test_the_routed_moe_word_follows_the_rows_and_not_the_other_way_round(contract):
    """Re-scoring the record must be able to MOVE the word (#327).

    A derivation nothing can flip is a restatement.  Each mutation below is a
    different smoke result the same fourteen prompts could have returned, and
    each one changes the published word or the attribution -- so a re-run that
    came back differently could not leave this cell's `recorded` standing.
    """
    cells = _WITHDRAWN_V38  # the one real record, quoted since contract v38
    for cell_id in ("tessera_e4m3_k1_routed_moe_sm121_decode_resident",
                    "tessera_e4m3_k1_routed_moe_sm121_batch_resident"):
        smoke = cells[cell_id]["evidence"]["smoke"]
        assert derive_smoke_status(smoke) == "recorded"
        assert derive_smoke_attribution(smoke) == "shared_with_reference"

        # The student cycles where the source answered: the degeneration is
        # this route's, and the word refuses.
        route_specific = copy.deepcopy(smoke)
        row = next(r for r in route_specific["record"]["rows"]
                   if r["interface"] == "chat_template")
        row["status"], row["reference_status"] = "repetitive", "recorded"
        assert derive_smoke_status(route_specific) == "repetitive"
        assert derive_smoke_attribution(route_specific) == "not_shared_with_reference"

        # The chat interface stops answering on both arms: every cycle is still
        # shared, but there is no positive record left to publish.
        no_positive = copy.deepcopy(smoke)
        for row in no_positive["record"]["rows"]:
            if row["interface"] == "chat_template":
                row["status"] = row["reference_status"] = "repetitive"
        assert derive_smoke_status(no_positive) == "repetitive"
        assert derive_smoke_attribution(no_positive) == "shared_with_reference"

        # The raw continuations stop cycling: nothing is left to attribute.
        never_cycled = copy.deepcopy(smoke)
        for row in never_cycled["record"]["rows"]:
            if row["interface"] == "raw_completion":
                row["status"] = row["reference_status"] = "recorded"
        assert derive_smoke_status(never_cycled) == "recorded"
        assert derive_smoke_attribution(never_cycled) == "unattributed"


def test_an_empty_completion_cannot_flip_the_word(contract):
    """The hole #327 names at the aggregation, not only at the instrument.

    ``experiments/moe_greedy_smoke.py`` scores an empty completion
    ``not_recorded``, so a row made of two non-answers cannot be a positive
    record here either: strip the six real ones and the word falls to
    ``repetitive``, however many empty rows are added beside them.
    """
    smoke = copy.deepcopy(
        _WITHDRAWN_V38["tessera_e4m3_k1_routed_moe_sm121_batch_resident"]["evidence"]["smoke"])
    for row in smoke["record"]["rows"]:
        if row["interface"] == "chat_template":
            row["status"] = row["reference_status"] = "not_recorded"
    assert derive_smoke_status(smoke) == "repetitive"
    assert instrument.classify([])["status"] == "not_recorded"


def test_the_record_quotes_the_rule_and_the_instrument_that_owns_it(contract):
    """One rule, one home: the contract quotes the instrument's own statement
    of the per-completion rule and names the file it came from, and both are
    checked against the tree.  A paraphrase would pass on the day the rule
    changed under it."""
    for cell in contract["lane_eligibility"]["cells"]:
        record = cell["evidence"]["smoke"]["record"]
        if record is None:
            continue
        assert record["instrument"] == INSTRUMENT, cell["id"]
        assert (ROOT / record["instrument"]).is_file(), cell["id"]
        assert record["rule"] == instrument.RULE, cell["id"]


def test_the_instrument_writes_the_block_the_cell_carries(contract, tmp_path):
    """The receipt and the contract cannot disagree about the word.

    ``moe_greedy_smoke.py compare --subject --reference`` emits the
    ``evidence.smoke.record`` a cell transcribes AND the word
    ``contract.derive_smoke_status`` reads off it -- the same function the
    validator calls -- so the pair a future run writes is the block that goes
    into the cell, not a shape somebody re-types.  Reconstructed here from the
    shipped record's own rows.
    """
    import json

    shipped = _WITHDRAWN_V38[MOE_DECODE]["evidence"]["smoke"]
    tokens = {"recorded": list(range(40)), "repetitive": [1, 2] * 8}

    def results(which):
        rows = []
        for row in shipped["record"]["rows"]:
            verdict = row["status"] if which == "tessera" else row["reference_status"]
            request = {"max_tokens": 64}
            request["messages" if row["interface"] == "chat_template" else "prompt"] = "x"
            rows.append({"id": row["prompt"], "form": row["form"],
                         "interface": row["interface"], "request": request,
                         "completion": f"{which}-{row['prompt']}-{row['form']}",
                         "finish_reason": "stop", "token_ids": tokens[verdict]})
        return rows

    paths = {}
    for arm in ("bf16", "tessera"):
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(
            {"schema": "tessera.moe-greedy-smoke/1", "arm": arm, "rule": instrument.RULE,
             "tokenizer": {"tokenizer_json_sha256": "t"}, "prompts_sha256": "p",
             "results": results(arm)}))
        paths[arm] = str(path)

    out = tmp_path / "pair.json"
    instrument.main(["compare", paths["bf16"], paths["tessera"], "--out", str(out),
                     "--subject", "tessera", "--reference", "bf16_source"])
    emitted = json.loads(out.read_text())["contract_record"]
    assert emitted["record"] == shipped["record"]
    assert emitted["derived"] == {"status": shipped["status"],
                                  "attribution": shipped["attribution"]}


def test_the_records_vocabularies_are_the_instruments_own(contract):
    """The forms and interfaces a row may name are the ones the instrument
    exercises, derived from the module that owns them rather than restated."""
    assert EVIDENCE_SMOKE_FORMS == instrument.FORMS
    assert EVIDENCE_SMOKE_INTERFACES == instrument.INTERFACES
    assert instrument.interface_of({"messages": [{"role": "user", "content": "x"}]}) \
        == "chat_template"
    assert instrument.interface_of({"prompt": "The capital of France is"}) == "raw_completion"


def test_which_cells_publish_a_derived_word_and_which_assert_one(contract):
    """An asserted status is a named state, not a silence (#327).

    A cell with no record publishes a word nothing here can check, and
    ``smoke_status_is_derived`` is what says so.  The two dense BF16 cells were
    in that state -- their receipt predates the instrument: one greedy
    continuation from four census arms of the same route, no per-completion
    scoring, no reference arm -- and no test invented a record they did not
    measure.  Contract v31 withdrew them with the dispatch they attested
    (tessera#538), so the roster is now empty and the RULE is the whole claim:
    a cell has a derived word exactly when it carries the record to derive it
    from.  The empty list below is asserted rather than skipped, so the day a
    cell publishes an underivable word it lands here.
    """
    for cell in contract["lane_eligibility"]["cells"]:
        smoke = cell["evidence"]["smoke"]
        assert smoke_status_is_derived(smoke) is (smoke["record"] is not None), cell["id"]
    asserted = sorted(cell["id"] for cell in contract["lane_eligibility"]["cells"]
                      if cell["evidence"]["smoke"]["status"] != "not_recorded"
                      and not smoke_status_is_derived(cell["evidence"]["smoke"]))
    assert asserted == [], (
        "a cell publishing a smoke word with no record to derive it from is the defect #327 "
        "closed; if a new one appears, give it a record or say here why it cannot have one")


def test_the_routed_moe_smoke_carries_the_record_its_word_is_derived_from(contract):
    """What a consumer at the pin reads (#327).

    Under contract v21 the two routed-MoE cells' smoke was byte-identical to
    the two dense BF16 cells' -- same status, same ``unattributed``, same null
    control -- while seven of its fourteen completions ended in a cycle and
    none of the BF16 smoke did.  The record and the attribution it derives are
    what tell them apart, in fields a gate reads and not in prose.

    The BF16 half of that contrast was withdrawn at contract v31 (tessera#538),
    so what is left to pin is the positive side: the record is there, the
    attribution is derived from it, and no cell anywhere publishes the
    ``unattributed``-with-no-record shape the contrast was about.
    """
    cells = _WITHDRAWN_V38  # withdrawn at contract v38; the record is quoted
    moe = [cells[cid]["evidence"]["smoke"] for cid in
           ("tessera_e4m3_k1_routed_moe_sm121_decode_resident",
            "tessera_e4m3_k1_routed_moe_sm121_batch_resident")]
    for smoke in moe:
        assert smoke["status"] == "recorded"
        assert smoke["attribution"] == "shared_with_reference"
        cycled = [row for row in smoke["record"]["rows"] if row["status"] == "repetitive"]
        assert len(cycled) == 7
        assert {row["interface"] for row in cycled} == {"raw_completion"}
        assert all(row["reference_status"] == "repetitive" for row in cycled)
        # The counts `docs/ARCHITECTURE.md` states for the raw-continuation
        # interface, asserted off the record rather than restated in prose:
        # eight pairs, the SOURCE cycling on all eight and the STUDENT on seven.
        raw = [row for row in smoke["record"]["rows"]
               if row["interface"] == "raw_completion"]
        assert len(raw) == 8
        assert sum(row["reference_status"] == "repetitive" for row in raw) == 8
        assert sum(row["status"] == "repetitive" for row in raw) == 7
        answered = [row for row in smoke["record"]["rows"] if row["status"] == "recorded"]
        assert {row["interface"] for row in answered} == {"raw_completion", "chat_template"}
        assert len([row for row in answered if row["interface"] == "chat_template"]) == 6
    assert not [cell["id"] for cell in contract["lane_eligibility"]["cells"]
                if cell["evidence"]["smoke"]["status"] != "not_recorded"
                and cell["evidence"]["smoke"]["record"] is None]


def test_which_decode_cells_carry_a_decode_regime_bound(contract):
    """Regime coverage is the axis that separates the decode cells.

    A decode cell carries a KL only where a decode-regime dump was scored
    against that route.  Two were: the sm_121 window-GEMV lane (eager
    2026-09-03, compiled r6) and the gfx1201 ``torch_window`` lane (#460,
    2026-09-13).  Every other decode cell is ``route_only``, and a prefill
    bound may not stand in for one -- the validator refuses a ``batch``
    kl entry on a ``decode`` cell, which is the rule re-read below.

    This was ``test_only_the_streamed_e4m3_decode_cell_has_a_decode_regime_bound``
    until #460 scored the second one; the count was never the claim.
    Contract v67 (tessera#1152) adds the next decode-regime record, and it is
    not a bound: every decode cell cites the open graph gap as
    ``graph_equality`` and still grades ``route_only``.
    """
    with_decode_kl = sorted(
        cell["id"] for cell in contract["lane_eligibility"]["cells"]
        if cell["regime"] == "decode"
        and any(entry["kind"] in _BOUND_KINDS for entry in cell["evidence"]["kl"]))
    # Both of them -- the sm_121 window-GEMV lane's and the gfx1201
    # ``torch_window`` lane's -- were scored against the dense dispatch that
    # ``1b767a207`` retired, and contract v31 withdrew their cells with it
    # (tessera#538).  No decode cell carries a bound today.  The list is
    # asserted, not skipped: the next decode-regime dump has to land here.
    assert with_decode_kl == []
    for cell in contract["lane_eligibility"]["cells"]:
        for entry in cell["evidence"]["kl"]:
            assert entry["regime"] == cell["regime"], cell["id"]


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
    assert EVIDENCE_KL_KINDS == ("topk_intersection_lower_bound", "full_vocab", "graph_equality")
    assert EVIDENCE_SMOKE_STATUSES == ("recorded", "repetitive", "not_recorded")
    assert EVIDENCE_GRADES == ("route_only", "kl_lower_bound", "kl_full_vocab")
    assert EVIDENCE_RECEIPT_ROOT == "docs/measurements/"
    assert EVIDENCE_SMOKE_ATTRIBUTIONS == (
        "unattributed", "shared_with_reference", "not_shared_with_reference")
    assert EVIDENCE_CONTROL_REFERENCES == ("bf16_source",)
    assert EVIDENCE_CONTROL_OUTCOMES == ("identical_completion", "different_completion")
    assert EVIDENCE_SMOKE_INTERFACES == ("raw_completion", "chat_template")
    assert EVIDENCE_SMOKE_FORMS == ("campaign", "pure_greedy")


# --- the validator refuses what a gate could not read ------------------------

# Select an eager dense pair that can carry the fixture's measured rung.
_BATCH_CELL = next(cell for cell in _GRAMMAR_CONTRACT["lane_eligibility"]["cells"]
                   if cell["structure"] == "dense" and cell["regime"] == "batch"
                   and cell["runtime"]["execution_modes"] == ["eager"]
                   and 896 in cell["rungs_q256"])
BATCH = _BATCH_CELL["id"]
DECODE = next(cell["id"] for cell in _GRAMMAR_CONTRACT["lane_eligibility"]["cells"]
              if cell["structure"] == "dense" and cell["regime"] == "decode"
              and cell["family"] == _BATCH_CELL["family"]
              and cell["runtime"] == _BATCH_CELL["runtime"])
_KL_BASE = {"grade": "kl_lower_bound",
            "kl": [{**_bound("batch", ["eager"], PLUGIN), "q256": 896}],
            "smoke": _NO_SMOKE}


def test_a_cell_without_evidence_is_refused(contract):
    doc = copy.deepcopy(contract)
    del _cells(doc)[BATCH]["evidence"]
    with pytest.raises(ValueError, match=r"missing \['evidence'\]"):
        validate_serving_contract(doc)


def test_evidence_is_a_closed_object(contract):
    good = _KL_BASE
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
    under = {**_KL_BASE, "grade": "route_only"}
    with pytest.raises(ValueError, match="grade is 'route_only' but its kl entries derive 'kl_lower_bound'"):
        validate_serving_contract(_with_evidence(contract, BATCH, under))
    with pytest.raises(ValueError, match="grade 'measured' is not one of"):
        validate_serving_contract(_with_evidence(
            contract, BATCH, {**_KL_BASE, "grade": "measured"}))


def test_a_bound_in_another_regime_is_not_this_cells_evidence(contract):
    """The confusion #133 is about, refused at the bytes: a prefill bound
    written into a decode cell."""
    borrowed = {"grade": "kl_lower_bound",
                "kl": [_bound("batch", ["eager"], PLUGIN)], "smoke": _NO_SMOKE}
    with pytest.raises(ValueError, match="regime 'batch' is not the cell's regime 'decode'"):
        validate_serving_contract(_with_evidence(contract, DECODE, borrowed))


def test_a_decode_cell_may_cite_the_open_graph_gap(contract):
    """tessera#1152: the validator accepts a decode graph-equality record, and
    it grades ``route_only`` -- the cell cites the gap receipt and claims no
    compiled KL."""
    evidence = {"grade": "route_only", "kl": [_graph_equality("decode", ["eager"])],
                "smoke": _NO_SMOKE}
    doc = _with_evidence(contract, DECODE, evidence)
    validate_serving_contract(doc)
    parsed = cell_evidence(_cells(doc)[DECODE], DECODE)
    assert parsed["grade"] == "route_only"
    assert parsed["kl"] == [_graph_equality("decode", ["eager"])]


def test_a_graph_equality_record_is_decode_only(contract):
    """tessera#1152: a graph serve replays decode batches, so a batch-regime
    graph-equality record attests nothing, in either cell; and a decode-regime
    one is not a batch cell's evidence."""
    batch_regime = {"grade": "route_only", "kl": [_graph_equality("batch", ["eager"])],
                    "smoke": _NO_SMOKE}
    with pytest.raises(ValueError, match="decode-only"):
        validate_serving_contract(_with_evidence(contract, DECODE, batch_regime))
    with pytest.raises(ValueError, match="decode-only"):
        validate_serving_contract(_with_evidence(contract, BATCH, batch_regime))
    misplaced = {"grade": "route_only", "kl": [_graph_equality("decode", ["eager"])],
                 "smoke": _NO_SMOKE}
    with pytest.raises(ValueError, match="is not the cell's regime"):
        validate_serving_contract(_with_evidence(contract, BATCH, misplaced))


def test_a_graph_equality_record_carries_no_top_k(contract):
    """tessera#1152: the record is not a top-K bound, so a number beside it is refused."""
    numbered = {"grade": "route_only",
                "kl": [{**_graph_equality("decode", ["eager"]), "top_k": 1024}],
                "smoke": _NO_SMOKE}
    with pytest.raises(ValueError, match="top_k must be null"):
        validate_serving_contract(_with_evidence(contract, DECODE, numbered))


def test_decode_cells_cite_the_graph_gap_and_batch_cells_carry_nothing(contract):
    """tessera#1152, read off the shipped file: every decode cell cites the
    gap receipt with grade ``route_only`` and eager scope; every batch cell
    carries no ``kl`` entry."""
    for cell in contract["lane_eligibility"]["cells"]:
        assert cell["runtime"]["execution_modes"] == ["eager"], cell["id"]
        if cell["regime"] == "decode":
            assert cell["evidence"]["grade"] == "route_only", cell["id"]
            assert [entry["kind"] for entry in cell["evidence"]["kl"]] == [
                "graph_equality"], cell["id"]
            entry = cell["evidence"]["kl"][0]
            assert entry["receipt"] == GRAPH_GAP, cell["id"]
            assert entry["top_k"] is None and entry["regime"] == "decode", cell["id"]
            assert entry["execution_modes"] == ["eager"], cell["id"]
        else:
            assert cell["evidence"]["kl"] == [], cell["id"]


@pytest.mark.parametrize("entry, match", [
    ({"kind": "screen", "top_k": 1024, "regime": "batch", "execution_modes": ["eager"],
      "receipt": PLUGIN}, "kind 'screen' is not one of"),
    ({"kind": "topk_intersection_lower_bound", "top_k": None, "regime": "batch",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "top_k must be a positive integer"),
    ({"kind": "topk_intersection_lower_bound", "top_k": 0, "regime": "batch",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "top_k must be a positive integer"),
    ({"kind": "full_vocab", "top_k": 1024, "regime": "batch",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "top_k must be null"),
    ({"kind": "graph_equality", "top_k": 1024, "regime": "batch",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "top_k must be null"),
    ({"kind": "graph_equality", "top_k": None, "regime": "batch",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "decode-only"),
    ({"kind": "graph_equality", "top_k": None, "regime": "decode",
      "execution_modes": ["eager"], "receipt": PLUGIN}, "is not the cell's regime"),
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
    entry = _bound("batch", ["eager"], PLUGIN)
    evidence = {"grade": "kl_lower_bound", "kl": [entry, entry], "smoke": _NO_SMOKE}
    with pytest.raises(ValueError, match="repeats"):
        validate_serving_contract(_with_evidence(contract, BATCH, evidence))


def test_a_kl_entry_may_stamp_the_rung_its_receipt_measured(contract):
    """D2b (tessera#560): the receipt's scope travels with the entry.

    A KL bound scored at one rung bounds nothing at another (a top-k
    intersection bound is rate-dependent), so an entry may name the rung its
    receipt measured in ``q256`` -- and when it does, the rung must be one the
    cell publishes.  The pin-image dense batch cell stamped the 896 rung its
    2026-09-02 receipt measured until contract v39 withdrew it; a stamp outside
    the cell's rungs is refused.
    """
    stamped = {**_bound("batch", ["eager"], PLUGIN), "q256": 896}
    doc = _with_evidence(contract, BATCH, {
        "grade": "kl_lower_bound", "kl": [stamped], "smoke": _NO_SMOKE})
    validate_serving_contract(doc)
    assert cell_evidence(_cells(doc)[BATCH], BATCH)["kl"] == [stamped]
    outside = {**_bound("batch", ["eager"], PLUGIN),
               "q256": max(_cells(contract)[BATCH]["rungs_q256"]) + 1}
    with pytest.raises(ValueError):
        validate_serving_contract(_with_evidence(contract, BATCH, {
            "grade": "kl_lower_bound", "kl": [outside], "smoke": _NO_SMOKE}))


@pytest.mark.parametrize("smoke, match", [
    (_uncontrolled("coherent", BF16), "smoke.status 'coherent' is not one of"),
    (_uncontrolled("recorded", None), "names no receipt"),
    (_uncontrolled("repetitive", None), "names no receipt"),
    (_uncontrolled("not_recorded", BF16), "not_recorded names a receipt"),
    (_uncontrolled("recorded", "/an/absolute/path/x.md"),
     "receipt must be a repository path under 'docs/measurements/'"),
    ({**_uncontrolled("recorded", BF16), "text": "France is"},
     r"smoke carries unknown field\(s\) \['text'\]"),
    ({"receipt": BF16, "attribution": "unattributed", "control": None, "record": None},
     r"smoke is missing \['status'\]"),
    ({"status": "recorded", "receipt": BF16},
     r"smoke is missing \['attribution', 'control', 'record'\]"),
], ids=lambda x: x if isinstance(x, str) else "smoke")
def test_a_smoke_outside_the_grammar_is_refused(contract, smoke, match):
    evidence = {**_KL_BASE, "smoke": smoke}
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(_with_evidence(contract, BATCH, evidence))


# --- the control a smoke was compared against (#195, schema v7) --------------

MOE_DECODE = "tessera_e4m3_k1_routed_moe_sm121_decode_resident"


def _smoke(status=None, control=None, **over):
    """A smoke record whose attribution is the DERIVED one, so a test that is
    not about the derivation cannot fail on it."""
    status = status or "repetitive"
    record = {"status": status, "receipt": LFM, "control": control, "record": None}
    record["attribution"] = derive_smoke_attribution(record)
    return {**record, **over}


def test_a_control_tells_a_shared_symptom_from_a_route_specific_one(contract):
    """The field #195 added, and where the shipped table stands on it.

    Contract v18 put the BF16 SOURCE's identical completion on the two
    routed-MoE cells as a control, so a consumer could tell a shared symptom
    from a route-specific one.  Contract v21 (#198) retired that control from
    the cells with the symptom, and the distinction went with it: after v21 no
    cell carried a control, all ten read `unattributed`, and the two cells whose
    smoke cycles on seven of fourteen completions were byte-identical to the two
    whose smoke never cycled.  Schema v9 (#327) puts the distinction back where
    the measurement is, not where a hand-written outcome word is: `attribution`
    is derived from the per-(prompt, form) `record`, so the routed-MoE cells read
    `shared_with_reference` on a scored comparison rather than on a summary.  The
    v7 control branch stays for a smoke with no record, and is exercised below.
    """
    for cell in contract["lane_eligibility"]["cells"]:
        smoke = cell["evidence"]["smoke"]
        assert smoke["control"] is None, cell["id"]
        assert smoke["attribution"] == derive_smoke_attribution(smoke), cell["id"]
        # The distinction lives on the cells that carry a record.  The E2M1
        # routed pair (v28) carries none and publishes `not_recorded`: being a
        # routed_moe cell does not make a smoke exist.
        if cell["structure"] == "routed_moe" and smoke["record"] is not None:
            assert smoke["status"] == "recorded", cell["id"]
            assert smoke["attribution"] == "shared_with_reference", cell["id"]
        else:
            assert smoke["attribution"] == "unattributed", cell["id"]
    routed_without_record = sorted(
        cell["id"] for cell in contract["lane_eligibility"]["cells"]
        if cell["structure"] == "routed_moe" and cell["evidence"]["smoke"]["record"] is None)

    for cell in _WITHDRAWN_V38.values():
        smoke = cell["evidence"]["smoke"]
        assert smoke["status"] == "recorded", cell["id"]
        assert smoke["attribution"] == "shared_with_reference", cell["id"]
    for cell_id in routed_without_record:
        assert _cells(contract)[cell_id]["evidence"]["smoke"] == _NO_SMOKE, cell_id


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
    under = {**_KL_BASE,
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
                "smoke": {"status": "repetitive", "receipt": LFM, "record": None,
                          "attribution": "shared_with_reference", "control": control}}
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(_with_evidence(contract, MOE_DECODE, evidence))


# --- the record a smoke's word is derived from (#327, schema v9) ------------

def _record(**over):
    return {**_moe_smoke_record(), **over}


def _rows(**over):
    return [{**_moe_smoke_record()["rows"][0], **over}]


def test_a_smoke_cannot_carry_both_a_control_and_a_record(contract):
    """One derivation, one home.  `attribution` is read off the record when there
    is one and off the control when there is not; a smoke holding both is two
    sources for one field, which is how they drift."""
    evidence = {**_WITHDRAWN_V38_EVIDENCE[MOE_DECODE],
                "smoke": {**_WITHDRAWN_V38_EVIDENCE[MOE_DECODE]["smoke"], "control": _BF16_CONTROL}}
    with pytest.raises(ValueError, match="carries BOTH a control and a record"):
        validate_serving_contract(_with_evidence(contract, MOE_DECODE, evidence))


def test_a_status_cannot_be_asserted_beside_its_record(contract):
    """`status` is to `record` what `grade` is to `kl`: read off it, never
    written beside it.  A word the rows do not derive is refused at the bytes,
    which is what makes a re-run of the smoke a test failure rather than a
    silent disagreement."""
    over = {**_WITHDRAWN_V38_EVIDENCE[MOE_DECODE],
            "smoke": {**_WITHDRAWN_V38_EVIDENCE[MOE_DECODE]["smoke"], "status": "repetitive"}}
    with pytest.raises(ValueError,
                       match="status is 'repetitive' but its record derives 'recorded'"):
        validate_serving_contract(_with_evidence(contract, MOE_DECODE, over))
    under = {**_WITHDRAWN_V38_EVIDENCE[MOE_DECODE],
             "smoke": {**_WITHDRAWN_V38_EVIDENCE[MOE_DECODE]["smoke"], "attribution": "unattributed"}}
    with pytest.raises(ValueError,
                       match="attribution is 'unattributed' but its record derives "
                             "'shared_with_reference'"):
        validate_serving_contract(_with_evidence(contract, MOE_DECODE, under))


@pytest.mark.parametrize("record, match", [
    # The instrument's own file, spelled absolutely: refused because it is
    # absolute, not because it is wrong.  Derived from the repository root this
    # file already resolves -- a path literal into somebody's home is a claim
    # about which machine the suite runs on (`tests/test_box_artifacts.py`).
    (_record(instrument=str(ROOT / INSTRUMENT)), "instrument must be a repository path"),
    (_record(instrument="experiments/../etc/passwd"), "instrument must be a repository path"),
    (_record(rule="  "), "rule must state the per-completion rule"),
    (_record(reference="the bf16 model"), "reference 'the bf16 model' is not one of"),
    (_record(rows=[]), "rows must be a non-empty JSON array"),
    (_record(rows=_rows(form="chat")), "form 'chat' is not one of"),
    (_record(rows=_rows(interface="/v1/completions")),
     "interface '/v1/completions' is not one of"),
    (_record(rows=_rows(status="coherent")), "status 'coherent' is not one of"),
    (_record(rows=_rows(reference_status=None)), "reference_status None is not one of"),
    (_record(reference=None), "but the record names no reference arm"),
    (_record(rows=_rows() * 2), r"repeats \(prompt, form\)"),
    (_record(rows=[{"prompt": "P0", "form": "campaign", "status": "repetitive",
                    "reference_status": "repetitive"}]),
     r"rows\[0\] is missing \['interface'\]"),
    ({k: v for k, v in _record().items() if k != "rule"}, r"record is missing \['rule'\]"),
    (_record(note="from the receipt"), r"record carries unknown field\(s\) \['note'\]"),
], ids=["absolute", "traversal", "rule", "reference", "empty", "form", "interface", "status",
        "reference_status", "unreferenced", "duplicate", "row_closed", "required", "closed"])
def test_a_record_outside_the_grammar_is_refused(contract, record, match):
    evidence = {**_EVIDENCE[MOE_DECODE],
                "smoke": {**_EVIDENCE[MOE_DECODE]["smoke"], "record": record}}
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(_with_evidence(contract, MOE_DECODE, evidence))


def test_a_smoke_nobody_ran_cannot_carry_a_control(contract):
    """`not_recorded` means no completion came back, so there is nothing for a
    reference to have matched."""
    evidence = {**_KL_BASE,
                "smoke": {"status": "not_recorded", "receipt": None, "record": None,
                          "attribution": "shared_with_reference", "control": _BF16_CONTROL}}
    with pytest.raises(ValueError, match="status not_recorded names a control"):
        validate_serving_contract(_with_evidence(contract, BATCH, evidence))


def test_a_full_vocabulary_bound_grades_above_a_top_k_one(contract):
    """The grammar is ready for the measurement #133 asks for, so a future
    full-vocabulary KL is a receipt and an entry, never a schema change."""
    full = {"kind": "full_vocab", "top_k": None, "regime": "batch",
            "execution_modes": ["eager"], "receipt": PLUGIN}
    evidence = {"grade": "kl_full_vocab",
                "kl": [_bound("batch", ["eager"], PLUGIN), full], "smoke": _NO_SMOKE}
    doc = _with_evidence(contract, BATCH, evidence)
    validate_serving_contract(doc)
    cell = _cells(doc)[BATCH]
    assert derive_evidence_grade(cell) == "kl_full_vocab"
    parsed = cell_evidence(cell, BATCH)
    assert parsed["grade"] == "kl_full_vocab" and len(parsed["kl"]) == 2


def test_the_cells_v39_withdrew_are_quoted_with_the_evidence_they_carried():
    """Contract v39 withdrew the pin-image dense E2M1 pair (tessera#604): the
    fixture quotes them with the evidence they published, the batch cell's
    KL bound included, so the receipt stays attached to the claim it backed."""
    from withdrawn_cells import withdrawn_v39_cells

    quoted = {cell["id"]: cell for cell in withdrawn_v39_cells()}
    for cell_id, expected in _WITHDRAWN_V39_EVIDENCE.items():
        assert {k: v for k, v in quoted[cell_id]["evidence"].items()
                if k != "artifact"} == expected, cell_id
