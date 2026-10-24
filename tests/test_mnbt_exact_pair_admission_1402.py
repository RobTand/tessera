"""Offline acceptance of the narrow admission handoff, NOT serving admission.

The implementation supplies docs/measurements/mnbt-exact-pair-1402.json.
Evidence entries have value, references (durable locators), and missing. A null
value means unavailable, never an inferred historical/default value; missing
then names an issue URL and the exact owner deliverable. Non-null values have
missing=null. The reviewer must authenticate the references and their scope:
these CPU checks cannot grant authorization or attest external fleet state.
No network, historical Git objects, shared mounts, GPU work or submission tools
are needed to read this record on either test arm.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/measurements/mnbt-exact-pair-1402.json"
ISSUES = "https://github.com/RobTand/tessera/issues/"
RECOVERY = ("O2", "comparison_O3", "retained_record", "configuration_checks")
BINDINGS = (
    "source_effective_source", "producer", "model_artifact", "runtime_dependencies",
    "participants", "instruments_collection_scope", "execution_owner",
    "budget_window", "admission", "authorization",
)


def _record():
    assert RECORD.is_file(), (
        "Missing current exact-pair admission handoff: write "
        "docs/measurements/mnbt-exact-pair-1402.json; retain HOLD with precise "
        "owner deliverables where source-bound admission is unavailable"
    )
    result = json.loads(RECORD.read_text())
    assert isinstance(result, dict)
    return result


def _text(value):
    assert isinstance(value, str) and value.strip(), value


def _references(value):
    assert isinstance(value, list) and value, "evidence needs durable references"
    for ref in value:
        _text(ref)


def _entry(entry):
    assert isinstance(entry, dict)
    _references(entry["references"])
    if entry["value"] is None:
        missing = entry["missing"]
        assert isinstance(missing, dict)
        assert re.fullmatch(re.escape(ISSUES) + r"\d+", missing["issue"])
        _text(missing["deliverable"])
    else:
        assert entry["value"], "use null and a missing deliverable, not an empty identity"
        assert entry["missing"] is None


def test_stopped_contract_and_check_reuse_are_accounted_for():
    record = _record()
    recovery = record["recovery"]
    for name in RECOVERY:
        _entry(recovery[name])
    # The stop comment is evidence of the stop, not the missing objective text
    # or the source identities behind the reported 264 checks per arm.
    assert ISSUES + "1392#issuecomment-6366265912" in record["stop_references"]
    _text(record["recovery_investigation"])
    checks = recovery["configuration_checks"]
    assert checks["reuse_disposition"] in ("withheld", "identity-matched")
    if checks["reuse_disposition"] == "identity-matched":
        assert checks["value"] is not None
        for arm in ("x86", "gpu"):
            receipt = checks["value"][arm]
            _text(receipt["source_identity"])
            _text(receipt["effective_source_identity"])
            _references(receipt["references"])
        _text(checks["identity_reconciliation"])
    # A recovered count or recipe alone never establishes an authorized serve.
    assert record["configuration_checks_are_serving_evidence"] is False


def test_exact_pair_scope_and_current_configuration_are_explicit():
    record = _record()
    assert record["issue"] == ISSUES + "1402"
    scope = record["scope"]
    assert scope["window_mode"] == "ship-graph-mnbt-matrix"
    assert scope["matrix_cells"] == [[2048, 8192, 1], [4096, 8192, 1]]
    assert scope["tp"] == 2
    assert scope["serve_mode"] == "resident"
    assert scope["fabric"] == "socket"
    assert scope["max_num_seqs"] == 8, "request c1 does not change serving geometry"
    assert scope["max_model_len"] == 8448
    assert scope["graph_decode"] is True
    assert scope["mtp"] is False
    assert scope["lever_env"] == {
        "TESSERA_E4M3_DECODE_ONCE": "1",
        "TESSERA_GLM53_KDA_CONV_SPLIT": "on",
        "TESSERA_ROUTED_PIECE_MAJOR": "1",
    }
    inspected = record["inspected_source"]
    assert re.fullmatch(r"[0-9a-f]{40}", inspected["commit"])
    _references(inspected["references"])
    _text(record["configuration_reconciliation"])
    for name in BINDINGS:
        _entry(record["bindings"][name])


def test_disposition_requires_scoped_authority_or_precise_hold():
    record = _record()
    assert record["execution_owner_issues"] == [ISSUES + "1351", ISSUES + "1353"]
    assert record["campaign_owner_issue"] == ISSUES + "1348"
    _entry(record["evidence_reuse_1341"])
    assert record["serving_submitted"] is False, "this handoff is not a serving campaign"
    assert record["status"] in ("HOLD", "ADMITTED")
    entries = [record["recovery"][key] for key in RECOVERY]
    entries += [record["bindings"][key] for key in BINDINGS]
    missing = [entry for entry in entries if entry["value"] is None]
    if record["status"] == "HOLD":
        assert missing, "HOLD must identify a concrete missing owner deliverable"
        assert record["ready_for_serving_submission"] is False
        _text(record["hold_reason"])
    else:
        assert not missing, "unresolved contract/identity/authority cannot be ADMITTED"
        assert record["ready_for_serving_submission"] is True
        for name in ("admission", "authorization"):
            authority = record["bindings"][name]["value"]
            assert authority["matrix_cells"] == record["scope"]["matrix_cells"]
            _text(authority["issuer"])
            _text(authority["issued_at"])
            _text(authority["valid_window"])
            _references(authority["binding_references"])
