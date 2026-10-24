"""Retained evidence/decision contract for issue 1404, owned through issue 1387.

These CPU-only checks validate the retained index, not the truth of an owner's
identity or consent. Review must authenticate the cited decision independently.
No fleet paths are opened and no workload, admission or publication is triggered.
The JSON is a documentation record, not a new executable admission mechanism.
"""
from datetime import date
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DISPOSITION = ROOT / "docs/measurements/issue-652-owner-disposition.json"
NARRATIVE = ROOT / "docs/measurements/tessera-window-best-form-2026-09-09.md"
OWNER = "https://github.com/RobTand/tessera/issues/1387"
PACKET_SHA256 = "490b4fd62e20be1cbf832c7720ce6f4ab65119ab6dbc2743ec40c15f41f9567a"
# Identities recovered from the eight original receipts, not common HEAD labels.
SNAPSHOTS = {
    "df7dc3ca89f1e763a11c80b06cf682330664104a874ea308d3d8657994d2d597":
        "6a0f3e23b61d7f9ece87ff8c2339614592e1911601f2aa40abc048b1a1a18b00",
    "32dc9e7144502ac241ac9490488de7ec2122a07f5ef13c4cf53ac252c59e0769":
        "6e6de6abb87a1e760890cfde39f80edc59243e361c2bbd0f354094f12cf208be",
    "127fd4230045041e7974027024ce8d637d44322f5e8f6bedd8dceb5556ddac85":
        "748a5d0f6153f01dbf3427ee177bbc4bfbad2104c82cfba88d8a78534d83b0fb",
    "cdb631626dd1f9f76160b5a507965f73b91598aa4da303ae724d30290f4bcb7d":
        "5e401424e65cefba21243a4874e4b0dccfec2b10f2eff1a1c077c053eb0998df",
    "a7fc61ae6ef7ab2e274839d15739e73537e8a6ca620ccb59a9213c6b99bc9b5b":
        "680794177ac4e3c5ae59e3c7c97a861c499110e62aa7e8fc8a958cff33525602",
    "0b79337e87335b97de41650d776b5ac14bf8a69094758921e06152d1995e2ea6":
        "a65bbc7c9c5e9cef5a4d4c86bd7540eead6eefe1f81cf5b34d7a8c44a2f4a0d8",
    "925589c5cad78e160e34f1a3f2a366c677f31c3978f1f7ffcdd9e149c5abf889":
        "9a04c0905e962e24b631108e089e9441679d32c0a9d3159f78a17c6ef0352c45",
    "5e12c1972a2583bb3574819cbce5c74a0c70db54a08f77e2808feb8cdcbfd5ab":
        "c6084d6f84fca387f8cd855c6bc47fbd89fbb4ca0a2456d672cc877ca7889be1",
}


def _record():
    assert DISPOSITION.is_file(), (
        "Missing retained eight-admission disposition and attributable owner "
        "decision through existing issue 1387; a worker recommendation is not approval"
    )
    return json.loads(DISPOSITION.read_text(encoding="utf-8"))


def _text(value):
    assert isinstance(value, str) and value.strip(), value


def _sha256(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), value


def test_all_eight_admissions_have_scoped_digest_and_log_dispositions():
    record = _record()
    assert record["owner_issue"] == OWNER
    assert record["packet"]["raw_file_sha256"] == PACKET_SHA256
    _text(record["packet"]["reference"])
    _text(record["packet"]["verification"])
    _text(record["source_reconciliation"])
    rows = record["historical_admissions"]
    assert len(rows) == 8
    assert {row["action_key"] for row in rows} == set(SNAPSHOTS)
    for row in rows:
        assert row["snapshot_sha256"] == SNAPSHOTS[row["action_key"]]
        assert row["admission"]["gpu_exclusive"] is False
        _text(row["admission"]["scope"])
        assert row["completion"]["returncode"] == 0
        _text(row["completion"]["scope"])
        # Receipt canonical identity and raw-file identity have different contracts.
        receipt = row["receipt"]
        _sha256(receipt["canonical_sha256"])
        _sha256(receipt["raw_file_sha256"])
        assert receipt["canonical_sha256"] != receipt["raw_file_sha256"]
        for evidence in (receipt, row["manifest"], row["payload"]):
            _text(evidence["reference"])
            _text(evidence["verification"])
        _sha256(row["manifest"]["sha256"])
        _sha256(row["payload"]["sha256"])
        assert row["disposition"] in {"accepted", "rejected", "deferred"}
        _text(row["disposition_scope"])
        logs = row["logs"]
        assert len(logs) == 2
        assert {log["stream"] for log in logs} == {"stdout", "stderr"}
        for log in logs:
            _text(log["linked_reference"])
            _sha256(log["sha256"])
            assert type(log["bytes"]) is int and log["bytes"] >= 0
            assert log["availability"] in {"verified", "unavailable", "unreadable"}
            _text(log["verification"])
            # Missing standalone files can have independently verified embedded copies.
            recovered = log.get("recovered_copy")
            if recovered is not None:
                _text(recovered["reference"])
                assert recovered["sha256"] == log["sha256"]
                assert recovered["bytes"] == log["bytes"]
                _text(recovered["verification"])


def test_retained_decision_identifies_authority_scope_and_approved_tuple():
    decision = _record()["owner_decision"]
    assert isinstance(decision, dict), "Pending/recommended is not owner approval"
    date.fromisoformat(decision["date"])
    assert decision["kind"] in {"qualification", "scoped_adoption", "fresh_exact_tuple"}
    for field in ("authority", "authority_reference", "decision_reference",
                  "integration_custody", "scope"):
        _text(decision[field])
    approved = decision["approved_tuple"]
    # Structured evidence or precise artifact references are allowed; authenticity,
    # digest binding and actual delegation are separate, mandatory review checks.
    for field in ("effective_source", "harness_configuration", "input_artifacts_and_digests",
                  "runtime_device_conditions"):
        assert approved[field], f"Missing approved tuple component: {field}"
    assert decision["accepted_historical_evidence"], "State the exact accepted subset or 'none'"
    assert decision["remaining_measurements"], "State remaining work or an evidenced 'none'"


def test_measurement_narrative_links_disposition_and_existing_owner():
    text = NARRATIVE.read_text(encoding="utf-8")
    assert "issue-652-owner-disposition.json" in text, "No retained disposition link"
    assert OWNER in text or "tessera#1387" in text, "Missing existing owner linkage"
