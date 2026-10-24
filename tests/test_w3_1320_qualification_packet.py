"""Evidence acceptance for the existing Tessera 1320 owner handoff.

This is a retained index, NOT a new authorization protocol or a GPU launcher.
Small supporting records live beside the index; large source/binary/input custody
can be referenced from those records. Review must authenticate the records and
check the actual W3 protocol, thresholds, freshness and authority independently.
No fixture or synthetic receipt can discharge that review.
"""
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/measurements/w3-1320/qualification.json"


def _packet():
    assert PACKET.is_file(), (
        "Missing owner-supplied Tessera 1320 W3 qualification packet: "
        "retain the successful setup action, adoption, exact-tuple decision and "
        "independent reviews; PENDING-GPU is not qualification")
    packet = json.loads(PACKET.read_text())
    assert packet["handoff"] == "tessera#1320"
    return packet


def _record(reference):
    """Verify retained bytes, not just a URL or an assertion that evidence exists."""
    relative = Path(reference["path"])
    assert not relative.is_absolute() and ".." not in relative.parts
    path = ROOT / relative
    assert path.is_file(), f"missing retained owner record: {relative}"
    data = path.read_bytes()
    assert data.strip(), f"empty retained owner record: {relative}"
    assert hashlib.sha256(data).hexdigest() == reference["sha256"], relative


def _identity(identity):
    assert re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])
    assert identity["custody"], "actual accessible artifact custody must be named"


def _tuple(value):
    assert re.fullmatch(r"[0-9a-f]{40}", value["source"]["commit"])
    _identity(value["source"])
    _identity(value["native_binary"])
    assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", value["runtime"]["image"])
    assert value["runtime"]["toolchain"]
    assert value["device"] and value["geometry"] and value["launches"]
    assert value["encoded_inputs"]
    for item in value["encoded_inputs"]:
        _identity(item)


def _run(run, authorized):
    assert run["tuple"] in authorized, "run is outside the reviewed exact tuples"
    _tuple(run["tuple"])
    _record(run["receipt"])
    _record(run["population"])
    assert run["command"].strip()
    assert run["duration_seconds"] > 0
    assert run["exit_code"] == 0
    for name in ("numerical", "replay", "frozen_storage", "no_redecode"):
        _record(run["results"][name])


def test_owner_prerequisites_bind_the_actual_execution_tuples():
    packet = _packet()
    prerequisites = packet["prerequisites"]
    for name in ("successful_setup_action", "adoption_note", "exact_tuple_decision",
                 "independent_prerun_review"):
        _record(prerequisites[name])
    assert prerequisites["execution_owner"] == "v6 Kernels Lead"
    assert prerequisites["review_owner"] == "v6 Review and Integration Lead"
    assert prerequisites["authorized_tuples"]
    for value in prerequisites["authorized_tuples"]:
        _tuple(value)


def test_protected_regression_precedes_any_missing_case_sweep():
    packet = _packet()
    pilot = packet["protected_regression"]
    assert pilot["tuple"]["geometry"] == {
        "rows": 416, "columns": 512, "q256": 1024, "m": 256}
    _run(pilot, packet["prerequisites"]["authorized_tuples"])
    # The retained ordering record must identify review -> pilot -> any sweep;
    # chronological authenticity is independently checked, not inferred here.
    _record(packet["execution_order"])


def test_six_case_w3_packet_has_receipts_and_independent_postrun_review():
    packet = _packet()
    _record(packet["w3_protocol"])
    cases = packet["cases"]
    assert len(cases) == 6 and len({case["id"] for case in cases}) == 6
    assert all(case["id"] for case in cases)
    for case in cases:
        _run(case, packet["prerequisites"]["authorized_tuples"])
    _record(packet["independent_postrun_review"])
