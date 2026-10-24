"""Report-completeness checks, NOT served shared-fold qualification.

The report embeds one JSON evidence index alongside its dated narrative. These
CPU checks never launch a serve, trust a queued action as execution, or require
measurement mounts. A truthful blocked report may pass: attributed authority,
receipt authenticity, matched settings and numerical outcomes still need review.
"""
from datetime import date
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "measurements" / "shared-add-fold-1360.md"
RUNTIME = "4c0fd6aca04f1c3e95b094ae006c2479043c3372"


def _report():
    assert REPORT.is_file(), (
        "missing dated shared-fold campaign report: "
        "docs/measurements/shared-add-fold-1360.md"
    )
    text = REPORT.read_text(encoding="utf-8")
    records = [json.loads(block) for block in
               re.findall(r"```json\s*\n(.*?)\n```", text, re.DOTALL)]
    indices = [record for record in records if isinstance(record, dict)
               and record.get("campaign") == "tessera#1204"]
    assert len(indices) == 1, "report needs one #1204 JSON evidence index"
    return text, indices[0]


def _evidence(item):
    assert isinstance(item, dict), "each evidence slot needs an explicit outcome"
    assert item.get("status") in {"reused", "observed", "unavailable", "not_run"}
    assert isinstance(item.get("detail"), str) and item["detail"].strip(), (
        "name the actual outcome or concrete missing evidence, not just a status"
    )
    refs = item.get("refs")
    assert isinstance(refs, list)
    if item["status"] in {"reused", "observed"}:
        assert refs, "an observed/reused claim needs immutable evidence references"
    for ref in refs:
        assert isinstance(ref.get("uri"), str) and ref["uri"].strip()
        assert re.fullmatch(r"[0-9a-f]{64}", ref.get("sha256", "")), (
            "bind the referenced receipt/log/config bytes, not only a mutable path"
        )


def test_report_recovers_parent_and_records_admission_without_inventing_it():
    _, index = _report()
    assert date.fromisoformat(index["date"]).isoformat() == index["date"]
    assert index["runtime"] == RUNTIME
    recovery = index.get("parent_recovery")
    assert isinstance(recovery, list) and recovery, "recover existing arms first"
    run_ids = [entry["run_id"] for entry in recovery]
    assert len(run_ids) == len(set(run_ids)), "do not count a receipt twice"
    for entry in recovery:
        assert isinstance(entry["run_id"], str) and entry["run_id"].strip()
        _evidence(entry["outcome"])
    for name in ("owner_decision", "w1_artifact_readiness",
                 "source_qualification", "matched_arm_config"):
        assert name in index["evidence"], f"unaccounted prerequisite: {name}"
        _evidence(index["evidence"][name])


def test_report_accounts_for_both_served_outcomes_including_unavailable_evidence():
    _, index = _report()
    evidence = index["evidence"]
    required = (
        "base_readiness", "gate_readiness",
        "base_rank0_log", "base_rank1_log", "gate_rank0_log", "gate_rank1_log",
        "gate_install_and_folded_calls", "tr3_outputs_and_baseline_comparison",
        "rank0_base_gate_traces", "rank1_base_gate_traces",
        "latency", "both_host_power_and_work_per_joule", "execution_population",
    )
    for name in required:
        assert name in evidence, f"unaccounted served evidence: {name}"
        _evidence(evidence[name])
    assert index.get("disposition") in {"blocked", "partial", "measured"}
    if index["disposition"] == "measured":
        assert all(item["status"] in {"reused", "observed"}
                   for item in evidence.values()), (
            "unavailable/queued/not-run evidence cannot establish a served result"
        )
    assert index.get("default_changed") is False
    assert index.get("promotion_decision") == "out_of_scope"
