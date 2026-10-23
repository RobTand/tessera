"""Receipt-shape checks for tessera#1371, not synthetic GPU/C4 qualification.

The new dated Markdown record contains one fenced JSON object with schema
``tessera.f16-ring-c4.v1``. These checks make missing evidence and inconsistent
verdicts visible on both arms without rereading an entire measurement directory.
A reviewer must still authenticate remote receipts and the actual PC/SASS rows.
An ``unavailable`` disposition is NOT C4 success. Do not fabricate samples to
satisfy these checks when W2 or exact-tuple profiler qualification is unavailable.

Measured rows use the fused gate/up projection (one launch may cover both),
rank 0, R1024/M1, R1024/M512 and R1088/M512. Percentages are percentage points
of an explicitly documented sample denominator, not NCU timing. For each row,
baseline/candidate contain loop_pc, barrier_pc, scoreboard_dependency,
sass_excerpt, correspondence and samples. Samples name run_id, native_report,
pc_table, total_samples, long_scoreboard_pct and producer_barrier_pct.

The conservative predeclared noise estimator here is the range of independent
baseline shares. Every R1024 baseline mean minus candidate mean must exceed
that range for status=passed; a measured failed verdict retains the same matrix.
"""
from pathlib import Path
import json
import math
import re
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/measurements/2026-10-23-f16-ring-c4.md"
HISTORICAL = {
    "R1024": {"long_scoreboard_pct": [8.6, 7.0],
              "producer_barrier_pct": [11.9, 12.2]},
    "R1088": {"long_scoreboard_pct": [1.2],
              "producer_barrier_pct": [6.1]},
}


def _record():
    assert RECORD.is_file(), (
        "Missing new dated f16 C4 record: retain measured evidence or a concrete "
        "unavailability disposition; aggregate/MMA8 history is not C4."
    )
    blocks = re.findall(r"```json\s*\n(.*?)\n```", RECORD.read_text(), re.S)
    records = [json.loads(block) for block in blocks]
    records = [record for record in records
               if record.get("schema") == "tessera.f16-ring-c4.v1"]
    assert len(records) == 1, "Include one tessera.f16-ring-c4.v1 evidence summary"
    return records[0]


def _text(value):
    assert isinstance(value, str) and value.strip(), value


def _sha(value):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), value


def _pc(value):
    assert isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]+", value), value


def test_record_binds_provenance_and_never_promotes_missing_evidence():
    record = _record()
    assert record["status"] in {"passed", "failed", "unavailable"}
    assert record["c4_passed"] is (record["status"] == "passed")
    assert record["issue_739_accepted"] is False
    assert record["historical"] == HISTORICAL
    _text(record["reuse_audit"])
    for key, admitted in (("qualification", "qualified"), ("w2", "eligible")):
        assert record[key]["status"] in {admitted, "unavailable"}
        _text(record[key]["evidence"])
        if record["status"] != "unavailable":
            assert record[key]["status"] == admitted

    artifacts = record["artifacts"]
    assert artifacts, "Even unavailable work needs retained diagnostic/qualification evidence"
    paths = set()
    for artifact in artifacts:
        _text(artifact["kind"])
        _text(artifact["path"])
        _sha(artifact["sha256"])
        assert artifact["path"] not in paths
        paths.add(artifact["path"])

    if record["status"] == "unavailable":
        gap = record["unavailable"]
        for key in ("missing_capability_or_prerequisite", "evidence", "next_owner"):
            _text(gap[key])
        # Partial/failed probes can be described in prose, never as a full matrix.
        assert record["measurements"] == []
        return

    assert record["unavailable"] is None
    assert {"native_ncu", "pc_table", "sass", "command_log"} <= {
        artifact["kind"] for artifact in artifacts
    }
    for arm in ("baseline", "candidate"):
        binary = record["binaries"][arm]
        _sha(binary["source_sha256"])
        _sha(binary["binary_sha256"])
        _text(binary["build_manifest"])
    emitted = record["binaries"]["candidate"]["async_instructions"]
    assert emitted, "Identify actual emitted asynchronous global-to-shared SASS"
    for instruction in emitted:
        _pc(instruction["pc"])
        _text(instruction["sass"])
        assert "cp.async" not in instruction["sass"].lower(), "PTX is not emitted SASS"


def test_pc_matrix_has_samples_repeats_and_a_noise_aware_verdict():
    record = _record()
    if record["status"] == "unavailable":
        assert record["c4_passed"] is False
        assert record["measurements"] == []
        return  # The preceding test requires concrete retained gap evidence.

    assert record["noise_method"] == "baseline_range_pp"
    rows = record["measurements"]
    expected = {(1024, 1), (1024, 512), (1088, 512)}
    assert len(rows) == len(expected)
    assert {(row["r"], row["m"]) for row in rows} == expected
    paths = {artifact["path"] for artifact in record["artifacts"]}
    reductions = []
    for row in rows:
        # A report may contain several shapes; it cannot count as two repeats
        # of this same shape. Do not force one GPU invocation per matrix cell.
        all_runs = set()
        all_reports = set()
        assert row["rank"] == 0
        assert row["projection"] == "gate_up"
        _text(row["denominator"])
        shares = {}
        for arm, minimum in (("baseline", 2), ("candidate", 1)):
            profile = row[arm]
            _pc(profile["loop_pc"])
            _pc(profile["barrier_pc"])
            for key in ("scoreboard_dependency", "sass_excerpt", "correspondence"):
                _text(profile[key])
            samples = profile["samples"]
            assert len(samples) >= minimum
            shares[arm] = []
            for sample in samples:
                _text(sample["run_id"])
                assert sample["run_id"] not in all_runs, "Baseline repeats must be independent profiles"
                all_runs.add(sample["run_id"])
                assert sample["native_report"] in paths
                assert sample["native_report"] not in all_reports, "Do not reuse one native report as repeats"
                all_reports.add(sample["native_report"])
                assert sample["pc_table"] in paths
                assert isinstance(sample["total_samples"], int) and sample["total_samples"] > 0
                for metric in ("long_scoreboard_pct", "producer_barrier_pct"):
                    value = sample[metric]
                    assert isinstance(value, (int, float)) and not isinstance(value, bool)
                    assert math.isfinite(value) and 0 <= value <= 100
                shares[arm].append(sample["long_scoreboard_pct"])
        if row["r"] == 1024:
            noise = max(shares["baseline"]) - min(shares["baseline"])
            reduction = mean(shares["baseline"]) - mean(shares["candidate"])
            reductions.append(reduction > noise)
    assert len(reductions) == 2
    assert record["c4_passed"] is all(reductions), (
        "C4 passes only when BOTH R1024 loop-PC reductions exceed independent "
        "baseline repeat range; failed measurements must remain failed"
    )
