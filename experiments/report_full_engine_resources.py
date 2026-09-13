"""Assemble the full-engine resource report for one finished capture.

Reads the observer plan and the single worker capture a resource pass wrote,
replays the raw ledger, and assembles the frozen seven-member report. The three
members the ledger cannot supply -- reference, workload, execution -- are read
from the plan and the selected configuration, never invented here.
"""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.full_engine_resources import analyze_engine_resource_ledger
from experiments.full_engine_resource_partition import assemble_full_engine_resource_report


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def declared_members(plan):
    """The reference, workload and execution coordinates one plan declares."""
    roster = plan["canonical_roster"]
    assignment = plan["assignment"]["units"]
    units = [row["unit_id"] for row in roster]
    reference = {"canonical_census": {"units": units},
                 "runtime_binding": {"member_formats": {unit: assignment[unit] for unit in units}},
                 "selected_rows": [{"unit": unit, "format": assignment[unit]} for unit in units]}
    workload = plan["workload"]
    if "calibration" not in workload:
        raise ValueError("a report needs the calibration workload; a chat prompt names no fixture")
    calibration = workload["calibration"]
    declared_workload = {"calibration": {"sha256": calibration["sha256"]},
                         "prompt_ids": [calibration["row"]], "sampling": workload["sampling"]}
    engine = plan["selected_configuration"]["engine_args"]
    execution = {"graph_mode": "eager" if engine.get("enforce_eager") else "graph",
                 "residency": plan["selected_configuration"]["environment"].get("TESSERA_SERVE_MODE", "resident"),
                 "topology": f"tp{engine.get('tensor_parallel_size', 1)}"}
    return reference, declared_workload, execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True, help="the observer output directory")
    parser.add_argument("--output", type=Path, required=True, help="directory for ledger.json and report.json")
    args = parser.parse_args()
    plan = json.loads((args.capture_dir / "observer-plan.json").read_text())
    run = json.loads((args.capture_dir / "run.json").read_text())
    if len(run["workers"]) != 1:
        raise ValueError("a report covers exactly one worker capture")
    worker_dir = Path(run["workers"][0]["directory"])
    if not worker_dir.is_absolute() or not worker_dir.exists():
        worker_dir = args.capture_dir / worker_dir.name
    capture_path = worker_dir / "capture.json"
    raw = json.loads(capture_path.read_text())
    ledger = analyze_engine_resource_ledger(raw)
    reference, workload, execution = declared_members(plan)
    artifacts = [{"schema": "tessera.full_engine_capture_files.v1",
                  "observer_plan": {"path": str(args.capture_dir / "observer-plan.json"),
                                    "sha256": run["plan_sha256"]},
                  "capture": {"path": str(capture_path), "sha256": _digest(capture_path)},
                  "run": {"path": str(args.capture_dir / "run.json")}}]
    report = assemble_full_engine_resource_report(ledger, reference=reference, workload=workload,
                                                  execution=execution, artifacts=artifacts)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ledger.json").write_text(json.dumps(ledger, sort_keys=True, indent=1) + "\n")
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=1) + "\n")
    summary = {"report": str(report_path), "sha256": _digest(report_path), "ledger_status": ledger["status"],
               "issues": len(ledger["issues"]), "admission": ledger["admission"],
               "domains": {name: domain["state"] for name, domain in report["partition"]["domains"].items()},
               "step_coverage": report["observations"]["step_coverage"]["state"],
               "scalar_budget_bytes": report["derived"]["scalar_budget_bytes"],
               "terms": report["derived"]["terms"]}
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
