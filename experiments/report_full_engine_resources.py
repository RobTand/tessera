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
from experiments.full_engine_kv import COMMON_RUN_DIGESTS

JOIN_SCHEMA = "tessera.full_engine_observation_join.v1"


def read_observation_list(path, *keys):
    """One observation sidecar, as a list, or ``None`` when it is not present.

    Both sidecars are written by a pass and read here; a bare list and a named
    block are the two spellings a pass may use, and anything else is refused
    rather than guessed.
    """
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return value
    for key in keys:
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return value[key]
    raise ValueError(f"{path}: observation sidecar carries no {'/'.join(keys)} list")


def first_existing(*paths):
    """The first path that exists, or the last one as the named default.

    The startup sample is written by the worker at arm, before its ledger
    directory exists, so it lives beside the capture rather than inside it;
    newer passes may also write it into the worker directory. Reading either is
    not a guess about the shape -- both are the same sidecar -- and the chosen
    path is recorded in the report's artifacts.
    """
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return Path(paths[-1])


def join_observation_passes(ledger, *, resource_process_id, startup_records, kv_records,
                            capacity_witness=None):
    """Bind the two passes to ONE run and ONE configured capacity.

    The intrusive resource pass owns the ledger and the startup sample; the KV
    record that may close ``cache_capacity`` has to come from a READ-ONLY pass,
    which is a different process.  What must agree is the run identity the two
    passes were started under and the capacity the engine was configured with;
    what must NOT be compared is anything about their pointers.  Two processes
    allocate their own backings, so their addresses and their process ids are
    expected to differ and the extent in bytes is the only thing the consumer
    compares.
    """
    identity = ledger.get("identity") or {}
    rank, world_size = identity.get("rank"), identity.get("world_size")
    notes = {"schema": JOIN_SCHEMA,
             "run": {name: identity.get(name) for name in COMMON_RUN_DIGESTS},
             "rank": rank, "world_size": world_size,
             "resource_pass": {"capture_sha256": ledger.get("capture_sha256"),
                               "process_id": resource_process_id,
                               "startup_records": len(startup_records or [])},
             "read_only_pass": None, "capacity_witness": None,
             "different_processes": None, "pointer_identities_compared": False,
             "scope": ("the two passes are the same configured run on the same box; their "
                       "process ids and pointer identities are their own and are never "
                       "compared, only the configured capacity and the byte extent are")}
    if (startup_records or kv_records) and (rank is None or world_size is None):
        raise ValueError("per-rank observations require a rank-scoped run identity")
    if kv_records is not None:
        if len(kv_records) != 1:
            raise ValueError("a report carries exactly one KV observation")
        record = kv_records[0]
        run = record.get("run_identity") or {}
        for name, value in notes["run"].items():
            if value is not None and run.get(name) != value:
                raise ValueError(
                    f"kv observation {name} differs from this capture's run identity; the two "
                    "passes are not the same configured run")
        if record.get("rank") != rank or record.get("world_size") != world_size:
            raise ValueError(
                f"kv observation is rank {record.get('rank')}/{record.get('world_size')} and "
                f"this capture is {rank}/{world_size}")
        notes["read_only_pass"] = {
            "process_id": record.get("process_id"),
            "admission_evidence": record.get("admission_evidence"),
            "runtime_admission": record.get("runtime_admission"),
            "pointer_identities": "this pass's own; not compared with the resource pass"}
        notes["different_processes"] = (resource_process_id is not None
                                        and record.get("process_id") != resource_process_id)
        if capacity_witness is not None:
            witness = capacity_witness[0] if isinstance(capacity_witness, list) else capacity_witness
            if (witness.get("num_blocks") != record.get("num_blocks")
                    or witness.get("group_page_size_bytes") != record.get("group_page_size_bytes")):
                raise ValueError(
                    "the resource pass and the read-only pass were configured with different "
                    "KV capacity; their observations are not of the same pool")
            notes["capacity_witness"] = {
                "source": "the resource pass's own KV record",
                "num_blocks": witness.get("num_blocks"),
                "group_page_size_bytes": witness.get("group_page_size_bytes"),
                "runtime_admission": witness.get("runtime_admission"),
                "note": ("the witness states the configured capacity; it cannot close the domain, "
                         "because the intrusive pass is not admission-eligible")}
    return notes


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
    parser.add_argument("--startup-observation", type=Path,
                        help="the resource pass's worker-startup sidecar (default: the worker directory)")
    parser.add_argument("--kv-observation", type=Path,
                        help="the READ-ONLY pass's KV observation (default: the worker directory, "
                             "which is the intrusive pass's own record and cannot close the domain)")
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
    # The two observations travel as their own artifacts: the startup record is
    # this pass's own, and the KV record that may close `cache_capacity` comes
    # from a read-only pass, which is a different process by design.  The join
    # binds them to one run and one configured capacity and says so.
    startup_path = args.startup_observation or first_existing(
        worker_dir / "worker-startup.json", args.capture_dir / "worker-startup.json")
    kv_path = args.kv_observation or (worker_dir / "kv-observation.json")
    startup_records = read_observation_list(startup_path, "worker_startup_records", "records")
    kv_records = read_observation_list(kv_path, "kv_observations", "records")
    capacity_witness = read_observation_list(worker_dir / "kv-observation.json",
                                             "kv_observations", "records")
    join = join_observation_passes(ledger, resource_process_id=raw.get("process_id"),
                                   startup_records=startup_records, kv_records=kv_records,
                                   capacity_witness=(capacity_witness
                                                     if capacity_witness != kv_records else None))
    if startup_records is not None:
        ledger["worker_startup_records"] = startup_records
    if kv_records is not None:
        ledger["kv_observations"] = kv_records
    reference, workload, execution = declared_members(plan)
    artifacts = [{"schema": "tessera.full_engine_capture_files.v1",
                  "observer_plan": {"path": str(args.capture_dir / "observer-plan.json"),
                                    "sha256": run["plan_sha256"]},
                  "capture": {"path": str(capture_path), "sha256": _digest(capture_path)},
                  "run": {"path": str(args.capture_dir / "run.json")}}]
    for name, records, path in (("worker_startup_records", startup_records,
                                 startup_path),
                                ("kv_observations", kv_records, kv_path)):
        if records is not None:
            path = Path(path)
            artifacts.append({"schema": "tessera.full_engine_observation_artifact.v1",
                              "observation": name, "path": str(path),
                              "sha256": _digest(path) if path.exists() else None,
                              "records": len(records)})
    artifacts.append(join)
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
