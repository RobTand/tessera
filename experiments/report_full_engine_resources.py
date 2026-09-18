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

from experiments.full_engine_resources import (
    analyze_engine_resource_ledger, runtime_provenance_relation,
)
from experiments.full_engine_resource_partition import assemble_full_engine_resource_report
from experiments.full_engine_kv import COMMON_RUN_DIGESTS

JOIN_SCHEMA = "tessera.full_engine_observation_join.v1"

#: The launcher's container environment names the roots the ownership rules
#: read: the observer tree on ``PYTHONPATH``, the plugin JIT extension dir and
#: the stock runtime's JIT caches. Read from the launch summary's own docker
#: argv, never assumed.
_JIT_CACHE_ENV = ("TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR")


def _container_environment(launch):
    """``{name: value}`` from the docker ``--env`` arguments of the capture phase."""
    environment = {}
    for phase in launch.get("phases") or ():
        if phase.get("phase") != "capture":
            continue
        argv = phase.get("command") or ()
        for index, item in enumerate(argv[:-1]):
            if item in ("-e", "--env") and "=" in argv[index + 1]:
                name, _, value = argv[index + 1].partition("=")
                environment[name] = value
    return environment


def ownership_evidence(*, plan, runtime_observation, per_job, core_manifest, launch,
                       jit_preflight, dense_startup):
    """The run's own inventories, in the shape ``full_engine_ownership`` reads.

    Nothing here is inferred from a path string alone: the plugin package path
    and roster come from the worker's loaded-package record and the installer's
    file table; the vLLM root and files from the attested core manifest; the
    observer roots and JIT prefixes from the launcher's recorded container
    environment; the observer libraries from the plan that loaded them.
    """
    loaded = runtime_observation["loaded_package"]
    environment = _container_environment(launch)
    observer_roots = [root for root in (environment.get("PYTHONPATH") or "").split(":") if root]
    if per_job.get("plugin_source_tree") and per_job["plugin_source_tree"] not in observer_roots:
        observer_roots.append(per_job["plugin_source_tree"])
    ext_dir = (jit_preflight or {}).get("ext_dir") or environment.get("TESSERA_EXT_DIR")
    observer_libraries = [plan["collector_library"]]
    workspace = plan.get("blas_workspace_observer")
    if isinstance(workspace, dict) and workspace.get("path"):
        observer_libraries.append(workspace["path"])
    return {
        "plugin_package_path": loaded["package_path"],
        "plugin_files": set(per_job["plugin_files"]),
        "vllm_root": core_manifest["root"],
        "vllm_files": set(core_manifest["files"]),
        "observer_roots": observer_roots,
        "observer_libraries": observer_libraries,
        "plugin_jit_prefix": (ext_dir.rstrip("/") + "/tessera_nvfp4/") if ext_dir else None,
        "jit_cache_prefixes": [environment[name] for name in _JIT_CACHE_ENV if environment.get(name)],
        "inventory_digests": {
            "plugin_source_sha256": per_job.get("plugin_source_sha256"),
            "plugin_installer_evidence_sha256": loaded.get("installer_evidence_sha256"),
            "core_manifest_sha256": per_job.get("core_manifest_sha256"),
            "collector_library_sha256": plan.get("collector_library_sha256"),
            "blas_workspace_observer_sha256": workspace.get("sha256") if isinstance(workspace, dict) else None,
            "plugin_jit_library_sha256": (jit_preflight or {}).get("library_sha256"),
        },
        "roster": plan["canonical_roster"],
        "dense_startup": dense_startup,
    }


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


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
    parser.add_argument("--timing-observation", type=Path,
                        help="the same-run TIMING pass's observation (timing-observation.json); "
                             "without it timing_partition stays open and timing_terms null")
    parser.add_argument("--launch-dir", type=Path,
                        help="the step-4 launcher output directory holding per-job-runtime.json, "
                             "launch-summary.json and jit-preflight.json (default: the parent of "
                             "--capture-dir)")
    parser.add_argument("--core-manifest", type=Path,
                        help="the attested vLLM core manifest (default: the plan's core_manifest path)")
    parser.add_argument("--without-ownership", action="store_true",
                        help="replay without the ownership derivation (the pre-#399 ledger)")
    parser.add_argument("--boundary-classification", type=Path,
                        help="tessera#548's two-capture comparison, built by "
                             "experiments/full_engine_boundary_classification.py from this "
                             "capture's ledger and the substitution capture's; without it every "
                             "census-shared site stays pending_548")
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
    launch_dir = args.launch_dir or args.capture_dir.parent
    runtime_observation = _read_json(worker_dir / "runtime-observation.json")
    per_job = _read_json(launch_dir / "per-job-runtime.json")
    launch = _read_json(launch_dir / "launch-summary.json")
    jit_preflight = _read_json(launch_dir / "jit-preflight.json")
    core_manifest = _read_json(args.core_manifest or plan["core_manifest"])
    startup_sidecar = _read_json(first_existing(worker_dir / "worker-startup.json",
                                                args.capture_dir / "worker-startup.json"))
    dense_startup = startup_sidecar.get("dense") if isinstance(startup_sidecar, dict) else None
    evidence_sources = {"runtime_observation": runtime_observation is not None,
                        "per_job_runtime": per_job is not None, "launch_summary": launch is not None,
                        "jit_preflight": jit_preflight is not None,
                        "core_manifest": core_manifest is not None,
                        "dense_startup": dense_startup is not None}
    evidence = None
    if not args.without_ownership:
        missing = [name for name in ("runtime_observation", "per_job_runtime", "launch_summary",
                                     "core_manifest") if not evidence_sources[name]]
        if missing:
            raise ValueError("ownership derivation needs the run's own inventories; missing: "
                             + ", ".join(missing) + " (pass --without-ownership for the raw replay)")
        evidence = ownership_evidence(plan=plan, runtime_observation=runtime_observation,
                                      per_job=per_job, core_manifest=core_manifest, launch=launch,
                                      jit_preflight=jit_preflight, dense_startup=dense_startup)
    classification = None
    if args.boundary_classification is not None:
        classification = json.loads(args.boundary_classification.read_text())
        if args.without_ownership:
            raise ValueError("a boundary classification is read by the ownership derivation; "
                             "--without-ownership replays without it")
    ledger = analyze_engine_resource_ledger(raw, evidence, classification)
    if evidence is not None and ledger.get("identity") is not None:
        ledger["runtime_provenance_relation"] = runtime_provenance_relation(
            ledger["identity"], plan=plan, launch=launch, per_job=per_job,
            runtime_observation=runtime_observation, core_file_count=len(core_manifest["files"]))
    timing_records = None
    if args.timing_observation is not None:
        timing_records = json.loads(Path(args.timing_observation).read_text())
        if not isinstance(timing_records, dict):
            raise ValueError("a timing observation is one record, not a list")
        ledger["timing_captures"] = timing_records
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
                                ("kv_observations", kv_records, kv_path),
                                ("timing_captures", timing_records, args.timing_observation)):
        if records is not None:
            path = Path(path)
            artifacts.append({"schema": "tessera.full_engine_observation_artifact.v1",
                              "observation": name, "path": str(path),
                              "sha256": _digest(path) if path.exists() else None,
                              "records": len(records) if isinstance(records, list) else 1})
    artifacts.append(join)
    artifacts.append({"schema": "tessera.full_engine_ownership_evidence_sources.v1",
                      "launch_dir": str(launch_dir), "sources": evidence_sources,
                      "ownership_derived": evidence is not None})
    report = assemble_full_engine_resource_report(ledger, reference=reference, workload=workload,
                                                  execution=execution, artifacts=artifacts)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ledger.json").write_text(json.dumps(ledger, sort_keys=True, indent=1) + "\n")
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=1) + "\n")
    derived = report["derived"]
    views = (ledger.get("owner_views") or {}).get("views") or {}
    summary = {"report": str(report_path), "sha256": _digest(report_path), "ledger_status": ledger["status"],
               "issues": len(ledger["issues"]),
               "admission": derived["admission"]["verdict"], "admission_reason": derived["admission"]["reason"],
               "domains": {name: domain["state"] for name, domain in report["partition"]["domains"].items()},
               "step_coverage": report["observations"]["step_coverage"]["state"],
               "allocations": len(ledger["torch_allocations"]),
               "classified": len(report["partition"]["membership"]),
               "unclassified": report["partition"]["scope"]["unclassified_allocation_count"],
               "uncharged": report["partition"]["scope"]["uncharged_allocation_count"],
               "non_step": report["partition"]["scope"]["non_step_allocation_count"],
               "observer": report["partition"]["scope"]["observer_allocation_count"],
               "owner_views": views.get("summary"),
               "external_records": {key: (ledger["owner_views"]["external_records"][key]
                                          if key != "unresolved" else len(ledger["owner_views"]["external_records"][key]))
                                    for key in ("record_count", "external_native_peak_bytes", "unresolved")}
               if ledger.get("owner_views") else None,
               "scalar_budget_bytes": derived["scalar_budget_bytes"],
               "fixed_resources_state": derived["fixed_resources"]["state"],
               "timing_terms": derived["timing_terms"],
               "terms": derived["terms"]}
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
