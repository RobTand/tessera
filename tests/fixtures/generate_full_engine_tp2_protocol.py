"""Materialize visibly synthetic, content-bound TP2 protocol fixtures.

Run through PrismaBuild. These records exercise parsers and refusals; they are
not a GPU capture, runtime receipt, quality measurement or admission evidence.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from experiments.full_engine_resources import analyze_engine_resource_ledger
from experiments.full_engine_resource_partition import (assemble_full_engine_resource_report,
                                                       timing_partition_closed)
from experiments.full_engine_timing_observation import build_timing_observation
from experiments.report_full_engine_resources import resource_rank_world


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"path": str(path), "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture_source = Path(__file__).resolve().parents[1] / "test_full_engine_timing_tp2.py"
    spec = importlib.util.spec_from_file_location("tp2_cpu_fixture", fixture_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    timing_root = out / "timing"
    timing_root.mkdir()
    timing_capture, timing_launch = module._write_world(timing_root)
    timing = build_timing_observation(timing_capture, launch_dir=timing_launch)
    if (timing["partition"]["established"] is not False
            or not all(rank.get("raw_terms") for rank in timing["ranks"])):
        raise ValueError("synthetic timing fixture lost raw terms or failed to refuse unsupported health")
    timing_ref = write(out / "timing-observation.json", timing)

    base = json.loads((Path(__file__).parent / "full_engine_resource_ledger.json").read_text())
    resource_dir = out / "resource"
    rank_runtime = {str(rank): write(resource_dir / f"runtime-rank{rank}.json",
        {"schema": "tessera.synthetic_installer_fixture.v1",
         "fixture_provenance": "synthetic CPU-only; not a runtime installation"})
        for rank in (0, 1)}
    plan = {"schema": "tessera.stock_engine_resource_observer_plan.v1",
            "world_size": 2, "identity": dict(module.DIGESTS),
            "canonical_source": module.SYNTHETIC_SOURCE,
            "selected_configuration": {"engine_args": {"tensor_parallel_size": 2,
                                                        "speculative_config": None},
                                       "environment": {"TESSERA_SERVE_MODE": "resident"}},
            "rank_runtime_evidence": rank_runtime,
            "fixture_provenance": "synthetic_cpu_protocol_fixture"}
    plan_ref = write(resource_dir / "observer-plan.json", plan)
    armed, workers, raw_by_rank = [], [], {}
    for rank in (0, 1):
        host = {"ip": f"192.168.1.{107 + rank}", "interface": "eth0",
                "source": "synthetic CPU fixture; no actual interface observed"}
        raw = copy.deepcopy(base)
        raw["identity"].update(module.DIGESTS, rank=rank, world_size=2,
                               device_id=0, device_uuid=f"GPU-rank-{rank}", host=host)
        raw["process_id"] = 4242 + rank
        raw["fixture_provenance"] = "synthetic CPU-only parser fixture, not a GPU measurement"
        directory = resource_dir / f"rank-{rank}-worker-{4242 + rank}"
        capture_ref = write(directory / "capture.json", raw)
        worker = {"rank": rank, "world_size": 2, "pid": 4242 + rank,
                  "device_id": 0, "device_uuid": f"GPU-rank-{rank}", "host": host,
                  "directory": str(directory),
                  "receipt": {"artifacts": {"capture.json": {"sha256": capture_ref["sha256"]}}}}
        workers.append(worker)
        armed.append({key: worker[key] for key in ("rank", "world_size", "pid", "device_id", "device_uuid", "host")})
        raw_by_rank[rank] = raw
    run = {"schema": "tessera.stock_engine_raw_resource_run.v1",
           "plan_sha256": plan_ref["sha256"], "workload_arm": armed, "workers": workers,
           "fixture_provenance": "synthetic_cpu_protocol_fixture"}
    run_ref = write(resource_dir / "run.json", run)
    rank_world = resource_rank_world(run, plan, resource_dir)
    report_refs = []
    for rank in (0, 1):
        ledger = analyze_engine_resource_ledger(raw_by_rank[rank])
        ledger["rank_world"] = rank_world
        ledger["timing_captures"] = timing
        if timing_partition_closed(ledger):
            raise ValueError("unsupported CUPTI health unexpectedly closed the synthetic timing world")
        unit = "synthetic:unit"
        report = assemble_full_engine_resource_report(
            ledger,
            reference={"canonical_census": {"units": [unit]},
                       "runtime_binding": {"member_formats": {unit: "synthetic"}},
                       "selected_rows": [{"unit": unit, "format": "synthetic"}]},
            workload={"calibration": {"sha256": "d" * 64}, "prompt_ids": [0],
                      "sampling": {"temperature": 0.0, "seed": 0,
                                   "max_tokens": 2, "ignore_eos": True}},
            execution={"graph_mode": "eager", "residency": "resident", "topology": "tp2"},
            artifacts=[{"schema": "tessera.synthetic_protocol_fixture.v1",
                        "raw_run": run_ref, "raw_plan": plan_ref},
                       {"schema": "tessera.full_engine_observation_artifact.v1",
                        "observation": "timing_captures", "path": timing_ref["path"],
                        "sha256": timing_ref["sha256"], "records": 1}])
        report_refs.append(write(out / f"report-rank{rank}.json", report))
    write(out / "manifest.json", {
        "schema": "tessera.synthetic_tp2_protocol_fixture.v1",
        "scope": "CPU parser fixture only; no GPU measurement or admission",
        "timing_observation": timing_ref, "reports": report_refs,
        "raw_resource_run": run_ref, "raw_resource_plan": plan_ref})
    print(json.dumps({"manifest": str(out / "manifest.json"), "reports": report_refs,
                      "timing_observation": timing_ref}, sort_keys=True))


if __name__ == "__main__":
    main()
