#!/usr/bin/env python3
"""Measure one explicitly frozen native dense operator (#688).

Repository application only: no scheduler, encoding, serving or cell publication.
Agents submit the command through PB; the library validator does not know PB.
The first slice is E4M3 dense, TP1, eager/resident, with one owned output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # Producer-only process, never shared with native worker.

from tessera.serving import census_plan, timing_panel as tp
from tools.run_glm_cached_cpu_export import fsync_path
if Path(tp.__file__).resolve() != ROOT / "src/tessera/serving/timing_panel.py":
    raise RuntimeError("producer schema origin differs from owned source")

def require_producer_origins():
    for name in ("timing_panel", "census_plan", "contract", "census", "scheme"):
        module = sys.modules["tessera.serving." + name]
        if Path(module.__file__).resolve() != ROOT / ("src/tessera/serving/" + name + ".py"):
            raise ValueError("foreign producer schema owner: " + name)


REQUEST_SCHEMA = "tessera.dense_shape_time_request.v1"


def publish_json(value, path):
    """A durable name, before any committed progress is reported."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(tp.canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_path(path.parent)
    return tp.file_binding(path)


PRODUCER_SCHEMA = "tessera.native_panel_producer_identity.v1"


def producer_source_identity():
    """Existing source-tree and tool-closure owners; no in-container Git required."""
    from experiments.full_engine_plugin_install import source_tree_identity
    from tessera.source_profiles import source_profiles
    tree_sha, members = source_tree_identity(ROOT)
    owners = [__file__, ROOT / "tools/tessera_shape_time_worker.py",
              ROOT / "experiments/bench_native_operator.py", ROOT / "experiments/step4_capture_launch.py",
              ROOT / "experiments/full_engine_plugin_install.py", ROOT / "tools/run_glm_cached_cpu_export.py",
              ROOT / "experiments/box_power_window.py", ROOT / "experiments/routed_pair_oracle.py"]
    profile = "tessera.shape_panel.tools.v1"
    value = source_profiles(((str(Path(p).relative_to(ROOT)), Path(p).read_bytes()) for p in owners),
                            legacy_profile=profile, legacy_prefix=profile.encode() + b"\0")
    return {"source_tree_sha256": tree_sha, "source_tree_members": members,
            "tool_source_sha256": value[profile]}


def seal_producer(path):
    """Host-only attestation, sealed as a request input before the admitted launch."""
    require_producer_origins()
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    tp._sha(commit, "host producer commit", 40)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
                           capture_output=True, text=True, check=True).stdout
    if dirty.strip():raise ValueError("host producer checkout is dirty; seal only an immutable source")
    value = {"schema": PRODUCER_SCHEMA, "commit": commit, "commit_source": "sealed_checkout",
             **producer_source_identity()}
    return publish_json(value, path)


def producer_identity(binding):
    """The source is independently host-attested; this process verifies every byte."""
    value = tp.json_bytes(tp.read_bound(binding))
    tp._object(value, {"schema", "commit", "commit_source", "source_tree_sha256", "source_tree_members", "tool_source_sha256"}, "sealed producer")
    tp._sha(value["commit"], "sealed producer commit", 40)
    if value["schema"] != PRODUCER_SCHEMA or value["commit_source"] != "sealed_checkout":
        raise ValueError("requires an independent sealed host checkout identity")
    observed = producer_source_identity()
    if any(value[k] != observed[k] for k in observed):
        raise ValueError("producer source differs from independent sealed checkout")
    return value


def read_request(path, *, expected_sha256=None):
    require_producer_origins()
    raw = Path(path).read_bytes()
    if expected_sha256 is not None:
        tp._sha(expected_sha256, "expected request sha256")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("owned request bytes differ from the sealed SHA-256")
    request = tp.json_bytes(raw)
    tp._object(request, {"schema", "expected_runtime", "scope", "prefix", "scheme", "wire",
                         "sampling", "netdata_hosts", "contract", "runtime_python", "worker_timeout_s", "record_verifier", "producer_identity"}, "dense request")
    if request["schema"] != REQUEST_SCHEMA:
        raise ValueError("unknown dense request schema")
    producer_identity(request["producer_identity"])
    runtime = tp.runtime_context(request["expected_runtime"])
    raw_contract = tp.read_bound(request["contract"])
    tp.read_bound(request["runtime_python"])
    tp.read_bound(request["record_verifier"])
    if Path(request["record_verifier"]["path"]) != Path("/mnt/shared/prismabuild-fleet/repo/tools/pbtest_pins.py").resolve(strict=True):
        raise ValueError("requires the published installation verifier")
    tp._integer(request["worker_timeout_s"], "native worker timeout")
    if hashlib.sha256(raw_contract).hexdigest() != runtime["contract_sha256"]:
        raise ValueError("request requires a different immutable runtime contract")
    scope = census_plan._request_scope(request["scope"])
    if (scope["route"], scope["structure"], scope["mode"], scope["execution_mode"], scope["tp_degree"]) != ("TESSERA_FP8", "dense", "resident", "eager", 1) or scope["requested_platform"] != runtime["platform"]:
        raise ValueError("first slice requires one E4M3 dense TP1 eager/resident scope")
    tp._object(request["sampling"], {"samples", "warmup_iterations", "steady_s", "seed"}, "sampling")
    tp._integer(request["sampling"]["samples"], "sample count", 3)
    tp._integer(request["sampling"]["warmup_iterations"], "warmups")
    tp._integer(request["sampling"]["seed"], "seed", 0)
    tp._number(request["sampling"]["steady_s"], "steady duration")
    if not isinstance(request["prefix"], str) or not request["prefix"]:
        raise ValueError("requires explicit native module prefix")
    tp._object(request["netdata_hosts"], {"sparky", "sparklina"}, "Netdata hosts")
    if any(not isinstance(v, str) or not v for v in request["netdata_hosts"].values()):
        raise ValueError("requires both explicit Netdata endpoints")
    wire = tp.read_bound(request["wire"])
    declared, roles = tp.wire_facts(wire, request["scheme"])
    if (declared["rows"], declared["columns"], declared["q256"]) != (scope["shape"]["N"], scope["shape"]["K"], scope["q256"]):
        raise ValueError("request wire geometry/rung differs")
    doc = tp.json_bytes(raw_contract)
    possible = []
    for launch in tp.scheme.route_launches(scope["route"], structure="dense", regime=scope["regime"], mode="resident"):
        try:
            tp.admitted_cell(doc, scope, runtime, (launch["symbol"], launch["decoder"]), roles)
        except ValueError:
            continue
        possible.append(launch)
    if not possible:
        raise ValueError("request has no positively backed native cell/wire in its frozen runtime")
    return request, scope, wire


def phase_command(request, job_source, output, *, preflight=False):
    worker=ROOT/"tools/tessera_shape_time_worker.py"
    command=["env","-u","PYTHONPATH","OMP_NUM_THREADS=1","MKL_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1"]
    command += [key+"="+value for key,value in request["expected_runtime"]["serve_flags"].items()]
    if preflight:command.append("CUDA_VISIBLE_DEVICES=")
    command += [request["runtime_python"]["path"],"-I","-B",str(worker),"--job",job_source["path"],
                "--job-sha256",job_source["sha256"],"--output",str(output)]
    if preflight:command.append("--preflight")
    return command


def run_runtime_preflight(request, job_source, output):
    # Only this actual execution path issues the in-memory external validation result.
    from experiments.step4_capture_launch import run_phase
    if Path(run_phase.__code__.co_filename).resolve()!=ROOT/"experiments/step4_capture_launch.py":
        raise ValueError("foreign native phase helper")
    command=phase_command(request,job_source,output,preflight=True)
    phase=run_phase("runtime-preflight",command,Path(output)/"runtime-preflight.log",request["worker_timeout_s"])
    publish_json(phase,Path(output)/"runtime-preflight-phase.json")
    if phase["returncode"]!=0:raise ValueError("installed runtime CPU preflight refused; inspect "+str(Path(output)/"runtime-preflight.log"))
    result=tp.json_bytes((Path(output)/"runtime-preflight.json").read_bytes())
    job=tp.json_bytes(tp.read_bound(job_source))
    validation=tp._verify_runtime_preflight(result,raw_contract=tp.read_bound(request["contract"]),
            expected_runtime=request["expected_runtime"],job_source=job_source,worker_source=job["worker_source"],
            request_source=job["request_source"],command=command,phase=phase)
    return validation


def owned_job(request_path, request, wire, output):
    producer=producer_identity(request["producer_identity"])
    _,roles=tp.wire_facts(wire,request["scheme"])
    worker=ROOT/"tools/tessera_shape_time_worker.py";worker_source=tp.file_binding(worker)
    request_source=tp.file_binding(request_path)
    if tp.json_bytes(tp.read_bound(request_source))!=request:raise ValueError("original request changed after entry")
    job={"schema":"tessera.native_shape_worker_job.v1","request":request,"request_source":request_source,
         "wire_roles":roles,"producer":producer,"worker_source":worker_source}
    job_source=publish_json(job,Path(output)/"worker-job.json")
    return producer,roles,worker_source,job_source


def measure(request_path, output, *, expected_request_sha256):
    request,scope,wire=read_request(request_path,expected_sha256=expected_request_sha256)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);fsync_path(output.parent)
    producer,roles,worker_source,job_source=owned_job(request_path,request,wire,output)
    if tp.json_bytes(tp.read_bound(job_source))["request_source"]["sha256"]!=expected_request_sha256:
        raise ValueError("owned original request changed")
    validation=run_runtime_preflight(request,job_source,output)
    plan=census_plan._build_validated_census_plan([scope],raw_contract=validation.raw_contract,
                                                contract=tp.json_bytes(validation.raw_contract))
    from experiments.step4_capture_launch import run_phase
    command=phase_command(request,job_source,output)
    phase=run_phase("native-dense",command,output/"native-phase.log",request["worker_timeout_s"])
    if phase["returncode"]!=0:raise ValueError("native phase refused; inspect "+str(output/"native-phase.log"))
    result=tp.json_bytes((output/"worker-result.json").read_bytes())
    tp._object(result,{"schema","evidence","worker_source","job_source","pair"},"native worker result")
    if result["schema"]!="tessera.native_shape_worker_result.v1" or result["worker_source"]!=worker_source or result["job_source"]!=job_source or tp.file_binding(output/"worker-job.json")!=job_source or tp.file_binding(ROOT/"tools/tessera_shape_time_worker.py")!=worker_source or producer_identity(request["producer_identity"])!=producer:
        raise ValueError("native worker or producer source identity differs")
    evidence=result["evidence"];runtime=tp.json_bytes(tp.read_bound(evidence["runtime"]))
    actual_contract=tp.json_bytes(tp.read_bound(evidence["contract"]))
    cell,_=tp.admitted_cell(actual_contract,request["scope"],runtime,result["pair"],roles)
    samples=tp.json_bytes(tp.read_bound(evidence["samples"]))["samples_ms"]
    panel={"schema":tp.SCHEMA,"status":"measured","claims":dict(tp.CLAIMS),"runtime":runtime,"plan":plan,
           "rows":[{"scope_id":plan["rows"][0]["id"],"prefix":request["prefix"],"scheme":request["scheme"],"timing":tp.timing_summary(samples),"cell_id":cell["id"]}],
           "evidence":evidence,"preflight":{"result":tp.file_binding(output/"runtime-preflight.json"),"phase":tp.file_binding(output/"runtime-preflight-phase.json")},"energy":{"status":"hold","reason":"cross_host_clock_alignment_unqualified","reference_w":140}}
    tp.validate_external_panel(panel,expected_runtime=request["expected_runtime"],runtime_validation=validation)
    require_producer_origins()
    bound=publish_json(panel,output/"panel.json")
    helper=os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
    if helper:runpy.run_path(helper)["commit"](1,"publish")
    return bound


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    check = sub.add_parser("check")
    check.add_argument("panel", type=Path);check.add_argument("--expected-runtime", type=Path, required=True)
    check.add_argument("--request",type=Path);check.add_argument("--request-sha256");check.add_argument("--preflight-output",type=Path)
    preflight = sub.add_parser("check-request")
    preflight.add_argument("request", type=Path)
    seal = sub.add_parser("seal-producer")
    seal.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("measure")
    run.add_argument("--request", type=Path, required=True);run.add_argument("--request-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.action == "seal-producer": result = seal_producer(args.output)
        elif args.action == "check":
            panel=tp.json_bytes(args.panel.read_bytes());expected=tp.json_bytes(args.expected_runtime.read_bytes())
            if "preflight" in panel:
                if args.request is None or args.request_sha256 is None or args.preflight_output is None:
                    raise ValueError("external replay requires owned request/SHA and a fresh CPU preflight output")
                request,_,_=read_request(args.request,expected_sha256=args.request_sha256)
                if request["expected_runtime"]!=expected:raise ValueError("check context differs from owned request")
                prior=tp.json_bytes(tp.read_bound(panel["preflight"]["result"]))
                job_source=prior["job_source"]
                job=tp.json_bytes(tp.read_bound(job_source))
                if job["request"]!=request or job["request_source"]!=tp.file_binding(args.request):raise ValueError("panel job differs from owned request")
                output=args.preflight_output.resolve();output.mkdir(parents=True,exist_ok=False);fsync_path(output.parent)
                validation=run_runtime_preflight(request,job_source,output)
                result=tp.validate_external_panel(panel,expected_runtime=expected,runtime_validation=validation)
            else:result=tp.validate_panel(panel,expected_runtime=expected)
        elif args.action == "check-request":
            _,scope,_=read_request(args.request);result={"scope":scope,"status":"unmeasured","gpu_executed":False,"contract_validation":"pending_installed_preflight"}
        else: result = measure(args.request, args.output, expected_request_sha256=args.request_sha256)
        print(json.dumps(result, sort_keys=True));return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[native-dense-panel] REFUSED: {exc}", file=sys.stderr);return 2


if __name__ == "__main__": raise SystemExit(main())
