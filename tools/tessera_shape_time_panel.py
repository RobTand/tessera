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


def producer_source_identity(root=ROOT):
    """Existing source-tree and tool-closure owners; no in-container Git required.

    ``root`` names the producer checkout whose bytes are hashed. The supported
    replay path may point it at an original producer source tree independently
    of the running tool, so a changed replay tool cannot claim to be that
    producer by re-deriving the identity from its own tree.
    """
    from experiments.full_engine_plugin_install import source_tree_identity
    from tessera.source_profiles import source_profiles
    tree_sha, members = source_tree_identity(root)
    owners = [root / "tools/tessera_shape_time_panel.py",
              root / "tools/tessera_shape_time_worker.py",
              root / "experiments/bench_native_operator.py", root / "experiments/step4_capture_launch.py",
              root / "experiments/full_engine_plugin_install.py", root / "tools/run_glm_cached_cpu_export.py",
              root / "experiments/box_power_window.py", root / "experiments/routed_pair_oracle.py"]
    profile = "tessera.shape_panel.tools.v1"
    value = source_profiles(((str(Path(p).relative_to(root)), Path(p).read_bytes()) for p in owners),
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


def producer_identity(binding, root=ROOT):
    """The source is independently host-attested; this process verifies every byte.

    ``root`` is the producer checkout the sealed identity must equal; the
    replay caller supplies the original producer tree it executes against.
    """
    value = tp.json_bytes(tp.read_bound(binding))
    tp._object(value, {"schema", "commit", "commit_source", "source_tree_sha256", "source_tree_members", "tool_source_sha256"}, "sealed producer")
    tp._sha(value["commit"], "sealed producer commit", 40)
    if value["schema"] != PRODUCER_SCHEMA or value["commit_source"] != "sealed_checkout":
        raise ValueError("requires an independent sealed host checkout identity")
    observed = producer_source_identity(root)
    if any(value[k] != observed[k] for k in observed):
        raise ValueError("producer source differs from independent sealed checkout")
    return value


def read_request(path, *, expected_sha256=None, producer_root=ROOT):
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
    producer_identity(request["producer_identity"], producer_root)
    runtime = tp.runtime_context(request["expected_runtime"])
    raw_contract = tp.read_bound(request["contract"])
    tp.read_bound(request["runtime_python"])
    tp.read_bound(request["record_verifier"])
    from tools.tessera_shape_time_worker import record_verifier_bytes
    record_verifier_bytes(request["record_verifier"])
    tp._integer(request["worker_timeout_s"], "native worker timeout")
    if hashlib.sha256(raw_contract).hexdigest() != runtime["contract_sha256"]:
        raise ValueError("request requires a different immutable runtime contract")
    scope = census_plan._request_scope(request["scope"])
    tp.admitted_panel_scope(scope, runtime)
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
    if scope["structure"] == "routed_moe":
        if (declared["hidden_size"], declared["intermediate_size"], declared["experts"]) != (scope["shape"]["K"], scope["shape"]["N"], scope["shape"]["experts"]):
            raise ValueError("request routed stack geometry differs")
    elif (declared["rows"], declared["columns"], declared["q256"]) != (scope["shape"]["N"], scope["shape"]["K"], scope["q256"]):
        raise ValueError("request wire geometry/rung differs")
    doc = tp.json_bytes(raw_contract)
    possible = []
    for launch in tp.candidate_launches(scope["route"], scope["structure"], scope["regime"]):
        try:
            tp.admitted_cell(doc, scope, runtime, (launch["symbol"], launch["decoder"]), roles)
        except ValueError:
            continue
        possible.append(launch)
    if not possible:
        raise ValueError("request has no positively backed native cell/wire in its frozen runtime")
    return request, scope, wire


def phase_command(request, job_source, output, *, preflight=False, producer_root=ROOT):
    worker=producer_root/"tools/tessera_shape_time_worker.py"
    command=["env","-u","PYTHONPATH","OMP_NUM_THREADS=1","MKL_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1"]
    command += [key+"="+value for key,value in request["expected_runtime"]["serve_flags"].items()]
    if preflight:command.append("CUDA_VISIBLE_DEVICES=")
    command += [request["runtime_python"]["path"],"-I","-B",str(worker),"--job",job_source["path"],
                "--job-sha256",job_source["sha256"],"--output",str(output)]
    if preflight:command.append("--preflight")
    return command


def run_runtime_preflight(request, job_source, output, *, producer_root=ROOT):
    # Only this actual execution path issues the in-memory external validation result.
    from experiments.step4_capture_launch import run_phase
    if Path(run_phase.__code__.co_filename).resolve()!=ROOT/"experiments/step4_capture_launch.py":
        raise ValueError("foreign native phase helper")
    command=phase_command(request,job_source,output,preflight=True,producer_root=producer_root)
    phase=run_phase("runtime-preflight",command,Path(output)/"runtime-preflight.log",request["worker_timeout_s"])
    publish_json(phase,Path(output)/"runtime-preflight-phase.json")
    if phase["returncode"]!=0:raise ValueError("installed runtime CPU preflight refused; inspect "+str(Path(output)/"runtime-preflight.log"))
    result=tp.json_bytes((Path(output)/"runtime-preflight.json").read_bytes())
    job=tp.json_bytes(tp.read_bound(job_source))
    validation=tp._verify_runtime_preflight(result,raw_contract=tp.read_bound(request["contract"]),
            expected_runtime=request["expected_runtime"],job_source=job_source,worker_source=job["worker_source"],
            request_source=job["request_source"],command=command,phase=phase)
    return validation


def owned_job(request_path, request, wire, output, *, producer_root=ROOT, worker_root=None):
    producer=producer_identity(request["producer_identity"], producer_root)
    _,roles=tp.wire_facts(wire,request["scheme"])
    worker=(producer_root if worker_root is None else worker_root)/"tools/tessera_shape_time_worker.py";worker_source=tp.file_binding(worker)
    request_source=tp.file_binding(request_path)
    if tp.json_bytes(tp.read_bound(request_source))!=request:raise ValueError("original request changed after entry")
    job={"schema":"tessera.native_shape_worker_job.v1","request":request,"request_source":request_source,
         "wire_roles":roles,"producer":producer,"worker_source":worker_source}
    job_source=publish_json(job,Path(output)/"worker-job.json")
    return producer,roles,worker_source,job_source


def prepare_request_phase(request_path, output, expected_request_sha256, *, producer_root=ROOT):
    request,scope,wire=read_request(request_path,expected_sha256=expected_request_sha256,producer_root=producer_root)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);fsync_path(output.parent)
    producer,roles,worker_source,job_source=owned_job(request_path,request,wire,output,producer_root=producer_root)
    if tp.json_bytes(tp.read_bound(job_source))["request_source"]["sha256"]!=expected_request_sha256:
        raise ValueError("owned original request changed")
    return request,scope,wire,output,producer,roles,worker_source,job_source


def preflight_request(request_path, output, *, expected_request_sha256):
    request,scope,_,output,_,_,_,job_source=prepare_request_phase(request_path,output,expected_request_sha256)
    validation=run_runtime_preflight(request,job_source,output)
    plan=census_plan._build_validated_census_plan([scope],raw_contract=validation.raw_contract,
                                                contract=tp.json_bytes(validation.raw_contract))
    return publish_json({"status":"unmeasured","gpu_executed":False,"plan":plan,
                         "preflight":{"result":tp.file_binding(output/"runtime-preflight.json"),
                                      "phase":tp.file_binding(output/"runtime-preflight-phase.json")}},output/"preflight-plan.json")


def measure(request_path, output, *, expected_request_sha256):
    request,scope,wire,output,producer,roles,worker_source,job_source=prepare_request_phase(request_path,output,expected_request_sha256)
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
    check.add_argument("--producer-root",type=Path)
    check.add_argument("--expected-panel-sha256")
    check.add_argument("--observation-out",type=Path)
    preflight = sub.add_parser("check-request")
    preflight.add_argument("request", type=Path)
    seal = sub.add_parser("seal-producer")
    seal.add_argument("--output", type=Path, required=True)
    installed=sub.add_parser("preflight-request")
    installed.add_argument("--request",type=Path,required=True);installed.add_argument("--request-sha256",required=True)
    installed.add_argument("--output",type=Path,required=True)
    run = sub.add_parser("measure")
    run.add_argument("--request", type=Path, required=True);run.add_argument("--request-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.action == "seal-producer": result = seal_producer(args.output)
        elif args.action == "preflight-request":result=preflight_request(args.request,args.output,expected_request_sha256=args.request_sha256)
        elif args.action == "check":
            raw=args.panel.read_bytes();panel=tp.json_bytes(raw);expected=tp.json_bytes(args.expected_runtime.read_bytes())
            if args.expected_panel_sha256 is not None:
                tp._sha(args.expected_panel_sha256,"expected panel sha256")
                if hashlib.sha256(raw).hexdigest()!=args.expected_panel_sha256:
                    raise ValueError("panel bytes differ from the externally supplied digest")
            observation_out=args.observation_out is not None
            if "preflight" in panel:
                if args.request is None or args.request_sha256 is None or args.preflight_output is None:
                    raise ValueError("external replay requires owned request/SHA and a fresh CPU preflight output")
                if observation_out and args.expected_panel_sha256 is None:
                    raise ValueError("bound observation requires --expected-panel-sha256")
                producer_root=ROOT if args.producer_root is None else args.producer_root.resolve()
                request,_,_=read_request(args.request,expected_sha256=args.request_sha256,producer_root=producer_root)
                if request["expected_runtime"]!=expected:raise ValueError("check context differs from owned request")
                prior=tp.json_bytes(tp.read_bound(panel["preflight"]["result"]))
                job_source=prior["job_source"]
                job=tp.json_bytes(tp.read_bound(job_source))
                if job["request"]!=request or job["request_source"]!=tp.file_binding(args.request):raise ValueError("panel job differs from owned request")
                if job["worker_source"]!=tp.file_binding(producer_root/"tools/tessera_shape_time_worker.py"):
                    raise ValueError("original panel worker differs from its sealed producer")
                output=args.preflight_output.resolve();output.mkdir(parents=True,exist_ok=False);fsync_path(output.parent)
                _,_,_,replay_job=owned_job(args.request,request,tp.read_bound(request["wire"]),output,
                                          producer_root=producer_root,worker_root=ROOT)
                validation=run_runtime_preflight(request,replay_job,output,producer_root=ROOT)
                result=tp.validate_external_panel(panel,expected_runtime=expected,runtime_validation=validation)
                if observation_out:
                    binding={"path":str(args.panel.resolve()),"bytes":len(raw),
                             "sha256":hashlib.sha256(raw).hexdigest()}
                    replay={**producer_source_identity(ROOT),"tool":tp.file_binding(__file__)}
                    result=tp.observation(panel,panel_binding=binding,
                            expected_panel_sha256=args.expected_panel_sha256,
                            request_binding=tp.file_binding(args.request),request=request,
                            expected_runtime_binding=tp.file_binding(args.expected_runtime),
                            expected_runtime=expected,runtime_validation=validation,replay=replay)
                    publish_json(result,args.observation_out)
            else:
                if observation_out or args.expected_panel_sha256 is not None:
                    raise ValueError("bound observation requires an external panel with an installed CPU preflight")
                result=tp.validate_panel(panel,expected_runtime=expected)
        elif args.action == "check-request":
            _,scope,_=read_request(args.request);result={"scope":scope,"status":"unmeasured","gpu_executed":False,"contract_validation":"pending_installed_preflight"}
        else: result = measure(args.request, args.output, expected_request_sha256=args.request_sha256)
        print(json.dumps(result, sort_keys=True));return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[native-dense-panel] REFUSED: {exc}", file=sys.stderr);return 2


if __name__ == "__main__": raise SystemExit(main())
