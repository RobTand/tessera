#!/usr/bin/env python3
"""Measure one explicitly frozen native dense operator (#688).

Repository application only: no scheduler, encoding, serving or cell publication.
Agents submit the command through PB; the library validator does not know PB.
The first slice is E4M3 dense, TP1, eager/resident, with one owned output.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Tool utilities; never substitute the runtime's src.

from tessera.serving import census_plan, timing_panel as tp
from tools.run_glm_cached_cpu_export import fsync_path

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


def observed_commit(package):
    """The actual imported package's repository or installed VCS metadata."""
    location = Path(package.__file__).resolve().parent
    result = subprocess.run(["git", "-C", str(location), "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        root = Path(result.stdout.strip()).resolve()
        if location.is_relative_to(root):
            # Dirt anywhere in that immutable source checkout invalidates its label.
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
                                   capture_output=True, text=True, check=True).stdout
            if dirty.strip():
                raise ValueError("imported Tessera runtime checkout is dirty")
            tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(Path(package.__file__).resolve().relative_to(root))], capture_output=True, text=True)
            if tracked.returncode != 0:
                raise ValueError("imported Tessera runtime is not tracked by its observed repository")
            return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    try:
        distribution = importlib.metadata.distribution("tessera-quant")
        if not distribution.files or not any(Path(distribution.locate_file(p)).resolve() == Path(package.__file__).resolve() for p in distribution.files):
            raise ValueError("installed VCS metadata does not own the imported Tessera runtime")
        metadata = distribution.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        metadata = None
    if metadata:
        record = tp.json_bytes(metadata)
        commit = record.get("vcs_info", {}).get("commit_id")
        if commit:
            return tp._sha(commit, "installed runtime VCS commit", 40)
    raise ValueError("imported Tessera runtime has no observed immutable VCS identity")


def observe_runtime(expected):
    import torch
    import vllm
    import tessera
    from tessera.serving import backend, contract, source_identity, runtime_image

    declaration = runtime_image.declared_reference(expected["image"])
    if declaration["image"] != expected["image"]:
        raise ValueError("launcher image declaration differs")
    value = {"image": declaration["image"], "tessera_commit": observed_commit(tessera),
             "serving_source_sha256": source_identity.serving_source_sha256(),
             "contract_sha256": hashlib.sha256(contract.contract_path().read_bytes()).hexdigest(),
             "platform": backend.platform_of_this_process(torch), "torch": torch.__version__,
             "vllm": vllm.__version__, "execution_mode": "eager", "residency": "resident",
             "tp_rank": 0, "tp_degree": 1,
             "serve_flags": {key: os.environ[key] for key in expected["serve_flags"] if key in os.environ}}
    if tp.runtime_context(value) != tp.runtime_context(expected):
        raise ValueError("actually imported runtime differs from frozen expected context")
    return value


def producer_identity():
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    owners = [__file__, ROOT / "tools/a4_measure.py", ROOT / "tools/run_glm_cached_cpu_export.py",
              ROOT / "experiments/box_power_window.py", ROOT / "experiments/routed_pair_oracle.py"]
    from tessera.source_profiles import source_profiles
    profile = "tessera.shape_panel.tools.v1"
    value = source_profiles(((str(Path(p).relative_to(ROOT)), Path(p).read_bytes()) for p in owners),
                            legacy_profile=profile, legacy_prefix=profile.encode() + b"\0")
    return {"commit": commit, "tool_source_sha256": value[profile]}


def prepare_dense(request, wire):
    """The public plugin's create/load/finalize path, with explicit TP1 coordinates."""
    import torch
    from tessera.serving.lane import build_tessera_method

    scope = request["scope"]
    declaration, roles = tp.wire_facts(wire, request["scheme"])
    layer = torch.nn.Module()
    layer.tp_rank, layer.tp_size = 0, scope["tp_degree"]
    if layer.tp_size != 1:
        raise ValueError("native dense application supports explicit TP1 only")
    method = build_tessera_method(request["scheme"], request["prefix"], "resident")
    method.create_weights(layer, input_size_per_partition=declaration["columns"],
                          output_partition_sizes=[r for _, r in declaration["roles"]],
                          input_size=declaration["columns"], output_size=declaration["rows"],
                          params_dtype=torch.bfloat16, weight_loader=None)
    # Copy the exact hash-checked owned byte buffer into the plugin's parameter.
    source = torch.frombuffer(bytearray(wire), dtype=torch.uint8)
    layer.wire_bytes.data.copy_(source)
    method.process_weights_after_loading(layer)
    native = layer.tessera_native
    shape = {"M": scope["shape"]["M"], "N": native.rows, "K": native.columns}
    if shape != scope["shape"] or (layer.tessera_shard_plan.tp_rank, layer.tessera_shard_plan.tp_size) != (0, 1):
        raise ValueError("actual native preparation differs from explicit shape/partition")
    prep = {"builder": "tessera.serving.lane.build_tessera_method",
            "wire_sha256": hashlib.sha256(wire).hexdigest(), "roles": roles, "shape": shape,
            "tp_rank": layer.tp_rank, "tp_degree": layer.tp_size, "grid": declaration["grid"],
            "native_packed_bytes": native.packed_bytes()}
    return layer, method, prep


def fresh_call(layer, method, x):
    """A latest-route record is evidence only if this call actually emitted it."""
    from tessera.serving import telemetry
    setattr(layer, telemetry.ATTR_PREFIX + "state", None)
    output = method.apply(layer, x)
    record = telemetry.read_route(layer)
    if record is None or record.get("state") != "served":
        raise ValueError("native call emitted no fresh served route")
    return output, record


def raw_netdata(host, t0, t1):
    from experiments import box_power_window as instrument
    out = {}
    for context, dims in instrument.SERIES:
        if context not in tp.NETDATA_CONTEXTS:
            continue
        got = instrument._fetch(host, context, dims, int(t0), int(t1) + 1, 0)
        raw = got.get("raw_doc", got["doc"])
        out[context] = {"query": got["url"], "raw_response": raw, "returned_view": raw.get("view")}
    return out


def profile_call(call, path):
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            call()
        torch.cuda.synchronize()
    temporary = path.with_suffix(".json")
    prof.export_chrome_trace(str(temporary))
    with path.open("xb") as handle:
        handle.write(gzip.compress(temporary.read_bytes()))
        handle.flush();os.fsync(handle.fileno())
    fsync_path(path.parent)
    temporary.unlink()
    return tp.file_binding(path)


def read_request(path):
    request = tp.json_bytes(Path(path).read_bytes())
    tp._object(request, {"schema", "expected_runtime", "scope", "prefix", "scheme", "wire",
                         "sampling", "netdata_hosts"}, "dense request")
    if request["schema"] != REQUEST_SCHEMA:
        raise ValueError("unknown dense request schema")
    runtime = tp.runtime_context(request["expected_runtime"])
    from tessera.serving import contract
    raw_contract = contract.contract_path().read_bytes()
    if hashlib.sha256(raw_contract).hexdigest() != runtime["contract_sha256"]:
        raise ValueError("request requires a different immutable runtime contract")
    plan = census_plan.build_census_plan([request["scope"]], raw_contract=raw_contract)
    scope = plan["rows"][0]["scope"]
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
    return request, plan, wire


def measure(request_path, output):
    request, plan, wire = read_request(request_path)
    import torch
    from tessera.serving import contract
    from tools.a4_measure import time_call
    from experiments.routed_pair_oracle import PowerSampler

    runtime = observe_runtime(request["expected_runtime"])
    producer = producer_identity()
    if not torch.cuda.is_available():
        raise ValueError("native measurement requires an actual CUDA device")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fsync_path(output.parent)
    layer, method, prep = prepare_dense(request, wire)
    gen = torch.Generator(device="cuda").manual_seed(request["sampling"]["seed"])
    x = torch.randn(request["scope"]["shape"]["M"], layer.tessera_columns, device="cuda", generator=gen).bfloat16()
    records = []
    fingerprints = layer.tessera_native.fingerprints()
    def call(verify=False):
        y, record = fresh_call(layer, method, x)
        if tuple(y.shape) != (request["scope"]["shape"]["M"], layer.tessera_rows):
            raise ValueError("actual output geometry differs")
        if verify and not bool(torch.isfinite(y).all()):
            raise ValueError("native output contains nonfinite values")
        return record
    call(verify=True);torch.cuda.synchronize()
    actual_contract = contract.contract_path().read_bytes()
    pair = list(layer.tessera_native.launch_pair)
    cell, lane = tp.admitted_cell(tp.json_bytes(actual_contract), request["scope"], runtime, pair, prep["roles"])
    trace = profile_call(call, output / "profile.trace.json.gz")
    sampler = PowerSampler();sampler.start()
    t0 = time.time()
    samples = []
    try:
        for i in range(request["sampling"]["samples"]):
            sample_record = []
            def timed(): sample_record.append(call())
            ms = time_call(timed, 1, warmup=request["sampling"]["warmup_iterations"] if i == 0 else 0)
            records.append(sample_record[-1]);samples.append(ms)
        until = time.perf_counter() + request["sampling"]["steady_s"]
        while time.perf_counter() < until:
            call();torch.cuda.synchronize()
        t1 = time.time()
    finally:
        sampler.stop_flag = True
        sampler.join(timeout=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {box: pool.submit(raw_netdata, host, t0, t1) for box, host in request["netdata_hosts"].items()}
        telemetry = {"interval_unix": [t0, t1], "fast_power_samples": [[t, w] for t, w in sampler.samples if t0 <= t <= t1],
                     "netdata": {box: future.result() for box, future in pending.items()}}
    # Reuse the library already loaded during native preparation; no new build path.
    from tessera import routed_fused
    module = routed_fused._ext(routed_fused.library_for("e4m3"))
    evidence = {"trace": trace, "native_binary": tp.file_binding(module.__file__)}
    documents = {"runtime": runtime, "producer": producer, "preparation": prep,
                 "samples": {"samples_ms": samples, "warmup_iterations": request["sampling"]["warmup_iterations"], "interval_unix": [t0, t1]},
                 "routes": {"records": records}, "telemetry": telemetry}
    for name, value in documents.items(): evidence[name] = publish_json(value, output / (name + ".json"))
    for name, data in (("wire", wire), ("contract", actual_contract)):
        path = output / (name + ".bin")
        with path.open("xb") as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        fsync_path(path.parent);evidence[name] = tp.file_binding(path)
    panel = {"schema": tp.SCHEMA, "status": "measured", "claims": dict(tp.CLAIMS), "runtime": runtime, "plan": plan,
             "rows": [{"scope_id": plan["rows"][0]["id"], "prefix": request["prefix"], "scheme": request["scheme"],
                       "timing": tp.timing_summary(samples), "cell_id": cell["id"]}], "evidence": evidence,
             "energy": {"status": "hold", "reason": "cross_host_clock_alignment_unqualified", "reference_w": 140}}
    tp.validate_panel(panel, expected_runtime=request["expected_runtime"])
    if layer.tessera_native.fingerprints() != fingerprints:
        raise ValueError("prepared native weights changed during measurement")
    if observe_runtime(request["expected_runtime"]) != runtime or producer_identity() != producer:
        raise ValueError("runtime or producer source changed during measurement")
    bound = publish_json(panel, output / "panel.json")
    helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
    if helper: runpy.run_path(helper)["commit"](1, "publish")
    return bound


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    check = sub.add_parser("check")
    check.add_argument("panel", type=Path);check.add_argument("--expected-runtime", type=Path, required=True)
    preflight = sub.add_parser("check-request")
    preflight.add_argument("request", type=Path)
    run = sub.add_parser("measure")
    run.add_argument("--request", type=Path, required=True);run.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.action == "check":
            result = tp.validate_panel(tp.json_bytes(args.panel.read_bytes()), expected_runtime=tp.json_bytes(args.expected_runtime.read_bytes()))
        elif args.action == "check-request":
            _, plan, _ = read_request(args.request);result = {"scope_id": plan["rows"][0]["id"], "status": "unmeasured", "gpu_executed": False}
        else: result = measure(args.request, args.output)
        print(json.dumps(result, sort_keys=True));return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[native-dense-panel] REFUSED: {exc}", file=sys.stderr);return 2


if __name__ == "__main__": raise SystemExit(main())
