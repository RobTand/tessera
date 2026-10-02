#!/usr/bin/env python3
"""One native phase: installed runtime only; repository schema lives in parent."""
from __future__ import annotations
import argparse,gzip,hashlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
# Repository tools are allowed; repository src never enters runtime search paths.
sys.path.insert(0,str(ROOT))
from tools.run_glm_cached_cpu_export import fsync_path,read_bound
SCHEMA="tessera.native_shape_worker_result.v1"
MODULES=("tessera", "tessera.serving.backend", "tessera.serving.contract", "tessera.serving.source_identity", "tessera.serving.runtime_image", "tessera.serving.scheme", "tessera.serving.lane", "tessera.serving.telemetry")

def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def file_binding(path):
 path=Path(path).resolve(strict=True);raw=path.read_bytes()
 return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
def json_bytes(raw):
 def pairs(items):
  result={}
  for key,value in items:
   if key in result:raise ValueError("duplicate worker JSON key")
   result[key]=value
  return result
 def invalid(v):raise ValueError("nonfinite worker JSON")
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
def hex_identity(value,length):
 if not isinstance(value,str) or len(value)!=length or any(c not in "0123456789abcdef" for c in value):raise ValueError("invalid immutable source identity")
 return value
def publish_json(value, path):
    """A durable name, before any committed progress is reported."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_path(path.parent)
    return file_binding(path)


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
        record = json_bytes(metadata)
        commit = record.get("vcs_info", {}).get("commit_id")
        if commit:
            return hex_identity(commit,40)
    raise ValueError("imported Tessera runtime has no observed immutable VCS identity")



def runtime_origins(expected_root):
 import importlib
 root=Path(expected_root)
 if not root.is_absolute() or root.resolve()!=root or root.is_relative_to(ROOT):raise ValueError("runtime root must be the installed immutable package outside producer checkout")
 if os.environ.get("PYTHONPATH"):raise ValueError("native phase refuses PYTHONPATH")
 if any(Path(v or os.getcwd()).resolve().is_relative_to(ROOT/"src") for v in sys.path):raise ValueError("native phase refuses producer src in import paths")
 files={}
 for name in MODULES:
  module=importlib.import_module(name);path=Path(module.__file__).resolve()
  expected=root/("__init__.py" if name=="tessera" else name.removeprefix("tessera.").replace(".","/")+".py")
  if path!=expected:raise ValueError("foreign runtime module origin: "+name)
  files[name]=file_binding(path)
 for name,module in list(sys.modules.items()):
  if name=="tessera" or name.startswith("tessera."):
   path=getattr(module,"__file__",None)
   if path is None or not Path(path).resolve().is_relative_to(root):raise ValueError("foreign loaded runtime owner: "+name)
 return {"package_root":str(root),"modules":files}


def observe_runtime(expected):
 origins=runtime_origins(expected["package_root"])
 import torch,vllm,tessera
 from tessera.serving import backend,contract,source_identity,runtime_image
 declaration=runtime_image.declared_reference(expected["image"])
 actual_commit=observed_commit(tessera)
 actual_source=source_identity.serving_source_sha256()
 actual_contract=hashlib.sha256(contract.contract_path().read_bytes()).hexdigest()
 if (declaration["image"],actual_commit,actual_source,actual_contract,torch.__version__,vllm.__version__)!=(expected["image"],expected["tessera_commit"],expected["serving_source_sha256"],expected["contract_sha256"],expected["torch"],expected["vllm"]):raise ValueError("runtime source/contract/version differs before device setup")
 value={"image":declaration["image"],"tessera_commit":actual_commit,
  "serving_source_sha256":actual_source,
  "contract_sha256":actual_contract,
  "platform":backend.platform_of_this_process(torch),"torch":torch.__version__,"vllm":vllm.__version__,
  "execution_mode":"eager","residency":"resident","tp_rank":0,"tp_degree":1,
  "package_root":origins["package_root"],
  "serve_flags":{k:os.environ[k] for k in expected["serve_flags"] if k in os.environ}}
 if canonical(value)!=canonical(expected):raise ValueError("actually imported runtime differs from frozen expected context")
 return value,origins
def prepare_dense(request, wire):
    """The public plugin's create/load/finalize path, with explicit TP1 coordinates."""
    import torch
    from tessera.serving.lane import build_tessera_method

    scope = request["scope"]
    from tessera.serving import scheme
    declaration = scheme.validate_tessera_scheme(request["scheme"], request["prefix"])
    roles = request.pop("_wire_roles")
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
        if context not in {"nvidia_smi.gpu_power_draw","system.cpu","system.load","mem.swapio","mem.available"}:
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
    return file_binding(path)


def _native_measure(job, output):
    if set(job) != {"schema", "request", "wire_roles", "producer", "worker_source"} or job["schema"] != "tessera.native_shape_worker_job.v1":
        raise ValueError("unknown native phase job")
    request=job["request"];request["_wire_roles"]=job["wire_roles"]
    wire=read_bound(request["wire"])
    if len(wire)!=request["wire"]["bytes"]:raise ValueError("wire bytes differ")
    import torch
    from tessera.serving import contract
    from experiments.bench_native_operator import time_apply
    from experiments.routed_pair_oracle import PowerSampler

    runtime, origins = observe_runtime(request["expected_runtime"])
    producer=job["producer"]
    worker_source=file_binding(__file__)
    if not torch.cuda.is_available():
        raise ValueError("native measurement requires an actual CUDA device")
    output = Path(output).resolve()
    if not output.is_dir():raise ValueError("parent must own the fresh output directory")
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
    if hashlib.sha256(actual_contract).hexdigest()!=request["expected_runtime"]["contract_sha256"]:raise ValueError("runtime contract changed before native call")
    trace = profile_call(call, output / "profile.trace.json.gz")
    sampler = PowerSampler();sampler.start()
    t0 = time.time()
    samples = []
    try:
        def timed():records.append(call())
        measured=time_apply(timed,warmup_iterations=request["sampling"]["warmup_iterations"],iterations=request["sampling"]["samples"])
        samples=measured["samples_ms"];records=records[-len(samples):]
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
    evidence = {"trace": trace, "native_binary": file_binding(module.__file__)}
    documents = {"runtime": runtime, "runtime_origins": origins, "producer": producer, "preparation": prep,
                 "samples": {"samples_ms": samples, "warmup_iterations": request["sampling"]["warmup_iterations"], "interval_unix": [t0, t1]},
                 "routes": {"records": records}, "telemetry": telemetry}
    for name, value in documents.items(): evidence[name] = publish_json(value, output / (name + ".json"))
    for name, data in (("wire", wire), ("contract", actual_contract)):
        path = output / (name + ".bin")
        with path.open("xb") as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        fsync_path(path.parent);evidence[name] = file_binding(path)
    if layer.tessera_native.fingerprints() != fingerprints:
        raise ValueError("prepared native weights changed during measurement")
    after,after_origins=observe_runtime(request["expected_runtime"])
    if after!=runtime or after_origins!=origins or file_binding(__file__)!=worker_source:
        raise ValueError("runtime, contract, module origin or worker source changed during measurement")
    result={"schema":SCHEMA,"evidence":evidence,"worker_source":worker_source,"pair":pair}
    return publish_json(result,output/"worker-result.json")

def native_measure(job_path, output):
 # Observe identity before entering any distributed/CUDA setup.
 raw=Path(job_path).read_bytes();job=json_bytes(raw)
 if set(job)!={"schema","request","wire_roles","producer","worker_source"} or job["schema"]!="tessera.native_shape_worker_job.v1":raise ValueError("unknown native phase job")
 if file_binding(__file__)!=job["worker_source"]:raise ValueError("native worker source differs")
 for key in ("wire","contract","runtime_python"):
  bound=job["request"][key];data=read_bound(bound)
  if len(data)!=bound["bytes"]:raise ValueError("native input bytes differ")
 observe_runtime(job["request"]["expected_runtime"])
 if Path(job_path).read_bytes()!=raw:raise ValueError("native job changed during entry checks")
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from experiments.bench_native_operator import native_runtime_context
 with native_runtime_context():return _native_measure(job,output)

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument("--job",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args(argv)
 try:print(json.dumps(native_measure(args.job,args.output),sort_keys=True));return 0
 except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:print("[native-phase] REFUSED: "+str(exc),file=sys.stderr);return 2
if __name__=="__main__":raise SystemExit(main())
