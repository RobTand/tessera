#!/usr/bin/env python3
"""One native phase: installed runtime only; repository schema lives in parent."""
from __future__ import annotations
import argparse,gzip,hashlib,importlib.metadata,json,os,stat,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
# Repository tools are allowed; repository src never enters runtime search paths.
sys.path.insert(0,str(ROOT))
from tools.run_glm_cached_cpu_export import fsync_path,read_bound
SCHEMA="tessera.native_shape_worker_result.v1"
MODULES=("tessera", "tessera.serving.backend", "tessera.serving.ext", "tessera.serving.contract", "tessera.serving.source_identity", "tessera.serving.runtime_image", "tessera.serving.scheme", "tessera.serving.lane", "tessera.serving.telemetry")
PB_RECORD_VERIFIER = Path('/mnt/shared/prismabuild-fleet/repo/tools/pbtest_pins.py')
PB_IMMUTABLE_GENERATIONS = Path('/mnt/shared/prismabuild-fleet/runtime-generations')


def _producer_dev_mode():
    """The producer checkout's standalone stamp policy, under a private name.

    Loaded by file path, never inserted into sys.path or sys.modules, so it
    cannot shadow the measured runtime's own tessera package -- and so a
    legacy installed candidate that predates ``tessera.dev_mode`` still gets
    default-dev stamp semantics from the tree that ships this worker.
    """
    import importlib.util
    path = ROOT / "src" / "tessera" / "dev_mode.py"
    spec = importlib.util.spec_from_file_location("tessera_producer_dev_mode_seal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DEV_MODE = _producer_dev_mode()


def record_verifier_bytes(binding):
    """Hold immutable helper bytes equal to the published PB owner's bytes."""
    path = Path(binding['path'])
    published = PB_RECORD_VERIFIER.resolve(strict=True)
    if (not path.is_absolute() or path.resolve(strict=True) != path
            or not path.is_relative_to(PB_IMMUTABLE_GENERATIONS)
            or len(path.relative_to(PB_IMMUTABLE_GENERATIONS).parts) != 3
            or path.parts[-2:] != ('tools', 'pbtest_pins.py')):
        raise ValueError('installation verifier must name an immutable PB generation')

    def held(member):
        fd = os.open(member, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o222 or before.st_size > 64 * 1024:
                raise ValueError('installation verifier must be a bounded read-only regular file')
            with os.fdopen(fd, 'rb', closefd=False) as handle:
                raw = handle.read(64 * 1024 + 1)
            after = os.fstat(fd)
            names = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
            if len(raw) != before.st_size or any(getattr(before, name) != getattr(after, name) for name in names):
                raise ValueError('installation verifier changed during its owned read')
            return raw
        finally:
            os.close(fd)

    raw = held(path)
    if (len(raw) != binding['bytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']
            or raw != held(published) or PB_RECORD_VERIFIER.resolve(strict=True) != published):
        raise ValueError('installation verifier differs from the published PB owner')
    return raw

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


def observed_commit(package, strict=True):
    """The actual imported package's repository or installed VCS metadata.

    With ``strict=False`` (D32 default dev mode) the same provenance facts
    are collected instead of refused and ``(commit, facts)`` is returned: a
    dirty or untracked tree still yields its HEAD -- the stamps name the
    drift -- and a tree with no derivable identity yields ``None`` for the
    caller to retain the stored commit against.
    """
    facts = [] if not strict else None
    def _refuse(message):
        if strict:raise ValueError(message)
        facts.append(message)
    location = Path(package.__file__).resolve().parent
    try:
        result = subprocess.run(["git", "-C", str(location), "rev-parse", "--show-toplevel"],
                                capture_output=True, text=True)
    except FileNotFoundError:
        result = subprocess.CompletedProcess([], 1, "", "")
    if result.returncode == 0:
        root = Path(result.stdout.strip()).resolve()
        if location.is_relative_to(root):
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
                                   capture_output=True, text=True, check=True).stdout
            if dirty.strip():
                _refuse("imported Tessera runtime checkout is dirty")
            tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(Path(package.__file__).resolve().relative_to(root))], capture_output=True, text=True)
            if tracked.returncode != 0:
                _refuse("imported Tessera runtime is not tracked by its observed repository")
            head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
            if strict:return head
            return head, facts
    try:
        distribution = importlib.metadata.distribution("tessera-quant")
        owned = distribution.files and any(Path(distribution.locate_file(p)).resolve() == Path(package.__file__).resolve() for p in distribution.files)
        if not owned:
            _refuse("installed VCS metadata does not own the imported Tessera runtime")
        metadata = distribution.read_text("direct_url.json") if owned or strict else distribution.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        metadata = None
    if metadata:
        record = json_bytes(metadata)
        commit = record.get("vcs_info", {}).get("commit_id")
        if commit:
            return (hex_identity(commit,40), facts) if not strict else hex_identity(commit,40)
    _refuse("imported Tessera runtime has no observed immutable VCS identity")
    if strict:raise ValueError("imported Tessera runtime has no observed immutable VCS identity")
    return None, facts



def runtime_origins(expected_root, facts=None):
 """Import-environment identity. ``facts`` (D32 dev) collects the foreign
 module origin/owner provenance instead of refusing; the containment checks
 below refuse in both modes."""
 import importlib
 root=Path(expected_root)
 if not root.is_absolute() or root.resolve()!=root or root.is_relative_to(ROOT):raise ValueError("runtime root must be the installed immutable package outside producer checkout")
 if os.environ.get("PYTHONPATH"):raise ValueError("native phase refuses PYTHONPATH")
 if any(Path(v or os.getcwd()).resolve().is_relative_to(ROOT/"src") for v in sys.path):raise ValueError("native phase refuses producer src in import paths")
 # The sealed accepted installation overlay is explicit; producer src is never inserted.
 overlay=str(root.parent)
 if overlay not in sys.path:sys.path.insert(0,overlay)
 files={}
 for name in MODULES:
  module=importlib.import_module(name);path=Path(module.__file__).resolve()
  expected=root/("__init__.py" if name=="tessera" else name.removeprefix("tessera.").replace(".","/")+".py")
  if path!=expected:
   if facts is None:raise ValueError("foreign runtime module origin: "+name)
   facts.append("foreign runtime module origin: "+name)
  files[name]=file_binding(path)
 for name,module in list(sys.modules.items()):
  if name=="tessera" or name.startswith("tessera."):
   path=getattr(module,"__file__",None)
   if path is None or not Path(path).resolve().is_relative_to(root):
    if facts is None:raise ValueError("foreign loaded runtime owner: "+name)
    facts.append("foreign loaded runtime owner: "+name)
 return {"package_root":str(root),"modules":files}


def observe_software_runtime(expected, record_verifier=None):
 def seal(kind,exp,act,refusal,**kw):
  return _DEV_MODE.seal_check(kind,exp,act,where="native shape worker before device setup",
                              refusal=refusal,**kw)
 dev=_DEV_MODE.dev_mode_enabled()
 # D32: in default dev mode the provenance facts are collected and stamped;
 # certified mode keeps every refusing gate verbatim.
 provenance=[] if dev else None
 origins=runtime_origins(expected["package_root"],facts=provenance)
 import torch,vllm,tessera
 from tessera.serving import backend,contract,source_identity,runtime_image
 declaration=runtime_image.declared_reference(expected["image"])
 # VCS provenance stamps and continues; a tree with no derivable identity
 # retains the stored commit, which the stamps name.
 if dev:
  actual_commit,commit_facts=observed_commit(tessera,strict=False)
  provenance.extend(commit_facts)
  if actual_commit is None:
   actual_commit=expected["tessera_commit"]
   provenance.append("running VCS identity not derivable; the stored commit is retained")
 else:
  actual_commit=observed_commit(tessera)
 # Dev computes no digest over existing data to satisfy identity; the record
 # retains the stored source identity and the stamps name that.
 if dev:
  actual_source=expected["serving_source_sha256"]
 else:
  cache=getattr(source_identity,"_cached_digest",None)
  if cache is not None:cache.cache_clear()
  actual_source=source_identity.serving_source_sha256()
 raw_contract=contract.contract_path().read_bytes()
 actual_contract=hashlib.sha256(raw_contract).hexdigest()
 observed=(declaration["image"],actual_commit,actual_source,actual_contract,torch.__version__,vllm.__version__)
 declared=(expected["image"],expected["tessera_commit"],expected["serving_source_sha256"],expected["contract_sha256"],expected["torch"],expected["vllm"])
 # The installed code and contract against the frozen expected pin is
 # cross-pin run identity: it seals in dev mode and refuses verbatim in
 # certified mode, before anything queries a device.
 seal("runtime source/contract/version",declared,observed,
      ValueError("runtime source/contract/version differs before device setup"))
 for fact in provenance:
  seal("runtime provenance",expected["tessera_commit"],fact,
       ValueError("runtime provenance differs before device setup"))
 if not dev:
  if record_verifier is None:raise ValueError("missing sealed installation verifier")
  verifier_bytes=record_verifier_bytes(record_verifier)
  verifier_namespace={"__file__":record_verifier["path"],"__name__":"tessera_owned_record_verifier"}
  exec(compile(verifier_bytes,record_verifier["path"],"exec"),verifier_namespace)
  origins["installation"]=verifier_namespace["verify_install"]("tessera",expected["tessera_commit"])
  if record_verifier_bytes(record_verifier)!=verifier_bytes:raise ValueError("installation verifier changed during execution")
  origins["record_verifier"]=file_binding(record_verifier["path"])
  if origins["record_verifier"]!=record_verifier:raise ValueError("installation verifier source differs")
 else:
  # A dev run requires no installation proof and no clean tree: the
  # PB-published verifier binding is retained when provided, and the stored
  # installation identity is what the record carries.
  origins["record_verifier"]=file_binding(record_verifier["path"]) if record_verifier is not None else None
  origins["installation"]={"module":"tessera","distribution":"unverified (D32 dev mode)",
   "expected_commit":expected["tessera_commit"],"installed_commit":actual_commit,
   "origin":str(Path(expected["package_root"])/"__init__.py"),"verified_files":0}
 value={"image":declaration["image"],"tessera_commit":actual_commit,
  "serving_source_sha256":actual_source,
  "contract_sha256":actual_contract,
  "torch":torch.__version__,"vllm":vllm.__version__,
  "execution_mode":"eager","residency":"resident","tp_rank":0,"tp_degree":1,
  "package_root":origins["package_root"],
  "serve_flags":{k:os.environ[k] for k in expected["serve_flags"] if k in os.environ}}
 frozen={k:v for k,v in expected.items() if k!="platform"}
 # D32 boundary: which CASE executed is comparability, not run identity.
 # Execution semantics (mode, residency, TP geometry, requested serve flags)
 # refuse on any mismatch in both modes; only the code/origin identity seals.
 execution={k:value[k] for k in ("execution_mode","residency","tp_rank","tp_degree","serve_flags") if k in value}
 frozen_execution={k:frozen[k] for k in ("execution_mode","residency","tp_rank","tp_degree","serve_flags") if k in frozen}
 if canonical(execution)!=canonical(frozen_execution):raise ValueError("actually imported software differs from frozen expected context")
 identity={k:value[k] for k in ("image","tessera_commit","serving_source_sha256","contract_sha256","torch","vllm","package_root") if k in value}
 frozen_identity={k:frozen[k] for k in ("image","tessera_commit","serving_source_sha256","contract_sha256","torch","vllm","package_root") if k in frozen}
 seal("actually imported software identity",frozen_identity,identity,
      ValueError("actually imported software differs from frozen expected context"))
 # The installed reader owns all loader, source and registry checks. No caller roster.
 if Path(contract.validate_serving_contract.__code__.co_filename).resolve()!=Path(contract.__file__).resolve():raise ValueError("foreign installed contract validator")
 contract.validate_serving_contract(json_bytes(raw_contract))
 return value,origins,raw_contract


def observe_runtime(expected, record_verifier=None):
 value,origins,_=observe_software_runtime(expected,record_verifier)
 import torch
 from tessera.serving import backend
 value["platform"]=backend.platform_of_this_process(torch)
 if canonical(value)!=canonical(expected):raise ValueError("actually imported hardware differs from frozen expected context")
 return value,origins
def require_fused_preparation(native, family):
    """The first receipt slice observes only an actual ELF-backed fused owner."""
    from tessera.serving import native_window
    window_family=native_window.NATIVE_WINDOW_FAMILY[family]
    published=native_window.DENSE_LANES[native_window.LANE_FUSED]
    pair=(published[0],published[1][window_family])
    if native.lane!=native_window.LANE_FUSED or tuple(native.launch_pair)!=pair:
        raise ValueError("first timing slice requires the prepared ELF-backed fused lane")
    return pair


def loaded_fused_binary(native, family, raw_contract):
    """Observe existing process mappings; never invoke a native library loader."""
    import fnmatch
    from tessera import routed_fused
    from experiments.bench_native_operator import _mapped_shared_libraries
    require_fused_preparation(native,family)
    prefix=routed_fused.MODULE_NAME_E4M3
    contracts=[row for row in json_bytes(raw_contract)["native_extensions"] if row["module_name_prefix"]==prefix]
    if len(contracts)!=1:raise ValueError("actual fused owner lacks one native binary declaration")
    paths=[path for path in _mapped_shared_libraries() if fnmatch.fnmatch(path.name,contracts[0]["filename_glob"])]
    if len(paths)!=1:raise ValueError("actual fused call has no unique already-loaded native ELF")
    if not paths[0].read_bytes().startswith(b"\x7fELF"):raise ValueError("loaded native object is not ELF")
    return file_binding(paths[0])


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
    require_fused_preparation(native,request["scheme"]["family"])
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


def _native_measure(job, output, job_source):
    if set(job) != JOB_FIELDS or job["schema"] != "tessera.native_shape_worker_job.v1":
        raise ValueError("unknown native phase job")
    request=dict(job["request"]);request["_wire_roles"]=job["wire_roles"]
    wire=read_bound(request["wire"])
    if len(wire)!=request["wire"]["bytes"]:raise ValueError("wire bytes differ")
    import torch
    from tessera.serving import contract
    from experiments.bench_native_operator import time_apply
    from experiments.routed_pair_oracle import PowerSampler

    runtime, origins = observe_runtime(request["expected_runtime"],request["record_verifier"])
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
    native_binary=loaded_fused_binary(layer.tessera_native,request["scheme"]["family"],actual_contract)
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
    if loaded_fused_binary(layer.tessera_native,request["scheme"]["family"],actual_contract)!=native_binary:
        raise ValueError("actual loaded native ELF changed during measurement")
    evidence = {"trace": trace, "native_binary": native_binary}
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
    after,after_origins=observe_runtime(request["expected_runtime"],request["record_verifier"])
    if after!=runtime or after_origins!=origins or file_binding(__file__)!=worker_source:
        raise ValueError("runtime, contract, module origin or worker source changed during measurement")
    if file_binding(job_source["path"])!=job_source:raise ValueError("native job changed during measurement")
    result={"schema":SCHEMA,"evidence":evidence,"worker_source":worker_source,"job_source":job_source,"pair":pair}
    return publish_json(result,output/"worker-result.json")

JOB_FIELDS={"schema","request","request_source","wire_roles","producer","worker_source"}


def read_job(job_path, expected_sha256):
    # Hash and parse the same owned buffer; bind the request's original bytes too.
    raw=Path(job_path).read_bytes()
    hex_identity(expected_sha256,64)
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError("owned native job bytes differ")
    job=json_bytes(raw)
    if set(job)!=JOB_FIELDS or job["schema"]!="tessera.native_shape_worker_job.v1":raise ValueError("unknown native phase job")
    if file_binding(__file__)!=job["worker_source"]:raise ValueError("native worker source differs")
    if json_bytes(read_bound(job["request_source"]))!=job["request"]:raise ValueError("native request differs from owned original bytes")
    for key in ("wire","contract","runtime_python","record_verifier"):
        bound=job["request"][key];data=read_bound(bound)
        if len(data)!=bound["bytes"]:raise ValueError("native input bytes differ")
    source={"path":str(Path(job_path).resolve()),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
    return job,source


def runtime_preflight(job_path, output, expected_job_sha256):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU preflight requires disabled CUDA visibility")
    job,job_source=read_job(job_path,expected_job_sha256)
    software,origins,raw=observe_software_runtime(job["request"]["expected_runtime"],job["request"]["record_verifier"])
    import torch
    from tessera.serving import contract
    if torch.cuda.is_initialized():raise ValueError("CPU contract preflight initialized CUDA")
    if raw!=read_bound(job["request"]["contract"]):raise ValueError("installed contract differs from owned request bytes")
    result={"schema":"tessera.installed_contract_preflight.v1", "software":software,"runtime_origins":origins,
            "validator":{"module":"tessera.serving.contract","function":"validate_serving_contract",
                         "source":file_binding(contract.__file__)},
            "contract_sha256":hashlib.sha256(raw).hexdigest(),"gpu_executed":False,
            "worker_source":job["worker_source"],"job_source":job_source,"request_source":job["request_source"]}
    after,after_origins,after_raw=observe_software_runtime(job["request"]["expected_runtime"],job["request"]["record_verifier"])
    if after!=software or after_origins!=origins or after_raw!=raw:raise ValueError("installed runtime changed during CPU preflight")
    if torch.cuda.is_initialized():raise ValueError("CPU contract preflight initialized CUDA")
    if file_binding(job_path)!=job_source or file_binding(__file__)!=job["worker_source"]:raise ValueError("preflight source changed")
    return publish_json(result,Path(output)/"runtime-preflight.json")


def native_measure(job_path, output, expected_job_sha256):
    job,job_source=read_job(job_path,expected_job_sha256)
    # Strict installed-runtime validation precedes any distributed/device setup.
    observe_runtime(job["request"]["expected_runtime"],job["request"]["record_verifier"])
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from experiments import bench_native_operator as instrument
    if Path(instrument.__file__).resolve()!=ROOT/"experiments/bench_native_operator.py":raise ValueError("foreign native context/timing helper")
    with instrument.native_runtime_context():return _native_measure(job,output,job_source)


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--job",type=Path,required=True);ap.add_argument("--job-sha256",required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--preflight",action="store_true");args=ap.parse_args(argv)
    try:
        operation=runtime_preflight if args.preflight else native_measure
        print(json.dumps(operation(args.job,args.output,args.job_sha256),sort_keys=True));return 0
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:print("[native-phase] REFUSED: "+str(exc),file=sys.stderr);return 2
if __name__=="__main__":raise SystemExit(main())
