#!/usr/bin/env python3
"""Run the step-4 full-engine capture of a served Tessera artifact, fail-closed on the native lanes.

``g2_launch.py`` (the native-cell launcher staged beside each frontier run) is
the only thing that wired ``full_engine_plugin_install.py``, and it is shaped
for a bench driver.  This is the same wiring for the capture CLI, plus the one
thing that wiring did not have: a proof, taken in the same container before the
engine exists, that the native code the artifact's families dispatch through
is importable and buildable there, and a per-family qualification of the
serve's own route trace afterwards.

WHY THAT PROOF IS THE POINT.  A resource capture measures allocation, and two
decoders of the same bytes do not allocate alike, so a ledger is about a
decoder and the record has to say which one.  The pre-``37e89f576`` tree made
that a silent hazard: ``ext.NATIVE_EXTENSIONS`` published a ``torch_materialize_stock``
substitute for a container that could not compile the ``tessera_nvfp4``
extension, so the serve succeeded and the capture read as qualified while
describing the wrong decoder.  On master every dense route makes exactly one
native launch and raises rather than falling back (``fp8_route.DENSE_LAUNCH``,
``bf16_route.DENSE_LAUNCH``, ``kernel_a4.require_native_fp4_mma``), so the
hazard is now a loud failure -- and the launcher still refuses at two points:
the in-container preflight must import the native lanes (and pass the FP4
MMA gate when the artifact carries an NVFP4 family), and the serve's own
route trace must name each family's one native ``(symbol, decoder)`` on every
dispatch, over exactly the manifest's modules.  No dense launch loads a
``cpp_extension`` on this tree, so there is no library to bind; the worker's
mapped-library census is recorded, not required.

WHAT IT DOES NOT DO.  The construction census is run (``--with-census``) because
the stock image refuses the ``tessera`` quantization method and the plugin image
is where that census can run at all -- but it is NOT fed to the capture:
``capture_full_engine_resources.main`` makes ``--census`` and ``--artifact``
mutually exclusive, because a served artifact carries its roster in
``tessera_serving_manifest.json``.

THE THREE PASSES (tessera#399).  ``--observation-mode`` selects which of the
capture CLI's passes runs in the container: ``resources`` (the intrusive
ledger pass, the default), ``kv`` (the read-only stock-engine KV pass that
closes ``cache_capacity`` when joined with it) and ``timings`` (the profiled
all-native-unit event partition, ``--timing-samples`` interleaved arm pairs).
``prepare()`` admits an artifact into every pass since the #399 derivation
layer; each pass gets its own fresh ``--out`` and the same configuration
document, so the three ledgers share one ``configuration_sha256`` and the
report joins them by it.  The kv pass runs a stock worker, so it carries no
worker census: its qualification is the route trace alone, which on this tree
is the whole proof, and the record says the library census is absent.

``--preflight-only`` stops after the native proof and an observer load smoke
(both collector libraries ``dlopen``ed and their entry symbols resolved, the
observer modules imported from ``/tessera``, the stock runner's stream hooks
located) and writes ``observer-preflight.json``.  It exists so a new image or
a new tree is refused in minutes, before a capture is spent on it.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

# The launcher lives in ``experiments/``; the capture CLI is imported as a
# package member so the allocator-policy rule keeps ONE home (see
# :func:`bound_container_environment`).  The import is torch-free by
# construction -- ``prepare`` refuses a process that imported Torch or vLLM
# before bootstrap -- so it is safe on a host that has neither.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.capture_full_engine_resources import (  # noqa: E402
    ALLOCATOR_POLICY_KEY, UNSET_ALLOCATOR_POLICY, require_allocator_policy)
from experiments.full_engine_plugin_install import source_tree_identity  # noqa: E402

#: OOM discipline for a shared GB10: GPU and host share one pool, so a serve
#: started under memory pressure takes the box down with it.
MIN_MEMAVAILABLE_GIB = 16.0
MAX_PSI_FULL_AVG10 = 20.0
WORKER_UID = 1000  # The pinned plugin installer drops the engine worker to this UID.


def prepare_worker_jit_cache(jit_dir: Path) -> None:
    """Bind reusable explicit cache roots before the installer drops UID."""
    for name in ("home", "xdg", "tmp", "triton", "torch-extensions", "inductor", "cuda-cache"):
        (jit_dir / name).mkdir(parents=True, exist_ok=True)
    for name in ("home", "xdg", "tmp", "triton", "torch-extensions", "inductor", "cuda-cache"):
        path = jit_dir / name
        if path.is_symlink() or path.stat().st_uid != WORKER_UID or not os.access(path, os.W_OK | os.X_OK):
            raise RuntimeError(f"scoped JIT cache is not writable by pinned worker UID {WORKER_UID}: {path}")


def bound_container_environment(config) -> dict:
    """The configuration's environment block as the CONTAINER must receive it.

    ``require_allocator_policy`` is the one home of the rule that the
    configuration must bind ``PYTORCH_CUDA_ALLOC_CONF`` and that ``"unset"``
    names the variable's ABSENCE rather than a value (tessera#558, PR #565).
    ``capture_full_engine_resources.prepare`` applies it to the worker
    environment it builds.  This applies it one level further out, to the
    container this launcher starts, and it has to: the sentinel is not a token
    c10 can parse, and torch parses the variable at ``libc10_cuda.so`` load
    time, before any of our code runs.

    Measured on sparklina, 2026-09-21, the first capture attempt of the
    tessera#399 master re-run: ``terminate called after throwing an instance of
    'c10::Error' ... Index out of bounds in ConfigTokenizer``, preflight
    returncode 133, with no engine started and nothing in the ledger to say
    why.  Passing the block through verbatim turned an explicit "no policy"
    binding into a process abort.
    """
    environment = dict(config["environment"])
    if require_allocator_policy(config) == UNSET_ALLOCATOR_POLICY:
        environment.pop(ALLOCATOR_POLICY_KEY, None)
    return environment


def require_tp2_source(config, tree: Path, commit: str) -> str:
    runtime = config["runtime_identity"]
    if runtime.get("plugin_source_commit") != commit:
        raise ValueError("selected configuration and launcher name different source commits")
    source_sha, _members = source_tree_identity(tree)
    if source_sha != runtime.get("plugin_source_sha256"):
        raise ValueError("frozen plugin source bytes differ from the selected configuration")
    return source_sha


def digest(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def memory_pressure() -> dict:
    available_kb = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kb = int(line.split()[1])
            break
    psi = {}
    pressure = Path("/proc/pressure/memory")
    if pressure.exists():
        for line in pressure.read_text().splitlines():
            kind, _, rest = line.partition(" ")
            psi[kind] = {k: float(v) for k, v in
                         (item.split("=") for item in rest.split() if "=" in item)}
    return {"memavailable_gib": round(available_kb / 1024 / 1024, 2),
            "psi": psi, "psi_full_avg10": psi.get("full", {}).get("avg10")}


def require_headroom(record: dict) -> None:
    if record["memavailable_gib"] < MIN_MEMAVAILABLE_GIB:
        raise SystemExit(f"refusing to load: MemAvailable {record['memavailable_gib']} GiB "
                         f"< {MIN_MEMAVAILABLE_GIB} GiB")
    full = record["psi_full_avg10"]
    if full is not None and full >= MAX_PSI_FULL_AVG10:
        raise SystemExit(f"refusing to load: memory PSI full avg10 {full} >= {MAX_PSI_FULL_AVG10}")


def gpu_sample() -> dict:
    query = "uuid,power.draw,memory.used,temperature.gpu"
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True)
    rows = [line.strip() for line in out.strip().splitlines() if line.strip()]
    if len(rows) != 1:
        raise SystemExit("this launcher requires exactly one visible GPU")
    uid, power, used, temperature = (item.strip() for item in rows[0].split(","))

    def number(text):
        # On GB10 several of these read ``[N/A]``: the GPU shares the host's
        # memory pool, so ``memory.used`` has nothing device-local to report.
        # Recording the absence is the honest value; inventing 0.0 would make a
        # missing measurement look like an idle one.
        try:
            return float(text)
        except ValueError:
            return None

    watts = number(power)
    return {"uuid": uid, "power_w": watts, "memory_used_mib": number(used),
            "temperature_c": number(temperature), "envelope_w": 140.0,
            "envelope_fraction": None if watts is None else round(watts / 140.0, 3)}


def image_declaration(tessera_tree: Path, base: str, configuration_sha256: str, out: Path) -> dict:
    """Built by Tessera's OWN resolver from the frozen tree, never restated here."""
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", base]))[0]
    if base not in inspected.get("RepoDigests", []):
        raise SystemExit("pinned base must appear in the inspected RepoDigests")
    spec = importlib.util.spec_from_file_location(
        "tessera_runtime_image_step4", tessera_tree / "src/tessera/serving/runtime_image.py")
    runtime_image = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime_image)
    contract = json.loads((tessera_tree / "src/tessera/serving/runtime_contract.json").read_text())
    declaration = runtime_image.require_pinned(base, contract=contract, inspector=lambda reference: {
        "present": True, "local_id": inspected["Id"],
        "repo_digests": sorted(inspected["RepoDigests"]), "error": None})
    declaration["selection"] = {
        "scope": "explicit research observation configuration, not a packaged default or cell promotion",
        "configuration_sha256": configuration_sha256,
        "packaged_default_reference": runtime_image.pinned_reference(contract)}
    (out / "launcher-image-inspect.json").write_text(json.dumps(inspected, indent=2))
    (out / "runtime-image-declaration.json").write_text(json.dumps(declaration, indent=2))
    return {"inspected": inspected, "declaration": declaration,
            "environment": runtime_image.container_env(declaration)}


def family_modules(artifact: Path) -> dict:
    """``{family: {"count": n, "names": [module prefixes]}}`` from the artifact's manifest.

    Every family the manifest assigns a module to, so the driver qualifies
    each one against the route trace; the names are the manifest's module keys
    (vLLM prefixes such as ``model.layers.0.mlp.gate_up_proj``), which the
    trace's ``module_names`` must match where it names them.
    """
    manifest = json.loads((artifact / "tessera_serving_manifest.json").read_text())
    families: dict = {}
    for name, entry in manifest["modules"].items():
        family = entry.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError(f"manifest module {name!r} names no family")
        bucket = families.setdefault(family, {"count": 0, "names": []})
        bucket["count"] += 1
        bucket["names"].append(name)
    for bucket in families.values():
        bucket["names"].sort()
    if not families:
        raise ValueError(f"{artifact}: the manifest assigns no module to any family")
    return families


def docker_command(*, name: str, image_id: str, mounts, environment, entry, out_host: Path,
                   jit_host: Path, ext_host: Path, ext_readonly: bool, affinity=None,
                   tp2=False, extra_mounts=(), cidfile=None) -> list:
    command = ["docker", "run", "--rm", "--gpus", "all", "--ipc", "host", "--network",
               "host" if tp2 else "none",
               "--name", name, "--workdir", "/tessera"]
    if cidfile is not None:
        command += ["--cidfile", str(cidfile), "--label", f"org.prismaquant.pact-observer={name}"]
    if affinity is not None:
        # The CPU mask this launcher was granted, carried into the container.
        # PrismaBuild's execution policy requires its reservation to be
        # preserved "including inside containers", and a container started
        # without a mask is placed on every host CPU whatever the pool
        # reserved.  Here that is more than politeness: the whole output is a
        # resource and timing observation, and one taken on an oversubscribed
        # box is not the observation the configuration describes.
        # ``--cpuset-cpus`` and not ``--cpus`` because
        # ``experiments/container_limits_probe.sh`` measured that a CFS quota
        # changes no CPU count a library can read, so a quota-limited container
        # still sizes its pools for the whole box (owned_container.sh, lines
        # two and three of that probe).  ``run_glm_native_construction.py`` is
        # the existing docker-under-PrismaBuild launcher spelled the same way.
        cpus = sorted(int(cpu) for cpu in affinity)
        if not cpus:
            raise ValueError("empty CPU affinity: a mask that read as nothing would pin the "
                             "capture container to no CPU at all, which is not a reservation")
        command += ["--cpuset-cpus", ",".join(str(cpu) for cpu in cpus)]
    for source, target, mode in mounts:
        command += ["--volume", f"{source}:{target}:{mode}"]
    # The scratch roots stay writable even in the negative control: making the
    # whole container unwritable would refuse for a reason that is not the one
    # under test.  Only the extension build root is mutated.
    command += ["--volume", f"{out_host}:/out", "--volume", f"{jit_host}:/jit:rw",
                "--volume", f"{ext_host}:/jit-ext:" + ("ro" if ext_readonly else "rw")]
    for source, target, mode in extra_mounts:
        command += ["--volume", f"{source}:{target}:{mode}"]
    for key, value in environment.items():
        command += ["--env", f"{key}={value}"]
    command += ["--entrypoint", "python3", image_id, *entry]
    return command


def stop_owned_container(cidfile: Path, name: str) -> dict:
    """Stop only the exact CID this phase created, then attest its absence."""
    container_id = cidfile.read_text().strip() if cidfile.is_file() else None
    if container_id is not None and (len(container_id) != 64 or
                                     any(character not in "0123456789abcdef" for character in container_id)):
        raise RuntimeError("owned Docker CID file is malformed")

    def inspect(reference):
        result = subprocess.run(["docker", "inspect", reference],
                                capture_output=True, text=True, timeout=15)
        if result.returncode:
            if "no such object" in (result.stderr + result.stdout).lower():
                return None
            raise RuntimeError("cannot inspect owned observer container: " + result.stderr[-500:])
        record = json.loads(result.stdout)[0]
        if ((container_id is not None and record.get("Id") != container_id)
                or (record.get("Config", {}).get("Labels") or {}).get(
                    "org.prismaquant.pact-observer") != name):
            raise RuntimeError("Docker CID does not carry this phase's observer owner label")
        return record

    # Docker may create the named container before the client writes --cidfile.
    # The name is generated uniquely for this phase, but is never by itself
    # authority to stop anything: require our label before deriving its ID.
    record = inspect(container_id or name)
    if record is not None:
        container_id = record["Id"]
    actions = []
    if record is not None:
        try:
            stopped = subprocess.run(["docker", "stop", "--time", "5", container_id],
                                     capture_output=True, text=True, timeout=30)
            actions.append({"operation": "stop", "returncode": stopped.returncode})
        except subprocess.TimeoutExpired:
            actions.append({"operation": "stop", "timeout": True})
        record = inspect(container_id)
        if record is not None:
            try:
                removed = subprocess.run(["docker", "rm", "-f", container_id],
                                         capture_output=True, text=True, timeout=30)
                actions.append({"operation": "rm", "returncode": removed.returncode})
            except subprocess.TimeoutExpired:
                actions.append({"operation": "rm", "timeout": True})
            record = inspect(container_id)
    return {"cidfile": str(cidfile), "container_id": container_id,
            "absent": record is None, "actions": actions,
            "identity_source": "cidfile" if cidfile.is_file() else "owned_name_and_label"}


def write_head_finished(out: Path, session_id: str, returncode=None):
    path = out / "head-finished.json"
    if path.exists():
        return
    plan = out / "capture" / "observer-plan.json"
    path.write_text(json.dumps({"schema": "tessera.tp2_head_finished.v1",
                                "session_id": session_id,
                                "plan_sha256": digest(plan) if plan.is_file() else None,
                                "returncode": returncode}) + "\n")


def run_phase(label: str, command: list, log: Path, timeout_s: int) -> dict:
    started = time.time()
    cidfile = (Path(command[command.index("--cidfile") + 1])
               if "--cidfile" in command else None)
    owner = command[command.index("--name") + 1] if cidfile is not None else None
    pressure_stop = None
    cleanup = None
    returncode = None
    with log.open("wb") as stream:
        process = subprocess.Popen(["timeout", "--signal=INT", "--kill-after=60", str(timeout_s), *command],
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    returncode = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    if cidfile is None or not cidfile.is_file() or pressure_stop is not None:
                        continue
                    pressure = memory_pressure()
                    try:
                        require_headroom(pressure)
                    except SystemExit as exc:
                        pressure_stop = {"reason": str(exc), "memory": pressure}
                        try:
                            cleanup = stop_owned_container(cidfile, owner)
                        except Exception as cleanup_exc:
                            cleanup = {"absent": False,
                                       "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}"}
                            returncode = 97
                            break
        finally:
            if cidfile is not None:
                try:
                    cleanup = stop_owned_container(cidfile, owner)
                except Exception as exc:
                    cleanup = {"absent": False, "error": f"{type(exc).__name__}: {exc}"}
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    seconds = time.time() - started
    if pressure_stop is not None:
        returncode = 98
    if cleanup is not None and not cleanup["absent"]:
        returncode = 97
    print(json.dumps({"phase": label, "returncode": returncode,
                      "seconds": round(seconds, 1), "log": str(log)}), flush=True)
    return {"phase": label, "returncode": returncode, "seconds": round(seconds, 3),
            "command": command, "log": str(log), "pressure_stop": pressure_stop,
            "owned_container_cleanup": cleanup}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True,
                        help="fresh ledger directory; refuses a directory that already exists")
    parser.add_argument("--tessera-tree", type=Path, required=True,
                        help="the frozen producer source tree the configuration names")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--control", type=Path, required=True,
                        help="the checkout holding step4_capture_driver.py")
    parser.add_argument("--serving-config", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--workspaces", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--jit-dir", type=Path, required=True,
                        help="LOCAL-disk JIT/build root; an NFS bind fails to inspect the architecture")
    parser.add_argument("--jit-readonly", action="store_true",
                        help="negative control: make the build root unwritable, so the preflight "
                             "must refuse instead of letting the route substitute torch")
    parser.add_argument("--with-census", action="store_true",
                        help="also run the construction census in the plugin image; it is recorded, "
                             "not consumed (--census and --artifact are mutually exclusive)")
    parser.add_argument("--observation-mode", choices=("resources", "kv", "timings"), default="resources",
                        help="which capture pass runs: the intrusive ledger, the read-only KV pass, "
                             "or the profiled timing partition")
    parser.add_argument("--timing-samples", type=int, default=1,
                        help="timings only: interleaved control/partition arm pairs")
    parser.add_argument("--observer-impact-policy", type=Path,
                        help="explicit versioned policy bytes for TP2 timing qualification")
    parser.add_argument("--tp2-role", choices=("head", "peer"),
                        help="two-host joined MP observer role; peer --out is HEAD_OUT/rank-1")
    parser.add_argument("--tp2-host-ip",
                        help="this host's actual transport IPv4 address, checked by the worker")
    parser.add_argument("--preflight-only", action="store_true",
                        help="JIT proof plus observer load smoke in the pinned image; no engine, no capture")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--census-timeout-s", type=int, default=900)
    args = parser.parse_args()

    if args.tp2_role:
        if (not args.tp2_host_ip or args.preflight_only or args.with_census
                or args.observation_mode == "timings" and args.timing_samples < 3):
            parser.error("TP2 roles need a host IP, full capture, no census, and >=3 timing samples")
        if not args.out.resolve().is_relative_to(Path("/mnt/shared")):
            parser.error("TP2 role directories must share the mounted /mnt/shared path")
        if args.tp2_role == "peer" and args.out.name != "rank-1":
            parser.error("TP2 peer --out must be HEAD_OUT/rank-1")
        if args.tp2_role == "peer" and not (args.out.parent / "host-preconditions.json").is_file():
            parser.error("start TP2 head first; its host-preconditions record is absent")
        if args.tp2_role == "peer" and not (args.out.parent / "head-session.json").is_file():
            parser.error("TP2 head has not published its owned session identity")
    elif args.tp2_host_ip:
        parser.error("--tp2-host-ip needs an explicit TP2 role")

    out = args.out
    out.mkdir(parents=True)                       # fresh per run; refuses a reused directory
    session_id = None
    if args.tp2_role == "head":
        session_id = uuid.uuid4().hex
        (out / "head-session.json").write_text(json.dumps({
            "schema": "tessera.tp2_head_session.v1", "session_id": session_id}) + "\n")
        atexit.register(write_head_finished, out, session_id)
    elif args.tp2_role == "peer":
        session_id = json.loads((out.parent / "head-session.json").read_text())["session_id"]
    if args.tp2_role:
        def stop_signal(signum, _frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, stop_signal)
    args.jit_dir.mkdir(parents=True, exist_ok=True)
    configuration_sha256 = digest(args.serving_config)
    config = json.loads(args.serving_config.read_text())
    if args.tp2_role:
        engine = config["engine_args"]
        if (engine.get("tensor_parallel_size") != 2 or engine.get("nnodes") != 2
                or engine.get("node_rank") != 0 or engine.get("distributed_executor_backend") != "mp"
                or not engine.get("master_addr")):
            raise SystemExit("TP2 launcher requires explicit head MP/nnodes/master topology")
        head_ip = config["environment"].get("VLLM_HOST_IP")
        if args.tp2_role == "head" and args.tp2_host_ip != head_ip:
            raise SystemExit("TP2 head host IP differs from the sealed configuration")
        if args.tp2_role == "peer" and args.tp2_host_ip == head_ip:
            raise SystemExit("TP2 peer host IP equals the sealed head host IP")
    base = config["runtime_image"]
    declared_tree = config["runtime_identity"]["plugin_source_tree"]
    if str(args.tessera_tree.resolve()) != declared_tree:
        raise SystemExit(f"configuration names plugin source tree {declared_tree}, launcher was given "
                         f"{args.tessera_tree.resolve()}")
    if args.tp2_role:
        require_tp2_source(config, args.tessera_tree, args.source_commit)
    if str(args.artifact.resolve()) != str(Path(config["artifact"]["path"]).resolve()):
        raise SystemExit("configuration names a different artifact than the launcher was given")

    headroom = memory_pressure()
    require_headroom(headroom)
    before = gpu_sample()
    # Read once, here, and carried into every container this launcher starts.
    affinity = sorted(os.sched_getaffinity(0))
    (out / "host-preconditions.json").write_text(json.dumps(
        {"schema": "tessera.step4_launch_preconditions.v1", "hostname": os.uname().nodename,
         "memory": headroom, "gpu": before, "affinity": affinity,
         "gate": {"min_memavailable_gib": MIN_MEMAVAILABLE_GIB, "max_psi_full_avg10": MAX_PSI_FULL_AVG10}},
        indent=2, sort_keys=True) + "\n")

    resolved = image_declaration(args.tessera_tree, base, configuration_sha256, out)
    image_id = resolved["inspected"]["Id"]
    mounts = [(args.tessera_tree, "/tessera", "ro"), (args.control, "/control", "ro"),
              ("/mnt/shared", "/mnt/shared", "ro")]
    environment = {
        "PYTHONPATH": "/tessera", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MAX_JOBS": "4",
        # Every generated-code root on LOCAL disk: an NFS bind mount under
        # root_squash fails the architecture inspection torch's JIT does.
        "HOME": "/jit/home", "XDG_CACHE_HOME": "/jit/xdg", "TMPDIR": "/jit/tmp",
        "TRITON_CACHE_DIR": "/jit/triton", "TORCH_EXTENSIONS_DIR": "/jit/torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": "/jit/inductor", "CUDA_CACHE_PATH": "/jit/cuda-cache",
        "TESSERA_EXT_DIR": "/jit-ext",
        # The serve's own dispatch histogram: the dispatch leg of the proof.
        "TESSERA_ROUTE_TRACE": "/out/route-trace.json",
        **bound_container_environment(config), **resolved["environment"]}
    if args.tp2_role == "peer":
        environment["VLLM_HOST_IP"] = args.tp2_host_ip
        environment["TESSERA_ROUTE_TRACE"] = "/peer-out/route-trace.json"
    prepare_worker_jit_cache(args.jit_dir)
    # The negative control gets its OWN empty build root, mounted read-only: it
    # must never share a directory with a run that already built the library,
    # or a cache hit would hide the refusal it exists to show.
    ext_host = args.jit_dir / ("tessera-ext-readonly" if args.jit_readonly else "tessera-ext")
    ext_host.mkdir(parents=True, exist_ok=True)

    # Every sample also lists the compute processes the driver sees on the
    # device (host pids): the timing observation's device-exclusivity witness
    # reads this log over the arms' spans, at this cadence, and says so.
    vitals_dir = (out / "rank-0" if args.tp2_role == "head" else out)
    vitals_dir.mkdir(exist_ok=True)
    vitals = subprocess.Popen(
        ["bash", "-c",
         'while true; do printf "%s host_ip=%s gpu_uuid=%s MemAvailable_kB=%s gpu_W=%s gpu_used_MiB=%s compute_apps=%s\\n" '
         '"$(date -u +%FT%TZ)" "$TP2_HOST_IP" "$TP2_GPU_UUID" '
         '"$(awk \'/MemAvailable/{print $2}\' /proc/meminfo)" '
         '"$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)" '
         '"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" '
         '"$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits '
         '| tr -d " " | paste -sd";" -)"; sleep 5; done'],
        stdout=(vitals_dir / "host-vitals.log").open("wb"), stderr=subprocess.STDOUT,
        env=dict(os.environ, TP2_HOST_IP=args.tp2_host_ip or "", TP2_GPU_UUID=before["uuid"]))
    phases = []
    try:
        if args.tp2_role == "peer":
            peer_root = out.parent
            shutil.copy2(args.control / "step4_capture_driver.py", out / "step4_capture_driver.py")
            shutil.copy2(args.control / "step4_route_qualification.py", out / "step4_route_qualification.py")
            shutil.copy2(args.control / "step4_tp2_peer.py", out / "step4_tp2_peer.py")
            peer_entry = ["/tessera/experiments/full_engine_plugin_install.py",
                          "--evidence-dir", "/peer-out", "--base-reference", base,
                          "--launcher-image-id", image_id,
                          "--launcher-image-inspect", "/peer-out/launcher-image-inspect.json",
                          "--source-tree", "/tessera", "--source-commit", args.source_commit,
                          "--core-manifest", str(args.core_manifest), "--",
                          "python3", "-u", "/control/step4_tp2_peer.py",
                          "--plan", "/out/capture/observer-plan.json",
                          "--evidence", "/peer-out/per-job-runtime.json",
                          "--ready", "/peer-out/peer-ready.json",
                          "--out", "/peer-out", "--host-ip", args.tp2_host_ip,
                          "--serve-mode", environment["TESSERA_SERVE_MODE"],
                          "--expected-modules", json.dumps(family_modules(args.artifact), sort_keys=True),
                          "--collector", str(args.collector),
                          "--workspaces", str(args.workspaces) if args.workspaces else "-"]
            phases.append(run_phase("tp2-peer", docker_command(
                name="step4-peer-" + uuid.uuid4().hex[:12], image_id=image_id,
                mounts=mounts, environment=environment, out_host=peer_root,
                jit_host=args.jit_dir, ext_host=ext_host, ext_readonly=args.jit_readonly,
                affinity=affinity, tp2=True, cidfile=out / "container.cid",
                extra_mounts=[(out, "/peer-out", "rw")], entry=peer_entry),
                out / "container.log", args.timeout_s))
        if args.with_census:
            census_out = out / "census"
            census_out.mkdir()
            shutil.copy2(out / "launcher-image-inspect.json",
                         census_out / "launcher-image-inspect.json")
            census_env = dict(environment)
            census_env["PYTHONPATH"] = "/tessera/src"
            census_env.pop("TESSERA_ROUTE_TRACE", None)
            phases.append(run_phase("census", docker_command(
                name="step4-census-" + uuid.uuid4().hex[:12], image_id=image_id, mounts=mounts,
                environment=census_env, out_host=census_out, jit_host=args.jit_dir,
                ext_host=ext_host, ext_readonly=args.jit_readonly, affinity=affinity,
                entry=["/tessera/experiments/full_engine_plugin_install.py",
                       "--evidence-dir", "/out", "--base-reference", base,
                       "--launcher-image-id", image_id,
                       "--launcher-image-inspect", "/out/launcher-image-inspect.json",
                       "--source-tree", "/tessera", "--source-commit", args.source_commit,
                       "--core-manifest", str(args.core_manifest), "--",
                       "python3", "-u", "/tessera/tools/tessera_construction_census.py",
                       str(args.artifact), "/out/census.json", "--runtime-image", base]),
                census_out / "census.log", args.census_timeout_s))

        if args.tp2_role != "peer":
            capture_out = out / "capture"
            mode = args.observation_mode
            capture_argv = ["--config", str(args.serving_config), "--model", str(args.artifact),
                            "--core-manifest", str(args.core_manifest),
                            "--runtime-evidence", "/out/per-job-runtime.json",
                            "--output", "/out/capture", "--artifact", "--all-units",
                            "--observation-mode", mode]
            if args.tp2_role == "head":
                capture_argv += ["--world-size", "2", "--rank", "0"]
            if mode != "kv":
                # The read-only pass loads no collector: a stock engine and a stock
                # worker are what make it the read-only half of the two-pass pair.
                capture_argv += ["--collector", str(args.collector)]
            if mode == "resources" and args.workspaces is not None:
                capture_argv += ["--workspaces", str(args.workspaces)]
            if mode == "timings":
                capture_argv += ["--timing-samples", str(args.timing_samples)]
                if args.observer_impact_policy is not None:
                    shutil.copy2(args.observer_impact_policy, out / "observer-impact-policy.json")
                    capture_argv += ["--observer-impact-policy", "/out/observer-impact-policy.json"]
            if args.calibration is not None:
                capture_argv += ["--calibration", str(args.calibration),
                                 "--calibration-sha256", digest(args.calibration)]
            driver_argv = ["--out", "/out", "--capture-output", "/out/capture",
                           "--route-trace", "/out/route-trace.json",
                           "--expected-modules", json.dumps(family_modules(args.artifact), sort_keys=True),
                           "--observation-mode", mode]
            if args.preflight_only:
                driver_argv.append("--preflight-only")
            if args.tp2_role == "head":
                driver_argv.append("--tp2-head")
            entry = ["/tessera/experiments/full_engine_plugin_install.py",
                     "--evidence-dir", "/out", "--base-reference", base,
                     "--launcher-image-id", image_id,
                     "--launcher-image-inspect", "/out/launcher-image-inspect.json",
                     "--source-tree", "/tessera", "--source-commit", args.source_commit,
                     "--core-manifest", str(args.core_manifest), "--",
                     "python3", "-u", "/control/step4_capture_driver.py", *driver_argv, "--",
                     *capture_argv]
            shutil.copy2(args.control / "step4_capture_driver.py", out / "step4_capture_driver.py")
            shutil.copy2(args.control / "step4_route_qualification.py", out / "step4_route_qualification.py")
            phases.append(run_phase("preflight" if args.preflight_only else "capture", docker_command(
                name="step4-capture-" + uuid.uuid4().hex[:12], image_id=image_id, mounts=mounts,
                environment=environment, out_host=out, jit_host=args.jit_dir,
                ext_host=ext_host, ext_readonly=args.jit_readonly, entry=entry, affinity=affinity,
                tp2=args.tp2_role == "head",
                cidfile=out / "rank-0" / "container.cid" if args.tp2_role == "head" else None),
                out / "container.log", args.timeout_s))
    finally:
        vitals.terminate()
        vitals.wait(timeout=30)
        if args.tp2_role == "head":
            write_head_finished(out, session_id,
                                max((phase["returncode"] for phase in phases), default=None))

    if args.tp2_role == "peer":
        cleanup = phases[-1].get("owned_container_cleanup") if phases else None
        (out / "container-absence.json").write_text(json.dumps({
            "schema": "tessera.tp2_peer_container_absence.v1",
            "session_id": session_id, "cleanup": cleanup,
            "absent": bool(cleanup and cleanup.get("absent"))}) + "\n")
    elif args.tp2_role == "head":
        peer_absence = out / "rank-1" / "container-absence.json"
        deadline = time.monotonic() + 90
        while not peer_absence.is_file() and time.monotonic() < deadline:
            time.sleep(1)
        if peer_absence.is_file():
            absence = json.loads(peer_absence.read_text())
        else:
            absence = None
        if (not isinstance(absence, dict) or absence.get("session_id") != session_id
                or absence.get("absent") is not True):
            phases.append({"phase": "tp2-peer-cleanup", "returncode": 96,
                           "seconds": 0.0, "command": None,
                           "log": str(peer_absence), "reason": "peer CID absence unverified"})

    after = gpu_sample()
    returncode = max(phase["returncode"] for phase in phases)
    qualification = out / "native-route-qualification.json"
    qualified = False
    if returncode == 0 and qualification.exists():
        qualified = json.loads(qualification.read_text()).get("qualified") is True
    scopes = {
        "resources": "raw resource observation of a served artifact; no timing, fixed-resource "
                     "or release admission, and the census is recorded rather than consumed",
        "kv": "read-only stock-engine KV observation of a served artifact; no recorder, no "
              "snapshot, no timing claim; the route proof is the trace alone (a stock worker "
              "carries no library census, and no dense launch needs one)",
        "timings": "profiled all-native-unit event partition of a served artifact; observer "
                   "qualification only, no admitted timing or fixed-resource price"}
    summary = {"schema": "tessera.step4_capture_launch.v1", "out": str(out),
               "hostname": os.uname().nodename, "image": base, "image_id": image_id,
               "configuration_sha256": configuration_sha256, "artifact": str(args.artifact),
               "source_commit": args.source_commit, "jit_dir": str(args.jit_dir),
               "jit_readonly": args.jit_readonly, "ext_dir": str(ext_host), "phases": phases,
               "affinity": affinity,
               "allocator_segment_policy": require_allocator_policy(config),
               "observation_mode": args.observation_mode, "preflight_only": args.preflight_only,
               "timing_samples": args.timing_samples if args.observation_mode == "timings" else None,
               "gpu_before": before, "gpu_after": after,
               "wall_seconds": round(sum(phase["seconds"] for phase in phases), 3),
               "returncode": returncode,
               "qualified": qualified,
               "scope": ("JIT proof and observer load smoke only; no engine ran"
                         if args.preflight_only else scopes[args.observation_mode])}
    (out / "launch-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("returncode", "qualified", "wall_seconds", "out")}),
          flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
