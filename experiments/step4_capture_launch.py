#!/usr/bin/env python3
"""Run the step-4 full-engine capture of a served Tessera artifact, fail-closed on JIT.

``g2_launch.py`` (the native-cell launcher staged beside each frontier run) is
the only thing that wired ``full_engine_plugin_install.py``, and it is shaped
for a bench driver.  This is the same wiring for the capture CLI, plus the one
thing that wiring did not have: a proof, taken in the same container before the
engine exists, that the NVFP4 decode extension actually builds.

WHY THAT PROOF IS THE POINT.  ``nvfp4_route`` prepares its decoder with
``allow_torch_fallback=substitutes_when_unavailable(self._mode)`` and
``ext.NATIVE_EXTENSIONS`` publishes, for ``tessera_nvfp4_``,
``{"resident": {"status": "substituted", "decoder": "torch_materialize_stock"}}``.
A container that cannot compile therefore does not fail: it serves, the request
succeeds and the ledger closes.  The resident serve is numerically untouched by
the substitution, so no quality check would notice -- but a resource capture
measures allocation, and the two decoders do not allocate alike.  The capture
would read as qualified while describing the wrong decoder.  So the launcher
refuses at three points: the preflight must build the library, the worker's own
mapped-library census must show those exact bytes, and the serve's own route
trace must name ``native_span2`` on every NVFP4 dispatch.

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
worker census: its route trace is recorded and the qualification record says
``qualified: false`` with that reason rather than passing on half a proof.

``--preflight-only`` stops after the JIT proof and an observer load smoke
(both collector libraries ``dlopen``ed and their entry symbols resolved, the
observer modules imported from ``/tessera``, the stock runner's stream hooks
located) and writes ``observer-preflight.json``.  It exists so a new image or
a new tree is refused in minutes, before a capture is spent on it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

#: OOM discipline for a shared GB10: GPU and host share one pool, so a serve
#: started under memory pressure takes the box down with it.
MIN_MEMAVAILABLE_GIB = 16.0
MAX_PSI_FULL_AVG10 = 20.0


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


def fp4_module_count(artifact: Path) -> int:
    """How many manifest modules the artifact assigns to the NVFP4 family."""
    manifest = json.loads((artifact / "tessera_serving_manifest.json").read_text())
    return sum(1 for entry in manifest["modules"].values()
               if entry.get("family") == "TESSERA_NVFP4")


def docker_command(*, name: str, image_id: str, mounts, environment, entry, out_host: Path,
                   jit_host: Path, ext_host: Path, ext_readonly: bool) -> list:
    command = ["docker", "run", "--rm", "--gpus", "all", "--ipc", "host", "--network", "none",
               "--name", name, "--workdir", "/tessera"]
    for source, target, mode in mounts:
        command += ["--volume", f"{source}:{target}:{mode}"]
    # The scratch roots stay writable even in the negative control: making the
    # whole container unwritable would refuse for a reason that is not the one
    # under test.  Only the extension build root is mutated.
    command += ["--volume", f"{out_host}:/out", "--volume", f"{jit_host}:/jit:rw",
                "--volume", f"{ext_host}:/jit-ext:" + ("ro" if ext_readonly else "rw")]
    for key, value in environment.items():
        command += ["--env", f"{key}={value}"]
    command += ["--entrypoint", "python3", image_id, *entry]
    return command


def run_phase(label: str, command: list, log: Path, timeout_s: int) -> dict:
    started = time.time()
    with log.open("wb") as stream:
        result = subprocess.run(["timeout", "--signal=INT", "--kill-after=60", str(timeout_s), *command],
                                stdout=stream, stderr=subprocess.STDOUT)
    seconds = time.time() - started
    print(json.dumps({"phase": label, "returncode": result.returncode,
                      "seconds": round(seconds, 1), "log": str(log)}), flush=True)
    return {"phase": label, "returncode": result.returncode, "seconds": round(seconds, 3),
            "command": command, "log": str(log)}


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
    parser.add_argument("--preflight-only", action="store_true",
                        help="JIT proof plus observer load smoke in the pinned image; no engine, no capture")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--census-timeout-s", type=int, default=900)
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True)                       # fresh per run; refuses a reused directory
    args.jit_dir.mkdir(parents=True, exist_ok=True)
    configuration_sha256 = digest(args.serving_config)
    config = json.loads(args.serving_config.read_text())
    base = config["runtime_image"]
    declared_tree = config["runtime_identity"]["plugin_source_tree"]
    if str(args.tessera_tree.resolve()) != declared_tree:
        raise SystemExit(f"configuration names plugin source tree {declared_tree}, launcher was given "
                         f"{args.tessera_tree.resolve()}")
    if str(args.artifact.resolve()) != str(Path(config["artifact"]["path"]).resolve()):
        raise SystemExit("configuration names a different artifact than the launcher was given")

    headroom = memory_pressure()
    require_headroom(headroom)
    before = gpu_sample()
    (out / "host-preconditions.json").write_text(json.dumps(
        {"schema": "tessera.step4_launch_preconditions.v1", "hostname": os.uname().nodename,
         "memory": headroom, "gpu": before,
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
        "TESSERA_EXT_DIR": "/jit-ext",
        # The serve's own dispatch histogram: the dispatch leg of the proof.
        "TESSERA_ROUTE_TRACE": "/out/route-trace.json",
        **config["environment"], **resolved["environment"]}
    for name in ("home", "xdg", "tmp", "triton", "torch-extensions"):
        (args.jit_dir / name).mkdir(parents=True, exist_ok=True)
    # The negative control gets its OWN empty build root, mounted read-only: it
    # must never share a directory with a run that already built the library,
    # or a cache hit would hide the refusal it exists to show.
    ext_host = args.jit_dir / ("tessera-ext-readonly" if args.jit_readonly else "tessera-ext")
    ext_host.mkdir(parents=True, exist_ok=True)

    # Every sample also lists the compute processes the driver sees on the
    # device (host pids): the timing observation's device-exclusivity witness
    # reads this log over the arms' spans, at this cadence, and says so.
    vitals = subprocess.Popen(
        ["bash", "-c",
         'while true; do printf "%s MemAvailable_kB=%s gpu_W=%s gpu_used_MiB=%s compute_apps=%s\\n" '
         '"$(date -u +%FT%TZ)" "$(awk \'/MemAvailable/{print $2}\' /proc/meminfo)" '
         '"$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)" '
         '"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" '
         '"$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits '
         '| tr -d " " | paste -sd";" -)"; sleep 5; done'],
        stdout=(out / "host-vitals.log").open("wb"), stderr=subprocess.STDOUT)
    phases = []
    try:
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
                ext_host=ext_host, ext_readonly=args.jit_readonly,
                entry=["/tessera/experiments/full_engine_plugin_install.py",
                       "--evidence-dir", "/out", "--base-reference", base,
                       "--launcher-image-id", image_id,
                       "--launcher-image-inspect", "/out/launcher-image-inspect.json",
                       "--source-tree", "/tessera", "--source-commit", args.source_commit,
                       "--core-manifest", str(args.core_manifest), "--",
                       "python3", "-u", "/tessera/tools/tessera_construction_census.py",
                       str(args.artifact), "/out/census.json", "--runtime-image", base]),
                census_out / "census.log", args.census_timeout_s))

        capture_out = out / "capture"
        mode = args.observation_mode
        capture_argv = ["--config", str(args.serving_config), "--model", str(args.artifact),
                        "--core-manifest", str(args.core_manifest),
                        "--runtime-evidence", "/out/per-job-runtime.json",
                        "--output", "/out/capture", "--artifact", "--all-units",
                        "--observation-mode", mode]
        if mode != "kv":
            # The read-only pass loads no collector: a stock engine and a stock
            # worker are what make it the read-only half of the two-pass pair.
            capture_argv += ["--collector", str(args.collector)]
        if mode == "resources" and args.workspaces is not None:
            capture_argv += ["--workspaces", str(args.workspaces)]
        if mode == "timings":
            capture_argv += ["--timing-samples", str(args.timing_samples)]
        if args.calibration is not None:
            capture_argv += ["--calibration", str(args.calibration),
                             "--calibration-sha256", digest(args.calibration)]
        driver_argv = ["--out", "/out", "--capture-output", "/out/capture",
                       "--route-trace", "/out/route-trace.json",
                       "--expected-fp4-modules", str(fp4_module_count(args.artifact)),
                       "--observation-mode", mode]
        if args.preflight_only:
            driver_argv.append("--preflight-only")
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
            ext_host=ext_host, ext_readonly=args.jit_readonly, entry=entry),
            out / "container.log", args.timeout_s))
    finally:
        vitals.terminate()
        vitals.wait(timeout=30)

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
              "snapshot, no timing claim, and no route proof (a stock worker carries no census)",
        "timings": "profiled all-native-unit event partition of a served artifact; observer "
                   "qualification only, no admitted timing or fixed-resource price"}
    summary = {"schema": "tessera.step4_capture_launch.v1", "out": str(out),
               "hostname": os.uname().nodename, "image": base, "image_id": image_id,
               "configuration_sha256": configuration_sha256, "artifact": str(args.artifact),
               "source_commit": args.source_commit, "jit_dir": str(args.jit_dir),
               "jit_readonly": args.jit_readonly, "ext_dir": str(ext_host), "phases": phases,
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
