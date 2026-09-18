"""In-container driver: prove the fp4 JIT, then capture, then qualify the route.

``full_engine_plugin_install.py`` execs this after it has installed the frozen
plugin and dropped to the runtime uid, so everything here runs in the image,
plugin installation and GPU the capture itself runs in.  Three phases, in this
order, each refusing the whole run rather than continuing:

1. **preflight** -- ``require_tessera_ext`` with no fallback available to it.
   ``get_tessera_ext`` is a soft probe that returns ``None`` when the extension
   cannot build, and in ``resident`` residency the NVFP4 route then decodes
   through ``torch_materialize_stock`` and *serves*.  Calling ``require_`` here,
   before the engine exists, converts that silence into an exit code.  The
   built library's path and sha256 are recorded; the capture's own evidence has
   to show the same bytes mapped.
2. **capture** -- ``capture_full_engine_resources`` from the frozen tree, on the
   artifact roster (``--artifact --all-units``).  The census CLI is mutually
   exclusive with ``--artifact`` (``capture_full_engine_resources.main``), so no
   census is passed and none is consumed.
3. **qualification** -- ``step4_route_qualification.qualify_native_route`` over
   the worker's own ``runtime-observation.json`` and the serve's own
   ``TESSERA_ROUTE_TRACE`` histogram.  A capture that cannot be qualified is
   kept (it is evidence of the refusal) and reported as refused.

Building the extension in phase 1 also means the measured run does not pay for
a compile it would otherwise charge to the engine's startup ledger.

``--observation-mode`` (tessera#399) names which capture pass phase 2 runs.
The ``kv`` pass is a stock engine with a stock worker, so phase 3 has no
worker census to bind the library leg to: the dispatch leg is recorded from
the route trace and the record says ``qualified: false`` with that reason.
``--preflight-only`` runs phase 1 and an observer load smoke
(``observer_preflight``) in a child interpreter, writes
``observer-preflight.json`` and stops, so an image or a tree that cannot host
the observer is refused before an engine is started on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4_route_qualification import (  # noqa: E402
    QualificationRefused, qualify_native_route, refusal_record)

#: ``scheme.NVFP4_ACTIVATION_CONTRACT``; spelled as data because the
#: qualification module may not import the serving package.
NVFP4_ACTIVATION_CONTRACT = "e2m1_group16_ue4m3_static"


def digest(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def refuse(out: Path, phase: str, message: str, **context) -> int:
    record = refusal_record(phase, message, **context)
    write(out / "refusal.json", record)
    print("REFUSED[" + phase + "]: " + message, file=sys.stderr, flush=True)
    return 3


def preflight(out: Path) -> dict:
    """Build/load the NVFP4 decode extension, or raise.

    ``substitutes_when_unavailable("resident")`` is read from the runtime's own
    published table and recorded: it is the fact that makes this phase
    necessary, and a tree that stopped substituting would make the record say
    so instead of the launcher claiming it.
    """
    from tessera.serving.ext import (
        NVFP4_MODULE_PREFIX, require_tessera_ext, substitutes_when_unavailable, toolchain_report)
    substitutes = substitutes_when_unavailable("resident")
    module = require_tessera_ext("the step-4 capture's fp4 decode preflight")
    library = Path(module.__file__).resolve()
    record = {"schema": "tessera.step4_jit_preflight.v1",
              "module_name_prefix": NVFP4_MODULE_PREFIX,
              "library": str(library), "library_sha256": digest(library),
              "library_bytes": library.stat().st_size,
              "jit_identity": getattr(module, "__tessera_jit_identity__", None),
              "jit_platform": getattr(module, "__tessera_jit_platform__", None),
              "jit_abi_schema": getattr(module, "__tessera_jit_abi_schema__", None),
              "ext_dir": os.environ.get("TESSERA_EXT_DIR"),
              "resident_substitutes_without_it": substitutes,
              "toolchain": toolchain_report(),
              "scope": "the extension built and loaded in this container before the engine started; "
                       "it is the library the capture's mapped-library census must show"}
    write(out / "jit-preflight.json", record)
    return record


OBSERVER_SMOKE = r"""
import ctypes, hashlib, importlib, inspect, json, sys
from pathlib import Path
collector, workspaces, out = sys.argv[1:4]
def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
record = {"schema": "tessera.step4_observer_preflight.v1", "libraries": {}, "modules": {},
          "runner_hooks": {}, "python": sys.version, "sys_path": list(sys.path)}
symbols = {"collector": ("tessera_memory_start", "tessera_memory_mark", "tessera_memory_stop"),
           "workspaces": ("tessera_blas_workspace_snapshot",)}
for name, path in (("collector", collector), ("workspaces", workspaces)):
    if path == "-":
        continue
    handle = ctypes.CDLL(path)
    for symbol in symbols[name]:
        getattr(handle, symbol)
    record["libraries"][name] = {"path": path, "sha256": digest(path), "loaded": True,
                                 "symbols": list(symbols[name])}
for module in ("experiments.full_engine_worker", "experiments.full_engine_timing_worker",
               "experiments.capture_full_engine_resources", "experiments.report_full_engine_resources"):
    loaded = importlib.import_module(module)
    record["modules"][module] = {"file": loaded.__file__, "sha256": digest(loaded.__file__)}
import functools
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
main_stream = inspect.getattr_static(GPUModelRunner, "main_stream", None)
# A stream-valued attribute, however the stock runner spells it: vLLM 0.28.0
# makes it a functools.cached_property (gpu/model_runner.py), an older tree a
# property; a plain function would hand the worker a bound method instead of a
# stream, and that is the case refused here.
kind = ("cached_property" if isinstance(main_stream, functools.cached_property)
        else "property" if isinstance(main_stream, property)
        else None if main_stream is None else type(main_stream).__name__)
record["runner_hooks"] = {
    "main_stream_kind": kind,
    "main_stream_is_attribute": kind in ("cached_property", "property"),
    "output_copy_stream_in_init": "self.output_copy_stream = " in inspect.getsource(GPUModelRunner.__init__),
    "scope": "the two runner attributes TimingCaptureWorker.execute_model binds; located in source, "
             "not exercised -- the timing capture's stream-coverage check is what exercises them"}
Path(out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
if not (record["runner_hooks"]["main_stream_is_attribute"] and record["runner_hooks"]["output_copy_stream_in_init"]):
    raise SystemExit("stock runner lacks a stream hook the timing worker binds: " + json.dumps(record["runner_hooks"]))
print(json.dumps({"phase": "observer_preflight", "libraries": sorted(record["libraries"]),
                  "modules": sorted(record["modules"])}), flush=True)
"""


def observer_preflight(out: Path, capture_argv: list) -> dict:
    """Load the observer's libraries and modules in a child interpreter, or refuse.

    A child, not this process: the driver stays a process that never loaded a
    CUPTI collector, so the capture it launches inherits nothing from the
    smoke.  Nothing here starts CUDA or an engine.
    """
    def option(name):
        if name in capture_argv:
            return capture_argv[capture_argv.index(name) + 1]
        return "-"
    path = out / "observer-preflight.json"
    result = subprocess.run([sys.executable, "-c", OBSERVER_SMOKE, option("--collector"),
                             option("--workspaces"), str(path)])
    if result.returncode != 0 or not path.exists():
        raise RuntimeError(f"observer load smoke exited {result.returncode}")
    return json.loads(path.read_text())


def record_dispatch_leg_only(out: Path, trace_path: Path, expected_modules, reason: str) -> dict:
    """The kv pass: the dispatch leg from the trace, no library leg, not qualified."""
    from step4_route_qualification import trace_decoders_by_contract
    if not trace_path.exists():
        raise QualificationRefused(f"no route trace at {trace_path}; the dispatch leg is not verified")
    decoders = trace_decoders_by_contract(json.loads(trace_path.read_text()), NVFP4_ACTIVATION_CONTRACT)
    record = {"schema": "tessera.step4_native_route_qualification.v1",
              "activation_contract": NVFP4_ACTIVATION_CONTRACT, "library": None,
              "decoders": decoders, "expected_modules": expected_modules,
              "route_trace": str(trace_path), "route_trace_sha256": digest(trace_path),
              "scope": "dispatch leg only: " + reason, "qualified": False}
    write(out / "native-route-qualification.json", record)
    return record


def qualify(out: Path, capture_dir: Path, trace_path: Path, preflight_record: dict,
            expected_modules) -> dict:
    observations = sorted(capture_dir.glob("worker-*/runtime-observation.json"))
    if len(observations) != 1:
        raise QualificationRefused(
            f"expected exactly one worker runtime observation under {capture_dir}, found "
            f"{[str(p) for p in observations]}")
    if not trace_path.exists():
        raise QualificationRefused(f"no route trace at {trace_path}; the dispatch leg is not verified")
    record = qualify_native_route(
        json.loads(observations[0].read_text()), json.loads(trace_path.read_text()),
        expected_library_sha256=preflight_record["library_sha256"],
        activation_contract=NVFP4_ACTIVATION_CONTRACT, expected_modules=expected_modules)
    record["runtime_observation"] = str(observations[0])
    record["route_trace"] = str(trace_path)
    record["route_trace_sha256"] = digest(trace_path)
    record["preflight_library"] = preflight_record["library"]
    write(out / "native-route-qualification.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="the container-visible ledger directory")
    parser.add_argument("--capture-output", type=Path, required=True)
    parser.add_argument("--route-trace", type=Path, required=True)
    parser.add_argument("--expected-fp4-modules", type=int, default=None)
    parser.add_argument("--observation-mode", choices=("resources", "kv", "timings"), default="resources")
    parser.add_argument("--preflight-only", action="store_true",
                        help="JIT proof and observer load smoke, then stop; no engine, no capture")
    parser.add_argument("capture_argv", nargs=argparse.REMAINDER,
                        help="-- followed by the capture CLI arguments")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    try:
        record = preflight(out)
    except Exception as exc:  # noqa: BLE001 -- every failure here refuses the run
        return refuse(out, "jit_preflight", f"{type(exc).__name__}: {exc}",
                      ext_dir=os.environ.get("TESSERA_EXT_DIR"))
    print(json.dumps({"phase": "jit_preflight", "library": record["library"],
                      "sha256": record["library_sha256"],
                      "resident_substitutes_without_it": record["resident_substitutes_without_it"]}),
          flush=True)

    capture_argv = args.capture_argv
    if capture_argv and capture_argv[0] == "--":
        capture_argv = capture_argv[1:]
    if args.preflight_only:
        try:
            smoke = observer_preflight(out, capture_argv)
        except Exception as exc:  # noqa: BLE001
            return refuse(out, "observer_preflight", f"{type(exc).__name__}: {exc}")
        write(out / "capture-result.json", {"returncode": None, "seconds": round(time.time() - started, 3),
                                            "preflight_only": True, "observer_preflight": smoke["schema"]})
        return 0
    command = [sys.executable, "-u", "-m", "experiments.capture_full_engine_resources", *capture_argv]
    write(out / "capture-command.json", {"command": command, "cwd": os.getcwd()})
    result = subprocess.run(command)
    capture_seconds = time.time() - started
    write(out / "capture-result.json", {"returncode": result.returncode,
                                        "seconds": round(capture_seconds, 3)})
    if result.returncode != 0:
        return refuse(out, "capture", f"capture exited {result.returncode}",
                      capture_seconds=round(capture_seconds, 3))
    try:
        if args.observation_mode == "kv":
            qualified = record_dispatch_leg_only(
                out, args.route_trace, args.expected_fp4_modules,
                "the read-only KV pass runs a stock worker that writes no runtime observation, "
                "so no mapped decode library can be bound to the preflight's bytes")
        else:
            qualified = qualify(out, args.capture_output, args.route_trace, record,
                                args.expected_fp4_modules)
    except QualificationRefused as exc:
        return refuse(out, "native_route_qualification", str(exc),
                      capture_seconds=round(capture_seconds, 3))
    except Exception as exc:  # noqa: BLE001
        return refuse(out, "native_route_qualification", f"{type(exc).__name__}: {exc}")
    print(json.dumps({"phase": "native_route_qualification", **{
        k: qualified[k] for k in ("activation_contract", "decoders", "qualified")}}), flush=True)
    if args.observation_mode == "timings":
        # The same-run timing observation, derived from every arm the pass
        # wrote, beside the ledger files: the resource report joins it by run
        # identity. A partition that could not be established is exit 3 from
        # the builder and a refusal here, with the builder's reason on record.
        builder = [sys.executable, "-m", "experiments.full_engine_timing_observation",
                   "--capture-dir", str(args.capture_output), "--launch-dir", str(out),
                   "--output", str(out / "timing-observation.json")]
        write(out / "timing-observation-command.json", {"command": builder, "cwd": os.getcwd()})
        built = subprocess.run(builder)
        if built.returncode != 0:
            return refuse(out, "timing_observation",
                          f"timing observation builder exited {built.returncode}; see timing-observation.json",
                          capture_seconds=round(capture_seconds, 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
