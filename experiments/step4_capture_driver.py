"""In-container driver: prove the native lanes, then capture, then qualify the routes.

``full_engine_plugin_install.py`` execs this after it has installed the frozen
plugin and dropped to the runtime uid, so everything here runs in the image,
plugin installation and GPU the capture itself runs in.  Three phases, in this
order, each refusing the whole run rather than continuing:

1. **native preflight** -- a child interpreter (``NATIVE_SMOKE``) reads the
   runtime's own tables and proves the native code the artifact's families
   need is buildable here, before an engine exists: ``scheme.route_launches``
   for every family must name no extension lane (a lane would reintroduce a
   ``when_unavailable`` substitute the qualifier does not model, so one is a
   refusal), ``tessera.window_gemm`` must import (it is Triton; the dense FP8
   and BF16 routes multiply through it and their ``apply`` raises rather than
   falling back), and for an NVFP4 family ``kernel_a4.require_native_fp4_mma``
   must pass (the fused span-2 GEMM refuses a Triton that cannot lower
   block-scaled FP4 MMA rather than emulating it).  The Triton cache's state
   is recorded so a report can tell a warm run from a cold one: on this tree
   the kernels compile at first forward INSIDE the engine, so a cold cache
   charges compilation to the measured ledger.  A child, not this process, so
   the driver never initialises CUDA.
2. **capture** -- ``capture_full_engine_resources`` from the frozen tree, on the
   artifact roster (``--artifact --all-units``).  The census CLI is mutually
   exclusive with ``--artifact`` (``capture_full_engine_resources.main``), so no
   census is passed and none is consumed.
3. **qualification** -- ``step4_route_qualification.qualify_native_route`` over
   the serve's own ``TESSERA_ROUTE_TRACE`` histogram, per family: every
   dispatch on each family's activation contract must be the one native
   ``(symbol, decoder)`` its dense route stamps, and the module count (and,
   where the trace names them, the module names) must be the manifest's.  The
   worker's ``runtime-observation.json`` mapped-library census is recorded,
   not required: no dense launch on this tree loads a ``cpp_extension``.  A
   capture that cannot be qualified is kept (it is evidence of the refusal)
   and reported as refused.

WHAT RETIRED HERE.  The pre-``37e89f576`` driver built the ``tessera_nvfp4``
``cpp_extension`` in phase 1 and bound the worker's mapped ``.so`` to those
bytes, because ``ext.NATIVE_EXTENSIONS`` then published a silent
``torch_materialize_stock`` substitute for a container that could not
compile.  That extension, its loader and the substitute are gone; the dense
routes now refuse loudly (ImportError at load, a Triton compile error at first
launch, ``GrammarError`` from the FP4 MMA gate), so the hazard the preflight
closed has moved from "silently substitutes" to "the capture phase fails",
and the preflight's job is to make that failure cost minutes, not a capture.

``--observation-mode`` (tessera#399) names which capture pass phase 2 runs.
The ``kv`` pass is a stock engine with a stock worker, so it writes no runtime
observation; its qualification is the dispatch leg alone, which on this tree
is the whole proof, and the record says the library census is absent.
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
    DENSE_LAUNCHES, QUALIFICATION_SCHEMA, QualificationRefused, qualify_dispatch,
    qualify_native_route, refusal_record)

#: ``lane.TESSERA_MODE_ENV``: the residency the configuration binds into the
#: container; the driver reads the same variable the plugin latches.
SERVE_MODE_ENV = "TESSERA_SERVE_MODE"


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


NATIVE_SMOKE = r"""
import hashlib, json, os, sys, traceback
from pathlib import Path
out, mode, expected_json = sys.argv[1:4]
expected = json.loads(expected_json)
families = sorted(f for f, v in expected.items()
                  if (v["count"] if isinstance(v, dict) else v) > 0)
record = {"schema": "tessera.step4_native_preflight.v1", "mode": mode, "families": families,
          "python": sys.version, "refusal": None}
def finish(code):
    Path(out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    raise SystemExit(code)
try:
    from tessera.serving import ext, scheme
    # 1. The published extension table, and whether any dense launch of the
    #    artifact's families names one of its lanes.  A lane on the dispatch
    #    is a when_unavailable substitute the qualifier does not model.
    record["native_extensions"] = [
        {"module_name_prefix": e["module_name_prefix"], "filename_glob": e["filename_glob"],
         "routes": list(e["routes"]),
         "substitutes_when_unavailable": ext.substitutes_when_unavailable(mode, e["module_name_prefix"])}
        for e in ext.NATIVE_EXTENSIONS]
    launches = {}
    for family in families:
        launches[family] = [
            {"symbol": l["symbol"], "decoder": l["decoder"], "lane": l["lane"],
             "when_lane_absent": bool(l["when_lane_absent"])}
            for l in scheme.route_launches(family, structure=scheme.STRUCTURE_DENSE, mode=mode,
                                           include_experimental=True)]
    record["dense_launches"] = launches
    named = sorted({l["lane"] for ls in launches.values() for l in ls if l["lane"] is not None})
    if named:
        record["refusal"] = f"dense launches name extension lane(s) {named}; this preflight has no proof for a lane"
        finish(4)
    # 2. The window GEMM is Triton: import it (fp8_route/bf16_route reach it
    #    through serving.native_window at process_weights_after_loading).
    import triton
    import tessera.window_gemm  # noqa: F401
    record["triton"] = {"module": triton.__name__, "version": getattr(triton, "__version__", None),
                        "file": getattr(triton, "__file__", None)}
    # 3. The A4 lane: its Triton build and, when the artifact carries an NVFP4
    #    family, the block-scaled FP4 MMA gate the dense GEMM runs on every call.
    from tessera import kernel_a4
    record["fp4_backend"] = kernel_a4.native_fp4_backend()
    if "TESSERA_NVFP4" in families:
        kernel_a4.require_native_fp4_mma("the step-4 capture's native preflight")
        record["fp4_mma_ptx_tokens"] = kernel_a4.native_fp4_mma_ptx_tokens()
    # 4. Triton cache state, so the report can tell a warm run from a cold one.
    cache = os.environ.get("TRITON_CACHE_DIR")
    listing = (sorted(str(p.relative_to(cache)) for p in Path(cache).rglob("*") if p.is_file())
               if cache and Path(cache).is_dir() else [])
    record["triton_cache"] = {"dir": cache, "files": len(listing),
                              "listing_sha256": hashlib.sha256("\n".join(listing).encode()).hexdigest()}
    record["scope"] = ("the native lanes the artifact's families dispatch through import and, for "
                       "NVFP4, pass the FP4 MMA gate in this container before the engine started; the "
                       "window GEMM kernel itself compiles at first forward inside the engine and has "
                       "no standalone probe here -- a compile failure raises in apply and the capture "
                       "phase refuses")
except SystemExit:
    raise
except Exception as exc:  # noqa: BLE001 -- every failure is the refusal
    record["refusal"] = f"{type(exc).__name__}: {exc}"
    record["traceback"] = traceback.format_exc()[-4000:]
    finish(4)
print(json.dumps({"phase": "native_preflight", "families": families,
                  "fp4_backend": record.get("fp4_backend"),
                  "triton_cache_files": record["triton_cache"]["files"]}), flush=True)
finish(0)
"""


def native_preflight(out: Path, mode: str, expected_modules) -> dict:
    """Prove the native lanes in a child interpreter, or raise.

    ``expected_modules`` is the launcher's per-family module map; a family
    with no module is not proved.  The child writes ``native-preflight.json``
    whether it passes or refuses, so a refusal carries the tables it read.
    """
    path = out / "native-preflight.json"
    result = subprocess.run([sys.executable, "-c", NATIVE_SMOKE, str(path), mode,
                             json.dumps(expected_modules, sort_keys=True)])
    record = json.loads(path.read_text()) if path.exists() else None
    if result.returncode != 0 or record is None or record.get("refusal") is not None:
        reason = (record or {}).get("refusal") or f"native preflight exited {result.returncode}"
        raise RuntimeError(reason)
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


def _read_trace(trace_path: Path) -> dict:
    if not trace_path.exists():
        raise QualificationRefused(f"no route trace at {trace_path}; the dispatch leg is not verified")
    return json.loads(trace_path.read_text())


def record_dispatch_leg_only(out: Path, trace_path: Path, mode: str, expected_modules, reason: str) -> dict:
    """The kv pass: the dispatch leg from the trace; no worker census to record."""
    trace = _read_trace(trace_path)
    families = qualify_dispatch(trace, mode=mode, expected_modules=expected_modules)
    record = {"schema": QUALIFICATION_SCHEMA, "mode": mode, "families": families,
              "mapped_extension_libraries": None,
              "trace_identity": {key: trace.get(key) for key in
                                 ("schema", "identity_version", "rank", "world_size", "rank_source",
                                  "rank_conflict", "platform", "pid")},
              "route_trace": str(trace_path), "route_trace_sha256": digest(trace_path),
              "scope": "dispatch leg over every family the artifact carries; no library census: " + reason,
              "qualified": True}
    write(out / "native-route-qualification.json", record)
    return record


def qualify(out: Path, capture_dir: Path, trace_path: Path, preflight_record: dict,
            mode: str, expected_modules) -> dict:
    observations = sorted(capture_dir.glob("worker-*/runtime-observation.json"))
    if len(observations) != 1:
        raise QualificationRefused(
            f"expected exactly one worker runtime observation under {capture_dir}, found "
            f"{[str(p) for p in observations]}")
    trace = _read_trace(trace_path)
    record = qualify_native_route(json.loads(observations[0].read_text()), trace,
                                  mode=mode, expected_modules=expected_modules)
    record["runtime_observation"] = str(observations[0])
    record["route_trace"] = str(trace_path)
    record["route_trace_sha256"] = digest(trace_path)
    record["native_preflight"] = {"schema": preflight_record["schema"],
                                  "fp4_backend": preflight_record.get("fp4_backend"),
                                  "triton_cache": preflight_record.get("triton_cache")}
    write(out / "native-route-qualification.json", record)
    return record


def _expected_modules(argument: str) -> dict:
    """``--expected-modules``: the launcher's per-family module map, as JSON."""
    try:
        value = json.loads(argument)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--expected-modules is not JSON: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise argparse.ArgumentTypeError("--expected-modules must be a non-empty JSON object")
    unknown = sorted(set(value) - set(DENSE_LAUNCHES))
    if unknown:
        raise argparse.ArgumentTypeError(f"--expected-modules names unknown families {unknown}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="the container-visible ledger directory")
    parser.add_argument("--capture-output", type=Path, required=True)
    parser.add_argument("--route-trace", type=Path, required=True)
    parser.add_argument("--expected-modules", type=_expected_modules, required=True,
                        help='JSON {family: {"count": n, "names": [...]}} from the artifact manifest')
    parser.add_argument("--serve-mode", choices=("resident", "streamed"),
                        default=os.environ.get(SERVE_MODE_ENV),
                        help=f"the residency; defaults to ${SERVE_MODE_ENV} as the configuration bound it")
    parser.add_argument("--observation-mode", choices=("resources", "kv", "timings"), default="resources")
    parser.add_argument("--preflight-only", action="store_true",
                        help="native proof and observer load smoke, then stop; no engine, no capture")
    parser.add_argument("capture_argv", nargs=argparse.REMAINDER,
                        help="-- followed by the capture CLI arguments")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.serve_mode is None:
        return refuse(out, "native_preflight",
                      f"no residency: --serve-mode not given and ${SERVE_MODE_ENV} is unset")

    started = time.time()
    try:
        record = native_preflight(out, args.serve_mode, args.expected_modules)
    except Exception as exc:  # noqa: BLE001 -- every failure here refuses the run
        return refuse(out, "native_preflight", f"{type(exc).__name__}: {exc}",
                      triton_cache_dir=os.environ.get("TRITON_CACHE_DIR"))
    print(json.dumps({"phase": "native_preflight", "families": record["families"],
                      "fp4_backend": record.get("fp4_backend"),
                      "dense_launches": record["dense_launches"]}), flush=True)

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
                out, args.route_trace, args.serve_mode, args.expected_modules,
                "the read-only KV pass runs a stock worker that writes no runtime observation")
        else:
            qualified = qualify(out, args.capture_output, args.route_trace, record,
                                args.serve_mode, args.expected_modules)
    except QualificationRefused as exc:
        return refuse(out, "native_route_qualification", str(exc),
                      capture_seconds=round(capture_seconds, 3))
    except Exception as exc:  # noqa: BLE001
        return refuse(out, "native_route_qualification", f"{type(exc).__name__}: {exc}")
    print(json.dumps({"phase": "native_route_qualification", "qualified": qualified["qualified"],
                      "families": {family: {"launches": value["observed"]["launches"],
                                            "modules": value["observed"]["modules"]}
                                   for family, value in qualified["families"].items()}}), flush=True)
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
