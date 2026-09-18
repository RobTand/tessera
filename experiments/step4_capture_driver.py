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
        qualified = qualify(out, args.capture_output, args.route_trace, record,
                            args.expected_fp4_modules)
    except QualificationRefused as exc:
        return refuse(out, "native_route_qualification", str(exc),
                      capture_seconds=round(capture_seconds, 3))
    except Exception as exc:  # noqa: BLE001
        return refuse(out, "native_route_qualification", f"{type(exc).__name__}: {exc}")
    print(json.dumps({"phase": "native_route_qualification", **{
        k: qualified[k] for k in ("activation_contract", "decoders", "qualified")}}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
