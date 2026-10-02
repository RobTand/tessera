"""PB compile gate for the piece-major R4 reader (tessera#739).

Builds the E4M3 fused routed window extension with NO CUDA device present
(``CUDA_VISIBLE_DEVICES=''``): the point is to exercise nvcc/ptxas on the new
PM instantiation and report real compile diagnostics and ELF identity, not to
run anything.  The host device query is disabled, so the platform token is
given explicitly through ``TESSERA_PLATFORM_TOKEN`` (the serving backend reads
it); a build for an absent device is a compile gate, never a serving path.

Prints, as JSON on the last line:
  * ok: whether the build produced a loadable module
  * source_sha256: the compiled ``routed_fused_window.cu`` bytes
  * library_sha256 / library_path: the built ``.so``
  * flags: the exact ``_cflags`` argv (minus the token-derived -gencode)
  * env: the environment that pinned the build
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src", "tessera", "serving", "csrc", "routed_fused_window.cu")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    sys.path.insert(0, os.path.join(REPO, "src"))
    report = {"env": {k: os.environ.get(k) for k in
                      ("TESSERA_PLATFORM_TOKEN", "TORCH_EXTENSIONS_DIR", "MAX_JOBS",
                       "CUDA_VISIBLE_DEVICES")}}
    report["source_sha256"] = _sha256(SRC)
    try:
        from tessera.routed_fused import _cflags
        import torch
        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        token = os.environ.get("TESSERA_PLATFORM_TOKEN", "sm_121")
        # the E4M3 library the lane builds is the MMA one (default),
        # i.e. _cflags(token, fp8=True, mma8=True)
        report["flags"] = _cflags(token, True, True)
        from torch.utils.cpp_extension import load
        build = os.environ.get("TORCH_EXTENSIONS_DIR", os.path.expanduser("~/tmp/torch-ext-piece-major"))
        build = os.path.join(build, f"tessera_routed_fused_mma_e4m3_{token}")
        os.makedirs(build, exist_ok=True)
        report["build_directory"] = build
        lib = load(
            name="tessera_routed_fused_mma_e4m3",
            sources=[SRC],
            build_directory=build,
            extra_cuda_cflags=report["flags"],
            verbose=True,
        )
        report["ok"] = True
        report["module"] = getattr(lib, "__file__", None)
        # the returned module's backing .so
        so = None
        if getattr(lib, "__file__", None) and os.path.exists(lib.__file__):
            so = lib.__file__
        else:
            cands = glob.glob(os.path.join(build, "*.so"))
            so = cands[0] if cands else None
        if so and os.path.exists(so):
            report["library_path"] = so
            report["library_sha256"] = _sha256(so)
        # report the timing/capability the build pins, for the record
        for name in ("BM", "BN", "HALF", "BK", "BDESC_INTS", "WINDOW_BITS",
                     "FAMILY_FP8", "FAMILY_MMA8", "WORD_STAGES", "WORD_STAGES_MIN",
                     "ROUTED_RATE_MAX", "RATE_MIN"):
            try:
                report[f"module_{name}"] = getattr(lib, name)
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001 -- the compile result IS the report
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-4000:]
    print(json.dumps(report, sort_keys=True, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
