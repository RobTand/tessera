#!/usr/bin/env python3
"""KDA fusion build gate: Tessera's FlashKDA build against the image's own, inside the image.

Runs inside the serving image (``build_gate.sh`` starts the container).  Builds
FlashKDA's own ``fwd_launch.cu`` through ``tessera.flashkda.build_stock`` -- the
build path every fused variant uses -- extracts the cubin, and compares it with
the cubin of the image's ``_flashkda_C`` extension, extracted in the same run:

* every per-kernel ``.text.<mangled>`` section byte for byte (``cubin_cmp.py``,
  independent of any disassembler);
* ``cuobjdump -res-usage`` and ``cuobjdump -sass``, one tool for both sides.

PASS needs all three, the same kernel set, the vendored sources at FlashKDA
17a037d9 and the CUTLASS directory at 5c149f52 (every file against its
``SHA256SUMS``).  CPU only; no GPU is touched.  Writes ``build_gate.json``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tessera import flashkda  # noqa: E402

#: The image's FlashKDA library and its cubin as step 1 recorded them.
IMAGE_SO_SHA256 = "23cd6ed0"
IMAGE_CUBIN_SHA256 = "1344477a"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, **kw)


def first_line(cmd: list[str]) -> str:
    try:
        r = sh(cmd)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    text = (r.stdout or r.stderr).strip().splitlines()
    return f"rc={r.returncode}: " + (" | ".join(text[-2:]) if text else "")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_cutlass(root: Path) -> dict:
    """Every file the SHA256SUMS manifest lists, re-hashed."""
    listed = bad = 0
    mismatches = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        want, rel = line.split(None, 1)
        listed += 1
        got = sha256(root / rel.strip())
        if got != want:
            bad += 1
            mismatches.append(rel.strip())
    return {"root": str(root), "commit": (root / "COMMIT").read_text().strip(),
            "manifest_sha256": sha256(root / "SHA256SUMS"), "files": listed,
            "mismatched": bad, "mismatches": mismatches[:20]}


def extract_cubins(so: Path, into: Path, cuobjdump: str) -> list[Path]:
    into.mkdir(parents=True, exist_ok=True)
    r = sh([cuobjdump, "-xelf", "all", str(so)], cwd=into)
    if r.returncode != 0:
        raise SystemExit(f"cuobjdump -xelf {so} failed: {r.stderr.strip()}")
    return sorted(Path(p) for p in glob.glob(str(into / "*.cubin")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "host": os.environ.get("HOST_NAME", platform.node()),
                 "machine": platform.machine(), "python": sys.version.split()[0],
                 "pb_action_key": os.environ.get("PB_ACTION_KEY", ""),
                 "tessera_head": os.environ.get("TESSERA_HEAD", ""),
                 "tessera_state": os.environ.get("TESSERA_STATE", ""),
                 "image": os.environ.get("ORACLE_IMAGE", "")}
    reasons: list[str] = []

    import torch

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = shutil.which("nvcc") or os.path.join(cuda_home, "bin", "nvcc")
    cuobjdump = shutil.which("cuobjdump") or os.path.join(cuda_home, "bin", "cuobjdump")
    rec["toolchain"] = {
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "nvcc": nvcc, "nvcc_version": first_line([nvcc, "--version"]),
        "cxx_version": first_line(["c++", "--version"]).split(" | ")[0],
        "gxx_version": first_line(["g++", "--version"]).split(" | ")[0],
        "cuobjdump": cuobjdump, "cuobjdump_version": first_line([cuobjdump, "--version"]),
        "ninja": first_line(["ninja", "--version"]),
    }
    rec["flags"] = {"extra_cuda_cflags": flashkda.cuda_cflags()}

    # Inputs: vendored sources and CUTLASS, both against their pins.
    try:
        flashkda.verify_vendored()
        rec["vendored"] = {"dir": str(flashkda.vendored_dir()), "flashkda": flashkda.FLASHKDA_COMMIT,
                           "sha256": flashkda.VENDORED_SHA256, "verified": True}
    except flashkda.FlashKdaBuildError as exc:
        rec["vendored"] = {"verified": False, "error": str(exc)}
        reasons.append("vendored sources differ")
    cutlass = flashkda.cutlass_root()
    rec["cutlass"] = check_cutlass(cutlass)
    if rec["cutlass"]["mismatched"] or rec["cutlass"]["commit"] != flashkda.CUTLASS_COMMIT:
        reasons.append("CUTLASS directory is not 5c149f52's bytes")
    rec["include_dirs"] = flashkda.include_dirs(cutlass)

    # The image's library: located without importing vLLM.
    spec = importlib.util.find_spec("vllm")
    vllm_dir = Path(spec.origin).parent if spec and spec.origin else None
    image_sos = sorted(vllm_dir.glob("_flashkda_C*.so")) if vllm_dir else []
    if len(image_sos) != 1:
        rec["image_so"] = {"found": [str(p) for p in image_sos]}
        reasons.append("the image's _flashkda_C library is not exactly one file")
        image_cubin = None
    else:
        so = image_sos[0]
        cubins = extract_cubins(so, out / "image-cubin", cuobjdump)
        rec["image_so"] = {"path": str(so), "sha256": sha256(so),
                           "cubins": {c.name: sha256(c) for c in cubins}}
        if not rec["image_so"]["sha256"].startswith(IMAGE_SO_SHA256):
            reasons.append("the image's _flashkda_C is not the library step 1 read")
        image_cubin = cubins[0] if len(cubins) == 1 else None
        if image_cubin is None:
            reasons.append(f"the image's _flashkda_C holds {len(cubins)} cubins, not 1")
        elif not sha256(image_cubin).startswith(IMAGE_CUBIN_SHA256):
            reasons.append("the image's cubin is not the one step 1 compared against")

    # Tessera's build of stock FlashKDA.
    t0 = time.time()
    try:
        library, build = flashkda.build_stock(verbose=True)
    except Exception as exc:  # noqa: BLE001 - the record says what failed
        rec["build"] = {"ok": False, "seconds": round(time.time() - t0, 1),
                        "error": f"{type(exc).__name__}: {exc}"}
        reasons.append("the build failed")
        library = build = None
    if library:
        rec["build"] = {"ok": True, "seconds": round(time.time() - t0, 1), "library": library,
                        "library_sha256": sha256(Path(library)), "build_directory": build}
        ninja_file = Path(build) / "build.ninja"
        if ninja_file.is_file():
            shutil.copy(ninja_file, out / "build.ninja")
        cmds = sh(["ninja", "-C", build, "-t", "commands"])
        (out / "ninja-commands.txt").write_text(cmds.stdout + cmds.stderr)
        rec["build"]["commands"] = cmds.stdout.strip().splitlines()
        built = extract_cubins(Path(library), out / "built-cubin", cuobjdump)
        rec["build"]["cubins"] = {c.name: sha256(c) for c in built}
        if len(built) != 1:
            reasons.append(f"the built library holds {len(built)} cubins, not 1")
        elif image_cubin is not None:
            mine = built[0]
            cmp_json = out / "cubin_cmp.json"
            r = sh([sys.executable, str(HERE / "cubin_cmp.py"), str(image_cubin), str(mine),
                    "--out", str(cmp_json)])
            if r.returncode != 0:
                reasons.append(f"cubin_cmp failed: {r.stderr.strip()[-400:]}")
            else:
                verdict = json.loads(cmp_json.read_text())
                rec["cubin_cmp"] = {k: v for k, v in verdict.items() if k != "rows"}
                if not verdict["kernel_text_all_equal"]:
                    reasons.append(f"{len(verdict['kernel_text_differs'])} kernel .text sections differ")
                image_kernels = {r["section"] for r in verdict["rows"]
                                 if r["section"].startswith(".text.") and r["image"]}
                built_kernels = {r["section"] for r in verdict["rows"]
                                 if r["section"].startswith(".text.") and r["rebuilt"]}
                if image_kernels != built_kernels:
                    reasons.append("the kernel sets differ")
            for side, path in (("image", image_cubin), ("built", mine)):
                (out / f"{side}.res-usage.txt").write_text(
                    sh([cuobjdump, "-res-usage", str(path)]).stdout)
                (out / f"{side}.sass").write_text(sh([cuobjdump, "-sass", str(path)]).stdout)
            res_same = (out / "image.res-usage.txt").read_bytes() == (out / "built.res-usage.txt").read_bytes()
            sass_a = (out / "image.sass").read_text().splitlines()
            sass_b = (out / "built.sass").read_text().splitlines()
            instr = sum(1 for line in sass_a if re.match(r"^\s+/\*[0-9a-f]+\*/", line))
            rec["res_usage_identical"] = res_same
            rec["sass_identical"] = sass_a == sass_b
            rec["sass_lines"] = [len(sass_a), len(sass_b)]
            rec["sass_instruction_lines"] = instr
            if not res_same:
                reasons.append("res-usage differs")
            if sass_a != sass_b:
                reasons.append("SASS differs")
            if not sass_a:
                reasons.append("cuobjdump -sass printed nothing")

    rec["pass"] = not reasons
    rec["reasons"] = reasons
    rec["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "build_gate.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("pass", "reasons", "toolchain") if k in rec}, indent=1))
    print("cubin_cmp:", json.dumps(rec.get("cubin_cmp", {}).get("kernel_text_equal")),
          "of", json.dumps(rec.get("cubin_cmp", {}).get("kernel_text_sections")),
          "res_usage_identical", rec.get("res_usage_identical"), "sass_identical", rec.get("sass_identical"))
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
