"""x86 compile-only gate for a serving/csrc source (default regdirect_routed.cu; sm_121, -Xptxas -v): registers and spills.

Uses the CUDA 13 nvcc already unpacked on dl380g10 (read-only) inside an Ubuntu 24.04
container, as experiments/t8r_speed/compile_aligned.py did on the eng-rung-aligned-test
branch: CUDA 13.0 headers predate the host glibc 2.43 rsqrt declarations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="regdirect_routed.cu", help="a file under src/tessera/serving/csrc")
    ap.add_argument("--toolkit", type=Path, default=Path("/tmp/eng-rung-aligned-toolkit"))
    ap.add_argument("--container-image",
                    default="ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    nvcc = args.toolkit / "nvidia" / "cu13" / "bin" / "nvcc"      # the unpacked CUDA 13 wheel's layout
    if not nvcc.is_file():
        raise SystemExit(f"no nvcc at {nvcc}")
    src = Path(__file__).resolve().parents[2] / "src/tessera/serving/csrc" / args.source
    inc = Path(torch.__file__).parent / "include"
    cuda_inc = nvcc.parent.parent / "include"
    work = Path("/tmp") / f"regdirect-routed-compile-{os.environ.get('PRISMABUILD_ACTION_KEY', os.getpid())}"
    work.mkdir(exist_ok=True)
    cmd = [str(nvcc), "-c", str(src), "-o", str(work / "regdirect_routed.o"),
           "-DTORCH_EXTENSION_NAME=tessera_regdirect_routed",
           "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
           "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE", "-Xcompiler=-fPIC",
           "-O3", "-lineinfo", "-std=c++17", "-gencode", "arch=compute_121,code=sm_121", "-Xptxas", "-v"]
    cmd += [f"-I{p}" for p in (inc, inc / "torch/csrc/api/include", Path(sysconfig.get_paths()["include"]),
                                cuda_inc, cuda_inc / "cccl")]
    version = subprocess.run([str(nvcc), "--version"], text=True, capture_output=True, check=True).stdout
    print(version, flush=True)
    ro = (src.parent, args.toolkit.resolve(), inc, Path(sysconfig.get_paths()["include"]))
    cpus = ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    launcher = ["docker", "run", "--rm", "--network=host", "--cpuset-cpus", cpus]
    for p in ro:
        launcher += ["-v", f"{p}:{p}:ro"]
    launcher += ["-v", f"{work}:{work}", "--entrypoint", "bash", args.container_image, "-ec",
                 "export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; "
                 "apt-get install -y -qq --no-install-recommends g++ >/dev/null; g++ --version | head -1; exec \"$@\"",
                 "compile-regdirect", *cmd]
    r = subprocess.run(launcher, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.out / "ptxas.log").write_text(r.stdout)
    print(r.stdout[-20000:], flush=True)
    names = re.findall(r"Compiling entry function '([^']+)'", r.stdout)
    dem = subprocess.run(["c++filt", *names], text=True, capture_output=True).stdout.splitlines() if names else []
    blocks = re.split(r"ptxas info\s+: Compiling entry function", r.stdout)[1:]
    res = []
    for name, readable, block in zip(names, dem, blocks):
        regs = re.search(r"Used (\d+) registers", block)
        sp = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", block)
        res.append({"kernel": readable, "registers": int(regs.group(1)) if regs else None,
                    "stack_bytes": int(sp.group(1)) if sp else None,
                    "spill_store_bytes": int(sp.group(2)) if sp else None,
                    "spill_load_bytes": int(sp.group(3)) if sp else None})
    receipt = {"host": os.uname().nodename, "action": os.environ.get("PRISMABUILD_ACTION_KEY"),
               "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "compiler": version,
               "command": cmd, "container_image": args.container_image, "returncode": r.returncode,
               "kernels": res, "population": "x86 compile-only, sm_121 target, no device execution"}
    (args.out / "compile.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(res, indent=1), flush=True)
    shutil.rmtree(work)
    if r.returncode or not res:
        raise SystemExit(r.returncode or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
