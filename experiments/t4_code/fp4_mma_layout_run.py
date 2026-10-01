"""Build and run ``fp4_mma_layout.cu`` (sm_121a); write ``DIR/fp4_mma_layout.json``.

    python3 experiments/t4_code/fp4_mma_layout_run.py --out DIR [--trials N]

Exits with the binary's status: nonzero when the discovery finds an anomaly,
an unscaled (row, group) or (column, group), or any output differs from the
double-precision reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=4096)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent / "fp4_mma_layout.cu"
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    binary = out / "fp4_mma_layout"
    cmd = [nvcc, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a", "-o", str(binary), str(src)]
    subprocess.run(cmd, check=True)
    res = subprocess.run([str(binary), str(a.trials)], capture_output=True, text=True)
    sys.stderr.write(res.stderr)
    data = json.loads(res.stdout)
    data.update(build=cmd, source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                returncode=res.returncode)
    (out / "fp4_mma_layout.json").write_text(json.dumps(data, indent=1))
    print(json.dumps({k: v for k, v in data.items() if not k.endswith("_map")}))
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
