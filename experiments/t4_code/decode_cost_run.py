"""Build and run ``decode_cost.cu`` (sm_121a), sampling board power meanwhile.

    python3 experiments/t4_code/decode_cost_run.py --out DIR [--iters N]

Writes ``DIR/decode_cost.json`` (the binary's own JSON plus the build
command, the source digest and the power samples over the run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path


def power_samples(stop, samples):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        read = lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # noqa: E731
    except Exception:  # noqa: BLE001
        read = None
    while not stop.is_set():
        try:
            if read is not None:
                w = read()
            else:
                w = float(subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.split()[0])
            samples.append((time.time(), w))
        except Exception:  # noqa: BLE001
            pass
        stop.wait(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent / "decode_cost.cu"
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    binary = out / "decode_cost"
    cmd = [nvcc, "-O3", "-std=c++17", "-arch=sm_121a", "-o", str(binary), str(src)]
    subprocess.run(cmd, check=True)
    samples, stop = [], threading.Event()
    th = threading.Thread(target=power_samples, args=(stop, samples), daemon=True)
    th.start()
    t0 = time.time()
    res = subprocess.run([str(binary), str(a.iters)], check=True, capture_output=True, text=True)
    t1 = time.time()
    stop.set()
    th.join()
    data = json.loads(res.stdout)
    w = [v for ts, v in samples if t0 <= ts <= t1]
    data["build"] = cmd
    data["source_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    data["power"] = {"window_unix": [t0, t1], "samples": len(w),
                     "mean_w": sum(w) / len(w) if w else None, "max_w": max(w) if w else None,
                     "envelope_w": 140.0}
    (out / "decode_cost.json").write_text(json.dumps(data, indent=1))
    print(json.dumps({k: data[k] for k in ("device", "sm_count", "power")}))
    for r in data["results"]:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
