"""The TCQ graph capture's matched pair: bytes, wall, launches, power.

Issue #13 measured the H-aware encode at 19.2x (matched pair, two agreeing
weights-only arms around the treatment) and split the extra wall into ~28%
device work and ~72% launch/host.  ``encode.TcqGraph`` captures the per-segment
Viterbi once and replays it, which is aimed at the 72%.  This script is the
before/after, principle 15 shaped: the same unit encoded in ONE process with
the capture off, on, and off again, so the two off arms agreeing is what makes
the on/off ratio a factor and not a coincidence of scheduling.

Per arm it records

* the sha256 of the exported bytes -- the byte-identity proof is a re-encode
  and a digest, never a read of the diff;
* wall for an UNPROFILED encode (the profiler's own instrumentation inflates a
  launch-bound loop's wall; the receipt already warns not to quote it);
* device-busy seconds and device kernel launches from a second, profiled
  encode of the same arm; and
* mean/max power over the unprofiled encode against the box's ~140 W envelope,
  which is the only load signal GB10 reports honestly.

usage:
  tcq_graph_matched_pair.py --model DIR --h H.pt --unit NAME [--grid E2M1x2]
        [--q256 896] [--block 32] [--arms off,on,off] [--weights-only]
        [--out receipt.json]

``--digest-only`` runs one arm (graph off) and prints its digest: run it on
another tree for the cross-tree byte check.
"""
import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tessera import encode as enc
from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.compensate import block_ldl, regularize_hessian
from tessera.export import encode_linear_planes


def grid_by_name(name: str):
    for g in SERIALISABLE_GRIDS.values():
        if g.name == name:
            return g
    raise SystemExit(f"unknown grid {name!r}")


def sample_power(stop, out, period=0.25):
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            out.append(float(r.stdout.strip()))
        except Exception:
            pass
        stop.wait(period)


def timed(fn):
    """Wall of one call with the device drained on either side.

    A device-wide sync is legal here: this process is single-threaded and no
    capture is open between encodes (a capture opens and closes inside one
    ``TcqGraph`` call).  A threaded caller must not do this.
    """
    torch.cuda.synchronize()
    stop = threading.Event()
    watts: list = []
    t = threading.Thread(target=sample_power, args=(stop, watts), daemon=True)
    t.start()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    stop.set()
    t.join(timeout=2)
    return out, wall, watts


def profiled(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evs = prof.key_averages()

    def dev_s(e):
        return (getattr(e, "self_device_time_total", 0) or 0) / 1e6

    device_total = sum(dev_s(e) for e in evs)
    launches = sum(int(e.count) for e in evs if dev_s(e) > 0)
    top = sorted(evs, key=lambda e: -dev_s(e))
    rows = [{"op": e.key, "count": int(e.count), "self_device_s": round(dev_s(e), 4)}
            for e in top[:10] if dev_s(e) > 0]
    return {"device_busy_s": round(device_total, 3), "device_kernel_launches": launches,
            "top_ops": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--h", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--arms", default="off,on,off")
    ap.add_argument("--weights-only", action="store_true",
                    help="encode without the Hessian (the plain arm of #13's pair)")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--digest-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda"
    grid = grid_by_name(a.grid)
    with safe_open(f"{a.model}/model.safetensors", framework="pt") as f:
        W = f.get_tensor(a.unit + ".weight").to(dev, torch.float32).contiguous()
    kw = {}
    if not a.weights_only:
        payload = torch.load(a.h, map_location="cpu", weights_only=False)
        H = payload["H"][a.unit].to(dev, torch.float32)
        kw["ldl"] = block_ldl(regularize_hessian(H, sigma_reg=a.sigma), a.block)
        kw["ldl_block"] = a.block
        kw["refit_metric"] = H.diagonal() / H.diagonal().mean()

    def run():
        exported, unit, _ = encode_linear_planes(
            W, grid=grid, q256=a.q256, name=a.unit, verify=True, **kw)
        return hashlib.sha256(exported.blob).hexdigest(), exported.exact_bytes

    if a.digest_only:
        enc._TCQ_GRAPH = False
        digest, nbytes = run()
        print(json.dumps({"tree": str(Path(enc.__file__).resolve()), "unit": a.unit,
                          "graph": "off", "sha256": digest, "bytes": nbytes}))
        return

    # First call in the process absorbs first-launch cost (kernels, forests);
    # it is charged to no arm, exactly as a real export pays it once.
    enc._TCQ_GRAPH = False
    run()

    box = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    out = {"box": box, "unit": a.unit, "shape": list(W.shape), "grid": a.grid,
           "q256": a.q256, "hessian": not a.weights_only,
           "block": a.block if not a.weights_only else None,
           "segments": (W.shape[1] // a.block) if not a.weights_only else None,
           "envelope_w": 140, "arms": []}
    for arm in a.arms.split(","):
        enc._TCQ_GRAPH = {"on": True, "off": False}[arm]
        (digest, nbytes), wall, watts = timed(run)
        rec = {"arm": arm, "sha256": digest, "bytes": nbytes, "wall_s": round(wall, 2),
               "power_mean_w": round(sum(watts) / len(watts), 1) if watts else None,
               "power_max_w": round(max(watts), 1) if watts else None,
               "power_samples": len(watts)}
        if not a.no_profile:
            rec.update(profiled(run))
        out["arms"].append(rec)
        print(f"{arm:>4}  wall {wall:8.2f} s  {rec.get('device_busy_s', '-'):>8} dev  "
              f"{rec.get('device_kernel_launches', '-'):>10} launches  "
              f"{rec['power_mean_w']} W  sha256 {digest[:16]}", flush=True)
    digests = {r["sha256"] for r in out["arms"]}
    out["bytes_identical_across_arms"] = len(digests) == 1
    offs = [r["wall_s"] for r in out["arms"] if r["arm"] == "off"]
    ons = [r["wall_s"] for r in out["arms"] if r["arm"] == "on"]
    if offs and ons:
        out["off_over_on"] = round(sum(offs) / len(offs) / (sum(ons) / len(ons)), 2)
        out["off_arms_agree_within"] = round(max(offs) / min(offs), 3)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
