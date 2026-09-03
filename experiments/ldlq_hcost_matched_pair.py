"""Issue #13, leg 1 and 2: the matched pair in ONE process, and where it goes.

The 19.2x in the issue is a wall factor between two *exports* minutes apart.
This runs both arms on ONE tensor, in ONE process, interleaved A B A B, so the
box cannot drift between them -- the same discipline the issue's own control
used across runs, tightened to seconds.

Both instruments, because they answer different questions (CLAUDE.md principle
15): ``torch.profiler`` says where the time goes *inside* the run -- self CUDA
against self CPU is exactly the device/host split the issue claims -- and the
power sampler (and Netdata, queried from the printed epoch window) says whether
the board was loaded at all, which no in-process tool can see and which
``gpu_utilization`` misreports on GB10.

usage:
  ldlq_hcost_matched_pair.py --model DIR --h H.pt --unit NAME [...] [--out J]
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.compensate import block_ldl, regularize_hessian
from tessera.export import encode_linear_planes


def grid_by_name(name):
    for g in SERIALISABLE_GRIDS.values():
        if g.name == name:
            return g
    raise SystemExit(f"unknown grid {name!r}")


def sample_power(stop, out, period=0.25):
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            p, u = r.stdout.strip().split(",")
            out.append((float(p), float(u)))
        except Exception:
            pass
        stop.wait(period)


def timed(fn, watts=None):
    """Wall seconds for one call, with the sampler running if given."""
    torch.cuda.synchronize()
    t0 = time.time()
    r = fn()
    torch.cuda.synchronize()
    return time.time() - t0, r


def profile_arm(label, fn, weights_encoded):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    stop = threading.Event()
    watts = []
    t = threading.Thread(target=sample_power, args=(stop, watts), daemon=True)
    t.start()
    t_start = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    wall = time.time() - t_start
    stop.set(); t.join(timeout=2)

    evs = prof.key_averages()
    dev = lambda e: (getattr(e, "self_device_time_total", 0) or 0) / 1e6
    cpu = lambda e: (getattr(e, "self_cpu_time_total", 0) or 0) / 1e6
    device_total = sum(dev(e) for e in evs)
    cpu_total = sum(cpu(e) for e in evs)
    launches = sum(int(e.count) for e in evs if dev(e) > 0)
    mean_w = sum(w for w, _ in watts) / len(watts) if watts else None
    return {
        "arm": label,
        "epoch_start": round(t_start, 2), "epoch_end": round(t_start + wall, 2),
        "wall_s": round(wall, 3),
        "device_busy_s": round(device_total, 3),
        "host_s": round(cpu_total, 3),
        "device_frac": round(device_total / wall, 4) if wall else None,
        "host_gap_frac": round(1 - device_total / wall, 4) if wall else None,
        "device_kernel_launches": launches,
        "power_mean_w": round(mean_w, 1) if mean_w else None,
        "power_max_w": round(max(w for w, _ in watts), 1) if watts else None,
        "envelope_frac": round(mean_w / 140.0, 3) if mean_w else None,
        "util_mean_pct": round(sum(u for _, u in watts) / len(watts), 1) if watts else None,
        "weights_per_joule": (round(weights_encoded / (mean_w * wall), 1)
                              if mean_w else None),
        "top_device": [{"op": e.key, "n": int(e.count), "s": round(dev(e), 3)}
                       for e in sorted(evs, key=lambda e: -dev(e))[:10] if dev(e) > 0],
        "top_host": [{"op": e.key, "n": int(e.count), "s": round(cpu(e), 3)}
                     for e in sorted(evs, key=lambda e: -cpu(e))[:14] if cpu(e) > 0.01],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--h", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--label", default="tree")
    ap.add_argument("--levers", default="0",
                    help="comma-separated TESSERA_TCQ_GRAPH values to sweep, "
                         "or 'auto' for the shipping default (env unset); the "
                         "pre-change tree only understands the default")
    ap.add_argument("--crop", default=None,
                    help="ROWSxCOLS: crop the unit and its Hessian, so the "
                         "profiler leg stays a tractable number of events "
                         "while running the identical code path")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-pair", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import os

    dev = "cuda"
    grid = grid_by_name(a.grid)
    payload = torch.load(a.h, map_location="cpu", weights_only=False)
    H = payload["H"][a.unit].to(dev, torch.float32)
    with safe_open(f"{a.model}/model.safetensors", framework="pt") as f:
        W = f.get_tensor(a.unit + ".weight").to(dev, torch.float32).contiguous()
    del payload
    if a.crop:
        cr, cc = (int(x) for x in a.crop.lower().split("x"))
        W = W[:cr, :cc].contiguous()
        H = H[:cc, :cc].contiguous()
    rows, cols = W.shape
    L = block_ldl(regularize_hessian(H, sigma_reg=a.sigma), a.block)
    hn = (H.diagonal() / H.diagonal().mean()).clone()

    def enc(**kw):
        return encode_linear_planes(W, grid=grid, q256=a.q256, name=a.unit,
                                    verify=False, **kw)

    ldl_kw = dict(ldl=L, ldl_block=a.block, refit_metric=hn)

    def clear():
        try:
            from tessera.encode import tcq_plan_cache_clear
        except ImportError:
            return
        tcq_plan_cache_clear()

    levers = [x for x in a.levers.split(",") if x != ""]
    out = {
        "label": a.label,
        "unit": a.unit, "shape": [rows, cols], "grid": a.grid, "q256": a.q256,
        "block": a.block, "segments": cols // a.block, "sigma": a.sigma,
        "envelope_w": 140, "torch": torch.__version__,
        "load_at_start": open("/proc/loadavg").read().split()[:3],
        "pairs": {}, "arms": [],
    }
    for lever in levers:
        if lever == "auto":
            os.environ.pop("TESSERA_TCQ_GRAPH", None)   # the shipping default
        else:
            os.environ["TESSERA_TCQ_GRAPH"] = lever
        tag = {"0": "eager", "1": "graph", "auto": "auto"}.get(lever, lever)
        clear(); enc(); enc(**ldl_kw)          # warm this lever's path
        if not a.no_pair:
            pair = {"A_weights_only": [], "B_ldlq_h": []}
            for _ in range(a.reps):
                pair["A_weights_only"].append(round(timed(lambda: enc())[0], 3))
                pair["B_ldlq_h"].append(round(timed(lambda: enc(**ldl_kw))[0], 3))
            med = lambda v: sorted(v)[len(v) // 2]
            pair["h_factor"] = round(
                med(pair["B_ldlq_h"]) / med(pair["A_weights_only"]), 2)
            pair["load"] = open("/proc/loadavg").read().split()[:3]
            out["pairs"][tag] = pair
        if a.profile:
            out["arms"].append(profile_arm(f"A weights-only [{tag}]",
                                           lambda: enc(), rows * cols))
            out["arms"].append(profile_arm(f"B LDLQ+refit [{tag}]",
                                           lambda: enc(**ldl_kw), rows * cols))
    out["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)

    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2))
    hdr = (f"{'arm':<30} {'wall':>8} {'devbusy':>8} {'devfrac':>8} {'hostgap':>8} "
           f"{'launches':>9} {'W':>6} {'env':>6} {'util%':>6} {'wt/J':>10}")
    print("\n" + hdr)
    for arm in out["arms"]:
        print(f"{arm['arm']:<30} {arm['wall_s']:8.2f} {arm['device_busy_s']:8.2f} "
              f"{arm['device_frac']:8.1%} {arm['host_gap_frac']:8.1%} "
              f"{arm['device_kernel_launches']:9d} {arm['power_mean_w'] or 0:6.1f} "
              f"{arm['envelope_frac'] or 0:6.2f} {arm['util_mean_pct'] or 0:6.1f} "
              f"{arm['weights_per_joule'] or 0:10.1f}")
    for arm in out["arms"]:
        print(f"\ntop DEVICE ops, {arm['arm']}:")
        for r in arm["top_device"][:6]:
            print(f"  {r['op'][:56]:<58} n={r['n']:<9} {r['s']:8.3f} s")
        print(f"top HOST ops, {arm['arm']}:")
        for r in arm["top_host"][:10]:
            print(f"  {r['op'][:56]:<58} n={r['n']:<9} {r['s']:8.3f} s")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
