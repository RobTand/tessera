"""Where the H-aware encode's cost goes -- profiler evidence for the factor.

The 9.2x this file was written for is retracted; issue #13's controlled
matched pair reads 18.6-19.2x on the pre-change encoder and 5.27x after it.
Superseded as a driver by ``ldlq_hcost_matched_pair.py``, which runs both arms
in one process and sweeps the graph lever; kept because its per-arm profile
shape is the same and the numbers it produced are cited in the issue.

The receipt claims LDLQ's cost on the TCQ body is per-call overhead rather than
work: the pass splits a row into ``cols/block`` sequential segments and calls
the trellis once per segment, so the total *work* is unchanged and what
multiplies is everything a call pays around its kernels.  That is a claim about
where the time goes, so it needs a profiler and not a stopwatch (AGENTS.md
principle 13 / CLAUDE.md principle 15).

Both legs, because they answer different questions:

* ``torch.profiler`` says what happens *inside* the process -- self device time
  per kernel, launch counts, and the wall-vs-device gap that is the actual
  signature of a launch-bound loop; and
* a power sampler against the box's ~140 W envelope says whether the GPU was
  loaded at all, which no in-process tool can see and which ``gpu_utilization``
  actively misreports on GB10 (96% for a stalled kernel and a saturated one
  alike).

usage:
  ldlq_cost_profile.py --model DIR --h H.pt --unit NAME [--grid E2M1x2]
                       [--q256 896] [--block 32] [--out prof.json]
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


def grid_by_name(name: str):
    for g in SERIALISABLE_GRIDS.values():
        if g.name == name:
            return g
    raise SystemExit(f"unknown grid {name!r}; one of "
                     f"{[g.name for g in SERIALISABLE_GRIDS.values()]}")


def sample_power(stop, out, period=0.5):
    """nvidia-smi power against the envelope, the only honest load signal here."""
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


def profile_arm(label, fn):
    """Wall, device-busy and launch counts for one encode."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    stop = threading.Event()
    watts: list = []
    t = threading.Thread(target=sample_power, args=(stop, watts), daemon=True)
    t.start()
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        fn()
        torch.cuda.synchronize()
    wall = time.time() - t0
    stop.set()
    t.join(timeout=2)

    evs = prof.key_averages()

    def dev_s(e):
        return (getattr(e, "self_device_time_total", 0) or 0) / 1e6

    device_total = sum(dev_s(e) for e in evs)
    launches = sum(int(e.count) for e in evs if dev_s(e) > 0)
    top = sorted(evs, key=lambda e: -dev_s(e))
    rows = [{"op": e.key, "count": int(e.count), "self_device_s": round(dev_s(e), 4)}
            for e in top[:12] if dev_s(e) > 0]
    return {
        "arm": label,
        "wall_s": round(wall, 2),
        "device_busy_s": round(device_total, 2),
        "gpu_idle_frac": round(1 - device_total / wall, 4) if wall else None,
        "device_kernel_launches": launches,
        "power_mean_w": round(sum(w for w, _ in watts) / len(watts), 1) if watts else None,
        "power_max_w": round(max(w for w, _ in watts), 1) if watts else None,
        "util_mean_pct": round(sum(u for _, u in watts) / len(watts), 1) if watts else None,
        "power_samples": len(watts),
        "top_ops": rows,
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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda"
    grid = grid_by_name(a.grid)
    payload = torch.load(a.h, map_location="cpu", weights_only=False)
    H = payload["H"][a.unit].to(dev, torch.float32)
    with safe_open(f"{a.model}/model.safetensors", framework="pt") as f:
        W = f.get_tensor(a.unit + ".weight").to(dev, torch.float32).contiguous()
    rows, cols = W.shape
    L = block_ldl(regularize_hessian(H, sigma_reg=a.sigma), a.block)
    hn = H.diagonal() / H.diagonal().mean()

    def enc(**kw):
        encode_linear_planes(W, grid=grid, q256=a.q256, name=a.unit,
                             verify=False, **kw)

    # Warm the kernels and any lazily built forest, so the measured arms are not
    # charged for a first-call compile that a real export pays once per process.
    enc()

    out = {
        "unit": a.unit, "shape": [rows, cols], "grid": a.grid, "q256": a.q256,
        "block": a.block, "segments": cols // a.block, "sigma": a.sigma,
        "envelope_w": 140,
        "arms": [
            profile_arm("weights-only", lambda: enc()),
            profile_arm("refit h^1.0 only", lambda: enc(refit_metric=hn)),
            profile_arm(f"LDLQ {a.sigma}/{a.block}",
                        lambda: enc(ldl=L, ldl_block=a.block)),
            profile_arm(f"LDLQ {a.sigma}/{a.block} + refit h^1.0",
                        lambda: enc(ldl=L, ldl_block=a.block, refit_metric=hn)),
        ],
    }
    base = out["arms"][0]
    for arm in out["arms"]:
        arm["vs_weights_only"] = round(arm["wall_s"] / base["wall_s"], 2)
        arm["launches_vs_weights_only"] = round(
            arm["device_kernel_launches"] / max(base["device_kernel_launches"], 1), 2)

    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2))
    print(f"\n{'arm':<34} {'wall':>8} {'dev':>8} {'idle':>7} {'launches':>10} "
          f"{'x_lau':>6} {'W':>6} {'util%':>6} {'x':>6}")
    for arm in out["arms"]:
        print(f"{arm['arm']:<34} {arm['wall_s']:8.2f} {arm['device_busy_s']:8.2f} "
              f"{arm['gpu_idle_frac']:7.1%} {arm['device_kernel_launches']:10d} "
              f"{arm['launches_vs_weights_only']:6.2f} "
              f"{arm['power_mean_w'] or 0:6.1f} {arm['util_mean_pct'] or 0:6.1f} "
              f"{arm['vs_weights_only']:6.2f}")
    for i, arm in enumerate(out["arms"]):
        print(f"\ntop device ops, {arm['arm']}:")
        for r in arm["top_ops"][:6]:
            print(f"  {r['op'][:58]:<60} n={r['count']:<8} {r['self_device_s']:8.3f} s")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
