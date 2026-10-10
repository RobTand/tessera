#!/usr/bin/env python3
"""tessera#619: fused window Viterbi power, throughput and profiler receipt.

Encode GLM-5.3-Flash routed-expert projections (gate/up 2048x4096) at the two
rungs the issue measured (BF16 q256 1024, E4M3 q256 896, L=14 window body over
the CHANNEL plane, weights-only fresh encode) two ways:

  single: one unit per encode_linear_planes call (the issue baseline);
  batch4: four same-shape units in one encode_linears_planes call, whose
          Viterbi calls ride _run_joined along the column axis.

Record wall units/s, nvidia-smi power against the 140 W envelope, J/unit, a
torch.profiler kernel table for one single-unit encode (it names the
time-holding kernel), and sha256 of every unit blob in both arms (wires stay
byte-identical). verify=False: the claim is encode throughput, and the
verify decode is not on that path.

Width arms: --l2-div sets TESSERA_WINDOW_L2_BYTES to L2//div before tessera
imports, so the step kernel tiles wider than the default sixth of L2.
--best-form sets TESSERA_WINDOW_BEST_FORM. Both keep bytes identical by
contract. The receipt records the resolved budget and each plan width.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

ENVELOPE_W = 140.0


class Power:
    """Sample nvidia-smi power.draw as one streaming subprocess for the run.

    One ``nvidia-smi -lms`` process, read line by line with arrival
    timestamps: no spawn per sample, so a loaded box cannot starve the
    sampler behind fork storms. ``errors`` counts unparsed lines, so a
    silent window is distinguishable from an idle one.
    """

    def __init__(self, period_ms: int = 500):
        self._period_ms = period_ms
        self._samples: list[tuple[float, float]] = []
        self._errors = 0
        self._stop = threading.Event()
        self._proc = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits", f"-lms={self._period_ms}"],
                stdout=subprocess.PIPE, text=True)
        except Exception:
            self._proc = None
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        try:
            if self._proc is not None:
                self._proc.terminate()
        except Exception:
            pass
        self._thread.join(timeout=10)

    def _run(self):
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self._stop.is_set():
                break
            try:
                watts = float(line.strip().split()[0])
            except Exception:
                self._errors += 1
                continue
            self._samples.append((time.time(), watts))

    def window(self, t0: float, t1: float) -> dict:
        sel = [s for s in self._samples if t0 <= s[0] <= t1]
        out: dict = {"points": len(sel), "sampler_errors": self._errors}
        if not sel:
            return out
        watts = [s[1] for s in sel]
        out.update({
            "mean_w": round(sum(watts) / len(watts), 2),
            "max_w": round(max(watts), 2),
            "min_w": round(min(watts), 2),
            "mean_fraction_of_envelope": round(sum(watts) / len(watts) / ENVELOPE_W, 4),
            "max_fraction_of_envelope": round(max(watts) / ENVELOPE_W, 4),
        })
        return out

def _netdata_power(host: str, after: float, before: float) -> dict:
    """GPU power over a UTC window from the box's own Netdata, GB10 vs 140 W.

    Reads ``nvidia_smi.gpu_power_draw`` at the box's own 10 s cadence and
    keeps only groups fully inside the window. A second instrument beside
    the in-process sampler, and the one the issue's baseline used.
    """
    import urllib.parse
    import urllib.request
    span = max(1, int(before - after))
    q = urllib.parse.urlencode({
        "contexts": "nvidia_smi.gpu_power_draw", "after": int(after),
        "before": int(before), "points": max(2, span // 10),
        "group_by": "dimension", "format": "json2",
        "time_group": "average", "dimensions": "power_draw",
        "options": "unaligned",
    })
    url = f"http://{host}:19999/api/v2/data?{q}"
    try:
        with urllib.request.urlopen(url, timeout=60) as fh:
            doc = json.loads(fh.read().decode())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "url": url}
    try:
        width = doc.get("view", {}).get("update_every")
        labels = doc["result"]["labels"]
        col = labels.index("power_draw")
        kept = [r[col][0] for r in doc["result"]["data"]
                if r[0] - width >= after and r[0] <= before]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "url": url}
    if not kept:
        return {"url": url, "groups_accepted": 0,
                "requested_window_unix": [after, before]}
    return {"url": url, "groups_accepted": len(kept),
            "requested_window_unix": [after, before],
            "mean_w": round(sum(kept) / len(kept), 2),
            "max_w": round(max(kept), 2), "min_w": round(min(kept), 2),
            "mean_fraction_of_envelope": round(sum(kept) / len(kept) / ENVELOPE_W, 4),
            "max_fraction_of_envelope": round(max(kept) / ENVELOPE_W, 4)}


def _blob_sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _encode_grid(grid_name: str, q256: int, rows: int, cols: int,
                 units: int, power: Power) -> dict:
    import torch
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.export import encode_linear_planes, encode_linears_planes

    grid = BF16_GRID if grid_name == "bf16" else E4M3_GRID
    gen = torch.Generator(device="cpu").manual_seed(61900 + q256)
    weights = [(torch.randn(rows, cols, generator=gen) * 0.02)
               .to(device="cuda", dtype=torch.bfloat16).contiguous()
               for _ in range(units)]

    rec: dict = {"grid": grid_name, "q256": q256, "rows": rows, "cols": cols,
                 "units": units}
    names = [f"u{i}" for i in range(units)]
    # Warmup pays Triton compile and each shape's persistent plan capture:
    # the timed single calls replay the single shape, the timed batch call
    # replays the joined shape. Without the second warmup the batch arm runs
    # eager and the comparison measures capture, not width.
    encode_linear_planes(weights[0], grid=grid, q256=q256, name=names[0],
                         verify=False)
    encode_linears_planes(weights, grid=grid, q256=q256, names=names,
                          verify=False)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    w0 = time.time()
    singles = [encode_linear_planes(w, grid=grid, q256=q256, name=name,
                                    verify=False)
               for w, name in zip(weights, names)]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    w1 = time.time()
    single_blobs = [s[0].blob for s in singles]

    t2 = time.perf_counter()
    w2 = time.time()
    batched = encode_linears_planes(weights, grid=grid, q256=q256,
                                    names=names, verify=False)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    w3 = time.time()
    batch_blobs = [b[0].blob for b in batched]
    identical = all(a == b for a, b in zip(single_blobs, batch_blobs))
    ups_single = units / (t1 - t0)
    ups_batch = units / (t3 - t2)
    p_single = power.window(w0, w1)
    p_batch = power.window(w2, w3)
    rec.update({
        "single_s": round(t1 - t0, 3), "batch_s": round(t3 - t2, 3),
        "single_units_per_s": round(ups_single, 4),
        "batch_units_per_s": round(ups_batch, 4),
        "single_power": p_single, "batch_power": p_batch,
        "single_utc_unix": [w0, w1], "batch_utc_unix": [w2, w3],
        "single_j_per_unit": (round(p_single["mean_w"] / ups_single, 2)
                              if p_single.get("points") else None),
        "batch_j_per_unit": (round(p_batch["mean_w"] / ups_batch, 2)
                             if p_batch.get("points") else None),
        "byte_identical": identical,
        "single_sha256": [_blob_sha(b) for b in single_blobs],
        "batch_sha256": [_blob_sha(b) for b in batch_blobs],
    })
    return rec


def _profile_one(rows: int, cols: int) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from tessera.alphabet import BF16_GRID
    from tessera.export import encode_linear_planes

    gen = torch.Generator(device="cpu").manual_seed(61977)
    w = ((torch.randn(rows, cols, generator=gen) * 0.02)
         .to(device="cuda", dtype=torch.bfloat16).contiguous())
    encode_linear_planes(w, grid=BF16_GRID, q256=1024, verify=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        encode_linear_planes(w, grid=BF16_GRID, q256=1024, verify=False)
    torch.cuda.synchronize()
    table = prof.key_averages()
    rows_out = []
    for e in sorted(table, key=lambda x: x.self_device_time_total or 0,
                    reverse=True)[:14]:
        rows_out.append({
            "name": e.key[:96],
            "self_device_ms": round((e.self_device_time_total or 0) / 1e3, 3),
            "total_device_ms": round((e.device_time_total or 0) / 1e3, 3),
            "count": e.count,
        })
    return {"top_by_self_device_time": rows_out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--idle-s", type=float, default=3.0)
    ap.add_argument("--out", default="/tmp/ig619_receipt.json")
    ap.add_argument("--l2-div", type=int, default=None)
    ap.add_argument("--best-form", type=int, default=None, choices=(0, 1))
    args = ap.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"error": "no CUDA device"}))
        return 2
    l2_cache = int(torch.cuda.get_device_properties(0).L2_cache_size)
    if args.l2_div is not None:
        if args.l2_div < 1:
            print(json.dumps({"error": "l2-div is not a positive divisor"}))
            return 2
        os.environ["TESSERA_WINDOW_L2_BYTES"] = str(l2_cache // args.l2_div)
    if args.best_form is not None:
        os.environ["TESSERA_WINDOW_BEST_FORM"] = str(args.best_form)
    from tessera import window_viterbi
    rec: dict = {
        "schema": "tessera.ig619_viterbi_power/1",
        "device": torch.cuda.get_device_name(),
        "uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                 capture_output=True, text=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__))
                                 ).stdout.strip(),
        "fused_available": window_viterbi.fused_available(),
        "env": {k: os.environ.get(k, "") for k in
                ("TESSERA_WINDOW_BEST_FORM", "TESSERA_WINDOW_GRAPH",
                 "TESSERA_WINDOW_L2_BYTES", "TESSERA_WINDOW_RATE_STREAMS")},
        "envelope_w": ENVELOPE_W,
        "l2_cache_size": l2_cache,
        "l2_budget_bytes": window_viterbi._L2_BUDGET,
        "l2_div": args.l2_div,
        "best_form": args.best_form,
    }
    with Power() as power:
        time.sleep(args.idle_s)
        t_idle = time.time()
        time.sleep(args.idle_s)
        rec["idle_power"] = power.window(t_idle - args.idle_s, t_idle)
        grids = []
        for grid_name, q256 in (("bf16", 1024), ("e4m3", 896)):
            try:
                row = _encode_grid(grid_name, q256, args.rows,
                                   args.cols, args.units, power)
                grids.append(row)
                print(f"grid {grid_name}: single {row['single_s']} s "
                      f"({row['single_units_per_s']} u/s, "
                      f"{row['single_power'].get('mean_w')} W) batch {row['batch_s']} s "
                      f"({row['batch_units_per_s']} u/s, "
                      f"{row['batch_power'].get('mean_w')} W) "
                      f"identical={row['byte_identical']}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep the other grid
                grids.append({"grid": grid_name, "q256": q256,
                              "error": f"{type(exc).__name__}: {exc}"})
        rec["grids"] = grids
        try:
            rec["profile_bf16"] = _profile_one(args.rows, args.cols)
        except Exception as exc:  # noqa: BLE001
            rec["profile_bf16"] = {"error": f"{type(exc).__name__}: {exc}"}
        netdata = {}
        for row in grids:
            if "error" in row:
                continue
            key = f"{row['grid']}@{row['q256']}"
            netdata[key] = {
                "single": _netdata_power("localhost", *row["single_utc_unix"]),
                "batch": _netdata_power("localhost", *row["batch_utc_unix"]),
            }
        rec["netdata_power"] = netdata
    rec["plans"] = _plan_diag()
    text = json.dumps(rec, indent=1)
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    return 0


def _plan_diag() -> list:
    """Each cached plan width, batch count and step count, per shape.

    Names whether an arm runs launch-bound: batches times steps is the
    _step launch count, width times low is the grid behind each launch.
    """
    from tessera import window_viterbi

    plans, _seen = window_viterbi._window_maps()
    out = []
    for key, plan in plans.items():
        (device, rows, cols, arity, size, rate, chunk, has_w, budget, scan,
         best, tile) = key
        out.append({
            "rows": rows, "cols": cols, "arity": arity, "size": size,
            "rate": rate, "chunk": chunk, "has_weights": has_w,
            "budget_bytes": budget, "best_form": best,
            "steps": plan.steps, "width": plan.width,
            "batches": plan.batches, "grid": list(plan.grid),
        })
    return out

if __name__ == "__main__":
    sys.exit(main())
