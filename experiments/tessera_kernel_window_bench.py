"""Kernel-lane timing for the window body (schema minor 2), principle 15.

Same tensor and same protocol as ``tessera_kernel_span_bench.py`` -- one GLM
routed expert (2048 x 4096, the shipping shape) -- so the two are comparable
arm for arm.  Arms: the span-2 trellis GEMV at the E2M1x2 cap (the lane's
current default, **re-timed in this session** rather than quoted from the
2026-09-01 run, because the box is not idle), the window GEMV over the same
grid at L in {12, 14, 16}, and the window GEMV over E4M3 at R=4 and the same
widths.  A bf16 torch GEMV is the bandwidth anchor.

The two bodies do **not** weigh the same at the E2M1x2 cap: span-2 spends one
select bit and two label bits per pair plus six point bits per code -- 3.75
b/wt of body -- where the window body spends R=7 bits per code over two
weights, 3.5.  Both carry the 0.25 b/wt LUT scale nibble.  Bytes per call are
printed per arm so the ms are read against them and not against each other.

Every arm is checked bit-exact on a one-hot column against ``reconstruct_unit``
at the full 2048x4096 shape before it is timed: the pytest sweep runs the
widths at 256x512 and only L=12 at this shape, because encoding 2048x4096 at
L=16 costs ~5.5 minutes.

In-process: CUDA-event ms/call plus a ``torch.profiler`` pass for self CUDA
time.  Box-level: ``nvidia-smi`` power sampled once a second for the whole run
into ``--power-log``, with each arm's epoch window in the JSON so the draw can
be read against the ~140 W envelope (``CLAUDE.md`` §4.15; on GB10
``gpu_utilization`` is non-diagnostic).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera_kernel_span_bench import profiled, timed  # noqa: E402

from tessera.alphabet import E2M1_GRID, E4M3_GRID, build_forest, tuple_grid  # noqa: E402
from tessera.decode import reconstruct_unit  # noqa: E402
from tessera.encode import encode_unit  # noqa: E402
from tessera.manifest import BodyKind, ScalePlaneKind  # noqa: E402
from tessera.trellis import ConvCode  # noqa: E402

CODE = ConvCode(memory=6)


def power_window(log: Path, start: float, stop: float) -> "dict | None":
    """Mean/max GPU power over ``[start, stop]`` from the nvidia-smi log."""
    if not log.exists():
        return None
    import datetime

    draws = []
    for line in log.read_text().splitlines()[1:]:
        parts = line.split(", ")
        if len(parts) != 2:
            continue
        try:
            stamp = datetime.datetime.strptime(
                parts[0], "%Y/%m/%d %H:%M:%S.%f"
            ).timestamp()
            watts = float(parts[1].split()[0])
        except ValueError:
            continue
        if start <= stamp <= stop:
            draws.append(watts)
    if not draws:
        return None
    return {"mean_w": sum(draws) / len(draws), "max_w": max(draws), "samples": len(draws)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--proj", default="gate_proj")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--windows", type=int, nargs="+", default=[12, 14, 16])
    ap.add_argument("--arms", nargs="+", default=["bf16", "span2", "k2", "e4m3"])
    ap.add_argument("--lanes", type=int, default=64)
    ap.add_argument("--split-k", type=int, default=128)
    ap.add_argument("--label", default="")
    ap.add_argument("--power-log", default="experiments/results/tessera_kernel_window_power.csv")
    ap.add_argument("--out", default="experiments/results/tessera_kernel_window_bench.json")
    a = ap.parse_args()
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    power_log = Path(a.power_log)
    power_log.parent.mkdir(parents=True, exist_ok=True)
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw", "--format=csv", "-l", "1"],
        stdout=open(power_log, "w"), stderr=subprocess.DEVNULL,
    )
    log = lambda s: print(s, flush=True)
    try:
        index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
        name = f"model.language_model.layers.{a.layer}.mlp.experts.0.{a.proj}.weight"
        with safe_open(f"{F.SRC}/{index[name]}", framework="pt") as f:
            w = f.get_tensor(name).contiguous().cuda().float()
        rows, cols = w.shape
        torch.manual_seed(0)
        x = torch.randn(cols, device="cuda")
        records = []
        log(f"{name} {tuple(w.shape)}  commit {commit}  label {a.label!r}  "
            f"lanes {a.lanes} split_k {a.split_k}  windows {a.windows}")
        log("  concurrent GPU load: " + subprocess.run(
            ["bash", "-lc", "pgrep -af tessera_ | grep -v window_bench | wc -l"],
            capture_output=True, text=True).stdout.strip() + " other tessera process(es)")

        if "bf16" in a.arms:
            wb, xb = w.to(torch.bfloat16), x.to(torch.bfloat16)
            ms, n, t0, t1 = timed(lambda: wb @ xb, a.seconds)
            rec = {"arm": "bf16 torch gemv", "ms_per_call": ms, "calls": n,
                   "bytes_per_call": wb.numel() * 2, "epoch": [t0, t1],
                   "profile": profiled(lambda: wb @ xb)}
            rec["GBps"] = rec["bytes_per_call"] / ms / 1e6
            records.append(rec)
            log(f"  {'bf16 torch gemv':<34} {ms:8.4f} ms/call  {rec['GBps']:7.1f} GB/s")
            del wb, xb

        arms = []
        if "span2" in a.arms:
            arms.append(("span2 trellis E2M1x2 R=7", "span2", None, None))
        for window in a.windows:
            if "k2" in a.arms:
                arms.append((f"window E2M1x2 R=7 L={window}", "window",
                             tuple_grid(E2M1_GRID, 2), window))
            if "e4m3" in a.arms:
                arms.append((f"window E4M3   R=4 L={window}", "window",
                             E4M3_GRID, window))

        for label, kind, grid, window in arms:
            t_enc = time.time()
            if kind == "span2":
                grid = tuple_grid(E2M1_GRID, 2)
                rate = grid.rate_cap
                forests = {rate: build_forest(rate, grid=grid)}
                unit = encode_unit(w, forests, (rate,) * cols, CODE, span=2,
                                   scale_plane=ScalePlaneKind.LUT, scale_refit=0,
                                   completion=0)
                reference = reconstruct_unit(unit, forests, CODE, completion=0).float()
                packed = pack_unit_for_kernel(unit, forests[rate], CODE)
                body = packed["select"].numel() + packed["label"].numel() + packed["point"].numel()
                scale_bytes = packed["nibbles"].numel()
            else:
                rate = 7 if grid.arity == 2 else 4
                unit = encode_unit(w, grid, (rate,) * cols, CODE, body=BodyKind.WINDOW,
                                   window_bits=window, scale_plane=ScalePlaneKind.LUT,
                                   scale_refit=0)
                reference = reconstruct_unit(unit, grid, None).float()
                packed = pack_unit_for_kernel(unit, grid, CODE)
                body = packed["plane"].numel()
                scale_bytes = packed["scale_plane"].numel()
            encode_s = time.time() - t_enc

            # bit-exactness at THIS shape, before any timing number is taken
            probe = torch.zeros(cols, device="cuda")
            probe[17] = 1.0
            exact = bool(torch.equal(
                gemv_from_packed(probe, packed, lanes=a.lanes, split_k=a.split_k),
                reference[:, 17],
            ))
            fn = lambda: gemv_from_packed(x, packed, lanes=a.lanes, split_k=a.split_k)
            rel = float((fn() - reference @ x).norm() / (reference @ x).norm())
            assert exact and rel < 1e-5, f"{label}: exact={exact} rel={rel}"

            ms, n, t0, t1 = timed(fn, a.seconds)
            table_bytes = packed["table"].numel() if kind == "window" else 0
            rec = {"arm": label, "kind": kind, "window_bits": window, "rate": rate,
                   "grid": grid.name, "ms_per_call": ms, "calls": n,
                   "body_bytes": body, "scale_bytes": scale_bytes,
                   "table_bytes": table_bytes,
                   "bytes_per_call": body + scale_bytes + table_bytes,
                   "body_bits_per_weight": body * 8 / (rows * cols),
                   "scale_bits_per_weight": scale_bytes * 8 / (rows * cols),
                   "one_hot_bit_exact": exact, "rel_err_vs_reference": rel,
                   "encode_seconds": encode_s, "epoch": [t0, t1],
                   "profile": profiled(fn)}
            rec["GBps"] = rec["bytes_per_call"] / ms / 1e6
            records.append(rec)
            log(f"  {label:<34} {ms:8.4f} ms/call  {rec['GBps']:7.1f} GB/s  "
                f"({rec['bytes_per_call']/1e6:.2f} MB/call, "
                f"{rec['body_bits_per_weight']:.2f}+{rec['scale_bits_per_weight']:.2f} b/wt"
                f"{f' + {table_bytes} B table' if table_bytes else ''})  "
                f"exact={exact} rel={rel:.2e}  encode {encode_s:.0f}s")
            for p in rec["profile"][:2]:
                log(f"      {p['name'][:58]:<58} {p['self_cuda_ms_per_call']:8.4f} ms self CUDA")
            del unit, reference, packed
            torch.cuda.empty_cache()
    finally:
        time.sleep(2)
        sampler.terminate()
        sampler.wait(timeout=10)

    for rec in records:
        rec["power"] = power_window(power_log, rec["epoch"][0], rec["epoch"][1])
        if rec["power"]:
            log(f"  {rec['arm']:<34} power {rec['power']['mean_w']:5.1f} W mean, "
                f"{rec['power']['max_w']:5.1f} W max over {rec['power']['samples']} samples")
    out = Path(a.out)
    prior = json.load(open(out)) if out.exists() else []
    prior.append({"commit": commit, "label": a.label, "tensor": name,
                  "shape": [rows, cols], "seconds": a.seconds, "lanes": a.lanes,
                  "split_k": a.split_k, "envelope_w": 140, "records": records,
                  "time": time.time()})
    json.dump(prior, open(out, "w"), indent=1)
    log(f"wrote {out} and {power_log}")


if __name__ == "__main__":
    main()
