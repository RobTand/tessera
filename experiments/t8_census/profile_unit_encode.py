"""Where one routed census unit's encode time goes (the T8R1 export's own path).

The T-8 routed census encodes 5.3 s per expert projection at 0.56-0.60 of the
GB10 power envelope, and a second exporter on the same GPU adds nothing
(KERNELS-20260930.md, T8R1 first layer quantum).  This times and profiles the
census's own unit path, not a weights-only stand-in: the serving recipe for a
routed stack (``served_recipe(grid, q256, routed_moe)``), the producer
authority's Hessian capture through ``ActivationSource.for_unit`` (LDLQ and the
activation-aware refit), ``encode_linear_planes`` with ``verify`` on, then the
manifest parse and the container pack the exporter does per unit.

Three legs, on real GLM-5.3 expert bytes from the census source:
  timed     wall time per unit, CUDA-synchronised, with NVML power at 10 Hz
  counted   one unit with the window-Viterbi call counters of
            ``moe_encode_rate_profile`` (which path ran, and how often)
  profiled  one unit under torch.profiler (CPU + CUDA): kernel table by device
            time, host ops by self CPU time, the device-busy fraction of the
            wall, and the host-side synchronisations (D2H copies, stream and
            device syncs) that stall the GPU

usage (inside the image; profile_unit_encode.sh runs it there):
  python3 experiments/t8_census/profile_unit_encode.py OUT_DIR --q256 1280 \\
      --hessian H.json --producer-authority A.py [--units 4] [--layer 3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tessera.export import (ActivationSource, encode_linear_planes,        # noqa: E402
                            served_recipe)
from tessera.export_serving import grid_for, load_producer_authority      # noqa: E402
from tessera.structure import STRUCTURE_ROUTED_MOE                        # noqa: E402

SRC = Path("/mnt/shared/tessera-runs/moe/u1-stubs-20260926/source-l8")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class Power:
    """NVML power at ``hz`` on a thread, between start() and stop()."""

    def __init__(self, hz: float = 10.0):
        import pynvml
        pynvml.nvmlInit()
        self._nvml = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        self._dt = 1.0 / hz
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.samples.append((time.time(), self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0))
            time.sleep(self._dt)

    def start(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> dict:
        self._stop.set()
        self._t.join()
        w = sorted(p for _, p in self.samples)
        if not w:
            return {"samples": 0}
        n = len(w)
        return {"samples": n, "mean_w": round(sum(w) / n, 2), "p10_w": w[n // 10],
                "p50_w": w[n // 2], "p90_w": w[int(n * 0.9)], "max_w": w[-1], "envelope_w": 140.0,
                "mean_envelope_frac": round(sum(w) / n / 140.0, 3)}


def unit_names(src: Path, layer: int, experts: list[int]) -> list[str]:
    return [f"model.language_model.layers.{layer}.mlp.experts.{e}.{p}.weight"
            for e in experts for p in PROJECTIONS]


def read_tensor(src: Path, name: str) -> torch.Tensor:
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(src / index[name]), framework="pt") as h:
        return h.get_tensor(name)


def encode_one(name, source, grid, q256, recipe, activation, verify=True):
    """The exporter's per-unit fresh-encode path (export_serving, cached_units None)."""
    from tessera.fused import pack_fused
    from tessera.unit_artifact import parse_unit_artifact
    weight = source.to("cuda", torch.float32).contiguous()
    extra = activation.for_unit(name, weight.shape[1], "cuda", scale_plane=recipe.scale_plane)
    exported, unit_artifact, _forests = encode_linear_planes(
        weight, grid=grid, q256=q256, body=recipe.body, name=name, verify=verify, **extra)
    extra.clear()
    manifest = parse_unit_artifact(exported.blob, device="cuda").manifest
    blob = pack_fused([(name.split(".")[-2], exported.rows, exported.blob)])
    payload = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    return exported, manifest, payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--grid", default="E4M3")
    ap.add_argument("--q256", type=int, required=True)
    ap.add_argument("--hessian", type=Path, required=True)
    ap.add_argument("--producer-authority", type=Path, required=True)
    ap.add_argument("--units", type=int, default=4, help="timed units (after one warm unit)")
    ap.add_argument("--first-expert", type=int, default=112)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a GPU measurement with no GPU")
    args.out.mkdir(parents=True, exist_ok=True)

    # The exporter's own construction (export_serving.main, --hessian with defaults).
    from tessera.export import DEFAULT_LDLQ_BLOCK, DEFAULT_LDLQ_SIGMA
    _authority, canonical_capture = load_producer_authority(args.producer_authority)
    activation = ActivationSource.from_capture(
        args.hessian, canonical_capture=canonical_capture, ldlq_sigma=DEFAULT_LDLQ_SIGMA,
        ldlq_block=DEFAULT_LDLQ_BLOCK, refit_reach_floor=False)
    grid = grid_for(args.grid)
    recipe = served_recipe(grid, args.q256, STRUCTURE_ROUTED_MOE)

    n_experts = -(-(args.units + 3) // 3)        # warm + timed + counted + profiled
    names = unit_names(args.src, args.layer, list(range(args.first_expert, args.first_expert + n_experts)))
    t_read = time.perf_counter()
    sources = {n: read_tensor(args.src, n) for n in names}
    read_s = time.perf_counter() - t_read
    rec = {
        "schema": "tessera.t8_census.profile_unit_encode.v1",
        "tree": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                               cwd=ROOT).stdout.strip() or None,
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "src": str(args.src), "layer": args.layer, "grid": args.grid, "q256": args.q256,
        "recipe": {"body": recipe.body.name, "plane": recipe.scale_plane.name,
                   "window_bits": recipe.window_bits, "span": recipe.span},
        "hessian": str(args.hessian), "verify": True,
        "source_read_s": round(read_s, 3), "source_read_units": len(names),
    }
    queue = list(names)

    # Warm: compiles, captures and plan caches are paid once.
    t0 = time.perf_counter()
    n = queue.pop(0)
    encode_one(n, sources[n], grid, args.q256, recipe, activation)
    torch.cuda.synchronize()
    rec["warm_unit"] = {"name": n, "s": round(time.perf_counter() - t0, 3)}

    # Timed.
    power = Power()
    power.start()
    times = []
    for _ in range(args.units):
        n = queue.pop(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        exported, manifest, _payload = encode_one(n, sources[n], grid, args.q256, recipe, activation)
        torch.cuda.synchronize()
        times.append({"name": n, "s": round(time.perf_counter() - t0, 3),
                      "wire_bytes": int(exported.exact_bytes), "rows": int(exported.rows),
                      "cols": int(exported.columns)})
    rec["timed"] = {"units": times, "mean_s": round(sum(t["s"] for t in times) / len(times), 3),
                    "power": power.stop()}

    # Counted: which window-Viterbi path ran, and how often.
    from moe_encode_rate_profile import Counters
    counters = Counters()
    real_window, real_fused = counters.install()
    n = queue.pop(0)
    t0 = time.perf_counter()
    encode_one(n, sources[n], grid, args.q256, recipe, activation)
    torch.cuda.synchronize()
    rec["counted"] = {"name": n, "s_with_counter_syncs": round(time.perf_counter() - t0, 3),
                      **counters.snapshot()}
    from tessera import encode as encode_module
    from tessera import window_viterbi
    encode_module.viterbi_window = real_window
    window_viterbi.viterbi_window_fused = real_fused

    # Profiled.
    from torch.profiler import ProfilerActivity, profile
    n = queue.pop(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        encode_one(n, sources[n], grid, args.q256, recipe, activation)
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    trace = args.out / "unit_trace.json"
    prof.export_chrome_trace(str(trace))
    events = prof.key_averages()

    def dev_us(e):
        return float(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0)))

    kernels = [e for e in events if dev_us(e) > 0]
    device_us = sum(dev_us(e) for e in kernels)
    by_dev = sorted(kernels, key=dev_us, reverse=True)[:30]
    by_cpu = sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:30]
    sync_names = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync",
                  "cudaMemcpy", "aten::item", "aten::_local_scalar_dense", "aten::nonzero",
                  "cudaEventSynchronize", "aten::to", "aten::copy_")
    rec["profiled"] = {
        "name": n, "wall_s": round(wall, 3),
        "device_busy_s": round(device_us / 1e6, 3),
        "device_busy_frac_of_wall": round(device_us / 1e6 / wall, 3),
        "note": ("device_busy sums self device time over ops, so overlapping streams can "
                 "exceed the wall; one stream here"),
        "kernels_by_device_time": [
            {"name": e.key[:160], "calls": e.count, "device_ms": round(dev_us(e) / 1e3, 3),
             "frac_of_device": round(dev_us(e) / device_us, 4) if device_us else None}
            for e in by_dev],
        "host_ops_by_self_cpu": [
            {"name": e.key[:160], "calls": e.count, "self_cpu_ms": round(e.self_cpu_time_total / 1e3, 3),
             "cpu_total_ms": round(e.cpu_time_total / 1e3, 3)} for e in by_cpu],
        "sync_points": {e.key: {"calls": e.count, "self_cpu_ms": round(e.self_cpu_time_total / 1e3, 3)}
                        for e in events if e.key in sync_names},
        "trace": str(trace),
    }
    (args.out / "profile_unit_encode.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("timed", "counted")}, indent=1))
    print(json.dumps({k: v for k, v in rec["profiled"].items()
                      if k in ("wall_s", "device_busy_s", "device_busy_frac_of_wall", "sync_points")}, indent=1))
    for k in rec["profiled"]["kernels_by_device_time"][:15]:
        print(f"  {k['device_ms']:>10.3f} ms  {k['calls']:>6}  {k['frac_of_device']}  {k['name'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
