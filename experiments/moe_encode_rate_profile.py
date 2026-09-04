#!/usr/bin/env python
"""Per-unit encode rate at the real routed-expert shapes, and WHICH machine ran (#5).

Issue #5 records the encode cost as unconfirmed -- "~4.9-5.7 s per E4M3 window
unit ... ~10x below the fused-Viterbi rate elsewhere in this repo, so the fused
path may not be being taken" -- and asks for that to be settled by profile
before hours of encode are committed.  ``docs/tessera-serving-and-moe-contract.md``
§9.4 answered it on 2026-09-02; this harness is that answer as a re-runnable
measurement, on the tree it is standing in, because the encoder has moved since
(the window Viterbi's capture gate, #94; the scale-refit landing, #50/#75).

WHICH PATH RAN IS A CALL COUNT, NOT A LOG LINE.  ``encode.viterbi_window``
chooses between a Triton kernel and a pure-torch reference by
``impl``/``rate``/``fused_available()``, and neither branch announces itself.
So both are counted at their own call sites: every call into
``encode.viterbi_window`` and every call that reached
``window_viterbi.viterbi_window_fused``.  ``reference = total - fused`` is then
a subtraction over two counters rather than an inference from a timing.  The
CUDA-side half is ``torch.profiler``: the fused step kernel's share of device
time says the same thing in the profiler's own vocabulary, and a run whose
counters say "fused" while the kernel table shows an elementwise soup would be
a counter that is measuring the wrong thing.

THE SHAPES AND THE RECIPE ARE THE REAL ONES.  Weights come from the routed
experts of ``GLM-5.3-Flash-4layer`` -- ``gate_proj``/``up_proj`` ``[2048,
4096]``, ``down_proj`` ``[4096, 2048]`` -- read straight out of the shard the
index maps them to, so the rate is measured on the data it will be paid on and
not on ``randn``.  The recipe is ``TESSERA_FP8``'s: the E4M3 grid at
``q256=1024``, which ``wire_recipe`` resolves to the window body over the
CHANNEL plane at ``window_bits=14``.

Power is read against the box's envelope, never utilisation: on GB10
``gpu_utilization`` reads ~96% for a stalled kernel and a saturated one alike.
``nvidia-smi`` is sampled here for a coarse in-run figure; the Netdata series
on the box is the instrument that can see whether anything else was loaded, and
the receipt quotes it.

usage (through the pool, one GPU slot):
    pbrun.py --gpu --cwd <checkout> -- <venv>/bin/python experiments/moe_encode_rate_profile.py OUT.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import encode as encode_module           # noqa: E402
from tessera import window_viterbi                    # noqa: E402
from tessera.alphabet import E4M3_GRID                # noqa: E402
from tessera.export import encode_linear_planes, wire_recipe   # noqa: E402

SRC = Path("/mnt/shared/models/GLM-5.3-Flash-4layer")
#: The MoE layers on this model start at ``first_k_dense_replace = 1``.
LAYER = 1
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class Counters:
    """The two call sites, counted where they are, plus their own wall time."""

    def __init__(self):
        self.total = 0
        self.fused = 0
        self.fused_seconds = 0.0
        self.rates: dict[int, int] = {}

    def install(self):
        real_window = encode_module.viterbi_window
        real_fused = window_viterbi.viterbi_window_fused

        def counted_window(targets, vectors, window_bits, rate, *a, **kw):
            self.total += 1
            self.rates[int(rate)] = self.rates.get(int(rate), 0) + 1
            return real_window(targets, vectors, window_bits, rate, *a, **kw)

        def counted_fused(targets, vectors, window_bits, rate, *a, **kw):
            self.fused += 1
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = real_fused(targets, vectors, window_bits, rate, *a, **kw)
            torch.cuda.synchronize()
            self.fused_seconds += time.perf_counter() - t0
            return out

        encode_module.viterbi_window = counted_window
        window_viterbi.viterbi_window_fused = counted_fused
        return real_window, real_fused

    def snapshot(self):
        return {"viterbi_window_calls": self.total,
                "reached_viterbi_window_fused": self.fused,
                "took_the_reference": self.total - self.fused,
                "fused_seconds": round(self.fused_seconds, 4),
                "calls_by_rate": {str(k): v for k, v in sorted(self.rates.items())}}

    def reset(self):
        self.total = self.fused = 0
        self.fused_seconds = 0.0
        self.rates = {}


def _utc() -> str:
    """The wall clock, so the Netdata window can be cut to this run.

    An in-process profiler cannot see whether the box was otherwise loaded, and
    the box-level series can only be read against a run whose start and end are
    written down.  Without these two fields the power half of the measurement
    is unrecoverable after the fact.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def power_w() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return float("nan")


def expert_tensors(src: Path, layer: int, experts: int):
    """``(name, tensor)`` for the first ``experts`` experts of one MoE layer."""
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    picked = []
    for expert in range(experts):
        for projection in PROJECTIONS:
            name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            if name not in weight_map:
                raise SystemExit(f"{name} is not in {src}'s index; this is not the layout measured")
            picked.append(name)
    by_shard: dict[str, list[str]] = {}
    for name in picked:
        by_shard.setdefault(weight_map[name], []).append(name)
    out = {}
    for shard, names in by_shard.items():
        with safe_open(str(src / shard), framework="pt") as handle:
            for name in names:
                out[name] = handle.get_tensor(name)
    return [(name, out[name]) for name in picked]


def encode_one(name: str, weight: torch.Tensor, q256: int, verify: bool):
    exported, _unit, _forests = encode_linear_planes(
        weight.to("cuda", torch.float32).contiguous(),
        grid=E4M3_GRID, q256=q256, name=name, verify=verify)
    return exported


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--experts", type=int, default=2, help="experts to time (3 units each)")
    ap.add_argument("--q256", type=int, default=1024)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this is a GPU measurement and there is no GPU here")
    recipe = wire_recipe(E4M3_GRID, args.q256)
    counters = Counters()
    counters.install()

    units = expert_tensors(args.src, args.layer, args.experts)
    record = {
        "tree": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                               cwd=Path(__file__).parent).stdout.strip(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "src": str(args.src), "layer": args.layer,
        "grid": E4M3_GRID.name, "q256": args.q256,
        "recipe": {"body": recipe.body.name, "plane": recipe.scale_plane.name,
                   "window_bits": recipe.window_bits, "span": recipe.span},
        "fused_available": bool(window_viterbi.fused_available()),
        "window_fused_max_rate": encode_module.WINDOW_FUSED_MAX_RATE,
        "verify": not args.no_verify,
        # WEIGHTS-ONLY: no Hessian is supplied, so LDLQ and the activation-aware
        # refit are off and the window Viterbi is called once per chunk per
        # pass.  Under LDLQ the call pattern is different by construction
        # (narrow per-block calls, #94), so a Hessian-fed MoE export needs its
        # own number and does not inherit this one.
        "arm": "weights_only (no --hessian, so no LDLQ)",
        "idle_power_w": power_w(),
        "started_utc": _utc(),
    }

    # Warm: compile, capture and any plan cache are paid once and are not the
    # per-unit rate.  The warm unit is one of the real ones, re-encoded below.
    name0, weight0 = units[0]
    t0 = time.perf_counter()
    encode_one(name0, weight0, args.q256, not args.no_verify)
    record["warm_first_call_seconds"] = round(time.perf_counter() - t0, 4)
    record["warm_counters"] = counters.snapshot()

    rows = []
    powers = []
    counters.reset()
    for name, weight in units:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        exported = encode_one(name, weight, args.q256, not args.no_verify)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        powers.append(power_w())
        rows.append({"tensor": name, "shape": list(weight.shape),
                     "seconds": round(seconds, 4),
                     "wire_bytes": int(exported.exact_bytes),
                     "bpp": float(exported.bpp),
                     "counters": counters.snapshot()})
        counters.reset()
        print(f"  {name.split('.mlp.')[-1]:34s} {list(weight.shape)}  {seconds:.3f}s  "
              f"{rows[-1]['counters']}", flush=True)
    record["units"] = rows
    record["power_w_during_encode"] = powers

    # The CUDA-side answer, in the profiler's own vocabulary.
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        encode_one(name0, weight0, args.q256, not args.no_verify)
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total_cuda = sum(e.self_device_time_total for e in events)
    top = sorted(events, key=lambda e: -e.self_device_time_total)[:8]
    record["profiler"] = {
        "unit": name0, "total_self_cuda_us": total_cuda,
        "top_self_cuda": [{"name": e.key, "self_cuda_us": e.self_device_time_total,
                           "share": round(e.self_device_time_total / total_cuda, 4),
                           "calls": e.count} for e in top],
    }

    per_unit = [r["seconds"] for r in rows]
    record["summary"] = {
        "units_timed": len(per_unit),
        "seconds_per_unit_min": round(min(per_unit), 4),
        "seconds_per_unit_max": round(max(per_unit), 4),
        "seconds_per_unit_mean": round(sum(per_unit) / len(per_unit), 4),
        "units_per_moe_layer": 288 * 3,
        "estimated_minutes_per_moe_layer": round(sum(per_unit) / len(per_unit) * 288 * 3 / 60, 1),
    }
    record["finished_utc"] = _utc()
    record["netdata"] = {"metric": "nvidia_smi.gpu_power_draw",
                         "envelope_w": 140,
                         "window": [record["started_utc"], record["finished_utc"]]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(json.dumps(record["summary"], indent=1))
    print(json.dumps(record["profiler"], indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
