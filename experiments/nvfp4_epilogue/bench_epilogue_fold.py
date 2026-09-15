#!/usr/bin/env python3
"""Price the NVFP4 dense route's scalar epilogue pass, and the ways to remove it (tessera#522).

``nvfp4_route.apply`` used to run ``y = y * layer.tessera_epilogue_scale`` after
the fp4 ``torch._scaled_mm``: a full M x N bf16 read and write to apply one
scalar.  This driver measures that route against candidates that apply the
scalar differently, on the same prepared cells, in one process, with the arms
interleaved inside each M:

* ``route``: the route's own ``method.apply``.  With ``--baseline-route`` the
  baseline is built from that module file instead (the pre-change source), and
  the installed route becomes the ``route_new`` arm.
* ``inplace``: the same GEMM, then ``y.mul_(scale)``.  Same bytes, no second
  output buffer.
* ``v2_host`` / ``v2_device``: ``torch._scaled_mm_v2`` with the two-level NVFP4
  recipe ``[BlockWise1x16, TensorWise]``.  torch passes ``global_a * global_b``
  to cuBLASLt as ``alpha``, so the scalar is applied inside the GEMM.  The
  globals are ``1.0`` and ``float32(scale)``, on the host or on the device.
* ``cutlass``: vLLM's ``_C.cutlass_scaled_fp4_mm`` with ``alpha``, the kernel
  vLLM's own compressed-tensors NVFP4 W4A4 scheme calls.
* ``fp8``: the byte-matched fp8 cell's route, for the fp4/fp8 ratio.

THREE REGIMES, one process.  ``--rows`` at or below 512 slices the frozen
prefill panel and is the DECODE shape a serve actually runs; above 512 it tiles
it.  ``--graph-rows`` repeats the decode set under ``torch.cuda.CUDAGraph``
capture and replay, which is how a serve issues those shapes -- at one row the
launch path, not the mainloop, is the cost.  ``--compile-rows`` runs
``torch._dynamo.explain`` and a ``fullgraph=True`` compile of each arm, and
captures the COMPILED callable into a graph as well.

Numerics: every fp4 candidate is compared with the route's output at every M
(bit equality, max abs, max relative, fraction of differing elements), and both
are compared with a float32-output GEMM times the float64 scalar.  At M=512 and
at the cell's own decode row the route and every candidate are compared with
the cell's frozen reference at the panel's own tolerance.  M above 512 tiles the
frozen 512-row input, and the replica blocks of each arm's output must be
exactly equal.  Every graph replay is compared with its arm's eager output, and
every compiled output with its arm's eager output, for bit equality.

Instruments: CUDA events per apply (bootstrap interval on the sum of the three
units' medians), a sustained apply loop with an in-process NVML power sampler
per arm and M, and a final torch.profiler pass per arm, M and unit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import threading
import time
import traceback
from pathlib import Path

SCHEMA = "tessera.nvfp4_epilogue_fold_bench.v2"
ENVELOPE_W = 140.0
BASE_ROWS = 512
FP4_FORMAT = "TESSERA_E2M1_K2_R896"
FP8_FORMAT = "TESSERA_E4M3_K1_R1006"
UNITS = ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj")


def utc(when=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bootstrap_sum_of_medians(per_unit, *, resamples=10000, confidence=0.95, seed=20260915):
    """Resample each unit's own samples, sum the unit medians per draw."""
    rng = random.Random(seed)
    units = [sorted(float(v) for v in samples) for samples in per_unit]
    for values in units:
        if len(values) < 3:
            raise ValueError("bootstrap needs at least three samples per unit")

    def median(values):
        n = len(values)
        return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])

    draws = []
    for _ in range(resamples):
        total = 0.0
        for values in units:
            n = len(values)
            total += median(sorted(values[int(rng.random() * n)] for _ in range(n)))
        draws.append(total)
    draws.sort()
    return {"median_of_sums_ms": sum(median(values) for values in units),
            "low_ms": draws[int((1.0 - confidence) / 2.0 * resamples)],
            "high_ms": draws[min(resamples - 1, int((1.0 + confidence) / 2.0 * resamples))],
            "resamples": resamples, "confidence": confidence, "seed": seed}


class PowerSeries:
    """One NVML reader thread for the whole run."""

    def __init__(self, interval_s=0.1):
        self.interval_s = interval_s
        self.samples = []
        self.error = None
        self.uuid = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw = pynvml.nvmlDeviceGetUUID(handle)
            self.uuid = raw.decode() if isinstance(raw, bytes) else str(raw)
        except Exception as exc:  # noqa: BLE001 -- recorded, never invented
            self.error = f"pynvml unavailable: {exc}"
            return self

        def run():
            while not self._stop.is_set():
                try:
                    self.samples.append({
                        "t": time.time(), "watts": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                        "sm_mhz": int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
                        "temperature_c": int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))})
                except Exception as exc:  # noqa: BLE001
                    self.error = f"pynvml sample failed: {exc}"
                    return
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def window(self, start, end, *, settle_s):
        rows = [s for s in self.samples if start + settle_s <= s["t"] <= end]
        if not rows:
            return {"status": "no_samples"}
        watts = [s["watts"] for s in rows]
        mean_w = statistics.fmean(watts)
        return {"status": "observed", "samples": len(rows), "mean_w": mean_w, "max_w": max(watts),
                "envelope_fraction": mean_w / ENVELOPE_W,
                "sm_mhz_mean": statistics.fmean(s["sm_mhz"] for s in rows),
                "temperature_c_max": max(s["temperature_c"] for s in rows)}


def load_module_from(path, name):
    """Load a route module file under ``tessera.serving`` so its relative imports resolve."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arms(torch, native_ops, have_cutlass):
    """The fp4 candidates.  Each replicates the route's A side exactly and differs only in the epilogue."""
    from torch.nn.functional import ScalingType, SwizzleType
    block, tensor = int(ScalingType.BlockWise1x16.value), int(ScalingType.TensorWise.value)
    swizzled, flat = int(SwizzleType.SWIZZLE_32_4_4.value), int(SwizzleType.NO_SWIZZLE.value)

    def a_side(layer, x):
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        gs = layer.trellis_input_global_scale.data.reshape(())
        a_q, a_scale = native_ops.native_fp4_quant(x2.contiguous(), gs)
        if a_q.dtype == torch.uint8:
            a_q = a_q.view(torch.float4_e2m1fn_x2)
        return a_q, a_scale

    def inplace(layer, x):
        a_q, a_scale = a_side(layer, x)
        y = torch._scaled_mm(a_q, layer.weight_fp4.t(), scale_a=a_scale, scale_b=layer.scale_b,
                             out_dtype=torch.bfloat16)
        return y.mul_(layer.tessera_epilogue_scale)

    def v2(layer, x, ga, gb):
        a_q, a_scale = a_side(layer, x)
        return torch._scaled_mm_v2(a_q, layer.weight_fp4.t(), [a_scale, ga], [block, tensor], [swizzled, flat],
                                   [layer.scale_b, gb], [block, tensor], [swizzled, flat], None, torch.bfloat16)

    def v2_host(layer, x):
        return v2(layer, x, layer.bench_ga_host, layer.bench_gb_host)

    def v2_device(layer, x):
        return v2(layer, x, layer.bench_ga_device, layer.bench_gb_device)

    def cutlass(layer, x):
        a_q, a_scale = a_side(layer, x)
        out = torch.empty((a_q.shape[0], layer.weight_fp4.shape[0]), dtype=torch.bfloat16, device=x.device)
        torch.ops._C.cutlass_scaled_fp4_mm(out, a_q.view(torch.uint8), layer.weight_fp4.view(torch.uint8),
                                           a_scale, layer.bench_scale_b_2d, layer.bench_gb_device)
        return out

    def reference_fp32(layer, x):
        """float32-output GEMM times the float64 scalar: the rounding-free comparator."""
        a_q, a_scale = a_side(layer, x)
        y = torch._scaled_mm(a_q, layer.weight_fp4.t(), scale_a=a_scale, scale_b=layer.scale_b,
                             out_dtype=torch.float32)
        return y.double() * float(layer.tessera_epilogue_scale)

    arms = {"inplace": inplace, "v2_host": v2_host, "v2_device": v2_device}
    if have_cutlass:
        arms["cutlass"] = cutlass
    return arms, reference_fp32


def numerics(torch, candidate, baseline, chunk_rows=8192):
    """Elementwise comparison in float64, streamed over row chunks so a 131072-row
    output never materialises several float64 copies at once."""
    if candidate.shape != baseline.shape:
        raise ValueError(f"shape mismatch {tuple(candidate.shape)} vs {tuple(baseline.shape)}")
    bit_equal = bool(torch.equal(candidate, baseline)) if candidate.dtype == baseline.dtype else False
    max_abs = max_rel = sum_abs = base_max = 0.0
    differing = 0
    for start in range(0, candidate.shape[0], chunk_rows):
        a = candidate[start:start + chunk_rows].double()
        b = baseline[start:start + chunk_rows].double()
        delta = (a - b).abs()
        rel = torch.where(b != 0, delta / b.abs().clamp_min(1e-300), torch.zeros_like(delta))
        max_abs = max(max_abs, float(delta.max()))
        max_rel = max(max_rel, float(rel.max()))
        sum_abs += float(delta.sum())
        base_max = max(base_max, float(b.abs().max()))
        differing += int((delta != 0).sum())
        del a, b, delta, rel
    numel = candidate.numel()
    return {"bit_equal": bit_equal, "max_abs": max_abs, "max_rel": max_rel, "mean_abs": sum_abs / numel,
            "fraction_differing": differing / numel, "baseline_abs_max": base_max}


def input_rows(tensor, m):
    """The M-row input for a cell: a slice of the frozen panel below 512 rows, a
    tiling of it above.  The decode shapes are rows of the SAME panel, so an arm
    is never compared across inputs."""
    if m <= BASE_ROWS:
        return tensor[:m].contiguous()
    if m % BASE_ROWS:
        raise ValueError(f"M={m} above {BASE_ROWS} must be a multiple of it")
    return tensor.repeat(m // BASE_ROWS, 1).contiguous()


def error_text(exc):
    return "".join(traceback.format_exception_only(exc)).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-route", type=Path, help="module file for the pre-change route")
    parser.add_argument("--arms", default="inplace,v2_host,v2_device,cutlass",
                        help="bench-local fp4 candidates to include")
    parser.add_argument("--warmup-iterations", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--power-seconds", type=float, default=5.0)
    parser.add_argument("--power-settle-seconds", type=float, default=1.5)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--graph-rows", default="", help="M values to capture and replay as CUDA graphs")
    parser.add_argument("--compile-rows", default="1,512,8192",
                        help="M values for the dynamo explain, fullgraph compile and compiled-graph checks")
    parser.add_argument("--compile-arms", default="route,route_new,v2_device,cutlass")
    args = parser.parse_args(argv)

    def parse_rows(text, what):
        values = [int(v) for v in str(text).split(",") if v.strip()]
        for m in values:
            if m < 1 or (m > BASE_ROWS and m % BASE_ROWS):
                raise SystemExit(f"{what}: M={m} must be >= 1, and a multiple of {BASE_ROWS} above it")
        return values

    rows = parse_rows(args.rows, "--rows")
    graph_rows = parse_rows(args.graph_rows, "--graph-rows")
    compile_rows = parse_rows(args.compile_rows, "--compile-rows")

    import torch
    from safetensors.torch import load_file
    from experiments.bench_native_operator import (compare_tensors, native_runtime_context,
                                                   prepare_native_operator, time_apply)

    report = {"schema": SCHEMA, "started_utc": utc(), "rows": rows, "graph_rows": graph_rows,
              "compile_rows": compile_rows, "base_rows": BASE_ROWS,
              "bench_sha256": digest(__file__), "torch": torch.__version__,
              "warmup_iterations": args.warmup_iterations, "iterations": args.iterations,
              "power": {"seconds": args.power_seconds, "settle_seconds": args.power_settle_seconds,
                        "envelope_w": ENVELOPE_W},
              "units": {}, "points": {}, "graph_points": {}, "arm_errors": {},
              "compile_check": {}, "profile": {}}
    try:
        import vllm
        report["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        report["vllm"] = None
    report["gpu"] = torch.cuda.get_device_name(0)
    report["capability"] = list(torch.cuda.get_device_capability(0))

    power = PowerSeries().start()
    try:
        with native_runtime_context():
            from tessera.serving import native_ops
            from tessera.serving import nvfp4_route
            from tessera.serving.telemetry import read_route
            report["installed_route_sha256"] = digest(nvfp4_route.__file__)
            baseline_module = None
            if args.baseline_route is not None:
                baseline_module = load_module_from(args.baseline_route, "tessera.serving._nvfp4_route_baseline")
                report["baseline_route_sha256"] = digest(args.baseline_route)

            cells = {}
            for unit in UNITS:
                for fmt in (FP4_FORMAT, FP8_FORMAT):
                    directory = args.cells_root / f"{unit}__{fmt}"
                    request = json.loads((directory / "request.json").read_text())
                    inputs = json.loads((directory / "inputs.json").read_text())
                    record = json.loads((directory / request["wire_record_path"]).read_text())
                    tensors = load_file(str(directory / request["tensors_path"]), device="cuda")
                    prepared = prepare_native_operator(
                        (directory / request["wire_path"]).read_bytes(), record, tensors["source_weight"],
                        tensors["rendered_weight"], unit=request["unit"], format_name=request["format"],
                        runtime_image=request["runtime_image"], input_global_scale=request.get("input_global_scale"),
                        execution=request["execution"])
                    cells[(unit, fmt)] = {"prepared": prepared, "inputs": inputs, "tensors": tensors}
                fp4 = cells[(unit, FP4_FORMAT)]
                layer = fp4["prepared"]["layer"]
                scale = float(layer.tessera_epilogue_scale)
                layer.bench_ga_host = torch.tensor(1.0, dtype=torch.float32)
                layer.bench_gb_host = torch.tensor(scale, dtype=torch.float32)
                layer.bench_ga_device = layer.bench_ga_host.cuda()
                layer.bench_gb_device = layer.bench_gb_host.cuda()
                rows_n, cols_k = fp4["inputs"]["shape"]
                padded_rows = (rows_n + 127) // 128 * 128
                padded_groups = ((cols_k // 16) + 3) // 4 * 4
                layer.bench_scale_b_2d = layer.scale_b.view(padded_rows, padded_groups)
                report["units"][unit] = {
                    "shape": [rows_n, cols_k], "epilogue_scale_float64": scale,
                    "epilogue_scale_float32": float(layer.bench_gb_host),
                    "epilogue_scale_bfloat16": float(torch.tensor(scale, dtype=torch.bfloat16)),
                    "input_global_scale": float(layer.trellis_input_global_scale.reshape(())),
                    "fp4_wire_bytes": fp4["inputs"]["wire"]["blob_bytes"],
                    "fp8_wire_bytes": cells[(unit, FP8_FORMAT)]["inputs"]["wire"]["blob_bytes"],
                    "numerics_tolerance": fp4["inputs"]["numerics"]}

            requested = [name for name in args.arms.split(",") if name]
            have_cutlass = "cutlass" in requested and hasattr(torch.ops._C, "cutlass_scaled_fp4_mm")
            candidates, reference_fp32 = build_arms(torch, native_ops, have_cutlass)
            candidates = {name: fn for name, fn in candidates.items() if name in requested}
            if "cutlass" in requested and not have_cutlass:
                report["arm_errors"]["cutlass"] = "torch.ops._C.cutlass_scaled_fp4_mm is absent"

            baseline_methods = {}
            if baseline_module is not None:
                for unit in UNITS:
                    scheme = cells[(unit, FP4_FORMAT)]["prepared"]["operator"]["scheme"]
                    baseline_methods[unit] = baseline_module.build_tessera_nvfp4_method(scheme, unit, "resident")

            def arm_layer(name, unit):
                fmt = FP8_FORMAT if name == "fp8" else FP4_FORMAT
                return cells[(unit, fmt)]["prepared"]["layer"]

            def bound_apply(name, unit):
                """The arm as a one-argument callable over this unit's layer: what a
                compile or a capture is handed, with no bench dispatch inside it."""
                layer = arm_layer(name, unit)
                if name == "route" and baseline_module is not None:
                    method = baseline_methods[unit]
                elif name in ("route", "route_new"):
                    method = cells[(unit, FP4_FORMAT)]["prepared"]["method"]
                elif name == "fp8":
                    method = cells[(unit, FP8_FORMAT)]["prepared"]["method"]
                else:
                    fn = candidates[name]
                    return lambda t: fn(layer, t)
                return lambda t: method.apply(layer, t)

            arm_names = (["route"] + (["route_new"] if baseline_module is not None else [])
                         + list(candidates) + ["fp8"])
            bound = {name: {unit: bound_apply(name, unit) for unit in UNITS} for name in arm_names}
            arms = {name: (lambda unit, x, _n=name: bound[_n][unit](x)) for name in arm_names}
            report["arms"] = list(arms)

            # Admission: an arm that raises or is not finite at M=512, or on the
            # cell's own decode row, is recorded and dropped.
            with torch.inference_mode():
                for name in list(arms):
                    try:
                        for unit in UNITS:
                            fmt = FP8_FORMAT if name == "fp8" else FP4_FORMAT
                            cell = cells[(unit, fmt)]
                            for key, source, reference in (
                                    ("anchor_512", "prefill.input", "prefill.reference_output"),
                                    ("anchor_decode", "decode.input", "decode.reference_output")):
                                if source not in cell["tensors"]:
                                    continue
                                y = arms[name](unit, cell["tensors"][source])
                                torch.cuda.synchronize()
                                anchor = compare_tensors(y, cell["tensors"][reference],
                                                         **cell["inputs"]["numerics"])
                                report["units"][unit].setdefault(key, {})[name] = anchor
                                if not anchor["finite"]:
                                    raise ValueError(f"non-finite output at {key}")
                    except Exception as exc:  # noqa: BLE001
                        report["arm_errors"][name] = error_text(exc)
                        del arms[name]
            if "route" not in arms or "fp8" not in arms:
                raise SystemExit(f"a required arm failed: {report['arm_errors']}")
            report["admitted_arms"] = list(arms)

            def sustained(name, xs, m, seconds):
                """One power window over a sustained loop of this arm's three units."""
                torch.cuda.synchronize()
                start = time.time()
                applies = 0
                while time.time() - start < seconds:
                    for unit in UNITS:
                        out = arms[name](unit, xs[unit])
                        del out
                    torch.cuda.synchronize()
                    applies += 1
                end = time.time()
                window = power.window(start, end, settle_s=args.power_settle_seconds)
                flop = sum(2 * m * report["units"][u]["shape"][0] * report["units"][u]["shape"][1] for u in UNITS)
                window.update(three_unit_applies=applies, elapsed_s=end - start,
                              wall_ms_per_three_unit_apply=1000.0 * (end - start) / applies)
                if window.get("status") == "observed":
                    flops = applies * flop / (end - start)
                    window["achieved_tflop_s"] = flops / 1e12
                    window["gflop_per_joule"] = flops / window["mean_w"] / 1e9
                return window

            order = list(arms)
            for index, m in enumerate(rows):
                replicas = m // BASE_ROWS if m > BASE_ROWS else 1
                rotation = order[index % len(order):] + order[:index % len(order)]
                point = {"m": m, "arm_order": rotation, "arms": {}, "numerics": {}}
                with torch.inference_mode():
                    xs = {unit: {fmt: input_rows(cells[(unit, fmt)]["tensors"]["prefill.input"], m)
                                 for fmt in (FP4_FORMAT, FP8_FORMAT)} for unit in UNITS}
                    # Numerics, before any timing.
                    for unit in UNITS:
                        x = xs[unit][FP4_FORMAT]
                        base = arms["route"](unit, x)
                        torch.cuda.synchronize()
                        n = base.shape[1]
                        per_unit = {}
                        if replicas > 1:
                            blocks = base.view(replicas, BASE_ROWS, n)
                            per_unit["route_replica_spread"] = float((blocks - blocks[0:1]).abs().max())
                        try:
                            ref = reference_fp32(cells[(unit, FP4_FORMAT)]["prepared"]["layer"], x)
                            per_unit["route_vs_fp32_reference"] = numerics(torch, base, ref)
                        except Exception as exc:  # noqa: BLE001
                            ref = None
                            per_unit["fp32_reference_error"] = str(exc)
                        for name in arms:
                            if name in ("route", "fp8"):
                                continue
                            y = arms[name](unit, x)
                            torch.cuda.synchronize()
                            entry = numerics(torch, y, base)
                            if replicas > 1:
                                blocks = y.view(replicas, BASE_ROWS, n)
                                entry["replica_spread"] = float((blocks - blocks[0:1]).abs().max())
                            if ref is not None:
                                entry["vs_fp32_reference"] = numerics(torch, y, ref)
                            per_unit[name] = entry
                            del y
                        point["numerics"][unit] = per_unit
                        del base, ref
                    torch.cuda.empty_cache()

                    for name in rotation:
                        fmt = FP8_FORMAT if name == "fp8" else FP4_FORMAT
                        samples = {}
                        for unit in UNITS:
                            x = xs[unit][fmt]
                            timing = time_apply(lambda: arms[name](unit, x), warmup_iterations=args.warmup_iterations,
                                                iterations=args.iterations)
                            samples[unit] = timing["samples_ms"]
                        entry = {"samples_ms": samples,
                                 "unit_median_ms": {u: statistics.median(s) for u, s in samples.items()}}
                        entry.update(bootstrap_sum_of_medians([samples[u] for u in UNITS]))
                        window = sustained(name, {u: xs[u][fmt] for u in UNITS}, m, args.power_seconds)
                        entry["sustained"] = window
                        point["arms"][name] = entry
                        torch.cuda.empty_cache()
                        print(json.dumps({"mode": "eager", "m": m, "arm": name,
                                          "median_of_sums_ms": entry["median_of_sums_ms"],
                                          "low_ms": entry["low_ms"], "high_ms": entry["high_ms"],
                                          "mean_w": window.get("mean_w")}), flush=True)
                    del xs
                route_ms = point["arms"]["route"]["median_of_sums_ms"]
                fp8_ms = point["arms"]["fp8"]["median_of_sums_ms"]
                point["ratios"] = {name: {"vs_route": e["median_of_sums_ms"] / route_ms,
                                          "vs_fp8": e["median_of_sums_ms"] / fp8_ms}
                                   for name, e in point["arms"].items()}
                report["points"][str(m)] = point
                torch.cuda.empty_cache()

            # -- the same M set under CUDA-graph capture ----------------------
            # A decode serve replays graphs; at one row the launch path is the
            # cost, so an arm that wins eager can lose here and the two tables
            # are reported side by side.
            def capture(call, x):
                """Capture ``call(x)`` on a static input after the standard
                side-stream warmup, and return the graph and its output."""
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        call(x)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = call(x)
                return graph, out

            for index, m in enumerate(graph_rows):
                rotation = order[index % len(order):] + order[:index % len(order)]
                point = {"m": m, "arm_order": rotation, "arms": {}, "numerics": {}, "capture_errors": {}}
                with torch.inference_mode():
                    xs = {unit: {fmt: input_rows(cells[(unit, fmt)]["tensors"]["prefill.input"], m)
                                 for fmt in (FP4_FORMAT, FP8_FORMAT)} for unit in UNITS}
                    for name in rotation:
                        fmt = FP8_FORMAT if name == "fp8" else FP4_FORMAT
                        graphs, outputs, samples, checks = {}, {}, {}, {}
                        try:
                            for unit in UNITS:
                                x = xs[unit][fmt]
                                eager = arms[name](unit, x).clone()
                                graph, out = capture(bound[name][unit], x)
                                out.zero_()          # the replay must be what rewrites it
                                graph.replay()
                                torch.cuda.synchronize()
                                checks[unit] = numerics(torch, out, eager)
                                graphs[unit], outputs[unit] = graph, out
                                del eager
                        except Exception as exc:  # noqa: BLE001
                            point["capture_errors"][name] = error_text(exc)
                            graphs.clear()
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            continue
                        point["numerics"][name] = checks
                        for unit in UNITS:
                            timing = time_apply(graphs[unit].replay,
                                                warmup_iterations=args.warmup_iterations,
                                                iterations=args.iterations)
                            samples[unit] = timing["samples_ms"]
                        entry = {"samples_ms": samples,
                                 "unit_median_ms": {u: statistics.median(s) for u, s in samples.items()}}
                        entry.update(bootstrap_sum_of_medians([samples[u] for u in UNITS]))
                        torch.cuda.synchronize()
                        start = time.time()
                        replays = 0
                        while time.time() - start < args.power_seconds:
                            for unit in UNITS:
                                graphs[unit].replay()
                            torch.cuda.synchronize()
                            replays += 1
                        end = time.time()
                        window = power.window(start, end, settle_s=args.power_settle_seconds)
                        flop = sum(2 * m * report["units"][u]["shape"][0] * report["units"][u]["shape"][1]
                                   for u in UNITS)
                        window.update(three_unit_applies=replays, elapsed_s=end - start,
                                      wall_ms_per_three_unit_apply=1000.0 * (end - start) / replays)
                        if window.get("status") == "observed":
                            flops = replays * flop / (end - start)
                            window["achieved_tflop_s"] = flops / 1e12
                            window["gflop_per_joule"] = flops / window["mean_w"] / 1e9
                        entry["sustained"] = window
                        point["arms"][name] = entry
                        print(json.dumps({"mode": "graph", "m": m, "arm": name,
                                          "median_of_sums_ms": entry["median_of_sums_ms"],
                                          "low_ms": entry["low_ms"], "high_ms": entry["high_ms"],
                                          "bit_equal": all(c["bit_equal"] for c in checks.values()),
                                          "mean_w": window.get("mean_w")}), flush=True)
                        graphs.clear()
                        outputs.clear()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    del xs
                if "route" in point["arms"]:
                    route_ms = point["arms"]["route"]["median_of_sums_ms"]
                    point["ratios"] = {name: {"vs_route": e["median_of_sums_ms"] / route_ms}
                                       for name, e in point["arms"].items()}
                report["graph_points"][str(m)] = point
                torch.cuda.empty_cache()

            # Route telemetry is read once more so the record names the symbol the route emitted.
            report["route_records"] = {unit: read_route(cells[(unit, FP4_FORMAT)]["prepared"]["layer"])
                                       for unit in UNITS}

            # -- compile: what Dynamo does with each arm's forward ------------
            unit = UNITS[0]
            compile_arms = [n for n in args.compile_arms.split(",") if n and n in arms]
            for name in compile_arms:
                fmt = FP8_FORMAT if name == "fp8" else FP4_FORMAT
                call = bound[name][unit]
                for m in compile_rows:
                    key = f"{name}:M{m}"
                    entry = {}
                    x = input_rows(cells[(unit, fmt)]["tensors"]["prefill.input"], m)
                    try:
                        with torch.inference_mode():
                            eager = call(x).clone()
                            torch.cuda.synchronize()
                    except Exception as exc:  # noqa: BLE001
                        report["compile_check"][key] = {"eager_error": error_text(exc)}
                        continue
                    try:
                        torch._dynamo.reset()
                        with torch.inference_mode():
                            explanation = torch._dynamo.explain(call)(x)
                        entry["graph_count"] = int(explanation.graph_count)
                        entry["graph_break_count"] = int(explanation.graph_break_count)
                        entry["break_reasons"] = [str(getattr(reason, "reason", reason))[:400]
                                                  for reason in explanation.break_reasons]
                        entry["op_count"] = int(getattr(explanation, "op_count", -1))
                    except Exception as exc:  # noqa: BLE001
                        entry["explain_error"] = error_text(exc)
                    try:
                        torch._dynamo.reset()
                        compiled = torch.compile(call, dynamic=False, fullgraph=True)
                        with torch.inference_mode():
                            got = compiled(x)
                            torch.cuda.synchronize()
                        entry["fullgraph"] = numerics(torch, got, eager)
                        del got
                    except Exception as exc:  # noqa: BLE001
                        entry["fullgraph"] = {"error": error_text(exc)}
                        compiled = None
                    if compiled is not None:
                        try:
                            with torch.inference_mode():
                                graph, out = capture(compiled, x)
                                out.zero_()
                                graph.replay()
                                torch.cuda.synchronize()
                                entry["compiled_graph"] = numerics(torch, out, eager)
                            del graph, out
                        except Exception as exc:  # noqa: BLE001
                            entry["compiled_graph"] = {"error": error_text(exc)}
                    try:
                        torch._dynamo.reset()
                        with torch.inference_mode():
                            graph, out = capture(call, x)
                            out.zero_()
                            graph.replay()
                            torch.cuda.synchronize()
                            entry["eager_graph"] = numerics(torch, out, eager)
                        del graph, out
                    except Exception as exc:  # noqa: BLE001
                        entry["eager_graph"] = {"error": error_text(exc)}
                    report["compile_check"][key] = entry
                    print(json.dumps({"mode": "compile", "arm": name, "m": m,
                                      "graph_breaks": entry.get("graph_break_count"),
                                      "fullgraph_bit_equal": entry.get("fullgraph", {}).get("bit_equal"),
                                      "eager_graph_bit_equal": entry.get("eager_graph", {}).get("bit_equal")},
                                     default=str), flush=True)
                    del eager, x
                    torch.cuda.empty_cache()
            torch._dynamo.reset()

            # Profiler last: nothing timed runs under this instrumentation.
            from torch.profiler import ProfilerActivity, profile
            directory = args.profile_dir or args.out.parent / "profile"
            directory.mkdir(parents=True, exist_ok=True)

            def profile_calls(call, label, replays=5):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(replays):
                        out = call()
                        del out
                    torch.cuda.synchronize()
                events = prof.key_averages()
                kernels = sorted(({"name": e.key, "calls": e.count, "self_device_us": e.self_device_time_total}
                                  for e in events if e.device_type == torch.autograd.DeviceType.CUDA),
                                 key=lambda row: -row["self_device_us"])
                host_ops = sorted(({"name": e.key, "calls": e.count, "self_cpu_us": e.self_cpu_time_total}
                                   for e in events if e.device_type == torch.autograd.DeviceType.CPU),
                                  key=lambda row: -row["self_cpu_us"])[:12]
                trace = directory / f"{label}.trace.json"
                prof.export_chrome_trace(str(trace))
                return {"replays": replays, "kernels": kernels, "host_ops": host_ops,
                        "total_self_device_us": sum(k["self_device_us"] for k in kernels),
                        "total_self_cpu_us": sum(h["self_cpu_us"] for h in host_ops),
                        "trace_file": trace.name, "trace_sha256": digest(trace)}

            for m in rows:
                for name in arms:
                    fmt = FP8_FORMAT if name == "fp8" else FP4_FORMAT
                    for unit in UNITS:
                        with torch.inference_mode():
                            x = input_rows(cells[(unit, fmt)]["tensors"]["prefill.input"], m)
                            call = bound[name][unit]
                            for _ in range(3):
                                call(x)
                            torch.cuda.synchronize()
                            record = profile_calls(lambda: call(x),
                                                   f"{name}.{unit.rsplit('.', 1)[-1]}.M{m}")
                            del x
                        report["profile"].setdefault(str(m), {}).setdefault(name, {})[unit] = record
                        torch.cuda.empty_cache()

            # One graph-replay profile per arm at the smallest captured M: what a
            # decode replay actually launches.
            if graph_rows:
                m = min(graph_rows)
                for name in arms:
                    fmt = FP8_FORMAT if name == "fp8" else FP4_FORMAT
                    for unit in UNITS:
                        try:
                            with torch.inference_mode():
                                x = input_rows(cells[(unit, fmt)]["tensors"]["prefill.input"], m)
                                graph, out = capture(bound[name][unit], x)
                                record = profile_calls(graph.replay,
                                                       f"graph.{name}.{unit.rsplit('.', 1)[-1]}.M{m}")
                                del graph, out, x
                        except Exception as exc:  # noqa: BLE001
                            record = {"error": error_text(exc)}
                        report["profile"].setdefault(f"graph_M{m}", {}).setdefault(name, {})[unit] = record
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
    finally:
        power.stop()

    report["power"].update(gpu_uuid=power.uuid, error=power.error, series_samples=len(power.samples))
    report["finished_utc"] = utc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.out.parent / (args.out.stem + "-power-series.json")).write_text(
        json.dumps({"gpu_uuid": power.uuid, "samples": power.samples}) + "\n")
    print(json.dumps({"artifact": str(args.out), "sha256": digest(args.out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
