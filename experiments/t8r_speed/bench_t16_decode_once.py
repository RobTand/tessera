"""Measure per-step T16 decode and the same-wire BF16 control."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tests"))
from bench_rates import build_projection
from bench_dense_module import cold_scratch, graph_time_cold
from bench_t8r import PowerSampler, kernel_profile
import fused_bound as fb

MS = (16, 2048, 4096)
THRESHOLDS = {"go_prefill_ratio": 1.15, "kill_prefill_ratio": 1.5}
WIRE_SEED = 1301
INPUT_SEED = 1907


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def shape(config, tp):
    text = config["text_config"]
    linear = text["linear_attn_config"]
    heads, dim = int(linear["num_heads"]), int(linear["head_dim"])
    if tp <= 0 or heads % tp:
        raise ValueError("The head count must divide by the TP size.")
    roles = [(name, heads * dim // tp) for name in ("q", "k", "v")]
    roles += [("b", heads // tp), ("f_a", dim), ("g_a", dim)]
    return roles, int(text["hidden_size"])


def require_shape(config, tp):
    roles, cols = shape(config, tp)
    if tp != 2 or roles != [("q", 4096), ("k", 4096), ("v", 4096),
                            ("b", 32), ("f_a", 128), ("g_a", 128)] or cols != 4096:
        raise ValueError("The experiment requires the TP2 KDA shape 12576 by 4096.")
    return roles, cols


def paired_stats(passes):
    if set(passes) != {"F", "R"}:
        raise ValueError("Both arm orders are required.")
    values = []
    for order in ("F", "R"):
        samples = passes[order]["samples_ms"]
        if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
            raise ValueError("Time samples must be finite and positive.")
        values.append(statistics.median(samples))
    f, r = values
    mean = (f + r) / 2
    return {"forward_median_ms": f, "reverse_median_ms": r,
            "mean_ms": mean, "spread_fraction": abs(f - r) / mean}


def screen_ratio(ratio):
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("The full-lane ratio must be finite and positive.")
    if ratio <= THRESHOLDS["go_prefill_ratio"]:
        return "GO"
    if ratio > THRESHOLDS["kill_prefill_ratio"]:
        return "KILL"
    return "INCONCLUSIVE"


def reference_weight(p, rows, cols, device):
    """Read the MSB-first wire with independent CPU integer operations."""
    words = p["words"][0].cpu().to(torch.int64) & 0xFFFFFFFF
    init = p["init"][0].cpu().to(torch.int64)
    table = p["table"][0].cpu().view(torch.bfloat16)
    scale = p["scale"][0].cpu()
    perm = p["perm"][0].cpu().long()
    runs = p["runs"][0].cpu().reshape(-1, 4)
    bits = int(p["window_bits"])
    result = torch.empty(rows, cols, dtype=torch.bfloat16, device=device)
    for n0 in range(0, rows, 128):
        n1 = min(rows, n0 + 128)
        n = torch.arange(n0, n1, dtype=torch.int64)
        tile = n // 512
        block = torch.empty(n1 - n0, cols, dtype=torch.bfloat16)
        for rate, c0, count, offset in runs.tolist():
            if count == 0:
                continue
            columns = torch.arange(count, dtype=torch.int64)
            chunk_words = 16 * rate
            end = (n % 512 + 1) * rate
            first = (end - bits) // 32
            base = tile[:, None] * p["tile_words"] + offset + columns[None, :] * chunk_words
            history = first[:, None] < 0
            previous = base - p["tile_words"] + chunk_words - 1
            index0 = torch.where(history, previous, base + first[:, None])
            w0 = words[index0.clamp(0, words.numel() - 1)]
            w0 = torch.where(history & (tile[:, None] == 0), init[None, c0:c0 + count], w0)
            w1 = words[base + (first + 1)[:, None]]
            shift = (64 - end + 32 * first)[:, None]
            states = (((w0 << 32) | w1) >> shift) & ((1 << bits) - 1)
            values = (table[states].float() * scale[n0:n1, None]).bfloat16()
            block[:, perm[c0:c0 + count]] = values
        result[n0:n1].copy_(block)
    return result


def runtime_imports():
    from tessera import routed_fused as rf
    from tessera.window_gemm import PreparedWindowGemm
    from tessera.serving.native_window import PreparedDenseNativeModule
    from tessera.serving.bf16_prefill import prepare_bf16_prefill, prefill_apply, decode_into
    from tessera.serving.scheme import BF16_DECODE_ONCE_DENSE_SYMBOL
    from tessera.serving.telemetry import DECODER_NATIVE_WINDOW_DECODE_ONCE_BF16_FOLDED
    return (rf, PreparedWindowGemm, PreparedDenseNativeModule, prepare_bf16_prefill,
            prefill_apply, decode_into, BF16_DECODE_ONCE_DENSE_SYMBOL,
            DECODER_NATIVE_WINDOW_DECODE_ONCE_BF16_FOLDED)


def make_wire(rows, cols, device, owners):
    rf, Bundle, Module = owners[:3]
    p = build_projection(rf, 1, rows, cols, 8, 0, WIRE_SEED, device,
                         False, bf16_table=True)
    empty = torch.empty(0, dtype=torch.uint8, device=device)
    bundle = Bundle(words=p["words"][0], table=p["table"][0].view(torch.bfloat16),
                    codes=empty, native=empty, scale=p["scale"][0],
                    runs=p["runs"][0, :4].reshape(1, 4), init_perm=p["init"][0],
                    perm=p["perm"][0], tile_words=p["tile_words"],
                    total_words=p["words"].shape[1], rows=rows, cols=cols,
                    window_bits=p["window_bits"], family="value", has_init=True,
                    block_m=64, block_n=64, block_k=64, arithmetic="folded")
    control = reference_weight(p, rows, cols, device)
    if device == "cpu":
        return p, bundle, control, None
    role = SimpleNamespace(name="merged_kda", rows=rows, bundle=bundle, facts=None)
    module = Module([role], rows=rows, columns=cols, device=torch.device(device),
                    family="value", lane="fused", fused_roles=[rf.prepare_dense_role(bundle)])
    return p, bundle, control, module


def at_split(module, x, split):
    from tessera import routed_fused as rf
    previous = rf.dense_k_split
    rf.dense_k_split = lambda *args, **kwargs: split
    try:
        return module.apply(x)
    finally:
        rf.dense_k_split = previous


def check_numeric(call, x, control, name):
    """Check every output with bounded fp64 reference blocks."""
    got = call()
    if got.dtype != torch.bfloat16 or got.shape != (x.shape[0], control.shape[0]):
        raise AssertionError(f"{name}: the output shape or dtype differs.")
    worst = 0.0
    k = x.shape[1]
    for m0 in range(0, x.shape[0], 128):
        a64 = x[m0:m0 + 128].double()
        for n0 in range(0, control.shape[0], 256):
            w64 = control[n0:n0 + 256].double()
            ref, bound = fb.dense_bound("value", a64, w64, k, s=k)
            worst = max(worst, fb.check_within(got[m0:m0 + 128, n0:n0 + 256],
                                             ref, bound, name))
    return {"worst_ratio": worst, "elements": got.numel(),
            "reference_block": [128, 256], "bound_split_charge": k}


def check_one_hot(module, prepared, control, apply):
    """Check each original column without a full identity matrix."""
    rows, cols = control.shape
    for c0 in range(0, cols, 64):
        count = min(64, cols - c0)
        x = torch.zeros(count, cols, device="cuda", dtype=torch.bfloat16)
        x[torch.arange(count, device="cuda"), torch.arange(c0, c0 + count, device="cuda")] = 1
        expected = control[:, c0:c0 + count].t()
        for name, got in (("window", at_split(module, x, 1)),
                          ("decode_once", apply(prepared, x)),
                          ("control", torch.nn.functional.linear(x, control))):
            if not torch.equal(got, expected):
                raise AssertionError(f"{name}: one-hot decoded values differ at column {c0}.")
    return {"exact": True, "columns": cols, "rows": rows, "block_m": 64}


def unique_storage(tensors):
    storage = {}
    for name, tensor in tensors:
        raw = tensor.untyped_storage()
        key = (str(tensor.device), raw.data_ptr())
        entry = storage.setdefault(key, {"bytes": raw.nbytes(), "names": []})
        entry["names"].append(name)
    return {"physical_bytes": sum(v["bytes"] for v in storage.values()),
            "storages": list(storage.values())}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(HERE / "t16_decode_once_config.json"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--ms", default="16,2048,4096")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--pilot-iters", type=int, default=20)
    ap.add_argument("--power-s", type=float, default=1.0)
    ap.add_argument("--cpu-preflight", action="store_true")
    args = ap.parse_args(argv)
    if tuple(int(v) for v in args.ms.split(",")) != MS:
        raise ValueError("The experiment requires M=16,2048,4096.")
    if (args.warmup, args.iters, args.pilot_iters, args.power_s) != (10, 80, 20, 1.0):
        raise ValueError("The experiment requires the prior protocol sample counts and power interval.")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "t16_decode_once.json"
    config_bytes = Path(args.config).read_bytes()
    roles, cols = require_shape(json.loads(config_bytes), args.tp)
    rows = sum(n for _, n in roles)
    owners = runtime_imports()
    rf, _Bundle, _Module, prepare, apply, decode = owners[:6]
    result = {"schema": "tessera.t16_decode_once.v1", "thresholds": THRESHOLDS,
              "shape": {"rows": rows, "cols": cols, "tp": args.tp, "roles": roles},
              "scope": "Synthetic packed WINDOW wire. One merged role. No checkpoint or served qualification.",
              "statistic": "Mean of both order medians. Spread is abs(F-R)/mean.",
              "cache": "Cold L2 before each graph replay. CUDA events exclude eviction.",
              "decode_cost": "The full lane includes a direct scratch decode on every step.",
              "graph_scope": "Each graph uses one serving stream. Concurrent graph replay is not qualified.",
              "protocol": {"ms": MS, "warmup": args.warmup, "samples_per_order": args.iters,
                           "pilot_samples_per_order": args.pilot_iters, "power_seconds": args.power_s,
                           "wire_seed": WIRE_SEED, "input_seed": INPUT_SEED},
              "meta": {"start_unix": time.time(), "host": os.environ.get("HOST_NAME"),
                       "head": os.environ.get("TESSERA_HEAD"), "image": os.environ.get("ORACLE_IMAGE"),
                       "pb_action": os.environ.get("PB_ACTION_KEY"), "torch": torch.__version__,
                       "kernel_sha": os.environ.get("KERNEL_SHA"),
                       "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "launch_pair": list(owners[6:]), "correctness": {}, "pilot": {}, "cells": {}}
    if args.cpu_preflight:
        p, bundle, control, _ = make_wire(32, 128, "cpu", owners)
        if control.shape != (32, 128) or not bool(torch.isfinite(control).all()):
            raise AssertionError("The CPU reference shape or values differ.")
        if p["tile_words"] != rf.pair_tile_words(p["runs"][0]) or bundle.rows != 32:
            raise AssertionError("The CPU wire geometry differs.")
        result["cpu_preflight"] = {"passed": True, "slice": [32, 128],
                                   "imports": "The real measurement and shared scratch imports passed.",
                                   "cuda_executed": False}
        result["meta"].update(end_unix=time.time(), host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(path, result)
        print(json.dumps(result), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The experiment requires CUDA.")
    torch.cuda.reset_peak_memory_stats()
    result["meta"].update(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                          bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    p, _bundle, control, module = make_wire(rows, cols, "cuda", owners)
    prepared = prepare(module)
    if prepared.weight.shape != control.shape or prepared.weight.dtype != torch.bfloat16:
        raise AssertionError("The shared scratch shape or dtype differs.")
    decode(prepared)
    if not torch.equal(prepared.weight, control):
        raise AssertionError("The scratch differs from the independently decoded wire.")
    result["correctness"]["materialization"] = {"exact": True, "elements": rows * cols}
    result["memory"] = {"scratch_per_shape_bytes": prepared.weight.numel() * prepared.weight.element_size(),
                         "scratch_shape": [rows, cols], "scratch_pool_total_bytes": prepared.weight.untyped_storage().nbytes(),
                         "scratch_pool_shapes": [[rows, cols]],
                         "experiment_control_bytes": control.untyped_storage().nbytes(),
                         "prepared_module": unique_storage(module.named_tensors()),
                         "packed_body_bytes": p["words"].numel() * p["words"].element_size(),
                         "control_scope": "The separate same-wire control is experiment memory, not serving storage."}
    generator = torch.Generator(device="cuda").manual_seed(INPUT_SEED)
    xs = {m: (torch.randn(m, cols, device="cuda", generator=generator) * 0.5).bfloat16() for m in MS}
    max_split = rf.dense_split_max(cols)
    splits = sorted(s for s in {1, 2, 4, 8, 16, 32, max_split} if s <= max_split)
    result["split_scope"] = {"legal_max": max_split, "pilot_m16": splits, "prefill": [1],
                             "choice": "The lowest paired pilot time selects M16. Final samples are separate."}
    result["correctness"]["one_hot"] = check_one_hot(module, prepared, control, apply)
    for m, x in xs.items():
        checks = {}
        for split in splits if m == 16 else [1]:
            checks[f"window:S{split}"] = check_numeric(lambda s=split: at_split(module, x, s), x, control, f"window M{m} S{split}")
        prepared.weight.fill_(float("nan"))
        checks["per_step_refresh"] = check_numeric(lambda: apply(prepared, x), x, control, f"decode-once M{m}")
        checks["same_wire_control"] = check_numeric(lambda: torch.nn.functional.linear(x, control), x, control, f"BF16 M{m}")
        result["correctness"][f"M{m}"] = checks
        save(path, result)
    result["correctness_complete_unix"] = time.time()
    save(path, result)
    eviction = cold_scratch()
    pilot = {str(s): {} for s in splits}
    for order in ("F", "R"):
        for split in splits if order == "F" else list(reversed(splits)):
            start = time.time()
            samples = graph_time_cold(lambda s=split: at_split(module, xs[16], s), args.warmup, args.pilot_iters, eviction)
            pilot[str(split)][order] = {"samples_ms": samples, "window_unix": [start, time.time()]}
    for split in splits:
        pilot[str(split)]["paired"] = paired_stats({o: pilot[str(split)][o] for o in ("F", "R")})
    winner = min(splits, key=lambda s: (pilot[str(s)]["paired"]["mean_ms"], s))
    result["pilot"]["M16"] = {"splits": pilot, "selected_split": winner}
    calls = {}
    for m in MS:
        for arm in ("T16_R2048_window", "T16_decode_once_BF16", "cuBLAS_same_wire_BF16"):
            split = winner if m == 16 else 1
            key = f"{arm}:M{m}"
            if arm == "T16_R2048_window":
                calls[key] = lambda x=xs[m], s=split: at_split(module, x, s)
            elif arm == "T16_decode_once_BF16":
                calls[key] = lambda x=xs[m]: apply(prepared, x)
            else:
                calls[key] = lambda x=xs[m]: torch.nn.functional.linear(x, control)
            result["cells"][key] = {"arm": arm, "m": m, "split": split if arm == "T16_R2048_window" else None,
                                     "includes_step_decode": arm == "T16_decode_once_BF16", "passes": {},
                                     "launch_pair": list(module.launch_pair) if arm == "T16_R2048_window" else
                                        list(owners[6:]) if arm == "T16_decode_once_BF16" else None}
    calls["T16_decode_only"] = lambda: decode(prepared)
    result["cells"]["T16_decode_only"] = {"arm": "T16_decode_only", "diagnostic_only": True, "passes": {}}
    for order in ("F", "R"):
        keys = list(calls) if order == "F" else list(reversed(calls))
        result.setdefault("orders", {})[order] = keys
        for key in keys:
            start = time.time()
            samples = graph_time_cold(calls[key], args.warmup, args.iters, eviction)
            cell = result["cells"][key]
            cell["passes"][order] = {"samples_ms": samples, "median_ms": statistics.median(samples),
                                       "window_unix": [start, time.time()]}
            save(path, result)
    result["timing_complete_unix"] = time.time()
    power = PowerSampler()
    for key, call in calls.items():
        cell = result["cells"][key]
        cell.update(paired_stats(cell["passes"]))
        trace = out / (key.replace(":", "-") + ".trace.json")
        cell["profile"] = kernel_profile(call, reps=3, full_names=True, trace_path=trace)
        cell["trace"] = str(trace)
        cell["power"] = power.sample_during(call, args.power_s, capture_series=True)
        cell["diagnostic_scope"] = "Profiles and power use resident eager calls after the time samples. Energy remains unqualified."
        save(path, result)
    ratios = {}
    for m in MS:
        base = result["cells"][f"cuBLAS_same_wire_BF16:M{m}"]["mean_ms"]
        ratios[str(m)] = {arm: result["cells"][f"{arm}:M{m}"]["mean_ms"] / base
                          for arm in ("T16_R2048_window", "T16_decode_once_BF16")}
    ratio = ratios["2048"]["T16_decode_once_BF16"]
    result["screen"] = {"ratios": ratios, "m2048_full_lane_ratio": ratio,
                        "threshold_band": screen_ratio(ratio), "verdict_owner": "The parent applies the verdict.",
                        "scope": "This kernel screen does not qualify served KL or whole-model throughput."}
    result["memory"].update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            l2_eviction_bytes=eviction.numel() * eviction.element_size())
    result["meta"]["end_unix"] = time.time()
    save(path, result)
    save(out / "memory_receipt.json", result["memory"])
    print(json.dumps(result["screen"]), flush=True)


if __name__ == "__main__":
    main()
