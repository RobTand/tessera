"""Dense E4M3 modules through the served path, against their floors (tessera#750 WP2).

Each module is encoded the way the exporter writes it (``export.encode_linear_planes``
per role, ``fused.pack_fused``), parsed from its blob and prepared by
``native_window.prepare_dense_native_module`` -- the load path a serve runs -- so
the lane the module takes is the lane a serve would take, and it is recorded.
The timed call is ``module.apply`` on the route's own per-token FP8 activation
(``native_fp8_quant``), and separately the quantizer plus ``apply``.

Modules (TP2 per-rank shapes of GLM-5.3-Flash):

- ``kda_in``: the KDA input module ``in_proj_qkvbfg_a``: q, k, v at 4096 rows,
  b 32, f_a 64, g_a 64 (12448 rows) over K = 4096.
- ``o_proj``: KDA ``o_proj``, one 4096-row role over K = 4096 (the N % 128
  control: a shape the N-tail change leaves on the same launch).
- One role each, for a K-split sweep of one geometry per launch: KDA
  ``q_proj`` (4096 x 4096), ``b_proj`` (32 x 4096), ``f_a_proj`` (64 x 4096),
  ``f_b_proj`` (4096 x 128); MLA ``q_a_proj`` (1536 x 4096),
  ``kv_a_proj_with_mqa`` (512 x 4096), ``q_b_proj`` (8192 x 1536), read from
  ``--mla-layer``.

Weights: ``--model DIR`` encodes the real GLM-5.3 bytes of ``--layer L``'s
tensors (the TP2 rank-0 shard: the leading rows of a column-parallel role, the
leading columns of the row-parallel ``o_proj``); without it, seeded Gaussian
weights.  Before timing, each module group records a numerics check: the fused
lane against the Triton lane on the same prepared wire (both decode the same
bytes, so the difference is accumulation order), each lane against the source
``F.linear`` in fp32, and fused determinism.

References at each module's whole shape: bf16 ``F.linear`` (the BF16 source
passthrough a serve runs today) and ``torch._scaled_mm`` FP8 row-wise.

Floors: bytes (the wire at q256 bits per 256 weights plus the fp32 row scales,
the activation and the bf16 output) over the measured read rate, and FLOPs
over the E4M3 ``mma.sync`` peak; the floor is the larger.  ``copy_frac`` is
the byte floor over the time (the WP2 M = 1 criterion: 0.90 or more);
``roof_frac`` is the floor over the time (the M >= 512 criterion: within 20%).

Each cell is timed as a CUDA-graph replay in a forward pass over the cell list
and again in a reverse pass; the cell's time is the mean of the two medians.
A replayed module stays in the 24 MB L2 when it fits, so the default (``--l2
warm``) times decode from L2.  ``--l2 cold`` (or ``warm,cold``) also times
``apply_cold``: a read of four L2s before each replay evicts the module, as a
served forward finds it after the rest of the model has passed through.
The forward pass also records torch.profiler device time per kernel and, at
``--power-ms``, NVML board power over a back-to-back loop, with unix times for
the Netdata series.

Usage: bench_dense_module.py --out DIR [--modules kda_in,o_proj] [--q256 1024,1088]
       [--ms 1,2,4,8,16,64,512,2048,8192] [--refs] [--model DIR --layer L]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_t8r import ENVELOPE_W, PowerSampler, kernel_profile, time_events  # noqa: E402

READ_GBPS = 232.2            # GB10 measured read rate (docs/measurements/2026-09-30-fp8-prefill-roofline.md)
MMA_E4M3_TFLOPS = 246.5      # sm_121 mma.sync E4M3 peak (same note)
MODULES = {
    "kda_in": ([("q_proj", 4096), ("k_proj", 4096), ("v_proj", 4096), ("b_proj", 32),
                ("f_a_proj", 64), ("g_a_proj", 64)], 4096),
    "o_proj": ([("o_proj", 4096)], 4096),
    # One role each, so a K-split sweep times one geometry per launch.
    "q_proj": ([("q_proj", 4096)], 4096),
    "b_proj": ([("b_proj", 32)], 4096),
    "f_a_proj": ([("f_a_proj", 64)], 4096),
    "f_b_proj": ([("f_b_proj", 4096)], 128),
    # MLA (a full-attention layer, ``--mla-layer``).
    "q_a_proj": ([("q_a_proj", 1536)], 4096),
    "kv_a_proj_with_mqa": ([("kv_a_proj_with_mqa", 512)], 4096),
    "q_b_proj": ([("q_b_proj", 8192)], 1536),
}
MLA_MODULES = {"q_a_proj", "kv_a_proj_with_mqa", "q_b_proj"}
SOURCE_PREFIX = "model.language_model.layers.{layer}.self_attn.{name}.weight"


def source_weight(model, layer, name, rows, cols):
    """The TP2 rank-0 shard of one real source tensor, bf16 on the GPU."""
    from safetensors import safe_open

    key = SOURCE_PREFIX.format(layer=layer, name=name)
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(model, index[key]), framework="pt", device="cuda") as fh:
        w = fh.get_tensor(key)
    if w.shape[0] < rows or w.shape[1] < cols:
        raise ValueError(f"{key} is {tuple(w.shape)}; the shard needs [{rows}, {cols}]")
    return w[:rows, :cols].to(torch.bfloat16).contiguous()


def graph_time(call, warmup, iters):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    samples = time_events(graph.replay, warmup, iters)
    del graph
    return samples


def cold_scratch():
    """A read-only buffer of four L2s (at least 64 MB): reading it before a
    timed call leaves the call's weights out of L2, as a served forward
    finds them after the rest of the model has passed through.  Read, not
    written, so the timed call writes back no dirty lines of it."""
    l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or (24 << 20))
    return torch.ones(max(4 * l2, 64 << 20) // 4, dtype=torch.float32, device="cuda")


def graph_time_cold(call, warmup, iters, scratch):
    """``graph_time`` with L2 cleared before each replay (``cold_scratch``);
    the events bracket the replay alone."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(warmup):
        scratch.sum()
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        scratch.sum()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        out.append(float(a.elapsed_time(b)))
    del graph
    return out


def kernel_profile_cold(call, scratch, reps=3):
    """``kernel_profile`` with L2 cleared before each call; the clearing
    read's own kernels (profiled alone first) are left out."""
    alone = kernel_profile(lambda: scratch.sum(), reps=1)["top"]
    prof = kernel_profile(lambda: (scratch.sum(), call()), reps=reps)
    top = {k: v for k, v in prof["top"].items() if k not in alone}
    return {"kernel_us_per_call": sum(v["us_per_call"] for v in top.values()),
            "launches_per_call": prof["launches_per_call"] - len(alone), "top": top}


def encode_module(roles, cols, q256, seed, source=None):
    """The served module: encode each role on the E4M3 grid, pack, parse, prepare.
    ``source(name, rows)`` gives a role's real weight; otherwise seeded Gaussian."""
    from tessera import export, fused
    from tessera.alphabet import E4M3_GRID
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import (TESSERA_FP8, parse_compact_blob_for_scheme,
                                        validate_tessera_scheme)
    from tessera.serving.sharding import plan_shard

    torch.manual_seed(seed)
    blobs = []
    weights = []
    t0 = time.time()
    for name, rows in roles:
        w = (source(name, rows).float() if source is not None
             else torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
        exported, _unit, _forests = export.encode_linear_planes(w, grid=E4M3_GRID, q256=q256, name=name,
                                                                verify=False)
        blobs.append((name, rows, exported.blob))
        weights.append(w.bfloat16())
        del w
    blob = fused.pack_fused(blobs)
    rows = sum(r for _, r in roles)
    scheme = {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL", "q256": q256,
              "rows": rows, "columns": cols, "wire_bytes": len(blob), "roles": [[n, r] for n, r in roles]}
    declared = validate_tessera_scheme(scheme, "bench")
    plan = plan_shard("bench", roles=[(n, r) for n, r in roles], columns=cols,
                      out_partitions=[r for _, r in roles], in_size=cols, tp_rank=0, tp_size=1,
                      input_size=cols, output_size=rows)

    def prepare(fused_lane):
        prev = os.environ.get("TESSERA_DENSE_FUSED")
        os.environ["TESSERA_DENSE_FUSED"] = "1" if fused_lane else "0"
        try:
            compact = parse_compact_blob_for_scheme(blob, scheme, "bench", device="cuda")
            return prepare_dense_native_module(compact, plan, family=declared["family"], device="cuda")
        finally:
            if prev is None:
                os.environ.pop("TESSERA_DENSE_FUSED", None)
            else:
                os.environ["TESSERA_DENSE_FUSED"] = prev
    return prepare, len(blob), time.time() - t0, torch.cat(weights)


def numerics(lanes, w_src, cols, ms, seed):
    """Fused vs Triton on one prepared wire, each vs the source ``F.linear`` (fp32),
    and fused determinism, at each M."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    from tessera.serving.native_ops import native_fp8_quant

    out = {}
    for m in ms:
        x = (torch.randn(m, cols, device="cuda", generator=g) * 0.5).bfloat16()
        xq, a = native_fp8_quant(x)
        a = a.reshape(-1).contiguous().float()
        ref = torch.nn.functional.linear(x.float(), w_src.float())
        got = {k: v.apply(xq, a).float() for k, v in lanes.items()}
        rec = {}
        for k, y in got.items():
            rec[f"{k}_vs_source_rel_fro"] = float((y - ref).norm() / ref.norm())
        if "fused" in got and "triton" in got:
            d = got["fused"] - got["triton"]
            rec["fused_vs_triton_rel_fro"] = float(d.norm() / got["triton"].norm())
            rec["fused_vs_triton_max_rel"] = float(d.abs().max() / got["triton"].abs().max())
        if "fused" in lanes:
            rec["fused_deterministic"] = bool(torch.equal(lanes["fused"].apply(xq, a).float(), got["fused"]))
        out[str(m)] = rec
        print(json.dumps({"numerics_m": m, **{k: (round(v, 8) if isinstance(v, float) else v)
                                               for k, v in rec.items()}}), flush=True)
    torch.cuda.synchronize()
    return out


def floors(wire_bytes, m, rows, cols, out_bytes=2, a_bytes=1):
    bytes_ = wire_bytes + m * cols * a_bytes + m * rows * out_bytes
    byte_ms = bytes_ / (READ_GBPS * 1e9) * 1e3
    flop_ms = 2.0 * m * rows * cols / (MMA_E4M3_TFLOPS * 1e12) * 1e3
    return {"bytes": bytes_, "byte_ms": byte_ms, "flop_ms": flop_ms, "floor_ms": max(byte_ms, flop_ms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modules", default="kda_in,o_proj")
    ap.add_argument("--q256", default="1024,1088")
    ap.add_argument("--ms", default="1,2,4,8,16,64,512,2048,8192")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--lanes", default="fused,triton", help="which lanes of each module to time")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--power-ms", default="1,512,8192")
    ap.add_argument("--power-s", type=float, default=0.5)
    ap.add_argument("--ncu", action="store_true",
                    help="one apply per (module, lane, M) between cudaProfilerStart/Stop; no timing")
    ap.add_argument("--model", default=None, help="source checkpoint: encode its real bytes")
    ap.add_argument("--layer", type=int, default=1, help="the KDA layer read from --model")
    ap.add_argument("--mla-layer", type=int, default=3, help="the MLA layer the MLA modules read")
    ap.add_argument("--numerics-ms", default="1,64,2048")
    ap.add_argument("--l2", default="warm",
                    help="warm, cold or warm,cold: cold also times apply with L2 cleared before "
                         "each replay (call apply_cold), the state a served forward finds a module in")
    ap.add_argument("--k-splits", default="",
                    help="measurement only: also time the fused lane at these fixed K splits "
                         "(lane fused@S<s>; the launch's own model picks S on lane fused)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from tessera import routed_fused as rf
    from tessera.serving.native_ops import native_fp8_quant, require_native_fp8_quant
    require_native_fp8_quant("bench_dense_module")

    dev = torch.device("cuda")
    ms = [int(v) for v in args.ms.split(",")]
    power_ms = {int(v) for v in args.power_ms.split(",") if v}
    l2_modes = {v for v in args.l2.split(",") if v}
    if not l2_modes <= {"warm", "cold"} or not l2_modes:
        raise SystemExit(f"--l2 takes warm, cold or both, not {args.l2!r}")
    scratch = cold_scratch() if "cold" in l2_modes else None
    power = PowerSampler()
    meta = {"device": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
            "library": rf.library_for("e4m3"), "read_gbps": READ_GBPS, "mma_tflops": MMA_E4M3_TFLOPS,
            "kernel_sha": os.environ.get("KERNEL_SHA"), "tessera_head": os.environ.get("TESSERA_HEAD"),
            "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "power_source": power.source,
            "envelope_w": ENVELOPE_W, "start_unix": time.time(), "torch": torch.__version__,
            "dense_row_quantum": getattr(rf, "DENSE_ROW_QUANTUM", None),
            "weights": ({"model": args.model, "layer": args.layer, "mla_layer": args.mla_layer,
                         "shard": "TP2 rank 0"} if args.model
                        else "seeded Gaussian"),
            "statistic": "mean of the forward and reverse passes' medians (graph replay); spread = |F - R| / mean",
            "l2": sorted(l2_modes),
            "cold_scratch_bytes": (scratch.numel() * 4 if scratch is not None else 0)}
    groups = []          # (key, builder) -> builder() returns (head, make) with make(m) -> (meta, call)
    for name in args.modules.split(","):
        roles, cols = MODULES[name]
        rows = sum(r for _, r in roles)
        for q in (int(v) for v in args.q256.split(",")):
            groups.append((f"{name}:q{q}", ("module", name, roles, cols, rows, q)))
        if args.refs:
            groups.append((f"{name}:bf16_linear", ("bf16", name, roles, cols, rows, None)))
            groups.append((f"{name}:scaled_mm", ("fp8", name, roles, cols, rows, None)))
    cells = {}
    path = os.path.join(args.out, "bench_dense_module.json")

    def save():
        json.dump({"meta": meta, "groups": cells}, open(path, "w"), indent=1)

    def ref_calls(call):
        calls = {"apply": call} if "warm" in l2_modes else {}
        if "cold" in l2_modes:
            calls["apply_cold"] = call
        return calls

    def build(spec):
        kind, name, roles, cols, rows, q = spec
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"{name}:{kind}:{q}".encode()))
        if kind == "module":
            layer = args.mla_layer if name in MLA_MODULES else args.layer
            source = ((lambda n, r: source_weight(args.model, layer, n, r, cols)) if args.model else None)
            prepare, blob_bytes, enc_s, w_src = encode_module(roles, cols, q, zlib.crc32(f"{name}:{q}".encode()),
                                                              source)
            lanes = {}
            for lane in args.lanes.split(","):
                mod = prepare(lane == "fused")
                lanes[lane] = mod
            head = {"kind": kind, "module": name, "q256": q, "rows": rows, "cols": cols, "roles": roles,
                    "blob_bytes": blob_bytes, "encode_s": enc_s,
                    "lanes": {k: {"lane": v.lane, "reason": v.lane_reason, "launch_pair": list(v.launch_pair)}
                              for k, v in lanes.items()}}
            head["numerics"] = numerics(lanes, w_src, cols,
                                        [int(v) for v in args.numerics_ms.split(",") if v],
                                        zlib.crc32(f"num:{name}:{q}".encode()))
            del w_src
            wire = sum(r * cols * q // 256 // 8 + 4 * r for _, r in roles)
            head["wire_bytes"] = wire

            def make(m, lane):
                base, _, forced = lane.partition("@S")
                mod = lanes[base]
                x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).bfloat16()
                xq, a = native_fp8_quant(x)
                a = a.reshape(-1).contiguous().float()
                holder = {}

                def split(fn):
                    if not forced:
                        return fn

                    def at_split():
                        # the split is read at capture; replay runs what was captured
                        model = rf.dense_k_split
                        rf.dense_k_split = lambda m_, rows_, cols_, sms_, tile_words=None: min(
                            int(forced), cols_ // rf.BK)
                        try:
                            fn()
                        finally:
                            rf.dense_k_split = model
                    return at_split

                @split
                def apply():
                    holder["out"] = mod.apply(xq, a)

                @split
                def quant_apply():
                    q8, s8 = native_fp8_quant(x)
                    holder["out"] = mod.apply(q8, s8.reshape(-1))
                calls = {"apply": apply, "quant_apply": quant_apply}
                if "cold" in l2_modes:
                    calls["apply_cold"] = apply
                if "warm" not in l2_modes:
                    calls.pop("apply")
                    calls.pop("quant_apply")
                return {"floor": floors(wire, m, rows, cols)}, calls, holder
            extra = [f"fused@S{v}" for v in args.k_splits.split(",") if v] if "fused" in lanes else []
            return head, make, list(lanes) + extra
        if kind == "bf16":
            w = (torch.randn(rows, cols, device=dev, generator=g) * 0.02).bfloat16()
            head = {"kind": kind, "module": name, "rows": rows, "cols": cols, "wire_bytes": rows * cols * 2}

            def make(m, _lane):
                x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).bfloat16()
                holder = {}

                def call():
                    holder["out"] = torch.nn.functional.linear(x, w)
                return {"floor": floors(rows * cols * 2, m, rows, cols, a_bytes=2)}, ref_calls(call), holder
            return head, make, ["bf16"]
        w8 = (torch.randn(rows, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
        sw = torch.rand(1, rows, device=dev, generator=g) * 1e-2 + 1e-3
        head = {"kind": kind, "module": name, "rows": rows, "cols": cols, "wire_bytes": rows * cols + 4 * rows}

        def make(m, _lane):
            x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
            sa = torch.rand(m, 1, device=dev, generator=g) * 0.1 + 0.01
            holder = {}

            def call():
                holder["out"] = torch._scaled_mm(x, w8.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16)
            return {"floor": floors(rows * cols + 4 * rows, m, rows, cols)}, ref_calls(call), holder
        return head, make, ["scaled_mm"]

    built = {}
    for pas in (("F",) if args.ncu else ("F", "R")):
        order = groups if pas == "F" else list(reversed(groups))
        for key, spec in order:
            if key not in built:
                built[key] = build(spec)
                cells[key] = dict(built[key][0], cells={})
            head, make, lanes = built[key]
            seq = [(m, lane) for lane in lanes for m in ms]
            for m, lane in (seq if pas == "F" else list(reversed(seq))):
                ckey = f"{lane}:{m}"
                cell = cells[key]["cells"].setdefault(ckey, {})
                try:
                    cmeta, calls, holder = make(m, lane)
                    if args.ncu:
                        # ncu --profile-from-start off: exactly one profiled apply per cell.
                        for _ in range(3):
                            calls["apply"]()
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStart()
                        calls["apply"]()
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                        cell["ncu"] = True
                        print(json.dumps({"g": key, "cell": ckey, "ncu": True}), flush=True)
                        del calls, holder
                        continue
                    for cname, call in calls.items():
                        cold = cname.endswith("_cold")
                        samples = (graph_time_cold(call, args.warmup, args.iters, scratch) if cold
                                   else graph_time(call, args.warmup, args.iters))
                        rec = cell.setdefault(cname, {})
                        rec[pas] = {"median_ms": statistics.median(samples), "min_ms": min(samples),
                                    "unix": time.time()}
                        if pas == "F":
                            rec["floor"] = cmeta["floor"]
                            if cname == "apply":
                                rec["profile"] = kernel_profile(call, reps=3)
                            elif cold:
                                rec["profile"] = kernel_profile_cold(call, scratch, reps=3)
                                if m in power_ms:
                                    rec["power"] = power.sample_during(call, args.power_s)
                        else:
                            f, r = rec.get("F", {}).get("median_ms"), rec["R"]["median_ms"]
                            if f:
                                rec["ms"] = 0.5 * (f + r)
                                rec["spread"] = abs(f - r) / rec["ms"]
                                fl = rec["floor"]
                                rec["copy_frac"] = fl["byte_ms"] / rec["ms"]
                                rec["roof_frac"] = fl["floor_ms"] / rec["ms"]
                                print(json.dumps({"g": key, "cell": ckey, "call": cname,
                                                  "ms": round(rec["ms"], 4), "spread": round(rec["spread"], 4),
                                                  "copy": round(rec["copy_frac"], 3),
                                                  "roof": round(rec["roof_frac"], 3)}), flush=True)
                    del calls, holder
                except Exception as exc:  # noqa: BLE001
                    cell.setdefault("error", {})[pas] = repr(exc)[:500]
                    print(json.dumps({"g": key, "cell": ckey, "pass": pas, "error": repr(exc)[:300]}), flush=True)
            save()
    meta["end_unix"] = time.time()
    save()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
