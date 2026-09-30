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

References at each module's whole shape: bf16 ``F.linear`` (the BF16 source
passthrough a serve runs today) and ``torch._scaled_mm`` FP8 row-wise.

Floors: bytes (the wire at q256 bits per 256 weights plus the fp32 row scales,
the activation and the bf16 output) over the measured read rate, and FLOPs
over the E4M3 ``mma.sync`` peak; the floor is the larger.  ``copy_frac`` is
the byte floor over the time (the WP2 M = 1 criterion: 0.90 or more);
``roof_frac`` is the floor over the time (the M >= 512 criterion: within 20%).

Each cell is timed as a CUDA-graph replay in a forward pass over the cell list
and again in a reverse pass; the cell's time is the mean of the two medians.
The forward pass also records torch.profiler device time per kernel and, at
``--power-ms``, NVML board power over a back-to-back loop, with unix times for
the Netdata series.

Usage: bench_dense_module.py --out DIR [--modules kda_in,o_proj] [--q256 1024,1088]
       [--ms 1,2,4,8,16,64,512,2048,8192] [--refs]
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
}


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


def encode_module(roles, cols, q256, seed):
    """The served module: encode each role on the E4M3 grid, pack, parse, prepare."""
    from tessera import export, fused
    from tessera.alphabet import E4M3_GRID
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import (TESSERA_FP8, parse_compact_blob_for_scheme,
                                        validate_tessera_scheme)
    from tessera.serving.sharding import plan_shard

    torch.manual_seed(seed)
    blobs = []
    t0 = time.time()
    for name, rows in roles:
        w = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
        exported, _unit, _forests = export.encode_linear_planes(w, grid=E4M3_GRID, q256=q256, name=name,
                                                                verify=False)
        blobs.append((name, rows, exported.blob))
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
    return prepare, len(blob), time.time() - t0


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
    ap.add_argument("--ncu", action="store_true", help="accepted for the wrapper; not used")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from tessera import routed_fused as rf
    from tessera.serving.native_ops import native_fp8_quant, require_native_fp8_quant
    require_native_fp8_quant("bench_dense_module")

    dev = torch.device("cuda")
    ms = [int(v) for v in args.ms.split(",")]
    power_ms = {int(v) for v in args.power_ms.split(",") if v}
    power = PowerSampler()
    meta = {"device": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
            "library": rf.library_for("e4m3"), "read_gbps": READ_GBPS, "mma_tflops": MMA_E4M3_TFLOPS,
            "kernel_sha": os.environ.get("KERNEL_SHA"), "tessera_head": os.environ.get("TESSERA_HEAD"),
            "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "power_source": power.source,
            "envelope_w": ENVELOPE_W, "start_unix": time.time(), "torch": torch.__version__,
            "dense_row_quantum": getattr(rf, "DENSE_ROW_QUANTUM", None),
            "statistic": "mean of the forward and reverse passes' medians (graph replay); spread = |F - R| / mean"}
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

    def build(spec):
        kind, name, roles, cols, rows, q = spec
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"{name}:{kind}:{q}".encode()))
        if kind == "module":
            prepare, blob_bytes, enc_s = encode_module(roles, cols, q, zlib.crc32(f"{name}:{q}".encode()))
            lanes = {}
            for lane in args.lanes.split(","):
                mod = prepare(lane == "fused")
                lanes[lane] = mod
            head = {"kind": kind, "module": name, "q256": q, "rows": rows, "cols": cols, "roles": roles,
                    "blob_bytes": blob_bytes, "encode_s": enc_s,
                    "lanes": {k: {"lane": v.lane, "reason": v.lane_reason, "launch_pair": list(v.launch_pair)}
                              for k, v in lanes.items()}}
            wire = sum(r * cols * q // 256 // 8 + 4 * r for _, r in roles)
            head["wire_bytes"] = wire

            def make(m, lane):
                mod = lanes[lane]
                x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).bfloat16()
                xq, a = native_fp8_quant(x)
                a = a.reshape(-1).contiguous().float()
                holder = {}

                def apply():
                    holder["out"] = mod.apply(xq, a)

                def quant_apply():
                    q8, s8 = native_fp8_quant(x)
                    holder["out"] = mod.apply(q8, s8.reshape(-1))
                return {"floor": floors(wire, m, rows, cols)}, {"apply": apply, "quant_apply": quant_apply}, holder
            return head, make, list(lanes)
        if kind == "bf16":
            w = (torch.randn(rows, cols, device=dev, generator=g) * 0.02).bfloat16()
            head = {"kind": kind, "module": name, "rows": rows, "cols": cols, "wire_bytes": rows * cols * 2}

            def make(m, _lane):
                x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).bfloat16()
                holder = {}

                def call():
                    holder["out"] = torch.nn.functional.linear(x, w)
                return {"floor": floors(rows * cols * 2, m, rows, cols, a_bytes=2)}, {"apply": call}, holder
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
            return {"floor": floors(rows * cols + 4 * rows, m, rows, cols)}, {"apply": call}, holder
        return head, make, ["scaled_mm"]

    built = {}
    for pas in ("F", "R"):
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
                    for cname, call in calls.items():
                        samples = graph_time(call, args.warmup, args.iters)
                        rec = cell.setdefault(cname, {})
                        rec[pas] = {"median_ms": statistics.median(samples), "min_ms": min(samples),
                                    "unix": time.time()}
                        if pas == "F":
                            rec["floor"] = cmeta["floor"]
                            if cname == "apply":
                                rec["profile"] = kernel_profile(call, reps=3)
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
