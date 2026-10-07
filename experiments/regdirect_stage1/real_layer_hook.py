"""build_layer on the class build's real construction bundles (GPU; eng-regdirect-build stage 1).

Builds layer ``--layer`` at TP2 rank ``--rank`` the way the routed loader does
(``compact_prep.parse_compact_wire`` -> ``prepare_window_compact`` with the rank's cut ->
``native_window_moe.WindowUnitAxis`` -> ``prepare_grouped_window_gemm_from_soa``), then
``regdirect_routed.build_layer``.  Two checks:

1. Every payload plane equals the ``real_stack.py`` stack of the same layer and rank, bit for
   bit (those stacks passed the kernel's decode, run-to-run and output checks on GB10).
2. One forward through ``routed_class_dispatch.dispatch_class_projection`` (two classes, two
   streams) equals one direct launch, at decode and prefill M, for both modes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_units import Artifact  # noqa: E402

PROJ = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def bundles(art, layer, rank, experts, device, tp=2):
    from tessera.compact_prep import parse_compact_wire, prepare_window_compact
    from tessera.native_window_moe import WindowUnitAxis
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa
    units = {}
    for role, proj in PROJ.items():
        units[role] = []
        for e in range(experts):
            wire = parse_compact_wire(art.blob(f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}.wire"), device)
            rows, cols = wire.metadata.rows, wire.metadata.manifest.geometry.columns
            if role == "down":
                step = cols // tp
                cut = dict(rows=(0, rows), cols=(rank * step, (rank + 1) * step))
            else:
                step = rows // tp
                cut = dict(rows=(rank * step, (rank + 1) * step), cols=(0, cols))
            units[role].append(prepare_window_compact(wire, device=device, family="e4m3", **cut))
    axis = WindowUnitAxis(experts, tuple(PROJ), family="e4m3",
                          word_runs={role: [(u.rep.words.numel(), u.rep.runs.shape[0]) for u in us]
                                     for role, us in units.items()})
    for role, us in units.items():
        for e, u in enumerate(us):
            axis.put(role, e, u)
    soa = axis.finish()
    out = []
    for role in PROJ:
        s = soa[role]
        out.append(prepare_grouped_window_gemm_from_soa(
            words_all=s["words"], table_all=s["table"], codes_all=s["codes"], native_all=s["native"],
            scale_all=s["scale"], runs_all=s["runs"], init_all=s["init"], has_init=s["has_init"],
            word_off=s["word_off"], tile_words=s["tile_words"], total_words=s["total_words"],
            run_off=s["run_off"], perm_all=s["perm"], rows=s["rows"], cols=s["cols"], experts=experts,
            window_bits=s["window_bits"], family="e4m3", block_m=32, block_n=64, block_k=64, arithmetic="epilogue"))
    return out


class _Resources:
    def __init__(self, kernel, device):
        self.kernel = kernel
        self.streams = tuple(torch.cuda.Stream(device=device) for _ in range(2))
        self.ready, self.finished = torch.cuda.Event(), tuple(torch.cuda.Event() for _ in range(2))
        self.empty = torch.empty(0, dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--stack", required=True, help="real_stack.py path prefix (…/stack-L10-rank0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tessera import regdirect_routed as rr
    from tessera import routed_class_dispatch as rcd
    from tessera.routed_fused import _routing_tables
    device = torch.device("cuda", torch.cuda.current_device())   # an indexed device: tensors report cuda:N
    t0 = time.time()
    gate, up, down = bundles(Artifact(a.artifact), a.layer, a.rank, a.experts, device)
    half = a.experts // 2
    classes = [{"start": 0, "end": half, "q256": {"w13": [1024, 1024], "w2": [1024]}},
               {"start": half, "end": a.experts, "q256": {"w13": [1024, 1024], "w2": [1024]}}]
    top_k = 8
    torch.cuda.synchronize()
    t_load = time.time() - t0
    parameters, kernel = rr.build_layer(gate, up, down, classes, device, top_k=top_k, max_tokens=2048)
    torch.cuda.synchronize()
    t_build = time.time() - t0 - t_load
    # Step 1a/1d of the 14:22Z plan: the per-layer cost of each path, apart from the loader's own work.
    from tessera.routed_fused import compose_table8
    tables = tuple(compose_table8(b) for b in (gate, up, down))
    timings = {}
    for name, fn in (("layer_stacks", rr.layer_stacks), ("transcode_stacks", rr.transcode_stacks)):
        fn(gate, up, down, tables)                       # warm: extension build and allocator
        torch.cuda.synchronize()
        t1 = time.time()
        planes = fn(gate, up, down, tables)
        torch.cuda.synchronize()
        timings[name] = time.time() - t1
        if name == "layer_stacks":
            ref_planes = planes
    same = {mode: {k: bool(torch.equal(planes[mode][k], v)) if torch.is_tensor(v) else planes[mode][k] == v
                   for k, v in ref_planes[mode].items()} for mode in (0, 2)}
    rep = {"layer": a.layer, "rank": a.rank, "experts": a.experts, "build_s": round(t_load + t_build, 1),
           "loader_s": round(t_load, 2), "build_layer_s": round(t_build, 2),
           "layer_stacks_s": round(timings["layer_stacks"], 3), "transcode_stacks_s": round(timings["transcode_stacks"], 4),
           "transcode_equals_layer_stacks": same, "planes": {}}
    for mode in (0, 2):
        ref = torch.load(f"{a.stack}-mode{mode}.pt")
        payload = parameters["regdirect"][mode]
        rep["planes"][mode] = {name: bool(torch.equal(payload[i].cpu(), ref[name]))
                               for i, name in enumerate(rr.PAYLOAD_FIELDS[:-3])}
    resources = _Resources(kernel, device)
    starts, ends = [c["start"] for c in classes], [c["end"] for c in classes]
    rep["dispatch"] = {}
    for m in (1, 16, 2048):
        g = torch.Generator(device=device).manual_seed(m)
        ids = torch.stack([torch.randperm(a.experts, device=device, generator=g)[:top_k] for _ in range(m)]).to(torch.int32)
        w = torch.rand(m, top_k, device=device, generator=g)
        widths = rcd.declared_route_widths(kernel, m, range(len(classes)), parameters)
        routing = _routing_tables(ids, w, a.experts, device, widths)
        for mode in (0, 2):
            rows_in, k = (m, 4096) if mode == 0 else (m * top_k, 1024)
            x = (torch.randn(rows_in, k, device=device, generator=g) * 0.5).to(torch.float8_e4m3fn)
            s = torch.rand(rows_in, device=device, generator=g) * 0.1 + 0.01
            xz, sc = kernel.prepare_input(x, s, rows_in, "e4m3", device)
            n_out = int(parameters["regdirect"][mode][rr.PAYLOAD_FIELDS.index("wscale")].shape[2])
            outs = []
            for spans in (([0], [a.experts]), (starts, ends)):
                out = torch.zeros(m * top_k, n_out, dtype=torch.bfloat16, device=device)
                rcd.dispatch_class_projection(mode, xz, sc, routing, parameters=parameters, starts=spans[0],
                                              ends=spans[1], issue_order=list(range(len(spans[0]))), counters=None,
                                              resources=resources, mul_weight=mode == 2, limit=10.0,
                                              a_row_mode=0 if mode == 0 else 1, out=out)
                torch.cuda.synchronize()
                outs.append(out)
            rep["dispatch"][f"mode{mode}.M{m}"] = {"equal": bool(torch.equal(outs[0], outs[1])),
                                                   "nonzero": bool(outs[0].abs().sum() > 0)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, f"real-layer-hook-rank{a.rank}.json"), "w"), indent=1)
    print(json.dumps(rep))
    ok = (all(all(v.values()) for v in rep["planes"].values()) and all(all(v.values()) for v in rep["dispatch"].values())
          and all(all(v.values()) for v in rep["transcode_equals_layer_stacks"].values()))
    assert ok, "build_layer on real bundles differs from the checked stacks or the direct launch"
    print("real layer hook passed")


if __name__ == "__main__":
    main()
